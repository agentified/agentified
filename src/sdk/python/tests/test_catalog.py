"""Tests for `ToolCatalog` — mirrors `src/sdk/ts/src/catalog.test.ts`."""

import asyncio
import gc
import os
import threading
from pathlib import Path

import pytest

from ratel_ai import (
    DimensionMismatchError,
    EmbedderError,
    ExecutableTool,
    Tool,
    ToolCatalog,
    ToolRegistry,
    TraceSinkConfig,
)

_HUGGING_FACE_HUB = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
    )
)
_CACHED_CANDLE_MODEL = (
    _HUGGING_FACE_HUB
    / "models--BAAI--bge-small-en-v1.5"
    / "snapshots"
    / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
_HAS_CACHED_CANDLE_MODEL = (
    (_CACHED_CANDLE_MODEL / "config.json").is_file()
    and (_CACHED_CANDLE_MODEL / "tokenizer.json").is_file()
    and (
        (_CACHED_CANDLE_MODEL / "model.safetensors").is_file()
        or (_CACHED_CANDLE_MODEL / "pytorch_model.bin").is_file()
    )
)


def _read_file_tool(execute) -> ExecutableTool:
    return ExecutableTool(
        id="read_file",
        name="read_file",
        description="Read a file from local disk and return its textual contents.",
        input_schema={"properties": {"path": {"type": "string"}}},
        output_schema={"properties": {"contents": {"type": "string"}}},
        execute=execute,
    )


async def _register_when_idle(catalog: ToolCatalog, tool: ExecutableTool) -> None:
    deadline = asyncio.get_running_loop().time() + 2
    while True:
        try:
            await catalog.register(tool)
            return
        except RuntimeError as err:
            if str(err) != "registry busy; await the active operation":
                raise
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("dense worker did not finish") from err
            await asyncio.sleep(0.01)


class _PausingTool(Tool):
    def __init__(
        self, reached: threading.Event, release: threading.Event, *, tool_id: str = "raced"
    ) -> None:
        super().__init__(id=tool_id, name=tool_id, description="Race marker")
        self._reached = reached
        self._release = release
        self._paused = False

    def __getattribute__(self, name: str):
        if name == "id" and not object.__getattribute__(self, "_paused"):
            object.__setattr__(self, "_paused", True)
            object.__getattribute__(self, "_reached").set()
            if not object.__getattribute__(self, "_release").wait(timeout=5):
                raise TimeoutError("registration was not released")
        return super().__getattribute__(name)


async def test_register_then_get_and_has() -> None:
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {"contents": "hi"}))
    assert catalog.has("read_file")
    assert not catalog.has("missing")
    tool = catalog.get("read_file")
    assert tool is not None and tool.name == "read_file"
    # get() returns metadata only — no execute attribute leaked as a Tool
    assert not hasattr(tool, "execute")
    executable = catalog.get_executable("read_file")
    assert executable is not None and executable.execute is not None


async def test_register_removed_methods_are_gone() -> None:
    # register_many / build_embeddings / rebuild_embeddings were folded into the
    # variadic, self-embedding `register` (RAT-379/async-register).
    catalog = ToolCatalog()
    registry = ToolRegistry()
    for obj in (catalog, registry):
        assert not hasattr(obj, "register_many")
        assert not hasattr(obj, "build_embeddings")
        assert not hasattr(obj, "rebuild_embeddings")


async def test_register_accepts_an_iterable_and_adds_every_tool() -> None:
    catalog = ToolCatalog()

    await catalog.register(
        [
            _read_file_tool(lambda args: {}),
            ExecutableTool(
                id="send", name="send", description="Send email", execute=lambda args: {}
            ),
        ]
    )

    assert catalog.has("read_file") and catalog.has("send")


@pytest.mark.parametrize(
    "invalid_schema",
    [
        {"bad": object()},
        {1: "non-string key"},
        {"too_large": 2**1000},
    ],
)
async def test_register_iterable_validation_failure_commits_nothing(
    invalid_schema: dict[object, object],
) -> None:
    catalog = ToolCatalog()
    invalid = ExecutableTool(
        id="invalid",
        name="invalid",
        description="Invalid schema",
        input_schema=invalid_schema,  # type: ignore[arg-type]
        execute=lambda args: {},
    )

    with pytest.raises((TypeError, ValueError)):
        await catalog.register([_read_file_tool(lambda args: {}), invalid])

    assert not catalog.has("read_file")
    assert catalog.search("read a file", 5) == []


async def test_register_rejects_tool_without_execute() -> None:
    catalog = ToolCatalog()
    with pytest.raises(ValueError, match="no execute handler"):
        await catalog.register(
            ExecutableTool(id="x", name="x", description="d", execute=None)  # type: ignore[arg-type]
        )


async def test_search_ranks_the_relevant_tool_first() -> None:
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {}))
    await catalog.register(
        ExecutableTool(
            id="send_email",
            name="send_email",
            description="Send an email message to a recipient.",
            input_schema={},
            output_schema={},
            execute=lambda args: {},
        )
    )
    hits = catalog.search("read a file from disk", 5)
    assert hits[0].tool_id == "read_file"


async def test_search_defaults_to_bm25_stage() -> None:
    # Semantic/hybrid load a real model (network) and are covered in Rust; this
    # stays offline and asserts the model-free default + selection plumbing.
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="m"))
    await catalog.register(_read_file_tool(lambda args: {}))
    hits = catalog.search("read a file", 5)
    assert hits[0].tool_id == "read_file"
    events = catalog.drain_trace_events()
    search = next(e for e in events if e["type"] == "search")
    assert any(stage["name"] == "bm25" for stage in search["stages"])


async def test_per_call_bm25_matches_the_default() -> None:
    # Stays on BM25 so no model loads; dense override behavior is covered below.
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {}))
    via_default = [h.tool_id for h in catalog.search("read a file", 5)]
    via_explicit = [h.tool_id for h in catalog.search("read a file", 5, method="bm25")]
    assert via_explicit == via_default
    assert via_explicit[0] == "read_file"


async def test_per_call_method_overrides_default_and_reroutes() -> None:
    # Default is semantic, but with no registrations no model loads. A per-call
    # "bm25" must route to the bm25 engine — provable offline via the trace stage
    # the semantic default (empty corpus) never emits.
    catalog = ToolCatalog(method="semantic", trace=TraceSinkConfig(kind="memory", session_id="o"))
    await catalog.search_async("anything", 5)  # default: semantic engine
    catalog.search("anything", 5, method="bm25")  # per-call override: bm25 engine
    searches = [e for e in catalog.drain_trace_events() if e["type"] == "search"]
    assert len(searches) == 2
    assert not any(s["name"] == "bm25" for s in searches[0]["stages"])
    assert any(s["name"] == "bm25" for s in searches[1]["stages"])


async def test_unknown_method_raises() -> None:
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {}))
    with pytest.raises(ValueError, match="unknown search method"):
        catalog.search("read", 5, method="keyword")


async def test_register_empty_batch_on_semantic_is_a_noop() -> None:
    # Empty corpus short-circuits before any embedder load, even for the eager
    # per-register build a semantic/hybrid catalog now runs.
    catalog = ToolCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})
    await catalog.register([])  # no tools → no model load, must not raise
    assert not catalog.has("anything")


async def test_semantic_registration_embeds_and_search_finds_hit(
    delayed_embedding_endpoint: str,
) -> None:
    # Registration on a semantic/hybrid catalog now embeds eagerly (inside
    # `register`), so "metadata only" no longer holds — prove the end-to-end
    # effect instead: search_async finds the hit right after registration.
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )

    await catalog.register(_read_file_tool(lambda args: {}))

    assert catalog.has("read_file")
    hits = await catalog.search_async("read a file", 5)
    assert hits[0].tool_id == "read_file"


def test_synchronous_dense_search_points_to_search_async() -> None:
    # search() rejects a resolved semantic/hybrid method before ever touching
    # the registry, so this needs no registration (and no working model).
    catalog = ToolCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})

    with pytest.raises(RuntimeError, match=r"await .*search_async"):
        catalog.search("read", 5)


async def test_semantic_on_bm25_without_embeddings_raises() -> None:
    # A BM25 catalog with no embeddings built → a per-call semantic search refuses with a
    # clear error instead of silently embedding. Guard runs before any model
    # load, so this is offline-safe.
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {}))
    with pytest.raises(RuntimeError, match="not computed for semantic"):
        await catalog.search_async("read", 5, method="semantic")


async def test_unawaited_register_commits_bm25_metadata_synchronously() -> None:
    # register() indexes metadata the instant it is called — a forgotten `await`
    # never silently drops the corpus. BM25 search finds the tool without awaiting.
    catalog = ToolCatalog()
    pending = catalog.register(_read_file_tool(lambda args: {}))  # deliberately not awaited
    assert catalog.has("read_file")
    assert catalog.search("read a file", 5)[0].tool_id == "read_file"
    await pending  # drain the (no-op) awaitable so no coroutine is left dangling


async def test_unawaited_semantic_register_raises_on_dense_search(
    delayed_embedding_endpoint: str,
) -> None:
    # A forgotten `await` on a semantic catalog leaves the embedding pass undriven;
    # a dense search then fails loudly with an await-specific message instead of
    # ranking an empty corpus. Awaiting the pending build afterwards recovers.
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )
    pending = catalog.register(_read_file_tool(lambda args: {}))  # forgot to await

    assert catalog.has("read_file")  # metadata still committed synchronously
    with pytest.raises(RuntimeError, match="was not awaited"):
        await catalog.search_async("read a file", 5)

    await pending  # driving the build clears the guard; search then works
    hits = await catalog.search_async("read a file", 5)
    assert hits[0].tool_id == "read_file"


async def test_awaited_semantic_register_leaves_no_pending_build(
    delayed_embedding_endpoint: str,
) -> None:
    # The happy path is unchanged: an awaited register drives the build to
    # completion, so the guard never fires on the next dense search.
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )
    await catalog.register(_read_file_tool(lambda args: {}))
    assert catalog._registry._undriven_builds == 0
    hits = await catalog.search_async("read a file", 5)
    assert hits[0].tool_id == "read_file"


async def test_delayed_endpoint_register_keeps_asyncio_responsive(
    delayed_embedding_endpoint: str,
) -> None:
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )

    register = asyncio.create_task(catalog.register(_read_file_tool(lambda args: {})))
    heartbeats = 0
    while not register.done():
        heartbeats += 1
        await asyncio.sleep(0.01)
    await register

    assert heartbeats >= 5


@pytest.mark.skipif(
    not _HAS_CACHED_CANDLE_MODEL,
    reason="pinned bge-small model is not present in the HuggingFace cache",
)
async def test_cached_candle_register_keeps_asyncio_responsive() -> None:
    catalog = ToolCatalog(
        method="semantic",
        embedding={"local": str(_CACHED_CANDLE_MODEL), "pooling": "cls"},
    )

    register = asyncio.create_task(catalog.register(_read_file_tool(lambda args: {})))
    heartbeats = 0
    while not register.done():
        heartbeats += 1
        await asyncio.sleep(0.001)
    await register

    assert heartbeats > 1


async def test_delayed_endpoint_search_keeps_asyncio_responsive(
    delayed_embedding_endpoint: str,
) -> None:
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )
    await catalog.register(_read_file_tool(lambda args: {}))

    search = asyncio.create_task(catalog.search_async("read", 5))
    heartbeats = 0
    while not search.done():
        heartbeats += 1
        await asyncio.sleep(0.01)
    hits = await search

    assert hits[0].tool_id == "read_file" and heartbeats >= 5


async def test_cancelled_dense_register_keeps_registration_busy_until_worker_finishes(
    delayed_embedding_endpoint: str,
) -> None:
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )
    register = asyncio.create_task(catalog.register(_read_file_tool(lambda args: {})))
    await asyncio.sleep(0.02)
    register.cancel()
    with pytest.raises(asyncio.CancelledError):
        await register

    with pytest.raises(RuntimeError, match=r"^registry busy; await the active operation$"):
        await catalog.register(
            ExecutableTool(
                id="send", name="send", description="Send email", execute=lambda args: {}
            )
        )

    await _register_when_idle(
        catalog,
        ExecutableTool(id="send", name="send", description="Send email", execute=lambda args: {}),
    )
    assert catalog.has("send")


async def test_cancelled_failing_dense_worker_exception_is_consumed(
    delayed_embedding_endpoint: str,
) -> None:
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "fail-model"},
    )
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        register = asyncio.create_task(catalog.register(_read_file_tool(lambda args: {})))
        await asyncio.sleep(0.02)
        register.cancel()
        with pytest.raises(asyncio.CancelledError):
            await register

        # Every register on this catalog embeds against the same broken
        # "fail-model" endpoint, so the retry below cannot succeed — it only
        # needs to stop seeing "busy", proving the cancelled worker's leftover
        # failure released the registry rather than wedging it.
        deadline = asyncio.get_running_loop().time() + 2
        probe = ExecutableTool(
            id="probe", name="probe", description="Probe", execute=lambda args: {}
        )
        while True:
            try:
                await catalog.register(probe)
                break
            except EmbedderError:
                break
            except RuntimeError as err:
                if str(err) != "registry busy; await the active operation":
                    raise
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("dense worker did not finish") from err
                await asyncio.sleep(0.01)

        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)


async def test_second_registration_raises_busy_while_first_still_embedding(
    delayed_embedding_endpoint: str,
) -> None:
    # No standalone build/rebuild remains to queue two dense ops behind one
    # another; a single in-flight eager register already exercises the same
    # "busy while a dense op owns the registry" invariant.
    catalog = ToolCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )
    first = asyncio.create_task(catalog.register(_read_file_tool(lambda args: {})))
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match=r"^registry busy; await the active operation$"):
        await catalog.register(
            ExecutableTool(
                id="send", name="send", description="Send email", execute=lambda args: {}
            )
        )

    await first


@pytest.mark.parametrize("batch", [False, True])
def test_registration_and_batch_cannot_race_dense_native_borrow(
    controlled_embedding_endpoint: tuple[str, threading.Event, threading.Event],
    batch: bool,
) -> None:
    # `_build` is the internal dense-build primitive an eager `register` now
    # drives; exercised directly here (no public standalone build call remains
    # on `ToolRegistry`) to prove registration and a dense build still cannot
    # race the native borrow — the invariant this test has always covered.
    endpoint, request_started, send_response = controlled_embedding_endpoint
    registry = ToolRegistry(embedding={"url": endpoint, "model": "test-model"})
    asyncio.run(registry.register(Tool(id="base", name="base", description="Base tool")))
    reached = threading.Event()
    release = threading.Event()
    raced = _PausingTool(reached, release)
    registration_errors: list[BaseException] = []
    build_errors: list[BaseException] = []

    def register() -> None:
        try:
            if batch:
                asyncio.run(
                    registry.register([Tool(id="first", name="first", description="First"), raced])
                )
            else:
                asyncio.run(registry.register(raced))
        except BaseException as err:
            registration_errors.append(err)

    def build() -> None:
        try:
            asyncio.run(registry._build())
        except BaseException as err:
            build_errors.append(err)

    registration = threading.Thread(target=register)
    registration.start()
    assert reached.wait(timeout=2)
    dense = threading.Thread(target=build)
    dense.start()

    request_started.wait(timeout=0.5)
    release.set()
    assert request_started.wait(timeout=2)
    send_response.set()
    registration.join(timeout=2)
    dense.join(timeout=2)

    assert not registration.is_alive() and not dense.is_alive()
    assert registration_errors == []
    assert build_errors == []
    assert registry.search("race marker", 5)[0].tool_id == "raced"


async def test_trace_sink_mutation_reports_busy_during_dense_native_borrow(
    controlled_embedding_endpoint: tuple[str, threading.Event, threading.Event],
) -> None:
    endpoint, request_started, send_response = controlled_embedding_endpoint
    registry = ToolRegistry(embedding={"url": endpoint, "model": "test-model"})
    await registry.register(Tool(id="base", name="base", description="Base tool"))
    build = asyncio.create_task(registry._build())
    for _ in range(200):
        if request_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert request_started.is_set()
    try:
        with pytest.raises(RuntimeError, match=r"^registry busy; await the active operation$"):
            registry.set_trace_sink("noop")
    finally:
        send_response.set()
    await build


def test_embedding_invalid_config_raises_valueerror_at_construction() -> None:
    # A bare URL has no model name → construction-time ValueError.
    with pytest.raises(ValueError, match="model"):
        ToolCatalog(method="semantic", embedding="https://api.openai.com/v1/embeddings")


def test_embedding_bare_repo_id_string_points_to_huggingface() -> None:
    # A string is a local path only; a repo id must use {"huggingface": ...}.
    with pytest.raises(ValueError, match="huggingface"):
        ToolCatalog(method="semantic", embedding="BAAI/bge-base-en-v1.5")


def test_embedding_conflicting_endpoint_keys_raise() -> None:
    with pytest.raises(ValueError, match="conflicting"):
        ToolCatalog(
            method="semantic",
            embedding={"ollama": "nomic", "url": "http://h:11434/v1/embeddings", "model": "nomic"},
        )


def test_embedding_unknown_dict_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown embedding keys"):
        ToolCatalog(method="semantic", embedding={"bogus": "x"})


def test_embedding_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ToolCatalog(embedding={})


def test_embedding_pooling_override_and_doc_prefix_accepted() -> None:
    # Construction validates (does not load); a valid pooling + doc_prefix is fine.
    ToolCatalog(
        method="semantic",
        embedding={"huggingface": "org/m", "pooling": "mean", "doc_prefix": "passage: "},
    )


def test_embedding_invalid_pooling_value_raises() -> None:
    with pytest.raises(ValueError, match="pooling"):
        ToolCatalog(method="semantic", embedding={"huggingface": "org/m", "pooling": "median"})


def test_embedding_pooling_on_endpoint_raises() -> None:
    with pytest.raises(ValueError, match="pooling"):
        ToolCatalog(method="semantic", embedding={"ollama": "nomic", "pooling": "mean"})


def test_embedding_api_key_env_on_ollama_raises() -> None:
    with pytest.raises(ValueError, match="api_key_env"):
        ToolCatalog(
            method="semantic",
            embedding={"ollama": "nomic", "api_key_env": "OLLAMA_API_KEY"},
        )


def test_embedding_download_opt_in_accepted() -> None:
    ToolCatalog(method="semantic", embedding={"huggingface": "org/m", "download": True})


def test_embedding_download_on_non_huggingface_raises() -> None:
    with pytest.raises(ValueError, match="download"):
        ToolCatalog(method="semantic", embedding={"local": "/opt/models/x", "download": True})


async def test_embedding_is_retained_under_bm25_without_eager_model_load() -> None:
    # A bm25-default catalog never loads the model during register — even with a
    # config that would fail to load. Constructing a semantic catalog with the
    # very same (broken) config proves it is the mode, not the model spec, that
    # decides whether register embeds and can fail.
    embedding = {"local": "/missing/ratel-model"}
    catalog = ToolCatalog(embedding=embedding)
    await catalog.register(_read_file_tool(lambda args: {}))
    assert catalog.search("read", 5)[0].tool_id == "read_file"

    semantic_catalog = ToolCatalog(method="semantic", embedding=embedding)
    with pytest.raises(EmbedderError, match="/missing/ratel-model"):
        await semantic_catalog.register(_read_file_tool(lambda args: {}))


def test_typed_embedder_exceptions_are_runtimeerror_subclasses() -> None:
    assert issubclass(EmbedderError, RuntimeError)
    assert issubclass(DimensionMismatchError, EmbedderError)


async def test_invoke_runs_sync_executor() -> None:
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {"contents": f"read {args['path']}"}))
    result = await catalog.invoke("read_file", {"path": "/tmp/x"})
    assert result == {"contents": "read /tmp/x"}


async def test_invoke_runs_async_executor() -> None:
    async def handler(args):
        return {"contents": f"async {args['path']}"}

    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(handler))
    result = await catalog.invoke("read_file", {"path": "/tmp/y"})
    assert result == {"contents": "async /tmp/y"}


async def test_invoke_unknown_tool_raises() -> None:
    catalog = ToolCatalog()
    with pytest.raises(ValueError, match="unknown toolId"):
        await catalog.invoke("nope", {})


async def test_invoke_emits_start_then_end_telemetry() -> None:
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="s"))
    await catalog.register(_read_file_tool(lambda args: {"ok": True}))
    catalog.drain_trace_events()  # clear registration churn
    await catalog.invoke("read_file", {"path": "/a"})
    events = catalog.drain_trace_events()
    types = [e["type"] for e in events]
    assert types == ["invoke_start", "invoke_end"]
    assert events[0]["tool_id"] == "read_file"
    assert events[0]["args_size_bytes"] > 0
    assert "took_ms" in events[1]


async def test_invoke_emits_error_telemetry_and_reraises() -> None:
    def boom(args):
        raise RuntimeError("kaboom")

    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="s"))
    await catalog.register(_read_file_tool(boom))
    catalog.drain_trace_events()
    with pytest.raises(RuntimeError, match="kaboom"):
        await catalog.invoke("read_file", {"path": "/a"})
    events = catalog.drain_trace_events()
    types = [e["type"] for e in events]
    assert types == ["invoke_start", "invoke_error"]
    assert events[1]["error"] == "kaboom"


async def test_re_register_replaces_in_place() -> None:
    # Re-registering an id replaces it in the native corpus, not appends a
    # duplicate: the id ranks once and the latest executor wins (RAT-378).
    catalog = ToolCatalog()
    await catalog.register(_read_file_tool(lambda args: {"contents": "v1"}))
    await catalog.register(
        ExecutableTool(
            id="read_file",
            name="read_file",
            description="Fetch and return a document over the network.",
            input_schema={"properties": {"path": {"type": "string"}}},
            output_schema={"properties": {"contents": {"type": "string"}}},
            execute=lambda args: {"contents": "v2"},
        )
    )
    hits = catalog.search("fetch a document over the network", 10)
    assert [h.tool_id for h in hits].count("read_file") == 1
    assert await catalog.invoke("read_file", {"path": "/x"}) == {"contents": "v2"}
