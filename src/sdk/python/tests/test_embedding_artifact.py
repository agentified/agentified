"""Embedding artifact build/warm lifecycle (ADR-0018) — Python Phase 2."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ratel_ai import (
    ArtifactWarmError,
    ExecutableTool,
    IncompatibleMergeError,
    Skill,
    SkillCatalog,
    SkillRegistry,
    Tool,
    ToolCatalog,
    ToolRegistry,
    experimental_build_embedding_artifact,
)
from ratel_ai._native import merge_embedding_artifacts
from ratel_ai.embedding_artifact import resolve_embedding_artifact


class _CountingEndpoint:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, object]] = []
        self.delay_s = 0.0
        self.hold_min_batch = 0
        self.response_models: list[str] = []
        self._hold: threading.Event | None = None
        self._release: threading.Event | None = None

    def set_response_models(self, models: Sequence[str]) -> None:
        """Override response ``model`` per request (FIFO)."""
        with self.lock:
            self.response_models = list(models)

    def record(self, payload: dict[str, object]) -> None:
        with self.lock:
            self.requests.append(payload)

    def arm_batch_hold(self, min_batch: int = 2) -> None:
        """Hold the next request whose input length is ``>= min_batch`` (once)."""
        with self.lock:
            self.hold_min_batch = min_batch
            self._hold = threading.Event()
            self._release = threading.Event()

    def release_held(self) -> None:
        assert self._release is not None
        self._release.set()


def _payload_inputs(payload: Mapping[str, object]) -> list[str]:
    raw = payload["input"]
    if isinstance(raw, str):
        return [raw]
    assert isinstance(raw, Sequence)
    return [str(item) for item in raw]


def _corpus_inputs(requests: Sequence[Mapping[str, object]]) -> set[str]:
    texts: set[str] = set()
    for payload in requests:
        texts.update(_payload_inputs(payload))
    return texts


def _assert_corpus_not_reembedded(
    corpus: set[str], after: Sequence[Mapping[str, object]]
) -> None:
    """Corpus document texts from build must not appear again (probe/query ok)."""
    for payload in after:
        for text in _payload_inputs(payload):
            assert text not in corpus, f"corpus document re-embedded: {text!r}"


@pytest.fixture
def counting_embedding_endpoint() -> Iterator[tuple[str, _CountingEndpoint]]:
    state = _CountingEndpoint()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            state.record(payload)
            inputs = _payload_inputs(payload)
            should_hold = False
            with state.lock:
                if state.hold_min_batch > 0 and len(inputs) >= state.hold_min_batch:
                    state.hold_min_batch = 0
                    should_hold = True
            if should_hold:
                assert state._hold is not None and state._release is not None
                state._hold.set()
                if not state._release.wait(timeout=10):
                    self.send_response(504)
                    self.end_headers()
                    return
            if state.delay_s:
                import time

                time.sleep(state.delay_s)
            with state.lock:
                model = (
                    state.response_models.pop(0)
                    if state.response_models
                    else str(payload["model"])
                )
            data = [
                {"embedding": [1.0, float(index + 1)], "index": index}
                for index, _ in enumerate(inputs)
            ]
            body = json.dumps({"data": data, "model": model}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/embeddings", state
    finally:
        if state._release is not None:
            state._release.set()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


async def _wait_hold(state: _CountingEndpoint, timeout: float = 5.0) -> None:
    """Wait for the HTTP hold without blocking the asyncio event loop."""
    assert state._hold is not None
    deadline = asyncio.get_running_loop().time() + timeout
    while not state._hold.is_set():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("embedding endpoint hold was never signaled")
        await asyncio.sleep(0.01)


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition was never met")
        await asyncio.sleep(0.01)


READ_FILE = Tool(
    id="read_file",
    name="read_file",
    description="Read a file from local disk and return its textual contents.",
    input_schema={"properties": {"path": {"type": "string"}}},
    output_schema={},
)
WRITE_FILE = Tool(
    id="write_file",
    name="write_file",
    description="Write textual contents to a file on local disk.",
    input_schema={
        "properties": {
            "path": {"type": "string"},
            "contents": {"type": "string"},
        }
    },
    output_schema={},
)
SLIDES = Skill(
    id="frontend-slides",
    name="frontend-slides",
    description="Build animation-rich HTML presentations from scratch.",
    tags=["frontend", "presentations"],
    body="# Frontend Slides\n\nStep 1: pick an aesthetic.",
)
API_DESIGN = Skill(
    id="api-design",
    name="api-design",
    description="REST API design patterns: resource naming, status codes, pagination.",
    tags=["backend", "api"],
    body="# API Design\n\nUse nouns for resources.",
)
SEARCH_TOOL = Tool(
    id="search",
    name="search",
    description="Search the tool catalog for matching tools.",
    input_schema={"properties": {"q": {"type": "string"}}},
    output_schema={},
)
SEARCH_SKILL = Skill(
    id="search",
    name="search",
    description="Search skill playbooks for matching guidance.",
    tags=["search"],
    body="# Search skill",
)


def _emb(url: str) -> dict[str, str]:
    return {"url": url, "model": "test-model"}


def _executable(tool: Tool) -> ExecutableTool:
    return ExecutableTool(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        output_schema=tool.output_schema,
        execute=lambda _args: {},
    )


async def test_build_mixed_artifact_warms_both_kinds(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "catalog.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output,
        embedding=embedding,
        tools=[READ_FILE, WRITE_FILE],
        skills=[SLIDES, API_DESIGN],
    )
    bytes_ = output.read_bytes()
    assert bytes_[:4] == b"RAT1"

    tools = ToolRegistry(embedding, method="bm25")
    await tools.register([READ_FILE, WRITE_FILE])
    await tools.experimental_warm_embeddings_from_artifact(bytes_, "error")
    tool_hits = await tools.search_async("read a file", 5, method="semantic")
    assert tool_hits[0].tool_id == "read_file"

    skills = SkillRegistry(embedding, method="bm25")
    await skills.register([SLIDES, API_DESIGN])
    await skills.experimental_warm_embeddings_from_artifact(bytes_, "error")
    skill_hits = await skills.search_async("html presentations", 5, method="semantic")
    assert skill_hits[0].skill_id == "frontend-slides"


async def test_build_embeds_each_document_at_most_once(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    await experimental_build_embedding_artifact(
        output=tmp_path / "once.ratel-embeddings",
        embedding=_emb(url),
        tools=[READ_FILE],
        skills=[SLIDES],
    )
    # Tool batch + skill batch only (no catalog dense build).
    assert len(state.requests) == 2


async def test_tool_only_and_skill_only_and_both_empty(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    tools_out = tmp_path / "tools.ratel-embeddings"
    skills_out = tmp_path / "skills.ratel-embeddings"
    empty_out = tmp_path / "empty.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=tools_out, embedding=embedding, tools=[READ_FILE]
    )
    await experimental_build_embedding_artifact(
        output=skills_out, embedding=embedding, skills=[SLIDES]
    )
    await experimental_build_embedding_artifact(output=empty_out, tools=[], skills=[])
    assert tools_out.read_bytes()[:4] == b"RAT1"
    assert skills_out.read_bytes()[:4] == b"RAT1"
    empty = empty_out.read_bytes()
    assert empty[:4] == b"RAT1"
    tools = ToolRegistry(method="bm25")
    await tools.experimental_warm_embeddings_from_artifact(empty, "error")


async def test_overwrite_and_missing_parent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "overwrite.ratel-embeddings"
    output.write_text("stale")
    await experimental_build_embedding_artifact(output=output, tools=[], skills=[])
    assert output.read_bytes()[:4] == b"RAT1"

    missing = tmp_path / "nope" / "missing" / "out.ratel-embeddings"
    with pytest.raises(FileNotFoundError):
        await experimental_build_embedding_artifact(output=missing, tools=[], skills=[])


async def test_same_textual_id_across_kinds_and_merge(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    tools = ToolRegistry(embedding, method="bm25")
    await tools.register(SEARCH_TOOL)
    tool_bytes = await tools.experimental_build_embedding_artifact()
    skills = SkillRegistry(embedding, method="bm25")
    await skills.register(SEARCH_SKILL)
    skill_bytes = await skills.experimental_build_embedding_artifact()
    merged = merge_embedding_artifacts([tool_bytes, skill_bytes])
    assert isinstance(merged, (bytes, bytearray))

    tools2 = ToolRegistry(embedding, method="bm25")
    await tools2.register(SEARCH_TOOL)
    await tools2.experimental_warm_embeddings_from_artifact(merged, "error")
    skills2 = SkillRegistry(embedding, method="bm25")
    await skills2.register(SEARCH_SKILL)
    await skills2.experimental_warm_embeddings_from_artifact(merged, "error")


async def test_incomplete_preserves_missing_ids(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "partial.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE]
    )
    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"bytes": output.read_bytes()},
    )
    with pytest.raises(ArtifactWarmError) as caught:
        await catalog.register([_executable(READ_FILE), _executable(WRITE_FILE)])
    err = caught.value
    assert err.code == "Incomplete"
    assert err.missing == ["write_file"]
    assert type(err).__name__ == "ArtifactWarmError"
    assert not hasattr(err, "name")
    expected = (
        "embedding artifact incomplete for the current corpus: 1 id(s) missing (write_file)"
    )
    assert str(err) == expected


async def test_on_miss_embed_fills_missing(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "partial.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE]
    )
    after_build = len(state.requests)
    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={
            "bytes": output.read_bytes(),
            "on_miss": "embed",
        },
    )
    await catalog.register([_executable(READ_FILE), _executable(WRITE_FILE)])
    # Warm may probe identity (+0/1) then embed the one missing id (+1).
    assert len(state.requests) >= after_build + 1
    assert len(state.requests) <= after_build + 2
    hits = await catalog.search_async("write a file", 5)
    ids = [h.tool_id for h in hits]
    assert "write_file" in ids
    assert "read_file" in ids


async def test_path_warm_avoids_corpus_reembedding(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "catalog.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output,
        embedding=embedding,
        tools=[READ_FILE],
        skills=[SLIDES],
    )
    corpus = _corpus_inputs(state.requests)
    docs_after_build = len(state.requests)

    tools = ToolCatalog(
        method="hybrid",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    await tools.register(_executable(READ_FILE))
    _assert_corpus_not_reembedded(corpus, state.requests[docs_after_build:])
    hits = await tools.search_async("read a file", 5)
    assert hits[0].tool_id == "read_file"


async def test_mixed_artifact_serves_tool_and_skill_catalogs(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "mixed.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output,
        embedding=embedding,
        tools=[SEARCH_TOOL],
        skills=[SEARCH_SKILL],
    )
    corpus = _corpus_inputs(state.requests)
    docs_after_build = len(state.requests)
    artifact = {"path": str(output)}
    tools = ToolCatalog(
        method="hybrid",
        embedding=embedding,
        experimental_embedding_artifact=artifact,
    )
    skills = SkillCatalog(
        method="hybrid",
        embedding=embedding,
        experimental_embedding_artifact=artifact,
    )
    await tools.register(_executable(SEARCH_TOOL))
    await skills.register(SEARCH_SKILL)
    _assert_corpus_not_reembedded(corpus, state.requests[docs_after_build:])
    assert (await tools.search_async("matching tools", 5))[0].tool_id == "search"
    assert (await skills.search_async("matching guidance", 5))[0].skill_id == "search"


async def test_bm25_with_artifact_still_warms(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "bm25.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE]
    )
    corpus = _corpus_inputs(state.requests)
    docs_after_build = len(state.requests)
    catalog = ToolCatalog(
        method="bm25",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    await catalog.register(_executable(READ_FILE))
    _assert_corpus_not_reembedded(corpus, state.requests[docs_after_build:])
    hits = await catalog.search_async("read a file", 5, method="semantic")
    assert hits[0].tool_id == "read_file"


async def test_build_concurrent_with_semantic_search(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
) -> None:
    """Build must overlap search: search reaches the endpoint while build is held.

    Fails if ``experimental_build_embedding_artifact`` ever takes ``_dense_gate``.
    """
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    # Prefill dense cache so semantic search only needs a query embed (batch 1).
    reg = ToolRegistry(embedding, method="semantic")
    await reg.register([READ_FILE, WRITE_FILE])

    state.arm_batch_hold(2)
    build_task = asyncio.create_task(reg.experimental_build_embedding_artifact())
    await _wait_hold(state)
    held_at = len(state.requests)

    search_task = asyncio.create_task(reg.search_async("read a file", 5, method="semantic"))
    await _wait_until(lambda: len(state.requests) > held_at)
    assert not build_task.done()
    assert not state._release.is_set()  # type: ignore[union-attr]

    state.release_held()
    artifact, hits = await asyncio.gather(build_task, search_task)
    assert artifact[:4] == b"RAT1"
    assert hits[0].tool_id == "read_file"


async def test_mutation_rejected_while_build_pending(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    reg = ToolRegistry(embedding, method="bm25")
    await reg.register(READ_FILE)

    state.arm_batch_hold(1)
    build_task = asyncio.create_task(reg.experimental_build_embedding_artifact())
    await _wait_hold(state)
    with pytest.raises(RuntimeError, match="registry busy"):
        await reg.register(WRITE_FILE)
    state.release_held()
    await build_task
    await reg.register(WRITE_FILE)


async def test_cancel_await_keeps_mutation_blocked_until_build_finishes(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    reg = ToolRegistry(embedding, method="bm25")
    await reg.register(READ_FILE)

    state.arm_batch_hold(1)
    build_task = asyncio.create_task(reg.experimental_build_embedding_artifact())
    await _wait_hold(state)
    build_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await build_task
    with pytest.raises(RuntimeError, match="registry busy"):
        await reg.register(WRITE_FILE)
    state.release_held()
    # Give the worker a moment to finish and clear pending.
    for _ in range(50):
        try:
            await reg.register(WRITE_FILE)
            break
        except RuntimeError as err:
            if "registry busy" not in str(err):
                raise
            await asyncio.sleep(0.05)
    else:
        pytest.fail("mutation stayed busy after build finished")


async def test_skill_replace_all_warms_configured_artifact(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "skills.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output,
        embedding=embedding,
        skills=[SLIDES, API_DESIGN],
    )
    corpus = _corpus_inputs(state.requests)
    docs_after_build = len(state.requests)

    catalog = SkillCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    reload = catalog.replace_all([SLIDES, API_DESIGN])
    assert reload.added == 2
    await reload
    _assert_corpus_not_reembedded(corpus, state.requests[docs_after_build:])
    hits = await catalog.search_async("html presentations", 5)
    assert hits[0].skill_id == "frontend-slides"


async def test_second_register_rewarms_without_document_inference(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    """PY-1: split-register re-warm with zero document inference."""
    url, state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "both-tools.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output,
        embedding=embedding,
        tools=[READ_FILE, WRITE_FILE],
    )
    # Artifact holds A+B, but warm only applies entries in the current corpus.
    # First register ({A}) must not populate B; the second register must warm B
    # from the artifact (no document inference) — search for B proves it.
    corpus = _corpus_inputs(state.requests)
    docs_after_build = len(state.requests)

    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    await catalog.register(_executable(READ_FILE))
    _assert_corpus_not_reembedded(corpus, state.requests[docs_after_build:])
    hits_after_a = await catalog.search_async("write a file", 5)
    assert "write_file" not in [h.tool_id for h in hits_after_a]

    docs_after_first = len(state.requests)
    await catalog.register(_executable(WRITE_FILE))
    _assert_corpus_not_reembedded(corpus, state.requests[docs_after_first:])

    hits = await catalog.search_async("write a file", 5)
    assert "write_file" in [h.tool_id for h in hits]


async def test_path_artifact_reread_on_second_register(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    """PY-2: same path config re-reads changed artifact bytes between registers."""
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    path = tmp_path / "path-swap.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=path, embedding=embedding, tools=[READ_FILE]
    )
    artifact = {"path": str(path)}

    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact=artifact,
    )
    await catalog.register(_executable(READ_FILE))
    hits_after_a = await catalog.search_async("write a file", 5)
    assert "write_file" not in [h.tool_id for h in hits_after_a]

    await experimental_build_embedding_artifact(
        output=path,
        embedding=embedding,
        tools=[READ_FILE, WRITE_FILE],
    )
    await catalog.register(_executable(WRITE_FILE))

    hits = await catalog.search_async("write a file", 5)
    assert "write_file" in [h.tool_id for h in hits]


async def test_incremental_register_fails_closed_on_second_register(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    """PY-3: second register beyond artifact coverage fails with exact missing ids."""
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "tool-a-only.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE]
    )

    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    await catalog.register(_executable(READ_FILE))

    with pytest.raises(ArtifactWarmError) as caught:
        await catalog.register(_executable(WRITE_FILE))
    err = caught.value
    assert isinstance(err, ArtifactWarmError)
    assert err.code == "Incomplete"
    assert err.missing == ["write_file"]


async def test_replace_all_twice_rereads_path_artifact(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    """PY-4: replace_all twice observes new bytes at the same configured path."""
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    path = tmp_path / "skills-path-swap.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=path, embedding=embedding, skills=[SLIDES]
    )
    artifact = {"path": str(path)}

    catalog = SkillCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact=artifact,
    )
    reload1 = catalog.replace_all([SLIDES])
    assert reload1.added == 1
    assert reload1.removed == 0
    await reload1

    await experimental_build_embedding_artifact(
        output=path, embedding=embedding, skills=[API_DESIGN]
    )
    reload2 = catalog.replace_all([API_DESIGN])
    assert reload2.added == 1
    assert reload2.removed == 1
    await reload2

    hits = await catalog.search_async("REST API design", 5)
    assert hits[0].skill_id == "api-design"


async def test_artifact_bytes_rejects_bytearray_and_memoryview() -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        await resolve_embedding_artifact({"bytes": bytearray(b"RAT1")})
    with pytest.raises(TypeError, match="must be bytes"):
        await resolve_embedding_artifact({"bytes": memoryview(b"RAT1")})  # type: ignore[typeddict-item]


async def test_resolve_embedding_artifact_presence_semantics_none(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.rat1"
    path.write_bytes(b"RAT1")
    valid_bytes = b"\x01\x02\x03"

    # explicit None behaves like “absent”
    resolved, on_miss = await resolve_embedding_artifact(
        {"path": str(path), "bytes": None}  # type: ignore[typeddict-item]
    )
    assert on_miss == "error"
    assert resolved == b"RAT1"

    resolved2, on_miss2 = await resolve_embedding_artifact(
        {"bytes": valid_bytes, "path": None}  # type: ignore[typeddict-item]
    )
    assert on_miss2 == "error"
    assert resolved2 == valid_bytes

    _, on_miss3 = await resolve_embedding_artifact(
        {"path": str(path), "on_miss": None}  # type: ignore[typeddict-item]
    )
    assert on_miss3 == "error"

    with pytest.raises(TypeError, match="requires exactly one of 'path' or 'bytes'"):
        await resolve_embedding_artifact(  # both explicit None
            {"path": None, "bytes": None}  # type: ignore[typeddict-item]
        )

    with pytest.raises(TypeError, match="requires exactly one of 'path' or 'bytes'"):
        await resolve_embedding_artifact(
            {"path": str(path), "bytes": valid_bytes}  # type: ignore[typeddict-item]
        )

    with pytest.raises(ValueError, match="unknown on-artifact-miss policy"):
        await resolve_embedding_artifact(
            {"path": str(path), "on_miss": "nope"}  # type: ignore[typeddict-item]
        )

    with pytest.raises(TypeError, match="unknown keys"):
        await resolve_embedding_artifact(
            {"path": str(path), "extra": 1}  # type: ignore[typeddict-item]
        )

    # type validation only applies to defined (non-None) values
    with pytest.raises(TypeError, match="path must be a str"):
        await resolve_embedding_artifact({"path": 1, "bytes": None})  # type: ignore[typeddict-item]

    with pytest.raises(TypeError, match="bytes must be bytes"):
        await resolve_embedding_artifact({"bytes": 123, "path": None})  # type: ignore[typeddict-item]


async def test_incompatible_merge_not_corrupt(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
) -> None:
    url, _state = counting_embedding_endpoint
    a = ToolRegistry({"url": url, "model": "model-a"}, method="bm25")
    await a.register(READ_FILE)
    bytes_a = await a.experimental_build_embedding_artifact()
    b = ToolRegistry({"url": url, "model": "model-b"}, method="bm25")
    await b.register(WRITE_FILE)
    bytes_b = await b.experimental_build_embedding_artifact()
    with pytest.raises(IncompatibleMergeError):
        merge_embedding_artifacts([bytes_a, bytes_b])


async def test_high_level_build_raises_incompatible_merge_on_model_drift(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, state = counting_embedding_endpoint
    state.set_response_models(["model-a", "model-b"])
    with pytest.raises(IncompatibleMergeError):
        await experimental_build_embedding_artifact(
            output=tmp_path / "drift.ratel-embeddings",
            embedding=_emb(url),
            tools=[READ_FILE],
            skills=[SLIDES],
        )


async def test_corrupt_artifact_is_warm_error(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    reg = ToolRegistry(embedding, method="bm25")
    await reg.register(READ_FILE)
    with pytest.raises(ArtifactWarmError) as caught:
        await reg.experimental_warm_embeddings_from_artifact(b"not-a-rat1-file", "error")
    assert caught.value.code == "Warm"
    assert type(caught.value).__name__ == "ArtifactWarmError"
    assert not hasattr(caught.value, "name")


async def test_public_exports_and_no_build_embeddings() -> None:
    import ratel_ai

    assert hasattr(ratel_ai, "experimental_build_embedding_artifact")
    assert hasattr(ratel_ai, "ArtifactWarmError")
    assert hasattr(ratel_ai, "ArtifactError")
    assert hasattr(ratel_ai, "IncompatibleMergeError")
    assert not hasattr(ratel_ai, "merge_embedding_artifacts")

    tools = ToolRegistry()
    skills = SkillRegistry()
    assert hasattr(tools, "experimental_build_embedding_artifact")
    assert hasattr(tools, "experimental_warm_embeddings_from_artifact")
    assert hasattr(skills, "experimental_build_embedding_artifact")
    assert hasattr(skills, "experimental_warm_embeddings_from_artifact")
    assert not hasattr(tools, "build_embedding_artifact")
    assert not hasattr(tools, "warm_embeddings_from_artifact")
    assert not hasattr(skills, "build_embedding_artifact")
    assert not hasattr(skills, "warm_embeddings_from_artifact")
    assert not hasattr(tools, "build_embeddings")
    assert not hasattr(skills, "build_embeddings")


def _blocked_resolve(
    artifact_bytes: bytes, entered: asyncio.Event, release: asyncio.Event
) -> Callable[[object], Awaitable[tuple[bytes, str]]]:
    """Stand in for path resolution: signal entered, then wait until release.

    Python ``{bytes}`` is not the racy configuration (no ``asyncio.to_thread``);
    tests still construct catalogs with ``{"path": ...}`` so they stay
    semantically path-backed.
    """

    async def resolve(_config: object) -> tuple[bytes, str]:
        entered.set()
        await release.wait()
        return artifact_bytes, "error"

    return resolve


async def test_mutation_rejected_while_artifact_warm_pending(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "tools-path.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE, WRITE_FILE]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        "ratel_ai.catalog.resolve_embedding_artifact",
        _blocked_resolve(output.read_bytes(), entered, release),
    )
    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    first = asyncio.ensure_future(catalog.register(_executable(READ_FILE)))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="registry busy"):
            catalog.register(_executable(WRITE_FILE))
    finally:
        release.set()
    await first
    assert catalog.has("read_file")
    assert not catalog.has("write_file")


async def test_skill_register_rejected_while_artifact_warm_pending(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "skills-path.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, skills=[SLIDES, API_DESIGN]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        "ratel_ai.skill_catalog.resolve_embedding_artifact",
        _blocked_resolve(output.read_bytes(), entered, release),
    )
    catalog = SkillCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    first = asyncio.ensure_future(catalog.register(SLIDES))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="registry busy"):
            catalog.register(API_DESIGN)
    finally:
        release.set()
    await first
    assert catalog.has("frontend-slides")
    assert not catalog.has("api-design")
    assert catalog.size() == 1


async def test_replace_all_rejected_while_artifact_warm_pending(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "skills-replace.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, skills=[SLIDES]
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    monkeypatch.setattr(
        "ratel_ai.skill_catalog.resolve_embedding_artifact",
        _blocked_resolve(output.read_bytes(), entered, release),
    )
    catalog = SkillCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    first = catalog.replace_all([SLIDES])
    assert (first.added, first.removed) == (1, 0)
    task = asyncio.ensure_future(first)
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="registry busy"):
            catalog.replace_all([API_DESIGN])
    finally:
        release.set()
    await task
    assert (first.added, first.removed) == (1, 0)
    assert catalog.has("frontend-slides")
    assert not catalog.has("api-design")
    assert catalog.size() == 1


async def test_artifact_read_failure_releases_pending(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    missing = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": "/nonexistent/ratel-artifact.rat1"},
    )
    with pytest.raises(OSError):
        await missing.register(_executable(READ_FILE))
    assert missing._registry._dense_pending == 0
    with pytest.raises(OSError):
        await missing.register(_executable(READ_FILE))
    assert missing._registry._dense_pending == 0

    output = tmp_path / "after-enoent.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE]
    )
    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    await catalog.register(_executable(READ_FILE))
    assert catalog.has("read_file")
    assert catalog._registry._dense_pending == 0


async def test_incomplete_warm_failure_releases_pending(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    output = tmp_path / "partial.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=output, embedding=embedding, tools=[READ_FILE]
    )
    catalog = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(output)},
    )
    with pytest.raises(ArtifactWarmError) as caught:
        await catalog.register([_executable(READ_FILE), _executable(WRITE_FILE)])
    assert caught.value.code == "Incomplete"
    assert catalog._registry._dense_pending == 0
    with pytest.raises(ArtifactWarmError) as second:
        await catalog.register(_executable(READ_FILE))
    assert "registry busy" not in str(second.value)
    assert catalog._registry._dense_pending == 0


async def test_sequential_artifact_register_and_replace_all(
    counting_embedding_endpoint: tuple[str, _CountingEndpoint], tmp_path: Path
) -> None:
    url, _state = counting_embedding_endpoint
    embedding = _emb(url)
    tools_out = tmp_path / "seq-tools.ratel-embeddings"
    skills_out = tmp_path / "seq-skills.ratel-embeddings"
    await experimental_build_embedding_artifact(
        output=tools_out, embedding=embedding, tools=[READ_FILE, WRITE_FILE]
    )
    await experimental_build_embedding_artifact(
        output=skills_out, embedding=embedding, skills=[SLIDES, API_DESIGN]
    )
    tools = ToolCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(tools_out)},
    )
    await tools.register(_executable(READ_FILE))
    await tools.register(_executable(WRITE_FILE))
    skills = SkillCatalog(
        method="semantic",
        embedding=embedding,
        experimental_embedding_artifact={"path": str(skills_out)},
    )
    first = skills.replace_all([SLIDES])
    await first
    second = skills.replace_all([SLIDES, API_DESIGN])
    assert second.added == 1
    await second
    assert skills.size() == 2
