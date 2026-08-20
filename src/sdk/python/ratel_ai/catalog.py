"""Tool catalog with executors — the Python mirror of `src/sdk/ts/src/catalog.ts`.

`ToolRegistry` is a typed facade over the private native index; `ToolCatalog`
layers executable handlers on top and emits the same trace events the TS SDK does
(see ADR-0007 for the core-owned schema).
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import threading
import time
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, TypedDict, TypeVar, Union, overload

from ._native import IntentGraph as IntentGraph  # re-exported for `ratel_ai.IntentGraph`
from ._native import NativeEventSubscription, SearchHit
from ._native import ToolRegistry as _NativeToolRegistry
from .embedding_artifact import (
    ExperimentalEmbeddingArtifact,
    OnArtifactMiss,
    resolve_embedding_artifact,
)
from .runtime_events import new_runtime_event_id
from .telemetry import (
    SEARCH_TARGET_TOOL,
    RuntimeEventProjection,
    record_catalog_definitions,
    trace_execute_tool,
    trace_search,
    trace_search_async,
)

Executor = Callable[[dict[str, Any]], Union[Awaitable[Any], Any]]
"""A tool handler: takes the tool's arguments dict, returns the result.

May be sync or async (tool inputs are heterogeneous across the catalog);
`ToolCatalog.invoke` absorbs the difference.
"""

SearchOrigin = str
"""Who initiated a search: ``"direct"`` (host code, the default), ``"agent"``
(a model calling a capability tool), or ``"baseline"`` (a query recorded while
Ratel was observing but not serving retrieval, so the invocations that follow
can be attributed to it). Labels the emitted trace event only — ranking is
unaffected.
"""

SearchMethod = str
"""Retrieval engine: ``"bm25"`` (lexical, model-free, the default),
``"semantic"`` (dense embeddings) or ``"hybrid"`` (both, fused).
"""

OriginFilterOption = Literal["any", "agent", "baseline"]
"""Which searches open an observation window when learning.

- ``"any"`` (the default) — every search in the stream.
- ``"baseline"`` — only turns recorded with ``experimental_baseline_turn``, so
  Ratel's own searches during a capture period do not become clusters.
- ``"agent"`` — only searches the model made through the capability tools, for
  rebuilding a graph from a period when Ratel was already serving.

``"direct"`` is a valid ``SearchOrigin`` but not a filter: learning only from
searches your own code made means learning from your plumbing.
"""

ProvenanceOption = Literal["live", "seeded"]
"""Whether what is learned is marked as coming from a seeding pass.

``"seeded"`` records it on each cluster's provenance count; ``"live"`` (the
default) does not. Never affects ranking.

These three are closed sets rather than plain ``str`` so a typo is caught by a
type checker instead of at runtime. The native still validates, for callers
without types.
"""


class _PrefixOptions(TypedDict, total=False):
    query_prefix: str
    doc_prefix: str


class _HuggingFaceOptions(_PrefixOptions, total=False):
    revision: str
    pooling: Literal["cls", "mean"]
    download: bool


class HuggingFaceEmbeddingConfig(_HuggingFaceOptions):
    """In-process HuggingFace embedding model configuration."""

    huggingface: str


class _LocalOptions(_PrefixOptions, total=False):
    pooling: Literal["cls", "mean"]


class LocalEmbeddingConfig(_LocalOptions):
    """In-process local-directory embedding model configuration."""

    local: str


class OllamaEmbeddingConfig(_PrefixOptions):
    """Local Ollama embedding endpoint configuration."""

    ollama: str


class _EndpointOptions(_PrefixOptions, total=False):
    api_key_env: str


class EndpointEmbeddingConfig(_EndpointOptions):
    """OpenAI-compatible embedding endpoint configuration."""

    url: str
    model: str


EmbeddingModelConfig = Union[
    HuggingFaceEmbeddingConfig,
    LocalEmbeddingConfig,
    OllamaEmbeddingConfig,
    EndpointEmbeddingConfig,
]
"""Mutually exclusive keyed embedding-source configurations."""

EmbeddingSpec = Union[str, EmbeddingModelConfig]
"""Embedding selection; a bare string is a local model directory path."""

_DenseResult = TypeVar("_DenseResult")
_REGISTRY_BUSY = "registry busy; await the active operation"
_UNAWAITED_REGISTER = (
    "a register() call was not awaited; dense preparation did not complete — "
    "`await catalog.register(...)` (or `registry.register(...)`) before a "
    "semantic/hybrid search"
)

_EMBEDDING_KEYS = frozenset(
    {
        "huggingface",
        "local",
        "ollama",
        "url",
        "model",
        "revision",
        "api_key_env",
        "query_prefix",
        "doc_prefix",
        "pooling",
        "download",
    }
)


def _embedding_kwargs(embedding: EmbeddingSpec) -> dict[str, Any]:
    """Normalize the public string|dict embedding form into native constructor kwargs.

    A string becomes the inferred ``spec``; a dict is passed through after a key
    check (the native layer validates the combination). Values are heterogeneous
    (``download`` is a bool), so the native constructor's typed params apply.
    """
    if isinstance(embedding, str):
        return {"spec": embedding}
    if isinstance(embedding, dict):
        if not embedding:
            raise ValueError("embedding config must not be empty")
        unknown = set(embedding) - _EMBEDDING_KEYS
        if unknown:
            raise ValueError(
                f"unknown embedding keys {sorted(unknown)}; allowed: {sorted(_EMBEDDING_KEYS)}"
            )
        return dict(embedding)
    raise TypeError("embedding must be a local-path string or a keyed config dict")


def _registry_embedding_kwargs(
    embedding: EmbeddingSpec | None,
    *,
    spec: str | None,
    huggingface: str | None,
    local: str | None,
    ollama: str | None,
    url: str | None,
    model: str | None,
    revision: str | None,
    api_key_env: str | None,
    query_prefix: str | None,
    doc_prefix: str | None,
    pooling: str | None,
    download: bool | None,
) -> dict[str, Any]:
    legacy = {
        key: value
        for key, value in {
            "spec": spec,
            "huggingface": huggingface,
            "local": local,
            "ollama": ollama,
            "url": url,
            "model": model,
            "revision": revision,
            "api_key_env": api_key_env,
            "query_prefix": query_prefix,
            "doc_prefix": doc_prefix,
            "pooling": pooling,
            "download": download,
        }.items()
        if value is not None
    }
    if embedding is not None:
        if legacy:
            raise TypeError("pass either embedding or legacy embedding kwargs, not both")
        return _embedding_kwargs(embedding)
    return legacy


@dataclass
class Tool:
    """Tool metadata: what the index ranks and the capability tools surface."""

    id: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    # Experimental (ADR-0021): retrieval-only description replacement.
    experimental_searchable_description: str | None = None


@dataclass
class ExecutableTool(Tool):
    """A `Tool` plus the handler that runs it. Registered into a `ToolCatalog`."""

    # No default so `execute` stays required; placed last after the inherited fields.
    execute: Executor = field(default=None)  # type: ignore[assignment]


@dataclass
class TraceSinkConfig:
    """Where the catalog's trace events go. Mirrors the TS `TraceSinkConfig` union.

    kind: "noop" | "memory" | "jsonl". `session_id` is required for memory/jsonl;
    `path` is required for jsonl.
    """

    kind: str
    session_id: str | None = None
    path: str | None = None


class AdaptiveRankingStatus(str):
    """The adaptive-ranking status, enriched with the models a pause involves.

    A plain ``str`` — ``== "active"`` and ``.startswith("paused")`` work as before
    — that also exposes, when the arm is paused by a model change, the fingerprint
    the graph was ``built`` with, the currently ``active`` model, and whether it is
    a ``dim_mismatch`` (different width) vs a same-width different model. All three
    are ``None`` unless the status is a ``paused: ...``. Mirrors the object the TS
    SDK returns, so the detail is reachable without parsing stderr.
    """

    built: str | None
    active: str | None
    dim_mismatch: bool | None

    def __new__(
        cls,
        status: str,
        built: str | None = None,
        active: str | None = None,
        dim_mismatch: bool | None = None,
    ) -> AdaptiveRankingStatus:
        """Build a status string carrying the model detail behind a pause."""
        self = super().__new__(cls, status)
        self.built = built
        self.active = active
        self.dim_mismatch = dim_mismatch
        return self


class ToolRegistry:
    """Typed Python facade over the private native tool registry."""

    @overload
    def __init__(
        self,
        embedding: EmbeddingSpec | None = None,
        *,
        method: SearchMethod = "bm25",
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        spec: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        pooling: Literal["cls", "mean"] | None = None,
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        huggingface: str,
        revision: str | None = None,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        pooling: Literal["cls", "mean"] | None = None,
        download: bool | None = None,
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        local: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        pooling: Literal["cls", "mean"] | None = None,
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        ollama: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        url: str,
        model: str,
        api_key_env: str | None = None,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None: ...

    def __init__(
        self,
        embedding: EmbeddingSpec | None = None,
        *,
        method: SearchMethod = "bm25",
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
        spec: str | None = None,
        huggingface: str | None = None,
        local: str | None = None,
        ollama: str | None = None,
        url: str | None = None,
        model: str | None = None,
        revision: str | None = None,
        api_key_env: str | None = None,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        pooling: str | None = None,
        download: bool | None = None,
    ) -> None:
        """Create a metadata registry with an optional embedding model.

        A "semantic"/"hybrid" `method` makes `register` prepare the dense cache
        eagerly (embed on a worker thread); "bm25" keeps registration model-free
        unless ``experimental_embedding_artifact`` is set (then the artifact is
        warmed on register for any method).
        """
        kwargs = _registry_embedding_kwargs(
            embedding,
            spec=spec,
            huggingface=huggingface,
            local=local,
            ollama=ollama,
            url=url,
            model=model,
            revision=revision,
            api_key_env=api_key_env,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
            pooling=pooling,
            download=download,
        )
        self._native = _NativeToolRegistry(**kwargs)
        self._eager = method in ("semantic", "hybrid")
        self._embedding_artifact = experimental_embedding_artifact
        self._warn_on_model_mismatch = True
        self._adaptive_warned = False
        self._rebuild_on_model_change = False
        self._dense_gate = threading.Lock()
        self._dense_state = threading.Lock()
        self._dense_pending = 0
        # Builds scheduled by `register` but not yet driven to completion. Bumped
        # synchronously so a *forgotten* `await register(...)` leaves it > 0; the
        # dense search path reads it to fail loudly instead of ranking an empty
        # corpus. Touched only on the event-loop thread (register / `_drive` /
        # search_async), so it needs no lock.
        self._undriven_builds = 0
        self._dense_tasks: set[asyncio.Task[Any]] = set()
        self._emitted_definition_hashes: dict[str, str] = {}

    @overload
    def register(self, item: Tool) -> Awaitable[None]: ...

    @overload
    def register(self, item: Iterable[Tool]) -> Awaitable[None]: ...

    @overload
    def register(
        self,
        item: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> Awaitable[None]: ...

    def register(
        self,
        item: Tool | Iterable[Tool] | str,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> Awaitable[None]:
        """Register one `Tool`, many `Tool`s, or a flat (id, name, …) tuple.

        Metadata is indexed **synchronously**, the instant `register(...)` is
        called, so a forgotten `await` can never silently drop the corpus. The
        returned awaitable drives only dense preparation — on a
        "semantic"/"hybrid" registry it embeds in one batched, off-thread pass
        (errors surface when awaited); with ``experimental_embedding_artifact``
        it warms that artifact first (any method); plain "bm25" without an
        artifact is a no-op. Always `await` the result.
        """
        flat_args = (name, description, input_schema, output_schema)
        if isinstance(item, Tool):
            if any(value is not None for value in flat_args):
                raise TypeError("item register accepts only the Tool argument")
            tools: list[Tool] = [item]
        elif isinstance(item, str):
            if any(value is None for value in flat_args):
                raise TypeError("flat register requires all metadata arguments")
            tools = [Tool(item, name, description, input_schema, output_schema)]  # type: ignore[arg-type]
        else:
            if any(value is not None for value in flat_args):
                raise TypeError("iterable register accepts only the items argument")
            tools = list(item)
            if not all(isinstance(tool, Tool) for tool in tools):
                raise TypeError("register requires Tool items")
        self._register_items(tools)
        return self._build_tracked(bool(tools))

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Run synchronous, model-free BM25 retrieval."""
        return self._native.search(query, top_k)

    def search_with_origin(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin,
        projection: RuntimeEventProjection | None = None,
    ) -> list[SearchHit]:
        """Run BM25 retrieval with an explicit trace origin."""
        return self._native.search_with_origin(query, top_k, origin, projection)

    def search_with_method(
        self, query: str, top_k: int, origin: SearchOrigin, method: SearchMethod
    ) -> list[SearchHit]:
        """Run BM25 synchronously; dense retrieval is async-only."""
        if method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {method}")
        if method != "bm25":
            raise RuntimeError(
                f"{method} search is asynchronous; use `await registry.search_async(..., "
                f'method="{method}")`'
            )
        return self.search_with_origin(query, top_k, origin)

    async def experimental_build_embedding_artifact(self) -> bytes:
        """Build a RAT1 artifact from the registered corpus (ADR-0018).

        Off the event loop and mutation-blocking via ``_dense_pending``, but does
        **not** take ``_dense_gate`` — semantic search may run concurrently.
        Cancelling the await does not clear pending until the native build finishes.

        Raises:
            EmbedderError: Embedding failed during artifact build.
            ArtifactError: Non-embedder artifact encode failure from native build.
        """
        self._queue_dense()
        runner = self._run_artifact_build_task()
        try:
            task = asyncio.create_task(runner)
        except BaseException:
            runner.close()
            self._finish_dense()
            raise
        self._dense_tasks.add(task)
        task.add_done_callback(self._dense_task_done)
        await asyncio.wait({task})
        return task.result()

    async def _run_artifact_build_task(self) -> bytes:
        try:
            return await asyncio.to_thread(self._native._build_embedding_artifact)
        finally:
            self._finish_dense()

    async def experimental_warm_embeddings_from_artifact(
        self, artifact: bytes, on_miss: OnArtifactMiss = "error"
    ) -> None:
        """Warm the dense cache from artifact bytes (serialized via ``_run_dense``)."""
        await self._run_dense(
            lambda: self._native._warm_embeddings_from_artifact(artifact, on_miss)
        )

    async def _ensure_dense_ready(self) -> None:
        if self._embedding_artifact is not None:
            self._queue_dense()
            try:
                artifact_bytes, on_miss = await resolve_embedding_artifact(self._embedding_artifact)
                await self.experimental_warm_embeddings_from_artifact(artifact_bytes, on_miss)
            finally:
                self._finish_dense()
            return
        if self._eager:
            await self._build()

    async def _build(self) -> None:
        """Embed not-yet-embedded items on a worker thread (used by `register`)."""
        await self._run_dense(self._native._build_embeddings)

    async def _rebuild(self) -> None:
        """Recompute and atomically replace the full embedding cache (internal)."""
        await self._run_dense(self._native._rebuild_embeddings)

    def _build_tracked(self, has_items: bool) -> Awaitable[None]:
        """Return the awaitable `register` hands back — it drives dense preparation.

        `_undriven_builds` is bumped **now** (synchronously, while `register`
        runs), not inside the coroutine, so it stays > 0 when the coroutine is
        never driven — that is how a forgotten `await` becomes detectable.
        """
        schedule = self._embedding_artifact is not None or (self._eager and has_items)
        if schedule:
            self._undriven_builds += 1

        async def _drive() -> None:
            if not schedule:
                return
            try:
                await self._ensure_dense_ready()
            finally:
                self._undriven_builds -= 1

        return _drive()

    async def search_async(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod = "bm25",
        projection: RuntimeEventProjection | None = None,
    ) -> list[SearchHit]:
        """Search immediately with BM25 or run dense retrieval on a worker thread."""
        if method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {method}")
        if method == "bm25":
            return self.search_with_origin(query, top_k, origin, projection)
        if self._undriven_builds > 0:
            raise RuntimeError(_UNAWAITED_REGISTER)
        await self._maybe_rebuild_on_model_change()
        return await self._run_dense(
            lambda: self._native._search_with_method(query, top_k, origin, method, projection)
        )

    def record_event(
        self,
        event: dict[str, Any],
        projection: RuntimeEventProjection | None = None,
    ) -> None:
        """Record an SDK-layer trace event."""
        if projection is None:
            self._native.record_event(event)
        else:
            self._native.record_event_with_context(event, projection)

    def subscribe_events(
        self,
        handler: Callable[[list[dict[str, Any]]], object],
        *,
        session_id: str,
        source_id: str,
        queue_capacity: int = 1_024,
        batch_size: int = 64,
    ) -> NativeEventSubscription:
        """Attach one public batched runtime-event subscriber."""
        with self._dense_state:
            self._raise_if_busy()
            return self._native.subscribe_trace_events(
                handler,
                session_id,
                source_id,
                queue_capacity,
                batch_size,
            )

    def set_trace_sink(
        self, kind: str, session_id: str | None = None, path: str | None = None
    ) -> None:
        """Replace the native trace sink."""
        with self._dense_state:
            self._raise_if_busy()
            self._native.set_trace_sink(kind, session_id, path)

    def experimental_enable_catalog_definitions(self) -> None:
        """Enable experimental complete catalog-definition events."""
        with self._dense_state:
            self._raise_if_busy()
            self._native.experimental_enable_catalog_definitions()

    def experimental_enable_adaptive_ranking(
        self,
        graph: IntentGraph,
        *,
        warn_on_model_mismatch: bool = True,
        rebuild_on_model_change: bool = False,
        origins: OriginFilterOption | None = None,
        provenance: ProvenanceOption | None = None,
    ) -> None:
        """Turn on adaptive usage ranking against ``graph`` (ADR-0014).

        Wires both halves: this registry ranks against what users have actually
        invoked after similar queries, and keeps learning as it is used. Pass
        the same :class:`IntentGraph` to the other registry so both learn into one
        set of clusters.

        Only queries matching a cluster are affected. With a graph attached the
        hit ``score`` becomes a fusion score rather than a raw BM25 score, so
        use ``rank`` for ordering and ``fused`` to detect the scale, not the
        raw ``score``.

        On a model change the arm pauses and a one-time warning is issued unless
        ``warn_on_model_mismatch`` is False; call :meth:`experimental_rebuild_intent_graph`.

        Set ``rebuild_on_model_change`` to recover automatically instead: the
        next dense (semantic/hybrid) search re-embeds the graph under the current
        model before searching, so a persisted graph self-heals after a model
        swap. It is off by default because the rebuild is an embedding pass —
        expensive, able to raise :class:`EmbedderError`, and it mutates the graph
        (new centroids, bumped ``rev``). Recovery is lazy: status stays
        ``paused`` until that first dense search.
        """
        # `experimental_enable_adaptive_ranking` takes `&mut self` natively, so it must not run
        # while an in-flight dense build holds the registry — guard it like
        # `set_trace_sink`, surfacing the typed busy error rather than a raw
        # pyo3 "Already borrowed".
        with self._dense_state:
            self._raise_if_busy()
            self._warn_on_model_mismatch = warn_on_model_mismatch
            self._rebuild_on_model_change = rebuild_on_model_change
            self._adaptive_warned = False
            self._native.enable_adaptive_ranking(graph, origins, provenance)
        self._maybe_warn_model_mismatch()

    def experimental_disable_adaptive_ranking(self) -> None:
        """Turn adaptive usage ranking off; the graph keeps what it learned."""
        with self._dense_state:
            self._raise_if_busy()
            self._rebuild_on_model_change = False
            self._native.disable_adaptive_ranking()

    async def _maybe_rebuild_on_model_change(self) -> None:
        """Auto-recover a model-mismatched graph before a dense search, opt-in.

        A no-op unless ``rebuild_on_model_change`` was set and the arm is paused.
        Re-checks each dense search: once rebuilt the status reads ``active`` and
        this stops rebuilding, so a model that is briefly unavailable heals on a
        later search rather than staying paused forever.
        """
        if not self._rebuild_on_model_change:
            return
        status, _built, _active, _dim = self._native.adaptive_ranking_status()
        if status.startswith("paused"):
            await self.experimental_rebuild_intent_graph()

    async def experimental_rebuild_intent_graph(self) -> None:
        """Re-embed the graph's members under the current model; preserves learning.

        Call after changing the embedding model: a graph's centroids are only
        comparable to queries from the model that built them, so on a swap the
        usage arm pauses until this runs.
        """
        await self._run_dense(self._native._rebuild_intent_graph)
        self._adaptive_warned = False
        self._maybe_warn_model_mismatch()

    async def experimental_build_intent_graph(
        self,
        jsonl: str,
        *,
        origins: OriginFilterOption | None = None,
        provenance: ProvenanceOption | None = None,
    ) -> IntentGraph:
        """Build an IntentGraph from a JSONL trace log — offline baseline seeding.

        Every distinct query is embedded up front, so clusters form at the dense
        tier exactly as the live path would grow them. A model-free replay would
        cluster on word overlap instead, and ``experimental_rebuild_intent_graph``
        cannot repair that later: it replaces centroids without revisiting
        cluster boundaries.

        The returned graph is **detached** — pass it to
        ``experimental_enable_adaptive_ranking`` once you decide it is ready. One
        call covers both catalogs, so do not run it again on the skill registry.

        Args:
            jsonl: the trace log, exactly as the ``jsonl`` sink writes it. Blank
                lines are skipped; a malformed line raises, naming its line number.
            origins: which searches count. Defaults to ``baseline`` here, since
                building from a log is seeding; pass ``agent`` to re-derive a
                graph from a period when Ratel was serving, or ``any`` for a log
                you know holds only one kind.
            provenance: defaults to ``seeded`` here, since building from a log is
                a seeding pass. Pass ``live`` when re-deriving a graph that was
                grown from live traffic.

        Raises:
            ValueError: an unknown policy value, or a malformed log line.
            EmbedderError: the queries could not be embedded.
        """
        json = await self._run_dense(
            lambda: self._native._build_intent_graph(jsonl, origins, provenance)
        )
        return IntentGraph.from_json(json)

    @property
    def experimental_adaptive_ranking_status(self) -> AdaptiveRankingStatus:
        """Adaptive-ranking status; a str that also carries a pause's model detail."""
        status, built, active, dim_mismatch = self._native.adaptive_ranking_status()
        return AdaptiveRankingStatus(status, built, active, dim_mismatch)

    def _maybe_warn_model_mismatch(self) -> None:
        if self._adaptive_warned or not self._warn_on_model_mismatch:
            return
        status, built, active, dim_mismatch = self._native.adaptive_ranking_status()
        if not status.startswith("paused"):
            return
        self._adaptive_warned = True
        how = (
            f"built with a {built}-dim embedding model but the active model outputs {active} dims"
            if dim_mismatch
            else f"built with embedding model '{built}' but the active model is '{active}'"
        )
        warnings.warn(
            f"ratel: intent graph was {how}. Adaptive usage ranking is PAUSED — "
            "call experimental_rebuild_intent_graph() to rebuild it with the current model.",
            stacklevel=2,
        )

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured native trace events."""
        return self._native.drain_trace_events()

    async def _run_dense(self, operation: Callable[[], _DenseResult]) -> _DenseResult:
        self._queue_dense()
        runner = self._run_dense_task(operation)
        try:
            task = asyncio.create_task(runner)
        except BaseException:
            runner.close()
            self._finish_dense()
            raise
        self._dense_tasks.add(task)
        task.add_done_callback(self._dense_task_done)
        # Wait for the worker WITHOUT asyncio.shield. Like shield, `asyncio.wait`
        # never cancels the awaited task, so a cancelled caller leaves the worker
        # running (it holds the dense gate and must finish) — but it avoids
        # shield's Python-3.14 `_log_on_exception` callback, which would
        # unconditionally re-report the inner exception even after
        # `_dense_task_done` has already consumed it.
        await asyncio.wait({task})
        return task.result()

    async def _run_dense_task(self, operation: Callable[[], _DenseResult]) -> _DenseResult:
        try:
            return await asyncio.to_thread(self._run_dense_worker, operation)
        finally:
            # Also runs when the default executor rejects submission, before a
            # worker exists to clear the queued-operation state.
            self._finish_dense()

    def _run_dense_worker(self, operation: Callable[[], _DenseResult]) -> _DenseResult:
        with self._dense_gate:
            return operation()

    def _dense_task_done(self, task: asyncio.Task[Any]) -> None:
        self._dense_tasks.discard(task)
        if not task.cancelled():
            # A shielded worker outlives a cancelled caller. Retrieve any later
            # failure so asyncio does not report an unhandled task exception.
            task.exception()

    def _queue_dense(self) -> None:
        with self._dense_state:
            self._dense_pending += 1

    def _finish_dense(self) -> None:
        with self._dense_state:
            self._dense_pending -= 1

    def _register_items(self, tools: Iterable[Tool]) -> None:
        tools = list(tools)
        with self._dense_state:
            self._raise_if_busy()
            self._native._register_many(
                [
                    (
                        tool.id,
                        tool.name,
                        tool.description,
                        tool.experimental_searchable_description,
                        tool.input_schema,
                        tool.output_schema,
                    )
                    for tool in tools
                ]
            )
            record_catalog_definitions("tool", tools, self._emitted_definition_hashes)

    def _raise_if_busy(self) -> None:
        if self._dense_pending:
            raise RuntimeError(_REGISTRY_BUSY)


class BaselineTurn:
    """One baseline turn being assembled — the query, plus what the agent chose.

    Created by :meth:`ToolCatalog.experimental_baseline_turn`, not directly.

    Buffered: nothing reaches the trace log until :meth:`record`, so a turn that
    fails your quality gate can simply be dropped. Recording twice raises, as
    does adding to a turn already recorded — both are the same mistake, evidence
    counted more than once.
    """

    def __init__(self, catalog: ToolCatalog, query: str) -> None:
        """Open a turn for ``query``; use `ToolCatalog.experimental_baseline_turn`."""
        self._catalog = catalog
        self._recorded = False
        self._events: list[dict[str, Any]] = [
            {
                "type": "search",
                "query": query,
                "origin": "baseline",
                "top_k": 0,
                "hits": [],
                "stages": [],
                "took_ms": 0,
            }
        ]

    def _still_open(self) -> None:
        if self._recorded:
            raise RuntimeError("this baseline turn was already recorded")

    def invoked(self, tool_id: str) -> BaselineTurn:
        """Attribute a tool invocation to this turn. Chainable."""
        self._still_open()
        self._events.append({"type": "invoke_start", "tool_id": tool_id, "args_size_bytes": 0})
        return self

    def invoked_skill(self, skill_id: str) -> BaselineTurn:
        """Attribute a skill load to this turn. Chainable."""
        self._still_open()
        self._events.append({"type": "skill_invoke", "skill_id": skill_id, "took_ms": 0})
        return self

    def record(self) -> None:
        """Write the turn to the trace log. Raises if called twice."""
        self._still_open()
        self._recorded = True
        for event in self._events:
            self._catalog.record_event(event)

    def __enter__(self) -> BaselineTurn:
        """Enter the turn; the block names what the agent invoked."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Record the turn on a clean exit; discard it if the block raised."""
        # A raising turn is exactly the turn you would not want the graph to
        # learn from, so the block failing discards it rather than recording a
        # half-finished one.
        if exc_type is None:
            self.record()


class ToolCatalog:
    """Registry + executors. Register tools once, then search and invoke by id."""

    def __init__(
        self,
        trace: TraceSinkConfig | None = None,
        method: SearchMethod = "bm25",
        embedding: EmbeddingSpec | None = None,
        experimental_embedding_artifact: ExperimentalEmbeddingArtifact | None = None,
    ) -> None:
        """Create an empty catalog.

        Args:
            trace: where trace events go; `None` keeps the default no-op sink.
            method: default retrieval method for `search` — "bm25" (the
                historical, model-free behavior), "semantic" or "hybrid". A
                per-call `method=` overrides it. Dense ranking uses
                `search_async`. BM25 search is model-free, but an explicitly
                configured embedding artifact is still warmed during registration.
            embedding: model for semantic/hybrid retrieval (a path string or a
                keyed dict — see `EmbeddingSpec`). Retained and validated even
                under "bm25" so a later async semantic override can use it.
            experimental_embedding_artifact: build-time RAT1 to warm on register
                (any method; default ``on_miss`` is ``"error"``). Each
                ``register`` re-resolves and re-warms over the whole current
                corpus — intended for one batch at startup; incremental
                register calls repeat I/O and id+hash matching. With the
                default ``on_miss="error"``, warm fails when the catalog's
                current corpus includes one or more ids missing from the
                artifact. If you configure the same artifact for both this
                ToolCatalog and a SkillCatalog, it must cover every
                non-empty corpus that actually registers; a tool-only artifact
                is valid when the Skill corpus stays empty. The normal remedy
                is a mixed artifact built via
                ``experimental_build_embedding_artifact``; the runtime remedy
                for uncovered current-kind entries is ``on_miss="embed"``.
        """
        self._executors: dict[str, Executor] = {}
        self._tools: dict[str, Tool] = {}
        self._method: SearchMethod = method
        self._registry = ToolRegistry(
            embedding,
            method=method,
            experimental_embedding_artifact=experimental_embedding_artifact,
        )
        if trace is not None:
            self._registry.set_trace_sink(trace.kind, trace.session_id, trace.path)

    def register(self, tools: ExecutableTool | Iterable[ExecutableTool]) -> Awaitable[None]:
        """Register one tool or many — the single entry point for both.

        Metadata and the executor handler are stored **synchronously**, the
        instant `register(...)` is called, so a forgotten `await` can never
        silently drop the corpus. The returned awaitable drives dense
        preparation: on a "semantic"/"hybrid" catalog without an artifact it
        embeds the batch on a worker thread; with
        ``experimental_embedding_artifact`` it warms that artifact first (any
        method). Dense-preparation errors surface when awaited; metadata still
        persists if that phase fails. A BM25 catalog without an artifact never
        loads a model. **Always `await` the result** — a semantic/hybrid search
        after an un-awaited `register` raises rather than ranking an empty
        corpus. Re-registering an id replaces it in place; the index never holds
        a duplicate.

        A model or dimension change is not recovered in place — construct a new
        catalog and re-register.

        Args:
            tools: a single `ExecutableTool` or an iterable of them; each
                `execute` must be set. Pass the whole batch at once for a single
                dense-preparation request; separate `register` calls prepare
                separately.

        Raises:
            ValueError: if any `execute` is `None`, or a schema isn't JSON-serializable.
            EmbedderError: when embedding fails (when awaited).
            ArtifactWarmError: when a configured artifact fails (when awaited).
            RuntimeError: if a dense operation already owns the registry.
        """
        batch = [tools] if isinstance(tools, ExecutableTool) else list(tools)
        for tool in batch:
            if tool.execute is None:
                raise ValueError(f"tool {tool.id!r} has no execute handler")
        self._registry._register_items(batch)
        for tool in batch:
            self._executors[tool.id] = tool.execute
            self._tools[tool.id] = Tool(
                id=tool.id,
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
                output_schema=tool.output_schema,
            )
        return self._registry._build_tracked(bool(batch))

    def search(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod | None = None,
    ) -> list[SearchHit]:
        """Rank registered tools synchronously with BM25.

        Args:
            query: what the caller wants to do.
            top_k: max hits to return.
            origin: who initiated the search — labels the trace event only.
            method: per-call override of the catalog's default retrieval
                method ("bm25" | "semantic" | "hybrid").

        Returns:
            Up to `top_k` `SearchHit`s, best first.

        Raises:
            ValueError: if `method` is not "bm25", "semantic" or "hybrid".
            RuntimeError: if the resolved method is semantic/hybrid; use
                `search_async` for dense retrieval.
        """
        resolved_method = method or self._method
        if resolved_method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {resolved_method}")
        if resolved_method != "bm25":
            raise RuntimeError(
                f"{resolved_method} search is asynchronous; use `await catalog.search_async(..., "
                f'method="{resolved_method}")`'
            )
        return trace_search(
            SEARCH_TARGET_TOOL,
            query,
            top_k,
            origin,
            lambda projection: self._registry.search_with_origin(query, top_k, origin, projection),
        )

    async def search_async(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod | None = None,
    ) -> list[SearchHit]:
        """Rank tools asynchronously with BM25, semantic, or hybrid retrieval.

        Dense methods require the corpus to have been embedded by `register` on a
        semantic/hybrid catalog; searching never embeds missing corpus vectors.
        """
        resolved_method = method or self._method
        return await trace_search_async(
            SEARCH_TARGET_TOOL,
            query,
            top_k,
            origin,
            lambda projection: self._registry.search_async(
                query, top_k, origin, resolved_method, projection
            ),
        )

    def has(self, tool_id: str) -> bool:
        """Return whether a tool with this id is registered."""
        return tool_id in self._executors

    def get(self, tool_id: str) -> Tool | None:
        """Return the metadata-only `Tool` for an id, or `None` if unknown."""
        return self._tools.get(tool_id)

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the complete deterministic executor-free tool definitions."""
        return [
            {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "input_schema": copy.deepcopy(tool.input_schema),
                "output_schema": copy.deepcopy(tool.output_schema),
            }
            for tool in sorted(self._tools.values(), key=lambda item: item.id)
        ]

    def get_executable(self, tool_id: str) -> ExecutableTool | None:
        """Return the `ExecutableTool` (metadata plus handler) for an id, or `None`."""
        tool = self._tools.get(tool_id)
        execute = self._executors.get(tool_id)
        if tool is None or execute is None:
            return None
        return ExecutableTool(
            id=tool.id,
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            output_schema=tool.output_schema,
            execute=execute,
        )

    def record_event(
        self,
        event: dict[str, Any],
        projection: RuntimeEventProjection | None = None,
    ) -> None:
        """Record a trace event into the catalog's sink.

        Args:
            event: a dict matching one of the core-owned `TraceEvent` shapes
                (ADR-0007), e.g. `{"type": "gateway_search", ...}`.
            projection: optional event identity and OTel correlation shared with the envelope.

        Raises:
            ValueError: if the dict doesn't match any known event shape.
        """
        self._registry.record_event(event, projection)

    def subscribe_events(
        self,
        handler: Callable[[list[dict[str, Any]]], object],
        *,
        session_id: str,
        source_id: str,
        queue_capacity: int = 1_024,
        batch_size: int = 64,
    ) -> NativeEventSubscription:
        """Attach one public runtime-event subscriber."""
        return self._registry.subscribe_events(
            handler,
            session_id=session_id,
            source_id=source_id,
            queue_capacity=queue_capacity,
            batch_size=batch_size,
        )

    def experimental_enable_catalog_definitions(self) -> None:
        """Enable experimental complete catalog-definition events."""
        self._registry.experimental_enable_catalog_definitions()

    def experimental_enable_adaptive_ranking(
        self,
        graph: IntentGraph,
        *,
        warn_on_model_mismatch: bool = True,
        rebuild_on_model_change: bool = False,
        origins: OriginFilterOption | None = None,
        provenance: ProvenanceOption | None = None,
    ) -> None:
        """Turn on adaptive usage ranking against ``graph`` (ADR-0014).

        Wires both halves: this catalog ranks against what users have actually
        invoked after similar queries, and keeps learning as it is used. Pass
        the same :class:`IntentGraph` to the other catalog so both learn into one
        set of clusters.

        Only queries matching a cluster are affected. With a graph attached the
        hit ``score`` becomes a fusion score rather than a raw BM25 score, so
        use ``rank`` for ordering and ``fused`` to detect the scale, not the
        raw ``score``.

        Set ``rebuild_on_model_change`` to auto-recover a model-mismatched graph
        on the next dense search rather than staying paused until you call
        :meth:`experimental_rebuild_intent_graph` yourself. Off by default — the rebuild is an
        embedding pass (cost, possible :class:`EmbedderError`, mutates the graph).
        """
        self._registry.experimental_enable_adaptive_ranking(
            graph,
            warn_on_model_mismatch=warn_on_model_mismatch,
            rebuild_on_model_change=rebuild_on_model_change,
            origins=origins,
            provenance=provenance,
        )

    async def experimental_rebuild_intent_graph(self) -> None:
        """Re-embed the graph's members under the current model; preserves learning."""
        await self._registry.experimental_rebuild_intent_graph()

    def experimental_baseline_turn(self, query: str) -> BaselineTurn:
        """Begin recording a turn observed while Ratel is *not* serving retrieval.

        Name the turn's query, then name what the agent chose after it::

            catalog.experimental_baseline_turn("why is the build broken").invoked(
                "gh_run_list"
            ).record()

        Nothing reaches the trace log until :meth:`BaselineTurn.record`, so the
        turn is also where your own quality gate goes — a turn you would not
        want the graph to learn from is simply never recorded. Also usable as a
        context manager, which records on a clean exit and discards on an
        exception.

        Sugar over :meth:`record_event`: it writes one ``search`` event with
        origin ``"baseline"`` followed by one event per invocation, which is the
        adjacency the graph builder pairs on.
        """
        return BaselineTurn(self, query)

    def experimental_record_baseline_turn(
        self,
        query: str,
        invoked: Sequence[str] | None = None,
        invoked_skills: Sequence[str] | None = None,
    ) -> None:
        """Record a complete baseline turn in one call.

        The same evidence :meth:`experimental_baseline_turn` collects, for hosts
        that cannot hold a turn open while it happens::

            catalog.experimental_record_baseline_turn(
                query="why is the build broken", invoked=["gh_run_list"]
            )

        Use this when the query and the invocations arrive separately — a
        process-per-request server, where the search and the invocation that
        follows are different requests on possibly different machines.
        Reassemble the turn from your own storage, then hand it over whole.

        One turn stays one observation. Splitting a search with three
        invocations into three recorded turns counts the query three times,
        which inflates the support that scales the boost and gates the flip.

        Nothing is buffered, so the quality gate is simply whether you call it:
        a turn you would not want the graph to learn from is never recorded.
        """
        # Deliberately not shared with `experimental_baseline_turn`: that path
        # emits invocations in call order, which separate `invoked` and
        # `invoked_skills` sequences cannot express for a turn mixing the two.
        # `test_recording_a_whole_turn_matches_the_chained_builder` holds the
        # two in parity instead.
        self.record_event(
            {
                "type": "search",
                "query": query,
                "origin": "baseline",
                "top_k": 0,
                "hits": [],
                "stages": [],
                "took_ms": 0,
            }
        )
        for tool_id in invoked or ():
            self.record_event({"type": "invoke_start", "tool_id": tool_id, "args_size_bytes": 0})
        for skill_id in invoked_skills or ():
            self.record_event({"type": "skill_invoke", "skill_id": skill_id, "took_ms": 0})

    async def experimental_build_intent_graph(
        self,
        jsonl: str,
        *,
        origins: OriginFilterOption | None = None,
        provenance: ProvenanceOption | None = None,
    ) -> IntentGraph:
        """Build an IntentGraph from a JSONL trace log; returns a detached graph.

        See `ToolRegistry.experimental_build_intent_graph`. One call covers
        both the tool and skill catalogs.
        """
        return await self._registry.experimental_build_intent_graph(
            jsonl, origins=origins, provenance=provenance
        )

    @property
    def experimental_adaptive_ranking_status(self) -> AdaptiveRankingStatus:
        """Adaptive-ranking status: active, inactive, unknown, or paused."""
        return self._registry.experimental_adaptive_ranking_status

    def experimental_disable_adaptive_ranking(self) -> None:
        """Turn adaptive usage ranking off; the graph keeps what it learned."""
        self._registry.experimental_disable_adaptive_ranking()

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured trace envelopes; `[]` unless the sink is "memory"."""
        return self._registry.drain_trace_events()

    async def invoke(self, tool_id: str, args: dict[str, Any]) -> Any:
        """Run a registered tool's handler and return its result.

        This is the canonical place that absorbs the sync/async executor
        difference: the handler is called first and the result awaited only if
        it is awaitable, so plain functions and `async def` executors (e.g.
        MCP/HTTP tools) are both supported. Callers must route invocations
        here rather than re-deriving that logic. Emits `invoke_start` /
        `invoke_end` / `invoke_error` trace events and wraps the call in an
        `execute_tool` OTel span (ADR-0007).

        Args:
            tool_id: id of a registered tool.
            args: the arguments dict passed to the handler.

        Returns:
            Whatever the handler returns (awaited if it returned an awaitable).

        Raises:
            ValueError: if `tool_id` is not registered.
            asyncio.CancelledError: after recording a cancelled `invoke_error`.
            Exception: whatever the handler raises, re-raised after an
                `invoke_error` trace event is recorded.
        """
        fn = self._executors.get(tool_id)
        if fn is None:
            raise ValueError(f"unknown toolId: {tool_id}")

        async def _run(projection: RuntimeEventProjection) -> Any:
            self._registry.record_event(
                {
                    "type": "invoke_start",
                    "tool_id": tool_id,
                    "args_size_bytes": _args_size_bytes(args),
                },
                projection,
            )
            started = time.monotonic()
            try:
                # Call first, await only if awaitable (see the `invoke` docstring).
                # Never bare-`await fn(args)`: in Python that raises on a sync result.
                result = fn(args)
                if inspect.isawaitable(result):
                    result = await result
                terminal_projection = projection.copy()
                terminal_projection["event_id"] = new_runtime_event_id()
                self._registry.record_event(
                    {
                        "type": "invoke_end",
                        "tool_id": tool_id,
                        "took_ms": _elapsed_ms(started),
                    },
                    terminal_projection,
                )
                return result
            except asyncio.CancelledError as err:
                terminal_projection = projection.copy()
                terminal_projection["event_id"] = new_runtime_event_id()
                self._registry.record_event(
                    {
                        "type": "invoke_error",
                        "tool_id": tool_id,
                        "took_ms": _elapsed_ms(started),
                        "error": _error_message(err),
                    },
                    terminal_projection,
                )
                raise
            except Exception as err:
                terminal_projection = projection.copy()
                terminal_projection["event_id"] = new_runtime_event_id()
                self._registry.record_event(
                    {
                        "type": "invoke_error",
                        "tool_id": tool_id,
                        "took_ms": _elapsed_ms(started),
                        "error": _error_message(err),
                    },
                    terminal_projection,
                )
                raise

        # The `execute_tool` OTel span wraps the local trace stream; both record the
        # same invocation, on their two independent channels (ADR-0007).
        return await trace_execute_tool(tool_id, args, _run)


def _args_size_bytes(args: Any) -> int:
    try:
        return len(json.dumps(args))
    except Exception:
        return 0


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _error_message(err: BaseException) -> str:
    return str(err) or err.__class__.__name__
