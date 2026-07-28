"""Skill catalog — the Python mirror of `src/sdk/ts/src/skill-catalog.ts`.

`SkillRegistry` is a typed facade over the private native index; `SkillCatalog`
is the on-demand analogue of `ToolCatalog`: registered skills are ranked by
relevance, and the matching body is fetched only on `invoke`.
"""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

from ._native import IntentGraph as IntentGraph  # re-exported for `ratel_ai.IntentGraph`
from ._native import SkillHit
from ._native import SkillRegistry as _NativeSkillRegistry
from .catalog import (
    _REGISTRY_BUSY,
    _UNAWAITED_REGISTER,
    AdaptiveRankingStatus,
    EmbeddingSpec,
    SearchMethod,
    SearchOrigin,
    TraceSinkConfig,
    _registry_embedding_kwargs,
)
from .telemetry import SEARCH_TARGET_SKILL, trace_search, trace_search_async, trace_skill_load

__all__ = [
    "PendingReplace",
    "ReplaceOutcome",
    "Skill",
    "SkillCatalog",
    "SkillHit",
    "SkillRegistry",
]

_DenseResult = TypeVar("_DenseResult")


@dataclass
class Skill:
    """Skill metadata: what the index ranks and the capability tools surface."""

    id: str
    name: str
    description: str
    # Author-declared labels and task phrases ("frontend", "login form");
    # indexed for ranking.
    tags: list[str] = field(default_factory=list)
    # Ids of tools this skill's instructions call; surfaced into the
    # search_capabilities tools bucket — not indexed as query terms.
    tools: list[str] = field(default_factory=list)
    # Free-form, non-indexed context for higher layers — e.g.
    # {"stacks": ["react"]} for the push ranker to boost by project context.
    metadata: dict[str, list[str]] = field(default_factory=dict)
    body: str = ""


@dataclass(frozen=True)
class ReplaceOutcome:
    """What a `replace_all` changed, counted by id.

    `updated` covers any field edit (including a body-only rewrite);
    `unchanged` ids are identical to what was registered and keep their cached
    embedding, so a reload that changed nothing costs no embedding calls.
    """

    added: int = 0
    removed: int = 0
    updated: int = 0
    unchanged: int = 0


@dataclass(frozen=True)
class PendingReplace(ReplaceOutcome):
    """What `replace_all` returns: final counts over a pending embedding pass.

    The corpus swap commits synchronously, so the counts are already final when
    `replace_all` returns — read them without awaiting. Awaiting drives the
    embedding pass and resolves to the plain `ReplaceOutcome`, or raises if
    embedding fails, so a source can report what a reload changed even when the
    embedding pass failed.
    """

    _driver: Awaitable[None] | None = None

    def __await__(self) -> Generator[Any, None, ReplaceOutcome]:
        """Drive the embedding pass, then resolve to the plain counts."""
        if self._driver is not None:
            yield from self._driver.__await__()
        return ReplaceOutcome(self.added, self.removed, self.updated, self.unchanged)


class SkillRegistry:
    """Typed Python facade over the private native skill registry."""

    @overload
    def __init__(
        self, embedding: EmbeddingSpec | None = None, *, method: SearchMethod = "bm25"
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        spec: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        pooling: Literal["cls", "mean"] | None = None,
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
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        local: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        pooling: Literal["cls", "mean"] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        ollama: str,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
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
    ) -> None: ...

    def __init__(
        self,
        embedding: EmbeddingSpec | None = None,
        *,
        method: SearchMethod = "bm25",
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

        A "semantic"/"hybrid" `method` makes `register` embed eagerly (inside the
        call, on a worker thread); "bm25" keeps registration model-free.
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
        self._native = _NativeSkillRegistry(**kwargs)
        self._eager = method in ("semantic", "hybrid")
        self._warn_on_model_mismatch = True
        self._adaptive_warned = False
        self._rebuild_on_model_change = False
        self._dense_gate = threading.Lock()
        self._dense_state = threading.Lock()
        self._dense_pending = 0
        # See `ToolRegistry.__init__`: scheduled-but-undriven embedding builds, so
        # a forgotten `await register(...)` is caught at the next dense search.
        self._undriven_builds = 0
        self._dense_tasks: set[asyncio.Task[Any]] = set()

    @overload
    def register(self, item: Skill) -> Awaitable[None]: ...

    @overload
    def register(self, item: Iterable[Skill]) -> Awaitable[None]: ...

    @overload
    def register(
        self,
        item: str,
        name: str,
        description: str,
        tags: list[str],
        tools: list[str],
        metadata: dict[str, list[str]],
        body: str,
    ) -> Awaitable[None]: ...

    def register(
        self,
        item: Skill | Iterable[Skill] | str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        tools: list[str] | None = None,
        metadata: dict[str, list[str]] | None = None,
        body: str | None = None,
    ) -> Awaitable[None]:
        """Register one `Skill`, many `Skill`s, or a flat (id, name, …) tuple.

        Metadata is indexed **synchronously** when `register(...)` is called (a
        forgotten `await` never drops the corpus); the returned awaitable drives
        only the embedding pass. On a "semantic"/"hybrid" registry it embeds in
        one batched, off-thread pass (errors surface when awaited); "bm25" has
        nothing to embed. Always `await` the result.
        """
        flat_args = (name, description, tags, tools, metadata, body)
        if isinstance(item, Skill):
            if any(value is not None for value in flat_args):
                raise TypeError("item register accepts only the Skill argument")
            skills: list[Skill] = [item]
        elif isinstance(item, str):
            if any(value is None for value in flat_args):
                raise TypeError("flat register requires all metadata arguments")
            skills = [Skill(item, name, description, tags, tools, metadata, body)]  # type: ignore[arg-type]
        else:
            if any(value is not None for value in flat_args):
                raise TypeError("iterable register accepts only the items argument")
            skills = list(item)
            if not all(isinstance(skill, Skill) for skill in skills):
                raise TypeError("register requires Skill items")
        self._register_items(skills)
        return self._build_tracked(bool(skills))

    def replace_all(self, skills: Iterable[Skill]) -> PendingReplace:
        """Make `skills` the entire corpus — see `SkillCatalog.replace_all`.

        The corpus swap lands **synchronously** (a forgotten `await` never
        leaves a half-applied reload); the returned `PendingReplace` already
        carries the counts and drives only the embedding pass when awaited.
        Always `await` the result.
        """
        batch = list(skills)
        if not all(isinstance(skill, Skill) for skill in batch):
            raise TypeError("replace_all requires Skill items")
        outcome = self._replace_all_items(batch)
        return self._outcome_tracked(outcome, bool(batch))

    def search(self, query: str, top_k: int) -> list[SkillHit]:
        """Run synchronous, model-free BM25 retrieval."""
        return self._native.search(query, top_k)

    def search_with_origin(self, query: str, top_k: int, origin: SearchOrigin) -> list[SkillHit]:
        """Run BM25 retrieval with an explicit trace origin."""
        return self._native.search_with_origin(query, top_k, origin)

    def search_with_method(
        self, query: str, top_k: int, origin: SearchOrigin, method: SearchMethod
    ) -> list[SkillHit]:
        """Run BM25 synchronously; dense retrieval is async-only."""
        if method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {method}")
        if method != "bm25":
            raise RuntimeError(
                f"{method} search is asynchronous; use `await registry.search_async(..., "
                f'method="{method}")`'
            )
        return self.search_with_origin(query, top_k, origin)

    async def _build(self) -> None:
        """Embed not-yet-embedded items on a worker thread (used by `register`)."""
        await self._run_dense(self._native._build_embeddings)

    async def _rebuild(self) -> None:
        """Recompute and atomically replace the full embedding cache (internal)."""
        await self._run_dense(self._native._rebuild_embeddings)

    def _build_tracked(self, has_items: bool) -> Awaitable[None]:
        """The awaitable `register` returns — see `ToolRegistry._build_tracked`."""
        schedule = self._eager and has_items
        if schedule:
            self._undriven_builds += 1

        async def _drive() -> None:
            if not schedule:
                return
            try:
                await self._build()
            finally:
                self._undriven_builds -= 1

        return _drive()

    async def search_async(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod = "bm25",
    ) -> list[SkillHit]:
        """Search immediately with BM25 or run dense retrieval on a worker thread."""
        if method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {method}")
        if method == "bm25":
            return self.search_with_origin(query, top_k, origin)
        if self._undriven_builds > 0:
            raise RuntimeError(_UNAWAITED_REGISTER)
        await self._maybe_rebuild_on_model_change()
        return await self._run_dense(
            lambda: self._native._search_with_method(query, top_k, origin, method)
        )

    def record_event(self, event: dict[str, Any]) -> None:
        """Record an SDK-layer trace event."""
        self._native.record_event(event)

    def set_trace_sink(
        self, kind: str, session_id: str | None = None, path: str | None = None
    ) -> None:
        """Replace the native trace sink."""
        with self._dense_state:
            self._raise_if_busy()
            self._native.set_trace_sink(kind, session_id, path)

    def experimental_enable_adaptive_ranking(
        self,
        graph: IntentGraph,
        *,
        warn_on_model_mismatch: bool = True,
        rebuild_on_model_change: bool = False,
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
            self._native.enable_adaptive_ranking(graph)
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
        """Re-embed the graph's members under the current model; preserves learning."""
        await self._run_dense(self._native._rebuild_intent_graph)
        self._adaptive_warned = False
        self._maybe_warn_model_mismatch()

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
        # Wait for the worker WITHOUT asyncio.shield (see `ToolRegistry._run_dense`):
        # `asyncio.wait` never cancels the awaited task, so a cancelled caller leaves
        # the worker running, but it avoids shield's Python-3.14 callback that
        # re-reports the inner exception after `_dense_task_done` consumed it.
        await asyncio.wait({task})
        return task.result()

    async def _run_dense_task(self, operation: Callable[[], _DenseResult]) -> _DenseResult:
        try:
            return await asyncio.to_thread(self._run_dense_worker, operation)
        finally:
            self._finish_dense()

    def _run_dense_worker(self, operation: Callable[[], _DenseResult]) -> _DenseResult:
        with self._dense_gate:
            return operation()

    def _dense_task_done(self, task: asyncio.Task[Any]) -> None:
        self._dense_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _queue_dense(self) -> None:
        with self._dense_state:
            self._dense_pending += 1

    def _finish_dense(self) -> None:
        with self._dense_state:
            self._dense_pending -= 1

    def _register_items(self, skills: Iterable[Skill]) -> None:
        skills = list(skills)
        with self._dense_state:
            self._raise_if_busy()
            self._native._register_many(
                [
                    (
                        skill.id,
                        skill.name,
                        skill.description,
                        skill.tags,
                        skill.tools,
                        skill.metadata,
                        skill.body,
                    )
                    for skill in skills
                ]
            )

    def _replace_all_items(self, skills: Iterable[Skill]) -> ReplaceOutcome:
        """Swap the corpus without embedding — the metadata half of `replace_all`."""
        skills = list(skills)
        with self._dense_state:
            self._raise_if_busy()
            added, removed, updated, unchanged = self._native._replace_all(
                [
                    (
                        skill.id,
                        skill.name,
                        skill.description,
                        skill.tags,
                        skill.tools,
                        skill.metadata,
                        skill.body,
                    )
                    for skill in skills
                ]
            )
        return ReplaceOutcome(added, removed, updated, unchanged)

    def _outcome_tracked(self, outcome: ReplaceOutcome, has_items: bool) -> PendingReplace:
        """`_build_tracked`, carrying the already-computed replace counts.

        The counts ride on the awaitable rather than through it, so a failed
        embedding pass still reports what the (already committed) swap changed.
        """
        return PendingReplace(
            outcome.added,
            outcome.removed,
            outcome.updated,
            outcome.unchanged,
            self._build_tracked(has_items),
        )

    def _raise_if_busy(self) -> None:
        if self._dense_pending:
            raise RuntimeError(_REGISTRY_BUSY)


class SkillCatalog:
    """Registry of skills. Register once, then search and load bodies by id."""

    def __init__(
        self,
        trace: TraceSinkConfig | None = None,
        method: SearchMethod = "bm25",
        embedding: EmbeddingSpec | None = None,
    ) -> None:
        """Create an empty skill catalog.

        Args:
            trace: where trace events go; `None` keeps the default no-op sink.
            method: default retrieval method for `search` — see
                `ToolCatalog.__init__`; a "semantic"/"hybrid" catalog embeds
                inside `register`, and dense results come from `search_async`.
            embedding: model for semantic/hybrid retrieval — see
                `ToolCatalog.__init__`; retained and validated under "bm25" too.
        """
        self._skills: dict[str, Skill] = {}
        self._method: SearchMethod = method
        self._registry = SkillRegistry(embedding, method=method)
        if trace is not None:
            self._registry.set_trace_sink(trace.kind, trace.session_id, trace.path)

    def register(self, skills: Skill | Iterable[Skill]) -> Awaitable[None]:
        """Register one skill or many — the single entry point for both.

        Metadata is stored **synchronously** when `register(...)` is called
        (name/description/tags are indexed; `tools`, `metadata`, `body` are
        stored but not indexed), so a forgotten `await` never drops the corpus.
        The returned awaitable drives only the embedding pass: on a
        "semantic"/"hybrid" catalog it embeds the batch off-thread and surfaces
        errors when awaited; a BM25 catalog never loads a model. **Always
        `await` the result.** Re-registering an id replaces it in place.

        A model or dimension change is not recovered in place — construct a new
        catalog and re-register.

        Args:
            skills: a single `Skill` or an iterable of them. Pass the whole batch
                at once for a single embedding request.

        Raises:
            EmbedderError: on a semantic/hybrid catalog, if embedding fails (when awaited).
            RuntimeError: if another operation already owns the registry —
                dense work, but also an in-flight BM25 `search_async` holding
                the read lock. `registry busy` is retryable, not fatal.
        """
        batch = [skills] if isinstance(skills, Skill) else list(skills)
        self._registry._register_items(batch)
        for skill in batch:
            self._skills[skill.id] = skill
        return self._registry._build_tracked(bool(batch))

    def replace_all(self, skills: Iterable[Skill]) -> PendingReplace:
        """Make `skills` the entire content of the catalog.

        Ids absent from the batch are removed, the rest are added or updated —
        the whole-catalog counterpart to `register`, for a source that reloads a
        catalog rather than pushing individual changes. Mutates in place, so
        every holder of this catalog observes the reload without being rebuilt.

        **Removal is unconditional.** Skills registered in-process are dropped
        just like any others, since the batch *is* the catalog. A host that keeps
        local skills alongside a remote source composes the batch itself:
        `await catalog.replace_all([*local_skills, *remote_skills])`.

        Like `register`, the corpus swap lands synchronously and the returned
        awaitable drives only the embedding pass — which embeds just the new and
        re-worded skills, so reloading an unchanged catalog costs nothing. If it
        fails, the new corpus is already live and BM25 still ranks it; semantic
        search raises until a later pass succeeds. **Always `await` the result.**

        Args:
            skills: the complete catalog contents. A repeated id keeps its last
                entry; an empty iterable clears the catalog.

        Returns:
            A `PendingReplace`: the counts, already final, over the pending
            embedding pass. Read them without awaiting; `await` the result to
            drive the embedding pass.

        Raises:
            TypeError: if any item is not a `Skill`.
            EmbedderError: on a semantic/hybrid catalog, if embedding fails (when awaited).
            RuntimeError: if another operation already owns the registry —
                dense work, but also an in-flight BM25 `search_async` holding
                the read lock. `registry busy` is retryable, not fatal.
        """
        batch = list(skills)
        awaitable = self._registry.replace_all(batch)
        self._skills = {skill.id: skill for skill in batch}
        return awaitable

    def search(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod | None = None,
    ) -> list[SkillHit]:
        """Rank registered skills synchronously with BM25.

        The skill twin of `ToolCatalog.search`: a dense resolved method raises
        immediately with guidance to use `search_async`.

        Returns:
            Up to `top_k` `SkillHit`s, best first.
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
            SEARCH_TARGET_SKILL,
            query,
            top_k,
            origin,
            lambda: self._registry.search_with_origin(query, top_k, origin),
        )

    async def search_async(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod | None = None,
    ) -> list[SkillHit]:
        """Rank skills asynchronously with BM25, semantic, or hybrid retrieval.

        Dense methods require a complete cache built explicitly beforehand.
        """
        resolved_method = method or self._method
        return await trace_search_async(
            SEARCH_TARGET_SKILL,
            query,
            top_k,
            origin,
            lambda: self._registry.search_async(query, top_k, origin, resolved_method),
        )

    def has(self, skill_id: str) -> bool:
        """Return whether a skill with this id is registered."""
        return skill_id in self._skills

    def get(self, skill_id: str) -> Skill | None:
        """Return the registered `Skill` for an id, or `None` if unknown."""
        return self._skills.get(skill_id)

    def size(self) -> int:
        """Return the number of registered skills."""
        return len(self._skills)

    def record_event(self, event: dict[str, Any]) -> None:
        """Record a trace event into the catalog's sink.

        See `ToolCatalog.record_event` for the event contract.
        """
        self._registry.record_event(event)

    def experimental_enable_adaptive_ranking(
        self,
        graph: IntentGraph,
        *,
        warn_on_model_mismatch: bool = True,
        rebuild_on_model_change: bool = False,
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
        )

    async def experimental_rebuild_intent_graph(self) -> None:
        """Re-embed the graph's members under the current model; preserves learning."""
        await self._registry.experimental_rebuild_intent_graph()

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

    def invoke(self, skill_id: str) -> str:
        """Return a skill's body for dispatch, recording a `skill_invoke` event.

        Synchronous, unlike `ToolCatalog.invoke` — the body is already in
        memory, there is no handler to run.

        Args:
            skill_id: id of a registered skill.

        Returns:
            The skill's body (Markdown), verbatim as registered.

        Raises:
            ValueError: on an unknown id — callers at the capability-tool
                boundary translate this into a structured error for the agent.
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            raise ValueError(f"unknown skillId: {skill_id}")

        def _run() -> str:
            started = time.monotonic()
            body = skill.body
            self._registry.record_event(
                {
                    "type": "skill_invoke",
                    "skill_id": skill_id,
                    "took_ms": int((time.monotonic() - started) * 1000),
                }
            )
            return body

        return trace_skill_load(skill_id, _run)
