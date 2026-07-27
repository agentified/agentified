"""Fact catalog — the Python mirror of `src/sdk/ts/src/fact-catalog.ts`.

`FactRegistry` is a typed facade over the private native index; `FactCatalog` is
the push-path analogue of `SkillCatalog`: where a skill is a playbook the agent
*pulls* and runs on demand, a fact is constant grounding content the grounding
layer *pushes* into the context so the model is never missing it (a barbershop's
address and hours, a brand's voice).

A fact's `pin` splits two tiers: `"always"` facts are injected on every
applicable turn (`FactCatalog.pinned`); `"retrieved"` facts (the default)
surface only when a query ranks them in (`FactCatalog.search`). Both are ranked
by the native registry, so a pinned fact stays discoverable. The re-injection
freshness gate (`plan_injection` in `grounding.py`) decides, per turn, which of
these actually need injecting.
"""

from __future__ import annotations

import asyncio
import os
import threading
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeVar, cast, overload

from ._native import FactHit
from ._native import FactRegistry as _NativeFactRegistry
from .catalog import (
    _REGISTRY_BUSY,
    _UNAWAITED_REGISTER,
    EmbeddingSpec,
    SearchMethod,
    SearchOrigin,
    TraceSinkConfig,
    _registry_embedding_kwargs,
)
from .grounding import (
    FACT_ID_PATTERN,
    FactCandidate,
    GroundingItem,
    GroundingResult,
    GroundingSnapshotItem,
    InjectionReason,
    PinTier,
    plan_injection,
)
from .telemetry import SEARCH_TARGET_FACT, trace_search, trace_search_async

__all__ = ["ExperimentalWarning", "Fact", "FactCatalog", "FactHit", "FactRegistry", "Pin"]

_DenseResult = TypeVar("_DenseResult")
_DEFAULT_FACTS_TOP_K = 3
_MAX_TOP_K = 50

# One-time gate for the experimental-API warning `FactCatalog.__init__` emits.
# Module-level so a test can reset it (the warning is once per process).
_warned = False


class ExperimentalWarning(UserWarning):
    """Warning that the facts/grounding API is experimental and may change.

    Emitted once per process the first time a `FactCatalog` is constructed,
    unless `RATEL_EXPERIMENTAL_SILENCE` is set. A dedicated `UserWarning`
    subclass so callers can filter it precisely by category.
    """


def _warn_experimental_once() -> None:
    """Emit the experimental-API warning at most once per process.

    Skipped entirely when `RATEL_EXPERIMENTAL_SILENCE` is set to any truthy
    value; otherwise fires a single `ExperimentalWarning` and latches the
    module-level `_warned` flag so later `FactCatalog` constructions stay quiet.
    """
    global _warned
    if _warned or os.environ.get("RATEL_EXPERIMENTAL_SILENCE"):
        return
    _warned = True
    warnings.warn(
        "ratel: FactCatalog is experimental — the facts/grounding API may change. "
        "Set RATEL_EXPERIMENTAL_SILENCE=1 to silence.",
        ExperimentalWarning,
        stacklevel=3,
    )


class Pin(str, Enum):
    """The two tiers a fact's `pin` splits into — always-on vs retrieval-gated.

    A `str` enum, so `Pin.ALWAYS` is interchangeable with the wire string
    `"always"` — pass either to `Fact(pin=...)`.
    """

    ALWAYS = "always"
    RETRIEVED = "retrieved"


def _clamp_facts_top_k(value: int | None) -> int:
    """Clamp a facts top-K to [1, 50], falling back to the default for junk."""
    if not isinstance(value, int) or value < 1:
        return _DEFAULT_FACTS_TOP_K
    return min(value, _MAX_TOP_K)


@dataclass
class Fact:
    """Fact metadata plus the grounding content the freshness gate injects.

    The push-path twin of `Skill`: `name`, `description`, and `tags` drive
    ranking exactly as on a skill, so the retrieval-gated tier is discoverable by
    query; `body` is the injected content (not indexed), and `pin` splits the
    always-on tier from the retrieval-gated one. A fact has no `tools` field (it
    is content, not a playbook that calls tools).
    """

    id: str
    name: str
    description: str
    # Author-declared labels and task phrases ("location", "booking"); indexed
    # for ranking alongside the description.
    tags: list[str] = field(default_factory=list)
    # Free-form, non-indexed context for higher layers (push-path
    # boosting/filtering); never matched as query terms.
    metadata: dict[str, list[str]] = field(default_factory=dict)
    # The injected content — stored but not indexed, so a long body never skews
    # relevance.
    body: str = ""
    # "always" (injected every turn) or "retrieved" (the default: surfaced only
    # when a query ranks it in).
    pin: str = "retrieved"


class FactRegistry:
    """Typed Python facade over the private native fact registry."""

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
        call, on a worker thread); "bm25" keeps registration model-free. Mirrors
        `SkillRegistry.__init__`.
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
        self._native = _NativeFactRegistry(**kwargs)
        self._eager = method in ("semantic", "hybrid")
        self._dense_gate = threading.Lock()
        self._dense_state = threading.Lock()
        self._dense_pending = 0
        # See `SkillRegistry.__init__`: scheduled-but-undriven embedding builds,
        # so a forgotten `await register(...)` is caught at the next dense search.
        self._undriven_builds = 0
        self._dense_tasks: set[asyncio.Task[Any]] = set()

    @overload
    def register(self, item: Fact) -> Awaitable[None]: ...

    @overload
    def register(self, item: Iterable[Fact]) -> Awaitable[None]: ...

    @overload
    def register(
        self,
        item: str,
        name: str,
        description: str,
        tags: list[str],
        metadata: dict[str, list[str]],
        body: str,
        pin: str,
    ) -> Awaitable[None]: ...

    def register(
        self,
        item: Fact | Iterable[Fact] | str,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, list[str]] | None = None,
        body: str | None = None,
        pin: str | None = None,
    ) -> Awaitable[None]:
        """Register one `Fact`, many `Fact`s, or a flat (id, name, …, pin) tuple.

        Metadata is indexed **synchronously** when `register(...)` is called (a
        forgotten `await` never drops the corpus); the returned awaitable drives
        only the embedding pass. On a "semantic"/"hybrid" registry it embeds in
        one batched, off-thread pass (errors surface when awaited); "bm25" has
        nothing to embed. Always `await` the result. Mirrors
        `SkillRegistry.register`, minus `tools` and plus `pin`.
        """
        flat_args = (name, description, tags, metadata, body, pin)
        if isinstance(item, Fact):
            if any(value is not None for value in flat_args):
                raise TypeError("item register accepts only the Fact argument")
            facts: list[Fact] = [item]
        elif isinstance(item, str):
            if any(value is None for value in flat_args):
                raise TypeError("flat register requires all metadata arguments")
            facts = [Fact(item, name, description, tags, metadata, body, pin)]  # type: ignore[arg-type]
        else:
            if any(value is not None for value in flat_args):
                raise TypeError("iterable register accepts only the items argument")
            facts = list(item)
            if not all(isinstance(fact, Fact) for fact in facts):
                raise TypeError("register requires Fact items")
        self._register_items(facts)
        return self._build_tracked(bool(facts))

    def search(self, query: str, top_k: int) -> list[FactHit]:
        """Run synchronous, model-free BM25 retrieval."""
        return self._native.search(query, top_k)

    def search_with_origin(self, query: str, top_k: int, origin: SearchOrigin) -> list[FactHit]:
        """Run BM25 retrieval with an explicit trace origin."""
        return self._native.search_with_origin(query, top_k, origin)

    def search_with_method(
        self, query: str, top_k: int, origin: SearchOrigin, method: SearchMethod
    ) -> list[FactHit]:
        """Run BM25 synchronously; dense retrieval is async-only."""
        if method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {method}")
        if method != "bm25":
            raise RuntimeError(
                f'{method} search is asynchronous; use `await registry.search_async(..., '
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
        """The awaitable `register` returns — see `SkillRegistry._build_tracked`."""
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
    ) -> list[FactHit]:
        """Search immediately with BM25 or run dense retrieval on a worker thread."""
        if method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {method}")
        if method == "bm25":
            return self.search_with_origin(query, top_k, origin)
        if self._undriven_builds > 0:
            raise RuntimeError(_UNAWAITED_REGISTER)
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
        # Wait for the worker WITHOUT asyncio.shield (see `SkillRegistry._run_dense`):
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

    def _register_items(self, facts: Iterable[Fact]) -> None:
        facts = list(facts)
        with self._dense_state:
            self._raise_if_busy()
            self._native._register_many(
                [
                    (
                        fact.id,
                        fact.name,
                        fact.description,
                        fact.tags,
                        fact.metadata,
                        fact.body,
                        fact.pin,
                    )
                    for fact in facts
                ]
            )

    def _raise_if_busy(self) -> None:
        if self._dense_pending:
            raise RuntimeError(_REGISTRY_BUSY)


class FactCatalog:
    """Registry of facts. Register once, then search, ground, and look up by id."""

    def __init__(
        self,
        trace: TraceSinkConfig | None = None,
        method: SearchMethod = "bm25",
        embedding: EmbeddingSpec | None = None,
        facts_top_k: int | None = None,
    ) -> None:
        """Create an empty fact catalog.

        Args:
            trace: where trace events go; `None` keeps the default no-op sink.
            method: default retrieval method for `search` — see
                `SkillCatalog.__init__`; a "semantic"/"hybrid" catalog embeds
                inside `register`, and dense results come from `search_async`.
            embedding: model for semantic/hybrid retrieval — see
                `SkillCatalog.__init__`; retained and validated under "bm25" too.
            facts_top_k: max retrieval-gated facts `ground` considers (default 3).
        """
        _warn_experimental_once()
        self._facts: dict[str, Fact] = {}
        self._method: SearchMethod = method
        self._facts_top_k = facts_top_k
        # Session bookkeeping for the freshness gate: the body this catalog last
        # injected per fact id via `ground`. Lets an absent body be classified
        # as `evicted` (same body, gone from the window) or `mutated` (body has
        # since changed) instead of `never`. The transcript itself carries the
        # rest.
        self._injected_bodies: dict[str, str] = {}
        self._registry = FactRegistry(embedding, method=method)
        if trace is not None:
            self._registry.set_trace_sink(trace.kind, trace.session_id, trace.path)

    def register(self, facts: Fact | Iterable[Fact]) -> Awaitable[None]:
        """Register one fact or many — the single entry point for both.

        Metadata is stored **synchronously** when `register(...)` is called
        (name/description/tags are indexed; `metadata`, `body`, and `pin` are
        stored but not indexed), so a forgotten `await` never drops the corpus.
        The returned awaitable drives only the embedding pass: on a
        "semantic"/"hybrid" catalog it embeds the batch off-thread and surfaces
        errors when awaited; a BM25 catalog never loads a model. **Always
        `await` the result.** Re-registering an id replaces it in place.

        Each fact is validated at this boundary: the `id` must match
        `FACT_ID_PATTERN` (ids ride in trace events and structured payloads) and
        `pin`, if set, must be `"always"` or `"retrieved"`. A bad value raises
        `ValueError` before anything is indexed.

        A model or dimension change is not recovered in place — construct a new
        catalog and re-register.

        Args:
            facts: a single `Fact` or an iterable of them. Pass the whole batch at
                once for a single embedding request.

        Raises:
            ValueError: if a fact's `id` or `pin` is invalid.
            EmbedderError: on a semantic/hybrid catalog, if embedding fails (when awaited).
            RuntimeError: if a dense operation already owns the registry.
        """
        batch = [facts] if isinstance(facts, Fact) else list(facts)
        for fact in batch:
            _assert_valid_fact(fact)
        self._registry._register_items(batch)
        for fact in batch:
            self._facts[fact.id] = fact
        return self._registry._build_tracked(bool(batch))

    def search(
        self,
        query: str,
        top_k: int,
        origin: SearchOrigin = "direct",
        method: SearchMethod | None = None,
    ) -> list[FactHit]:
        """Rank registered facts synchronously with BM25 — ranks both tiers.

        A pinned fact can still be a query hit. The fact twin of
        `SkillCatalog.search`: a dense resolved method raises immediately with
        guidance to use `search_async`.

        Returns:
            Up to `top_k` `FactHit`s, best first.
        """
        resolved_method = method or self._method
        if resolved_method not in ("bm25", "semantic", "hybrid"):
            raise ValueError(f"unknown search method: {resolved_method}")
        if resolved_method != "bm25":
            raise RuntimeError(
                f'{resolved_method} search is asynchronous; use `await catalog.search_async(..., '
                f'method="{resolved_method}")`'
            )
        return trace_search(
            SEARCH_TARGET_FACT,
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
    ) -> list[FactHit]:
        """Rank facts asynchronously with BM25, semantic, or hybrid retrieval.

        Dense methods require a complete cache built explicitly beforehand.
        """
        resolved_method = method or self._method
        return await trace_search_async(
            SEARCH_TARGET_FACT,
            query,
            top_k,
            origin,
            lambda: self._registry.search_async(query, top_k, origin, resolved_method),
        )

    async def ground(
        self,
        query: str,
        transcript: Sequence[str],
        top_k: int | None = None,
    ) -> GroundingResult:
        """Decide which facts to (re-)inject given the current transcript.

        The grounding freshness gate. Considers the always-on tier (`pinned`)
        plus the retrieval-gated facts `query` ranks in, then injects only those
        whose body is not already in `transcript`: never injected (``never``),
        gone from the window (``evicted``), or changed since injection
        (``mutated``). Records a ``fact_inject`` / ``fact_inject_skip`` event
        per fact.

        Presence is the fact's own body text — no markers, no tags: render each
        `GroundingItem.body` **verbatim** as (part of) the message you append
        (decorate around it, don't rewrite it), and its presence dedupes the
        next turn. Stateless across conversations — the transcript *is* the
        record — with one piece of session memory: the last body injected per
        id, so an absent body reads as ``evicted``/``mutated`` instead of
        ``never``.

        Args:
            query: the current turn's text, for the retrieval-gated tier.
            transcript: per-message text of the current history, oldest first.
            top_k: max retrieval-gated facts to consider (defaults to the
                catalog's `facts_top_k`, then 3).

        Returns:
            The facts to inject (always-on first) and the ids left fresh.
        """
        candidates = await self._candidate_facts(query, top_k)
        decisions = plan_injection(
            [FactCandidate(fact.id, fact.body) for fact in candidates],
            transcript,
            previously_injected=self._injected_bodies,
        )

        by_id = {fact.id: fact for fact in candidates}
        inject: list[GroundingItem] = []
        skipped: list[str] = []
        for decision in decisions:
            fact = by_id.get(decision.id)
            if fact is None:  # unreachable: decisions mirror candidates
                continue
            if decision.inject:
                tier: PinTier = "always" if fact.pin == Pin.ALWAYS else "retrieved"
                inject.append(
                    GroundingItem(
                        id=fact.id,
                        body=fact.body,
                        reason=cast(InjectionReason, decision.reason),
                        pin=tier,
                    )
                )
                self._injected_bodies[fact.id] = fact.body
                self.record_event(
                    {"type": "fact_inject", "fact_id": fact.id, "reason": decision.reason}
                )
            else:
                skipped.append(fact.id)
                self.record_event({"type": "fact_inject_skip", "fact_id": fact.id})
        return GroundingResult(inject=inject, skipped=skipped)

    async def ground_snapshot(
        self, query: str, top_k: int | None = None
    ) -> list[GroundingSnapshotItem]:
        """Return the full grounding set for one model call — the stateless mode.

        The per-call twin of `ground`: always-on facts plus the retrieval-gated
        facts `query` ranks in, recomputed fresh every call. No freshness gate,
        no transcript, nothing persisted — render the items into the call's
        message override and discard them with it. The persist/per-call split
        mirrors the recall idiom's ``appendRecall``-vs-``prepareStep``: use this
        for one-shot or stateless calls (or to keep grounding out of your stored
        history), and `ground` for a long-lived transcript where the freshness
        gate earns its keep. Records a ``fact_snapshot`` event per fact.

        Args:
            query: the current turn's text, for the retrieval-gated tier.
            top_k: max retrieval-gated facts to consider (defaults to the
                catalog's `facts_top_k`, then 3).

        Returns:
            The snapshot items (always-on first).
        """
        items: list[GroundingSnapshotItem] = []
        for fact in await self._candidate_facts(query, top_k):
            tier: PinTier = "always" if fact.pin == Pin.ALWAYS else "retrieved"
            items.append(GroundingSnapshotItem(id=fact.id, body=fact.body, pin=tier))
            self.record_event({"type": "fact_snapshot", "fact_id": fact.id})
        return items

    def pinned(self) -> list[Fact]:
        """Return the always-on facts (`pin == "always"`), in registration order.

        The push tier `ground` considers every turn, bypassing ranking. The
        freshness gate still decides whether each actually needs (re-)injecting.
        """
        return [fact for fact in self._facts.values() if fact.pin == Pin.ALWAYS]

    async def _candidate_facts(self, query: str, top_k: int | None) -> list[Fact]:
        """Assemble a grounding pass's candidates: pinned plus query-ranked facts.

        Deduped by id (a pinned fact that also ranks appears once, as pinned).
        Shared by `ground` and `ground_snapshot` so the two modes can never
        disagree on the set.
        """
        k = _clamp_facts_top_k(top_k if top_k is not None else self._facts_top_k)
        pinned = self.pinned()
        pinned_ids = {fact.id for fact in pinned}
        hits = await self.search_async(query, k, "direct")
        retrieved = [
            fact
            for hit in hits
            if (fact := self._facts.get(hit.fact_id)) is not None and fact.id not in pinned_ids
        ]
        return pinned + retrieved

    def has(self, fact_id: str) -> bool:
        """Return whether a fact with this id is registered."""
        return fact_id in self._facts

    def get(self, fact_id: str) -> Fact | None:
        """Return the registered `Fact` for an id (including `body`), or `None`."""
        return self._facts.get(fact_id)

    def size(self) -> int:
        """Return the number of registered facts."""
        return len(self._facts)

    def record_event(self, event: dict[str, Any]) -> None:
        """Record a trace event into the catalog's sink.

        The `fact_inject` / `fact_inject_skip` grounding events ride this. See
        `SkillCatalog.record_event` for the event contract.
        """
        self._registry.record_event(event)

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured trace envelopes; `[]` unless the sink is "memory"."""
        return self._registry.drain_trace_events()


def _assert_valid_fact(fact: Fact) -> None:
    """Reject a fact whose id or pin can't be trusted at the catalog boundary.

    Args:
        fact: the fact to validate.

    Raises:
        ValueError: if `id` doesn't match `FACT_ID_PATTERN`, or `pin` isn't
            `"always"` / `"retrieved"`.
    """
    if not isinstance(fact.id, str) or not FACT_ID_PATTERN.match(fact.id):
        raise ValueError(
            f"ratel: fact id {fact.id!r} must match {FACT_ID_PATTERN.pattern} "
            "(letters, digits, and . _ : - only; ids ride in trace events and "
            "structured payloads)"
        )
    if fact.pin not in ("always", "retrieved"):
        raise ValueError(
            f'ratel: fact {fact.id} has invalid pin {fact.pin!r} (expected "always" or "retrieved")'
        )
