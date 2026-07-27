"""Type stubs for the compiled PyO3 extension (`ratel_ai._native`).

Mirrors `src/sdk/python/native/src/lib.rs` — docstrings here are adapted from
that file's `///` docs, so what an IDE shows matches the runtime `__doc__`. The
native layer is a pure pass-through over `ratel-ai-core`; the ergonomic SDK
surface lives in the pure Python modules of this package.
"""

from typing import Any

class SearchHit:
    """A single search result: the matched tool id and its relevance score."""

    @property
    def tool_id(self) -> str:
        """Id of the matched tool, as passed to `register`."""

    @property
    def score(self) -> float:
        """Relevance score; higher ranks first.

        The scale depends on the search method (raw BM25 / cosine / RRF) AND on
        `fused` — with adaptive ranking a matched query returns small RRF scores
        while an unmatched one on the same catalog returns the raw score. Order
        by `rank`, branch on `fused`; treat `score` as a within-list hint only.
        """

    @property
    def rank(self) -> int:
        """0-based position in this result list (best is 0).

        Stable across methods and across the `fused` switch — the field to order
        or threshold on, in place of the scale-shifting `score`.
        """

    @property
    def fused(self) -> bool:
        """Whether `score` is a Reciprocal Rank Fusion score (ordering-only).

        `True` when the usage arm fused into this search or the method is hybrid;
        `False` for a plain BM25/semantic result. Uniform across one result list;
        lets you detect which scale `score` is on.
        """

class IntentGraph:
    """A shared usage-ranking intent graph (ADR-0014).

    Clusters of past queries, each remembering the capabilities invoked after
    them. Hand the *same* instance to a tool catalog and a skill catalog: one
    cluster carries both a tool and a skill edge map, so sharing gives one set
    of clusters with all the evidence behind it, while separate graphs
    duplicate every cluster and split the evidence.
    """

    def __init__(self) -> None:
        """An empty graph — knows nothing until a search is followed by an invoke."""

    @staticmethod
    def from_json(json: str) -> IntentGraph:
        """Adopt a graph in the `protocol/v1` wire form.

        Accepts output from `ratel-graph build`, a previous `to_json()`, or
        Ratel Cloud. Raises `ValueError` if it is malformed or declares a
        schema version this build does not read.
        """

    def to_json(self) -> str:
        """Serialize to the `protocol/v1` wire form.

        For inspection, or to carry what was learned across processes. The graph
        is in-process only; persistence is yours. It mutates on every confirmed
        invoke, so unsaved observations are lost on a crash — persist on a cadence
        or at shutdown. Use ``rev`` to save only when it changed and to detect a
        concurrent writer; single-writer is the supported model.

        SENSITIVE: the output contains the raw text of past user queries (the
        cluster ``members``). Treat a persisted graph like your query/telemetry
        log — restrict permissions (``0600``), keep it out of version control and
        images, and do not ship it to a less-trusted store.
        """

    @property
    def cluster_count(self) -> int:
        """How many clusters the graph holds.

        `0` is the cold-start state, in which it contributes nothing to
        ranking.
        """

    @property
    def rev(self) -> int:
        """Monotonic write counter, bumped once per mutation.

        Never affects ranking — a primitive for your storage layer. Snapshot it
        after each save: a later value means unsaved learning (save-when-changed),
        and a stored graph whose ``rev`` is higher than the one you loaded was
        written by another process (stale-base detection).
        """

class ToolRegistry:
    """Private native metadata registry over `ratel-ai-core`.

    BM25 is synchronous; GIL-releasing dense primitives support the public
    pure-Python async facade. Executors and capability-tool / MCP layers also
    live above this binding.
    """

    def __init__(
        self,
        spec: str | None = ...,
        huggingface: str | None = ...,
        local: str | None = ...,
        ollama: str | None = ...,
        url: str | None = ...,
        model: str | None = ...,
        revision: str | None = ...,
        api_key_env: str | None = ...,
        query_prefix: str | None = ...,
        doc_prefix: str | None = ...,
        pooling: str | None = ...,
        download: bool | None = ...,
    ) -> None: ...
    def register(
        self,
        id: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Register a tool's metadata into the index.

        Replaces in place when `id` is already registered. The schemas must be
        JSON-serializable dicts; anything else raises `ValueError`.
        """

    def _register_many(
        self,
        tools: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]],
    ) -> None:
        """Atomically convert, then register a tool metadata batch."""

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Lexical BM25 search: the top `top_k` tools for `query`, best first.

        Model-free and infallible; the trace event records origin "direct".
        """

    def search_with_origin(self, query: str, top_k: int, origin: str) -> list[SearchHit]:
        """BM25 search tagged with who initiated it.

        `origin` is "agent" (a model calling a capability tool) or anything
        else → "direct" (host code). The origin only labels the emitted trace
        event — ranking is identical to `search`.
        """

    def _search_with_method(
        self, query: str, top_k: int, origin: str, method: str
    ) -> list[SearchHit]:
        """Search with an explicit method ("bm25" | "semantic" | "hybrid").

        "bm25" is infallible; "semantic"/"hybrid" rank against the prebuilt
        embedding cache and raise `RuntimeError` (`EmbeddingsNotBuilt`) if it
        isn't built. Private worker-thread primitive; the public Python wrapper
        exposes `search_async`. An unknown method raises `ValueError`.
        """

    def _build_embeddings(self) -> None:
        """Pre-compute embeddings for not-yet-embedded tools (incremental).

        A later semantic/hybrid search then only embeds the query. Private
        worker-thread primitive used by the public async wrapper.
        """

    def _rebuild_embeddings(self) -> None:
        """Recompute and atomically replace every tool embedding."""

    def record_event(self, event: dict[str, Any]) -> None:
        """Record an SDK-layer trace event into the active sink.

        `event` must be a dict matching one of the core-owned `TraceEvent`
        shapes (ADR-0007, e.g. `{"type": "gateway_search", ...}`); anything
        else raises `ValueError`.
        """

    def set_trace_sink(
        self,
        kind: str,
        session_id: str | None = ...,
        path: str | None = ...,
    ) -> None:
        """Route trace events to a sink.

        `kind` is "noop" (drop everything, the initial state), "memory"
        (buffer for `drain_trace_events`; requires `session_id`) or "jsonl"
        (append to a file; requires `session_id` and `path`). Raises
        `ValueError` on an unknown kind, a missing required argument, or a
        jsonl path that cannot be opened.
        """

    def _rebuild_intent_graph(self) -> None:
        """Re-embed the intent graph's members under the current model (worker)."""

    def adaptive_ranking_status(
        self,
    ) -> tuple[str, str | None, str | None, bool | None]:
        """(status, built, active, dim_mismatch) — adaptive ranking model check."""

    def enable_adaptive_ranking(self, graph: IntentGraph) -> None:
        """Turn on adaptive usage ranking against `graph` (ADR-0014).

        Wires both halves: this registry ranks against the graph, and its trace
        sink is decorated with a learner that grows it from search-then-invoke
        pairs. Pass the same graph to the other registry so both learn into one
        set of clusters.

        Only queries matching a cluster are affected. With a graph attached
        `SearchHit.score` becomes a fusion score rather than a raw BM25 score,
        so use `rank` for ordering and `fused` to detect the scale.
        """

    def disable_adaptive_ranking(self) -> None:
        """Turn adaptive usage ranking off.

        Ranking returns to the base engine and the graph stops growing; the
        graph keeps what it learned, so re-enabling resumes rather than
        restarts.
        """

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured envelopes from the active sink.

        Returns `[]` unless the active sink is "memory".
        """

class EmbedderError(RuntimeError):
    """Embedding model load / inference failure (subclass of RuntimeError)."""

class DimensionMismatchError(EmbedderError):
    """A query/corpus embedding dimension mismatch."""

class SkillHit:
    """A single skill search result: the matched skill id and its relevance score.

    The skill analogue of `SearchHit` (`tool_id` → `skill_id`).
    """

    @property
    def skill_id(self) -> str:
        """Id of the matched skill, as passed to `register`."""

    @property
    def score(self) -> float:
        """Relevance score; higher ranks first.

        Scale depends on the method and on `fused`, as on `SearchHit.score`.
        Order by `rank`, branch on `fused`.
        """

    @property
    def rank(self) -> int:
        """0-based position — as on `SearchHit.rank`."""

    @property
    def fused(self) -> bool:
        """Whether `score` is an RRF score — as on `SearchHit.fused`."""

class SkillRegistry:
    """Private native metadata registry over the skill corpus.

    The on-demand analogue of `ToolRegistry`: a separate index, so skills are
    ranked independently of tools (own corpus statistics, own top-K).
    """

    def __init__(
        self,
        spec: str | None = ...,
        huggingface: str | None = ...,
        local: str | None = ...,
        ollama: str | None = ...,
        url: str | None = ...,
        model: str | None = ...,
        revision: str | None = ...,
        api_key_env: str | None = ...,
        query_prefix: str | None = ...,
        doc_prefix: str | None = ...,
        pooling: str | None = ...,
        download: bool | None = ...,
    ) -> None: ...
    def register(
        self,
        id: str,
        name: str,
        description: str,
        tags: list[str],
        tools: list[str],
        metadata: dict[str, list[str]],
        body: str,
    ) -> None:
        """Register a skill's metadata into the index.

        Replaces in place when `id` is already registered. `tags` are indexed
        for ranking; `tools` and `metadata` ride along un-indexed for higher
        layers; `body` is the full instruction text, stored for on-demand load.
        """

    def _register_many(
        self,
        skills: list[
            tuple[
                str,
                str,
                str,
                list[str],
                list[str],
                dict[str, list[str]],
                str,
            ]
        ],
    ) -> None:
        """Atomically convert, then register a skill metadata batch."""

    def search(self, query: str, top_k: int) -> list[SkillHit]:
        """Lexical BM25 search over the skill corpus — see `ToolRegistry.search`."""

    def search_with_origin(self, query: str, top_k: int, origin: str) -> list[SkillHit]:
        """BM25 search tagged with who initiated it — see `ToolRegistry.search_with_origin`."""

    def _search_with_method(
        self, query: str, top_k: int, origin: str, method: str
    ) -> list[SkillHit]:
        """Private worker-thread search primitive."""

    def _build_embeddings(self) -> None:
        """Private incremental-build primitive."""

    def _rebuild_embeddings(self) -> None:
        """Recompute and atomically replace every skill embedding."""

    def record_event(self, event: dict[str, Any]) -> None:
        """Record an SDK-layer trace event — see `ToolRegistry.record_event`."""

    def set_trace_sink(
        self,
        kind: str,
        session_id: str | None = ...,
        path: str | None = ...,
    ) -> None:
        """Route trace events to a sink — see `ToolRegistry.set_trace_sink`."""

    def _rebuild_intent_graph(self) -> None:
        """Re-embed the intent graph's members under the current model (worker)."""

    def adaptive_ranking_status(
        self,
    ) -> tuple[str, str | None, str | None, bool | None]:
        """(status, built, active, dim_mismatch) — adaptive ranking model check."""

    def enable_adaptive_ranking(self, graph: IntentGraph) -> None:
        """Turn on adaptive usage ranking against `graph` (ADR-0014).

        Wires both halves: this registry ranks against the graph, and its trace
        sink is decorated with a learner that grows it from search-then-invoke
        pairs. Pass the same graph to the other registry so both learn into one
        set of clusters.

        Only queries matching a cluster are affected. With a graph attached
        `SearchHit.score` becomes a fusion score rather than a raw BM25 score,
        so use `rank` for ordering and `fused` to detect the scale.
        """

    def disable_adaptive_ranking(self) -> None:
        """Turn adaptive usage ranking off.

        Ranking returns to the base engine and the graph stops growing; the
        graph keeps what it learned, so re-enabling resumes rather than
        restarts.
        """

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured envelopes — see `ToolRegistry.drain_trace_events`."""
