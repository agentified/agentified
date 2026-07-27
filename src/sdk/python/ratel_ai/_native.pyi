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

        The scale depends on the search method: raw BM25 (unbounded) for
        "bm25", cosine similarity for "semantic", reciprocal-rank-fusion for
        "hybrid" — scores from different methods are not comparable.
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

        Same method-dependent scale as `SearchHit.score`, computed against the
        skill corpus.
        """

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

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured envelopes — see `ToolRegistry.drain_trace_events`."""

class FactHit:
    """A single fact search result: the matched fact id and its relevance score.

    The fact analogue of `SearchHit` (`tool_id` → `fact_id`), twin of `SkillHit`.
    """

    @property
    def fact_id(self) -> str:
        """Id of the matched fact, as passed to `register`."""

    @property
    def score(self) -> float:
        """Relevance score; higher ranks first.

        Same method-dependent scale as `SearchHit.score`, computed against the
        fact corpus.
        """

class FactRegistry:
    """Private native metadata registry over the fact corpus.

    The grounding-side twin of `SkillRegistry`: a separate index, so facts are
    ranked independently of tools and skills. Unlike a skill, a fact has no
    `tools` field and carries a `pin` (`"always"` / `"retrieved"`).
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
        metadata: dict[str, list[str]],
        body: str,
        pin: str,
    ) -> None:
        """Register a fact's metadata into the index.

        Replaces in place when `id` is already registered. `tags` are indexed
        for ranking; `metadata` rides along un-indexed for higher layers; `body`
        is the injected content, stored but not indexed; `pin` is `"always"` or
        `"retrieved"` (an unknown value raises `ValueError`).
        """

    def _register_many(
        self,
        facts: list[
            tuple[
                str,
                str,
                str,
                list[str],
                dict[str, list[str]],
                str,
                str,
            ]
        ],
    ) -> None:
        """Atomically convert (validating each `pin`), then register a fact batch."""

    def search(self, query: str, top_k: int) -> list[FactHit]:
        """Lexical BM25 search over the fact corpus — see `ToolRegistry.search`."""

    def search_with_origin(self, query: str, top_k: int, origin: str) -> list[FactHit]:
        """BM25 search tagged with who initiated it — see `ToolRegistry.search_with_origin`."""

    def _search_with_method(
        self, query: str, top_k: int, origin: str, method: str
    ) -> list[FactHit]:
        """Private worker-thread search primitive."""

    def _build_embeddings(self) -> None:
        """Private incremental-build primitive."""

    def _rebuild_embeddings(self) -> None:
        """Recompute and atomically replace every fact embedding."""

    def record_event(self, event: dict[str, Any]) -> None:
        """Record an SDK-layer trace event — see `ToolRegistry.record_event`."""

    def set_trace_sink(
        self,
        kind: str,
        session_id: str | None = ...,
        path: str | None = ...,
    ) -> None:
        """Route trace events to a sink — see `ToolRegistry.set_trace_sink`."""

    def drain_trace_events(self) -> list[dict[str, Any]]:
        """Drain captured envelopes — see `ToolRegistry.drain_trace_events`."""
