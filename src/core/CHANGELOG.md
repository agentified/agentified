# Changelog

All notable changes to `ratel-ai-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.6.0-rc.0] - 2026-07-27

### Added

- **Adaptive usage ranking (ADR-0014).** A capability search followed by an invoke becomes a weighted edge in an in-memory `IntentGraph`; matched clusters then boost future rankings through a sub-unit RRF arm fused beside BM25/dense retrieval. Clusters carry support counts and are matched per-member lexically (Jaccard) or by centroid cosine on a semantic catalog, labelled by medoid + c-TF-IDF terms, and aged out by recency (a grace window then a half-life) with eviction and a per-cluster member cap. The arm never overrides a strong lexical/dense match — it lifts capabilities usage history supports.
- `IntentGraph` value type with `protocol/v1` serialization: `to_json` / `from_json` (semantically validated on load, so a malformed or incompatible graph is rejected rather than silently degrading), a `rev` write-counter for save-when-changed persistence and stale-base detection, and `cluster_count`.
- `rank` (0-based, scale-invariant position) and `fused` (whether the usage arm was mixed into this ranking) on `SearchHit`, so callers can order on `rank` and see when `score` switched from raw BM25/cosine to an RRF scale.
- Embedding-model-change detection: a graph built under one model is detected against the active model (fingerprint / dimension) and its arm pauses rather than boost on stale centroids; `rebuild_embeddings` re-embeds cluster members under the current model, preserving support and edges.

### Changed

- **BREAKING:** `TraceEvent` is now `#[non_exhaustive]`. Downstream `match`es over it must include a `_ =>` arm; in return, future event variants (such as the new `usage_boost`) are non-breaking. In-crate matches and the serde wire form are unaffected.

## [0.5.0] - 2026-07-20

### Added

- Configurable dense retrieval via public `EmbeddingModel`, `EmbeddingSpec`, and `Pooling` types: built-in default, HuggingFace, local Candle directories, and OpenAI-compatible endpoints.
- `ToolRegistry::rebuild_embeddings` and `SkillRegistry::rebuild_embeddings` atomically recompute the full dense corpus. Failed rebuilds preserve the prior complete cache.
- `EmbedderError::ModelMismatch` rejects model-identity drift with guidance to rebuild.
- Embedding download, pooling-assumption, and model-mismatch trace events.

### Changed

- **BREAKING:** `EmbedderError` and `TraceEvent` add public variants for configurable-model validation and lifecycle failures; exhaustive matches must handle them.
- Dense cache batches are validated and committed atomically. Endpoint embedding requests are chunked at 64 inputs, responses are capped at 64 MiB, optional response model identity is enforced, and malformed indices/vectors are rejected.
- Endpoint client-cache identity includes the `api_key_env` name without including its secret value, preventing credential cross-talk while preserving vector-space identity.
- Dense searches and rebuilds share an operation guard, preventing a rebuild from swapping vector spaces between query validation and ranking. Fingerprint fields are length-delimited to prevent configuration collisions.
- Public `EmbeddingModel` values can be checked with `validate()` and are validated before a lazy model load; SDK `EmbeddingSpec` construction remains fail-fast.
- In-process (Candle) embedding runs one padded forward pass per batch chunk instead of one per document, speeding up embedding a corpus on a `"semantic"`/`"hybrid"` catalog. Produced vectors are bit-for-bit identical, so rankings and reproducibility are unchanged.

### Fixed

- Failed incremental embedding batches can no longer leave partial vectors, dimensions, or model fingerprints in the cache.
- Distinct embedding models now load concurrently. The process-wide embedder cache held its lock across a full model load, so a cold load of one model blocked loading — or a warm-cache hit for — another; loads now run under a per-key slot lock while same-model loads stay single-flight (one load, reported once).

## [0.4.0] - 2026-07-09

### Fixed

- Re-registering a tool or skill (MCP re-sync, hot-reload) left a stale duplicate in the corpus instead of replacing it in place, causing BM25 score drift and an unbounded memory leak. `ToolRegistry`/`SkillRegistry` are now id-keyed so `register` replaces in place, and the dense embedding cache invalidates on replace so `build_embeddings` re-embeds the changed id.

## [0.3.0] - 2026-07-06

### Added

- **Selectable retrieval methods** (ADR-0011): a `SearchMethod` enum — `Bm25` (default), `Semantic`, `Hybrid` — chosen per registry or per call via `ToolRegistry::search_with_method` / `SkillRegistry::search_with_method`. Semantic ranks a local `BAAI/bge-small-en-v1.5` embedding (pure-Rust Candle); hybrid fuses the BM25 and dense arms with Reciprocal Rank Fusion (no reranker).
- `EmbedderError` (surfaced from `search_with_method` on the semantic/hybrid path) and a `TraceEvent::EmbedderLoad` / `EmbedderLoadStatus` flagging a slow (possibly underpowered machine) or failed model load.
- `ToolRegistry::build_embeddings` / `SkillRegistry::build_embeddings` — pre-compute embeddings for not-yet-embedded tools/skills so a later semantic/hybrid search only embeds the query.

### Changed

- BM25 remains the default engine. `search` / `search_with_origin` keep their infallible `Vec<SearchHit>` signature and BM25 behavior unchanged.
- The dense embedding cache is now **incremental** — a growing prefix of the corpus. `register` only appends (never invalidates), and `build_embeddings` embeds only newly-registered tools, so an existing vector is never recomputed (adding one tool costs one embedding, not N). A BM25-only registry still never loads the model.
- A semantic/hybrid search over an un-built corpus now returns `EmbedderError::EmbeddingsNotBuilt` instead of embedding inside the search path — a search never silently pays the corpus-embedding cost. Populate the cache with `build_embeddings()` first.

## [0.2.1-rc.1] - 2026-07-04

### Changed

- First release cut under the per-package release scheme (ADR-0008): `ratel-ai-core` now versions and ships independently, tagged `core-v*`. No crate API changes since 0.2.0.

## [0.2.0] - 2026-06-16

### Added

- First-class **skills**: a `Skill { id, name, description, tags, tools, metadata, body }` type and a separate `SkillRegistry` BM25 index — ranked independently of tools. Only `name`/`description`/`tags` are indexed; `tools` (a declared dependency edge surfaced at the gateway), `metadata` (non-indexed context such as `stacks`), and `body` are not. Plus `skill_search` / `skill_churn` / `skill_invoke` trace events for the retrieval funnel.

## [0.1.6] - 2026-06-10

### Changed

- Version bump for the coordinated v0.1.6 release (first release shipping the `ratel-ai` Python SDK). No crate source changes since 0.1.5; re-published in lockstep to keep all artifacts version-aligned.

## [0.1.5] - 2026-05-10

### Added

- Initial release on the v1 (revamp) line. BM25 tool retrieval, MCP ingestion, framework-neutral catalog. See the [crate README](README.md) for the full surface.
- `trace` module: `TraceEvent` tagged enum, `TraceEnvelope`, `TraceSink` trait with `NoopSink`, `MemorySink`, and `JsonlSink` (synchronous `O_APPEND`, mode `0600` on Unix) — single tagged event stream per [ADR-0007](../../docs/adr/0007-telemetry-two-streams.md). `ToolRegistry::with_trace_sink` / `set_trace_sink` / `record_event` plus a `search_with_origin` method. `register` emits `index_churn{Add}`; `search` emits `search` with a `bm25` stage. The origin enum tags each search as `direct` (Rust callers, pre-fetch helpers, benchmarks) or `agent` (LLM-synthesized via the gateway), to let downstream consumers separate the two paths.
