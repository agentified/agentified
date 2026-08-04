# Changelog

All notable changes to `ratel-ai-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `ObservationPolicy` now drives one shared `classify` step used by both the live learner and trace-log initialization, so the pairing rule — which event opens an observation window, which confirms one — exists once rather than in two implementations that could drift.
- `Origin::Baseline` (wire value `baseline`): a query recorded while Ratel was observing but not serving retrieval — the agent chose from its own full tool list and the host captured the turn's text so the invocations that follow can be attributed to it. Ratel's own search path never produces it.
- `ToolRegistry::initialize_intent_graph` builds a graph by stepping through a trace log — the offline half of baseline seeding. Every distinct query is embedded up front so clusters form at the **dense** tier, exactly as the live path would; a model-free replay clusters lexically and a later `rebuild_intent_graph` cannot undo that, since it replaces centroids without revisiting cluster boundaries. Pairing is per session over the log's own order (never re-sorted — `ts` stamps observations, it does not order them), so interleaved sessions sharing one graph cannot cross-pair. One call populates both `tools` and `skills` edges. Returns a detached graph; attaching stays an explicit `set_intent_graph` call.
- `ObservationPolicy` on `UsageLearner`, via `with_policy` — which search origins open an observation window (`OriginFilter`), whether a tool confirms on the attempt or on completion (`Confirmation`), and whether what is learned is marked as seeded (`Provenance`). `Default` reproduces existing behavior exactly and `UsageLearner::new` delegates to it, so nothing changes unless a policy is passed. A rejected search is *ignored*, not cleared, so an unrelated internal search between a captured query and its invokes cannot discard the turn's evidence.
- `seeded_support` on `Intent` (`protocol/v1`): how many of a cluster's `support` observations came from a seeding pass rather than live traffic. Provenance only — nothing reads it during ranking, and two graphs differing only here rank identically and compare equal. Bumped in lockstep with `support`, so one fanned-out question stays one observation. Omitted from the wire form when zero, so a live-only graph serializes byte-identically to one produced before the field existed; a value exceeding `support` is rejected on load.
- `dropped` on the `usage_boost` trace event: how many capability ids a matched cluster remembers that the registry no longer defines, so they were filtered out of the arm. Previously a cluster whose every edge had left the catalog emitted an event byte-identical to a query that matched nothing — the two are different problems (catalog drift vs a coverage gap) with different fixes, and `intent: Some(_)` with `promoted: 0` and `dropped > 0` now tells them apart. Ranking is unchanged: dropping ids the agent cannot invoke is still correct, and only an armed outcome reaches the fusion. Older log lines without the field replay as `dropped: 0`.

### Fixed

- Trace-log initialization under-counted `support` when two sessions asked the same question with their events interleaved — the second session's edge landed but its observation did not. The credit mark is a single slot on the graph keyed by query text, which is enough for the live path (two learners sharing one graph need somewhere common to agree, and identical text from concurrent sessions is rare there) but not for a replay, where sessions interleave by construction and popular questions repeat verbatim. Replay now tracks the mark per session, which it can do exactly because it knows the session id.

### Changed

- **BREAKING:** `Origin` is now `#[non_exhaustive]`. Downstream `match`es over it must include a `_ =>` arm; in return, future origins are non-breaking. Constructing existing variants is unaffected, as are the serde wire form and in-crate matches.

## [0.9.0] - 2026-08-13

### Added

- **Build-time embedding artifacts (ADR-0018).** Binary RAT1 format (magic/version/length/checksum, projection header, Tool/Skill entries with id + projection hash + L2-normalized vector). Semantic validation on load (checksum plus structure and vector semantics). `ToolRegistry` / `SkillRegistry` build a single-kind artifact and warm the dense cache from bytes (`OnArtifactMiss::Error` or `Embed`). `merge_embedding_artifacts` combines compatible parts into one mixed RAT1. For `Local` models, RAT1 build/warm stamps an artifact compatibility fingerprint (content-derived, lazy); runtime `Local` dense-cache identity remains path-based. Artifact persistence remains host-owned; the core artifact APIs accept/return bytes and perform no artifact filesystem I/O. Public exports: `ArtifactError`, `merge_embedding_artifacts`, `OnArtifactMiss`, `ArtifactWarmError`, `ParseOnArtifactMissError`, `WarmError`.

## [0.8.0] - 2026-08-11

### Added

- **Facts: a third capability primitive (ADR-0017).** `Fact { id, name, description, tags, metadata, body, pin }` and `FactRegistry` join `Tool` and `Skill` — constant grounding content an agent should always have on hand (a shop's address and hours, a brand's voice) rather than a playbook it pulls and runs. `PinMode::Always` marks the push tier injected every applicable turn; `PinMode::Retrieved` (the default) is ranked like a skill and surfaces only when a query pulls it in. Name, description, and tags are indexed; `body` is the injected payload and is never indexed. The registry is a parallel of `SkillRegistry`: same selectable BM25 / semantic / hybrid engines, same replace-in-place semantics, same `IndexMap` insertion order.
- Fact telemetry on its own stream — `fact_search`, `fact_churn`, `fact_inject` (carrying a `FactInjectReason` of `Never` / `Evicted` / `Mutated`), `fact_inject_skip`, and `fact_snapshot` — so fact activity stands on its own rather than borrowing the skill events. `TraceEvent` is `#[non_exhaustive]`, so the new variants are additive for downstream matches.

## [0.7.0] - 2026-08-07

### Added

- **Whole-corpus skill reload (ADR-0015).** `SkillRegistry::replace_all` makes the batch the *entire* skill corpus and diffs it against the live one, so an id removed upstream stops being searchable — until now the registry was append-only (`register` replaced an id in place, nothing removed one). The dense cache is touched only where it must be: removed ids' vectors are dropped, changed indexed text is invalidated, everything else is kept, so reloading an unchanged catalog costs zero embeddings. Returns the new public `ReplaceOutcome` (added / removed / updated / unchanged counts). It is the only source of `ChurnKind::Remove` for skills; `TraceEvent::SkillChurn` fires for real changes only.
- `Skill` derives `PartialEq` / `Eq`.

### Changed

- **The BM25 index is cached across searches and rebuilt only on catalog mutation.** Every search used to rebuild the whole BM25 engine (two full-corpus tokenization passes) and re-derive `searchable_text` for every item — roughly 100× the cost of the query itself (61 ms vs 0.6 ms at 1k tools). The first search after a mutation now builds the index through the same full-corpus path and later searches reuse it; `ToolRegistry::register`, `SkillRegistry::register`, and `SkillRegistry::replace_all` invalidate it (`replace_all` keeps it when no indexed text changed), and the hybrid arms share the same cached index instead of deriving `searchable_text` a second time. Scores are byte-for-byte unchanged (ADR-0011) — this is purely a latency win, no API change.

## [0.6.0] - 2026-07-28

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
