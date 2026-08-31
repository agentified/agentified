# Changelog

All notable changes to `ratel-ai-core` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this crate adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **A query joins an intent cluster by member coverage, not centroid proximity alone.** Admission now prefilters on `cos(query, centroid) >= similarity`, counts how many of the cluster's vector-bearing members the query also clears that threshold against, and admits only when `count >= max(2, ceil(coverage * members))`. Matching one member is single-link chaining — A joins because of B, and the cluster grows into whatever B happened to bridge to — and matching an average is worse, because the average of two intents resembles neither. The dense tier had no per-member guard while the lexical tier always had one; that gap collapsed 12 distinct questions into a single cluster in production. On the 50-turn fixture the graph goes from 8 clusters (largest holding 39 of 50 turns) to 20 (largest 12), purity 0.362 to 0.766, merge F1 0.209 to 0.423. Merge recall falls 0.872 to 0.402 and that is the point: the old figure came from one cluster containing almost everything, which scores well on recall for the same reason it is useless.
- **`ClusterPolicy { similarity, coverage }` — the two numbers that draw every cluster boundary, configurable per catalog and recorded on the graph they clustered.** The threshold is model-dependent (a cosine of 0.70 does not mean the same thing on two embedding models) and corpus-dependent (a narrow catalog and a broad one want different granularity), so it cannot be a constant. Recording it is what keeps that safe: two producers at different settings would otherwise disagree about what a cluster means while both claiming `v: 1`. `#[non_exhaustive]` with `with_similarity` / `with_coverage` builders; out-of-range values are rejected rather than clamped, since a clamp clusters at something the caller did not ask for and boundaries once drawn are never redrawn. A graph clustered under one policy and served under another reports `AdaptiveRankingStatus::PolicyDrift` — deliberately not `Paused`, because the clusters are still coherent, merely coarser or finer, and rebuilding cannot revisit boundaries. Recorded on the wire as `cluster_policy`, omitted at the defaults so an untuned graph stays byte-identical to one written before the field existed.
- **`SearchHit::normalized` and `SkillHit::normalized`: `score` mapped onto `[0, 1]`.** The raw `score` is on three incomparable scales — unbounded BM25, bounded cosine, and an RRF sum whose magnitude is rank arithmetic — so it has never been displayable, and normalizing it yourself is how a rank position gets read as certainty. Each method now carries the rule its own scale admits: `(cos + 1) / 2` for cosine; `score / sum of idf(query terms)`, clamped, for raw BM25 — the share of the query's discriminating mass a hit captured, which also exposes when the catalog has no vocabulary for the question at all; and min-max across the **full candidate set** for RRF, computed before the `top_k` cut so the weakest returned hit is not forced to `0.00` and the value does not move with `top_k`. The first two are absolute and compare across queries. The RRF rule does not, because rank fusion has no achievable maximum tied to the query — a `1.0` there means "best of what came back", not "right". **None of them is a calibrated confidence:** nothing was fitted to whether a hit was the one the caller went on to invoke.
- **`Intent::surfaced_tools` and `Intent::surfaced_skills`: how many of a cluster's searches put each capability in front of the caller.** The denominator the edge maps never had — a capability shown twelve times and invoked once was indistinguishable from one shown once and invoked once, because only invocations were recorded. `TraceEvent::Search` has always carried its ranked hits and the learner has always discarded them. Counted at confirm, so an abandoned search still teaches nothing, and once per search window, so one search followed by three invokes is three edges but one impression each. Only ids ranked **at or above** the invoked one count: a capability listed below the one the caller took was very likely never read, and counting it as refused would penalise it for where the ranker put it. Two maps rather than one because tool and skill ids are distinct id spaces. Provenance on the wire, omitted when empty; ADR-0014's rule that edges come from invocations and never from retrievals is unchanged, and an id present here but absent from the matching edge map has no edge and cannot be promoted.
- **`Intent` persists `cohesion` and `vector_n`.** A normalized centroid has length 1 however far apart its members are, so a cluster that drifted into the generic direction of its domain reads as tight and keeps absorbing. `cohesion` is the un-normalized mean's length and `vector_n` the fold count; together they let a reloaded graph keep folding correctly instead of treating its centroid as a single prior sample that one further observation can drag. Both optional on the wire, absent meaning `1`.

### Changed

- **Cluster edges are ranked by inverse cluster frequency, not raw invocation count.** An edge weight is a count of confirmed invocations, but serving that order raw lets a capability invoked across many different intents rank on volume rather than on answering *this* question — the mechanism behind the reported failure, where the most-invoked write op rode into every task-phrased search. Each edge is now scaled by `1 + ln(clusters / clusters naming that capability)`, counted over every cluster's raw edge maps rather than the edges a consumer's own catalog still defines, so the order is a property of the graph and not of which catalog is attached. Where counts previously tied, the order fell through to the id tie-break, so a write op could lead a read query on the strength of sorting earlier alphabetically. Derived, so nothing about it crosses the wire.
- **An edge is damped by how often it was offered and refused.** `passed_over = min(1, (invoked + 3) / (considered + 3))` multiplies the invocation count, so a capability shown constantly and taken rarely falls behind one shown rarely and always taken. Three properties earn that shape: a cluster that recorded no impressions gets `3/3 = 1` exactly, so existing graphs and callers that do not report their hits rank unchanged and no feature flag is needed; it is clamped at `1`, so an impression can only ever cost an edge and never promote one; and it multiplies rather than replaces the count, so nine invocations still outweigh one. The pseudo-counts stop a single unlucky impression halving an edge.
- **BM25 `b` raised 0.4 to the standard 0.75.** **Breaking:** lexical scores and therefore ranking change for every corpus. Note that this was raised alongside a projection change that has since been reverted, and it is the one change on this branch the fixture argues *against*: on 47 queries it leaves top-1 accuracy against the invoked tool unchanged at 12 of 47 while raising read-queries-served-a-write-op from 8 of 25 to 11. It stays only because 0.75 is the field standard. See the `b` sweep in `harness-results.md`.
- **The clustering policy is read from the graph rather than from constants**, so a graph carries the boundaries it was drawn under. **Breaking** for anything that assumed the built-in thresholds.

### Fixed

- **The centroid accumulated wrongly.** `absorb_vector` rescaled the already-normalized centroid by the fold count before folding in the next vector, which asserted that the accumulated history had full length `k` — true only when every member points the same way. That over-weighted early members and erased the spread, and the error grew with exactly the diffuse clusters where it mattered most. The mean is now accumulated un-normalized and the centroid re-derived from it.
- **A cluster the dense tier rejected could still be joined lexically while learning**, so a query the coverage rule had just refused entered the cluster by the back door and the guard bought nothing.
- **Intent-graph members are embedded query-side.** They are queries, and embedding them with the document-side prefix put them in a different region of the space than the queries they are matched against.
- **A stale embedding artifact is named rather than silently re-embedded.** `projection_version` was decoded from a prebuilt RAT1 artifact and discarded, so an artifact built under a different projection either failed closed with a list of missing ids and no stated cause, or — under `OnArtifactMiss::Embed` — silently re-embedded the whole catalog on every registration, which is the exact cost the artifact exists to remove. It now fails with `WarmError::ArtifactProjectionMismatch`, naming the cause and the remedy.

## [0.11.0] - 2026-08-21

### Added

- Experimental, opt-in `CatalogDefinition` trace events for changed tool, skill, and fact definitions. Events carry the public definition fields, effective searchable description, override state, and a canonical SHA-256 content hash; unchanged definitions are suppressed.

- **Per-entry searchable-description projections (ADR-0021).** `Tool`, `Skill`, and `Fact` each gained `experimental_searchable_description: Option<String>`, an override for the description component BM25 and dense retrieval actually rank — so retrieval text can be tuned without touching the model-facing `description`, the schemas, or the payload. The override replaces *only* that component: the name stays indexed (whole and identifier-split) and skill/fact tags stay indexed, including when the override is the empty string, so optimizing a description can never make an entry undiscoverable by its own identifier. For a tool, supplying the override additionally opts that entry out of schema indexing — input and output schemas stay model-facing but stop contributing tokens. `None` leaves the stable ADR-0004 projection untouched: tools still rank description plus schema tokens, skills and facts still rank their authored description. Schema exclusion is per-entry opt-in, not a new default.

  Existing RAT1 artifacts still warm: warming reuses a vector when the entry's stored `projection_hash` matches the text projected now, and the no-override arm of each projection is the previous implementation unchanged, so those hashes still match. What does change is `merge_embedding_artifacts`, which requires every part to share a `projection_version` — a hash of the projection sources, all three of which this release touched. A part built against 0.10.0 therefore no longer merges with one built against this version; rebuild the parts together. `fact_indexing.rs` is also now hashed alongside `indexing.rs` and `skill_indexing.rs`, which it should have been since facts shipped — a fact-projection change previously left that version stamp unmoved.

### Changed

- **Source-breaking: `Tool`, `Skill`, and `Fact` each gained a public field.** None of the three is `#[non_exhaustive]`, none derives `Default`, and none has a builder, so every downstream struct literal — `Tool { id, name, description, input_schema, output_schema }` and its skill/fact equivalents — stops compiling until it adds `experimental_searchable_description: None`. This crate's own quickstart and `examples/search_demo.rs` had to be edited, which is the proof: code written against the published docs breaks. **At 0.x a breaking change is a MINOR bump, never a patch — the next release is 0.11.0, not 0.10.1.**

## [0.10.0] - 2026-08-17

### Added

- `FnSink`: a `TraceSink` that hands each envelope to a closure as a JSON line, for hosts whose trace destination this crate cannot own — a process-per-request server writing to a database, a language binding forwarding to its runtime. The line is the same wire form `JsonlSink` writes for the same event, session, and source — field for field, bar the per-record `ts` and `event_id` every envelope-aware sink mints for itself, neither of which replay reads — so lines collected across processes can be joined with newlines and passed straight back to `build_intent_graph` with no re-derivation. The session id is a default rather than an identity: replay pairs searches with invokes per session, so a host reassembling turns from its own storage should restamp each line with an id unique to each concurrent turn.
- `ObservationPolicy` now drives one shared `classify` step used by both the live learner and the offline log replay, so the pairing rule — which event opens an observation window, which confirms one — exists once rather than in two implementations that could drift.
- `Origin::Baseline` (wire value `baseline`): a query recorded while Ratel was observing but not serving retrieval — the agent chose from its own full tool list and the host captured the turn's text so the invocations that follow can be attributed to it. Ratel's own search path never produces it.
- `ToolRegistry::build_intent_graph` builds a graph by stepping through a trace log — the offline half of baseline seeding. Every distinct query is embedded up front so clusters form at the **dense** tier, exactly as the live path would; a model-free replay clusters lexically and a later `rebuild_intent_graph` cannot undo that, since it replaces centroids without revisiting cluster boundaries. Pairing is per session over the log's own order (never re-sorted — `ts` stamps observations, it does not order them), so interleaved sessions sharing one graph cannot cross-pair. One call populates both `tools` and `skills` edges. Returns a detached graph; attaching stays an explicit `set_intent_graph` call.

  A graph built this way **is** the graph live learning would have grown from the same events, asserted by test rather than claimed. It diverges in exactly one place, also pinned by test: the credit that makes a fanned-out question *one* observation is a single slot on the graph keyed by query text, which is enough live (identical text from concurrent sessions is rare there) but not in a replay, where sessions interleave by construction and popular questions repeat verbatim. Replay tracks that credit per session — which it can do exactly because it knows the session id — so it counts interleaved sessions that live learning would have under-counted.
- `ObservationPolicy` on `UsageLearner`, via `with_policy` — which search origins open an observation window (`OriginFilter`) and whether what is learned is marked as seeded (`Provenance`). There is deliberately no knob for *which* event confirms an observation: a trace records which capability was called, never whether calling it was right, so the choice is the only signal there is. `Default` reproduces existing behavior exactly and `UsageLearner::new` delegates to it, so nothing changes unless a policy is passed. A rejected search is *ignored*, not cleared, so an unrelated internal search between a captured query and its invokes cannot discard the turn's evidence.
- `seeded_support` on `Intent` (`protocol/v1`): how many of a cluster's `support` observations came from a seeding pass rather than live traffic. Provenance only — nothing reads it during ranking, and two graphs differing only here rank identically and compare equal. Bumped in lockstep with `support`, so one fanned-out question stays one observation. Omitted from the wire form when zero, so a live-only graph serializes byte-identically to one produced before the field existed; a value exceeding `support` is rejected on load.
- `dropped` on the `usage_boost` trace event: how many capability ids a matched cluster remembers that the registry no longer defines, so they were filtered out of the arm. Previously a cluster whose every edge had left the catalog emitted an event byte-identical to a query that matched nothing — the two are different problems (catalog drift vs a coverage gap) with different fixes, and `intent: Some(_)` with `promoted: 0` and `dropped > 0` now tells them apart. Ranking is unchanged: dropping ids the agent cannot invoke is still correct, and only an armed outcome reaches the fusion. Older log lines without the field replay as `dropped: 0`.

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
