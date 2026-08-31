# 11. Selectable retrieval methods: BM25, semantic, hybrid

Date: 2026-07-06

## Status

Accepted

Builds on ADR-0004 (retrieval: the stable `searchable_text` projection and BM25 scorer), which
anticipated a semantic ranker merging with the lexical signal. ADR-0006 (native FFI bindings)
and ADR-0007 (telemetry schema) frame how the choice surfaces through the SDKs and traces.

## Context

BM25 is the shipped ranker. Two parallel spikes each replaced it wholesale — one with dense
(semantic) retrieval over a local embedding model, one with a BM25+dense hybrid. Neither
coexistence nor selection existed: each was the single engine on its branch.

Users need all three at once, choosing per deployment. BM25 must stay the default and stay
model-free: it needs no download, never fails, and many callers neither want nor can run a
local model. Semantic and hybrid must be opt-in, and their one cost — loading the embedding
model — must not leak into a BM25-only process.

## Decision

**A `SearchMethod` enum — `Bm25` (default) | `Semantic` | `Hybrid` — selects the ranker.** It
is chosen per catalog (a construction-time default) or per call (an explicit override); the
override wins. Across the SDKs the identifier is the string `"bm25" | "semantic" | "hybrid"`,
parallel to `SearchOrigin`.

- **BM25** is unchanged from ADR-0004 (`Bm25Index::search`, `k1 = 0.9`, `b = 0.75` since
  ADR-0023, which also narrowed the projection those scores are computed over) and stays
  the default. The legacy `search` / `search_with_origin` entry points keep their infallible
  `Vec<SearchHit>` signature and BM25 behavior byte-for-byte — upgrading callers need no code
  change. The index is cached per registry (`Bm25Cache`) and rebuilt in full on the first
  search after a corpus mutation, so scores always equal a fresh build's.
- **Semantic** embeds the query with a local `BAAI/bge-small-en-v1.5` model (pure-Rust Candle,
  pinned revision, CLS-pool + L2-normalize) and cosine-ranks it against embedded tool/skill
  text (`dense_search`). The same `searchable_text` projection feeds it. (The model is the
  default; **ADR-0012** makes it configurable per catalog — HuggingFace/local/endpoint.)
- **Hybrid** runs the BM25 and dense arms to a fixed retrieval depth and fuses their rankings
  by **Reciprocal Rank Fusion** (`RRF_K = 60`), no cross-encoder reranker. RRF fuses on rank
  position, so BM25's unbounded scores and cosine's `[-1, 1]` never need reconciling.
- **Fallibility is confined to the new path.** A method-carrying `search_with_method` returns
  `Result<_, EmbedderError>`; `Bm25` is always `Ok`, while `Semantic`/`Hybrid` surface a failed
  model load (network, cache, underpowered machine) as a catchable error — a Python
  `RuntimeError` / a thrown JS error at the SDK edge.
- **The embedding cache is incremental, transactional, and in-process.** It is an **id-keyed**
  map of per-item vectors: `register` inserts metadata by id (replacing an existing id in place
  and invalidating its stale vector), and `build_embeddings()` embeds every id not currently
  cached. Each batch is fully validated before one atomic commit, so a failed incremental build
  adds nothing. `rebuild_embeddings()` recomputes the full corpus and atomically swaps it in;
  failure preserves the prior complete cache. A shared/exclusive operation guard keeps query
  validation and ranking in one vector space while a rebuild waits. Core
  `register(&mut self, tool) -> ()` stays infallible and model-free. The one-time model load emits
  a `TraceEvent::EmbedderLoad` flagging a slow (possibly underpowered) or failed load.
- **SDK registration folds embedding in.** An SDK caller `await register(...)`s one item or
  many; on a semantic/hybrid catalog that embeds the batch on a worker thread (so a model /
  endpoint failure surfaces from `register`), while a BM25 catalog registers metadata only.
  Model loading, downloads, HTTP, corpus embedding, and dense queries run on worker threads at
  the TypeScript/Python boundary; a model or dimension change is recovered by constructing a new
  catalog. Synchronous SDK `search` is BM25-only; `searchAsync` / `search_async` supports all
  three methods. Rust core APIs remain synchronous, exposing the incremental `build_embeddings`
  and atomic `rebuild_embeddings` primitives the SDK drives internally.
- **A search never embeds the corpus.** A semantic/hybrid search over a corpus whose cache is not
  fully built returns `EmbedderError::EmbeddingsNotBuilt` (a catchable `RuntimeError` / thrown error) —
  it does *not* silently embed inside the search path. So a BM25 catalog handed a per-call
  `"semantic"` override errors with a remediation hint to build, then use async search, rather
  than incurring a one-off slow search. The guard loads no model.

## Consequences

- The default stays lightweight and infallible; the ML dependency (Candle, tokenizers with the
  pure-Rust `fancy-regex` backend, `hf-hub`) is compiled in but never exercised unless a caller
  opts into semantic/hybrid. The `searchable_text` contract (ADR-0004, experimentally extended
  by ADR-0023) is shared, so all
  three engines rank the same projection.
- Capability tools inherit the catalog's construction-time default and await async search. MCP
  ingestion registers metadata only, so several upstreams can be ingested before one batched
  embedding build.
- The ML native deps (`ring`, Candle's `esaxx-rs`) do not zig-cross-compile, so the linux-arm64
  SDK binaries build on a native ARM64 runner instead of `--use-napi-cross`.
- Rejected: a compile-time feature flag to pick the engine (the earlier spikes' shape) —
  runtime selection is required for per-call and per-catalog choice, and for one binary serving
  all three. Rejected: embedding at `register` — it makes registration fallible and can block a
  Node event loop or Python interpreter on model/HTTP work. Explicit asynchronous build keeps
  registration metadata-only in every mode. Rejected: full-corpus re-embed on `register` (the first cut of
  this ADR) — the incremental id-keyed cache re-embeds only ids not currently cached (newly
  registered, or invalidated when a re-register replaces an id). Rejected:
  score-normalization fusion for hybrid — RRF needs no per-arm score calibration.
