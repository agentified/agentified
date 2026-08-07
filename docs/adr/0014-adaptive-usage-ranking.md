# 14. Adaptive usage ranking: an online usage arm over the existing fusion

Date: 2026-07-20

## Status

Accepted

Builds on [ADR-0004](0004-retrieval-and-tool-selection.md) (the `searchable_text`
projection), [ADR-0011](0011-selectable-retrieval-methods.md) (the three methods and RRF),
and [ADR-0007](0007-telemetry-two-streams.md) (the local trace stream, whose sink seam this
subscribes to). Ratifies the 2026-07-06 adaptive-ranking brief with the amendments below.

## Context

Every ranker in the engine scores **text similarity only** — BM25 over the flattened
projection, dense cosine, or their fusion. None of them has memory. A query like
`"why is the build broken"` ranks a `docker_build` tool first on the token *build*, and it
will do so on the thousandth repetition, in a project where every such query has ended in
`gh_run_list`.

The engine already emits the evidence that would fix this. `TraceEvent::Search` carries the
query and its ranked hits; `TraceEvent::InvokeStart` carries the tool that was actually
called. Adjacent, in one session, they are a user-confirmed relevance judgment — the search
literature's impression/click pair — and today nothing consumes them.

The 2026-07-06 brief proposed a "co-usage boost" over an intent graph, extracted by an LLM,
synced down the catalog loader seam. Specifying it surfaced that the loader seam does not
exist in code, that the LLM buys nothing load-bearing, and that the brief's multiplicative
combination is arithmetically worse than the additive one the crate already implements.

## Decision

**A third RRF arm, ranked by what users actually invoked after semantically similar
queries, learned online.**

### The read model

Queries are clustered; each cluster carries weighted edges to the capabilities invoked
after its members. A cluster is a row: `members` (the match key), an optional `centroid`,
a `label`, `terms`, `support`, and `tools` / `skills` edge maps.

- **Edges come from invocations, never from retrievals.** Recording retrieved ids would
  memorize the ranker's own output — including its errors — and reinforce them on every
  update. Invocations are the only point at which information the ranker does not already
  hold enters the system.
- **Online.** Clusters are created and grow as queries arrive; a cluster may boost from
  its first confirmed pair. There is no build step.
- **The learner clusters at whatever tier the registry runs.** A `TraceEvent::Search`
  carries the query text, not its embedding, so the sink alone could only cluster on
  words. But a semantic/hybrid registry has *already embedded the query* for its own
  ranking, so it stashes that vector on the graph (a `Mutex` slot, written under the read
  lock) and the learner grows a real centroid from it — free, since the embedding was
  computed anyway. A `Bm25` registry loads no model and its clusters carry no centroid,
  reaching repeats and near-repeats only. The slot is keyed by query text: sessions share
  a graph, so a clobbered slot degrades to lexical clustering rather than attaching one
  session's embedding to another's question.
- **Support-scaled, not support-gated.** The arm's weight is `W · min(1, support/3)`, so
  one observation nudges and three or more get full weight. A batch design could filter
  weak clusters before use; an online one cannot, so the ramp does that job without making
  the user wait.
- **An observation is a *search* that was acted on, not an invoke.** One search that leads
  to three tool calls adds three edges but counts once: the agent used three capabilities
  to answer one question. Counting invokes let a single query reach full weight
  immediately, defeating the ramp — which is the normal shape of `search_capabilities`,
  not an edge case. This holds across catalogs too: `search_capabilities` fans one query
  to the tool and skill catalogs, each with its own learner, so the credit that makes it
  *one* observation lives on the shared graph, not per-learner (`CreditSlot`). That credit
  is keyed by query text with a single slot, so it is exact for the fan-out but under-counts
  two *concurrent* sessions that ask the same text and resolve different catalogs into one
  cluster — a rare, conservative trade accepted over threading a per-turn correlation id
  through the trace events; see `CreditSlot`.
- **Clusters age out.** The arm weight is `W · min(1, support/3) · recency`, where recency is
  `1` for a grace period (90d) after a cluster's last use and then halves every half-life
  (90d), evaluated against the newest observed event — so a topic that falls out of use fades
  and, past a floor, is evicted, while support stays a pure count (confidence). Recency is on
  the *arm weight*, not the edges: edge decay was invisible because RRF fuses on rank. Members
  are capped per cluster. Together these bound relevance drift and the unbounded growth of
  memory and cluster count (which is the search cost). The constants are unswept defaults.
- **Edge weights are plain invocation counts.** No recency term: only their *order* within
  a cluster reaches the fusion, so a decay factor applied uniformly to a cluster changed
  nothing that ranking could observe (see Rejected).

### The scorer

`score(id) = Σ_arms w_arm · 1/(RRF_K + rank_arm(id))`, with `w_bm25 = w_dense = 1` and
`w_usage = W · min(1, support/3)`, `W < 1`. The arm is **absent**, not zero-weighted, on a
miss, so an unmatched query ranks bit-identically to a registry with no graph.

`W < 1` is deliberate: at equal rank a capability the current query lexically matched
outranks one only history supports. The arm still promotes a low-ranked capability past
BM25's rank-0 (it contributes from both arms), but it cannot conjure one the base ranker
did not retrieve at all.

### Two similarity tiers

Online clustering needs a query-to-cluster similarity at search time.

| Method | similarity | reach |
|---|---|---|
| `Semantic` / `Hybrid` | cosine against `centroid` | groups phrasings that share no words |
| `Bm25` | best Jaccard overlap with any single member | repeats and near-repeats only |

On semantic/hybrid the marginal cost is zero — the dense arm already embedded the query for
its own ranking. On `Bm25` no model is loaded at any point, so ADR-0011's model-free
default is preserved. The Bm25 tier is genuinely weaker and is documented as such.

**Lexical scoring is per member, never against their union.** A union only grows, so scoring
against it let a mature cluster recognize most of the vocabulary, absorb unrelated asks, and
grow further — 100 distinct topics measured as 18 clusters, once as 1. Per-member Jaccard
keeps a cluster exactly as discriminating on its 200th member as on its first.

The cost is recall, and it is the right trade. Two queries sharing one word out of two are
structurally identical whether they are the same question phrased differently or two
unrelated asks; no word-overlap rule can accept one and reject the other. This tier rejects
both, because **a false merge degrades ranking while a false split only misses a boost**.
Bridging distant wording is what the dense tier is for.

Because `members` is the match key and `centroid` is optional, a graph grown under one tier
is consumable by the other. **The tier is chosen from what the graph carries, not from the
caller's search method**: a semantic catalog handed a centroid-less graph matches it
lexically rather than seeing nothing. Without that fallback the in-process learner's own
output would be invisible to the very methods it is meant to improve.

### Embedding-model changes

Centroids are model-specific — cosine only means anything against a query embedded by the
*same* model. The graph therefore records its model as an optional `model` fingerprint
(absent for a lexical graph), and a semantic/hybrid search compares it against the active
model. On a mismatch — a different output dimension, or the same width with a different
identity (a fine-tune; a length check cannot catch this) — the usage arm **pauses** (base
ranking is untouched) rather than cosine across incompatible spaces, and the learner
**freezes** centroid growth so it never blends two spaces. This deliberately differs from the
corpus cache, which hard-errors: the corpus cannot produce dense results without matching
embeddings, but the usage arm has a valid fall-through (no boost), so breaking search over a
stale *enhancement* would be worse than the problem.

The mismatch is surfaced three ways: a `TraceEvent::UsageModelMismatch` (structured, always),
a one-time SDK stderr warning (default on, `warnOnModelMismatch: false` to suppress), and an
`experimentalAdaptiveRankingStatus` the app can gate on. `experimentalRebuildIntentGraph()` re-embeds the graph's
members under the current model and restamps — members, support, and edges are
model-independent, so all learning survives; only the centroids move.

Recovery is **explicit by default**: the arm stays paused until the caller invokes
`experimentalRebuildIntentGraph()`, because a rebuild is an embedding pass (cost, can fail, mutates the
graph and bumps `rev`) and the paused fall-through is safe in the meantime. For zero-touch
recovery, `experimentalEnableAdaptiveRanking(graph, { rebuildOnModelChange: true })` (`rebuild_on_model_change=True`
in Python) opts in: the next dense search re-embeds the graph before searching, then proceeds.
It lives on the catalog, not `IntentGraph` — the graph is a pure wire type with no embedder;
only the catalog owns the model. Recovery is lazy (dense search is async-only, `enable` is
sync), so `experimentalAdaptiveRankingStatus` reads `paused` until that first dense search; a failed
rebuild raises the same `EmbedderError` the dense query itself would. Off by default keeps the
expensive, fallible operation from being implicit.

### Persistence, change-tracking, and forward compatibility

The graph is in-process; the caller owns storage (`toJson`/`fromJson`). That is deliberate —
Ratel runs with no infra and must not pick a backend (file, SQLite, the app's own DB, S3).
But delegating storage without the primitives to do it safely is a trap, so the wire form
carries a **monotonic `rev` counter**, bumped once per mutation (a confirmed observation, a
rebuild) and never read during ranking. It answers two questions the caller otherwise cannot:

- **Save-when-changed.** Learning happens on every confirmed invoke, in memory; a crash loses
  whatever was not persisted. Rather than serialize on every invoke (wasteful) or guess
  (lossy), the caller snapshots `rev` after each save and writes again only when it differs.
- **Stale-base detection.** Two writers that both load, learn, and save would silently clobber
  each other. Before overwriting a stored graph, the caller compares its `rev` to the one it
  loaded; a higher value means another writer moved ahead. **Single-writer is the supported
  model** — `rev` makes a collision *detectable*, not merged. Automatic merge is rejected: it
  would mean owning the storage layer, the very thing delegation avoids.

Forward compatibility is a contract, not a hope: within schema `v: 1` fields are **additive**,
a consumer **ignores unknown fields** (no `deny_unknown_fields`; `additionalProperties: true`
in the schema), and an unrecognized `v` is a **typed error** (`UnsupportedVersion`), never a
panic or a silent degrade. An older graph missing a newer field (`model`, `last_ts`, `rev`)
loads with a safe default. Conformance vectors pin all three guarantees so a future change
cannot quietly regress them.

### Opt-in, per registry

A usage arm turns `SearchHit.score` from a BM25 score into an RRF score. ADR-0011 promises
`search` / `search_with_origin` keep BM25 behavior byte-for-byte, so the graph attaches per
registry and those entry points are untouched — the same containment ADR-0011 used for
fallibility.

Because that makes `score` switch scale *between calls* on one catalog (raw when a query
matches no cluster, RRF when it matches one), each hit carries `rank` (0-based position,
scale-invariant) and `fused` (whether `score` is an RRF score). Callers order/threshold on
`rank` and branch on `fused`; `score` is a within-list hint only. Deliberately no normalized
confidence — BM25, cosine, and RRF have no honest common scale, so any single number would
be fabricated.

The SDK entry points ship behind an `experimental` prefix —
`experimentalEnableAdaptiveRanking` / `experimental_enable_adaptive_ranking` and its
`rebuild` / `disable` / `status` siblings — per the additive-evolution convention
([AGENTS.md](../../AGENTS.md)): a new, unproven capability is marked until it earns promotion,
then the prefix is dropped. The marker sits on the *behavior* entry points only. `IntentGraph`
and its `to_json` / `from_json` / `rev` keep stable names — it is a `protocol/v1` wire type
whose versioning already governs its evolution, and the experimental methods are the sole way
to activate it, so they gate all use on their own.

### Where learning happens

The learner is a `TraceSink` decorator. `Search` and `InvokeStart` arrive through separate
API calls and the registries have no session concept, but sinks do. ADR-0007 already frames
the sink as the subscription seam ("rerankers, suggestion analysis, and inspection
subscribe to different cuts of the same producer"), so this needs no new plumbing.

### What is open source

The **format** is specified in [`protocol/v1`](../../protocol/v1/README.md) beside
`CatalogSkillWire`, and the local learner ships in `ratel-ai-core` under Apache-2.0. Ratel
Cloud is a second producer of the same format from hosted traces — the "usage-ranking read
model" ADR-0003 named as the first candidate to open. Labels are medoid + c-TF-IDF, both
counted from the member strings; no model, no key, no vendor in the OSS path. Cloud's
LLM-extracted intents populate the same `members` field.

## Consequences

- Retrieval gains a signal no amount of description-writing can supply: what users chose
  when the ranker was wrong. The tool author's vocabulary stops being the only lever
  (ADR-0004).
- **Ranking becomes order-dependent.** The same events in a different order produce a
  different graph. This is a real reversal of the determinism posture `Bm25Index::search` and
  `sort_and_truncate` hold elsewhere, accepted as the cost of learning without a build
  step; replaying the JSONL trace log is the escape hatch when a reproducible artifact is
  needed (CI, benchmarks, bug reports).
- **The graph accumulates from a stream ADR-0007 permits to drop events.** A dropped
  invoke is lost permanently rather than recovered on the next rebuild. Replay is the
  repair path.
- Persistence is the caller's: `toJson`/`fromJson` plus a monotonic `rev` for save-when-changed
  and stale-base detection (single-writer supported; no built-in merge). Replaying
  `~/.ratel/telemetry` at construction is an alternative cross-session path, and needs no new
  storage because `JsonlSink` already writes that log.
- `members` holds raw query text. Whatever persists it must match the `0600` treatment
  `JsonlSink` already applies.
- A feedback loop is inherent — boosting used capabilities makes them more used. `W < 1`
  and the support ramp bound it; they do not remove it.
- **Clusters fade and evict, but `support` only rises.** A topic that falls out of use
  loses arm weight through the recency multiplier and, past the eviction floor, is dropped
  (see Decision). `support` itself is a pure count that never decreases, so it records
  confidence, not currency — staleness is handled on the arm weight and cluster lifetime,
  never on edge magnitudes.

## Rejected

- **Recency decay on edge weights** (`Σ 2^(−Δt/half_life)`, built and then removed). It
  discounted old invocations correctly, but the fusion consumes *rank position*: decaying
  every edge in a cluster by the same factor left their order unchanged, so ranking could
  not observe it. It only reordered when two capabilities in one cluster differed sharply
  in recency — a narrow case bought with a wire field, a tuning constant, and a time
  parameter threaded through the learning path. The staleness it appeared to address (a
  whole cluster going cold) it never addressed, since `support` and cluster lifetime are
  untouched by it.

- **Multiplicative fusion with intent similarity as a factor** (the brief's
  `BM25/dense × intent-similarity × co-usage`): a `W·(cos−τ)/(1−τ)` ramp scores 0.53 at
  cos 0.78 and ranks the correct capability *below* where no boost at all would leave it.
  Real match similarities occupy 0.70–0.90, so a ramp normalized to 1.0 spends its range
  where nothing lives. Similarity is a gate; the arms combine additively.
- **Recording retrieved capabilities as edges**: self-reinforcing, adds no information.
- **An LLM for intent extraction or labeling in the OSS path**: labels are cosmetic —
  identical retrieval results if every label were `intent_17`. The crate is encoder-only
  (`candle_transformers::models::bert`); adding a generation path, and a download or an
  API key, to produce display strings is not a trade worth making. Medoid + c-TF-IDF are
  counted from the members and cannot hallucinate. Cloud may label however it likes; the
  format carries a plain string.
- **An optional OpenAI-compatible labeling endpoint**: the precedent in
  `embedding_config.rs` does not transfer, because embedding quality is load-bearing and
  label text is not. An OSS artifact that reaches for a service to produce cosmetic strings
  reads as a hole where a product should be.
- **Tool↔tool co-usage** (the brief's "co-usage"): a different signal from the intent→tool
  edges the impression/click pairing yields. Possible fourth arm later; not carried by
  this decision.
- **Batch rebuild**: reproducible and able to sweep thresholds corpus-wide, but a query's
  evidence would not affect ranking until the next build. Immediacy was ratified over
  determinism; replay preserves the batch path where it is needed.
- **Shipping the graph down the catalog loader seam** (the brief's "no new machinery"): no
  `RATEL_URL` or `CatalogSource` exists in `src/` — the seam is specified, not built.
  Revisit when PSKS-5 lands.
