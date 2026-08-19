# 14. Adaptive usage ranking: an online usage arm over the existing fusion

Date: 2026-07-20

## Status

Accepted

Builds on [ADR-0004](0004-retrieval-and-tool-selection.md) (the stable `searchable_text`
projection), [ADR-0011](0011-selectable-retrieval-methods.md) (the three methods and RRF),
and [ADR-0007](0007-telemetry-two-streams.md) (the local trace stream, whose sink seam this
subscribes to). Ratifies the 2026-07-06 adaptive-ranking brief with the amendments below.

Amended 2026-08-13 by [ADR-0020](0020-runtime-events-lane.md): the sink is now a fan-out
subscription seam. The learner remains an internal consumer and cannot be replaced by attaching
another subscriber.

Amended 2026-08-17: a graph may be **seeded offline from a captured baseline** before ranking is
switched on — see [Seeding from a baseline capture](#seeding-from-a-baseline-capture). Online
learning remains how ranking learns while serving; the "no build step" claim below is narrowed
to the serving path, not the bootstrap.

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
  its first confirmed pair. There is no build step **on the serving path** — nothing has to
  be rebuilt for a query's evidence to reach ranking. A graph may still be *seeded* once,
  offline, before serving begins (see [Seeding from a baseline
  capture](#seeding-from-a-baseline-capture)); that is a bootstrap, not a build the serving
  path waits on.
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

The learner consumes `Search` and `InvokeStart` through the core fan-out specified by
[ADR-0020](0020-runtime-events-lane.md). It is composed as an internal subscriber/decorator,
not installed into a replace-only sink slot: adding a JSONL, SDK, or Cloud subscriber MUST NOT
silently disable learning. The public queue and callback machinery stays outside the learner,
so stalled subscribers cannot block it. Adaptive ranking remains experimental; its decoration
may be simplified if that is required to keep fan-out robust.

### Seeding from a baseline capture

Online learning has two cold-start problems, both of which this ADR states elsewhere and
neither of which the ramp addresses. A fresh deployment has no evidence, so it gets no boost
until pairs accumulate. And once Ratel *is* ranking, what the agent invokes is partly Ratel's
own doing — the feedback loop under Consequences. Both are the same shape: the cleanest
evidence available is what an agent invoked while Ratel was **not** in the retrieval path, and
until now there was no way to record it.

**A host may capture turns while Ratel serves nothing, build a graph from that log offline,
inspect it, and only then enable ranking.**

- **A fourth origin, `Origin::Baseline`.** Ratel's own search path never produces it. The host
  names the turn's query text — a run where nobody searches has no query, and a graph is keyed
  on query text — and Ratel is a recorder for the duration.
- **The turn is the unit, and it is buffered.** Nothing reaches the log until `record()`, so
  declining to call it is how a host drops a turn it would not want learned from. One turn
  stays **one** observation, holding the line the `CreditSlot` bullet draws: a search with
  three invocations recorded as three turns would count the query three times and defeat the
  ramp.
- **Building embeds every distinct query up front**, so clusters form at the **dense** tier —
  the tier the live path would have grown them at. A model-free replay clusters lexically, and
  `rebuild_intent_graph` cannot repair that later: it replaces centroids without revisiting
  cluster boundaries. Getting the tier right is therefore a property of the build, not
  something a caller can fix afterwards.
- **The pairing rule exists once.** The live learner and the offline replay share one
  `classify` step. They differ only in where pending state lives — a per-session learner holds
  a slot, a replay holds a map keyed by `session_id` — because a log interleaves sessions by
  construction and a single slot would cross-pair them. Replay walks the log in **its own
  order, never re-sorted**: file order is arrival order, and sorting by `ts` would produce a
  graph the live path could not have grown, since cluster membership depends on which clusters
  existed when each query arrived. `ts` stamps observations; it does not order them.
- **Equivalence is asserted by test, not by this document** — a graph built from a log *is*
  the graph live learning would have grown from the same events. It diverges in exactly one
  place, and that divergence is also pinned: interleaved sessions asking identical text, where
  the live path's single global credit slot under-counts (the trade the `CreditSlot` bullet
  accepts) and replay, knowing the session, does not.
- **Policy is a closed set, and applies to both paths.** `origins` (`any` | `agent` |
  `baseline`) selects which searches may open an observation window; `provenance` (`live` |
  `seeded`) selects whether what is learned is marked as seeded. What counts as evidence must
  not depend on which path produced the graph, so both take the same policy. Building defaults
  to `baseline`/`seeded` — that is what an offline build *is* — while enabling live learning
  keeps `any`/`live`, so existing behavior is unchanged. `direct` is a valid search origin but
  deliberately **not** a filter: learning only from searches your own plumbing made is
  learning from your plumbing.
- **A rejected search is ignored, not cleared.** One of Ratel's own internal searches landing
  between a captured query and its invocations must not discard the turn's evidence.
- **Provenance is recorded, and is inert.** `seeded_support` on a cluster counts how many of
  its `support` observations came from a seeding pass. It is `protocol/v1`-additive, omitted
  when zero, and **ranking must not read it**: two graphs differing only here rank identically
  and compare equal. Its use is operational — after the flip, `support` grows while
  `seeded_support` stays put, so the gap says how much still rests on the baseline.
- **Building never enables ranking.** The returned graph is detached; attaching stays an
  explicit call. Inspecting before the flip is the whole point of seeding first, so the API
  must not make enabling the default outcome of building.
- **The destination may belong to the host.** A single process holding the turn open is the
  wrong model for a process-per-request server across N instances, which is the first real
  deployment this targets: the search and the invocation that follows are different requests,
  on possibly different machines. Two seams follow, both specified in
  [ADR-0007](0007-telemetry-two-streams.md) — a closure sink (`FnSink`, the SDKs' `"callback"`)
  so the host writes envelopes to storage it already runs, and a **whole-turn** recording call
  for a turn the host reassembled itself. The whole-turn call is not mere ergonomics: the
  chained builder permits an `await` between invocations, so concurrent turns can interleave
  their events in one sink and break the search-then-invoke adjacency the pairing depends on.

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
  and the support ramp bound it; they do not remove it. Seeding from a baseline gives the
  loop a starting point Ratel had no hand in, which bounds where it *begins*; it does not
  bound where it goes.
- **Every invocation is evidence, and seeding assumes it is good evidence.** Nothing in a
  trace says whether a turn went well — a trace records which capability was called, never
  whether calling it was right — so nothing is filtered, and the quality gate is the host's:
  seed from an agent you already trust. The exposure is bounded by the same property that
  makes edge decay pointless (see Rejected): edge weights set only *order* within a cluster,
  so one wrong invocation of a capability the base ranker already favours is close to free —
  and, symmetrically, more good data does not dislodge it.
- **Readiness is measured by coverage, not by cluster count.** Clusters, observations, and
  support all rise whether or not the graph generalises; only held-out queries that match a
  cluster say it does. The threshold behind the support ramp is not exposed, so a caller
  currently hardcodes it — a first-class readiness surface is still owed.
- **Rebuilding is O(whole log).** Fine as a nightly or one-time seed, wasteful per turn. The
  offline path is a bootstrap and a repair path, not a serving-time operation.
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
- **Batch rebuild _as the serving model_**: reproducible and able to sweep thresholds
  corpus-wide, but a query's evidence would not affect ranking until the next build.
  Immediacy was ratified over determinism. This rejection is about what serving *waits on*
  and still stands. The offline build added by the 2026-08-17 amendment is the batch path
  this entry pointed at ("replay preserves the batch path where it is needed"), scoped to a
  bootstrap and a repair path: it runs before ranking is enabled, or out of band, and no
  search ever blocks on it.
- **Shipping the graph down the catalog loader seam** (the brief's "no new machinery"): no
  `RATEL_URL` or `CatalogSource` exists in `src/` — the seam is specified, not built.
  Revisit when PSKS-5 lands.
