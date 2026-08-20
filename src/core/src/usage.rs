//! The usage-ranking read model: clusters of past queries, each carrying
//! weighted edges to the capabilities users actually invoked after them
//! (ADR-0014).
//!
//! A query is matched to at most one cluster, and that cluster's capabilities
//! become an extra ranked arm for [`crate::fusion::rrf_fuse_weighted`] beside
//! BM25 and dense retrieval. Two things follow from that and are easy to lose:
//!
//! - **Only the arm's *order* is used.** Edge weights choose the order and are
//!   then discarded; RRF fuses on rank position, so a weight never has to be
//!   reconciled with a BM25 or cosine score.
//! - **A miss produces no arm at all**, not a zero-weighted one. A query that
//!   matches nothing ranks bit-identically to a registry with no graph.
//!
//! Matching has two tiers, because the graph must work on a `Bm25` catalog that
//! has no embedder ([`crate::SearchMethod`], ADR-0011):
//!
//! - [`IntentGraph::arm_dense`] — cosine against a cluster's stored centroid.
//!   Groups phrasings that share no words. Used by semantic/hybrid, where the
//!   query embedding was already computed for the dense arm, so it costs nothing.
//! - [`IntentGraph::arm_lexical`] — token overlap against a cluster's member
//!   bag. No model is ever loaded. Reaches repeats and near-repeats only; it
//!   cannot connect "why is the build broken" to "did CI pass".
//!
//! The wire shape is `protocol/v1/schema/intent-graph.schema.json`; this is its
//! consumer. An edge weight is a plain count of confirmed invocations: it orders
//! the arm and nothing more, since RRF then fuses on rank position.

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::fusion::sort_and_truncate;

/// Fraction of the usage arm's full weight granted per unit of support, capped
/// at 1.0 once `SUPPORT_FULL` observations agree. One confirmed observation
/// nudges the ranking; it must never dictate it, or a single misclick becomes
/// policy (ADR-0014).
pub(crate) const SUPPORT_FULL: u32 = 3;

/// The usage arm's full weight, relative to the BM25/dense arms at 1.0.
///
/// **Deliberately below 1.0**: at the same rank, a capability the query
/// lexically matched outranks one only usage history supports. The arm still
/// promotes a deeply-ranked capability past another arm's top hit, because that
/// id accumulates from both arms — sub-unit damps the arm without disabling it.
/// Like `BM25_K1` / `RRF_K`, this is fixed tuning, not a public knob (ADR-0004).
pub(crate) const USAGE_WEIGHT: f32 = 0.5;

/// Default for [`ClusterPolicy::similarity`].
pub(crate) const TAU_COSINE: f32 = 0.70;

/// Default for [`ClusterPolicy::coverage`].
pub(crate) const COVERAGE_FRACTION: f32 = 0.5;

/// How similar a query must be to a cluster, and to how much of it, before it
/// joins — the two numbers that draw every cluster boundary.
///
/// **Configurable, and recorded on the graph that was clustered under it.**
/// The threshold is model-dependent: an endpoint catalog can carry any embedding
/// model, and a cosine of 0.70 does not mean the same thing on two of them. It is
/// corpus-dependent too — a narrow catalog and a broad one want different
/// granularity. Recording it is what keeps that safe: two producers at different
/// settings would otherwise disagree about what a cluster means while both
/// claiming the same protocol version (ADR-0014).
///
/// `#[non_exhaustive]` so a later dimension is additive, which is also why the
/// builders exist — such a struct cannot be written as a literal outside this
/// crate at all, `..Default::default()` included.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ClusterPolicy {
    /// Minimum cosine between a query and a **single cluster member** for that
    /// member to count toward [`Intent::coverage`]. Also the centroid prefilter,
    /// which is what makes admission provably *the centroid rule AND coverage* —
    /// a strict subset, never looser. Default `0.70`.
    pub similarity: f32,
    /// Fraction of a cluster's vector-bearing members a query must match before
    /// it joins — a majority by default, floored at two members.
    ///
    /// Matching one member is single-link chaining: A joins because of B, and the
    /// cluster grows into whatever B happened to bridge to. Matching an average is
    /// worse, because the average of two intents resembles neither. Counting
    /// members is what distinguishes "close to this whole cluster" from "close to
    /// one thing in it", and a fraction rather than a count keeps the test
    /// comparable across clusters of different sizes. Default `0.5`.
    pub coverage: f32,
}

impl Default for ClusterPolicy {
    fn default() -> Self {
        Self {
            similarity: TAU_COSINE,
            coverage: COVERAGE_FRACTION,
        }
    }
}

impl ClusterPolicy {
    /// Set how similar a query must be to a single member.
    #[must_use]
    pub fn with_similarity(mut self, similarity: f32) -> Self {
        self.similarity = similarity;
        self
    }

    /// Set the share of members it must be that similar to.
    #[must_use]
    pub fn with_coverage(mut self, coverage: f32) -> Self {
        self.coverage = coverage;
        self
    }

    /// Whether this is the built-in policy. `skip_serializing_if` reads it, so a
    /// graph that never moved off the defaults stays byte-identical on the wire
    /// to one written before the field existed.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether both values are in `(0, 1]` — the range a cosine and a fraction
    /// can mean anything in.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        (0.0 < self.similarity && self.similarity <= 1.0)
            && (0.0 < self.coverage && self.coverage <= 1.0)
    }
}

/// How many of a cluster's most recent members keep their query vector in memory.
///
/// Coverage over a bounded recent sample is the same statistical test as coverage
/// over all 50, at a third of the memory: `Option<Vec<f32>>` at [`MEMBER_CAP`] and
/// 384 dims is ~77 KB per cluster, this caps it near ~25 KB. Older members keep
/// their text and tokens — only the vector is dropped.
pub(crate) const VECTOR_RETAIN: usize = 16;

/// Minimum Jaccard overlap between a query and a cluster's closest single
/// member for a lexical match — `|q ∩ m| / |q ∪ m|`.
///
/// Scored per member rather than against the members' union: a union only
/// grows, so union scoring let a mature cluster absorb unrelated asks and grow
/// further still. Per-member scoring reaches repeats and near-repeats, which is
/// this tier's documented ceiling (ADR-0014) — distant wording is the dense
/// tier's job.
pub(crate) const TAU_LEXICAL: f32 = 0.5;

/// How many c-TF-IDF terms a cluster's display label carries.
const MAX_TERMS: usize = 5;

const MS_PER_DAY: f64 = 86_400_000.0;

/// `skip_serializing_if` predicate for count fields that default to zero.
fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// Wire default for [`Intent::cohesion`]: a graph produced before the field
/// existed, or by a producer that does not track spread, is treated as perfectly
/// tight — which reproduces the pre-cohesion bar exactly.
fn one_f32() -> f32 {
    1.0
}

fn is_one_f32(x: &f32) -> bool {
    *x == 1.0
}

/// Wire default for [`Intent::vector_n`]: a reloaded centroid counts as a single
/// prior sample, the weight the pre-accumulator code gave it.
fn one_u32() -> u32 {
    1
}

fn is_one_u32(n: &u32) -> bool {
    *n == 1
}

/// A cluster keeps full weight for this long after its last use, then decays.
/// Recent work should not be discounted at all; only topics that have genuinely
/// gone quiet fade (ADR-0014, blocker #3).
const RECENCY_GRACE_DAYS: f64 = 90.0;

/// After the grace period, the recency factor halves every this many days —
/// gentle: a topic idle for a year still weighs ~0.12, only near-zero by ~2y.
const RECENCY_HALF_LIFE_DAYS: f64 = 90.0;

/// A cluster whose recency factor falls below this is evicted on the next
/// observation — it no longer boosts, and dropping it bounds cluster count (the
/// search cost) and memory. `0.01` ≈ idle ~2 years at the defaults above.
const EVICTION_FLOOR: f32 = 0.01;

/// Cap on members kept per cluster. Bounds the lexical token bags and per-cluster
/// memory; the centroid is a running mean and is unaffected by dropping members.
const MEMBER_CAP: usize = 50;

/// Recency weight for a cluster last touched at `last_ts`, evaluated against the
/// graph's newest observed event `now_ts`. `1.0` within the grace period, then
/// `2^(−(Δdays − grace)/half_life)`.
///
/// Measured against the newest **observed** event, not the wall clock, so the
/// graph stays a pure function of its trace log — a topic fades relative to how
/// much other activity has happened since, and an idle graph does not decay.
fn recency_factor(now_ts: u64, last_ts: u64) -> f32 {
    let dt_days = now_ts.saturating_sub(last_ts) as f64 / MS_PER_DAY;
    if dt_days <= RECENCY_GRACE_DAYS {
        return 1.0;
    }
    2f64.powf(-(dt_days - RECENCY_GRACE_DAYS) / RECENCY_HALF_LIFE_DAYS) as f32
}

/// The effective weight of the usage arm for a cluster with `support`
/// observations: `USAGE_WEIGHT · min(1, support / SUPPORT_FULL)`.
pub(crate) fn usage_weight(support: u32) -> f32 {
    let ramp = (support as f32 / SUPPORT_FULL as f32).min(1.0);
    USAGE_WEIGHT * ramp
}

/// One confirmed observation to fold into a graph — a query, and the capability
/// invoked after it.
///
/// A struct rather than positional arguments because the two booleans read
/// identically at a call site and mean very different things: one is "this
/// search was acted on", the other is "this evidence came from a seeding pass".
#[derive(Debug, Clone, Copy)]
pub(crate) struct Observation<'a> {
    /// The query text the invocation is attributed to — the cluster match key.
    pub query: &'a str,
    /// Which edge map the invoked capability belongs to.
    pub kind: Capability,
    /// The capability that was invoked.
    pub capability_id: &'a str,
    /// When it happened, epoch-millis. Records how current the graph is and
    /// drives recency; never affects ranking order directly.
    pub ts_ms: u64,
    /// Whether this is the search's **first** confirming invoke — the only kind
    /// that raises `support`. Later invokes of the same question add edges only.
    pub first_confirmation: bool,
    /// Whether this came from a seeding pass (a baseline capture or a replay)
    /// rather than live serving traffic. Recorded on
    /// [`Intent::seeded_support`]; never reaches ranking.
    pub seeded: bool,
}

/// Which edge map of a cluster to rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Capability {
    /// Rank the cluster's `tools` edges.
    Tool,
    /// Rank the cluster's `skills` edges.
    Skill,
}

/// A matched cluster's contribution to one search: the capabilities it
/// remembers, best-first, plus what is needed to weight and trace the arm.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UsageArm {
    /// Id of the cluster that matched — carried into `TraceEvent::UsageBoost`.
    pub intent_id: String,
    /// How well the query matched: cosine against the centroid on the dense
    /// tier, the best per-member Jaccard overlap on the lexical one. Both are in
    /// `[0, 1]`, but they are **different scales** — compare within a tier, not
    /// across. Reported so near-misses are visible, not just hits.
    pub similarity: f32,
    /// The cluster's observation count. Sets the confidence ramp of the weight
    /// and is reported on the trace event; the final weight also folds in
    /// recency (see [`Self::weight`]).
    pub support: u32,
    /// The arm's full fusion weight — the support ramp times the cluster's
    /// recency factor, precomputed at match time because recency needs the
    /// graph's newest-event anchor.
    pub weight: f32,
    /// Capability ids, best-first. Already filtered to ids the registry knows.
    pub ids: Vec<String>,
    /// How many of the cluster's ids were dropped because the registry no
    /// longer defines them. Never reaches the fusion — it is carried so
    /// `TraceEvent::UsageBoost` can report catalog drift.
    pub dropped: u32,
}

/// What a query's usage lookup produced.
///
/// Distinguishes the two ways a search ends up with no arm, which a bare
/// `Option<UsageArm>` collapsed into one. Both leave ranking untouched, but they
/// are different problems: [`Self::NoMatch`] means the graph does not cover the
/// question, while [`Self::AllFiltered`] means it does and the *catalog* has
/// moved out from under it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ArmOutcome {
    /// No cluster cleared the match threshold.
    NoMatch,
    /// A cluster matched, but every capability it remembers of this kind names
    /// an id the registry does not define.
    AllFiltered {
        /// The cluster that matched.
        intent_id: String,
        /// How well it matched.
        similarity: f32,
        /// Its observation count.
        support: u32,
        /// How many ids were dropped — all of them, by definition.
        dropped: u32,
    },
    /// A cluster matched and contributed ids to the fusion.
    Armed(UsageArm),
}

impl ArmOutcome {
    /// The arm, if one was produced — the view the fusion path takes, and the
    /// reason adding the outcome distinction leaves ranking bit-identical.
    pub(crate) fn into_arm(self) -> Option<UsageArm> {
        match self {
            ArmOutcome::Armed(arm) => Some(arm),
            _ => None,
        }
    }

    /// `(intent, similarity, support, promoted, dropped)` for
    /// [`crate::TraceEvent::UsageBoost`].
    pub(crate) fn describe(&self) -> (Option<String>, f64, u32, u32, u32) {
        match self {
            ArmOutcome::NoMatch => (None, 0.0, 0, 0, 0),
            ArmOutcome::AllFiltered {
                intent_id,
                similarity,
                support,
                dropped,
            } => (
                Some(intent_id.clone()),
                *similarity as f64,
                *support,
                0,
                *dropped,
            ),
            ArmOutcome::Armed(a) => (
                Some(a.intent_id.clone()),
                a.similarity as f64,
                a.support,
                a.ids.len() as u32,
                a.dropped,
            ),
        }
    }

    /// Prefer an armed outcome, then a drift report, then a plain miss — used
    /// where a dense attempt falls through to a lexical one and only the more
    /// informative of the two failures is worth reporting.
    fn or_else(self, next: impl FnOnce() -> ArmOutcome) -> ArmOutcome {
        match self {
            ArmOutcome::Armed(_) => self,
            _ => match next() {
                ArmOutcome::NoMatch => self,
                other => other,
            },
        }
    }
}

/// Sugar for the many tests that predate [`Observation`] and mean "a live
/// observation" — the default provenance. Keeps those call sites reading as the
/// behaviour they assert rather than as struct literals.
#[cfg(test)]
impl IntentGraph {
    pub(crate) fn observe_live(
        &mut self,
        query: &str,
        kind: Capability,
        capability_id: &str,
        ts_ms: u64,
        first_confirmation: bool,
    ) {
        self.observe(Observation {
            query,
            kind,
            capability_id,
            ts_ms,
            first_confirmation,
            seeded: false,
        });
    }
}

/// `Option`-shaped sugar for the tests that predate the outcome distinction.
/// They assert on *whether an arm reached the fusion*, which is exactly what
/// these expose; the `NoMatch`/`AllFiltered` split is asserted separately.
#[cfg(test)]
impl ArmOutcome {
    fn expect(self, msg: &str) -> UsageArm {
        self.into_arm().expect(msg)
    }

    fn unwrap(self) -> UsageArm {
        self.into_arm().expect("expected an arm")
    }

    fn is_none(&self) -> bool {
        !matches!(self, ArmOutcome::Armed(_))
    }

    fn is_some(&self) -> bool {
        matches!(self, ArmOutcome::Armed(_))
    }
}

impl UsageArm {
    /// This arm's fusion weight — the support ramp times recency, precomputed
    /// when the arm was built.
    pub(crate) fn weight(&self) -> f32 {
        self.weight
    }
}

/// A graph that could not be adopted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentGraphError {
    /// The bytes were not the expected JSON shape, or a value broke a semantic
    /// rule of the wire contract (e.g. a zero-support cluster, a duplicate intent
    /// id) that the shape alone cannot enforce.
    Malformed(String),
    /// The graph declares a schema version this build does not know. A consumer
    /// rejects rather than degrading, since an unknown version may have changed
    /// what the fields mean.
    UnsupportedVersion(u32),
}

impl std::fmt::Display for IntentGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntentGraphError::Malformed(e) => write!(f, "malformed intent graph: {e}"),
            IntentGraphError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported intent graph version {v} (this build reads 1)"
                )
            }
        }
    }
}

impl std::error::Error for IntentGraphError {}

/// The schema version this build reads.
const GRAPH_VERSION: u32 = 1;

/// The most recent query and its embedding, stashed by the search path so the
/// learner can grow a real centroid.
///
/// Transient scratch, **not part of the graph's value**: skipped on the wire,
/// empty after a clone, and ignored by equality — two graphs that differ only
/// here are the same graph. It lives on [`IntentGraph`] because the search path
/// and the learner share nothing else, and it is a `Mutex` so the search path
/// can write it while holding only a read lock.
#[derive(Debug, Default)]
struct PendingQuery(Mutex<Option<(String, Vec<f32>, String)>>);

impl Clone for PendingQuery {
    /// A clone starts empty: a half-finished search is not worth copying.
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for PendingQuery {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl PendingQuery {
    fn set(&self, query: &str, vector: &[f32], fingerprint: &str) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some((query.to_string(), vector.to_vec(), fingerprint.to_string()));
        }
    }

    /// The stashed vector and the fingerprint of the model that produced it, but
    /// **only if it belongs to `query`**. Reads without clearing: several invokes
    /// may follow one search, and each needs to see it.
    ///
    /// Sessions share a graph, so a concurrent search can overwrite the slot
    /// between one session's search and its invoke. Keying by the query text
    /// means a clobbered slot degrades to lexical clustering rather than
    /// attaching one session's embedding to another's question.
    fn vector_for(&self, query: &str) -> Option<(Vec<f32>, String)> {
        let slot = self.0.lock().ok()?;
        match slot.as_ref() {
            Some((q, v, fp)) if q == query => Some((v.clone(), fp.clone())),
            _ => None,
        }
    }
}

/// Which query is currently owed a support credit, and whether an invoke has
/// already claimed it. Lives on the shared graph — not the learner — so the
/// per-registry tool and skill learners that a `search_capabilities` fan-out
/// drives (same query, two learners) credit **one** observation between them,
/// not one each. A `Mutex` so a search can arm it while holding only a read
/// lock, mirroring [`PendingQuery`].
///
/// Identity is the **query text**, and there is one slot per graph — the same
/// single-slot, best-effort posture as [`PendingQuery`]. This is exact for the
/// fan-out it targets (one question, two catalogs, searches before invokes),
/// but it cannot distinguish that from two *concurrent* sessions that ask the
/// same text and each resolve a different catalog into the same cluster: those
/// share the one slot and credit once, an under-count. The trade is deliberate
/// — it removes the systematic over-count on every fanned-out question at the
/// cost of a rare, order-of-magnitude-smaller concurrent edge, and it errs
/// conservative (under-, not over-count, and support caps regardless). Making
/// concurrent same-text sessions exact needs a per-turn correlation id threaded
/// through the trace events, deferred as not worth the plumbing.
#[derive(Debug, Default)]
struct CreditSlot(Mutex<Option<(String, bool)>>);

impl Clone for CreditSlot {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for CreditSlot {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl CreditSlot {
    /// Arm `query` for a support credit. Called on every search; re-arming with
    /// the same text before any invoke is idempotent, so a fanned-out capability
    /// search still yields a single credit.
    fn arm(&self, query: &str) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some((query.to_string(), false));
        }
    }

    /// `true` for the first invoke of an armed `query` — and marks it claimed so
    /// later invokes of the same question (a tool *and* a skill) do not re-credit.
    /// Keyed by query text: a slot clobbered by another session's search reads as
    /// "not first" rather than crediting the wrong question.
    fn claim(&self, query: &str) -> bool {
        let Ok(mut slot) = self.0.lock() else {
            return false;
        };
        match slot.as_mut() {
            Some((q, credited)) if q == query && !*credited => {
                *credited = true;
                true
            }
            _ => false,
        }
    }
}

/// One cluster: the queries it covers and the capabilities invoked after them.
///
/// `label` and `terms` are **derived**, not stored: they are computed from the
/// members at read time and deliberately excluded from equality. c-TF-IDF scores
/// a term against *the other clusters*, so a value frozen when this cluster was
/// last written is wrong the moment another cluster appears.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Cluster id, unique within the graph. Opaque — it names a row.
    pub id: String,
    /// Display name (the medoid member). Never affects ranking.
    pub label: String,
    /// Distinguishing keywords. Never affects ranking.
    #[serde(default)]
    pub terms: Vec<String>,
    /// The texts this cluster covers — **the match key**.
    pub members: Vec<String>,
    /// Optional precomputed L2-normalized mean of the members' embeddings.
    /// Absent when the producer clustered lexically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub centroid: Option<Vec<f32>>,
    /// Confirmed search-then-invoke observations behind this cluster.
    pub support: u32,
    /// How many of [`Self::support`] came from a **seeding** pass — a baseline
    /// capture or a trace replay — rather than from live serving traffic.
    ///
    /// Provenance only: nothing reads it during ranking, and two graphs
    /// differing only here rank identically and compare equal. It exists so a
    /// caller can see how much of a cluster's confidence rests on seeded
    /// evidence, and discount it if the seed turns out to have taught something
    /// wrong.
    ///
    /// Invariant: `seeded_support <= support`, enforced on load. Omitted from
    /// the wire form when zero, so a live-only graph serializes byte-identically
    /// to one produced before this field existed.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub seeded_support: u32,
    /// Epoch-millis of this cluster's most recent observation. Drives the
    /// recency factor and eviction; `0` (default) means "as old as the graph".
    #[serde(default)]
    pub last_ts: u64,
    /// Tool id → count of confirmed invocations. Orders the arm; the
    /// magnitude is discarded by the fusion.
    #[serde(default)]
    pub tools: BTreeMap<String, f32>,
    /// Skill id → count of confirmed invocations. Orders the arm; the
    /// magnitude is discarded by the fusion.
    #[serde(default)]
    pub skills: BTreeMap<String, f32>,
    /// Tokens of each member, positionally parallel to `members`.
    ///
    /// Matching scores a query against **individual members**, not their union:
    /// the union only grows, so scoring against it made a mature cluster
    /// recognize most of the vocabulary and absorb unrelated asks, which grew it
    /// further (ADR-0014). Derived from `members`, so never serialized and never
    /// part of identity.
    #[serde(skip)]
    member_bags: Vec<std::collections::HashSet<String>>,
    /// Every distinct content token across `members`, cached — retained as a
    /// cheap prefilter for [`Self::lexical_score`], not as the score itself.
    ///
    /// Derived from `members` and kept in step with them, so it is never
    /// serialized and never part of identity. It exists because lexical
    /// matching needs this set on **every search**, and rebuilding it from the
    /// member strings each time cost ~99% of that search — the set does not
    /// change between searches, so it is built once and extended in place.
    /// Rebuilt after deserialization by [`IntentGraph::rebuild_caches`].
    #[serde(skip)]
    bag: std::collections::HashSet<String>,
    /// How many query vectors have been folded into `centroid` — the weight of
    /// the running mean in [`Self::absorb_vector`]. Distinct from `members.len()`
    /// because a cluster can gain members lexically (no vector), which must not
    /// inflate the weight.
    ///
    /// Carried on the wire so learning resumes at the right weight after a
    /// round-trip. Without it the running mean restarted at one sample, and the
    /// next single observation could drag a fifty-member cluster's centroid
    /// halfway across — every save/load cycle made centroids plastic again.
    /// Absent on the wire means one, the weight the pre-accumulator code gave a
    /// reloaded centroid.
    #[serde(default = "one_u32", skip_serializing_if = "is_one_u32")]
    vector_n: u32,
    /// `‖mean‖` — how tightly this cluster's members agree. `1.0` when they all
    /// point the same way, `sqrt(n)/n` for n mutually orthogonal ones.
    ///
    /// Normalizing the centroid divides this out, so it is the one thing the
    /// stored centroid cannot tell a consumer, and it is what lets a cluster with
    /// no retained member vectors still guard itself: the bar it must clear is
    /// `TAU_COSINE / cohesion`, which rises as the cluster spreads. Without it a
    /// diffuse cluster presents as tight and keeps absorbing.
    ///
    /// Absent on the wire means `1.0` — the pre-cohesion bar, unchanged.
    #[serde(default = "one_f32", skip_serializing_if = "is_one_f32")]
    pub(crate) cohesion: f32,
    /// Running mean of the folded query vectors, kept **unnormalized** — the
    /// accumulator [`Self::centroid`] is derived from.
    ///
    /// Held separately because normalizing destroys the one thing the magnitude
    /// carries: `‖mean‖` is the cluster's cohesion. Folding the normalized
    /// centroid back into itself instead (the earlier `c * k + v`) stretched the
    /// accumulated history to full length at every step, so the centroid
    /// over-weighted early members and the spread was lost outright.
    ///
    /// Live-learning scratch like [`Self::vector_n`]: never serialized, restored
    /// from the centroid at weight 1 by [`Self::restore_accumulator`], and so
    /// never part of identity.
    #[serde(skip)]
    mean: Option<Vec<f32>>,
    /// Query vector of each member, positionally parallel to `members` and
    /// `member_bags` — the evidence [`Self::coverage`] counts over.
    ///
    /// `None` for a member the lexical tier added (no embedding was in flight)
    /// and for one whose vector has aged past [`VECTOR_RETAIN`]. The `Option` is
    /// load-bearing rather than an empty-vector sentinel: [`cosine`] returns 0.0
    /// for a zero-norm input, so a sentinel would be a member that can never
    /// qualify yet still counts in the denominator, silently raising the bar in
    /// proportion to how much a cluster grew lexically.
    ///
    /// Never serialized. Fifty 384-dim vectors per cluster is ~77 KB, and
    /// `to_json` crosses the SDK boundary as a string on every save — that is
    /// ~230 KB of JSON floats per cluster, on an artifact already documented as
    /// carrying sensitive query text. A graph reloaded without them falls back to
    /// the cohesion-scaled centroid bar until its members are re-observed or
    /// `rebuild_intent_graph` refills them.
    #[serde(skip)]
    member_vectors: Vec<Option<Vec<f32>>>,
}

/// Identity is the evidence — members, centroid, cohesion, support, edges.
/// `cohesion` counts because it is read while ranking (it sets the bar a
/// centroid-only cluster must clear); `vector_n` does not, because it only
/// weights the *next* fold, the same reason `seeded_support` is excluded. The
/// derived
/// display fields are ignored, so a graph compares equal to its own round-trip
/// whether or not labels have been materialized.
impl PartialEq for Intent {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.members == other.members
            && self.centroid == other.centroid
            && self.cohesion == other.cohesion
            && self.support == other.support
            && self.tools == other.tools
            && self.skills == other.skills
    }
}

impl Intent {
    fn edges(&self, kind: Capability) -> &BTreeMap<String, f32> {
        match kind {
            Capability::Tool => &self.tools,
            Capability::Skill => &self.skills,
        }
    }

    /// The cluster's capabilities of `kind`, best-first, dropping any id the
    /// registry does not currently define, plus **how many were dropped**.
    /// Ordered `(weight desc, id asc)` — the same total order the rankers use,
    /// so the arm is deterministic.
    ///
    /// The drop count is returned rather than discarded because an arm that
    /// loses every id is indistinguishable, from the outside, from a cluster
    /// that never matched — see [`ArmOutcome`].
    fn ranked(&self, kind: Capability, known: &dyn Fn(&str) -> bool) -> (Vec<String>, u32) {
        let edges = self.edges(kind);
        let mut ranked: Vec<(String, f32)> = edges
            .iter()
            .filter(|(id, _)| known(id.as_str()))
            .map(|(id, w)| (id.clone(), *w))
            .collect();
        let dropped = (edges.len() - ranked.len()) as u32;
        let len = ranked.len();
        sort_and_truncate(&mut ranked, len);
        (ranked.into_iter().map(|(id, _)| id).collect(), dropped)
    }

    /// Fold `vector` into this cluster's running mean over its members, and
    /// re-derive the centroid from it — renormalized so cosine stays a plain dot
    /// product.
    ///
    /// **The mean is accumulated, never re-derived from the centroid.** A
    /// normalized centroid has length 1 however far apart its members are, so
    /// re-weighting it by the fold count (`c * k + v`) asserted that the
    /// accumulated history had full length `k` — which is true only when every
    /// member points the same way. That over-weighted early members and erased
    /// the spread, and the error grew with exactly the diffuse clusters where it
    /// mattered most.
    ///
    /// The update is the standard incremental mean, `m += (v - m) / n`. It stays
    /// bounded at `‖m‖ <= 1`, where a raw running sum would grow without bound
    /// and lose f32 precision as a long-lived cluster keeps folding.
    ///
    /// Weighted by vectors ALREADY folded (`vector_n`), not by `members.len()`:
    /// a cluster can gain members lexically with no vector, and counting those
    /// would pin the centroid to the first vector after such growth (it would
    /// arrive with weight ~n).
    ///
    /// A first vector — or one of a different width, meaning the embedding model
    /// changed — starts the mean fresh rather than blending across two spaces.
    fn absorb_vector(&mut self, vector: &[f32]) {
        // A centroid set without going through the accumulator — a producer-built
        // cluster, or any future path that writes one directly — would otherwise
        // be thrown away by the fresh-start arm below. Adopt it first, at the
        // same weight 1 a reloaded centroid gets.
        if self.mean.is_none() && self.centroid.is_some() {
            self.restore_accumulator();
        }
        let fresh = self.mean.as_deref().is_none_or(|m| m.len() != vector.len());
        if fresh {
            self.mean = Some(vector.to_vec());
            self.vector_n = 1;
        } else {
            self.vector_n = self.vector_n.saturating_add(1);
            let n = self.vector_n as f32;
            let mean = self.mean.as_mut().expect("not fresh, so the mean is set");
            for (m, v) in mean.iter_mut().zip(vector) {
                *m += (v - *m) / n;
            }
        }
        let mean = self.mean.clone().expect("set on both arms above");
        self.cohesion = norm(&mean);
        self.centroid = Some(normalize(mean));
    }

    /// Add `member` and everything derived from it in one step: the text, its
    /// token bag, its query vector, and the centroid fold.
    ///
    /// One method rather than four calls at the call site because the vector is
    /// **optional while the tokens are not** — a member the lexical tier added has
    /// no embedding. Pushed separately, the two vectors would fall out of step on
    /// the first such member and [`Self::cap_members`] would then evict a
    /// mismatched pair. Here the parallel arrays cannot diverge.
    fn absorb_member(&mut self, member: &str, vector: Option<&[f32]>) {
        self.members.push(member.to_string());
        self.absorb_tokens(member);
        self.member_vectors.push(vector.map(<[f32]>::to_vec));
        if let Some(v) = vector {
            self.absorb_vector(v);
        }
        self.retain_recent_vectors();
        self.cap_members();
    }

    /// Drop all but the newest [`VECTOR_RETAIN`] member vectors, leaving their
    /// members and token bags in place.
    fn retain_recent_vectors(&mut self) {
        let mut budget = VECTOR_RETAIN;
        for slot in self.member_vectors.iter_mut().rev() {
            if slot.is_none() {
                continue;
            }
            if budget == 0 {
                *slot = None;
            } else {
                budget -= 1;
            }
        }
    }

    /// How much of this cluster `query` actually matches: the count of members it
    /// clears [`ClusterPolicy::similarity`] against, the count it needs, and the mean cosine over
    /// those it cleared.
    ///
    /// `None` when the cluster holds no comparable member vector at all — a
    /// lexically-grown cluster, or one reloaded from the wire. That is "no dense
    /// evidence available", **not** "rejected": the caller falls back to the
    /// centroid bar rather than treating an unanswerable question as a no.
    pub(crate) fn coverage(&self, query: &[f32], policy: ClusterPolicy) -> Option<Coverage> {
        let mut total = 0u32;
        let mut qualifying = 0u32;
        let mut sum = 0.0f32;
        for v in self.member_vectors.iter().flatten() {
            // A width mismatch means the member was embedded by a different model
            // — not comparable, and the arm is paused on that mismatch anyway.
            if v.len() != query.len() {
                continue;
            }
            total += 1;
            let c = cosine(query, v);
            if c >= policy.similarity {
                qualifying += 1;
                sum += c;
            }
        }
        if total == 0 {
            return None;
        }
        Some(Coverage {
            qualifying,
            required: required_matches(total, policy.coverage),
            fraction: qualifying as f32 / total as f32,
            mean_cos: if qualifying == 0 {
                0.0
            } else {
                sum / qualifying as f32
            },
        })
    }

    /// Drop the oldest members past [`MEMBER_CAP`], keeping the token caches in
    /// step. Bounds per-cluster memory and lexical-match cost; the centroid is a
    /// cumulative mean, so trimming members does not disturb it.
    fn cap_members(&mut self) {
        while self.members.len() > MEMBER_CAP {
            self.members.remove(0);
            self.member_bags.remove(0);
            self.member_vectors.remove(0);
        }
        // The union bag is derived from the surviving members.
        self.bag = self.member_bags.iter().flatten().cloned().collect();
    }

    /// Fold a newly added member's tokens into the cache. O(tokens in that one
    /// member) — the other members are already accounted for.
    fn absorb_tokens(&mut self, member: &str) {
        let tokens: std::collections::HashSet<String> = tokenize(member).into_iter().collect();
        self.bag.extend(tokens.iter().cloned());
        self.member_bags.push(tokens);
    }

    /// Restore the live accumulator after deserialization, where it is skipped
    /// on the wire.
    ///
    /// A normalized centroid has had its magnitude divided out, so the
    /// accumulator is rebuilt from the two scalars that survive the wire:
    /// `mean = centroid × cohesion`, weighted by the fold count it was built
    /// from. Both default to their identity values, so a graph written before
    /// they existed — or by a producer that does not track them — reconstructs to
    /// exactly the pre-accumulator behaviour.
    fn restore_accumulator(&mut self) {
        match self.centroid.as_deref() {
            Some(c) => {
                self.mean = Some(c.iter().map(|x| x * self.cohesion).collect());
                // A centroid exists, so at least one vector went into it. Leaving
                // the weight at zero would let the next fold replace it outright
                // rather than average with it.
                self.vector_n = self.vector_n.max(1);
            }
            None => {
                self.mean = None;
                self.vector_n = 0;
                self.cohesion = 1.0;
            }
        }
    }

    /// Rebuild the cache from `members` — after deserialization, where the
    /// cache is skipped on the wire.
    fn rebuild_bag(&mut self) {
        self.member_bags = self
            .members
            .iter()
            .map(|m| tokenize(m).into_iter().collect())
            .collect();
        self.bag = self.members.iter().flat_map(|m| tokenize(m)).collect();
        // Member vectors cannot be rebuilt — that needs an embedder, which the
        // load path does not have. Size the array to match so the three stay
        // parallel; the cluster matches on the centroid bar until a rebuild or
        // fresh observations refill it.
        self.member_vectors = vec![None; self.members.len()];
    }

    /// How well `q` matches this cluster: the **best Jaccard overlap with any
    /// single member**, `|q ∩ m| / |q ∪ m|`.
    ///
    /// Per-member rather than against the union, because the union only grows —
    /// so a union score rises with cluster size regardless of whether any actual
    /// past question resembles the query. Per-member, a cluster is exactly as
    /// discriminating on its 200th member as on its first.
    ///
    /// The union is still useful as a cheap **necessary condition**: from
    /// `J = i/(|q|+|m|-i) ≥ τ` and `|m| ≥ 1`, any matching member needs
    /// `i ≥ τ(|q|+1)/(1+τ)` shared tokens, and `|q ∩ union| ≥ |q ∩ m|` for every
    /// member. Clusters that cannot clear that are skipped without touching
    /// their members.
    fn lexical_score(&self, q: &std::collections::HashSet<String>) -> f32 {
        let needed = (TAU_LEXICAL * (q.len() as f32 + 1.0) / (1.0 + TAU_LEXICAL)).ceil() as usize;
        if q.iter().filter(|t| self.bag.contains(*t)).count() < needed {
            return 0.0;
        }
        self.member_bags
            .iter()
            .map(|m| {
                // Length alone can rule a member out: the intersection is at most
                // `min(|q|,|m|)` and the union at least `max(|q|,|m|)`, so a
                // 2-token query can never reach 0.5 against a 5-token member
                // (best case 2/5). Checking that first skips the hashing entirely,
                // and it is exact rather than heuristic.
                let (lo, hi) = if q.len() < m.len() {
                    (q.len(), m.len())
                } else {
                    (m.len(), q.len())
                };
                if hi == 0 || (lo as f32 / hi as f32) < TAU_LEXICAL {
                    return 0.0;
                }
                let inter = q.intersection(m).count() as f32;
                let union = (q.len() + m.len()) as f32 - inter;
                if union == 0.0 { 0.0 } else { inter / union }
            })
            .fold(0.0f32, f32::max)
    }
}

/// The usage-ranking read model — a set of query clusters with capability edges.
///
/// Built either in-process by the local learner or offline by Ratel Cloud; both
/// emit the shape in `protocol/v1`. Attach one to a registry to add the usage
/// arm to its ranking.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct IntentGraph {
    /// Schema version. Always [`GRAPH_VERSION`] for a graph this build accepts.
    pub v: u32,
    /// Epoch-millis of the newest event folded in. Provenance only — it says how
    /// current the graph is, and nothing reads it during ranking.
    pub built_from_ts: u64,
    /// Monotonic write counter, bumped once on every mutation ([`Self::observe`],
    /// a centroid rebuild). Nothing reads it during ranking; it exists for the
    /// caller's storage layer, which owns persistence (the graph is in-process
    /// only). Two uses: **save-when-changed** — persist only when `rev` differs
    /// from the last saved value; and **stale-base detection** — before
    /// overwriting a stored graph, compare its `rev` to the one you loaded, and
    /// if it advanced another writer got there first (single-writer is the
    /// supported model; this makes a clobber *detectable*, not merged). Carried
    /// in the wire form; an older graph without it loads as 0 and continues up.
    #[serde(default)]
    pub rev: u64,
    /// The clusters. Order is not significant.
    pub intents: Vec<Intent>,
    /// Fingerprint of the embedding model the centroids were built with, or
    /// `None` for a lexically-grown graph that has none.
    ///
    /// Centroids are only comparable to a query embedded by the **same** model.
    /// This lets a consumer detect a model swap (`GraphModelStatus`) instead of
    /// cosine-ing across incompatible vector spaces. Stamped when the first
    /// centroid is grown, or by a producer (e.g. Ratel Cloud) that builds
    /// centroids offline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The policy this graph's existing cluster boundaries were drawn under.
    ///
    /// Provenance, not configuration — it says how the clusters below came to be,
    /// which is the whole reason the policy is safe to expose: without it, two
    /// producers at different settings would disagree about what a cluster means
    /// while both claiming the same protocol version (ADR-0014).
    ///
    /// Absent on the wire means the default, which is also historically exact:
    /// before the policy was configurable the constants were the only value a
    /// producer could have used. A graph still at the default therefore
    /// serializes byte-identically to one written before the field existed.
    #[serde(default, skip_serializing_if = "ClusterPolicy::is_default")]
    pub cluster_policy: ClusterPolicy,
    /// The policy in force now — what every admission decision is measured
    /// against. Comes from configuration, never from the wire: it describes how
    /// *this process* is clustering, not how the loaded graph was clustered.
    /// Those can differ, and [`Self::cluster_policy`] is what lets a consumer
    /// see that they do.
    #[serde(skip)]
    active_policy: ClusterPolicy,
    /// Scratch for the search path → learner handoff; never serialized.
    #[serde(skip)]
    pending: PendingQuery,
    /// Which query is owed a support credit, shared across the tool and skill
    /// learners so one fanned-out question counts once. Never serialized.
    #[serde(skip)]
    credit: CreditSlot,
}

/// Whether an [`IntentGraph`]'s centroids can be trusted against the currently
/// active embedding model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GraphModelStatus {
    /// Usable: no centroids (lexical graph), or the model matches.
    Ok,
    /// Centroid width differs from the active model's output — a different model
    /// family. Dense matching is meaningless; the arm must pause.
    DimMismatch { built: usize, active: usize },
    /// Same width but a different model fingerprint (a fine-tune, or another
    /// model of the same dimension). Cosine across the two spaces is garbage; the
    /// arm must pause. A length check alone cannot catch this.
    ModelMismatch { built: String, active: String },
}

impl GraphModelStatus {
    /// `(built, active, dim_mismatch)` for [`crate::TraceEvent::UsageModelMismatch`],
    /// or `None` when there is no mismatch. Dimensions are stringified so both
    /// cases share one event shape.
    pub(crate) fn describe(&self) -> Option<(String, String, bool)> {
        match self {
            GraphModelStatus::Ok => None,
            GraphModelStatus::DimMismatch { built, active } => {
                Some((built.to_string(), active.to_string(), true))
            }
            GraphModelStatus::ModelMismatch { built, active } => {
                Some((built.clone(), active.clone(), false))
            }
        }
    }
}

/// Serializing materializes the derived display fields, so the wire form always
/// carries labels computed against the graph being written — never a stale
/// snapshot from whenever a cluster last happened to change.
impl Serialize for IntentGraph {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let len =
            4 + usize::from(self.model.is_some()) + usize::from(!self.cluster_policy.is_default());
        let mut out = serializer.serialize_struct("IntentGraph", len)?;
        out.serialize_field("v", &self.v)?;
        out.serialize_field("built_from_ts", &self.built_from_ts)?;
        out.serialize_field("rev", &self.rev)?;
        if let Some(model) = &self.model {
            out.serialize_field("model", model)?;
        }
        if !self.cluster_policy.is_default() {
            out.serialize_field("cluster_policy", &self.cluster_policy)?;
        }
        out.serialize_field("intents", &self.labeled())?;
        out.end()
    }
}

impl Default for IntentGraph {
    fn default() -> Self {
        Self::empty()
    }
}

impl IntentGraph {
    /// Parse a graph from its JSON wire form.
    ///
    /// # Errors
    ///
    /// [`IntentGraphError::Malformed`] if the bytes are not the expected shape,
    /// or [`IntentGraphError::UnsupportedVersion`] if `v` is not 1.
    pub fn from_json(json: &str) -> Result<Self, IntentGraphError> {
        let mut graph: IntentGraph =
            serde_json::from_str(json).map_err(|e| IntentGraphError::Malformed(e.to_string()))?;
        if graph.v != GRAPH_VERSION {
            return Err(IntentGraphError::UnsupportedVersion(graph.v));
        }
        graph.validate()?;
        // A cluster with no recorded `last_ts` (an older or cloud-built graph
        // that didn't track it) is treated as current at load — decay begins
        // from the graph's own timestamp, not epoch 0, so a freshly loaded graph
        // is not instantly stale.
        let anchor = graph.built_from_ts;
        for it in &mut graph.intents {
            if it.last_ts == 0 {
                it.last_ts = anchor;
            }
        }
        // Absent configuration, keep clustering the way this graph already was —
        // a reload on its own must not move a boundary.
        graph.active_policy = graph.cluster_policy;
        graph.rebuild_caches();
        Ok(graph)
    }

    /// Reject a structurally-parseable graph that breaks a semantic rule the wire
    /// contract requires (`protocol/v1/conformance/vectors.json`, the `invalid`
    /// set). Serde already catches shape errors (a missing `members`, a negative
    /// `rev`); these are the value-level rules it cannot express.
    fn validate(&self) -> Result<(), IntentGraphError> {
        // The first graph-level rule; every check below it is per-intent. A
        // cosine and a fraction both live in (0, 1], and a value outside that is
        // not something any producer can have meant.
        if !self.cluster_policy.is_valid() {
            return Err(IntentGraphError::Malformed(format!(
                "cluster_policy similarity {} / coverage {} outside (0, 1]",
                self.cluster_policy.similarity, self.cluster_policy.coverage
            )));
        }
        let mut seen = std::collections::HashSet::with_capacity(self.intents.len());
        for it in &self.intents {
            if !seen.insert(it.id.as_str()) {
                return Err(IntentGraphError::Malformed(format!(
                    "duplicate intent id {:?}",
                    it.id
                )));
            }
            if it.members.is_empty() {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} has no members",
                    it.id
                )));
            }
            if it.support < 1 {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} has support 0 (a confirmed cluster is at least 1)",
                    it.id
                )));
            }
            if it.centroid.as_ref().is_some_and(|c| c.is_empty()) {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} has an empty centroid",
                    it.id
                )));
            }
            // `seeded_support` counts a subset of `support`, so no producer can
            // emit a larger value. Equality is legal and is the normal state
            // right after a seeding pass.
            if !(0.0 < it.cohesion && it.cohesion <= 1.0) {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} has cohesion {} outside (0, 1]",
                    it.id, it.cohesion
                )));
            }
            if it.cohesion != 1.0 && it.centroid.is_none() {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} records a cohesion but carries no centroid",
                    it.id
                )));
            }
            if it.seeded_support > it.support {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} has seeded_support {} exceeding support {}",
                    it.id, it.seeded_support, it.support
                )));
            }
            if it
                .tools
                .values()
                .chain(it.skills.values())
                .any(|w| *w <= 0.0)
            {
                return Err(IntentGraphError::Malformed(format!(
                    "intent {:?} has a non-positive edge weight",
                    it.id
                )));
            }
        }
        Ok(())
    }

    /// Rebuild every cluster's derived state. The token cache and the centroid
    /// accumulator are both skipped on the wire, so a deserialized graph must
    /// restore them before it can match lexically or keep folding.
    fn rebuild_caches(&mut self) {
        for it in &mut self.intents {
            it.rebuild_bag();
            it.restore_accumulator();
        }
    }

    /// An empty graph at the current version — the starting state of a learner.
    pub fn empty() -> Self {
        Self {
            v: GRAPH_VERSION,
            built_from_ts: 0,
            rev: 0,
            intents: Vec::new(),
            model: None,
            cluster_policy: ClusterPolicy::default(),
            active_policy: ClusterPolicy::default(),
            pending: PendingQuery::default(),
            credit: CreditSlot::default(),
        }
    }

    /// Number of clusters.
    pub fn len(&self) -> usize {
        self.intents.len()
    }

    /// Whether the graph holds no clusters — the cold-start state, in which it
    /// contributes no arm to any query.
    pub fn is_empty(&self) -> bool {
        self.intents.is_empty()
    }

    /// Stash the embedded query so a later [`Self::observe`] can grow a real
    /// centroid from it.
    ///
    /// Called on the search path of a semantic/hybrid registry, which has
    /// already embedded the query for its own ranking — so this costs nothing
    /// beyond a copy. Takes `&self`: the slot is a `Mutex`, so the search path
    /// never needs the write lock.
    pub(crate) fn note_query_vector(&self, query: &str, vector: &[f32], fingerprint: &str) {
        self.pending.set(query, vector, fingerprint);
    }

    /// Arm `query` for a support credit on the shared credit slot — called by the
    /// learner on every search. See [`CreditSlot`] for why this lives on the
    /// graph rather than the learner.
    pub(crate) fn arm_credit(&self, query: &str) {
        self.credit.arm(query);
    }

    /// Whether this invoke is the first confirmation of `query` across every
    /// learner sharing the graph. Marks the credit claimed, so a tool invoke and
    /// a skill invoke for one fanned-out question yield a single support bump.
    pub(crate) fn claim_credit(&self, query: &str) -> bool {
        self.credit.claim(query)
    }

    /// Fold one confirmed observation — a query, and the capability invoked
    /// after it — into the graph.
    ///
    /// This is the whole learning step (ADR-0014). It:
    ///
    /// 1. finds the cluster this query belongs to — by centroid when the search
    ///    path stashed an embedding, else by token overlap — or **seeds a new
    ///    one**;
    /// 2. adds the query as a member and adds `1.0` to the invoked capability's
    ///    edge, bumping `support` only when this is the search's **first**
    ///    confirming invoke;
    /// 3. recomputes the cluster's display label and terms.
    ///
    /// `ts_ms` records how current the graph is; it never affects ranking.
    /// Traces are loosely ordered (ADR-0007), so a late-arriving older event
    /// leaves the recorded high-water mark alone.
    /// [`Observation::first_confirmation`] distinguishes *this search was acted
    /// on* from *another capability was used for the same search*. Both add an
    /// edge; only the former is an observation, so only the former raises
    /// `support`. The caller owns that distinction because it is the one holding
    /// the pending search — see [`crate::UsageLearner`].
    pub(crate) fn observe(&mut self, obs: Observation<'_>) {
        let Observation {
            query,
            kind,
            capability_id,
            ts_ms,
            first_confirmation,
            seeded,
        } = obs;
        // A query vector is available only when the search path was
        // semantic/hybrid AND the slot still belongs to this query.
        let stashed = self.pending.vector_for(query);
        if stashed.is_none() && tokenize(query).is_empty() {
            return; // no words to cluster on and no embedding either
        }
        self.built_from_ts = self.built_from_ts.max(ts_ms);
        let first_cluster = self.intents.is_empty();

        // Only fold the vector if it was produced by the graph's model. On a
        // model swap (fingerprint differs from `self.model`) we FREEZE: the
        // member, support, and edge still update — they are model-independent —
        // but the centroid is left untouched rather than blended across two
        // vector spaces. `None` model means no centroids yet; the first fold
        // stamps it.
        let usable = match (&self.model, &stashed) {
            (Some(m), Some((_, fp))) => m == fp,
            _ => true,
        };
        let vector: Option<Vec<f32>> = if usable {
            stashed.as_ref().map(|(v, _)| v.clone())
        } else {
            None
        };
        let fingerprint: Option<String> = stashed.as_ref().map(|(_, fp)| fp.clone());

        let idx = match self.best_match(query, vector.as_deref()) {
            Some(i) => i,
            None => {
                let id = format!("intent_{}", self.next_intent_seq());
                self.intents.push(Intent {
                    id,
                    // Derived on read — see `labeled`. Never written while learning.
                    label: String::new(),
                    terms: Vec::new(),
                    members: Vec::new(),
                    centroid: None,
                    support: 0,
                    seeded_support: 0,
                    last_ts: 0,
                    tools: BTreeMap::new(),
                    skills: BTreeMap::new(),
                    bag: std::collections::HashSet::new(),
                    member_bags: Vec::new(),
                    vector_n: 0,
                    cohesion: 1.0,
                    mean: None,
                    member_vectors: Vec::new(),
                });
                self.intents.len() - 1
            }
        };

        {
            let it = &mut self.intents[idx];
            // Members are the match key, so a repeated phrasing must not inflate
            // the token bag — dedupe. The centroid is the mean of the DISTINCT
            // member texts, so it moves exactly when a new member arrives: the
            // same condition, and what stops a second invoke from folding the
            // same query vector in twice.
            if !it.members.iter().any(|m| m == query) {
                it.absorb_member(query, vector.as_deref());
            }
            // `|| support == 0` is load-bearing: a cluster moves as it learns, so
            // a later invoke from the same search can match a cluster the first
            // one did not — and a freshly seeded cluster must still start at 1.
            // `protocol/v1` requires support >= 1, and a zero-support cluster
            // would contribute a weightless arm.
            if first_confirmation || it.support == 0 {
                it.support = it.support.saturating_add(1);
                // In lockstep with `support`, never per edge — one fanned-out
                // question adds two edges but is one observation, and bumping
                // per edge would break `seeded_support <= support` on the very
                // first fanned-out baseline turn.
                if seeded {
                    it.seeded_support = it.seeded_support.saturating_add(1);
                }
            }
            it.last_ts = it.last_ts.max(ts_ms);
            let edges = match kind {
                Capability::Tool => &mut it.tools,
                Capability::Skill => &mut it.skills,
            };
            *edges.entry(capability_id.to_string()).or_insert(0.0) += 1.0;
        }

        // Stamp the model the first time a centroid actually exists, so later
        // observations under a different model can be detected and frozen. Done
        // before eviction, while `idx` is still valid.
        if self.model.is_none() && self.intents[idx].centroid.is_some() {
            self.model = fingerprint;
        }
        // Stamped on the observation that creates the graph's first cluster, and
        // frozen from then on. It records the policy these boundaries were drawn
        // under, and boundaries are never redrawn in place — so a later policy
        // change stays visible precisely because this does not follow it.
        if first_cluster {
            self.cluster_policy = self.active_policy;
        }

        // Evict clusters decayed past the floor — last, since it renumbers
        // `intents`. The just-touched cluster has `last_ts == built_from_ts`, so
        // it is never evicted here.
        let now = self.built_from_ts;
        self.intents
            .retain(|it| recency_factor(now, it.last_ts) >= EVICTION_FLOOR);

        // Every path that reaches here changed a member, an edge, or support, so
        // count exactly one write. The early returns above (no words and no
        // vector) leave `rev` alone — nothing was persisted-worthy.
        self.rev += 1;
    }

    /// The write counter — see [`Self::rev`]. Snapshot it after each save; a
    /// later value means unsaved learning, or another writer moved ahead of you.
    pub fn rev(&self) -> u64 {
        self.rev
    }

    /// Whether this graph's boundaries were drawn under the policy now in force.
    ///
    /// `None` when they match, or when the graph is still at the defaults it was
    /// necessarily written under. `Some((built, active))` is a **notice, not a
    /// pause**: the vectors are fine and the clusters are still coherent, merely
    /// coarser or finer than the current setting would draw them. What it warns
    /// about is that they will not be redrawn.
    pub(crate) fn cluster_policy_drift(&self) -> Option<(ClusterPolicy, ClusterPolicy)> {
        (self.cluster_policy != self.active_policy)
            .then_some((self.cluster_policy, self.active_policy))
    }

    /// Set the policy future admissions are measured against.
    ///
    /// Does **not** redraw existing boundaries — nothing can, since a cluster's
    /// edges are aggregate counts with no member attribution to split on. The
    /// graph keeps reporting the policy it was clustered under, and the
    /// difference surfaces as a drift notice.
    pub fn set_cluster_policy(&mut self, policy: ClusterPolicy) {
        self.active_policy = policy;
    }

    /// The policy future admissions are measured against.
    #[must_use]
    pub fn active_cluster_policy(&self) -> ClusterPolicy {
        self.active_policy
    }

    /// Whether this graph's centroids can be trusted against the currently active
    /// embedding model, whose vectors are `query_dim`-wide with identity
    /// `active_fingerprint`.
    ///
    /// A lexical graph (no centroids) is always [`GraphModelStatus::Ok`] — it has
    /// nothing model-specific. A dense graph must agree on both width and model
    /// identity; the width check alone cannot catch a same-dimension model swap.
    pub(crate) fn model_status(
        &self,
        active_fingerprint: &str,
        query_dim: usize,
    ) -> GraphModelStatus {
        let Some(built_dim) = self
            .intents
            .iter()
            .find_map(|i| i.centroid.as_ref().map(Vec::len))
        else {
            return GraphModelStatus::Ok; // no centroids — lexical, model-agnostic
        };
        if built_dim != query_dim {
            return GraphModelStatus::DimMismatch {
                built: built_dim,
                active: query_dim,
            };
        }
        match &self.model {
            Some(built) if built != active_fingerprint => GraphModelStatus::ModelMismatch {
                built: built.clone(),
                active: active_fingerprint.to_string(),
            },
            _ => GraphModelStatus::Ok,
        }
    }

    /// Re-embed every cluster's members under a new model and replace the
    /// centroids, restamping [`Self::model`]. Each entry is a cluster **id** and
    /// the embeddings of its `members` (in member order).
    ///
    /// Assignment is **by id, not position**. `rebuild_intent_graph` snapshots
    /// members and embeds them without the graph lock (so searches are not
    /// blocked), then re-locks to apply here — and a concurrent `observe()` in
    /// that window can evict or seed a cluster, shifting positions since the
    /// snapshot. Zipping by position would stamp a centroid onto the wrong
    /// cluster, silently, because the fresh model fingerprint hides the swap. An
    /// id absent now (evicted since the snapshot) is skipped; a cluster seeded
    /// since is simply left for the next rebuild.
    ///
    /// Members, support, and edges are model-independent and untouched, so all
    /// learning survives a model change — only the centroids move to the new
    /// space. A cluster with no members (or none embedded) keeps whatever
    /// centroid it had.
    pub(crate) fn rebuild_centroids(
        &mut self,
        per_cluster: Vec<(String, Vec<Vec<f32>>)>,
        fingerprint: String,
    ) {
        let index: std::collections::HashMap<String, usize> = self
            .intents
            .iter()
            .enumerate()
            .map(|(i, it)| (it.id.clone(), i))
            .collect();
        for (id, vectors) in per_cluster {
            if vectors.is_empty() {
                continue;
            }
            let Some(&i) = index.get(&id) else {
                continue; // evicted since the snapshot — nothing to attach to
            };
            let dim = vectors[0].len();
            let mut sum = vec![0.0f32; dim];
            for v in &vectors {
                for (s, x) in sum.iter_mut().zip(v) {
                    *s += x;
                }
            }
            let folded = vectors.len();
            for s in &mut sum {
                *s /= folded as f32;
            }
            let it = &mut self.intents[i];
            it.cohesion = norm(&sum);
            it.centroid = Some(normalize(sum.clone()));
            // Keep the per-member vectors, not just their mean. They are what
            // `coverage` counts, they never cross the wire, and this is the only
            // path that has them in hand — so a rebuild is also how a graph that
            // came off disk, or was grown before coverage existed, gets a dense
            // tier that can tell a cluster's members apart again.
            //
            // Only when the pairing is provably intact. The caller snapshots
            // members, embeds without the graph lock, then re-locks, and a
            // concurrent `observe` in that window can append a member and evict
            // the oldest — shifting every position by one. The centroid does not
            // care, being an order-insensitive mean, but these are matched to
            // members by position. On any disagreement leave them alone: the
            // cluster keeps the cohesion bar until fresh observations refill it,
            // which is a degraded tier rather than a silently wrong one.
            if vectors.len() == it.members.len() {
                it.member_vectors = vectors.into_iter().map(Some).collect();
                it.retain_recent_vectors();
            }
            // Reset the accumulator to what was actually rebuilt. Leaving the
            // stale fold count let the next single observation yank a whole
            // cluster's centroid halfway across, no matter how many members it
            // held — and since the count never crosses the wire, every
            // save/load/rebuild cycle made centroids plastic again.
            it.mean = Some(sum);
            it.vector_n = folded as u32;
        }
        self.model = Some(fingerprint);
        // A rebuild rewrites every centroid and restamps the model — a change the
        // caller will want to persist.
        self.rev += 1;
    }

    /// The cluster this query belongs to: by cosine when an embedding is
    /// available and some cluster carries a centroid, otherwise by token
    /// overlap.
    ///
    /// Dense first, lexical as a fallback — a graph can hold both kinds while
    /// centroids are still being filled in, and a query that no centroid
    /// recognizes may still share words with a cluster.
    fn best_match(&self, query: &str, vector: Option<&[f32]>) -> Option<usize> {
        match vector {
            // Same guard the serving path applies in `arm`: once a query has been
            // put to the centroid-bearing clusters and refused, token overlap must
            // not hand it back to one of them. Only clusters the dense tier cannot
            // see — those with no centroid — remain eligible.
            //
            // Learning needs this more than serving does, not less. A bad serve is
            // one bad ranking; a bad admission is written into the graph and every
            // later query is matched against a cluster that has drifted.
            Some(v) if self.has_centroids() => self
                .best_dense_match(v)
                .or_else(|| self.best_lexical_matching(query, true)),
            _ => self.best_lexical_matching(query, false),
        }
    }

    /// Index of the nearest cluster centroid clearing [`TAU_COSINE`]. Ties break
    /// by cluster id so growth does not depend on `Vec` order.
    fn best_dense_match(&self, vector: &[f32]) -> Option<usize> {
        self.intents
            .iter()
            .enumerate()
            .filter_map(|(i, it)| dense_verdict(it, vector, self.active_policy).map(|v| (i, v)))
            .filter(|(_, v)| v.admitted)
            .max_by(|a, b| {
                rank_verdicts(&a.1, &b.1)
                    .then_with(|| self.intents[b.0].id.cmp(&self.intents[a.0].id))
            })
            .map(|(i, _)| i)
    }

    /// Index of the cluster whose member-token bag best covers `query`, if any
    /// clears [`TAU_LEXICAL`]. Ties break by cluster id so growth does not
    /// depend on `Vec` order.
    fn best_lexical_matching(&self, query: &str, centroidless_only: bool) -> Option<usize> {
        let q: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
        if q.is_empty() {
            return None;
        }
        self.intents
            .iter()
            .enumerate()
            .filter(|(_, it)| !centroidless_only || it.centroid.is_none())
            .map(|(i, it)| (i, it.lexical_score(&q)))
            .filter(|(_, score)| *score >= TAU_LEXICAL)
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| self.intents[b.0].id.cmp(&self.intents[a.0].id))
            })
            .map(|(i, _)| i)
    }

    /// The next free `intent_N` sequence number, so ids stay unique even after
    /// clusters are merged away by a future compaction.
    fn next_intent_seq(&self) -> usize {
        self.intents
            .iter()
            .filter_map(|i| i.id.strip_prefix("intent_")?.parse::<usize>().ok())
            .max()
            .map_or(0, |m| m + 1)
    }

    /// The most central member — the one whose tokens the *rest* of the cluster
    /// shares most — as a real past query rather than a generated summary, so it
    /// can never misdescribe the cluster. Ties break by the member text.
    ///
    /// Scored against the *other* members, not the cluster's union bag: the union
    /// contains every member's tokens by construction, so coverage-of-the-union
    /// is a constant `1.0` and would leave the label to the tie-break alone.
    fn medoid(&self, idx: usize) -> String {
        let it = &self.intents[idx];
        let tokenized: Vec<(&String, Vec<String>)> =
            it.members.iter().map(|m| (m, tokenize(m))).collect();
        tokenized
            .iter()
            .enumerate()
            .map(|(i, (m, t))| {
                let shared = t
                    .iter()
                    .filter(|tok| {
                        tokenized
                            .iter()
                            .enumerate()
                            .any(|(j, (_, other))| j != i && other.contains(*tok))
                    })
                    .count();
                let score = if t.is_empty() {
                    0.0
                } else {
                    shared as f32 / t.len() as f32
                };
                (*m, score)
            })
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(a.0))
            })
            .map(|(m, _)| m.clone())
            .unwrap_or_default()
    }

    /// The distinguishing terms for one cluster: class-based TF-IDF (BERTopic's
    /// method — each cluster is one document, so a term ranks by how much it sets
    /// this cluster apart, not how common it is within it).
    ///
    /// Takes the corpus-wide stats — `total` tokens across the graph, `avg`
    /// tokens per cluster, and `global` per-token occurrence counts — as
    /// arguments because they are identical for every cluster. [`Self::labeled`]
    /// builds them once and hands them to each call rather than rebuilding the
    /// whole-corpus index per cluster (which made labeling O(N²) in cluster
    /// count on a path `toJson` may run often).
    fn c_tf_idf_terms(
        cluster_tokens: &[String],
        total: usize,
        avg: f32,
        global: &std::collections::HashMap<&str, usize>,
    ) -> Vec<String> {
        use std::collections::HashMap;
        if total == 0 || cluster_tokens.is_empty() {
            return Vec::new();
        }
        let mut local: HashMap<&str, usize> = HashMap::new();
        for t in cluster_tokens {
            *local.entry(t.as_str()).or_insert(0) += 1;
        }
        let len = cluster_tokens.len() as f32;

        let mut scored: Vec<(String, f32)> = local
            .into_iter()
            .map(|(t, count)| {
                let f = global[t] as f32;
                (t.to_string(), (count as f32 / len) * (1.0 + avg / f).ln())
            })
            .collect();
        sort_and_truncate(&mut scored, MAX_TERMS);
        scored.into_iter().map(|(t, _)| t).collect()
    }

    /// The clusters with their display fields materialized against the graph as
    /// it is **now**.
    ///
    /// Labels are derived rather than stored for two reasons. c-TF-IDF ranks a
    /// term by how rare it is across the *other* clusters, so a value computed
    /// when a cluster was last written goes stale as soon as the graph grows.
    /// And computing them on write meant re-tokenizing every member of every
    /// cluster on every invocation — for strings ranking never reads.
    ///
    /// The whole-corpus token index (`per_cluster`, `global`, `avg`) is built
    /// **once** here and shared by every cluster's c-TF-IDF; building it inside
    /// each call made this quadratic in cluster count.
    pub fn labeled(&self) -> Vec<Intent> {
        use std::collections::HashMap;
        // Tokenize every member of every cluster once, in `intents` order.
        let per_cluster: Vec<Vec<String>> = self
            .intents
            .iter()
            .map(|it| it.members.iter().flat_map(|m| tokenize(m)).collect())
            .collect();
        let total: usize = per_cluster.iter().map(|c| c.len()).sum();
        let avg = if per_cluster.is_empty() {
            0.0
        } else {
            total as f32 / per_cluster.len() as f32
        };
        // Corpus-wide occurrence count per token, shared across all clusters.
        let mut global: HashMap<&str, usize> = HashMap::new();
        for c in &per_cluster {
            for t in c {
                *global.entry(t.as_str()).or_insert(0) += 1;
            }
        }

        self.intents
            .iter()
            .enumerate()
            .map(|(i, it)| Intent {
                label: self.medoid(i),
                terms: Self::c_tf_idf_terms(&per_cluster[i], total, avg, &global),
                ..it.clone()
            })
            .collect()
    }

    /// Resolve the usage arm, choosing the match tier from **what this graph
    /// carries** rather than from the caller's search method.
    ///
    /// Dense matching needs both a query vector *and* stored centroids. A
    /// producer that clustered lexically — the in-process learner, or Ratel
    /// Cloud's Jaccard clusterer — emits no centroids, so a semantic catalog
    /// handed such a graph must still match it lexically rather than see nothing
    /// at all. Falling back here is what makes the format portable across
    /// producers in practice, not just on paper.
    ///
    /// On a **mixed** graph (some clusters carry centroids, some don't), a vector
    /// query matches the centroid-bearing clusters densely and, on a miss, the
    /// centroid-less clusters lexically. The lexical fallback is restricted to
    /// centroid-less clusters on purpose: a fingerprinted cluster dense matching
    /// already rejected must never be rescued by token overlap (that would let
    /// words override meaning), but a centroid-less cluster has no other way to
    /// match at all, so it gets the lexical shot it is due (#5).
    pub(crate) fn arm(
        &self,
        query: &str,
        query_vec: Option<&[f32]>,
        kind: Capability,
        known: &dyn Fn(&str) -> bool,
    ) -> ArmOutcome {
        match query_vec {
            Some(v) if self.has_centroids() => self
                .arm_dense(v, kind, known)
                .or_else(|| self.arm_lexical_matching(query, kind, known, true)),
            _ => self.arm_lexical(query, kind, known),
        }
    }

    /// Whether any cluster carries a centroid, i.e. whether dense matching is
    /// possible at all against this graph.
    fn has_centroids(&self) -> bool {
        self.intents.iter().any(|i| i.centroid.is_some())
    }

    /// Match `query_vec` to the nearest cluster centroid and return its arm.
    ///
    /// `None` when nothing clears [`TAU_COSINE`], when the matched cluster has
    /// no surviving edges of `kind`, or when no cluster carries a centroid of
    /// the query's dimension (a changed embedding model — mismatched vector
    /// spaces are skipped, never compared).
    pub(crate) fn arm_dense(
        &self,
        query_vec: &[f32],
        kind: Capability,
        known: &dyn Fn(&str) -> bool,
    ) -> ArmOutcome {
        let Some((intent, verdict)) = self
            .intents
            .iter()
            .filter_map(|it| dense_verdict(it, query_vec, self.active_policy).map(|v| (it, v)))
            .filter(|(_, v)| v.admitted)
            .max_by(|a, b| rank_verdicts(&a.1, &b.1).then_with(|| b.0.id.cmp(&a.0.id)))
        else {
            return ArmOutcome::NoMatch;
        };
        // The reported similarity stays the centroid cosine: integrators already
        // dashboard it, and swapping in a coverage fraction would silently change
        // what the number means. The prefilter computed it anyway.
        arm_from(
            intent,
            verdict.centroid_cos,
            self.built_from_ts,
            kind,
            known,
        )
    }

    /// Match `query` lexically against each cluster's members and return the best
    /// cluster's arm.
    ///
    /// The score is the best **per-member Jaccard overlap** — `|q ∩ m| / |q ∪ m|`
    /// against the cluster's closest single member, not against the members'
    /// union (which only grows, letting a mature cluster absorb unrelated asks;
    /// see [`Intent::lexical_score`]). Bounded in `[0, 1]`, so it thresholds
    /// meaningfully — unlike a raw BM25 score, which is unbounded and
    /// corpus-relative. `None` when nothing clears [`TAU_LEXICAL`] or the match
    /// has no surviving edges.
    pub(crate) fn arm_lexical(
        &self,
        query: &str,
        kind: Capability,
        known: &dyn Fn(&str) -> bool,
    ) -> ArmOutcome {
        self.arm_lexical_matching(query, kind, known, false)
    }

    /// Lexical match, optionally limited to centroid-less clusters. The dense
    /// serving fallback sets `centroidless_only` so a fingerprinted cluster that
    /// dense matching already rejected is never rescued by token overlap; only
    /// clusters dense cannot see (no centroid) get a lexical match (see [`arm`]).
    ///
    /// [`arm`]: Self::arm
    fn arm_lexical_matching(
        &self,
        query: &str,
        kind: Capability,
        known: &dyn Fn(&str) -> bool,
        centroidless_only: bool,
    ) -> ArmOutcome {
        let q: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
        if q.is_empty() {
            return ArmOutcome::NoMatch;
        }
        let Some(best) = self
            .intents
            .iter()
            .filter(|it| !centroidless_only || it.centroid.is_none())
            .map(|it| (it, it.lexical_score(&q)))
            .filter(|(_, score)| *score >= TAU_LEXICAL)
            .max_by(pick_best)
        else {
            return ArmOutcome::NoMatch;
        };
        arm_from(best.0, best.1, self.built_from_ts, kind, known)
    }
}

/// Break a score tie by id ascending, so the chosen cluster does not depend on
/// iteration order. (`max_by` keeps the last maximum, so the comparison is
/// reversed on id to leave the alphabetically-first winner in place.)
fn pick_best(a: &(&Intent, f32), b: &(&Intent, f32)) -> std::cmp::Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| b.0.id.cmp(&a.0.id))
}

fn arm_from(
    intent: &Intent,
    similarity: f32,
    now_ts: u64,
    kind: Capability,
    known: &dyn Fn(&str) -> bool,
) -> ArmOutcome {
    let (ids, dropped) = intent.ranked(kind, known);
    if ids.is_empty() {
        // Matched, but nothing it remembers of this kind still exists. Two very
        // different reasons: the catalog dropped every id it knew (`dropped >
        // 0` — drift, worth reporting), or the cluster simply holds no edges of
        // this kind at all, which a tools-only cluster asked for skills does
        // legitimately and is not drift.
        return if dropped > 0 {
            ArmOutcome::AllFiltered {
                intent_id: intent.id.clone(),
                similarity,
                support: intent.support,
                dropped,
            }
        } else {
            ArmOutcome::NoMatch
        };
    }
    let weight = usage_weight(intent.support) * recency_factor(now_ts, intent.last_ts);
    ArmOutcome::Armed(UsageArm {
        intent_id: intent.id.clone(),
        similarity,
        support: intent.support,
        weight,
        ids,
        dropped,
    })
}

/// How much of a cluster a query matched — see [`Intent::coverage`].
///
/// `pub(crate)` so the measurement harness can report the rule's own numbers
/// rather than recomputing them beside it, where the two could drift.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Coverage {
    /// Members whose cosine cleared [`ClusterPolicy::similarity`].
    pub(crate) qualifying: u32,
    /// Members that had to clear it — see [`required_matches`].
    pub(crate) required: u32,
    /// `qualifying / comparable members`, in `[0, 1]`. Comparable across clusters
    /// of different sizes, which a raw count is not: a 50-member cluster would
    /// otherwise outrank a 4-member one on count alone.
    pub(crate) fraction: f32,
    /// Mean cosine over the members that qualified. Breaks ties between clusters
    /// a query covers equally.
    pub(crate) mean_cos: f32,
}

/// How many of `n` comparable members a query must match to join.
///
/// `ceil(COVERAGE_FRACTION * n)`, floored at 2 and capped at `n`. The floor
/// matters because a plain fraction drops the requirement to 1 at `n = 2` —
/// single-link chaining, at exactly the size where one bad admission defines what
/// the cluster becomes. The cap is cold start: a cluster with one member must
/// still be able to gain its second.
fn required_matches(n: u32, coverage: f32) -> u32 {
    let by_fraction = (coverage * n as f32).ceil() as u32;
    n.min(by_fraction.max(2))
}

/// One cluster's dense verdict for a query: whether it admits, how well it
/// matched, and the centroid cosine to report on the trace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseVerdict {
    pub(crate) admitted: bool,
    /// Whether the verdict rests on member coverage rather than the centroid
    /// alone. Clusters with real per-member evidence outrank those without, so a
    /// legacy or freshly-loaded cluster cannot beat one that actually matched.
    pub(crate) covered: bool,
    /// Coverage fraction, or the centroid cosine when the cluster has no
    /// comparable member vectors. Both are in `[0, 1]`, and they are only ever
    /// compared within the same `covered` tier.
    pub(crate) score: f32,
    pub(crate) mean_cos: f32,
    /// Always the centroid cosine — what `UsageBoost.similarity` reports, whose
    /// meaning must not change under integrators.
    pub(crate) centroid_cos: f32,
}

/// Score `query` against one cluster's dense tier, or `None` if the cluster is
/// not comparable (no centroid, different embedding model) or fails the
/// prefilter.
///
/// The centroid check is kept as a **prefilter**, one dot product, so the
/// per-member scan only runs on clusters that could plausibly match — the same
/// shape the lexical tier uses, where `bag` prefilters `lexical_score`. Because
/// the prefilter threshold is [`TAU_COSINE`] itself, admission here is the old
/// rule *and* coverage, never looser.
pub(crate) fn dense_verdict(
    it: &Intent,
    query: &[f32],
    policy: ClusterPolicy,
) -> Option<DenseVerdict> {
    let centroid = it.centroid.as_deref()?;
    if centroid.len() != query.len() {
        return None; // a different embedding model — not comparable
    }
    let centroid_cos = cosine(query, centroid);
    if centroid_cos < policy.similarity {
        return None;
    }
    Some(match it.coverage(query, policy) {
        Some(cov) => DenseVerdict {
            admitted: cov.qualifying >= cov.required,
            covered: true,
            score: cov.fraction,
            mean_cos: cov.mean_cos,
            centroid_cos,
        },
        // No comparable member vector: nothing to count, so fall back to the
        // centroid — but scaled by how tight the cluster actually is. Normalizing
        // divided that spread out, which is what let a diffuse cluster present as
        // tight and keep absorbing; dividing it back in raises the bar exactly as
        // the cluster diversifies. A cluster with no recorded spread has
        // `cohesion == 1.0`, so this is the pre-coverage rule unchanged.
        None => DenseVerdict {
            admitted: centroid_cos >= policy.similarity / it.cohesion.max(f32::MIN_POSITIVE),
            covered: false,
            score: centroid_cos,
            mean_cos: centroid_cos,
            centroid_cos,
        },
    })
}

/// Order two dense verdicts: real coverage beats none, then how much of the
/// cluster matched, then how well. Callers append an id tie-break so the winner
/// never depends on iteration order.
fn rank_verdicts(a: &DenseVerdict, b: &DenseVerdict) -> std::cmp::Ordering {
    a.covered
        .cmp(&b.covered)
        .then_with(|| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            a.mean_cos
                .partial_cmp(&b.mean_cos)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// L2 norm. For an accumulator of unit vectors this doubles as the cluster's
/// **cohesion**: 1.0 when every member points the same way, `sqrt(n)/n` for n
/// mutually orthogonal ones — the spread that normalizing throws away.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Scale to unit length. A zero vector is returned unchanged — there is no
/// direction to preserve, and dividing would produce NaNs that would poison
/// every later comparison.
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = norm(&v);
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

/// Cosine similarity. Computed in full rather than as a bare dot product: the
/// contract says centroids are L2-normalized, but a producer that rounds or
/// truncates would otherwise silently depress every score.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Content tokens of a text: lowercased alphanumeric runs, minus a small
/// closed-class stopword list. Deliberately tiny — the lexical tier is a
/// fallback for catalogs with no embedder, not a search engine.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect()
}

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "why", "with", "you", "your",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(id: &str, members: &[&str], tools: &[(&str, f32)]) -> Intent {
        let mut it = Intent {
            id: id.into(),
            label: members.first().copied().unwrap_or_default().into(),
            terms: Vec::new(),
            members: members.iter().map(|m| m.to_string()).collect(),
            centroid: None,
            support: 5,
            seeded_support: 0,
            last_ts: 0,
            tools: tools.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            skills: BTreeMap::new(),
            bag: std::collections::HashSet::new(),
            member_bags: Vec::new(),
            vector_n: 0,
            cohesion: 1.0,
            mean: None,
            member_vectors: Vec::new(),
        };
        it.rebuild_bag(); // the cache is derived from members — keep them in step
        it
    }

    fn graph(intents: Vec<Intent>) -> IntentGraph {
        IntentGraph {
            v: 1,
            built_from_ts: 1_753_000_000_000,
            rev: 0,
            intents,
            model: None,
            cluster_policy: ClusterPolicy::default(),
            active_policy: ClusterPolicy::default(),
            pending: PendingQuery::default(),
            credit: CreditSlot::default(),
        }
    }

    fn all_known(_: &str) -> bool {
        true
    }

    // ---- mixed-graph serving fallback (#5) ---------------------------------

    #[test]
    fn a_vector_query_boosts_a_centroidless_cluster_when_dense_misses() {
        // Mixed graph: one dense cluster (fingerprinted) and one word-only
        // cluster carrying its own tool edge. A vector query orthogonal to the
        // dense centroid must still reach the word-only cluster lexically — its
        // learned evidence is otherwise permanently invisible to dense queries.
        let mut dense = intent(
            "dense",
            &["why is the build broken"],
            &[("gh_run_list", 1.0)],
        );
        dense.centroid = Some(normalize(vec![1.0, 0.0, 0.0]));
        let lexical = intent(
            "lexical",
            &["deploy the app to prod"],
            &[("deploy_tool", 1.0)],
        );
        let g = graph(vec![dense, lexical]);

        // Orthogonal to the dense centroid → dense miss; shares every word with
        // the word-only cluster → lexical hit.
        let arm = g.arm(
            "deploy the app to prod",
            Some(&[0.0, 1.0, 0.0]),
            Capability::Tool,
            &all_known,
        );

        let arm = arm.expect("word-only cluster must still boost on a dense miss");
        assert_eq!(arm.intent_id, "lexical");
        assert_eq!(arm.ids, vec!["deploy_tool".to_string()]);
    }

    #[test]
    fn a_dense_rejected_cluster_is_not_rescued_by_word_overlap() {
        // The guard on the fallback: a fingerprinted cluster that dense matching
        // rejected must NOT be pulled back by token overlap — that would let a
        // shallow word match override the embedder's "not similar" verdict.
        let mut dense = intent(
            "dense",
            &["deploy the app to prod"],
            &[("gh_run_list", 1.0)],
        );
        dense.centroid = Some(normalize(vec![1.0, 0.0, 0.0]));
        let g = graph(vec![dense]);

        // Orthogonal → dense miss; but the query shares every word with the
        // cluster's member, so an unrestricted lexical fallback would match it.
        let arm = g.arm(
            "deploy the app to prod",
            Some(&[0.0, 1.0, 0.0]),
            Capability::Tool,
            &all_known,
        );

        assert!(
            arm.is_none(),
            "a fingerprinted cluster dense rejected must not match lexically, got {arm:?}"
        );
    }

    // ---- centroid running mean ---------------------------------------------

    #[test]
    fn absorb_vector_weights_by_vectors_folded_not_member_count() {
        // A cluster that grew lexically (members added with no vector) must not
        // let those members inflate the running-mean weight — otherwise the first
        // vector after lexical growth dominates the centroid.
        let mut it = intent("i0", &["a", "b", "c", "d"], &[("t", 1.0)]);
        assert!(
            it.centroid.is_none(),
            "four lexical members, no centroid yet"
        );

        it.absorb_vector(&[1.0, 0.0, 0.0]); // first vector → centroid is e_x
        it.absorb_vector(&[0.0, 1.0, 0.0]); // second → equal-weight mean of the two

        // Two equal-weight orthogonal unit vectors → normalize(e_x + e_y).
        let c = it.centroid.as_ref().unwrap();
        assert!(
            (c[0] - c[1]).abs() < 1e-6,
            "equal-weight vectors → symmetric centroid, got {c:?}"
        );
        assert!(
            (c[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "expected ~0.707 per axis, got {c:?}"
        );
    }

    #[test]
    fn the_centroid_is_the_mean_of_its_members() {
        // Three mutually orthogonal unit vectors: the true mean is
        // `(e_x + e_y + e_z) / 3`, whose direction is `1/sqrt(3)` per axis.
        let mut it = intent("i0", &["a"], &[("t", 1.0)]);
        it.absorb_vector(&[1.0, 0.0, 0.0]);
        it.absorb_vector(&[0.0, 1.0, 0.0]);
        it.absorb_vector(&[0.0, 0.0, 1.0]);

        let c = it.centroid.as_ref().unwrap();
        let want = 1.0 / 3.0f32.sqrt();
        for (axis, x) in c.iter().enumerate() {
            assert!(
                (x - want).abs() < 1e-6,
                "axis {axis}: expected {want}, got {x} (centroid {c:?})"
            );
        }
    }

    #[test]
    fn the_accumulator_norm_falls_as_a_cluster_diversifies() {
        // `‖mean‖` IS the cluster's spread: 1.0 when every member points the same
        // way, `sqrt(n)/n` for n mutually orthogonal ones. It is the signal a
        // diffuse cluster needs to price itself out of new members.
        //
        // The pre-accumulator code rebuilt the mean from the RENORMALIZED
        // centroid — `c * k + v`, where `c` has length 1, so `c * k` has length
        // `k` no matter how far apart the members actually were. That stretched
        // the history back to full length at every step, and this sequence read
        // 1.000 / 0.707 / 0.745 / 0.825: *rising* exactly as the cluster spread
        // out, which is backwards.
        let mut it = intent("i0", &["a"], &[("t", 1.0)]);
        for n in 1..=4usize {
            let mut v = vec![0.0f32; 4];
            v[n - 1] = 1.0;
            it.absorb_vector(&v);

            let want = (n as f32).sqrt() / n as f32;
            let got = norm(it.mean.as_deref().unwrap());
            assert!(
                (got - want).abs() < 1e-6,
                "after {n} orthogonal vectors: expected ‖mean‖ {want}, got {got}"
            );
        }
    }

    #[test]
    fn a_reloaded_centroid_folds_as_a_single_prior_sample() {
        // The accumulator is derived state and never crosses the wire, so a
        // reloaded centroid arrives normalized with no record of how many members
        // it averaged. Seeding the accumulator with it at weight 1 reproduces
        // exactly what the pre-accumulator code did (`vector_n.max(1)`), so a
        // round-trip does not change what the next observation does — and, more
        // importantly, does not DISCARD the centroid by starting fresh.
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "m");
        g.observe_live("build broken", Capability::Tool, "a", T0, true);

        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();
        let mut it = back.intents[0].clone();
        it.absorb_vector(&[0.0, 1.0, 0.0]);

        // Equal weight with the reloaded centroid → normalize(e_x + e_y).
        let c = it.centroid.as_ref().unwrap();
        assert!(
            (c[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6
                && (c[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "expected ~0.707 per axis, got {c:?}"
        );
    }

    #[test]
    fn rebuild_centroids_resets_the_fold_count_to_what_it_rebuilt() {
        // Rebuild rewrote the centroid but left the stale fold count. Combined
        // with the count never crossing the wire, a loaded-then-rebuilt cluster
        // resumed its running mean at k=1 — so the next single observation yanked
        // a whole cluster's centroid halfway across, no matter how many members
        // it actually held.
        let mut g = graph(vec![intent("i0", &["a", "b", "c"], &[("t", 1.0)])]);
        let along_x = vec![vec![1.0f32, 0.0, 0.0]; 3];
        g.rebuild_centroids(vec![("i0".into(), along_x)], "m".into());

        let mut it = g.intents[0].clone();
        it.absorb_vector(&[0.0, 1.0, 0.0]);

        // Three members at e_x plus one at e_y → mean (0.75, 0.25, 0).
        let c = it.centroid.as_ref().unwrap();
        let want = normalize(vec![0.75, 0.25, 0.0]);
        assert!(
            (c[0] - want[0]).abs() < 1e-6 && (c[1] - want[1]).abs() < 1e-6,
            "expected {want:?} (one new vector against three), got {c:?} \
             — 0.707 per axis means the fold count was reset to 1"
        );
    }

    #[test]
    fn a_centroid_set_outside_the_accumulator_is_adopted_not_discarded() {
        // The accumulator is the fold's source of truth, so a cluster carrying a
        // centroid it never accumulated (a producer-built graph, a direct write)
        // must be adopted rather than dropped on the next fold — otherwise the
        // first live observation silently erases everything the producer knew.
        let mut it = dense_intent("i0", vec![1.0, 0.0, 0.0]);
        assert!(
            it.mean.is_none(),
            "centroid written directly, no accumulator"
        );

        it.absorb_vector(&[0.0, 1.0, 0.0]);

        let c = it.centroid.as_ref().unwrap();
        assert!(
            (c[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "expected the prior centroid at weight 1 → ~0.707 per axis, got {c:?}"
        );
    }

    #[test]
    fn medoid_labels_the_most_central_member_not_the_alphabetical_first() {
        // The old scoring was a constant 1.0 (every member's tokens are in the
        // union bag), so the label was just the tie-break — the alphabetically
        // first member. It should be the member most shared with the rest.
        let g = graph(vec![intent(
            "i0",
            &[
                "a lonely unique phrase",
                "build broken ci",
                "build broken pipeline",
            ],
            &[("t", 1.0)],
        )]);
        // "a lonely…" sorts first but shares no tokens; the build-broken members
        // are central. Most-covered wins, tie broken alphabetically among equals.
        assert_eq!(g.medoid(0), "build broken ci");
    }

    // ---- support ramp ------------------------------------------------------

    #[test]
    fn support_ramps_the_arm_weight_then_caps() {
        assert!((usage_weight(1) - USAGE_WEIGHT / 3.0).abs() < 1e-6);
        assert!((usage_weight(2) - USAGE_WEIGHT * 2.0 / 3.0).abs() < 1e-6);
        assert!((usage_weight(3) - USAGE_WEIGHT).abs() < 1e-6);
        assert!((usage_weight(900) - USAGE_WEIGHT).abs() < 1e-6);
    }

    #[test]
    fn a_single_observation_is_weaker_than_a_confirmed_cluster() {
        // The whole point of the ramp: one misclick must not rank like a pattern.
        assert!(usage_weight(1) < usage_weight(3));
    }

    // ---- parsing -----------------------------------------------------------

    #[test]
    fn parses_a_graph_without_a_centroid() {
        // The Bm25 / Jaccard-producer case: `centroid` is optional by contract.
        let json = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "tools":{"t":1.0},"skills":{}}]}"#;
        let g = IntentGraph::from_json(json).expect("valid graph");
        assert_eq!(g.len(), 1);
        assert!(g.intents[0].centroid.is_none());
    }

    #[test]
    fn rejects_an_unknown_version_instead_of_degrading() {
        let json = r#"{"v":2,"built_from_ts":1,"intents":[]}"#;
        assert_eq!(
            IntentGraph::from_json(json),
            Err(IntentGraphError::UnsupportedVersion(2))
        );
    }

    #[test]
    fn rejects_malformed_bytes() {
        assert!(matches!(
            IntentGraph::from_json("not json"),
            Err(IntentGraphError::Malformed(_))
        ));
    }

    // ---- seeded_support: baseline provenance -------------------------------

    #[test]
    fn a_seeded_observation_records_provenance_beside_support() {
        let mut g = IntentGraph::empty();
        g.observe(Observation {
            query: "why is the build broken",
            kind: Capability::Tool,
            capability_id: "gh_run_list",
            ts_ms: T0,
            first_confirmation: true,
            seeded: true,
        });
        assert_eq!(g.intents[0].support, 1);
        assert_eq!(g.intents[0].seeded_support, 1);
    }

    #[test]
    fn a_live_observation_leaves_seeded_support_alone() {
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        assert_eq!(g.intents[0].support, 1);
        assert_eq!(g.intents[0].seeded_support, 0);
    }

    #[test]
    fn a_seeded_observation_that_adds_only_an_edge_does_not_bump_seeded_support() {
        // The `search_capabilities` fan-out shape: one question, two edges, one
        // observation. Bumping seeded_support per EDGE rather than per
        // observation would break `seeded_support <= support` on the very first
        // fanned-out baseline turn.
        let mut g = IntentGraph::empty();
        let obs = |id, first| Observation {
            query: "why is the build broken",
            kind: Capability::Tool,
            capability_id: id,
            ts_ms: T0,
            first_confirmation: first,
            seeded: true,
        };
        g.observe(obs("gh_run_list", true));
        g.observe(obs("gh_run_view", false));

        assert_eq!(g.intents[0].tools.len(), 2, "two capabilities were used");
        assert_eq!(g.intents[0].support, 1, "but one question was asked");
        assert_eq!(g.intents[0].seeded_support, 1, "and it was one seeded turn");
    }

    #[test]
    fn seeded_and_live_observations_accumulate_in_the_same_cluster() {
        // The post-flip state: a seeded base that live traffic builds on. The
        // gap between the two counts is how much of the cluster's confidence
        // came from the baseline.
        let mut g = IntentGraph::empty();
        g.observe(Observation {
            query: "why is the build broken",
            kind: Capability::Tool,
            capability_id: "gh_run_list",
            ts_ms: T0,
            first_confirmation: true,
            seeded: true,
        });
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );

        assert_eq!(g.intents[0].support, 2);
        assert_eq!(g.intents[0].seeded_support, 1);
    }

    #[test]
    fn a_graph_with_no_seeded_observations_serializes_without_the_field() {
        // Zero-skip keeps a live-only graph byte-identical to one produced
        // before the field existed, so existing wire fixtures do not move.
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("seeded_support"),
            "absent means zero; got {json}"
        );
    }

    #[test]
    fn seeded_support_round_trips_through_the_wire_form() {
        let mut g = IntentGraph::empty();
        g.observe(Observation {
            query: "why is the build broken",
            kind: Capability::Tool,
            capability_id: "t",
            ts_ms: T0,
            first_confirmation: true,
            seeded: true,
        });
        let json = serde_json::to_string(&g).unwrap();
        assert!(json.contains(r#""seeded_support":1"#), "got {json}");
        let back = IntentGraph::from_json(&json).unwrap();
        assert_eq!(back.intents[0].seeded_support, 1);
    }

    #[test]
    fn a_graph_without_seeded_support_loads_as_zero() {
        let json = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "tools":{"t":1.0},"skills":{}}]}"#;
        let g = IntentGraph::from_json(json).expect("valid graph");
        assert_eq!(g.intents[0].seeded_support, 0);
    }

    #[test]
    fn a_graph_whose_seeded_support_exceeds_its_support_is_rejected() {
        // seeded_support counts a SUBSET of support; no producer can emit a
        // larger value, so accepting one would mean trusting a broken producer.
        let json = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "seeded_support":3,"tools":{"t":1.0},"skills":{}}]}"#;
        assert!(matches!(
            IntentGraph::from_json(json),
            Err(IntentGraphError::Malformed(_))
        ));
    }

    #[test]
    fn seeded_support_equal_to_support_is_the_normal_post_seeding_state() {
        let json = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":4,
            "seeded_support":4,"tools":{"t":1.0},"skills":{}}]}"#;
        assert!(IntentGraph::from_json(json).is_ok());
    }

    #[test]
    fn two_graphs_differing_only_in_seeded_support_compare_equal() {
        // Identity is the evidence — members, centroid, support, edges.
        // Provenance is not evidence: a re-seeded graph must compare equal to an
        // equivalent live one, or round-trip and rebuild assertions start
        // failing for a field that cannot affect retrieval.
        let live = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "tools":{"t":1.0},"skills":{}}]}"#;
        let seeded = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "seeded_support":2,"tools":{"t":1.0},"skills":{}}]}"#;
        assert_eq!(
            IntentGraph::from_json(live).unwrap(),
            IntentGraph::from_json(seeded).unwrap()
        );
    }

    // ---- rev: the persistence write-counter --------------------------------

    #[test]
    fn observe_bumps_rev_once_per_mutation() {
        let mut g = IntentGraph::empty();
        assert_eq!(g.rev(), 0, "an empty graph has written nothing");
        g.observe_live("build broken", Capability::Tool, "a", T0, true);
        assert_eq!(g.rev(), 1);
        // A second observe on the same search adds an edge — a real change even
        // though it seeds no new member — so it must still count as one write.
        g.observe_live("build broken", Capability::Tool, "b", T0, false);
        assert_eq!(g.rev(), 2);
    }

    #[test]
    fn a_no_op_observe_does_not_bump_rev() {
        // No words to cluster on and no stashed vector: `observe` returns before
        // changing anything, so the write-counter must not move. Guards against a
        // "bump unconditionally" regression.
        let mut g = IntentGraph::empty();
        g.observe_live("   ", Capability::Tool, "a", T0, true);
        assert_eq!(g.len(), 0);
        assert_eq!(g.rev(), 0);
    }

    #[test]
    fn rev_survives_a_round_trip() {
        let mut g = IntentGraph::empty();
        g.observe_live("build broken", Capability::Tool, "a", T0, true);
        g.observe_live("rotate the signing key", Capability::Tool, "b", T0, true);
        let before = g.rev();
        assert_eq!(before, 2);
        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back.rev(), before, "rev must persist across the wire form");
    }

    #[test]
    fn a_graph_without_rev_loads_as_zero_then_continues() {
        // An older or cloud-built graph carries no `rev`; it loads as 0 and the
        // counter continues up from there — monotonic across the gap.
        let json = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "tools":{"t":1.0},"skills":{}}]}"#;
        let mut g = IntentGraph::from_json(json).expect("valid graph");
        assert_eq!(g.rev(), 0);
        g.observe_live("something new", Capability::Tool, "t", T0, true);
        assert_eq!(g.rev(), 1);
    }

    #[test]
    fn an_unknown_field_is_ignored_on_load() {
        // Forward compatibility: a field a future build adds must be dropped, not
        // rejected — both at the graph and the intent level. Locks the current
        // (no `deny_unknown_fields`) behavior against regression.
        let json = r#"{"v":1,"built_from_ts":1,"future_top_level":42,
            "intents":[{"id":"i0","label":"l","members":["q"],"support":2,
            "tools":{"t":1.0},"skills":{},"future_intent_field":"x"}]}"#;
        let g = IntentGraph::from_json(json).expect("unknown fields must be ignored");
        assert_eq!(g.len(), 1);
        assert_eq!(g.intents[0].support, 2);
    }

    #[test]
    fn an_empty_graph_contributes_no_arm() {
        let g = IntentGraph::empty();
        assert!(g.is_empty());
        assert!(
            g.arm_lexical("anything", Capability::Tool, &all_known)
                .is_none()
        );
        assert!(g.arm_dense(&[1.0], Capability::Tool, &all_known).is_none());
    }

    // ---- dense matching ----------------------------------------------------

    #[test]
    fn dense_match_returns_edges_best_first() {
        let mut it = intent("i0", &["why is the build broken"], &[]);
        it.centroid = Some(vec![1.0, 0.0, 0.0]);
        it.tools = [
            ("gh_run_view".to_string(), 0.2),
            ("gh_run_list".to_string(), 0.8),
        ]
        .into_iter()
        .collect();
        let g = graph(vec![it]);
        let arm = g
            .arm_dense(&[1.0, 0.0, 0.0], Capability::Tool, &all_known)
            .expect("exact match");
        assert_eq!(arm.ids, vec!["gh_run_list", "gh_run_view"]);
        assert_eq!(arm.intent_id, "i0");
    }

    #[test]
    fn dense_match_below_tau_yields_no_arm() {
        let mut it = intent("i0", &["q"], &[("t", 1.0)]);
        it.centroid = Some(vec![1.0, 0.0]);
        let g = graph(vec![it]);
        // Orthogonal query: cosine 0, far below TAU_COSINE.
        assert!(
            g.arm_dense(&[0.0, 1.0], Capability::Tool, &all_known)
                .is_none()
        );
    }

    #[test]
    fn dense_match_skips_centroids_of_a_different_dimension() {
        // A changed embedding model must never be compared across vector spaces.
        let mut it = intent("i0", &["q"], &[("t", 1.0)]);
        it.centroid = Some(vec![1.0, 0.0, 0.0]);
        let g = graph(vec![it]);
        assert!(
            g.arm_dense(&[1.0, 0.0], Capability::Tool, &all_known)
                .is_none()
        );
    }

    #[test]
    fn dense_match_picks_the_closest_of_several_clusters() {
        let mut a = intent("a", &["q"], &[("ta", 1.0)]);
        a.centroid = Some(vec![1.0, 0.0]);
        let mut b = intent("b", &["q"], &[("tb", 1.0)]);
        b.centroid = Some(vec![0.8, 0.6]);
        let g = graph(vec![a, b]);
        let arm = g
            .arm_dense(&[0.8, 0.6], Capability::Tool, &all_known)
            .expect("match");
        assert_eq!(arm.intent_id, "b");
    }

    // ---- lexical matching --------------------------------------------------

    #[test]
    fn lexical_match_finds_a_repeat_phrasing() {
        let g = graph(vec![intent(
            "i0",
            &["why is the build broken", "is the build green"],
            &[("gh_run_list", 1.0)],
        )]);
        let arm = g
            .arm_lexical("is the build broken", Capability::Tool, &all_known)
            .expect("shares 'build' and 'broken'");
        assert_eq!(arm.ids, vec!["gh_run_list"]);
    }

    #[test]
    fn lexical_match_cannot_bridge_disjoint_vocabulary() {
        // The documented ceiling of the Bm25 tier (ADR-0014): no shared content
        // tokens means no match, however semantically close the two queries are.
        // This is what the dense tier exists to fix — pinned so the boundary is a
        // test, not a claim in prose.
        let g = graph(vec![intent(
            "i0",
            &["why is the build broken"],
            &[("gh_run_list", 1.0)],
        )]);
        assert!(
            g.arm_lexical("did CI pass", Capability::Tool, &all_known)
                .is_none()
        );
    }

    #[test]
    fn lexical_match_ignores_stopwords_only_queries() {
        let g = graph(vec![intent("i0", &["build"], &[("t", 1.0)])]);
        assert!(
            g.arm_lexical("is the", Capability::Tool, &all_known)
                .is_none()
        );
    }

    // ---- edge filtering ----------------------------------------------------

    #[test]
    fn edges_naming_capabilities_the_registry_lacks_are_dropped() {
        // A graph outlives a catalog change; ranking a ghost id would surface a
        // capability that cannot be invoked.
        let g = graph(vec![intent(
            "i0",
            &["build broken"],
            &[("gh_run_list", 0.8), ("since_deleted", 0.9)],
        )]);
        let arm = g
            .arm_lexical("build broken", Capability::Tool, &|id| {
                id != "since_deleted"
            })
            .expect("match");
        assert_eq!(arm.ids, vec!["gh_run_list"]);
    }

    #[test]
    fn a_match_whose_every_edge_is_gone_yields_no_arm() {
        let g = graph(vec![intent("i0", &["build broken"], &[("gone", 1.0)])]);
        let outcome = g.arm_lexical("build broken", Capability::Tool, &|_| false);
        assert!(outcome.is_none(), "nothing reaches the fusion");
        // ...but the cluster DID match, and saying so is the whole point: a
        // caller that only saw "no arm" would read catalog drift as a coverage
        // gap and re-derive a graph that was never the problem.
        assert_eq!(
            outcome,
            ArmOutcome::AllFiltered {
                intent_id: "i0".into(),
                similarity: 1.0,
                support: 5,
                dropped: 1,
            }
        );
    }

    #[test]
    fn a_cluster_with_no_edges_of_the_asked_kind_is_a_plain_miss() {
        // A tools-only cluster asked for skills has nothing to drop — that is
        // not drift, and reporting it as such would cry wolf on every mixed
        // graph.
        let g = graph(vec![intent("i0", &["build broken"], &[("a_tool", 1.0)])]);
        assert_eq!(
            g.arm_lexical("build broken", Capability::Skill, &all_known),
            ArmOutcome::NoMatch
        );
    }

    #[test]
    fn tool_and_skill_edges_are_ranked_independently() {
        let mut it = intent("i0", &["build broken"], &[("a_tool", 1.0)]);
        it.skills = [("a_skill".to_string(), 1.0)].into_iter().collect();
        let g = graph(vec![it]);
        assert_eq!(
            g.arm_lexical("build broken", Capability::Tool, &all_known)
                .unwrap()
                .ids,
            vec!["a_tool"]
        );
        assert_eq!(
            g.arm_lexical("build broken", Capability::Skill, &all_known)
                .unwrap()
                .ids,
            vec!["a_skill"]
        );
    }

    #[test]
    fn a_capability_spread_across_clusters_loses_to_one_specific_to_this_intent() {
        // Sorting edges by raw count alone lets a capability rank on volume
        // rather than on answering *this* question. Here every edge in the
        // matched cluster sits at one observation, so the order is decided
        // entirely by the tie-break — and by id ascending that hands it to
        // `apply_migration`, a capability every other cluster also reaches for,
        // over one only this cluster has ever used.
        let mut intents: Vec<Intent> = (0..4)
            .map(|i| {
                intent(
                    &format!("other_{i}"),
                    &[&format!("unrelated question {i}")],
                    &[("apply_migration", 3.0)],
                )
            })
            .collect();
        intents.insert(
            0,
            intent(
                "matched",
                &["build broken"],
                &[("apply_migration", 1.0), ("gh_run_list", 1.0)],
            ),
        );
        let g = graph(intents);

        let arm = g
            .arm_lexical("build broken", Capability::Tool, &all_known)
            .unwrap();
        assert_eq!(
            arm.ids,
            vec!["gh_run_list", "apply_migration"],
            "a capability that answers every intent identifies none of them"
        );
    }

    #[test]
    fn a_real_count_gap_still_beats_cluster_frequency() {
        // The guard on the other side. Down-weighting spread must not be strong
        // enough to invert genuine evidence: `gh_run_list` was chosen four times
        // here against one, and stays ahead even though it is the more widely
        // used of the two.
        let mut intents: Vec<Intent> = (0..4)
            .map(|i| {
                intent(
                    &format!("other_{i}"),
                    &[&format!("unrelated question {i}")],
                    &[("gh_run_list", 2.0)],
                )
            })
            .collect();
        intents.insert(
            0,
            intent(
                "matched",
                &["build broken"],
                &[("gh_run_list", 4.0), ("apply_migration", 1.0)],
            ),
        );
        let g = graph(intents);

        let arm = g
            .arm_lexical("build broken", Capability::Tool, &all_known)
            .unwrap();
        assert_eq!(arm.ids, vec!["gh_run_list", "apply_migration"]);
    }

    #[test]
    fn cluster_frequency_is_read_from_the_graph_not_the_registry() {
        // `known` is a REGISTRY closure, so counting cluster frequency over
        // surviving edges would make the statistic depend on which catalog is
        // attached: two agents sharing a graph would rank the same cluster
        // differently, and the harness — which knows everything — would disagree
        // with both. Counting over the raw edges keeps it a property of the graph.
        let mut intents: Vec<Intent> = (0..4)
            .map(|i| {
                intent(
                    &format!("other_{i}"),
                    &[&format!("unrelated question {i}")],
                    &[("apply_migration", 3.0)],
                )
            })
            .collect();
        intents.insert(
            0,
            intent(
                "matched",
                &["build broken"],
                &[("apply_migration", 1.0), ("gh_run_list", 1.0)],
            ),
        );
        let g = graph(intents);

        // A registry that has never heard of the other clusters' capability —
        // it is dropped from the arm, but it still counted toward the spread.
        let narrow = |id: &str| id != "ghost";
        let arm = g
            .arm_lexical("build broken", Capability::Tool, &narrow)
            .unwrap();
        assert_eq!(
            arm.ids,
            vec!["gh_run_list", "apply_migration"],
            "the order must not depend on what the registry happens to define"
        );
    }

    #[test]
    fn edges_rank_by_weight_not_by_id() {
        // The edges live in a BTreeMap, which already iterates id-ascending — so a
        // fixture whose weight order happens to agree with alphabetical order proves
        // nothing about the sort. Here they DISAGREE: `zulu` is the strongest edge
        // and must lead despite sorting last by id.
        let g = graph(vec![intent(
            "i0",
            &["build broken"],
            &[("alpha", 0.1), ("mike", 0.5), ("zulu", 0.9)],
        )]);
        let arm = g
            .arm_lexical("build broken", Capability::Tool, &all_known)
            .unwrap();
        assert_eq!(arm.ids, vec!["zulu", "mike", "alpha"]);
    }

    #[test]
    fn tied_edge_weights_break_by_id_ascending() {
        // One cluster, so every id has cluster frequency 1 and the inverse-
        // cluster-frequency weight is exactly 1.0 — this pins the tie-break that
        // remains once cf has nothing to say. The multi-cluster case, where cf
        // decides instead, is the sibling below.
        let g = graph(vec![intent(
            "i0",
            &["build broken"],
            &[("zeta", 1.0), ("alpha", 1.0), ("mid", 1.0)],
        )]);
        let arm = g
            .arm_lexical("build broken", Capability::Tool, &all_known)
            .unwrap();
        assert_eq!(arm.ids, vec!["alpha", "mid", "zeta"]);
    }

    #[test]
    fn round_trips_through_json() {
        let mut it = intent("i0", &["q"], &[("t", 1.0)]);
        it.centroid = Some(vec![0.8, 0.6]);
        let g = graph(vec![it]);
        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(g, back);
    }

    // ---- observe: the online learning step ---------------------------------

    const T0: u64 = 1_753_000_000_000;

    #[test]
    fn the_first_observation_seeds_a_cluster() {
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );

        assert_eq!(g.len(), 1);
        assert_eq!(g.intents[0].support, 1);
        assert_eq!(g.intents[0].members, vec!["why is the build broken"]);
        assert_eq!(g.intents[0].tools.get("gh_run_list"), Some(&1.0));
        // Grown lexically, so no centroid — `arm` must still match it.
        assert!(g.intents[0].centroid.is_none());
    }

    #[test]
    fn a_similar_query_joins_the_existing_cluster() {
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe_live(
            "is the build broken now",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );

        assert_eq!(g.len(), 1, "should not have seeded a second cluster");
        assert_eq!(g.intents[0].support, 2);
        assert_eq!(g.intents[0].tools.get("gh_run_list"), Some(&2.0));
    }

    #[test]
    fn a_dissimilar_query_seeds_its_own_cluster() {
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe_live(
            "rotate the signing key",
            Capability::Tool,
            "vault_rotate",
            T0,
            true,
        );

        assert_eq!(g.len(), 2);
        let ids: Vec<&str> = g.intents.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, vec!["intent_0", "intent_1"]);
    }

    #[test]
    fn a_repeated_phrasing_is_not_duplicated_in_members() {
        // Members are the match key; repeating one must not inflate the token
        // bag and make the cluster match ever more loosely.
        let mut g = IntentGraph::empty();
        for _ in 0..3 {
            g.observe_live(
                "why is the build broken",
                Capability::Tool,
                "gh_run_list",
                T0,
                true,
            );
        }
        assert_eq!(g.intents[0].members.len(), 1);
        assert_eq!(
            g.intents[0].support, 3,
            "support still counts every observation"
        );
    }

    #[test]
    fn learning_then_searching_closes_the_loop() {
        // The whole feature in one assertion: observe, then match a query that
        // was never observed verbatim.
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe_live(
            "is the build broken again",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );

        let arm = g
            .arm(
                "the build broken on main",
                None,
                Capability::Tool,
                &all_known,
            )
            .expect("a near-repeat of a member");
        assert_eq!(arm.ids, vec!["gh_run_list"]);
        assert_eq!(arm.support, 2);
    }

    #[test]
    fn the_lexical_tier_does_not_reach_distant_wording() {
        // "is the build ok" and "why is the build broken" are the same question,
        // and this tier will not connect them — they share one word out of two,
        // which is indistinguishable from two unrelated asks that happen to
        // share a word (`one_shared_word_does_not_merge_distinct_intents`).
        //
        // No word-overlap rule can accept one and reject the other, so this tier
        // rejects both: a false merge degrades ranking, a false split only misses
        // a boost. Bridging distant wording is the dense tier's job.
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        assert!(
            g.arm("is the build ok", None, Capability::Tool, &all_known)
                .is_none()
        );
    }

    #[test]
    fn a_lexically_grown_graph_is_matchable_even_when_a_query_vector_is_offered() {
        // A semantic catalog hands `arm` a query vector, but a locally-learned
        // graph has no centroids to compare it against. It must fall back to
        // lexical matching rather than silently returning nothing.
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );

        let arm = g.arm(
            "why is the build broken",
            Some(&[0.1, 0.2, 0.3]),
            Capability::Tool,
            &all_known,
        );
        assert!(arm.is_some(), "must not be invisible to a semantic catalog");
    }

    #[test]
    fn edges_rank_by_how_often_a_capability_was_chosen() {
        let mut g = IntentGraph::empty();
        for _ in 0..3 {
            g.observe_live(
                "why is the build broken",
                Capability::Tool,
                "chosen_often",
                T0,
                true,
            );
        }
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "chosen_once",
            T0,
            true,
        );

        let arm = g
            .arm(
                "why is the build broken",
                None,
                Capability::Tool,
                &all_known,
            )
            .unwrap();
        assert_eq!(arm.ids, vec!["chosen_often", "chosen_once"]);
    }

    #[test]
    fn built_from_ts_tracks_the_newest_event_and_never_rewinds() {
        // Provenance only — it says how current the graph is. Traces are loosely
        // ordered (ADR-0007), so a late-arriving older event must not drag it back.
        let mut g = IntentGraph::empty();
        g.observe_live("build broken", Capability::Tool, "a", T0 + 10, true);
        g.observe_live("build broken", Capability::Tool, "b", T0, true);
        assert_eq!(g.built_from_ts, T0 + 10);
    }

    #[test]
    fn the_token_cache_stays_in_step_with_members() {
        // The cache is derived from `members`; if the two drift, a query stops
        // matching a cluster that plainly covers it. Silent, and invisible to
        // every other test — so pin it directly.
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe_live("the pipeline is broken", Capability::Tool, "t", T0, true);

        let it = &g.intents[0];
        let fresh: std::collections::HashSet<String> =
            it.members.iter().flat_map(|m| tokenize(m)).collect();
        assert_eq!(&it.bag, &fresh, "union cache drifted from members");

        // The per-member sets are what scoring actually reads, and they are
        // positional — a drift here silently stops a cluster matching queries it
        // plainly covers.
        assert_eq!(it.member_bags.len(), it.members.len(), "one set per member");
        for (m, bag) in it.members.iter().zip(&it.member_bags) {
            let fresh: std::collections::HashSet<String> = tokenize(m).into_iter().collect();
            assert_eq!(bag, &fresh, "member set drifted for {m:?}");
        }
    }

    #[test]
    fn a_deserialized_graph_can_still_match_lexically() {
        // The cache is skipped on the wire, so `from_json` must rebuild it —
        // otherwise a reloaded graph silently matches nothing.
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();

        assert!(
            back.arm(
                "why is the build broken",
                None,
                Capability::Tool,
                &all_known
            )
            .is_some(),
            "a reloaded graph must still match"
        );
    }

    #[test]
    fn the_centroid_is_folded_once_per_distinct_member() {
        // The centroid is the mean of the cluster's DISTINCT member texts, so
        // extra invokes from one search must not fold that query's vector again.
        //
        // Two members are essential here: with a single member the running mean
        // is `c*(n-1) + v = v`, so re-folding is idempotent and a one-member
        // fixture passes even when the guard is removed.
        let v1 = [1.0f32, 0.0, 0.0];
        let v2 = [0.0f32, 1.0, 0.0];
        let build = |extra: usize| {
            let mut g = IntentGraph::empty();
            g.note_query_vector("build broken", &v1, "m");
            g.observe_live("build broken", Capability::Tool, "a", T0, true);
            g.note_query_vector("build broken again", &v2, "m");
            g.observe_live("build broken again", Capability::Tool, "b", T0, true);
            for i in 0..extra {
                g.observe_live(
                    "build broken again",
                    Capability::Tool,
                    &format!("x{i}"),
                    T0,
                    false,
                );
            }
            g.intents[0].centroid.clone().unwrap()
        };

        let once = build(0);
        let with_extra_invokes = build(3);
        for (a, b) in once.iter().zip(&with_extra_invokes) {
            assert!(
                (a - b).abs() < 1e-6,
                "extra invokes moved the centroid: {once:?} vs {with_extra_invokes:?}"
            );
        }
    }

    #[test]
    fn a_later_invoke_landing_elsewhere_still_has_support() {
        // A cluster moves as it learns, so a second invoke from the same search
        // can match a DIFFERENT cluster than the first did. That cluster is new,
        // so it must still start at 1 — `protocol/v1` requires support >= 1, and
        // a zero-support cluster would contribute a weightless arm.
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "a", T0, false);
        assert_eq!(g.intents[0].support, 1);
    }

    // ---- lexical clustering must not over-merge -----------------------------

    #[test]
    fn one_shared_word_does_not_merge_distinct_intents() {
        // The bug, minimally. Two unrelated asks sharing a single word were
        // exactly 50% "covered" by each other and merged.
        let mut g = IntentGraph::empty();
        g.observe_live("deploy0 rollback3", Capability::Tool, "a", T0, true);
        g.observe_live("deploy0 migrate5", Capability::Tool, "b", T0, true);
        assert_eq!(g.len(), 2, "one shared word is not the same question");
    }

    #[test]
    fn a_large_cluster_does_not_absorb_an_unrelated_query() {
        // The runaway: the old score was measured against the UNION of every
        // member, which only grows — so a mature cluster recognized most of the
        // vocabulary and swallowed anything, which grew it further.
        let mut g = IntentGraph::empty();
        for i in 0..30 {
            g.observe_live(
                &format!("build broken variant{i}"),
                Capability::Tool,
                "gh_run_list",
                T0,
                true,
            );
        }
        assert_eq!(g.len(), 1, "those really are one ask");

        // Every word of this query appears somewhere in that cluster's 32-word
        // union — but no single member shares more than one of them. Scoring
        // against the union called it a perfect match; scoring against members
        // calls it 0.25.
        g.observe_live("variant7 variant12", Capability::Tool, "vault", T0, true);
        assert_eq!(
            g.len(),
            2,
            "a big cluster must not absorb by sheer vocabulary"
        );
    }

    #[test]
    fn distinct_topics_do_not_collapse_at_scale() {
        // Collapse only shows once unions have grown, which is why small
        // fixtures never caught it: 40 separable topics used to end up as 11
        // clusters. These phrasings are deliberately adversarial — two words
        // each, low overlap — so a HIGH cluster count is the right outcome here.
        // This asserts the absence of collapse; `near_repeats_still_merge`
        // covers the other direction.
        const WORDS: [&str; 20] = [
            "deploy", "rollback", "migrate", "schema", "invoice", "refund", "tenant", "webhook",
            "cursor", "throttle", "quota", "shard", "replica", "index", "vault", "rotate", "lease",
            "beacon", "harvest", "prune",
        ];
        let mut g = IntentGraph::empty();
        for topic in 0..40 {
            for phrasing in 0..10 {
                let q = format!(
                    "{}{topic} {}{phrasing}",
                    WORDS[topic % 20],
                    WORDS[(topic + phrasing) % 20]
                );
                g.observe_live(&q, Capability::Tool, &format!("t{topic}"), T0, true);
            }
        }
        assert!(
            g.len() >= 35,
            "40 distinct topics collapsed into {} clusters",
            g.len()
        );
    }

    #[test]
    fn near_repeats_still_merge() {
        // The fix must not over-split: rephrasings of one ask stay together.
        let mut g = IntentGraph::empty();
        for q in [
            "why is the build broken",
            "is the build broken again",
            "the build broken on main",
        ] {
            g.observe_live(q, Capability::Tool, "gh_run_list", T0, true);
        }
        for q in ["rotate the signing key", "rotate the signing key now"] {
            g.observe_live(q, Capability::Tool, "vault_rotate", T0, true);
        }
        assert_eq!(g.len(), 2, "two asks, however phrased");
        assert_eq!(g.intents[0].members.len(), 3);
        assert_eq!(g.intents[1].members.len(), 2);
    }

    // ---- labels ------------------------------------------------------------

    #[test]
    fn the_label_is_always_one_of_the_members() {
        // Counted from the data, so it cannot describe the cluster wrongly.
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe_live("is the build broken now", Capability::Tool, "t", T0, true);

        let it = &g.labeled()[0];
        assert!(
            it.members.contains(&it.label),
            "label {:?} not a member",
            it.label
        );
    }

    #[test]
    fn terms_distinguish_a_cluster_from_its_neighbours() {
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe_live("the build is broken again", Capability::Tool, "t", T0, true);
        g.observe_live("rotate the signing key", Capability::Tool, "v", T0, true);

        let build = &g.labeled()[0];
        assert!(
            build.terms.contains(&"build".to_string()),
            "got {:?}",
            build.terms
        );
        assert!(!build.terms.contains(&"rotate".to_string()));
    }

    #[test]
    fn terms_are_scored_against_the_graph_as_it_is_now() {
        // c-TF-IDF ranks a term by how rare it is across the OTHER clusters, so a
        // value frozen when a cluster was last written goes stale the moment the
        // graph grows. This used to be computed inside `observe`, which made every
        // label describe a graph that no longer existed.
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe_live("the build broken again", Capability::Tool, "t", T0, true);
        let alone = g.labeled()[0].terms.clone();

        // A second cluster that also uses "again" makes that term less
        // distinguishing for the first — which must be reflected even though the
        // first cluster was never touched again.
        g.observe_live(
            "tail the service log again",
            Capability::Tool,
            "u",
            T0,
            true,
        );
        let with_neighbour = g.labeled()[0].terms.clone();

        let rank = |terms: &[String], t: &str| terms.iter().position(|x| x == t);
        assert!(
            rank(&with_neighbour, "again") >= rank(&alone, "again"),
            "\"again\" should not gain rank once a neighbour shares it: {alone:?} -> {with_neighbour:?}"
        );
    }

    #[test]
    fn the_label_is_derived_not_stored() {
        // Nothing writes `label` during learning; it is materialized on read, so
        // two graphs holding the same evidence are equal regardless.
        let mut g = IntentGraph::empty();
        g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        assert!(
            g.intents[0].label.is_empty(),
            "not stored on the write path"
        );
        assert!(!g.labeled()[0].label.is_empty(), "materialized on read");
    }

    #[test]
    fn a_stopword_only_query_teaches_nothing() {
        let mut g = IntentGraph::empty();
        g.observe_live("is the", Capability::Tool, "t", T0, true);
        assert!(g.is_empty());
    }

    #[test]
    fn tool_and_skill_observations_land_on_separate_edge_maps() {
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe_live(
            "why is the build broken",
            Capability::Skill,
            "ci-triage",
            T0,
            true,
        );

        assert_eq!(g.len(), 1);
        assert_eq!(g.intents[0].tools.len(), 1);
        assert_eq!(g.intents[0].skills.len(), 1);
    }

    #[test]
    fn a_learned_graph_round_trips_through_the_wire_form() {
        let mut g = IntentGraph::empty();
        g.observe_live(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(g, back);
    }
    // ---- embedding-model change detection (centroids are model-specific) -----

    fn dense_intent(id: &str, centroid: Vec<f32>) -> Intent {
        let mut it = intent(id, &["why is the build broken"], &[("gh_run_list", 1.0)]);
        it.centroid = Some(normalize(centroid));
        it
    }

    #[test]
    fn model_status_ok_for_a_lexical_graph() {
        // No centroids → nothing model-specific → always usable.
        let g = graph(vec![intent("i0", &["build broken"], &[("t", 1.0)])]);
        assert_eq!(g.model_status("any-model", 384), GraphModelStatus::Ok);
    }

    #[test]
    fn model_status_flags_a_dimension_change() {
        let mut g = graph(vec![dense_intent("i0", vec![1.0, 0.0, 0.0])]);
        g.model = Some("bge-small".into());
        assert_eq!(
            g.model_status("bge-base", 768),
            GraphModelStatus::DimMismatch {
                built: 3,
                active: 768
            }
        );
    }

    #[test]
    fn model_status_flags_a_same_dim_model_change() {
        // The case a length check cannot catch: same width, different model.
        let mut g = graph(vec![dense_intent("i0", vec![1.0, 0.0, 0.0])]);
        g.model = Some("model-a".into());
        assert_eq!(
            g.model_status("model-b", 3),
            GraphModelStatus::ModelMismatch {
                built: "model-a".into(),
                active: "model-b".into()
            }
        );
    }

    #[test]
    fn model_status_ok_when_the_model_matches() {
        let mut g = graph(vec![dense_intent("i0", vec![1.0, 0.0, 0.0])]);
        g.model = Some("model-a".into());
        assert_eq!(g.model_status("model-a", 3), GraphModelStatus::Ok);
    }

    #[test]
    fn intent_graph_accepts_pre_pr_local_path_model_fingerprint() {
        let mut g = graph(vec![dense_intent("i0", vec![1.0, 0.0, 0.0])]);
        let pre_pr = "local|path=11:/models/foo";
        g.model = Some(pre_pr.into());
        assert_eq!(
            g.model_status(pre_pr, 3),
            GraphModelStatus::Ok,
            "pre-PR Local path fingerprints must remain IntentGraph-compatible"
        );
        assert!(matches!(
            g.model_status(
                "local|content=64:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                3
            ),
            GraphModelStatus::ModelMismatch { .. }
        ));
    }

    #[test]
    fn observe_stamps_the_model_on_the_first_centroid() {
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "model-a");
        g.observe_live("build broken", Capability::Tool, "t", T0, true);
        assert_eq!(g.model.as_deref(), Some("model-a"));
    }

    #[test]
    fn observe_freezes_the_centroid_on_a_model_change() {
        // Grow under model-a, then an observation arrives embedded by model-b.
        // Member/support must still update, but the centroid must NOT blend the
        // two vector spaces.
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "model-a");
        g.observe_live("build broken", Capability::Tool, "t", T0, true);
        let frozen = g.intents[0].centroid.clone();

        g.note_query_vector("build broken again", &[0.0, 1.0, 0.0], "model-b");
        g.observe_live("build broken again", Capability::Tool, "t", T0, true);

        assert_eq!(
            g.intents[0].centroid, frozen,
            "centroid must not blend models"
        );
        assert_eq!(g.intents[0].members.len(), 2, "member still recorded");
        assert_eq!(g.intents[0].support, 2, "support still counts");
        assert_eq!(g.model.as_deref(), Some("model-a"), "model unchanged");
    }

    #[test]
    fn rebuild_centroids_re_embeds_members_and_restamps() {
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "model-a");
        g.observe_live("build broken", Capability::Tool, "gh_run_list", T0, true);
        let rev_before = g.rev();

        // Members re-embedded under model-b (here, just different vectors).
        let id = g.intents[0].id.clone();
        g.rebuild_centroids(vec![(id, vec![vec![0.0, 1.0, 0.0]])], "model-b".into());

        // A rebuild is a persistable change — it must advance the write counter.
        assert_eq!(g.rev(), rev_before + 1);
        assert_eq!(g.model.as_deref(), Some("model-b"));
        assert_eq!(g.model_status("model-b", 3), GraphModelStatus::Ok);
        // Learning preserved.
        assert_eq!(g.intents[0].support, 1);
        assert_eq!(g.intents[0].tools.get("gh_run_list"), Some(&1.0));
        // Centroid moved to the new (normalized) vector.
        let c = g.intents[0].centroid.as_ref().unwrap();
        assert!((c[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rebuild_centroids_follows_ids_when_a_cluster_is_evicted_mid_rebuild() {
        // `rebuild_intent_graph` snapshots members, embeds WITHOUT the lock, then
        // re-locks to apply — a concurrent `observe()` can evict a cluster in
        // that window and shift positions. Centroids must reattach by id, not by
        // position, or a survivor silently inherits the evicted cluster's vector
        // (and the fresh model stamp hides it).
        let mut g = IntentGraph::empty();
        g.observe_live("alpha query", Capability::Tool, "a", T0, true);
        g.observe_live("bravo query", Capability::Tool, "b", T0, true);
        g.observe_live("charlie query", Capability::Tool, "c", T0, true);
        assert_eq!(g.len(), 3, "three disjoint queries → three clusters");
        let id_a = g.intents[0].id.clone();
        let id_b = g.intents[1].id.clone();
        let id_c = g.intents[2].id.clone();

        // Embeddings computed from the snapshot order [a, b, c], each a distinct
        // axis so a misassignment is unambiguous.
        let per_cluster = vec![
            (id_a.clone(), vec![vec![1.0, 0.0, 0.0]]),
            (id_b.clone(), vec![vec![0.0, 1.0, 0.0]]),
            (id_c.clone(), vec![vec![0.0, 0.0, 1.0]]),
        ];

        // ...but by apply time an observe() evicted cluster A, shifting b and c
        // down one slot. Position-zip would give b the [1,0,0] meant for a.
        g.intents.retain(|it| it.id != id_a);
        assert_eq!(g.len(), 2);

        g.rebuild_centroids(per_cluster, "model-b".into());

        let b = g.intents.iter().find(|it| it.id == id_b).unwrap();
        assert!(
            (b.centroid.as_ref().unwrap()[1] - 1.0).abs() < 1e-6,
            "cluster b must keep its own centroid, got {:?}",
            b.centroid
        );
        let c = g.intents.iter().find(|it| it.id == id_c).unwrap();
        assert!(
            (c.centroid.as_ref().unwrap()[2] - 1.0).abs() < 1e-6,
            "cluster c must keep its own centroid, got {:?}",
            c.centroid
        );
    }

    #[test]
    fn the_model_field_round_trips_through_json() {
        let mut g = graph(vec![dense_intent("i0", vec![0.6, 0.8, 0.0])]);
        g.model = Some("bge-small".into());
        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back.model.as_deref(), Some("bge-small"));
    }

    #[test]
    fn from_json_matches_the_conformance_valid_and_invalid_sets() {
        // The protocol mandates every consumer reject the `invalid` set and
        // accept the `valid` set (protocol/v1/conformance/vectors.json). Drive
        // both directly off the fixtures so this consumer stays conformant.
        let vectors: serde_json::Value = serde_json::from_str(include_str!(
            "../../../protocol/v1/conformance/vectors.json"
        ))
        .expect("conformance vectors parse");
        let graph = &vectors["graph"];

        for case in graph["invalid"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let doc = serde_json::to_string(&case["doc"]).unwrap();
            assert!(
                IntentGraph::from_json(&doc).is_err(),
                "invalid vector `{name}` must be rejected"
            );
        }
        for case in graph["valid"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let doc = serde_json::to_string(&case["doc"]).unwrap();
            assert!(
                IntentGraph::from_json(&doc).is_ok(),
                "valid vector `{name}` must be accepted"
            );
        }
    }

    #[test]
    fn a_lexical_graph_omits_the_model_on_the_wire() {
        let g = graph(vec![intent("i0", &["build broken"], &[("t", 1.0)])]);
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            !json.contains("model"),
            "no model field for a centroid-less graph"
        );
    }
    // ---- recency: decay, eviction, member cap (blocker #3) -----------------

    const DAY: u64 = 86_400_000;

    #[test]
    fn a_recent_cluster_keeps_full_weight_within_the_grace() {
        let mut g = IntentGraph::empty();
        for _ in 0..3 {
            g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        }
        // now == last_ts → Δt 0 → recency 1; support 3 → ramp 1.
        let arm = g
            .arm(
                "why is the build broken",
                None,
                Capability::Tool,
                &all_known,
            )
            .unwrap();
        assert!((arm.weight() - USAGE_WEIGHT).abs() < 1e-6);
    }

    #[test]
    fn a_stale_cluster_decays_after_the_grace() {
        let mut g = IntentGraph::empty();
        for _ in 0..3 {
            g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        }
        // Advance the graph's clock 200 days via a different topic.
        g.observe_live(
            "rotate the signing key",
            Capability::Tool,
            "v",
            T0 + 200 * DAY,
            true,
        );

        let arm = g
            .arm(
                "why is the build broken",
                None,
                Capability::Tool,
                &all_known,
            )
            .unwrap();
        let expected = USAGE_WEIGHT * 2f32.powf(-((200.0 - 90.0) / 90.0));
        assert!(
            (arm.weight() - expected).abs() < 1e-3,
            "got {} expected {expected}",
            arm.weight()
        );
        assert!(
            arm.weight() < USAGE_WEIGHT,
            "a cold cluster must weigh less"
        );
    }

    #[test]
    fn a_long_idle_cluster_is_evicted() {
        let mut g = IntentGraph::empty();
        for _ in 0..3 {
            g.observe_live("why is the build broken", Capability::Tool, "t", T0, true);
        }
        assert_eq!(g.len(), 1);
        // ~2 years of other activity later: the build cluster is past the floor.
        g.observe_live(
            "rotate the signing key",
            Capability::Tool,
            "v",
            T0 + 700 * DAY,
            true,
        );
        assert_eq!(
            g.len(),
            1,
            "the stale cluster was evicted, the fresh one stays"
        );
        assert!(
            g.arm(
                "why is the build broken",
                None,
                Capability::Tool,
                &all_known
            )
            .is_none(),
            "an evicted cluster contributes no arm"
        );
    }

    #[test]
    fn members_are_capped_per_cluster() {
        let mut g = IntentGraph::empty();
        for i in 0..MEMBER_CAP + 20 {
            g.observe_live(
                &format!("build broken variant{i}"),
                Capability::Tool,
                "t",
                T0,
                true,
            );
        }
        assert_eq!(g.len(), 1, "near-repeats form one cluster");
        assert!(
            g.intents[0].members.len() <= MEMBER_CAP,
            "members capped, got {}",
            g.intents[0].members.len()
        );
        // The token cache stays in step with the trimmed members.
        let fresh: std::collections::HashSet<String> = g.intents[0]
            .members
            .iter()
            .flat_map(|m| tokenize(m))
            .collect();
        assert_eq!(&g.intents[0].bag, &fresh);
    }

    #[test]
    fn a_tight_cluster_serializes_without_cohesion_or_fold_count() {
        // Both fields default to their identity values, so a cluster that has
        // nothing to say about spread must not say it — a graph written by this
        // build stays byte-identical to one written before the fields existed.
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "m");
        g.observe_live("build broken", Capability::Tool, "a", T0, true);

        let json = serde_json::to_string(&g).unwrap();
        assert!(!json.contains("cohesion"), "unexpected cohesion in {json}");
        assert!(!json.contains("vector_n"), "unexpected vector_n in {json}");
    }

    #[test]
    fn cohesion_and_the_fold_count_round_trip_and_rebuild_the_accumulator() {
        // Normalizing the centroid divides the spread out, so without these two
        // scalars a reloaded cluster cannot say how tightly its members agreed,
        // and its running mean restarts at one sample.
        let mut it = intent("i0", &[], &[("t", 1.0)]);
        it.absorb_member("a", Some(&topic_vec(0, 0)));
        it.absorb_member("b", Some(&topic_vec(1, 0)));
        it.last_ts = T0;
        let (cohesion, folded, mean) = (it.cohesion, it.vector_n, it.mean.clone().unwrap());
        assert!(cohesion < 1.0, "two topics should not read as tight");

        let g = graph(vec![it]);
        let json = serde_json::to_string(&g).unwrap();
        assert!(
            json.contains("cohesion"),
            "a diffuse cluster must record it"
        );

        let back = IntentGraph::from_json(&json).unwrap();
        let reloaded = &back.intents[0];
        assert!((reloaded.cohesion - cohesion).abs() < 1e-6);
        assert_eq!(reloaded.vector_n, folded);
        for (a, b) in reloaded.mean.as_ref().unwrap().iter().zip(&mean) {
            assert!((a - b).abs() < 1e-6, "accumulator not rebuilt: {a} vs {b}");
        }
    }

    #[test]
    fn a_reloaded_diffuse_cluster_raises_its_own_bar() {
        // The payoff, and the reason cohesion is on the wire at all. Member
        // vectors are not, so after a reload this cluster has no coverage to
        // count — and the centroid alone would admit the outsider at 0.72. Scaled
        // by how far apart the members actually are, the bar it has to clear is
        // higher than that, so the cluster stops absorbing instead of drifting
        // further.
        let mut it = intent("i0", &[], &[("t", 1.0)]);
        it.absorb_member("a", Some(&topic_vec(0, 0)));
        it.absorb_member("b", Some(&topic_vec(1, 0)));
        it.last_ts = T0;
        let g = graph(vec![it]);

        let outsider = topic_vec(2, 0);
        let centroid_cos = cosine(&outsider, g.intents[0].centroid.as_deref().unwrap());
        assert!(
            centroid_cos >= TAU_COSINE,
            "the unscaled centroid must still admit it, got {centroid_cos}"
        );

        let back = IntentGraph::from_json(&serde_json::to_string(&g).unwrap()).unwrap();
        assert!(
            back.intents[0].member_vectors.iter().all(Option::is_none),
            "member vectors do not cross the wire"
        );
        assert!(
            back.arm("outsider", Some(&outsider), Capability::Tool, &all_known)
                .is_none(),
            "a cluster spread this wide must not arm on a query no member knows"
        );
    }

    #[test]
    fn a_graph_whose_cohesion_is_out_of_range_is_rejected() {
        for bad in ["0.0", "-0.5", "1.5"] {
            let json = format!(
                r#"{{"v":1,"built_from_ts":1,"intents":[{{"id":"i0","label":"q","terms":[],
                   "members":["q"],"support":1,"tools":{{}},"skills":{{}},
                   "centroid":[1.0],"cohesion":{bad}}}]}}"#
            );
            assert!(
                IntentGraph::from_json(&json).is_err(),
                "cohesion {bad} is not a spread any producer can mean"
            );
        }
    }

    #[test]
    fn a_cohesion_without_a_centroid_is_rejected() {
        // Cohesion describes a centroid's spread. Recorded without one it is
        // unattached to anything, which means the producer lost a field.
        let json = r#"{"v":1,"built_from_ts":1,"intents":[{"id":"i0","label":"q","terms":[],
                       "members":["q"],"support":1,"tools":{},"skills":{},"cohesion":0.5}]}"#;
        assert!(IntentGraph::from_json(json).is_err());
    }

    // ---- the cluster policy -------------------------------------------------

    #[test]
    fn the_default_policy_is_the_constants_it_replaced() {
        let p = ClusterPolicy::default();
        assert_eq!(p.similarity, TAU_COSINE);
        assert_eq!(p.coverage, COVERAGE_FRACTION);
        assert!(p.is_valid());
    }

    #[test]
    fn a_stricter_policy_refuses_what_the_default_admits() {
        // The policy has to be read at the decision point, not just stored. A
        // refactor that threaded the parameter through and then ignored it would
        // pass every other test in this file, because they all run at the default.
        let mut it = intent("i0", &[], &[("t", 1.0)]);
        it.absorb_member("a", Some(&topic_vec(0, 0)));
        it.absorb_member("b", Some(&topic_vec(0, 1)));

        let query = topic_vec(0, 2); // ~0.98 against both members
        assert!(
            dense_verdict(&it, &query, ClusterPolicy::default()).is_some_and(|v| v.admitted),
            "the default must admit a same-topic query"
        );
        assert!(
            !dense_verdict(&it, &query, ClusterPolicy::default().with_similarity(0.995))
                .is_some_and(|v| v.admitted),
            "a threshold above the members' own similarity must refuse it"
        );
    }

    #[test]
    fn a_looser_coverage_admits_what_a_majority_refuses() {
        // The other knob, on its own: a query matching one member of three is a
        // third of the cluster — refused by a majority, admitted at a third.
        let mut it = intent("i0", &[], &[("t", 1.0)]);
        it.absorb_member("a", Some(&topic_vec(0, 0)));
        it.absorb_member("b", Some(&topic_vec(1, 0)));
        it.absorb_member("c", Some(&topic_vec(2, 0)));

        let query = topic_vec(0, 1); // ~0.98 to `a`, ~0.67 to the rest
        let at = |coverage| {
            dense_verdict(
                &it,
                &query,
                ClusterPolicy::default().with_coverage(coverage),
            )
            .is_some_and(|v| v.admitted)
        };
        assert!(!at(0.5), "one of three is not a majority");
        // The floor of two members still applies, so a third of three is two.
        assert_eq!(required_matches(3, 0.34), 2);
    }

    /// Build a graph whose boundaries were drawn under `policy`.
    fn graph_clustered_at(policy: ClusterPolicy) -> IntentGraph {
        let mut g = IntentGraph::empty();
        g.active_policy = policy;
        g.note_query_vector("q0", &topic_vec(0, 0), "m");
        g.observe_live("q0", Capability::Tool, "t", T0, true);
        g
    }

    #[test]
    fn a_graph_at_the_default_policy_omits_it_from_the_wire() {
        // Absent means the default, which is historically exact: before the
        // policy was configurable the constants were the only value a producer
        // could have used. So a default graph must be byte-identical to one
        // written before the field existed.
        let json = serde_json::to_string(&graph_clustered_at(ClusterPolicy::default())).unwrap();
        assert!(
            !json.contains("cluster_policy"),
            "unexpected policy in {json}"
        );
    }

    #[test]
    fn a_tuned_graph_records_the_policy_it_was_clustered_under() {
        let tuned = ClusterPolicy::default()
            .with_similarity(0.82)
            .with_coverage(0.4);
        let g = graph_clustered_at(tuned);
        let json = serde_json::to_string(&g).unwrap();
        assert!(json.contains("cluster_policy"), "missing policy in {json}");

        let back = IntentGraph::from_json(&json).unwrap();
        assert_eq!(back.cluster_policy, tuned);
        assert_eq!(
            back.active_policy, tuned,
            "a reload on its own must not move a boundary — the active policy \
             starts as the one the graph was clustered under"
        );
    }

    #[test]
    fn the_recorded_policy_is_stamped_once_and_then_frozen() {
        // It describes how the existing boundaries were drawn, and boundaries are
        // never redrawn in place. If it followed later configuration it would
        // report that the clusters are something they are not — and the mismatch
        // it exists to expose could never be detected.
        let mut g = graph_clustered_at(ClusterPolicy::default().with_similarity(0.75));

        g.active_policy = ClusterPolicy::default().with_similarity(0.95);
        g.note_query_vector("q1", &topic_vec(3, 0), "m");
        g.observe_live("q1", Capability::Tool, "t", T0, true);

        assert_eq!(
            g.cluster_policy.similarity, 0.75,
            "the stamp must not follow"
        );
    }

    #[test]
    fn rebuild_does_not_restamp_the_policy() {
        // Unlike the model, which a rebuild does overwrite. A rebuild replaces
        // centroids without revisiting cluster boundaries, so the policy those
        // boundaries came from is still the recorded one.
        let mut g = graph_clustered_at(ClusterPolicy::default().with_coverage(0.9));
        g.active_policy = ClusterPolicy::default();

        g.rebuild_centroids(
            vec![("intent_0".into(), vec![topic_vec(0, 0)])],
            "m2".into(),
        );

        assert_eq!(g.cluster_policy.coverage, 0.9);
        assert_eq!(g.model.as_deref(), Some("m2"), "the model IS restamped");
    }

    #[test]
    fn a_graph_whose_policy_is_out_of_range_is_rejected() {
        for bad in [
            r#"{"similarity":0.0,"coverage":0.5}"#,
            r#"{"similarity":0.7,"coverage":1.5}"#,
        ] {
            let json = format!(
                r#"{{"v":1,"built_from_ts":1,"cluster_policy":{bad},"intents":[{{"id":"i0",
                   "label":"q","terms":[],"members":["q"],"support":1,"tools":{{}},"skills":{{}}}}]}}"#
            );
            assert!(
                IntentGraph::from_json(&json).is_err(),
                "a cosine and a fraction both live in (0, 1]: {bad}"
            );
        }
    }

    #[test]
    fn a_policy_change_is_visible_as_drift() {
        let mut g = graph_clustered_at(ClusterPolicy::default().with_similarity(0.75));
        assert!(
            g.cluster_policy_drift().is_none(),
            "nothing has changed yet"
        );

        g.set_cluster_policy(ClusterPolicy::default().with_similarity(0.9));

        let (built, active) = g
            .cluster_policy_drift()
            .expect("the change must be visible");
        assert_eq!(built.similarity, 0.75);
        assert_eq!(active.similarity, 0.9);
        assert_eq!(
            g.intents.len(),
            1,
            "and it must not have redrawn anything — nothing can"
        );
    }

    // ---- dense clustering must not collapse ---------------------------------

    /// A deterministic stand-in for a real sentence embedding, composed of three
    /// disjoint axis groups: a **shared** component every query in the domain
    /// carries, a **topic** component for what the query wants, and a small
    /// **phrasing** component for how it is worded.
    ///
    /// The existing dense fixtures use orthogonal basis vectors — cosine 0.0
    /// between topics — and that is precisely why they never caught this. Real
    /// embeddings of same-domain English do not look like that: measured on the
    /// checked-in incident fixture, bge-small puts distinct-intent query pairs at
    /// a median cosine of 0.64 and same-intent pairs at 0.69. The distributions
    /// overlap almost entirely, and *that* is the geometry the clustering rule
    /// has to survive.
    ///
    /// Tuned to reproduce it: ~0.98 within a topic, **~0.65 across topics** —
    /// below `TAU_COSINE`, so no two members are individually similar enough to
    /// merge, yet the drifting centroid merges them anyway. That gap is the bug.
    fn topic_vec(topic: usize, phrasing: usize) -> Vec<f32> {
        const TOPICS: usize = 4;
        const PHRASINGS: usize = 4;
        let mut v = vec![0.0f32; 1 + TOPICS + PHRASINGS];
        v[0] = 0.806; // shared: every query here is about tasks
        v[1 + topic] = 0.574; // what the query wants
        v[1 + TOPICS + phrasing] = 0.141; // how it happens to be worded
        normalize(v)
    }

    /// Grow a graph by observing `(topic, phrasing)` pairs through the dense tier.
    fn cluster_topics(pairs: &[(usize, usize)]) -> IntentGraph {
        let mut g = IntentGraph::empty();
        for (i, (topic, phrasing)) in pairs.iter().enumerate() {
            let q = format!("t{topic} p{phrasing} q{i}");
            g.note_query_vector(&q, &topic_vec(*topic, *phrasing), "m");
            g.observe_live(&q, Capability::Tool, &format!("tool_{topic}"), T0, true);
        }
        g
    }

    #[test]
    fn a_dense_rejected_cluster_is_not_joined_by_word_overlap_while_learning() {
        // The learning twin of a_dense_rejected_cluster_is_not_rescued_by_word_
        // overlap. Learning needs the guard more than serving does: a bad serve is
        // one bad ranking, but a bad admission is written into the graph, and
        // every later query is then matched against a cluster that has drifted.
        //
        // It also gets louder with coverage in place, because the dense tier now
        // refuses far more often — every refusal is another chance for token
        // overlap to hand the query straight back.
        let mut dense = intent("dense", &["deploy the app to prod"], &[("t", 1.0)]);
        dense.centroid = Some(normalize(vec![1.0, 0.0, 0.0]));
        dense.last_ts = T0;
        let mut g = graph(vec![dense]);

        // Orthogonal to the centroid, so the dense tier refuses it — while it
        // shares every word with the cluster's only member.
        let q = "deploy the app to prod now";
        g.note_query_vector(q, &[0.0, 1.0, 0.0], "m");
        g.observe_live(q, Capability::Tool, "t", T0, true);

        assert_eq!(
            g.intents.len(),
            2,
            "a query the dense tier refused must seed its own cluster, not rejoin \
             on word overlap: {:?}",
            g.intents.iter().map(|i| &i.members).collect::<Vec<_>>()
        );
    }

    #[test]
    fn genuine_paraphrases_still_share_one_cluster() {
        // The guard in the other direction. Splitting is only a fix if real
        // paraphrases still merge — a graph of singletons has learned nothing,
        // and "cluster count went up" on its own proves neither.
        let g = cluster_topics(&[(0, 0), (0, 1), (0, 2), (0, 3)]);

        assert_eq!(
            g.intents.len(),
            1,
            "four phrasings of one intent must stay together, got {:?}",
            g.intents.iter().map(|i| &i.members).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_diffuse_cluster_does_not_absorb_a_distinct_query() {
        // The mechanism in isolation. Two topics in one cluster, then a third
        // query that is BELOW tau against every single member — and yet above
        // tau against their centroid, because averaging two topics cancels what
        // distinguishes them and leaves the shared component pointing at
        // everything in the domain.
        // Built by hand: with the rule in place two topics no longer merge
        // through the live path, so a cluster this diffuse can only arrive from
        // history — one grown under the old rule, or refilled by a rebuild.
        let mut it = intent("i0", &[], &[("t", 1.0)]);
        it.absorb_member("a", Some(&topic_vec(0, 0)));
        it.absorb_member("b", Some(&topic_vec(1, 0)));

        let outsider = topic_vec(2, 0);
        let centroid_cos = cosine(&outsider, it.centroid.as_deref().unwrap());
        assert!(
            centroid_cos >= TAU_COSINE,
            "fixture must reproduce the bug: the centroid should admit the \
             outsider, got {centroid_cos}"
        );

        it.last_ts = T0; // hand-built, so stamp it current or eviction claims it
        let mut g = IntentGraph::empty();
        g.intents.push(it);
        g.note_query_vector("outsider", &outsider, "m");
        g.observe_live("outsider", Capability::Tool, "other", T0, true);

        assert_eq!(
            g.intents.len(),
            2,
            "a query no member recognizes must seed its own cluster, not join on \
             the average"
        );
    }

    #[test]
    fn the_cohesion_bar_is_non_decreasing_in_cluster_diversity() {
        // The property that makes the rule self-limiting rather than
        // self-accelerating: as a cluster takes on distinct topics, `‖mean‖`
        // falls, so the bar derived from it (`TAU_COSINE / cohesion`) rises.
        // Under the pre-accumulator arithmetic this ran the other way.
        let mut it = intent("i0", &["a"], &[("t", 1.0)]);
        let mut previous = f32::INFINITY;
        for topic in 0..4 {
            it.absorb_vector(&topic_vec(topic, 0));
            let bar = TAU_COSINE / norm(it.mean.as_deref().unwrap());
            assert!(
                bar >= previous || previous.is_infinite(),
                "absorbing topic {topic} lowered the bar from {previous} to {bar}"
            );
            previous = bar;
        }
        // Concretely: once spread across four topics, the bar has risen past the
        // similarity a same-domain query actually carries, so a fifth distinct
        // topic can no longer reach it — the cluster has stopped growing on its
        // own rather than accelerating.
        let cross_topic = cosine(&topic_vec(0, 0), &topic_vec(1, 0));
        assert!(
            previous > cross_topic,
            "the bar ({previous}) must outrun the similarity a distinct topic \
             carries ({cross_topic}), or the cluster keeps absorbing"
        );
    }

    #[test]
    fn required_matches_floors_at_two_and_caps_at_the_cluster() {
        // The floor is what stops single-link chaining at the size where it does
        // the most damage: a plain 50% of two members is one, and one member is
        // exactly how a cluster grows into something nobody asked for. The cap is
        // cold start — a one-member cluster must still be able to gain a second.
        for (members, want) in [(1, 1), (2, 2), (3, 2), (4, 2), (5, 3), (10, 5), (16, 8)] {
            assert_eq!(
                required_matches(members, COVERAGE_FRACTION),
                want,
                "{members} members should require {want}"
            );
        }
    }

    #[test]
    fn a_cluster_without_member_vectors_matches_on_the_centroid() {
        // Member vectors never cross the wire, so a reloaded or producer-built
        // cluster has none. That is "no dense evidence", not "reject": it must
        // still match, reproducing the pre-coverage rule exactly, or every graph
        // loaded from disk would silently stop arming.
        let mut it = intent("i0", &["deploy the app"], &[("t", 1.0)]);
        it.centroid = Some(normalize(vec![1.0, 0.0, 0.0]));
        it.last_ts = T0;
        assert!(
            it.member_vectors.iter().all(Option::is_none),
            "the helper rebuilds derived state, so no member carries a vector"
        );
        let g = graph(vec![it]);

        let close = normalize(vec![0.95, 0.31, 0.0]); // ~0.95 to the centroid
        assert!(
            g.arm("deploy the app", Some(&close), Capability::Tool, &all_known)
                .is_some(),
            "a centroid-only cluster must still arm"
        );

        let far = vec![0.0, 1.0, 0.0];
        assert!(
            g.arm("deploy the app", Some(&far), Capability::Tool, &all_known)
                .is_none(),
            "and must still reject below tau"
        );
    }

    #[test]
    fn rebuild_refills_the_member_vectors_it_is_the_repair_path_for() {
        // Member vectors never cross the wire, so a graph off disk has none and
        // falls back to the centroid bar. A rebuild already embeds every member —
        // keeping those vectors rather than only their mean is what makes it the
        // repair path for a graph grown before coverage existed.
        let mut g = graph(vec![intent("i0", &["a", "b"], &[("t", 1.0)])]);
        assert!(
            g.intents[0].member_vectors.iter().all(Option::is_none),
            "a wire-shaped cluster starts with none"
        );

        g.rebuild_centroids(
            vec![("i0".into(), vec![topic_vec(0, 0), topic_vec(0, 1)])],
            "m".into(),
        );

        assert!(
            g.intents[0].member_vectors.iter().all(Option::is_some),
            "the rebuild had them in hand and must keep them"
        );
        // And the refilled tier discriminates again: same topic in, distinct out.
        assert!(
            dense_verdict(&g.intents[0], &topic_vec(0, 2), ClusterPolicy::default())
                .is_some_and(|v| v.admitted && v.covered)
        );
        assert!(
            !dense_verdict(&g.intents[0], &topic_vec(1, 0), ClusterPolicy::default())
                .is_some_and(|v| v.admitted),
            "a distinct topic must not be admitted once coverage can see the members"
        );
    }

    #[test]
    fn rebuild_leaves_member_vectors_alone_when_the_pairing_shifted() {
        // The caller snapshots members, embeds without the graph lock, then
        // re-locks — and a concurrent observe in that window can append a member,
        // shifting every position. The centroid is an order-insensitive mean and
        // does not care; these are matched by position and would silently pair a
        // member's text with its neighbour's vector.
        let mut g = graph(vec![intent("i0", &["a", "b", "c"], &[("t", 1.0)])]);

        g.rebuild_centroids(
            vec![("i0".into(), vec![topic_vec(0, 0), topic_vec(0, 1)])],
            "m".into(),
        );

        assert!(
            g.intents[0].member_vectors.iter().all(Option::is_none),
            "a degraded tier beats a silently wrong one"
        );
        assert!(
            g.intents[0].centroid.is_some(),
            "the centroid is order-insensitive and still rebuilds"
        );
    }

    // ---- the reported incident, against real embeddings ---------------------

    /// Query-side bge-small embeddings of the 12 queries from Experiment E4 of
    /// the 2026-08-11 misranking investigation.
    ///
    /// The crate's only checked-in real-embedding fixture, and it earns the
    /// exception: every other dense fixture pins geometry we invented, so it can
    /// only prove the rule works on the geometry we claim exists. This one is a
    /// literal regression vector for the reported incident. It downloads nothing
    /// at test time (see `crate::fusion` — core tests run on every build without
    /// a model), and the queries are the report's own examples, not customer data.
    const INCIDENT_FIXTURE: &str = include_str!("../tests/fixtures/incident-queries.json");

    fn incident_queries() -> Vec<(String, String, Vec<f32>)> {
        let doc: serde_json::Value = serde_json::from_str(INCIDENT_FIXTURE).unwrap();
        doc["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                (
                    q["query"].as_str().unwrap().to_string(),
                    q["tool"].as_str().unwrap().to_string(),
                    q["vector"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap() as f32)
                        .collect(),
                )
            })
            .collect()
    }

    fn incident_graph() -> IntentGraph {
        let mut g = IntentGraph::empty();
        for (query, tool, vector) in incident_queries() {
            g.note_query_vector(&query, &vector, "bge-small");
            g.observe_live(&query, Capability::Tool, &tool, T0, true);
        }
        g
    }

    #[test]
    fn the_incident_queries_do_not_collapse_into_one_cluster() {
        // Reproduced from the fixture: matching on the centroid puts 11 of these
        // 12 into a single cluster, which is what let the most-invoked write op
        // ride into every task-phrased search.
        let g = incident_graph();
        let sizes: Vec<usize> = g.intents.iter().map(|i| i.members.len()).collect();

        assert!(
            g.intents.len() >= 5,
            "12 queries across four intents collapsed into {} clusters, sizes {sizes:?}",
            g.intents.len()
        );
        assert!(
            sizes.iter().all(|n| *n <= 5),
            "no cluster should hold most of the corpus, sizes {sizes:?}"
        );
    }

    #[test]
    fn a_read_query_and_a_write_query_do_not_arm_the_same_cluster() {
        // The user-visible failure, structurally: "find tasks related to
        // authentication" served create_task because it armed the same cluster a
        // create-phrased query arms — one cluster holding both intents boosts
        // whatever it saw invoked most onto every query it recognizes.
        let g = incident_graph();
        let qs = incident_queries();
        let pick = |text: &str| {
            let (q, _, v) = qs
                .iter()
                .find(|(q, _, _)| q == text)
                .expect("query is in the fixture");
            g.arm(q, Some(v), Capability::Tool, &all_known)
                .expect("the fixture's own query must match the cluster it grew")
                .intent_id
        };

        assert_ne!(
            pick("find tasks related to authentication"),
            pick("create a task for the login bug"),
            "a read intent and a write intent armed the same cluster"
        );
    }

    /// Regenerate [`INCIDENT_FIXTURE`] against the real model. Ignored: it is a
    /// tool, not a test, and it needs the model on disk.
    ///
    /// `cargo test -p ratel-ai-core --lib regenerate_incident_fixture -- --ignored`
    #[test]
    #[ignore]
    fn regenerate_incident_fixture() {
        use crate::embedding::embedder_with_telemetry;
        use crate::embedding_config::EmbeddingModel;

        let embedder =
            embedder_with_telemetry(&EmbeddingModel::Default, &crate::trace::NoopSink).unwrap();
        let doc: serde_json::Value = serde_json::from_str(INCIDENT_FIXTURE).unwrap();
        let rows: Vec<String> = doc["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let (intent, tool, text) = (
                    q["intent"].as_str().unwrap(),
                    q["tool"].as_str().unwrap(),
                    q["query"].as_str().unwrap(),
                );
                let nums: Vec<String> = embedder
                    .embed_query(text)
                    .unwrap()
                    .iter()
                    .map(|x| format!("{x:.6}"))
                    .collect();
                format!(
                    "    {{ \"intent\": \"{intent}\", \"tool\": \"{tool}\", \"query\": \"{text}\", \"vector\": [{}] }}",
                    nums.join(",")
                )
            })
            .collect();
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/incident-queries.json"),
            format!(
                "{{\n  \"note\": {},\n  \"model\": {},\n  \"revision\": {},\n  \"queries\": [\n{}\n  ]\n}}\n",
                doc["note"], doc["model"], doc["revision"], rows.join(",\n")
            ),
        )
        .unwrap();
    }
}
