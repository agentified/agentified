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

/// Minimum cosine between a query and a cluster centroid to count as a match.
pub(crate) const TAU_COSINE: f32 = 0.70;

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
    /// inflate the weight. Live-learning scratch: not serialized (a reloaded
    /// centroid is treated as a single prior sample), so never part of identity.
    #[serde(skip)]
    vector_n: u32,
}

/// Identity is the evidence — members, centroid, support, edges. The derived
/// display fields are ignored, so a graph compares equal to its own round-trip
/// whether or not labels have been materialized.
impl PartialEq for Intent {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.members == other.members
            && self.centroid == other.centroid
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
    /// registry does not currently define. Ordered `(weight desc, id asc)` —
    /// the same total order the rankers use, so the arm is deterministic.
    fn ranked(&self, kind: Capability, known: &dyn Fn(&str) -> bool) -> Vec<String> {
        let mut ranked: Vec<(String, f32)> = self
            .edges(kind)
            .iter()
            .filter(|(id, _)| known(id.as_str()))
            .map(|(id, w)| (id.clone(), *w))
            .collect();
        let len = ranked.len();
        sort_and_truncate(&mut ranked, len);
        ranked.into_iter().map(|(id, _)| id).collect()
    }

    /// Fold `vector` into this cluster's centroid as a running mean over its
    /// members, renormalized so cosine stays a plain dot product.
    ///
    /// The mean of unit vectors falls inside the sphere, so skipping the
    /// renormalize would depress every later similarity by the cluster's own
    /// spread. A first vector — or one of a different width, meaning the
    /// embedding model changed — replaces the centroid rather than being
    /// averaged into a space it does not share.
    fn absorb_vector(&mut self, vector: &[f32]) {
        let merged: Vec<f32> = match self.centroid.as_deref() {
            // Weight the running mean by vectors ALREADY folded (`vector_n`), not
            // by `members.len()`: a cluster can gain members lexically with no
            // vector, and counting those would pin the centroid to the first
            // vector after such growth (it would arrive with weight ~n).
            Some(c) if c.len() == vector.len() => {
                let k = self.vector_n.max(1) as f32;
                c.iter().zip(vector).map(|(c, v)| c * k + v).collect()
            }
            // A first vector — or one of a different width (the embedding model
            // changed) — starts the mean fresh rather than blending across spaces.
            _ => {
                self.vector_n = 0;
                vector.to_vec()
            }
        };
        self.vector_n = self.vector_n.saturating_add(1);
        self.centroid = Some(normalize(merged));
    }

    /// Drop the oldest members past [`MEMBER_CAP`], keeping the token caches in
    /// step. Bounds per-cluster memory and lexical-match cost; the centroid is a
    /// cumulative mean, so trimming members does not disturb it.
    fn cap_members(&mut self) {
        while self.members.len() > MEMBER_CAP {
            self.members.remove(0);
            self.member_bags.remove(0);
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

    /// Rebuild the cache from `members` — after deserialization, where the
    /// cache is skipped on the wire.
    fn rebuild_bag(&mut self) {
        self.member_bags = self
            .members
            .iter()
            .map(|m| tokenize(m).into_iter().collect())
            .collect();
        self.bag = self.members.iter().flat_map(|m| tokenize(m)).collect();
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
        let len = 4 + usize::from(self.model.is_some());
        let mut out = serializer.serialize_struct("IntentGraph", len)?;
        out.serialize_field("v", &self.v)?;
        out.serialize_field("built_from_ts", &self.built_from_ts)?;
        out.serialize_field("rev", &self.rev)?;
        if let Some(model) = &self.model {
            out.serialize_field("model", model)?;
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
        graph.rebuild_caches();
        Ok(graph)
    }

    /// Reject a structurally-parseable graph that breaks a semantic rule the wire
    /// contract requires (`protocol/v1/conformance/vectors.json`, the `invalid`
    /// set). Serde already catches shape errors (a missing `members`, a negative
    /// `rev`); these are the value-level rules it cannot express.
    fn validate(&self) -> Result<(), IntentGraphError> {
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

    /// Rebuild every cluster's derived token cache. The cache is skipped on the
    /// wire, so a deserialized graph must restore it before it can match
    /// lexically.
    fn rebuild_caches(&mut self) {
        for it in &mut self.intents {
            it.rebuild_bag();
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
    /// `first_confirmation` distinguishes *this search was acted on* from
    /// *another capability was used for the same search*. Both add an edge; only
    /// the former is an observation, so only the former raises `support`. The
    /// caller owns that distinction because it is the one holding the pending
    /// search — see [`crate::UsageLearner`].
    pub(crate) fn observe(
        &mut self,
        query: &str,
        kind: Capability,
        capability_id: &str,
        ts_ms: u64,
        first_confirmation: bool,
    ) {
        // A query vector is available only when the search path was
        // semantic/hybrid AND the slot still belongs to this query.
        let stashed = self.pending.vector_for(query);
        if stashed.is_none() && tokenize(query).is_empty() {
            return; // no words to cluster on and no embedding either
        }
        self.built_from_ts = self.built_from_ts.max(ts_ms);

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
                    last_ts: 0,
                    tools: BTreeMap::new(),
                    skills: BTreeMap::new(),
                    bag: std::collections::HashSet::new(),
                    member_bags: Vec::new(),
                    vector_n: 0,
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
                it.members.push(query.to_string());
                it.absorb_tokens(query);
                if let Some(v) = vector.as_deref() {
                    it.absorb_vector(v);
                }
                it.cap_members();
            }
            // `|| support == 0` is load-bearing: a cluster moves as it learns, so
            // a later invoke from the same search can match a cluster the first
            // one did not — and a freshly seeded cluster must still start at 1.
            // `protocol/v1` requires support >= 1, and a zero-support cluster
            // would contribute a weightless arm.
            if first_confirmation || it.support == 0 {
                it.support = it.support.saturating_add(1);
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
            self.intents[i].centroid = Some(normalize(sum));
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
        if let Some(v) = vector
            && let Some(i) = self.best_dense_match(v)
        {
            return Some(i);
        }
        self.best_lexical_match(query)
    }

    /// Index of the nearest cluster centroid clearing [`TAU_COSINE`]. Ties break
    /// by cluster id so growth does not depend on `Vec` order.
    fn best_dense_match(&self, vector: &[f32]) -> Option<usize> {
        self.intents
            .iter()
            .enumerate()
            .filter_map(|(i, it)| {
                let c = it.centroid.as_deref()?;
                if c.len() != vector.len() {
                    return None; // a different embedding model — not comparable
                }
                Some((i, cosine(vector, c)))
            })
            .filter(|(_, sim)| *sim >= TAU_COSINE)
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| self.intents[b.0].id.cmp(&self.intents[a.0].id))
            })
            .map(|(i, _)| i)
    }

    /// Index of the cluster whose member-token bag best covers `query`, if any
    /// clears [`TAU_LEXICAL`]. Ties break by cluster id so growth does not
    /// depend on `Vec` order.
    fn best_lexical_match(&self, query: &str) -> Option<usize> {
        let q: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
        if q.is_empty() {
            return None;
        }
        self.intents
            .iter()
            .enumerate()
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
    ) -> Option<UsageArm> {
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
    ) -> Option<UsageArm> {
        let best = self
            .intents
            .iter()
            .filter_map(|it| {
                let c = it.centroid.as_deref()?;
                if c.len() != query_vec.len() {
                    return None; // different embedding model — not comparable
                }
                Some((it, cosine(query_vec, c)))
            })
            .filter(|(_, sim)| *sim >= TAU_COSINE)
            .max_by(pick_best)?;
        arm_from(best.0, best.1, self.built_from_ts, kind, known)
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
    ) -> Option<UsageArm> {
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
    ) -> Option<UsageArm> {
        let q: std::collections::HashSet<String> = tokenize(query).into_iter().collect();
        if q.is_empty() {
            return None;
        }
        let best = self
            .intents
            .iter()
            .filter(|it| !centroidless_only || it.centroid.is_none())
            .map(|it| (it, it.lexical_score(&q)))
            .filter(|(_, score)| *score >= TAU_LEXICAL)
            .max_by(pick_best)?;
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
) -> Option<UsageArm> {
    let ids = intent.ranked(kind, known);
    if ids.is_empty() {
        return None; // matched, but nothing it remembers still exists
    }
    let weight = usage_weight(intent.support) * recency_factor(now_ts, intent.last_ts);
    Some(UsageArm {
        intent_id: intent.id.clone(),
        similarity,
        support: intent.support,
        weight,
        ids,
    })
}

/// Scale to unit length. A zero vector is returned unchanged — there is no
/// direction to preserve, and dividing would produce NaNs that would poison
/// every later comparison.
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
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
            last_ts: 0,
            tools: tools.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            skills: BTreeMap::new(),
            bag: std::collections::HashSet::new(),
            member_bags: Vec::new(),
            vector_n: 0,
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

    // ---- rev: the persistence write-counter --------------------------------

    #[test]
    fn observe_bumps_rev_once_per_mutation() {
        let mut g = IntentGraph::empty();
        assert_eq!(g.rev(), 0, "an empty graph has written nothing");
        g.observe("build broken", Capability::Tool, "a", T0, true);
        assert_eq!(g.rev(), 1);
        // A second observe on the same search adds an edge — a real change even
        // though it seeds no new member — so it must still count as one write.
        g.observe("build broken", Capability::Tool, "b", T0, false);
        assert_eq!(g.rev(), 2);
    }

    #[test]
    fn a_no_op_observe_does_not_bump_rev() {
        // No words to cluster on and no stashed vector: `observe` returns before
        // changing anything, so the write-counter must not move. Guards against a
        // "bump unconditionally" regression.
        let mut g = IntentGraph::empty();
        g.observe("   ", Capability::Tool, "a", T0, true);
        assert_eq!(g.len(), 0);
        assert_eq!(g.rev(), 0);
    }

    #[test]
    fn rev_survives_a_round_trip() {
        let mut g = IntentGraph::empty();
        g.observe("build broken", Capability::Tool, "a", T0, true);
        g.observe("rotate the signing key", Capability::Tool, "b", T0, true);
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
        g.observe("something new", Capability::Tool, "t", T0, true);
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
        assert_eq!(
            g.arm_lexical("anything", Capability::Tool, &all_known),
            None
        );
        assert_eq!(g.arm_dense(&[1.0], Capability::Tool, &all_known), None);
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
        assert_eq!(g.arm_dense(&[0.0, 1.0], Capability::Tool, &all_known), None);
    }

    #[test]
    fn dense_match_skips_centroids_of_a_different_dimension() {
        // A changed embedding model must never be compared across vector spaces.
        let mut it = intent("i0", &["q"], &[("t", 1.0)]);
        it.centroid = Some(vec![1.0, 0.0, 0.0]);
        let g = graph(vec![it]);
        assert_eq!(g.arm_dense(&[1.0, 0.0], Capability::Tool, &all_known), None);
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
        assert_eq!(
            g.arm_lexical("did CI pass", Capability::Tool, &all_known),
            None
        );
    }

    #[test]
    fn lexical_match_ignores_stopwords_only_queries() {
        let g = graph(vec![intent("i0", &["build"], &[("t", 1.0)])]);
        assert_eq!(g.arm_lexical("is the", Capability::Tool, &all_known), None);
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
        assert_eq!(
            g.arm_lexical("build broken", Capability::Tool, &|_| false),
            None
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
        g.observe(
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
        g.observe(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe(
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
        g.observe(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe(
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
            g.observe(
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
        g.observe(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe(
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
        g.observe(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        assert_eq!(
            g.arm("is the build ok", None, Capability::Tool, &all_known),
            None
        );
    }

    #[test]
    fn a_lexically_grown_graph_is_matchable_even_when_a_query_vector_is_offered() {
        // A semantic catalog hands `arm` a query vector, but a locally-learned
        // graph has no centroids to compare it against. It must fall back to
        // lexical matching rather than silently returning nothing.
        let mut g = IntentGraph::empty();
        g.observe(
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
            g.observe(
                "why is the build broken",
                Capability::Tool,
                "chosen_often",
                T0,
                true,
            );
        }
        g.observe(
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
        g.observe("build broken", Capability::Tool, "a", T0 + 10, true);
        g.observe("build broken", Capability::Tool, "b", T0, true);
        assert_eq!(g.built_from_ts, T0 + 10);
    }

    #[test]
    fn the_token_cache_stays_in_step_with_members() {
        // The cache is derived from `members`; if the two drift, a query stops
        // matching a cluster that plainly covers it. Silent, and invisible to
        // every other test — so pin it directly.
        let mut g = IntentGraph::empty();
        g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe("the pipeline is broken", Capability::Tool, "t", T0, true);

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
        g.observe(
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
            g.observe("build broken", Capability::Tool, "a", T0, true);
            g.note_query_vector("build broken again", &v2, "m");
            g.observe("build broken again", Capability::Tool, "b", T0, true);
            for i in 0..extra {
                g.observe(
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
        g.observe("why is the build broken", Capability::Tool, "a", T0, false);
        assert_eq!(g.intents[0].support, 1);
    }

    // ---- lexical clustering must not over-merge -----------------------------

    #[test]
    fn one_shared_word_does_not_merge_distinct_intents() {
        // The bug, minimally. Two unrelated asks sharing a single word were
        // exactly 50% "covered" by each other and merged.
        let mut g = IntentGraph::empty();
        g.observe("deploy0 rollback3", Capability::Tool, "a", T0, true);
        g.observe("deploy0 migrate5", Capability::Tool, "b", T0, true);
        assert_eq!(g.len(), 2, "one shared word is not the same question");
    }

    #[test]
    fn a_large_cluster_does_not_absorb_an_unrelated_query() {
        // The runaway: the old score was measured against the UNION of every
        // member, which only grows — so a mature cluster recognized most of the
        // vocabulary and swallowed anything, which grew it further.
        let mut g = IntentGraph::empty();
        for i in 0..30 {
            g.observe(
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
        g.observe("variant7 variant12", Capability::Tool, "vault", T0, true);
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
                g.observe(&q, Capability::Tool, &format!("t{topic}"), T0, true);
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
            g.observe(q, Capability::Tool, "gh_run_list", T0, true);
        }
        for q in ["rotate the signing key", "rotate the signing key now"] {
            g.observe(q, Capability::Tool, "vault_rotate", T0, true);
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
        g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe("is the build broken now", Capability::Tool, "t", T0, true);

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
        g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe("the build is broken again", Capability::Tool, "t", T0, true);
        g.observe("rotate the signing key", Capability::Tool, "v", T0, true);

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
        g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        g.observe("the build broken again", Capability::Tool, "t", T0, true);
        let alone = g.labeled()[0].terms.clone();

        // A second cluster that also uses "again" makes that term less
        // distinguishing for the first — which must be reflected even though the
        // first cluster was never touched again.
        g.observe(
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
        g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        assert!(
            g.intents[0].label.is_empty(),
            "not stored on the write path"
        );
        assert!(!g.labeled()[0].label.is_empty(), "materialized on read");
    }

    #[test]
    fn a_stopword_only_query_teaches_nothing() {
        let mut g = IntentGraph::empty();
        g.observe("is the", Capability::Tool, "t", T0, true);
        assert!(g.is_empty());
    }

    #[test]
    fn tool_and_skill_observations_land_on_separate_edge_maps() {
        let mut g = IntentGraph::empty();
        g.observe(
            "why is the build broken",
            Capability::Tool,
            "gh_run_list",
            T0,
            true,
        );
        g.observe(
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
        g.observe(
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
    fn observe_stamps_the_model_on_the_first_centroid() {
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "model-a");
        g.observe("build broken", Capability::Tool, "t", T0, true);
        assert_eq!(g.model.as_deref(), Some("model-a"));
    }

    #[test]
    fn observe_freezes_the_centroid_on_a_model_change() {
        // Grow under model-a, then an observation arrives embedded by model-b.
        // Member/support must still update, but the centroid must NOT blend the
        // two vector spaces.
        let mut g = IntentGraph::empty();
        g.note_query_vector("build broken", &[1.0, 0.0, 0.0], "model-a");
        g.observe("build broken", Capability::Tool, "t", T0, true);
        let frozen = g.intents[0].centroid.clone();

        g.note_query_vector("build broken again", &[0.0, 1.0, 0.0], "model-b");
        g.observe("build broken again", Capability::Tool, "t", T0, true);

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
        g.observe("build broken", Capability::Tool, "gh_run_list", T0, true);
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
        g.observe("alpha query", Capability::Tool, "a", T0, true);
        g.observe("bravo query", Capability::Tool, "b", T0, true);
        g.observe("charlie query", Capability::Tool, "c", T0, true);
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
            g.observe("why is the build broken", Capability::Tool, "t", T0, true);
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
            g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        }
        // Advance the graph's clock 200 days via a different topic.
        g.observe(
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
            g.observe("why is the build broken", Capability::Tool, "t", T0, true);
        }
        assert_eq!(g.len(), 1);
        // ~2 years of other activity later: the build cluster is past the floor.
        g.observe(
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
            g.observe(
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
}
