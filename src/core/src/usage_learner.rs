//! The online learner: turns the trace stream into an [`IntentGraph`]
//! (ADR-0014).
//!
//! # Why this is a sink
//!
//! The two halves of a relevance judgment arrive through *different* API calls —
//! a search, then an invoke — and the registries have no session concept to join
//! them with. Trace sinks do: each is constructed per session. ADR-0007 already
//! frames the sink as the subscription seam ("rerankers, suggestion analysis,
//! and inspection subscribe to different cuts of the same producer"), so the
//! learner needs no new plumbing — it decorates whatever sink is already
//! installed and forwards every event untouched.
//!
//! **One learner per session.** Two sessions sharing one learner would cross-pair
//! their searches and invokes and record edges nobody produced.
//!
//! # What counts as evidence
//!
//! ```text
//! Search{query}      → remembered as this session's pending query
//! InvokeStart{tool}  → paired with it → one confirmed observation
//! ```
//!
//! Only **invocations** become edges. What retrieval *returned* is the ranker's
//! own guess; recording it would teach the graph what it already believes and
//! reinforce its mistakes. A search nobody acts on teaches nothing and is
//! dropped.
//!
//! A pending query survives until the next search replaces it, so an agent that
//! searches once and invokes three tools records three observations — that is
//! genuinely what happened.
//!
//! # How far a cluster reaches
//!
//! A [`TraceEvent::Search`] carries the query *text*, not its embedding, so the
//! sink alone could only cluster on words. A semantic/hybrid registry closes
//! that gap: it has already embedded the query for its own ranking and stashes
//! that vector on the graph, so the learner grows a real centroid and clusters
//! phrasings that share **no vocabulary** — "delete a path" with "remove
//! something". A `Bm25` registry loads no model (ADR-0011), so its clusters
//! carry no centroid and reach repeats and near-repeats only.
//! [`IntentGraph::arm`] picks the tier from what the graph carries, so either
//! kind works on every [`crate::SearchMethod`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::trace::{Origin, TraceEnvelope, TraceEvent, TraceEventContext, TraceSink};
use crate::usage::{Capability, IntentGraph, Observation};

/// The learner's most recent search — the query an invoke attributes to. Kept
/// per-learner (not read from the shared graph) so a concurrent search from
/// another session cannot misattribute this learner's invoke. Whether the
/// question has already been credited a support bump lives on the shared graph
/// ([`IntentGraph::claim_credit`]), so per-catalog tool and skill learners count
/// one fanned-out question once between them, not once each.
struct Pending {
    query: String,
}

/// Which searches may open an observation window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OriginFilter {
    /// Pair an invoke with the most recent search of **any** origin.
    #[default]
    Any,
    /// Pair only with searches carrying exactly this origin — the setting a
    /// baseline capture uses, with [`crate::Origin::Baseline`].
    ///
    /// A search of any other origin is **ignored entirely**: it does not become
    /// the pending query and it does not arm a support credit. Ignoring rather
    /// than clearing is deliberate — one of Ratel's own internal searches
    /// landing between a baseline query and its invokes must not discard the
    /// turn's evidence.
    Exactly(Origin),
}

/// Whether observations are recorded as seeded evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Provenance {
    /// Live serving traffic — raises `support` only.
    #[default]
    Live,
    /// A seeding pass (a baseline capture, or a replay of one) — additionally
    /// raises [`crate::Intent::seeded_support`].
    Seeded,
}

/// How a [`UsageLearner`] turns a trace stream into observations.
///
/// [`Default`] reproduces today's behavior exactly, field for field — pinned by
/// `the_default_policy_reproduces_todays_pairing_exactly`. Build from it with
/// the `with_*` setters:
///
/// ```
/// use ratel_ai_core::{ObservationPolicy, Origin, OriginFilter, Provenance};
///
/// // A baseline capture: only observed queries teach, and everything learned
/// // is marked as seeded.
/// let seeding = ObservationPolicy::default()
///     .with_origins(OriginFilter::Exactly(Origin::Baseline))
///     .with_provenance(Provenance::Seeded);
///
/// assert_eq!(seeding.provenance, Provenance::Seeded);
/// assert_eq!(ObservationPolicy::default().origins, OriginFilter::Any);
/// ```
///
/// The struct is `#[non_exhaustive]`, which is what makes a future field
/// additive rather than breaking — and is also why the setters exist rather
/// than struct-literal syntax: a `#[non_exhaustive]` struct cannot be built
/// with a struct expression outside its defining crate at all, `..Default`
/// included. Fields stay public for reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ObservationPolicy {
    /// Which searches open an observation window.
    pub origins: OriginFilter,
    /// Whether what is learned is marked as seeded.
    pub provenance: Provenance,
}

impl ObservationPolicy {
    /// Set which searches open an observation window.
    pub fn with_origins(mut self, origins: OriginFilter) -> Self {
        self.origins = origins;
        self
    }

    /// Set whether what is learned is marked as seeded.
    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }
}

/// What one trace event means for learning, under a policy.
///
/// **The pairing rule, in one place.** The live path and the replay path differ
/// in where they keep pending state — a per-session learner holds a `Mutex`
/// slot, a replay holds a map keyed by `session_id` — but they must agree
/// exactly on *which event does what*, or a graph built from a log stops
/// matching the one live learning would have grown from the same events. Having
/// written that match twice, a later change (a new confirming event, a pairing
/// strategy) would have to land in both with nothing forcing the second.
#[derive(Debug, PartialEq)]
pub(crate) enum Step<'a> {
    /// This search opens an observation window for `query`.
    Remember(&'a str),
    /// This invocation closes one, confirming `capability_id` of `kind`.
    Confirm(Capability, &'a str),
    /// Not evidence — including a search the policy rejects, which is **ignored
    /// rather than treated as a boundary**, so one of Ratel's own internal
    /// searches landing mid-turn cannot discard the turn's evidence.
    Ignore,
}

pub(crate) fn classify(event: &TraceEvent, policy: ObservationPolicy) -> Step<'_> {
    match event {
        // Both search kinds open a window: a capability search hits the tool and
        // skill registries in turn with the same text.
        TraceEvent::Search { query, origin, .. }
        | TraceEvent::SkillSearch { query, origin, .. }
            if accepts(policy, *origin) =>
        {
            Step::Remember(query)
        }
        // The invocation the agent CHOSE to make. A trace records which tool was
        // called, never whether calling it was right, so completion is not a
        // second signal: filtering on `invoke_end` would drop good choices that
        // failed on their arguments while keeping wrong ones that ran fine.
        TraceEvent::InvokeStart { tool_id, .. } => Step::Confirm(Capability::Tool, tool_id),
        TraceEvent::SkillInvoke { skill_id, .. } => Step::Confirm(Capability::Skill, skill_id),
        _ => Step::Ignore,
    }
}

/// Replay a whole trace log into `graph`, pairing searches with invokes
/// **per session** while walking the log in its own order.
///
/// Both halves of that are load-bearing:
///
/// - **Per-session pending state.** Sessions interleave in one log and share one
///   graph; feeding them through a single pending slot would cross-pair one
///   session's search with another's invoke and record edges nobody produced
///   (the rule the module doc states for [`UsageLearner`] itself).
/// - **Log order, never re-sorted.** `JsonlSink` appends, so file order *is*
///   arrival order — what the live path saw. Sorting by `ts` would produce a
///   graph the live path could not have grown, since cluster membership depends
///   on which clusters existed when each query arrived. `ts` is used to *stamp*
///   observations (so recency reflects when the work happened), never to order
///   them.
///
/// `embeddings` maps query text to its vector. When present, an entry is stashed
/// on the graph immediately before the observation that consumes it, so
/// clustering happens at the **dense** tier — the same tier the live path uses,
/// rather than the lexical fallback a model-free replay would fall back to. An
/// absent entry simply clusters lexically.
pub(crate) fn replay_log_into(
    graph: &mut IntentGraph,
    envelopes: &[TraceEnvelope],
    policy: ObservationPolicy,
    embeddings: &HashMap<String, Vec<f32>>,
    fingerprint: Option<&str>,
) {
    // session id -> (the query its next invoke attributes to, already credited).
    //
    // The credit is tracked HERE rather than through [`IntentGraph::arm_credit`]
    // / [`claim_credit`]. That slot is global and keyed by query text, which is
    // enough live — two learners sharing one graph need somewhere common to
    // agree, and identical text from two concurrent sessions is rare. In a
    // replay it is not rare: sessions interleave by construction and popular
    // questions repeat verbatim, so a shared slot loses the second session's
    // observation every time. Replay knows the session, so it can be exact.
    let mut pending: HashMap<&str, (&str, bool)> = HashMap::new();

    for env in envelopes {
        let session = env.session_id.as_str();
        let (kind, capability_id) = match classify(&env.event, policy) {
            Step::Remember(query) => {
                // Re-arming with the same text is idempotent: a capability
                // search fans one question to both catalogs, and both of those
                // land before any invoke, so the turn still credits once.
                pending.insert(session, (query, false));
                continue;
            }
            Step::Confirm(kind, id) => (kind, id),
            Step::Ignore => continue,
        };

        let Some(entry) = pending.get_mut(session) else {
            continue; // an invoke with no accepted search before it proves nothing
        };
        let query = entry.0;
        // The first confirming invoke of THIS session's question is what makes
        // it an observation; later ones add edges for the same question.
        let first_confirmation = !entry.1;
        entry.1 = true;
        // Stash this query's vector right before the observation reads it. The
        // slot holds one entry, so with sessions interleaved anything set
        // earlier may belong to another session's question.
        if let (Some(vector), Some(fp)) = (embeddings.get(query), fingerprint) {
            graph.note_query_vector(query, vector, fp);
        }
        graph.observe(Observation {
            query,
            kind,
            capability_id,
            ts_ms: env.ts,
            first_confirmation,
            seeded: policy.provenance == Provenance::Seeded,
        });
    }
}

/// Every distinct query a `policy`-accepted search carries, in first-appearance
/// order — the texts a caller must embed for [`replay_log_into`] to cluster
/// densely.
pub(crate) fn queries_to_embed(
    envelopes: &[TraceEnvelope],
    policy: ObservationPolicy,
) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for env in envelopes {
        if let TraceEvent::Search { query, origin, .. }
        | TraceEvent::SkillSearch { query, origin, .. } = &env.event
            && accepts(policy, *origin)
            && seen.insert(query.as_str())
        {
            out.push(query.clone());
        }
    }
    out
}

/// Whether a search of `origin` may open an observation window under `policy`.
fn accepts(policy: ObservationPolicy, origin: Origin) -> bool {
    match policy.origins {
        OriginFilter::Any => true,
        OriginFilter::Exactly(wanted) => origin == wanted,
    }
}

/// A [`TraceSink`] decorator that grows an [`IntentGraph`] from the events
/// passing through it, then forwards them unchanged.
///
/// Install it in place of the sink you would otherwise use, and hand the same
/// graph handle to the registries so searches read what invocations write:
///
/// ```
/// use std::sync::{Arc, RwLock};
/// use ratel_ai_core::{IntentGraph, NoopSink, Tool, ToolRegistry, UsageLearner};
///
/// let graph = Arc::new(RwLock::new(IntentGraph::empty()));
/// let learner = Arc::new(UsageLearner::new(graph.clone(), Arc::new(NoopSink)));
///
/// let mut registry = ToolRegistry::new();
/// registry.set_trace_sink(learner);                  // writes the graph
/// registry.set_intent_graph(Some(graph.clone()));    // reads it
/// registry.register(Tool {
///     id: "gh_run_list".into(),
///     name: "gh_run_list".into(),
///     description: "List CI runs".into(),
///     experimental_searchable_description: None,
///     input_schema: serde_json::json!({}),
///     output_schema: serde_json::json!({}),
/// });
///
/// registry.search("why is the build broken", 5);
/// registry.record_event(ratel_ai_core::TraceEvent::InvokeStart {
///     tool_id: "gh_run_list".into(),
///     args_size_bytes: 0,
/// });
/// assert_eq!(graph.read().unwrap().len(), 1); // learned
/// ```
pub struct UsageLearner {
    inner: Arc<dyn TraceSink>,
    graph: Arc<RwLock<IntentGraph>>,
    /// The session's most recent search, awaiting an invoke to confirm it.
    pending: Mutex<Option<Pending>>,
    policy: ObservationPolicy,
}

impl UsageLearner {
    /// Wrap `inner`, learning into `graph` under [`ObservationPolicy::default`].
    /// Pass [`crate::NoopSink`] for `inner` when the only thing you want is the
    /// learning.
    pub fn new(graph: Arc<RwLock<IntentGraph>>, inner: Arc<dyn TraceSink>) -> Self {
        Self::with_policy(graph, inner, ObservationPolicy::default())
    }

    /// Wrap `inner`, learning into `graph` under `policy` — the entry point a
    /// baseline capture or a replay uses. [`Self::new`] is this at the default
    /// policy.
    pub fn with_policy(
        graph: Arc<RwLock<IntentGraph>>,
        inner: Arc<dyn TraceSink>,
        policy: ObservationPolicy,
    ) -> Self {
        Self {
            inner,
            graph,
            pending: Mutex::new(None),
            policy,
        }
    }

    /// The policy in force.
    pub fn policy(&self) -> ObservationPolicy {
        self.policy
    }

    /// The graph this learner writes — hand it to a registry to read.
    pub fn graph(&self) -> Arc<RwLock<IntentGraph>> {
        self.graph.clone()
    }

    /// Record the search awaiting confirmation, and arm the shared credit.
    ///
    /// Arming on the graph re-arms unconditionally: `search_capabilities` emits
    /// a `Search` and a `SkillSearch` for one question, but both arrive *before*
    /// any invoke, so one credit follows either way. Because the credit lives on
    /// the graph the two catalogs share, this holds even though each catalog has
    /// its own learner — the previous per-learner flag credited once *each*.
    /// Over-counting still needs a credit, then another search of the same text,
    /// then another credit — two real searches, which should count twice.
    fn remember_query(&self, query: &str) {
        if let Ok(mut pending) = self.pending.lock() {
            *pending = Some(Pending {
                query: query.to_string(),
            });
        }
        if let Ok(graph) = self.graph.read() {
            graph.arm_credit(query);
        }
    }

    /// Pair `capability_id` with the pending query, if there is one.
    ///
    /// Best-effort throughout: trace events are observations, so a poisoned lock
    /// or a missing pending query drops the evidence rather than disturbing the
    /// agent loop (ADR-0007's query-log semantics).
    fn confirm(&self, kind: Capability, capability_id: &str, ts_ms: u64) {
        let Ok(pending) = self.pending.lock() else {
            return;
        };
        let Some(query) = pending.as_ref().map(|p| p.query.clone()) else {
            return; // an invoke with no search before it proves nothing
        };
        drop(pending);
        if let Ok(mut graph) = self.graph.write() {
            // The first invoke of this question, across every learner sharing the
            // graph, is what makes it an observation; the rest add edges for the
            // same question without re-bumping support.
            let first_confirmation = graph.claim_credit(&query);
            graph.observe(Observation {
                query: &query,
                kind,
                capability_id,
                ts_ms,
                first_confirmation,
                seeded: self.policy.provenance == Provenance::Seeded,
            });
        }
    }

    /// Learn from a **historical** envelope instead of a live event.
    ///
    /// Identical to [`TraceSink::record`] except that the observation is stamped
    /// with the envelope's own `ts` rather than the wall clock, so decay
    /// reflects when the work actually happened. Replaying a trace log through
    /// this therefore reproduces the graph the live path would have grown —
    /// which is what makes a JSONL replay a faithful reconstruction rather than
    /// an approximation.
    ///
    /// Does **not** forward to the inner sink: replaying an old log must not
    /// re-emit its events into a live stream.
    ///
    /// One learner covers one session. Feed envelopes from different
    /// `session_id`s through separate learners, or their searches and invokes
    /// cross-pair into edges nobody produced.
    pub fn replay(&self, envelope: &TraceEnvelope) {
        self.learn_from(&envelope.event, envelope.ts);
    }

    /// The shared pairing step behind [`Self::replay`] and [`TraceSink::record`].
    ///
    /// A search the policy rejects falls through to `_ => {}` — **ignored, not
    /// cleared**. Clearing would let one of Ratel's own internal searches,
    /// landing between a captured query and its invokes, silently discard the
    /// turn's evidence.
    fn learn_from(&self, event: &TraceEvent, ts_ms: u64) {
        match classify(event, self.policy) {
            Step::Remember(query) => self.remember_query(query),
            Step::Confirm(kind, capability_id) => self.confirm(kind, capability_id, ts_ms),
            Step::Ignore => {}
        }
    }
}

impl TraceSink for UsageLearner {
    fn record(&self, event: TraceEvent) {
        self.learn_from(&event, now_ms());
        self.inner.record(event);
    }

    fn record_with_context(&self, event: TraceEvent, context: TraceEventContext) {
        self.learn_from(&event, now_ms());
        self.inner.record_with_context(event, context);
    }

    fn record_envelope(&self, envelope: TraceEnvelope) {
        self.learn_from(&envelope.event, envelope.ts);
        self.inner.record_envelope(envelope);
    }

    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{MemorySink, NoopSink, Origin};

    fn learner() -> (Arc<UsageLearner>, Arc<RwLock<IntentGraph>>) {
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        let l = Arc::new(UsageLearner::new(graph.clone(), Arc::new(NoopSink)));
        (l, graph)
    }

    fn search(query: &str) -> TraceEvent {
        TraceEvent::Search {
            query: query.into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: Vec::new(),
            stages: Vec::new(),
            took_ms: 0,
        }
    }

    fn invoke(tool_id: &str) -> TraceEvent {
        TraceEvent::InvokeStart {
            tool_id: tool_id.into(),
            args_size_bytes: 0,
        }
    }

    #[test]
    fn a_search_then_invoke_becomes_one_observation() {
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g.intents[0].support, 1);
        assert_eq!(g.intents[0].tools.get("gh_run_list"), Some(&1.0));
    }

    #[test]
    fn a_search_nobody_acts_on_teaches_nothing() {
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        assert!(graph.read().unwrap().is_empty());
    }

    #[test]
    fn an_invoke_with_no_preceding_search_teaches_nothing() {
        // Nothing ties the tool to an intent, so there is no judgment to record.
        let (l, graph) = learner();
        l.record(invoke("gh_run_list"));
        assert!(graph.read().unwrap().is_empty());
    }

    #[test]
    fn what_retrieval_returned_never_becomes_an_edge() {
        // The central rule (ADR-0014): only invocations are evidence. This search
        // reports `docker_build` as its top hit and the user invokes something
        // else — the graph must learn the invoke, not the hit.
        let (l, graph) = learner();
        l.record(TraceEvent::Search {
            query: "why is the build broken".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: vec![crate::trace::SearchHitTrace {
                tool_id: "docker_build".into(),
                score: 9.9,
            }],
            stages: Vec::new(),
            took_ms: 0,
        });
        l.record(invoke("gh_run_list"));

        let g = graph.read().unwrap();
        assert_eq!(
            g.intents[0].tools.keys().collect::<Vec<_>>(),
            vec!["gh_run_list"]
        );
    }

    #[test]
    fn several_invokes_after_one_search_all_count_as_capabilities() {
        // An agent that searches once and uses three tools genuinely confirmed
        // three capabilities — so three EDGES. But it asked one question, so it
        // is one observation. This assertion on `support` is what was missing:
        // the edge count alone passed while support inflated to 3.
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));
        l.record(invoke("gh_run_view"));
        l.record(invoke("read_file"));

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g.intents[0].tools.len(), 3, "three capabilities were used");
        assert_eq!(g.intents[0].support, 1, "but only one question was asked");
        for (id, w) in &g.intents[0].tools {
            assert_eq!(*w, 1.0, "{id} was used once");
        }
    }

    #[test]
    fn the_same_question_asked_twice_counts_twice() {
        // Two real searches, even with identical text, are two observations.
        // An earlier attempt to dedupe the capability-search double-emit by
        // comparing query text broke exactly this — and protected nothing,
        // since both of those searches arrive before any invoke.
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));
        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));

        let g = graph.read().unwrap();
        assert_eq!(g.intents[0].support, 2);
        assert_eq!(g.intents[0].tools["gh_run_list"], 2.0);
    }

    #[test]
    fn separate_searches_each_count() {
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));
        l.record(search("is the build broken again"));
        l.record(invoke("gh_run_list"));

        let g = graph.read().unwrap();
        assert_eq!(g.intents[0].support, 2, "two questions, two observations");
    }

    #[test]
    fn a_capability_search_across_both_registries_counts_once() {
        // `search_capabilities` searches the tool and skill catalogs with the
        // same text, so ONE logical search emits both a Search and a SkillSearch
        // (src/sdk/ts/src/capabilities.ts). Crediting each would reintroduce the
        // very inflation this guards against.
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(TraceEvent::SkillSearch {
            query: "why is the build broken".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: Vec::new(),
            stages: Vec::new(),
            took_ms: 0,
        });
        l.record(invoke("gh_run_list"));
        l.record(TraceEvent::SkillInvoke {
            skill_id: "ci-triage".into(),
            took_ms: 1,
        });

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(
            g.intents[0].support, 1,
            "one question, however many catalogs it hit"
        );
        assert_eq!(g.intents[0].tools.len(), 1);
        assert_eq!(g.intents[0].skills.len(), 1);
    }

    #[test]
    fn two_learners_sharing_a_graph_count_a_capability_search_once() {
        // The real SDK topology: `search_capabilities` fans one query to a tool
        // catalog and a skill catalog, each with its OWN learner but ONE shared
        // graph. Per-learner crediting double-counts support; the shared credit
        // slot on the graph collapses them into a single observation.
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        let tools = Arc::new(UsageLearner::new(graph.clone(), Arc::new(NoopSink)));
        let skills = Arc::new(UsageLearner::new(graph.clone(), Arc::new(NoopSink)));

        // Fan-out: both searches (same query) arrive before any invoke.
        tools.record(search("why is the build broken"));
        skills.record(TraceEvent::SkillSearch {
            query: "why is the build broken".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: Vec::new(),
            stages: Vec::new(),
            took_ms: 0,
        });
        // The agent uses a tool AND a skill for the one question.
        tools.record(invoke("gh_run_list"));
        skills.record(TraceEvent::SkillInvoke {
            skill_id: "ci-triage".into(),
            took_ms: 1,
        });

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(
            g.intents[0].support, 1,
            "one question, even across two per-catalog learners"
        );
        assert_eq!(g.intents[0].tools.get("gh_run_list"), Some(&1.0));
        assert_eq!(g.intents[0].skills.get("ci-triage"), Some(&1.0));
    }

    #[test]
    fn a_new_search_replaces_the_pending_query() {
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(search("rotate the signing key"));
        l.record(invoke("vault_rotate"));

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1, "only the later query should have been credited");
        assert!(
            g.intents[0]
                .members
                .contains(&"rotate the signing key".to_string())
        );
    }

    #[test]
    fn skill_searches_and_skill_invokes_pair_on_the_skill_edges() {
        let (l, graph) = learner();
        l.record(TraceEvent::SkillSearch {
            query: "why is the build broken".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: Vec::new(),
            stages: Vec::new(),
            took_ms: 0,
        });
        l.record(TraceEvent::SkillInvoke {
            skill_id: "ci-triage".into(),
            took_ms: 1,
        });

        let g = graph.read().unwrap();
        assert_eq!(g.intents[0].skills.get("ci-triage"), Some(&1.0));
        assert!(g.intents[0].tools.is_empty());
    }

    // ---- the shared pairing rule -------------------------------------------

    #[test]
    fn a_rejected_search_is_ignored_not_a_boundary() {
        // The distinction the live and replay paths must agree on: `Ignore`
        // leaves whatever window is open alone, so a stray internal search
        // between a captured query and its invokes cannot discard the turn.
        let policy =
            ObservationPolicy::default().with_origins(OriginFilter::Exactly(Origin::Baseline));
        assert_eq!(
            classify(&search_from("q", Origin::Direct), policy),
            Step::Ignore
        );
        assert_eq!(
            classify(&search_from("q", Origin::Baseline), policy),
            Step::Remember("q")
        );
    }

    #[test]
    fn only_the_attempt_confirms_an_observation() {
        // A trace records which tool was called, never whether calling it was
        // right. `invoke_end` is not a second, better signal — it filters on
        // execution outcome, which is leaky in both directions: a wrong tool
        // that ran fine is kept, a right one that failed on arguments is
        // dropped. So the choice is the only signal, and there is nothing to
        // configure.
        let policy = ObservationPolicy::default();
        assert_eq!(
            classify(&invoke("t"), policy),
            Step::Confirm(Capability::Tool, "t")
        );
        assert_eq!(classify(&invoke_end("t"), policy), Step::Ignore);
        assert_eq!(classify(&invoke_error("t"), policy), Step::Ignore);
    }

    #[test]
    fn an_unrelated_event_is_never_evidence() {
        assert_eq!(
            classify(
                &TraceEvent::AuthNeeds {
                    upstream: "gh".into()
                },
                ObservationPolicy::default()
            ),
            Step::Ignore
        );
    }

    // ---- ObservationPolicy -------------------------------------------------

    fn search_from(query: &str, origin: Origin) -> TraceEvent {
        TraceEvent::Search {
            query: query.into(),
            origin,
            top_k: 5,
            hits: Vec::new(),
            stages: Vec::new(),
            took_ms: 0,
        }
    }

    fn invoke_end(tool_id: &str) -> TraceEvent {
        TraceEvent::InvokeEnd {
            tool_id: tool_id.into(),
            took_ms: 1,
        }
    }

    fn invoke_error(tool_id: &str) -> TraceEvent {
        TraceEvent::InvokeError {
            tool_id: tool_id.into(),
            took_ms: 1,
            error: "bad args".into(),
        }
    }

    fn policy_learner(policy: ObservationPolicy) -> (Arc<UsageLearner>, Arc<RwLock<IntentGraph>>) {
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        let l = Arc::new(UsageLearner::with_policy(
            graph.clone(),
            Arc::new(NoopSink),
            policy,
        ));
        (l, graph)
    }

    #[test]
    fn the_default_policy_reproduces_todays_pairing_exactly() {
        // The additive-evolution guarantee: `new` and `with_policy(default())`
        // must be the same learner. Without this, every later policy field is a
        // chance to silently move the default path.
        let events = || {
            vec![
                search("why is the build broken"),
                invoke("gh_run_list"),
                invoke("gh_run_view"),
                search("rotate the signing key"),
                invoke("vault_rotate"),
            ]
        };

        let (old, old_graph) = learner();
        for e in events() {
            old.record(e);
        }

        let (new, new_graph) = policy_learner(ObservationPolicy::default());
        for e in events() {
            new.record(e);
        }

        let old_g = old_graph.read().unwrap();
        let new_g = new_graph.read().unwrap();
        // Compare the LEARNING, not the graph wholesale: `built_from_ts` and
        // `last_ts` are stamped from the wall clock, so two runs a millisecond
        // apart differ there and nowhere else. `Intent`'s equality already
        // excludes those, which is exactly the cut this assertion wants.
        assert_eq!(old_g.intents, new_g.intents);
        assert_eq!(old_g.rev(), new_g.rev());
        for it in &new_g.intents {
            assert_eq!(it.seeded_support, 0, "the default policy is live");
        }
    }

    #[test]
    fn only_the_required_origin_opens_an_observation_window() {
        let (l, graph) = policy_learner(
            ObservationPolicy::default().with_origins(OriginFilter::Exactly(Origin::Baseline)),
        );

        l.record(search_from("why is the build broken", Origin::Agent));
        l.record(invoke("gh_run_list"));
        assert!(
            graph.read().unwrap().is_empty(),
            "an agent search must not teach a baseline-only learner"
        );

        l.record(search_from("why is the build broken", Origin::Baseline));
        l.record(invoke("gh_run_list"));
        assert_eq!(graph.read().unwrap().len(), 1);
    }

    #[test]
    fn a_filtered_out_search_leaves_the_pending_query_intact() {
        // The subtle one. A baseline capture runs Ratel's own searches too (a
        // pre-fetch helper, a health check). If a filtered search CLEARED the
        // pending query instead of being ignored, one stray internal search
        // between the turn's query and its invokes would silently discard the
        // turn's evidence.
        let (l, graph) = policy_learner(
            ObservationPolicy::default().with_origins(OriginFilter::Exactly(Origin::Baseline)),
        );

        l.record(search_from("why is the build broken", Origin::Baseline));
        l.record(search_from("some internal probe", Origin::Direct));
        l.record(invoke("gh_run_list"));

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1);
        assert!(
            g.intents[0]
                .members
                .contains(&"why is the build broken".to_string()),
            "the baseline query still owns the invoke, got {:?}",
            g.intents[0].members
        );
    }

    #[test]
    fn the_default_policy_still_pairs_on_the_attempt() {
        // Choice is the relevance signal: which tool the agent reached for says
        // what it thought fit, and a later argument error does not retract that.
        let (l, graph) = learner();
        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));
        assert_eq!(graph.read().unwrap().len(), 1);
    }

    #[test]
    fn a_seeded_policy_stamps_provenance_on_what_it_credits() {
        let (l, graph) = policy_learner(
            ObservationPolicy::default()
                .with_origins(OriginFilter::Exactly(Origin::Baseline))
                .with_provenance(Provenance::Seeded),
        );

        l.record(search_from("why is the build broken", Origin::Baseline));
        l.record(invoke("gh_run_list"));
        l.record(invoke("gh_run_view"));

        let g = graph.read().unwrap();
        assert_eq!(g.intents[0].support, 1, "one question");
        assert_eq!(g.intents[0].seeded_support, 1, "and it was seeded");
        assert_eq!(g.intents[0].tools.len(), 2, "two capabilities");
    }

    #[test]
    fn a_skill_invoke_confirms_like_a_tool_invoke() {
        let (l, graph) = policy_learner(ObservationPolicy::default());
        l.record(TraceEvent::SkillSearch {
            query: "why is the build broken".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: Vec::new(),
            stages: Vec::new(),
            took_ms: 0,
        });
        l.record(TraceEvent::SkillInvoke {
            skill_id: "ci-triage".into(),
            took_ms: 1,
        });
        assert_eq!(
            graph.read().unwrap().intents[0].skills.get("ci-triage"),
            Some(&1.0)
        );
    }

    #[test]
    fn every_event_is_forwarded_to_the_inner_sink() {
        // Decorating must be transparent: installing a learner cannot cost the
        // caller their JSONL/inspector stream.
        let inner = Arc::new(MemorySink::new("s"));
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        let l = UsageLearner::new(graph, inner.clone());

        l.record(search("why is the build broken"));
        l.record(invoke("gh_run_list"));
        l.record(TraceEvent::AuthNeeds {
            upstream: "gh".into(),
        });

        assert_eq!(inner.snapshot().len(), 3);
    }

    #[test]
    fn unrelated_events_are_forwarded_without_learning() {
        let (l, graph) = learner();
        l.record(TraceEvent::AuthNeeds {
            upstream: "gh".into(),
        });
        assert!(graph.read().unwrap().is_empty());
    }
}
