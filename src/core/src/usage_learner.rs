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

use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::trace::{TraceEnvelope, TraceEvent, TraceSink};
use crate::usage::{Capability, IntentGraph};

/// The learner's most recent search — the query an invoke attributes to. Kept
/// per-learner (not read from the shared graph) so a concurrent search from
/// another session cannot misattribute this learner's invoke. Whether the
/// question has already been credited a support bump lives on the shared graph
/// ([`IntentGraph::claim_credit`]), so per-catalog tool and skill learners count
/// one fanned-out question once between them, not once each.
struct Pending {
    query: String,
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
}

impl UsageLearner {
    /// Wrap `inner`, learning into `graph`. Pass [`crate::NoopSink`] for `inner`
    /// when the only thing you want is the learning.
    pub fn new(graph: Arc<RwLock<IntentGraph>>, inner: Arc<dyn TraceSink>) -> Self {
        Self {
            inner,
            graph,
            pending: Mutex::new(None),
        }
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
            graph.observe(&query, kind, capability_id, ts_ms, first_confirmation);
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
    fn learn_from(&self, event: &TraceEvent, ts_ms: u64) {
        match event {
            // Both search kinds set the pending query: a capability search hits
            // the tool and skill registries in turn with the same text.
            TraceEvent::Search { query, .. } | TraceEvent::SkillSearch { query, .. } => {
                self.remember_query(query)
            }
            TraceEvent::InvokeStart { tool_id, .. } => {
                self.confirm(Capability::Tool, tool_id, ts_ms)
            }
            TraceEvent::SkillInvoke { skill_id, .. } => {
                self.confirm(Capability::Skill, skill_id, ts_ms)
            }
            _ => {}
        }
    }
}

impl TraceSink for UsageLearner {
    fn record(&self, event: TraceEvent) {
        self.learn_from(&event, now_ms());
        self.inner.record(event);
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
