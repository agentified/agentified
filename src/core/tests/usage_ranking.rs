//! Adaptive usage ranking end to end through the public API (ADR-0014).
//!
//! The load-bearing guarantee is the negative one: a registry with no intent
//! graph — or one whose graph matches nothing — must rank *byte-identically* to
//! today, because ADR-0011 promises `search`/`search_with_origin` keep BM25
//! behavior unchanged.

use std::sync::Arc;

use ratel_ai_core::{IntentGraph, MemorySink, Tool, ToolRegistry, TraceEvent};
use serde_json::json;

fn tool(id: &str, description: &str) -> Tool {
    Tool {
        id: id.into(),
        name: id.into(),
        description: description.into(),
        input_schema: json!({}),
        output_schema: json!({}),
    }
}

/// A catalog where lexical retrieval is confidently wrong: the query
/// "why is the build broken" hits `docker_build` on the token *build*, while the
/// tool people actually invoke is `gh_run_list`.
fn registry() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register(tool(
        "docker_build",
        "Build a Docker image from a Dockerfile",
    ));
    r.register(tool(
        "gh_run_list",
        "List CI workflow runs and whether the build passed",
    ));
    r.register(tool("read_file", "Read a file from disk"));
    r
}

/// A graph whose one cluster covers build/CI questions and remembers
/// `gh_run_list`. `support` is a parameter because the arm's weight ramps with
/// it.
fn graph_json(support: u32) -> String {
    json!({
        "v": 1,

        "built_from_ts": 1_753_000_000_000u64,
        "intents": [{
            "id": "intent_0",
            "label": "is the build green",
            "terms": ["build", "ci"],
            "members": ["why is the build broken", "did CI pass"],
            "support": support,
            "tools": { "gh_run_list": 0.8 },
            "skills": {}
        }]
    })
    .to_string()
}

fn graph(support: u32) -> IntentGraph {
    IntentGraph::from_json(&graph_json(support)).expect("valid graph")
}

fn ids(hits: &[ratel_ai_core::SearchHit]) -> Vec<&str> {
    hits.iter().map(|h| h.tool_id.as_str()).collect()
}

// ---- the negative guarantee ------------------------------------------------

#[test]
fn a_graph_that_matches_nothing_ranks_identically_to_no_graph() {
    let baseline = registry().search("rotate the signing key", 5);

    let mut r = registry();
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    let with_graph = r.search("rotate the signing key", 5);

    assert_eq!(ids(&baseline), ids(&with_graph));
    // Scores too, not just order: a miss must not switch the result to the RRF
    // scale, or callers reading `score` see a silent semantic change.
    for (a, b) in baseline.iter().zip(with_graph.iter()) {
        assert_eq!(a.score, b.score, "score changed for {}", a.tool_id);
    }
}

#[test]
fn an_empty_graph_ranks_identically_to_no_graph() {
    let baseline = registry().search("why is the build broken", 5);

    let mut r = registry();
    r.set_intent_graph(Some(Arc::new(IntentGraph::empty().into())));
    let with_graph = r.search("why is the build broken", 5);

    assert_eq!(ids(&baseline), ids(&with_graph));
    for (a, b) in baseline.iter().zip(with_graph.iter()) {
        assert_eq!(a.score, b.score);
    }
}

// ---- the headline behavior -------------------------------------------------

#[test]
fn without_a_graph_lexical_retrieval_gets_it_wrong() {
    // Establishes the "before" the feature exists to fix.
    let hits = registry().search("why is the build broken", 5);
    assert_eq!(
        hits.first().map(|h| h.tool_id.as_str()),
        Some("docker_build")
    );
}

#[test]
fn the_usage_arm_promotes_the_tool_people_actually_invoke() {
    let mut r = registry();
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    let hits = r.search("why is the build broken", 5);

    let order = ids(&hits);
    let gh = order.iter().position(|id| *id == "gh_run_list");
    let docker = order.iter().position(|id| *id == "docker_build");
    assert!(
        gh < docker,
        "usage history should outrank the lexical decoy, got {order:?}"
    );
}

#[test]
fn support_scales_the_arms_contribution() {
    // The ramp is what makes immediate learning safe: one observation nudges,
    // several dictate. Asserted on the SCORE, not on position — in a small
    // corpus both supports can land the same ordering, so a position assertion
    // silently tests nothing (this test previously did exactly that, and a
    // mutant that ignored `support` entirely survived it).
    let score_of = |support: u32| {
        let mut r = registry();
        r.set_intent_graph(Some(Arc::new(graph(support).into())));
        r.search("why is the build broken", 5)
            .into_iter()
            .find(|h| h.tool_id == "gh_run_list")
            .expect("boosted tool is present")
            .score
    };

    let weak = score_of(1);
    let strong = score_of(9);
    assert!(
        weak < strong,
        "support 1 ({weak}) must contribute less than support 9 ({strong})"
    );

    // And by exactly the ramp's amount: both share the same BM25 rank term, so
    // the gap is (W - W/3)/RRF_K with W = 0.5.
    let expected_gap = (0.5 - 0.5 / 3.0) / 60.0;
    assert!(
        (strong - weak - expected_gap).abs() < 1e-6,
        "gap {} should be the ramp delta {expected_gap}",
        strong - weak
    );
}

#[test]
fn support_at_the_cap_and_beyond_contribute_equally() {
    // The ramp saturates at SUPPORT_FULL: a wildly popular cluster must not
    // outvote a merely confirmed one without bound.
    let score_of = |support: u32| {
        let mut r = registry();
        r.set_intent_graph(Some(Arc::new(graph(support).into())));
        r.search("why is the build broken", 5)
            .into_iter()
            .find(|h| h.tool_id == "gh_run_list")
            .expect("present")
            .score
    };
    assert_eq!(score_of(3), score_of(900));
}

#[test]
fn edges_naming_an_unregistered_tool_are_not_ranked() {
    // The graph outlives a catalog change; a ghost id must never be returned,
    // since the agent could not invoke it.
    let json = json!({
        "v": 1,  "built_from_ts": 1u64,
        "intents": [{
            "id": "intent_0", "label": "l", "terms": [],
            "members": ["why is the build broken"], "support": 9,
            "tools": { "since_deleted_tool": 0.9, "gh_run_list": 0.1 }, "skills": {}
        }]
    })
    .to_string();
    let mut r = registry();
    r.set_intent_graph(Some(Arc::new(
        IntentGraph::from_json(&json).unwrap().into(),
    )));
    let hits = r.search("why is the build broken", 5);
    assert!(!ids(&hits).contains(&"since_deleted_tool"));
}

// ---- tracing ---------------------------------------------------------------

fn usage_events(sink: &MemorySink) -> Vec<(Option<String>, u32, u32)> {
    sink.snapshot()
        .into_iter()
        .filter_map(|e| match e.event {
            TraceEvent::UsageBoost {
                intent,
                support,
                promoted,
                ..
            } => Some((intent, support, promoted)),
            _ => None,
        })
        .collect()
}

/// The match score carried by each `UsageBoost` — cosine on the dense tier,
/// token-overlap share on the lexical one.
fn usage_similarities(sink: &MemorySink) -> Vec<f64> {
    sink.snapshot()
        .into_iter()
        .filter_map(|e| match e.event {
            TraceEvent::UsageBoost { similarity, .. } => Some(similarity),
            _ => None,
        })
        .collect()
}

#[test]
fn a_hit_reports_how_well_it_matched_and_a_miss_reports_zero() {
    // The score makes near-misses visible: without it, a query that scored 0.69
    // against a 0.70 threshold is indistinguishable from one that scored 0.05.
    let sink = Arc::new(MemorySink::new("s"));
    let mut r = registry();
    r.set_trace_sink(sink.clone());
    r.set_intent_graph(Some(Arc::new(graph(9).into())));

    r.search("why is the build broken", 5);
    let hit = usage_similarities(&sink)[0];
    assert!(hit > 0.0 && hit <= 1.0, "expected a match score, got {hit}");

    let sink2 = Arc::new(MemorySink::new("s"));
    r.set_trace_sink(sink2.clone());
    r.search("rotate the signing key", 5);
    assert_eq!(usage_similarities(&sink2), vec![0.0]);
}

#[test]
fn a_registry_without_a_graph_emits_no_usage_event() {
    // The event's presence is itself the "adaptive ranking is on" signal.
    let sink = Arc::new(MemorySink::new("s"));
    let mut r = registry();
    r.set_trace_sink(sink.clone());
    r.search("why is the build broken", 5);
    assert!(usage_events(&sink).is_empty());
}

#[test]
fn a_match_records_the_cluster_and_how_much_it_promoted() {
    let sink = Arc::new(MemorySink::new("s"));
    let mut r = registry();
    r.set_trace_sink(sink.clone());
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    r.search("why is the build broken", 5);

    assert_eq!(
        usage_events(&sink),
        vec![(Some("intent_0".to_string()), 9, 1)]
    );
}

#[test]
fn a_miss_is_recorded_so_staleness_is_observable() {
    let sink = Arc::new(MemorySink::new("s"));
    let mut r = registry();
    r.set_trace_sink(sink.clone());
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    r.search("rotate the signing key", 5);

    assert_eq!(usage_events(&sink), vec![(None, 0, 0)]);
}

#[test]
fn a_boosted_search_reports_its_usage_and_fusion_stages() {
    let sink = Arc::new(MemorySink::new("s"));
    let mut r = registry();
    r.set_trace_sink(sink.clone());
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    r.search("why is the build broken", 5);

    let stages: Vec<String> = sink
        .snapshot()
        .into_iter()
        .find_map(|e| match e.event {
            TraceEvent::Search { stages, .. } => Some(stages),
            _ => None,
        })
        .expect("a search event")
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(stages, vec!["bm25", "usage", "rrf"]);
}

#[test]
fn an_unboosted_search_keeps_its_single_bm25_stage() {
    let sink = Arc::new(MemorySink::new("s"));
    let mut r = registry();
    r.set_trace_sink(sink.clone());
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    r.search("rotate the signing key", 5);

    let stages: Vec<String> = sink
        .snapshot()
        .into_iter()
        .find_map(|e| match e.event {
            TraceEvent::Search { stages, .. } => Some(stages),
            _ => None,
        })
        .expect("a search event")
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(stages, vec!["bm25"]);
}

// ---- the documented ceiling ------------------------------------------------

#[test]
fn a_tool_the_base_ranker_never_retrieved_is_not_promoted() {
    // ADR-0014's honest boundary for W < 1: the arm can lift a capability the
    // base ranker scored poorly, but not one it never returned at all. Pinned as
    // a test so the limitation is discovered here rather than in production.
    let mut r = ToolRegistry::new();
    r.register(tool(
        "docker_build",
        "Build a Docker image from a Dockerfile",
    ));
    // Shares no vocabulary with the query, so BM25 never retrieves it.
    r.register(tool(
        "vault_rotate",
        "Rotate a credential in the secret store",
    ));

    let json = json!({
        "v": 1,  "built_from_ts": 1u64,
        "intents": [{
            "id": "intent_0", "label": "l", "terms": [],
            "members": ["why is the build broken"], "support": 9,
            "tools": { "vault_rotate": 1.0 }, "skills": {}
        }]
    })
    .to_string();
    r.set_intent_graph(Some(Arc::new(
        IntentGraph::from_json(&json).unwrap().into(),
    )));

    let hits = r.search("why is the build broken", 5);
    assert_eq!(
        hits.first().map(|h| h.tool_id.as_str()),
        Some("docker_build"),
        "at W<1 an arm-only capability cannot outrank the base ranker's top hit"
    );
}

// ---- the skill-side twin ---------------------------------------------------

fn skill(id: &str, description: &str) -> ratel_ai_core::Skill {
    ratel_ai_core::Skill {
        id: id.into(),
        name: id.into(),
        description: description.into(),
        tags: Vec::new(),
        tools: Vec::new(),
        metadata: Default::default(),
        body: String::new(),
    }
}

#[test]
fn skills_are_boosted_from_the_same_graphs_skill_edges() {
    // One graph serves both registries: `tools` feeds ToolRegistry, `skills`
    // feeds SkillRegistry. A cluster can remember both.
    use ratel_ai_core::SkillRegistry;

    let mut r = SkillRegistry::new();
    r.register(skill(
        "docker-build-guide",
        "How to build and publish a Docker image",
    ));
    r.register(skill(
        "ci-triage",
        "Diagnose why the build failed in CI and what to check",
    ));

    let before = r.search("why is the build broken", 5);
    let pos_before = before
        .iter()
        .position(|h| h.skill_id == "ci-triage")
        .expect("present");

    let json = json!({
        "v": 1,  "built_from_ts": 1u64,
        "intents": [{
            "id": "intent_0", "label": "l", "terms": [],
            "members": ["why is the build broken"], "support": 9,
            "tools": {}, "skills": { "ci-triage": 1.0 }
        }]
    })
    .to_string();
    r.set_intent_graph(Some(Arc::new(
        IntentGraph::from_json(&json).unwrap().into(),
    )));

    let after = r.search("why is the build broken", 5);
    let pos_after = after
        .iter()
        .position(|h| h.skill_id == "ci-triage")
        .expect("present");
    assert!(
        pos_after <= pos_before,
        "the skills arm should not rank it lower"
    );
    assert_eq!(after[0].skill_id, "ci-triage");
}

#[test]
fn a_tools_only_cluster_contributes_no_skill_arm() {
    // Edge maps are independent: a cluster that only remembers tools must leave
    // skill ranking untouched.
    use ratel_ai_core::SkillRegistry;

    let build = || {
        let mut r = SkillRegistry::new();
        r.register(skill("docker-build-guide", "How to build a Docker image"));
        r.register(skill("ci-triage", "Diagnose why the build failed in CI"));
        r
    };
    let baseline = build().search("why is the build broken", 5);

    let mut r = build();
    r.set_intent_graph(Some(Arc::new(graph(9).into()))); // tools-only cluster
    let with_graph = r.search("why is the build broken", 5);

    assert_eq!(baseline.len(), with_graph.len());
    for (a, b) in baseline.iter().zip(with_graph.iter()) {
        assert_eq!(a.skill_id, b.skill_id);
        assert_eq!(a.score, b.score, "score changed for {}", a.skill_id);
    }
}

// ---- the closed loop: learn from use, then rank better ---------------------

use ratel_ai_core::{NoopSink, UsageLearner};
use std::sync::RwLock;

/// Wire a registry so that what it learns is what it reads: the learner writes
/// the graph, the registry ranks against it.
fn self_teaching_registry() -> (ToolRegistry, std::sync::Arc<RwLock<IntentGraph>>) {
    let graph = std::sync::Arc::new(RwLock::new(IntentGraph::empty()));
    let learner = Arc::new(UsageLearner::new(graph.clone(), Arc::new(NoopSink)));
    let mut r = registry();
    r.set_trace_sink(learner);
    r.set_intent_graph(Some(graph.clone()));
    (r, graph)
}

/// Search, then invoke `chosen` — one confirmed observation, exactly as an agent
/// would produce it.
fn use_it(r: &ToolRegistry, query: &str, chosen: &str) {
    r.search(query, 5);
    r.record_event(TraceEvent::InvokeStart {
        tool_id: chosen.into(),
        args_size_bytes: 0,
    });
}

#[test]
fn a_registry_learns_from_use_and_then_ranks_better() {
    let (r, graph) = self_teaching_registry();

    // Cold: no graph, so lexical retrieval makes its usual mistake.
    let cold = r.search("why is the build broken", 5);
    assert_eq!(
        cold.first().map(|h| h.tool_id.as_str()),
        Some("docker_build"),
        "before learning, the token 'build' wins"
    );

    // Three sessions where people ask about builds and reach for gh_run_list.
    use_it(&r, "why is the build broken", "gh_run_list");
    use_it(&r, "is the build broken again", "gh_run_list");
    use_it(&r, "the build broken on main", "gh_run_list");

    assert_eq!(graph.read().unwrap().len(), 1, "one cluster formed");
    assert_eq!(graph.read().unwrap().intents[0].support, 3);

    // Warm: the same catalog, the same query, a better answer — learned entirely
    // from observed behavior, with no graph authored by hand.
    let warm = r.search("why is the build broken", 5);
    let order = ids(&warm);
    let gh = order.iter().position(|id| *id == "gh_run_list");
    let docker = order.iter().position(|id| *id == "docker_build");
    assert!(
        gh < docker,
        "after learning, usage should outrank the lexical decoy, got {order:?}"
    );
}

#[test]
fn learning_generalizes_to_a_phrasing_never_observed() {
    // The reason clusters exist rather than exact-match memoization: a query the
    // system has never seen benefits from ones it has.
    let (r, _graph) = self_teaching_registry();
    use_it(&r, "why is the build broken", "gh_run_list");
    use_it(&r, "is the build broken again", "gh_run_list");
    use_it(&r, "the build broken on main", "gh_run_list");

    let novel = r.search("build broken somehow", 5);
    let order = ids(&novel);
    assert!(
        order.iter().position(|id| *id == "gh_run_list")
            < order.iter().position(|id| *id == "docker_build"),
        "a novel phrasing should inherit the cluster's evidence, got {order:?}"
    );
}

#[test]
fn an_unrelated_query_is_untouched_by_what_was_learned() {
    let (r, _graph) = self_teaching_registry();
    let baseline = registry().search("read a file from disk", 5);

    use_it(&r, "why is the build broken", "gh_run_list");
    use_it(&r, "is the build broken again", "gh_run_list");
    use_it(&r, "the build broken on main", "gh_run_list");

    let after = r.search("read a file from disk", 5);
    assert_eq!(ids(&baseline), ids(&after));
    for (a, b) in baseline.iter().zip(after.iter()) {
        assert_eq!(
            a.score, b.score,
            "unrelated query perturbed for {}",
            a.tool_id
        );
    }
}

#[test]
fn a_single_stray_invocation_does_not_reshape_ranking() {
    // The support ramp in practice: one misclick must not dictate policy.
    let (r, _graph) = self_teaching_registry();
    let baseline = registry().search("why is the build broken", 5);

    use_it(&r, "why is the build broken", "read_file"); // one odd choice
    let after = r.search("why is the build broken", 5);

    assert_eq!(
        after.first().map(|h| h.tool_id.as_str()),
        baseline.first().map(|h| h.tool_id.as_str()),
        "a lone observation must not flip the top result"
    );
}

// ---- rank and fused expose the score-scale switch (problem #2) --------------

#[test]
fn rank_is_the_zero_based_position() {
    let hits = registry().search("why is the build broken", 5);
    for (i, h) in hits.iter().enumerate() {
        assert_eq!(h.rank, i as u32, "rank must be the list position");
    }
}

#[test]
fn without_a_graph_scores_are_raw_and_unfused() {
    // The byte-identical promise, now also asserting the flag: no graph means no
    // fusion, so scores stay on the raw BM25 scale.
    let hits = registry().search("why is the build broken", 5);
    assert!(hits.iter().all(|h| !h.fused), "no graph → not fused");
    assert!(hits[0].score > 0.1, "raw BM25 score, not a ~0.03 RRF score");
}

#[test]
fn a_matched_query_is_flagged_fused() {
    let mut r = registry();
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    let hits = r.search("why is the build broken", 5);
    assert!(hits.iter().all(|h| h.fused), "the usage arm fused → fused");
    assert!(hits[0].score < 0.1, "RRF score, small magnitude");
    // rank still contiguous from 0 through the fused list
    for (i, h) in hits.iter().enumerate() {
        assert_eq!(h.rank, i as u32);
    }
}

#[test]
fn a_missed_query_stays_unfused_even_with_a_graph() {
    // Same catalog, different query: this one matches no cluster, so it must
    // report the raw scale — the exact between-calls switch `fused` exists for.
    let mut r = registry();
    r.set_intent_graph(Some(Arc::new(graph(9).into())));
    let hits = r.search("rotate the signing key", 5);
    assert!(
        hits.iter().all(|h| !h.fused),
        "no cluster matched → not fused"
    );
}
