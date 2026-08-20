use std::sync::{Arc, Condvar, Mutex, RwLock, mpsc};

use ratel_ai_core::{
    CatalogKind, ChurnKind, Fact, FactRegistry, FanoutSink, FnSink, IntentGraph, JsonlSink,
    MemorySink, NoopSink, Origin, PinMode, Skill, SkillRegistry, Tool, ToolRegistry, TraceEnvelope,
    TraceEvent, TraceEventContext, TraceSink, UsageLearner,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn catalog_canonicalization_vectors() -> Value {
    serde_json::from_str(include_str!("../../telemetry/conformance/fixtures.json")).unwrap()
}

struct BlockingSink {
    released: (Mutex<bool>, Condvar),
    started: Mutex<Option<mpsc::SyncSender<()>>>,
    events: Mutex<Vec<TraceEnvelope>>,
}

impl BlockingSink {
    fn new(started: mpsc::SyncSender<()>) -> Self {
        Self {
            released: (Mutex::new(false), Condvar::new()),
            started: Mutex::new(Some(started)),
            events: Mutex::new(Vec::new()),
        }
    }

    fn release(&self) {
        *self.released.0.lock().unwrap() = true;
        self.released.1.notify_all();
    }

    fn snapshot(&self) -> Vec<TraceEnvelope> {
        self.events.lock().unwrap().clone()
    }
}

impl TraceSink for BlockingSink {
    fn record(&self, _event: TraceEvent) {}

    fn record_envelope(&self, envelope: TraceEnvelope) {
        if let Some(started) = self.started.lock().unwrap().take() {
            started.send(()).unwrap();
            let mut released = self.released.0.lock().unwrap();
            while !*released {
                released = self.released.1.wait(released).unwrap();
            }
        }
        self.events.lock().unwrap().push(envelope);
    }
}

fn empty_schema() -> Value {
    json!({})
}

fn lookup_tool(id: &str) -> Tool {
    Tool {
        id: id.into(),
        name: id.into(),
        description: "lookup".into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    }
}

#[test]
fn default_registry_uses_noop_sink_and_does_not_panic() {
    let mut registry = ToolRegistry::new();
    registry.register(lookup_tool("t"));
    let _ = registry.search("lookup", 5);
}

#[test]
fn register_emits_index_churn_add() {
    let sink = Arc::new(MemorySink::with_source("session-1", "ratel"));
    let mut registry = ToolRegistry::with_trace_sink(sink.clone());
    registry.register(lookup_tool("alpha"));

    let events = sink.snapshot();
    assert_eq!(events.len(), 1);
    let env: &TraceEnvelope = &events[0];
    assert_eq!(env.session_id, "session-1");
    assert_eq!(env.source_id, "ratel");
    assert_eq!(env.v, 2);
    assert_eq!(env.event_id.len(), 26);
    assert!(env.ts > 0);
    match &env.event {
        TraceEvent::IndexChurn { kind, tool_id } => {
            assert_eq!(*kind, ChurnKind::Add);
            assert_eq!(tool_id, "alpha");
        }
        other => panic!("expected IndexChurn, got {other:?}"),
    }
}

#[test]
fn catalog_definitions_require_explicit_experimental_opt_in() {
    let sink = Arc::new(MemorySink::with_source("session-1", "ratel"));
    let mut tools = ToolRegistry::with_trace_sink(sink.clone());
    tools.register(lookup_tool("read"));
    assert!(
        sink.snapshot()
            .iter()
            .all(|envelope| !matches!(envelope.event, TraceEvent::CatalogDefinition { .. }))
    );

    tools.experimental_enable_catalog_definitions();
    let mut changed = lookup_tool("read");
    changed.description = "Read changed records".into();
    tools.register(changed);
    assert!(
        sink.snapshot()
            .iter()
            .any(|envelope| matches!(envelope.event, TraceEvent::CatalogDefinition { .. }))
    );
}

#[test]
fn registration_emits_complete_catalog_definitions_for_every_kind() {
    let sink = Arc::new(MemorySink::with_source("session-1", "ratel"));
    let mut tools = ToolRegistry::with_trace_sink(sink.clone());
    tools.experimental_enable_catalog_definitions();
    let mut tool = lookup_tool("read");
    tool.name = "read_records".into();
    tool.description = "Read records".into();
    tool.experimental_searchable_description = Some("find archive".into());
    tool.input_schema = json!({
        "type": "object",
        "properties": { "path": { "type": "string" } }
    });
    tool.output_schema = json!({ "type": "string" });
    tools.register(tool);

    let mut skills = SkillRegistry::with_trace_sink(sink.clone());
    skills.experimental_enable_catalog_definitions();
    skills.register(Skill {
        id: "review".into(),
        name: "review_code".into(),
        description: "Review source".into(),
        experimental_searchable_description: None,
        tags: vec!["quality".into()],
        tools: vec!["read".into()],
        metadata: Default::default(),
        body: "private instructions".into(),
    });

    let mut facts = FactRegistry::with_trace_sink(sink.clone());
    facts.experimental_enable_catalog_definitions();
    facts.register(Fact {
        id: "address".into(),
        name: "shop_address".into(),
        description: "Where the shop is".into(),
        experimental_searchable_description: None,
        tags: vec!["location".into()],
        metadata: Default::default(),
        body: "private address".into(),
        pin: PinMode::Always,
    });

    let definitions: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter_map(|envelope| match envelope.event {
            TraceEvent::CatalogDefinition {
                kind,
                id,
                name,
                description,
                tags,
                input_schema,
                output_schema,
                searchable_description,
                searchable_description_overridden,
                content_hash,
            } => Some((
                kind,
                id,
                name,
                description,
                tags,
                input_schema,
                output_schema,
                searchable_description,
                searchable_description_overridden,
                content_hash,
            )),
            _ => None,
        })
        .collect();

    assert_eq!(definitions.len(), 3);
    assert_eq!(definitions[0].0, CatalogKind::Tool);
    assert_eq!(definitions[0].1, "read");
    assert_eq!(definitions[0].2, "read_records");
    assert_eq!(definitions[0].3, "Read records");
    assert!(definitions[0].4.is_empty());
    assert_eq!(
        definitions[0].5.as_deref(),
        Some(&json!({
            "type": "object",
            "properties": { "path": { "type": "string" } }
        }))
    );
    assert_eq!(
        definitions[0].6.as_deref(),
        Some(&json!({ "type": "string" }))
    );
    assert_eq!(definitions[0].7, "find archive");
    assert!(definitions[0].8);
    assert_eq!(
        definitions[0].9,
        "b114cd9a32169e1b0f05b4cf993ef7babc63c27e4944af3dbf7ec984ca507e4f"
    );

    assert_eq!(definitions[1].0, CatalogKind::Skill);
    assert_eq!(definitions[1].1, "review");
    assert_eq!(definitions[1].4, vec!["quality"]);
    assert_eq!(definitions[1].5, None);
    assert_eq!(definitions[1].6, None);
    assert_eq!(definitions[1].7, "Review source");
    assert!(!definitions[1].8);

    assert_eq!(definitions[2].0, CatalogKind::Fact);
    assert_eq!(definitions[2].1, "address");
    assert_eq!(definitions[2].4, vec!["location"]);
    assert_eq!(definitions[2].7, "Where the shop is");
    assert!(!definitions[2].8);
}

#[test]
fn catalog_definitions_match_shared_rfc8785_vectors() {
    let fixtures = catalog_canonicalization_vectors();
    for vector in fixtures["catalog_definition_canonicalization"]["canonicalizer_only_vectors"]
        .as_array()
        .unwrap()
    {
        assert_eq!(
            serde_json_canonicalizer::to_string(&vector["input"]).unwrap(),
            vector["canonical"].as_str().unwrap()
        );
    }
    let vectors = fixtures["catalog_definition_canonicalization"]["vectors"]
        .as_array()
        .unwrap();

    for vector in vectors {
        let input = &vector["input"];
        let canonical = serde_json_canonicalizer::to_string(input).unwrap();
        assert_eq!(canonical, vector["canonical"].as_str().unwrap());

        let sink = Arc::new(MemorySink::with_source("jcs", "ratel"));
        let mut registry = ToolRegistry::with_trace_sink(sink.clone());
        registry.experimental_enable_catalog_definitions();
        registry.register(Tool {
            id: input["id"].as_str().unwrap().into(),
            name: input["name"].as_str().unwrap().into(),
            description: input["description"].as_str().unwrap().into(),
            experimental_searchable_description: input["searchable_description_overridden"]
                .as_bool()
                .unwrap()
                .then(|| input["searchable_description"].as_str().unwrap().into()),
            input_schema: input["input_schema"].clone(),
            output_schema: input["output_schema"].clone(),
        });

        let hash = sink
            .drain()
            .into_iter()
            .find_map(|envelope| match envelope.event {
                TraceEvent::CatalogDefinition { content_hash, .. } => Some(content_hash),
                _ => None,
            })
            .unwrap();
        assert_eq!(hash, vector["sha256"].as_str().unwrap());
    }
}

#[test]
fn catalog_definitions_skip_shared_unsafe_integer_vectors_and_recover() {
    let fixtures = catalog_canonicalization_vectors();
    let vectors = fixtures["catalog_definition_canonicalization"]["rejected_vectors"]
        .as_array()
        .unwrap();

    for vector in vectors {
        let input = &vector["input"];
        let tool = |input: &Value| Tool {
            id: input["id"].as_str().unwrap().into(),
            name: input["name"].as_str().unwrap().into(),
            description: input["description"].as_str().unwrap().into(),
            experimental_searchable_description: None,
            input_schema: input["input_schema"].clone(),
            output_schema: input["output_schema"].clone(),
        };
        let sink = Arc::new(MemorySink::with_source("unsafe", "ratel"));
        let mut registry = ToolRegistry::with_trace_sink(sink.clone());
        registry.experimental_enable_catalog_definitions();

        registry.register(tool(input));

        assert_eq!(registry.len(), 1);
        assert!(
            sink.drain()
                .into_iter()
                .all(|event| !matches!(event.event, TraceEvent::CatalogDefinition { .. }))
        );

        let mut safe = input.clone();
        safe["input_schema"] = json!({
            "type": "integer",
            "minimum": -9_007_199_254_740_991_i64,
            "maximum": 9_007_199_254_740_991_u64,
        });
        registry.register(tool(&safe));
        assert!(
            sink.drain()
                .into_iter()
                .any(|event| matches!(event.event, TraceEvent::CatalogDefinition { .. }))
        );
    }
}

#[test]
fn catalog_definition_canonicalizer_rejects_non_json_numbers() {
    for number in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(serde_json_canonicalizer::to_vec(&number).is_err());
    }
}

#[test]
fn byte_identical_registration_emits_no_duplicate_definition() {
    let sink = Arc::new(MemorySink::new("session-1"));
    let mut registry = ToolRegistry::with_trace_sink(sink.clone());
    registry.experimental_enable_catalog_definitions();

    registry.register(lookup_tool("read"));
    registry.register(lookup_tool("read"));

    let definitions = sink
        .snapshot()
        .into_iter()
        .filter(|envelope| matches!(envelope.event, TraceEvent::CatalogDefinition { .. }))
        .count();
    assert_eq!(definitions, 1);
}

#[test]
fn envelope_v2_carries_explicit_source_and_span_context() {
    let sink = MemorySink::with_source("session", "worker-a");

    sink.record_with_context(
        TraceEvent::AuthNeeds {
            upstream: "github".into(),
        },
        TraceEventContext {
            trace_id: Some("trace-1".into()),
            span_id: Some("span-1".into()),
            ..TraceEventContext::default()
        },
    );

    let events = sink.snapshot();
    let envelope = &events[0];
    assert_eq!(envelope.source_id, "worker-a");
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-1"));
    assert_eq!(envelope.span_id.as_deref(), Some("span-1"));
    assert!(envelope.invocation_id.is_none());
}

#[test]
fn invocation_lifecycle_events_share_one_invocation_id() {
    let sink = MemorySink::new("session");

    sink.record(TraceEvent::InvokeStart {
        tool_id: "alpha".into(),
        args_size_bytes: 1,
    });
    sink.record(TraceEvent::InvokeEnd {
        tool_id: "alpha".into(),
        took_ms: 2,
    });
    sink.record(TraceEvent::InvokeStart {
        tool_id: "alpha".into(),
        args_size_bytes: 3,
    });
    sink.record(TraceEvent::InvokeError {
        tool_id: "alpha".into(),
        took_ms: 4,
        error: "boom".into(),
    });

    let events = sink.snapshot();
    assert_eq!(events[0].invocation_id, events[1].invocation_id);
    assert_eq!(events[2].invocation_id, events[3].invocation_id);
    assert_ne!(events[0].invocation_id, events[2].invocation_id);
    assert!(events.iter().all(|event| event.invocation_id.is_some()));
    assert!(
        events
            .windows(2)
            .all(|pair| pair[0].event_id != pair[1].event_id)
    );
}

#[test]
fn explicit_invocation_context_pairs_concurrent_same_tool_calls() {
    let sink = MemorySink::new("session");
    let first = TraceEventContext::new_invocation();
    let second = TraceEventContext::new_invocation();

    sink.record_with_context(
        TraceEvent::InvokeStart {
            tool_id: "alpha".into(),
            args_size_bytes: 1,
        },
        first.clone(),
    );
    sink.record_with_context(
        TraceEvent::InvokeStart {
            tool_id: "alpha".into(),
            args_size_bytes: 2,
        },
        second.clone(),
    );
    sink.record_with_context(
        TraceEvent::InvokeEnd {
            tool_id: "alpha".into(),
            took_ms: 3,
        },
        second.clone(),
    );
    sink.record_with_context(
        TraceEvent::InvokeError {
            tool_id: "alpha".into(),
            took_ms: 4,
            error: "boom".into(),
        },
        first.clone(),
    );

    let events = sink.snapshot();
    assert_eq!(events[0].invocation_id, first.invocation_id);
    assert_eq!(events[1].invocation_id, second.invocation_id);
    assert_eq!(events[2].invocation_id, second.invocation_id);
    assert_eq!(events[3].invocation_id, first.invocation_id);
}

#[test]
fn explicit_invocation_start_does_not_shift_legacy_pairing() {
    let sink = MemorySink::new("session");
    let explicit = TraceEventContext::new_invocation();

    sink.record_with_context(
        TraceEvent::InvokeStart {
            tool_id: "alpha".into(),
            args_size_bytes: 1,
        },
        explicit.clone(),
    );
    sink.record(TraceEvent::InvokeStart {
        tool_id: "alpha".into(),
        args_size_bytes: 2,
    });
    sink.record(TraceEvent::InvokeEnd {
        tool_id: "alpha".into(),
        took_ms: 3,
    });

    let events = sink.snapshot();
    assert_eq!(events[0].invocation_id, explicit.invocation_id);
    assert_eq!(events[1].invocation_id, events[2].invocation_id);
    assert_ne!(events[0].invocation_id, events[2].invocation_id);
}

#[test]
fn legacy_invocation_pairing_drops_oldest_after_the_per_tool_limit() {
    let sink = MemorySink::new("session");

    for _ in 0..=1_024 {
        sink.record(TraceEvent::InvokeStart {
            tool_id: "alpha".into(),
            args_size_bytes: 0,
        });
    }
    sink.record(TraceEvent::InvokeEnd {
        tool_id: "alpha".into(),
        took_ms: 1,
    });

    let events = sink.snapshot();
    assert_eq!(events[1].invocation_id, events[1_025].invocation_id);
    assert_ne!(events[0].invocation_id, events[1_025].invocation_id);
}

#[test]
fn registry_forwards_event_context_to_its_sink() {
    let sink = Arc::new(MemorySink::new("session"));
    let registry = ToolRegistry::with_trace_sink(sink.clone());

    registry.record_event_with_context(
        TraceEvent::AuthNeeds {
            upstream: "github".into(),
        },
        TraceEventContext {
            environment: Some("production".into()),
            end_user_id: Some("user-1".into()),
            ..TraceEventContext::default()
        },
    );

    let events = sink.snapshot();
    assert_eq!(events[0].environment.as_deref(), Some("production"));
    assert_eq!(events[0].end_user_id.as_deref(), Some("user-1"));
}

#[test]
fn fanout_is_non_blocking_and_reports_drop_oldest_loss() {
    let fanout = FanoutSink::with_source("session", "source");
    let fast = Arc::new(MemorySink::new("ignored"));
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let slow = Arc::new(BlockingSink::new(started_tx));
    let fast_subscription = fanout.subscribe(fast.clone(), 8);
    let slow_subscription = fanout.subscribe(slow.clone(), 2);

    fanout.record(TraceEvent::AuthNeeds {
        upstream: "one".into(),
    });
    started_rx.recv().unwrap();
    for upstream in ["two", "three", "four"] {
        fanout.record(TraceEvent::AuthNeeds {
            upstream: upstream.into(),
        });
    }

    assert_eq!(slow_subscription.dropped_count(), 1);
    assert_eq!(fanout.dropped_count(), 1);
    slow.release();
    fanout.flush();

    let fast_events = fast.snapshot();
    let slow_events = slow.snapshot();
    assert_eq!(fast_events.len(), 4);
    assert_eq!(slow_events.len(), 4);
    assert_eq!(slow_events[0].event_id, fast_events[0].event_id);
    assert_eq!(slow_events[2].event_id, fast_events[2].event_id);
    assert_eq!(slow_events[3].event_id, fast_events[3].event_id);
    assert!(fast_events.iter().all(|event| event.source_id == "source"));
    assert_eq!(fast_subscription.dropped_count(), 0);

    match &slow_events[1].event {
        TraceEvent::EventsDropped {
            dropped_count,
            reason,
            window_start_ts,
            window_end_ts,
        } => {
            assert_eq!(*dropped_count, 1);
            assert_eq!(reason, "queue_overflow");
            assert!(*window_start_ts > 0);
            assert!(*window_end_ts >= *window_start_ts);
        }
        event => panic!("expected events_dropped, got {event:?}"),
    }
}

#[test]
fn usage_learner_subscriber_learns_and_preserves_fanout_identity() {
    let fanout = FanoutSink::with_source("session", "source");
    let graph = Arc::new(RwLock::new(IntentGraph::empty()));
    let recorded = Arc::new(MemorySink::new("ignored"));
    let learner = Arc::new(UsageLearner::new(graph.clone(), recorded.clone()));
    let subscription = fanout.subscribe(learner, 8);

    fanout.record(TraceEvent::Search {
        query: "find logs".into(),
        origin: Origin::Agent,
        top_k: 5,
        hits: vec![],
        stages: vec![],
        took_ms: 1,
    });
    fanout.record(TraceEvent::InvokeStart {
        tool_id: "logs".into(),
        args_size_bytes: 0,
    });
    subscription.flush();

    assert_eq!(graph.read().unwrap().len(), 1);
    let events = recorded.snapshot();
    assert_eq!(events.len(), 2);
    assert!(events.iter().all(|event| event.source_id == "source"));
}

#[test]
fn search_emits_search_event_with_bm25_stage_and_hits() {
    let sink = Arc::new(MemorySink::new("session-2"));
    let mut registry = ToolRegistry::with_trace_sink(sink.clone());
    registry.register(lookup_tool("alpha"));

    let hits = registry.search("lookup", 5);
    assert!(!hits.is_empty());

    let events = sink.snapshot();
    let search_event = events
        .iter()
        .find(|e| matches!(e.event, TraceEvent::Search { .. }))
        .expect("expected a search event");

    match &search_event.event {
        TraceEvent::Search {
            query,
            origin,
            top_k,
            hits,
            stages,
            ..
        } => {
            assert_eq!(query, "lookup");
            assert_eq!(*origin, Origin::Direct);
            assert_eq!(*top_k, 5);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].tool_id, "alpha");
            assert!(hits[0].score > 0.0);
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].name, "bm25");
            assert_eq!(stages[0].top_score, Some(hits[0].score));
        }
        _ => unreachable!(),
    }
}

#[test]
fn search_with_origin_propagates_origin() {
    let sink = Arc::new(MemorySink::new("session-3"));
    let mut registry = ToolRegistry::with_trace_sink(sink.clone());
    registry.register(lookup_tool("alpha"));

    let _ = registry.search_with_origin("lookup", 3, Origin::Agent);

    let events = sink.snapshot();
    let search_event = events
        .iter()
        .find(|e| matches!(e.event, TraceEvent::Search { .. }))
        .expect("expected a search event");
    if let TraceEvent::Search { origin, .. } = &search_event.event {
        assert_eq!(*origin, Origin::Agent);
    }
}

#[test]
fn empty_registry_search_still_emits_event() {
    let sink = Arc::new(MemorySink::new("session-4"));
    let registry = ToolRegistry::with_trace_sink(sink.clone());

    let _ = registry.search("anything", 5);

    let events = sink.snapshot();
    assert_eq!(events.len(), 1);
    match &events[0].event {
        TraceEvent::Search { hits, stages, .. } => {
            assert!(hits.is_empty());
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].name, "bm25");
            assert!(stages[0].top_score.is_none());
        }
        _ => panic!("expected Search event"),
    }
}

#[test]
fn record_event_passes_through_sink() {
    let sink = Arc::new(MemorySink::new("session-5"));
    let registry = ToolRegistry::with_trace_sink(sink.clone());

    registry.record_event(TraceEvent::InvokeStart {
        tool_id: "x".into(),
        args_size_bytes: 42,
    });

    let events = sink.snapshot();
    assert_eq!(events.len(), 1);
    match &events[0].event {
        TraceEvent::InvokeStart {
            tool_id,
            args_size_bytes,
        } => {
            assert_eq!(tool_id, "x");
            assert_eq!(*args_size_bytes, 42);
        }
        _ => panic!("expected InvokeStart"),
    }
}

#[test]
fn set_trace_sink_swaps_sink() {
    let mut registry = ToolRegistry::new();
    let sink = Arc::new(MemorySink::new("session-6"));
    registry.set_trace_sink(sink.clone());

    registry.record_event(TraceEvent::AuthNeeds {
        upstream: "github".into(),
    });

    assert_eq!(sink.snapshot().len(), 1);
}

#[test]
fn noop_sink_drops_everything() {
    let sink = Arc::new(NoopSink);
    let registry = ToolRegistry::with_trace_sink(sink);
    registry.record_event(TraceEvent::AuthRefresh {
        upstream: "x".into(),
        ok: true,
    });
    // Test that nothing panics; NoopSink has no observable side effect.
}

#[test]
fn jsonl_sink_writes_one_line_per_event() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let sink = Arc::new(JsonlSink::with_source("session-7", "ratel", &path).expect("open sink"));
    sink.record(TraceEvent::AuthNeeds {
        upstream: "github".into(),
    });
    sink.record(TraceEvent::AuthRefresh {
        upstream: "github".into(),
        ok: true,
    });
    drop(sink);

    let body = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 2);

    let first: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["v"], 2);
    assert_eq!(first["event_id"].as_str().unwrap().len(), 26);
    assert_eq!(first["session_id"], "session-7");
    assert_eq!(first["source_id"], "ratel");
    assert_eq!(first["type"], "auth_needs");
    assert_eq!(first["upstream"], "github");
    assert!(first["ts"].as_u64().unwrap() > 0);

    let second: Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(second["type"], "auth_refresh");
    assert_eq!(second["ok"], true);
}

#[test]
fn jsonl_sink_creates_parent_directory() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested").join("subdir").join("trace.jsonl");
    let sink = JsonlSink::new("session-8", &path).expect("open sink in nested dir");
    sink.record(TraceEvent::AuthFlowStart {
        upstream: "x".into(),
    });
    drop(sink);
    assert!(path.exists());
}

/// Lines collected from an [`FnSink`], shared with the closure that fills them.
type CollectedLines = Arc<Mutex<Vec<String>>>;

/// An [`FnSink`] that appends every line it is handed, for the tests below.
/// Returned as `dyn TraceSink` so callers can pass it straight to
/// `with_trace_sink` without naming the closure type.
fn collecting_sink(session_id: &str) -> (Arc<dyn TraceSink>, CollectedLines) {
    let lines: CollectedLines = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let lines = lines.clone();
        Arc::new(FnSink::new(session_id, move |line: &str| {
            lines.lock().expect("lines poisoned").push(line.to_string());
        }))
    };
    (sink, lines)
}

#[test]
fn fn_sink_hands_one_line_per_event() {
    let (sink, lines) = collecting_sink("session-fn-1");
    sink.record(TraceEvent::AuthNeeds {
        upstream: "github".into(),
    });
    sink.record(TraceEvent::AuthRefresh {
        upstream: "github".into(),
        ok: true,
    });

    let lines = lines.lock().unwrap();
    assert_eq!(lines.len(), 2);

    let first: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(first["v"], 2);
    assert_eq!(first["session_id"], "session-fn-1");
    assert_eq!(first["type"], "auth_needs");
    assert_eq!(first["upstream"], "github");
    assert!(first["ts"].as_u64().unwrap() > 0);
    assert!(
        !first["event_id"].as_str().unwrap().is_empty(),
        "envelope v2 stamps a per-event id, same as every other envelope-aware sink"
    );

    let second: Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(second["type"], "auth_refresh");
    assert_eq!(second["ok"], true);
}

/// The contract a host reassembling turns depends on: what the callback hands
/// out is what `JsonlSink` would have written, so `lines.join("\n")` is a valid
/// input to `build_intent_graph` with no re-derivation.
///
/// Two fields legitimately differ, both of them per-record identity rather than
/// content: `ts` is sampled at wrap time, and `event_id` is a fresh ULID minted
/// per event (envelope v2). Neither is read by replay, so normalizing them is
/// what "the same line" means here — every other field must match exactly.
#[test]
fn fn_sink_lines_match_jsonl_modulo_per_record_identity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("trace.jsonl");
    let event = || TraceEvent::Search {
        query: "why is the build broken".into(),
        origin: Origin::Baseline,
        top_k: 0,
        hits: Vec::new(),
        stages: Vec::new(),
        took_ms: 0,
    };

    let jsonl = JsonlSink::new("session-fn-2", &path).expect("open sink");
    jsonl.record(event());
    drop(jsonl);

    let (sink, lines) = collecting_sink("session-fn-2");
    sink.record(event());

    let from_file = std::fs::read_to_string(&path).unwrap();
    let mut expected: Value = serde_json::from_str(from_file.lines().next().unwrap()).unwrap();
    let mut actual: Value = serde_json::from_str(&lines.lock().unwrap()[0]).unwrap();
    for field in ["ts", "event_id"] {
        expected[field] = Value::Null;
        actual[field] = Value::Null;
    }
    assert_eq!(actual, expected);
}

#[test]
fn fn_sink_composes_as_a_registry_sink() {
    let (sink, lines) = collecting_sink("session-fn-3");
    let mut registry = ToolRegistry::with_trace_sink(sink);
    registry.experimental_enable_catalog_definitions();
    registry.register(lookup_tool("alpha"));

    let lines = lines.lock().unwrap();
    // Registration emits index_churn followed by the catalog_definition record.
    assert_eq!(lines.len(), 2);
    let env: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(env["session_id"], "session-fn-3");
    assert_eq!(env["type"], "index_churn");
    let definition: Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(definition["session_id"], "session-fn-3");
    assert_eq!(definition["type"], "catalog_definition");
}

#[test]
fn trace_event_round_trips_through_json() {
    let originals = vec![
        TraceEvent::Search {
            query: "q".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: vec![ratel_ai_core::SearchHitTrace {
                tool_id: "t".into(),
                score: 1.5,
            }],
            stages: vec![ratel_ai_core::SearchStage {
                name: "bm25".into(),
                took_ms: 1,
                top_score: Some(1.5),
            }],
            took_ms: 1,
        },
        TraceEvent::InvokeStart {
            tool_id: "x".into(),
            args_size_bytes: 12,
        },
        TraceEvent::InvokeEnd {
            tool_id: "x".into(),
            took_ms: 7,
        },
        TraceEvent::InvokeError {
            tool_id: "x".into(),
            took_ms: 7,
            error: "boom".into(),
        },
        TraceEvent::GatewaySearch {
            query: "q".into(),
            origin: Origin::Agent,
            top_k: 5,
            hits: 1,
            took_ms: 1,
        },
        TraceEvent::GatewayInvoke {
            tool_id: "x".into(),
            took_ms: 1,
        },
        TraceEvent::GatewayError {
            tool_id: "x".into(),
            error: "boom".into(),
        },
        TraceEvent::UpstreamRegister {
            server: "s".into(),
            transport: "stdio".into(),
            tool_count: 3,
        },
        TraceEvent::UpstreamInvoke {
            server: "s".into(),
            tool_id: "s.t".into(),
            took_ms: 1,
        },
        TraceEvent::UpstreamError {
            server: "s".into(),
            tool_id: "s.t".into(),
            error: "boom".into(),
        },
        TraceEvent::AuthRefresh {
            upstream: "u".into(),
            ok: false,
        },
        TraceEvent::AuthNeeds {
            upstream: "u".into(),
        },
        TraceEvent::AuthFlowStart {
            upstream: "u".into(),
        },
        TraceEvent::AuthFlowEnd {
            upstream: "u".into(),
            ok: true,
        },
        TraceEvent::EventsDropped {
            dropped_count: 2,
            reason: "queue_overflow".into(),
            window_start_ts: 10,
            window_end_ts: 11,
        },
        TraceEvent::IndexChurn {
            kind: ChurnKind::Add,
            tool_id: "t".into(),
        },
    ];

    for original in originals {
        let serialized = serde_json::to_string(&original).expect("serialize");
        let back: TraceEvent = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(back, original);
    }
}

#[test]
fn a_usage_boost_written_before_dropped_existed_still_replays() {
    // Logs outlive builds: an envelope recorded by an older core has no
    // `dropped` field, and refusing to parse it would make the trace log
    // unreplayable across an upgrade (ADR-0007 additive-field rule).
    let line = r#"{"v":1,"ts":1,"session_id":"s","type":"usage_boost",
        "intent":"intent_0","similarity":0.9,"support":3,"promoted":2}"#;
    let env: ratel_ai_core::TraceEnvelope = serde_json::from_str(line).expect("older line parses");
    match env.event {
        ratel_ai_core::TraceEvent::UsageBoost {
            promoted, dropped, ..
        } => {
            assert_eq!(promoted, 2);
            assert_eq!(dropped, 0, "absent means none were dropped");
        }
        other => panic!("expected UsageBoost, got {other:?}"),
    }
}

#[test]
fn the_baseline_origin_round_trips_through_the_wire_form() {
    // Baseline capture records the turn's query while Ratel serves nothing, so
    // the origin has to survive the JSONL round trip that offline graph
    // construction reads back.
    let sink = Arc::new(MemorySink::new("session-baseline"));
    let mut registry = ToolRegistry::with_trace_sink(sink.clone());
    registry.register(lookup_tool("alpha"));

    let _ = registry.search_with_origin("lookup", 3, Origin::Baseline);

    let envelope = sink
        .snapshot()
        .into_iter()
        .find(|e| matches!(e.event, TraceEvent::Search { .. }))
        .expect("expected a search event");

    let json = serde_json::to_string(&envelope).expect("serializes");
    assert!(
        json.contains(r#""origin":"baseline""#),
        "wire value must be `baseline`, got {json}"
    );

    let back: ratel_ai_core::TraceEnvelope = serde_json::from_str(&json).expect("parses");
    match back.event {
        TraceEvent::Search { origin, .. } => assert_eq!(origin, Origin::Baseline),
        other => panic!("expected Search, got {other:?}"),
    }
}
