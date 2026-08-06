use std::sync::{Arc, Mutex};

use ratel_ai_core::{
    ChurnKind, FnSink, JsonlSink, MemorySink, NoopSink, Origin, Tool, ToolRegistry, TraceEnvelope,
    TraceEvent, TraceSink,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn empty_schema() -> Value {
    json!({})
}

fn lookup_tool(id: &str) -> Tool {
    Tool {
        id: id.into(),
        name: id.into(),
        description: "lookup".into(),
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
    let sink = Arc::new(MemorySink::new("session-1"));
    let mut registry = ToolRegistry::with_trace_sink(sink.clone());
    registry.register(lookup_tool("alpha"));

    let events = sink.snapshot();
    assert_eq!(events.len(), 1);
    let env: &TraceEnvelope = &events[0];
    assert_eq!(env.session_id, "session-1");
    assert_eq!(env.v, 1);
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
    let sink = Arc::new(JsonlSink::new("session-7", &path).expect("open sink"));
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
    assert_eq!(first["v"], 1);
    assert_eq!(first["session_id"], "session-7");
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
    assert_eq!(first["v"], 1);
    assert_eq!(first["session_id"], "session-fn-1");
    assert_eq!(first["type"], "auth_needs");
    assert_eq!(first["upstream"], "github");
    assert!(first["ts"].as_u64().unwrap() > 0);

    let second: Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(second["type"], "auth_refresh");
    assert_eq!(second["ok"], true);
}

/// The contract a host reassembling turns depends on: what the callback hands
/// out is what `JsonlSink` would have written, so `lines.join("\n")` is a valid
/// input to `build_intent_graph` with no re-derivation. `ts` is sampled per
/// record, so it is the one field that legitimately differs.
#[test]
fn fn_sink_lines_match_jsonl_modulo_timestamp() {
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
    expected["ts"] = Value::Null;
    actual["ts"] = Value::Null;
    assert_eq!(actual, expected);
}

#[test]
fn fn_sink_composes_as_a_registry_sink() {
    let (sink, lines) = collecting_sink("session-fn-3");
    let mut registry = ToolRegistry::with_trace_sink(sink);
    registry.register(lookup_tool("alpha"));

    let lines = lines.lock().unwrap();
    assert_eq!(lines.len(), 1);
    let env: Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(env["session_id"], "session-fn-3");
    assert_eq!(env["type"], "index_churn");
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
