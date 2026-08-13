use std::sync::{Arc, Condvar, Mutex, RwLock, mpsc};

use ratel_ai_core::{
    ChurnKind, FanoutSink, IntentGraph, JsonlSink, MemorySink, NoopSink, Origin, Tool,
    ToolRegistry, TraceEnvelope, TraceEvent, TraceEventContext, TraceSink, UsageLearner,
};
use serde_json::{Value, json};
use tempfile::tempdir;

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
