use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::trace::event::{TraceEnvelope, TraceEvent, TraceEventContext};

const DEFAULT_SOURCE_ID: &str = "ratel";
const ENVELOPE_VERSION: u32 = 2;

struct EnvelopeFactory {
    session_id: String,
    source_id: String,
    pending_invocations: Mutex<HashMap<String, VecDeque<String>>>,
}

impl EnvelopeFactory {
    fn new(session_id: impl Into<String>, source_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            source_id: source_id.into(),
            pending_invocations: Mutex::new(HashMap::new()),
        }
    }

    fn wrap(&self, event: TraceEvent, mut context: TraceEventContext) -> TraceEnvelope {
        self.correlate_invocation(&event, &mut context);
        TraceEnvelope {
            v: ENVELOPE_VERSION,
            event_id: ulid::Ulid::new().to_string(),
            ts: now_ms(),
            session_id: self.session_id.clone(),
            source_id: self.source_id.clone(),
            invocation_id: context.invocation_id,
            catalog_version: context.catalog_version,
            environment: context.environment,
            end_user_id: context.end_user_id,
            trace_id: context.trace_id,
            span_id: context.span_id,
            event,
        }
    }

    fn correlate_invocation(&self, event: &TraceEvent, context: &mut TraceEventContext) {
        match event {
            TraceEvent::InvokeStart { tool_id, .. } => {
                let invocation_id = context.invocation_id.get_or_insert_with(new_ulid).clone();
                if let Ok(mut pending) = self.pending_invocations.lock() {
                    pending
                        .entry(tool_id.clone())
                        .or_default()
                        .push_back(invocation_id);
                }
            }
            TraceEvent::InvokeEnd { tool_id, .. } | TraceEvent::InvokeError { tool_id, .. } => {
                if context.invocation_id.is_none() {
                    context.invocation_id =
                        self.take_invocation(tool_id).or_else(|| Some(new_ulid()));
                } else {
                    self.remove_invocation(tool_id, context.invocation_id.as_deref());
                }
            }
            TraceEvent::SkillInvoke { .. }
            | TraceEvent::GatewayInvoke { .. }
            | TraceEvent::GatewayError { .. }
            | TraceEvent::UpstreamInvoke { .. }
            | TraceEvent::UpstreamError { .. } => {
                context.invocation_id.get_or_insert_with(new_ulid);
            }
            _ => {}
        }
    }

    fn take_invocation(&self, tool_id: &str) -> Option<String> {
        let mut pending = self.pending_invocations.lock().ok()?;
        let ids = pending.get_mut(tool_id)?;
        let invocation_id = ids.pop_front();
        if ids.is_empty() {
            pending.remove(tool_id);
        }
        invocation_id
    }

    fn remove_invocation(&self, tool_id: &str, invocation_id: Option<&str>) {
        let Some(invocation_id) = invocation_id else {
            return;
        };
        let Ok(mut pending) = self.pending_invocations.lock() else {
            return;
        };
        let Some(ids) = pending.get_mut(tool_id) else {
            return;
        };
        ids.retain(|id| id != invocation_id);
        if ids.is_empty() {
            pending.remove(tool_id);
        }
    }
}

/// A best-effort sink for trace events. Implementations must be cheap on the
/// hot path — see ADR-0007 for the query-log reliability profile (lossy on
/// backpressure is fine, blocking the agent loop is not).
///
/// Three implementations ship with the crate: [`NoopSink`] (discard — the
/// registries' default), [`MemorySink`] (in-memory buffer for tests and
/// introspection), and [`JsonlSink`] (append-to-file local persistence).
pub trait TraceSink: Send + Sync {
    /// Record one event. Called synchronously on the hot path, so it must be
    /// cheap and non-blocking; on failure, drop the event rather than
    /// propagate (trace events are observations, never load-bearing).
    fn record(&self, event: TraceEvent);

    /// Record an event with correlation fields known at the emission site.
    /// Legacy sinks may ignore the context; envelope-aware sinks preserve it.
    fn record_with_context(&self, event: TraceEvent, _context: TraceEventContext) {
        self.record(event);
    }

    /// Accept an event already wrapped by an upstream fan-out sink. Envelope-aware
    /// sinks override this so identity survives fan-out; legacy sinks still see
    /// the underlying event.
    fn record_envelope(&self, envelope: TraceEnvelope) {
        self.record(envelope.event);
    }

    /// Per-sink rate limit hint. Currently a documentation knob — nothing
    /// rate-limits yet — but the contract is in place so consumers can adopt
    /// it without a breaking change.
    fn sample_rate(&self) -> f64 {
        1.0
    }
}

/// A sink that discards every event — the default of a registry built with
/// [`crate::ToolRegistry::new`] / [`crate::SkillRegistry::new`], and the
/// right choice when tracing is off.
pub struct NoopSink;

impl TraceSink for NoopSink {
    fn record(&self, _event: TraceEvent) {}
}

/// A sink that buffers enveloped events in memory, for tests and in-process
/// introspection: record, then assert on [`Self::snapshot`] or
/// [`Self::drain`]. The buffer is unbounded, so drain it periodically if the
/// producer is long-lived.
pub struct MemorySink {
    factory: EnvelopeFactory,
    events: Mutex<Vec<TraceEnvelope>>,
}

impl MemorySink {
    /// An empty sink whose envelopes are stamped with `session_id`.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_source(session_id, DEFAULT_SOURCE_ID)
    }

    /// An empty sink with explicit stable `source_id` identity.
    pub fn with_source(session_id: impl Into<String>, source_id: impl Into<String>) -> Self {
        Self {
            factory: EnvelopeFactory::new(session_id, source_id),
            events: Mutex::new(Vec::new()),
        }
    }

    /// A copy of the recorded envelopes, oldest first, leaving the buffer in
    /// place.
    pub fn snapshot(&self) -> Vec<TraceEnvelope> {
        self.events.lock().expect("trace sink poisoned").clone()
    }

    /// Remove and return the recorded envelopes, oldest first, emptying the
    /// buffer.
    pub fn drain(&self) -> Vec<TraceEnvelope> {
        let mut guard = self.events.lock().expect("trace sink poisoned");
        std::mem::take(&mut *guard)
    }

    /// The session id stamped on every envelope this sink records.
    pub fn session_id(&self) -> &str {
        &self.factory.session_id
    }
}

impl TraceSink for MemorySink {
    fn record(&self, event: TraceEvent) {
        self.record_with_context(event, TraceEventContext::default());
    }

    fn record_with_context(&self, event: TraceEvent, context: TraceEventContext) {
        let envelope = self.factory.wrap(event, context);
        self.record_envelope(envelope);
    }

    fn record_envelope(&self, envelope: TraceEnvelope) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(envelope);
        }
    }
}

/// A sink that appends events to a JSONL file, one [`TraceEnvelope`] per
/// line — local persistence for the offline inspector and reporting
/// (ADR-0007; the consuming shells bucket files under `~/.ratel/telemetry/`,
/// but the sink accepts any path). Writes are best-effort: a serialization or
/// I/O failure drops the event rather than disturb the agent loop.
pub struct JsonlSink {
    factory: EnvelopeFactory,
    file: Mutex<BufWriter<File>>,
}

impl JsonlSink {
    /// Open (or create) the JSONL file at `path` in append mode, creating
    /// missing parent directories. On Unix the file's permissions are
    /// tightened to `0600` (best-effort) since traces can carry query text.
    ///
    /// # Errors
    ///
    /// Any [`std::io::Error`] from creating the parent directories or opening
    /// the file.
    pub fn new(session_id: impl Into<String>, path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::with_source(session_id, DEFAULT_SOURCE_ID, path)
    }

    /// Open a JSONL sink with explicit stable `source_id` identity.
    pub fn with_source(
        session_id: impl Into<String>,
        source_id: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(Self {
            factory: EnvelopeFactory::new(session_id, source_id),
            file: Mutex::new(BufWriter::new(file)),
        })
    }
}

impl TraceSink for JsonlSink {
    fn record(&self, event: TraceEvent) {
        self.record_with_context(event, TraceEventContext::default());
    }

    fn record_with_context(&self, event: TraceEvent, context: TraceEventContext) {
        let envelope = self.factory.wrap(event, context);
        self.record_envelope(envelope);
    }

    fn record_envelope(&self, envelope: TraceEnvelope) {
        let Ok(line) = serde_json::to_string(&envelope) else {
            return;
        };
        if let Ok(mut guard) = self.file.lock() {
            // Best-effort: a write failure should not crash the agent loop.
            let _ = writeln!(guard, "{line}");
            let _ = guard.flush();
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn new_ulid() -> String {
    ulid::Ulid::new().to_string()
}
