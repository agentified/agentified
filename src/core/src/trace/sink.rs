use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::trace::event::{TraceEnvelope, TraceEvent};

const ENVELOPE_VERSION: u32 = 1;

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
    session_id: String,
    events: Mutex<Vec<TraceEnvelope>>,
}

impl MemorySink {
    /// An empty sink whose envelopes are stamped with `session_id`.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
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
        &self.session_id
    }
}

impl TraceSink for MemorySink {
    fn record(&self, event: TraceEvent) {
        let envelope = wrap(self.session_id.clone(), event);
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
    session_id: String,
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
            session_id: session_id.into(),
            file: Mutex::new(BufWriter::new(file)),
        })
    }
}

impl TraceSink for JsonlSink {
    fn record(&self, event: TraceEvent) {
        let envelope = wrap(self.session_id.clone(), event);
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

/// A sink that hands each enveloped event to a closure, for hosts whose trace
/// destination this crate cannot own — a process-per-request server writing to
/// a database, a language binding forwarding to its runtime, anything
/// distributed enough that a local file is the wrong answer.
///
/// The closure receives the **serialized envelope line**, byte-for-byte what
/// [`JsonlSink`] would have written for the same event and session. That
/// identity is the point: a host can collect lines from many processes, join
/// them with newlines, and feed the result straight back to
/// [`crate::ToolRegistry::build_intent_graph`] without re-deriving the wire
/// form. It also keeps language bindings trivial — they forward a string
/// rather than re-modelling [`TraceEnvelope`].
///
/// The session id stamped here is a **default, not an identity**: a host that
/// reassembles a turn from its own storage generally knows a better one (per
/// turn, or per client/actor/workspace unit) and may restamp the line before
/// persisting it. Replay tracks one pending query per session in log order, so
/// a shared id is only a problem once lines from several producers are merged
/// without preserving each turn's search-then-invoke adjacency.
///
/// Best-effort like every sink: a serialization failure drops the event. A
/// closure that panics is the host's bug and unwinds normally — keep it cheap
/// and non-blocking, per [`TraceSink::record`].
pub struct FnSink<F: Fn(&str) + Send + Sync> {
    session_id: String,
    emit: F,
}

impl<F: Fn(&str) + Send + Sync> FnSink<F> {
    /// A sink whose envelopes are stamped with `session_id` and handed to
    /// `emit` as JSON lines.
    pub fn new(session_id: impl Into<String>, emit: F) -> Self {
        Self {
            session_id: session_id.into(),
            emit,
        }
    }
}

impl<F: Fn(&str) + Send + Sync> TraceSink for FnSink<F> {
    fn record(&self, event: TraceEvent) {
        let envelope = wrap(self.session_id.clone(), event);
        let Ok(line) = serde_json::to_string(&envelope) else {
            return;
        };
        (self.emit)(&line);
    }
}

fn wrap(session_id: String, event: TraceEvent) -> TraceEnvelope {
    TraceEnvelope {
        v: ENVELOPE_VERSION,
        ts: now_ms(),
        session_id,
        event,
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
