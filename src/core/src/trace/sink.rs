use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::trace::event::{TraceEnvelope, TraceEvent, TraceEventContext};

const DEFAULT_SOURCE_ID: &str = "ratel";
const ENVELOPE_VERSION: u32 = 2;
const MAX_PENDING_INVOCATIONS_PER_TOOL: usize = 1_024;
const QUEUE_OVERFLOW: &str = "queue_overflow";

/// A handle to one [`FanoutSink`] subscriber.
#[must_use = "dropping the handle unsubscribes the sink"]
pub struct FanoutSubscription {
    id: u64,
    inner: Arc<Subscriber>,
    owner: Weak<FanoutInner>,
}

/// A sink that wraps each event once, then asynchronously dispatches the same
/// envelope to any number of bounded subscribers.
#[derive(Clone)]
pub struct FanoutSink {
    inner: Arc<FanoutInner>,
}

struct FanoutInner {
    factory: Arc<EnvelopeFactory>,
    subscribers: Mutex<HashMap<u64, Arc<Subscriber>>>,
    dropped: AtomicU64,
    next_id: AtomicU64,
}

struct Subscriber {
    capacity: usize,
    dropped: AtomicU64,
    sink: Arc<dyn TraceSink>,
    state: Mutex<SubscriberState>,
    changed: Condvar,
}

#[derive(Default)]
struct SubscriberState {
    queue: VecDeque<TraceEnvelope>,
    pending_loss: Option<DropWindow>,
    delivering: bool,
    closed: bool,
}

struct DropWindow {
    count: u64,
    start_ts: u64,
    end_ts: u64,
}

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
            event_id: context.event_id.take().unwrap_or_else(new_ulid),
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
        // Explicit context is the concurrency-safe path. The pending queues keep
        // legacy sequential record(event) callers conformant until higher layers
        // can carry context; same-tool calls that may finish out of order must use
        // TraceEventContext::new_invocation().
        match event {
            TraceEvent::InvokeStart { tool_id, .. } => {
                let has_explicit_invocation = context.invocation_id.is_some();
                let invocation_id = context.invocation_id.get_or_insert_with(new_ulid).clone();
                if has_explicit_invocation {
                    return;
                }
                if let Ok(mut pending) = self.pending_invocations.lock() {
                    let ids = pending.entry(tool_id.clone()).or_default();
                    if ids.len() == MAX_PENDING_INVOCATIONS_PER_TOOL {
                        ids.pop_front();
                    }
                    ids.push_back(invocation_id);
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
/// Five implementations ship with the crate: [`NoopSink`] (discard — the
/// registries' default), [`FanoutSink`] (bounded asynchronous subscribers),
/// [`MemorySink`] (unbounded in-memory buffer for tests and introspection),
/// [`JsonlSink`] (append-to-file local persistence), and [`FnSink`] (hands each
/// envelope line to a closure, for hosts this crate cannot write to itself).
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

impl FanoutSink {
    /// Create an empty fan-out whose source defaults to `OTEL_SERVICE_NAME`,
    /// falling back to `ratel`.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_source(session_id, default_source_id())
    }

    /// Create an empty fan-out for one session and explicit stable source.
    pub fn with_source(session_id: impl Into<String>, source_id: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(FanoutInner {
                factory: Arc::new(EnvelopeFactory::new(session_id, source_id)),
                subscribers: Mutex::new(HashMap::new()),
                dropped: AtomicU64::new(0),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Add an asynchronously dispatched subscriber with a bounded queue.
    /// Zero capacity is treated as one so emission always has a usable slot.
    pub fn subscribe(&self, sink: Arc<dyn TraceSink>, queue_capacity: usize) -> FanoutSubscription {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let subscriber = Arc::new(Subscriber {
            capacity: queue_capacity.max(1),
            dropped: AtomicU64::new(0),
            sink,
            state: Mutex::new(SubscriberState::default()),
            changed: Condvar::new(),
        });
        self.inner
            .subscribers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, subscriber.clone());
        spawn_dispatcher(subscriber.clone(), self.inner.factory.clone());
        FanoutSubscription {
            id,
            inner: subscriber,
            owner: Arc::downgrade(&self.inner),
        }
    }

    /// Wait until every current subscriber finishes all accepted work.
    pub fn flush(&self) {
        let subscribers: Vec<_> = self
            .inner
            .subscribers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect();
        for subscriber in subscribers {
            subscriber.flush();
        }
    }

    /// Total events dropped across every subscriber since this fan-out was created.
    /// This monotonic counter is the core seam for an OpenTelemetry counter projection.
    pub fn dropped_count(&self) -> u64 {
        self.inner.dropped.load(Ordering::Relaxed)
    }
}

impl TraceSink for FanoutSink {
    fn record(&self, event: TraceEvent) {
        self.record_with_context(event, TraceEventContext::default());
    }

    fn record_with_context(&self, event: TraceEvent, context: TraceEventContext) {
        let envelope = self.inner.factory.wrap(event, context);
        self.record_envelope(envelope);
    }

    fn record_envelope(&self, envelope: TraceEnvelope) {
        let subscribers: Vec<_> = self
            .inner
            .subscribers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect();
        for subscriber in subscribers {
            if subscriber.enqueue(envelope.clone()) {
                self.inner.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl FanoutSubscription {
    /// Total events dropped from this subscriber's queue since subscription.
    pub fn dropped_count(&self) -> u64 {
        self.inner.dropped.load(Ordering::Relaxed)
    }

    /// Wait until this subscriber finishes accepted work and loss reports.
    pub fn flush(&self) {
        self.inner.flush();
    }
}

impl Drop for FanoutSubscription {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade()
            && let Ok(mut subscribers) = owner.subscribers.lock()
        {
            subscribers.remove(&self.id);
        }
        self.inner.close();
    }
}

impl Drop for FanoutInner {
    fn drop(&mut self) {
        let subscribers = self
            .subscribers
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for subscriber in subscribers.values() {
            subscriber.close();
        }
    }
}

impl Subscriber {
    fn enqueue(&self, envelope: TraceEnvelope) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.closed {
            return false;
        }
        let dropped = state.queue.len() == self.capacity;
        if dropped {
            state.queue.pop_front();
            let dropped_at = now_ms();
            let loss = state.pending_loss.get_or_insert(DropWindow {
                count: 0,
                start_ts: dropped_at,
                end_ts: dropped_at,
            });
            loss.count += 1;
            loss.end_ts = dropped_at;
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        state.queue.push_back(envelope);
        self.changed.notify_one();
        dropped
    }

    fn flush(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !state.queue.is_empty() || state.pending_loss.is_some() || state.delivering {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
            self.changed.notify_all();
        }
    }
}

fn spawn_dispatcher(subscriber: Arc<Subscriber>, factory: Arc<EnvelopeFactory>) {
    thread::spawn(move || dispatch(subscriber, factory));
}

fn dispatch(subscriber: Arc<Subscriber>, factory: Arc<EnvelopeFactory>) {
    loop {
        let envelope = {
            let mut state = subscriber
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while state.queue.is_empty() && state.pending_loss.is_none() && !state.closed {
                state = subscriber
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            let envelope = if let Some(loss) = state.pending_loss.take() {
                factory.wrap(
                    TraceEvent::EventsDropped {
                        dropped_count: loss.count,
                        reason: QUEUE_OVERFLOW.into(),
                        window_start_ts: loss.start_ts,
                        window_end_ts: loss.end_ts,
                    },
                    TraceEventContext::default(),
                )
            } else if let Some(envelope) = state.queue.pop_front() {
                envelope
            } else {
                return;
            };
            state.delivering = true;
            envelope
        };

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            subscriber.sink.record_envelope(envelope);
        }));
        if let Ok(mut state) = subscriber.state.lock() {
            state.delivering = false;
            subscriber.changed.notify_all();
        }
    }
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
    /// An empty sink stamped with `session_id` and a source defaulting to
    /// `OTEL_SERVICE_NAME`, falling back to `ratel`.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::with_source(session_id, default_source_id())
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
    /// missing parent directories. The source defaults to `OTEL_SERVICE_NAME`,
    /// falling back to `ratel`. On Unix the file's permissions are
    /// tightened to `0600` (best-effort) since traces can carry query text.
    ///
    /// # Errors
    ///
    /// Any [`std::io::Error`] from creating the parent directories or opening
    /// the file.
    pub fn new(session_id: impl Into<String>, path: impl AsRef<Path>) -> std::io::Result<Self> {
        Self::with_source(session_id, default_source_id(), path)
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

/// A sink that hands each enveloped event to a closure, for hosts whose trace
/// destination this crate cannot own — a process-per-request server writing to
/// a database, a language binding forwarding to its runtime, anything
/// distributed enough that a local file is the wrong answer.
///
/// The closure receives the **serialized envelope line** — the same wire form
/// [`JsonlSink`] would have written for the same event, session, and source,
/// field for field, differing only in the two per-record identity fields every
/// envelope-aware sink mints for itself (`ts`, sampled at wrap time, and
/// `event_id`, a fresh ULID). Replay reads neither. That identity is the point:
/// a host can collect lines from many processes, join them with newlines, and
/// feed the result straight back to
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
    factory: EnvelopeFactory,
    emit: F,
}

impl<F: Fn(&str) + Send + Sync> FnSink<F> {
    /// A sink whose envelopes are stamped with `session_id` and a source
    /// defaulting to `OTEL_SERVICE_NAME` (falling back to `ratel`), handed to
    /// `emit` as JSON lines.
    pub fn new(session_id: impl Into<String>, emit: F) -> Self {
        Self::with_source(session_id, default_source_id(), emit)
    }

    /// A sink with explicit stable `source_id` identity.
    pub fn with_source(
        session_id: impl Into<String>,
        source_id: impl Into<String>,
        emit: F,
    ) -> Self {
        Self {
            factory: EnvelopeFactory::new(session_id, source_id),
            emit,
        }
    }
}

impl<F: Fn(&str) + Send + Sync> TraceSink for FnSink<F> {
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
        (self.emit)(&line);
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

fn default_source_id() -> String {
    std::env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_SOURCE_ID.into())
}
