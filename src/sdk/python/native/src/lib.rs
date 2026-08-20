//! PyO3 binding to `ratel-ai-core` — the Python analogue of the TS SDK's NAPI
//! binding (`src/sdk/ts/native/src/lib.rs`). Pure pass-through over the public
//! API of `ratel-ai-core`; see ADR-0006 for the binding-strategy rationale and
//! ADR-0007 for the core-owned trace schema this emits into.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use ratel_ai_core as core;
use ratel_ai_core::{
    ArtifactError as CoreArtifactError, ArtifactWarmError as CoreArtifactWarmError, JsonlSink,
    MemorySink, NoopSink, OnArtifactMiss, Origin, TraceEnvelope, TraceEvent, UsageLearner,
};
use serde_json::Value;

/// Map a `SearchOrigin` wire string to its core variant.
///
/// **One function, called from every site.** Each search entry point used to
/// inline this `match`, which meant a new `Origin` variant was silently
/// swallowed by the `_ =>` fallback at whichever sites nobody remembered to
/// update — a wrong-but-plausible `direct` rather than a compile error. Adding
/// a variant is still not a compile error here (unknown strings must stay
/// tolerated, since `search_with_origin` is infallible), so the guard is
/// `every_origin_wire_value_maps_to_its_own_variant` below, not the match.
fn parse_origin(s: &str) -> Origin {
    match s {
        "agent" => Origin::Agent,
        "baseline" => Origin::Baseline,
        _ => Origin::Direct,
    }
}

/// Resolve the observation-policy strings, **rejecting** unknown values.
///
/// Deliberately stricter than [`parse_origin`], which tolerates an unknown wire
/// string so version skew cannot fail an infallible search. A policy is a
/// deliberate configuration a caller typed: silently reading `"seedd"` as
/// `"live"` would produce a graph with no provenance and no error.
/// Defaults differ per entry point: building from a log IS a seeding pass, so
/// it defaults to `baseline` + `seeded`; enabling live learning keeps `any` +
/// `live`, which is what it has always done.
fn parse_policy(
    origins: Option<&str>,
    provenance: Option<&str>,
    default_origins: core::OriginFilter,
    default_provenance: core::Provenance,
) -> PyResult<core::ObservationPolicy> {
    let mut policy = core::ObservationPolicy::default()
        .with_origins(default_origins)
        .with_provenance(default_provenance);
    if let Some(o) = origins {
        policy = policy.with_origins(match o {
            "any" => core::OriginFilter::Any,
            "agent" => core::OriginFilter::Exactly(Origin::Agent),
            "baseline" => core::OriginFilter::Exactly(Origin::Baseline),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown origins {other:?}: expected any | agent | baseline"
                )));
            }
        });
    }
    if let Some(p) = provenance {
        policy = policy.with_provenance(match p {
            "live" => core::Provenance::Live,
            "seeded" => core::Provenance::Seeded,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown provenance {other:?}: expected live | seeded"
                )));
            }
        });
    }
    Ok(policy)
}

/// Parse a JSONL trace log into envelopes.
///
/// Blank lines are skipped — common and harmless. A malformed line is an error
/// naming its line number rather than a silent skip: a log is the only record of
/// a baseline capture, and quietly dropping part of it would produce a thinner
/// graph with nothing to explain why.
fn parse_trace_log(jsonl: &str) -> PyResult<Vec<core::TraceEnvelope>> {
    jsonl
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(i, line)| {
            serde_json::from_str::<core::TraceEnvelope>(line)
                .map_err(|e| PyValueError::new_err(format!("trace log line {}: {e}", i + 1)))
        })
        .collect()
}

type ToolBatchItem = (String, String, String, Py<PyAny>, Py<PyAny>);
type SkillBatchItem = (
    String,
    String,
    String,
    Vec<String>,
    Vec<String>,
    HashMap<String, Vec<String>>,
    String,
);
// The fact-side twin of `SkillBatchItem`: a fact has no `tools` Vec and adds a
// `pin` string (parsed into `core::PinMode`), so the shape is
// (id, name, description, tags, metadata, body, pin).
type FactBatchItem = (
    String,
    String,
    String,
    Vec<String>,
    HashMap<String, Vec<String>>,
    String,
    String,
);

const DEFAULT_EVENT_QUEUE_CAPACITY: usize = 1_024;
const DEFAULT_EVENT_BATCH_SIZE: usize = 64;
const EVENT_BATCH_DELAY: Duration = Duration::from_millis(1);

fn trace_event_context(context: Option<&Bound<'_, PyAny>>) -> PyResult<core::TraceEventContext> {
    let Some(context) = context else {
        return Ok(core::TraceEventContext::default());
    };
    let value: Value = pythonize::depythonize(context)
        .map_err(|e| PyValueError::new_err(format!("invalid trace event context: {e}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| PyValueError::new_err("trace event context must be a dict"))?;
    let string = |key: &str| {
        object
            .get(key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    };
    Ok(core::TraceEventContext {
        event_id: string("event_id"),
        invocation_id: string("invocation_id"),
        catalog_version: string("catalog_version"),
        environment: string("environment"),
        end_user_id: string("end_user_id"),
        trace_id: string("trace_id"),
        span_id: string("span_id"),
    })
}

/// Handle for one private native runtime-event callback subscription.
#[pyclass]
struct NativeEventSubscription {
    subscription: Arc<Mutex<Option<Arc<core::FanoutSubscription>>>>,
    // Retain progress, not the sink: unsubscribe must release its sender so
    // the callback dispatcher observes a disconnected receiver and exits.
    progress: Arc<EventCallbackProgress>,
    final_dropped_count: AtomicU64,
}

struct EventStream {
    fanout: Arc<core::FanoutSink>,
    session_id: String,
    source_id: Option<String>,
}

/// Records to the pre-existing base sink synchronously on the producer thread,
/// then fans out to bounded push subscribers. Keeping the base sink out of the
/// bounded fan-out queues preserves the lossless local memory/jsonl projection;
/// only push subscribers may drop on overflow. Mirrors the TS binding.
struct RuntimeEventSink {
    base: Arc<dyn core::TraceSink>,
    fanout: Arc<core::FanoutSink>,
}

struct EventCallbackSink {
    sender: SyncSender<TraceEnvelope>,
    progress: Arc<EventCallbackProgress>,
}

struct EventCallbackProgress {
    state: Mutex<EventCallbackState>,
    changed: Condvar,
}

#[derive(Default)]
struct EventCallbackState {
    accepted: u64,
    delivered: u64,
}

#[pymethods]
impl NativeEventSubscription {
    /// Wait without the GIL until accepted callback work has completed.
    fn flush(&self, py: Python<'_>) {
        py.allow_threads(|| {
            let subscription = self
                .subscription
                .lock()
                .ok()
                .and_then(|subscription| subscription.clone());
            if let Some(subscription) = subscription {
                subscription.flush();
            }
            self.progress.flush();
        });
    }

    /// Stop accepting new events. Envelopes already queued still drain.
    fn unsubscribe(&self, py: Python<'_>) {
        py.allow_threads(|| {
            if let Ok(mut subscription) = self.subscription.lock()
                && let Some(subscription) = subscription.take()
            {
                self.final_dropped_count
                    .store(subscription.dropped_count(), Ordering::Relaxed);
            }
        });
    }

    /// Number of envelopes displaced by this subscriber's bounded queue.
    #[getter]
    fn dropped_count(&self, py: Python<'_>) -> u64 {
        py.allow_threads(|| {
            let current = self.subscription.lock().ok().and_then(|subscription| {
                subscription
                    .as_ref()
                    .map(|subscription| subscription.dropped_count())
            });
            current.unwrap_or_else(|| self.final_dropped_count.load(Ordering::Relaxed))
        })
    }
}

impl EventStream {
    fn new(session_id: String, source_id: Option<String>) -> Self {
        let fanout = Arc::new(match source_id.as_ref() {
            Some(source_id) => core::FanoutSink::with_source(&session_id, source_id),
            None => core::FanoutSink::new(&session_id),
        });
        Self {
            fanout,
            session_id,
            source_id,
        }
    }

    fn validate_identity(&self, session_id: &str, source_id: Option<&str>) -> PyResult<()> {
        if self.session_id != session_id || self.source_id.as_deref() != source_id {
            return Err(PyValueError::new_err(
                "runtime event stream already configured with a different session_id/source_id",
            ));
        }
        Ok(())
    }
}

impl core::TraceSink for RuntimeEventSink {
    fn record(&self, event: TraceEvent) {
        self.base.record(event.clone());
        self.fanout.record(event);
    }

    fn record_with_context(&self, event: TraceEvent, context: core::TraceEventContext) {
        self.base
            .record_with_context(event.clone(), context.clone());
        self.fanout.record_with_context(event, context);
    }

    fn record_envelope(&self, envelope: TraceEnvelope) {
        self.base.record_envelope(envelope.clone());
        self.fanout.record_envelope(envelope);
    }
}

impl EventCallbackSink {
    fn new(callback: Py<PyAny>, batch_size: usize) -> Arc<Self> {
        let (sender, receiver) = sync_channel(batch_size);
        let progress = Arc::new(EventCallbackProgress {
            state: Mutex::new(EventCallbackState::default()),
            changed: Condvar::new(),
        });
        let sink = Arc::new(Self {
            sender,
            progress: progress.clone(),
        });
        spawn_event_callback_dispatcher(receiver, callback, batch_size, progress);
        sink
    }
}

impl EventCallbackProgress {
    fn flush(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.delivered < state.accepted {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn mark_delivered(&self, count: usize) {
        if let Ok(mut state) = self.state.lock() {
            state.delivered += count as u64;
            self.changed.notify_all();
        }
    }
}

impl core::TraceSink for EventCallbackSink {
    fn record(&self, _event: TraceEvent) {}

    fn record_envelope(&self, envelope: TraceEnvelope) {
        if let Ok(mut state) = self.progress.state.lock() {
            state.accepted += 1;
        }
        if self.sender.send(envelope).is_err() {
            self.progress.mark_delivered(1);
        }
    }
}

fn subscribe_event_callback(
    stream: &mut Option<EventStream>,
    callback: Py<PyAny>,
    session_id: String,
    source_id: Option<String>,
    queue_capacity: usize,
    batch_size: usize,
) -> PyResult<NativeEventSubscription> {
    let callback_sink = EventCallbackSink::new(callback, batch_size.max(1));
    let stream = match stream {
        Some(stream) => {
            stream.validate_identity(&session_id, source_id.as_deref())?;
            stream
        }
        slot @ None => slot.insert(EventStream::new(session_id, source_id)),
    };
    let subscription = stream
        .fanout
        .subscribe(callback_sink.clone(), queue_capacity.max(1));
    Ok(NativeEventSubscription {
        subscription: Arc::new(Mutex::new(Some(Arc::new(subscription)))),
        progress: callback_sink.progress.clone(),
        final_dropped_count: AtomicU64::new(0),
    })
}

fn spawn_event_callback_dispatcher(
    receiver: Receiver<TraceEnvelope>,
    callback: Py<PyAny>,
    batch_size: usize,
    progress: Arc<EventCallbackProgress>,
) {
    thread::spawn(move || dispatch_event_callbacks(receiver, callback, batch_size, progress));
}

/// Marshal batches from the fan-out subscriber thread onto a dedicated
/// callback thread. This thread alone acquires the GIL and invokes the Python
/// callable; core emission and dense-search workers never run subscriber code.
fn dispatch_event_callbacks(
    receiver: Receiver<TraceEnvelope>,
    callback: Py<PyAny>,
    batch_size: usize,
    progress: Arc<EventCallbackProgress>,
) {
    while let Ok(first) = receiver.recv() {
        let mut envelopes = vec![first];
        while envelopes.len() < batch_size {
            match receiver.recv_timeout(EVENT_BATCH_DELAY) {
                Ok(envelope) => envelopes.push(envelope),
                Err(_) => break,
            }
        }
        let delivered = envelopes.len();
        let _ = Python::with_gil(|py| -> PyResult<()> {
            let batch = PyList::empty(py);
            for envelope in envelopes {
                let Ok(value) = pythonize::pythonize(py, &envelope) else {
                    continue;
                };
                batch.append(value)?;
            }
            callback.call1(py, (batch,))?;
            Ok(())
        });
        progress.mark_delivered(delivered);
    }
}

create_exception!(
    _native,
    EmbedderError,
    PyRuntimeError,
    "Embedding model load / inference failure (subclass of RuntimeError)."
);
create_exception!(
    _native,
    DimensionMismatchError,
    EmbedderError,
    "A query/corpus embedding dimension mismatch — the model changed under an existing set."
);
create_exception!(
    _native,
    ArtifactError,
    PyRuntimeError,
    "Build-time embedding artifact encode/decode/merge failure (subclass of RuntimeError)."
);
create_exception!(
    _native,
    IncompatibleMergeError,
    ArtifactError,
    "Valid RAT1 parts that cannot be merged (header mismatch or duplicate kind+id)."
);
create_exception!(
    _native,
    ArtifactWarmError,
    PyRuntimeError,
    "Warming the dense cache from an embedding artifact failed (subclass of RuntimeError)."
);

/// Map a core embedding error to a typed Python exception (base `EmbedderError`,
/// with `DimensionMismatchError` for the dimension case), keeping `RuntimeError`
/// as the common ancestor so existing `except RuntimeError` still catches them.
fn map_embedder_err(e: core::EmbedderError) -> PyErr {
    let msg = e.to_string();
    match e {
        core::EmbedderError::DimensionMismatch { .. } => DimensionMismatchError::new_err(msg),
        _ => EmbedderError::new_err(msg),
    }
}

fn map_artifact_build_err(e: CoreArtifactError) -> PyErr {
    match e {
        CoreArtifactError::Embedder(inner) => map_embedder_err(inner),
        CoreArtifactError::IncompatibleMerge { .. } => {
            IncompatibleMergeError::new_err(e.to_string())
        }
        other => ArtifactError::new_err(other.to_string()),
    }
}

fn artifact_warm_pyerr(code: &str, message: String, missing: Option<Vec<String>>) -> PyErr {
    Python::with_gil(|py| {
        let err = ArtifactWarmError::new_err(message);
        let value = err.value(py);
        if let Err(attr_err) = value.setattr("code", code) {
            return attr_err;
        }
        match missing {
            Some(ids) => {
                if let Err(attr_err) = value.setattr("missing", ids) {
                    return attr_err;
                }
            }
            None => {
                if let Err(attr_err) = value.setattr("missing", py.None()) {
                    return attr_err;
                }
            }
        };
        err
    })
}

fn map_artifact_warm_err(e: CoreArtifactWarmError) -> PyErr {
    // Compute once from the core enum so the Python exception message cannot drift.
    let message = e.to_string();
    match e {
        CoreArtifactWarmError::Incomplete { missing } => {
            artifact_warm_pyerr("Incomplete", message, Some(missing))
        }
        CoreArtifactWarmError::Warm(_) => artifact_warm_pyerr("Warm", message, None),
        CoreArtifactWarmError::Embedder(_) => artifact_warm_pyerr("Embedder", message, None),
    }
}

fn parse_on_artifact_miss(on_miss: &str) -> PyResult<OnArtifactMiss> {
    on_miss
        .parse::<OnArtifactMiss>()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Merge valid RAT1 parts into one mixed Tool+Skill artifact.
#[pyfunction]
fn merge_embedding_artifacts(py: Python<'_>, parts: Vec<Vec<u8>>) -> PyResult<Py<PyBytes>> {
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    let bytes = core::merge_embedding_artifacts(&refs).map_err(map_artifact_build_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Resolve the flat embedding-config kwargs to a core model, or `None` when none
/// are given (→ built-in default). A config error is a construction-time
/// `ValueError`. Mirrors the TS binding's `resolve_embedding`.
#[allow(clippy::too_many_arguments)]
fn resolve_embedding(
    spec: Option<String>,
    huggingface: Option<String>,
    local: Option<String>,
    ollama: Option<String>,
    url: Option<String>,
    model: Option<String>,
    revision: Option<String>,
    api_key_env: Option<String>,
    query_prefix: Option<String>,
    doc_prefix: Option<String>,
    pooling: Option<String>,
    download: Option<bool>,
) -> PyResult<Option<core::EmbeddingModel>> {
    // No fields given → no override (the default model).
    let all_none = download.is_none()
        && [
            &spec,
            &huggingface,
            &local,
            &ollama,
            &url,
            &model,
            &revision,
            &api_key_env,
            &query_prefix,
            &doc_prefix,
            &pooling,
        ]
        .iter()
        .all(|f| f.is_none());
    if all_none {
        return Ok(None);
    }
    let s = core::EmbeddingSpec {
        spec,
        huggingface,
        local,
        ollama,
        url,
        model,
        revision,
        api_key_env,
        query_prefix,
        doc_prefix,
        pooling,
        download,
    };
    core::EmbeddingModel::resolve(s)
        .map(Some)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// A single search result: the matched tool id and its relevance score. Mirrors
/// the TS SDK's `SearchHit` (camelCase `toolId` there → snake_case `tool_id` here).
#[pyclass(frozen)]
pub struct SearchHit {
    /// Id of the matched tool, as passed to `register`.
    #[pyo3(get)]
    pub tool_id: String,
    /// Relevance score; higher ranks first. Its scale depends on the method
    /// (raw BM25 / cosine / RRF) AND on `fused` — with adaptive ranking a matched
    /// query returns small RRF scores while an unmatched one on the same catalog
    /// returns the raw score. Order by `rank`, branch on `fused`; treat `score`
    /// as a within-list hint only.
    #[pyo3(get)]
    pub score: f64,
    /// 0-based position in this result list (best is `0`). Stable across methods
    /// and across the `fused` switch — the field to order or threshold on.
    #[pyo3(get)]
    pub rank: u32,
    /// `true` when `score` is an RRF score (ordering-only) rather than the raw
    /// method score. Uniform across one result list; lets a caller detect the
    /// scale their `score` is on.
    #[pyo3(get)]
    pub fused: bool,
}

#[pymethods]
impl SearchHit {
    fn __repr__(&self) -> String {
        format!(
            "SearchHit(tool_id={:?}, score={}, rank={}, fused={})",
            self.tool_id, self.score, self.rank, self.fused
        )
    }
}

/// A single skill search result: the matched skill id and its relevance score.
/// The skill analogue of [`SearchHit`] (`tool_id` → `skill_id`).
#[pyclass(frozen)]
pub struct SkillHit {
    /// Id of the matched skill, as passed to `register`.
    #[pyo3(get)]
    pub skill_id: String,
    /// Relevance score; scale depends on the method and on `fused`, as on
    /// [`SearchHit::score`]. Order by `rank`, branch on `fused`.
    #[pyo3(get)]
    pub score: f64,
    /// 0-based position — as on [`SearchHit::rank`].
    #[pyo3(get)]
    pub rank: u32,
    /// `true` when `score` is an RRF score — as on [`SearchHit::fused`].
    #[pyo3(get)]
    pub fused: bool,
}

#[pymethods]
impl SkillHit {
    fn __repr__(&self) -> String {
        format!(
            "SkillHit(skill_id={:?}, score={}, rank={}, fused={})",
            self.skill_id, self.score, self.rank, self.fused
        )
    }
}

/// A single fact search result: the matched fact id and its relevance score.
/// The fact analogue of [`SearchHit`] (`tool_id` → `fact_id`), twin of
/// [`SkillHit`].
#[pyclass(frozen)]
pub struct FactHit {
    /// Id of the matched fact, as passed to `register`.
    #[pyo3(get)]
    pub fact_id: String,
    /// Relevance score; higher ranks first. Same method-dependent scale as
    /// [`SearchHit::score`], computed against the fact corpus.
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl FactHit {
    fn __repr__(&self) -> String {
        format!("FactHit(fact_id={:?}, score={})", self.fact_id, self.score)
    }
}

/// Map the core status enum to the tuple the Python facade renders.
/// Read the cluster policy, rejecting a value outside `(0, 1]` rather than
/// clamping it: a clamp would silently cluster at something the caller did not
/// ask for, and boundaries once drawn are never redrawn.
fn parse_cluster_policy(
    similarity: Option<f64>,
    coverage: Option<f64>,
) -> PyResult<core::ClusterPolicy> {
    let mut policy = core::ClusterPolicy::default();
    if let Some(v) = similarity {
        policy = policy.with_similarity(v as f32);
    }
    if let Some(v) = coverage {
        policy = policy.with_coverage(v as f32);
    }
    if !policy.is_valid() {
        return Err(PyValueError::new_err(format!(
            "cluster_similarity {} / cluster_coverage {}: both must be in (0, 1]",
            policy.similarity, policy.coverage
        )));
    }
    Ok(policy)
}

fn map_adaptive_status(
    s: core::AdaptiveRankingStatus,
) -> (String, Option<String>, Option<String>, Option<bool>) {
    use core::AdaptiveRankingStatus as S;
    match s {
        S::Inactive => ("inactive".into(), None, None, None),
        S::Active => ("active".into(), None, None, None),
        S::Unknown => ("unknown".into(), None, None, None),
        S::Paused {
            dim_mismatch,
            built,
            active,
        } => (
            if dim_mismatch {
                "paused: dim mismatch"
            } else {
                "paused: model mismatch"
            }
            .into(),
            Some(built),
            Some(active),
            Some(dim_mismatch),
        ),
        // Begins "active" on purpose: the arm is still serving, and a caller that
        // recovers from anything "paused" by rebuilding would summon a remedy
        // that cannot revisit cluster boundaries.
        S::PolicyDrift {
            built_similarity,
            built_coverage,
            active_similarity,
            active_coverage,
        } => (
            "active: policy drift".into(),
            Some(format!(
                "similarity {built_similarity} / coverage {built_coverage}"
            )),
            Some(format!(
                "similarity {active_similarity} / coverage {active_coverage}"
            )),
            None,
        ),
        // Non-exhaustive upstream: degrade rather than fail to compile.
        _ => ("unknown".into(), None, None, None),
    }
}

/// Decorate `sink` with a [`UsageLearner`] when adaptive ranking is on, so the
/// graph keeps growing across a sink change, under `policy`. Without this,
/// `set_trace_sink` would quietly stop learning — or resume it at the default
/// policy, silently dropping a configured one.
fn wrap_learner(
    sink: Arc<dyn core::TraceSink>,
    graph: Option<&Arc<RwLock<core::IntentGraph>>>,
    policy: core::ObservationPolicy,
) -> Arc<dyn core::TraceSink> {
    match graph {
        Some(graph) => Arc::new(UsageLearner::with_policy(graph.clone(), sink, policy)),
        None => sink,
    }
}

/// The sink a registry should actually be holding, given its base sink, whether
/// a graph is attached, and whether the runtime-events stream is live.
///
/// **Every path that installs a sink goes through here** — the two decorations
/// compose in a fixed order (learner outermost, fan-out beneath it, base sink
/// innermost) and reconstructing that at each call site is how one of them
/// silently goes missing.
fn active_trace_sink(
    base_sink: &Arc<dyn core::TraceSink>,
    graph: Option<&Arc<RwLock<core::IntentGraph>>>,
    event_stream: &Option<EventStream>,
    policy: core::ObservationPolicy,
) -> Arc<dyn core::TraceSink> {
    let sink: Arc<dyn core::TraceSink> = match event_stream {
        Some(stream) => Arc::new(RuntimeEventSink {
            base: base_sink.clone(),
            fanout: stream.fanout.clone(),
        }),
        None => base_sink.clone(),
    };
    wrap_learner(sink, graph, policy)
}

/// A shared usage-ranking intent graph (ADR-0014): clusters of past queries,
/// each remembering the capabilities invoked after them.
///
/// Hand the **same** instance to a tool catalog and a skill catalog. One cluster
/// carries both a tool and a skill edge map, so sharing gives one set of
/// clusters with all the evidence behind it; separate graphs duplicate every
/// cluster and split the evidence.
#[pyclass]
#[derive(Clone)]
pub struct IntentGraph {
    inner: Arc<RwLock<core::IntentGraph>>,
}

#[pymethods]
impl IntentGraph {
    /// An empty graph — knows nothing until a search is followed by an invoke.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(core::IntentGraph::empty())),
        }
    }

    /// Adopt a graph in the `protocol/v1` wire form — from a previous
    /// `to_json()` or Ratel Cloud. Raises `ValueError` if it is malformed or
    /// declares a schema version this build does not read.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let graph =
            core::IntentGraph::from_json(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(graph)),
        })
    }

    /// Serialize to the `protocol/v1` wire form — for inspection, or to carry
    /// what was learned across processes.
    ///
    /// The graph is in-process only; persistence is yours. It mutates on every
    /// confirmed invoke, so unsaved observations are lost on a crash — persist on
    /// a cadence or at shutdown. Use `rev` to save only when it changed and to
    /// detect a concurrent writer; single-writer is the supported model.
    ///
    /// SENSITIVE: the output contains the raw text of past user queries (the
    /// cluster `members`). Treat a persisted graph like your query/telemetry log
    /// — restrict permissions (`0600`), keep it out of version control and
    /// images, and do not ship it to a less-trusted store.
    fn to_json(&self) -> PyResult<String> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("intent graph lock poisoned"))?;
        serde_json::to_string(&*guard)
            .map_err(|e| PyValueError::new_err(format!("serialize intent graph: {e}")))
    }

    /// How many clusters the graph holds. `0` is the cold-start state, in which
    /// it contributes nothing to ranking.
    #[getter]
    fn cluster_count(&self) -> PyResult<usize> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("intent graph lock poisoned"))?;
        Ok(guard.len())
    }

    /// Monotonic write counter — bumped once per mutation (a confirmed
    /// observation, a rebuild). Never affects ranking; it is a primitive for your
    /// storage layer. Snapshot it after each save: a later value means unsaved
    /// learning (save-when-changed), and a stored graph whose `rev` is higher than
    /// the one you loaded was written by another process (stale-base detection).
    #[getter]
    fn rev(&self) -> PyResult<u64> {
        let guard = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("intent graph lock poisoned"))?;
        Ok(guard.rev())
    }
}

/// Metadata registry over `ratel-ai-core`. BM25 is exposed synchronously;
/// GIL-releasing dense primitives are private to the pure-Python async facade.
/// Executors and capability-tool / MCP layers also live above this binding.
#[pyclass]
pub struct ToolRegistry {
    inner: core::ToolRegistry,
    memory_sink: Option<Arc<MemorySink>>,
    /// The current undecorated sink, whatever its kind. Retained so
    /// enable/disable adaptive ranking can re-wrap or restore it; rebuilding
    /// from `memory_sink` alone would drop a configured jsonl sink to noop.
    base_sink: Arc<dyn core::TraceSink>,
    /// Retained so `set_trace_sink` can re-wrap the new sink in a learner —
    /// otherwise changing sinks would silently switch learning off.
    graph: Option<Arc<RwLock<core::IntentGraph>>>,
    /// The policy the attached learner runs under. Retained beside `graph` for
    /// the same reason: a sink change re-wraps the learner, and rebuilding it at
    /// the default would silently drop a configured policy.
    usage_policy: core::ObservationPolicy,
    event_stream: Option<EventStream>,
}

#[pymethods]
impl ToolRegistry {
    /// Construct a registry. The optional embedding kwargs select the
    /// semantic/hybrid model (default bge-small when none given); an invalid
    /// config raises `ValueError` here, at construction.
    #[new]
    #[pyo3(signature = (spec=None, huggingface=None, local=None, ollama=None, url=None, model=None, revision=None, api_key_env=None, query_prefix=None, doc_prefix=None, pooling=None, download=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        spec: Option<String>,
        huggingface: Option<String>,
        local: Option<String>,
        ollama: Option<String>,
        url: Option<String>,
        model: Option<String>,
        revision: Option<String>,
        api_key_env: Option<String>,
        query_prefix: Option<String>,
        doc_prefix: Option<String>,
        pooling: Option<String>,
        download: Option<bool>,
    ) -> PyResult<Self> {
        let inner = match resolve_embedding(
            spec,
            huggingface,
            local,
            ollama,
            url,
            model,
            revision,
            api_key_env,
            query_prefix,
            doc_prefix,
            pooling,
            download,
        )? {
            Some(model) => core::ToolRegistry::with_embedding(model),
            None => core::ToolRegistry::new(),
        };
        Ok(Self {
            inner,
            memory_sink: None,
            base_sink: Arc::new(NoopSink),
            graph: None,
            usage_policy: core::ObservationPolicy::default(),
            event_stream: None,
        })
    }

    /// Register a tool's metadata into the index (or replace it in place when
    /// `id` is already registered). The schemas must be JSON-serializable dicts;
    /// anything else raises `ValueError`.
    fn register(
        &mut self,
        id: String,
        name: String,
        description: String,
        input_schema: &Bound<'_, PyAny>,
        output_schema: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let input_schema: Value = pythonize::depythonize(input_schema)
            .map_err(|e| PyValueError::new_err(format!("invalid input_schema: {e}")))?;
        let output_schema: Value = pythonize::depythonize(output_schema)
            .map_err(|e| PyValueError::new_err(format!("invalid output_schema: {e}")))?;
        self.inner.register(core::Tool {
            id,
            name,
            description,
            input_schema,
            output_schema,
        });
        Ok(())
    }

    /// Convert the complete batch before mutating the core registry, so a bad
    /// schema in a later item cannot leave earlier items partially registered.
    fn _register_many(&mut self, py: Python<'_>, tools: Vec<ToolBatchItem>) -> PyResult<()> {
        let tools = tools
            .into_iter()
            .map(|(id, name, description, input_schema, output_schema)| {
                let input_schema: Value = pythonize::depythonize(input_schema.bind(py))
                    .map_err(|e| PyValueError::new_err(format!("invalid input_schema: {e}")))?;
                let output_schema: Value = pythonize::depythonize(output_schema.bind(py))
                    .map_err(|e| PyValueError::new_err(format!("invalid output_schema: {e}")))?;
                Ok(core::Tool {
                    id,
                    name,
                    description,
                    input_schema,
                    output_schema,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        for tool in tools {
            self.inner.register(tool);
        }
        Ok(())
    }

    /// Lexical BM25 search: the top `top_k` tools for `query`, best first.
    /// Model-free and infallible; the trace event records origin `"direct"`.
    fn search(&self, query: String, top_k: u32) -> Vec<SearchHit> {
        self.inner
            .search(&query, top_k as usize)
            .into_iter()
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect()
    }

    /// BM25 search tagged with who initiated it: `"agent"` (a model calling a
    /// capability tool) or anything else → `"direct"` (host code). The origin
    /// only labels the emitted trace event — ranking is identical to `search`.
    #[pyo3(signature = (query, top_k, origin, context=None))]
    fn search_with_origin(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<SearchHit>> {
        let parsed = parse_origin(origin.as_str());
        let hits = self
            .inner
            .search_with_method_and_context(
                &query,
                top_k as usize,
                parsed,
                core::SearchMethod::Bm25,
                trace_event_context(context)?,
            )
            .map_err(map_embedder_err)?;
        Ok(hits
            .into_iter()
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect())
    }

    /// Search with an explicit method (`"bm25"` | `"semantic"` | `"hybrid"`).
    /// `bm25` is infallible; `semantic`/`hybrid` rank against the prebuilt embedding
    /// cache and raise `RuntimeError` (`EmbeddingsNotBuilt`) if it isn't built — the
    /// model loads at `_build_embeddings`, never inside a search. Private worker
    /// primitive; releases the GIL. An unknown method raises `ValueError`.
    #[pyo3(signature = (query, top_k, origin, method, context=None))]
    fn _search_with_method(
        &self,
        py: Python<'_>,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<SearchHit>> {
        let parsed_origin = parse_origin(origin.as_str());
        let parsed_method = method
            .parse::<core::SearchMethod>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let context = trace_event_context(context)?;
        let hits = py
            .allow_threads(|| {
                self.inner.search_with_method_and_context(
                    &query,
                    top_k as usize,
                    parsed_origin,
                    parsed_method,
                    context,
                )
            })
            .map_err(map_embedder_err)?;
        Ok(hits
            .into_iter()
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect())
    }

    /// Pre-compute embeddings for not-yet-embedded tools (incremental) so a later
    /// semantic/hybrid search only embeds the query. Private worker primitive;
    /// releases the GIL while loading models, calling HTTP, and running inference.
    fn _build_embeddings(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.build_embeddings())
            .map_err(map_embedder_err)
    }

    /// Recompute the complete tool embedding cache without holding the GIL.
    fn _rebuild_embeddings(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.rebuild_embeddings())
            .map_err(map_embedder_err)
    }

    /// Build a binary embedding artifact from the registered corpus (ADR-0018).
    /// Releases the GIL. Does not touch the mutable dense cache.
    fn _build_embedding_artifact<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .allow_threads(|| self.inner.build_embedding_artifact())
            .map_err(map_artifact_build_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Warm the dense cache from build-time artifact bytes. `on_miss` is
    /// `"error"` or `"embed"`.
    fn _warm_embeddings_from_artifact(
        &self,
        py: Python<'_>,
        bytes: Vec<u8>,
        on_miss: String,
    ) -> PyResult<()> {
        let policy = parse_on_artifact_miss(&on_miss)?;
        py.allow_threads(|| self.inner.warm_embeddings_from_artifact(&bytes, policy))
            .map_err(map_artifact_warm_err)
    }

    /// Re-embed the intent graph's members under the current model and replace
    /// its centroids. GIL-releasing worker; the Python facade wraps it as
    /// `rebuild_intent_graph`.
    fn _rebuild_intent_graph(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.rebuild_intent_graph())
            .map_err(map_embedder_err)
    }

    /// Build an intent graph from a JSONL trace log, returning its wire form.
    ///
    /// Embeds every distinct query so clusters form densely — the same tier the
    /// live path uses. Releases the GIL for the embedding pass. The returned
    /// graph is NOT attached to this registry; enabling adaptive ranking stays a
    /// separate, explicit call.
    #[pyo3(signature = (jsonl, origins=None, provenance=None))]
    fn _build_intent_graph(
        &self,
        py: Python<'_>,
        jsonl: &str,
        origins: Option<&str>,
        provenance: Option<&str>,
    ) -> PyResult<String> {
        // Policy errors surface before the embedding pass, so a typo fails fast.
        let policy = parse_policy(
            origins,
            provenance,
            core::OriginFilter::Exactly(Origin::Baseline),
            core::Provenance::Seeded,
        )?;
        let envelopes = parse_trace_log(jsonl)?;
        let graph = py
            .allow_threads(|| self.inner.build_intent_graph(envelopes, policy))
            .map_err(map_embedder_err)?;
        serde_json::to_string(&graph).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `(status, built, active, dim_mismatch)` — whether adaptive usage ranking
    /// is contributing, paused by a model change, or off.
    fn adaptive_ranking_status(&self) -> (String, Option<String>, Option<String>, Option<bool>) {
        map_adaptive_status(self.inner.adaptive_ranking_status())
    }

    /// Record an SDK-layer trace event into the active sink. `event` must be a
    /// dict matching one of the core-owned `TraceEvent` shapes (ADR-0007, e.g.
    /// `{"type": "gateway_search", ...}`); anything else raises `ValueError`.
    fn record_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: Value = pythonize::depythonize(event)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        let event: TraceEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        self.inner.record_event(event);
        Ok(())
    }

    /// Record an event with caller-supplied identity and OTel correlation.
    fn record_event_with_context(
        &self,
        event: &Bound<'_, PyAny>,
        context: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value: Value = pythonize::depythonize(event)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        let event: TraceEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        self.inner
            .record_event_with_context(event, trace_event_context(Some(context))?);
        Ok(())
    }

    /// Attach the private batched callback seam consumed by the public SDK in
    /// Phase 5. A dedicated Rust thread acquires the GIL for each callback;
    /// producer and dense-search worker threads only enqueue.
    #[pyo3(signature = (callback, session_id, source_id=None, queue_capacity=DEFAULT_EVENT_QUEUE_CAPACITY, batch_size=DEFAULT_EVENT_BATCH_SIZE))]
    fn subscribe_trace_events(
        &mut self,
        callback: Py<PyAny>,
        session_id: String,
        source_id: Option<String>,
        queue_capacity: usize,
        batch_size: usize,
    ) -> PyResult<NativeEventSubscription> {
        let subscription = subscribe_event_callback(
            &mut self.event_stream,
            callback,
            session_id,
            source_id,
            queue_capacity,
            batch_size,
        )?;
        let sink = active_trace_sink(
            &self.base_sink,
            self.graph.as_ref(),
            &self.event_stream,
            self.usage_policy,
        );
        self.inner.set_trace_sink(sink);
        Ok(subscription)
    }

    /// Route trace events to a sink. `kind` is `"noop"` (drop everything, the
    /// initial state), `"memory"` (buffer for `drain_trace_events`; requires
    /// `session_id`) or `"jsonl"` (append to a file; requires `session_id` and
    /// `path`). Raises `ValueError` on an unknown kind, a missing required
    /// argument, or a jsonl path that cannot be opened.
    #[pyo3(signature = (kind, session_id=None, path=None))]
    fn set_trace_sink(
        &mut self,
        kind: String,
        session_id: Option<String>,
        path: Option<String>,
    ) -> PyResult<()> {
        match kind.as_str() {
            "noop" => {
                self.memory_sink = None;
                self.base_sink = Arc::new(NoopSink);
                let sink = active_trace_sink(
                    &self.base_sink,
                    self.graph.as_ref(),
                    &self.event_stream,
                    self.usage_policy,
                );
                self.inner.set_trace_sink(sink);
            }
            "memory" => {
                let session_id = session_id
                    .ok_or_else(|| PyValueError::new_err("memory sink requires session_id"))?;
                let sink = Arc::new(MemorySink::new(session_id));
                self.memory_sink = Some(sink.clone());
                self.base_sink = sink;
                let sink = active_trace_sink(
                    &self.base_sink,
                    self.graph.as_ref(),
                    &self.event_stream,
                    self.usage_policy,
                );
                self.inner.set_trace_sink(sink);
            }
            "jsonl" => {
                let session_id = session_id
                    .ok_or_else(|| PyValueError::new_err("jsonl sink requires session_id"))?;
                let path = path.ok_or_else(|| PyValueError::new_err("jsonl sink requires path"))?;
                let sink = JsonlSink::new(session_id, &path)
                    .map_err(|e| PyValueError::new_err(format!("open jsonl sink: {e}")))?;
                self.memory_sink = None;
                self.base_sink = Arc::new(sink);
                let sink = active_trace_sink(
                    &self.base_sink,
                    self.graph.as_ref(),
                    &self.event_stream,
                    self.usage_policy,
                );
                self.inner.set_trace_sink(sink);
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown trace sink kind: {other}"
                )));
            }
        }
        Ok(())
    }

    /// Turn on adaptive usage ranking against `graph` (ADR-0014).
    ///
    /// Wires both halves: this registry ranks against the graph, and its trace
    /// sink is decorated with a learner that grows it from search-then-invoke
    /// pairs. Pass the same graph to the other registry so both learn into one
    /// set of clusters.
    ///
    /// Only queries matching a cluster are affected. With a graph attached
    /// `SearchHit.score` becomes a fusion score rather than a raw BM25 score, so
    /// compare ordering rather than magnitudes.
    #[pyo3(signature = (graph, origins=None, provenance=None, cluster_similarity=None, cluster_coverage=None))]
    fn enable_adaptive_ranking(
        &mut self,
        graph: &IntentGraph,
        origins: Option<&str>,
        provenance: Option<&str>,
        cluster_similarity: Option<f64>,
        cluster_coverage: Option<f64>,
    ) -> PyResult<()> {
        let cluster_policy = parse_cluster_policy(cluster_similarity, cluster_coverage)?;
        self.usage_policy = parse_policy(
            origins,
            provenance,
            core::OriginFilter::Any,
            core::Provenance::Live,
        )?;
        let handle = graph.inner.clone();
        // Sets what FUTURE admissions are measured against. Existing boundaries
        // stay as they are — nothing can redraw them in place — and the graph
        // keeps reporting the policy it was clustered under, so the difference
        // surfaces as drift rather than silently changing what a cluster means.
        handle
            .write()
            .map_err(|_| PyValueError::new_err("intent graph lock poisoned"))?
            .set_cluster_policy(cluster_policy);
        let sink = active_trace_sink(
            &self.base_sink,
            Some(&handle),
            &self.event_stream,
            self.usage_policy,
        );
        self.inner.set_trace_sink(sink);
        self.inner.set_intent_graph(Some(handle.clone()));
        self.graph = Some(handle);
        Ok(())
    }

    /// Turn adaptive usage ranking off: ranking returns to the base engine and
    /// the graph stops growing. The graph keeps what it learned.
    fn disable_adaptive_ranking(&mut self) {
        let sink = active_trace_sink(&self.base_sink, None, &self.event_stream, self.usage_policy);
        self.inner.set_trace_sink(sink);
        self.inner.set_intent_graph(None);
        self.graph = None;
    }

    /// Drain captured envelopes from the active sink. Returns `[]` unless the
    /// active sink is "memory".
    fn drain_trace_events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let Some(sink) = self.memory_sink.as_ref() else {
            return Ok(out);
        };
        for env in sink.drain() {
            let obj = pythonize::pythonize(py, &env)
                .map_err(|e| PyValueError::new_err(format!("serialize trace envelope: {e}")))?;
            out.append(obj)?;
        }
        Ok(out)
    }
}

/// Metadata registry over the skill corpus — the on-demand analogue of
/// [`ToolRegistry`]. A separate index keeps skill ranking independent of tools.
#[pyclass]
pub struct SkillRegistry {
    inner: core::SkillRegistry,
    memory_sink: Option<Arc<MemorySink>>,
    /// The current undecorated sink, whatever its kind. Retained so
    /// enable/disable adaptive ranking can re-wrap or restore it; rebuilding
    /// from `memory_sink` alone would drop a configured jsonl sink to noop.
    base_sink: Arc<dyn core::TraceSink>,
    /// Retained so `set_trace_sink` can re-wrap the new sink in a learner —
    /// otherwise changing sinks would silently switch learning off.
    graph: Option<Arc<RwLock<core::IntentGraph>>>,
    /// The policy the attached learner runs under. Retained beside `graph` for
    /// the same reason: a sink change re-wraps the learner, and rebuilding it at
    /// the default would silently drop a configured policy.
    usage_policy: core::ObservationPolicy,
    event_stream: Option<EventStream>,
}

#[pymethods]
impl SkillRegistry {
    #[new]
    #[pyo3(signature = (spec=None, huggingface=None, local=None, ollama=None, url=None, model=None, revision=None, api_key_env=None, query_prefix=None, doc_prefix=None, pooling=None, download=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        spec: Option<String>,
        huggingface: Option<String>,
        local: Option<String>,
        ollama: Option<String>,
        url: Option<String>,
        model: Option<String>,
        revision: Option<String>,
        api_key_env: Option<String>,
        query_prefix: Option<String>,
        doc_prefix: Option<String>,
        pooling: Option<String>,
        download: Option<bool>,
    ) -> PyResult<Self> {
        let inner = match resolve_embedding(
            spec,
            huggingface,
            local,
            ollama,
            url,
            model,
            revision,
            api_key_env,
            query_prefix,
            doc_prefix,
            pooling,
            download,
        )? {
            Some(model) => core::SkillRegistry::with_embedding(model),
            None => core::SkillRegistry::new(),
        };
        Ok(Self {
            inner,
            memory_sink: None,
            base_sink: Arc::new(NoopSink),
            graph: None,
            usage_policy: core::ObservationPolicy::default(),
            event_stream: None,
        })
    }

    /// Register a skill's metadata into the index (or replace it in place when
    /// `id` is already registered). `tags` are indexed for ranking; `tools` and
    /// `metadata` ride along un-indexed for higher layers; `body` is the full
    /// instruction text, stored for on-demand load.
    // Mirrors `ToolRegistry::register`'s flat-param style (PyO3 has no by-value
    // object arg like the TS NAPI `Skill`); a skill simply has more fields.
    #[allow(clippy::too_many_arguments)]
    fn register(
        &mut self,
        id: String,
        name: String,
        description: String,
        tags: Vec<String>,
        tools: Vec<String>,
        metadata: HashMap<String, Vec<String>>,
        body: String,
    ) {
        self.inner.register(core::Skill {
            id,
            name,
            description,
            tags,
            tools,
            metadata,
            body,
        });
    }

    /// Register a batch only after PyO3 has converted every item's full shape.
    /// A bad later item therefore fails before this method mutates the registry.
    fn _register_many(&mut self, skills: Vec<SkillBatchItem>) {
        for (id, name, description, tags, tools, metadata, body) in skills {
            self.inner.register(core::Skill {
                id,
                name,
                description,
                tags,
                tools,
                metadata,
                body,
            });
        }
    }

    /// Replace the whole corpus: ids absent from `skills` are removed, the rest
    /// are added or updated. Embeds nothing — the caller builds after, which then
    /// embeds only what this replace invalidated. Returns the
    /// `(added, removed, updated, unchanged)` counts; the Python facade wraps
    /// them in its `ReplaceOutcome` dataclass, as it does for `Skill`.
    fn _replace_all(&mut self, skills: Vec<SkillBatchItem>) -> (u32, u32, u32, u32) {
        let outcome = self.inner.replace_all(
            skills
                .into_iter()
                .map(
                    |(id, name, description, tags, tools, metadata, body)| core::Skill {
                        id,
                        name,
                        description,
                        tags,
                        tools,
                        metadata,
                        body,
                    },
                )
                .collect(),
        );
        (
            outcome.added as u32,
            outcome.removed as u32,
            outcome.updated as u32,
            outcome.unchanged as u32,
        )
    }

    /// Lexical BM25 search over the skill corpus — see [`ToolRegistry::search`].
    fn search(&self, query: String, top_k: u32) -> Vec<SkillHit> {
        self.inner
            .search(&query, top_k as usize)
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect()
    }

    /// BM25 search tagged with who initiated it — see
    /// [`ToolRegistry::search_with_origin`].
    #[pyo3(signature = (query, top_k, origin, context=None))]
    fn search_with_origin(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<SkillHit>> {
        let parsed = parse_origin(origin.as_str());
        let hits = self
            .inner
            .search_with_method_and_context(
                &query,
                top_k as usize,
                parsed,
                core::SearchMethod::Bm25,
                trace_event_context(context)?,
            )
            .map_err(map_embedder_err)?;
        Ok(hits
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect())
    }

    /// Private GIL-releasing method search — see [`ToolRegistry::_search_with_method`].
    #[pyo3(signature = (query, top_k, origin, method, context=None))]
    fn _search_with_method(
        &self,
        py: Python<'_>,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
        context: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<SkillHit>> {
        let parsed_origin = parse_origin(origin.as_str());
        let parsed_method = method
            .parse::<core::SearchMethod>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let context = trace_event_context(context)?;
        let hits = py
            .allow_threads(|| {
                self.inner.search_with_method_and_context(
                    &query,
                    top_k as usize,
                    parsed_origin,
                    parsed_method,
                    context,
                )
            })
            .map_err(map_embedder_err)?;
        Ok(hits
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect())
    }

    /// See [`ToolRegistry::_build_embeddings`].
    fn _build_embeddings(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.build_embeddings())
            .map_err(map_embedder_err)
    }

    /// Recompute the complete skill embedding cache without holding the GIL.
    fn _rebuild_embeddings(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.rebuild_embeddings())
            .map_err(map_embedder_err)
    }

    /// See [`ToolRegistry::_build_embedding_artifact`].
    fn _build_embedding_artifact<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .allow_threads(|| self.inner.build_embedding_artifact())
            .map_err(map_artifact_build_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// See [`ToolRegistry::_warm_embeddings_from_artifact`].
    fn _warm_embeddings_from_artifact(
        &self,
        py: Python<'_>,
        bytes: Vec<u8>,
        on_miss: String,
    ) -> PyResult<()> {
        let policy = parse_on_artifact_miss(&on_miss)?;
        py.allow_threads(|| self.inner.warm_embeddings_from_artifact(&bytes, policy))
            .map_err(map_artifact_warm_err)
    }

    /// Re-embed the intent graph's members under the current model and replace
    /// its centroids. GIL-releasing worker; the Python facade wraps it as
    /// `rebuild_intent_graph`.
    fn _rebuild_intent_graph(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.rebuild_intent_graph())
            .map_err(map_embedder_err)
    }

    /// `(status, built, active, dim_mismatch)` — whether adaptive usage ranking
    /// is contributing, paused by a model change, or off.
    fn adaptive_ranking_status(&self) -> (String, Option<String>, Option<String>, Option<bool>) {
        map_adaptive_status(self.inner.adaptive_ranking_status())
    }

    /// Record an SDK-layer trace event into the active sink — see
    /// [`ToolRegistry::record_event`].
    fn record_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: Value = pythonize::depythonize(event)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        let event: TraceEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        self.inner.record_event(event);
        Ok(())
    }

    /// Record an event with caller-supplied identity and OTel correlation.
    fn record_event_with_context(
        &self,
        event: &Bound<'_, PyAny>,
        context: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value: Value = pythonize::depythonize(event)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        let event: TraceEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        self.inner
            .record_event_with_context(event, trace_event_context(Some(context))?);
        Ok(())
    }

    /// Attach the private batched callback seam — see
    /// [`ToolRegistry::subscribe_trace_events`].
    #[pyo3(signature = (callback, session_id, source_id=None, queue_capacity=DEFAULT_EVENT_QUEUE_CAPACITY, batch_size=DEFAULT_EVENT_BATCH_SIZE))]
    fn subscribe_trace_events(
        &mut self,
        callback: Py<PyAny>,
        session_id: String,
        source_id: Option<String>,
        queue_capacity: usize,
        batch_size: usize,
    ) -> PyResult<NativeEventSubscription> {
        let subscription = subscribe_event_callback(
            &mut self.event_stream,
            callback,
            session_id,
            source_id,
            queue_capacity,
            batch_size,
        )?;
        let sink = active_trace_sink(
            &self.base_sink,
            self.graph.as_ref(),
            &self.event_stream,
            self.usage_policy,
        );
        self.inner.set_trace_sink(sink);
        Ok(subscription)
    }

    /// Route trace events to a sink — see [`ToolRegistry::set_trace_sink`] for
    /// the kind / session_id / path rules and `ValueError` conditions.
    #[pyo3(signature = (kind, session_id=None, path=None))]
    fn set_trace_sink(
        &mut self,
        kind: String,
        session_id: Option<String>,
        path: Option<String>,
    ) -> PyResult<()> {
        match kind.as_str() {
            "noop" => {
                self.memory_sink = None;
                self.base_sink = Arc::new(NoopSink);
                let sink = active_trace_sink(
                    &self.base_sink,
                    self.graph.as_ref(),
                    &self.event_stream,
                    self.usage_policy,
                );
                self.inner.set_trace_sink(sink);
            }
            "memory" => {
                let session_id = session_id
                    .ok_or_else(|| PyValueError::new_err("memory sink requires session_id"))?;
                let sink = Arc::new(MemorySink::new(session_id));
                self.memory_sink = Some(sink.clone());
                self.base_sink = sink;
                let sink = active_trace_sink(
                    &self.base_sink,
                    self.graph.as_ref(),
                    &self.event_stream,
                    self.usage_policy,
                );
                self.inner.set_trace_sink(sink);
            }
            "jsonl" => {
                let session_id = session_id
                    .ok_or_else(|| PyValueError::new_err("jsonl sink requires session_id"))?;
                let path = path.ok_or_else(|| PyValueError::new_err("jsonl sink requires path"))?;
                let sink = JsonlSink::new(session_id, &path)
                    .map_err(|e| PyValueError::new_err(format!("open jsonl sink: {e}")))?;
                self.memory_sink = None;
                self.base_sink = Arc::new(sink);
                let sink = active_trace_sink(
                    &self.base_sink,
                    self.graph.as_ref(),
                    &self.event_stream,
                    self.usage_policy,
                );
                self.inner.set_trace_sink(sink);
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown trace sink kind: {other}"
                )));
            }
        }
        Ok(())
    }

    /// Turn on adaptive usage ranking against `graph` (ADR-0014).
    ///
    /// Wires both halves: this registry ranks against the graph, and its trace
    /// sink is decorated with a learner that grows it from search-then-invoke
    /// pairs. Pass the same graph to the other registry so both learn into one
    /// set of clusters.
    ///
    /// Only queries matching a cluster are affected. With a graph attached
    /// `SearchHit.score` becomes a fusion score rather than a raw BM25 score, so
    /// compare ordering rather than magnitudes.
    #[pyo3(signature = (graph, origins=None, provenance=None, cluster_similarity=None, cluster_coverage=None))]
    fn enable_adaptive_ranking(
        &mut self,
        graph: &IntentGraph,
        origins: Option<&str>,
        provenance: Option<&str>,
        cluster_similarity: Option<f64>,
        cluster_coverage: Option<f64>,
    ) -> PyResult<()> {
        let cluster_policy = parse_cluster_policy(cluster_similarity, cluster_coverage)?;
        self.usage_policy = parse_policy(
            origins,
            provenance,
            core::OriginFilter::Any,
            core::Provenance::Live,
        )?;
        let handle = graph.inner.clone();
        // Sets what FUTURE admissions are measured against. Existing boundaries
        // stay as they are — nothing can redraw them in place — and the graph
        // keeps reporting the policy it was clustered under, so the difference
        // surfaces as drift rather than silently changing what a cluster means.
        handle
            .write()
            .map_err(|_| PyValueError::new_err("intent graph lock poisoned"))?
            .set_cluster_policy(cluster_policy);
        let sink = active_trace_sink(
            &self.base_sink,
            Some(&handle),
            &self.event_stream,
            self.usage_policy,
        );
        self.inner.set_trace_sink(sink);
        self.inner.set_intent_graph(Some(handle.clone()));
        self.graph = Some(handle);
        Ok(())
    }

    /// Turn adaptive usage ranking off: ranking returns to the base engine and
    /// the graph stops growing. The graph keeps what it learned.
    fn disable_adaptive_ranking(&mut self) {
        let sink = active_trace_sink(&self.base_sink, None, &self.event_stream, self.usage_policy);
        self.inner.set_trace_sink(sink);
        self.inner.set_intent_graph(None);
        self.graph = None;
    }

    /// Drain captured envelopes from the active sink — see
    /// [`ToolRegistry::drain_trace_events`].
    fn drain_trace_events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let Some(sink) = self.memory_sink.as_ref() else {
            return Ok(out);
        };
        for env in sink.drain() {
            let obj = pythonize::pythonize(py, &env)
                .map_err(|e| PyValueError::new_err(format!("serialize trace envelope: {e}")))?;
            out.append(obj)?;
        }
        Ok(out)
    }
}

/// Metadata registry over the fact corpus — the grounding-side twin of
/// [`SkillRegistry`]. A separate index keeps fact ranking independent of tools
/// and skills. Unlike a skill, a fact has no `tools` field and carries a `pin`
/// (`"always"` / `"retrieved"`) parsed into `core::PinMode` at registration.
#[pyclass]
pub struct FactRegistry {
    inner: core::FactRegistry,
    memory_sink: Option<Arc<MemorySink>>,
}

#[pymethods]
impl FactRegistry {
    #[new]
    #[pyo3(signature = (spec=None, huggingface=None, local=None, ollama=None, url=None, model=None, revision=None, api_key_env=None, query_prefix=None, doc_prefix=None, pooling=None, download=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        spec: Option<String>,
        huggingface: Option<String>,
        local: Option<String>,
        ollama: Option<String>,
        url: Option<String>,
        model: Option<String>,
        revision: Option<String>,
        api_key_env: Option<String>,
        query_prefix: Option<String>,
        doc_prefix: Option<String>,
        pooling: Option<String>,
        download: Option<bool>,
    ) -> PyResult<Self> {
        let inner = match resolve_embedding(
            spec,
            huggingface,
            local,
            ollama,
            url,
            model,
            revision,
            api_key_env,
            query_prefix,
            doc_prefix,
            pooling,
            download,
        )? {
            Some(model) => core::FactRegistry::with_embedding(model),
            None => core::FactRegistry::new(),
        };
        Ok(Self {
            inner,
            memory_sink: None,
        })
    }

    /// Register a fact's metadata into the index (or replace it in place when
    /// `id` is already registered). `tags` are indexed for ranking; `metadata`
    /// rides along un-indexed for higher layers; `body` is the injected content,
    /// stored but not indexed; `pin` is `"always"` or `"retrieved"` (parsed into
    /// `core::PinMode` — an unknown value raises `ValueError`).
    // Mirrors `SkillRegistry::register`, minus the `tools` Vec and plus `pin`.
    #[allow(clippy::too_many_arguments)]
    fn register(
        &mut self,
        id: String,
        name: String,
        description: String,
        tags: Vec<String>,
        metadata: HashMap<String, Vec<String>>,
        body: String,
        pin: String,
    ) -> PyResult<()> {
        let pin = pin
            .parse::<core::PinMode>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.register(core::Fact {
            id,
            name,
            description,
            tags,
            metadata,
            body,
            pin,
        });
        Ok(())
    }

    /// Register a batch only after every item's full shape (including a valid
    /// `pin`) has been converted. A bad later `pin` therefore fails before this
    /// method mutates the registry — no partial registration.
    fn _register_many(&mut self, facts: Vec<FactBatchItem>) -> PyResult<()> {
        let facts = facts
            .into_iter()
            .map(|(id, name, description, tags, metadata, body, pin)| {
                let pin = pin
                    .parse::<core::PinMode>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(core::Fact {
                    id,
                    name,
                    description,
                    tags,
                    metadata,
                    body,
                    pin,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        for fact in facts {
            self.inner.register(fact);
        }
        Ok(())
    }

    /// Lexical BM25 search over the fact corpus — see [`ToolRegistry::search`].
    fn search(&self, query: String, top_k: u32) -> Vec<FactHit> {
        self.inner
            .search(&query, top_k as usize)
            .into_iter()
            .map(|hit| FactHit {
                fact_id: hit.fact_id,
                score: hit.score as f64,
            })
            .collect()
    }

    /// BM25 search tagged with who initiated it — see
    /// [`ToolRegistry::search_with_origin`].
    fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<FactHit> {
        let parsed = parse_origin(origin.as_str());
        self.inner
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| FactHit {
                fact_id: hit.fact_id,
                score: hit.score as f64,
            })
            .collect()
    }

    /// Private GIL-releasing method search — see [`ToolRegistry::_search_with_method`].
    fn _search_with_method(
        &self,
        py: Python<'_>,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> PyResult<Vec<FactHit>> {
        let parsed_origin = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        let parsed_method = method
            .parse::<core::SearchMethod>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let hits = py
            .allow_threads(|| {
                self.inner
                    .search_with_method(&query, top_k as usize, parsed_origin, parsed_method)
            })
            .map_err(map_embedder_err)?;
        Ok(hits
            .into_iter()
            .map(|hit| FactHit {
                fact_id: hit.fact_id,
                score: hit.score as f64,
            })
            .collect())
    }

    /// See [`ToolRegistry::_build_embeddings`].
    fn _build_embeddings(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.build_embeddings())
            .map_err(map_embedder_err)
    }

    /// Recompute the complete fact embedding cache without holding the GIL.
    fn _rebuild_embeddings(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.rebuild_embeddings())
            .map_err(map_embedder_err)
    }

    /// Record an SDK-layer trace event into the active sink — see
    /// [`ToolRegistry::record_event`].
    fn record_event(&self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: Value = pythonize::depythonize(event)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        let event: TraceEvent = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("invalid trace event: {e}")))?;
        self.inner.record_event(event);
        Ok(())
    }

    /// Route trace events to a sink — see [`ToolRegistry::set_trace_sink`] for
    /// the kind / session_id / path rules and `ValueError` conditions.
    #[pyo3(signature = (kind, session_id=None, path=None))]
    fn set_trace_sink(
        &mut self,
        kind: String,
        session_id: Option<String>,
        path: Option<String>,
    ) -> PyResult<()> {
        match kind.as_str() {
            "noop" => {
                self.memory_sink = None;
                self.inner.set_trace_sink(Arc::new(NoopSink));
            }
            "memory" => {
                let session_id = session_id
                    .ok_or_else(|| PyValueError::new_err("memory sink requires session_id"))?;
                let sink = Arc::new(MemorySink::new(session_id));
                self.memory_sink = Some(sink.clone());
                self.inner.set_trace_sink(sink);
            }
            "jsonl" => {
                let session_id = session_id
                    .ok_or_else(|| PyValueError::new_err("jsonl sink requires session_id"))?;
                let path = path.ok_or_else(|| PyValueError::new_err("jsonl sink requires path"))?;
                let sink = JsonlSink::new(session_id, &path)
                    .map_err(|e| PyValueError::new_err(format!("open jsonl sink: {e}")))?;
                self.memory_sink = None;
                self.inner.set_trace_sink(Arc::new(sink));
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown trace sink kind: {other}"
                )));
            }
        }
        Ok(())
    }

    /// Drain captured envelopes from the active sink — see
    /// [`ToolRegistry::drain_trace_events`].
    fn drain_trace_events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        let Some(sink) = self.memory_sink.as_ref() else {
            return Ok(out);
        };
        for env in sink.drain() {
            let obj = pythonize::pythonize(py, &env)
                .map_err(|e| PyValueError::new_err(format!("serialize trace envelope: {e}")))?;
            out.append(obj)?;
        }
        Ok(out)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeEventSubscription>()?;
    m.add_class::<ToolRegistry>()?;
    m.add_class::<IntentGraph>()?;
    m.add_class::<SearchHit>()?;
    m.add_class::<SkillRegistry>()?;
    m.add_class::<SkillHit>()?;
    m.add_function(wrap_pyfunction!(merge_embedding_artifacts, m)?)?;
    m.add_class::<FactRegistry>()?;
    m.add_class::<FactHit>()?;
    m.add("EmbedderError", m.py().get_type::<EmbedderError>())?;
    m.add(
        "DimensionMismatchError",
        m.py().get_type::<DimensionMismatchError>(),
    )?;
    m.add("ArtifactError", m.py().get_type::<ArtifactError>())?;
    m.add(
        "IncompatibleMergeError",
        m.py().get_type::<IncompatibleMergeError>(),
    )?;
    m.add("ArtifactWarmError", m.py().get_type::<ArtifactWarmError>())?;
    Ok(())
}
