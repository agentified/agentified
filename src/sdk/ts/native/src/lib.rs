#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

use napi::bindgen_prelude::{AsyncTask, Buffer};
use napi::{Env, Task};
use ratel_ai_core as core;
use ratel_ai_core::{
    ArtifactError, EmbeddingModel, EmbeddingSpec, JsonlSink, MemorySink, NoopSink, OnArtifactMiss,
    Origin, SearchMethod, TraceEvent, UsageLearner,
};
use serde_json::{Value, json};

/// A constructed sink plus the `MemorySink` handle when the kind is `"memory"`
/// (so the owner can drain it later).
type BuiltTraceSink = (Arc<dyn core::TraceSink>, Option<Arc<MemorySink>>);

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

const REGISTRY_BUSY_MESSAGE: &str =
    "registry busy; await the active operation before registering more items";

/// Private NAPI→TypeScript transport prefix for structured artifact-warm errors.
/// Must stay identical to the constant in `src/sdk/ts/src/errors.ts`.
const ARTIFACT_WARM_ERROR_PREFIX: &str = "RATEL_ARTIFACT_WARM_ERROR:";

/// Private NAPI→TypeScript transport prefix for artifact build/merge errors.
/// Must stay identical to the constant in `src/sdk/ts/src/errors.ts`.
const ARTIFACT_ERROR_PREFIX: &str = "RATEL_ARTIFACT_ERROR:";

fn artifact_error_variant_code(error: &ArtifactError) -> &'static str {
    match error {
        ArtifactError::TooShort { .. } => "TooShort",
        ArtifactError::InvalidMagic { .. } => "InvalidMagic",
        ArtifactError::UnsupportedFormatVersion { .. } => "UnsupportedFormatVersion",
        ArtifactError::ChecksumMismatch => "ChecksumMismatch",
        ArtifactError::CorruptPayload { .. } => "CorruptPayload",
        ArtifactError::InconsistentVectorWidth { .. } => "InconsistentVectorWidth",
        ArtifactError::VectorNotNormalized { .. } => "VectorNotNormalized",
        ArtifactError::NonEmptyZeroDim => "NonEmptyZeroDim",
        ArtifactError::InvalidVector { .. } => "InvalidVector",
        ArtifactError::IncompatibleMerge { .. } => "IncompatibleMerge",
        ArtifactError::Embedder(_) => "Embedder",
    }
}

fn map_artifact_warm_error(error: core::ArtifactWarmError) -> napi::Error {
    let message = error.to_string();
    let payload = match &error {
        core::ArtifactWarmError::Warm(_) => json!({
            "code": "Warm",
            "message": message,
        }),
        core::ArtifactWarmError::Incomplete { missing } => json!({
            "code": "Incomplete",
            "message": message,
            "missing": missing,
        }),
        core::ArtifactWarmError::Embedder(_) => json!({
            "code": "Embedder",
            "message": message,
        }),
    };
    // Value::to_string is infallible for this JSON-safe payload shape.
    let serialized = payload.to_string();
    napi::Error::from_reason(format!("{ARTIFACT_WARM_ERROR_PREFIX}{serialized}"))
}

fn map_artifact_build_error(error: ArtifactError) -> napi::Error {
    match error {
        ArtifactError::Embedder(inner) => napi::Error::from_reason(inner.to_string()),
        other => {
            let code = artifact_error_variant_code(&other);
            let message = other.to_string();
            let payload = json!({
                "code": code,
                "message": message,
            });
            // Value::to_string is infallible for this JSON-safe payload shape.
            let serialized = payload.to_string();
            napi::Error::from_reason(format!("{ARTIFACT_ERROR_PREFIX}{serialized}"))
        }
    }
}

/// Merge valid RAT1 parts into one mixed artifact (see core
/// `merge_embedding_artifacts`).
#[napi]
pub fn merge_embedding_artifacts(parts: Vec<Buffer>) -> napi::Result<Buffer> {
    let refs: Vec<&[u8]> = parts.iter().map(|part| part.as_ref()).collect();
    core::merge_embedding_artifacts(&refs)
        .map(Buffer::from)
        .map_err(map_artifact_build_error)
}
#[derive(Clone, Copy)]
enum EmbeddingOperation {
    Build,
    Rebuild,
    RebuildIntentGraph,
}

struct DenseOperationPermit {
    pending: Arc<AtomicUsize>,
}

impl DenseOperationPermit {
    fn new(pending: Arc<AtomicUsize>) -> Self {
        pending.fetch_add(1, Ordering::AcqRel);
        Self { pending }
    }
}

impl Drop for DenseOperationPermit {
    fn drop(&mut self) {
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

pub struct ToolEmbeddingTask {
    inner: Arc<RwLock<core::ToolRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    operation: EmbeddingOperation,
    _permit: DenseOperationPermit,
}

pub struct ToolSearchTask {
    inner: Arc<RwLock<core::ToolRegistry>>,
    dense_gate: Option<Arc<Mutex<()>>>,
    query: String,
    top_k: u32,
    origin: String,
    method: String,
    _permit: Option<DenseOperationPermit>,
}

pub struct SkillEmbeddingTask {
    inner: Arc<RwLock<core::SkillRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    operation: EmbeddingOperation,
    _permit: DenseOperationPermit,
}

pub struct SkillSearchTask {
    inner: Arc<RwLock<core::SkillRegistry>>,
    dense_gate: Option<Arc<Mutex<()>>>,
    query: String,
    top_k: u32,
    origin: String,
    method: String,
    _permit: Option<DenseOperationPermit>,
}

pub struct ToolBuildArtifactTask {
    inner: Arc<RwLock<core::ToolRegistry>>,
    _permit: DenseOperationPermit,
}

pub struct ToolWarmArtifactTask {
    inner: Arc<RwLock<core::ToolRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    bytes: Buffer,
    on_miss: String,
    _permit: DenseOperationPermit,
}

pub struct SkillBuildArtifactTask {
    inner: Arc<RwLock<core::SkillRegistry>>,
    _permit: DenseOperationPermit,
}

pub struct SkillWarmArtifactTask {
    inner: Arc<RwLock<core::SkillRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    bytes: Buffer,
    on_miss: String,
    _permit: DenseOperationPermit,
}

impl Task for ToolEmbeddingTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let _dense = self
            .dense_gate
            .lock()
            .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?;
        match self.operation {
            EmbeddingOperation::Build => registry.build_embeddings(),
            EmbeddingOperation::Rebuild => registry.rebuild_embeddings(),
            EmbeddingOperation::RebuildIntentGraph => registry.rebuild_intent_graph(),
        }
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for ToolBuildArtifactTask {
    type Output = Vec<u8>;
    type JsValue = Buffer;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?;
        registry
            .build_embedding_artifact()
            .map_err(map_artifact_build_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(Buffer::from(output))
    }
}

impl Task for ToolWarmArtifactTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let on_miss: OnArtifactMiss =
            self.on_miss
                .parse()
                .map_err(|e: ratel_ai_core::ParseOnArtifactMissError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        let _dense = self
            .dense_gate
            .lock()
            .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?;
        registry
            .warm_embeddings_from_artifact(self.bytes.as_ref(), on_miss)
            .map_err(map_artifact_warm_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for ToolSearchTask {
    type Output = Vec<SearchHit>;
    type JsValue = Vec<SearchHit>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let parsed_origin = parse_origin(self.origin.as_str());
        let parsed_method: SearchMethod =
            self.method
                .parse()
                .map_err(|e: ratel_ai_core::ParseSearchMethodError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        let _dense = self
            .dense_gate
            .as_ref()
            .map(|gate| {
                gate.lock()
                    .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))
            })
            .transpose()?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?;
        registry
            .search_with_method(
                &self.query,
                self.top_k as usize,
                parsed_origin,
                parsed_method,
            )
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SearchHit {
                        tool_id: hit.tool_id,
                        score: hit.score as f64,
                        rank: hit.rank,
                        fused: hit.fused,
                    })
                    .collect()
            })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for SkillEmbeddingTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let _dense = self
            .dense_gate
            .lock()
            .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?;
        match self.operation {
            EmbeddingOperation::Build => registry.build_embeddings(),
            EmbeddingOperation::Rebuild => registry.rebuild_embeddings(),
            EmbeddingOperation::RebuildIntentGraph => registry.rebuild_intent_graph(),
        }
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for SkillBuildArtifactTask {
    type Output = Vec<u8>;
    type JsValue = Buffer;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?;
        registry
            .build_embedding_artifact()
            .map_err(map_artifact_build_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(Buffer::from(output))
    }
}

impl Task for SkillWarmArtifactTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let on_miss: OnArtifactMiss =
            self.on_miss
                .parse()
                .map_err(|e: ratel_ai_core::ParseOnArtifactMissError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        let _dense = self
            .dense_gate
            .lock()
            .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?;
        registry
            .warm_embeddings_from_artifact(self.bytes.as_ref(), on_miss)
            .map_err(map_artifact_warm_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for SkillSearchTask {
    type Output = Vec<SkillHit>;
    type JsValue = Vec<SkillHit>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let parsed_origin = parse_origin(self.origin.as_str());
        let parsed_method: SearchMethod =
            self.method
                .parse()
                .map_err(|e: ratel_ai_core::ParseSearchMethodError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        let _dense = self
            .dense_gate
            .as_ref()
            .map(|gate| {
                gate.lock()
                    .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))
            })
            .transpose()?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?;
        registry
            .search_with_method(
                &self.query,
                self.top_k as usize,
                parsed_origin,
                parsed_method,
            )
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| SkillHit {
                        skill_id: hit.skill_id,
                        score: hit.score as f64,
                        rank: hit.rank,
                        fused: hit.fused,
                    })
                    .collect()
            })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

fn write_registry<'a, T>(
    inner: &'a RwLock<T>,
    pending_dense: &AtomicUsize,
) -> napi::Result<RwLockWriteGuard<'a, T>> {
    if pending_dense.load(Ordering::Acquire) > 0 {
        return Err(napi::Error::from_reason(REGISTRY_BUSY_MESSAGE));
    }
    inner.try_write().map_err(|error| match error {
        std::sync::TryLockError::WouldBlock => napi::Error::from_reason(REGISTRY_BUSY_MESSAGE),
        std::sync::TryLockError::Poisoned(_) => napi::Error::from_reason("registry lock poisoned"),
    })
}

/// Build a trace sink from a [`TraceSinkConfig`].
fn build_trace_sink(config: TraceSinkConfig) -> napi::Result<BuiltTraceSink> {
    match config.kind.as_str() {
        "noop" => Ok((Arc::new(NoopSink), None)),
        "memory" => {
            let session_id = config
                .session_id
                .ok_or_else(|| napi::Error::from_reason("memory sink requires sessionId"))?;
            let sink = Arc::new(MemorySink::new(session_id));
            Ok((sink.clone(), Some(sink)))
        }
        "jsonl" => {
            let session_id = config
                .session_id
                .ok_or_else(|| napi::Error::from_reason("jsonl sink requires sessionId"))?;
            let path = config
                .path
                .ok_or_else(|| napi::Error::from_reason("jsonl sink requires path"))?;
            let sink = JsonlSink::new(session_id, &path)
                .map_err(|e| napi::Error::from_reason(format!("open jsonl sink: {e}")))?;
            Ok((Arc::new(sink), None))
        }
        other => Err(napi::Error::from_reason(format!(
            "unknown trace sink kind: {other}"
        ))),
    }
}

/// A tool's searchable metadata: what the registry indexes and what a search
/// hit resolves back to. Execution lives a layer up (the SDK's `ToolCatalog`
/// pairs each `Tool` with its executor).
#[napi(object)]
pub struct Tool {
    /// Unique id, the registry key. Re-registering an existing id replaces the
    /// entry in place. MCP-proxied tools use the `<server>__<tool>` convention.
    pub id: String,
    /// Callable name (typically the same as `id` for local tools); indexed for
    /// ranking both whole and split on `snake_case`/`camelCase` boundaries.
    pub name: String,
    /// What the tool does and when to use it — the main ranking signal.
    pub description: String,
    /// JSON Schema of the arguments. Property names and their `description`s
    /// (nested included) are indexed for ranking.
    #[napi(ts_type = "import('json-schema').JSONSchema7")]
    pub input_schema: Value,
    /// JSON Schema of the result; indexed the same way as `inputSchema`.
    #[napi(ts_type = "import('json-schema').JSONSchema7")]
    pub output_schema: Value,
}

/// One ranked tool from a registry search, best-first.
#[napi(object)]
pub struct SearchHit {
    /// Id of the matched tool, as registered.
    pub tool_id: String,
    /// Relevance score; higher is better, ties break by id ascending. Its scale
    /// depends on the method (raw BM25 / cosine / RRF) AND on `fused` — with
    /// adaptive ranking a matched query returns small RRF scores while an
    /// unmatched one on the same catalog returns the raw score. Order by `rank`
    /// and branch on `fused`; treat `score` as a within-list hint only.
    pub score: f64,
    /// 0-based position in this result list (best is `0`). Stable across methods
    /// and across the `fused` switch — the field to order or threshold on.
    pub rank: u32,
    /// `true` when `score` is an RRF score (ordering-only) rather than the raw
    /// method score: the usage arm fused into this search, or the method is
    /// hybrid. Uniform across one result list; lets a caller detect the scale.
    pub fused: bool,
}

/// Destination for the local trace stream (ADR-0007): `"noop"` discards,
/// `"memory"` buffers envelopes for `drainTraceEvents`, `"jsonl"` appends one
/// JSON envelope per line to `path`.
#[napi(object)]
pub struct TraceSinkConfig {
    /// One of "noop" | "memory" | "jsonl".
    pub kind: String,
    /// Stamped on every envelope. Required for "memory" and "jsonl".
    pub session_id: Option<String>,
    /// Required for "jsonl".
    pub path: Option<String>,
}

/// Cross-SDK embedding-model config. The high-level catalog normalizes the
/// public `string | object` form into these fields; core [`EmbeddingModel::resolve`]
/// infers/validates the source. Exactly one of `spec`/`huggingface`/`local`/
/// `ollama`/`url` is a primary source; the rest are modifiers.
#[napi(object)]
pub struct EmbeddingConfig {
    pub spec: Option<String>,
    pub huggingface: Option<String>,
    pub local: Option<String>,
    pub ollama: Option<String>,
    pub url: Option<String>,
    pub model: Option<String>,
    pub revision: Option<String>,
    pub api_key_env: Option<String>,
    pub query_prefix: Option<String>,
    pub doc_prefix: Option<String>,
    /// `"cls"` | `"mean"` — overrides pooling auto-detection (in-process models).
    pub pooling: Option<String>,
    /// Opt in to downloading a not-yet-cached HuggingFace model (default false).
    pub download: Option<bool>,
}

/// Resolve an optional [`EmbeddingConfig`] to a core model, throwing config
/// errors at construction. `None` → the built-in default (no override).
fn resolve_embedding(config: Option<EmbeddingConfig>) -> napi::Result<Option<EmbeddingModel>> {
    let Some(c) = config else { return Ok(None) };
    let spec = EmbeddingSpec {
        spec: c.spec,
        huggingface: c.huggingface,
        local: c.local,
        ollama: c.ollama,
        url: c.url,
        model: c.model,
        revision: c.revision,
        api_key_env: c.api_key_env,
        query_prefix: c.query_prefix,
        doc_prefix: c.doc_prefix,
        pooling: c.pooling,
        download: c.download,
    };
    EmbeddingModel::resolve(spec)
        .map(Some)
        .map_err(|e| napi::Error::from_reason(e.to_string()))
}

/// SDK-facing view of whether adaptive usage ranking is contributing. `status`
/// is `"active" | "inactive" | "unknown" | "paused: dim mismatch" | "paused:
/// model mismatch"`; `built`/`active`/`dimMismatch` are set only when paused.
#[napi(object)]
pub struct AdaptiveRankingStatus {
    /// One of `"active" | "inactive" | "unknown" | "paused: dim mismatch" |
    /// "paused: model mismatch"`.
    pub status: String,
    /// When paused: the embedding model (or its dimension) the graph's centroids
    /// were built with. Absent unless paused.
    pub built: Option<String>,
    /// When paused: the currently active embedding model (or its dimension).
    /// Absent unless paused.
    pub active: Option<String>,
    /// When paused: `true` if the mismatch is a dimension difference, `false` if
    /// it is a same-dimension model-identity difference. Absent unless paused.
    pub dim_mismatch: Option<bool>,
}

fn map_status(s: core::AdaptiveRankingStatus) -> AdaptiveRankingStatus {
    use core::AdaptiveRankingStatus as S;
    match s {
        S::Inactive => AdaptiveRankingStatus {
            status: "inactive".into(),
            built: None,
            active: None,
            dim_mismatch: None,
        },
        S::Active => AdaptiveRankingStatus {
            status: "active".into(),
            built: None,
            active: None,
            dim_mismatch: None,
        },
        S::Unknown => AdaptiveRankingStatus {
            status: "unknown".into(),
            built: None,
            active: None,
            dim_mismatch: None,
        },
        S::Paused {
            dim_mismatch,
            built,
            active,
        } => AdaptiveRankingStatus {
            status: if dim_mismatch {
                "paused: dim mismatch"
            } else {
                "paused: model mismatch"
            }
            .into(),
            built: Some(built),
            active: Some(active),
            dim_mismatch: Some(dim_mismatch),
        },
    }
}

/// A shared usage-ranking intent graph (ADR-0014): clusters of past queries,
/// each remembering the capabilities invoked after them.
///
/// **Hand the same instance to both registries.** One cluster carries a `tools`
/// map and a `skills` map, so a tool catalog and a skill catalog sharing a graph
/// learn from and rank against one set of clusters. Giving them separate graphs
/// duplicates every cluster and halves the evidence behind each.
#[napi]
pub struct IntentGraph {
    inner: Arc<RwLock<core::IntentGraph>>,
}

impl Default for IntentGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[napi]
impl IntentGraph {
    /// An empty graph — knows nothing until a search is followed by an invoke.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(core::IntentGraph::empty())),
        }
    }

    /// Adopt a graph serialized in the `protocol/v1` wire form — produced by
    /// a previous `toJson` or by Ratel Cloud.
    ///
    /// Throws if the JSON is malformed or declares a schema version this build
    /// does not read (a consumer rejects rather than degrading).
    #[napi(factory)]
    pub fn from_json(json: String) -> napi::Result<Self> {
        let graph = core::IntentGraph::from_json(&json)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
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
    #[napi]
    pub fn to_json(&self) -> napi::Result<String> {
        let guard = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("intent graph lock poisoned"))?;
        serde_json::to_string(&*guard)
            .map_err(|e| napi::Error::from_reason(format!("serialize intent graph: {e}")))
    }

    /// How many clusters the graph holds. `0` is the cold-start state, in which
    /// the graph contributes nothing to ranking.
    #[napi(getter)]
    pub fn cluster_count(&self) -> napi::Result<u32> {
        let guard = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("intent graph lock poisoned"))?;
        Ok(guard.len() as u32)
    }

    /// Monotonic write counter — bumped once per mutation (a confirmed
    /// observation, a rebuild). Never affects ranking; it is a primitive for your
    /// storage layer. Snapshot it after each save: a later value means unsaved
    /// learning (save-when-changed), and a stored graph whose `rev` is higher than
    /// the one you loaded was written by another process (stale-base detection).
    #[napi(getter)]
    pub fn rev(&self) -> napi::Result<f64> {
        let guard = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("intent graph lock poisoned"))?;
        Ok(guard.rev() as f64)
    }
}

/// Node binding over the `ratel-ai-core` tool registry: an in-process index
/// that ranks registered tools against a natural-language query (BM25 by
/// default; semantic/hybrid once embeddings are built). Metadata-only — the
/// SDK's `ToolCatalog` layers executors, OTel spans, and defaults on top.
#[napi]
pub struct ToolRegistry {
    inner: Arc<RwLock<core::ToolRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    pending_dense: Arc<AtomicUsize>,
    memory_sink: Option<Arc<MemorySink>>,
    /// The current undecorated sink, whatever its kind. Retained so
    /// enable/disable adaptive ranking can re-wrap or restore it; rebuilding
    /// from `memory_sink` alone would drop a configured jsonl sink to noop.
    base_sink: Arc<dyn core::TraceSink>,
    /// Retained so `setTraceSink` can re-wrap the new sink in a learner —
    /// otherwise changing sinks would silently switch learning off.
    graph: Option<Arc<RwLock<core::IntentGraph>>>,
}

#[napi]
impl ToolRegistry {
    /// Construct a registry with a no-op trace sink. An optional `embedding`
    /// config selects the semantic/hybrid model (default bge-small when
    /// omitted); an invalid config throws here, at construction.
    #[napi(constructor)]
    pub fn new(embedding: Option<EmbeddingConfig>) -> napi::Result<Self> {
        let inner = match resolve_embedding(embedding)? {
            Some(model) => core::ToolRegistry::with_embedding(model),
            None => core::ToolRegistry::new(),
        };
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            dense_gate: Arc::new(Mutex::new(())),
            pending_dense: Arc::new(AtomicUsize::new(0)),
            memory_sink: None,
            base_sink: Arc::new(NoopSink),
            graph: None,
        })
    }

    /// Index a tool, or replace one in place if its id is already registered
    /// (the corpus never holds a duplicate). Infallible and model-free; a
    /// semantic caller embeds afterwards via `buildEmbeddings`.
    #[napi]
    pub fn register(&self, tool: Tool) -> napi::Result<()> {
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.register(core::Tool {
            id: tool.id,
            name: tool.name,
            description: tool.description,
            input_schema: tool.input_schema,
            output_schema: tool.output_schema,
        });
        Ok(())
    }

    /// Index a batch under one registry write lock.
    #[napi]
    pub fn register_many(&self, tools: Vec<Tool>) -> napi::Result<()> {
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        for tool in tools {
            registry.register(core::Tool {
                id: tool.id,
                name: tool.name,
                description: tool.description,
                input_schema: tool.input_schema,
                output_schema: tool.output_schema,
            });
        }
        Ok(())
    }

    /// Lexical BM25 search: up to `topK` hits, best-first with ties broken by
    /// id. Model-free and infallible; an empty registry returns `[]`. Records
    /// the query on the local trace stream with origin `"direct"`.
    #[napi]
    pub fn search(&self, query: String, top_k: u32) -> Vec<SearchHit> {
        self.inner
            .read()
            .expect("tool registry lock poisoned")
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

    /// BM25 search with an explicit origin — `"agent"` for a call the model
    /// synthesized (capability tools), anything else counts as `"direct"`
    /// (host code). Origin only annotates the trace event; ranking is
    /// identical to `search`.
    #[napi]
    pub fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<SearchHit> {
        let parsed = parse_origin(origin.as_str());
        self.inner
            .read()
            .expect("tool registry lock poisoned")
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect()
    }

    /// Synchronous method search. Accepts BM25 only; semantic/hybrid callers use
    /// `searchWithMethodAsync` so model and endpoint work stays off the event loop.
    #[napi]
    pub fn search_with_method(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> napi::Result<Vec<SearchHit>> {
        let parsed_origin = parse_origin(origin.as_str());
        let parsed_method: SearchMethod =
            method
                .parse()
                .map_err(|e: ratel_ai_core::ParseSearchMethodError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        if !matches!(parsed_method, SearchMethod::Bm25) {
            return Err(napi::Error::from_reason(
                "semantic and hybrid search are asynchronous; use searchWithMethodAsync() or ToolCatalog.searchAsync()",
            ));
        }
        let hits = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?
            .search_with_method(&query, top_k as usize, parsed_origin, parsed_method)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
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

    /// Search on a libuv worker. Supports BM25, semantic, and hybrid methods.
    #[napi(ts_return_type = "Promise<Array<SearchHit>>")]
    pub fn search_with_method_async(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> AsyncTask<ToolSearchTask> {
        let is_dense = matches!(method.as_str(), "semantic" | "dense" | "hybrid");
        AsyncTask::new(ToolSearchTask {
            inner: self.inner.clone(),
            dense_gate: is_dense.then(|| self.dense_gate.clone()),
            query,
            top_k,
            origin,
            method,
            _permit: is_dense.then(|| DenseOperationPermit::new(self.pending_dense.clone())),
        })
    }

    /// Pre-compute embeddings for not-yet-embedded tools on a worker. Registration
    /// is metadata-only; callers explicitly await this after populating the corpus.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn build_embeddings(&self) -> AsyncTask<ToolEmbeddingTask> {
        AsyncTask::new(ToolEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::Build,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Recompute the full tool corpus and atomically replace the dense cache.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn rebuild_embeddings(&self) -> AsyncTask<ToolEmbeddingTask> {
        AsyncTask::new(ToolEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::Rebuild,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Build a binary embedding artifact from the registered corpus (ADR-0018).
    /// Runs on a libuv worker. Takes a dense-operation permit so corpus
    /// mutations are rejected while the build is pending, but does **not** take
    /// the dense gate — semantic search may run concurrently (artifact build is
    /// independent of the mutable dense cache).
    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn build_embedding_artifact(&self) -> AsyncTask<ToolBuildArtifactTask> {
        AsyncTask::new(ToolBuildArtifactTask {
            inner: self.inner.clone(),
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Warm the dense cache from a build-time artifact. `on_miss` is `"error"`
    /// or `"embed"` (see [`OnArtifactMiss`]).
    #[napi(ts_return_type = "Promise<void>")]
    pub fn warm_embeddings_from_artifact(
        &self,
        bytes: Buffer,
        on_miss: String,
    ) -> AsyncTask<ToolWarmArtifactTask> {
        AsyncTask::new(ToolWarmArtifactTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            bytes,
            on_miss,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Record a custom event on the local trace stream (ADR-0007). `event` is
    /// the tagged wire shape — `{ type: "...", ... }` with snake_case fields —
    /// and an object that doesn't parse as a known event throws
    /// (`invalid trace event`). Higher layers use this to put their
    /// invoke/upstream/auth lifecycle events on the same stream as the
    /// registry's own search events.
    #[napi]
    pub fn record_event(&self, event: Value) -> napi::Result<()> {
        let event: TraceEvent = serde_json::from_value(event)
            .map_err(|e| napi::Error::from_reason(format!("invalid trace event: {e}")))?;
        self.inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?
            .record_event(event);
        Ok(())
    }

    /// Replace the trace sink; subsequent events go to the new destination,
    /// already-recorded ones are not replayed. Throws on an unknown `kind`, a
    /// missing `sessionId`/`path`, or a `"jsonl"` file that can't be opened.
    #[napi]
    pub fn set_trace_sink(&mut self, config: TraceSinkConfig) -> napi::Result<()> {
        let (sink, memory) = build_trace_sink(config)?;
        // Retain the raw sink so enable/disable can re-wrap or restore it —
        // rebuilding from `memory_sink` alone would drop a jsonl sink to noop.
        self.base_sink = sink.clone();
        // Re-wrap: adaptive ranking learns by decorating the sink, so replacing
        // the sink outright would quietly stop learning.
        let sink = match &self.graph {
            Some(graph) => {
                Arc::new(UsageLearner::new(graph.clone(), sink)) as Arc<dyn core::TraceSink>
            }
            None => sink,
        };
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(sink);
        drop(registry);
        self.memory_sink = memory;
        Ok(())
    }

    /// Turn on adaptive usage ranking against `graph` (ADR-0014).
    ///
    /// Wires both halves at once: the registry **ranks** against the graph, and
    /// the trace sink is decorated with a learner that **grows** it from
    /// search-then-invoke pairs. Pass the same `IntentGraph` to the tool and
    /// skill registries so both learn into one set of clusters.
    ///
    /// Ranking changes only where the graph has evidence: a query matching no
    /// cluster returns exactly what it would have without one. Note that with a
    /// graph attached `SearchHit.score` becomes a fusion score rather than a raw
    /// BM25 score — only ordering is comparable, as with hybrid search.
    #[napi]
    pub fn enable_adaptive_ranking(&mut self, graph: &IntentGraph) -> napi::Result<()> {
        let handle = graph.inner.clone();
        let inner_sink = self.base_sink.clone();
        let learner = Arc::new(UsageLearner::new(handle.clone(), inner_sink));
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(learner);
        registry.set_intent_graph(Some(handle.clone()));
        drop(registry);
        self.graph = Some(handle);
        Ok(())
    }

    /// Turn adaptive usage ranking off: ranking returns to the base engine and
    /// the graph stops growing. The graph itself is untouched, so re-enabling
    /// resumes from what it already learned.
    #[napi]
    pub fn disable_adaptive_ranking(&mut self) -> napi::Result<()> {
        let inner_sink = self.base_sink.clone();
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(inner_sink);
        registry.set_intent_graph(None);
        drop(registry);
        self.graph = None;
        Ok(())
    }

    /// Re-embed the intent graph's members under the current model and replace
    /// its centroids — call after changing the embedding model. Preserves
    /// members, support, and edges.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn rebuild_intent_graph(&self) -> AsyncTask<ToolEmbeddingTask> {
        AsyncTask::new(ToolEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::RebuildIntentGraph,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Whether adaptive usage ranking is contributing, paused (model changed),
    /// or inactive — see `AdaptiveRankingStatus`.
    #[napi]
    pub fn adaptive_ranking_status(&self) -> napi::Result<AdaptiveRankingStatus> {
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("tool registry lock poisoned"))?;
        Ok(map_status(registry.adaptive_ranking_status()))
    }

    /// Drain captured envelopes from the active sink. Returns `[]` unless the
    /// active sink is "memory".
    #[napi]
    pub fn drain_trace_events(&self) -> Vec<Value> {
        let Some(sink) = self.memory_sink.as_ref() else {
            return Vec::new();
        };
        sink.drain()
            .into_iter()
            .filter_map(|env| serde_json::to_value(&env).ok())
            .collect()
    }
}

pub struct FactEmbeddingTask {
    inner: Arc<RwLock<core::FactRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    operation: EmbeddingOperation,
    _permit: DenseOperationPermit,
}

pub struct FactSearchTask {
    inner: Arc<RwLock<core::FactRegistry>>,
    dense_gate: Option<Arc<Mutex<()>>>,
    query: String,
    top_k: u32,
    origin: String,
    method: String,
    _permit: Option<DenseOperationPermit>,
}

impl Task for FactEmbeddingTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let _dense = self
            .dense_gate
            .lock()
            .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("fact registry lock poisoned"))?;
        match self.operation {
            EmbeddingOperation::Build => registry.build_embeddings(),
            EmbeddingOperation::Rebuild => registry.rebuild_embeddings(),
            // Facts have no usage-ranking intent graph; the fact binding never
            // constructs this operation.
            EmbeddingOperation::RebuildIntentGraph => {
                return Err(napi::Error::from_reason(
                    "fact registry has no intent graph to rebuild",
                ));
            }
        }
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for FactSearchTask {
    type Output = Vec<FactHit>;
    type JsValue = Vec<FactHit>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let parsed_origin = match self.origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        let parsed_method: SearchMethod =
            self.method
                .parse()
                .map_err(|e: ratel_ai_core::ParseSearchMethodError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        let _dense = self
            .dense_gate
            .as_ref()
            .map(|gate| {
                gate.lock()
                    .map_err(|_| napi::Error::from_reason("dense operation mutex poisoned"))
            })
            .transpose()?;
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("fact registry lock poisoned"))?;
        registry
            .search_with_method(
                &self.query,
                self.top_k as usize,
                parsed_origin,
                parsed_method,
            )
            .map(|hits| {
                hits.into_iter()
                    .map(|hit| FactHit {
                        fact_id: hit.fact_id,
                        score: hit.score as f64,
                    })
                    .collect()
            })
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

/// Constant grounding content the agent should have on hand — a barbershop's
/// address and hours, a brand's voice. The push-path twin of `Skill`: name,
/// description, and tags are indexed for ranking (the retrieval-gated tier);
/// the `body` is the injected content, never indexed. `pin` splits the tiers.
#[napi(object)]
pub struct Fact {
    /// Unique id, the registry key. Re-registering an existing id replaces the
    /// entry in place.
    pub id: String,
    /// Human-readable name; indexed for ranking both whole and split on
    /// `snake_case`/`camelCase` boundaries.
    pub name: String,
    /// What the fact is about — the main ranking signal (not the content itself).
    pub description: String,
    /// Author-declared labels and task phrases; indexed for ranking. Optional
    /// (defaults to `[]`).
    pub tags: Option<Vec<String>>,
    /// Free-form, non-indexed context for higher layers — e.g.
    /// `{ stacks: ["react"] }` for the push ranker to boost by project context.
    pub metadata: Option<HashMap<String, Vec<String>>>,
    /// The content injected into the context on grounding — the payload, never
    /// indexed for ranking. Optional (defaults to `""`).
    pub body: Option<String>,
    /// `"always"` (always-on) or `"retrieved"` (retrieval-gated). Optional
    /// (defaults to `"retrieved"`); an unknown value throws at registration.
    pub pin: Option<String>,
}

/// One ranked fact from a registry search, best-first — the fact twin of
/// `SkillHit`, with the same score semantics per method.
#[napi(object)]
pub struct FactHit {
    /// Id of the matched fact, as registered.
    pub fact_id: String,
    /// Relevance score; higher is better, ties break by id ascending. Scale
    /// depends on the method (BM25 / cosine / RRF), as on `SearchHit.score`.
    pub score: f64,
}

/// Convert a binding [`Fact`] to a core one, parsing the `pin` string (default
/// `"retrieved"`). An unknown pin value throws — validated at the boundary.
fn core_fact(fact: Fact) -> napi::Result<core::Fact> {
    let pin = fact
        .pin
        .as_deref()
        .unwrap_or("retrieved")
        .parse::<core::PinMode>()
        .map_err(|e: core::ParsePinModeError| napi::Error::from_reason(e.to_string()))?;
    Ok(core::Fact {
        id: fact.id,
        name: fact.name,
        description: fact.description,
        tags: fact.tags.unwrap_or_default(),
        metadata: fact.metadata.unwrap_or_default(),
        body: fact.body.unwrap_or_default(),
        pin,
    })
}

/// Node binding over the `ratel-ai-core` fact registry — the fact twin of
/// `SkillRegistry`, ranking registered facts against a natural-language query.
/// Fact bodies are stored but never indexed; the SDK's `FactCatalog` injects
/// them on the grounding path.
#[napi]
pub struct FactRegistry {
    inner: Arc<RwLock<core::FactRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    pending_dense: Arc<AtomicUsize>,
    memory_sink: Option<Arc<MemorySink>>,
}

#[napi]
impl FactRegistry {
    /// Create an empty registry with a no-op trace sink.
    #[napi(constructor)]
    pub fn new(embedding: Option<EmbeddingConfig>) -> napi::Result<Self> {
        let inner = match resolve_embedding(embedding)? {
            Some(model) => core::FactRegistry::with_embedding(model),
            None => core::FactRegistry::new(),
        };
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            dense_gate: Arc::new(Mutex::new(())),
            pending_dense: Arc::new(AtomicUsize::new(0)),
            memory_sink: None,
        })
    }

    /// Index a fact, or replace one in place if its id is already registered.
    /// Omitted optional fields default to empty (`tags`/`metadata`), `""`
    /// (`body`), and `"retrieved"` (`pin`). An unknown `pin` throws.
    #[napi]
    pub fn register(&self, fact: Fact) -> napi::Result<()> {
        let core_fact = core_fact(fact)?;
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.register(core_fact);
        Ok(())
    }

    /// Index a batch under one registry write lock. An unknown `pin` on any
    /// item throws before any of the batch is indexed.
    #[napi]
    pub fn register_many(&self, facts: Vec<Fact>) -> napi::Result<()> {
        let core_facts = facts
            .into_iter()
            .map(core_fact)
            .collect::<napi::Result<Vec<_>>>()?;
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        for fact in core_facts {
            registry.register(fact);
        }
        Ok(())
    }

    /// Lexical BM25 search over facts — see `ToolRegistry.search` for the
    /// contract (best-first, ties by id, infallible, traced as `"direct"`).
    #[napi]
    pub fn search(&self, query: String, top_k: u32) -> Vec<FactHit> {
        self.inner
            .read()
            .expect("fact registry lock poisoned")
            .search(&query, top_k as usize)
            .into_iter()
            .map(|hit| FactHit {
                fact_id: hit.fact_id,
                score: hit.score as f64,
            })
            .collect()
    }

    /// BM25 search with an explicit origin — see `ToolRegistry.searchWithOrigin`.
    #[napi]
    pub fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<FactHit> {
        let parsed = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        self.inner
            .read()
            .expect("fact registry lock poisoned")
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| FactHit {
                fact_id: hit.fact_id,
                score: hit.score as f64,
            })
            .collect()
    }

    /// Search with an explicit method — see [`ToolRegistry::search_with_method`].
    #[napi]
    pub fn search_with_method(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> napi::Result<Vec<FactHit>> {
        let parsed_origin = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        let parsed_method: SearchMethod =
            method
                .parse()
                .map_err(|e: ratel_ai_core::ParseSearchMethodError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        if !matches!(parsed_method, SearchMethod::Bm25) {
            return Err(napi::Error::from_reason(
                "semantic and hybrid search are asynchronous; use searchWithMethodAsync() or FactCatalog.searchAsync()",
            ));
        }
        let hits = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("fact registry lock poisoned"))?
            .search_with_method(&query, top_k as usize, parsed_origin, parsed_method)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
        Ok(hits
            .into_iter()
            .map(|hit| FactHit {
                fact_id: hit.fact_id,
                score: hit.score as f64,
            })
            .collect())
    }

    /// Search on a libuv worker. Supports BM25, semantic, and hybrid methods.
    #[napi(ts_return_type = "Promise<Array<FactHit>>")]
    pub fn search_with_method_async(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> AsyncTask<FactSearchTask> {
        let is_dense = matches!(method.as_str(), "semantic" | "dense" | "hybrid");
        AsyncTask::new(FactSearchTask {
            inner: self.inner.clone(),
            dense_gate: is_dense.then(|| self.dense_gate.clone()),
            query,
            top_k,
            origin,
            method,
            _permit: is_dense.then(|| DenseOperationPermit::new(self.pending_dense.clone())),
        })
    }

    /// See `ToolRegistry.build_embeddings`.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn build_embeddings(&self) -> AsyncTask<FactEmbeddingTask> {
        AsyncTask::new(FactEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::Build,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Recompute the full fact corpus and atomically replace the dense cache.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn rebuild_embeddings(&self) -> AsyncTask<FactEmbeddingTask> {
        AsyncTask::new(FactEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::Rebuild,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Record a custom event on the local trace stream — see
    /// `ToolRegistry.recordEvent`. The `fact_inject` / `fact_inject_skip`
    /// grounding events ride this. Throws on an object that doesn't parse as a
    /// known trace event.
    #[napi]
    pub fn record_event(&self, event: Value) -> napi::Result<()> {
        let event: TraceEvent = serde_json::from_value(event)
            .map_err(|e| napi::Error::from_reason(format!("invalid trace event: {e}")))?;
        self.inner
            .read()
            .map_err(|_| napi::Error::from_reason("fact registry lock poisoned"))?
            .record_event(event);
        Ok(())
    }

    /// Replace the trace sink — see `ToolRegistry.setTraceSink`.
    #[napi]
    pub fn set_trace_sink(&mut self, config: TraceSinkConfig) -> napi::Result<()> {
        let (sink, memory) = build_trace_sink(config)?;
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(sink);
        drop(registry);
        self.memory_sink = memory;
        Ok(())
    }

    /// Drain captured envelopes from the active sink. Returns `[]` unless the
    /// active sink is "memory".
    #[napi]
    pub fn drain_trace_events(&self) -> Vec<Value> {
        let Some(sink) = self.memory_sink.as_ref() else {
            return Vec::new();
        };
        sink.drain()
            .into_iter()
            .filter_map(|env| serde_json::to_value(&env).ok())
            .collect()
    }
}

/// A reusable playbook: instructions the agent *reads* and follows, in
/// contrast to a `Tool` it executes. Name, description, and tags are indexed
/// for ranking; the `body` is the dispatch payload, deliberately excluded from
/// the index so it can't drown the description's term weights.
#[napi(object)]
pub struct Skill {
    /// Unique id, the registry key. Re-registering an existing id replaces the
    /// entry in place.
    pub id: String,
    /// Human-readable name; indexed for ranking both whole and split on
    /// `snake_case`/`camelCase` boundaries.
    pub name: String,
    /// What the skill covers and when to reach for it — the main ranking signal.
    pub description: String,
    /// Author-declared labels and task phrases ("frontend", "login form");
    /// indexed for ranking. Optional (defaults to `[]`) — a minimal
    /// `Skill(id, name, description)` is valid, in parity with the Python SDK.
    pub tags: Option<Vec<String>>,
    /// Ids of tools this skill's instructions call; surfaced into the
    /// `search_capabilities` tools bucket — not indexed as query terms.
    pub tools: Option<Vec<String>>,
    /// Free-form, non-indexed context for higher layers — e.g.
    /// `{ stacks: ["react"] }` for the push ranker to boost by project context.
    pub metadata: Option<HashMap<String, Vec<String>>>,
    /// The full instructions (Markdown) returned on load — the dispatch
    /// payload, never indexed for ranking.
    /// Optional (defaults to `""`) — parity with the Python SDK's default body.
    pub body: Option<String>,
}

/// One ranked skill from a registry search, best-first — the skill twin of
/// `SearchHit`, with the same score semantics per method.
#[napi(object)]
pub struct SkillHit {
    /// Id of the matched skill, as registered.
    pub skill_id: String,
    /// Relevance score; scale depends on the method and on `fused`, as on
    /// `SearchHit.score`. Order by `rank`, branch on `fused`.
    pub score: f64,
    /// 0-based position in this result list — as on `SearchHit.rank`.
    pub rank: u32,
    /// `true` when `score` is an RRF score — as on `SearchHit.fused`.
    pub fused: bool,
}

/// What a `SkillRegistry.replaceAll` changed, counted by id. `updated` covers
/// any field edit (including a body-only rewrite); `unchanged` ids are identical
/// to what was registered and keep their cached embedding.
#[napi(object)]
pub struct ReplaceOutcome {
    /// Ids in the new corpus that were not in the old one.
    pub added: u32,
    /// Ids in the old corpus that are absent from the new one.
    pub removed: u32,
    /// Ids present in both whose content differs in any field.
    pub updated: u32,
    /// Ids present in both with identical content.
    pub unchanged: u32,
}

/// Node binding over the `ratel-ai-core` skill registry — the skill twin of
/// `ToolRegistry`, ranking registered skills against a natural-language query.
/// Skill bodies are stored but never indexed; fetch them a layer up (the SDK's
/// `SkillCatalog.invoke`).
#[napi]
pub struct SkillRegistry {
    inner: Arc<RwLock<core::SkillRegistry>>,
    dense_gate: Arc<Mutex<()>>,
    pending_dense: Arc<AtomicUsize>,
    memory_sink: Option<Arc<MemorySink>>,
    /// The current undecorated sink, whatever its kind. Retained so
    /// enable/disable adaptive ranking can re-wrap or restore it; rebuilding
    /// from `memory_sink` alone would drop a configured jsonl sink to noop.
    base_sink: Arc<dyn core::TraceSink>,
    /// Retained so `setTraceSink` can re-wrap the new sink in a learner —
    /// otherwise changing sinks would silently switch learning off.
    graph: Option<Arc<RwLock<core::IntentGraph>>>,
}

#[napi]
impl SkillRegistry {
    /// Create an empty registry with a no-op trace sink.
    #[napi(constructor)]
    pub fn new(embedding: Option<EmbeddingConfig>) -> napi::Result<Self> {
        let inner = match resolve_embedding(embedding)? {
            Some(model) => core::SkillRegistry::with_embedding(model),
            None => core::SkillRegistry::new(),
        };
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            dense_gate: Arc::new(Mutex::new(())),
            pending_dense: Arc::new(AtomicUsize::new(0)),
            memory_sink: None,
            base_sink: Arc::new(NoopSink),
            graph: None,
        })
    }

    /// Index a skill, or replace one in place if its id is already registered.
    /// Omitted optional fields default to empty (`tags`/`tools`/`metadata`)
    /// and `""` (`body`). See `ToolRegistry.register`.
    #[napi]
    pub fn register(&self, skill: Skill) -> napi::Result<()> {
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.register(core::Skill {
            id: skill.id,
            name: skill.name,
            description: skill.description,
            tags: skill.tags.unwrap_or_default(),
            tools: skill.tools.unwrap_or_default(),
            metadata: skill.metadata.unwrap_or_default(),
            body: skill.body.unwrap_or_default(),
        });
        Ok(())
    }

    /// Index a batch under one registry write lock.
    #[napi]
    pub fn register_many(&self, skills: Vec<Skill>) -> napi::Result<()> {
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        for skill in skills {
            registry.register(core::Skill {
                id: skill.id,
                name: skill.name,
                description: skill.description,
                tags: skill.tags.unwrap_or_default(),
                tools: skill.tools.unwrap_or_default(),
                metadata: skill.metadata.unwrap_or_default(),
                body: skill.body.unwrap_or_default(),
            });
        }
        Ok(())
    }

    /// Replace the whole corpus under one registry write lock: ids absent from
    /// `skills` are removed, the rest are added or updated. Embeds nothing —
    /// call `buildEmbeddings()` after on a semantic/hybrid registry, which then
    /// embeds only what this replace invalidated. Rejects while a dense
    /// operation owns the registry, exactly as `registerMany` does.
    #[napi]
    pub fn replace_all(&self, skills: Vec<Skill>) -> napi::Result<ReplaceOutcome> {
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        let outcome = registry.replace_all(
            skills
                .into_iter()
                .map(|skill| core::Skill {
                    id: skill.id,
                    name: skill.name,
                    description: skill.description,
                    tags: skill.tags.unwrap_or_default(),
                    tools: skill.tools.unwrap_or_default(),
                    metadata: skill.metadata.unwrap_or_default(),
                    body: skill.body.unwrap_or_default(),
                })
                .collect(),
        );
        Ok(ReplaceOutcome {
            added: outcome.added as u32,
            removed: outcome.removed as u32,
            updated: outcome.updated as u32,
            unchanged: outcome.unchanged as u32,
        })
    }

    /// Lexical BM25 search over skills — see `ToolRegistry.search` for the
    /// contract (best-first, ties by id, infallible, traced as `"direct"`).
    #[napi]
    pub fn search(&self, query: String, top_k: u32) -> Vec<SkillHit> {
        self.inner
            .read()
            .expect("skill registry lock poisoned")
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

    /// BM25 search with an explicit origin — see `ToolRegistry.searchWithOrigin`.
    #[napi]
    pub fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<SkillHit> {
        let parsed = parse_origin(origin.as_str());
        self.inner
            .read()
            .expect("skill registry lock poisoned")
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
                score: hit.score as f64,
                rank: hit.rank,
                fused: hit.fused,
            })
            .collect()
    }

    /// Search with an explicit method — see [`ToolRegistry::search_with_method`].
    #[napi]
    pub fn search_with_method(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> napi::Result<Vec<SkillHit>> {
        let parsed_origin = parse_origin(origin.as_str());
        let parsed_method: SearchMethod =
            method
                .parse()
                .map_err(|e: ratel_ai_core::ParseSearchMethodError| {
                    napi::Error::from_reason(e.to_string())
                })?;
        if !matches!(parsed_method, SearchMethod::Bm25) {
            return Err(napi::Error::from_reason(
                "semantic and hybrid search are asynchronous; use searchWithMethodAsync() or SkillCatalog.searchAsync()",
            ));
        }
        let hits = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?
            .search_with_method(&query, top_k as usize, parsed_origin, parsed_method)
            .map_err(|e| napi::Error::from_reason(e.to_string()))?;
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

    /// Search on a libuv worker. Supports BM25, semantic, and hybrid methods.
    #[napi(ts_return_type = "Promise<Array<SkillHit>>")]
    pub fn search_with_method_async(
        &self,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> AsyncTask<SkillSearchTask> {
        let is_dense = matches!(method.as_str(), "semantic" | "dense" | "hybrid");
        AsyncTask::new(SkillSearchTask {
            inner: self.inner.clone(),
            dense_gate: is_dense.then(|| self.dense_gate.clone()),
            query,
            top_k,
            origin,
            method,
            _permit: is_dense.then(|| DenseOperationPermit::new(self.pending_dense.clone())),
        })
    }

    /// See `ToolRegistry.build_embeddings`.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn build_embeddings(&self) -> AsyncTask<SkillEmbeddingTask> {
        AsyncTask::new(SkillEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::Build,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Recompute the full skill corpus and atomically replace the dense cache.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn rebuild_embeddings(&self) -> AsyncTask<SkillEmbeddingTask> {
        AsyncTask::new(SkillEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::Rebuild,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// See `ToolRegistry.build_embedding_artifact`.
    #[napi(ts_return_type = "Promise<Buffer>")]
    pub fn build_embedding_artifact(&self) -> AsyncTask<SkillBuildArtifactTask> {
        AsyncTask::new(SkillBuildArtifactTask {
            inner: self.inner.clone(),
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// See `ToolRegistry.warm_embeddings_from_artifact`.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn warm_embeddings_from_artifact(
        &self,
        bytes: Buffer,
        on_miss: String,
    ) -> AsyncTask<SkillWarmArtifactTask> {
        AsyncTask::new(SkillWarmArtifactTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            bytes,
            on_miss,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Record a custom event on the local trace stream — see
    /// `ToolRegistry.recordEvent`. Throws on an object that doesn't parse as a
    /// known trace event.
    #[napi]
    pub fn record_event(&self, event: Value) -> napi::Result<()> {
        let event: TraceEvent = serde_json::from_value(event)
            .map_err(|e| napi::Error::from_reason(format!("invalid trace event: {e}")))?;
        self.inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?
            .record_event(event);
        Ok(())
    }

    /// Replace the trace sink — see `ToolRegistry.setTraceSink`.
    #[napi]
    pub fn set_trace_sink(&mut self, config: TraceSinkConfig) -> napi::Result<()> {
        let (sink, memory) = build_trace_sink(config)?;
        // Retain the raw sink so enable/disable can re-wrap or restore it —
        // rebuilding from `memory_sink` alone would drop a jsonl sink to noop.
        self.base_sink = sink.clone();
        // Re-wrap: adaptive ranking learns by decorating the sink, so replacing
        // the sink outright would quietly stop learning.
        let sink = match &self.graph {
            Some(graph) => {
                Arc::new(UsageLearner::new(graph.clone(), sink)) as Arc<dyn core::TraceSink>
            }
            None => sink,
        };
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(sink);
        drop(registry);
        self.memory_sink = memory;
        Ok(())
    }

    /// Turn on adaptive usage ranking against `graph` (ADR-0014).
    ///
    /// Wires both halves at once: the registry **ranks** against the graph, and
    /// the trace sink is decorated with a learner that **grows** it from
    /// search-then-invoke pairs. Pass the same `IntentGraph` to the tool and
    /// skill registries so both learn into one set of clusters.
    ///
    /// Ranking changes only where the graph has evidence: a query matching no
    /// cluster returns exactly what it would have without one. Note that with a
    /// graph attached `SearchHit.score` becomes a fusion score rather than a raw
    /// BM25 score — only ordering is comparable, as with hybrid search.
    #[napi]
    pub fn enable_adaptive_ranking(&mut self, graph: &IntentGraph) -> napi::Result<()> {
        let handle = graph.inner.clone();
        let inner_sink = self.base_sink.clone();
        let learner = Arc::new(UsageLearner::new(handle.clone(), inner_sink));
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(learner);
        registry.set_intent_graph(Some(handle.clone()));
        drop(registry);
        self.graph = Some(handle);
        Ok(())
    }

    /// Turn adaptive usage ranking off: ranking returns to the base engine and
    /// the graph stops growing. The graph itself is untouched, so re-enabling
    /// resumes from what it already learned.
    #[napi]
    pub fn disable_adaptive_ranking(&mut self) -> napi::Result<()> {
        let inner_sink = self.base_sink.clone();
        let mut registry = write_registry(&self.inner, &self.pending_dense)?;
        registry.set_trace_sink(inner_sink);
        registry.set_intent_graph(None);
        drop(registry);
        self.graph = None;
        Ok(())
    }

    /// Re-embed the intent graph's members under the current model and replace
    /// its centroids — call after changing the embedding model. Preserves
    /// members, support, and edges.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn rebuild_intent_graph(&self) -> AsyncTask<SkillEmbeddingTask> {
        AsyncTask::new(SkillEmbeddingTask {
            inner: self.inner.clone(),
            dense_gate: self.dense_gate.clone(),
            operation: EmbeddingOperation::RebuildIntentGraph,
            _permit: DenseOperationPermit::new(self.pending_dense.clone()),
        })
    }

    /// Whether adaptive usage ranking is contributing, paused (model changed),
    /// or inactive — see `AdaptiveRankingStatus`.
    #[napi]
    pub fn adaptive_ranking_status(&self) -> napi::Result<AdaptiveRankingStatus> {
        let registry = self
            .inner
            .read()
            .map_err(|_| napi::Error::from_reason("skill registry lock poisoned"))?;
        Ok(map_status(registry.adaptive_ranking_status()))
    }

    /// Drain captured envelopes from the active sink. Returns `[]` unless the
    /// active sink is "memory".
    #[napi]
    pub fn drain_trace_events(&self) -> Vec<Value> {
        let Some(sink) = self.memory_sink.as_ref() else {
            return Vec::new();
        };
        sink.drain()
            .into_iter()
            .filter_map(|env| serde_json::to_value(&env).ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every documented `SearchOrigin` wire value must reach its own variant.
    ///
    /// This is the real guard behind `parse_origin`. The `_ => Origin::Direct`
    /// fallback is deliberate — an unknown string from an older or newer SDK
    /// must not fail an infallible search — but it also means a newly added
    /// `Origin` variant would be silently mapped to `direct` with no compile
    /// error anywhere. Asserting each value maps somewhere distinct turns that
    /// silent degrade into a failing test.
    #[test]
    fn every_origin_wire_value_maps_to_its_own_variant() {
        assert_eq!(parse_origin("direct"), Origin::Direct);
        assert_eq!(parse_origin("agent"), Origin::Agent);
        assert_eq!(parse_origin("baseline"), Origin::Baseline);
    }

    #[test]
    fn an_unrecognized_origin_degrades_to_direct_rather_than_failing() {
        // Version skew between the SDK and the native must not break search.
        assert_eq!(parse_origin("from_the_future"), Origin::Direct);
        assert_eq!(parse_origin(""), Origin::Direct);
    }
}
