#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

use napi::bindgen_prelude::AsyncTask;
use napi::{Env, Task};
use ratel_ai_core as core;
use ratel_ai_core::{
    EmbeddingModel, EmbeddingSpec, JsonlSink, MemorySink, NoopSink, Origin, SearchMethod,
    TraceEvent,
};
use serde_json::Value;

/// A constructed sink plus the `MemorySink` handle when the kind is `"memory"`
/// (so the owner can drain it later).
type BuiltTraceSink = (Arc<dyn core::TraceSink>, Option<Arc<MemorySink>>);

const REGISTRY_BUSY_MESSAGE: &str =
    "registry busy; await the active operation before registering more items";

#[derive(Clone, Copy)]
enum EmbeddingOperation {
    Build,
    Rebuild,
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
        }
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for ToolSearchTask {
    type Output = Vec<SearchHit>;
    type JsValue = Vec<SearchHit>;

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
        }
        .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

impl Task for SkillSearchTask {
    type Output = Vec<SkillHit>;
    type JsValue = Vec<SkillHit>;

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
    /// Relevance score; higher is better, ties break by id ascending. The scale
    /// depends on the method: raw BM25 (unbounded) for `"bm25"`, cosine
    /// similarity for `"semantic"`, Reciprocal Rank Fusion for `"hybrid"` —
    /// comparable within one result list, not across methods.
    pub score: f64,
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
            })
            .collect()
    }

    /// BM25 search with an explicit origin — `"agent"` for a call the model
    /// synthesized (capability tools), anything else counts as `"direct"`
    /// (host code). Origin only annotates the trace event; ranking is
    /// identical to `search`.
    #[napi]
    pub fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<SearchHit> {
        let parsed = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        self.inner
            .read()
            .expect("tool registry lock poisoned")
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
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
    /// Relevance score; higher is better, ties break by id ascending. Scale
    /// depends on the method (BM25 / cosine / RRF), as on `SearchHit.score`.
    pub score: f64,
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
            })
            .collect()
    }

    /// BM25 search with an explicit origin — see `ToolRegistry.searchWithOrigin`.
    #[napi]
    pub fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<SkillHit> {
        let parsed = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        self.inner
            .read()
            .expect("skill registry lock poisoned")
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
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
    ) -> napi::Result<Vec<SkillHit>> {
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
