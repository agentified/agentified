//! PyO3 binding to `ratel-ai-core` — the Python analogue of the TS SDK's NAPI
//! binding (`src/sdk/ts/native/src/lib.rs`). Pure pass-through over the public
//! API of `ratel-ai-core`; see ADR-0006 for the binding-strategy rationale and
//! ADR-0007 for the core-owned trace schema this emits into.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use ratel_ai_core as core;
use ratel_ai_core::{JsonlSink, MemorySink, NoopSink, Origin, TraceEvent};
use serde_json::Value;

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
    /// Relevance score; higher ranks first. The scale depends on the search
    /// method: raw BM25 (unbounded) for `"bm25"`, cosine similarity for
    /// `"semantic"`, reciprocal-rank-fusion for `"hybrid"` — scores from
    /// different methods are not comparable.
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl SearchHit {
    fn __repr__(&self) -> String {
        format!(
            "SearchHit(tool_id={:?}, score={})",
            self.tool_id, self.score
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
    /// Relevance score; higher ranks first. Same method-dependent scale as
    /// [`SearchHit::score`], computed against the skill corpus.
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl SkillHit {
    fn __repr__(&self) -> String {
        format!(
            "SkillHit(skill_id={:?}, score={})",
            self.skill_id, self.score
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

/// Metadata registry over `ratel-ai-core`. BM25 is exposed synchronously;
/// GIL-releasing dense primitives are private to the pure-Python async facade.
/// Executors and capability-tool / MCP layers also live above this binding.
#[pyclass]
pub struct ToolRegistry {
    inner: core::ToolRegistry,
    memory_sink: Option<Arc<MemorySink>>,
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
            })
            .collect()
    }

    /// BM25 search tagged with who initiated it: `"agent"` (a model calling a
    /// capability tool) or anything else → `"direct"` (host code). The origin
    /// only labels the emitted trace event — ranking is identical to `search`.
    fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<SearchHit> {
        let parsed = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        self.inner
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
            })
            .collect()
    }

    /// Search with an explicit method (`"bm25"` | `"semantic"` | `"hybrid"`).
    /// `bm25` is infallible; `semantic`/`hybrid` rank against the prebuilt embedding
    /// cache and raise `RuntimeError` (`EmbeddingsNotBuilt`) if it isn't built — the
    /// model loads at `_build_embeddings`, never inside a search. Private worker
    /// primitive; releases the GIL. An unknown method raises `ValueError`.
    fn _search_with_method(
        &self,
        py: Python<'_>,
        query: String,
        top_k: u32,
        origin: String,
        method: String,
    ) -> PyResult<Vec<SearchHit>> {
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
            .map(|hit| SearchHit {
                tool_id: hit.tool_id,
                score: hit.score as f64,
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

    /// Lexical BM25 search over the skill corpus — see [`ToolRegistry::search`].
    fn search(&self, query: String, top_k: u32) -> Vec<SkillHit> {
        self.inner
            .search(&query, top_k as usize)
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
                score: hit.score as f64,
            })
            .collect()
    }

    /// BM25 search tagged with who initiated it — see
    /// [`ToolRegistry::search_with_origin`].
    fn search_with_origin(&self, query: String, top_k: u32, origin: String) -> Vec<SkillHit> {
        let parsed = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
        self.inner
            .search_with_origin(&query, top_k as usize, parsed)
            .into_iter()
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
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
    ) -> PyResult<Vec<SkillHit>> {
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
            .map(|hit| SkillHit {
                skill_id: hit.skill_id,
                score: hit.score as f64,
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
        let parsed = match origin.as_str() {
            "agent" => Origin::Agent,
            _ => Origin::Direct,
        };
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
    m.add_class::<ToolRegistry>()?;
    m.add_class::<SearchHit>()?;
    m.add_class::<SkillRegistry>()?;
    m.add_class::<SkillHit>()?;
    m.add_class::<FactRegistry>()?;
    m.add_class::<FactHit>()?;
    m.add("EmbedderError", m.py().get_type::<EmbedderError>())?;
    m.add(
        "DimensionMismatchError",
        m.py().get_type::<DimensionMismatchError>(),
    )?;
    Ok(())
}
