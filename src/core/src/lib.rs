//! Tool and skill retrieval for AI agents — the Rust core of the Ratel
//! context engineering platform.
//!
//! Agents degrade when every tool definition is stuffed into the context
//! window. This crate keeps the full catalog *outside* the context and
//! retrieves only the entries relevant to the task at hand: register tools
//! and skills once, then search them per turn. The engine runs in-process;
//! BM25 and local dense retrieval need no server, while dense retrieval may
//! instead use a configured OpenAI-compatible embedding endpoint.
//!
//! # Mental model
//!
//! Two registries hold the corpus, one per capability kind:
//!
//! - [`ToolRegistry`] indexes [`Tool`]s — callable endpoints described by a
//!   name, a description, and JSON schemas.
//! - [`SkillRegistry`] indexes [`Skill`]s — reusable instruction playbooks
//!   whose body is dispatched on demand.
//!
//! Both rank a query with one of three engines, selected by [`SearchMethod`]:
//!
//! - [`SearchMethod::Bm25`] (default) — lexical BM25. Needs no model and
//!   never fails; [`ToolRegistry::search`] and [`SkillRegistry::search`] use
//!   it unconditionally.
//! - [`SearchMethod::Semantic`] — cosine similarity over dense embeddings
//!   from a configurable in-process HuggingFace/local model (default
//!   `bge-small-en-v1.5`) or OpenAI-compatible endpoint (ADR-0011/ADR-0012).
//! - [`SearchMethod::Hybrid`] — the BM25 and dense rankings fused with
//!   Reciprocal Rank Fusion.
//!
//! Semantic and hybrid searches rank against an embedding cache built by
//! [`ToolRegistry::build_embeddings`] / [`SkillRegistry::build_embeddings`];
//! a search itself never embeds the corpus and never downloads the model.
//!
//! Every register and search also emits a [`TraceEvent`] on the registry's
//! [`TraceSink`] — the local trace stream behind the inspector and usage
//! reporting (ADR-0007). The default sink is [`NoopSink`] (discard);
//! [`MemorySink`] buffers for tests and introspection, [`JsonlSink`] appends
//! to a local file.
//!
//! # Example: register and search (BM25)
//!
//! ```
//! use ratel_ai_core::{Tool, ToolRegistry};
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(Tool {
//!     id: "read_file".into(),
//!     name: "read_file".into(),
//!     description: "Read a file from disk".into(),
//!     input_schema: serde_json::json!({
//!         "properties": {
//!             "path": { "type": "string", "description": "absolute path" }
//!         }
//!     }),
//!     output_schema: serde_json::json!({}),
//! });
//! registry.register(Tool {
//!     id: "send_email".into(),
//!     name: "send_email".into(),
//!     description: "Send an email to a recipient".into(),
//!     input_schema: serde_json::json!({}),
//!     output_schema: serde_json::json!({}),
//! });
//!
//! let hits = registry.search("read a file", 5);
//! assert_eq!(hits[0].tool_id, "read_file");
//! ```
//!
//! The language SDKs (`@ratel-ai/sdk` on npm, `ratel-ai` on PyPI) bundle this
//! crate and surface the same model; the agent-facing capability tools
//! (`search_capabilities` / `invoke_tool` / `get_skill_content`) sit on top
//! of them. Design rationale lives in the repo's `docs/adr/`.

#![warn(missing_docs)]

mod dense_cache;
mod dense_search;
mod embedding;
mod embedding_config;
mod fusion;
mod indexing;
mod method;
mod search;
mod skill;
mod skill_indexing;
mod skill_registry;
mod tool;
mod tool_registry;
mod trace;
mod usage;
mod usage_learner;

pub use embedding::EmbedderError;
pub use embedding_config::{EmbeddingModel, EmbeddingSpec, Pooling};
pub use method::{ParseSearchMethodError, SearchMethod};
pub use skill::Skill;
pub use skill_registry::{SkillHit, SkillRegistry};
pub use tool::Tool;
pub use tool_registry::{AdaptiveRankingStatus, SearchHit, ToolRegistry};
pub use trace::{
    ChurnKind, EmbedderLoadStatus, JsonlSink, MemorySink, NoopSink, Origin, SearchHitTrace,
    SearchStage, SkillHitTrace, TraceEnvelope, TraceEvent, TraceSink,
};
pub use usage::{Intent, IntentGraph, IntentGraphError};
pub use usage_learner::UsageLearner;
