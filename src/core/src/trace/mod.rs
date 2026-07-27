//! Trace events emitted from every layer of Ratel — the substrate for the
//! inspector, the suggestion analyzer, the reranker training, and the optional
//! self-hosted consolidation server. See ADR-0007 for the schema-ownership
//! and reliability story.

mod event;
mod sink;

pub use event::{
    ChurnKind, EmbedderLoadStatus, FactHitTrace, FactInjectReason, Origin, SearchHitTrace,
    SearchStage, SkillHitTrace, TraceEnvelope, TraceEvent,
};
pub use sink::{JsonlSink, MemorySink, NoopSink, TraceSink};
