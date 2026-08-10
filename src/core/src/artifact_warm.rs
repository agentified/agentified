//! Registry-level policy and errors for warming dense embeddings from a
//! build-time artifact — shared by [`crate::ToolRegistry`] and
//! [`crate::SkillRegistry`].

use crate::dense_cache::WarmError;
use crate::embedding::EmbedderError;

/// What to do when some corpus ids are not covered by the artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnArtifactMiss {
    /// Fail if any corpus id was not reused from the artifact.
    Error,
    /// Call [`crate::ToolRegistry::build_embeddings`] /
    /// [`crate::SkillRegistry::build_embeddings`] to embed only the still-missing ids.
    Embed,
}

/// Failure of [`crate::ToolRegistry::warm_embeddings_from_artifact`] /
/// [`crate::SkillRegistry::warm_embeddings_from_artifact`].
#[derive(Debug, Clone)]
pub enum ArtifactWarmError {
    /// Parse / kind / model-mismatch from [`crate::dense_cache::DenseCache::warm_from_artifact`].
    Warm(WarmError),
    /// Policy Error: corpus ids not covered by the artifact.
    Incomplete {
        /// Corpus ids that were not reused from the artifact.
        missing: Vec<String>,
    },
    /// Policy Embed: failure from the follow-up [`build_embeddings`](crate::ToolRegistry::build_embeddings).
    Embedder(EmbedderError),
}

impl std::fmt::Display for ArtifactWarmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactWarmError::Warm(e) => write!(f, "{e}"),
            ArtifactWarmError::Incomplete { missing } => write!(
                f,
                "embedding artifact incomplete for the current corpus: {} id(s) missing ({})",
                missing.len(),
                missing.join(", ")
            ),
            ArtifactWarmError::Embedder(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ArtifactWarmError {}

impl From<WarmError> for ArtifactWarmError {
    fn from(value: WarmError) -> Self {
        Self::Warm(value)
    }
}

impl From<EmbedderError> for ArtifactWarmError {
    fn from(value: EmbedderError) -> Self {
        Self::Embedder(value)
    }
}
