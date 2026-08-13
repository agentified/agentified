//! Registry-level policy and errors for warming dense embeddings from a
//! build-time artifact — shared by [`crate::ToolRegistry`] and
//! [`crate::SkillRegistry`].

use std::fmt;
use std::str::FromStr;

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

impl OnArtifactMiss {
    /// Stable SDK identifier: `"error"` or `"embed"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            OnArtifactMiss::Error => "error",
            OnArtifactMiss::Embed => "embed",
        }
    }
}

/// Rejected [`OnArtifactMiss`] string from the SDK binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseOnArtifactMissError(pub String);

impl fmt::Display for ParseOnArtifactMissError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown on-artifact-miss policy {:?} (expected \"error\" or \"embed\")",
            self.0
        )
    }
}

impl std::error::Error for ParseOnArtifactMissError {}

impl FromStr for OnArtifactMiss {
    type Err = ParseOnArtifactMissError;

    /// Parse the SDK identifier: `"error"` or `"embed"`.
    ///
    /// # Errors
    ///
    /// Any other string is a [`ParseOnArtifactMissError`] naming the rejected
    /// input.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "error" => Ok(OnArtifactMiss::Error),
            "embed" => Ok(OnArtifactMiss::Embed),
            other => Err(ParseOnArtifactMissError(other.to_string())),
        }
    }
}

/// Failure of [`crate::ToolRegistry::warm_embeddings_from_artifact`] /
/// [`crate::SkillRegistry::warm_embeddings_from_artifact`].
#[derive(Debug, Clone)]
pub enum ArtifactWarmError {
    /// Parse, RAT1 header mismatch, or wrapped embedder error from warm.
    Warm(WarmError),
    /// Policy Error: corpus ids not covered by the artifact.
    Incomplete {
        /// Corpus ids that were not reused from the artifact.
        missing: Vec<String>,
    },
    /// Policy Embed: failure from the follow-up [`build_embeddings`](crate::ToolRegistry::build_embeddings).
    Embedder(EmbedderError),
}

impl fmt::Display for ArtifactWarmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_artifact_miss_round_trips_through_str() {
        for policy in [OnArtifactMiss::Error, OnArtifactMiss::Embed] {
            assert_eq!(policy.as_str().parse::<OnArtifactMiss>().unwrap(), policy);
        }
    }

    #[test]
    fn on_artifact_miss_rejects_unknown() {
        assert!("reuse".parse::<OnArtifactMiss>().is_err());
    }

    #[test]
    fn incomplete_display_lists_missing_ids() {
        let incomplete = ArtifactWarmError::Incomplete {
            missing: vec!["a".into(), "b".into()],
        };
        let message = incomplete.to_string();
        assert!(message.contains("embedding artifact incomplete for the current corpus"));
        assert!(message.contains("a, b"));
    }
}
