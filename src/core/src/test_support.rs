//! Shared test helpers for artifact/warm embedder stubs (crate-internal, tests only).

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::dense_cache::Embeddable;
use crate::embedding::{Embedded, Embedder, EmbedderError};
use crate::embedding_artifact::{ArtifactEntryKind, build_artifact};

/// L2-normalizes a fixed-dimension vector for deterministic artifact tests.
pub(crate) fn unit<const N: usize>(values: [f32; N]) -> Vec<f32> {
    let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    values.iter().map(|x| x / norm).collect()
}

/// Builds artifacts with a fixed identity and explicit per-item vectors.
pub(crate) struct ArtifactBuildStub {
    fingerprint: String,
    vectors: Vec<Vec<f32>>,
}

impl ArtifactBuildStub {
    pub(crate) fn new(fingerprint: impl Into<String>, vectors: Vec<Vec<f32>>) -> Self {
        Self {
            fingerprint: fingerprint.into(),
            vectors,
        }
    }
}

impl Embedder for ArtifactBuildStub {
    fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        unreachable!("artifact build uses batch")
    }
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        unreachable!("artifact build uses batch")
    }
    fn embed_batch_with_identity(
        &self,
        texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        assert_eq!(texts.len(), self.vectors.len());
        Ok(Embedded {
            value: self.vectors.clone(),
            fingerprint: self.fingerprint.clone(),
        })
    }
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

/// Resolves identity without inference — panics if any embed path is hit.
pub(crate) struct PanicOnEmbedStub {
    fingerprint: String,
}

impl PanicOnEmbedStub {
    pub(crate) fn new(fingerprint: impl Into<String>) -> Self {
        Self {
            fingerprint: fingerprint.into(),
        }
    }
}

impl Embedder for PanicOnEmbedStub {
    fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        panic!("embed_doc must not be called")
    }
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        panic!("embed_query must not be called")
    }
    fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        panic!("embed_batch must not be called")
    }
    fn embed_batch_with_identity(
        &self,
        _texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        panic!("embed_batch_with_identity must not be called")
    }
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

/// Identity matches the artifact; every embed path fails (post-warm Embed policy).
pub(crate) struct FailOnEmbedStub {
    fingerprint: String,
}

impl FailOnEmbedStub {
    pub(crate) fn new(fingerprint: impl Into<String>) -> Self {
        Self {
            fingerprint: fingerprint.into(),
        }
    }
}

impl Embedder for FailOnEmbedStub {
    fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        Err(EmbedderError::Inference {
            source: "forced embed failure".into(),
        })
    }
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
        Err(EmbedderError::Inference {
            source: "forced embed failure".into(),
        })
    }
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

/// Counts embeds under a fixed fingerprint (warm + extend chains).
pub(crate) struct FpCountingEmbedder {
    fingerprint: String,
    doc_calls: AtomicUsize,
    vec_for: fn(&str) -> Vec<f32>,
}

impl FpCountingEmbedder {
    pub(crate) fn new(fingerprint: &str, vec_for: fn(&str) -> Vec<f32>) -> Self {
        Self {
            fingerprint: fingerprint.into(),
            doc_calls: AtomicUsize::new(0),
            vec_for,
        }
    }

    pub(crate) fn docs(&self) -> usize {
        self.doc_calls.load(Ordering::SeqCst)
    }
}

impl Embedder for FpCountingEmbedder {
    fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.doc_calls.fetch_add(1, Ordering::SeqCst);
        Ok((self.vec_for)(text))
    }
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok((self.vec_for)(text))
    }
    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

pub(crate) fn build_test_artifact<'a, T: Embeddable + 'a>(
    kind: ArtifactEntryKind,
    items: impl IntoIterator<Item = &'a T>,
    fingerprint: &str,
    vectors: Vec<Vec<f32>>,
) -> Vec<u8> {
    build_artifact(kind, items, &ArtifactBuildStub::new(fingerprint, vectors)).unwrap()
}
