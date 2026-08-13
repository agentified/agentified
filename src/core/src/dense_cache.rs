//! Shared incremental dense-embedding cache backing the registries' semantic and
//! hybrid engines.
//!
//! [`crate::ToolRegistry`] and [`crate::SkillRegistry`] rank two different item
//! types (tools, skills) but embed and cosine-rank them identically: an **id-keyed
//! map** of per-item vectors, extended incrementally. This type owns that
//! structure and its operations so the two registries share one implementation
//! instead of two copies that could drift. The registries keep their own
//! trace-emitting search wrappers, since the trace event shapes differ (tool vs
//! skill). See ADR-0011.
//!
//! Keying by id (not by position) is what lets `register` replace an item in
//! place: on replace the registry calls [`DenseCache::invalidate`] to drop the
//! stale vector, and the next [`DenseCache::extend`] re-embeds that id like any
//! other missing one — so a re-registered id never leaves a stale embedding
//! behind (RAT-378). The same keying lets `replace_all` drop the vector of an
//! id that leaves the corpus outright, which no `extend` then re-embeds.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::dense_search::dense_search;
use crate::embedding::{Embedder, EmbedderError, embedder_with_telemetry};
use crate::embedding_artifact::{
    ArtifactEntry, ArtifactEntryKind, ArtifactError, build_empty_artifact, hash_projection_text,
    load_and_validate,
};
use crate::embedding_config::EmbeddingModel;
use crate::trace::{TraceEvent, TraceSink};

/// Single-input placeholder for the Endpoint identity probe.
const ARTIFACT_WARM_PROBE_TEXT: &str = "__ratel_artifact_probe__";

/// Result of [`DenseCache::warm_from_artifact`]: which corpus ids were taken from
/// the artifact and which still need embedding (or another policy) by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WarmOutcome {
    /// Corpus ids whose artifact vectors were committed.
    pub reused: Vec<String>,
    /// Corpus ids with no usable artifact entry (absent or projection_hash mismatch).
    pub missing: Vec<String>,
}

/// Failure loading or applying an embedding artifact into the dense cache.
#[derive(Debug, Clone)]
pub enum WarmError {
    /// Binary artifact failed [`crate::embedding_artifact::load_and_validate`].
    Artifact(ArtifactError),
    /// RAT1 header fingerprint does not match the configured embedder.
    ArtifactModelMismatch {
        /// Fingerprint recorded in the RAT1 header.
        artifact: String,
        /// Artifact-compat identity of the configured embedder.
        active: String,
    },
    /// Embedder load, identity probe, or dimension check failed during warm.
    Embedder(EmbedderError),
}

impl std::fmt::Display for WarmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarmError::Artifact(e) => write!(f, "{e}"),
            WarmError::ArtifactModelMismatch { artifact, active } => write!(
                f,
                "embedding artifact was built with model {artifact}, but the configured embedding model is {active} — the artifact cannot warm this catalog (hint: rebuild the artifact with the current model, or configure the model the artifact was built with; artifact identities are opaque build-time values, so compare them rather than reading them)"
            ),
            WarmError::Embedder(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WarmError {}

impl From<ArtifactError> for WarmError {
    fn from(value: ArtifactError) -> Self {
        Self::Artifact(value)
    }
}

impl From<EmbedderError> for WarmError {
    fn from(value: EmbedderError) -> Self {
        Self::Embedder(value)
    }
}

/// An item the cache can embed and rank: its stable id and the flat searchable
/// text fed to the embedder. Implemented by `Tool` and `Skill` in their
/// respective registries, so the cache stays agnostic to the item type.
pub(crate) trait Embeddable {
    fn embed_id(&self) -> &str;
    fn embed_text(&self) -> String;
}

/// A dense ranking paired with the query vector that produced it — see
/// [`DenseCache::search_returning_query_vec`], which lets the usage-ranking arm
/// reuse the embedding instead of paying for a second inference.
pub(crate) type RankedWithQuery = (Vec<(String, f32)>, Vec<f32>);

/// Per-item dense vectors, keyed by item id.
#[derive(Default)]
struct DenseCacheState {
    /// `vectors[id]` is the embedding for the registry item with that id.
    vectors: HashMap<String, Vec<f32>>,
    /// Resolved identity of the model that produced every vector in `vectors`.
    built_fingerprint: Option<String>,
    /// Shared width of every vector in `vectors`.
    dim: Option<usize>,
}

/// Per-item dense vectors, keyed by item id.
pub(crate) struct DenseCache {
    /// Vectors, dimension, and vector identity move together under one mutex so a
    /// reader can never observe a partially-committed embedding batch.
    state: Mutex<DenseCacheState>,
    /// Builds take the write side; searches take the read side across query
    /// embedding and ranking. This keeps one search in one vector space while
    /// still allowing independent searches to run concurrently.
    operation_lock: RwLock<()>,
    /// Which embedding model backs this cache. Chosen per catalog; drives which
    /// embedder [`Self::resolve_embedder`] loads. `Default` = built-in bge-small.
    model: EmbeddingModel,
    /// Test-only embedder override (`None` → the `model`'s embedder, loaded
    /// lazily on first use). Lets tests inject a deterministic/failing embedder
    /// without touching the network.
    embedder_override: Option<Arc<dyn Embedder>>,
}

impl DenseCache {
    pub(crate) fn new() -> Self {
        Self::with_model(EmbeddingModel::Default)
    }

    /// A cache backed by an explicit embedding model (the configurable-model
    /// path). The model is resolved lazily on first embed.
    pub(crate) fn with_model(model: EmbeddingModel) -> Self {
        Self {
            state: Mutex::new(DenseCacheState::default()),
            operation_lock: RwLock::new(()),
            model,
            embedder_override: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_embedder(embedder: Arc<dyn Embedder>) -> Self {
        Self::with_embedder_and_model(embedder, EmbeddingModel::Default)
    }

    /// Test helper that tags the cache with an [`EmbeddingModel`] variant while
    /// still injecting a stub embedder — needed to exercise Endpoint warm probe
    /// logic without a real network client.
    #[cfg(test)]
    pub(crate) fn with_embedder_and_model(
        embedder: Arc<dyn Embedder>,
        model: EmbeddingModel,
    ) -> Self {
        Self {
            state: Mutex::new(DenseCacheState::default()),
            operation_lock: RwLock::new(()),
            model,
            embedder_override: Some(embedder),
        }
    }

    /// The embedder to use: an injected one (tests) or the configured model's,
    /// whose one-time load telemetry is recorded on `sink`.
    fn resolve_embedder(&self, sink: &dyn TraceSink) -> Result<Arc<dyn Embedder>, EmbedderError> {
        match &self.embedder_override {
            Some(e) => Ok(e.clone()),
            None => {
                self.model.validate()?;
                embedder_with_telemetry(&self.model, sink)
            }
        }
    }

    /// Error unless the cache covers the whole corpus (`corpus_len` distinct ids).
    /// A semantic/hybrid search never embeds inside the search path — the caller
    /// must have built first, so no search silently pays the embedding cost.
    /// Loads no model.
    ///
    /// The registries key their corpus by id, so `corpus_len` is the number of
    /// distinct ids and `vectors.len()` is the number of embedded ids; the two
    /// match exactly once every id is embedded. A re-register that
    /// [`Self::invalidate`]s an id drops `vectors.len()` below `corpus_len`, so a
    /// search after churn correctly reports `EmbeddingsNotBuilt` until the next
    /// [`Self::extend`] — the same "build after registering" contract as a fresh
    /// register.
    /// The output width of the cached vectors, or `None` before any embedding
    /// has run. Equals the active model's output dimension.
    pub(crate) fn dim(&self) -> Option<usize> {
        self.state.lock().expect("dense cache mutex poisoned").dim
    }

    /// The fingerprint of the model that built the current cache, or `None`
    /// before any embedding has run. After a successful search this is the active
    /// model's identity (query and corpus must agree, or `embed_query` errored),
    /// so it doubles as "the model the query was embedded with" for the intent
    /// graph's model check.
    pub(crate) fn built_fingerprint(&self) -> Option<String> {
        self.state
            .lock()
            .expect("dense cache mutex poisoned")
            .built_fingerprint
            .clone()
    }

    /// Embed arbitrary texts under the active model, returning the vectors and
    /// the model fingerprint. Used to re-embed an intent graph's members when
    /// rebuilding its centroids after a model change.
    pub(crate) fn embed_texts_with_identity(
        &self,
        texts: &[String],
        sink: &dyn TraceSink,
    ) -> Result<(Vec<Vec<f32>>, String), EmbedderError> {
        if texts.is_empty() {
            return Ok((Vec::new(), self.built_fingerprint().unwrap_or_default()));
        }
        let embedder = self.resolve_embedder(sink)?;
        let embedded = embedder.embed_batch_with_identity(texts)?;
        Ok((embedded.value, embedded.fingerprint))
    }

    pub(crate) fn require_built(&self, corpus_len: usize) -> Result<(), EmbedderError> {
        let cached = self
            .state
            .lock()
            .expect("dense cache mutex poisoned")
            .vectors
            .len();
        if cached < corpus_len {
            return Err(EmbedderError::EmbeddingsNotBuilt);
        }
        Ok(())
    }

    /// Run `f` while holding the dense operation write lock. Used by registries to
    /// compose warm + optional extend without releasing between steps (and without
    /// re-entering [`Self::extend`], which would deadlock on this non-reentrant lock).
    pub(crate) fn with_operation_write<R>(&self, f: impl FnOnce(&Self) -> R) -> R {
        let _guard = self
            .operation_lock
            .write()
            .expect("dense operation lock poisoned");
        f(self)
    }

    /// Load vectors from a build-time embedding artifact for corpus ids whose
    /// projection text still matches, without running inference on those ids.
    ///
    /// Matching (id + projection hash) runs before any embedder load. If nothing
    /// can be reused, returns immediately with all corpus ids in `missing`.
    /// Model identity is checked only when there is at least one reuse candidate
    /// (Local/HF via [`Embedder::artifact_identity`]; Endpoint may probe once).
    /// `built_fingerprint` continues to store the **runtime** identity.
    /// Does not call [`Self::extend`] — the caller decides how to cover `missing`.
    pub(crate) fn warm_from_artifact<'a, T: Embeddable + 'a>(
        &self,
        bytes: &[u8],
        expected_kind: ArtifactEntryKind,
        items: impl IntoIterator<Item = &'a T>,
        sink: &dyn TraceSink,
    ) -> Result<WarmOutcome, WarmError> {
        self.with_operation_write(|cache| {
            cache.warm_from_artifact_locked(bytes, expected_kind, items, sink)
        })
    }

    /// Like [`Self::warm_from_artifact`], but assumes the caller already holds
    /// [`Self::operation_lock`] for write.
    pub(crate) fn warm_from_artifact_locked<'a, T: Embeddable + 'a>(
        &self,
        bytes: &[u8],
        expected_kind: ArtifactEntryKind,
        items: impl IntoIterator<Item = &'a T>,
        sink: &dyn TraceSink,
    ) -> Result<WarmOutcome, WarmError> {
        let (header, entries) = load_and_validate(bytes)?;
        // Known other kinds are ignored so one mixed RAT1 can warm either registry.
        let by_id: HashMap<&str, &ArtifactEntry> = entries
            .iter()
            .filter(|e| e.kind == expected_kind)
            .map(|e| (e.id.as_str(), e))
            .collect();

        let mut reused: Vec<(String, Vec<f32>)> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        for item in items {
            let id = item.embed_id();
            let text = item.embed_text();
            match by_id.get(id) {
                Some(entry) if entry.projection_hash == hash_projection_text(&text) => {
                    reused.push((id.to_string(), entry.vector.clone()));
                }
                _ => missing.push(id.to_string()),
            }
        }

        if reused.is_empty() {
            return Ok(WarmOutcome {
                reused: Vec::new(),
                missing,
            });
        }

        let embedder = self.resolve_embedder(sink)?;
        let mut active_artifact = embedder.artifact_identity()?;
        let mut runtime_identity = embedder.fingerprint();
        if matches!(self.model, EmbeddingModel::Endpoint { .. })
            && active_artifact != header.model_fingerprint
        {
            let probed = embedder.embed_batch_with_identity(&[ARTIFACT_WARM_PROBE_TEXT.into()])?;
            active_artifact = probed.fingerprint.clone();
            runtime_identity = probed.fingerprint;
        }
        if active_artifact != header.model_fingerprint {
            let artifact = header.model_fingerprint.clone();
            sink.record(TraceEvent::EmbedderModelMismatch {
                built: artifact.clone(),
                active: active_artifact.clone(),
            });
            return Err(WarmError::ArtifactModelMismatch {
                artifact,
                active: active_artifact,
            });
        }

        let staged: Vec<Vec<f32>> = reused.iter().map(|(_, v)| v.clone()).collect();
        let existing_dim = self.state.lock().expect("dense cache mutex poisoned").dim;
        let expected_dim = validate_batch(
            &staged,
            reused.len(),
            Some(existing_dim.unwrap_or(header.dim)),
            &header.model_fingerprint,
        )?;

        let mut state = self.state.lock().expect("dense cache mutex poisoned");
        if let Some(built) = &state.built_fingerprint
            && built != &runtime_identity
        {
            let built = built.clone();
            let active = runtime_identity.clone();
            sink.record(TraceEvent::EmbedderModelMismatch {
                built: built.clone(),
                active: active.clone(),
            });
            return Err(WarmError::Embedder(EmbedderError::ModelMismatch {
                built,
                active,
            }));
        }
        state.dim.get_or_insert(expected_dim);
        state.built_fingerprint.get_or_insert(runtime_identity);
        let reused_ids: Vec<String> = reused.iter().map(|(id, _)| id.clone()).collect();
        state.vectors.extend(reused);
        Ok(WarmOutcome {
            reused: reused_ids,
            missing,
        })
    }

    /// Serialize corpus embeddings into a build-time artifact. Does not take
    /// [`Self::operation_lock`] (does not touch cache state). An empty corpus
    /// returns a valid zero-entry artifact without resolving the embedder.
    pub(crate) fn build_artifact<'a, T: Embeddable + 'a>(
        &self,
        kind: ArtifactEntryKind,
        items: impl IntoIterator<Item = &'a T>,
        sink: &dyn TraceSink,
    ) -> Result<Vec<u8>, ArtifactError> {
        let corpus: Vec<&T> = items.into_iter().collect();
        if corpus.is_empty() {
            return build_empty_artifact();
        }
        let embedder = self.resolve_embedder(sink)?;
        crate::embedding_artifact::build_artifact(kind, corpus, embedder.as_ref())
    }

    /// Embed any item whose id is not yet cached and insert it by id — the
    /// incremental core of the cache. Skips ids already present, so an
    /// already-embedded item is never recomputed (O(k) for k missing ids: newly
    /// registered *or* invalidated-on-replace). Idempotent: a no-op once every id
    /// is cached.
    pub(crate) fn extend<'a, T: Embeddable + 'a>(
        &self,
        items: impl IntoIterator<Item = &'a T>,
        sink: &dyn TraceSink,
    ) -> Result<(), EmbedderError> {
        self.with_operation_write(|cache| cache.extend_locked(items, sink))
    }

    /// Like [`Self::extend`], but assumes the caller already holds
    /// [`Self::operation_lock`] for write.
    pub(crate) fn extend_locked<'a, T: Embeddable + 'a>(
        &self,
        items: impl IntoIterator<Item = &'a T>,
        sink: &dyn TraceSink,
    ) -> Result<(), EmbedderError> {
        // Gather the not-yet-cached ids so a fully-cached corpus never loads the
        // model (empty batch → early return).
        let missing: Vec<(String, String)> = {
            let state = self.state.lock().expect("dense cache mutex poisoned");
            items
                .into_iter()
                .filter(|item| !state.vectors.contains_key(item.embed_id()))
                .map(|item| (item.embed_id().to_string(), item.embed_text()))
                .collect()
        };
        if missing.is_empty() {
            return Ok(());
        }
        let embedder = self.resolve_embedder(sink)?;
        // One batch call: cheap for an in-process model and essential for an
        // endpoint when an explicit build embeds the missing corpus.
        let texts: Vec<String> = missing.iter().map(|(_, text)| text.clone()).collect();
        let embedded = embedder.embed_batch_with_identity(&texts)?;
        let vectors = embedded.value;

        // Validate the entire batch against one staged dimension before mutating
        // the live cache. A failure leaves all missing ids missing.
        let existing_dim = self.state.lock().expect("dense cache mutex poisoned").dim;
        let expected_dim =
            validate_batch(&vectors, missing.len(), existing_dim, &embedded.fingerprint)?;

        // Commit vectors and their metadata as one state transition.
        let mut state = self.state.lock().expect("dense cache mutex poisoned");
        if let Some(built) = &state.built_fingerprint
            && built != &embedded.fingerprint
        {
            let built = built.clone();
            let active = embedded.fingerprint;
            sink.record(TraceEvent::EmbedderModelMismatch {
                built: built.clone(),
                active: active.clone(),
            });
            return Err(EmbedderError::ModelMismatch { built, active });
        }
        state.dim.get_or_insert(expected_dim);
        state.built_fingerprint.get_or_insert(embedded.fingerprint);
        state
            .vectors
            .extend(missing.into_iter().map(|(id, _)| id).zip(vectors));
        Ok(())
    }

    /// Recompute the complete corpus, then replace vectors and metadata in one
    /// commit. Any load, inference, identity, or dimension failure leaves the
    /// previously searchable cache untouched.
    pub(crate) fn rebuild<'a, T: Embeddable + 'a>(
        &self,
        items: impl IntoIterator<Item = &'a T>,
        sink: &dyn TraceSink,
    ) -> Result<(), EmbedderError> {
        let _build = self
            .operation_lock
            .write()
            .expect("dense operation lock poisoned");
        let corpus: Vec<(String, String)> = items
            .into_iter()
            .map(|item| (item.embed_id().to_string(), item.embed_text()))
            .collect();
        if corpus.is_empty() {
            *self.state.lock().expect("dense cache mutex poisoned") = DenseCacheState::default();
            return Ok(());
        }

        let embedder = self.resolve_embedder(sink)?;
        let texts: Vec<String> = corpus.iter().map(|(_, text)| text.clone()).collect();
        let embedded = embedder.embed_batch_with_identity(&texts)?;
        let dim = validate_batch(&embedded.value, corpus.len(), None, &embedded.fingerprint)?;
        let vectors = corpus
            .into_iter()
            .map(|(id, _)| id)
            .zip(embedded.value)
            .collect();
        let replacement = DenseCacheState {
            vectors,
            built_fingerprint: Some(embedded.fingerprint),
            dim: Some(dim),
        };
        *self.state.lock().expect("dense cache mutex poisoned") = replacement;
        Ok(())
    }

    /// Drop a cached embedding for an id whose vector must not survive.
    ///
    /// Two callers, for two different reasons: a registry's `register` (and the
    /// update half of `replace_all`) drops a superseded vector so the next
    /// [`Self::extend`] re-embeds that id; the removal half of `replace_all`
    /// drops the vector of an id leaving the corpus entirely, with nothing to
    /// re-embed. The second case is load-bearing rather than tidiness —
    /// [`Self::require_built`] compares *counts*, so a vector left behind for a
    /// departed id could offset a new, unembedded one and let the guard pass.
    pub(crate) fn invalidate(&self, id: &str) {
        self.state
            .lock()
            .expect("dense cache mutex poisoned")
            .vectors
            .remove(id);
    }

    /// Validate, embed, and rank one query against one immutable cache version,
    /// returning the ranking **and** the embedded query. A concurrent
    /// build/rebuild cannot replace the vector space between the query identity
    /// check and cosine ranking.
    ///
    /// The query vector escapes because the usage-ranking arm (ADR-0014) matches
    /// it against intent centroids: reusing it here is what makes adaptive
    /// ranking free on the semantic and hybrid paths instead of costing a second
    /// inference. The match happens after this returns but within the same
    /// caller, so it observes the vector this ranking used.
    pub(crate) fn search_returning_query_vec<'a, T: Embeddable + 'a>(
        &self,
        items: impl IntoIterator<Item = &'a T>,
        query: &str,
        depth: usize,
        sink: &dyn TraceSink,
    ) -> Result<RankedWithQuery, EmbedderError> {
        let _search = self
            .operation_lock
            .read()
            .expect("dense operation lock poisoned");
        let items: Vec<&T> = items.into_iter().collect();
        self.require_built(items.len())?;
        let query_vec = self.embed_query(query, sink)?;
        let ranked = self.ranked(items, &query_vec, depth);
        Ok((ranked, query_vec))
    }

    /// Embed a query for cosine ranking (uses the same embedder as
    /// [`Self::extend`], so the one-time model load is shared).
    ///
    /// Two hard guards protect against silently-wrong cosine results: a model
    /// mismatch if the active model differs from the one that built the cache,
    /// and a dimension mismatch if the query vector's width differs from the
    /// corpus's (cosine over mismatched dims is meaningless, not merely worse).
    pub(crate) fn embed_query(
        &self,
        query: &str,
        sink: &dyn TraceSink,
    ) -> Result<Vec<f32>, EmbedderError> {
        let embedder = self.resolve_embedder(sink)?;

        let built = self
            .state
            .lock()
            .expect("dense cache mutex poisoned")
            .built_fingerprint
            .clone();
        let embedded = embedder.embed_query_with_identity(query)?;
        if let Some((built, active)) = model_drift(built.as_deref(), &embedded.fingerprint) {
            sink.record(TraceEvent::EmbedderModelMismatch {
                built: built.clone(),
                active: active.clone(),
            });
            return Err(EmbedderError::ModelMismatch { built, active });
        }

        let vector = embedded.value;
        if let Some(dim) = self.state.lock().expect("dense cache mutex poisoned").dim
            && vector.len() != dim
        {
            return Err(EmbedderError::DimensionMismatch {
                expected: dim,
                got: vector.len(),
                model: embedded.fingerprint,
            });
        }
        Ok(vector)
    }

    /// Cosine-rank `query_vec` against the cached vectors, best-first with ties
    /// broken by id. Assumes [`Self::extend`] already ran (`require_built`
    /// passed), so every item's id resolves to a vector; an id missing from the
    /// cache (shouldn't happen post-`require_built`) is simply skipped. The
    /// id-keyed corpus holds one entry per id, so there are no duplicates to
    /// collapse.
    pub(crate) fn ranked<'a, T: Embeddable + 'a>(
        &self,
        items: impl IntoIterator<Item = &'a T>,
        query_vec: &[f32],
        depth: usize,
    ) -> Vec<(String, f32)> {
        let guard = self.state.lock().expect("dense cache mutex poisoned");
        let docs: Vec<(String, &[f32])> = items
            .into_iter()
            .filter_map(|item| {
                guard
                    .vectors
                    .get(item.embed_id())
                    .map(|v| (item.embed_id().to_string(), v.as_slice()))
            })
            .collect();
        dense_search(docs, query_vec, depth)
    }
}

/// Validate one complete embedder batch without mutating cache state. Returns
/// the common vector width that may be committed.
fn validate_batch(
    vectors: &[Vec<f32>],
    expected_len: usize,
    expected_dim: Option<usize>,
    fingerprint: &str,
) -> Result<usize, EmbedderError> {
    if vectors.len() != expected_len {
        return Err(EmbedderError::Inference {
            source: format!(
                "embedder returned {} embeddings for {expected_len} inputs",
                vectors.len()
            ),
        });
    }
    let first_dim = vectors
        .first()
        .map(Vec::len)
        .ok_or_else(|| EmbedderError::Inference {
            source: "embedder returned no embeddings".into(),
        })?;
    let dim = expected_dim.unwrap_or(first_dim);
    for vector in vectors {
        if vector.len() != dim {
            return Err(EmbedderError::DimensionMismatch {
                expected: dim,
                got: vector.len(),
                model: fingerprint.to_string(),
            });
        }
    }
    Ok(dim)
}

/// Detect a model-identity mismatch: the fingerprint that built the cache vs the
/// one now in use. `None` if the cache is unbuilt (`built` is `None`) or they match.
/// Pure, so the drift logic is unit-tested without forcing the (currently
/// impossible in-process) state through the cache.
fn model_drift(built: Option<&str>, active: &str) -> Option<(String, String)> {
    match built {
        Some(b) if b != active => Some((b.to_string(), active.to_string())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::embedding::Embedded;
    use crate::test_support::{FpCountingEmbedder, PanicOnEmbedStub, build_test_artifact, unit};
    use crate::trace::{MemorySink, NoopSink, TraceEvent};

    struct Doc {
        id: String,
        text: String,
    }
    impl Embeddable for Doc {
        fn embed_id(&self) -> &str {
            &self.id
        }
        fn embed_text(&self) -> String {
            self.text.clone()
        }
    }
    fn doc(id: &str, text: &str) -> Doc {
        Doc {
            id: id.into(),
            text: text.into(),
        }
    }

    /// One-hot embedder keyed on the word "read", counting `embed_doc` calls so
    /// the incremental contract is provable without the network.
    struct CountingStub {
        docs: AtomicUsize,
    }
    impl CountingStub {
        fn new() -> Self {
            Self {
                docs: AtomicUsize::new(0),
            }
        }
        fn docs(&self) -> usize {
            self.docs.load(Ordering::SeqCst)
        }
    }
    fn vec_for(text: &str) -> Vec<f32> {
        if text.to_lowercase().contains("read") {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        }
    }
    impl Embedder for CountingStub {
        fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            self.docs.fetch_add(1, Ordering::SeqCst);
            Ok(vec_for(text))
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec_for(text))
        }
    }

    #[test]
    fn require_built_errors_until_the_cache_covers_the_corpus() {
        let cache = DenseCache::with_embedder(Arc::new(CountingStub::new()));
        let items = vec![doc("a", "read"), doc("b", "write")];
        assert!(matches!(
            cache.require_built(items.len()),
            Err(EmbedderError::EmbeddingsNotBuilt)
        ));
        cache.extend(&items, &NoopSink).unwrap();
        assert!(cache.require_built(items.len()).is_ok());
    }

    #[test]
    fn extend_embeds_only_the_new_tail() {
        let stub = Arc::new(CountingStub::new());
        let cache = DenseCache::with_embedder(stub.clone());
        let mut items = vec![doc("a", "read"), doc("b", "write")];
        cache.extend(&items, &NoopSink).unwrap();
        assert_eq!(stub.docs(), 2);
        items.push(doc("c", "read"));
        cache.extend(&items, &NoopSink).unwrap();
        assert_eq!(stub.docs(), 3, "only the newly-appended item is embedded");
        // Idempotent once caught up.
        cache.extend(&items, &NoopSink).unwrap();
        assert_eq!(stub.docs(), 3);
    }

    #[test]
    fn invalidate_forces_re_embed_of_an_id() {
        // Replace-in-place path: an id embedded as "read", then invalidated and
        // re-embedded from "write". The new vector must win — the mechanism a
        // re-registered tool/skill relies on (RAT-378).
        let stub = Arc::new(CountingStub::new());
        let cache = DenseCache::with_embedder(stub.clone());
        cache.extend([&doc("x", "read")], &NoopSink).unwrap();
        assert_eq!(stub.docs(), 1);

        cache.invalidate("x");
        // Re-embed x from its new content; the query matches "write".
        cache.extend([&doc("x", "write")], &NoopSink).unwrap();
        assert_eq!(stub.docs(), 2, "invalidated id is re-embedded, once");

        let item = doc("x", "write");
        let ranked = cache.ranked([&item], &[0.0, 1.0], 10);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].0, "x");
        assert!(ranked[0].1 > 0.9, "ranks with the re-embedded vector");
    }

    /// Embeds docs at one width but the query at another — forces the query-time
    /// dimension guard.
    struct WidthStub {
        doc_dim: usize,
        query_dim: usize,
    }
    impl Embedder for WidthStub {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![1.0; self.doc_dim])
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![1.0; self.query_dim])
        }
    }

    #[test]
    fn query_dimension_mismatch_is_a_hard_error() {
        let cache = DenseCache::with_embedder(Arc::new(WidthStub {
            doc_dim: 2,
            query_dim: 3,
        }));
        cache.extend([&doc("a", "x")], &NoopSink).unwrap(); // stamps dim = 2
        let err = cache.embed_query("q", &NoopSink).unwrap_err();
        assert!(
            matches!(
                err,
                EmbedderError::DimensionMismatch {
                    expected: 2,
                    got: 3,
                    ..
                }
            ),
            "got: {err:?}"
        );
    }

    struct RetryAfterMixedDimensions {
        batches: Mutex<Vec<usize>>,
        attempts: AtomicUsize,
    }

    impl RetryAfterMixedDimensions {
        fn new() -> Self {
            Self {
                batches: Mutex::new(Vec::new()),
                attempts: AtomicUsize::new(0),
            }
        }
    }

    impl Embedder for RetryAfterMixedDimensions {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("test exercises the batch seam")
        }

        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![1.0, 0.0])
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            self.batches
                .lock()
                .expect("batches mutex poisoned")
                .push(texts.len());
            if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(vec![vec![1.0, 0.0], vec![1.0, 0.0, 0.0]])
            } else {
                Ok(vec![vec![1.0, 0.0]; texts.len()])
            }
        }
    }

    #[test]
    fn failed_incremental_batch_commits_nothing_and_retries_every_missing_item() {
        let stub = Arc::new(RetryAfterMixedDimensions::new());
        let cache = DenseCache::with_embedder(stub.clone());
        let items = vec![doc("a", "read"), doc("b", "write")];

        assert!(matches!(
            cache.extend(&items, &NoopSink),
            Err(EmbedderError::DimensionMismatch { .. })
        ));
        cache.extend(&items, &NoopSink).unwrap();

        assert_eq!(
            *stub.batches.lock().expect("batches mutex poisoned"),
            vec![2, 2],
            "a failed batch must leave every item missing"
        );
        assert!(cache.require_built(items.len()).is_ok());
    }

    struct ChangingIdentityStub {
        batch_identities: Mutex<std::collections::VecDeque<&'static str>>,
        query_identity: &'static str,
    }

    impl Embedder for ChangingIdentityStub {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![1.0, 0.0])
        }

        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![1.0, 0.0])
        }

        fn embed_batch_with_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            let fingerprint = self
                .batch_identities
                .lock()
                .expect("identities mutex poisoned")
                .pop_front()
                .expect("scripted batch identity");
            Ok(Embedded {
                value: vec![vec![1.0, 0.0]; texts.len()],
                fingerprint: fingerprint.into(),
            })
        }

        fn embed_query_with_identity(
            &self,
            _text: &str,
        ) -> Result<Embedded<Vec<f32>>, EmbedderError> {
            Ok(Embedded {
                value: vec![1.0, 0.0],
                fingerprint: self.query_identity.into(),
            })
        }
    }

    #[test]
    fn incremental_model_mismatch_is_hard_and_commits_nothing() {
        let stub = Arc::new(ChangingIdentityStub {
            batch_identities: Mutex::new(std::collections::VecDeque::from(["a", "b", "a"])),
            query_identity: "a",
        });
        let cache = DenseCache::with_embedder(stub);
        let mut items = vec![doc("a", "read")];
        cache.extend(&items, &NoopSink).unwrap();
        items.push(doc("b", "write"));

        assert!(matches!(
            cache.extend(&items, &NoopSink),
            Err(EmbedderError::ModelMismatch { .. })
        ));
        cache.extend(&items, &NoopSink).unwrap();
        assert!(cache.require_built(items.len()).is_ok());
    }

    #[test]
    fn query_model_mismatch_is_a_hard_error() {
        let stub = Arc::new(ChangingIdentityStub {
            batch_identities: Mutex::new(std::collections::VecDeque::from(["built"])),
            query_identity: "active",
        });
        let cache = DenseCache::with_embedder(stub);
        cache.extend([&doc("a", "read")], &NoopSink).unwrap();

        assert!(matches!(
            cache.embed_query("q", &NoopSink),
            Err(EmbedderError::ModelMismatch { built, active })
                if built == "built" && active == "active"
        ));
    }

    #[test]
    fn model_drift_detects_a_changed_fingerprint() {
        assert_eq!(model_drift(None, "a"), None, "unbuilt cache never drifts");
        assert_eq!(model_drift(Some("a"), "a"), None, "same model never drifts");
        assert_eq!(
            model_drift(Some("a"), "b"),
            Some(("a".to_string(), "b".to_string())),
            "a changed model drifts"
        );
    }

    #[test]
    fn require_built_fails_after_invalidate_until_rebuilt() {
        let cache = DenseCache::with_embedder(Arc::new(CountingStub::new()));
        let items = vec![doc("a", "read"), doc("b", "write")];
        cache.extend(&items, &NoopSink).unwrap();
        assert!(cache.require_built(items.len()).is_ok());

        cache.invalidate("a");
        assert!(
            matches!(
                cache.require_built(items.len()),
                Err(EmbedderError::EmbeddingsNotBuilt)
            ),
            "an invalidated id drops the cache below the corpus until rebuilt"
        );
        cache.extend(&items, &NoopSink).unwrap();
        assert!(cache.require_built(items.len()).is_ok());
    }

    /// Warm stub with distinct runtime vs artifact identities (Local AD-1 shape).
    struct SplitIdentityStub {
        runtime: String,
        artifact: String,
    }

    impl Embedder for SplitIdentityStub {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            panic!("embed_doc must not be called during pure warm")
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            panic!("embed_query must not be called during pure warm")
        }
        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            panic!("embed_batch must not be called during pure warm")
        }
        fn fingerprint(&self) -> String {
            self.runtime.clone()
        }
        fn artifact_identity(&self) -> Result<String, EmbedderError> {
            Ok(self.artifact.clone())
        }
    }

    /// Endpoint probe stub: static `fingerprint()`, probe identity via batch.
    struct EndpointProbeStub {
        static_fingerprint: String,
        probe_fingerprint: String,
        probe_calls: AtomicUsize,
        allow_probe: bool,
    }

    impl Embedder for EndpointProbeStub {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            panic!("embed_doc must not be called")
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            panic!("embed_query must not be called")
        }
        fn embed_batch_with_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            assert!(
                self.allow_probe,
                "Endpoint probe must not run when static fingerprint already matches"
            );
            assert_eq!(texts.len(), 1);
            self.probe_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Embedded {
                value: vec![unit([1.0, 0.0])],
                fingerprint: self.probe_fingerprint.clone(),
            })
        }
        fn fingerprint(&self) -> String {
            self.static_fingerprint.clone()
        }
    }

    fn sample_endpoint_model() -> EmbeddingModel {
        EmbeddingModel::Endpoint {
            url: "http://example.test/v1/embeddings".into(),
            model: "configured-model".into(),
            api_key_env: None,
            query_prefix: None,
            doc_prefix: None,
        }
    }

    #[test]
    fn warm_reuses_matching_id_and_hash_without_calling_embedder() {
        let items = [doc("a", "read"), doc("b", "write")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "fp-warm",
            vec![unit([1.0, 0.0]), unit([0.0, 1.0])],
        );
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a", "b"]);
        assert!(outcome.missing.is_empty());
        assert_eq!(cache.built_fingerprint().as_deref(), Some("fp-warm"));
        assert_eq!(cache.dim(), Some(2));
        assert!(cache.require_built(items.len()).is_ok());
    }

    #[test]
    fn warm_reports_missing_when_id_absent_from_artifact() {
        let artifact_items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &artifact_items,
            "fp-warm",
            vec![unit([1.0, 0.0])],
        );
        let corpus = [doc("a", "read"), doc("b", "write")];
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &corpus, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a"]);
        assert_eq!(outcome.missing, vec!["b"]);
        assert!(cache.require_built(1).is_ok());
        assert!(matches!(
            cache.require_built(2),
            Err(EmbedderError::EmbeddingsNotBuilt)
        ));
    }

    #[test]
    fn warm_reports_missing_when_projection_hash_differs() {
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &[doc("a", "read")],
            "fp-warm",
            vec![unit([1.0, 0.0])],
        );
        let corpus = [doc("a", "read changed")];
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &corpus, &NoopSink)
            .unwrap();
        assert!(outcome.reused.is_empty());
        assert_eq!(outcome.missing, vec!["a"]);
        assert!(cache.built_fingerprint().is_none());
        assert!(cache.dim().is_none());
    }

    #[test]
    fn warm_model_fingerprint_mismatch_leaves_cache_untouched() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "fp-artifact",
            vec![unit([1.0, 0.0])],
        );
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-active")));
        assert!(matches!(
            cache.warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink),
            Err(WarmError::ArtifactModelMismatch {
                artifact,
                active
            }) if artifact == "fp-artifact" && active == "fp-active"
        ));
        assert!(cache.built_fingerprint().is_none());
        assert!(cache.dim().is_none());
        assert!(matches!(
            cache.require_built(1),
            Err(EmbedderError::EmbeddingsNotBuilt)
        ));
    }

    #[test]
    fn warm_ignores_other_known_entry_kind() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            &items,
            "fp-warm",
            vec![unit([1.0, 0.0])],
        );
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink)
            .unwrap();
        assert!(outcome.reused.is_empty());
        assert_eq!(outcome.missing, vec!["a"]);
        assert!(cache.built_fingerprint().is_none());
        assert!(cache.dim().is_none());
    }

    #[test]
    fn warm_mixed_artifact_reuses_matching_kind_only() {
        let tool = doc("search", "tool search text");
        let skill = doc("search", "skill search text");
        let tool_bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            std::slice::from_ref(&tool),
            "fp-warm",
            vec![unit([1.0, 0.0])],
        );
        let skill_bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            std::slice::from_ref(&skill),
            "fp-warm",
            vec![unit([0.0, 1.0])],
        );
        let bytes =
            crate::embedding_artifact::merge_embedding_artifacts(&[&tool_bytes, &skill_bytes])
                .unwrap();

        let tool_cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let tool_outcome = tool_cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &[tool], &NoopSink)
            .unwrap();
        assert_eq!(tool_outcome.reused, vec!["search"]);
        assert!(tool_outcome.missing.is_empty());
        assert_eq!(
            tool_cache.state.lock().unwrap().vectors.get("search"),
            Some(&unit([1.0, 0.0]))
        );

        let skill_cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let skill_outcome = skill_cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Skill, &[skill], &NoopSink)
            .unwrap();
        assert_eq!(skill_outcome.reused, vec!["search"]);
        assert!(skill_outcome.missing.is_empty());
        assert_eq!(
            skill_cache.state.lock().unwrap().vectors.get("search"),
            Some(&unit([0.0, 1.0]))
        );
    }

    #[test]
    fn warm_subset_corpus_from_superset_artifact() {
        let artifact_items = [doc("a", "alpha"), doc("b", "bravo"), doc("c", "charlie")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &artifact_items,
            "fp-warm",
            vec![unit([1.0, 0.0]), unit([0.0, 1.0]), unit([0.6, 0.8])],
        );
        let corpus = [doc("a", "alpha"), doc("c", "charlie")];
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("fp-warm")));
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &corpus, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a", "c"]);
        assert!(outcome.missing.is_empty());
        assert!(cache.require_built(2).is_ok());
    }

    #[test]
    fn warm_then_extend_embeds_only_missing_ids() {
        let artifact_items = [doc("a", "read file")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &artifact_items,
            "fp-warm",
            vec![unit([1.0, 0.0])],
        );
        let corpus = [doc("a", "read file"), doc("b", "write file")];

        let count_stub = Arc::new(FpCountingEmbedder::new("fp-warm", vec_for));
        let cache = DenseCache::with_embedder(count_stub.clone());
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &corpus, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a"]);
        assert_eq!(outcome.missing, vec!["b"]);
        assert_eq!(count_stub.docs(), 0, "warm must not embed reused ids");

        cache.extend(&corpus, &NoopSink).unwrap();
        assert_eq!(count_stub.docs(), 1, "only the missing id is embedded");
        assert!(cache.require_built(corpus.len()).is_ok());
    }

    #[test]
    fn warm_on_endpoint_skips_probe_when_static_fingerprint_already_matches() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "fp-static",
            vec![unit([1.0, 0.0])],
        );
        let stub = Arc::new(EndpointProbeStub {
            static_fingerprint: "fp-static".into(),
            probe_fingerprint: "fp-should-not-matter".into(),
            probe_calls: AtomicUsize::new(0),
            allow_probe: false,
        });
        let cache = DenseCache::with_embedder_and_model(stub.clone(), sample_endpoint_model());
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a"]);
        assert_eq!(stub.probe_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn warm_on_endpoint_probes_and_accepts_on_resolved_match() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "fp-resolved",
            vec![unit([1.0, 0.0])],
        );
        let stub = Arc::new(EndpointProbeStub {
            static_fingerprint: "fp-configured".into(),
            probe_fingerprint: "fp-resolved".into(),
            probe_calls: AtomicUsize::new(0),
            allow_probe: true,
        });
        let cache = DenseCache::with_embedder_and_model(stub.clone(), sample_endpoint_model());
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a"]);
        assert_eq!(stub.probe_calls.load(Ordering::SeqCst), 1);
        assert_eq!(cache.built_fingerprint().as_deref(), Some("fp-resolved"));
    }

    #[test]
    fn warm_on_endpoint_probes_and_rejects_on_genuine_mismatch() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "fp-artifact",
            vec![unit([1.0, 0.0])],
        );
        let stub = Arc::new(EndpointProbeStub {
            static_fingerprint: "fp-configured".into(),
            probe_fingerprint: "fp-other".into(),
            probe_calls: AtomicUsize::new(0),
            allow_probe: true,
        });
        let cache = DenseCache::with_embedder_and_model(stub.clone(), sample_endpoint_model());
        assert!(matches!(
            cache.warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink),
            Err(WarmError::ArtifactModelMismatch {
                artifact,
                active
            }) if artifact == "fp-artifact" && active == "fp-other"
        ));
        assert_eq!(stub.probe_calls.load(Ordering::SeqCst), 1);
        assert!(cache.built_fingerprint().is_none());
        assert!(matches!(
            cache.require_built(1),
            Err(EmbedderError::EmbeddingsNotBuilt)
        ));
    }

    #[test]
    fn warm_compares_artifact_identity_stamps_runtime() {
        let items = [doc("a", "read"), doc("b", "write")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "content-x",
            vec![unit([1.0, 0.0]), unit([0.0, 1.0])],
        );
        let cache = DenseCache::with_embedder(Arc::new(SplitIdentityStub {
            runtime: "runtime-path-b".into(),
            artifact: "content-x".into(),
        }));
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["a", "b"]);
        assert_eq!(
            cache.built_fingerprint().as_deref(),
            Some("runtime-path-b"),
            "warm must stamp runtime identity, not the RAT1 artifact identity"
        );
    }

    #[test]
    fn warm_rejects_artifact_identity_mismatch() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "content-a",
            vec![unit([1.0, 0.0])],
        );
        let cache = DenseCache::with_embedder(Arc::new(SplitIdentityStub {
            runtime: "runtime-path".into(),
            artifact: "content-b".into(),
        }));
        assert!(matches!(
            cache.warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink),
            Err(WarmError::ArtifactModelMismatch {
                artifact,
                active
            }) if artifact == "content-a" && active == "content-b"
        ));
        assert!(cache.built_fingerprint().is_none());
        assert!(cache.dim().is_none());
    }

    #[test]
    fn artifact_model_mismatch_message_names_the_artifact() {
        let items = [doc("a", "read")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &items,
            "content-a",
            vec![unit([1.0, 0.0])],
        );
        let cache = DenseCache::with_embedder(Arc::new(SplitIdentityStub {
            runtime: "runtime-path".into(),
            artifact: "content-b".into(),
        }));
        let err = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink)
            .expect_err("header mismatch");
        let message = err.to_string();
        assert!(
            message.contains("embedding artifact was built with"),
            "{message}"
        );
        assert!(message.contains("rebuild the artifact"), "{message}");
        assert!(!message.contains("cache was built with"), "{message}");
        assert!(!message.contains("re-embed the corpus"), "{message}");
    }

    #[test]
    fn warm_rejects_nonempty_zero_dim_before_cache_mutation() {
        let items = [doc("a", "read")];
        let bytes = crate::embedding_artifact::test_hand_artifact(
            crate::embedding_artifact::projection_version(),
            0,
            "fp-zero-dim",
            &[ArtifactEntry {
                kind: ArtifactEntryKind::Tool,
                id: "a".into(),
                projection_hash: hash_projection_text("read"),
                vector: vec![],
            }],
        );
        let cache = DenseCache::with_embedder(Arc::new(PanicOnEmbedStub::new("unused")));
        assert!(matches!(
            cache.warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &items, &NoopSink),
            Err(WarmError::Artifact(ArtifactError::NonEmptyZeroDim))
        ));
        assert!(cache.built_fingerprint().is_none());
        assert!(cache.dim().is_none());
    }

    #[test]
    fn warm_into_prebuilt_rejects_dimension_mismatch() {
        let cache = DenseCache::with_embedder(Arc::new(WidthStub {
            doc_dim: 3,
            query_dim: 3,
        }));
        let prebuilt = doc("a", "x");
        cache.extend([&prebuilt], &NoopSink).unwrap();
        assert_eq!(cache.dim(), Some(3));
        assert_eq!(cache.built_fingerprint().as_deref(), Some("unknown"));
        let a_before = cache
            .state
            .lock()
            .unwrap()
            .vectors
            .get("a")
            .cloned()
            .expect("prebuilt id a");

        let warm_items = [doc("b", "y")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &warm_items,
            "unknown",
            vec![unit([1.0, 0.0])],
        );
        assert!(matches!(
            cache.warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &warm_items, &NoopSink),
            Err(WarmError::Embedder(EmbedderError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            }))
        ));
        assert_eq!(cache.dim(), Some(3));
        assert_eq!(cache.built_fingerprint().as_deref(), Some("unknown"));
        let state = cache.state.lock().unwrap();
        assert_eq!(state.vectors.get("a"), Some(&a_before));
        assert!(!state.vectors.contains_key("b"));
        assert_eq!(state.vectors.len(), 1);
    }

    #[test]
    fn warm_into_prebuilt_rejects_runtime_fingerprint_mismatch() {
        let stub = Arc::new(EndpointProbeStub {
            static_fingerprint: "fp-static-Y".into(),
            probe_fingerprint: "fp-probed-X".into(),
            probe_calls: AtomicUsize::new(0),
            allow_probe: true,
        });
        let cache = DenseCache::with_embedder_and_model(stub.clone(), sample_endpoint_model());
        let prebuilt = doc("a", "read");
        cache.extend([&prebuilt], &NoopSink).unwrap();
        assert_eq!(cache.built_fingerprint().as_deref(), Some("fp-probed-X"));
        assert_eq!(cache.dim(), Some(2));
        let a_before = cache
            .state
            .lock()
            .unwrap()
            .vectors
            .get("a")
            .cloned()
            .expect("prebuilt id a");
        let probes_before = stub.probe_calls.load(Ordering::SeqCst);
        assert_eq!(probes_before, 1);

        let warm_items = [doc("b", "write")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &warm_items,
            "fp-static-Y",
            vec![unit([0.0, 1.0])],
        );
        let sink = MemorySink::new("warm-second-guard");
        let err = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &warm_items, &sink)
            .expect_err("runtime fingerprint mismatch");
        assert!(matches!(
            &err,
            WarmError::Embedder(EmbedderError::ModelMismatch {
                built,
                active
            }) if built == "fp-probed-X" && active == "fp-static-Y"
        ));
        assert!(err.to_string().contains("cache was built with"), "{err}");
        assert_eq!(stub.probe_calls.load(Ordering::SeqCst), probes_before);
        let mismatch: Vec<_> = sink
            .drain()
            .into_iter()
            .filter_map(|e| match e.event {
                TraceEvent::EmbedderModelMismatch { built, active } => Some((built, active)),
                _ => None,
            })
            .collect();
        assert_eq!(
            mismatch,
            vec![("fp-probed-X".into(), "fp-static-Y".into())],
            "second guard must emit exactly one EmbedderModelMismatch"
        );
        assert_eq!(cache.dim(), Some(2));
        assert_eq!(cache.built_fingerprint().as_deref(), Some("fp-probed-X"));
        let state = cache.state.lock().unwrap();
        assert_eq!(state.vectors.get("a"), Some(&a_before));
        assert!(!state.vectors.contains_key("b"));
        assert_eq!(state.vectors.len(), 1);
    }

    #[test]
    fn warm_into_prebuilt_adds_compatible_id() {
        let stub = Arc::new(FpCountingEmbedder::new("fp-add", vec_for));
        let cache = DenseCache::with_embedder(stub.clone());
        let prebuilt = doc("a", "read");
        cache.extend([&prebuilt], &NoopSink).unwrap();
        assert_eq!(stub.docs(), 1);
        assert_eq!(cache.dim(), Some(2));
        assert_eq!(cache.built_fingerprint().as_deref(), Some("fp-add"));

        let warm_items = [doc("b", "write")];
        let bytes = build_test_artifact(
            ArtifactEntryKind::Tool,
            &warm_items,
            "fp-add",
            vec![unit([0.0, 1.0])],
        );
        let docs_before_warm = stub.docs();
        let outcome = cache
            .warm_from_artifact(&bytes, ArtifactEntryKind::Tool, &warm_items, &NoopSink)
            .unwrap();
        assert_eq!(outcome.reused, vec!["b"]);
        assert!(outcome.missing.is_empty());
        assert_eq!(stub.docs(), docs_before_warm);
        assert_eq!(cache.dim(), Some(2));
        assert_eq!(cache.built_fingerprint().as_deref(), Some("fp-add"));
        let state = cache.state.lock().unwrap();
        assert!(state.vectors.contains_key("a"));
        assert!(state.vectors.contains_key("b"));
        assert_eq!(state.vectors.len(), 2);
    }
}
