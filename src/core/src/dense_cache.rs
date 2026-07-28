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
//! behind (RAT-378).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::dense_search::dense_search;
use crate::embedding::{Embedder, EmbedderError, embedder_with_telemetry};
use crate::embedding_config::EmbeddingModel;
use crate::trace::{TraceEvent, TraceSink};

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
        Self {
            state: Mutex::new(DenseCacheState::default()),
            operation_lock: RwLock::new(()),
            model: EmbeddingModel::Default,
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
        let _build = self
            .operation_lock
            .write()
            .expect("dense operation lock poisoned");
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

    /// Drop a cached embedding so the next [`Self::extend`] re-embeds this id.
    /// Called by a registry's `register` when it replaces an existing id in place.
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
    use crate::trace::NoopSink;

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
}
