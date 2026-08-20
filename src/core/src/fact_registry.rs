use std::sync::Arc;
use std::time::Instant;

use indexmap::IndexMap;

use crate::dense_cache::{DenseCache, Embeddable};
use crate::embedding::EmbedderError;
use crate::embedding_config::EmbeddingModel;
use crate::fact::{Fact, PinMode};
use crate::fact_indexing::searchable_text;
use crate::fusion::{RETRIEVE_DEPTH, RRF_K, rrf_fuse_weighted, sort_and_truncate};
use crate::method::SearchMethod;
use crate::search::Bm25Cache;
use crate::trace::{ChurnKind, FactHitTrace, NoopSink, Origin, SearchStage, TraceEvent, TraceSink};

/// One ranked match from a [`FactRegistry`] search, best-first in the returned
/// `Vec` — the fact-side twin of [`crate::SkillHit`].
pub struct FactHit {
    /// Id of the matching fact ([`Fact::id`]).
    pub fact_id: String,
    /// Relevance score — higher is better; the scale depends on the
    /// [`SearchMethod`] exactly as documented on [`crate::SearchHit::score`].
    /// Ties break by `fact_id` ascending.
    pub score: f32,
}

impl Embeddable for Fact {
    fn embed_id(&self) -> &str {
        &self.id
    }
    fn embed_text(&self) -> String {
        searchable_text(self)
    }
}

/// Retrieval index over [`Fact`]s — the push-path analog of
/// [`crate::SkillRegistry`]. Same selectable BM25/semantic/hybrid engines; a
/// parallel type keeps the skill path untouched and lets fact telemetry
/// (`fact_search` / `fact_churn` / `fact_inject`) stand on its own.
pub struct FactRegistry {
    /// Corpus keyed by fact id, in insertion order — the fact-side twin of
    /// [`crate::SkillRegistry`]'s field. `register` replaces an existing id in
    /// place, never duplicating it. Insertion order is also the order
    /// [`Self::pinned`] injects always-on facts in.
    facts: IndexMap<String, Fact>,
    sink: Arc<dyn TraceSink>,
    experimental_catalog_definitions: bool,
    /// Prebuilt BM25 index over `facts`, cached across searches and invalidated
    /// on any mutation of the indexed text (mirrors the skill/tool registries).
    bm25: Bm25Cache,
    /// Dense embeddings for `facts`, keyed by id and built on demand.
    dense: DenseCache,
}

impl Default for FactRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FactRegistry {
    /// An empty registry with tracing off ([`NoopSink`]) — see
    /// [`crate::SkillRegistry::new`].
    pub fn new() -> Self {
        Self {
            facts: IndexMap::new(),
            sink: Arc::new(NoopSink),
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
            dense: DenseCache::new(),
        }
    }

    /// An empty registry recording trace events to `sink` from the start.
    pub fn with_trace_sink(sink: Arc<dyn TraceSink>) -> Self {
        Self {
            facts: IndexMap::new(),
            sink,
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
            dense: DenseCache::new(),
        }
    }

    /// A registry whose semantic/hybrid engines use an explicit embedding model.
    /// BM25 is unaffected. See [`crate::SkillRegistry::with_embedding`].
    pub fn with_embedding(model: EmbeddingModel) -> Self {
        Self {
            facts: IndexMap::new(),
            sink: Arc::new(NoopSink),
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
            dense: DenseCache::with_model(model),
        }
    }

    /// Replace the trace sink; subsequent events go to `sink`.
    pub fn set_trace_sink(&mut self, sink: Arc<dyn TraceSink>) {
        self.sink = sink;
    }

    /// Enable experimental complete catalog-definition events for later registrations.
    pub fn experimental_enable_catalog_definitions(&mut self) {
        self.experimental_catalog_definitions = true;
    }

    /// Record an arbitrary [`TraceEvent`] on the registry's sink. The SDK fact
    /// grounding path emits its `fact_inject` / `fact_inject_skip` events
    /// through this.
    pub fn record_event(&self, event: TraceEvent) {
        self.sink.record(event);
    }

    /// Register a fact, or replace one in place if its id is already present.
    /// Replacing invalidates the old id's cached embedding; the corpus never
    /// holds a duplicate.
    pub fn register(&mut self, fact: Fact) {
        let fact_id = fact.id.clone();
        let definition = self
            .experimental_catalog_definitions
            .then(|| TraceEvent::catalog_definition_for_fact(&fact))
            .flatten();
        let definition_changed = definition.as_ref().is_some_and(|definition| {
            self.facts.get(&fact_id).is_none_or(|existing| {
                let existing_definition = TraceEvent::catalog_definition_for_fact(existing);
                existing_definition
                    .as_ref()
                    .and_then(TraceEvent::catalog_definition_hash)
                    != definition.catalog_definition_hash()
            })
        });
        if self.facts.insert(fact_id.clone(), fact).is_some() {
            // Replaced an existing id: drop its stale embedding.
            self.dense.invalidate(&fact_id);
        }
        // The corpus changed: the cached BM25 index is stale.
        self.bm25.invalidate();
        self.sink.record(TraceEvent::FactChurn {
            kind: ChurnKind::Add,
            fact_id,
        });
        if definition_changed && let Some(definition) = definition {
            self.sink.record(definition);
        }
    }

    /// Number of registered facts (distinct ids).
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// Whether no facts are registered.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    /// The always-on facts ([`PinMode::Always`]), in registration order — the
    /// push tier the grounding layer injects every applicable turn, bypassing
    /// ranking entirely. Retrieval-gated facts are excluded; reach them via
    /// [`Self::search`].
    pub fn pinned(&self) -> Vec<&Fact> {
        self.facts
            .values()
            .filter(|f| f.pin == PinMode::Always)
            .collect()
    }

    /// Look up a fact by id (including its `body`), or `None` for an unknown id.
    pub fn get(&self, fact_id: &str) -> Option<&Fact> {
        self.facts.get(fact_id)
    }

    /// Lexical BM25 retrieval — the fact-side twin of
    /// [`crate::SkillRegistry::search`]: no model, never fails. Returns at most
    /// `top_k` hits, best-first. Ranks both tiers (a pinned fact can still be a
    /// query hit). Traced as [`Origin::Direct`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ratel_ai_core::{Fact, FactRegistry, PinMode};
    ///
    /// let mut registry = FactRegistry::new();
    /// registry.register(Fact {
    ///     id: "cancellation".into(),
    ///     name: "cancellation-policy".into(),
    ///     description: "How to cancel or reschedule a booking".into(),
    ///     experimental_searchable_description: None,
    ///     tags: vec!["booking".into()],
    ///     metadata: std::collections::HashMap::new(),
    ///     body: "Cancel at least 24h ahead for a full refund.".into(),
    ///     pin: PinMode::Retrieved,
    /// });
    ///
    /// let hits = registry.search("how do I reschedule my appointment", 5);
    /// assert_eq!(hits[0].fact_id, "cancellation");
    /// ```
    pub fn search(&self, query: &str, top_k: usize) -> Vec<FactHit> {
        self.search_with_origin(query, top_k, Origin::Direct)
    }

    /// [`Self::search`] with an explicit trace [`Origin`].
    pub fn search_with_origin(&self, query: &str, top_k: usize, origin: Origin) -> Vec<FactHit> {
        self.bm25_search_traced(query, top_k, origin)
    }

    /// Retrieve with an explicit [`SearchMethod`]. See
    /// [`crate::SkillRegistry::search_with_method`].
    ///
    /// # Errors
    ///
    /// Never errors for [`SearchMethod::Bm25`]; for `Semantic`/`Hybrid`, the
    /// same [`EmbedderError`] cases as the skill path.
    pub fn search_with_method(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        method: SearchMethod,
    ) -> Result<Vec<FactHit>, EmbedderError> {
        match method {
            SearchMethod::Bm25 => Ok(self.bm25_search_traced(query, top_k, origin)),
            SearchMethod::Semantic => self.semantic_search_traced(query, top_k, origin),
            SearchMethod::Hybrid => self.hybrid_search_traced(query, top_k, origin),
        }
    }

    /// Pre-compute embeddings for not-yet-embedded facts.
    ///
    /// # Errors
    ///
    /// The same [`EmbedderError`] cases as [`crate::SkillRegistry::build_embeddings`].
    pub fn build_embeddings(&self) -> Result<(), EmbedderError> {
        self.dense.extend(self.facts.values(), self.sink.as_ref())
    }

    /// Recompute embeddings for the full fact corpus and atomically replace the
    /// dense cache.
    ///
    /// # Errors
    ///
    /// Any [`EmbedderError`] from loading or embedding the complete corpus.
    pub fn rebuild_embeddings(&self) -> Result<(), EmbedderError> {
        self.dense.rebuild(self.facts.values(), self.sink.as_ref())
    }

    /// The corpus as `(id, searchable_text)` pairs for BM25.
    fn bm25_docs(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.facts
            .values()
            .map(|f| (f.id.clone(), searchable_text(f)))
    }

    /// The prebuilt BM25 index for the current corpus — cached across searches,
    /// rebuilt on the first search after a mutation.
    fn bm25_index(&self) -> Arc<crate::search::Bm25Index> {
        self.bm25.get_or_build(|| self.bm25_docs())
    }

    // ---- engines -----------------------------------------------------------

    fn bm25_search_traced(&self, query: &str, top_k: usize, origin: Origin) -> Vec<FactHit> {
        let started = Instant::now();
        let hits: Vec<FactHit> = self
            .bm25_index()
            .search(query, top_k)
            .into_iter()
            .map(|(fact_id, score)| FactHit { fact_id, score })
            .collect();
        let took_ms = started.elapsed().as_millis() as u64;
        let top_score = hits.first().map(|h| h.score as f64);
        self.record_search(
            query,
            origin,
            top_k,
            &hits,
            vec![SearchStage {
                name: "bm25".into(),
                took_ms,
                top_score,
            }],
            took_ms,
        );
        hits
    }

    fn semantic_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
    ) -> Result<Vec<FactHit>, EmbedderError> {
        let started = Instant::now();
        if self.facts.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0);
            return Ok(Vec::new());
        }
        let t = Instant::now();
        // Facts have no usage arm, so retrieval depth stays `top_k` and the
        // query vector the dense arm embeds is not reused.
        let (ranked, _query_vec) = self.dense.search_returning_query_vec(
            self.facts.values(),
            query,
            top_k,
            self.sink.as_ref(),
        )?;
        let stage_ms = t.elapsed().as_millis() as u64;
        let hits: Vec<FactHit> = ranked
            .into_iter()
            .map(|(fact_id, score)| FactHit { fact_id, score })
            .collect();
        let took_ms = started.elapsed().as_millis() as u64;
        let top_score = hits.first().map(|h| h.score as f64);
        self.record_search(
            query,
            origin,
            top_k,
            &hits,
            vec![SearchStage {
                name: "dense".into(),
                took_ms: stage_ms,
                top_score,
            }],
            took_ms,
        );
        Ok(hits)
    }

    fn hybrid_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
    ) -> Result<Vec<FactHit>, EmbedderError> {
        let started = Instant::now();
        if self.facts.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0);
            return Ok(Vec::new());
        }
        let depth = RETRIEVE_DEPTH.max(top_k);

        let t = Instant::now();
        let bm25_ranked = self.bm25_index().search(query, depth);
        let bm25_stage = SearchStage {
            name: "bm25".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: bm25_ranked.first().map(|(_, s)| *s as f64),
        };

        let t = Instant::now();
        let (dense_ranked, _query_vec) = self.dense.search_returning_query_vec(
            self.facts.values(),
            query,
            depth,
            self.sink.as_ref(),
        )?;
        let dense_stage = SearchStage {
            name: "dense".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: dense_ranked.first().map(|(_, s)| *s as f64),
        };

        let t = Instant::now();
        let bm25_ids: Vec<String> = bm25_ranked.into_iter().map(|(id, _)| id).collect();
        let dense_ids: Vec<String> = dense_ranked.into_iter().map(|(id, _)| id).collect();
        let mut fused = rrf_fuse_weighted(&[(&bm25_ids, 1.0), (&dense_ids, 1.0)], RRF_K);
        sort_and_truncate(&mut fused, top_k);
        let rrf_stage = SearchStage {
            name: "rrf".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: fused.first().map(|(_, s)| *s as f64),
        };

        let hits: Vec<FactHit> = fused
            .into_iter()
            .map(|(fact_id, score)| FactHit { fact_id, score })
            .collect();
        let took_ms = started.elapsed().as_millis() as u64;
        self.record_search(
            query,
            origin,
            top_k,
            &hits,
            vec![bm25_stage, dense_stage, rrf_stage],
            took_ms,
        );
        Ok(hits)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_search(
        &self,
        query: &str,
        origin: Origin,
        top_k: usize,
        hits: &[FactHit],
        stages: Vec<SearchStage>,
        took_ms: u64,
    ) {
        self.sink.record(TraceEvent::FactSearch {
            query: query.to_string(),
            origin,
            top_k: top_k as u32,
            hits: hits
                .iter()
                .map(|h| FactHitTrace {
                    fact_id: h.fact_id.clone(),
                    score: h.score as f64,
                })
                .collect(),
            stages,
            took_ms,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::{Embedder, EmbedderError};
    use crate::trace::MemorySink;

    struct StubEmbedder;
    impl StubEmbedder {
        fn vec_for(text: &str) -> Vec<f32> {
            let t = text.to_lowercase();
            if t.contains("address") || t.contains("location") {
                vec![1.0, 0.0, 0.0]
            } else if t.contains("cancel") || t.contains("refund") {
                vec![0.0, 1.0, 0.0]
            } else {
                vec![0.0, 0.0, 1.0]
            }
        }
    }
    impl Embedder for StubEmbedder {
        fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(StubEmbedder::vec_for(text))
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(StubEmbedder::vec_for(text))
        }
    }

    fn with_embedder(embedder: Arc<dyn Embedder>) -> FactRegistry {
        FactRegistry {
            facts: IndexMap::new(),
            sink: Arc::new(NoopSink),
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
            dense: DenseCache::with_embedder(embedder),
        }
    }

    fn fact(id: &str, name: &str, description: &str, tags: &[&str], pin: PinMode) -> Fact {
        Fact {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            experimental_searchable_description: None,
            tags: tags.iter().map(|t| (*t).into()).collect(),
            metadata: std::collections::HashMap::new(),
            body: format!("{name} body"),
            pin,
        }
    }

    fn catalog() -> FactRegistry {
        let mut reg = FactRegistry::new();
        reg.register(fact(
            "shop-address",
            "shop-address",
            "Where the barbershop is located and its opening hours",
            &["location"],
            PinMode::Always,
        ));
        reg.register(fact(
            "cancellation",
            "cancellation-policy",
            "How to cancel or reschedule a booking and get a refund",
            &["booking"],
            PinMode::Retrieved,
        ));
        reg
    }

    /// A builder that must never run — proof the search path already populated
    /// the cache (the skill-side twin explains why the public seam can't pin this).
    fn no_build() -> Vec<(String, String)> {
        unreachable!("cache should already be populated by the search path")
    }

    #[test]
    fn bm25_cache_is_warmed_by_search_and_dropped_by_register() {
        // Lifecycle pin: a public search populates the cache, and `register` —
        // the registry's only mutator — drops it. Results are byte-identical
        // either way, so only this test fails if `register` stops invalidating;
        // a stale index means `ground()` ranks against the old corpus and an
        // edited fact never surfaces in the retrieved tier. Twin of
        // `SkillRegistry::bm25_cache_is_warmed_by_search_and_dropped_by_every_mutator`.
        let mut reg = catalog();
        let _ = reg.search("how do I cancel", 5);
        let warmed = reg.bm25.get_or_build(no_build);
        let _ = reg.search("where is the shop", 5);
        let reused = reg.bm25.get_or_build(no_build);
        assert!(
            Arc::ptr_eq(&warmed, &reused),
            "searches between mutations must reuse one index"
        );

        reg.register(fact(
            "extra",
            "extra-fact",
            "an extra fact",
            &[],
            PinMode::Retrieved,
        ));
        let builds = std::cell::Cell::new(0);
        let after_register = reg.bm25.get_or_build(|| {
            builds.set(builds.get() + 1);
            reg.bm25_docs()
        });
        assert_eq!(builds.get(), 1, "register must drop the cached index");
        assert!(!Arc::ptr_eq(&warmed, &after_register));
    }

    #[test]
    fn a_replaced_fact_is_searchable_under_its_new_text() {
        // The behavioral consequence of the invalidation above, through the
        // public seam only: re-registering an id must re-index it, so the new
        // description ranks and the old one no longer does.
        let mut reg = catalog();
        assert_eq!(
            reg.search("cancel or reschedule a booking", 5)
                .first()
                .map(|h| h.fact_id.as_str()),
            Some("cancellation")
        );
        reg.register(fact(
            "cancellation",
            "cancellation-policy",
            "Gift cards, vouchers and store credit",
            &["payment"],
            PinMode::Retrieved,
        ));
        let hits = reg.search("gift cards and vouchers", 5);
        assert_eq!(
            hits.first().map(|h| h.fact_id.as_str()),
            Some("cancellation"),
            "a replaced fact must be searchable under its new text"
        );
    }

    #[test]
    fn search_ranks_the_relevant_fact_first() {
        let reg = catalog();
        let hits = reg.search("how do I cancel and get my money back", 5);
        assert_eq!(
            hits.first().map(|h| h.fact_id.as_str()),
            Some("cancellation")
        );
    }

    #[test]
    fn experimental_searchable_description_replaces_fact_description_but_keeps_name_and_tags() {
        let mut reg = FactRegistry::new();
        let mut overridden = fact(
            "billing",
            "billing_policy",
            "orchestrate zeppelin manifests",
            &["finance_ops"],
            PinMode::Retrieved,
        );
        overridden.experimental_searchable_description = Some("reconcile overdue invoices".into());
        reg.register(overridden);

        assert_eq!(reg.search("overdue invoices", 5)[0].fact_id, "billing");
        assert!(reg.search("zeppelin manifests", 5).is_empty());
        assert_eq!(reg.search("billing", 5)[0].fact_id, "billing");
        assert_eq!(reg.search("finance ops", 5)[0].fact_id, "billing");
    }

    #[test]
    fn search_on_empty_registry_returns_no_hits() {
        let reg = FactRegistry::new();
        assert!(reg.search("anything", 5).is_empty());
    }

    #[test]
    fn pinned_returns_only_always_facts_in_registration_order() {
        let mut reg = FactRegistry::new();
        reg.register(fact("a", "a", "always one", &[], PinMode::Always));
        reg.register(fact("r", "r", "retrieved one", &[], PinMode::Retrieved));
        reg.register(fact("b", "b", "always two", &[], PinMode::Always));
        let pinned: Vec<&str> = reg.pinned().iter().map(|f| f.id.as_str()).collect();
        assert_eq!(
            pinned,
            vec!["a", "b"],
            "only Always facts, in insertion order"
        );
    }

    #[test]
    fn pinned_facts_are_still_query_rankable() {
        // A pinned fact is always-on *and* discoverable — search ranks both tiers.
        let reg = catalog();
        let hits = reg.search("barbershop location and address", 5);
        assert_eq!(
            hits.first().map(|h| h.fact_id.as_str()),
            Some("shop-address")
        );
    }

    #[test]
    fn re_register_replaces_not_appends() {
        let mut reg = FactRegistry::new();
        reg.register(fact("s", "s", "barbershop address", &[], PinMode::Always));
        reg.register(fact(
            "s",
            "s",
            "cancellation and refund",
            &[],
            PinMode::Retrieved,
        ));
        assert_eq!(reg.len(), 1, "re-register replaces, not appends");
        // The pin flag was updated too — no longer pinned.
        assert!(reg.pinned().is_empty(), "replaced fact adopts the new pin");
        let hits = reg.search("cancellation refund", 5);
        assert_eq!(hits.first().map(|h| h.fact_id.as_str()), Some("s"));
        assert_eq!(hits.len(), 1, "one id in the corpus yields at most one hit");
    }

    #[test]
    fn get_returns_the_body() {
        let reg = catalog();
        assert_eq!(
            reg.get("cancellation").map(|f| f.body.as_str()),
            Some("cancellation-policy body")
        );
        assert!(reg.get("nope").is_none());
    }

    #[test]
    fn semantic_ranks_via_injected_embedder() {
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(fact(
            "shop-address",
            "shop-address",
            "shop location",
            &["location"],
            PinMode::Always,
        ));
        reg.register(fact(
            "cancellation",
            "cancellation",
            "cancel and refund",
            &["booking"],
            PinMode::Retrieved,
        ));
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method(
                "the shop location and address",
                5,
                Origin::Direct,
                SearchMethod::Semantic,
            )
            .unwrap();
        assert_eq!(
            hits.first().map(|h| h.fact_id.as_str()),
            Some("shop-address")
        );
    }

    #[test]
    fn semantic_uses_experimental_searchable_description_and_keeps_name_and_tags() {
        let mut overridden_reg = with_embedder(Arc::new(StubEmbedder));
        let mut overridden = fact(
            "target",
            "catalog",
            "shop address",
            &["general"],
            PinMode::Retrieved,
        );
        overridden.experimental_searchable_description = Some("cancel refund policy".into());
        overridden_reg.register(overridden);
        overridden_reg.register(fact(
            "decoy",
            "decoy",
            "shop address",
            &["general"],
            PinMode::Retrieved,
        ));
        overridden_reg.build_embeddings().unwrap();
        let override_hits = overridden_reg
            .search_with_method("cancel refund", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            override_hits.first().map(|h| h.fact_id.as_str()),
            Some("target")
        );

        let mut name_reg = with_embedder(Arc::new(StubEmbedder));
        let mut named = fact(
            "named",
            "shop_address",
            "unrelated",
            &["general"],
            PinMode::Retrieved,
        );
        named.experimental_searchable_description = Some("cancel refund".into());
        name_reg.register(named);
        name_reg.register(fact(
            "name-decoy",
            "decoy",
            "cancel refund",
            &[],
            PinMode::Retrieved,
        ));
        name_reg.build_embeddings().unwrap();
        let name_hits = name_reg
            .search_with_method("shop address", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(name_hits.first().map(|h| h.fact_id.as_str()), Some("named"));

        let mut tag_reg = with_embedder(Arc::new(StubEmbedder));
        let mut tagged = fact(
            "tagged",
            "catalog",
            "unrelated",
            &["location"],
            PinMode::Retrieved,
        );
        tagged.experimental_searchable_description = Some("cancel refund".into());
        tag_reg.register(tagged);
        tag_reg.register(fact(
            "tag-decoy",
            "decoy",
            "cancel refund",
            &[],
            PinMode::Retrieved,
        ));
        tag_reg.build_embeddings().unwrap();
        let tag_hits = tag_reg
            .search_with_method("shop location", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(tag_hits.first().map(|h| h.fact_id.as_str()), Some("tagged"));
    }

    #[test]
    fn re_register_updates_the_ranked_vector() {
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(fact(
            "s",
            "s",
            "shop location address",
            &["location"],
            PinMode::Retrieved,
        ));
        reg.build_embeddings().unwrap();
        reg.register(fact(
            "s",
            "s",
            "cancel booking refund",
            &["booking"],
            PinMode::Retrieved,
        ));
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method(
                "cancel and refund",
                5,
                Origin::Direct,
                SearchMethod::Semantic,
            )
            .unwrap();
        assert_eq!(hits.first().map(|h| h.fact_id.as_str()), Some("s"));
        assert!(hits[0].score > 0.9, "ranks with the re-embedded vector");
    }

    #[test]
    fn hybrid_emits_three_stages() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(fact(
            "shop-address",
            "shop-address",
            "shop location",
            &["location"],
            PinMode::Always,
        ));
        reg.build_embeddings().unwrap();
        reg.search_with_method("location", 5, Origin::Agent, SearchMethod::Hybrid)
            .unwrap();
        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::FactSearch { stages, .. }
                if stages.iter().any(|s| s.name == "bm25")
                && stages.iter().any(|s| s.name == "dense")
                && stages.iter().any(|s| s.name == "rrf")
        )));
    }

    #[test]
    fn register_and_search_emit_trace_events() {
        let sink = Arc::new(MemorySink::new("test-session"));
        let mut reg = FactRegistry::with_trace_sink(sink.clone());
        reg.register(fact(
            "shop-address",
            "shop-address",
            "shop location address",
            &["location"],
            PinMode::Always,
        ));
        reg.search_with_origin("shop address", 5, Origin::Agent);

        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            e.event,
            TraceEvent::FactChurn {
                kind: ChurnKind::Add,
                ..
            }
        )));
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::FactSearch { origin: Origin::Agent, hits, .. } if !hits.is_empty()
        )));
    }
}
