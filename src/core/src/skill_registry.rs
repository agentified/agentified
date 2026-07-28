use std::sync::{Arc, PoisonError, RwLock};
use std::time::Instant;

use indexmap::IndexMap;

use crate::dense_cache::{DenseCache, Embeddable};
use crate::embedding::EmbedderError;
use crate::embedding_config::EmbeddingModel;
use crate::fusion::{RETRIEVE_DEPTH, RRF_K, WeightedArm, rrf_fuse_weighted};
use crate::method::SearchMethod;
use crate::search::bm25_search;
use crate::skill::Skill;
use crate::skill_indexing::searchable_text;
use crate::tool_registry::AdaptiveRankingStatus;
use crate::trace::{
    ChurnKind, NoopSink, Origin, SearchStage, SkillHitTrace, TraceEvent, TraceSink,
};
use crate::usage::{Capability, IntentGraph, UsageArm};

/// One ranked match from a [`SkillRegistry`] search, best-first in the
/// returned `Vec` — the skill-side twin of [`crate::SearchHit`].
pub struct SkillHit {
    /// Id of the matching skill ([`Skill::id`]).
    pub skill_id: String,
    /// Relevance score — higher is better; the scale depends on the
    /// [`SearchMethod`] exactly as documented on [`crate::SearchHit::score`]:
    /// raw BM25 relevance for `Bm25`, cosine similarity (at most `1.0`) for
    /// `Semantic`, a Reciprocal Rank Fusion sum for `Hybrid`. Ties break by
    /// `skill_id` ascending. **Scale also depends on [`fused`](Self::fused)** —
    /// order by [`rank`](Self::rank), branch on [`fused`](Self::fused).
    pub score: f32,
    /// 0-based position in this result list (best is `0`) — the scale-invariant
    /// signal to order or threshold on, in place of [`score`](Self::score). The
    /// skill-side twin of [`crate::SearchHit::rank`].
    pub rank: u32,
    /// Whether [`score`](Self::score) is an RRF score (ordering-only) rather than
    /// the raw method score — the skill-side twin of [`crate::SearchHit::fused`].
    pub fused: bool,
}

/// Build hits from an already-ranked, best-first `(id, score)` list — the
/// skill-side twin of [`crate::tool_registry`]'s `to_search_hits`.
fn to_skill_hits(ranked: Vec<(String, f32)>, fused: bool) -> Vec<SkillHit> {
    ranked
        .into_iter()
        .enumerate()
        .map(|(i, (skill_id, score))| SkillHit {
            skill_id,
            score,
            rank: i as u32,
            fused,
        })
        .collect()
}

impl Embeddable for Skill {
    fn embed_id(&self) -> &str {
        &self.id
    }
    fn embed_text(&self) -> String {
        searchable_text(self)
    }
}

/// Retrieval index over [`Skill`]s — the on-demand analog of
/// [`crate::ToolRegistry`]. Same selectable BM25/semantic/hybrid engines; a
/// parallel type keeps the tool path untouched and lets skill telemetry stand on
/// its own.
pub struct SkillRegistry {
    /// Corpus keyed by skill id, in insertion order — the skill-side twin of
    /// [`crate::ToolRegistry`]'s field. `register` replaces an existing id in
    /// place, never duplicating it (RAT-378).
    skills: IndexMap<String, Skill>,
    sink: Arc<dyn TraceSink>,
    /// Dense embeddings for `skills`, keyed by id and built on demand — the
    /// skill-side twin of [`crate::ToolRegistry`]'s field (see [`DenseCache`]).
    dense: DenseCache,
    /// Optional usage-ranking read model (ADR-0014). `None` — the default — is
    /// today's behavior exactly. Shared behind a lock because the learner writes
    /// to the same graph the search path reads.
    graph: Option<Arc<RwLock<IntentGraph>>>,
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillRegistry {
    /// An empty registry with tracing off ([`NoopSink`]) — see
    /// [`crate::ToolRegistry::new`].
    pub fn new() -> Self {
        Self {
            skills: IndexMap::new(),
            sink: Arc::new(NoopSink),
            dense: DenseCache::new(),
            graph: None,
        }
    }

    /// An empty registry recording trace events to `sink` from the start —
    /// see [`crate::ToolRegistry::with_trace_sink`].
    pub fn with_trace_sink(sink: Arc<dyn TraceSink>) -> Self {
        Self {
            skills: IndexMap::new(),
            sink,
            dense: DenseCache::new(),
            graph: None,
        }
    }

    /// A registry whose semantic/hybrid engines use an explicit embedding model
    /// (the configurable-model path). BM25 is unaffected. Direct enum variants
    /// are validated on the first embedding build; call
    /// [`EmbeddingModel::validate`] first for construction-time feedback. The
    /// trace sink is set separately via [`Self::set_trace_sink`].
    pub fn with_embedding(model: EmbeddingModel) -> Self {
        Self {
            skills: IndexMap::new(),
            sink: Arc::new(NoopSink),
            dense: DenseCache::with_model(model),
            graph: None,
        }
    }

    /// Replace the trace sink; subsequent events go to `sink` — see
    /// [`crate::ToolRegistry::set_trace_sink`].
    pub fn set_trace_sink(&mut self, sink: Arc<dyn TraceSink>) {
        self.sink = sink;
    }

    /// Record an arbitrary [`TraceEvent`] on the registry's sink — see
    /// [`crate::ToolRegistry::record_event`]. The SDK skill catalogs emit
    /// their `skill_invoke` (content-load) events through this.
    pub fn record_event(&self, event: TraceEvent) {
        self.sink.record(event);
    }

    /// Attach (or with `None`, detach) the usage-ranking read model — the
    /// skill-side twin of [`crate::ToolRegistry::set_intent_graph`], reading the
    /// same graph's `skills` edges (ADR-0014).
    ///
    /// Opt-in for the same reason: with an arm in play [`SkillHit::score`]
    /// becomes an RRF score rather than a BM25 one.
    pub fn set_intent_graph(&mut self, graph: Option<Arc<RwLock<IntentGraph>>>) {
        self.graph = graph;
    }

    /// A snapshot of whether adaptive usage ranking is currently contributing, so
    /// the SDK can surface a model-mismatch to the user without draining the trace
    /// stream. Computed from the attached graph's model vs the active embedder.
    /// A dense graph on a catalog with no built embeddings (a BM25 catalog) still
    /// boosts lexically, so it reads `Active`; `Unknown` is reserved for a
    /// poisoned lock, where the state genuinely can't be read.
    pub fn adaptive_ranking_status(&self) -> AdaptiveRankingStatus {
        let Some(graph) = self.graph.as_ref() else {
            return AdaptiveRankingStatus::Inactive;
        };
        // A poisoned lock must not crash a status query — the same policy the
        // search path uses. "Can't tell" is the honest answer, not a panic.
        let Ok(g) = graph.read() else {
            return AdaptiveRankingStatus::Unknown;
        };
        // A lexical graph (no centroids) is model-agnostic — always active.
        if !g.intents.iter().any(|i| i.centroid.is_some()) {
            return AdaptiveRankingStatus::Active;
        }
        // Centroids exist, but no embeddings are built here (a BM25 catalog, which
        // never builds, or one not yet built). Dense matching cannot run, so a
        // query with no vector takes the lexical path and the arm still boosts —
        // model-agnostic and live, so Active rather than Unknown.
        let Some(active_fp) = self.dense.built_fingerprint() else {
            return AdaptiveRankingStatus::Active;
        };
        let active_dim = self.dense.dim().unwrap_or(0);
        match g.model_status(&active_fp, active_dim).describe() {
            None => AdaptiveRankingStatus::Active,
            Some((built, active, dim_mismatch)) => AdaptiveRankingStatus::Paused {
                dim_mismatch,
                built,
                active,
            },
        }
    }

    /// Re-embed the attached intent graph's members under the current model and
    /// replace its centroids — the skill-side twin of
    /// [`crate::ToolRegistry::rebuild_intent_graph`]. Preserves members, support,
    /// and edges.
    ///
    /// # Errors
    ///
    /// Any [`EmbedderError`] from embedding the members under the current model.
    pub fn rebuild_intent_graph(&self) -> Result<(), EmbedderError> {
        let Some(graph) = self.graph.as_ref() else {
            return Ok(());
        };
        // A poisoned lock is recovered rather than a panic: rebuild overwrites
        // every centroid wholesale, so it has no reason to refuse a graph whose
        // state an earlier panic left in doubt (mirrors the tool registry).
        // Snapshot `(id, members)` — centroids are reattached by id, so a
        // concurrent `observe()` that reorders `intents` between here and the
        // write lock cannot misassign them (see `rebuild_centroids`).
        let members: Vec<(String, Vec<String>)> = {
            let g = graph.read().unwrap_or_else(PoisonError::into_inner);
            g.intents
                .iter()
                .map(|i| (i.id.clone(), i.members.clone()))
                .collect()
        };
        let mut per_cluster = Vec::with_capacity(members.len());
        let mut fingerprint = None;
        for (id, cluster_members) in &members {
            let (vectors, fp) = self
                .dense
                .embed_texts_with_identity(cluster_members, self.sink.as_ref())?;
            if !cluster_members.is_empty() {
                fingerprint = Some(fp);
            }
            per_cluster.push((id.clone(), vectors));
        }
        if let Some(fp) = fingerprint {
            let mut g = graph.write().unwrap_or_else(PoisonError::into_inner);
            g.rebuild_centroids(per_cluster, fp);
        }
        Ok(())
    }

    /// Resolve the usage arm for one query and record the outcome. See
    /// `ToolRegistry::usage_arm`; this reads the `skills` edge map instead.
    fn usage_arm(&self, query: &str, query_vec: Option<&[f32]>) -> Option<UsageArm> {
        let graph = self.graph.as_ref()?;
        // The model that embedded this query (semantic/hybrid only), compared
        // against the graph's model so a swap pauses the arm.
        let fingerprint = self.dense.built_fingerprint();
        // Usage ranking is an enhancement; a poisoned lock degrades to today's
        // behavior rather than failing the search.
        let (arm, mismatch) = {
            let guard = graph.read().ok()?;
            let mismatch = match (query_vec, &fingerprint) {
                (Some(v), Some(fp)) => guard.model_status(fp, v.len()).describe(),
                _ => None,
            };
            if mismatch.is_some() {
                (None, mismatch)
            } else {
                if let (Some(v), Some(fp)) = (query_vec, &fingerprint) {
                    guard.note_query_vector(query, v, fp);
                }
                let known = |id: &str| self.skills.contains_key(id);
                (guard.arm(query, query_vec, Capability::Skill, &known), None)
            }
        };
        // The read guard is released BEFORE the sink runs (RwLock is not
        // reentrant and a `UsageLearner` sink takes the write lock).
        if let Some((built, active, dim_mismatch)) = mismatch {
            self.sink.record(TraceEvent::UsageModelMismatch {
                built,
                active,
                dim_mismatch,
            });
        }
        self.sink.record(TraceEvent::UsageBoost {
            intent: arm.as_ref().map(|a| a.intent_id.clone()),
            similarity: arm.as_ref().map_or(0.0, |a| a.similarity as f64),
            support: arm.as_ref().map_or(0, |a| a.support),
            promoted: arm.as_ref().map_or(0, |a| a.ids.len() as u32),
        });
        arm
    }

    /// The corpus as `(id, searchable_text)` pairs for BM25.
    fn bm25_docs(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.skills
            .values()
            .map(|s| (s.id.clone(), searchable_text(s)))
    }

    /// Fuse the ranked arms into the final top-`top_k`, returning the hits and
    /// the `rrf` stage — one implementation for all three engines.
    fn fuse_arms(arms: &[WeightedArm<'_>], top_k: usize) -> (Vec<SkillHit>, SearchStage) {
        let t = Instant::now();
        let mut fused = rrf_fuse_weighted(arms, RRF_K);
        fused.truncate(top_k);
        let stage = SearchStage {
            name: "rrf".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: fused.first().map(|(_, s)| *s as f64),
        };
        // Ordering-only RRF scores.
        let hits = to_skill_hits(fused, true);
        (hits, stage)
    }

    /// The `usage` stage descriptor for a matched arm; `top_score` carries the
    /// arm's fusion weight, the only scalar it has.
    fn usage_stage(arm: &UsageArm, took_ms: u64) -> SearchStage {
        SearchStage {
            name: "usage".into(),
            took_ms,
            top_score: Some(arm.weight() as f64),
        }
    }

    /// Register a skill, or replace one in place if its id is already present —
    /// see [`crate::ToolRegistry::register`]. Replacing invalidates the old id's
    /// cached embedding; the corpus never holds a duplicate.
    pub fn register(&mut self, skill: Skill) {
        let skill_id = skill.id.clone();
        if self.skills.insert(skill_id.clone(), skill).is_some() {
            // Replaced an existing id: drop its stale embedding.
            self.dense.invalidate(&skill_id);
        }
        self.sink.record(TraceEvent::SkillChurn {
            kind: ChurnKind::Add,
            skill_id,
        });
    }

    /// Number of registered skills (distinct ids).
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Whether no skills are registered.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Lexical BM25 retrieval — the skill-side twin of
    /// [`crate::ToolRegistry::search`]: no model, never fails. Returns at most
    /// `top_k` hits, best-first (see [`SkillHit::score`]). Traced as
    /// [`Origin::Direct`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ratel_ai_core::{Skill, SkillRegistry};
    ///
    /// let mut registry = SkillRegistry::new();
    /// registry.register(Skill {
    ///     id: "api-design".into(),
    ///     name: "api-design".into(),
    ///     description: "REST API design patterns: resource naming, pagination".into(),
    ///     tags: vec!["backend".into(), "api".into()],
    ///     tools: vec![],
    ///     metadata: std::collections::HashMap::new(),
    ///     body: "# API design\n...".into(),
    /// });
    ///
    /// let hits = registry.search("design a REST endpoint", 5);
    /// assert_eq!(hits[0].skill_id, "api-design");
    /// ```
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SkillHit> {
        self.search_with_origin(query, top_k, Origin::Direct)
    }

    /// [`Self::search`] with an explicit trace [`Origin`] — see
    /// [`crate::ToolRegistry::search_with_origin`].
    pub fn search_with_origin(&self, query: &str, top_k: usize, origin: Origin) -> Vec<SkillHit> {
        self.bm25_search_traced(query, top_k, origin)
    }

    /// Retrieve with an explicit [`SearchMethod`]. See
    /// [`crate::ToolRegistry::search_with_method`].
    ///
    /// # Errors
    ///
    /// Never errors for [`SearchMethod::Bm25`]; for `Semantic`/`Hybrid`, the
    /// same [`EmbedderError`] cases as
    /// [`crate::ToolRegistry::search_with_method`].
    pub fn search_with_method(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        method: SearchMethod,
    ) -> Result<Vec<SkillHit>, EmbedderError> {
        match method {
            SearchMethod::Bm25 => Ok(self.bm25_search_traced(query, top_k, origin)),
            SearchMethod::Semantic => self.semantic_search_traced(query, top_k, origin),
            SearchMethod::Hybrid => self.hybrid_search_traced(query, top_k, origin),
        }
    }

    /// Pre-compute embeddings for not-yet-embedded skills — see
    /// [`crate::ToolRegistry::build_embeddings`].
    ///
    /// # Errors
    ///
    /// The same [`EmbedderError`] cases as
    /// [`crate::ToolRegistry::build_embeddings`]: model download/cache/load
    /// failures on first use, or an `Inference` failure embedding a skill.
    pub fn build_embeddings(&self) -> Result<(), EmbedderError> {
        self.dense.extend(self.skills.values(), self.sink.as_ref())
    }

    /// Recompute embeddings for the full skill corpus and atomically replace the
    /// dense cache. A changed model identity or dimension is adopted only after
    /// the complete rebuild succeeds; failures preserve the prior cache.
    ///
    /// # Errors
    ///
    /// Any [`EmbedderError`] from loading or embedding the complete corpus.
    pub fn rebuild_embeddings(&self) -> Result<(), EmbedderError> {
        self.dense.rebuild(self.skills.values(), self.sink.as_ref())
    }

    // ---- engines -----------------------------------------------------------

    fn bm25_search_traced(&self, query: &str, top_k: usize, origin: Origin) -> Vec<SkillHit> {
        let started = Instant::now();
        let t = Instant::now();
        let arm = self.usage_arm(query, None);
        let usage_ms = t.elapsed().as_millis() as u64;

        let Some(arm) = arm else {
            // No graph, or nothing matched: the original path with raw BM25
            // scores, unchanged.
            // Raw BM25 scores — not fused.
            let hits = to_skill_hits(bm25_search(self.bm25_docs(), query, top_k), false);
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
            return hits;
        };

        let depth = RETRIEVE_DEPTH.max(top_k);
        let t = Instant::now();
        let bm25_ranked = bm25_search(self.bm25_docs(), query, depth);
        let bm25_stage = SearchStage {
            name: "bm25".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: bm25_ranked.first().map(|(_, s)| *s as f64),
        };
        let bm25_ids: Vec<String> = bm25_ranked.into_iter().map(|(id, _)| id).collect();

        let (hits, rrf_stage) =
            Self::fuse_arms(&[(&bm25_ids, 1.0), (&arm.ids, arm.weight())], top_k);
        let took_ms = started.elapsed().as_millis() as u64;
        self.record_search(
            query,
            origin,
            top_k,
            &hits,
            vec![bm25_stage, Self::usage_stage(&arm, usage_ms), rrf_stage],
            took_ms,
        );
        hits
    }

    fn semantic_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
    ) -> Result<Vec<SkillHit>, EmbedderError> {
        let started = Instant::now();
        if self.skills.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0);
            return Ok(Vec::new());
        }
        // Retrieve deeper only when a graph is attached; without one the depth,
        // scores, and stages stay exactly as they were.
        let depth = if self.graph.is_some() {
            RETRIEVE_DEPTH.max(top_k)
        } else {
            top_k
        };
        let t = Instant::now();
        let (ranked, query_vec) = self.dense.search_returning_query_vec(
            self.skills.values(),
            query,
            depth,
            self.sink.as_ref(),
        )?;
        let stage_ms = t.elapsed().as_millis() as u64;

        // Reuses the vector the dense arm just embedded — no second inference.
        let t = Instant::now();
        let arm = self.usage_arm(query, Some(&query_vec));
        let usage_ms = t.elapsed().as_millis() as u64;

        let Some(arm) = arm else {
            // Raw cosine scores — not fused. Retrieval ran deeper than `top_k`
            // to give the usage arm room to re-rank; with no arm to fuse, trim
            // back to what the caller asked for (the fused path does this in
            // `fuse_arms`).
            let mut hits = to_skill_hits(ranked, false);
            hits.truncate(top_k);
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
            return Ok(hits);
        };

        let dense_stage = SearchStage {
            name: "dense".into(),
            took_ms: stage_ms,
            top_score: ranked.first().map(|(_, s)| *s as f64),
        };
        let dense_ids: Vec<String> = ranked.into_iter().map(|(id, _)| id).collect();
        let (hits, rrf_stage) =
            Self::fuse_arms(&[(&dense_ids, 1.0), (&arm.ids, arm.weight())], top_k);
        let took_ms = started.elapsed().as_millis() as u64;
        self.record_search(
            query,
            origin,
            top_k,
            &hits,
            vec![dense_stage, Self::usage_stage(&arm, usage_ms), rrf_stage],
            took_ms,
        );
        Ok(hits)
    }

    fn hybrid_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
    ) -> Result<Vec<SkillHit>, EmbedderError> {
        let started = Instant::now();
        if self.skills.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0);
            return Ok(Vec::new());
        }
        let depth = RETRIEVE_DEPTH.max(top_k);

        let t = Instant::now();
        let bm25_ranked = bm25_search(
            self.skills
                .values()
                .map(|s| (s.id.clone(), searchable_text(s))),
            query,
            depth,
        );
        let bm25_stage = SearchStage {
            name: "bm25".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: bm25_ranked.first().map(|(_, s)| *s as f64),
        };

        let t = Instant::now();
        let (dense_ranked, query_vec) = self.dense.search_returning_query_vec(
            self.skills.values(),
            query,
            depth,
            self.sink.as_ref(),
        )?;
        let dense_stage = SearchStage {
            name: "dense".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: dense_ranked.first().map(|(_, s)| *s as f64),
        };

        // Usage arm, matched on the vector the dense arm already embedded.
        let t = Instant::now();
        let arm = self.usage_arm(query, Some(&query_vec));
        let usage_ms = t.elapsed().as_millis() as u64;

        let bm25_ids: Vec<String> = bm25_ranked.into_iter().map(|(id, _)| id).collect();
        let dense_ids: Vec<String> = dense_ranked.into_iter().map(|(id, _)| id).collect();
        let mut arms: Vec<WeightedArm<'_>> = vec![(&bm25_ids, 1.0), (&dense_ids, 1.0)];
        if let Some(arm) = &arm {
            arms.push((&arm.ids, arm.weight()));
        }
        let (hits, rrf_stage) = Self::fuse_arms(&arms, top_k);

        let mut stages = vec![bm25_stage, dense_stage];
        if let Some(arm) = &arm {
            stages.push(Self::usage_stage(arm, usage_ms));
        }
        stages.push(rrf_stage);

        let took_ms = started.elapsed().as_millis() as u64;
        self.record_search(query, origin, top_k, &hits, stages, took_ms);
        Ok(hits)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_search(
        &self,
        query: &str,
        origin: Origin,
        top_k: usize,
        hits: &[SkillHit],
        stages: Vec<SearchStage>,
        took_ms: u64,
    ) {
        self.sink.record(TraceEvent::SkillSearch {
            query: query.to_string(),
            origin,
            top_k: top_k as u32,
            hits: hits
                .iter()
                .map(|h| SkillHitTrace {
                    skill_id: h.skill_id.clone(),
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
    use crate::embedding::Embedder;
    use crate::trace::MemorySink;

    struct StubEmbedder;
    impl StubEmbedder {
        fn vec_for(text: &str) -> Vec<f32> {
            let t = text.to_lowercase();
            if t.contains("api") || t.contains("rest") {
                vec![1.0, 0.0, 0.0]
            } else if t.contains("frontend") || t.contains("slides") {
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

    /// Counts `embed_doc` calls (see `tool_registry`'s `CountingEmbedder`).
    struct CountingEmbedder {
        doc_calls: std::sync::atomic::AtomicUsize,
    }
    impl CountingEmbedder {
        fn new() -> Self {
            Self {
                doc_calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn doc_calls(&self) -> usize {
            self.doc_calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    impl Embedder for CountingEmbedder {
        fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            self.doc_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(StubEmbedder::vec_for(text))
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(StubEmbedder::vec_for(text))
        }
    }

    fn with_embedder(embedder: Arc<dyn Embedder>) -> SkillRegistry {
        SkillRegistry {
            skills: IndexMap::new(),
            sink: Arc::new(NoopSink),
            dense: DenseCache::with_embedder(embedder),
            graph: None,
        }
    }

    fn skill(id: &str, name: &str, description: &str, tags: &[&str]) -> Skill {
        Skill {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            tags: tags.iter().map(|t| (*t).into()).collect(),
            tools: vec![],
            metadata: std::collections::HashMap::new(),
            body: format!("# {name}\n\nbody"),
        }
    }

    fn catalog() -> SkillRegistry {
        let mut reg = SkillRegistry::new();
        reg.register(skill(
            "frontend-slides",
            "frontend-slides",
            "Build animation-rich HTML presentations from scratch",
            &["frontend", "presentations"],
        ));
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design patterns: resource naming, status codes, pagination",
            &["backend", "api"],
        ));
        reg
    }

    #[test]
    fn semantic_search_truncates_to_top_k_when_the_graph_matches_no_cluster() {
        // Cold start: an empty graph is still attached, so retrieval depth jumps
        // to RETRIEVE_DEPTH — but no cluster can ever match. The no-match path
        // must still hand back exactly `top_k`, not the deep candidate list.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        // Four skills that all embed to the stub's "api" vector, so dense ranks
        // every one of them at cosine 1.0 — the deep list holds 4 entries.
        reg.register(skill("api_a", "api_a", "rest api design", &[]));
        reg.register(skill("api_b", "api_b", "rest api pagination", &[]));
        reg.register(skill("api_c", "api_c", "rest api auth", &[]));
        reg.register(skill("api_d", "api_d", "rest api versioning", &[]));
        reg.build_embeddings().unwrap();
        reg.set_intent_graph(Some(Arc::new(RwLock::new(IntentGraph::empty()))));

        let hits = reg
            .search_with_method("api", 2, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        assert_eq!(hits.len(), 2, "no-match dense path must honor top_k");
    }

    #[test]
    fn skill_hits_carry_rank_and_unfused_scores_without_a_graph() {
        let mut reg = SkillRegistry::new();
        reg.register(skill(
            "design-api",
            "design-api",
            "design a REST endpoint",
            &[],
        ));
        reg.register(skill(
            "html-slides",
            "html-slides",
            "build html slide decks",
            &[],
        ));
        let hits = reg.search("design a REST endpoint", 5);
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.rank, i as u32);
            assert!(!h.fused, "no graph → not fused");
        }
    }

    #[test]
    fn search_ranks_the_relevant_skill_first() {
        let reg = catalog();
        let hits = reg.search("design a REST endpoint with pagination", 5);
        assert_eq!(
            hits.first().map(|h| h.skill_id.as_str()),
            Some("api-design")
        );
    }

    #[test]
    fn search_on_empty_registry_returns_no_hits() {
        let reg = SkillRegistry::new();
        assert!(reg.search("anything", 5).is_empty());
    }

    #[test]
    fn re_register_replaces_not_appends() {
        // Re-registering a skill id replaces it in place — the corpus holds one
        // entry per id, no duplicate (RAT-378, mirror of the tool path).
        let mut reg = SkillRegistry::new();
        reg.register(skill("s", "s", "REST API design", &["api"]));
        reg.register(skill("s", "s", "HTML slides frontend", &["frontend"]));
        assert_eq!(reg.len(), 1, "re-register replaces, not appends");
        let hits = reg.search("html slides frontend", 5);
        assert_eq!(hits.first().map(|h| h.skill_id.as_str()), Some("s"));
        assert_eq!(hits.len(), 1, "one id in the corpus yields at most one hit");
    }

    #[test]
    fn re_register_updates_the_ranked_vector() {
        // Replace-in-place invalidates the old embedding; after rebuild a semantic
        // query for the new content ranks the re-registered skill first.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(skill("s", "s", "REST API design", &["api"])); // dense: api bucket
        reg.build_embeddings().unwrap();
        reg.register(skill("s", "s", "HTML slides frontend", &["frontend"])); // → frontend bucket
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method("frontend slides", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(hits.first().map(|h| h.skill_id.as_str()), Some("s"));
        assert!(
            hits[0].score > 0.9,
            "ranks with the re-embedded frontend vector"
        );
    }

    #[test]
    fn semantic_ranks_via_injected_embedder() {
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.register(skill(
            "frontend-slides",
            "frontend-slides",
            "HTML slides",
            &["frontend"],
        ));
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method("rest api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            hits.first().map(|h| h.skill_id.as_str()),
            Some("api-design")
        );
    }

    #[test]
    fn build_embeddings_after_register_embeds_only_the_new_skill() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.register(skill("frontend", "frontend", "HTML slides", &["frontend"]));
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 2);
        reg.register(skill("api-v2", "api-v2", "REST API v2", &["api"]));
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 3, "only the new skill is embedded");
    }

    #[test]
    fn build_embeddings_precomputes_so_search_embeds_no_docs() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 1);
        reg.search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            counter.doc_calls(),
            1,
            "a search after build_embeddings embeds only the query"
        );
    }

    #[test]
    fn rebuild_embeddings_recomputes_the_full_skill_corpus() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.register(skill("frontend", "frontend", "HTML slides", &["frontend"]));
        reg.build_embeddings().unwrap();
        reg.rebuild_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 4, "rebuild embeds every skill again");
    }

    #[test]
    fn hybrid_emits_three_stages() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.build_embeddings().unwrap();
        reg.search_with_method("api", 5, Origin::Agent, SearchMethod::Hybrid)
            .unwrap();
        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::SkillSearch { stages, .. }
                if stages.iter().any(|s| s.name == "bm25")
                && stages.iter().any(|s| s.name == "dense")
                && stages.iter().any(|s| s.name == "rrf")
        )));
    }

    #[test]
    fn register_and_search_emit_trace_events() {
        let sink = Arc::new(MemorySink::new("test-session"));
        let mut reg = SkillRegistry::with_trace_sink(sink.clone());
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.search_with_origin("api design", 5, Origin::Agent);

        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            e.event,
            TraceEvent::SkillChurn {
                kind: ChurnKind::Add,
                ..
            }
        )));
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::SkillSearch { origin: Origin::Agent, hits, .. } if !hits.is_empty()
        )));
    }

    // ---- usage ranking on the dense paths (ADR-0014) -----------------------
    // Twins of the ToolRegistry battery: the skill registry runs its own copy of
    // `usage_arm`/`semantic_search_traced`/`rebuild_intent_graph`, so the
    // model-mismatch, poison, and rebuild guarantees need their own coverage.

    /// A graph whose single cluster boosts `skill_id`, carrying `centroid`
    /// stamped with `model`. The member text embeds to the stub's "api" vector.
    fn graph_with_model(
        skill_id: &str,
        centroid: Vec<f32>,
        model: &str,
    ) -> Arc<RwLock<IntentGraph>> {
        let c: Vec<String> = centroid.iter().map(|x| x.to_string()).collect();
        let json = format!(
            r#"{{"v":1,"built_from_ts":1,"model":"{model}",
                 "intents":[{{"id":"i0","label":"l","terms":[],
                 "members":["rest api design"],"centroid":[{}],
                 "support":9,"tools":{{}},"skills":{{"{skill_id}":1.0}}}}]}}"#,
            c.join(",")
        );
        Arc::new(RwLock::new(IntentGraph::from_json(&json).expect("valid")))
    }

    fn model_mismatch_events(sink: &MemorySink) -> Vec<(String, String, bool)> {
        sink.drain()
            .into_iter()
            .filter_map(|e| match e.event {
                TraceEvent::UsageModelMismatch {
                    built,
                    active,
                    dim_mismatch,
                } => Some((built, active, dim_mismatch)),
                _ => None,
            })
            .collect()
    }

    /// Two skills: `api-design` matches an "api" query at cosine 1.0; `frontend`
    /// embeds orthogonally, so dense ranks it last.
    fn mismatch_registry(sink: Arc<MemorySink>) -> SkillRegistry {
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink);
        reg.register(skill("api-design", "api-design", "rest api design", &[]));
        reg.register(skill("frontend", "frontend", "frontend slides", &[]));
        reg.build_embeddings().unwrap();
        reg
    }

    #[test]
    fn a_same_dim_model_mismatch_pauses_the_arm_and_warns() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = mismatch_registry(sink.clone());

        // Graph says boost frontend for this intent — but its centroid was built
        // by a different model (same 3-dim width the stub uses).
        reg.set_intent_graph(Some(graph_with_model(
            "frontend",
            vec![1.0, 0.0, 0.0],
            "a-different-model",
        )));
        let hits = reg
            .search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        // Paused: frontend is NOT lifted; it stays last, as with no graph.
        assert_eq!(hits.last().map(|h| h.skill_id.as_str()), Some("frontend"));
        assert!(hits.iter().all(|h| !h.fused), "no fusion — the arm paused");

        let events = model_mismatch_events(&sink);
        assert_eq!(events.len(), 1);
        assert!(!events[0].2, "same-dim swap → dim_mismatch false");
    }

    #[test]
    fn a_dim_mismatch_pauses_the_arm_and_warns() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = mismatch_registry(sink.clone());

        // Centroid is 5-dim; the stub embeds queries to 3-dim.
        reg.set_intent_graph(Some(graph_with_model(
            "frontend",
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            "some-model",
        )));
        let hits = reg
            .search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        assert_eq!(hits.last().map(|h| h.skill_id.as_str()), Some("frontend"));
        let events = model_mismatch_events(&sink);
        assert_eq!(events.len(), 1);
        assert!(events[0].2, "different width → dim_mismatch true");
    }

    #[test]
    fn rebuild_intent_graph_restores_the_arm_after_a_model_change() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = mismatch_registry(sink.clone());
        reg.set_intent_graph(Some(graph_with_model(
            "api-design",
            vec![1.0, 0.0, 0.0],
            "a-different-model",
        )));

        // Paused before rebuild.
        reg.search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(model_mismatch_events(&sink).len(), 1);

        // Rebuild re-embeds members under the stub model → arm active again.
        reg.rebuild_intent_graph().unwrap();
        let after = reg
            .search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert!(after.iter().all(|h| h.fused), "arm resumed → fused ranking");
        assert!(
            model_mismatch_events(&sink).is_empty(),
            "no mismatch after rebuild"
        );
    }

    /// Poison a graph's lock the only way locks poison: panic while holding the
    /// write guard. The join swallows the panic; the lock stays poisoned.
    fn poison(graph: &Arc<RwLock<IntentGraph>>) {
        let g = graph.clone();
        let _ = std::thread::spawn(move || {
            let _guard = g.write().expect("first writer takes the lock");
            panic!("intentional poison");
        })
        .join();
        assert!(
            graph.is_poisoned(),
            "lock should be poisoned after the panic"
        );
    }

    #[test]
    fn a_dense_graph_on_a_bm25_catalog_reports_active_not_unknown() {
        // A BM25 catalog never builds embeddings, so a centroid-bearing graph can
        // only boost lexically — and it does. Status must report the arm is live.
        let mut reg = SkillRegistry::new();
        reg.register(skill("api-design", "api-design", "rest api design", &[]));
        reg.set_intent_graph(Some(graph_with_model(
            "api-design",
            vec![1.0, 0.0, 0.0],
            "some-model",
        )));
        assert_eq!(reg.adaptive_ranking_status(), AdaptiveRankingStatus::Active);
    }

    #[test]
    fn a_poisoned_graph_lock_reports_unknown_not_a_panic() {
        // The search path degrades on a poisoned lock; a status query must too —
        // a read-only getter that can crash the caller is a footgun.
        let mut reg = SkillRegistry::new();
        reg.register(skill("api-design", "api-design", "rest api design", &[]));
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        poison(&graph);
        reg.set_intent_graph(Some(graph));

        assert_eq!(
            reg.adaptive_ranking_status(),
            AdaptiveRankingStatus::Unknown
        );
    }

    #[test]
    fn a_poisoned_graph_lock_degrades_the_search_path_not_a_panic() {
        // The load-bearing guarantee: a search over a poisoned graph must fall
        // back to plain ranking, never panic.
        let mut reg = mismatch_registry(Arc::new(MemorySink::new("s")));
        let graph = graph_with_model("frontend", vec![0.0, 1.0, 0.0], "m");
        poison(&graph);
        reg.set_intent_graph(Some(graph));

        let hits = reg
            .search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert!(hits.iter().all(|h| !h.fused), "poisoned lock → arm paused");
    }

    #[test]
    fn rebuild_recovers_a_poisoned_graph_lock_not_a_panic() {
        // rebuild is the repair path — it overwrites every centroid, so a lock an
        // earlier panic poisoned is recovered and the call completes, never panics.
        let mut reg = SkillRegistry::new();
        reg.register(skill("api-design", "api-design", "rest api design", &[]));
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        poison(&graph);
        reg.set_intent_graph(Some(graph));

        assert!(reg.rebuild_intent_graph().is_ok());
    }
}
