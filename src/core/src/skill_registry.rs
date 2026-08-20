use std::sync::{Arc, PoisonError, RwLock};
use std::time::Instant;

use indexmap::IndexMap;

use crate::artifact_warm::{ArtifactWarmError, OnArtifactMiss};
use crate::dense_cache::{DenseCache, Embeddable};
use crate::embedding::EmbedderError;
use crate::embedding_artifact::{ArtifactEntryKind, ArtifactError};
use crate::embedding_config::EmbeddingModel;
use crate::fusion::{RETRIEVE_DEPTH, RRF_K, WeightedArm, rrf_fuse_weighted};
use crate::method::SearchMethod;
use crate::search::Bm25Cache;
use crate::skill::Skill;
use crate::skill_indexing::searchable_text;
use crate::tool_registry::AdaptiveRankingStatus;
use crate::trace::{
    ChurnKind, NoopSink, Origin, SearchStage, SkillHitTrace, TraceEvent, TraceEventContext,
    TraceSink,
};
use crate::usage::{ArmOutcome, Capability, IntentGraph, UsageArm};

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

/// What a whole-corpus [`SkillRegistry::replace_all`] actually changed, counted
/// by id. `updated` covers any field edit (including a body-only rewrite);
/// `unchanged` ids are byte-identical and keep their cached embedding. A reload
/// that changed nothing reports zeros across `added`/`removed`/`updated` — the
/// cheap case a periodic source hits most of the time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplaceOutcome {
    /// Ids in the new corpus that were not in the old one.
    pub added: usize,
    /// Ids in the old corpus that are absent from the new one.
    pub removed: usize,
    /// Ids present in both whose content differs in any field.
    pub updated: usize,
    /// Ids present in both with identical content.
    pub unchanged: usize,
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
    experimental_catalog_definitions: bool,
    /// Prebuilt BM25 index over `skills` — the skill-side twin of
    /// [`crate::ToolRegistry`]'s field: built lazily by the first search,
    /// reused until [`Self::register`] or [`Self::replace_all`] invalidates it.
    bm25: Bm25Cache,
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
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
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
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
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
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
            dense: DenseCache::with_model(model),
            graph: None,
        }
    }

    /// Replace the trace sink; subsequent events go to `sink` — see
    /// [`crate::ToolRegistry::set_trace_sink`].
    pub fn set_trace_sink(&mut self, sink: Arc<dyn TraceSink>) {
        self.sink = sink;
    }

    /// Enable experimental complete catalog-definition events for later registrations.
    pub fn experimental_enable_catalog_definitions(&mut self) {
        self.experimental_catalog_definitions = true;
    }

    /// Record an arbitrary [`TraceEvent`] on the registry's sink — see
    /// [`crate::ToolRegistry::record_event`]. The SDK skill catalogs emit
    /// their `skill_invoke` (content-load) events through this.
    pub fn record_event(&self, event: TraceEvent) {
        self.sink.record(event);
    }

    /// Record an arbitrary event with per-emission envelope correlation.
    pub fn record_event_with_context(&self, event: TraceEvent, context: TraceEventContext) {
        self.sink.record_with_context(event, context);
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
        let (outcome, mismatch) = {
            let guard = graph.read().ok()?;
            let mismatch = match (query_vec, &fingerprint) {
                (Some(v), Some(fp)) => guard.model_status(fp, v.len()).describe(),
                _ => None,
            };
            if mismatch.is_some() {
                (ArmOutcome::NoMatch, mismatch)
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
        let (intent, similarity, support, promoted, dropped) = outcome.describe();
        self.sink.record(TraceEvent::UsageBoost {
            intent,
            similarity,
            support,
            promoted,
            dropped,
        });
        // Trace-only: a drift report never reaches the fusion, so ranking is
        // bit-identical to before this distinction existed.
        outcome.into_arm()
    }

    /// The corpus as `(id, searchable_text)` pairs for BM25.
    fn bm25_docs(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.skills
            .values()
            .map(|s| (s.id.clone(), searchable_text(s)))
    }

    /// The prebuilt BM25 index for the current corpus — cached across
    /// searches, rebuilt on the first search after a mutation.
    fn bm25_index(&self) -> Arc<crate::search::Bm25Index> {
        self.bm25.get_or_build(|| self.bm25_docs())
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
        let definition = self
            .experimental_catalog_definitions
            .then(|| TraceEvent::catalog_definition_for_skill(&skill))
            .flatten();
        let definition_changed = definition.as_ref().is_some_and(|definition| {
            self.skills.get(&skill_id).is_none_or(|existing| {
                let existing_definition = TraceEvent::catalog_definition_for_skill(existing);
                existing_definition
                    .as_ref()
                    .and_then(TraceEvent::catalog_definition_hash)
                    != definition.catalog_definition_hash()
            })
        });
        // Add or replace, the corpus changed either way: the prebuilt BM25
        // index no longer matches it.
        self.bm25.invalidate();
        if self.skills.insert(skill_id.clone(), skill).is_some() {
            // Replaced an existing id: drop its stale embedding.
            self.dense.invalidate(&skill_id);
        }
        self.sink.record(TraceEvent::SkillChurn {
            kind: ChurnKind::Add,
            skill_id,
        });
        if definition_changed && let Some(definition) = definition {
            self.sink.record(definition);
        }
    }

    /// Replace the entire corpus with `skills`: ids absent from the batch are
    /// removed, ids present are added or updated. The whole-catalog counterpart
    /// to [`Self::register`], for a source that reloads a catalog rather than
    /// pushing individual changes. Within one batch a repeated id keeps its last
    /// entry, exactly as a repeated [`Self::register`] would.
    ///
    /// Synchronous and infallible: it swaps the corpus and maintains the dense
    /// cache, but never embeds. Call [`Self::build_embeddings`] afterwards on a
    /// semantic/hybrid catalog — it then embeds only the ids this replace
    /// actually invalidated.
    ///
    /// Cache handling is deliberately narrow, so a reload of a mostly-unchanged
    /// catalog costs no embeddings: a removed id's vector is dropped, an id whose
    /// indexed text (`name`/effective searchable description/`tags`) changed is invalidated for
    /// re-embedding, and everything else keeps the vector it already had —
    /// including an id whose `body`, `tools`, or `metadata` changed, since none
    /// of those are embedded.
    ///
    /// Dropping a removed id's vector is load-bearing, not just tidiness: the
    /// dense cache's built-ness guard compares *counts*, so a stale vector left
    /// behind could offset a new, unembedded id and let a semantic search
    /// silently omit it.
    ///
    /// # Examples
    ///
    /// ```
    /// use ratel_ai_core::{Skill, SkillRegistry};
    ///
    /// fn skill(id: &str, description: &str) -> Skill {
    ///     Skill {
    ///         id: id.into(),
    ///         name: id.into(),
    ///         description: description.into(),
    ///         experimental_searchable_description: None,
    ///         tags: vec![],
    ///         tools: vec![],
    ///         metadata: std::collections::HashMap::new(),
    ///         body: String::new(),
    ///     }
    /// }
    ///
    /// let mut registry = SkillRegistry::new();
    /// registry.register(skill("api-design", "REST API design patterns"));
    /// registry.register(skill("slides", "Build HTML presentations"));
    ///
    /// // The reloaded catalog no longer carries `slides`.
    /// let outcome = registry.replace_all(vec![
    ///     skill("api-design", "REST API design patterns"),
    ///     skill("migrations", "Write reversible database migrations"),
    /// ]);
    ///
    /// assert_eq!((outcome.added, outcome.removed, outcome.unchanged), (1, 1, 1));
    /// assert!(registry.search("build HTML presentations", 5).is_empty());
    /// ```
    pub fn replace_all(&mut self, skills: Vec<Skill>) -> ReplaceOutcome {
        let mut next: IndexMap<String, Skill> = IndexMap::with_capacity(skills.len());
        for skill in skills {
            next.insert(skill.id.clone(), skill);
        }

        let mut outcome = ReplaceOutcome::default();
        // Whether any id's *indexed* text changed. Only then does the prebuilt
        // BM25 index go stale — the same diff the dense cache keys on, so a
        // reload of an unchanged catalog (or one that only edited bodies)
        // keeps both caches and the next search rebuilds nothing.
        let mut indexed_text_changed = false;

        for id in self.skills.keys() {
            if !next.contains_key(id) {
                self.dense.invalidate(id);
                indexed_text_changed = true;
                self.sink.record(TraceEvent::SkillChurn {
                    kind: ChurnKind::Remove,
                    skill_id: id.clone(),
                });
                outcome.removed += 1;
            }
        }

        for (id, skill) in &next {
            let definition = self
                .experimental_catalog_definitions
                .then(|| TraceEvent::catalog_definition_for_skill(skill))
                .flatten();
            let definition_changed = definition.as_ref().is_some_and(|definition| {
                self.skills.get(id).is_none_or(|existing| {
                    let existing_definition = TraceEvent::catalog_definition_for_skill(existing);
                    existing_definition
                        .as_ref()
                        .and_then(TraceEvent::catalog_definition_hash)
                        != definition.catalog_definition_hash()
                })
            });
            match self.skills.get(id) {
                Some(current) if current == skill => {
                    outcome.unchanged += 1;
                    continue;
                }
                Some(current) => {
                    if searchable_text(current) != searchable_text(skill) {
                        self.dense.invalidate(id);
                        indexed_text_changed = true;
                    }
                    outcome.updated += 1;
                }
                None => {
                    indexed_text_changed = true;
                    outcome.added += 1;
                }
            }
            self.sink.record(TraceEvent::SkillChurn {
                kind: ChurnKind::Add,
                skill_id: id.clone(),
            });
            if definition_changed && let Some(definition) = definition {
                self.sink.record(definition);
            }
        }

        if indexed_text_changed {
            self.bm25.invalidate();
        }
        self.skills = next;
        outcome
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
    ///     experimental_searchable_description: None,
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
        self.search_with_method_and_context(
            query,
            top_k,
            origin,
            method,
            TraceEventContext::default(),
        )
    }

    /// Search with caller-supplied event identity and OTel correlation.
    pub fn search_with_method_and_context(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        method: SearchMethod,
        context: TraceEventContext,
    ) -> Result<Vec<SkillHit>, EmbedderError> {
        match method {
            SearchMethod::Bm25 => {
                Ok(self.bm25_search_traced_with_context(query, top_k, origin, context))
            }
            SearchMethod::Semantic => self.semantic_search_traced(query, top_k, origin, context),
            SearchMethod::Hybrid => self.hybrid_search_traced(query, top_k, origin, context),
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

    /// Load corpus vectors from a build-time embedding artifact, then apply [`OnArtifactMiss`].
    ///
    /// For [`OnArtifactMiss::Embed`], reuse commit and embedding of missing ids run
    /// under one dense operation write lock so semantic search cannot observe a
    /// partially warmed cache. This is serialization only — if the follow-up
    /// embed fails after reuse commit, prior committed vectors are retained
    /// (no rollback).
    ///
    /// # Errors
    ///
    /// [`ArtifactWarmError::Warm`] from parse / kind / model-mismatch during warm;
    /// [`ArtifactWarmError::Incomplete`] when `on_miss` is [`OnArtifactMiss::Error`]
    /// and some corpus ids were not reused; [`ArtifactWarmError::Embedder`] when
    /// `on_miss` is [`OnArtifactMiss::Embed`] and embedding the missing ids fails.
    pub fn warm_embeddings_from_artifact(
        &self,
        bytes: &[u8],
        on_miss: OnArtifactMiss,
    ) -> Result<(), ArtifactWarmError> {
        match on_miss {
            OnArtifactMiss::Error => {
                let outcome = self.dense.warm_from_artifact(
                    bytes,
                    ArtifactEntryKind::Skill,
                    self.skills.values(),
                    self.sink.as_ref(),
                )?;
                if outcome.missing.is_empty() {
                    Ok(())
                } else {
                    Err(ArtifactWarmError::Incomplete {
                        missing: outcome.missing,
                    })
                }
            }
            OnArtifactMiss::Embed => self.dense.with_operation_write(|cache| {
                let outcome = cache.warm_from_artifact_locked(
                    bytes,
                    ArtifactEntryKind::Skill,
                    self.skills.values(),
                    self.sink.as_ref(),
                )?;
                if outcome.missing.is_empty() {
                    return Ok(());
                }
                cache
                    .extend_locked(self.skills.values(), self.sink.as_ref())
                    .map_err(ArtifactWarmError::from)
            }),
        }
    }

    /// Serialize the current corpus embeddings into a build-time artifact (bytes only).
    ///
    /// # Errors
    ///
    /// Any [`ArtifactError`] from resolving the embedder or building the artifact
    /// (including [`ArtifactError::Embedder`] when inference fails).
    pub fn build_embedding_artifact(&self) -> Result<Vec<u8>, ArtifactError> {
        self.dense.build_artifact(
            ArtifactEntryKind::Skill,
            self.skills.values(),
            self.sink.as_ref(),
        )
    }

    // ---- engines -----------------------------------------------------------

    fn bm25_search_traced(&self, query: &str, top_k: usize, origin: Origin) -> Vec<SkillHit> {
        self.bm25_search_traced_with_context(query, top_k, origin, TraceEventContext::default())
    }

    fn bm25_search_traced_with_context(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        context: TraceEventContext,
    ) -> Vec<SkillHit> {
        let started = Instant::now();
        let t = Instant::now();
        let arm = self.usage_arm(query, None);
        let usage_ms = t.elapsed().as_millis() as u64;

        let Some(arm) = arm else {
            // No graph, or nothing matched: the original path with raw BM25
            // scores, unchanged.
            // Raw BM25 scores — not fused.
            let hits = to_skill_hits(self.bm25_index().search(query, top_k), false);
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
                context,
            );
            return hits;
        };

        let depth = RETRIEVE_DEPTH.max(top_k);
        let t = Instant::now();
        let bm25_ranked = self.bm25_index().search(query, depth);
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
            context,
        );
        hits
    }

    fn semantic_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        context: TraceEventContext,
    ) -> Result<Vec<SkillHit>, EmbedderError> {
        let started = Instant::now();
        if self.skills.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0, context);
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
                context,
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
            context,
        );
        Ok(hits)
    }

    fn hybrid_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        context: TraceEventContext,
    ) -> Result<Vec<SkillHit>, EmbedderError> {
        let started = Instant::now();
        if self.skills.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0, context);
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
        self.record_search(query, origin, top_k, &hits, stages, took_ms, context);
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
        context: TraceEventContext,
    ) {
        self.sink.record_with_context(
            TraceEvent::SkillSearch {
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
            },
            context,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Embedder;
    use crate::test_support::{
        FailOnEmbedStub, FpCountingEmbedder, PanicOnEmbedStub, build_test_artifact, unit,
    };
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
            experimental_catalog_definitions: false,
            bm25: Bm25Cache::new(),
            dense: DenseCache::with_embedder(embedder),
            graph: None,
        }
    }

    fn skill(id: &str, name: &str, description: &str, tags: &[&str]) -> Skill {
        Skill {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            experimental_searchable_description: None,
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
    fn mutation_after_a_warmed_search_is_visible_in_the_next_search() {
        // Same staleness guard as the tool-side integration test: searches
        // must not pin ranking state across register or replace_all.
        let mut reg = catalog();
        for _ in 0..3 {
            let _ = reg.search("REST API design", 5);
        }

        reg.register(skill(
            "migrations",
            "migrations",
            "Write reversible database migrations",
            &["backend"],
        ));
        assert_eq!(
            reg.search("reversible database migrations", 5)[0].skill_id,
            "migrations",
            "a skill registered after searches must rank immediately"
        );

        // replace_all drops one skill and rewrites another; both must be
        // visible in the next search.
        let _ = reg.search("presentations", 5); // re-warm
        reg.replace_all(vec![
            skill(
                "api-design",
                "api-design",
                "GraphQL schema federation",
                &["backend"],
            ),
            skill(
                "migrations",
                "migrations",
                "Write reversible database migrations",
                &["backend"],
            ),
        ]);
        assert!(
            reg.search("animation-rich HTML presentations", 5)
                .is_empty(),
            "a skill removed by replace_all must stop matching immediately"
        );
        assert_eq!(
            reg.search("GraphQL schema federation", 5)[0].skill_id,
            "api-design",
            "content rewritten by replace_all must match immediately"
        );
    }

    /// A builder that must never run — proof the search path already
    /// populated the cache (the tool-side twin explains why the public seam
    /// can't pin this).
    fn no_build() -> Vec<(String, String)> {
        unreachable!("cache should already be populated by the search path")
    }

    /// Rebuild count when the builder IS expected to run — asserts the cache
    /// was dropped by the preceding mutation.
    fn assert_rebuilds(reg: &SkillRegistry) -> Arc<crate::search::Bm25Index> {
        let builds = std::cell::Cell::new(0);
        let index = reg.bm25.get_or_build(|| {
            builds.set(builds.get() + 1);
            reg.bm25_docs()
        });
        assert_eq!(builds.get(), 1, "mutation must drop the cached index");
        index
    }

    #[test]
    fn bm25_cache_is_warmed_by_search_and_dropped_by_every_mutator() {
        // Lifecycle pin: a public search populates the cache, and both
        // mutators — register and replace_all — drop it. Results are
        // byte-identical either way, so only this test fails if the
        // search→cache wiring regresses to build-per-call.
        let mut reg = catalog();
        let _ = reg.search("REST API design", 5);
        let warmed = reg.bm25.get_or_build(no_build);
        let _ = reg.search("presentations", 5);
        let reused = reg.bm25.get_or_build(no_build);
        assert!(
            Arc::ptr_eq(&warmed, &reused),
            "searches between mutations must reuse one index"
        );

        reg.register(skill("extra", "extra", "an extra skill", &[]));
        let after_register = assert_rebuilds(&reg);
        assert!(!Arc::ptr_eq(&warmed, &after_register));

        reg.replace_all(vec![skill("only", "only", "the only skill", &[])]);
        let after_replace = assert_rebuilds(&reg);
        assert!(!Arc::ptr_eq(&after_register, &after_replace));
    }

    #[test]
    fn an_unchanged_replace_all_keeps_the_cached_bm25_index() {
        // The periodic-source steady state: a reload that changes nothing (or
        // only un-indexed fields like `body`) must not throw the index away —
        // same diff the dense cache keys on.
        let reload = || {
            vec![
                skill(
                    "frontend-slides",
                    "frontend-slides",
                    "Build animation-rich HTML presentations from scratch",
                    &["frontend", "presentations"],
                ),
                skill(
                    "api-design",
                    "api-design",
                    "REST API design patterns: resource naming, status codes, pagination",
                    &["backend", "api"],
                ),
            ]
        };
        let mut reg = catalog();
        let _ = reg.search("REST API design", 5);
        let warmed = reg.bm25.get_or_build(no_build);

        reg.replace_all(reload());
        let after_noop = reg.bm25.get_or_build(no_build);
        assert!(
            Arc::ptr_eq(&warmed, &after_noop),
            "an unchanged reload must keep the cached index"
        );

        let mut body_edit = reload();
        body_edit[0].body = "rewritten body — not part of searchable_text".into();
        reg.replace_all(body_edit);
        let after_body_edit = reg.bm25.get_or_build(no_build);
        assert!(
            Arc::ptr_eq(&warmed, &after_body_edit),
            "a body-only edit is not indexed and must keep the cached index"
        );
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
    fn experimental_searchable_description_replaces_skill_description_but_keeps_name_and_tags() {
        let mut reg = SkillRegistry::new();
        let mut overridden = skill(
            "billing",
            "billing_helper",
            "orchestrate zeppelin manifests",
            &["finance_ops"],
        );
        overridden.experimental_searchable_description = Some("reconcile overdue invoices".into());
        reg.register(overridden);

        assert_eq!(reg.search("overdue invoices", 5)[0].skill_id, "billing");
        assert!(reg.search("zeppelin manifests", 5).is_empty());
        assert_eq!(reg.search("billing", 5)[0].skill_id, "billing");
        assert_eq!(reg.search("finance ops", 5)[0].skill_id, "billing");
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
    fn replace_all_drops_ids_absent_from_the_batch() {
        let mut reg = catalog();
        let outcome = reg.replace_all(vec![skill(
            "api-design",
            "api-design",
            "REST API design patterns: resource naming, status codes, pagination",
            &["backend", "api"],
        )]);
        assert_eq!(reg.len(), 1);
        assert_eq!(outcome.removed, 1);
        assert!(
            reg.search("animation-rich HTML presentations", 5)
                .is_empty(),
            "a dropped skill must leave the corpus, not linger in the index"
        );
    }

    #[test]
    fn replace_all_with_an_empty_batch_clears_the_corpus() {
        let mut reg = catalog();
        reg.replace_all(Vec::new());
        assert!(reg.is_empty());
        assert!(reg.search("anything", 5).is_empty());
    }

    #[test]
    fn replace_all_keeps_the_last_of_duplicate_ids() {
        // Parity with `register`, which replaces an id in place: within one
        // batch the later entry wins and the corpus holds a single entry.
        let mut reg = SkillRegistry::new();
        reg.replace_all(vec![
            skill("s", "s", "REST API design", &["api"]),
            skill("s", "s", "HTML slides frontend", &["frontend"]),
        ]);
        assert_eq!(reg.len(), 1);
        let hits = reg.search("html slides frontend", 5);
        assert_eq!(hits.first().map(|h| h.skill_id.as_str()), Some("s"));
    }

    #[test]
    fn replace_all_reports_what_changed() {
        let mut reg = SkillRegistry::new();
        reg.register(skill("keep", "keep", "REST API design", &["api"]));
        reg.register(skill("edit", "edit", "HTML slides", &["frontend"]));
        reg.register(skill("drop", "drop", "database migrations", &["data"]));

        let outcome = reg.replace_all(vec![
            skill("keep", "keep", "REST API design", &["api"]),
            skill("edit", "edit", "HTML slides and animations", &["frontend"]),
            skill("add", "add", "queue consumers", &["backend"]),
        ]);

        assert_eq!(
            outcome,
            ReplaceOutcome {
                added: 1,
                removed: 1,
                updated: 1,
                unchanged: 1,
            }
        );
    }

    #[test]
    fn replace_all_embeds_only_what_changed() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(skill("a", "a", "REST API design", &["api"]));
        reg.register(skill("b", "b", "HTML slides", &["frontend"]));
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 2);

        // `a` is byte-identical, `b` is dropped, `c` is new.
        reg.replace_all(vec![
            skill("a", "a", "REST API design", &["api"]),
            skill("c", "c", "database migrations", &["data"]),
        ]);
        reg.build_embeddings().unwrap();
        assert_eq!(
            counter.doc_calls(),
            3,
            "an unchanged id keeps its vector; only the new skill is embedded"
        );
    }

    #[test]
    fn replace_all_keeps_the_vector_when_only_the_body_changed() {
        // The body is never embedded (`searchable_text` excludes it), so a
        // body-only edit must not cost an embedding on the next build.
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(skill("a", "a", "REST API design", &["api"]));
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 1);

        let mut rewritten = skill("a", "a", "REST API design", &["api"]);
        rewritten.body = "# a\n\nrewritten instructions".into();
        reg.replace_all(vec![rewritten]);
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 1);
    }

    #[test]
    fn replace_all_re_embeds_a_skill_whose_indexed_text_changed() {
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(skill("s", "s", "REST API design", &["api"])); // dense: api bucket
        reg.build_embeddings().unwrap();

        reg.replace_all(vec![skill("s", "s", "HTML slides frontend", &["frontend"])]); // → frontend bucket
        reg.build_embeddings().unwrap();

        let hits = reg
            .search_with_method("frontend slides", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(hits.first().map(|h| h.skill_id.as_str()), Some("s"));
        assert!(hits[0].score > 0.9, "ranks with the re-embedded vector");
    }

    #[test]
    fn replace_all_re_embeds_only_the_experimental_searchable_description_edit() {
        let counter = Arc::new(CountingEmbedder::new());
        let sink = Arc::new(MemorySink::new("test-session"));
        let mut reg = with_embedder(counter.clone());
        reg.set_trace_sink(sink.clone());
        reg.register(skill("keep", "keep", "REST API design", &["api"]));
        reg.register(skill("edit", "edit", "REST API design", &["api"]));
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 2);
        sink.drain();

        let mut edited = skill("edit", "edit", "REST API design", &["api"]);
        edited.experimental_searchable_description = Some("HTML slides frontend".into());
        let outcome = reg.replace_all(vec![
            skill("keep", "keep", "REST API design", &["api"]),
            edited,
        ]);
        assert_eq!(
            outcome,
            ReplaceOutcome {
                added: 0,
                removed: 0,
                updated: 1,
                unchanged: 1,
            }
        );
        let churn: Vec<(ChurnKind, String)> = sink
            .drain()
            .into_iter()
            .filter_map(|envelope| match envelope.event {
                TraceEvent::SkillChurn { kind, skill_id } => Some((kind, skill_id)),
                _ => None,
            })
            .collect();
        assert_eq!(churn, vec![(ChurnKind::Add, "edit".to_string())]);

        reg.build_embeddings().unwrap();
        assert_eq!(
            counter.doc_calls(),
            3,
            "only the override-edited skill is re-embedded"
        );
    }

    #[test]
    fn replace_all_drops_the_vector_of_a_removed_id() {
        // The dense guard is a count (`vectors.len() < corpus_len`), so leaving a
        // removed id's vector in the cache would let a *new*, unembedded id slip
        // past `require_built` and be silently skipped in ranking. Removal must
        // drop the vector, not just the corpus entry.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(skill(
            "api-design",
            "api-design",
            "REST API design",
            &["api"],
        ));
        reg.register(skill("frontend", "frontend", "HTML slides", &["frontend"]));
        reg.build_embeddings().unwrap();

        reg.replace_all(vec![
            skill("api-design", "api-design", "REST API design", &["api"]),
            skill("db", "db", "database migrations", &["data"]),
        ]);

        assert!(
            matches!(
                reg.search_with_method("database", 5, Origin::Direct, SearchMethod::Semantic),
                Err(EmbedderError::EmbeddingsNotBuilt)
            ),
            "a removed id's stale vector must not mask an unembedded new id"
        );

        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method(
                "database migrations",
                5,
                Origin::Direct,
                SearchMethod::Semantic,
            )
            .unwrap();
        assert_eq!(hits.first().map(|h| h.skill_id.as_str()), Some("db"));
    }

    #[test]
    fn replace_all_emits_churn_only_for_real_changes() {
        let sink = Arc::new(MemorySink::new("test-session"));
        let mut reg = SkillRegistry::with_trace_sink(sink.clone());
        reg.register(skill("keep", "keep", "REST API design", &["api"]));
        reg.register(skill("drop", "drop", "HTML slides", &["frontend"]));
        sink.drain();

        reg.replace_all(vec![
            skill("keep", "keep", "REST API design", &["api"]),
            skill("add", "add", "database migrations", &["data"]),
        ]);

        let churn: Vec<(ChurnKind, String)> = sink
            .drain()
            .into_iter()
            .filter_map(|envelope| match envelope.event {
                TraceEvent::SkillChurn { kind, skill_id } => Some((kind, skill_id)),
                _ => None,
            })
            .collect();
        assert_eq!(churn.len(), 2, "an unchanged id emits nothing: {churn:?}");
        assert!(churn.contains(&(ChurnKind::Remove, "drop".to_string())));
        assert!(churn.contains(&(ChurnKind::Add, "add".to_string())));
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
    fn semantic_uses_experimental_searchable_description_and_keeps_name_and_tags() {
        let mut overridden_reg = with_embedder(Arc::new(StubEmbedder));
        let mut overridden = skill("target", "catalog", "REST API design", &["general"]);
        overridden.experimental_searchable_description = Some("frontend slides".into());
        overridden_reg.register(overridden);
        overridden_reg.register(skill("decoy", "decoy", "REST API design", &["general"]));
        overridden_reg.build_embeddings().unwrap();
        let override_hits = overridden_reg
            .search_with_method("frontend slides", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            override_hits.first().map(|h| h.skill_id.as_str()),
            Some("target")
        );

        let mut name_reg = with_embedder(Arc::new(StubEmbedder));
        let mut named = skill("named", "api_helper", "unrelated", &["general"]);
        named.experimental_searchable_description = Some("frontend slides".into());
        name_reg.register(named);
        name_reg.register(skill("name-decoy", "decoy", "frontend slides", &[]));
        name_reg.build_embeddings().unwrap();
        let name_hits = name_reg
            .search_with_method("REST API", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            name_hits.first().map(|h| h.skill_id.as_str()),
            Some("named")
        );

        let mut tag_reg = with_embedder(Arc::new(StubEmbedder));
        let mut tagged = skill("tagged", "catalog", "unrelated", &["rest_ops"]);
        tagged.experimental_searchable_description = Some("frontend slides".into());
        tag_reg.register(tagged);
        tag_reg.register(skill("tag-decoy", "decoy", "frontend slides", &[]));
        tag_reg.build_embeddings().unwrap();
        let tag_hits = tag_reg
            .search_with_method("REST API", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            tag_hits.first().map(|h| h.skill_id.as_str()),
            Some("tagged")
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

    #[test]
    fn warm_embeddings_error_ok_when_artifact_covers_corpus() {
        let a = skill("api-design", "api-design", "rest api design", &[]);
        let b = skill("frontend", "frontend", "frontend slides", &[]);
        let bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            [&a, &b],
            "fp-warm",
            vec![unit([1.0, 0.0, 0.0]), unit([0.0, 1.0, 0.0])],
        );
        let counter = Arc::new(FpCountingEmbedder::new("fp-warm", StubEmbedder::vec_for));
        let mut reg = with_embedder(counter.clone());
        reg.register(a);
        reg.register(b);
        reg.warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Error)
            .unwrap();
        assert_eq!(counter.docs(), 0);
        assert!(
            reg.search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
                .is_ok()
        );
    }

    #[test]
    fn warm_embeddings_error_fails_when_ids_missing() {
        let a = skill("api-design", "api-design", "rest api design", &[]);
        let b = skill("frontend", "frontend", "frontend slides", &[]);
        let bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            [&a],
            "fp-warm",
            vec![unit([1.0, 0.0, 0.0])],
        );
        let counter = Arc::new(FpCountingEmbedder::new("fp-warm", StubEmbedder::vec_for));
        let mut reg = with_embedder(counter.clone());
        reg.register(a);
        reg.register(b);
        assert!(matches!(
            reg.warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Error),
            Err(ArtifactWarmError::Incomplete { missing }) if missing == ["frontend"]
        ));
        assert_eq!(counter.docs(), 0);
        assert!(matches!(
            reg.search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic),
            Err(EmbedderError::EmbeddingsNotBuilt)
        ));
    }

    #[test]
    fn warm_embeddings_embed_completes_only_missing_ids() {
        let a = skill("api-design", "api-design", "rest api design", &[]);
        let b = skill("frontend", "frontend", "frontend slides", &[]);
        let bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            [&a],
            "fp-warm",
            vec![unit([1.0, 0.0, 0.0])],
        );
        let counter = Arc::new(FpCountingEmbedder::new("fp-warm", StubEmbedder::vec_for));
        let mut reg = with_embedder(counter.clone());
        reg.register(a);
        reg.register(b);
        reg.warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Embed)
            .unwrap();
        assert_eq!(counter.docs(), 1, "only the missing skill is embedded");
        assert!(
            reg.search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
                .is_ok()
        );
    }

    #[test]
    fn warm_embeddings_embed_policy_propagates_build_embeddings_failure() {
        let a = skill("api-design", "api-design", "rest api design", &[]);
        let b = skill("frontend", "frontend", "frontend slides", &[]);
        let bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            [&a],
            "fp-warm",
            vec![unit([1.0, 0.0, 0.0])],
        );
        let mut reg = with_embedder(Arc::new(FailOnEmbedStub::new("fp-warm")));
        reg.register(a);
        reg.register(b);
        assert!(matches!(
            reg.warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Embed),
            Err(ArtifactWarmError::Embedder(EmbedderError::Inference { .. }))
        ));
    }

    #[test]
    fn warm_embeddings_propagates_warm_error_without_embed() {
        let a = skill("api-design", "api-design", "rest api design", &[]);
        let bytes = build_test_artifact(
            ArtifactEntryKind::Skill,
            [&a],
            "fp-artifact",
            vec![unit([1.0, 0.0, 0.0])],
        );
        let counter = Arc::new(FpCountingEmbedder::new("fp-active", StubEmbedder::vec_for));
        let mut reg = with_embedder(counter.clone());
        reg.register(a);
        assert!(matches!(
            reg.warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Embed),
            Err(ArtifactWarmError::Warm(
                crate::WarmError::ArtifactModelMismatch { .. }
            ))
        ));
        assert_eq!(counter.docs(), 0);
    }

    #[test]
    fn build_embedding_artifact_round_trips_via_warm() {
        let a = skill("api-design", "api-design", "rest api design", &[]);
        let b = skill("frontend", "frontend", "frontend slides", &[]);
        let builder = Arc::new(FpCountingEmbedder::new("fp-warm", StubEmbedder::vec_for));
        let mut reg_a = with_embedder(builder.clone());
        reg_a.register(skill("api-design", "api-design", "rest api design", &[]));
        reg_a.register(skill("frontend", "frontend", "frontend slides", &[]));
        let bytes = reg_a.build_embedding_artifact().unwrap();
        assert_eq!(
            builder.docs(),
            2,
            "build embeds each corpus document exactly once"
        );

        let warmer = Arc::new(FpCountingEmbedder::new("fp-warm", StubEmbedder::vec_for));
        let mut reg_b = with_embedder(warmer.clone());
        reg_b.register(a);
        reg_b.register(b);
        reg_b
            .warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Error)
            .unwrap();
        assert_eq!(warmer.docs(), 0, "warm must not re-embed covered ids");
        assert!(
            reg_b
                .search_with_method("api", 5, Origin::Direct, SearchMethod::Semantic)
                .is_ok()
        );
    }

    #[test]
    fn build_embedding_artifact_propagates_embedder_failure() {
        let mut reg = with_embedder(Arc::new(FailOnEmbedStub::new("fp-warm")));
        reg.register(skill("api-design", "api-design", "rest api design", &[]));
        assert!(matches!(
            reg.build_embedding_artifact(),
            Err(ArtifactError::Embedder(EmbedderError::Inference { .. }))
        ));
    }

    #[test]
    fn build_embedding_artifact_empty_corpus_is_valid() {
        let reg = with_embedder(Arc::new(PanicOnEmbedStub::new("unused")));
        let bytes = reg.build_embedding_artifact().unwrap();
        reg.warm_embeddings_from_artifact(&bytes, OnArtifactMiss::Error)
            .unwrap();
    }
}
