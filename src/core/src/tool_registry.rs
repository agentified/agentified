use std::sync::{Arc, PoisonError, RwLock};
use std::time::Instant;

use indexmap::IndexMap;

use crate::dense_cache::{DenseCache, Embeddable};
use crate::embedding::EmbedderError;
use crate::embedding_config::EmbeddingModel;
use crate::fusion::{RETRIEVE_DEPTH, RRF_K, WeightedArm, rrf_fuse_weighted};
use crate::indexing::searchable_text;
use crate::method::SearchMethod;
use crate::search::bm25_search;
use crate::tool::Tool;
use crate::trace::{
    ChurnKind, NoopSink, Origin, SearchHitTrace, SearchStage, TraceEvent, TraceSink,
};
use crate::usage::{Capability, IntentGraph, UsageArm};

/// Whether adaptive usage ranking is currently contributing to a registry's
/// results — the SDK-facing view of the model-compatibility check (ADR-0014).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveRankingStatus {
    /// No intent graph attached.
    Inactive,
    /// Attached and usable (matching model, or a lexical graph with no centroids).
    Active,
    /// Attached, but embeddings have not been built yet, so the active model's
    /// identity — and thus compatibility — is not yet known.
    Unknown,
    /// Attached but paused: the graph's centroids were built with a different
    /// embedding model, so the usage arm is off until `rebuild_intent_graph`.
    Paused {
        /// `true` when the models differ in output dimension, `false` for a
        /// same-width model change.
        dim_mismatch: bool,
        /// The graph's model (fingerprint, or centroid width when dimensional).
        built: String,
        /// The active model, in the same units.
        active: String,
    },
}

/// One ranked match from a [`ToolRegistry`] search, best-first in the
/// returned `Vec`.
pub struct SearchHit {
    /// Id of the matching tool ([`Tool::id`]).
    pub tool_id: String,
    /// Relevance score — higher is better, and the scale depends on the
    /// [`SearchMethod`] that produced the hit:
    ///
    /// - `Bm25`: raw BM25 relevance (non-negative, unbounded; `k1=0.9`,
    ///   `b=0.4`, tuned for short tool text per ADR-0004).
    /// - `Semantic`: cosine similarity of the L2-normalized query and
    ///   document embeddings (at most `1.0`).
    /// - `Hybrid`: the Reciprocal Rank Fusion sum `Σ 1/(60 + rank)` over the
    ///   BM25 and dense rankings — with two arms at most `2/60 ≈ 0.033`, so
    ///   only the ordering is meaningful, not the magnitude.
    ///
    /// Scores are comparable within one result list, not across methods or
    /// corpora. Ties are broken by `tool_id` ascending, so ordering is
    /// deterministic across processes.
    ///
    /// **The scale also depends on [`fused`](Self::fused).** With adaptive
    /// ranking on, a matched query returns RRF scores (~0.03) while an unmatched
    /// one on the same catalog returns the raw method score (~0.9) — so `score`
    /// switches scale *between calls*. Order by [`rank`](Self::rank) and branch
    /// on [`fused`](Self::fused); treat `score` as a within-list hint only.
    pub score: f32,
    /// 0-based position in this result list (best is `0`). Scale-invariant and
    /// stable across methods and across the [`fused`](Self::fused) switch — the
    /// signal to order or threshold on, in place of the wobbling [`score`](Self::score).
    pub rank: u32,
    /// Whether [`score`](Self::score) is a Reciprocal Rank Fusion score (only its
    /// *ordering* is meaningful) rather than the raw method score. Uniform across
    /// all hits of one search: `true` when the usage arm fused into this search
    /// (or the method is hybrid, always RRF), `false` for a plain BM25/semantic
    /// result. Lets a caller detect the scale their `score` is on.
    pub fused: bool,
}

/// Build hits from an already-ranked, best-first `(id, score)` list: `rank` is
/// the position, `fused` says whether these are RRF scores. One place so the two
/// new fields cannot drift between the fused and raw paths.
fn to_search_hits(ranked: Vec<(String, f32)>, fused: bool) -> Vec<SearchHit> {
    ranked
        .into_iter()
        .enumerate()
        .map(|(i, (tool_id, score))| SearchHit {
            tool_id,
            score,
            rank: i as u32,
            fused,
        })
        .collect()
}

impl Embeddable for Tool {
    fn embed_id(&self) -> &str {
        &self.id
    }
    fn embed_text(&self) -> String {
        searchable_text(self)
    }
}

/// Retrieval index over [`Tool`]s — the registry behind the SDKs' tool
/// catalogs.
///
/// Tools are [`Self::register`]ed into an id-keyed corpus (re-registering an
/// id replaces it in place) and ranked by one of three engines selected per
/// call via [`SearchMethod`]. The plain [`Self::search`] path is lexical BM25:
/// infallible, model-free, ready as soon as tools are registered. Semantic
/// and hybrid go through [`Self::search_with_method`] and require
/// [`Self::build_embeddings`] first — a search never embeds the corpus (see
/// ADR-0011).
///
/// Every register and search emits a [`TraceEvent`] on the registry's
/// [`TraceSink`] ([`NoopSink`] by default). [`SkillRegistry`] is the
/// skill-side twin with the same shape.
///
/// [`SkillRegistry`]: crate::SkillRegistry
pub struct ToolRegistry {
    /// Corpus keyed by tool id, in insertion order. Keying by id makes `register`
    /// replace an existing id in place (never a duplicate), so the BM25 corpus
    /// stays one-entry-per-id — no `avgdl` drift, no leak (RAT-378).
    tools: IndexMap<String, Tool>,
    sink: Arc<dyn TraceSink>,
    /// Dense embeddings for `tools`, keyed by id and built on demand. `register`
    /// invalidates a replaced id; the missing ids are embedded by
    /// [`Self::build_embeddings`] — a search never embeds the corpus (it requires
    /// the cache built first). A pure BM25 user never populates it and never loads
    /// the model (see ADR-0011 and [`DenseCache`]).
    dense: DenseCache,
    /// Optional usage-ranking read model (ADR-0014). `None` — the default — is
    /// today's behavior exactly: no extra arm, no `UsageBoost` event, BM25
    /// scores unchanged. Shared behind a lock because the learner writes to the
    /// same graph the search path reads.
    graph: Option<Arc<RwLock<IntentGraph>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    /// An empty registry with tracing off ([`NoopSink`]); attach a real sink
    /// later with [`Self::set_trace_sink`].
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
            sink: Arc::new(NoopSink),
            dense: DenseCache::new(),
            graph: None,
        }
    }

    /// An empty registry recording trace events to `sink` from the start.
    pub fn with_trace_sink(sink: Arc<dyn TraceSink>) -> Self {
        Self {
            tools: IndexMap::new(),
            sink,
            dense: DenseCache::new(),
            graph: None,
        }
    }

    /// A registry whose semantic/hybrid engines use an explicit embedding model
    /// (the configurable-model path). BM25 is unaffected — it needs no model.
    /// Direct enum variants are validated on the first embedding build; call
    /// [`EmbeddingModel::validate`] first when construction-time feedback is
    /// required. The trace sink is set separately via [`Self::set_trace_sink`].
    pub fn with_embedding(model: EmbeddingModel) -> Self {
        Self {
            tools: IndexMap::new(),
            sink: Arc::new(NoopSink),
            dense: DenseCache::with_model(model),
            graph: None,
        }
    }

    /// Replace the trace sink; subsequent register/search/`record_event`
    /// events go to `sink`. Already-recorded events are not replayed.
    pub fn set_trace_sink(&mut self, sink: Arc<dyn TraceSink>) {
        self.sink = sink;
    }

    /// Record an arbitrary [`TraceEvent`] on the registry's sink. Higher
    /// layers (the SDK catalogs and capability tools) use this to emit their
    /// invoke/upstream/auth lifecycle events into the same stream as the
    /// registry's own search and churn events (ADR-0007).
    pub fn record_event(&self, event: TraceEvent) {
        self.sink.record(event);
    }

    /// Attach (or with `None`, detach) the usage-ranking read model — adaptive
    /// ranking is **opt-in** (ADR-0014).
    ///
    /// With a graph attached, a query that matches one of its clusters gains a
    /// third fusion arm ranked by what users invoked after similar queries. That
    /// changes [`SearchHit::score`] from a BM25 score to an RRF score on the
    /// `Bm25` path, which is exactly why it is opt-in: ADR-0011 promises
    /// [`Self::search`] keeps BM25 behavior byte-for-byte, and it does for every
    /// caller that never calls this.
    ///
    /// A query that matches nothing is unaffected — same order, same scores —
    /// so attaching a graph is only ever felt where it has evidence.
    ///
    /// The graph is shared behind a lock so a learner may keep updating it while
    /// searches read it.
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
    /// replace its centroids, restamping the graph's model fingerprint.
    ///
    /// Call this after changing the embedding model: a graph's centroids are only
    /// comparable to queries embedded by the same model, so on a swap the usage
    /// arm pauses (a [`TraceEvent::UsageModelMismatch`] is emitted) until this
    /// runs. Members, support, and edges are model-independent and preserved —
    /// only the centroids move to the new space.
    ///
    /// No-op when no graph is attached, or when the graph is lexical (no
    /// centroids). Loads the model, so callers run it off the hot path (the SDK
    /// exposes it as an async method).
    ///
    /// # Errors
    ///
    /// Any [`EmbedderError`] from embedding the members under the current model.
    pub fn rebuild_intent_graph(&self) -> Result<(), EmbedderError> {
        let Some(graph) = self.graph.as_ref() else {
            return Ok(());
        };
        // Snapshot members under the read lock, then embed WITHOUT holding it —
        // embedding can be slow and must not block concurrent searches.
        //
        // A poisoned lock is recovered rather than a panic: rebuild is the repair
        // path and overwrites every centroid wholesale, so it has no reason to
        // refuse a graph whose state an earlier panic left in doubt — and
        // `members` are model-independent text, untouched by that overwrite.
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

    /// Resolve the usage arm for one query, and record the outcome.
    ///
    /// `query_vec` carries the already-computed query embedding on the
    /// semantic/hybrid paths, so dense matching costs no extra inference; `None`
    /// selects the lexical tier, which loads no model (ADR-0011's model-free
    /// `Bm25` guarantee).
    ///
    /// Emits [`TraceEvent::UsageBoost`] on hit *and* miss — but only when a
    /// graph is attached, so a registry without one is silent and behaves
    /// exactly as before.
    fn usage_arm(&self, query: &str, query_vec: Option<&[f32]>) -> Option<UsageArm> {
        let graph = self.graph.as_ref()?;
        // The model that embedded this query (semantic/hybrid only). Compared
        // against the graph's model so a swap pauses the arm instead of cosine-ing
        // across incompatible vector spaces.
        let fingerprint = self.dense.built_fingerprint();
        // A poisoned lock must not take down a search: usage ranking is an
        // enhancement, and losing it degrades to today's behavior.
        let (arm, mismatch) = {
            let guard = graph.read().ok()?;
            let mismatch = match (query_vec, &fingerprint) {
                (Some(v), Some(fp)) => guard.model_status(fp, v.len()).describe(),
                _ => None,
            };
            if mismatch.is_some() {
                // Paused: the graph's centroids belong to another model. Base
                // ranking is untouched; the query is NOT noted, so the learner
                // does not fold a foreign-model vector.
                (None, mismatch)
            } else {
                // Hand the embedded query to the learner: it only sees trace
                // events, which carry text and not vectors, so this is how a
                // locally-grown cluster gets a real centroid.
                if let (Some(v), Some(fp)) = (query_vec, &fingerprint) {
                    guard.note_query_vector(query, v, fp);
                }
                let known = |id: &str| self.tools.contains_key(id);
                // The graph picks the match tier from what it carries; a lexically
                // grown graph has no centroids to compare a query vector against.
                (guard.arm(query, query_vec, Capability::Tool, &known), None)
            }
        };
        // The read guard is released BEFORE the sink runs. A `UsageLearner` sink
        // takes the write lock on this same graph, and `RwLock` is not
        // reentrant — recording while still holding the read guard would
        // deadlock the search path.
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

    /// Register a tool, or replace one in place if its id is already present.
    /// Replacing invalidates the old id's cached embedding so the next
    /// `build_embeddings` re-embeds the new content; the corpus never holds a
    /// duplicate. Registration stays infallible and model-free (a search never
    /// embeds), so BM25 users are unaffected (see ADR-0011).
    ///
    /// # Examples
    ///
    /// ```
    /// use ratel_ai_core::{Tool, ToolRegistry};
    ///
    /// let tool = |desc: &str| Tool {
    ///     id: "read_file".into(),
    ///     name: "read_file".into(),
    ///     description: desc.into(),
    ///     input_schema: serde_json::json!({}),
    ///     output_schema: serde_json::json!({}),
    /// };
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(tool("Read a file"));
    /// registry.register(tool("Read a file from disk")); // same id: replaced
    /// assert_eq!(registry.len(), 1);
    /// ```
    pub fn register(&mut self, tool: Tool) {
        let tool_id = tool.id.clone();
        if self.tools.insert(tool_id.clone(), tool).is_some() {
            // Replaced an existing id: drop its stale embedding.
            self.dense.invalidate(&tool_id);
        }
        self.sink.record(TraceEvent::IndexChurn {
            kind: ChurnKind::Add,
            tool_id,
        });
    }

    /// Number of registered tools (distinct ids).
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Lexical BM25 retrieval. The default engine — needs no model and never
    /// fails, so the public `search`/`search_with_origin` stay infallible and
    /// byte-for-byte compatible with the BM25-only releases.
    ///
    /// Returns at most `top_k` hits, best-first (see [`SearchHit::score`] for
    /// the score semantics). Traced as [`Origin::Direct`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ratel_ai_core::{Tool, ToolRegistry};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Tool {
    ///     id: "read_file".into(),
    ///     name: "read_file".into(),
    ///     description: "Read a file from disk".into(),
    ///     input_schema: serde_json::json!({}),
    ///     output_schema: serde_json::json!({}),
    /// });
    ///
    /// let hits = registry.search("read a file", 5);
    /// assert_eq!(hits[0].tool_id, "read_file");
    /// ```
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SearchHit> {
        self.search_with_origin(query, top_k, Origin::Direct)
    }

    /// [`Self::search`] with an explicit trace [`Origin`], so consumers of the
    /// trace stream can tell agent-synthesized searches from direct library
    /// calls. Same BM25 engine, same infallibility.
    pub fn search_with_origin(&self, query: &str, top_k: usize, origin: Origin) -> Vec<SearchHit> {
        self.bm25_search_traced(query, top_k, origin)
    }

    /// Retrieve with an explicit [`SearchMethod`]. `Bm25` is infallible; `Semantic`
    /// and `Hybrid` rank against the prebuilt embedding cache and return an
    /// [`EmbedderError`] (`EmbeddingsNotBuilt`) if it isn't built — they never load
    /// the model or embed the corpus in-search (the model loads at
    /// `build_embeddings`). The SDK layer picks the method (a per-catalog default or
    /// a per-call override) and calls this.
    ///
    /// # Errors
    ///
    /// Never errors for [`SearchMethod::Bm25`]. For `Semantic` and `Hybrid`:
    /// [`EmbedderError::EmbeddingsNotBuilt`] when the cache does not cover the
    /// corpus (call [`Self::build_embeddings`] after registering), or any
    /// other [`EmbedderError`] from loading the model / embedding the query.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ratel_ai_core::{Origin, SearchMethod, Tool, ToolRegistry};
    /// # fn main() -> Result<(), ratel_ai_core::EmbedderError> {
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Tool {
    ///     id: "delete_file".into(),
    ///     name: "delete_file".into(),
    ///     description: "Delete a path from the filesystem".into(),
    ///     input_schema: serde_json::json!({}),
    ///     output_schema: serde_json::json!({}),
    /// });
    /// registry.build_embeddings()?; // loads the model on first use
    ///
    /// // Lexically unrelated query, semantically close:
    /// let hits = registry.search_with_method(
    ///     "remove a document",
    ///     5,
    ///     Origin::Direct,
    ///     SearchMethod::Semantic,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn search_with_method(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
        method: SearchMethod,
    ) -> Result<Vec<SearchHit>, EmbedderError> {
        match method {
            SearchMethod::Bm25 => Ok(self.bm25_search_traced(query, top_k, origin)),
            SearchMethod::Semantic => self.semantic_search_traced(query, top_k, origin),
            SearchMethod::Hybrid => self.hybrid_search_traced(query, top_k, origin),
        }
    }

    /// Pre-compute embeddings for any not-yet-embedded tools so a later
    /// semantic/hybrid search only has to embed the query (never the corpus).
    /// Incremental — embeds only tools registered since the last call. Callers
    /// invoke this explicitly before semantic/hybrid search; a BM25-only user
    /// never calls it and never loads the model.
    ///
    /// # Errors
    ///
    /// Any [`EmbedderError`] from resolving or using the embedding source,
    /// including `Config`, `NotCached`, `Download`, `CacheUnwritable`, `Load`,
    /// or `Inference`; retained caches can also surface `DimensionMismatch` or
    /// `ModelMismatch`. A failed build leaves the prior cache unchanged and a
    /// later call can retry once the cause clears.
    pub fn build_embeddings(&self) -> Result<(), EmbedderError> {
        self.dense.extend(self.tools.values(), self.sink.as_ref())
    }

    /// Recompute embeddings for the full tool corpus and atomically replace the
    /// dense cache. Unlike [`Self::build_embeddings`], this adopts a changed model
    /// identity or dimension. A failed rebuild preserves the prior cache.
    ///
    /// # Errors
    ///
    /// Any [`EmbedderError`] from loading or embedding the complete corpus.
    pub fn rebuild_embeddings(&self) -> Result<(), EmbedderError> {
        self.dense.rebuild(self.tools.values(), self.sink.as_ref())
    }

    // ---- engines -----------------------------------------------------------

    /// The corpus as `(id, searchable_text)` pairs for BM25.
    fn bm25_docs(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.tools
            .values()
            .map(|t| (t.id.clone(), searchable_text(t)))
    }

    /// Fuse the ranked arms into the final top-`top_k`, returning the hits and
    /// the `rrf` stage. Shared by all three engines so the fusion, truncation,
    /// and `(score desc, id asc)` ordering have exactly one implementation.
    fn fuse_arms(arms: &[WeightedArm<'_>], top_k: usize) -> (Vec<SearchHit>, SearchStage) {
        let t = Instant::now();
        let mut fused = rrf_fuse_weighted(arms, RRF_K);
        fused.truncate(top_k);
        let stage = SearchStage {
            name: "rrf".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: fused.first().map(|(_, s)| *s as f64),
        };
        // These came out of RRF, so their magnitude is ordering-only.
        let hits = to_search_hits(fused, true);
        (hits, stage)
    }

    /// The `usage` stage descriptor for a matched arm. `top_score` carries the
    /// arm's fusion weight — the only scalar the stage has, since the arm
    /// contributes rank positions rather than scores.
    fn usage_stage(arm: &UsageArm, took_ms: u64) -> SearchStage {
        SearchStage {
            name: "usage".into(),
            took_ms,
            top_score: Some(arm.weight() as f64),
        }
    }

    fn bm25_search_traced(&self, query: &str, top_k: usize, origin: Origin) -> Vec<SearchHit> {
        let started = Instant::now();
        let t = Instant::now();
        let arm = self.usage_arm(query, None);
        let usage_ms = t.elapsed().as_millis() as u64;

        let Some(arm) = arm else {
            // No graph, or nothing matched: the original path, unchanged, with
            // raw BM25 scores. ADR-0011's byte-for-byte promise lives here.
            // Raw BM25 scores — not fused.
            let hits = to_search_hits(bm25_search(self.bm25_docs(), query, top_k), false);
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

        // Matched: retrieve deeper than `top_k` so a tool the usage arm favors
        // still has BM25 rank signal to fuse, then fuse. Scores become RRF
        // scores — the opt-in cost documented on `set_intent_graph`.
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
    ) -> Result<Vec<SearchHit>, EmbedderError> {
        let started = Instant::now();
        if self.tools.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0);
            return Ok(Vec::new());
        }
        // Retrieve deeper than `top_k` only when a graph is attached; without
        // one the depth, scores, and stages stay exactly as they were.
        let depth = if self.graph.is_some() {
            RETRIEVE_DEPTH.max(top_k)
        } else {
            top_k
        };
        let t = Instant::now();
        let (ranked, query_vec) = self.dense.search_returning_query_vec(
            self.tools.values(),
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
            let mut hits = to_search_hits(ranked, false);
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

    /// Hybrid retrieval (ADR-0011): BM25 and dense each rank the corpus deeper
    /// than `top_k`, then Reciprocal Rank Fusion combines the two rankings into
    /// the final order (no reranker). Emits `bm25`, `dense`, and `rrf` stages.
    fn hybrid_search_traced(
        &self,
        query: &str,
        top_k: usize,
        origin: Origin,
    ) -> Result<Vec<SearchHit>, EmbedderError> {
        let started = Instant::now();
        if self.tools.is_empty() || top_k == 0 {
            self.record_search(query, origin, top_k, &[], Vec::new(), 0);
            return Ok(Vec::new());
        }
        // Retrieve deeper than `top_k` so a tool ranked low by one arm but high
        // by the other still has rank signal to fuse.
        let depth = RETRIEVE_DEPTH.max(top_k);

        // 1. BM25 (lexical).
        let t = Instant::now();
        let bm25_ranked = bm25_search(
            self.tools
                .values()
                .map(|t| (t.id.clone(), searchable_text(t))),
            query,
            depth,
        );
        let bm25_stage = SearchStage {
            name: "bm25".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: bm25_ranked.first().map(|(_, s)| *s as f64),
        };

        // 2. Dense (semantic) — requires embeddings to be built; never embeds in-search.
        let t = Instant::now();
        let (dense_ranked, query_vec) = self.dense.search_returning_query_vec(
            self.tools.values(),
            query,
            depth,
            self.sink.as_ref(),
        )?;
        let dense_stage = SearchStage {
            name: "dense".into(),
            took_ms: t.elapsed().as_millis() as u64,
            top_score: dense_ranked.first().map(|(_, s)| *s as f64),
        };

        // 3. Usage (ADR-0014), matched on the vector the dense arm already
        //    embedded. Absent unless a graph is attached and the query matches.
        let t = Instant::now();
        let arm = self.usage_arm(query, Some(&query_vec));
        let usage_ms = t.elapsed().as_millis() as u64;

        // 4. RRF fusion → final top_k.
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
        hits: &[SearchHit],
        stages: Vec<SearchStage>,
        took_ms: u64,
    ) {
        self.sink.record(TraceEvent::Search {
            query: query.to_string(),
            origin,
            top_k: top_k as u32,
            hits: hits
                .iter()
                .map(|h| SearchHitTrace {
                    tool_id: h.tool_id.clone(),
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Barrier, mpsc};
    use std::time::Duration;

    use super::*;
    use crate::embedding::{Embedded, Embedder};
    use crate::trace::MemorySink;
    use crate::usage::Capability;

    /// Deterministic, network-free embedder: a 3-d one-hot keyed on a keyword so
    /// dense ranking is predictable ("read" docs/queries collide, etc.).
    struct StubEmbedder;
    impl StubEmbedder {
        fn vec_for(text: &str) -> Vec<f32> {
            let t = text.to_lowercase();
            if t.contains("read") {
                vec![1.0, 0.0, 0.0]
            } else if t.contains("delete") || t.contains("remove") {
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

    /// Stands in for a machine that can't run the model: every embed fails.
    struct FailingEmbedder;
    impl Embedder for FailingEmbedder {
        fn embed_doc(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
            Err(EmbedderError::Inference {
                source: "stub failure".into(),
            })
        }
        fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
            Err(EmbedderError::Inference {
                source: "stub failure".into(),
            })
        }
    }

    /// Counts `embed_doc` calls so tests can prove the cache is incremental
    /// (registering a tool re-embeds only that tool, not the whole corpus).
    struct CountingEmbedder {
        doc_calls: std::sync::atomic::AtomicUsize,
        query_calls: std::sync::atomic::AtomicUsize,
    }
    impl CountingEmbedder {
        fn new() -> Self {
            Self {
                doc_calls: std::sync::atomic::AtomicUsize::new(0),
                query_calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        fn doc_calls(&self) -> usize {
            self.doc_calls.load(std::sync::atomic::Ordering::SeqCst)
        }
        fn query_calls(&self) -> usize {
            self.query_calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }
    impl Embedder for CountingEmbedder {
        fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            self.doc_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(StubEmbedder::vec_for(text))
        }
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            self.query_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(StubEmbedder::vec_for(text))
        }
    }

    fn with_embedder(embedder: Arc<dyn Embedder>) -> ToolRegistry {
        ToolRegistry {
            tools: IndexMap::new(),
            sink: Arc::new(NoopSink),
            dense: DenseCache::with_embedder(embedder),
            graph: None,
        }
    }

    fn tool(id: &str, description: &str) -> Tool {
        Tool {
            id: id.into(),
            name: id.into(),
            description: description.into(),
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
        }
    }

    fn catalog(embedder: Arc<dyn Embedder>) -> ToolRegistry {
        let mut reg = with_embedder(embedder);
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a file"));
        reg
    }

    #[test]
    fn default_search_is_bm25_and_infallible() {
        let mut reg = ToolRegistry::new();
        reg.register(tool("read_file", "read the contents of a file"));
        reg.register(tool("delete_file", "delete a file"));
        // No embedder override, no model load — pure lexical.
        let hits = reg.search("read a file", 5);
        assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("read_file"));
    }

    #[test]
    fn bm25_never_loads_the_model() {
        // A registry with a failing embedder must still serve BM25 without ever
        // touching the embedder (lazy) — proves the default path is ML-free.
        let reg = catalog(Arc::new(FailingEmbedder));
        let hits = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Bm25)
            .expect("bm25 is infallible");
        assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("read_file"));
    }

    #[test]
    fn semantic_ranks_via_injected_embedder() {
        let reg = catalog(Arc::new(StubEmbedder));
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method("read something", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("read_file"));
    }

    #[test]
    fn semantic_without_embeddings_errors() {
        // Registered but embeddings never built: a semantic search must refuse with a clear
        // error rather than silently embedding the corpus in the search path.
        let reg = catalog(Arc::new(StubEmbedder));
        assert!(matches!(
            reg.search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic),
            Err(EmbedderError::EmbeddingsNotBuilt)
        ));
    }

    #[test]
    fn build_embeddings_surfaces_embedder_error_instead_of_panicking() {
        // The failing embedder's error surfaces at build_embeddings() (where embedding
        // happens) as a catchable error.
        let reg = catalog(Arc::new(FailingEmbedder));
        assert!(matches!(
            reg.build_embeddings(),
            Err(EmbedderError::Inference { .. })
        ));
    }

    #[test]
    fn hybrid_fuses_bm25_and_dense() {
        let reg = catalog(Arc::new(StubEmbedder));
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method("read a file", 5, Origin::Direct, SearchMethod::Hybrid)
            .unwrap();
        // Both arms rank read_file first (lexical "read a file" + dense "read").
        assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("read_file"));
    }

    #[test]
    fn hybrid_recalls_a_tool_bm25_alone_misses() {
        // The two arms disagree: `records_mgr` matches the query lexically, while
        // `deleter` matches only in the dense bucket (zero lexical overlap with the
        // query). Hybrid must fuse both arms and surface `deleter` — the semantic
        // recall a pure BM25 search never returns.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(tool("records_mgr", "manage old records archive")); // dense: other
        reg.register(tool("deleter", "delete entries")); // dense: delete bucket
        reg.build_embeddings().unwrap();
        let q = "remove old records"; // lexical -> records_mgr; dense bucket -> deleter

        let bm25 = reg
            .search_with_method(q, 5, Origin::Direct, SearchMethod::Bm25)
            .unwrap();
        let semantic = reg
            .search_with_method(q, 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        let hybrid = reg
            .search_with_method(q, 5, Origin::Direct, SearchMethod::Hybrid)
            .unwrap();

        // The arms genuinely disagree on the top hit...
        assert_eq!(
            bm25.first().map(|h| h.tool_id.as_str()),
            Some("records_mgr")
        );
        assert_eq!(
            semantic.first().map(|h| h.tool_id.as_str()),
            Some("deleter")
        );
        // ...and pure BM25 never surfaces the lexically-invisible tool.
        assert!(!bm25.iter().any(|h| h.tool_id == "deleter"));
        // Hybrid fuses both arms, so it recalls both.
        let ids: Vec<&str> = hybrid.iter().map(|h| h.tool_id.as_str()).collect();
        assert!(
            ids.contains(&"records_mgr") && ids.contains(&"deleter"),
            "hybrid should fuse both arms, got {ids:?}"
        );
    }

    #[test]
    fn semantic_stage_is_named_dense() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = catalog(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.build_embeddings().unwrap();
        reg.search_with_method("read", 5, Origin::Agent, SearchMethod::Semantic)
            .unwrap();
        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::Search { stages, .. } if stages.iter().any(|s| s.name == "dense")
        )));
    }

    #[test]
    fn hybrid_emits_three_stages() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = catalog(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.build_embeddings().unwrap();
        reg.search_with_method("read", 5, Origin::Agent, SearchMethod::Hybrid)
            .unwrap();
        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::Search { stages, .. }
                if stages.iter().any(|s| s.name == "bm25")
                && stages.iter().any(|s| s.name == "dense")
                && stages.iter().any(|s| s.name == "rrf")
        )));
    }

    #[test]
    fn build_embeddings_after_register_embeds_only_the_new_tool() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a file"));
        // build_embeddings embeds the 2-tool corpus.
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 2);
        // Registering one more then building embeddings must embed ONLY it — the two existing
        // vectors are reused, never recomputed (the O(N) regression).
        reg.register(tool("reader_v2", "read a file too"));
        reg.build_embeddings().unwrap();
        assert_eq!(
            counter.doc_calls(),
            3,
            "only the newly-registered tool should be embedded"
        );
        let hits = reg
            .search_with_method("read", 10, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert!(hits.iter().any(|h| h.tool_id == "reader_v2"));
    }

    #[test]
    fn build_embeddings_precomputes_so_search_embeds_no_docs() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a file"));
        reg.build_embeddings().unwrap();
        assert_eq!(
            counter.doc_calls(),
            2,
            "build_embeddings embeds the corpus up front"
        );
        reg.search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            counter.doc_calls(),
            2,
            "a search after build_embeddings embeds only the query, no documents"
        );
    }

    #[test]
    fn build_embeddings_is_idempotent() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();
        reg.build_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 1);
    }

    #[test]
    fn rebuild_embeddings_recomputes_the_full_corpus() {
        let counter = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counter.clone());
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a file"));
        reg.build_embeddings().unwrap();
        reg.rebuild_embeddings().unwrap();
        assert_eq!(counter.doc_calls(), 4, "rebuild embeds every tool again");
    }

    struct FailSecondBatch {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Embedder for FailSecondBatch {
        fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(StubEmbedder::vec_for(text))
        }

        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(StubEmbedder::vec_for(text))
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
            if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1 {
                return Err(EmbedderError::Inference {
                    source: "rebuild failed".into(),
                });
            }
            texts.iter().map(|text| self.embed_doc(text)).collect()
        }
    }

    #[test]
    fn failed_rebuild_preserves_the_previous_searchable_cache() {
        let mut reg = with_embedder(Arc::new(FailSecondBatch {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }));
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();

        assert!(matches!(
            reg.rebuild_embeddings(),
            Err(EmbedderError::Inference { .. })
        ));
        let hits = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            hits.first().map(|hit| hit.tool_id.as_str()),
            Some("read_file")
        );
    }

    struct RebuildRaceEmbedder {
        rebuilding: AtomicBool,
        query_started: Barrier,
        release_query: Barrier,
        rebuild_entered: mpsc::Sender<()>,
    }

    impl RebuildRaceEmbedder {
        fn batch(&self, texts: &[String], rebuilding: bool) -> Vec<Vec<f32>> {
            texts
                .iter()
                .map(|text| match (text.contains("alpha"), rebuilding) {
                    (true, false) | (false, true) => vec![1.0, 0.0],
                    (false, false) | (true, true) => vec![0.0, 1.0],
                })
                .collect()
        }
    }

    impl Embedder for RebuildRaceEmbedder {
        fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(self.batch(&[text.to_string()], false).remove(0))
        }

        fn embed_query(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
            Ok(vec![1.0, 0.0])
        }

        fn embed_query_with_identity(&self, _: &str) -> Result<Embedded<Vec<f32>>, EmbedderError> {
            self.query_started.wait();
            self.release_query.wait();
            Ok(Embedded {
                value: vec![1.0, 0.0],
                fingerprint: "old".into(),
            })
        }

        fn embed_batch_with_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            let rebuilding = self.rebuilding.load(Ordering::SeqCst);
            if rebuilding {
                self.rebuild_entered.send(()).unwrap();
            }
            Ok(Embedded {
                value: self.batch(texts, rebuilding),
                fingerprint: if rebuilding { "new" } else { "old" }.into(),
            })
        }
    }

    #[test]
    fn rebuild_cannot_swap_vector_space_during_dense_search() {
        let (rebuild_entered_tx, rebuild_entered_rx) = mpsc::channel();
        let embedder = Arc::new(RebuildRaceEmbedder {
            rebuilding: AtomicBool::new(false),
            query_started: Barrier::new(2),
            release_query: Barrier::new(2),
            rebuild_entered: rebuild_entered_tx,
        });
        let mut reg = with_embedder(embedder.clone());
        reg.register(tool("alpha", "alpha"));
        reg.register(tool("beta", "beta"));
        reg.build_embeddings().unwrap();
        embedder.rebuilding.store(true, Ordering::SeqCst);
        let reg = Arc::new(reg);

        let search_reg = reg.clone();
        let search = std::thread::spawn(move || {
            search_reg.search_with_method("query", 2, Origin::Direct, SearchMethod::Semantic)
        });
        embedder.query_started.wait();

        let rebuild_started = Arc::new(Barrier::new(2));
        let rebuild_reg = reg.clone();
        let rebuild_thread_started = rebuild_started.clone();
        let rebuild = std::thread::spawn(move || {
            rebuild_thread_started.wait();
            rebuild_reg.rebuild_embeddings()
        });
        rebuild_started.wait();
        let swapped_during_query = rebuild_entered_rx
            .recv_timeout(Duration::from_millis(500))
            .is_ok();

        embedder.release_query.wait();
        let hits = search.join().unwrap().unwrap();
        rebuild.join().unwrap().unwrap();

        assert!(
            !swapped_during_query,
            "rebuild entered the embedder while a query from the old vector space was in flight"
        );
        assert_eq!(hits.first().map(|hit| hit.tool_id.as_str()), Some("alpha"));
    }

    #[test]
    fn direct_embedding_model_is_validated_before_loading() {
        let mut reg = ToolRegistry::with_embedding(EmbeddingModel::Endpoint {
            url: " ".into(),
            model: "model".into(),
            api_key_env: None,
            query_prefix: None,
            doc_prefix: None,
        });
        reg.register(tool("alpha", "alpha"));

        assert!(matches!(
            reg.build_embeddings(),
            Err(EmbedderError::Config { .. })
        ));
    }

    #[test]
    fn re_register_replaces_not_appends() {
        // Re-registering an id must REPLACE it in place, not append a duplicate.
        // A duplicate would inflate the BM25 corpus (avgdl drift, degrading scores
        // corpus-wide) and leak the old Tool + its embedding. The corpus must hold
        // exactly one entry per id (RAT-378).
        let mut reg = ToolRegistry::new();
        reg.register(tool("shared", "read a file"));
        reg.register(tool("shared", "delete a file"));
        assert_eq!(reg.len(), 1, "re-register replaces, not appends");
        // The single surviving entry ranks with the latest content.
        let hits = reg.search("delete a file", 5);
        assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("shared"));
        assert_eq!(hits.len(), 1, "one id in the corpus yields at most one hit");
    }

    #[test]
    fn re_register_updates_the_ranked_vector() {
        // Re-registering an id replaces it in place and invalidates its cached
        // embedding; the next build_embeddings re-embeds the new content, so the
        // updated content wins.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(tool("t", "read a file")); // dense vec keyed on "read"
        reg.build_embeddings().unwrap();
        reg.register(tool("t", "delete a file")); // re-register → invalidated, keyed on "delete"
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method("delete", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("t"));
        assert!(hits[0].score > 0.9, "ranks with the re-registered vector");
    }

    #[test]
    fn empty_registry_semantic_returns_no_hits_without_loading() {
        // Failing embedder + empty corpus: must short-circuit before any load.
        let reg = with_embedder(Arc::new(FailingEmbedder));
        let hits = reg
            .search_with_method("anything", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn register_and_search_emit_trace_events() {
        let sink = Arc::new(MemorySink::new("test-session"));
        let mut reg = ToolRegistry::with_trace_sink(sink.clone());
        reg.register(tool("read_file", "read a file"));
        reg.search_with_origin("read", 5, Origin::Agent);

        let events = sink.drain();
        assert!(events.iter().any(|e| matches!(
            e.event,
            TraceEvent::IndexChurn {
                kind: ChurnKind::Add,
                ..
            }
        )));
        assert!(events.iter().any(|e| matches!(
            &e.event,
            TraceEvent::Search { origin: Origin::Agent, hits, .. } if !hits.is_empty()
        )));
    }

    #[test]
    fn a_semantic_registry_learns_clusters_that_bridge_disjoint_vocabulary() {
        // THE headline behaviour, and what the lexical tier cannot do:
        // "delete a path" and "remove something" share NO content words, but the
        // embedder places them together. Learning on a semantic registry must
        // group them into one cluster, so evidence from one lifts the other.
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(Arc::new(crate::UsageLearner::new(
            graph.clone(),
            Arc::new(NoopSink),
        )));
        reg.set_intent_graph(Some(graph.clone()));
        reg.register(tool("delete_file", "delete a path"));
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();

        let observe = |q: &str, id: &str| {
            reg.search_with_method(q, 5, Origin::Agent, SearchMethod::Semantic)
                .unwrap();
            reg.record_event(TraceEvent::InvokeStart {
                tool_id: id.into(),
                args_size_bytes: 0,
            });
        };
        observe("delete a path", "delete_file");
        observe("remove something", "delete_file");

        let g = graph.read().unwrap();
        assert_eq!(
            g.len(),
            1,
            "disjoint wording, same meaning — one cluster, got {:?}",
            g.intents.iter().map(|i| &i.members).collect::<Vec<_>>()
        );
        assert!(
            g.intents[0].centroid.is_some(),
            "a semantic registry must grow a real centroid"
        );
    }

    #[test]
    fn a_bm25_registry_still_learns_without_any_centroid() {
        // No embedder is ever touched on this path (ADR-0011), so the cluster is
        // lexical and carries no centroid — the documented lesser tier.
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        let mut reg = ToolRegistry::new();
        reg.set_trace_sink(Arc::new(crate::UsageLearner::new(
            graph.clone(),
            Arc::new(NoopSink),
        )));
        reg.set_intent_graph(Some(graph.clone()));
        reg.register(tool("delete_file", "delete a path"));

        reg.search_with_origin("delete a path", 5, Origin::Agent);
        reg.record_event(TraceEvent::InvokeStart {
            tool_id: "delete_file".into(),
            args_size_bytes: 0,
        });

        let g = graph.read().unwrap();
        assert_eq!(g.len(), 1);
        assert!(g.intents[0].centroid.is_none());
    }

    #[test]
    fn a_stashed_vector_from_a_different_query_is_not_reused() {
        // Sessions share a graph, so a concurrent search can clobber the slot.
        // Keying it by query text means the mismatch degrades to lexical
        // clustering rather than attaching the wrong embedding to a question.
        let graph = IntentGraph::empty();
        graph.note_query_vector("some other query", &[1.0, 0.0, 0.0], "m");
        let mut graph = graph;
        graph.observe("delete a path", Capability::Tool, "delete_file", 1, true);

        assert_eq!(graph.len(), 1);
        assert!(
            graph.intents[0].centroid.is_none(),
            "must not borrow another query's embedding"
        );
    }

    // ---- usage ranking on the dense paths (ADR-0014) -----------------------

    /// A graph whose single cluster carries the stub embedder's "read" vector,
    /// so a query containing "read" matches it at cosine 1.0.
    fn read_graph(tool_id: &str, support: u32) -> Arc<RwLock<IntentGraph>> {
        let json = format!(
            r#"{{"v":1,"built_from_ts":1,
                 "intents":[{{"id":"i0","label":"l","terms":[],
                 "members":["read a file"],"centroid":[1.0,0.0,0.0],
                 "support":{support},"tools":{{"{tool_id}":1.0}},"skills":{{}}}}]}}"#
        );
        Arc::new(RwLock::new(IntentGraph::from_json(&json).expect("valid")))
    }

    #[test]
    fn semantic_search_is_unchanged_when_no_graph_is_attached() {
        // ADR-0011's byte-for-byte promise, on the dense path.
        let build = || {
            let mut reg = with_embedder(Arc::new(StubEmbedder));
            reg.register(tool("read_file", "read a file"));
            reg.register(tool("delete_file", "delete a path"));
            reg.build_embeddings().unwrap();
            reg
        };
        let baseline = build()
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        let mut with_graph = build();
        // A cluster the query cannot match (orthogonal centroid).
        let json = r#"{"v":1,"built_from_ts":1,
            "intents":[{"id":"i0","label":"l","terms":[],"members":["x"],
            "centroid":[0.0,1.0,0.0],"support":9,
            "tools":{"delete_file":1.0},"skills":{}}]}"#;
        with_graph.set_intent_graph(Some(Arc::new(RwLock::new(
            IntentGraph::from_json(json).unwrap(),
        ))));
        let missed = with_graph
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        assert_eq!(baseline.len(), missed.len());
        for (a, b) in baseline.iter().zip(missed.iter()) {
            assert_eq!(a.tool_id, b.tool_id);
            assert_eq!(a.score, b.score, "cosine score changed for {}", a.tool_id);
        }
    }

    #[test]
    fn semantic_search_truncates_to_top_k_when_the_graph_matches_no_cluster() {
        // Cold start: an empty graph is still attached, so retrieval depth jumps
        // to RETRIEVE_DEPTH — but no cluster can ever match. The no-match path
        // must still hand back exactly `top_k`, not the deep candidate list.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        // Four tools that all embed to the stub's "read" vector, so dense ranks
        // every one of them at cosine 1.0 — the deep list holds 4 entries.
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("read_path", "read a path"));
        reg.register(tool("read_bytes", "read bytes"));
        reg.register(tool("read_conf", "read config"));
        reg.build_embeddings().unwrap();
        reg.set_intent_graph(Some(Arc::new(RwLock::new(IntentGraph::empty()))));

        let hits = reg
            .search_with_method("read", 2, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        assert_eq!(hits.len(), 2, "no-match dense path must honor top_k");
    }

    /// A graph with an explicit model fingerprint and a chosen centroid width,
    /// for the model-change tests.
    fn graph_with_model(
        tool_id: &str,
        centroid: Vec<f32>,
        model: &str,
    ) -> Arc<RwLock<IntentGraph>> {
        let c: Vec<String> = centroid.iter().map(|x| x.to_string()).collect();
        let json = format!(
            r#"{{"v":1,"built_from_ts":1,"model":"{model}",
                 "intents":[{{"id":"i0","label":"l","terms":[],
                 "members":["read a file"],"centroid":[{}],
                 "support":9,"tools":{{"{tool_id}":1.0}},"skills":{{}}}}]}}"#,
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

    #[test]
    fn a_same_dim_model_mismatch_pauses_the_arm_and_warns() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a path")); // dense-orthogonal → ranks last
        reg.build_embeddings().unwrap();

        // Graph says boost delete_file for this intent — but its centroid was
        // built by a different model (same 3-dim width the stub uses).
        reg.set_intent_graph(Some(graph_with_model(
            "delete_file",
            vec![1.0, 0.0, 0.0],
            "a-different-model",
        )));
        let hits = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        // Paused: delete_file is NOT lifted; it stays last, as with no graph.
        assert_eq!(hits.last().map(|h| h.tool_id.as_str()), Some("delete_file"));
        assert!(hits.iter().all(|h| !h.fused), "no fusion — the arm paused");

        let events = model_mismatch_events(&sink);
        assert_eq!(events.len(), 1);
        assert!(!events[0].2, "same-dim swap → dim_mismatch false");
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
        // only ever boost lexically — `arm()` takes the lexical path with no query
        // vector, and it does boost. Status must report the arm is live, not sit
        // at Unknown forever just because no active model fingerprint exists.
        let mut reg = ToolRegistry::new();
        reg.register(tool("read_file", "read a file"));
        reg.set_intent_graph(Some(graph_with_model(
            "read_file",
            vec![1.0, 0.0, 0.0],
            "some-model",
        )));
        assert_eq!(reg.adaptive_ranking_status(), AdaptiveRankingStatus::Active);
    }

    #[test]
    fn a_poisoned_graph_lock_reports_unknown_not_a_panic() {
        // The search path degrades on a poisoned lock; a status query must too —
        // a read-only getter that can crash the caller is a footgun.
        let mut reg = ToolRegistry::new();
        reg.register(tool("read_file", "read a file"));
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
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a path"));
        reg.build_embeddings().unwrap();
        let graph = graph_with_model("delete_file", vec![0.0, 1.0, 0.0], "m");
        poison(&graph);
        reg.set_intent_graph(Some(graph));

        let hits = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert!(hits.iter().all(|h| !h.fused), "poisoned lock → arm paused");
    }

    #[test]
    fn rebuild_recovers_a_poisoned_graph_lock_not_a_panic() {
        // rebuild is the repair path — it overwrites every centroid, so a lock an
        // earlier panic poisoned is recovered and the call completes, never panics.
        let mut reg = ToolRegistry::new();
        reg.register(tool("read_file", "read a file"));
        let graph = Arc::new(RwLock::new(IntentGraph::empty()));
        poison(&graph);
        reg.set_intent_graph(Some(graph));

        assert!(reg.rebuild_intent_graph().is_ok());
    }

    #[test]
    fn a_dim_mismatch_pauses_the_arm_and_warns() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a path"));
        reg.build_embeddings().unwrap();

        // Centroid is 5-dim; the stub embeds queries to 3-dim.
        reg.set_intent_graph(Some(graph_with_model(
            "delete_file",
            vec![1.0, 0.0, 0.0, 0.0, 0.0],
            "some-model",
        )));
        let hits = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();

        assert_eq!(hits.last().map(|h| h.tool_id.as_str()), Some("delete_file"));
        let events = model_mismatch_events(&sink);
        assert_eq!(events.len(), 1);
        assert!(events[0].2, "different width → dim_mismatch true");
    }

    #[test]
    fn rebuild_intent_graph_restores_the_arm_after_a_model_change() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(tool("read_file", "read a file"));
        reg.register(tool("delete_file", "delete a path"));
        reg.build_embeddings().unwrap();
        reg.set_intent_graph(Some(graph_with_model(
            "read_file",
            vec![1.0, 0.0, 0.0],
            "a-different-model",
        )));

        // Paused before rebuild.
        reg.search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(model_mismatch_events(&sink).len(), 1);

        // Rebuild re-embeds members under the stub model → arm active again.
        reg.rebuild_intent_graph().unwrap();
        let after = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert!(after.iter().all(|h| h.fused), "arm resumed → fused ranking");
        assert!(
            model_mismatch_events(&sink).is_empty(),
            "no mismatch after rebuild"
        );
    }

    #[test]
    fn semantic_search_promotes_the_usage_arms_tool() {
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(tool("read_file", "read a file"));
        // Embeds orthogonally to the query, so dense ranks it last...
        reg.register(tool("delete_file", "delete a path"));
        reg.build_embeddings().unwrap();

        let before = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(before[0].tool_id, "read_file");

        // ...but usage history says people invoke it for this intent.
        reg.set_intent_graph(Some(read_graph("delete_file", 9)));
        let after = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        let pos = |hits: &[SearchHit], id: &str| hits.iter().position(|h| h.tool_id == id).unwrap();
        assert!(
            pos(&after, "delete_file") < pos(&before, "delete_file"),
            "the usage arm should lift it, got {:?}",
            after.iter().map(|h| &h.tool_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_usage_match_reuses_the_dense_query_vector() {
        // The claim that adaptive ranking is free on semantic/hybrid rests on
        // this: matching against centroids must not trigger a second inference.
        let counting = Arc::new(CountingEmbedder::new());
        let mut reg = with_embedder(counting.clone());
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();
        reg.set_intent_graph(Some(read_graph("read_file", 9)));

        reg.search_with_method("read", 5, Origin::Direct, SearchMethod::Semantic)
            .unwrap();
        assert_eq!(
            counting.query_calls(),
            1,
            "one search must embed the query exactly once"
        );
    }

    #[test]
    fn hybrid_scores_are_always_flagged_fused() {
        // Hybrid is RRF-scale with or without a graph, so `fused` is always true
        // — retroactively documenting what was already the case.
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();
        let hits = reg
            .search_with_method("read", 5, Origin::Direct, SearchMethod::Hybrid)
            .unwrap();
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.fused), "hybrid is always RRF-scale");
    }

    #[test]
    fn hybrid_search_reports_all_four_stages_when_the_arm_fires() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();
        reg.set_intent_graph(Some(read_graph("read_file", 9)));

        reg.search_with_method("read", 5, Origin::Direct, SearchMethod::Hybrid)
            .unwrap();

        let stages: Vec<String> = sink
            .drain()
            .into_iter()
            .find_map(|e| match e.event {
                TraceEvent::Search { stages, .. } => Some(stages),
                _ => None,
            })
            .expect("a search event")
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(stages, vec!["bm25", "dense", "usage", "rrf"]);
    }

    #[test]
    fn hybrid_search_keeps_three_stages_when_the_query_misses() {
        let sink = Arc::new(MemorySink::new("s"));
        let mut reg = with_embedder(Arc::new(StubEmbedder));
        reg.set_trace_sink(sink.clone());
        reg.register(tool("read_file", "read a file"));
        reg.build_embeddings().unwrap();
        reg.set_intent_graph(Some(read_graph("read_file", 9)));

        // "delete" embeds orthogonally to the cluster centroid.
        reg.search_with_method("delete", 5, Origin::Direct, SearchMethod::Hybrid)
            .unwrap();

        let stages: Vec<String> = sink
            .drain()
            .into_iter()
            .find_map(|e| match e.event {
                TraceEvent::Search { stages, .. } => Some(stages),
                _ => None,
            })
            .expect("a search event")
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(stages, vec!["bm25", "dense", "rrf"]);
    }
}
