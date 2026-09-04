use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use bm25::{DefaultTokenizer, Document, Language, SearchEngine, SearchEngineBuilder, Tokenizer};

/// Shipped `k1`, unless a caller overrides it via [`Bm25Params`].
///
/// Below the crate default because tool text is short, so a term repeating
/// adds little (ADR-0004).
pub(crate) const BM25_K1: f32 = 0.9;
/// Shipped `b`, unless a caller overrides it via [`Bm25Params`].
///
/// Standard-value length normalisation. A tool's indexed document is its
/// description plus every schema token, so length differences partly reflect
/// how many arguments a tool takes rather than how much it says — that is why
/// ADR-0004 originally discounted `b` to 0.4 below standard.
///
/// **BFCL later re-measured `b` on its own** — 599 scenarios, lexical arm
/// only, no intent graph — and 0.75 won narrowly there: hit@1 0.8998 to
/// 0.9065, MRR@5 0.9366 to 0.9410, recall unchanged at every `k`. On the
/// in-repo 47-query fixture, though, 0.75 raises how often a write op crowds
/// out a read query (8/25 to 11/25) — a failure mode BFCL's taxonomy cannot
/// see at all. See the `b` sweep in `harness-results.md` and ADR-0023/ADR-0024.
///
/// **The shipped default is 0.4**, not the BFCL-favored 0.75: different
/// deployments have differently-shaped documents than the tool-call corpora
/// this constant was tuned against — a catalog that sets
/// `experimental_searchable_description` everywhere skips schema flattening
/// entirely ([`crate::indexing::searchable_text`]) and has near-uniform
/// document length, and a catalog of long-form skill documents looks nothing
/// like a BFCL scenario. [`Bm25Params`] lets a caller pick 0.75 (or any other
/// value) back for a corpus that does resemble what BFCL measured.
pub(crate) const BM25_B: f32 = 0.4;

/// `k1`/`b` for a BM25 index build (ADR-0004, ADR-0023/ADR-0024).
///
/// Unlike most of this module's tuning, this one is a deliberate public knob
/// (not "fixed tuning" in the ADR-0004 sense) — the corpus-shape assumption
/// behind the shipped defaults does not hold for every deployment. See the
/// [`BM25_B`] doc comment for when a caller would want to override it. No
/// built-in evaluation ships alongside this: a caller who overrides it has no
/// way to tell, from this crate alone, whether the override helped their
/// corpus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation. Must be finite and non-negative.
    pub k1: f32,
    /// Length normalisation. Must be finite and in `[0, 1]`.
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self {
            k1: BM25_K1,
            b: BM25_B,
        }
    }
}

impl Bm25Params {
    /// Set `k1`.
    #[must_use]
    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Set `b`.
    #[must_use]
    pub fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Whether `k1` and `b` are both in their mathematically valid domain —
    /// `k1` finite and non-negative, `b` finite and in `[0, 1]`. Rejected
    /// rather than clamped at the SDK boundary, same reasoning as
    /// [`crate::ClusterPolicy`]: a clamp would silently search at a tuning the
    /// caller did not ask for.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.k1.is_finite() && self.k1 >= 0.0 && self.b.is_finite() && (0.0..=1.0).contains(&self.b)
    }
}

/// A prebuilt BM25 index over `(id, searchable_text)` documents: build once
/// per corpus state, query many times. Building tokenizes and stems the whole
/// corpus — ~100x the cost of one query — so the registries cache one of these
/// (see [`Bm25Cache`]) and rebuild only after a corpus mutation, instead of the
/// historical build-per-search. Every query ranks the full corpus in
/// `(score desc, id asc)` order and then cuts to `top_k`.
pub(crate) struct Bm25Index {
    /// `None` when the corpus is empty — the engine is never built for zero
    /// documents (its `avgdl` fit has no corpus to fit).
    engine: Option<SearchEngine<String>>,
    doc_count: usize,
    /// How many documents contain each stemmed token. The engine keeps this
    /// internally but does not expose it, and [`Self::query_ceiling`] needs IDF
    /// to say what a good score for a given query even *is*.
    df: HashMap<String, usize>,
}

/// The tokenizer the engine is built with — same language, same defaults
/// (normalize, stem, drop stopwords). Rebuilt here rather than borrowed because
/// the crate keeps its own private; if the two ever diverge, `query_ceiling`
/// would divide by a ceiling computed over different terms than the score.
fn tokenizer() -> DefaultTokenizer {
    DefaultTokenizer::new(Language::English)
}

impl Bm25Index {
    /// [`Self::build_with`] at the shipped [`Bm25Params::default`]. Production
    /// always goes through [`Bm25Cache`], which reads the registry's current
    /// params instead — this exists for tests and the harness, which want the
    /// default without threading it explicitly.
    #[cfg(test)]
    pub(crate) fn build<I>(docs: I) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
    {
        Self::build_with(docs, BM25_K1, BM25_B)
    }

    /// Tokenize and index `docs` at explicit `k1`/`b`. Corpus statistics
    /// (`avgdl`, IDF) come from a complete build over exactly these documents,
    /// so a rebuild after any mutation scores byte-for-byte like a fresh
    /// one-shot search — never incrementally upsert into the engine, that
    /// freezes a stale `avgdl` into new documents' scores.
    ///
    /// [`Bm25Cache`] calls this directly with the registry's current
    /// [`Bm25Params`] on every rebuild — this is the one production path.
    pub(crate) fn build_with<I>(docs: I, k1: f32, b: f32) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let pairs: Vec<(String, String)> = docs.into_iter().collect();
        let doc_count = pairs.len();
        if doc_count == 0 {
            return Self {
                engine: None,
                doc_count,
                df: HashMap::new(),
            };
        }
        let tk = tokenizer();
        let mut df: HashMap<String, usize> = HashMap::new();
        for (_, contents) in &pairs {
            let mut seen = tk.tokenize(contents);
            seen.sort_unstable();
            seen.dedup();
            for token in seen {
                *df.entry(token).or_insert(0) += 1;
            }
        }
        let engine = SearchEngineBuilder::<String>::with_documents(
            Language::English,
            pairs
                .into_iter()
                .map(|(id, contents)| Document { id, contents }),
        )
        .k1(k1)
        .b(b)
        .build();
        Self {
            engine: Some(engine),
            doc_count,
            df,
        }
    }

    /// The score a hypothetical *ideal* document would earn for `query` — the
    /// yardstick that turns an unbounded BM25 score into a readable fraction.
    ///
    /// A term contributes `idf · tf(k1+1) / (tf + k1(1 - b + b·dl/avgdl))`. At
    /// `tf = 1` and `dl = avgdl` the length factor collapses to `k1`, so the
    /// fraction is `(k1+1)/(1+k1) = 1` and the term contributes its IDF
    /// unchanged. Summing IDF over the query's distinct terms is therefore the
    /// score of an average-length document containing each query term once.
    ///
    /// **Terms absent from the corpus still count.** A query word no document
    /// contains has the highest IDF of all and contributes nothing to any real
    /// score, so it drags every ratio down — correctly. That gap is the honest
    /// reading "this catalog cannot fully answer you", and hiding it would make
    /// a query the corpus has no vocabulary for look as answerable as one it does.
    ///
    /// Not a hard maximum: a short document repeating a term can exceed it, so
    /// callers clamp.
    pub(crate) fn query_ceiling(&self, query: &str) -> f32 {
        let n = self.doc_count as f32;
        if n == 0.0 {
            return 0.0;
        }
        let mut terms = tokenizer().tokenize(query);
        terms.sort_unstable();
        terms.dedup();
        terms
            .iter()
            .map(|t| {
                let df = self.df.get(t).copied().unwrap_or(0) as f32;
                (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
            })
            .sum()
    }

    /// Top-`top_k` matches as `(id, score)`, best-first with ties broken by
    /// `id` so the result is deterministic across processes.
    pub(crate) fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        let Some(engine) = &self.engine else {
            return Vec::new();
        };
        // Rank against the full corpus, then truncate — never let the engine
        // cut to `top_k` itself. The bm25 crate sorts by score alone and
        // collects candidates through a HashSet, so equal scores fall back to
        // hash-seed iteration order and flip between processes; a tie
        // straddling the `top_k` boundary would make top-K *membership*
        // nondeterministic. We rank everything, break ties by id, then cut —
        // so both the tool and skill buckets are stable. (Centralizes #63,
        // which originally fixed only the tool path in ToolRegistry::search.)
        let mut ranked: Vec<(String, f32)> = engine
            .search(query, self.doc_count)
            .into_iter()
            .map(|r| (r.document.id, r.score))
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(top_k);
        ranked
    }
}

/// Dirty-flag holder for a registry's [`Bm25Index`]: `get_or_build` returns
/// the cached index or builds it from the caller's corpus snapshot, and every
/// corpus mutation calls [`Self::invalidate`] so the next search rebuilds.
/// Interior mutability (the [`DenseCache`] precedent) because searches take
/// `&self`; the `Arc` lets a search rank outside the lock, so a concurrent
/// invalidate never blocks on (or invalidates out from under) a live query.
///
/// [`DenseCache`]: crate::dense_cache::DenseCache
pub(crate) struct Bm25Cache {
    index: Mutex<Option<Arc<Bm25Index>>>,
    params: Mutex<Bm25Params>,
}

impl Bm25Cache {
    pub(crate) fn new() -> Self {
        Self {
            index: Mutex::new(None),
            params: Mutex::new(Bm25Params::default()),
        }
    }

    /// The cached index, or the one built from `docs()` and cached. `docs` is
    /// only called on a cache miss — the first search after construction, or
    /// after [`Self::invalidate`]/[`Self::set_params`].
    ///
    /// The corpus snapshot runs under the lock: registries mutate through
    /// `&mut self`, so no mutation can interleave, and concurrent first
    /// searches build once instead of racing.
    pub(crate) fn get_or_build<I>(&self, docs: impl FnOnce() -> I) -> Arc<Bm25Index>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let mut slot = self.index.lock().unwrap_or_else(PoisonError::into_inner);
        match &*slot {
            Some(index) => Arc::clone(index),
            None => {
                let params = *self.params.lock().unwrap_or_else(PoisonError::into_inner);
                let index = Arc::new(Bm25Index::build_with(docs(), params.k1, params.b));
                *slot = Some(Arc::clone(&index));
                index
            }
        }
    }

    /// Drop the cached index; the next [`Self::get_or_build`] rebuilds from
    /// the then-current corpus. Called by every registry mutation.
    pub(crate) fn invalidate(&self) {
        *self.index.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }

    /// `k1`/`b` the next build will use.
    pub(crate) fn params(&self) -> Bm25Params {
        *self.params.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Set `k1`/`b` for future builds and invalidate the cached index — a
    /// param change must rebuild, same as any corpus mutation, or a search
    /// right after `set_params` would still score under the old tuning.
    pub(crate) fn set_params(&self, params: Bm25Params) {
        *self.params.lock().unwrap_or_else(PoisonError::into_inner) = params;
        self.invalidate();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference ranking: a fresh index built for exactly this query —
    /// what the historical build-per-search path computed. Reused/cached
    /// indexes must match it byte-for-byte.
    fn fresh_search(docs: Vec<(String, String)>, query: &str, top_k: usize) -> Vec<(String, f32)> {
        Bm25Index::build(docs).search(query, top_k)
    }

    fn tie_corpus() -> Vec<(String, String)> {
        vec![
            (
                "zeta".to_string(),
                "send a notification message".to_string(),
            ),
            (
                "alpha".to_string(),
                "send a notification message".to_string(),
            ),
            ("mid".to_string(), "send a notification message".to_string()),
            (
                "reader".to_string(),
                "read a file from disk with an absolute path".to_string(),
            ),
        ]
    }

    #[test]
    fn empty_index_yields_no_hits() {
        let index = Bm25Index::build(Vec::new());
        assert!(index.search("anything", 5).is_empty());
    }

    #[test]
    fn ranks_the_lexically_closest_document_first() {
        let docs = vec![
            ("read".to_string(), "read a file from disk".to_string()),
            (
                "write".to_string(),
                "write bytes to a network socket".to_string(),
            ),
        ];
        let hits = fresh_search(docs, "read file", 5);
        assert_eq!(hits.first().map(|(id, _)| id.as_str()), Some("read"));
    }

    #[test]
    fn respects_top_k() {
        let docs: Vec<(String, String)> = (0..10)
            .map(|i| (format!("doc{i}"), "shared term content".to_string()))
            .collect();
        let hits = fresh_search(docs, "shared", 3);
        assert!(hits.len() <= 3);
    }

    #[test]
    fn a_reused_index_matches_a_fresh_build_exactly() {
        // The whole point of caching: querying one prebuilt index must return
        // exactly what a fresh build for each query would — ids AND f32
        // scores, including on tied scores.
        let index = Bm25Index::build(tie_corpus());
        for query in ["notification message", "read file", "absolute disk path"] {
            for top_k in [1, 2, 10] {
                assert_eq!(
                    index.search(query, top_k),
                    fresh_search(tie_corpus(), query, top_k),
                    "query={query} top_k={top_k}"
                );
            }
        }
    }

    #[test]
    fn cache_builds_once_and_reuses_the_index() {
        let cache = Bm25Cache::new();
        let builds = std::cell::Cell::new(0);
        let build = || {
            builds.set(builds.get() + 1);
            tie_corpus()
        };
        let first = cache.get_or_build(build);
        let second = cache.get_or_build(build);
        assert_eq!(builds.get(), 1, "second get must reuse, not rebuild");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(
            first.search("notification", 5),
            fresh_search(tie_corpus(), "notification", 5)
        );
    }

    #[test]
    fn invalidate_forces_a_rebuild_over_the_new_corpus() {
        let cache = Bm25Cache::new();
        let _ = cache.get_or_build(tie_corpus);
        cache.invalidate();
        let new_corpus = || vec![("fresh".to_string(), "compact a database table".to_string())];
        let rebuilt = cache.get_or_build(new_corpus);
        assert_eq!(
            rebuilt.search("compact database", 5),
            fresh_search(new_corpus(), "compact database", 5),
            "post-invalidate index must reflect only the new corpus"
        );
        assert!(rebuilt.search("notification", 5).is_empty());
    }

    #[test]
    fn bm25_params_default_is_the_shipped_tuning() {
        let params = Bm25Params::default();
        assert_eq!(params.k1, BM25_K1);
        assert_eq!(params.b, BM25_B);
        assert_eq!(params.b, 0.4, "shipped default reverted to 0.4");
    }

    #[test]
    fn bm25_params_rejects_b_outside_zero_one() {
        assert!(!Bm25Params::default().with_b(-0.01).is_valid());
        assert!(!Bm25Params::default().with_b(1.01).is_valid());
        assert!(!Bm25Params::default().with_b(f32::NAN).is_valid());
        assert!(Bm25Params::default().with_b(0.0).is_valid());
        assert!(Bm25Params::default().with_b(1.0).is_valid());
    }

    #[test]
    fn bm25_params_rejects_negative_or_nan_k1() {
        assert!(!Bm25Params::default().with_k1(-0.01).is_valid());
        assert!(!Bm25Params::default().with_k1(f32::NAN).is_valid());
        assert!(Bm25Params::default().with_k1(0.0).is_valid());
    }

    #[test]
    fn cache_set_params_rebuilds_and_changes_ranking() {
        // "long" repeats the query term 4x across 16 words; "short" has it once
        // in a 1-word document. At low b (default 0.4), the extra term
        // frequency outweighs the modest length cost and "long" wins. At b=1.0
        // (full length normalization), the length cost dominates and "short"
        // wins. The margin (~20-25%) is deliberately wide: an earlier,
        // narrower corpus flipped on a difference as small as one extra
        // indexed token, because a 30x-repetition term frequency saturates
        // with k1=0.9 and nearly cancels the length penalty regardless of b.
        let docs = || {
            vec![
                ("short".to_string(), "read".to_string()),
                ("long".to_string(), "read read read read filler1 filler2 filler3 filler4 filler5 filler6 filler7 filler8 filler9 filler10 filler11 filler12".to_string()),
            ]
        };
        let cache = Bm25Cache::new();
        let low_b = cache.get_or_build(docs);
        let low_b_top = low_b.search("read", 1)[0].0.clone();

        cache.set_params(Bm25Params::default().with_b(1.0));
        let high_b = cache.get_or_build(docs);
        let high_b_top = high_b.search("read", 1)[0].0.clone();

        assert_eq!(low_b_top, "long", "at low b, term frequency should win");
        assert_eq!(high_b_top, "short", "at high b, length penalty should win");
    }

    #[test]
    fn cache_params_defaults_and_reflects_what_was_set() {
        let cache = Bm25Cache::new();
        assert_eq!(cache.params(), Bm25Params::default());

        let overridden = Bm25Params::default().with_k1(1.2).with_b(0.75);
        cache.set_params(overridden);
        assert_eq!(cache.params(), overridden);
    }

    #[test]
    fn tied_scores_break_by_id_with_stable_top_k_membership() {
        // Identical searchable text → identical scores for any matching query.
        // The bm25 crate collects candidates through a HashSet, so the engine's
        // own order is hash-seed dependent; Bm25Index::search must impose a
        // stable (score desc, id asc) order so both the returned order AND
        // which docs survive the top_k cut are fixed across processes. Shared
        // by the tool and skill registries.
        let docs = vec![
            (
                "zeta".to_string(),
                "send a notification message".to_string(),
            ),
            (
                "alpha".to_string(),
                "send a notification message".to_string(),
            ),
            ("mid".to_string(), "send a notification message".to_string()),
        ];
        let hits = fresh_search(docs, "notification message", 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, "alpha");
        assert_eq!(hits[1].0, "mid");
    }
}
