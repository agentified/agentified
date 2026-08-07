use std::sync::{Arc, Mutex, PoisonError};

use bm25::{Document, Language, SearchEngine, SearchEngineBuilder};

// Tuned for short tool/skill descriptions; see ADR-0004.
pub(crate) const BM25_K1: f32 = 0.9;
pub(crate) const BM25_B: f32 = 0.4;

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
}

impl Bm25Index {
    /// Tokenize and index `docs`. Corpus statistics (`avgdl`, IDF) come from a
    /// complete build over exactly these documents, so a rebuild after any
    /// mutation scores byte-for-byte like a fresh one-shot search — never
    /// incrementally upsert into the engine, that freezes a stale `avgdl` into
    /// new documents' scores.
    pub(crate) fn build<I>(docs: I) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let pairs: Vec<(String, String)> = docs.into_iter().collect();
        let doc_count = pairs.len();
        if doc_count == 0 {
            return Self {
                engine: None,
                doc_count,
            };
        }
        let engine = SearchEngineBuilder::<String>::with_documents(
            Language::English,
            pairs
                .into_iter()
                .map(|(id, contents)| Document { id, contents }),
        )
        .k1(BM25_K1)
        .b(BM25_B)
        .build();
        Self {
            engine: Some(engine),
            doc_count,
        }
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
}

impl Bm25Cache {
    pub(crate) fn new() -> Self {
        Self {
            index: Mutex::new(None),
        }
    }

    /// The cached index, or the one built from `docs()` and cached. `docs` is
    /// only called on a cache miss — the first search after construction or
    /// [`Self::invalidate`].
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
                let index = Arc::new(Bm25Index::build(docs()));
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
