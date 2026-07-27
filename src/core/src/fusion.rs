//! Rank fusion and the shared deterministic ordering used across rankers.
//!
//! Reciprocal Rank Fusion (RRF) combines the BM25 and dense rankings into one
//! candidate list for the hybrid pipeline (see [`crate::tool_registry`] and
//! ADR-0011). It fuses on *rank position*, not raw scores, so it is immune to
//! the incomparable scales of BM25 (unbounded) and cosine ([-1, 1]). Pure Rust,
//! no heavy deps — its tests run on every build without a model download.

/// RRF damping constant. 60 is the Cormack et al. (2009) default and the field
/// standard; large enough that the reciprocal curve is gentle past the head of
/// each list, small enough that top ranks still dominate.
pub(crate) const RRF_K: f32 = 60.0;

/// How deep each arm (BM25, dense) retrieves before fusion. Deeper than `top_k`
/// so a tool the two arms rank differently still has rank signal to fuse.
pub(crate) const RETRIEVE_DEPTH: usize = 100;

/// One already-ranked, best-first id list plus the weight its rank positions
/// carry into the fusion. `1.0` is the baseline the BM25 and dense arms use.
pub(crate) type WeightedArm<'a> = (&'a [String], f32);

/// Reciprocal Rank Fusion with a **per-arm weight**:
/// `score(id) = Σ_arms w_arm · 1 / (k + rank_in_arm)`.
///
/// [`rrf_fuse`] is this at `w = 1.0` for every arm. The weight exists for the
/// usage-ranking arm (ADR-0014), which is deliberately sub-unit: at the same rank
/// a capability the query lexically matched outranks one only usage history
/// supports. A sub-unit arm still promotes a deeply-ranked id past another arm's
/// rank-0 hit, because the id accumulates from both arms — it damps the arm
/// without disabling it.
///
/// Weights scale contributions only; ordering, tie-breaking, and determinism are
/// unchanged (`(score desc, id asc)`, see [`sort_and_truncate`]). An arm at
/// weight `0.0` still contributes its ids to the candidate set at score `0.0`,
/// so muting an arm is not the same as omitting it — callers that want an arm
/// gone must not pass it.
pub(crate) fn rrf_fuse_weighted(lists: &[WeightedArm<'_>], k: f32) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    let mut scores: HashMap<&str, f32> = HashMap::new();
    for (list, weight) in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(id.as_str()).or_insert(0.0) += weight / (k + rank as f32);
        }
    }
    let mut ranked: Vec<(String, f32)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect();
    let len = ranked.len();
    sort_and_truncate(&mut ranked, len);
    ranked
}

/// The shared `(score desc, id asc)` ordering, then truncate to `top_k`. Ranking
/// the full set before the cut keeps top-K *membership* stable when a tie
/// straddles the boundary — see the rationale in [`crate::search::bm25_search`].
pub(crate) fn sort_and_truncate(ranked: &mut Vec<(String, f32)>, top_k: usize) {
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(top_k);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Fuse at the baseline weight — the plain RRF every arm used before the
    /// usage arm introduced weighting. Keeps the original behavioural tests
    /// expressed in terms of unweighted fusion.
    fn rrf_fuse(lists: &[&[String]], k: f32) -> Vec<(String, f32)> {
        let weighted: Vec<WeightedArm<'_>> = lists.iter().map(|l| (*l, 1.0)).collect();
        rrf_fuse_weighted(&weighted, k)
    }

    #[test]
    fn empty_input_yields_no_hits() {
        assert!(rrf_fuse(&[], RRF_K).is_empty());
        let empty: Vec<String> = Vec::new();
        assert!(rrf_fuse(&[empty.as_slice()], RRF_K).is_empty());
    }

    #[test]
    fn single_list_preserves_order_with_reciprocal_scores() {
        let list = ids(&["a", "b", "c"]);
        let fused = rrf_fuse(&[list.as_slice()], RRF_K);
        assert_eq!(
            fused.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        // score(rank r) = 1 / (60 + r)
        assert!((fused[0].1 - 1.0 / 60.0).abs() < 1e-6);
        assert!((fused[1].1 - 1.0 / 61.0).abs() < 1e-6);
        assert!((fused[2].1 - 1.0 / 62.0).abs() < 1e-6);
    }

    #[test]
    fn doc_in_both_lists_outranks_doc_in_one() {
        // "shared" sits rank 1 in both lists; "solo" sits rank 0 in one only.
        // 1/61 + 1/61 ≈ 0.0328 beats 1/60 ≈ 0.0167 — agreement across arms wins.
        let bm25 = ids(&["solo", "shared"]);
        let dense = ids(&["other", "shared"]);
        let fused = rrf_fuse(&[bm25.as_slice(), dense.as_slice()], RRF_K);
        assert_eq!(fused.first().map(|(id, _)| id.as_str()), Some("shared"));
    }

    #[test]
    fn tied_scores_break_by_id_ascending() {
        // Each id appears once at rank 0 in its own list → identical RRF scores.
        let l1 = ids(&["zeta"]);
        let l2 = ids(&["alpha"]);
        let l3 = ids(&["mid"]);
        let fused = rrf_fuse(&[l1.as_slice(), l2.as_slice(), l3.as_slice()], RRF_K);
        assert_eq!(
            fused.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "mid", "zeta"]
        );
    }

    #[test]
    fn arg_order_does_not_change_the_result() {
        let bm25 = ids(&["a", "b", "c"]);
        let dense = ids(&["b", "c", "d"]);
        let forward = rrf_fuse(&[bm25.as_slice(), dense.as_slice()], RRF_K);
        let swapped = rrf_fuse(&[dense.as_slice(), bm25.as_slice()], RRF_K);
        assert_eq!(forward, swapped);
    }

    #[test]
    fn baseline_weights_reproduce_the_classic_rrf_scores() {
        // Adding the weight parameter must not perturb the two-arm hybrid
        // pipeline: at w=1 the scores are still Σ 1/(60 + rank).
        let bm25 = ids(&["a", "b"]);
        let dense = ids(&["b", "a"]);
        let fused = rrf_fuse_weighted(&[(bm25.as_slice(), 1.0), (dense.as_slice(), 1.0)], RRF_K);
        // Both ids sit at ranks 0 and 1 once each, so both score 1/60 + 1/61.
        let expected = 1.0 / 60.0 + 1.0 / 61.0;
        for (id, score) in &fused {
            assert!((score - expected).abs() < 1e-6, "{id} scored {score}");
        }
    }

    #[test]
    fn a_zero_weight_arm_contributes_nothing() {
        let bm25 = ids(&["a", "b"]);
        let muted = ids(&["z"]);
        let fused = rrf_fuse_weighted(&[(bm25.as_slice(), 1.0), (muted.as_slice(), 0.0)], RRF_K);
        // "z" is present (the arm listed it) but scores 0, so it sorts last.
        assert_eq!(fused.last().map(|(id, _)| id.as_str()), Some("z"));
        assert_eq!(fused.last().map(|(_, s)| *s), Some(0.0));
    }

    #[test]
    fn a_heavier_arm_outvotes_a_lighter_one_at_the_same_rank() {
        // Rank 0 in the heavy arm beats rank 0 in the light arm.
        let light = ids(&["lex"]);
        let heavy = ids(&["usage"]);
        let fused = rrf_fuse_weighted(&[(light.as_slice(), 1.0), (heavy.as_slice(), 2.0)], RRF_K);
        assert_eq!(fused.first().map(|(id, _)| id.as_str()), Some("usage"));
    }

    #[test]
    fn sub_unit_usage_weight_lets_the_lexical_arm_win_at_equal_rank() {
        // ADR-0014 ships W < 1: at the SAME rank, a capability the query lexically
        // matched must outrank one only usage history supports. `c` sits at rank 2 of
        // the usage arm (w=0.5), `d` at rank 2 of the lexical arm (w=1) — so `d` wins.
        // At w=1 they would tie and fall back to id order, which is the boundary this
        // pins. See the risk note in ADR-0014 on W's direction.
        let lexical = ids(&["a", "b", "d"]);
        let usage = ids(&["a", "b", "c"]);
        let fused = rrf_fuse_weighted(&[(lexical.as_slice(), 1.0), (usage.as_slice(), 0.5)], RRF_K);
        let order: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(order, vec!["a", "b", "d", "c"]);
    }

    #[test]
    fn a_sub_unit_arm_still_promotes_a_deeply_ranked_id_past_the_lexical_top_hit() {
        // The headline case: BM25 ranks the wrong tool first and the right one deep.
        // Even at w=0.5 the usage arm lifts it past rank 0, because the id draws from
        // BOTH arms. This is why W<1 is still useful — it damps the arm without
        // disabling it.
        let mut lexical = vec!["docker_build".to_string()];
        lexical.extend((0..49).map(|i| format!("filler{i:02}")));
        lexical.push("gh_run_list".to_string()); // rank 50
        let usage = ids(&["gh_run_list"]);
        let fused = rrf_fuse_weighted(&[(lexical.as_slice(), 1.0), (usage.as_slice(), 0.5)], RRF_K);
        assert_eq!(
            fused.first().map(|(id, _)| id.as_str()),
            Some("gh_run_list")
        );
    }

    #[test]
    fn sort_and_truncate_keeps_stable_membership_across_a_tie_boundary() {
        // Three tied scores, cut to 2: membership must be the id-ascending head,
        // not whatever order they arrived in.
        let mut ranked = vec![
            ("zeta".to_string(), 1.0_f32),
            ("alpha".to_string(), 1.0),
            ("mid".to_string(), 1.0),
        ];
        sort_and_truncate(&mut ranked, 2);
        assert_eq!(
            ranked.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
            vec!["alpha", "mid"]
        );
    }
}
