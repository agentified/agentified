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
/// Plain RRF is this at `w = 1.0` for every arm. The weight exists for the
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
/// straddles the boundary — see the rationale in [`crate::search::Bm25Index::search`].
pub(crate) fn sort_and_truncate(ranked: &mut Vec<(String, f32)>, top_k: usize) {
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(top_k);
}

/// Default weight the dense arm carries in [`score_fuse`]; BM25 takes the
/// remainder. See [`DenseWeight`] for why it is a default rather than a
/// constant.
///
/// 0.7/0.3 rather than even, because the lexical arm is the weaker of the two on
/// every corpus measured so far — on the harness fixture BM25 alone recovers the
/// invoked tool 12 times in 47 against dense's 23 — and because a BM25 score is
/// zero for anything the arm never returned, so an even split lets a query with
/// no lexical purchase halve every candidate uniformly.
pub(crate) const SCORE_FUSION_DENSE_WEIGHT: f32 = 0.7;

/// The share of the hybrid **content** score the dense arm carries; BM25 takes
/// `1 - w`. Only [`SearchMethod::Hybrid`] reads it — the single-arm methods have
/// nothing to weigh.
///
/// **Why this is a knob and not a constant.** The balance between lexical and
/// semantic retrieval is a property of the *catalog*, not of the algorithm. A
/// catalog of natural-language descriptions gives the dense arm most of the
/// signal, which is what the two benchmark corpora measure and where `0.7` was
/// chosen (ADR-0024). A catalog keyed on exact function names, error codes, or
/// internal jargon gives BM25 lexical purchase neither corpus has, and `0.7`
/// would be actively wrong for it. No single value can be right for both, which
/// is the same argument that made [`ClusterPolicy`] configurable.
///
/// Both endpoints are reachable and meaningful: `0.0` is pure lexical, `1.0`
/// pure dense. That is deliberate — a control that cannot reach its own limits
/// is a nudge, not a choice.
///
/// It does **not** scale the usage arm. That arm's share is ADR-0014's guard
/// against history overriding a live match, and folding it in here would let a
/// caller disable the guard while believing they were only retuning retrieval.
///
/// [`SearchMethod::Hybrid`]: crate::SearchMethod::Hybrid
/// [`ClusterPolicy`]: crate::ClusterPolicy
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DenseWeight(f32);

/// A [`DenseWeight`] was built from a value outside `[0, 1]`.
///
/// Rejected rather than clamped, for the reason [`ClusterPolicy`] rejects: a
/// clamp searches at a weighting the caller did not ask for and never says so.
///
/// [`ClusterPolicy`]: crate::ClusterPolicy
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidDenseWeight(pub f32);

impl std::fmt::Display for InvalidDenseWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dense weight {} is out of range (expected 0.0 to 1.0 inclusive)",
            self.0
        )
    }
}

impl std::error::Error for InvalidDenseWeight {}

impl DenseWeight {
    /// The weight the dense arm carries, in `[0, 1]`.
    ///
    /// # Errors
    /// [`InvalidDenseWeight`] if `weight` is outside `[0, 1]` or is NaN.
    pub fn new(weight: f32) -> Result<Self, InvalidDenseWeight> {
        if weight.is_nan() || !(0.0..=1.0).contains(&weight) {
            return Err(InvalidDenseWeight(weight));
        }
        Ok(Self(weight))
    }

    /// The weight as a plain float.
    #[must_use]
    pub fn get(self) -> f32 {
        self.0
    }
}

impl Default for DenseWeight {
    fn default() -> Self {
        Self(SCORE_FUSION_DENSE_WEIGHT)
    }
}

/// How much of the content score the usage arm may add on top.
///
/// Chosen to preserve the influence the arm has under RRF rather than to be
/// round. There, an arm at weight `w` contributes `w/(k+r)` against two content
/// arms contributing up to `2/k`, so at rank 0 it is worth `w/2` of the content
/// maximum. Score fusion normalises the content arms to a combined maximum of
/// `1.0`, so the same relative say is `w/2` — this constant is that `1/2`.
///
/// Getting this wrong is not neutral: too large and history overrides a live
/// lexical match, which is the failure ADR-0014's sub-unit arm weight exists to
/// prevent.
const USAGE_SHARE: f32 = 0.5;

/// Fuse the content arms on their **normalised scores** rather than their rank
/// positions, and add the usage arm as a bounded bonus.
///
/// ```text
/// score(id) = (1-w)·bm25(id)/ceiling + w·max(0, cos(id))
///           + USAGE_SHARE · arm_weight · 1/(1 + rank_in_arm)
/// ```
///
/// `w` is the caller's [`DenseWeight`], defaulting to
/// [`SCORE_FUSION_DENSE_WEIGHT`]. It splits the *content* arms only; the usage
/// arm keeps its own share whatever `w` is.
///
/// **Why this is possible now and was not before.** ADR-0011 rejected score
/// fusion because BM25 is unbounded and cosine is `[-1, 1]`, so the two cannot
/// be added without one silently dominating. Both arms now produce an absolute
/// `[0, 1]` value — BM25 against the query's own IDF ceiling, cosine clamped —
/// so they share a scale for the first time.
///
/// **What it buys.** RRF's magnitude is rank arithmetic: its maximum is "first
/// in every arm", which says nothing about whether the match is any good, so a
/// query the catalog answers well and one it cannot answer at all both return
/// ~1.0 at the top. A fused score has a real maximum, so its magnitude means
/// something and can be compared across queries.
///
/// **What it costs.** Rank fusion bounds each arm's mistake to one rank
/// position; score fusion lets a confidently wrong arm carry its confidence. An
/// id absent from an arm scores `0` there rather than merely being unranked.
///
/// Missing from an arm is `0` for that arm, not a dropped candidate: BM25
/// returns nothing when no query term matches, and scoring that as zero is the
/// honest reading.
pub(crate) fn score_fuse(
    bm25: &[(String, f32)],
    ceiling: f32,
    dense: &[(String, f32)],
    usage: Option<(&[String], f32)>,
    dense_weight: DenseWeight,
) -> Vec<(String, f32)> {
    use std::collections::HashMap;

    let w = dense_weight.get();
    let mut scores: HashMap<&str, f32> = HashMap::new();
    if ceiling > 0.0 {
        for (id, s) in bm25 {
            *scores.entry(id.as_str()).or_insert(0.0) += (1.0 - w) * (s / ceiling).clamp(0.0, 1.0);
        }
    }
    for (id, s) in dense {
        *scores.entry(id.as_str()).or_insert(0.0) += w * s.clamp(0.0, 1.0);
    }
    if let Some((ids, weight)) = usage {
        for (rank, id) in ids.iter().enumerate() {
            *scores.entry(id.as_str()).or_insert(0.0) += USAGE_SHARE * weight / (1.0 + rank as f32);
        }
    }
    let mut ranked: Vec<(String, f32)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect();
    let len = ranked.len();
    sort_and_truncate(&mut ranked, len);
    // Clamp AFTER ordering. The content arms cap at 1.0 by construction, so any
    // excess is the usage arm's bonus. Clamping first collapsed every saturated
    // candidate to exactly 1.0 and handed the order to the id tie-break — the
    // one case the arm exists to influence.
    for (_, score) in &mut ranked {
        *score = score.min(1.0);
    }
    ranked
}

/// Which scale a ranked list's scores are on, which decides both
/// a hit's `fused` flag and how its `normalized` value is derived. Shared by
/// [`crate::tool_registry`] and [`crate::skill_registry`], so the two cannot
/// drift into two spellings of one scoring rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Scale {
    /// Raw Okapi BM25, carrying the query's score ceiling to divide by.
    Bm25 { ceiling: f32 },
    /// Cosine of L2-normalized vectors — bounded, model-anchored.
    Cosine,
    /// Reciprocal Rank Fusion — rank arithmetic; magnitude is ordering only.
    Rrf,
    /// A score fusion of already-normalised arms — absolute in `[0, 1]`, so it
    /// needs no further mapping.
    Fused,
}

/// Map each score onto `[0, 1]` by the rule its scale admits.
///
/// Cosine is bounded, so `(s + 1) / 2` keeps the absolute level: a weak best
/// match stays low instead of being promoted to `1.0`.
///
/// Raw BM25 has no ceiling in general, but it does have one *per query* — the
/// score an average-length document containing every query term once would earn
/// (`query_ceiling`). Dividing by that gives the same absolute property, and
/// clamping handles the short-document-with-repeats case that can exceed it.
///
/// RRF has neither. Its magnitude is `Σ 1/(60 + rank)`, so the only reachable
/// maximum is "first in every arm", which says nothing about the match. Min-max
/// is what is left, and its cost is a top pinned at `1.0`. Callers must hand in
/// the **full** candidate set rather than the slice they intend to return, or
/// the bottom of that slice reads `0.0` however good it was.
/// A zero range (one hit, or every score tied) yields `1.0` throughout rather
/// than dividing by zero: nothing in that set is worse than anything else.
pub(crate) fn normalize(ranked: &[(String, f32)], scale: Scale) -> Vec<f32> {
    match scale {
        Scale::Cosine => return ranked.iter().map(|(_, s)| (s + 1.0) / 2.0).collect(),
        // Already `[0, 1]` and already absolute — the whole point of fusing on
        // scores rather than ranks.
        Scale::Fused => return ranked.iter().map(|(_, s)| s.clamp(0.0, 1.0)).collect(),
        // A ceiling of zero means no query term appears anywhere in the corpus,
        // so every score is zero too and there is nothing to divide.
        Scale::Bm25 { ceiling } if ceiling > 0.0 => {
            return ranked
                .iter()
                .map(|(_, s)| (s / ceiling).clamp(0.0, 1.0))
                .collect();
        }
        Scale::Bm25 { .. } => return vec![0.0; ranked.len()],
        Scale::Rrf => {}
    }
    let (Some(max), Some(min)) = (
        ranked.first().map(|(_, s)| *s),
        ranked.last().map(|(_, s)| *s),
    ) else {
        return Vec::new();
    };
    let range = max - min;
    if range <= 0.0 {
        return vec![1.0; ranked.len()];
    }
    ranked.iter().map(|(_, s)| (s - min) / range).collect()
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

    fn scored(v: &[(&str, f32)]) -> Vec<(String, f32)> {
        v.iter().map(|(id, s)| ((*id).to_string(), *s)).collect()
    }

    #[test]
    fn the_usage_arm_orders_candidates_that_both_saturate() {
        // The content arms cap at 1.0 by construction, so anything above that is
        // the usage arm's bonus. Two ids equal on content and both in the usage
        // arm: `zeta` at rank 0 (raw 1.0 + 0.5 = 1.50), `alpha` at rank 1
        // (raw 1.0 + 0.25 = 1.25). Clamping before the sort collapsed both to
        // exactly 1.0 and handed the order to the id tie-break, which put
        // `alpha` first — defeating the arm in the one case it exists for.
        let bm25 = scored(&[("zeta", 1.0), ("alpha", 1.0)]);
        let dense = scored(&[("zeta", 1.0), ("alpha", 1.0)]);
        let usage = ids(&["zeta", "alpha"]);
        let fused = score_fuse(
            &bm25,
            1.0,
            &dense,
            Some((usage.as_slice(), 1.0)),
            DenseWeight::default(),
        );
        let order: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            order,
            vec!["zeta", "alpha"],
            "a stronger usage rank must lead, not the alphabetically earlier id"
        );
    }

    #[test]
    fn a_saturated_score_is_still_reported_within_the_unit_range() {
        // The companion to the test above: ordering on the raw score must not
        // become "stop clamping". Everything returned is still in [0, 1], so a
        // caller reading `score` sees the same range it always has.
        let bm25 = scored(&[("zeta", 1.0), ("alpha", 1.0)]);
        let dense = scored(&[("zeta", 1.0), ("alpha", 1.0)]);
        let usage = ids(&["zeta", "alpha"]);
        let fused = score_fuse(
            &bm25,
            1.0,
            &dense,
            Some((usage.as_slice(), 1.0)),
            DenseWeight::default(),
        );
        assert!(
            fused.iter().all(|(_, s)| *s <= 1.0),
            "fused scores must stay within [0, 1]: {fused:?}"
        );
    }

    #[test]
    fn the_default_dense_weight_is_the_documented_constant() {
        assert_eq!(DenseWeight::default().get(), SCORE_FUSION_DENSE_WEIGHT);
        assert_eq!(DenseWeight::default().get(), 0.7);
    }

    #[test]
    fn a_dense_weight_outside_the_unit_range_is_rejected_not_clamped() {
        // Rejected, so a caller who fat-fingers 70 for 0.7 is told, rather than
        // silently searching at 1.0 and drawing conclusions from it.
        for bad in [-0.1_f32, 1.1, 70.0, f32::NAN, f32::INFINITY] {
            assert!(DenseWeight::new(bad).is_err(), "{bad} should be rejected");
        }
        // Both endpoints are legal: pure lexical and pure dense are real choices.
        for good in [0.0_f32, 0.3, 0.7, 1.0] {
            assert_eq!(DenseWeight::new(good).unwrap().get(), good);
        }
    }

    #[test]
    fn the_weight_splits_the_content_arms() {
        // One id in each arm at full strength. At w the dense-only id scores w and
        // the lexical-only id scores 1-w, so the weight IS the split, exactly.
        let bm25 = scored(&[("lex", 1.0)]);
        let dense = scored(&[("sem", 1.0)]);
        for w in [0.0_f32, 0.3, 0.5, 0.7, 1.0] {
            let fused = score_fuse(&bm25, 1.0, &dense, None, DenseWeight::new(w).unwrap());
            let by: std::collections::HashMap<&str, f32> =
                fused.iter().map(|(id, s)| (id.as_str(), *s)).collect();
            assert!((by["sem"] - w).abs() < 1e-6, "dense at w={w}");
            assert!((by["lex"] - (1.0 - w)).abs() < 1e-6, "bm25 at w={w}");
        }
    }

    #[test]
    fn the_weight_can_flip_which_arm_wins() {
        // The point of the knob: the same two candidates order differently at
        // 0.3 and 0.7. A knob that cannot change the answer is not a knob.
        let bm25 = scored(&[("lex", 1.0)]);
        let dense = scored(&[("sem", 1.0)]);
        let lexical_heavy = score_fuse(&bm25, 1.0, &dense, None, DenseWeight::new(0.3).unwrap());
        let dense_heavy = score_fuse(&bm25, 1.0, &dense, None, DenseWeight::new(0.7).unwrap());
        assert_eq!(
            lexical_heavy.first().map(|(id, _)| id.as_str()),
            Some("lex")
        );
        assert_eq!(dense_heavy.first().map(|(id, _)| id.as_str()), Some("sem"));
    }

    #[test]
    fn the_endpoints_mute_an_arm_without_dropping_its_candidates() {
        // At w=1.0 BM25 contributes no score, but a BM25-only id is still a
        // candidate at 0.0 — the same "muted, not omitted" rule weighted RRF has.
        let bm25 = scored(&[("lex", 1.0)]);
        let dense = scored(&[("sem", 1.0)]);
        let pure_dense = score_fuse(&bm25, 1.0, &dense, None, DenseWeight::new(1.0).unwrap());
        assert_eq!(pure_dense.len(), 2);
        assert_eq!(pure_dense.last(), Some(&("lex".to_string(), 0.0)));

        let pure_lexical = score_fuse(&bm25, 1.0, &dense, None, DenseWeight::new(0.0).unwrap());
        assert_eq!(pure_lexical.len(), 2);
        assert_eq!(pure_lexical.last(), Some(&("sem".to_string(), 0.0)));
    }

    #[test]
    fn the_weight_does_not_scale_the_usage_arm() {
        // ADR-0014's guard is not part of the retrieval split. The usage arm
        // contributes the same at every content weighting, so a caller cannot
        // disable it by retuning dense-vs-lexical.
        let usage = ids(&["hist"]);
        let mut contributions = vec![];
        for w in [0.0_f32, 0.5, 1.0] {
            let fused = score_fuse(
                &[],
                1.0,
                &[],
                Some((usage.as_slice(), 1.0)),
                DenseWeight::new(w).unwrap(),
            );
            contributions.push(fused[0].1);
        }
        assert!(
            contributions.windows(2).all(|p| (p[0] - p[1]).abs() < 1e-6),
            "usage contribution moved with the content weight: {contributions:?}"
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
