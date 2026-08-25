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

/// How loudly each content arm votes, as a function of **its own** top score.
///
/// RRF fuses on rank position alone, so an arm's best hit contributes `1/(k+0)`
/// whether that hit was a near-perfect match or the least-bad of a bad set. An
/// arm with nothing useful to say still votes at full strength. This is the dial
/// that lets it go quiet instead.
///
/// **No score ever crosses an arm boundary.** Each arm's score is compared only
/// against its own thresholds, never against the other arm's, so this is not the
/// cross-arm score normalization ADR-0011 rejected — that would require putting
/// unbounded BM25 and bounded cosine on one scale, which nothing here does.
///
/// **Only the dense arm is gated, and that asymmetry is deliberate.** Cosine is
/// bounded by the geometry of a fixed model, so `0.9` and `0.3` mean the same
/// thing on any corpus and a threshold transfers. Raw BM25 is unbounded and
/// corpus-dependent — the same `8.3` is excellent on one catalog and mediocre on
/// another — so no floor written here would survive contact with a different
/// corpus. The usage arm is already score-weighted by its own support and
/// recency ramp (ADR-0014).
///
/// [`Default`] is **disabled**: every arm at `1.0`, byte-identical to plain RRF.
///
/// `#[non_exhaustive]` so a later dimension is additive, which is also why the
/// builders exist — the struct cannot be written as a literal outside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct FusionPolicy {
    /// `(floor, full)` cosines, or `None` for a flat weight of `1.0`.
    dense_confidence: Option<(f32, f32)>,
}

impl FusionPolicy {
    /// Scale the dense arm between silent at `floor` and full weight at `full`.
    ///
    /// Both are cosines against the query. Sensible values depend on the
    /// embedding model: bge-small packs related text into roughly `[0.6, 0.9]`,
    /// so a floor tuned for it means nothing on a model with a wider spread.
    /// That model-dependence is why this is configuration rather than a constant.
    #[must_use]
    pub fn with_dense_confidence(mut self, floor: f32, full: f32) -> Self {
        self.dense_confidence = Some((floor, full));
        self
    }

    /// Return the dense arm to a flat weight of `1.0`.
    #[must_use]
    pub fn without_dense_confidence(mut self) -> Self {
        self.dense_confidence = None;
        self
    }

    /// The configured `(floor, full)`, if any.
    #[must_use]
    pub fn dense_confidence(&self) -> Option<(f32, f32)> {
        self.dense_confidence
    }

    /// Whether this is the built-in policy — every arm flat at `1.0`.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether the ramp is orderable and inside the range a cosine lives in.
    ///
    /// `floor == full` is rejected rather than treated as a step: it makes the
    /// ramp's denominator zero, and a caller who wants a hard cutoff can say so
    /// with two values a hair apart.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.dense_confidence.is_none_or(|(floor, full)| {
            (0.0..=1.0).contains(&floor) && (0.0..=1.0).contains(&full) && floor < full
        })
    }

    /// The dense arm's fusion weight given its own top cosine.
    ///
    /// `1.0` when disabled or when the arm returned nothing — an empty arm
    /// contributes no ids either way, so damping it would only be misleading in
    /// a trace. Exactly `0.0` means **omit the arm**, which callers must honour:
    /// [`rrf_fuse_weighted`] still admits a zero-weight arm's ids to the
    /// candidate set at score `0.0`, where they would fill top-k in alphabetical
    /// order.
    #[must_use]
    pub(crate) fn dense_weight(&self, top_cos: Option<f32>) -> f32 {
        match (self.dense_confidence, top_cos) {
            (Some((floor, full)), Some(cos)) => ((cos - floor) / (full - floor)).clamp(0.0, 1.0),
            _ => 1.0,
        }
    }
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

    // ---- the fusion policy ----

    #[test]
    fn the_default_policy_leaves_every_arm_at_full_weight() {
        let p = FusionPolicy::default();
        assert!(p.is_default());
        assert_eq!(p.dense_weight(Some(0.05)), 1.0);
        assert_eq!(p.dense_weight(Some(0.95)), 1.0);
        assert_eq!(p.dense_weight(None), 1.0);
    }

    #[test]
    fn a_cosine_at_or_below_the_floor_silences_the_dense_arm() {
        let p = FusionPolicy::default().with_dense_confidence(0.6, 0.9);
        assert_eq!(p.dense_weight(Some(0.6)), 0.0);
        assert_eq!(p.dense_weight(Some(0.2)), 0.0);
    }

    #[test]
    fn a_cosine_at_or_above_full_carries_the_weight_it_carries_today() {
        let p = FusionPolicy::default().with_dense_confidence(0.6, 0.9);
        assert_eq!(p.dense_weight(Some(0.9)), 1.0);
        assert_eq!(p.dense_weight(Some(1.0)), 1.0);
    }

    #[test]
    fn the_ramp_is_linear_between_the_floor_and_full() {
        let p = FusionPolicy::default().with_dense_confidence(0.6, 0.8);
        assert!((p.dense_weight(Some(0.70)) - 0.5).abs() < 1e-6);
        assert!((p.dense_weight(Some(0.65)) - 0.25).abs() < 1e-6);
    }

    /// An empty arm is not a *quiet* arm. It contributes no ids at any weight, so
    /// damping it would put a number in a trace that explains nothing.
    #[test]
    fn an_arm_that_returned_nothing_is_not_damped() {
        let p = FusionPolicy::default().with_dense_confidence(0.6, 0.9);
        assert_eq!(p.dense_weight(None), 1.0);
    }

    /// The reason `dense_weight` returning `0.0` must mean *omit*, not *pass at
    /// zero*: at zero the arm's ids still enter the candidate set, all tied, and
    /// `sort_and_truncate`'s alphabetical tie-break then fills top-k with them.
    #[test]
    fn a_zero_weight_arm_still_places_its_ids_so_it_must_be_omitted() {
        let strong = vec!["real_hit".to_string()];
        let junk = vec!["aaa_junk".to_string(), "bbb_junk".to_string()];

        let passed_at_zero = rrf_fuse_weighted(&[(&strong, 1.0), (&junk, 0.0)], RRF_K);
        assert_eq!(
            passed_at_zero.len(),
            3,
            "a zero-weight arm still admits ids"
        );
        assert!(
            passed_at_zero.iter().any(|(id, _)| id == "aaa_junk"),
            "so omitting it is a different result, not a cosmetic one"
        );

        let omitted = rrf_fuse_weighted(&[(&strong, 1.0)], RRF_K);
        assert_eq!(omitted.len(), 1);
    }

    #[test]
    fn a_ramp_outside_the_cosine_range_or_out_of_order_is_invalid() {
        assert!(FusionPolicy::default().is_valid());
        assert!(
            FusionPolicy::default()
                .with_dense_confidence(0.6, 0.9)
                .is_valid()
        );
        assert!(
            !FusionPolicy::default()
                .with_dense_confidence(0.9, 0.6)
                .is_valid()
        );
        assert!(
            !FusionPolicy::default()
                .with_dense_confidence(0.7, 0.7)
                .is_valid()
        );
        assert!(
            !FusionPolicy::default()
                .with_dense_confidence(-0.1, 0.9)
                .is_valid()
        );
        assert!(
            !FusionPolicy::default()
                .with_dense_confidence(0.6, 1.4)
                .is_valid()
        );
    }

    /// Names chosen so the alphabetical tie-break favours the dense pick: at
    /// equal weight `alpha_semantic` wins the tie outright, and only the damping
    /// can move it. Two single-hit arms rather than a shared id, because RRF's
    /// gentle curve (`1/60` vs `1/61`) means an id both arms rank almost always
    /// survives damping — which is the point of `k`, not a bug.
    #[test]
    fn a_damped_arm_loses_the_tie_it_would_otherwise_win() {
        let bm25 = vec!["zeta_lexical".to_string()];
        let dense = vec!["alpha_semantic".to_string()];

        let flat = rrf_fuse_weighted(&[(&bm25, 1.0), (&dense, 1.0)], RRF_K);
        assert_eq!(
            flat.first().map(|(id, _)| id.as_str()),
            Some("alpha_semantic"),
            "equal weights leave nothing but the tie-break to separate them"
        );

        let damped = rrf_fuse_weighted(&[(&bm25, 1.0), (&dense, 0.5)], RRF_K);
        assert_eq!(
            damped.first().map(|(id, _)| id.as_str()),
            Some("zeta_lexical"),
            "a dense arm at half confidence should not still be setting top-1"
        );
    }
}
