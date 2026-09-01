# 24. Hybrid fuses on normalised scores, not rank positions

Date: 2026-09-01

## Status

Accepted — **provisionally, for an RC**.

Supersedes the **fusion rule** in [ADR-0011](0011-selectable-retrieval-methods.md), and reverses
its rejection of *"score-normalization fusion for hybrid — RRF needs no per-arm score
calibration."* The rest of ADR-0011 — three selectable methods, the arms themselves, retrieval
depth, fallibility confined to the semantic path — stands.

**Shipped to be measured, not because it is measured.** On the in-repo fixture this rule is
*worse*: top-1 against the invoked tool falls from 35 of 47 to 32. It ships in an RC so
`ratel-bench` can judge it on corpora where BM25 has lexical purchase the fixture does not give
it. If it does not hold up there, revert to RRF and keep the fused value as a display-only
number.

## Context

RRF fuses on rank position: `score(id) = Σ_arms w · 1/(k + rank)`. Two consequences.

The rejection in ADR-0011 was a scale argument, and it was correct at the time: BM25 is unbounded
and corpus-dependent, cosine is `[-1, 1]`, and adding them lets one silently dominate. **That
argument no longer applies.** Both arms now carry an absolute `[0, 1]` value — BM25 divided by
`Σ idf(query terms)`, the score an average-length document containing every query term once would
earn; cosine clamped at zero. Neither is normalised against the *slate*, so neither moves when
another candidate is added.

The second consequence is the one that motivated this. `RRF_K = 60` makes consecutive ranks
1.6% apart, and the maximum reachable score is "first in every arm" — which says nothing about
whether the match is good. A query the catalog answers precisely and a query it cannot answer at
all both return ~1.0 at the top. Kestral's integration displayed that number as a confidence and
read certainty into rank arithmetic; the SDK's own docs already said the magnitude was
meaningless. A fusion whose magnitude cannot be displayed is a fusion that will be displayed
wrongly.

## Decision

```text
score(id) = 0.3 · clamp(bm25(id) / Σ idf(query terms), 0, 1)
          + 0.7 · clamp(cos(id), 0, 1)
          + USAGE_SHARE · arm_weight · 1/(1 + rank_in_arm)
```

- **0.7 to dense.** The lexical arm is the weaker of the two on every corpus measured so far — on
  the fixture BM25 alone recovers the invoked tool 12 times in 47 against dense's 23. An even
  split also lets a query with no lexical purchase halve every candidate uniformly, since a tool
  BM25 never returned scores `0` on that arm.
- **Absent from an arm is `0`, not a dropped candidate.** BM25 returns nothing when no query term
  matches; scoring that as zero is the honest reading, and it is what makes the total comparable
  across queries.
- **The usage arm keeps the share it already had.** It carries no per-id score, only an order, so
  its rank is mapped into the same band and scaled by the weight it already computes. Under RRF
  an arm at weight `w` contributes `w/(k+r)` against content arms reaching `2/k`, so at rank 0 it
  is worth `w/2` of the content maximum; `USAGE_SHARE = 0.5` reproduces that. Getting this wrong
  is not neutral — too large and history overrides a live lexical match, the failure ADR-0014's
  sub-unit arm weight exists to prevent.
- **`SearchHit::normalized` is the fused score itself.** It is already absolute in `[0, 1]`, so
  hybrid stops min-maxing against the returned slate — which pinned the top to `1.0` and the
  bottom to `0.0` whatever they matched.
- **Only hybrid changes.** `Bm25` and `Semantic` still fuse the usage arm by rank, because a
  single content arm has nothing to be normalised against.

## Consequences

- **Breaking.** Hybrid's served order changes for every caller. On the fixture 11 of 47 queries
  get a different top-1 and 35 of 47 a different top-5.
- **The magnitude becomes usable.** `0.92` for a query the catalog answers, `0.27` for one it
  does not, where RRF gave `1.000` for both. This is the first hybrid number that could be
  calibrated against outcomes at all.
- **An arm's mistake is no longer bounded.** Rank fusion caps a wrong arm at one rank position;
  score fusion lets it carry its confidence. That is the mechanism behind the measured drop, and
  it is intrinsic to the choice rather than a tuning problem.
- The weight sweep climbs monotonically to dense-only (34 of 47), so on this fixture the answer
  to "how much BM25" is "none". Read that as evidence about *this corpus* — 47 semantic queries
  over 26 tools — not as a verdict on lexical retrieval.

## Rejected

- **Keeping RRF and exposing the fused value as a display-only score.** It removes all ranking
  risk and was the recommendation before this decision. Rejected *for the RC only*, because
  `ratel-bench`'s retrieval metrics are rank-based and therefore blind to a score that does not
  order anything: shipping it display-only would make the question unmeasurable on the one corpus
  set that can answer it. This is the fallback if the bench disagrees.
- **`max(nb, nd)` instead of a weighted sum**, so an absent arm cannot drag a candidate down. It
  discards the agreement signal — two arms both liking a tool should beat one — and was not
  measured.
- **Making the split configurable.** A knob shifts the decision to the caller without evidence
  for any setting, and the equivalent knob for the dense-confidence gate was reverted for exactly
  that reason.
