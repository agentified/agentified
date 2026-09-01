# 24. Hybrid fuses on normalised scores, not rank positions

Date: 2026-09-01

## Status

Accepted.

Supersedes the **fusion rule** in [ADR-0011](0011-selectable-retrieval-methods.md), and reverses
its rejection of *"score-normalization fusion for hybrid — RRF needs no per-arm score
calibration."* The rest of ADR-0011 — three selectable methods, the arms themselves, retrieval
depth, fallibility confined to the semantic path — stands.

**Measured on `ratel-bench` before acceptance.** The in-repo fixture said this rule was *worse*
— top-1 against the invoked tool falling from 35 of 47 to 32 — so it shipped in `0.12.0-rc.2`
unaccepted, to be judged on a corpus where BM25 has lexical purchase the fixture does not give
it. BFCL disagreed with the fixture, and the bench is the instrument that decides. See
**Measurement** below.

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

## Measurement

### BFCL — isolates this rule

100 scenarios shared across all four runs, pool size 100, `retrieval-0.12.0-rc.{1,2}-*`.
rc.1 is RRF; rc.2 is this rule. The `Bm25` and `Semantic` arms are unchanged between the two
RCs, so rc.1's numbers for them stand.

| arm | recall@1 | MRR@5 | nDCG@5 | mean gold score |
|---|---|---|---|---|
| `Bm25` | 0.8800 | 0.9242 | 0.9408 | 26.08 |
| `Semantic` | 0.8800 | **0.9312** | **0.9486** | 0.7502 |
| `Hybrid`, RRF (rc.1) | 0.8300 | 0.8953 | 0.9193 | 0.0332 |
| `Hybrid`, fused (rc.2) | **0.8500** | 0.9075 | 0.9285 | 0.7151 |

Score fusion beats RRF on every metric at every `k`, and closes roughly a third of hybrid's gap
to dense. Per-scenario at `k = 5`: 6 improved, 2 regressed, 92 unchanged — a small, one-sided
move, not noise in both directions. Top-1 gained 3 and lost 1.

Two things this does **not** show. Hybrid still trails pure dense on BFCL, which is the fixture's
verdict surviving contact with a real corpus: the weight already sits at 0.7 dense and the sweep
wanted more. And BFCL is a *tool* corpus; for skills see SR-Agents below.

The gold-score column is the readability claim, not a quality one. RRF returned `0.0332` for the
correct answer whatever the query — two arms times `1/(60+1)`, an artifact of `RRF_K`. Fusion
returns `0.7151`, on the same scale as the dense arm it mostly reflects.

### SR-Agents — the skill path, but does not isolate this rule

100 scenarios, pool size 100, hybrid. This is the **skill** corpus, so it answers the "unmeasured
on skills" caveat above. It is a `0.4.0` → `rc.2` comparison, not `rc.1` → `rc.2`.

| k | metric | 0.4.0 | rc.2 | Δ |
|---|---|---|---|---|
| 1 | hit@1 / MRR | 0.9700 | 0.9700 | 0 |
| 3 | recall@3 | 0.7963 | 0.8513 | +0.0550 |
| 3 | nDCG@3 | 0.8588 | 0.9056 | +0.0468 |
| 5 | recall@5 | 0.8777 | 0.9593 | **+0.0816** |
| 5 | nDCG@5 | 0.8836 | 0.9435 | +0.0599 |
| 5 | hit@5 | 0.9900 | 1.0000 | +0.0100 |

Larger than BFCL's gain, and on skills. **It is not attributable to this rule alone.** `0.4.0`
predates the whole branch, so the delta also contains `BM25_B` 0.4 → 0.75
([ADR-0023](0023-searchable-text-indexes-names-not-schema-prose.md), still unsettled) and the
skill-side normalize-before-truncate fix. Adaptive ranking is *not* in it — `sragents-candidates`
builds no intent graph — so the usage arm, ICF, and impressions are all inert here.

The shape is consistent with score fusion: hit@1 is flat and every gain is in the tail, which is
what replacing RRF's near-ties with real scores should do. But `b` also only reorders the tail,
so the shape is a hint, not an attribution. An `rc.1` hybrid run on the same slice would split
the two, and would be the first real-corpus evidence on `b`.

## Decision

```text
score(id) = (1-w) · clamp(bm25(id) / Σ idf(query terms), 0, 1)
          +    w  · clamp(cos(id), 0, 1)
          + USAGE_SHARE · arm_weight · 1/(1 + rank_in_arm)
```

- **`w` defaults to 0.7 and is a per-catalog setting.** The lexical arm is the weaker of the two
  wherever the two have been compared — decisively on the fixture, where BM25 alone recovers the
  invoked tool 12 times in 47 against dense's 23, and narrowly on BFCL, where the two tie on
  recall@1 and dense leads on MRR. An even split also lets a query with no lexical purchase halve
  every candidate uniformly, since a tool BM25 never returned scores `0` on that arm.

  It is a **default**, not a constant, because both corpora that produced it are natural-language
  queries against descriptive metadata — one catalog shape out of many. `DenseWeight` in the core,
  `experimentalDenseWeight` / `experimental_dense_weight` on the SDK catalogs. `0` is pure
  lexical and `1` pure dense: both endpoints reachable, because a control that cannot reach its
  own limits is a nudge rather than a choice. Out of range is rejected, not clamped — a clamp
  searches at a weighting the caller did not ask for and never says so. See the reversal under
  **Rejected**.
- **Absent from an arm is `0`, not a dropped candidate.** BM25 returns nothing when no query term
  matches; scoring that as zero is the honest reading, and it is what makes the total comparable
  across queries.
- **The usage arm keeps the share it already had.** It carries no per-id score, only an order, so
  its rank is mapped into the same band and scaled by the weight it already computes. Under RRF
  an arm at weight `w` contributes `w/(k+r)` against content arms reaching `2/k`, so at rank 0 it
  is worth `w/2` of the content maximum; `USAGE_SHARE = 0.5` reproduces that. Getting this wrong
  is not neutral — too large and history overrides a live lexical match, the failure ADR-0014's
  sub-unit arm weight exists to prevent.
- **The total is bounded to `[0, 1]` after ordering, never before.** Content caps at `1.0` by
  construction, so a candidate the usage arm also promotes can exceed it. The excess is
  discarded for display, but the **ordering is decided on the unbounded value** — clamping first
  collapses every saturated candidate to exactly `1.0` and hands the order to the id tie-break,
  which silently defeats the usage arm in the one case it exists for. That was the behaviour
  shipped in `0.12.0-rc.2`; see the Fixed entry in the core changelog.

  The consequence worth knowing: two hits can display an identical `1.00` while correctly
  ordered. `rank` is the ordering contract, not `normalized`. Giving the usage arm the remaining
  headroom instead (`content + (1 - content)·bonus`) would remove the saturation entirely, but it
  reweights the arm against content everywhere, so it needs its own measurement rather than
  riding in as a bug fix.

- **`SearchHit::normalized` is the fused score itself.** It is already absolute in `[0, 1]`, so
  hybrid stops min-maxing against the returned slate — which pinned the top to `1.0` and the
  bottom to `0.0` whatever they matched.
- **Only hybrid changes.** `Bm25` and `Semantic` still fuse the usage arm by rank, because a
  single content arm has nothing to be normalised against.

## Consequences

- **Breaking.** Hybrid's served order changes for every caller. On the fixture 11 of 47 queries
  get a different top-1 and 35 of 47 a different top-5; on BFCL 8 of 100 scenarios reorder inside
  the top 5. The blast radius is real even where the net effect is positive.
- **The magnitude becomes usable.** `0.92` for a query the catalog answers, `0.27` for one it
  does not, where RRF gave `1.000` for both. On BFCL the correct answer scores `0.7151` on
  average against RRF's flat `0.0332`. This is the first hybrid number that could be calibrated
  against outcomes at all — it is not yet *calibrated*, only meaningful.
- **An arm's mistake is no longer bounded.** Rank fusion caps a wrong arm at one rank position;
  score fusion lets it carry its confidence. On BFCL that cuts both ways and nets positive — 6
  scenarios improved against 2 regressed — but the downside is intrinsic to the choice, not a
  tuning problem, and a corpus with a noisier lexical arm could land on the other side of it.
- **The fixture was wrong about this, and was retired as an evaluator because of it.** 47
  semantic queries over 26 tools cannot move a served result when the usage arm enters fusion at
  sub-unit weight against two full-weight arms. It stays as a regression guard — it catches a
  rule that changed unintentionally — but ranking decisions are made on `ratel-bench`.
- **The weight is not settled.** Both the fixture sweep and BFCL want more dense than 0.7; the
  fixture sweep climbs monotonically to dense-only. 0.7 is defensible as the point where the
  lexical arm still contributes, not as an optimum. Revisit it with a corpus where BM25 wins
  outright.

## Rejected

- **Keeping RRF and exposing the fused value as a display-only score.** It removes all ranking
  risk and was the recommendation before the bench ran. It was rejected for the RC because
  `ratel-bench`'s retrieval metrics are rank-based and therefore blind to a score that orders
  nothing — shipping it display-only would have made the question unmeasurable on the one corpus
  set that could answer it. It is now rejected outright: the bench answered, and the ranking it
  produces is the better half of the change.
- **`max(nb, nd)` instead of a weighted sum**, so an absent arm cannot drag a candidate down. It
  discards the agreement signal — two arms both liking a tool should beat one — and was not
  measured.
- ~~**Making the split configurable.**~~ **Reversed.** The original argument — a knob shifts the
  decision to the caller without evidence for any setting, as the reverted dense-confidence gate
  did — mistook *which* decision was being shifted. The evidence gap is not "what should the
  number be" but "what should it be **for this catalog**", and that is not a question we can
  answer centrally at all. BFCL and SR-Agents are both natural-language queries against
  descriptive metadata; a catalog keyed on exact identifiers, error codes, or internal jargon
  gives BM25 purchase neither corpus has. That is the same argument that made `ClusterPolicy`
  configurable, and it applies here for the same reason. Shipped as
  `experimentalDenseWeight` / `experimental_dense_weight`, defaulting to `0.7` so an
  untouched catalog is byte-identical.

  The dense-confidence gate is not a counterexample: it was reverted because it changed *nothing*
  at any setting, not because it was configurable.
