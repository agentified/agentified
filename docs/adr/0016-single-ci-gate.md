# 16. Single required `ci-gate`, core-maintainers bypass

Date: 2026-08-07

## Status

Accepted

## Context

Pre-merge enforcement on `ratel-ai/ratel` was documentation-only: `RELEASING.md` described a
required `pr-gate` check with an `rstagi` bypass, but `gh api repos/ratel-ai/ratel/rulesets`
returned `[]` and branch protection 404'd — nothing was actually provisioned, and the setup
script the docs referenced (`scripts/setup-branch-ruleset.sh`) did not exist.

The checks themselves were fragmented. The everyday `rust.yml`, `ts.yml`, `python.yml`, and
`protocol.yml` were independent workflows, and `pr-gate.yml`'s terminal job aggregated only
its own release-verify matrix. GitHub `needs:` cannot cross workflow files, so no single
status context reflected the whole pipeline — you cannot require "all of them" with one check
while they live in separate files. Arming was via a `ready-to-merge` label, and the intended
override was scoped to one named person.

Our sibling repo `ratel-ai/ratel-cloud` had already converged on a cleaner shape (its ruleset
requires exactly one `ci-gate` context), and we want parity so contributors moving between the
repos meet the same gate.

## Decision

**One pre-merge workflow, one required check.** All pre-merge jobs live in
`.github/workflows/ci.yml`; a terminal `ci-gate` job `needs:` every leg (the everyday `rust` /
`ts` / `ai-sdk-compat` / `python` / `telemetry` / `protocol` legs **and** the heavy
release-verify matrix). `ci-gate` is the **sole** required status check on `main`. It runs
`if: always()`, so a *skipped* upstream job passes and only `failure`/`cancelled` fails the
gate — giving the ruleset a definitive answer even when skip-green elides untouched legs.

Per-leg contexts are **not** registered as required directly: a skipped matrix job collapses
to a single check under the bare job name and never emits its per-leg contexts (`verify (…)`,
`ai-sdk compat (…)`), which would sit "Expected" forever and block merge. The rule is: add a
new pre-merge job to `ci-gate`'s `needs:`, never to the ruleset.

**Ready-not-draft arming.** The `changes`, `setup`, and `ci-gate` jobs carry a non-draft
guard, so draft PRs run nothing and post no `ci-gate`. Drafts are unmergeable, so a missing
check cannot deadlock a merge; marking a PR ready fires `ready_for_review` and posts the check.
This replaces the `ready-to-merge` label.

**Team bypass.** The single ruleset bypass actor is the `core-maintainers` org team
(`bypass_mode: always`), not named users — membership is self-maintaining. Provisioned by the
idempotent `scripts/setup-branch-ruleset.sh` (which resolves the team id at runtime and
PUT/POSTs the ruleset), mirroring ratel-cloud: `deletion`, `non_fast_forward`,
`required_linear_history`, PR review (>=1 approval + thread resolution), and exactly `ci-gate`
as the strict required check.

## Consequences

- A single green `ci-gate` means the whole pre-merge pipeline is green; branch-protection
  config no longer drifts from what CI actually runs.
- Skip-green filters must stay conservative for this FFI monorepo (ADR-0006): the TS and Python
  SDK legs include the Rust core + `Cargo.*` paths so an FFI change re-runs them, and every
  filter includes `ci.yml` so editing CI re-runs all legs. On push to `main` and
  `workflow_dispatch` every leg runs regardless of filters.
- Enforcement is a checked-in script, not click-ops. Applying it is a privileged manual step
  (create the `core-maintainers` team, run the script, delete the obsolete `ready-to-merge`
  label) done by an admin after this lands.
- `docs/adr/0008-release-engineering.md` (tag-time publishing) is unaffected and stays as is.

## Rejected

- **Requiring the per-leg checks directly.** Skipped matrix legs never emit their contexts, so
  the ruleset would wait on checks that never arrive. The aggregator job is the standard fix.
- **Named-user bypass (`rstagi`).** Correct for a one-person escape hatch, but it rots as the
  maintainer set changes; a team actor tracks membership without ruleset edits.
- **Label arming.** A `ready-to-merge` label is extra state to add, remove, and police in the
  ruleset; PR draft status already encodes "not ready" and can't be forgotten-on.
