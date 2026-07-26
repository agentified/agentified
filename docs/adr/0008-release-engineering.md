# 8. Release engineering: per-unit versions, one routed workflow, RC-first OIDC

Date: 2026-07-05

## Status

Accepted

Compacted 2026-07 from pre-compaction ADR-0008 (per-package CHANGELOGs, 2026-05-07),
ADR-0016 (per-package versions and releases, 2026-07-04), and ADR-0018 (defer the CLI,
2026-07-04).

Amended 2026-07-26 to drop the `telemetry-ts-otlp` unit with the `@ratel-ai/telemetry-otlp`
package, and to record the `mastra` adapter unit.

## Context

The repo once released everything in lockstep on one `v*` tag: one bump shipped every
artifact, forcing no-op releases (a CLI-only fix riding a dead SDK bump). The product split
put independently-evolving units in one repo; lockstep became untenable. Two hard constraints
shape the design: publish trust is OIDC **Trusted Publishers** (each registration binds the
owning repo, the workflow **filename**, and the `release` GitHub **environment**; renaming
either de-registers everything), and the environment's tag policy must agree with the
workflow trigger.

## Decision

### Per-unit versions and tags

Each release unit carries its own version and ships on its own tag prefix,
`<prefix>-vX.Y.Z` (GA) or `<prefix>-vX.Y.Z-rc.N` (RC). The units are registered once in
`scripts/release-units.mjs`, the single source of truth the tag gate, changelog drafter, and
publish helpers read:

| Unit | Tag prefix | Registry |
|---|---|---|
| `core` | `core-v*` | `ratel-ai-core` on crates.io |
| `sdk-ts` | `sdk-ts-v*` | `@ratel-ai/sdk` + 5 platform packages on npm |
| `sdk-py` | `sdk-py-v*` | `ratel-ai` on PyPI |
| `telemetry-core` | `telemetry-core-v*` | `ratel-ai-telemetry` on crates.io |
| `telemetry-ts` | `telemetry-ts-v*` | `@ratel-ai/telemetry` on npm |
| `telemetry-py` | `telemetry-py-v*` | `ratel-ai-telemetry` on PyPI |
| `vercel-ai-sdk` | `vercel-ai-sdk-v*` | `@ratel-ai/vercel-ai-sdk` on npm |
| `mastra` | `mastra-v*` | `@ratel-ai/mastra` on npm |

The `sdk-ts` unit is **internally lockstep**: the loader, its five per-OS native packages,
and the `ts-native` crate must share a version, because the loader's `optionalDependencies`
pin each platform package exactly and a mismatch is a silent ABI break. The inject script
throws on divergence before publish. This is the only lockstep left, and it is a correctness
invariant, not a coupling convenience. Versions diverge across units by design; cross-unit
compatibility is expressed by dependency ranges, not a shared semver.

The `vercel-ai-sdk` framework adapter is an independent, pure-TypeScript unit. It peers on
`@ratel-ai/sdk` and takes a runtime dependency on `@ratel-ai/telemetry`; its release therefore
changes only the adapter version, while both workspace specifiers are replaced with the
compatible published ranges — which orders `telemetry-ts` and `sdk-ts` ahead of it. That
ordering is load-bearing rather than tidy: npm installs peers automatically, so a range naming
a version the registry does not have is an `ETARGET` at install time and not a warning — for
the runtime dependency and the peer alike. npm versions are immutable, so publishing out of
order ships a permanently uninstallable release.
It publishes over OIDC from `release.yml`'s `publish-vercel-ai-sdk` job. That job publishes the
package directory, not a `pnpm pack` tarball, so the substitution is explicit pin steps in the
job rather than something the pack step does for free. Only the first publish went through
`scripts/publish-rc.sh --unit vercel-ai-sdk`, to bootstrap the npm name before a Trusted
Publisher could exist.

### One `release.yml`, routed by prefix

For OIDC-wired units, a single `release.yml` fires on the prefix set and routes the tag to its
unit: only that unit's manifests and CHANGELOG are checked (`tag-version-check`), only its
build/publish jobs run. Splitting into per-unit workflow files would de-register every Trusted
Publisher. The `release` environment keeps its name; only its deployment tag policy lists
the prefixes (a repo-settings change, invisible in git, that must move together with the
trigger).

### RC-first, OIDC everywhere

Every release ships as `-rc.N` first and is promoted to GA only after the RC is exercised.
RCs publish under the npm `rc` dist-tag / PEP 440 pre-release; GA under `latest`. No
long-lived registry tokens anywhere. A new unit's first publish is a one-time manual push
(a Trusted Publisher cannot bind to a package that does not exist yet); every release after
that goes through the workflow.

### Per-package CHANGELOGs, skill-curated, workflow-gated

Every published package keeps a Keep-a-Changelog `CHANGELOG.md` in its directory. Drafts come
from git-cliff (repo-root `cliff.toml`, path-scoped per package at invocation); curation runs
through the repo-local `/changelog` skill; commit-prefix discipline (`feat`/`fix`/
`refactor`/`perf` appear, `docs`/`chore`/`ci` do not) is load-bearing for draft quality. At
GA, existing `X.Y.Z-rc.*` sections collapse into a single `X.Y.Z` section. For workflow-wired
units, the CI gate blocks publish if the tagged unit's CHANGELOG lacks the version heading.

### No first-party CLI

`@ratel-ai/cli` was removed from the repo: a leaf package with no in-repo dependents whose
verbs the local product now owns (ratel-local's `ratel-mcp` CLI). Its npm versions stay
frozen and undeprecated; its Trusted Publisher registration stays dormant so a future
reintroduction (a fresh decision) needs no first-publish dance.

## Consequences

- Units release on their own clock; the no-op-bump tax is gone. "The Ratel version" stops
  being a single value, and that is intended.
- The workflow gains prefix-routing complexity to keep the Trusted Publisher registrations
  and the `release` environment untouched: the highest-risk pieces of the system.
- Every release commit must touch the unit's CHANGELOG; forgetting blocks the release by
  construction.
- Rejected: staying lockstep (the tax compounds per unit); per-unit workflow files or a
  renamed environment (de-registers OIDC trust); long-lived tokens (the posture RC-first /
  Trusted Publishers exists to avoid); skipping RC for "small" units (the RC gate is what
  catches a bad publish before an immutable version goes live).
