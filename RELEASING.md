# Releasing Ratel

How a new version of a Ratel package is published. Read end-to-end before cutting a release.

Ratel releases **per unit** (ADR-0008): eight independently-versioned release units, each on
its own tag prefix, routed through one `release.yml`. There is no workspace-shared version —
each unit carries its own version in its own manifest and ships on its own cadence.

## Release units

| Unit | Tag prefix | Registry | Manifest (canonical version) |
|---|---|---|---|
| `core` | `core-v*` | `ratel-ai-core` on crates.io | `src/core/Cargo.toml` |
| `sdk-ts` | `sdk-ts-v*` | `@ratel-ai/sdk` + 5 platform packages on npm | `src/sdk/ts/package.json` |
| `sdk-py` | `sdk-py-v*` | `ratel-ai` on PyPI | `src/sdk/python/pyproject.toml` |
| `telemetry-core` | `telemetry-core-v*` | `ratel-ai-telemetry` on crates.io | `src/telemetry/core/Cargo.toml` |
| `telemetry-ts` | `telemetry-ts-v*` | `@ratel-ai/telemetry` on npm | `src/telemetry/ts/package.json` |
| `telemetry-py` | `telemetry-py-v*` | `ratel-ai-telemetry` on PyPI | `src/telemetry/python/pyproject.toml` |
| `vercel-ai-sdk` | `vercel-ai-sdk-v*` | `@ratel-ai/vercel-ai-sdk` on npm | `src/adapters/ts-vercel-ai-sdk/package.json` |
| `mastra` | `mastra-v*` | `@ratel-ai/mastra` on npm | `src/adapters/ts-mastra/package.json` |

The units are registered once, in [`scripts/release-units.mjs`](scripts/release-units.mjs)
— the single source of truth that the tag gate, the `releasable` helper, the changelog
drafter, and the manual publish helper all read. Adding a future unit is a one-place change.

The `vercel-ai-sdk` framework adapter is wired into `release.yml`'s triggers and its own
`publish-vercel-ai-sdk` job (pure-TS, built in CI then published from the package directory,
like the `telemetry-ts` job), publishing over OIDC via a Trusted Publisher. That path does
*not* `pnpm pack`, so unlike the manual helper it rewrites nothing: every `workspace:`
specifier the adapter ships needs an explicit pin step in the job. Its npm name was
bootstrapped by a manual first-publish (`scripts/publish-rc.sh --unit vercel-ai-sdk`) before
the Trusted Publisher could exist. The `release` environment's tag policy must allow
`vercel-ai-sdk-v*`, or the publish job hangs at the deploy gate.

The `sdk-ts` unit is internally lockstep: the loader `@ratel-ai/sdk`, its five per-OS native
packages (`@ratel-ai/sdk-darwin-arm64`, `-darwin-x64`, `-linux-x64-gnu`, `-linux-arm64-gnu`,
`-win32-x64-msvc`), and the `ratel-sdk-ts-native` crate all move together on one `sdk-ts-v*`
tag. Likewise `sdk-py` bundles the `ratel-sdk-python-native` crate with the wheel.

The three **telemetry** units are independent single-registry units — a fix to just the npm
vocabulary ships on `telemetry-ts-v*` alone. `telemetry-core` (crates.io), `telemetry-ts`
(npm), and `telemetry-py` (PyPI) carry the shared `ratel.*` vocabulary and its spec +
conformance fixtures, so they usually move together: tag the same commit with those prefixes
to release them in one run. All three are pure-language, so the manual helper builds them
locally (no prebuilt artifacts).

The **framework adapters** (`vercel-ai-sdk` → `@ratel-ai/vercel-ai-sdk`, `mastra` →
`@ratel-ai/mastra`; more to come) are npm-only, pure-language units that peer-depend on
`@ratel-ai/sdk` via `workspace:^`. `vercel-ai-sdk` additionally depends on
`@ratel-ai/telemetry` at publish time — a real runtime `dependencies` entry, not a peer, so
`telemetry-ts` must be released **before** it or the adapter's pinned range names a version
that is not on npm yet. `mastra` has no such dependency. Like `telemetry-ts`, the manual
helper builds them locally, then `pnpm pack`s them (the pack rewrites every `workspace:`
specifier, peer and runtime alike, to a concrete range); they need no prebuilt artifact.

`@ratel-ai/mcp-server` ships from a sibling repo, [ratel-ai/ratel-mcp](https://github.com/ratel-ai/ratel-mcp), on its own cadence.

## How the release pipeline is wired

- **`release.yml`** — fires on any `core-v*` / `sdk-ts-v*` / `sdk-py-v*` / `telemetry-core-v*` /
  `telemetry-ts-v*` / `telemetry-py-v*` / `mastra-v*` / `vercel-ai-sdk-v*` tag push (and
  supports `workflow_dispatch` with `dry_run: true` for rehearsal). Its first job,
  `tag-version-check`, runs [`scripts/check-release-tag.mjs`](scripts/check-release-tag.mjs) to
  route the tag to its unit and verify **only that unit's** manifests + CHANGELOG carry the
  version; the rest of the repo need not be in lockstep. The routed unit's build + publish
  jobs then run (the others are skipped), and a GitHub Release is created. Authentication is
  via Trusted Publishers (OIDC) — no `NPM_TOKEN` / `CARGO_REGISTRY_TOKEN` / PyPI token
  secrets. `*-rc.*` versions publish under the npm `rc` dist-tag (and are pre-release on PyPI
  by PEP 440); un-suffixed versions become `latest`.
- **`scripts/releasable.mjs`** — DX helper: run `node scripts/releasable.mjs` to see which
  units have commits since their last release tag (and how many), so you know what to cut.
- **`verify-install.yml`** — `workflow_dispatch` + daily cron. Installs a unit's published
  package from its public registry with no repo checkout / local toolchain and exercises it.
  Pick a `unit` (and optionally a `version`) to verify one; the daily cron verifies every unit
  at `latest`, `vercel-ai-sdk` included now that its GA holds npm's `latest` tag. A caller can
  still pin a version or select `rc` explicitly. Run after every release.
- **`build-binaries.yml`** / **`python-binaries.yml`** — `workflow_dispatch` only. Build the
  npm `.node` binaries (bundled into a `release-tarballs` artifact) and the PyPI `wheels-*` +
  sdist respectively. Used for the very first manual publish of a brand-new package, before a
  Trusted Publisher relationship exists (see First-time bootstrap).

## Pre-merge gate (catch breakage before it lands)

`release.yml` only builds the workflow-wired units' real distributables at tag time, and
`verify-install.yml` only smoke-tests them *after* publishing. To catch packaging breaks
(missing `files`, `optionalDependencies` injection, sdist/twine metadata, native-binding
load, cross-SDK drift) **before** they reach `main`, `pr-gate.yml` shifts that validation
onto the PR.

- **Opt-in to save CI.** The heavy jobs only run when a PR carries the **`ready-to-merge`**
  label (and re-run on every new commit while it stays on). Unlabeled PRs spend zero
  build minutes — the jobs are skipped.
- **Mandatory for everyone but rstagi.** The terminal `pr-gate` check is required on `main`.
  It **fails** any PR without the `ready-to-merge` label (so unlabeled PRs can't merge), and
  on a labeled PR it goes green only when the whole pipeline is green. So every contributor
  must arm + pass the gate to merge.
- **rstagi is a superadmin bypass.** rstagi can merge any PR at any time — red or green, with
  or without the label — via the branch ruleset's bypass. This is the deliberate escape hatch
  (no label, no bot). The bypass is scoped to him (admin role by default, or a one-member team
  for exact scoping via the branch ruleset). Everyone else is hard-blocked
  until `pr-gate` is green.
- **What it runs:** one **`verify` job per platform** that builds the real distributables
  (wheel, npm loader + native binding) and **installs each into a clean
  environment and runs the cross-SDK E2E** (`e2e/` — Python wheel, TS loader+native). The
  platform-independent packaging checks (sdist + `twine check`, `cargo publish
  --dry-run`, npm `optionalDependencies` injection) run once, folded into the linux leg.
  The Python and TS runners assert the same `e2e/scenario.json`, so a behavior divergence
  fails exactly one. (Kept to few check rows: `setup` + one row per platform + `pr-gate`;
  platforms run in parallel.)
- **Matrix (cost control):** armed-PR commits run a **reduced** matrix (`linux-x64` +
  `darwin-arm64` — the fast native runners). The **full 5-platform** matrix (adding Windows,
  `linux-arm64` cross-compile, `darwin-x64` Rosetta) runs on **every push to `main`**, so each
  merge is fully validated. A platform-specific break that slipped through the reduced PR matrix
  surfaces right after merge, not on every PR commit. (`workflow_dispatch` runs the full matrix
  on demand.)

Developer flow: open a PR → fast `rust/ts/python` checks run as usual → when ready to land,
add the `ready-to-merge` label → the gate runs on every commit → merge once `pr-gate` is
green. If the gate is red and the merge truly can't wait, **rstagi** can merge it directly
(his ruleset bypass); nobody else can.

Enable the required `pr-gate` check + create the `ready-to-merge` label once via repo settings
(a branch ruleset requiring `pr-gate` on `main`, plus the label); scope the bypass to exactly
rstagi with a one-member team. Run the E2E locally per `e2e/README.md`.

## Cutting a release

### Once-per-repo prep (already done; do not redo)

- `@ratel-ai` npm org exists; the publishing account is a member with `developer`+ role; 2FA enabled.
- `ratel-ai-core` (crates.io) and `ratel-ai` (PyPI) names are registered.
- Trusted Publishers are configured on the 6 SDK npm packages, the `ratel-ai-core` crate, the
  `ratel-ai` PyPI project, and the `@ratel-ai/vercel-ai-sdk` npm name — each pointing at this
  repo / `release.yml` / the `release` environment. `@ratel-ai/telemetry` is registered on npm
  (0.1.2, manual first-publish); confirm its Trusted Publisher before the next
  `telemetry-ts-v*` tag. **The `ratel-ai-telemetry` names (PyPI + crates.io) and the
  `@ratel-ai/mastra` npm name are not yet registered** — they are added at their first-time
  bootstrap. `@ratel-ai/telemetry-otlp@0.1.1` is retired: `npm deprecate`d by hand, out of the
  repo, no longer a release unit.
- A `release` GitHub Environment exists whose **deployment tag policy allows the unit
  prefixes** — `core-v*`, `sdk-ts-v*`, `sdk-py-v*`. Keep the environment *name* `release`
  unchanged (it's what binds the Trusted Publishers); only its tag policy lists the prefixes.
  A tag not matched by the policy hangs the publish job at the deploy gate. **Add
  `telemetry-core-v*`, `telemetry-ts-v*`, `telemetry-py-v*` to the policy before cutting the
  first telemetry release, and `mastra-v*` + `vercel-ai-sdk-v*` before their first CI-driven
  adapter release.**

### Per-release flow (one unit at a time)

1. **See what changed:** `node scripts/releasable.mjs` — pick the unit `$UNIT` to release.
2. **Bump that unit's version** to the new value (e.g. `0.2.1-rc.1`, then later `0.2.1`) in
   its manifest(s) — the tag gate checks every manifest the unit owns:
   - `core` → `src/core/Cargo.toml`
   - `sdk-ts` → `src/sdk/ts/package.json` **and** each `src/sdk/ts/npm/<triple>/package.json`
     **and** `src/sdk/ts/native/Cargo.toml` (all lockstep). The loader's
     `optionalDependencies` block is not stored in source; it is injected at publish time by
     `scripts/inject-sdk-optional-deps.mjs`.
   - `sdk-py` → `src/sdk/python/pyproject.toml` **and** `src/sdk/python/native/Cargo.toml`.
   - `telemetry-core` → `src/telemetry/core/Cargo.toml`.
   - `telemetry-ts` → `src/telemetry/ts/package.json`.
   - `telemetry-py` → `src/telemetry/python/pyproject.toml` (PEP 440 spelling, e.g. `0.1.0rc1`).
   - `vercel-ai-sdk` → `src/adapters/ts-vercel-ai-sdk/package.json`.
   - `mastra` → `src/adapters/ts-mastra/package.json`.
     The three telemetry units and the adapters version independently; bump only the unit(s)
     you are releasing.
3. **Update the CHANGELOG:** run the `/changelog` skill (`.claude/skills/changelog/`) for
   `$UNIT`. It drafts entries with [git-cliff](https://git-cliff.org) scoped to the unit,
   lets you curate, and writes the unit's `CHANGELOG.md`. For GA versions (no `-rc` suffix) it
   collapses the unit's existing `## [X.Y.Z-rc.*]` sections into one `## [X.Y.Z]` section.
4. **Verify locally** before tagging (whole workspace still builds):
   - `pnpm -r build && pnpm -r typecheck && pnpm -r lint && pnpm -r test`
   - `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo publish -p ratel-ai-core --dry-run --allow-dirty` (for a `core` release)
   - `pnpm --filter "@ratel-ai/vercel-ai-sdk..." build && pnpm --filter @ratel-ai/vercel-ai-sdk test`
     (for a `vercel-ai-sdk` release)
5. **(Optional dry-run, workflow-wired units only)** `workflow_dispatch` `release.yml` with
   the tag (e.g. `sdk-py-v0.2.1-rc.1`) and `dry_run: true` to validate the auth + publish path
   without consuming a version number.
6. **Commit, tag, push:**
   ```
   git commit -am "release: <unit>-vX.Y.Z"
   git tag <unit>-vX.Y.Z          # e.g. sdk-py-v0.2.1-rc.1
   git push origin main <unit>-vX.Y.Z
   ```
7. **Publish:** watch `release.yml` to completion and inspect the GitHub Release.
8. **Verify the install:** run `verify-install.yml` for the unit + version
   (`gh workflow run verify-install.yml -f unit=$UNIT -f version=X.Y.Z`).
9. **For RCs:** iterate (`-rc.2`, `-rc.3`, …) until happy, then bump to the un-suffixed
   version and tag again to promote to `latest`.

## Sharp edges

- **`tag-version-check` fails loudly on any manifest/CHANGELOG mismatch** for the tagged
  unit, short-circuiting the pipeline so nothing publishes. Fix the offending version, commit,
  re-tag. A tag that routes nowhere (e.g. the old lockstep `v0.2.0`) is rejected outright.
- **Never republish a version.** npm, crates.io, and PyPI all reject it. If a release goes
  wrong after a partial publish, bump to the next version and re-tag — the publish jobs are
  idempotent and will skip whatever already landed.
- **`@ratel-ai/sdk` `optionalDependencies` are injected, not committed.** `scripts/inject-sdk-optional-deps.mjs`
  writes the block into the in-flight `package.json` right before pack/publish, reading each
  `npm/<triple>/package.json`. Keeping it out of source prevents `pnpm install
  --frozen-lockfile` from failing on subpackages that don't yet exist on the registry, and it
  enforces that every subpackage version matches the loader — so a bump is "edit the version
  fields, push the tag".
- **macOS x64 is cross-compiled from `macos-14`** (Apple Silicon). GitHub's `macos-13` (Intel)
  pool has very long queues. Building `x86_64-apple-darwin` on `macos-14` with Rust's
  `--target` flag works because the Apple Silicon runners ship both SDKs. Don't switch back
  unless you've confirmed the Intel pool latency has improved.
- **Linux arm64-gnu** uses NAPI-RS's `--use-napi-cross` (its prebuilt sysroot containers).
  Don't switch to QEMU/`cross` without verifying glibc compatibility.

## First-time bootstrap

(Only when registering a brand-new package that has never existed on its registry — Trusted
Publishers can't be configured for a package that doesn't exist yet. Do this per unit.)

1. Build the unit's artifacts via `workflow_dispatch`:
   - `sdk-ts` → `build-binaries.yml` (produces the `release-tarballs` artifact).
   - `sdk-py` → `python-binaries.yml` (produces `wheels-*` + sdist).
   - `core` needs no prebuilt artifact — it publishes straight from the repo.
   - `telemetry-core` / `telemetry-ts` / `telemetry-py` / `mastra` / `vercel-ai-sdk` need no
     prebuilt artifact — they are pure-language, so `publish-rc.sh` builds the crate, npm
     package, wheel/sdist, or packed adapter locally.
2. Log in locally: `npm login` (npm requires 2FA on the publishing account for a first-publish
   of scoped public packages), `cargo login` for crates.io, and configure twine credentials
   (`TWINE_USERNAME=__token__` + a PyPI token, or `~/.pypirc`) for PyPI. The three telemetry
   units together need all three registries (`telemetry-ts` → npm, `telemetry-py` → PyPI,
   `telemetry-core` → crates.io); also `pip install build twine`.
3. Run `scripts/publish-rc.sh --unit <unit> --from-run <run-id>` (omit `--from-run` for
   `core`, the telemetry units, and the adapter units). It reads the unit's version from its
   manifest, finds the tarballs/wheels in the run's artifacts, and publishes — npm
   subpackages → loader for `sdk-ts`, `twine upload --skip-existing` for `sdk-py`, `cargo
   publish` for `core`, the locally-built npm / wheel / crate for `telemetry-ts` /
   `telemetry-py` / `telemetry-core`, and the pnpm-packed npm tarballs for `mastra` +
   `vercel-ai-sdk`.
   It's idempotent (skips anything already on the registry), so a partial failure is safe to
   resume. First-publish from a laptop ships **without provenance** (that requires GH Actions
   OIDC); that's expected for the bootstrap.
4. Configure Trusted Publishers on each registry name (npm web UI for the 6 SDK packages +
   `@ratel-ai/telemetry` + `@ratel-ai/mastra` + `@ratel-ai/vercel-ai-sdk`, crates.io for
   `ratel-ai-core` + `ratel-ai-telemetry`, PyPI for `ratel-ai` + `ratel-ai-telemetry`) pointing
   at `release.yml` in this repo, `release` environment.
5. Bump to the next version (e.g. `-rc.2`), tag `<unit>-v…`, push — `release.yml` should now
   publish via OIDC with no token errors, validating the trust relationship.
