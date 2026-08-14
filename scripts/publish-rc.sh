#!/usr/bin/env bash
# Manual per-unit publish helper for package bootstrap (Trusted Publishers can't
# bind before a package exists) and temporarily manual release units.
#
# After Trusted Publishers are configured, use the release.yml workflow instead —
# it publishes via OIDC with provenance and no stored tokens. `vercel-ai-sdk` is
# wired there now (`publish-vercel-ai-sdk`); it stays here only as the bootstrap
# path that first-published the npm name.
#
# This helper packs before publishing, so `pnpm pack` rewrites every `workspace:`
# specifier for it. What it does NOT do is check the rewritten range exists on npm
# — publish a unit ahead of a workspace dependency it names and the tarball is
# permanently uninstallable. The stable ordering at the bottom of this file only
# protects an all-units run; `--unit vercel-ai-sdk` skips straight past it.
#
# Release units (ADR-0008), each publishable independently:
#   core           -> ratel-ai-core on crates.io          (cargo publish)
#   sdk-ts         -> @ratel-ai/sdk + 5 platform pkgs, npm (npm publish, tarballs)
#   sdk-py         -> ratel-ai on PyPI                     (twine upload, wheels + sdist)
#   telemetry-core -> ratel-ai-telemetry on crates.io      (cargo publish)
#   telemetry-ts   -> @ratel-ai/telemetry on npm           (npm publish, built locally)
#   telemetry-py   -> ratel-ai-telemetry on PyPI           (twine upload, built locally)
#   vercel-ai-sdk  -> @ratel-ai/vercel-ai-sdk on npm       (npm publish, pnpm-packed locally)
#   mastra -> @ratel-ai/mastra on npm      (npm publish, pnpm-packed locally)
# Each unit's version is read from its own manifest via scripts/release-units.mjs
# (the single source of truth) — units are NOT assumed to share a version.
#
# The telemetry units are pure-language (no cross-compiled native binaries), so this
# helper BUILDS their npm/PyPI/crate artifacts locally — they need no --from-run/--from-dir.
#
# Usage:
#   scripts/publish-rc.sh --unit sdk-ts --from-run <run-id> [--tag rc] [--dry-run]
#   scripts/publish-rc.sh --unit sdk-py --from-dir <path>   [--dry-run]
#   scripts/publish-rc.sh --unit core                       [--dry-run]
#   scripts/publish-rc.sh --unit telemetry-ts               [--dry-run]
#   scripts/publish-rc.sh --unit vercel-ai-sdk [--tag rc|latest] [--dry-run]
#   scripts/publish-rc.sh --from-dir <path>                 # all units
#
# Options:
#   --unit <id>        core | sdk-ts | sdk-py | telemetry-core | telemetry-ts |
#                      telemetry-py | vercel-ai-sdk | mastra.
#                      Repeatable. Default: all.
#   --from-run <id>    Download all artifacts from the given GH Actions run
#                      (requires `gh auth login`); tarballs/wheels are found within.
#   --from-dir <path>  Use a directory already holding the tarballs/wheels.
#   --tag <name>       npm dist-tag (default: rc). PyPI/crates use the version's
#                      own pre-release semantics; there is no dist-tag concept.
#   --dry-run          Print what would be published; don't actually publish.
#
# Idempotent: every registry is queried (or `--skip-existing` used) before an
# upload, so a partial failure is safe to resume by re-running the same command.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORMS=(darwin-arm64 darwin-x64 linux-x64-gnu linux-arm64-gnu win32-x64-msvc)
TAG="rc"
FROM_DIR=""
FROM_RUN=""
DRY_RUN=0
declare -a SELECTED=()
# Adapter publishes pin the live manifests before pack; git checkout restores
# them on EXIT (crash included). Guarded so a core/telemetry-only run cannot
# clobber an unrelated uncommitted edit to those two files.
PINNED_ADAPTERS=0
trap '[[ $PINNED_ADAPTERS -eq 1 ]] && git -C "$REPO_ROOT" checkout -- src/adapters/ts-mastra/package.json src/adapters/ts-vercel-ai-sdk/package.json' EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)     SELECTED+=("$2"); shift 2 ;;
    --from-run) FROM_RUN="$2"; shift 2 ;;
    --from-dir) FROM_DIR="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,42p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Default to all units. Validate each against the shared registry.
# (while-read, not mapfile — this helper runs on macOS's bash 3.2 too.)
if [[ ${#SELECTED[@]} -eq 0 ]]; then
  while IFS= read -r u; do SELECTED+=("$u"); done < <(node "$REPO_ROOT/scripts/release-units.mjs" --list)
fi
VALID_UNITS="$(node "$REPO_ROOT/scripts/release-units.mjs" --list | tr '\n' ' ')"
for u in "${SELECTED[@]}"; do
  case " $VALID_UNITS " in
    *" $u "*) ;;
    *) echo "error: unknown unit '$u' (valid: $VALID_UNITS)" >&2; exit 2 ;;
  esac
done

selected() { [[ " ${SELECTED[*]} " == *" $1 "* ]]; }
resolve_version() { node "$REPO_ROOT/scripts/release-units.mjs" --version "$1"; }

# npm-only units need artifacts on disk; core (cargo) builds from the repo.
NEEDS_ARTIFACTS=0
selected sdk-ts && NEEDS_ARTIFACTS=1
selected sdk-py && NEEDS_ARTIFACTS=1
if [[ $NEEDS_ARTIFACTS -eq 1 && -z "$FROM_DIR" && -z "$FROM_RUN" ]]; then
  echo "error: sdk-ts/sdk-py need --from-run <id> or --from-dir <path>" >&2
  exit 2
fi
if [[ -n "$FROM_RUN" && -z "$FROM_DIR" ]]; then
  FROM_DIR="$(mktemp -d)"
  echo "==> downloading all artifacts from run $FROM_RUN to $FROM_DIR"
  gh run download "$FROM_RUN" -D "$FROM_DIR" --repo ratel-ai/ratel
fi

echo "==> units:   ${SELECTED[*]}"
echo "==> npm tag: $TAG"
[[ $DRY_RUN -eq 1 ]] && echo "==> DRY RUN (no actual publishes)"
echo

# Wrapper around `npm publish` that treats "previously published" as success
# (fallback when the registry's HTTP pre-check missed due to CDN/replication lag)
# without breaking npm's 2FA UX. stdout is left flowing so npm's TTY-detection
# picks the right 2FA flow; only stderr is captured (via tee) to inspect the 403.
publish_one_npm() {
  local file="$1" err_file
  err_file="$(mktemp)"
  if npm publish "$file" --access public --tag "$TAG" --provenance=false \
       2> >(tee "$err_file" >&2); then
    rm -f "$err_file"; return 0
  fi
  if grep -q "previously published versions" "$err_file"; then
    rm -f "$err_file"; echo "    already published (caught at publish time), continuing"; return 0
  fi
  rm -f "$err_file"; return 1
}

# ---------- sdk-ts: npm loader + 5 platform packages ----------
publish_sdk_ts() {
  local version; version="$(resolve_version sdk-ts)"
  echo "===== sdk-ts @ $version (npm) ====="
  # A dry-run only validates the publish plan, so it doesn't need credentials.
  if [[ $DRY_RUN -eq 0 ]]; then
    if ! npm whoami >/dev/null 2>&1; then
      echo "error: not logged in to npm. run 'npm login' first." >&2; exit 1
    fi
    echo "    npm user: $(npm whoami)"
  fi

  # Order matters: the 5 subpackages (referenced by the loader's
  # optionalDependencies) publish before the loader.
  local -a packages=()
  local t
  for t in "${PLATFORMS[@]}"; do
    packages+=("ratel-ai-sdk-${t}-${version}.tgz|@ratel-ai/sdk-${t}")
  done
  packages+=("ratel-ai-sdk-${version}.tgz|@ratel-ai/sdk")

  # Pre-flight: locate every expected tarball inside FROM_DIR.
  local -a files=()
  local entry file found cand
  for entry in "${packages[@]}"; do
    file="${entry%%|*}"
    # first match; while-read avoids a `find | head` pipeline tripping pipefail.
    found=""
    while IFS= read -r cand; do found="$cand"; break; done < <(find "$FROM_DIR" -name "$file")
    if [[ -z "$found" ]]; then echo "missing tarball: $file (in $FROM_DIR)" >&2; exit 1; fi
    files+=("$found")
  done

  local i name enc_name url status
  for i in "${!packages[@]}"; do
    name="${packages[$i]##*|}"; file="${files[$i]}"
    echo "----- $name@$version -----"
    enc_name="${name//\//%2F}"
    url="https://registry.npmjs.org/${enc_name}/${version}"
    status="$(curl -sS -o /dev/null -w '%{http_code}' "$url" || echo 000)"
    if [[ "$status" == "200" ]]; then echo "    already published, skipping"; continue; fi
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "    [dry-run] npm publish $file --access public --tag $TAG --provenance=false"; continue
    fi
    publish_one_npm "$file" || exit 1
  done
  echo "==> sdk-ts npm publishes complete"; echo
}

# ---------- sdk-py: PyPI wheels + sdist via twine ----------
publish_sdk_py() {
  local version; version="$(resolve_version sdk-py)"
  echo "===== sdk-py @ $version (PyPI) ====="
  if ! command -v twine >/dev/null 2>&1; then
    echo "error: twine not found. 'pip install twine' (auth via TWINE_* env or ~/.pypirc)." >&2; exit 1
  fi

  # Collect this version's wheels + sdist from FROM_DIR.
  local -a dists=()
  local f
  while IFS= read -r f; do dists+=("$f"); done < <(
    find "$FROM_DIR" \( -name "ratel_ai-*.whl" -o -name "ratel_ai-*.tar.gz" \) | sort
  )
  if [[ ${#dists[@]} -eq 0 ]]; then
    echo "missing: no ratel_ai wheels/sdist found in $FROM_DIR" >&2; exit 1
  fi
  echo "    artifacts:"; printf '      %s\n' "${dists[@]}"

  # `twine check` validates metadata/README rendering before any upload.
  twine check "${dists[@]}"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] twine upload --skip-existing ${dists[*]}"; echo; return 0
  fi
  # --skip-existing makes re-runs idempotent (already-uploaded files are skipped).
  twine upload --skip-existing "${dists[@]}"
  echo "==> sdk-py PyPI publish complete"; echo
}

# ---------- core: crates.io ----------
# crates.io has no dist-tag; a pre-release version (0.2.0-rc.1) is simply not
# selected by consumers unless they ask for it explicitly.
publish_core() {
  local version; version="$(resolve_version core)"
  echo "===== core @ $version (crates.io) ====="
  cd "$REPO_ROOT"
  local status
  # crates.io's data-access policy 403s a generic `curl/*` User-Agent, so a
  # UA-less check would ALWAYS look "not published" and never skip. Send a
  # descriptive UA (with contact) as their policy requires.
  status="$(curl -sS -o /dev/null -w '%{http_code}' \
    -H 'User-Agent: ratel-publish-rc (roberto@ratel.sh)' \
    "https://crates.io/api/v1/crates/ratel-ai-core/${version}" || echo 000)"
  if [[ "$status" == "200" ]]; then
    echo "    already published, skipping"; echo; return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] cargo publish -p ratel-ai-core"; echo; return 0
  fi
  local out
  if out="$(cargo publish -p ratel-ai-core 2>&1)"; then
    printf '%s\n' "$out"
  elif printf '%s' "$out" | grep -qE "already (exists|uploaded)|crate version .* is already uploaded"; then
    echo "    already published (caught at publish time), continuing"
  else
    printf '%s\n' "$out" >&2; exit 1
  fi
  echo
}

# ---------- telemetry-ts: @ratel-ai/telemetry on npm, built locally ----------
# Pure-language, so nothing comes from a CI run: build the npm package (tsc) and
# publish, idempotent against the registry.
publish_telemetry_ts() {
  local version; version="$(resolve_version telemetry-ts)"
  echo "===== telemetry-ts @ $version (npm) ====="
  echo "----- @ratel-ai/telemetry@$version (npm) -----"
  local status
  status="$(curl -sS -o /dev/null -w '%{http_code}' \
    "https://registry.npmjs.org/@ratel-ai%2Ftelemetry/${version}" || echo 000)"
  if [[ "$status" == "200" ]]; then
    echo "    already published, skipping npm"; echo; return 0
  fi
  if [[ $DRY_RUN -eq 0 ]] && ! npm whoami >/dev/null 2>&1; then
    echo "error: not logged in to npm. run 'npm login' first." >&2; exit 1
  fi
  ( cd "$REPO_ROOT" && pnpm --filter @ratel-ai/telemetry run build )
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] npm publish (src/telemetry/ts) --access public --tag $TAG --provenance=false"
  else
    ( cd "$REPO_ROOT/src/telemetry/ts" && npm publish --access public --tag "$TAG" --provenance=false )
  fi
  echo "==> telemetry-ts publish complete"; echo
}

# ---------- telemetry-py: ratel-ai-telemetry on PyPI (pure-python wheel + sdist) ----------
publish_telemetry_py() {
  local version; version="$(resolve_version telemetry-py)"
  echo "===== telemetry-py @ $version (PyPI) ====="
  echo "----- ratel-ai-telemetry@$version (PyPI) -----"
  if ! command -v twine >/dev/null 2>&1; then
    echo "error: twine not found. 'pip install twine build' (auth via TWINE_* env or ~/.pypirc)." >&2; exit 1
  fi
  local dist; dist="$(mktemp -d)"
  ( cd "$REPO_ROOT" && python -m build --outdir "$dist" src/telemetry/python )
  twine check "$dist"/*
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] twine upload --skip-existing $dist/*"
  else
    # --skip-existing keeps re-runs idempotent.
    twine upload --skip-existing "$dist"/*
  fi
  echo "==> telemetry-py publish complete"; echo
}

# ---------- vercel-ai-sdk: @ratel-ai/vercel-ai-sdk on npm, packed locally ----------
# The Vercel AI SDK framework adapter. Pure-TS like telemetry-ts, with a workspace:^
# peer on @ratel-ai/sdk (pinned to the ADR-0020 floor range before pack) and a
# workspace:^ runtime dep on @ratel-ai/telemetry. `pnpm pack` rewrites remaining
# workspace: specifiers in the tarball, so a plain `npm publish <tarball>`
# (bootstrap: no OIDC, no provenance) ships a valid manifest.
publish_vercel_ai_sdk() {
  local version; version="$(resolve_version vercel-ai-sdk)"
  echo "===== vercel-ai-sdk @ $version (npm) ====="
  echo "----- @ratel-ai/vercel-ai-sdk@$version (npm) -----"
  local status
  status="$(curl -sS -o /dev/null -w '%{http_code}' \
    "https://registry.npmjs.org/@ratel-ai%2Fvercel-ai-sdk/${version}" || echo 000)"
  if [[ "$status" == "200" ]]; then
    echo "    already published, skipping npm"; echo; return 0
  fi
  if [[ $DRY_RUN -eq 0 ]] && ! npm whoami >/dev/null 2>&1; then
    echo "error: not logged in to npm. run 'npm login' first." >&2; exit 1
  fi
  # Build the adapter + its @ratel-ai/sdk and @ratel-ai/telemetry workspace
  # dependencies (topo order).
  ( cd "$REPO_ROOT" && pnpm --filter "@ratel-ai/vercel-ai-sdk..." run build )
  PINNED_ADAPTERS=1
  ( cd "$REPO_ROOT" && node scripts/pin-adapter-sdk-peer.mjs src/adapters/ts-vercel-ai-sdk/package.json )
  local tgz
  tgz="$( cd "$REPO_ROOT/src/adapters/ts-vercel-ai-sdk" && pnpm pack --pack-destination "$(mktemp -d)" | tail -1 )"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] npm publish $tgz --access public --tag $TAG --provenance=false"
  else
    publish_one_npm "$tgz" || exit 1
  fi
  echo "==> vercel-ai-sdk publish complete"; echo
}

# ---------- telemetry-core: ratel-ai-telemetry on crates.io ----------
publish_telemetry_core() {
  local version; version="$(resolve_version telemetry-core)"
  echo "===== telemetry-core @ $version (crates.io) ====="
  echo "----- ratel-ai-telemetry@$version (crates.io) -----"
  local status
  status="$(curl -sS -o /dev/null -w '%{http_code}' \
    -H 'User-Agent: ratel-publish-rc (roberto@ratel.sh)' \
    "https://crates.io/api/v1/crates/ratel-ai-telemetry/${version}" || echo 000)"
  if [[ "$status" == "200" ]]; then
    echo "    already published, skipping crate"; echo; return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] cargo publish -p ratel-ai-telemetry"; echo; return 0
  fi
  local out
  if out="$(cd "$REPO_ROOT" && cargo publish -p ratel-ai-telemetry 2>&1)"; then
    printf '%s\n' "$out"
  elif printf '%s' "$out" | grep -qE "already (exists|uploaded)|crate version .* is already uploaded"; then
    echo "    already published (caught at publish time), continuing"
  else
    printf '%s\n' "$out" >&2; exit 1
  fi
  echo "==> telemetry-core publish complete"; echo
}

# ---------- mastra: @ratel-ai/mastra on npm, packed locally ----------
# Pure-language framework adapter with a `workspace:^` peer on @ratel-ai/sdk
# (pinned to the ADR-0020 floor range before pack) and a `workspace:^` runtime
# dep on @ratel-ai/telemetry. `pnpm pack` rewrites remaining workspace: specifiers
# in the tarball, so a plain `npm publish <tarball>` (bootstrap: no OIDC, no provenance)
# ships a valid manifest — same shape as vercel-ai-sdk.
publish_mastra() {
  local version; version="$(resolve_version mastra)"
  echo "===== mastra @ $version (npm) ====="
  echo "----- @ratel-ai/mastra@$version (npm) -----"
  local status
  status="$(curl -sS -o /dev/null -w '%{http_code}' \
    "https://registry.npmjs.org/@ratel-ai%2Fmastra/${version}" || echo 000)"
  if [[ "$status" == "200" ]]; then
    echo "    already published, skipping npm"; echo; return 0
  fi
  if [[ $DRY_RUN -eq 0 ]] && ! npm whoami >/dev/null 2>&1; then
    echo "error: not logged in to npm. run 'npm login' first." >&2; exit 1
  fi
  # Build the adapter and its @ratel-ai/sdk workspace dependency (topo order).
  ( cd "$REPO_ROOT" && pnpm --filter "@ratel-ai/mastra..." run build )
  PINNED_ADAPTERS=1
  ( cd "$REPO_ROOT" && node scripts/pin-adapter-sdk-peer.mjs src/adapters/ts-mastra/package.json )
  local tgz
  tgz="$( cd "$REPO_ROOT/src/adapters/ts-mastra" && pnpm pack --pack-destination "$(mktemp -d)" | tail -1 )"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "    [dry-run] npm publish $tgz --access public --tag $TAG --provenance=false"
  else
    publish_one_npm "$tgz" || exit 1
  fi
  echo "==> mastra publish complete"; echo
}

# Publish in a stable order regardless of --unit argument order.
selected core           && publish_core
selected sdk-ts         && publish_sdk_ts
selected sdk-py         && publish_sdk_py
selected telemetry-core && publish_telemetry_core
selected telemetry-ts   && publish_telemetry_ts
selected telemetry-py   && publish_telemetry_py
selected vercel-ai-sdk  && publish_vercel_ai_sdk
selected mastra && publish_mastra

echo "==> done"
echo
echo "next steps:"
echo "  1. verify on a clean machine (no local toolchain) via the Verify install"
echo "     workflow: gh workflow run verify-install.yml -f unit=<unit> -f version=<ver>"
echo "  2. configure Trusted Publishers for each published name (see RELEASING.md,"
echo "     'first-time bootstrap'), all pointing at release.yml / the release env"
echo "  3. push the next tag (e.g. <unit>-v<ver>) to validate the CI publish path"
