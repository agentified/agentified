#!/usr/bin/env node
// Single source of truth for Ratel's release units (ADR-0008).
//
// Every release-infra tool reads unit facts from HERE so the units can
// never drift apart:
//   - scripts/check-release-tag.mjs  — the per-tag manifest + CHANGELOG gate
//   - scripts/releasable.mjs         — "which units have commits since their tag"
//   - scripts/publish-rc.sh          — the manual first-publish helper (--unit)
//   - .claude/skills/changelog/draft.sh — via the `--changelog-map` CLI below
//
// Adding another unit later is a one-place change.
//
// Per unit:
//   tagPrefix        release tag prefix; a unit ships on `<tagPrefix><semver>`.
//   label            human one-liner (registry destination) for docs/CLIs.
//   versionManifest  the manifest whose version string is canonical for the unit.
//   manifests        every manifest that MUST carry the tag version (tag gate).
//   changelogs       CHANGELOG(s) that MUST record the version (tag gate).
//   srcPaths         git pathspecs that make the unit "releasable" when they
//                    carry commits since its last tag (releasable.mjs).
//   changelog        git-cliff scope: { name, includePaths } (draft.sh).

import { readFileSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const PLATFORMS = ["darwin-arm64", "darwin-x64", "linux-x64-gnu", "linux-arm64-gnu", "win32-x64-msvc"];

export const UNITS = {
  core: {
    tagPrefix: "core-v",
    label: "ratel-ai-core → crates.io",
    versionManifest: { path: "src/core/Cargo.toml", kind: "toml" },
    manifests: [{ path: "src/core/Cargo.toml", kind: "toml" }],
    changelogs: ["src/core/CHANGELOG.md"],
    srcPaths: ["src/core"],
    changelog: { name: "ratel-ai-core", includePaths: ["src/core/**", "Cargo.toml"] },
  },
  "sdk-ts": {
    tagPrefix: "sdk-ts-v",
    label: "@ratel-ai/sdk (loader + 5 platform packages) → npm",
    // The JS SDK is internally lockstep: loader + five platform packages + the
    // ts-native crate all move together on one sdk-ts-v* tag.
    versionManifest: { path: "src/sdk/ts/package.json", kind: "json" },
    manifests: [
      { path: "src/sdk/ts/package.json", kind: "json" },
      ...PLATFORMS.map((t) => ({ path: `src/sdk/ts/npm/${t}/package.json`, kind: "json" })),
      { path: "src/sdk/ts/native/Cargo.toml", kind: "toml" },
    ],
    changelogs: ["src/sdk/ts/CHANGELOG.md"],
    srcPaths: ["src/sdk/ts"],
    changelog: { name: "@ratel-ai/sdk", includePaths: ["src/sdk/ts/**"] },
  },
  "sdk-py": {
    tagPrefix: "sdk-py-v",
    label: "ratel-ai → PyPI",
    versionManifest: { path: "src/sdk/python/pyproject.toml", kind: "toml", pep440: true },
    manifests: [
      { path: "src/sdk/python/pyproject.toml", kind: "toml", pep440: true },
      { path: "src/sdk/python/native/Cargo.toml", kind: "toml" },
    ],
    changelogs: ["src/sdk/python/CHANGELOG.md"],
    srcPaths: ["src/sdk/python"],
    changelog: { name: "ratel-ai", includePaths: ["src/sdk/python/**"] },
  },
  // The telemetry helpers are INDEPENDENT units — one per registry — so a fix to
  // just the npm vocabulary ships alone, and they can still go out in one run by
  // tagging the same commit (ADR-0008's per-package principle; the packages have
  // no cross-registry install dependency, so nothing forces them lockstep).
  // core/js/py share the vocabulary spec + conformance fixtures, so a change
  // there marks all three releasable and drafts into their changelogs.
  "telemetry-core": {
    tagPrefix: "telemetry-core-v",
    label: "ratel-ai-telemetry → crates.io",
    versionManifest: { path: "src/telemetry/core/Cargo.toml", kind: "toml" },
    manifests: [{ path: "src/telemetry/core/Cargo.toml", kind: "toml" }],
    changelogs: ["src/telemetry/core/CHANGELOG.md"],
    srcPaths: ["src/telemetry/core", "src/telemetry/CONVENTIONS.md", "src/telemetry/conformance"],
    changelog: {
      name: "ratel-ai-telemetry (crate)",
      includePaths: ["src/telemetry/core/**", "src/telemetry/CONVENTIONS.md", "src/telemetry/conformance/**"],
    },
  },
  "telemetry-ts": {
    tagPrefix: "telemetry-ts-v",
    label: "@ratel-ai/telemetry → npm",
    versionManifest: { path: "src/telemetry/ts/package.json", kind: "json" },
    manifests: [{ path: "src/telemetry/ts/package.json", kind: "json" }],
    changelogs: ["src/telemetry/ts/CHANGELOG.md"],
    srcPaths: ["src/telemetry/ts", "src/telemetry/CONVENTIONS.md", "src/telemetry/conformance"],
    changelog: {
      name: "@ratel-ai/telemetry",
      includePaths: ["src/telemetry/ts/**", "src/telemetry/CONVENTIONS.md", "src/telemetry/conformance/**"],
    },
  },
  "telemetry-py": {
    tagPrefix: "telemetry-py-v",
    label: "ratel-ai-telemetry → PyPI",
    // The npm package.json is canonical for its unit; the pyproject carries the
    // PEP 440 spelling of the same semver (e.g. 0.1.0rc1).
    versionManifest: { path: "src/telemetry/python/pyproject.toml", kind: "toml", pep440: true },
    manifests: [{ path: "src/telemetry/python/pyproject.toml", kind: "toml", pep440: true }],
    changelogs: ["src/telemetry/python/CHANGELOG.md"],
    srcPaths: ["src/telemetry/python", "src/telemetry/CONVENTIONS.md", "src/telemetry/conformance"],
    changelog: {
      name: "ratel-ai-telemetry (PyPI)",
      includePaths: ["src/telemetry/python/**", "src/telemetry/CONVENTIONS.md", "src/telemetry/conformance/**"],
    },
  },
  // Framework adapters: npm-only, pure-language packages that peer-depend on
  // @ratel-ai/sdk via `workspace:^` (rewritten at pack time to
  // SDK_ADAPTER_PEER_RANGE — a 0.x caret would pin the minor).
  // Each ships independently on its own tag prefix.
  "vercel-ai-sdk": {
    tagPrefix: "vercel-ai-sdk-v",
    label: "@ratel-ai/vercel-ai-sdk → npm",
    versionManifest: { path: "src/adapters/ts-vercel-ai-sdk/package.json", kind: "json" },
    manifests: [{ path: "src/adapters/ts-vercel-ai-sdk/package.json", kind: "json" }],
    changelogs: ["src/adapters/ts-vercel-ai-sdk/CHANGELOG.md"],
    srcPaths: ["src/adapters/ts-vercel-ai-sdk"],
    changelog: {
      name: "@ratel-ai/vercel-ai-sdk",
      includePaths: ["src/adapters/ts-vercel-ai-sdk/**"],
    },
  },
  "mastra": {
    tagPrefix: "mastra-v",
    label: "@ratel-ai/mastra → npm",
    versionManifest: { path: "src/adapters/ts-mastra/package.json", kind: "json" },
    manifests: [{ path: "src/adapters/ts-mastra/package.json", kind: "json" }],
    changelogs: ["src/adapters/ts-mastra/CHANGELOG.md"],
    srcPaths: ["src/adapters/ts-mastra"],
    changelog: {
      name: "@ratel-ai/mastra",
      includePaths: ["src/adapters/ts-mastra/**"],
    },
  },
};

export const UNIT_IDS = Object.keys(UNITS);

// Floor of the published `@ratel-ai/sdk` peer on both adapters. Source manifests
// stay `workspace:^`; pin-adapter-sdk-peer.mjs rewrites this range at publish.
// Never lower it; bump it to the SDK the adapters are built against when cutting
// an adapter release (ADR-0020).
export const SDK_ADAPTER_PEER_FLOOR = "0.11.0";
export const SDK_ADAPTER_PEER_RANGE = `>=${SDK_ADAPTER_PEER_FLOOR} <1.0.0`;

// Accepted release version: semver with an optional `-rc.N` pre-release.
export const SEMVER = /^\d+\.\d+\.\d+(?:-rc\.\d+)?$/;

// Regex alternation of the registered unit ids, derived so a new unit needs no
// hand-edited regex anywhere. `sdk-ts` etc. are literal in a character run.
export function unitIdAlternation() {
  return UNIT_IDS.join("|");
}

// The canonical version string a unit currently carries, read from its
// versionManifest. Shared so bash tools (publish-rc.sh) never re-hardcode a
// manifest path. Returns the raw manifest string (PEP 440 form for sdk-py).
export function versionOf(unit, root = process.cwd()) {
  const spec = UNITS[unit];
  if (!spec) throw new Error(`unknown unit: ${unit}`);
  const { path, kind } = spec.versionManifest;
  const body = readFileSync(`${root}/${path}`, "utf8");
  if (kind === "json") return JSON.parse(body).version ?? null;
  return /^version\s*=\s*"([^"]+)"/m.exec(body)?.[1] ?? null;
}

// Robust "was this module run as the entry script?" check, shared by every CLI in
// this dir. ESM resolves symlinks in import.meta.url, but process.argv[1] keeps the
// path as it was invoked — so a plain string compare is false when the repo is
// reached through a symlink, and the CLI silently does nothing (e.g. `--list` prints
// an empty list, which made publish-rc.sh report `unknown unit … (valid: )`).
// Comparing real paths matches any symlinked / relative / absolute invocation.
export function isMainModule(importMetaUrl) {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(importMetaUrl));
  } catch {
    return false;
  }
}

// ---- tiny CLI so bash tools (draft.sh) share this one registry ----
//   node scripts/release-units.mjs --list
//   node scripts/release-units.mjs --changelog-map [unit]
//   node scripts/release-units.mjs --tag-prefix <unit>
function main(argv) {
  const [cmd, arg] = argv.slice(2);
  switch (cmd) {
    case "--list":
      process.stdout.write(`${UNIT_IDS.join("\n")}\n`);
      return;
    case "--changelog-map": {
      // One `name|glob1|glob2...` line per unit (or just `arg`), consumed by
      // draft.sh so the git-cliff package list lives in exactly one place.
      const ids = arg ? [arg] : UNIT_IDS;
      for (const id of ids) {
        const u = UNITS[id];
        if (!u) {
          process.stderr.write(`unknown unit: ${id}\n`);
          process.exit(2);
        }
        process.stdout.write(`${[u.changelog.name, ...u.changelog.includePaths].join("|")}\n`);
      }
      return;
    }
    case "--tag-prefix": {
      const u = UNITS[arg];
      if (!u) {
        process.stderr.write(`unknown unit: ${arg}\n`);
        process.exit(2);
      }
      process.stdout.write(`${u.tagPrefix}\n`);
      return;
    }
    case "--version": {
      try {
        process.stdout.write(`${versionOf(arg)}\n`);
      } catch (e) {
        process.stderr.write(`${e.message}\n`);
        process.exit(2);
      }
      return;
    }
    default:
      process.stderr.write(
        "usage: release-units.mjs --list | --changelog-map [unit] | --tag-prefix <unit> | --version <unit>\n",
      );
      process.exit(2);
  }
}

if (isMainModule(import.meta.url)) {
  main(process.argv);
}
