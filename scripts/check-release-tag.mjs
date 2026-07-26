#!/usr/bin/env node
// Per-unit release-tag gate (ADR-0008). A release is cut by pushing a prefixed
// tag — one of the registered unit prefixes (`core-v*`, `sdk-ts-v*`, `sdk-py-v*`,
// `telemetry-core-v*`, `telemetry-ts-v*`, `telemetry-py-v*`; the set is derived
// from release-units.mjs, not hard-coded here). This checks that ONLY the tagged
// unit's manifests carry the tag's version and that its CHANGELOG(s) record it —
// nothing else in the repo has to be in lockstep.
//
// Usage (from repo root):
//   node scripts/check-release-tag.mjs <tag> [--root <dir>]
// On success it writes `unit`, `version`, `dist_tag` to $GITHUB_OUTPUT (when set)
// and exits 0; on any mismatch it prints ::error:: lines and exits 1.

import { readFileSync } from "node:fs";
import { join } from "node:path";

// The release units (manifests, CHANGELOGs, tag prefixes) live in one registry
// shared with releasable.mjs / publish-rc.sh / draft.sh — see release-units.mjs.
// `isMainModule` is shared too so every CLI's entry check is symlink-robust.
import { UNITS, SEMVER, unitIdAlternation, isMainModule } from "./release-units.mjs";

// `<prefix>-v<semver>` -> { unit, version }. Anything else (the old lockstep
// `v0.2.0`, an unknown prefix, a non-semver body) returns null so the caller can
// fail loudly instead of routing a tag nowhere.
export function parseTag(tag) {
  const m = new RegExp(`^(${unitIdAlternation()})-v(.+)$`).exec(tag ?? "");
  if (!m) return null;
  const [, unit, version] = m;
  if (!SEMVER.test(version)) return null;
  return { unit, version };
}

export function distTagFor(version) {
  return version.includes("-rc.") ? "rc" : "latest";
}

// PyPI/PEP 440 normalizes pre-releases (`0.2.0-rc.1` -> `0.2.0rc1`), so pyproject
// may legitimately hold the normalized form. Accept either spelling.
function pep440(version) {
  return version.replace(/-?rc\.?(\d+)/, "rc$1");
}

function readVersion(abs, kind) {
  const body = readFileSync(abs, "utf8");
  if (kind === "json") return JSON.parse(body).version ?? null;
  const m = /^version\s*=\s*"([^"]+)"/m.exec(body);
  return m ? m[1] : null;
}

export function checkReleaseTag(tag, { root = process.cwd() } = {}) {
  const parsed = parseTag(tag);
  if (!parsed) {
    return {
      ok: false,
      unit: null,
      version: null,
      distTag: null,
      // Unit list derived from the registry so a new unit never leaves this stale.
      errors: [`unroutable tag "${tag}" — expected <${unitIdAlternation()}>-v<semver>`],
    };
  }

  const { unit, version } = parsed;
  const spec = UNITS[unit];
  const errors = [];

  for (const { path, kind, pep440: isPy } of spec.manifests) {
    let found;
    try {
      found = readVersion(join(root, path), kind);
    } catch {
      errors.push(`${path}: cannot read version (missing manifest?)`);
      continue;
    }
    const accepted = isPy ? [version, pep440(version)] : [version];
    if (!accepted.includes(found)) {
      errors.push(`${path}: version ${found ?? "(none)"} does not match tag ${version}`);
    }
  }

  const heading = `## [${version}]`;
  for (const path of spec.changelogs) {
    let body;
    try {
      body = readFileSync(join(root, path), "utf8");
    } catch {
      errors.push(`${path}: cannot read CHANGELOG (missing?)`);
      continue;
    }
    if (!body.includes(heading)) {
      errors.push(`${path}: missing "${heading}" — run /changelog before tagging`);
    }
  }

  return { ok: errors.length === 0, unit, version, distTag: distTagFor(version), errors };
}

function main(argv) {
  const args = argv.slice(2);
  let tag;
  let root = process.cwd();
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--root") root = args[++i];
    else if (!tag) tag = args[i];
  }
  if (!tag) {
    console.error("::error::usage: check-release-tag.mjs <tag> [--root <dir>]");
    process.exit(2);
  }

  const r = checkReleaseTag(tag, { root });
  if (!r.ok) {
    for (const e of r.errors) console.error(`::error::${e}`);
    process.exit(1);
  }

  const out = `unit=${r.unit}\nversion=${r.version}\ndist_tag=${r.distTag}\n`;
  if (process.env.GITHUB_OUTPUT) {
    // Lazy import so the module stays usable in tests without the fs write path.
    import("node:fs").then(({ appendFileSync }) => appendFileSync(process.env.GITHUB_OUTPUT, out));
  }
  console.log(`${r.unit} ${r.version} (${r.distTag}) — manifests + CHANGELOG verified`);
}

if (isMainModule(import.meta.url)) {
  main(process.argv);
}
