#!/usr/bin/env node
// Prepare src/sdk/ts/package.json for `npm publish` of the SDK loader. Two rewrites,
// both because `npm publish` (unlike `pnpm publish`) ships the manifest verbatim:
//
//   1. `optionalDependencies` — the five committed src/sdk/ts/npm/<triple>/package.json
//      platform packages, injected at the loader's version (their source of truth).
//   2. `workspace:` specifiers — the SDK depends on sibling workspace packages
//      (today just @ratel-ai/telemetry). npm keeps `workspace:^` verbatim, which is
//      uninstallable off-workspace, so pin each to a caret range on the sibling's
//      current (published-in-lockstep) version.
//
// A final guard throws if ANY `workspace:` specifier survives, so a new workspace dep
// can never silently ship a broken manifest again.
//
// Run this in CI right before `npm publish` for the SDK loader. Replaces
// `napi pre-publish --skip-optional-publish`, which has a hidden dependency on git log
// heuristics that breaks under shallow checkouts on non-tagged branches.

import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Sibling workspace packages the loader may reference, and where their canonical
// version lives. A `workspace:` specifier for any other name is a hard error.
const WORKSPACE_DEP_MANIFESTS = {
  "@ratel-ai/telemetry": "src/telemetry/ts/package.json",
};

const DEP_FIELDS = ["dependencies", "peerDependencies", "optionalDependencies"];

function readJson(root, rel) {
  return JSON.parse(readFileSync(join(root, rel), "utf8"));
}

// Pin every `workspace:*` specifier in `deps` to `^<sibling version>`.
function pinWorkspaceDeps(deps, root) {
  if (!deps) return;
  for (const [name, spec] of Object.entries(deps)) {
    if (typeof spec !== "string" || !spec.startsWith("workspace:")) continue;
    const manifest = WORKSPACE_DEP_MANIFESTS[name];
    if (!manifest) {
      throw new Error(
        `workspace dependency ${name} has no known manifest — add it to WORKSPACE_DEP_MANIFESTS`,
      );
    }
    deps[name] = `^${readJson(root, manifest).version}`;
  }
}

export function injectSdkOptionalDeps(root = process.cwd()) {
  const mainRel = "src/sdk/ts/package.json";
  const npmRel = "src/sdk/ts/npm";
  const pkg = readJson(root, mainRel);

  const optionalDependencies = {};
  for (const dir of readdirSync(join(root, npmRel))) {
    const sub = readJson(root, join(npmRel, dir, "package.json"));
    if (sub.version !== pkg.version) {
      throw new Error(
        `version mismatch: ${sub.name}@${sub.version} but loader is ${pkg.version} — bump the subpackage version to match before publishing`,
      );
    }
    optionalDependencies[sub.name] = sub.version;
  }
  pkg.optionalDependencies = optionalDependencies;

  pinWorkspaceDeps(pkg.dependencies, root);
  pinWorkspaceDeps(pkg.peerDependencies, root);

  // Nothing may reach the registry still speaking `workspace:` — it is uninstallable.
  for (const field of DEP_FIELDS) {
    for (const [name, spec] of Object.entries(pkg[field] ?? {})) {
      if (typeof spec === "string" && spec.startsWith("workspace:")) {
        throw new Error(`unresolved workspace specifier ${field}.${name}=${spec} — cannot publish`);
      }
    }
  }

  writeFileSync(join(root, mainRel), `${JSON.stringify(pkg, null, 2)}\n`);
  return {
    version: pkg.version,
    optionalDependencies,
    dependencies: pkg.dependencies,
    peerDependencies: pkg.peerDependencies,
  };
}

// Run as a script (not when imported by the test).
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const { version, optionalDependencies } = injectSdkOptionalDeps();
  console.log(
    `injected ${Object.keys(optionalDependencies).length} optionalDependencies + pinned workspace deps into src/sdk/ts/package.json at version ${version}`,
  );
}
