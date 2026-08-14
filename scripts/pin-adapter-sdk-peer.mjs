#!/usr/bin/env node
// Rewrite an adapter package.json's @ratel-ai/sdk peer + dev to the floor range
// from release-units.mjs. Telemetry is untouched (its pin stays inline in
// release.yml — mastra package-layout.test.ts asserts that text byte-for-byte).

import { readFileSync, writeFileSync } from "node:fs";

import { isMainModule, SDK_ADAPTER_PEER_RANGE } from "./release-units.mjs";

export function pinAdapterSdkPeerFile(file) {
  const pkg = JSON.parse(readFileSync(file, "utf8"));
  for (const field of ["peerDependencies", "devDependencies"]) {
    if (pkg[field]?.["@ratel-ai/sdk"]) pkg[field]["@ratel-ai/sdk"] = SDK_ADAPTER_PEER_RANGE;
  }
  writeFileSync(file, `${JSON.stringify(pkg, null, 2)}\n`);
}

if (isMainModule(import.meta.url)) {
  pinAdapterSdkPeerFile(process.argv[2]);
}
