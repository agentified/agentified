import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { pinAdapterSdkPeerFile } from "./pin-adapter-sdk-peer.mjs";

test("writes the floor range over the SDK peer and dev, leaving every other dep alone", () => {
  const dir = mkdtempSync(join(tmpdir(), "ratel-pin-"));
  const file = join(dir, "package.json");
  try {
    writeFileSync(
      file,
      `${JSON.stringify(
        {
          peerDependencies: { "@ratel-ai/sdk": "workspace:^", "@mastra/core": ">=1.11.0 <2" },
          devDependencies: { "@ratel-ai/sdk": "workspace:^" },
          dependencies: { "@ratel-ai/telemetry": "workspace:^" },
        },
        null,
        2,
      )}\n`,
    );
    pinAdapterSdkPeerFile(file);
    const written = JSON.parse(readFileSync(file, "utf8"));
    assert.equal(written.peerDependencies["@ratel-ai/sdk"], ">=0.11.0 <1.0.0");
    assert.equal(written.devDependencies["@ratel-ai/sdk"], ">=0.11.0 <1.0.0");
    assert.equal(written.dependencies["@ratel-ai/telemetry"], "workspace:^");
    assert.equal(written.peerDependencies["@mastra/core"], ">=1.11.0 <2");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
