import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

interface PackageManifest {
  private?: boolean;
  license?: string;
  engines?: Record<string, string>;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  publishConfig?: { access?: string; provenance?: boolean };
}

const manifest = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
) as PackageManifest;

describe("published mastra dependency layout", () => {
  it("carries no runtime dependencies — the adapter is pure glue", () => {
    // Everything the adapter touches (`@mastra/core`, `zod`, `@ratel-ai/sdk`) is
    // the host's to provide, so it ships as peers with zero runtime `dependencies`.
    expect(manifest.dependencies ?? {}).toEqual({});
  });

  it("supports @mastra/core from 1.11 through 1.x", () => {
    expect(manifest.peerDependencies?.["@mastra/core"]).toBe(">=1.11.0 <2");
    // zod is a runtime peer (unlike the AI SDK adapter): the exposed capability
    // tools carry hand-written zod schemas. The range matches @mastra/core's own
    // zod peer so a Mastra app on zod 3.25.x resolves cleanly.
    expect(manifest.peerDependencies?.zod).toBe("^3.25.0 || ^4.0.0");
    expect(manifest.peerDependencies?.["@ratel-ai/sdk"]).toBe("workspace:^");
  });

  it("dev-pins a concrete @mastra/core release and the workspace SDK", () => {
    // CI replaces this exact dev version with the exact supported floor and reruns
    // the suite; a range here would make either side of that matrix nondeterministic.
    expect(manifest.devDependencies?.["@mastra/core"]).toMatch(/^\d+\.\d+\.\d+$/);
    expect(manifest.devDependencies?.["@ratel-ai/sdk"]).toBe("workspace:^");
  });

  it("matches Mastra's Node.js floor", () => {
    expect(manifest.engines?.node).toBe(">=22.13.0");
  });

  it("publishes public with provenance under MIT", () => {
    expect(manifest.private).toBe(false);
    expect(manifest.license).toBe("MIT");
    expect(manifest.publishConfig?.access).toBe("public");
    expect(manifest.publishConfig?.provenance).toBe(true);
  });
});
