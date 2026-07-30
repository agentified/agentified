import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

interface PackageManifest {
  private?: boolean;
  license?: string;
  engines?: Record<string, string>;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  exports?: Record<string, unknown>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
  publishConfig?: { access?: string; provenance?: boolean };
}

const manifest = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
) as PackageManifest;
const releaseWorkflow = readFileSync(
  new URL("../../../../.github/workflows/release.yml", import.meta.url),
  "utf8",
);

describe("published mastra dependency layout", () => {
  it("depends only on the vocabulary its observability subpath imports", () => {
    // Mastra, zod, the SDK, and OTel are host-owned peers. The OTel-free
    // vocabulary is the one unconditional package import the published
    // observability module must always resolve.
    expect(manifest.dependencies ?? {}).toEqual({ "@ratel-ai/telemetry": "workspace:^" });
  });

  it("supports @mastra/core from 1.11 through 1.x", () => {
    expect(manifest.peerDependencies?.["@mastra/core"]).toBe(">=1.11.0 <2");
    // zod is a runtime peer (unlike the AI SDK adapter): the exposed capability
    // tools carry hand-written zod schemas. The range matches @mastra/core's own
    // zod peer so a Mastra app on zod 3.25.x resolves cleanly.
    expect(manifest.peerDependencies?.zod).toBe("^3.25.0 || ^4.0.0");
    expect(manifest.peerDependencies?.["@ratel-ai/sdk"]).toBe("workspace:^");
  });

  it("keeps experiment enrichment on an opt-in subpath with an optional OTel peer", () => {
    expect(manifest.peerDependencies?.["@opentelemetry/api"]).toBe("^1.9.0");
    expect(manifest.peerDependenciesMeta?.["@opentelemetry/api"]?.optional).toBe(true);
    expect(manifest.exports).toMatchObject({
      ".": {
        types: "./dist/index.d.ts",
        default: "./dist/index.js",
      },
      "./observability": {
        types: "./dist/observability.d.ts",
        default: "./dist/observability.js",
      },
      "./dist/*": "./dist/*",
    });

    const rootSource = readFileSync(new URL("./index.ts", import.meta.url), "utf8");
    expect(rootSource).not.toContain("./observability.js");
  });

  it("pins the telemetry workspace dependency before publishing", () => {
    const jobStart = releaseWorkflow.indexOf("\n  publish-mastra:");
    const jobEnd = releaseWorkflow.indexOf("\n  publish-vercel-ai-sdk:", jobStart);
    const publishJob = releaseWorkflow.slice(jobStart, jobEnd);

    expect(publishJob).toContain("p.dependencies['@ratel-ai/telemetry']='^'+process.env.TELE_VER");
    expect(publishJob).toContain(`npm view "@ratel-ai/telemetry@^\${tele_ver}" version`);
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
