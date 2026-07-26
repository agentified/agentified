import { describe, expect, it } from "vitest";
import { isPackageInstalled } from "./package-resolution.js";

const globals = globalThis as unknown as Record<string, unknown>;

describe("isPackageInstalled", () => {
  it("detects an installed package and an absent one", () => {
    // @opentelemetry/api is a real dependency, so it resolves; a nonsense name does not.
    expect(isPackageInstalled("@opentelemetry/api")).toBe(true);
    expect(isPackageInstalled("@ratel-ai/definitely-not-a-real-package")).toBe(false);
  });

  it("reads a resolvable-but-quirky package as installed", () => {
    // The crux: a *present* package that fails for its own reasons — here an exports-map
    // subpath that is not published — must not be misreported as absent; that would send
    // the caller to install what they already have.
    expect(isPackageInstalled("@opentelemetry/api/not-an-exported-subpath")).toBe(true);
  });

  it("resolves without evaluating the package", () => {
    // The probe must stay side-effect-free: it answers "is it in the tree", so a module
    // that throws (or spawns, or registers globals) on evaluation must not run. This
    // fixture sets a global the moment it is imported — resolving it is fine, importing
    // is not.
    // Specifier is relative to package-resolution.ts, where createRequire roots.
    expect(isPackageInstalled("./test-support/evaluation-witness.mjs")).toBe(true);
    expect(globals.__ratelEvaluationWitness).toBeUndefined();
  });
});
