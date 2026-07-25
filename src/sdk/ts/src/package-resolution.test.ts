import { describe, expect, it } from "vitest";
import { isModuleNotFound, isPackageInstalled } from "./package-resolution.js";

describe("isPackageInstalled", () => {
  it("detects an installed package and an absent one", () => {
    // @opentelemetry/api is a real dependency, so it resolves; a nonsense name does not.
    expect(isPackageInstalled("@opentelemetry/api")).toBe(true);
    expect(isPackageInstalled("@ratel-ai/definitely-not-a-real-package")).toBe(false);
  });

  it("resolves without evaluating the package", () => {
    // The probe must stay side-effect-free: it answers "is it in the tree", so a module
    // that throws (or spawns, or registers globals) on evaluation must not run. This
    // fixture throws immediately when imported — resolving it is fine, importing is not.
    // Specifier is relative to package-resolution.ts, where createRequire roots.
    expect(isPackageInstalled("./test-support/throws-on-import.mjs")).toBe(true);
  });
});

describe("isModuleNotFound", () => {
  it("is true only for a genuine module-not-found code", () => {
    expect(isModuleNotFound(Object.assign(new Error("x"), { code: "ERR_MODULE_NOT_FOUND" }))).toBe(
      true,
    );
    expect(isModuleNotFound(Object.assign(new Error("x"), { code: "MODULE_NOT_FOUND" }))).toBe(
      true,
    );
  });

  it("is false for a package that throws while loading, so it still reads as installed", () => {
    // The crux: a *present* package that fails for its own reasons must not be
    // misreported as absent — that would send the caller to install what they already have.
    expect(isModuleNotFound(Object.assign(new Error("boom"), { code: "ERR_DLOPEN_FAILED" }))).toBe(
      false,
    );
    expect(isModuleNotFound(new Error("plain error, no code"))).toBe(false);
    expect(isModuleNotFound(undefined)).toBe(false);
  });
});
