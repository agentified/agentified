import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Regression guard for the tree-shakeable split (ADR-0007): `@ratel-ai/telemetry`
// is the pure `ratel.*` vocabulary + config/gate and MUST stay OTel-free, so the
// SDK (emit side), the server (read side), and edge/serverless emitters can take
// it without pulling the OpenTelemetry SDK. This asserts the boundary at the two
// places that would silently re-couple it: a runtime dependency, or an import in
// a shipped source file.

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");

describe("@ratel-ai/telemetry stays OTel-free (tree-shakeable vocabulary)", () => {
  it("declares no @opentelemetry/* runtime dependency", () => {
    const pkg = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8"));
    const otel = Object.keys(pkg.dependencies ?? {}).filter((d) => d.startsWith("@opentelemetry/"));
    expect(otel).toEqual([]);
  });

  it("imports nothing from the OTel SDK in any shipped source file", () => {
    const shipped = readdirSync(here).filter((f) => f.endsWith(".ts") && !f.endsWith(".test.ts"));
    const offenders = shipped.filter((f) =>
      /from ["']@opentelemetry\//.test(readFileSync(join(here, f), "utf8")),
    );
    expect(offenders).toEqual([]);
  });

  // The published artifact is `dist/` (files: ["dist"]), so guard the actually-shipped
  // .js too: a stale build output (e.g. a left-over from a since-deleted source file)
  // could re-introduce an OTel import that the src scan above can't see. CI runs
  // `pnpm -r build` before `pnpm -r test`, so dist/ exists here; skip if unbuilt.
  it("ships no dist file that references the OTel SDK", () => {
    const dist = join(pkgRoot, "dist");
    if (!existsSync(dist)) return;
    const offenders: string[] = [];
    const walk = (dir: string): void => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, entry.name);
        if (entry.isDirectory()) walk(p);
        else if (entry.name.endsWith(".js") && /@opentelemetry\//.test(readFileSync(p, "utf8")))
          offenders.push(p.slice(pkgRoot.length + 1));
      }
    };
    walk(dist);
    expect(offenders).toEqual([]);
  });
});
