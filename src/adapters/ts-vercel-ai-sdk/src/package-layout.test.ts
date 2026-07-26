import { readFileSync } from "node:fs";
import ts from "typescript";
import { describe, expect, it } from "vitest";

interface PackageManifest {
  private?: boolean;
  license?: string;
  files?: string[];
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
  exports?: Record<string, unknown>;
  publishConfig?: { access?: string; provenance?: boolean };
}

const manifest = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
) as PackageManifest;

/**
 * Every package `entry` can pull in: walks relative imports transitively and
 * collects the bare specifiers found along the way. Specifiers resolve against
 * the importing file, not against `src/` — flattening them means a nested
 * `./aisdk.js` reads the wrong file, and the walk reports whatever that one
 * imports instead of missing it, which is a false green rather than an error.
 */
function packagesReachableFrom(entry: string): Set<string> {
  const visited = new Set<string>();
  const packages = new Set<string>();
  const queue = [new URL(entry, import.meta.url)];
  while (queue.length > 0) {
    const file = queue.shift() as URL;
    if (visited.has(file.href)) continue;
    visited.add(file.href);
    for (const specifier of specifiersIn(file)) {
      if (specifier.startsWith(".")) {
        queue.push(new URL(specifier.replace(/\.js$/, ".ts"), file));
      } else {
        packages.add(specifier);
      }
    }
  }
  return packages;
}

/**
 * Every module specifier `file` names, parsed rather than matched. A regex over
 * the source has to decide what is code and what is prose, and gets it wrong in
 * both directions: `from "…"` alone misses a bare `import "x"` and a dynamic
 * `import("x")`, while widening it to catch those starts matching the `import`
 * examples in this package's own docblocks. Stripping comments first only moves
 * the problem — `"https://"` in a string reads as a line comment and swallows a
 * real import to its right. The parser knows which is which.
 */
function specifiersIn(file: URL): string[] {
  const source = ts.createSourceFile(
    file.pathname,
    readFileSync(file, "utf8"),
    ts.ScriptTarget.ESNext,
    true,
  );
  const specifiers: string[] = [];
  const visit = (node: ts.Node): void => {
    if (
      (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
      node.moduleSpecifier !== undefined &&
      ts.isStringLiteral(node.moduleSpecifier)
    ) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword &&
      node.arguments[0] !== undefined &&
      ts.isStringLiteralLike(node.arguments[0])
    ) {
      specifiers.push(node.arguments[0].text);
    } else if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) {
      // `import("@ai-sdk/otel").Foo` never runs, but it still drags the package
      // into the host's *type* graph — which is where the duplicate-`ai` TS2403
      // this whole test guards against actually fires.
      const literal = node.argument.literal;
      if (ts.isStringLiteral(literal)) specifiers.push(literal.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  return specifiers;
}

describe("published vercel-ai-sdk dependency layout", () => {
  it("depends on nothing at runtime but the vocabulary `./otel` imports unconditionally", () => {
    // Everything the adapter touches (`ai`, `@ratel-ai/sdk`) is the host's to
    // provide, so it ships as peers. The one exception is deliberate:
    // `dist/otel.js` has a top-level `import ... from "@ratel-ai/telemetry"`
    // for the `ratel.*` constants. A peer (even a required one) can go unmet or
    // unhoisted and turn that into a load-time ERR_MODULE_NOT_FOUND, and no
    // typecheck would catch it — the constants never reach `otel.d.ts`. It is a
    // zero-dependency constants package with no `ai` of its own, so it carries
    // none of the duplicate-`ai` hazard that keeps `@ai-sdk/otel` a peer.
    expect(manifest.dependencies ?? {}).toEqual({ "@ratel-ai/telemetry": "workspace:^" });
  });

  it("peers on AI SDK 5–7 and the workspace SDK", () => {
    expect(manifest.peerDependencies?.ai).toBe("^5.0.0 || ^6.0.0 || ^7.0.0");
    expect(manifest.peerDependencies?.["@ratel-ai/sdk"]).toBe("workspace:^");
  });

  it("peers optionally on the OTel pieces RatelOtelIntegration delegates to", () => {
    // The host owns its OTel stack, exactly as it owns `ai`.
    expect(manifest.peerDependencies?.["@ai-sdk/otel"]).toBe("^1.0.0");
    expect(manifest.peerDependencies?.["@opentelemetry/api"]).toBe("^1.9.0");

    // Optional is load-bearing, not politeness. `@ai-sdk/otel` depends on an
    // exact `ai@7`, so a required peer makes every package manager install a
    // second `ai` next to an `ai@5`/`ai@6` host's own. Two `ai` copies in one
    // program redeclare `AI_SDK_DEFAULT_PROVIDER` with different types, and the
    // host's build fails (TS2403) without ever importing the integration.
    for (const name of ["@ai-sdk/otel", "@opentelemetry/api"]) {
      expect(manifest.peerDependenciesMeta?.[name]?.optional, `${name} must be optional`).toBe(
        true,
      );
    }
  });

  it("keeps the ai@7-only integration off the root entrypoint", () => {
    // Root stays importable with nothing but `ai` + the SDK installed. The
    // integration is a separate specifier so only hosts that ask for it resolve
    // `@ai-sdk/otel` — the same reason it is an optional peer.
    expect(manifest.exports?.["."]).toEqual({
      types: "./dist/index.d.ts",
      default: "./dist/index.js",
    });
    expect(manifest.exports?.["./otel"]).toEqual({
      types: "./dist/otel.d.ts",
      default: "./dist/otel.js",
    });

    // The exports map alone doesn't hold the line: a single `export * from
    // "./otel.js"` in the root would drag the optional peers into every host's
    // graph, and nothing else would notice — the repo tsconfig and both CI
    // consumer typechecks pass `--skipLibCheck`, which swallows the resulting
    // TS2307. So walk the root's real module graph.
    const rootPackages = packagesReachableFrom("index.ts");
    expect([...rootPackages].sort()).toEqual(["@ratel-ai/sdk", "ai"]);

    // ...and the integration's graph is the thing that legitimately reaches them.
    expect(packagesReachableFrom("otel.ts")).toContain("@ai-sdk/otel");
  });

  it("keeps the test doubles out of the build", () => {
    // `files: ["dist"]` publishes whatever `tsc` emits, and `src/test-support/`
    // is ordinary non-test source as far as the compiler is concerned — the
    // `*.test.ts` exclude does not cover it. Drop this entry and the mock model
    // ships in the tarball with every suite still green.
    const tsconfig = JSON.parse(
      readFileSync(new URL("../tsconfig.json", import.meta.url), "utf8"),
    ) as { exclude?: string[] };
    expect(tsconfig.exclude).toContain("src/test-support/**/*");
    expect(manifest.files).toContain("dist");
  });

  it("dev-pins the exact AI SDK release selected for this verification run", () => {
    // CI replaces this exact dev version per matrix row. The environment keeps
    // the assertion exact without hard-wiring every row to the default v7 pin.
    expect(manifest.devDependencies?.ai).toBe(process.env.AI_SDK_VERSION ?? "7.0.37");
    expect(manifest.devDependencies?.["@ratel-ai/sdk"]).toBe("workspace:^");
  });

  it("dev-pins @ai-sdk/otel to the release that carries the default ai pin", () => {
    // `@ai-sdk/otel@1.0.N` depends on `ai@7.0.N` exactly, so an unmatched pair
    // installs a second `ai` copy and the `Telemetry` types stop being identical.
    // Only the committed default row is checked: the compat matrix swaps `ai`
    // alone, and on its ai@5/6 rows `@ai-sdk/otel` deliberately keeps its own.
    expect(manifest.devDependencies?.["@ai-sdk/otel"]).toBe("1.0.37");
  });

  it("publishes public with provenance under MIT", () => {
    expect(manifest.private).toBe(false);
    expect(manifest.license).toBe("MIT");
    expect(manifest.publishConfig?.access).toBe("public");
    expect(manifest.publishConfig?.provenance).toBe(true);
  });
});
