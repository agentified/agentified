import { isSpanContextValid, trace } from "@opentelemetry/api";
import { logs } from "@opentelemetry/api-logs";
import { describe, expect, it } from "vitest";
import * as sdk from "./index.js";
import {
  ContentCapture,
  clearContentCapture,
  type SearchHit,
  type Skill,
  SkillRegistry,
  setContentCapture,
  type Tool,
  ToolCatalog,
  ToolRegistry,
} from "./index.js";
import { startDelayedEmbeddingServer } from "./test-support/delayed-embedding-server.js";

const readFile: Tool = {
  id: "read_file",
  name: "read_file",
  description: "Read a file from local disk and return its textual contents.",
  inputSchema: {
    properties: {
      path: { type: "string", description: "absolute path to the file" },
      encoding: {
        type: "string",
        enum: ["utf8", "binary"],
        description: "how to decode the bytes",
      },
    },
  },
  outputSchema: {
    properties: {
      contents: { type: "string", description: "decoded file contents" },
    },
  },
};

const writeFile: Tool = {
  id: "write_file",
  name: "write_file",
  description: "Write textual contents to a file on local disk.",
  inputSchema: {
    properties: {
      path: { type: "string", description: "absolute path to the file" },
      contents: { type: "string", description: "bytes to write" },
    },
  },
  outputSchema: {},
};

const searchFiles: Tool = {
  id: "search_files",
  name: "search_files",
  description: "Grep across files in a directory using a regular expression.",
  inputSchema: {
    properties: {
      root: { type: "string", description: "directory to scan recursively" },
      pattern: { type: "string", description: "regular expression to match" },
    },
  },
  outputSchema: {},
};

const BM25_CONCURRENCY_CORPUS_SIZE = 50_000;

describe("ToolRegistry", () => {
  it("normalizes a public bare string embedding spec", () => {
    expect(() => new ToolRegistry("")).toThrow(/must not be blank/);
  });

  it("returns no hits from an empty registry", () => {
    const registry = new ToolRegistry();
    expect(registry.search("anything", 5)).toEqual([]);
  });

  it("finds a registered tool by name with a positive score", async () => {
    const registry = new ToolRegistry();
    await registry.register(readFile);

    const hits = registry.search("read file", 5);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].toolId).toBe("read_file");
    expect(hits[0].score).toBeGreaterThan(0);
  });

  it("indexes content nested inside inputSchema property descriptions", async () => {
    // Tool's only signal for "regular expression" lives inside inputSchema.properties.pattern.description.
    // Verifies the binding forwards serde_json::Value across the FFI without dropping nested fields.
    const registry = new ToolRegistry();
    await registry.register([readFile, writeFile, searchFiles]);

    const hits = registry.search("regular expression", 3);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].toolId).toBe("search_files");
  });

  it("bounds the result count by topK", async () => {
    const registry = new ToolRegistry();
    await registry.register([readFile, writeFile, searchFiles]);

    const hits = registry.search("file", 2);
    expect(hits.length).toBeLessThanOrEqual(2);
  });

  it("exposes hit fields in camelCase (toolId, score)", async () => {
    const registry = new ToolRegistry();
    await registry.register(readFile);

    const [hit] = registry.search("read file", 1);
    expect(hit).toBeDefined();
    const typedHit: SearchHit = hit;
    expect(typeof typedHit.toolId).toBe("string");
    expect(typeof typedHit.score).toBe("number");
    expect(typedHit.score).toBeGreaterThan(0);
  });

  it("rejects a semantic search when embeddings were never built (bm25-default registry)", async () => {
    const registry = new ToolRegistry({ local: "/definitely/missing/ratel-embedding-model" });
    await registry.register([readFile, writeFile]);

    expect(() => registry.searchWithMethod("read", 5, "direct", "semantic")).toThrow(
      /searchWithMethodAsync/,
    );
    await expect(registry.searchWithMethodAsync("read", 5, "direct", "semantic")).rejects.toThrow(
      /not computed for semantic/,
    );
  });

  it("surfaces embedding load failures through register on a semantic registry", async () => {
    const registry = new ToolRegistry(
      { local: "/definitely/missing/ratel-embedding-model" },
      "semantic",
    );
    await expect(registry.register(readFile)).rejects.toThrow(/failed to load embedding model/);
  });

  it("serializes the raw dense method alias as semantic work", async () => {
    const registry = new ToolRegistry({ local: "/definitely/missing/ratel-embedding-model" });
    await registry.register(readFile);

    const search = registry.searchWithMethodAsync("read", 5, "direct", "dense");
    const rejection = expect(search).rejects.toThrow(/not computed for semantic/);
    await expect(registry.register(writeFile)).rejects.toThrow(/registry busy; await/);
    await rejection;
  });

  it("rejects registration promptly instead of blocking behind active asynchronous bm25 reads", async () => {
    const registry = new ToolRegistry();
    await registry.register(
      Array.from({ length: BM25_CONCURRENCY_CORPUS_SIZE }, (_, index) => ({
        id: `concurrent_${index}`,
        name: `concurrent_${index}`,
        description: `Deploy database file search operation ${index}`,
        inputSchema: {},
        outputSchema: {},
      })),
    );

    const searches = Array.from({ length: 4 }, () =>
      registry.searchWithMethodAsync("deploy database file", 10, "direct", "bm25"),
    );
    await new Promise((resolve) => setTimeout(resolve, 25));

    const startedAt = performance.now();
    await expect(registry.register(writeFile)).rejects.toThrow(/registry busy; await/);
    expect(performance.now() - startedAt).toBeLessThan(100);

    await Promise.all(searches);
  }, 15_000);
});

describe("ToolRegistry removed methods", () => {
  it("registerMany / buildEmbeddings / rebuildEmbeddings are gone at runtime", () => {
    const registry = new ToolRegistry() as unknown as Record<string, unknown>;
    expect(registry.registerMany).toBeUndefined();
    expect(registry.buildEmbeddings).toBeUndefined();
    expect(registry.rebuildEmbeddings).toBeUndefined();
  });
});

describe("SkillRegistry", () => {
  it("embeds inline on register and search_async finds the hit", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const registry = new SkillRegistry({ url: server.url, model: "test-model" }, "semantic");
      const skill: Skill = {
        id: "api-design",
        name: "api-design",
        description: "Design a REST API.",
      };
      await registry.register(skill);

      const hits = await registry.searchWithMethodAsync("REST API", 5, "direct", "semantic");
      expect(hits[0]?.skillId).toBe("api-design");
    } finally {
      await server.close();
    }
  });

  it("surfaces embedding load failures through register", async () => {
    const registry = new SkillRegistry(
      { local: "/definitely/missing/ratel-embedding-model" },
      "semantic",
    );
    const skill: Skill = {
      id: "api-design",
      name: "api-design",
      description: "Design a REST API.",
    };
    await expect(registry.register(skill)).rejects.toThrow(/failed to load embedding model/);
  });
});

describe("SkillRegistry removed methods", () => {
  it("registerMany / buildEmbeddings / rebuildEmbeddings are gone at runtime", () => {
    const registry = new SkillRegistry() as unknown as Record<string, unknown>;
    expect(registry.registerMany).toBeUndefined();
    expect(registry.buildEmbeddings).toBeUndefined();
    expect(registry.rebuildEmbeddings).toBeUndefined();
  });
});

/**
 * Telemetry v2: the SDK ships no provider bootstrap. Hosts own the OTel provider
 * (`new NodeSDK({ spanProcessors })`), so the bootstrap entry points and the
 * optional-peer machinery behind them are gone. Emission and the content-capture
 * gate stay: they are the SDK's actual telemetry surface.
 */
describe("removed telemetry bootstrap", () => {
  const surface = sdk as unknown as Record<string, unknown>;

  it("no longer exports the bootstrap entry points", () => {
    expect(surface.startTelemetry).toBeUndefined();
    expect(surface.configureTelemetry).toBeUndefined();
  });

  it("keeps the content-capture gate", () => {
    expect(typeof sdk.setContentCapture).toBe("function");
    expect(typeof sdk.clearContentCapture).toBe("function");
    expect(sdk.ContentCapture).toBeDefined();
  });
});

/**
 * The other half of "the host owns the provider", stated positively. The
 * `no provider configured` case in telemetry.test.ts calls `trace.disable()`
 * first, so it passes even against an SDK that registers one; this case never
 * touches the OTel globals, so it fails the moment the SDK does.
 */
describe("host-owned OTel providers", () => {
  it("leaves the global providers at the OTel no-op default after driving the SDK", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({ ...readFile, execute: async () => ({ contents: "contents" }) });
    setContentCapture(ContentCapture.SpanAndEvent); // so the EventRecord path runs too
    try {
      catalog.search("read file", 1);
      await catalog.invoke("read_file", { path: "/x" });
    } finally {
      clearContentCapture();
    }

    // Both probes are indirect on purpose: a provider registered without processors
    // still yields a non-recording span and a disabled logger. What it cannot fake is
    // a real span context, or the forceFlush an SDK LoggerProvider has and the API's
    // default proxy does not.
    const span = trace.getTracer("probe").startSpan("probe");
    expect(isSpanContextValid(span.spanContext())).toBe(false);
    expect((logs.getLoggerProvider() as { forceFlush?: unknown }).forceFlush).toBeUndefined();
  });
});
