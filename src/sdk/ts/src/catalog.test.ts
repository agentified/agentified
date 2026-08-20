import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { EmbedderError, type ExecutableTool, ToolCatalog } from "./index.js";
import { startDelayedEmbeddingServer } from "./test-support/delayed-embedding-server.js";

async function expectTimerBefore<T>(operation: Promise<T>): Promise<T> {
  const first = await Promise.race([
    operation.then(() => "operation" as const),
    new Promise<"timer">((resolve) => setTimeout(() => resolve("timer"), 20)),
  ]);
  expect(first).toBe("timer");
  return operation;
}

const huggingFaceHub =
  process.env.HF_HUB_CACHE ??
  join(process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"), "hub");
const cachedCandleModel = join(
  huggingFaceHub,
  "models--BAAI--bge-small-en-v1.5",
  "snapshots",
  "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
);
const hasCachedCandleModel =
  existsSync(join(cachedCandleModel, "config.json")) &&
  existsSync(join(cachedCandleModel, "tokenizer.json")) &&
  (existsSync(join(cachedCandleModel, "model.safetensors")) ||
    existsSync(join(cachedCandleModel, "pytorch_model.bin")));

const readFile: ExecutableTool = {
  id: "read_file",
  name: "read_file",
  description: "Read a file from local disk and return its textual contents.",
  inputSchema: {
    properties: {
      path: { type: "string", description: "absolute path to the file" },
    },
  },
  outputSchema: {
    properties: { contents: { type: "string" } },
  },
  execute: async ({ path }) => ({ contents: `contents of ${path}` }),
};

describe("ToolCatalog", () => {
  it("returns no hits from an empty catalog", () => {
    const catalog = new ToolCatalog();
    expect(catalog.search("anything", 5)).toEqual([]);
  });

  it("registers a tool and finds it by name", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    const hits = catalog.search("read file", 5);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].toolId).toBe("read_file");
    expect(hits[0].score).toBeGreaterThan(0);
  });

  it("ranks a tool by experimentalSearchableDescription instead of its description", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      id: "tool",
      name: "tool",
      description: "composedonlyterm",
      experimentalSearchableDescription: "overrideonlyterm",
    });

    expect(catalog.search("overrideonlyterm", 5).map((hit) => hit.toolId)).toEqual(["tool"]);
    expect(catalog.search("composedonlyterm", 5)).toEqual([]);
  });

  it("registers an iterable of tools as one batch", async () => {
    const catalog = new ToolCatalog();
    await catalog.register([
      readFile,
      {
        ...readFile,
        id: "send_email",
        name: "send_email",
        description: "Send an email message to a recipient.",
      },
    ]);

    expect(catalog.has("read_file")).toBe(true);
    expect(catalog.has("send_email")).toBe(true);
  });

  it("rejects a tool without an execute handler; nothing commits", async () => {
    const catalog = new ToolCatalog();
    await expect(
      catalog.register({
        id: "x",
        name: "x",
        description: "d",
        inputSchema: {},
        outputSchema: {},
        execute: undefined as never,
      }),
    ).rejects.toThrow(/no execute handler/);
    expect(catalog.has("x")).toBe(false);
  });

  it("a validation failure across a batch commits nothing", async () => {
    const catalog = new ToolCatalog();
    const invalid = { ...readFile, id: "invalid", execute: undefined as never };
    await expect(catalog.register([readFile, invalid])).rejects.toThrow(/no execute handler/);
    expect(catalog.has("read_file")).toBe(false);
    expect(catalog.search("read a file", 5)).toEqual([]);
  });

  it("invokes a registered tool by id with args", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    const result = await catalog.invoke("read_file", { path: "/tmp/x" });
    expect(result).toEqual({ contents: "contents of /tmp/x" });
  });

  it("validates and transforms input before invoking the executor", async () => {
    const catalog = new ToolCatalog();
    let received: unknown;
    await catalog.register({
      ...readFile,
      validateInput: async (input) => ({
        success: true,
        value: String((input as { path: unknown }).path).trim(),
      }),
      execute: (input) => {
        received = input;
        return { ok: true };
      },
    });

    await catalog.invoke("read_file", { path: " /tmp/x " });

    expect(received).toBe("/tmp/x");
    expect(catalog.getExecutable("read_file")?.validateInput).toBeTypeOf("function");
  });

  it("clears a framework validator when native registration replaces the tool", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      validateInput: async () => ({
        success: false,
        error: new Error("stale validator"),
      }),
    });
    let received: unknown;
    await catalog.register({
      ...readFile,
      execute: (input) => {
        received = input;
        return { ok: true };
      },
    });

    await catalog.invoke("read_file", { path: "/tmp/new" });

    expect(received).toEqual({ path: "/tmp/new" });
    expect(catalog.getExecutable("read_file")?.validateInput).toBeUndefined();
  });

  it("keeps legacy executors at one argument when no invocation context is supplied", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      execute: (...received) => ({ argumentCount: received.length }),
    });

    expect(await catalog.invoke("read_file", {})).toEqual({ argumentCount: 1 });
  });

  it("forwards an invocation context to the registered executor by identity", async () => {
    const context = { adapter: "test", value: { tenantId: "tenant-42" } };
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      execute: (_input, receivedContext) => ({ sameContext: receivedContext === context }),
    });

    expect(await catalog.invoke("read_file", {}, context)).toEqual({ sameContext: true });
  });

  it("throws on invoke of an unknown tool id", async () => {
    const catalog = new ToolCatalog();
    await expect(catalog.invoke("nope", {})).rejects.toThrow(/unknown toolId: nope/);
  });

  it("get(id) returns metadata; has(id) reports membership", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    expect(catalog.has("read_file")).toBe(true);
    expect(catalog.has("missing")).toBe(false);
    const tool = catalog.get("read_file");
    expect(tool?.description).toContain("Read a file");
  });

  it("getExecutable(id) returns metadata + execute together", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    const exec = catalog.getExecutable("read_file");
    expect(exec).toBeDefined();
    expect(exec?.id).toBe("read_file");
    const result = await exec?.execute({ path: "/etc/hosts" });
    expect(result).toEqual({ contents: "contents of /etc/hosts" });
  });
});

describe("ToolCatalog removed methods", () => {
  it("registerMany / buildEmbeddings / rebuildEmbeddings are gone at runtime", () => {
    // Folded into the variadic, self-embedding `register` (RAT-379/async-register).
    const catalog = new ToolCatalog() as unknown as Record<string, unknown>;
    expect(catalog.registerMany).toBeUndefined();
    expect(catalog.buildEmbeddings).toBeUndefined();
    expect(catalog.rebuildEmbeddings).toBeUndefined();
  });
});

describe("ToolCatalog search methods", () => {
  // Semantic/hybrid load a real model (network) and are covered in Rust; these
  // stay offline except where a working (fake) endpoint is needed to prove the
  // end-to-end embed-then-search effect.
  it("registers on a semantic catalog: embeds inline and search_async finds the hit", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding: { url: server.url, model: "test-model" },
      });
      await catalog.register(readFile);

      expect(catalog.has("read_file")).toBe(true);
      const hits = await catalog.searchAsync("read a file", 5);
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("surfaces embedding load failures through register, but metadata persists", async () => {
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });

    await expect(catalog.register(readFile)).rejects.toThrow(/failed to load embedding model/);
    // Metadata registration happens before the embedding pass inside `register`,
    // so it persists even though the embed itself failed.
    expect(catalog.has("read_file")).toBe(true);
  });

  it("surfaces a load failure as a typed EmbedderError with a stable code", async () => {
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });
    const error = await catalog.register(readFile).then(
      () => undefined,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(EmbedderError);
    expect((error as EmbedderError).code).toBe("Load");
    // Original message preserved so message-based matchers keep working.
    expect((error as EmbedderError).message).toMatch(/failed to load embedding model/);
  });

  it("keeps dense search behind the asynchronous API", () => {
    // search() rejects a resolved semantic/hybrid method before ever touching
    // the registry, so this needs no registration (and no working model).
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });
    expect(() => catalog.search("read", 5)).toThrow(/searchAsync/);
  });

  it("rejects a semantic override on a bm25 catalog with no embeddings built", async () => {
    // A bm25-default catalog's register never embeds; overriding searchAsync's
    // method to semantic hits the "not built" guard rather than a silent full scan.
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await expect(catalog.searchAsync("read", 5, "direct", "semantic")).rejects.toThrow(
      /not computed for semantic/,
    );
  });

  it("enriches the not-built dense error with an await-register hint", async () => {
    // The same failure a forgotten `await catalog.register(...)` produces on a
    // semantic catalog: the corpus is unembedded, so a dense search enriches the
    // core "not computed" error with an actionable await hint (original message
    // preserved). Non-breaking: it only augments an already-failing path.
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await expect(catalog.searchAsync("read", 5, "direct", "semantic")).rejects.toThrow(
      /without awaiting it/,
    );
  });

  it("keeps Node timers responsive during registration and query on a semantic catalog", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding: { url: server.url, model: "test-model" },
      });

      await expectTimerBefore(
        catalog.register([
          readFile,
          {
            ...readFile,
            id: "send_email",
            name: "send_email",
            description: "Send an email message to a recipient.",
          },
        ]),
      );
      expect(server.requests).toHaveLength(1);
      expect(server.requests[0]).toHaveLength(2);
      expect(server.requests[0]?.[0]).toContain("Read a file");
      expect(server.requests[0]?.[1]).toContain("Send an email");
      const hits = await expectTimerBefore(catalog.searchAsync("read a file", 5));
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("serializes queued dense searches and rejects registration until they settle", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding: { url: server.url, model: "test-model" },
      });
      await catalog.register(readFile);

      const started = Date.now();
      const first = catalog.searchAsync("read one", 5);
      const second = catalog.searchAsync("read two", 5);
      await expect(catalog.register({ ...readFile, id: "later" })).rejects.toThrow(
        /registry busy; await/,
      );
      await Promise.all([first, second]);
      expect(Date.now() - started).toBeGreaterThanOrEqual(200);
      await expect(catalog.register({ ...readFile, id: "later" })).resolves.toBeUndefined();
    } finally {
      await server.close();
    }
  });

  it.skipIf(!hasCachedCandleModel)(
    "keeps Node timers responsive during cached Candle model loading and inference",
    async () => {
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding: { local: cachedCandleModel, pooling: "cls" },
      });
      let ticks = 0;
      const heartbeat = setInterval(() => {
        ticks += 1;
      }, 1);
      try {
        await catalog.register(readFile);
      } finally {
        clearInterval(heartbeat);
      }
      expect(ticks).toBeGreaterThan(0);
    },
    60_000,
  );

  it("defaults to bm25 and never loads a model", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "m" } });
    await catalog.register(readFile);
    const hits = catalog.search("read file", 5);
    expect(hits[0]?.toolId).toBe("read_file");
    const events = catalog.drainTraceEvents() as Array<{
      type: string;
      stages?: { name: string }[];
    }>;
    const search = events.find((e) => e.type === "search");
    expect(search?.stages?.some((s) => s.name === "bm25")).toBe(true);
  });

  it("accepts an explicit per-call bm25 method matching the default", async () => {
    // Dense behavior is covered separately; this compares the synchronous BM25 path.
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    const viaDefault = catalog.search("read file", 5).map((h) => h.toolId);
    const viaExplicit = catalog.search("read file", 5, "direct", "bm25").map((h) => h.toolId);
    expect(viaExplicit).toEqual(viaDefault);
    expect(viaExplicit[0]).toBe("read_file");
  });

  it("rejects an unknown asynchronous method without making the registry busy", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    const search = catalog.searchAsync("read file", 5, "direct", "keyword" as never);
    const rejection = expect(search).rejects.toThrow(/unknown search method/);
    await expect(
      catalog.register({ ...readFile, id: "registered_after_invalid" }),
    ).resolves.toBeUndefined();
    await rejection;
  });

  it("per-call method overrides the catalog default and reroutes the engine", () => {
    // Default is semantic, but with no registrations no model loads. A per-call
    // "bm25" must route to the bm25 engine — provable offline via the trace stage
    // the semantic default (empty corpus) never emits.
    const catalog = new ToolCatalog({
      method: "semantic",
      trace: { kind: "memory", sessionId: "o" },
    });
    expect(() => catalog.search("anything", 5)).toThrow(/searchAsync/);
    catalog.search("anything", 5, "direct", "bm25"); // per-call override: bm25 engine
    const searches = (
      catalog.drainTraceEvents() as Array<{ type: string; stages?: { name: string }[] }>
    ).filter((e) => e.type === "search");
    expect(searches).toHaveLength(1);
    expect(searches[0].stages?.some((s) => s.name === "bm25")).toBe(true);
  });

  it("rejects an unknown method", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    expect(() =>
      // biome-ignore lint/suspicious/noExplicitAny: exercising the runtime guard
      catalog.search("read", 5, "direct", "keyword" as any),
    ).toThrow(/unknown search method/);
  });

  it("register([]) on a semantic catalog is an asynchronous no-op", async () => {
    // The empty batch short-circuits before any model load.
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });
    await expect(catalog.register([])).resolves.toBeUndefined();
    expect(catalog.has("anything")).toBe(false);
  });

  it("synchronous semantic override points to searchAsync", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    expect(() => catalog.search("read", 5, "direct", "semantic")).toThrow(/searchAsync/);
  });
});

describe("ToolCatalog embedding config", () => {
  it("accepts a bm25 catalog with no embedding (default model, no load)", () => {
    // Construction must not load a model; offline-safe.
    expect(() => new ToolCatalog()).not.toThrow();
  });

  it("rejects an explicitly empty embedding string", () => {
    expect(() => new ToolCatalog({ embedding: "" })).toThrow(/must not be blank/);
  });

  it("throws at construction on an invalid embedding config (bare url, no model)", () => {
    expect(
      () =>
        new ToolCatalog({ method: "semantic", embedding: "https://api.openai.com/v1/embeddings" }),
    ).toThrow(/model/);
  });

  it("rejects a bare repo-id string, pointing to the huggingface object", () => {
    // A string is a local path only; a repo id must use { huggingface }.
    expect(
      () => new ToolCatalog({ method: "semantic", embedding: "BAAI/bge-base-en-v1.5" }),
    ).toThrow(/huggingface/);
  });

  it("throws at construction on a conflicting endpoint config", () => {
    expect(
      () =>
        new ToolCatalog({
          method: "semantic",
          // ollama shorthand and an explicit url are contradictory
          embedding: {
            ollama: "nomic",
            url: "http://h:11434/v1/embeddings",
            model: "nomic",
          } as never,
        }),
    ).toThrow(/conflicting/);
  });

  it("accepts a pooling override and doc prefix on an in-process model", () => {
    expect(
      () =>
        new ToolCatalog({
          method: "semantic",
          embedding: { huggingface: "org/m", pooling: "mean", docPrefix: "passage: " },
        }),
    ).not.toThrow();
  });

  it("rejects an invalid pooling value at construction", () => {
    expect(
      () =>
        new ToolCatalog({
          method: "semantic",
          embedding: { huggingface: "org/m", pooling: "median" as never },
        }),
    ).toThrow(/pooling/);
  });

  it("rejects pooling on an endpoint (server pools)", () => {
    expect(
      () =>
        new ToolCatalog({
          method: "semantic",
          embedding: { ollama: "nomic", pooling: "mean" } as never,
        }),
    ).toThrow(/pooling/);
  });

  it("accepts an explicit download opt-in on a huggingface model", () => {
    expect(
      () =>
        new ToolCatalog({
          method: "semantic",
          embedding: { huggingface: "org/m", download: true },
        }),
    ).not.toThrow();
  });

  it("rejects download on a non-huggingface source", () => {
    expect(
      () =>
        new ToolCatalog({
          method: "semantic",
          embedding: { local: "/opt/models/x", download: true } as never,
        }),
    ).toThrow(/download/);
  });

  it("retains embedding configuration for a bm25 catalog without eager model load", async () => {
    // A bm25-default catalog never loads the model during register — even with a
    // config that would fail to load. Constructing a semantic catalog with the
    // very same (broken) config proves it is the mode, not the model spec, that
    // decides whether register embeds and can fail.
    const embedding = { local: "/definitely/missing/ratel-embedding-model" } as const;
    const catalog = new ToolCatalog({ embedding });
    await catalog.register(readFile);
    expect(catalog.search("read", 5)[0]?.toolId).toBe("read_file");

    const semanticCatalog = new ToolCatalog({ method: "semantic", embedding });
    await expect(semanticCatalog.register(readFile)).rejects.toThrow(
      /failed to load embedding model/,
    );
    expect(() => new ToolCatalog({ embedding: {} as never })).toThrow(/embedding source/);
  });
});

describe("ToolCatalog tracing", () => {
  it("does not capture events when no trace sink is configured (default noop)", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    catalog.search("read", 5);
    expect(catalog.drainTraceEvents()).toEqual([]);
  });

  it("captures index_churn on register and search on search via memory sink", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register(readFile);
    catalog.search("read", 5);

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const types = events.map((e) => e.type);
    expect(types).toContain("index_churn");
    expect(types).toContain("search");

    const search = events.find((e) => e.type === "search");
    expect(search?.origin).toBe("direct");
    expect((search?.hits as unknown[]).length).toBeGreaterThan(0);
  });

  it("emits invoke_start + invoke_end around a successful invoke", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register(readFile);
    catalog.drainTraceEvents(); // discard register/search noise

    await catalog.invoke("read_file", { path: "/x" });

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const types = events.map((e) => e.type);
    expect(types).toContain("invoke_start");
    expect(types).toContain("invoke_end");
    const start = events.find((e) => e.type === "invoke_start");
    expect(start?.tool_id).toBe("read_file");
    expect(typeof start?.args_size_bytes).toBe("number");
  });

  it("does not record opaque invocation context in the local trace", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register(readFile);
    catalog.drainTraceEvents();
    const context: { secret: string; self?: unknown } = { secret: "tenant-secret" };
    context.self = context;

    await catalog.invoke("read_file", { path: "/x" }, context);

    expect(JSON.stringify(catalog.drainTraceEvents())).not.toContain("tenant-secret");
  });

  it("emits invoke_error when the executor throws and re-throws to the caller", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register({
      id: "boom",
      name: "boom",
      description: "x",
      inputSchema: {},
      outputSchema: {},
      execute: async () => {
        throw new Error("kaboom");
      },
    });
    catalog.drainTraceEvents();

    await expect(catalog.invoke("boom", {})).rejects.toThrow(/kaboom/);

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const err = events.find((e) => e.type === "invoke_error");
    expect(err?.tool_id).toBe("boom");
    expect(err?.error).toMatch(/kaboom/);
  });

  it("search() defaults origin=direct; explicit origin=agent flows through to the event", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register(readFile);
    catalog.drainTraceEvents();

    catalog.search("read", 5, "agent");
    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const search = events.find((e) => e.type === "search");
    expect(search?.origin).toBe("agent");
  });

  it("re-registering an id replaces it in place — one hit, latest content wins", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      description: "Read a file from local disk.",
      execute: async () => ({ contents: "v1" }),
    });
    await catalog.register({
      ...readFile,
      description: "Fetch and return a document over the network.",
      execute: async () => ({ contents: "v2" }),
    });

    // Native corpus is deduped by id: the id ranks once, not twice (RAT-378).
    const hits = catalog.search("fetch a document over the network", 10);
    expect(hits.filter((h) => h.toolId === "read_file")).toHaveLength(1);
    // The latest executor wins.
    expect(await catalog.invoke("read_file", { path: "/x" })).toEqual({ contents: "v2" });
  });
});
