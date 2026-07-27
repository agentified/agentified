import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerMcpServer, ToolCatalog } from "./index.js";

interface ServerToolSpec {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
  handler?: (args: Record<string, unknown>) => unknown;
}

interface FakeMcp {
  server: Server;
  clientTransport: InMemoryTransport;
}

interface PaginatedToolsPage {
  tools: ServerToolSpec[];
  nextCursor?: string;
}

async function startPaginatedFakeMcpServer(
  pages: PaginatedToolsPage[] | ((cursor: string | undefined) => PaginatedToolsPage | Error),
  serverOptions?: { instructions?: string },
): Promise<FakeMcp> {
  const resolvePage = (cursor: string | undefined): PaginatedToolsPage | Error => {
    if (typeof pages === "function") {
      return pages(cursor);
    }
    if (cursor === undefined) {
      return pages[0] ?? new Error("paginated fake MCP has no first page");
    }
    for (let i = 0; i < pages.length - 1; i++) {
      if (pages[i].nextCursor === cursor) {
        return pages[i + 1];
      }
    }
    return new Error(`unexpected tools/list cursor: ${cursor}`);
  };

  const toolByName = new Map<string, ServerToolSpec>();

  const server = new Server(
    { name: "fake-mcp", version: "0.0.0" },
    {
      capabilities: { tools: {} },
      ...(serverOptions?.instructions ? { instructions: serverOptions.instructions } : {}),
    },
  );

  server.setRequestHandler(ListToolsRequestSchema, async (req) => {
    const cursor = req.params?.cursor as string | undefined;
    const page = resolvePage(cursor);
    if (page instanceof Error) {
      throw page;
    }
    for (const spec of page.tools) {
      toolByName.set(spec.name, spec);
    }
    return {
      tools: page.tools.map((t) => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema ?? { type: "object" },
        ...(t.outputSchema ? { outputSchema: t.outputSchema } : {}),
      })),
      ...(page.nextCursor !== undefined ? { nextCursor: page.nextCursor } : {}),
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (req) => {
    const spec = toolByName.get(req.params.name);
    if (!spec) {
      throw new Error(`unknown tool: ${req.params.name}`);
    }
    const args = (req.params.arguments ?? {}) as Record<string, unknown>;
    const out = await (spec.handler ?? ((a) => a))(args);
    return {
      content: [{ type: "text", text: JSON.stringify(out) }],
      structuredContent: out as Record<string, unknown>,
    };
  });

  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  return { server, clientTransport };
}

async function startFakeMcpServer(
  tools: ServerToolSpec[],
  serverOptions?: { instructions?: string },
): Promise<FakeMcp> {
  return startPaginatedFakeMcpServer([{ tools }], serverOptions);
}

describe("registerMcpServer", () => {
  let fake: FakeMcp;

  beforeEach(async () => {
    fake = await startFakeMcpServer([
      {
        name: "read_file",
        description: "Read a file from the local disk and return its textual contents.",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string", description: "absolute path to the file" } },
          required: ["path"],
        },
        outputSchema: {
          type: "object",
          properties: { contents: { type: "string" } },
        },
        handler: ({ path }) => ({ contents: `contents of ${path as string}` }),
      },
    ]);
  });

  afterEach(async () => {
    await fake.server.close();
    trace.disable(); // drop any provider a span test registered globally
  });

  it("registers each upstream tool with a server-namespaced id", async () => {
    const catalog = new ToolCatalog();

    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });

    expect(handle.toolIds).toEqual(["demo__read_file"]);
    expect(catalog.has("demo__read_file")).toBe(true);

    await handle.close();
  });

  it("embeds during registration; a broken model surfaces the error from registerMcpServer, but metadata persists", async () => {
    // Embedding now happens inside `catalog.register` (RAT-379/async-register),
    // which `registerMcpServer` awaits — so a broken model surfaces the error
    // right out of `registerMcpServer` itself, not from a later, separate build.
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });

    await expect(
      registerMcpServer(catalog, { name: "demo", transport: fake.clientTransport }),
    ).rejects.toThrow(/failed to load embedding model/);

    // Metadata registration happens before the embedding pass inside
    // `catalog.register`, so it persists even though the embed itself failed.
    expect(catalog.has("demo__read_file")).toBe(true);
  });

  it("makes upstream tools discoverable via catalog.search using their description", async () => {
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });

    const hits = catalog.search("read a file from disk", 5);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].toolId).toBe("demo__read_file");

    await handle.close();
  });

  it("namespaces by server label so two servers with overlapping tool names don't collide", async () => {
    const otherFake = await startFakeMcpServer([
      {
        name: "read_file",
        description: "Read a remote file from the cloud bucket.",
        handler: ({ path }) => ({ remoteContents: `cloud:${path as string}` }),
      },
    ]);

    const catalog = new ToolCatalog();
    const localHandle = await registerMcpServer(catalog, {
      name: "local",
      transport: fake.clientTransport,
    });
    const cloudHandle = await registerMcpServer(catalog, {
      name: "cloud",
      transport: otherFake.clientTransport,
    });

    expect(catalog.has("local__read_file")).toBe(true);
    expect(catalog.has("cloud__read_file")).toBe(true);

    const localResult = (await catalog.invoke("local__read_file", {
      path: "/etc/hosts",
    })) as { structuredContent?: { contents?: string } };
    const cloudResult = (await catalog.invoke("cloud__read_file", {
      path: "/etc/hosts",
    })) as { structuredContent?: { remoteContents?: string } };

    expect(localResult.structuredContent?.contents).toBe("contents of /etc/hosts");
    expect(cloudResult.structuredContent?.remoteContents).toBe("cloud:/etc/hosts");

    await localHandle.close();
    await cloudHandle.close();
    await otherFake.server.close();
  });

  it("disconnects on handle.close() so subsequent invokes reject", async () => {
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });

    await handle.close();

    await expect(catalog.invoke("demo__read_file", { path: "/x" })).rejects.toThrow();
  });

  it("rejects from catalog.invoke when the upstream tool handler throws", async () => {
    const failing = await startFakeMcpServer([
      {
        name: "boom",
        description: "always fails",
        handler: () => {
          throw new Error("kaboom");
        },
      },
    ]);
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: failing.clientTransport,
    });

    await expect(catalog.invoke("demo__boom", {})).rejects.toThrow(/kaboom/);

    await handle.close();
    await failing.server.close();
  });

  it("surfaces the upstream server's `instructions` field on the handle", async () => {
    const withInstructions = await startFakeMcpServer([{ name: "x", description: "anything" }], {
      instructions: "Use this server for filesystem ops.",
    });
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "fs",
      transport: withInstructions.clientTransport,
    });
    expect(handle.serverInstructions).toBe("Use this server for filesystem ops.");
    await handle.close();
    await withInstructions.server.close();
  });

  it("returns serverInstructions as undefined when the upstream did not supply any", async () => {
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });
    expect(handle.serverInstructions).toBeUndefined();
    await handle.close();
  });

  it("emits upstream_register on connect and upstream_invoke on each call", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });
    catalog.drainTraceEvents().filter((e) => (e as { type: string }).type === "upstream_register");

    await catalog.invoke("demo__read_file", { path: "/etc/hosts" });

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const inv = events.find((e) => e.type === "upstream_invoke");
    expect(inv?.server).toBe("demo");
    expect(inv?.tool_id).toBe("demo__read_file");
    expect(typeof inv?.took_ms).toBe("number");

    await handle.close();
  });

  it("emits a ratel.upstream.register OTel span with server, transport, and tool_count", async () => {
    const exporter = new InMemorySpanExporter();
    trace.setGlobalTracerProvider(
      new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }),
    );
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });

    const [span] = exporter.getFinishedSpans().filter((s) => s.name === "ratel.upstream.register");
    expect(span, "one ratel.upstream.register span").toBeTruthy();
    const a = span.attributes as Record<string, unknown>;
    expect(a["ratel.upstream.server"]).toBe("demo");
    expect(a["ratel.upstream.tool_count"]).toBe(1);
    expect(typeof a["ratel.upstream.transport"]).toBe("string");
    expect(span.status.code).toBe(1); // OK

    await handle.close();
  });

  it("emits upstream_register at connect time with the tool count and a transport label", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const reg = events.find((e) => e.type === "upstream_register");
    expect(reg?.server).toBe("demo");
    expect(reg?.tool_count).toBe(1);
    expect(typeof reg?.transport).toBe("string");

    await handle.close();
  });

  it("emits upstream_error and re-throws when the upstream tool handler throws", async () => {
    const failing = await startFakeMcpServer([
      {
        name: "boom",
        description: "always fails",
        handler: () => {
          throw new Error("kaboom");
        },
      },
    ]);
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: failing.clientTransport,
    });
    catalog.drainTraceEvents();

    await expect(catalog.invoke("demo__boom", {})).rejects.toThrow();

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const err = events.find((e) => e.type === "upstream_error");
    expect(err?.server).toBe("demo");
    expect(err?.tool_id).toBe("demo__boom");

    await handle.close();
    await failing.server.close();
  });

  it("invokes the upstream tool via tools/call and returns its structured payload", async () => {
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: fake.clientTransport,
    });

    const result = (await catalog.invoke("demo__read_file", {
      path: "/etc/hosts",
    })) as { structuredContent?: { contents?: string } };

    expect(result.structuredContent).toEqual({ contents: "contents of /etc/hosts" });

    await handle.close();
  });
});

describe("registerMcpServer tools/list pagination", () => {
  afterEach(async () => {
    vi.restoreAllMocks();
    trace.disable();
  });

  it("registers tools from every page and reports the aggregated tool count", async () => {
    const paginated = await startPaginatedFakeMcpServer([
      {
        tools: [{ name: "page_one", description: "first page tool", handler: () => ({ page: 1 }) }],
        nextCursor: "page-2",
      },
      {
        tools: [
          { name: "page_two", description: "second page tool", handler: () => ({ page: 2 }) },
        ],
      },
    ]);
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: paginated.clientTransport,
    });

    expect(handle.toolIds).toEqual(["demo__page_one", "demo__page_two"]);
    expect(catalog.has("demo__page_one")).toBe(true);
    expect(catalog.has("demo__page_two")).toBe(true);

    const hits = catalog.search("second page tool", 5);
    expect(hits.some((h) => h.toolId === "demo__page_two")).toBe(true);

    const events = catalog.drainTraceEvents() as Array<Record<string, unknown>>;
    const reg = events.find((e) => e.type === "upstream_register");
    expect(reg?.tool_count).toBe(2);

    await handle.close();
    await paginated.server.close();
  });

  it("follows nextCursor across an empty tools page", async () => {
    const paginated = await startPaginatedFakeMcpServer([
      { tools: [], nextCursor: "after-empty" },
      {
        tools: [
          {
            name: "after_empty",
            description: "tool listed after an empty page",
            handler: () => ({ ok: true }),
          },
        ],
      },
    ]);
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: paginated.clientTransport,
    });

    expect(handle.toolIds).toEqual(["demo__after_empty"]);
    expect(catalog.has("demo__after_empty")).toBe(true);

    await handle.close();
    await paginated.server.close();
  });

  it("rejects when the server repeats the same nextCursor", async () => {
    const paginated = await startPaginatedFakeMcpServer([
      {
        tools: [{ name: "stuck", description: "stuck in pagination", handler: () => ({}) }],
        nextCursor: "loop",
      },
      {
        tools: [{ name: "again", description: "another page", handler: () => ({}) }],
        nextCursor: "loop",
      },
    ]);
    const catalog = new ToolCatalog();

    await expect(
      registerMcpServer(catalog, { name: "demo", transport: paginated.clientTransport }),
    ).rejects.toMatchObject({
      code: "RepeatedCursor",
      message: expect.stringMatching(/repeated nextCursor/i),
    });
    expect(catalog.has("demo__stuck")).toBe(false);

    await paginated.server.close();
  });

  it("rejects when tools/list exceeds the page cap", async () => {
    let pageNum = 0;
    const paginated = await startPaginatedFakeMcpServer((cursor) => {
      if (cursor !== undefined && !cursor.startsWith("cap-page-")) {
        return new Error(`unexpected cursor: ${cursor}`);
      }
      pageNum += 1;
      return {
        tools:
          pageNum === 1
            ? [{ name: "endless", description: "never finishes", handler: () => ({}) }]
            : [],
        nextCursor: `cap-page-${pageNum}`,
      };
    });
    const catalog = new ToolCatalog();

    await expect(
      registerMcpServer(catalog, { name: "demo", transport: paginated.clientTransport }),
    ).rejects.toMatchObject({
      code: "PaginationExceeded",
      message: expect.stringMatching(/exceeded 64 pages/i),
    });
    expect(catalog.has("demo__endless")).toBe(false);

    await paginated.server.close();
  });

  it("closes the MCP client when a later list page fails", async () => {
    const closeSpy = vi.spyOn(Client.prototype, "close").mockResolvedValue(undefined);
    const paginated = await startPaginatedFakeMcpServer((cursor) => {
      if (cursor === undefined) {
        return {
          tools: [{ name: "first_page", description: "page one", handler: () => ({}) }],
          nextCursor: "page-2",
        };
      }
      return new Error("list page two failed");
    });
    const catalog = new ToolCatalog();

    await expect(
      registerMcpServer(catalog, { name: "demo", transport: paginated.clientTransport }),
    ).rejects.toThrow(/list page two failed/);
    expect(closeSpy).toHaveBeenCalled();
    expect(catalog.has("demo__first_page")).toBe(false);

    await paginated.server.close();
  });

  it("succeeds with zero tools when the server returns an empty unpaginated list", async () => {
    const paginated = await startPaginatedFakeMcpServer([{ tools: [] }]);
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "t" } });
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: paginated.clientTransport,
    });

    expect(handle.toolIds).toEqual([]);
    const reg = (catalog.drainTraceEvents() as Array<Record<string, unknown>>).find(
      (e) => e.type === "upstream_register",
    );
    expect(reg?.tool_count).toBe(0);

    await handle.close();
    await paginated.server.close();
  });

  it("does not mutate the catalog when a later list page fails", async () => {
    const paginated = await startPaginatedFakeMcpServer((cursor) => {
      if (cursor === undefined) {
        return {
          tools: [{ name: "first_page", description: "only on page one", handler: () => ({}) }],
          nextCursor: "page-2",
        };
      }
      return new Error("list page two failed");
    });
    const catalog = new ToolCatalog();

    await expect(
      registerMcpServer(catalog, { name: "demo", transport: paginated.clientTransport }),
    ).rejects.toThrow(/list page two failed/);
    expect(catalog.has("demo__first_page")).toBe(false);

    const retryFake = await startPaginatedFakeMcpServer((cursor) => {
      if (cursor === undefined) {
        return {
          tools: [{ name: "first_page", description: "only on page one", handler: () => ({}) }],
        };
      }
      return new Error("unexpected cursor on retry");
    });
    const retry = await registerMcpServer(catalog, {
      name: "demo",
      transport: retryFake.clientTransport,
    });
    expect(retry.toolIds).toEqual(["demo__first_page"]);
    await retry.close();
    await paginated.server.close();
    await retryFake.server.close();
  });

  it("resolves duplicate tool names across pages to the last listing (catalog batch semantics)", async () => {
    const paginated = await startPaginatedFakeMcpServer([
      {
        tools: [
          {
            name: "dup",
            description: "first listing",
            handler: () => ({ version: "first" }),
          },
        ],
        nextCursor: "page-2",
      },
      {
        tools: [
          {
            name: "dup",
            description: "second listing wins",
            handler: () => ({ version: "second" }),
          },
        ],
      },
    ]);
    const catalog = new ToolCatalog();
    const handle = await registerMcpServer(catalog, {
      name: "demo",
      transport: paginated.clientTransport,
    });

    expect(handle.toolIds).toEqual(["demo__dup", "demo__dup"]);
    const result = (await catalog.invoke("demo__dup", {})) as {
      structuredContent?: { version?: string };
    };
    expect(result.structuredContent?.version).toBe("second");

    await handle.close();
    await paginated.server.close();
  });
});
