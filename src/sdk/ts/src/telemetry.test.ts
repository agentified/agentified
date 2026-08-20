import { readFileSync } from "node:fs";
import { context, trace } from "@opentelemetry/api";
import { logs } from "@opentelemetry/api-logs";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  InMemoryLogRecordExporter,
  LoggerProvider,
  type ReadableLogRecord,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import canonicalize from "canonicalize";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FactCatalog } from "./experimental.js";
import {
  ContentCapture,
  clearContentCapture,
  type ExecutableTool,
  SkillCatalog,
  setContentCapture,
  ToolCatalog,
} from "./index.js";
import { recordAuthNeeded } from "./telemetry.js";

/**
 * Instrumentation is verified through the public OTel API: register an in-memory
 * exporter as the global provider, drive the SDK, and read the spans back. The
 * SDK code never imports the exporter — it emits to whatever provider is active,
 * exactly as a host deployment would wire it.
 */
let exporter: InMemorySpanExporter;
let logExporter: InMemoryLogRecordExporter;

const CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";
const EXPERIMENTAL_CATALOG_DEFINITIONS_ENV = "RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS";

interface CatalogDefinitionFixtureInput {
  kind: "tool";
  id: string;
  name: string;
  description: string;
  tags: string[];
  input_schema: ExecutableTool["inputSchema"];
  output_schema: ExecutableTool["outputSchema"];
  searchable_description: string;
  searchable_description_overridden: boolean;
}

interface CatalogCanonicalizationFixture {
  catalog_definition_canonicalization: {
    algorithm: string;
    canonicalizer_only_vectors: Array<{
      name: string;
      input: unknown;
      canonical: string;
    }>;
    vectors: Array<{
      name: string;
      input: CatalogDefinitionFixtureInput;
      canonical: string;
      input_schema_canonical: string;
      output_schema_canonical: string;
      sha256: string;
    }>;
    rejected_vectors: Array<{
      name: string;
      reason: "unsafe_integer";
      input: CatalogDefinitionFixtureInput;
    }>;
  };
}

const catalogCanonicalization = (
  JSON.parse(
    readFileSync(new URL("../../../telemetry/conformance/fixtures.json", import.meta.url), "utf8"),
  ) as CatalogCanonicalizationFixture
).catalog_definition_canonicalization;

beforeEach(() => {
  // Fresh exporter + provider each test. Don't shut the provider down in teardown
  // (that would also shut the exporter); just drop the global registration.
  exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  trace.setGlobalTracerProvider(provider);
  logExporter = new InMemoryLogRecordExporter();
  logs.setGlobalLoggerProvider(
    new LoggerProvider({
      processors: [new SimpleLogRecordProcessor({ exporter: logExporter })],
    }),
  );
});

afterEach(() => {
  delete process.env[CAPTURE_ENV];
  delete process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV];
  setContentCapture(null); // never leak a programmatic capture override across tests
  trace.disable(); // reset the global provider to the no-op default
  logs.disable();
});

const readFile: ExecutableTool = {
  id: "read_file",
  name: "read_file",
  description: "Read a file from local disk and return its textual contents.",
  inputSchema: { properties: { path: { type: "string" } } },
  outputSchema: { properties: { contents: { type: "string" } } },
  execute: async ({ path }) => ({ contents: `contents of ${path}` }),
};

const boom: ExecutableTool = {
  id: "boom",
  name: "boom",
  description: "Always throws, to exercise the error path.",
  inputSchema: { properties: {} },
  outputSchema: { properties: {} },
  execute: async () => {
    throw new Error("kaboom");
  },
};

// An MCP-proxied tool: `<server>__<tool>` id convention.
const gmailSend: ExecutableTool = {
  id: "gmail__send_email",
  name: "send_email",
  description: "Send an email through the Gmail upstream.",
  inputSchema: { properties: { to: { type: "string" } } },
  outputSchema: { properties: {} },
  execute: async () => ({ ok: true }),
};

/** All exported spans with the given name. */
function spansNamed(name: string): ReadableSpan[] {
  return exporter.getFinishedSpans().filter((s) => s.name === name);
}

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

function logEventsNamed(name: string): ReadableLogRecord[] {
  return logExporter.getFinishedLogRecords().filter((record) => record.eventName === name);
}

/** The single span event with the given name, or undefined. */
function eventNamed(span: ReadableSpan, name: string) {
  return span.events.find((e) => e.name === name);
}

const INFERENCE_DETAILS = "gen_ai.client.inference.operation.details";
const SEARCH_RESULTS = "ratel.search.results";

describe("execute_tool span", () => {
  it("wraps a tool invocation with gen_ai + ratel attributes", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/tmp/x" });

    const [span] = spansNamed("execute_tool read_file");
    expect(span, "one execute_tool span").toBeTruthy();
    expect(attrs(span)["gen_ai.operation.name"]).toBe("execute_tool");
    expect(attrs(span)["gen_ai.tool.name"]).toBe("read_file");
    expect(attrs(span)["ratel.tool.args_size_bytes"]).toBeGreaterThan(0);
    expect(span.status.code).toBe(1); // OK
  });

  it("does not capture argument/result content by default", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/secret" });

    const [span] = spansNamed("execute_tool read_file");
    expect(attrs(span)["gen_ai.tool.call.arguments"]).toBeUndefined();
    expect(attrs(span)["gen_ai.tool.call.result"]).toBeUndefined();
    expect(logEventsNamed("ratel.tool.execution.details")).toHaveLength(0);
    expect(logEventsNamed("ratel.catalog.definition")).toHaveLength(0);
  });

  it("emits gated tool, skill, and fact definitions once per unchanged content hash", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV] = "true";
    const tools = new ToolCatalog();
    const skills = new SkillCatalog();
    const facts = new FactCatalog();

    await tools.register(readFile);
    await tools.register(readFile);
    const review = {
      id: "review",
      name: "review_code",
      description: "Review source",
      tags: ["quality"],
    };
    await skills.register(review);
    await facts.register({
      id: "address",
      name: "shop_address",
      description: "Where the shop is",
      tags: ["location"],
    });
    await skills.replaceAll([review]);
    await skills.replaceAll([{ ...review, description: "Review changed source" }]);

    const events = logEventsNamed("ratel.catalog.definition");
    expect(events).toHaveLength(4);
    expect(events.map((event) => event.attributes["ratel.catalog.kind"])).toEqual([
      "tool",
      "skill",
      "fact",
      "skill",
    ]);
    expect(events[0]?.attributes).toEqual({
      "ratel.catalog.kind": "tool",
      "ratel.catalog.id": "read_file",
      "ratel.catalog.name": "read_file",
      "ratel.catalog.description": "Read a file from local disk and return its textual contents.",
      "ratel.catalog.tags": [],
      "ratel.catalog.input_schema": '{"properties":{"path":{"type":"string"}}}',
      "ratel.catalog.output_schema": '{"properties":{"contents":{"type":"string"}}}',
      "ratel.catalog.searchable_description":
        "Read a file from local disk and return its textual contents.",
      "ratel.catalog.searchable_description_overridden": false,
      "ratel.catalog.content_hash":
        "a6135789c27ce3a9cb35a0ca2303133e20d29bd17aad90ebebe2b99fb4fcd0eb",
    });
    expect(events[3]?.attributes["ratel.catalog.id"]).toBe("review");
    expect(events[3]?.attributes["ratel.catalog.content_hash"]).not.toBe(
      events[1]?.attributes["ratel.catalog.content_hash"],
    );
  });

  it("omits oversized catalog schemas without dropping unrelated logs", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV] = "true";
    const catalog = new ToolCatalog();

    await catalog.register({
      id: "oversized",
      name: "oversized",
      description: "Oversized schema",
      inputSchema: { type: "string", description: "x".repeat(100_000) },
      outputSchema: { type: "object" },
      execute: () => undefined,
    });
    logs.getLogger("test").emit({ eventName: "unrelated", attributes: { ok: true } });

    const [event] = logEventsNamed("ratel.catalog.definition");
    expect(event?.attributes).toMatchObject({
      "ratel.catalog.kind": "tool",
      "ratel.catalog.id": "oversized",
      "ratel.catalog.output_schema": '{"type":"object"}',
      "ratel.catalog.schema_omitted": true,
      "ratel.catalog.content_hash": expect.stringMatching(/^[0-9a-f]{64}$/),
    });
    expect(event?.attributes).not.toHaveProperty("ratel.catalog.input_schema");
    expect(logEventsNamed("unrelated")).toHaveLength(1);
  });

  it("does not emit experimental catalog definitions from the generic content gate alone", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    const catalog = new ToolCatalog();

    await catalog.register(readFile);

    expect(logEventsNamed("ratel.catalog.definition")).toHaveLength(0);
  });

  it("hashes mixed-case schema keys identically to the Rust core and Python", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV] = "true";
    const catalog = new ToolCatalog();
    await catalog.register({
      id: "mixed_case",
      name: "mixed_case",
      description: "Mixed-case schema keys.",
      inputSchema: { properties: { B: { type: "string" }, a: { type: "string" } } },
      outputSchema: { properties: { ok: { type: "string" } } },
      execute: async () => ({ ok: "ok" }),
    });

    const [event] = logEventsNamed("ratel.catalog.definition");
    // Pinned via the Python twin's canonicalization, which the Rust core reproduces
    // byte-identically; byte order puts "B" before "a", locale collation does not.
    expect(event?.attributes["ratel.catalog.content_hash"]).toBe(
      "03542abe36f96c27db84a086337f7b66737c2480848193fde046e770b63231b8",
    );
  });

  it("matches shared RFC 8785 catalog-definition bytes, schema attributes, and hashes", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV] = "true";
    const catalog = new ToolCatalog();

    for (const vector of catalogCanonicalization.canonicalizer_only_vectors) {
      expect(canonicalize(vector.input), vector.name).toBe(vector.canonical);
    }
    for (const vector of catalogCanonicalization.vectors) {
      expect(canonicalize(vector.input), vector.name).toBe(vector.canonical);
      await catalog.register({
        id: vector.input.id,
        name: vector.input.name,
        description: vector.input.description,
        inputSchema: vector.input.input_schema,
        outputSchema: vector.input.output_schema,
        ...(vector.input.searchable_description_overridden
          ? { experimentalSearchableDescription: vector.input.searchable_description }
          : {}),
        execute: () => undefined,
      });
    }

    const events = logEventsNamed("ratel.catalog.definition");
    expect(events).toHaveLength(catalogCanonicalization.vectors.length);
    for (const [index, vector] of catalogCanonicalization.vectors.entries()) {
      expect(events[index]?.attributes, vector.name).toMatchObject({
        "ratel.catalog.input_schema": vector.input_schema_canonical,
        "ratel.catalog.output_schema": vector.output_schema_canonical,
        "ratel.catalog.content_hash": vector.sha256,
      });
    }
  });

  it("skips shared unsafe-integer definitions and emits after a safe edit", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV] = "true";
    const catalog = new ToolCatalog();

    for (const vector of catalogCanonicalization.rejected_vectors) {
      const definition = vector.input;
      await catalog.register({
        id: definition.id,
        name: definition.name,
        description: definition.description,
        inputSchema: definition.input_schema,
        outputSchema: definition.output_schema,
        execute: () => undefined,
      });
    }
    expect(logEventsNamed("ratel.catalog.definition")).toHaveLength(0);

    for (const vector of catalogCanonicalization.rejected_vectors) {
      const definition = vector.input;
      await catalog.register({
        id: definition.id,
        name: definition.name,
        description: definition.description,
        inputSchema: { type: "integer", maximum: Number.MAX_SAFE_INTEGER },
        outputSchema: definition.output_schema,
        execute: () => undefined,
      });
    }
    expect(logEventsNamed("ratel.catalog.definition")).toHaveLength(
      catalogCanonicalization.rejected_vectors.length,
    );
  });

  it.each([
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ])("rejects non-JSON schema number %s", async (number) => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    const catalog = new ToolCatalog();

    await expect(
      catalog.register({
        id: "invalid-number",
        name: "invalid-number",
        description: "Invalid number",
        inputSchema: { const: number },
        outputSchema: {},
        execute: () => undefined,
      }),
    ).rejects.toThrow();
  });

  it("under SPAN_AND_EVENT captures content on both the span and the event", async () => {
    process.env[CAPTURE_ENV] = "SPAN_AND_EVENT";
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/p" });

    const [span] = spansNamed("execute_tool read_file");
    expect(attrs(span)["gen_ai.tool.call.arguments"]).toBe('{"path":"/p"}');
    expect(attrs(span)["gen_ai.tool.call.result"]).toContain("contents of /p");
    // Dual emission: the structured EventRecord is present too.
    expect(eventNamed(span, INFERENCE_DETAILS)).toBeUndefined();
    const [event] = logEventsNamed("ratel.tool.execution.details");
    expect(event.attributes["gen_ai.tool.call.arguments"]).toEqual({ path: "/p" });
    expect(event.attributes["gen_ai.tool.call.result"]).toEqual({
      contents: "contents of /p",
    });
    expect(event.spanContext?.traceId).toBe(span.spanContext().traceId);
    expect(event.spanContext?.spanId).toBe(span.spanContext().spanId);
  });

  it("under SPAN_ONLY captures content on the span but emits no content event", async () => {
    process.env[CAPTURE_ENV] = "SPAN_ONLY";
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/p" });

    const [span] = spansNamed("execute_tool read_file");
    expect(attrs(span)["gen_ai.tool.call.arguments"]).toBe('{"path":"/p"}');
    expect(eventNamed(span, INFERENCE_DETAILS)).toBeUndefined();
    expect(logEventsNamed("ratel.tool.execution.details")).toHaveLength(0);
  });

  it("under EVENT_ONLY a failed tool emits arguments without a result", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    const catalog = new ToolCatalog();
    await catalog.register(boom);
    await expect(catalog.invoke("boom", { x: 1 })).rejects.toThrow("kaboom");

    const [span] = spansNamed("execute_tool boom");
    expect(span.status.code).toBe(2); // ERROR
    expect(eventNamed(span, INFERENCE_DETAILS)).toBeUndefined();
    const [event] = logEventsNamed("ratel.tool.execution.details");
    expect(event.attributes["gen_ai.tool.call.arguments"]).toEqual({ x: 1 });
    expect(event.attributes["gen_ai.tool.call.result"]).toBeUndefined();
  });

  it("under EVENT_ONLY emits a content event and keeps content off the span", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/p" });

    const [span] = spansNamed("execute_tool read_file");
    // Content rides the event, not span attributes.
    expect(attrs(span)["gen_ai.tool.call.arguments"]).toBeUndefined();
    expect(attrs(span)["gen_ai.tool.call.result"]).toBeUndefined();

    expect(eventNamed(span, INFERENCE_DETAILS)).toBeUndefined();
    const [event] = logEventsNamed("ratel.tool.execution.details");
    expect(event, "ratel.tool.execution.details EventRecord").toBeTruthy();
    expect(event.attributes["gen_ai.operation.name"]).toBe("execute_tool");
    expect(event.attributes["gen_ai.tool.name"]).toBe("read_file");
    expect(event.attributes["gen_ai.tool.call.arguments"]).toEqual({ path: "/p" });
    expect(event.attributes["gen_ai.tool.call.result"]).toEqual({
      contents: "contents of /p",
    });
    expect(event.attributes["gen_ai.input.messages"]).toBeUndefined();
    expect(event.attributes["gen_ai.output.messages"]).toBeUndefined();
  });

  it("keeps content off the span and emits no event under explicit NO_CONTENT", async () => {
    process.env[CAPTURE_ENV] = "NO_CONTENT";
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/p" });

    const [span] = spansNamed("execute_tool read_file");
    expect(attrs(span)["gen_ai.tool.call.arguments"]).toBeUndefined();
    expect(attrs(span)["gen_ai.tool.call.result"]).toBeUndefined();
    expect(eventNamed(span, INFERENCE_DETAILS)).toBeUndefined();
    expect(logEventsNamed("ratel.tool.execution.details")).toHaveLength(0);
  });

  it("captures content from a programmatic setContentCapture and stops once cleared", async () => {
    // No env var here: the programmatic override is the only thing opening the gate.
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    const generation = setContentCapture(ContentCapture.SpanOnly);
    await catalog.invoke("read_file", { path: "/p" });
    clearContentCapture(generation);
    await catalog.invoke("read_file", { path: "/p" });

    const [captured, cleared] = spansNamed("execute_tool read_file");
    expect(attrs(captured)["gen_ai.tool.call.arguments"]).toBe('{"path":"/p"}');
    expect(attrs(cleared)["gen_ai.tool.call.arguments"]).toBeUndefined();
  });

  it("records args_size_bytes as UTF-8 bytes, not UTF-16 characters", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    // "café" is 4 UTF-16 chars but 5 UTF-8 bytes; the JSON wrapper adds ASCII bytes.
    await catalog.invoke("read_file", { path: "café" });

    const [span] = spansNamed("execute_tool read_file");
    const expected = new TextEncoder().encode(JSON.stringify({ path: "café" })).length;
    expect(attrs(span)["ratel.tool.args_size_bytes"]).toBe(expected);
  });

  it("tags an MCP-proxied invoke with ratel.upstream.server and omits it for a plain tool", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(gmailSend);
    await catalog.register(readFile);
    await catalog.invoke("gmail__send_email", { to: "a@b.com" });
    await catalog.invoke("read_file", { path: "/x" });

    const [proxied] = spansNamed("execute_tool gmail__send_email");
    expect(attrs(proxied)["ratel.upstream.server"]).toBe("gmail");
    const [plain] = spansNamed("execute_tool read_file");
    expect(attrs(plain)["ratel.upstream.server"]).toBeUndefined();
  });

  it("marks the span ERROR and rethrows when the tool throws", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(boom);
    await expect(catalog.invoke("boom", {})).rejects.toThrow("kaboom");

    const [span] = spansNamed("execute_tool boom");
    expect(span.status.code).toBe(2); // ERROR
    expect(span.events.some((e) => e.name === "exception")).toBe(true);
  });

  it("keeps an AsyncIterable span open until iteration completes", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      id: "watch",
      execute: () =>
        (async function* () {
          yield { progress: 25 };
          yield { progress: 100 };
        })(),
    });

    const result = catalog.invokeRaw("watch", {});
    expect(spansNamed("execute_tool watch")).toHaveLength(0);

    const outputs: unknown[] = [];
    for await (const output of result as AsyncIterable<unknown>) outputs.push(output);

    expect(outputs).toEqual([{ progress: 25 }, { progress: 100 }]);
    const [span] = spansNamed("execute_tool watch");
    expect(span.status.code).toBe(1);
  });

  it("marks an AsyncIterable span ERROR when iteration throws", async () => {
    const catalog = new ToolCatalog();
    await catalog.register({
      ...readFile,
      id: "broken_watch",
      execute: () =>
        (async function* () {
          yield { progress: 25 };
          throw new Error("stream failed");
        })(),
    });

    const consume = async () => {
      for await (const _output of catalog.invokeRaw("broken_watch", {}) as AsyncIterable<unknown>) {
        // consume through the failure
      }
    };
    await expect(consume()).rejects.toThrow("stream failed");

    const [span] = spansNamed("execute_tool broken_watch");
    expect(span.status.code).toBe(2);
    expect(span.events.some((event) => event.name === "exception")).toBe(true);
  });

  it("ends the span as ERROR when AsyncIterable cancellation cleanup throws", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "s" } });
    await catalog.register({
      ...readFile,
      id: "broken_cleanup",
      execute: () => ({
        [Symbol.asyncIterator]() {
          return {
            next: async () => ({ done: false as const, value: { progress: 25 } }),
            return: async () => {
              throw new Error("cleanup failed");
            },
          };
        },
      }),
    });
    catalog.drainTraceEvents();

    const consumeOne = async () => {
      for await (const _output of catalog.invokeRaw(
        "broken_cleanup",
        {},
      ) as AsyncIterable<unknown>) {
        break;
      }
    };
    await expect(consumeOne()).rejects.toThrow("cleanup failed");

    const [span] = spansNamed("execute_tool broken_cleanup");
    expect(span, "one completed execute_tool span").toBeTruthy();
    expect(span.status.code).toBe(2);
    expect(span.events.some((event) => event.name === "exception")).toBe(true);
    expect(
      (catalog.drainTraceEvents() as Array<{ type: string }>).map((event) => event.type),
    ).toEqual(["invoke_start", "invoke_error"]);
  });

  it("leaves the local trace stream intact alongside the span", async () => {
    const catalog = new ToolCatalog({ trace: { kind: "memory", sessionId: "s" } });
    await catalog.register(readFile);
    await catalog.invoke("read_file", { path: "/tmp/x" });

    const local = catalog.drainTraceEvents() as Array<{ type: string }>;
    const invokeEvents = local.map((e) => e.type).filter((t) => t.startsWith("invoke_"));
    expect(invokeEvents).toEqual(["invoke_start", "invoke_end"]);
    expect(spansNamed("execute_tool read_file")).toHaveLength(1);
  });
});

describe("ratel.search span", () => {
  it("records target=tool with top_k, origin, and hit_count", async () => {
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    catalog.search("read file", 5, "agent");

    const [span] = spansNamed("ratel.search");
    expect(attrs(span)["ratel.search.target"]).toBe("tool");
    expect(attrs(span)["ratel.search.top_k"]).toBe(5);
    expect(attrs(span)["ratel.origin"]).toBe("agent");
    expect(attrs(span)["ratel.search.hit_count"]).toBeGreaterThan(0);
    expect(attrs(span)["ratel.search.query"]).toBeUndefined(); // content off by default
  });

  it("records target=skill for the skill catalog and captures the query when gated", async () => {
    process.env[CAPTURE_ENV] = "SPAN_ONLY";
    const skills = new SkillCatalog();
    await skills.register({
      id: "pdf",
      name: "pdf",
      description: "fill pdf forms",
      tags: [],
      body: "b",
      tools: [],
    });
    skills.search("pdf", 3);

    const [span] = spansNamed("ratel.search");
    expect(attrs(span)["ratel.search.target"]).toBe("skill");
    expect(attrs(span)["ratel.search.query"]).toBe("pdf");
    // SPAN_ONLY: query on the span, no results event.
    expect(eventNamed(span, SEARCH_RESULTS)).toBeUndefined();
    expect(logEventsNamed(SEARCH_RESULTS)).toHaveLength(0);
  });

  it("records target=fact for the fact catalog", async () => {
    // The `ratel.search.target` vocabulary has three values; without this the
    // fact catalog could drop `traceSearch` entirely and stay green.
    const facts = new FactCatalog();
    await facts.register({
      id: "shop-address",
      name: "shop-address",
      description: "where the barbershop is",
      body: "12 Baker Street",
      pin: "always",
    });
    facts.search("where is the shop", 4, "agent");

    const [span] = spansNamed("ratel.search");
    expect(attrs(span)["ratel.search.target"]).toBe("fact");
    expect(attrs(span)["ratel.search.top_k"]).toBe(4);
    expect(attrs(span)["ratel.origin"]).toBe("agent");
  });

  it("under EVENT_ONLY carries the query on a ratel.search.results event, not the span", async () => {
    process.env[CAPTURE_ENV] = "EVENT_ONLY";
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    catalog.search("read file", 5, "agent");

    const [span] = spansNamed("ratel.search");
    expect(attrs(span)["ratel.search.query"]).toBeUndefined(); // content off the span
    expect(eventNamed(span, SEARCH_RESULTS)).toBeUndefined();
    const [event] = logEventsNamed(SEARCH_RESULTS);
    expect(event, "ratel.search.results EventRecord").toBeTruthy();
    expect(event.attributes["ratel.search.query"]).toBe("read file");
  });

  it("under SPAN_AND_EVENT carries the query on both the span and the results event", async () => {
    process.env[CAPTURE_ENV] = "SPAN_AND_EVENT";
    const catalog = new ToolCatalog();
    await catalog.register(readFile);
    catalog.search("read file", 5, "agent");

    const [span] = spansNamed("ratel.search");
    expect(attrs(span)["ratel.search.query"]).toBe("read file");
    expect(eventNamed(span, SEARCH_RESULTS)).toBeUndefined();
    expect(logEventsNamed(SEARCH_RESULTS)[0]?.attributes["ratel.search.query"]).toBe("read file");
  });
});

describe("ratel.skill.load span", () => {
  it("wraps a skill load with the skill id", async () => {
    const skills = new SkillCatalog();
    await skills.register({
      id: "pdf",
      name: "pdf",
      description: "d",
      tags: [],
      body: "BODY",
      tools: [],
    });
    expect(skills.invoke("pdf")).toBe("BODY");

    const [span] = spansNamed("ratel.skill.load");
    expect(attrs(span)["ratel.skill.id"]).toBe("pdf");
    expect(span.status.code).toBe(1);
  });
});

describe("ratel.auth.flow span", () => {
  it("records outcome=needs_auth with the upstream server", () => {
    recordAuthNeeded("gmail");

    const [span] = spansNamed("ratel.auth.flow");
    expect(span, "one ratel.auth.flow span").toBeTruthy();
    expect(attrs(span)["ratel.auth.outcome"]).toBe("needs_auth");
    expect(attrs(span)["ratel.upstream.server"]).toBe("gmail");
  });

  it("omits the server attribute when the upstream is unknown", () => {
    recordAuthNeeded();

    const [span] = spansNamed("ratel.auth.flow");
    expect(attrs(span)["ratel.auth.outcome"]).toBe("needs_auth");
    expect(attrs(span)["ratel.upstream.server"]).toBeUndefined();
  });
});

describe("no provider configured", () => {
  it("is a no-op: operations still work and the wired exporter records nothing", async () => {
    // Drop the beforeEach provider; the OTel API now hands back non-recording spans.
    // `exporter` is that dropped provider's exporter, so if the SDK still reached it
    // (e.g. by caching a tracer) this would catch it.
    trace.disable();
    const catalog = new ToolCatalog();
    await catalog.register(readFile);

    expect(catalog.search("read", 5).length).toBeGreaterThan(0);
    await expect(catalog.invoke("read_file", { path: "/x" })).resolves.toEqual({
      contents: "contents of /x",
    });
    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });
});

describe("span nesting", () => {
  // The attribute/status tests above register no ContextManager, so context.active()
  // is always ROOT and every span is rootless — parent/child linkage is invisible to
  // them. Register a real AsyncLocalStorageContextManager here so a regression that
  // swaps startActiveSpan for a non-active startSpan (which would detach nested spans)
  // is actually caught.
  it("parents an inner span to the wrapping execute_tool span", async () => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    try {
      const catalog = new ToolCatalog();
      await catalog.register(readFile);
      // Executor triggers a nested ratel.search while the outer execute_tool span is active.
      await catalog.register({
        id: "outer",
        name: "outer",
        description: "invokes a nested search",
        inputSchema: { properties: {} },
        outputSchema: { properties: {} },
        execute: async () => {
          catalog.search("read", 3);
          return { ok: true };
        },
      });
      await catalog.invoke("outer", {});

      const [outer] = spansNamed("execute_tool outer");
      const [inner] = spansNamed("ratel.search");
      expect(outer, "outer execute_tool span").toBeTruthy();
      expect(inner, "inner ratel.search span").toBeTruthy();
      expect(inner.parentSpanContext?.spanId).toBe(outer.spanContext().spanId);
      expect(inner.spanContext().traceId).toBe(outer.spanContext().traceId);
    } finally {
      context.disable(); // drop the context manager so other tests keep ROOT context
    }
  });

  it("keeps the execute_tool context active while an AsyncIterable advances", async () => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    try {
      const catalog = new ToolCatalog();
      await catalog.register(readFile);
      await catalog.register({
        id: "stream_outer",
        name: "stream_outer",
        description: "streams after a nested search",
        inputSchema: { properties: {} },
        outputSchema: { properties: {} },
        execute: () =>
          (async function* () {
            catalog.search("read", 3);
            yield { ok: true };
          })(),
      });

      for await (const _output of catalog.invokeRaw("stream_outer", {}) as AsyncIterable<unknown>) {
        // consume the stream under test
      }

      const [outer] = spansNamed("execute_tool stream_outer");
      const [inner] = spansNamed("ratel.search");
      expect(inner.parentSpanContext?.spanId).toBe(outer.spanContext().spanId);
      expect(inner.spanContext().traceId).toBe(outer.spanContext().traceId);
    } finally {
      context.disable();
    }
  });
});
