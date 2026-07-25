import { OpenTelemetry } from "@ai-sdk/otel";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { RATEL_ORIGIN } from "@ratel-ai/telemetry";
import * as ai from "ai";
import { generateText, type LanguageModel, tool } from "ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { RatelOtelIntegration } from "./otel.js";

// The integration seam (`registerTelemetry`, `telemetry.integrations`) is ai@7
// only, and the compat matrix runs this suite against ai@5 and ai@6 too. A
// namespace import reads the export without failing to link when it's absent,
// so those rows can skip the suite instead of erroring on it.
const registerTelemetry = (ai as { registerTelemetry?: (...integrations: unknown[]) => void })
  .registerTelemetry;
// Skip on the row identity, never on feature detection: a self-fulfilling
// `skipIf(!seamFound)` would silently skip all of this on a v7 row the day `ai`
// moves the seam, and report green. Matches `generate-text.test.ts`.
const aiSdkMajor = Number.parseInt(process.env.AI_SDK_VERSION ?? "7", 10);

/**
 * The integration is verified the way a host deployment wires it: the *test*
 * owns the provider, registers an in-memory exporter on it, and reads the spans
 * back. `RatelOtelIntegration` never registers a provider and never exports —
 * it only creates spans onto whatever provider is globally active (plan
 * decision 1). No network, no Ratel Cloud import.
 */
let exporter: InMemorySpanExporter;

beforeEach(() => {
  exporter = new InMemorySpanExporter();
  trace.setGlobalTracerProvider(
    new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }),
  );
  // `@ai-sdk/otel` parents tool and model spans via `context.with`, which only
  // survives the SDK's awaits when a real context manager is installed.
  context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
});

afterEach(() => {
  // `registerTelemetry()` only ever pushes onto this array — calling it with no
  // arguments clears nothing — so the global registry has to be emptied by hand
  // or a later test inherits an integration bound to a disposed provider.
  (globalThis as { AI_SDK_TELEMETRY_INTEGRATIONS?: unknown[] }).AI_SDK_TELEMETRY_INTEGRATIONS = [];
  trace.disable();
  context.disable();
});

const usage = { inputTokens: 1, outputTokens: 1, totalTokens: 2 };

/**
 * `OpenTelemetry`'s own internals. TypeScript's `private` is erased, so these
 * sit on the runtime prototype next to the real hooks and have to be named to
 * partition it. Anything upstream adds that is on neither list is, by
 * definition, a hook nobody has decided about yet.
 */
const EMITTER_INTERNALS = [
  "getCallState",
  "cleanupCallState",
  "getSpanAttributes",
  "onGenerateStart",
  "onGenerateEnd",
  "onObjectOperationStart",
  "onObjectOperationEnd",
  "onEmbedOperationStart",
  "onEmbedOperationEnd",
  "onRerankOperationStart",
  "onRerankOperationEnd",
] as const;

/** The `Telemetry` surface `@ai-sdk/otel`'s emitter implements, and so must we. */
const TELEMETRY_MEMBERS = [
  "executeTool",
  "executeLanguageModelCall",
  "onStart",
  "onStepStart",
  "onLanguageModelCallStart",
  "onLanguageModelCallEnd",
  "onToolExecutionStart",
  "onToolExecutionEnd",
  "onStepEnd",
  "onStepFinish",
  "onObjectStepStart",
  "onObjectStepEnd",
  "onEnd",
  "onEmbedStart",
  "onEmbedEnd",
  "onRerankStart",
  "onRerankEnd",
  "onAbort",
  "onError",
] as const;

interface ModelCall {
  prompt: unknown[];
}

class MockLanguageModelV2 {
  readonly specificationVersion = "v2";
  readonly provider = "mock-provider";
  readonly modelId = "mock-model";
  readonly supportedUrls = {};

  constructor(private readonly generateResults: unknown[] = []) {}

  async doGenerate(options: ModelCall): Promise<unknown> {
    const index = this.callCount++;
    void options;
    return this.generateResults[index];
  }

  private callCount = 0;
}

function textReply(text: string): unknown {
  return {
    content: [{ type: "text", text }],
    finishReason: "stop",
    usage,
    warnings: [],
  };
}

function toolCallThenText(toolName: string, input: unknown): unknown[] {
  return [
    {
      content: [
        {
          type: "tool-call",
          toolCallId: "call-0",
          toolName,
          input: JSON.stringify(input),
        },
      ],
      finishReason: "tool-calls",
      usage,
      warnings: [],
    },
    textReply("done"),
  ];
}

/** Every span the run produced, in export order. */
function spans(): ReadableSpan[] {
  return exporter.getFinishedSpans();
}

describe.skipIf(aiSdkMajor < 7)("RatelOtelIntegration", () => {
  it("finds the ai@7 telemetry seam this whole suite is written against", () => {
    // The load-bearing assertion for every skip decision below: on a v7 row this
    // fails loudly if `ai` moves or renames the seam, instead of the suite
    // quietly evaporating.
    expect(typeof registerTelemetry, "ai@7 no longer exports registerTelemetry").toBe("function");
  });

  it("stamps ratel.origin on every span the AI SDK emits", async () => {
    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: { integrations: [new RatelOtelIntegration()] },
    });

    const produced = spans();
    expect(produced.length).toBeGreaterThan(0);
    for (const span of produced) {
      expect(span.attributes[RATEL_ORIGIN]).toBe("agent");
    }
  });

  it("leaves the AI SDK's own gen_ai.* semantics intact", async () => {
    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: { integrations: [new RatelOtelIntegration()] },
    });

    // Delegation is only useful if the emitter's own attributes still arrive;
    // a hand-rolled emitter would silently drop them.
    const genAi = spans().filter((span) =>
      Object.keys(span.attributes).some((key) => key.startsWith("gen_ai.")),
    );
    expect(genAi.length).toBeGreaterThan(0);
    expect(genAi.some((span) => span.attributes["gen_ai.request.model"] === "mock-model")).toBe(
      true,
    );
  });

  it("forwards the execute* context hooks so tool spans nest under the run", async () => {
    await generateText({
      model: new MockLanguageModelV2(
        toolCallThenText("lookup", { q: "ratel" }),
      ) as unknown as LanguageModel,
      tools: {
        lookup: tool({
          description: "Look something up.",
          inputSchema: z.object({ q: z.string() }),
          execute: async () => ({ hit: true }),
        }),
      },
      prompt: "look up ratel",
      telemetry: { integrations: [new RatelOtelIntegration()] },
    });

    // `executeTool` / `executeLanguageModelCall` are context wrappers, not event
    // callbacks: drop them and the tool span silently reparents to the root (or
    // becomes a second root), which no attribute assertion would catch.
    const produced = spans();
    const roots = produced.filter((span) => span.parentSpanContext === undefined);
    expect(roots.map((span) => span.name)).toEqual(["invoke_agent mock-model"]);

    const step = produced.find((span) => span.name.startsWith("step "));
    const toolSpan = produced.find((span) => span.name === "execute_tool lookup");
    expect(step).toBeDefined();
    expect(toolSpan?.parentSpanContext?.spanId).toBe(step?.spanContext().spanId);
  });

  it("emits onto a caller-supplied tracer instead of the global one", async () => {
    const isolated = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(isolated)],
    });

    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: {
        integrations: [new RatelOtelIntegration({ tracer: provider.getTracer("host") })],
      },
    });

    expect(isolated.getFinishedSpans().length).toBeGreaterThan(0);
    expect(spans()).toHaveLength(0);
  });

  it("forwards every Telemetry member the embedded emitter implements", () => {
    const integration = new RatelOtelIntegration() as unknown as Record<string, unknown>;

    // A dropped hook is invisible at runtime — the AI SDK just stops calling it,
    // and the spans quietly go missing. `Telemetry`'s members are all optional,
    // so the typechecker won't catch it either.
    for (const name of TELEMETRY_MEMBERS) {
      expect(typeof integration[name], `missing delegate for ${name}`).toBe("function");
    }
  });

  it("accounts for the emitter's entire prototype, so a new upstream hook fails here", () => {
    // Checking only the names we already forward can never catch the dangerous
    // case: `ai` adds a `Telemetry` member, `@ai-sdk/otel` implements it, and we
    // silently don't. Partitioning the whole prototype turns that into a failure
    // on the next dependency bump, which is the only moment anyone can act on it.
    const actual = Object.getOwnPropertyNames(OpenTelemetry.prototype)
      .filter((name) => name !== "constructor")
      .sort();
    const accounted = [...TELEMETRY_MEMBERS, ...EMITTER_INTERNALS].sort();

    expect(actual).toEqual(accounted);
  });

  it("works through the global registerTelemetry path", async () => {
    registerTelemetry?.(new RatelOtelIntegration());

    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
    });

    const produced = spans();
    expect(produced.length).toBeGreaterThan(0);
    expect(produced.every((span) => span.attributes[RATEL_ORIGIN] === "agent")).toBe(true);
  });
});
