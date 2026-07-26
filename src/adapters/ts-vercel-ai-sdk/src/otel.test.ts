import { OpenTelemetry } from "@ai-sdk/otel";
import { context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { Origin, RATEL_ORIGIN } from "@ratel-ai/telemetry";
import * as ai from "ai";
import { generateText, type LanguageModel, tool } from "ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { z } from "zod";
import { RatelOtelIntegration } from "./otel.js";
import {
  aiSdkMajor,
  MockLanguageModelV2,
  type ModelCall,
  usage,
} from "./test-support/mock-model.js";

// The integration seam (`registerTelemetry`, `telemetry.integrations`) is ai@7
// only, and the compat matrix runs this suite against ai@5 and ai@6 too. A
// namespace import reads the export without failing to link when it's absent,
// so those rows can skip the suite instead of erroring on it.
const registerTelemetry = (ai as { registerTelemetry?: (...integrations: unknown[]) => void })
  .registerTelemetry;

/**
 * The integration is verified the way a host deployment wires it: the *test*
 * owns the provider, registers an in-memory exporter on it, and reads the spans
 * back. `RatelOtelIntegration` never registers a provider and never exports —
 * it only creates spans onto whatever provider is globally active. No network,
 * no Ratel Cloud import.
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

const IN_TOOL_SPAN = "inside tool execute";
const IN_MODEL_SPAN = "inside model call";

/**
 * Opens {@link IN_MODEL_SPAN} from inside `doGenerate` — the one place reached
 * only by the context `executeLanguageModelCall` activates.
 */
class ChildSpanningModel extends MockLanguageModelV2 {
  async doGenerate(options: ModelCall): Promise<unknown> {
    emitChildSpan(IN_MODEL_SPAN);
    return super.doGenerate(options);
  }
}

/** A span opened from inside the AI SDK's own work, so it records whatever parented it. */
function emitChildSpan(name: string): void {
  trace.getTracer("host-work").startActiveSpan(name, (span) => {
    span.end();
  });
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

/** A tool-calling step in which both the model and the tool open a child span. */
function runToolCallingStep(): Promise<unknown> {
  return generateText({
    model: new ChildSpanningModel(
      toolCallThenText("lookup", { q: "ratel" }),
    ) as unknown as LanguageModel,
    tools: {
      lookup: tool({
        description: "Look something up.",
        inputSchema: z.object({ q: z.string() }),
        execute: async () => {
          emitChildSpan(IN_TOOL_SPAN);
          return { hit: true };
        },
      }),
    },
    prompt: "look up ratel",
    telemetry: { integrations: [new RatelOtelIntegration()] },
  });
}

/** Every span the run produced, in export order. */
function spans(): ReadableSpan[] {
  return exporter.getFinishedSpans();
}

function spanNamed(name: string): ReadableSpan | undefined {
  return spans().find((span) => span.name === name);
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

  it("keeps a host enrichSpan's attributes next to ratel.origin", async () => {
    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: {
        integrations: [
          new RatelOtelIntegration({ enrichSpan: () => ({ "deployment.environment": "staging" }) }),
        ],
      },
    });

    // Taking the hook over instead of composing with it costs the host every
    // attribute it wired, with no way back: the emitter is private and a second
    // integration would duplicate the spans rather than enrich them.
    const produced = spans();
    expect(produced.length).toBeGreaterThan(0);
    for (const span of produced) {
      expect(span.attributes["deployment.environment"]).toBe("staging");
      expect(span.attributes[RATEL_ORIGIN]).toBe("agent");
    }
  });

  it("keeps ratel.origin when a host enrichSpan writes that key too", async () => {
    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: {
        integrations: [
          new RatelOtelIntegration({ enrichSpan: () => ({ [RATEL_ORIGIN]: "direct" }) }),
        ],
      },
    });

    // Merge order is the contract: `ratel.origin` is Ratel vocabulary, so a host
    // hook that happens to write it must not decide what the overlay reports.
    const produced = spans();
    expect(produced.length).toBeGreaterThan(0);
    expect(produced.every((span) => span.attributes[RATEL_ORIGIN] === "agent")).toBe(true);
  });

  it("keeps ratel.origin when a host enrichSpan throws", async () => {
    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: {
        integrations: [
          new RatelOtelIntegration({
            enrichSpan: () => {
              throw new Error("host bug");
            },
          }),
        ],
      },
    });

    // The emitter's own try/catch discards the whole return value, so composing
    // through it unguarded would let a host bug strip the one attribute this
    // class exists to add — from every span, with emission still succeeding.
    const produced = spans();
    expect(produced.length).toBeGreaterThan(0);
    expect(produced.every((span) => span.attributes[RATEL_ORIGIN] === "agent")).toBe(true);
  });

  it("stamps the origin the caller asked for instead of the agent default", async () => {
    await generateText({
      model: new MockLanguageModelV2([textReply("hi")]) as unknown as LanguageModel,
      prompt: "hello",
      telemetry: { integrations: [new RatelOtelIntegration({ origin: Origin.Direct })] },
    });

    // `embed` / `embedMany` / `rerank` are host-driven, not synthesized by an
    // agent loop, so `agent` is only a default. This also pins the constructor's
    // merge order: reverse the spread and the host hook could strip the key.
    const produced = spans();
    expect(produced.length).toBeGreaterThan(0);
    expect(produced.every((span) => span.attributes[RATEL_ORIGIN] === "direct")).toBe(true);
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

  it("nests the whole run under one root, tool span included", async () => {
    await runToolCallingStep();

    const produced = spans();
    const roots = produced.filter((span) => span.parentSpanContext === undefined);
    expect(roots.map((span) => span.name)).toEqual(["invoke_agent mock-model"]);

    const step = produced.find((span) => span.name.startsWith("step "));
    expect(step).toBeDefined();
    expect(spanNamed("execute_tool lookup")?.parentSpanContext?.spanId).toBe(
      step?.spanContext().spanId,
    );
  });

  it("runs tool and model work inside the context each execute* wrapper activates", async () => {
    await runToolCallingStep();

    // `executeTool` / `executeLanguageModelCall` are context wrappers, not event
    // callbacks. Dropping one loses no event and moves no AI SDK span — it
    // silently unparents everything the host's own code opens underneath, so
    // only a span opened from inside that code can catch it.
    const toolSpan = spanNamed("execute_tool lookup");
    const modelSpan = spanNamed("chat mock-model");
    expect(toolSpan).toBeDefined();
    expect(modelSpan).toBeDefined();
    expect(spanNamed(IN_TOOL_SPAN)?.parentSpanContext?.spanId).toBe(toolSpan?.spanContext().spanId);
    expect(spanNamed(IN_MODEL_SPAN)?.parentSpanContext?.spanId).toBe(
      modelSpan?.spanContext().spanId,
    );
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
