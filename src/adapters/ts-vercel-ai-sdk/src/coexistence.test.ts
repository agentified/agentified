import { LangfuseSpanProcessor } from "@langfuse/otel";
import { type Context, context, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  type ReadableSpan,
  SimpleSpanProcessor,
  type Span,
  type SpanExporter,
  type SpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { experimentalDefineExperiment, ratel } from "@ratel-ai/sdk";
import {
  EXECUTE_TOOL,
  RATEL_EXPERIMENT_ARM,
  RATEL_EXPERIMENT_ID,
  RATEL_EXPERIMENT_ROLE,
  RATEL_EXPERIMENT_SELECTION_ID,
  RATEL_EXPERIMENT_UNIT,
  RATEL_ORIGIN,
  RATEL_SEARCH,
} from "@ratel-ai/telemetry";
import * as ai from "ai";
import { generateText, type LanguageModel, tool } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { aiSdk } from "./aisdk.js";
import { RatelOtelIntegration } from "./otel.js";
import { aiSdkMajor, MockLanguageModelV2, usage } from "./test-support/mock-model.js";

/**
 * Coexistence proof for TS Telemetry v2 (RS-43): the SDK's `ratel.*` spans and
 * the AI SDK's `gen_ai.*` spans riding ONE host-owned provider out to several
 * destinations at once, including the real `@langfuse/otel` processor.
 *
 * In-memory capture only — no network — and no Ratel Cloud import: `@ratel-ai/cloud`
 * ships last precisely so the OSS path is proven against third-party backends
 * first. Everything here works with packages a host already runs.
 *
 * The host owns the provider. Ratel registers nothing.
 */

/** The `ai` integration seam is ai@7 only; the compat matrix runs this file on ai@5 and ai@6 too. */
const registerTelemetry = (ai as { registerTelemetry?: (...integrations: unknown[]) => void })
  .registerTelemetry;

/**
 * Emitter identity, which is `otel.scope.name` on the wire. Span *names* collide
 * across emitters (both the SDK and the AI SDK emit an `execute_tool <id>` span)
 * and `gen_ai.*` attributes are shared vocabulary, so scope is the only thing
 * that says who produced a span. Pinned here because a drift in either name
 * silently rewrites every scope-keyed query a backend has saved.
 */
const SDK_SCOPE = "@ratel-ai/sdk";
const AI_SDK_SCOPE = "gen_ai";
const HOST_SCOPE = "host";
const HOST_SPAN = "agent request";

const ACCEPT_ALL = () => true;

let provider: BasicTracerProvider | undefined;

afterEach(async () => {
  await provider?.forceFlush();
  await provider?.shutdown();
  provider = undefined;
  // `registerTelemetry()` only ever pushes onto this array, so the global
  // registry has to be emptied by hand or a later test inherits an integration
  // bound to a disposed provider.
  (globalThis as { AI_SDK_TELEMETRY_INTEGRATIONS?: unknown[] }).AI_SDK_TELEMETRY_INTEGRATIONS = [];
  trace.disable();
  context.disable();
  vi.restoreAllMocks();
});

/** One backend wired onto the host's provider, plus what actually reached it. */
interface Destination {
  processor: SpanProcessor;
  spans(): ReadableSpan[];
  names(): string[];
}

/**
 * Stand up the provider a host owns and register it globally. That registration
 * is the entire wiring contract: the SDK holds no provider reference and is
 * handed nothing, so nothing else connects emission to these destinations.
 */
function hostProvider(...destinations: Destination[]): void {
  provider = new BasicTracerProvider({
    spanProcessors: destinations.map((destination) => destination.processor),
  });
  trace.setGlobalTracerProvider(provider);
  // `@ai-sdk/otel` parents its spans via `context.with`, which only survives the
  // SDK's awaits once a real context manager is installed.
  context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
}

/** A generic OTel destination: the processor + exporter pair any host already runs. */
function inMemoryDestination(): Destination {
  const exporter = new InMemorySpanExporter();
  return destination(new SimpleSpanProcessor(exporter), () => exporter.getFinishedSpans());
}

/**
 * A generic destination that filters with a predicate of the host's own writing.
 * Ratel ships no span filter to pair against Langfuse's, so per-destination
 * filtering on the non-vendor side is exactly this: a few lines wrapping any
 * processor.
 */
function filteringDestination(keep: (span: ReadableSpan) => boolean): Destination {
  const exporter = new InMemorySpanExporter();
  const inner = new SimpleSpanProcessor(exporter);
  const processor: SpanProcessor = {
    onStart: (span: Span, parentContext: Context) => inner.onStart(span, parentContext),
    onEnd: (span: ReadableSpan) => {
      if (keep(span)) inner.onEnd(span);
    },
    forceFlush: () => inner.forceFlush(),
    shutdown: () => inner.shutdown(),
  };
  return destination(processor, () => exporter.getFinishedSpans());
}

/**
 * The real `LangfuseSpanProcessor`, captured at its own OTLP exporter so nothing
 * leaves the process. Reaching a private field is the price of proving the
 * *vendor's* processor works rather than a stand-in that merely resembles it:
 * `shouldExportSpan` is Langfuse's option, applied by Langfuse's code, on the
 * same shared stream as everyone else.
 */
function langfuseDestination(
  shouldExportSpan?: (arg: { otelSpan: ReadableSpan }) => boolean,
): Destination {
  const processor = new LangfuseSpanProcessor({
    publicKey: "pk",
    secretKey: "sk",
    baseUrl: "http://127.0.0.1:1",
    shouldExportSpan,
  });
  const exporter = (processor as unknown as { processor: { _exporter: SpanExporter } }).processor
    ._exporter;
  const exported: ReadableSpan[] = [];
  vi.spyOn(exporter, "export").mockImplementation((spans, resultCallback) => {
    exported.push(...spans);
    resultCallback({ code: 0 });
  });
  return destination(processor, () => exported);
}

function destination(processor: SpanProcessor, read: () => ReadableSpan[]): Destination {
  return {
    processor,
    spans: read,
    // Sorted, because export order across destinations is an implementation
    // detail and comparing sets is the actual claim.
    names: () =>
      read()
        .map((span) => span.name)
        .sort(),
  };
}

function fromScope(spans: ReadableSpan[], scope: string): ReadableSpan[] {
  return spans.filter((span) => span.instrumentationScope.name === scope);
}

function scopedNames(spans: ReadableSpan[], scope: string): string[] {
  return fromScope(spans, scope)
    .map((span) => span.name)
    .sort();
}

function hasGenAiAttribute(span: ReadableSpan): boolean {
  return Object.keys(span.attributes).some((key) => key.startsWith("gen_ai."));
}

/** Real SDK work, no AI SDK involved: a ranked search plus a tool execution. */
async function driveRatelWork(): Promise<void> {
  const r = ratel();
  await r.tools.register({
    id: "deploy_app",
    name: "deploy_app",
    description: "Deploy the app to production servers.",
    inputSchema: { type: "object", properties: {} },
    outputSchema: { type: "object" },
    execute: async () => ({ deployed: true }),
  });
  r.tools.search("deploy to production", 3);
  await r.tools.invoke("deploy_app", {});
}

/**
 * One agent loop producing BOTH span families: `prepareStep` recall ranks the
 * catalog (`ratel.search`), the model calls `invoke_tool` so the catalog executes
 * the target (`execute_tool deploy_app`), and the registered integration emits
 * the AI SDK's own `gen_ai.*` spans around all of it. A single shared stream, as
 * a real deployment produces it.
 */
async function driveMixedWork(): Promise<void> {
  const view = ratel().adaptTo(aiSdk());
  await view.tools.register({
    deploy_app: tool({
      description: "Deploy the app to production servers.",
      inputSchema: z.object({}),
      execute: async () => ({ deployed: true }),
    }),
  });
  const model = new MockLanguageModelV2([
    {
      content: [
        {
          type: "tool-call",
          toolCallId: "invoke-0",
          toolName: "invoke_tool",
          input: JSON.stringify({ toolId: "deploy_app", args: {} }),
        },
      ],
      finishReason: "tool-calls",
      usage,
      warnings: [],
    },
    { content: [{ type: "text", text: "deployed." }], finishReason: "stop", usage, warnings: [] },
  ]);

  await generateText({
    model: model as unknown as LanguageModel,
    tools: view.modelTools(),
    messages: [{ role: "user", content: "deploy to production" }],
    prepareStep: view.prepareStep,
    stopWhen: ai.stepCountIs(2),
  });
}

async function driveMixedWorkInHostSpan(): Promise<void> {
  await trace.getTracer(HOST_SCOPE).startActiveSpan(HOST_SPAN, async (span) => {
    try {
      await driveMixedWork();
    } finally {
      span.end();
    }
  });
}

async function driveExperimentWork(): Promise<string> {
  const experiment = experimentalDefineExperiment({
    id: "retrieval-v2",
    arms: {
      hybrid: {
        select: async () => {
          await driveMixedWork();
          return { ids: ["deploy_app"] };
        },
      },
    },
    ranking: (result) => result.ids.map((id) => ({ id })),
    evaluation: { references: ["peer-selection"] },
  });
  const selection = await experiment.select(
    { query: "deploy to production" },
    { arm: "hybrid", unitId: "unit-a" },
  );
  return selection.selectionId;
}

describe("telemetry coexistence", () => {
  it("fans the SDK's own spans to every processor on the host's provider", async () => {
    const langfuse = langfuseDestination(ACCEPT_ALL);
    const generic = inMemoryDestination();
    const second = inMemoryDestination();
    hostProvider(langfuse, generic, second);

    await driveRatelWork();
    await provider?.forceFlush();

    // The SDK emits through the *global* tracer and never touches a provider it
    // was handed, so a host that builds one without registering it globally
    // collects nothing. Fan-out is the provider's job: no destination can starve
    // another, and a vendor processor is just one more peer on the list.
    const expected = [`${EXECUTE_TOOL} deploy_app`, RATEL_SEARCH];
    expect(generic.names()).toEqual(expected);
    expect(second.names()).toEqual(expected);
    expect(langfuse.names()).toEqual(expected);
  });

  it("hides the SDK's non-gen_ai spans behind Langfuse's default filter", async () => {
    const langfuse = langfuseDestination();
    const generic = inMemoryDestination();
    hostProvider(langfuse, generic);

    await driveRatelWork();
    await provider?.forceFlush();

    // Langfuse's default `shouldExportSpan` keeps a span only when it carries a
    // `gen_ai.*` attribute or comes from a scope on Langfuse's known-instrumentor
    // list. `@ratel-ai/sdk` is on neither, so `execute_tool` survives on the
    // strength of its `gen_ai.tool.name` while `ratel.search` — and every other
    // purely `ratel.*` span — is dropped before export.
    //
    // Nothing here is broken: the stream reached Langfuse intact (the test
    // above), and Langfuse chose. But the difference is invisible from the host,
    // so a user wiring `new LangfuseSpanProcessor()` next to Ratel and expecting
    // their retrieval telemetry has to be told this, in words, in the docs.
    expect(generic.names()).toEqual([`${EXECUTE_TOOL} deploy_app`, RATEL_SEARCH]);
    expect(langfuse.names()).toEqual([`${EXECUTE_TOOL} deploy_app`]);
  });

  it("takes Ratel's spans into Langfuse once the host widens the predicate", async () => {
    const langfuse = langfuseDestination(
      ({ otelSpan }) => otelSpan.instrumentationScope.name === SDK_SCOPE,
    );
    hostProvider(langfuse);

    await driveRatelWork();
    await provider?.forceFlush();

    // The fix a host applies, and the reason it belongs in the docs rather than
    // in Ratel's code: the predicate is Langfuse's own option, so opting Ratel in
    // costs one line and needs nothing from us. Keying it on scope rather than a
    // `ratel.` name prefix also catches `execute_tool`, whose name carries no
    // hint of who emitted it.
    expect(langfuse.names()).toEqual([`${EXECUTE_TOOL} deploy_app`, RATEL_SEARCH]);
  });

  it("funnels a client-executed passthrough through the SDK tool span", async () => {
    const generic = inMemoryDestination();
    hostProvider(generic);
    const view = ratel({ trace: { kind: "memory", sessionId: "passthrough" } }).adaptTo(aiSdk());
    await view.tools.register({
      shell: {
        type: "provider",
        id: "acme.shell",
        args: {},
        isProviderExecuted: false,
        inputSchema: { type: "object" },
        execute: (input: unknown) => ({ input, ran: true }),
      } as unknown as Tool,
    });
    const exposed = view.modelTools().shell;
    const run = exposed.execute as (input: unknown, options: unknown) => unknown;

    const result = run({ command: "pwd" }, { toolCallId: "shell-0", messages: [] });
    await provider?.forceFlush();

    expect(result).toEqual({ input: { command: "pwd" }, ran: true });
    expect(scopedNames(generic.spans(), SDK_SCOPE)).toContain(`${EXECUTE_TOOL} shell`);
    expect(
      (view.tools.catalog.drainTraceEvents() as Array<{ type: string }>).map((event) => event.type),
    ).toEqual(["invoke_start", "invoke_end"]);
  });
});

describe.skipIf(aiSdkMajor < 7)("telemetry coexistence with the AI SDK integration", () => {
  it("finds the ai@7 telemetry seam this suite is written against", () => {
    // The load-bearing assertion for the skip above: on a v7 row this fails
    // loudly if `ai` moves the seam, instead of the suite quietly evaporating.
    expect(typeof registerTelemetry, "ai@7 no longer exports registerTelemetry").toBe("function");
  });

  it("correlates both span families under the host's active operation span", async () => {
    const generic = inMemoryDestination();
    hostProvider(generic);
    registerTelemetry?.(new RatelOtelIntegration());

    await driveMixedWorkInHostSpan();
    await provider?.forceFlush();

    const spans = generic.spans();
    const hostSpan = fromScope(spans, HOST_SCOPE).find((span) => span.name === HOST_SPAN);
    const searchSpans = fromScope(spans, SDK_SCOPE).filter((span) => span.name === RATEL_SEARCH);
    const genAiSpans = fromScope(spans, AI_SDK_SCOPE);
    const traceIds = new Set(
      [...searchSpans, ...genAiSpans].map((span) => span.spanContext().traceId),
    );

    expect(hostSpan).toBeDefined();
    expect(searchSpans).toHaveLength(2);
    expect(genAiSpans.length).toBeGreaterThan(0);
    expect(traceIds).toEqual(new Set([hostSpan?.spanContext().traceId]));
    expect(
      searchSpans.every(
        (span) => span.parentSpanContext?.spanId === hostSpan?.spanContext().spanId,
      ),
    ).toBe(true);
    expect(
      genAiSpans.some((span) => span.parentSpanContext?.spanId === hostSpan?.spanContext().spanId),
    ).toBe(true);
  });

  it("carries the integration's enrichment to every destination", async () => {
    const langfuse = langfuseDestination(ACCEPT_ALL);
    const generic = inMemoryDestination();
    hostProvider(langfuse, generic);
    registerTelemetry?.(
      new RatelOtelIntegration({ enrichSpan: () => ({ "deployment.environment": "staging" }) }),
    );

    await driveMixedWork();
    await provider?.forceFlush();

    // Enrichment happens at span creation, upstream of every processor, so no
    // destination can see a differently-attributed copy — which is the whole
    // reason this composes: Langfuse and a generic backend disagree about what
    // to *keep*, never about what a span *says*.
    for (const seen of [generic.spans(), langfuse.spans()]) {
      const emitted = fromScope(seen, AI_SDK_SCOPE);
      expect(emitted.length).toBeGreaterThan(0);
      for (const span of emitted) {
        expect(span.attributes[RATEL_ORIGIN]).toBe("agent");
        expect(span.attributes["deployment.environment"]).toBe("staging");
      }
    }
    expect(scopedNames(generic.spans(), AI_SDK_SCOPE)).toEqual(
      scopedNames(langfuse.spans(), AI_SDK_SCOPE),
    );
  });

  it("carries a real experiment join to every destination", async () => {
    const langfuse = langfuseDestination(ACCEPT_ALL);
    const generic = inMemoryDestination();
    hostProvider(langfuse, generic);
    registerTelemetry?.(new RatelOtelIntegration());

    const selectionId = await driveExperimentWork();
    await provider?.forceFlush();

    const expectedJoin = {
      [RATEL_EXPERIMENT_ID]: "retrieval-v2",
      [RATEL_EXPERIMENT_SELECTION_ID]: selectionId,
      [RATEL_EXPERIMENT_ARM]: "hybrid",
      [RATEL_EXPERIMENT_ROLE]: "serving",
      [RATEL_EXPERIMENT_UNIT]: "d7bce2267437426d",
    };
    for (const seen of [generic.spans(), langfuse.spans()]) {
      const emitted = fromScope(seen, AI_SDK_SCOPE);
      expect(emitted.length).toBeGreaterThan(0);
      for (const span of emitted) {
        expect(span.attributes).toMatchObject(expectedJoin);
      }
    }
    expect(scopedNames(generic.spans(), AI_SDK_SCOPE)).toEqual(
      scopedNames(langfuse.spans(), AI_SDK_SCOPE),
    );
  });

  it("leaves the SDK's own spans unenriched, gen_ai attributes notwithstanding", async () => {
    const generic = inMemoryDestination();
    hostProvider(generic);
    registerTelemetry?.(
      new RatelOtelIntegration({ enrichSpan: () => ({ "deployment.environment": "staging" }) }),
    );

    await driveMixedWork();
    await provider?.forceFlush();

    // The sharp edge behind the test above. `enrichSpan` is a hook on the AI
    // SDK's emitter, so it reaches only spans that emitter creates. The SDK's
    // `execute_tool` span carries `gen_ai.*` attributes of its own and is a
    // different emitter entirely — selecting spans by attribute prefix and
    // asserting enrichment on them would be asserting something untrue.
    const sdkSpans = fromScope(generic.spans(), SDK_SCOPE);
    expect(sdkSpans.map((span) => span.name)).toContain(`${EXECUTE_TOOL} deploy_app`);
    expect(sdkSpans.some(hasGenAiAttribute)).toBe(true);
    for (const span of sdkSpans) {
      expect(span.attributes["deployment.environment"]).toBeUndefined();
    }
  });

  it("lets each destination filter the one shared stream independently", async () => {
    const langfuse = langfuseDestination(
      ({ otelSpan }) => otelSpan.instrumentationScope.name === SDK_SCOPE,
    );
    const genAiOnly = filteringDestination(
      (span) => span.instrumentationScope.name === AI_SDK_SCOPE,
    );
    const everything = inMemoryDestination();
    hostProvider(langfuse, genAiOnly, everything);
    registerTelemetry?.(new RatelOtelIntegration());

    await driveMixedWork();
    await provider?.forceFlush();

    // Three destinations, one emission, three different export sets — a vendor
    // predicate on Langfuse's side, a hand-written one on the generic side, and
    // an unfiltered peer proving neither filter reached back into the stream.
    // Filtering is a property of a destination, never of the source.
    expect(langfuse.names()).toEqual([`${EXECUTE_TOOL} deploy_app`, RATEL_SEARCH, RATEL_SEARCH]);
    expect(genAiOnly.spans().length).toBeGreaterThan(0);
    expect(genAiOnly.spans().every(hasGenAiAttribute)).toBe(true);
    expect(genAiOnly.names()).not.toContain(RATEL_SEARCH);
    // The union closes the proof from the other side: the two predicates
    // partition the stream exactly, so nothing was invented and nothing leaked.
    expect(everything.names()).toEqual([...langfuse.names(), ...genAiOnly.names()].sort());
  });
});
