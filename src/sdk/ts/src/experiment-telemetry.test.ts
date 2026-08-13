import { context, propagation, SpanKind, trace } from "@opentelemetry/api";
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
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  ContentCapture,
  clearContentCapture,
  experimentalDefineExperiment,
  setContentCapture,
} from "./index.js";

interface SearchResult {
  ids: string[];
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

let exporter: InMemorySpanExporter;
let logExporter: InMemoryLogRecordExporter;

beforeEach(() => {
  exporter = new InMemorySpanExporter();
  trace.setGlobalTracerProvider(
    new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    }),
  );
  logExporter = new InMemoryLogRecordExporter();
  logs.setGlobalLoggerProvider(
    new LoggerProvider({
      processors: [new SimpleLogRecordProcessor({ exporter: logExporter })],
    }),
  );
});

afterEach(() => {
  setContentCapture(null);
  context.disable();
  logs.disable();
  trace.disable();
});

function experimentSpans(): ReadableSpan[] {
  return exporter.getFinishedSpans().filter((span) => span.name === "ratel.experiment.arm");
}

function experimentSpanForArm(arm: string): ReadableSpan | undefined {
  return experimentSpans().find((span) => span.attributes["ratel.experiment.arm"] === arm);
}

function experimentEvents(name: string): ReadableLogRecord[] {
  return logExporter.getFinishedLogRecords().filter((record) => record.eventName === name);
}

function deferred<T>(): Deferred<T> {
  let resolve = (_value: T): void => {};
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

describe("experiment telemetry", () => {
  it("directly stamps a serving arm when no ContextManager is registered", async () => {
    let activeBaggage = propagation.getBaggage(context.active());
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => {
            activeBaggage = propagation.getBaggage(context.active());
            return { ids: ["inspect-ci"] };
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      { arm: "legacy", unitId: "unit-a" },
    );

    const [span] = experimentSpans();
    expect(span, "one completed experiment arm span").toBeTruthy();
    expect(span.instrumentationScope.name).toBe("@ratel-ai/sdk");
    expect(span.kind).toBe(SpanKind.INTERNAL);
    expect(span.attributes).toMatchObject({
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "legacy",
      "ratel.experiment.role": "serving",
      "ratel.experiment.unit": "d7bce2267437426d",
      "ratel.experiment.cold": false,
      "ratel.experiment.outcome": "ok",
      "ratel.experiment.hit_count": 1,
    });
    expect(span.attributes["ratel.experiment.duration_ms"]).toEqual(expect.any(Number));
    expect(activeBaggage).toBeUndefined();
  });

  it("emits ranked ids and scores in an arm-correlated results EventRecord", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci", "read-logs"] }),
        },
      },
      ranking: (result: SearchResult) =>
        result.ids.map((id, index) => ({
          id,
          score: 0.9 - index / 10,
          attrs: { privateLabel: `label-${index}` },
        })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      {
        arm: "legacy",
        unitId: "unit-a",
        attributes: { "deployment.environment": "canary" },
      },
    );

    const [span] = experimentSpans();
    const [event] = experimentEvents("ratel.experiment.results");
    expect(event.attributes).toMatchObject({
      "deployment.environment": "canary",
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "legacy",
      "ratel.experiment.role": "serving",
      "ratel.experiment.unit": "d7bce2267437426d",
      "ratel.experiment.result_ids": ["inspect-ci", "read-logs"],
      "ratel.experiment.result_scores": [0.9, 0.8],
    });
    expect(event.attributes["ratel.experiment.result_attrs"]).toBeUndefined();
    expect(event.instrumentationScope.name).toBe("@ratel-ai/sdk");
    expect(event.spanContext).toEqual(span.spanContext());
  });

  it("captures aligned item attributes on both channels under SPAN_AND_EVENT", async () => {
    const generation = setContentCapture(ContentCapture.SpanAndEvent);
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci", "read-logs"] }),
        },
      },
      ranking: (result: SearchResult) =>
        result.ids.map((id, index) => ({
          id,
          ...(index === 0 ? { attrs: { domain: "ci", sequence: false } } : {}),
        })),
      evaluation: { references: ["peer-selection"] },
    });

    await experiment.select({ query: "broken build" }, { arm: "legacy", unitId: "unit-a" });
    clearContentCapture(generation);

    const [span] = experimentSpans();
    const [event] = experimentEvents("ratel.experiment.results");
    const aligned = [{ domain: "ci", sequence: false }, null];
    expect(span.attributes["ratel.experiment.result_attrs"]).toBe(JSON.stringify(aligned));
    expect(event.attributes["ratel.experiment.result_attrs"]).toEqual(aligned);
  });

  it.each([
    [ContentCapture.SpanOnly, true, false],
    [ContentCapture.EventOnly, false, true],
    [ContentCapture.NoContent, false, false],
  ] as const)("routes item attributes under %s", async (mode, expectedOnSpan, expectedOnEvent) => {
    const generation = setContentCapture(mode);
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id, attrs: { domain: "ci" } })),
      evaluation: { references: ["peer-selection"] },
    });

    await experiment.select({ query: "broken build" }, { arm: "legacy", unitId: "unit-a" });
    clearContentCapture(generation);

    const spanValue = experimentSpans()[0]?.attributes["ratel.experiment.result_attrs"];
    const eventValue = experimentEvents("ratel.experiment.results")[0]?.attributes[
      "ratel.experiment.result_attrs"
    ];
    expect(spanValue !== undefined).toBe(expectedOnSpan);
    expect(eventValue !== undefined).toBe(expectedOnEvent);
  });

  it("omits partial scores and contains item-attribute encoding failures", async () => {
    const generation = setContentCapture(ContentCapture.SpanAndEvent);
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci", "read-logs"] }),
        },
      },
      ranking: (result: SearchResult) =>
        result.ids.map((id, index) => ({
          id,
          ...(index === 0
            ? {
                score: 0.9,
                attrs: { invalid: 1n } as unknown as Record<
                  string,
                  string | number | boolean | null
                >,
              }
            : {}),
        })),
      evaluation: { references: ["peer-selection"] },
    });

    await experiment.select({ query: "broken build" }, { arm: "legacy", unitId: "unit-a" });
    clearContentCapture(generation);

    const [span] = experimentSpans();
    const [event] = experimentEvents("ratel.experiment.results");
    expect(span.attributes).toMatchObject({
      "ratel.experiment.outcome": "ok",
      "ratel.experiment.result_attrs_encoding_error": "TypeError",
    });
    expect(span.attributes["ratel.experiment.result_attrs"]).toBeUndefined();
    expect(event.attributes["ratel.experiment.result_scores"]).toBeUndefined();
    expect(event.attributes["ratel.experiment.result_attrs"]).toBeUndefined();
    expect(event.attributes["ratel.experiment.result_ids"]).toEqual(["inspect-ci", "read-logs"]);
  });

  it("records a serving timeout as an errored arm without a results event", async () => {
    const timeout = new Error("arm deadline exceeded");
    timeout.name = "TimeoutError";
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => {
            throw timeout;
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    await expect(
      experiment.select({ query: "broken build" }, { arm: "legacy", unitId: "unit-a" }),
    ).rejects.toThrow("arm deadline exceeded");

    const [span] = experimentSpans();
    expect(span.attributes).toMatchObject({
      "error.type": "TimeoutError",
      "ratel.experiment.outcome": "timeout",
    });
    expect(span.attributes["ratel.experiment.hit_count"]).toBeUndefined();
    expect(span.status).toEqual({ code: 2, message: "arm deadline exceeded" });
    expect(span.events.some((event) => event.name === "exception")).toBe(true);
    expect(experimentEvents("ratel.experiment.results")).toHaveLength(0);
  });

  it("records an empty successful ranking and its results event", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: [] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    await experiment.select({ query: "broken build" }, { arm: "legacy", unitId: "unit-a" });

    const [span] = experimentSpans();
    const [event] = experimentEvents("ratel.experiment.results");
    expect(span.attributes).toMatchObject({
      "ratel.experiment.outcome": "empty",
      "ratel.experiment.hit_count": 0,
    });
    expect(span.status).toEqual({ code: 1 });
    expect(event.attributes["ratel.experiment.result_ids"]).toEqual([]);
  });

  it("keeps a transform TimeoutError classified as error", async () => {
    const transformError = new Error("visibility transform failed");
    transformError.name = "TimeoutError";
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    await expect(
      experiment.select(
        { query: "broken build" },
        {
          arm: "legacy",
          unitId: "unit-a",
          transform: () => {
            throw transformError;
          },
        },
      ),
    ).rejects.toThrow("visibility transform failed");

    const [span] = experimentSpans();
    expect(span.attributes).toMatchObject({
      "error.type": "TimeoutError",
      "ratel.experiment.outcome": "error",
    });
    expect(span.status).toEqual({ code: 2, message: "visibility transform failed" });
  });

  it("diagnoses a ranking failure without replacing the transformed result", async () => {
    const rankingError = new Error("ranking unavailable");
    rankingError.name = "RankingProjectionError";
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: () => {
        throw rankingError;
      },
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      { arm: "legacy", unitId: "unit-a" },
    );

    expect(selection.result).toEqual({ ids: ["inspect-ci"] });
    const [span] = experimentSpans();
    expect(span.attributes).toMatchObject({
      "error.type": "RankingProjectionError",
      "ratel.experiment.outcome": "error",
      "ratel.experiment.ranking_error": "RankingProjectionError",
    });
    expect(span.attributes["ratel.experiment.hit_count"]).toBeUndefined();
    expect(span.status).toEqual({ code: 2, message: "ranking unavailable" });
    expect(span.events.some((event) => event.name === "exception")).toBe(true);
    expect(experimentEvents("ratel.experiment.results")).toHaveLength(0);
  });

  it("diagnoses result projection failure without failing the arm", async () => {
    const projectionError = new Error("result facts unavailable");
    projectionError.name = "ResultProjectionError";
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: {
        attributes: () => {
          throw projectionError;
        },
        references: ["peer-selection"],
      },
    });

    await experiment.select({ query: "broken build" }, { arm: "legacy", unitId: "unit-a" });

    const [span] = experimentSpans();
    expect(span.attributes).toMatchObject({
      "ratel.experiment.outcome": "ok",
      "ratel.experiment.hit_count": 1,
      "ratel.experiment.result_attributes_error": "ResultProjectionError",
    });
    expect(span.attributes["error.type"]).toBeUndefined();
    expect(span.status).toEqual({ code: 1 });
    expect(span.events.some((event) => event.name === "exception")).toBe(true);
    expect(experimentEvents("ratel.experiment.results")).toHaveLength(1);
  });

  it("emits a shadow-correlated served comparison", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci", "read-logs"] }),
        },
        candidate: {
          select: async () => ({ ids: ["read-logs", "retry-build"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: {
        k: 2,
        attributes: (result) => ({ first: result.ids[0] ?? null, length: result.ids.length }),
        references: ["peer-selection"],
      },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      {
        arm: "control",
        shadow: true,
        unitId: "unit-a",
        attributes: { "deployment.environment": "canary" },
      },
    );
    await experiment.drain();

    const shadowSpan = experimentSpanForArm("candidate");
    const [event] = experimentEvents("ratel.experiment.comparison");
    expect(event.attributes).toMatchObject({
      "deployment.environment": "canary",
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "candidate",
      "ratel.experiment.role": "shadow",
      "ratel.experiment.unit": "d7bce2267437426d",
      "ratel.experiment.served.arm": "control",
      "ratel.experiment.served.outcome": "ok",
      "ratel.experiment.served.hit_count": 2,
      "ratel.experiment.shadow.arm": "candidate",
      "ratel.experiment.shadow.outcome": "ok",
      "ratel.experiment.shadow.hit_count": 2,
      "ratel.experiment.agreement.top1": false,
      "ratel.experiment.agreement.exact_order": false,
      "ratel.experiment.agreement.overlap_count": 1,
      "ratel.experiment.agreement.jaccard_at_k": 1 / 3,
      "ratel.experiment.agreement.k": 2,
      "ratel.experiment.agreement.item_attrs": {},
      "ratel.experiment.agreement.result_attrs": { first: false, length: true },
    });
    expect(event.attributes["ratel.experiment.served.duration_ms"]).toEqual(expect.any(Number));
    expect(event.attributes["ratel.experiment.shadow.duration_ms"]).toEqual(expect.any(Number));
    expect(event.spanContext).toEqual(shadowSpan?.spanContext());
  });

  it("emits an assigned-arm-correlated event for a capacity-skipped shadow", async () => {
    const candidate = deferred<SearchResult>();
    let skippedCalls = 0;
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
        candidate: {
          select: () => candidate.promise,
        },
        skipped: {
          select: async () => {
            skippedCalls += 1;
            return { ids: ["retry-build"] };
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      shadowPolicy: { concurrency: 1 },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      {
        arm: "control",
        shadow: true,
        unitId: "unit-a",
        attributes: { "deployment.environment": "canary" },
      },
    );

    const servingSpan = experimentSpanForArm("control");
    const [event] = experimentEvents("ratel.experiment.skip");
    expect(skippedCalls).toBe(0);
    expect(event.attributes).toMatchObject({
      "deployment.environment": "canary",
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "control",
      "ratel.experiment.role": "serving",
      "ratel.experiment.skip.arm": "skipped",
      "ratel.experiment.skip.concurrency": 1,
      "ratel.experiment.skip.reason": "capacity",
    });
    expect(event.spanContext).toEqual(servingSpan?.spanContext());

    candidate.resolve({ ids: ["read-logs"] });
    await experiment.drain();
  });

  it("correlates a successful fresh fallback to the failed assigned arm", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        candidate: {
          select: async () => {
            throw new Error("candidate unavailable");
          },
        },
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "control",
    });

    const selection = await experiment.select(
      { query: "broken build" },
      {
        arm: "candidate",
        unitId: "unit-a",
        attributes: { "deployment.environment": "canary" },
      },
    );

    const failedSpan = experimentSpanForArm("candidate");
    const [event] = experimentEvents("ratel.experiment.fallback");
    expect(selection.effectiveArm).toBe("control");
    expect(event.attributes).toMatchObject({
      "deployment.environment": "canary",
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "candidate",
      "ratel.experiment.role": "serving",
      "ratel.experiment.fallback.effective_arm": "control",
      "ratel.experiment.fallback.reused_shadow": false,
    });
    expect(event.spanContext).toEqual(failedSpan?.spanContext());
  });

  it("reports a reused shadow fallback without opening a second fallback span", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        candidate: {
          select: async () => {
            throw new Error("candidate unavailable");
          },
        },
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "control",
    });

    const selection = await experiment.select(
      { query: "broken build" },
      { arm: "candidate", shadow: true, unitId: "unit-a" },
    );
    await experiment.drain();

    const controlSpans = experimentSpans().filter(
      (span) => span.attributes["ratel.experiment.arm"] === "control",
    );
    expect(controlSpans).toHaveLength(1);
    expect(controlSpans[0]?.attributes["ratel.experiment.role"]).toBe("shadow");
    expect(experimentEvents("ratel.experiment.fallback")[0]?.attributes).toMatchObject({
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.fallback.effective_arm": "control",
      "ratel.experiment.fallback.reused_shadow": true,
    });
    expect(experimentEvents("ratel.experiment.drop")[0]?.attributes).toMatchObject({
      "ratel.experiment.arm": "control",
      "ratel.experiment.drop.reason": "fallback-consumed",
    });
  });

  it("emits a terminal drop against the failed shadow arm", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
        candidate: {
          select: async () => {
            throw new Error("shadow unavailable");
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      {
        arm: "control",
        shadow: true,
        unitId: "unit-a",
        attributes: { "deployment.environment": "canary" },
      },
    );
    await experiment.drain();

    const shadowSpan = experimentSpanForArm("candidate");
    const [event] = experimentEvents("ratel.experiment.drop");
    expect(event.attributes).toMatchObject({
      "deployment.environment": "canary",
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "candidate",
      "ratel.experiment.role": "shadow",
      "ratel.experiment.drop.reason": "arm-failed",
    });
    expect(event.spanContext).toEqual(shadowSpan?.spanContext());
  });

  it("emits an attributed invocation in the active report-time context", async () => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci", "read-logs"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: {
        references: [{ kind: "invocation", window: { turns: 1 } }],
      },
    });
    const selection = await experiment.select(
      { query: "broken build" },
      { arm: "control", unitId: "unit-a" },
    );
    const hostSpan = trace.getTracer("host").startSpan("host-operation");
    const hostContext = trace.setSpan(context.active(), hostSpan);

    context.with(hostContext, () => {
      experiment.reportInvocation({
        unitId: "unit-a",
        toolId: "read-logs",
        turn: 7,
      });
    });
    hostSpan.end();

    const [event] = experimentEvents("ratel.experiment.invocation");
    expect(event.attributes).toMatchObject({
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.unit": "d7bce2267437426d",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.effective_arm": "control",
      "ratel.experiment.invocation.attributed": true,
      "ratel.experiment.invocation.rank": 1,
      "ratel.experiment.invocation.age_ms": expect.any(Number),
      "ratel.experiment.turn": 7,
      "gen_ai.tool.name": "read-logs",
    });
    expect(event.spanContext).toEqual(hostSpan.spanContext());
  });

  it("omits stale join fields from an unattributed invocation", () => {
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: {
        references: [{ kind: "invocation", window: { turns: 1 } }],
      },
    });

    experiment.reportInvocation({ unitId: "unit-without-selection", toolId: "inspect-ci" });

    const [event] = experimentEvents("ratel.experiment.invocation");
    expect(event.attributes).toEqual({
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.unit": "a763b7e51e4339b7",
      "ratel.experiment.invocation.attributed": false,
      "gen_ai.tool.name": "inspect-ci",
    });
  });

  it("emits each delayed outcome in the active report-time context", () => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { outcome: true, references: ["peer-selection"] },
    });
    const hostSpan = trace.getTracer("host").startSpan("outcome-report");
    const hostContext = trace.setSpan(context.active(), hostSpan);

    context.with(hostContext, () => {
      experiment.reportOutcome({
        selectionId: "selection-1",
        label: "accepted",
        score: 0.95,
      });
      experiment.reportOutcome({
        selectionId: "selection-1",
        label: "accepted",
        score: 0.95,
      });
    });
    hostSpan.end();

    const events = experimentEvents("ratel.experiment.outcome");
    expect(events).toHaveLength(2);
    for (const event of events) {
      expect(event.attributes).toEqual({
        "ratel.experiment.id": "search-v2",
        "ratel.experiment.selection_id": "selection-1",
        "ratel.experiment.outcome.label": "accepted",
        "ratel.experiment.outcome.score": 0.95,
      });
      expect(event.spanContext).toEqual(hostSpan.spanContext());
    }
  });

  it.each([
    "ratel.experiment.id",
    "ratel.experiment.agreement.k",
    "gen_ai.tool.name",
    "error.type",
  ])("rejects reserved caller attribute %s before dispatch", (reservedKey) => {
    let armCalls = 0;
    let warmupCalls = 0;
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          warmup: async () => {
            warmupCalls += 1;
          },
          select: async () => {
            armCalls += 1;
            return { ids: ["inspect-ci"] };
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    expect(() =>
      experiment.select(
        { query: "broken build" },
        {
          arm: "control",
          unitId: "unit-a",
          attributes: { [reservedKey]: "override" },
        },
      ),
    ).toThrow(/reserved telemetry attribute/i);
    expect({ armCalls, warmupCalls }).toEqual({ armCalls: 0, warmupCalls: 0 });
    expect(experimentSpans()).toHaveLength(0);
  });

  it("activates the full arm baggage while preserving the parent baggage", async () => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    const parentContext = propagation.setBaggage(
      context.active(),
      propagation.createBaggage({
        tenant: { value: "acme" },
        "ratel.experiment.id": { value: "outer-experiment" },
      }),
    );
    let activeEntries: Record<string, string | undefined> = {};
    let restoredExperimentId: string | undefined;
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        legacy: {
          select: async () => {
            const baggage = propagation.getBaggage(context.active());
            activeEntries = Object.fromEntries(
              [
                "tenant",
                "ratel.experiment.id",
                "ratel.experiment.selection_id",
                "ratel.experiment.arm",
                "ratel.experiment.role",
                "ratel.experiment.unit",
              ].map((key) => [key, baggage?.getEntry(key)?.value]),
            );
            return { ids: ["inspect-ci"] };
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await context.with(parentContext, async () => {
      const selected = await experiment.select(
        { query: "broken build" },
        { arm: "legacy", unitId: "unit-a" },
      );
      restoredExperimentId = propagation
        .getBaggage(context.active())
        ?.getEntry("ratel.experiment.id")?.value;
      return selected;
    });

    expect(activeEntries).toEqual({
      tenant: "acme",
      "ratel.experiment.id": "search-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "legacy",
      "ratel.experiment.role": "serving",
      "ratel.experiment.unit": "d7bce2267437426d",
    });
    expect(restoredExperimentId).toBe("outer-experiment");
  });

  it("keeps detached shadow baggage and parenting active after an await", async () => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    let shadowRole: string | undefined;
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
        candidate: {
          select: async () => {
            await new Promise((resolve) => setImmediate(resolve));
            shadowRole = propagation
              .getBaggage(context.active())
              ?.getEntry("ratel.experiment.role")?.value;
            const child = trace.getTracer("host").startSpan("shadow-child");
            child.end();
            return { ids: ["read-logs"] };
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    await experiment.select(
      { query: "broken build" },
      { arm: "control", shadow: true, unitId: "unit-a" },
    );
    await experiment.drain();

    const shadowSpan = experimentSpanForArm("candidate");
    const child = exporter.getFinishedSpans().find((span) => span.name === "shadow-child");
    expect(shadowRole).toBe("shadow");
    expect(child?.parentSpanContext?.spanId).toBe(shadowSpan?.spanContext().spanId);
    expect(child?.spanContext().traceId).toBe(shadowSpan?.spanContext().traceId);
  });

  it("is a clean no-op without trace or log providers", async () => {
    trace.disable();
    logs.disable();
    const experiment = experimentalDefineExperiment({
      id: "search-v2",
      arms: {
        control: {
          select: async () => ({ ids: ["inspect-ci"] }),
        },
        candidate: {
          select: async () => ({ ids: ["read-logs"] }),
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: {
        outcome: true,
        references: ["peer-selection", { kind: "invocation", window: { turns: 1 } }],
      },
    });

    const selection = await experiment.select(
      { query: "broken build" },
      { arm: "control", shadow: true, unitId: "unit-a" },
    );
    await expect(experiment.drain()).resolves.toBeUndefined();
    expect(() =>
      experiment.reportInvocation({ unitId: "unit-a", toolId: "inspect-ci" }),
    ).not.toThrow();
    expect(() =>
      experiment.reportOutcome({ selectionId: selection.selectionId, label: "accepted" }),
    ).not.toThrow();
    expect(selection.result).toEqual({ ids: ["inspect-ci"] });
    expect(exporter.getFinishedSpans()).toHaveLength(0);
    expect(logExporter.getFinishedLogRecords()).toHaveLength(0);
  });
});
