import { readFileSync } from "node:fs";
import { trace } from "@opentelemetry/api";
import { logs } from "@opentelemetry/api-logs";
import {
  InMemoryLogRecordExporter,
  LoggerProvider,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { RATEL_EVENT_ID } from "@ratel-ai/telemetry";
import { afterEach, describe, expect, it } from "vitest";
import {
  experimentalDefineExperiment,
  RUNTIME_EVENT_MAX_HITS,
  RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
  RUNTIME_EVENT_MAX_QUERY_BYTES,
  RUNTIME_EVENT_TYPES,
  type RuntimeEvent,
  RuntimeEvents,
  ratel,
} from "./index.js";

interface RuntimeEventsFixture {
  runtime_events: {
    version: number;
    max_payload_bytes: number;
    max_query_bytes: number;
    max_hits: number;
    otel_event_id_attribute: string;
    required_envelope_fields: string[];
    event_types: string[];
  };
}

const conformance = JSON.parse(
  readFileSync(new URL("../../../telemetry/conformance/fixtures.json", import.meta.url), "utf8"),
) as RuntimeEventsFixture;

describe("public runtime events", () => {
  afterEach(() => {
    delete process.env.OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT;
    logs.disable();
    trace.disable();
  });

  it("matches the frozen cross-language event vocabulary", () => {
    expect(conformance.runtime_events).toEqual({
      version: 2,
      max_payload_bytes: RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
      max_query_bytes: RUNTIME_EVENT_MAX_QUERY_BYTES,
      max_hits: RUNTIME_EVENT_MAX_HITS,
      otel_event_id_attribute: RATEL_EVENT_ID,
      required_envelope_fields: ["v", "event_id", "ts", "session_id", "source_id", "type"],
      event_types: [...RUNTIME_EVENT_TYPES],
    });
  });

  it("delivers small events unchanged", () => {
    const input = runtimeEvent({ query: "read docs", top_k: 3 });

    expect(deliverRuntimeEvent(input)).toEqual({
      v: 2,
      event_id: "01K2KB4QN2A9XJY5VKQCN8ZM1P",
      ts: 1_755_000_000_000,
      session_id: "session-test",
      source_id: "source-test",
      type: "search",
      query: "read docs",
      top_k: 3,
    });
  });

  it("preserves the existing oversize trimming result", () => {
    const padding = Object.fromEntries(
      Array.from({ length: 16 }, (_, index) => [
        `padding_${index.toString().padStart(2, "0")}`,
        "x".repeat(4_096),
      ]),
    );

    const event = deliverRuntimeEvent(runtimeEvent(padding));

    expect(event).toEqual(
      runtimeEvent({
        ...Object.fromEntries(Object.entries(padding).slice(1)),
        payload_truncated: true,
      }),
    );
  });

  it("enforces the payload cap immediately above the exact boundary", () => {
    const atLimit = runtimeEventWithSerializedSize(RUNTIME_EVENT_MAX_PAYLOAD_BYTES);
    const aboveLimit = runtimeEventWithSerializedSize(RUNTIME_EVENT_MAX_PAYLOAD_BYTES + 1);

    const atLimitDelivered = deliverRuntimeEvent(atLimit);
    const aboveLimitDelivered = deliverRuntimeEvent(aboveLimit);

    expect(atLimitDelivered).toEqual(atLimit);
    expect(atLimitDelivered.payload_truncated).toBeUndefined();
    expect(aboveLimitDelivered.payload_truncated).toBe(true);
    expect(Buffer.byteLength(JSON.stringify(aboveLimitDelivered), "utf8")).toBeLessThanOrEqual(
      RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
    );
  });

  it("enforces public query, hit, and payload bounds before delivery", async () => {
    const runtime = ratel();
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));

    runtime.events.emit({
      type: "search",
      query: "é".repeat(4_096),
      hits: Array.from({ length: 120 }, (_, rank) => ({ id: `tool-${rank}`, rank, score: 1 })),
      unrecognized_padding: "x".repeat(100_000),
    });
    await subscription.flush();

    const [event] = received;
    expect(Buffer.byteLength(String(event?.query), "utf8")).toBeLessThanOrEqual(
      RUNTIME_EVENT_MAX_QUERY_BYTES,
    );
    expect(event?.hits).toHaveLength(RUNTIME_EVENT_MAX_HITS);
    expect(Buffer.byteLength(JSON.stringify(event), "utf8")).toBeLessThanOrEqual(
      RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
    );
  });

  it("rolls back earlier native subscriptions when a later source rejects", () => {
    let firstUnsubscribed = false;
    const first = {
      subscribeEvents: () => ({
        unsubscribe: () => {
          firstUnsubscribed = true;
        },
        flush: async () => {},
        droppedCount: 0,
      }),
    };
    const second = {
      subscribeEvents: () => {
        throw new Error("registry busy");
      },
    };
    const events = new RuntimeEvents([first, second] as never);

    expect(() => events.subscribe(() => {})).toThrow("registry busy");
    expect(firstUnsubscribed).toBe(true);
  });

  it("waits for async native-event handlers before flush resolves", async () => {
    const runtime = ratel();
    let releaseHandler: () => void = () => {};
    const handlerReleased = new Promise<void>((resolve) => {
      releaseHandler = resolve;
    });
    let markHandlerStarted: () => void = () => {};
    const handlerStarted = new Promise<void>((resolve) => {
      markHandlerStarted = resolve;
    });
    let handlerCompleted = false;
    const subscription = runtime.events.subscribe(async (batch) => {
      if (!batch.some((event) => event.type === "search")) return;
      markHandlerStarted();
      await handlerReleased;
      handlerCompleted = true;
    });

    runtime.tools.search("read", 1);
    let flushCompleted = false;
    const flushing = subscription.flush().then(() => {
      flushCompleted = true;
    });
    await handlerStarted;
    const flushState = await Promise.race([
      flushing.then(() => "resolved" as const),
      new Promise<"pending">((resolve) => setTimeout(() => resolve("pending"), 100)),
    ]);

    releaseHandler();
    await flushing;
    expect(flushState).toBe("pending");
    expect(flushCompleted).toBe(true);
    expect(handlerCompleted).toBe(true);
  });

  it("protects envelope fields while preserving a shared event id", () => {
    const runtime = ratel({
      events: { sessionId: "session-owned", sourceId: "source-owned" },
    });

    const event = runtime.events.emit({
      type: "experiment_outcome",
      event_id: "01K2KB4QN2A9XJY5VKQCN8ZM1P",
      v: 1,
      ts: 0,
      session_id: "session-forged",
      source_id: "source-forged",
    });

    expect(event).toMatchObject({
      type: "experiment_outcome",
      event_id: "01K2KB4QN2A9XJY5VKQCN8ZM1P",
      v: 2,
      session_id: "session-owned",
      source_id: "source-owned",
    });
    expect(event.ts).toBeGreaterThan(0);
  });

  it("merges tool and skill streams under one session stamp", async () => {
    const runtime = ratel({
      events: {
        sessionId: "session-public",
        sourceId: "source-public",
        queueCapacity: 16,
        batchSize: 8,
      },
    });
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));

    await runtime.tools.register({
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      inputSchema: { type: "object", properties: { path: { type: "string" } } },
      outputSchema: { type: "object" },
      execute: () => ({ ok: true }),
    });
    await runtime.skills.register({
      id: "api-design",
      name: "API design",
      description: "Design an API",
      tags: ["backend"],
      metadata: { audience: ["developers"] },
      body: "Private dispatch instructions",
    });
    runtime.tools.search("read", 1);
    runtime.skills.search("api", 1);

    await subscription.flush();

    expect(received.map((event) => event.type)).toEqual(
      expect.arrayContaining(["index_churn", "skill_churn", "search", "skill_search"]),
    );
    expect(received.every((event) => event.session_id === "session-public")).toBe(true);
    expect(received.every((event) => event.source_id === "source-public")).toBe(true);
    expect(received.every((event) => /^[0-9A-HJKMNP-TV-Z]{26}$/.test(event.event_id))).toBe(true);

    subscription.unsubscribe();
  });

  it.each([
    "NO_CONTENT",
    "SPAN_AND_EVENT",
  ])("publishes catalog definitions independently of OTel capture mode %s", async (captureMode) => {
    process.env.OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = captureMode;
    const runtime = ratel({ events: { experimentalCatalogDefinitions: true } });
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));

    await runtime.tools.register({
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      inputSchema: { type: "object", properties: { path: { type: "string" } } },
      outputSchema: { type: "object" },
      execute: () => ({ secret: "executor result" }),
    });
    await subscription.flush();

    const definition = received.find((event) => event.type === "catalog_definition");
    expect(definition).toMatchObject({
      kind: "tool",
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      tags: [],
      input_schema: { type: "object", properties: { path: { type: "string" } } },
      output_schema: { type: "object" },
      searchable_description: "Read a file",
      searchable_description_overridden: false,
    });
    expect(definition).not.toHaveProperty("execute");
    subscription.unsubscribe();
  });

  it("keeps experimental catalog definitions off the runtime stream by default", async () => {
    const runtime = ratel();
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));
    await runtime.tools.register({
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      inputSchema: {},
      outputSchema: {},
      execute: () => undefined,
    });
    await subscription.flush();
    expect(received.some((event) => event.type === "catalog_definition")).toBe(false);
    subscription.unsubscribe();
  });

  it("stamps OTel search spans with the matching stream event id", async () => {
    const exporter = new InMemorySpanExporter();
    trace.setGlobalTracerProvider(
      new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }),
    );
    const runtime = ratel({ events: { sessionId: "session-join" } });
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));
    await runtime.tools.register({
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: () => ({ ok: true }),
    });

    runtime.tools.search("read", 1);
    await subscription.flush();

    const streamEvent = received.find((event) => event.type === "search" && event.query === "read");
    const span = exporter.getFinishedSpans().find((candidate) => candidate.name === "ratel.search");
    expect(span?.attributes["ratel.event.id"]).toBe(streamEvent?.event_id);
    expect(streamEvent?.trace_id).toBe(span?.spanContext().traceId);
    expect(streamEvent?.span_id).toBe(span?.spanContext().spanId);
  });

  it("stamps an invocation span with its matching invoke_start event id", async () => {
    const exporter = new InMemorySpanExporter();
    trace.setGlobalTracerProvider(
      new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }),
    );
    const runtime = ratel({ events: { sessionId: "session-invoke" } });
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));
    await runtime.tools.register({
      id: "ping",
      name: "ping",
      description: "Ping",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: () => ({ ok: true }),
    });

    await runtime.tools.invoke("ping", {});
    await subscription.flush();

    const start = received.find((event) => event.type === "invoke_start");
    const end = received.find((event) => event.type === "invoke_end");
    const span = exporter
      .getFinishedSpans()
      .find((candidate) => candidate.name === "execute_tool ping");
    expect(span?.attributes["ratel.event.id"]).toBe(start?.event_id);
    expect(start?.invocation_id).toBe(end?.invocation_id);
    expect(start?.event_id).not.toBe(end?.event_id);
  });

  it("merges experiment evaluations through the OTel and runtime-event sinks", async () => {
    const exporter = new InMemorySpanExporter();
    const logExporter = new InMemoryLogRecordExporter();
    trace.setGlobalTracerProvider(
      new BasicTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }),
    );
    logs.setGlobalLoggerProvider(
      new LoggerProvider({
        processors: [new SimpleLogRecordProcessor({ exporter: logExporter })],
      }),
    );
    const runtime = ratel({ events: { sessionId: "session-experiment" } });
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));
    const experiment = experimentalDefineExperiment(
      {
        id: "search-v2",
        arms: { stable: { select: async () => [{ id: "read_file", score: 0.9 }] } },
        ranking: (result) => result,
        evaluation: { references: ["peer-selection"], outcome: true },
      },
      runtime.events,
    );

    const selection = await experiment.select({}, { arm: "stable", unitId: "unit-a" });
    experiment.reportOutcome({ selectionId: selection.selectionId, label: "accepted" });
    await subscription.flush();

    expect(received.map((event) => event.type)).toEqual([
      "experiment_selection",
      "experiment_results",
      "experiment_outcome",
    ]);
    expect(received[0]).toMatchObject({
      experiment_id: "search-v2",
      selection_id: selection.selectionId,
      arm: "stable",
      role: "serving",
    });
    expect(received[1]).toMatchObject({
      selection_id: selection.selectionId,
      outcome: "ok",
      result_ids: ["read_file"],
      result_scores: [0.9],
    });
    expect(received[2]).toMatchObject({
      selection_id: selection.selectionId,
      label: "accepted",
    });
    const span = exporter
      .getFinishedSpans()
      .find((candidate) => candidate.name === "ratel.experiment.arm");
    expect(span?.attributes["ratel.event.id"]).toBe(received[0]?.event_id);
    const resultsLog = logExporter
      .getFinishedLogRecords()
      .find((record) => record.eventName === "ratel.experiment.results");
    expect(resultsLog?.attributes["ratel.event.id"]).toBe(received[1]?.event_id);
  });

  it("returns the complete serializable catalog without executable content", async () => {
    const runtime = ratel({ events: { sourceId: "service-a" } });
    const execute = () => ({ secret: "must never escape" });
    await runtime.tools.register({
      id: "z_tool",
      name: "Z tool",
      description: "Last by id",
      inputSchema: { type: "object", properties: { value: { type: "string" } } },
      outputSchema: { type: "string" },
      execute,
    });
    await runtime.tools.register({
      id: "a_tool",
      name: "A tool",
      description: "First by id",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute,
    });
    await runtime.skills.register({
      id: "skill-a",
      name: "Skill A",
      description: "Public skill metadata",
      tags: ["public"],
      tools: ["a_tool"],
      metadata: { stacks: ["typescript"] },
      body: "May contain private instructions",
    });

    const snapshot = runtime.catalog.snapshot();

    expect(snapshot).toEqual({
      source_id: "service-a",
      tools: [
        {
          id: "a_tool",
          name: "A tool",
          description: "First by id",
          inputSchema: { type: "object" },
          outputSchema: { type: "object" },
        },
        {
          id: "z_tool",
          name: "Z tool",
          description: "Last by id",
          inputSchema: { type: "object", properties: { value: { type: "string" } } },
          outputSchema: { type: "string" },
        },
      ],
      skills: [
        {
          id: "skill-a",
          name: "Skill A",
          description: "Public skill metadata",
          tags: ["public"],
          tools: ["a_tool"],
          metadata: { stacks: ["typescript"] },
        },
      ],
    });
    expect(JSON.stringify(snapshot)).not.toContain("execute");
    expect(JSON.stringify(snapshot)).not.toContain("private instructions");
  });
});

function runtimeEvent(fields: Record<string, unknown> = {}): RuntimeEvent {
  return {
    v: 2,
    event_id: "01K2KB4QN2A9XJY5VKQCN8ZM1P",
    ts: 1_755_000_000_000,
    session_id: "session-test",
    source_id: "source-test",
    type: "search",
    ...fields,
  };
}

function deliverRuntimeEvent(input: RuntimeEvent): RuntimeEvent {
  let deliver: (batch: RuntimeEvent[]) => void = () => {};
  const source = {
    subscribeEvents: (handler: (batch: RuntimeEvent[]) => void) => {
      deliver = handler;
      return { unsubscribe: () => {}, flush: async () => {}, droppedCount: 0 };
    },
  };
  const events = new RuntimeEvents([source] as never);
  let received: RuntimeEvent | undefined;
  events.subscribe((batch) => {
    received = batch[0];
  });

  deliver([input]);

  if (!received) throw new Error("runtime event was not delivered");
  return received;
}

function runtimeEventWithSerializedSize(size: number): RuntimeEvent {
  const padding = Object.fromEntries(
    Array.from({ length: 16 }, (_, index) => [`padding_${index.toString().padStart(2, "0")}`, ""]),
  );
  const event = runtimeEvent(padding);
  let remaining = size - Buffer.byteLength(JSON.stringify(event), "utf8");
  for (const key of Object.keys(padding)) {
    const length = Math.min(remaining, 4_096);
    event[key] = "x".repeat(length);
    remaining -= length;
  }
  if (remaining !== 0) throw new Error(`cannot create a ${size}-byte runtime event`);
  return event;
}
