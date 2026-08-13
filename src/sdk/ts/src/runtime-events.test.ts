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
import { afterEach, describe, expect, it } from "vitest";
import {
  experimentalDefineExperiment,
  RUNTIME_EVENT_MAX_HITS,
  RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
  RUNTIME_EVENT_MAX_QUERY_BYTES,
  RUNTIME_EVENT_TYPES,
  type RuntimeEvent,
  ratel,
} from "./index.js";

interface RuntimeEventsFixture {
  runtime_events: {
    version: number;
    max_payload_bytes: number;
    max_query_bytes: number;
    max_hits: number;
    required_envelope_fields: string[];
    event_types: string[];
  };
}

const conformance = JSON.parse(
  readFileSync(new URL("../../../telemetry/conformance/fixtures.json", import.meta.url), "utf8"),
) as RuntimeEventsFixture;

describe("public runtime events", () => {
  afterEach(() => {
    logs.disable();
    trace.disable();
  });

  it("matches the frozen cross-language event vocabulary", () => {
    expect(conformance.runtime_events).toEqual({
      version: 2,
      max_payload_bytes: RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
      max_query_bytes: RUNTIME_EVENT_MAX_QUERY_BYTES,
      max_hits: RUNTIME_EVENT_MAX_HITS,
      required_envelope_fields: ["v", "event_id", "ts", "session_id", "source_id", "type"],
      event_types: [...RUNTIME_EVENT_TYPES],
    });
  });

  it("enforces public query, hit, and payload bounds before delivery", async () => {
    const runtime = ratel();
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));

    runtime.events.emit({
      type: "search",
      search_id: "search-1",
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
