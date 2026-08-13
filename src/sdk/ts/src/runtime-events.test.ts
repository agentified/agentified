import { trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { afterEach, describe, expect, it } from "vitest";
import { type RuntimeEvent, ratel } from "./index.js";

describe("public runtime events", () => {
  afterEach(() => trace.disable());

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

    const streamEvent = received.find(
      (event) => event.type === "search" && event.query === "read",
    );
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
      sourceId: "service-a",
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
