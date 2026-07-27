import { readFileSync } from "node:fs";
import { type LogAttributes, logs } from "@opentelemetry/api-logs";
import {
  InMemoryLogRecordExporter,
  LoggerProvider,
  SimpleLogRecordProcessor,
} from "@opentelemetry/sdk-logs";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { describe, expect, it } from "vitest";
import {
  EXECUTE_TOOL,
  GEN_AI_OPERATION_NAME,
  GEN_AI_TOOL_CALL_ARGUMENTS,
  GEN_AI_TOOL_CALL_ID,
  GEN_AI_TOOL_CALL_RESULT,
  GEN_AI_TOOL_NAME,
  RATEL_AUTH_FLOW,
  RATEL_AUTH_OUTCOME,
  RATEL_ORIGIN,
  RATEL_SEARCH,
  RATEL_SEARCH_HIT_COUNT,
  RATEL_SEARCH_QUERY,
  RATEL_SEARCH_RESULTS,
  RATEL_SEARCH_TARGET,
  RATEL_SEARCH_TOP_K,
  RATEL_SKILL_ID,
  RATEL_SKILL_LOAD,
  RATEL_TOOL_ARGS_SIZE_BYTES,
  RATEL_TOOL_EXECUTION_DETAILS,
  RATEL_UPSTREAM_REGISTER,
  RATEL_UPSTREAM_SERVER,
  RATEL_UPSTREAM_TOOL_COUNT,
  RATEL_UPSTREAM_TRANSPORT,
  SEMCONV_VERSION,
} from "./index.js";

// Logical span id -> the span-name constant under test.
const SPAN_NAME: Record<string, string> = {
  execute_tool: EXECUTE_TOOL,
  ratel_search: RATEL_SEARCH,
  ratel_skill_load: RATEL_SKILL_LOAD,
  ratel_upstream_register: RATEL_UPSTREAM_REGISTER,
  ratel_auth_flow: RATEL_AUTH_FLOW,
};

// Logical attribute id -> the attribute-key constant under test.
const ATTR_KEY: Record<string, string> = {
  gen_ai_operation_name: GEN_AI_OPERATION_NAME,
  gen_ai_tool_name: GEN_AI_TOOL_NAME,
  gen_ai_tool_call_id: GEN_AI_TOOL_CALL_ID,
  gen_ai_tool_call_arguments: GEN_AI_TOOL_CALL_ARGUMENTS,
  gen_ai_tool_call_result: GEN_AI_TOOL_CALL_RESULT,
  ratel_origin: RATEL_ORIGIN,
  ratel_tool_args_size_bytes: RATEL_TOOL_ARGS_SIZE_BYTES,
  ratel_upstream_server: RATEL_UPSTREAM_SERVER,
  ratel_search_target: RATEL_SEARCH_TARGET,
  ratel_search_top_k: RATEL_SEARCH_TOP_K,
  ratel_search_hit_count: RATEL_SEARCH_HIT_COUNT,
  ratel_search_query: RATEL_SEARCH_QUERY,
  ratel_skill_id: RATEL_SKILL_ID,
  ratel_upstream_transport: RATEL_UPSTREAM_TRANSPORT,
  ratel_upstream_tool_count: RATEL_UPSTREAM_TOOL_COUNT,
  ratel_auth_outcome: RATEL_AUTH_OUTCOME,
};

// Logical event id -> the event-name constant under test.
const EVENT_NAME: Record<string, string> = {
  ratel_search_results: RATEL_SEARCH_RESULTS,
  ratel_tool_execution_details: RATEL_TOOL_EXECUTION_DETAILS,
};

interface EventFixture {
  event: string;
  attributes: Record<string, unknown>;
}

interface ExpectedEvent {
  name: string;
  attributes: Record<string, unknown>;
}

interface Fixture {
  name: string;
  span: string;
  set: Record<string, unknown>;
  emit_events?: EventFixture[];
  expect_name: string;
  expect_attributes: Record<string, unknown>;
  expect_events?: ExpectedEvent[];
}

interface FixtureFile {
  semconv_version: string;
  fixtures: Fixture[];
}

const fixtures: FixtureFile = JSON.parse(
  readFileSync(new URL("../../conformance/fixtures.json", import.meta.url), "utf8"),
);

async function emit(fixture: Fixture): Promise<{
  name: string;
  attributes: Record<string, unknown>;
  events: ExpectedEvent[];
}> {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  const tracer = provider.getTracer("conformance");
  const logExporter = new InMemoryLogRecordExporter();
  const loggerProvider = new LoggerProvider({
    processors: [new SimpleLogRecordProcessor({ exporter: logExporter })],
  });
  logs.setGlobalLoggerProvider(loggerProvider);
  const logger = logs.getLogger("conformance");
  const span = tracer.startSpan(SPAN_NAME[fixture.span]);
  for (const [field, value] of Object.entries(fixture.set)) {
    span.setAttribute(ATTR_KEY[field], value as string | number);
  }
  for (const event of fixture.emit_events ?? []) {
    const attributes = Object.fromEntries(
      Object.entries(event.attributes).map(([field, value]) => [ATTR_KEY[field], value]),
    ) as LogAttributes;
    logger.emit({ eventName: EVENT_NAME[event.event], attributes });
  }
  span.end();
  const [emitted] = exporter.getFinishedSpans();
  const result = {
    name: emitted.name,
    attributes: { ...emitted.attributes },
    events: logExporter.getFinishedLogRecords().map((record) => ({
      name: record.eventName ?? "",
      attributes: { ...record.attributes },
    })),
  };
  logs.disable();
  await Promise.all([provider.shutdown(), loggerProvider.shutdown()]);
  return result;
}

describe("telemetry conformance (contract against the pin)", () => {
  it("shares the pinned semconv version with the vocabulary", () => {
    expect(fixtures.semconv_version).toBe(SEMCONV_VERSION);
  });

  for (const fixture of fixtures.fixtures) {
    it(`emits the pinned keys: ${fixture.name}`, async () => {
      const { name, attributes, events } = await emit(fixture);
      expect(name).toBe(fixture.expect_name);
      expect(attributes).toEqual(fixture.expect_attributes);
      expect(events).toEqual(fixture.expect_events ?? []);
    });
  }
});
