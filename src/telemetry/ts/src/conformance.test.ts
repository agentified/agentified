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
  RATEL_CATALOG_CONTENT_HASH,
  RATEL_CATALOG_DEFINITION,
  RATEL_CATALOG_DESCRIPTION,
  RATEL_CATALOG_ID,
  RATEL_CATALOG_INPUT_SCHEMA,
  RATEL_CATALOG_KIND,
  RATEL_CATALOG_NAME,
  RATEL_CATALOG_OUTPUT_SCHEMA,
  RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
  RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
  RATEL_CATALOG_TAGS,
  RATEL_CATALOG_USE_CLOUD_DEFINITIONS,
  RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
  RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
  RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
  RATEL_EXPERIMENT_AGREEMENT_K,
  RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
  RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
  RATEL_EXPERIMENT_AGREEMENT_TOP1,
  RATEL_EXPERIMENT_ARM,
  RATEL_EXPERIMENT_COLD,
  RATEL_EXPERIMENT_COMPARISON,
  RATEL_EXPERIMENT_DROP,
  RATEL_EXPERIMENT_DROP_REASON,
  RATEL_EXPERIMENT_DURATION_MS,
  RATEL_EXPERIMENT_EFFECTIVE_ARM,
  RATEL_EXPERIMENT_FALLBACK,
  RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
  RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
  RATEL_EXPERIMENT_HIT_COUNT,
  RATEL_EXPERIMENT_ID,
  RATEL_EXPERIMENT_INVOCATION,
  RATEL_EXPERIMENT_INVOCATION_AGE_MS,
  RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
  RATEL_EXPERIMENT_INVOCATION_RANK,
  RATEL_EXPERIMENT_OUTCOME,
  RATEL_EXPERIMENT_OUTCOME_LABEL,
  RATEL_EXPERIMENT_OUTCOME_SCORE,
  RATEL_EXPERIMENT_RANKING_ERROR,
  RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
  RATEL_EXPERIMENT_RESULT_ATTRS,
  RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
  RATEL_EXPERIMENT_RESULT_IDS,
  RATEL_EXPERIMENT_RESULT_SCORES,
  RATEL_EXPERIMENT_RESULTS,
  RATEL_EXPERIMENT_ROLE,
  RATEL_EXPERIMENT_SELECTION_ID,
  RATEL_EXPERIMENT_SERVED_ARM,
  RATEL_EXPERIMENT_SERVED_DURATION_MS,
  RATEL_EXPERIMENT_SERVED_HIT_COUNT,
  RATEL_EXPERIMENT_SERVED_OUTCOME,
  RATEL_EXPERIMENT_SHADOW_ARM,
  RATEL_EXPERIMENT_SHADOW_DURATION_MS,
  RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
  RATEL_EXPERIMENT_SHADOW_OUTCOME,
  RATEL_EXPERIMENT_SKIP,
  RATEL_EXPERIMENT_SKIP_ARM,
  RATEL_EXPERIMENT_SKIP_CONCURRENCY,
  RATEL_EXPERIMENT_SKIP_REASON,
  RATEL_EXPERIMENT_TURN,
  RATEL_EXPERIMENT_UNIT,
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
  ratel_experiment_arm: RATEL_EXPERIMENT_ARM,
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
  ratel_experiment_agreement_exact_order: RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
  ratel_experiment_agreement_item_attrs: RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
  ratel_experiment_agreement_jaccard_at_k: RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
  ratel_experiment_agreement_k: RATEL_EXPERIMENT_AGREEMENT_K,
  ratel_experiment_agreement_overlap_count: RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
  ratel_experiment_agreement_result_attrs: RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
  ratel_experiment_agreement_top1: RATEL_EXPERIMENT_AGREEMENT_TOP1,
  ratel_experiment_arm: RATEL_EXPERIMENT_ARM,
  ratel_experiment_cold: RATEL_EXPERIMENT_COLD,
  ratel_experiment_drop_reason: RATEL_EXPERIMENT_DROP_REASON,
  ratel_experiment_duration_ms: RATEL_EXPERIMENT_DURATION_MS,
  ratel_experiment_effective_arm: RATEL_EXPERIMENT_EFFECTIVE_ARM,
  ratel_experiment_fallback_effective_arm: RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
  ratel_experiment_fallback_reused_shadow: RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
  ratel_experiment_hit_count: RATEL_EXPERIMENT_HIT_COUNT,
  ratel_experiment_id: RATEL_EXPERIMENT_ID,
  ratel_experiment_invocation_age_ms: RATEL_EXPERIMENT_INVOCATION_AGE_MS,
  ratel_experiment_invocation_attributed: RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
  ratel_experiment_invocation_rank: RATEL_EXPERIMENT_INVOCATION_RANK,
  ratel_experiment_outcome: RATEL_EXPERIMENT_OUTCOME,
  ratel_experiment_outcome_label: RATEL_EXPERIMENT_OUTCOME_LABEL,
  ratel_experiment_outcome_score: RATEL_EXPERIMENT_OUTCOME_SCORE,
  ratel_experiment_ranking_error: RATEL_EXPERIMENT_RANKING_ERROR,
  ratel_experiment_result_attributes_error: RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
  ratel_experiment_result_attrs: RATEL_EXPERIMENT_RESULT_ATTRS,
  ratel_experiment_result_attrs_encoding_error: RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
  ratel_experiment_result_ids: RATEL_EXPERIMENT_RESULT_IDS,
  ratel_experiment_result_scores: RATEL_EXPERIMENT_RESULT_SCORES,
  ratel_experiment_role: RATEL_EXPERIMENT_ROLE,
  ratel_experiment_served_arm: RATEL_EXPERIMENT_SERVED_ARM,
  ratel_experiment_served_duration_ms: RATEL_EXPERIMENT_SERVED_DURATION_MS,
  ratel_experiment_served_hit_count: RATEL_EXPERIMENT_SERVED_HIT_COUNT,
  ratel_experiment_served_outcome: RATEL_EXPERIMENT_SERVED_OUTCOME,
  ratel_experiment_selection_id: RATEL_EXPERIMENT_SELECTION_ID,
  ratel_experiment_shadow_arm: RATEL_EXPERIMENT_SHADOW_ARM,
  ratel_experiment_shadow_duration_ms: RATEL_EXPERIMENT_SHADOW_DURATION_MS,
  ratel_experiment_shadow_hit_count: RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
  ratel_experiment_shadow_outcome: RATEL_EXPERIMENT_SHADOW_OUTCOME,
  ratel_experiment_skip_arm: RATEL_EXPERIMENT_SKIP_ARM,
  ratel_experiment_skip_concurrency: RATEL_EXPERIMENT_SKIP_CONCURRENCY,
  ratel_experiment_skip_reason: RATEL_EXPERIMENT_SKIP_REASON,
  ratel_experiment_turn: RATEL_EXPERIMENT_TURN,
  ratel_experiment_unit: RATEL_EXPERIMENT_UNIT,
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
  ratel_catalog_content_hash: RATEL_CATALOG_CONTENT_HASH,
  ratel_catalog_description: RATEL_CATALOG_DESCRIPTION,
  ratel_catalog_id: RATEL_CATALOG_ID,
  ratel_catalog_input_schema: RATEL_CATALOG_INPUT_SCHEMA,
  ratel_catalog_kind: RATEL_CATALOG_KIND,
  ratel_catalog_name: RATEL_CATALOG_NAME,
  ratel_catalog_output_schema: RATEL_CATALOG_OUTPUT_SCHEMA,
  ratel_catalog_searchable_description: RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
  ratel_catalog_searchable_description_overridden: RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
  ratel_catalog_use_cloud_definitions: RATEL_CATALOG_USE_CLOUD_DEFINITIONS,
  ratel_catalog_tags: RATEL_CATALOG_TAGS,
};

// Logical event id -> the event-name constant under test.
const EVENT_NAME: Record<string, string> = {
  ratel_catalog_definition: RATEL_CATALOG_DEFINITION,
  ratel_experiment_comparison: RATEL_EXPERIMENT_COMPARISON,
  ratel_experiment_drop: RATEL_EXPERIMENT_DROP,
  ratel_experiment_fallback: RATEL_EXPERIMENT_FALLBACK,
  ratel_experiment_invocation: RATEL_EXPERIMENT_INVOCATION,
  ratel_experiment_outcome: RATEL_EXPERIMENT_OUTCOME,
  ratel_experiment_results: RATEL_EXPERIMENT_RESULTS,
  ratel_experiment_skip: RATEL_EXPERIMENT_SKIP,
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
  dedupe_catalog_definitions?: boolean;
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
  const seenDefinitions = new Set<string>();
  for (const [field, value] of Object.entries(fixture.set)) {
    span.setAttribute(ATTR_KEY[field], value as string | number);
  }
  for (const event of fixture.emit_events ?? []) {
    if (fixture.dedupe_catalog_definitions && event.event === "ratel_catalog_definition") {
      const key = `${event.attributes.ratel_catalog_id}\0${event.attributes.ratel_catalog_content_hash}`;
      if (seenDefinitions.has(key)) continue;
      seenDefinitions.add(key);
    }
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
