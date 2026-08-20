import { describe, expect, it } from "vitest";
import {
  AuthOutcome,
  CAPTURE_CONTENT_ENV,
  EXECUTE_TOOL,
  EXPERIMENTAL_CATALOG_DEFINITIONS_ENV,
  ExperimentArmOutcome,
  ExperimentArmRole,
  ExperimentDropReason,
  ExperimentSkipReason,
  GEN_AI_INFERENCE_DETAILS,
  GEN_AI_OPERATION_NAME,
  GEN_AI_TOOL_CALL_ARGUMENTS,
  GEN_AI_TOOL_CALL_ID,
  GEN_AI_TOOL_CALL_RESULT,
  GEN_AI_TOOL_NAME,
  Origin,
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
  RATEL_CATALOG_SCHEMA_OMITTED,
  RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
  RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
  RATEL_CATALOG_TAGS,
  RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
  RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
  RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
  RATEL_EXPERIMENT_AGREEMENT_K,
  RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
  RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
  RATEL_EXPERIMENT_AGREEMENT_TOP1,
  RATEL_EXPERIMENT_ARM,
  RATEL_EXPERIMENT_ARM_BAGGAGE_KEY,
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
  RATEL_EXPERIMENT_ID_BAGGAGE_KEY,
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
  RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY,
  RATEL_EXPERIMENT_SELECTION_ID,
  RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY,
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
  RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY,
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
  SearchTarget,
} from "./index.js";

describe("ratel telemetry vocabulary", () => {
  it("pins the OTel gen_ai semconv version", () => {
    expect(SEMCONV_VERSION).toBe("1.42.0");
  });

  it("gates content capture on the ecosystem instrumentation env var", () => {
    expect(CAPTURE_CONTENT_ENV).toBe("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT");
    expect(EXPERIMENTAL_CATALOG_DEFINITIONS_ENV).toBe("RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS");
  });

  it("exports no exporter configuration: the host resolves its own endpoint and auth", async () => {
    const telemetry = await import("./index.js");
    // The vocabulary carries the constants plus the capture gate, nothing about transport.
    for (const removed of [
      "ENDPOINT_ENV",
      "OTLP_ENDPOINT_ENV",
      "API_KEY_ENV",
      "DEFAULT_SERVICE_NAME",
      "resolveOtlpConfig",
    ]) {
      expect(telemetry).not.toHaveProperty(removed);
    }
  });

  it("names the ratel.* spans per the pin", () => {
    expect(RATEL_SEARCH).toBe("ratel.search");
    expect(RATEL_EXPERIMENT_ARM).toBe("ratel.experiment.arm");
    expect(RATEL_SKILL_LOAD).toBe("ratel.skill.load");
    expect(RATEL_UPSTREAM_REGISTER).toBe("ratel.upstream.register");
    expect(RATEL_AUTH_FLOW).toBe("ratel.auth.flow");
  });

  it("names the EventRecords per the pin", () => {
    expect(RATEL_CATALOG_DEFINITION).toBe("ratel.catalog.definition");
    expect(RATEL_EXPERIMENT_RESULTS).toBe("ratel.experiment.results");
    expect(RATEL_EXPERIMENT_COMPARISON).toBe("ratel.experiment.comparison");
    expect(RATEL_EXPERIMENT_SKIP).toBe("ratel.experiment.skip");
    expect(RATEL_EXPERIMENT_FALLBACK).toBe("ratel.experiment.fallback");
    expect(RATEL_EXPERIMENT_DROP).toBe("ratel.experiment.drop");
    expect(RATEL_EXPERIMENT_INVOCATION).toBe("ratel.experiment.invocation");
    expect(RATEL_EXPERIMENT_OUTCOME).toBe("ratel.experiment.outcome");
    expect(RATEL_SEARCH_RESULTS).toBe("ratel.search.results");
    expect(RATEL_TOOL_EXECUTION_DETAILS).toBe("ratel.tool.execution.details");
    expect(GEN_AI_INFERENCE_DETAILS).toBe("gen_ai.client.inference.operation.details");
  });

  it("pins the catalog definition attribute vocabulary", () => {
    expect([
      RATEL_CATALOG_KIND,
      RATEL_CATALOG_ID,
      RATEL_CATALOG_NAME,
      RATEL_CATALOG_DESCRIPTION,
      RATEL_CATALOG_TAGS,
      RATEL_CATALOG_INPUT_SCHEMA,
      RATEL_CATALOG_OUTPUT_SCHEMA,
      RATEL_CATALOG_SCHEMA_OMITTED,
      RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
      RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
      RATEL_CATALOG_CONTENT_HASH,
    ]).toEqual([
      "ratel.catalog.kind",
      "ratel.catalog.id",
      "ratel.catalog.name",
      "ratel.catalog.description",
      "ratel.catalog.tags",
      "ratel.catalog.input_schema",
      "ratel.catalog.output_schema",
      "ratel.catalog.schema_omitted",
      "ratel.catalog.searchable_description",
      "ratel.catalog.searchable_description_overridden",
      "ratel.catalog.content_hash",
    ]);
  });

  it("models tool invocation as the gen_ai execute_tool operation, not ratel.invoke", () => {
    expect(EXECUTE_TOOL).toBe("execute_tool");
    expect(EXECUTE_TOOL).not.toBe("ratel.invoke");
  });

  it("matches the ratel.* attribute keys to the pin", () => {
    expect(RATEL_ORIGIN).toBe("ratel.origin");
    expect(RATEL_SEARCH_TARGET).toBe("ratel.search.target");
    expect(RATEL_SEARCH_TOP_K).toBe("ratel.search.top_k");
    expect(RATEL_SEARCH_HIT_COUNT).toBe("ratel.search.hit_count");
    expect(RATEL_SEARCH_QUERY).toBe("ratel.search.query");
    expect(RATEL_TOOL_ARGS_SIZE_BYTES).toBe("ratel.tool.args_size_bytes");
    expect(RATEL_UPSTREAM_SERVER).toBe("ratel.upstream.server");
    expect(RATEL_UPSTREAM_TRANSPORT).toBe("ratel.upstream.transport");
    expect(RATEL_UPSTREAM_TOOL_COUNT).toBe("ratel.upstream.tool_count");
    expect(RATEL_SKILL_ID).toBe("ratel.skill.id");
    expect(RATEL_AUTH_OUTCOME).toBe("ratel.auth.outcome");
    expect(RATEL_EXPERIMENT_ID).toBe("ratel.experiment.id");
    expect(RATEL_EXPERIMENT_SELECTION_ID).toBe("ratel.experiment.selection_id");
    expect(RATEL_EXPERIMENT_ARM).toBe("ratel.experiment.arm");
    expect(RATEL_EXPERIMENT_ROLE).toBe("ratel.experiment.role");
    expect(RATEL_EXPERIMENT_UNIT).toBe("ratel.experiment.unit");
    expect(RATEL_EXPERIMENT_COLD).toBe("ratel.experiment.cold");
    expect(RATEL_EXPERIMENT_OUTCOME).toBe("ratel.experiment.outcome");
    expect(RATEL_EXPERIMENT_DURATION_MS).toBe("ratel.experiment.duration_ms");
    expect(RATEL_EXPERIMENT_HIT_COUNT).toBe("ratel.experiment.hit_count");
    expect(RATEL_EXPERIMENT_RANKING_ERROR).toBe("ratel.experiment.ranking_error");
    expect(RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR).toBe(
      "ratel.experiment.result_attributes_error",
    );
    expect(RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR).toBe(
      "ratel.experiment.result_attrs_encoding_error",
    );
    expect(RATEL_EXPERIMENT_RESULT_IDS).toBe("ratel.experiment.result_ids");
    expect(RATEL_EXPERIMENT_RESULT_SCORES).toBe("ratel.experiment.result_scores");
    expect(RATEL_EXPERIMENT_RESULT_ATTRS).toBe("ratel.experiment.result_attrs");
    expect(RATEL_EXPERIMENT_SERVED_ARM).toBe("ratel.experiment.served.arm");
    expect(RATEL_EXPERIMENT_SERVED_OUTCOME).toBe("ratel.experiment.served.outcome");
    expect(RATEL_EXPERIMENT_SERVED_DURATION_MS).toBe("ratel.experiment.served.duration_ms");
    expect(RATEL_EXPERIMENT_SERVED_HIT_COUNT).toBe("ratel.experiment.served.hit_count");
    expect(RATEL_EXPERIMENT_SHADOW_ARM).toBe("ratel.experiment.shadow.arm");
    expect(RATEL_EXPERIMENT_SHADOW_OUTCOME).toBe("ratel.experiment.shadow.outcome");
    expect(RATEL_EXPERIMENT_SHADOW_DURATION_MS).toBe("ratel.experiment.shadow.duration_ms");
    expect(RATEL_EXPERIMENT_SHADOW_HIT_COUNT).toBe("ratel.experiment.shadow.hit_count");
    expect(RATEL_EXPERIMENT_AGREEMENT_TOP1).toBe("ratel.experiment.agreement.top1");
    expect(RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER).toBe("ratel.experiment.agreement.exact_order");
    expect(RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT).toBe(
      "ratel.experiment.agreement.overlap_count",
    );
    expect(RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K).toBe("ratel.experiment.agreement.jaccard_at_k");
    expect(RATEL_EXPERIMENT_AGREEMENT_K).toBe("ratel.experiment.agreement.k");
    expect(RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS).toBe("ratel.experiment.agreement.item_attrs");
    expect(RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS).toBe("ratel.experiment.agreement.result_attrs");
    expect(RATEL_EXPERIMENT_EFFECTIVE_ARM).toBe("ratel.experiment.effective_arm");
    expect(RATEL_EXPERIMENT_SKIP_ARM).toBe("ratel.experiment.skip.arm");
    expect(RATEL_EXPERIMENT_SKIP_CONCURRENCY).toBe("ratel.experiment.skip.concurrency");
    expect(RATEL_EXPERIMENT_SKIP_REASON).toBe("ratel.experiment.skip.reason");
    expect(RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM).toBe("ratel.experiment.fallback.effective_arm");
    expect(RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW).toBe("ratel.experiment.fallback.reused_shadow");
    expect(RATEL_EXPERIMENT_DROP_REASON).toBe("ratel.experiment.drop.reason");
    expect(RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED).toBe("ratel.experiment.invocation.attributed");
    expect(RATEL_EXPERIMENT_INVOCATION_RANK).toBe("ratel.experiment.invocation.rank");
    expect(RATEL_EXPERIMENT_INVOCATION_AGE_MS).toBe("ratel.experiment.invocation.age_ms");
    expect(RATEL_EXPERIMENT_TURN).toBe("ratel.experiment.turn");
    expect(RATEL_EXPERIMENT_OUTCOME_LABEL).toBe("ratel.experiment.outcome.label");
    expect(RATEL_EXPERIMENT_OUTCOME_SCORE).toBe("ratel.experiment.outcome.score");
  });

  it("aliases experiment baggage keys to their matching span attributes", () => {
    expect(RATEL_EXPERIMENT_ID_BAGGAGE_KEY).toBe(RATEL_EXPERIMENT_ID);
    expect(RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY).toBe(RATEL_EXPERIMENT_SELECTION_ID);
    expect(RATEL_EXPERIMENT_ARM_BAGGAGE_KEY).toBe(RATEL_EXPERIMENT_ARM);
    expect(RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY).toBe(RATEL_EXPERIMENT_ROLE);
    expect(RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY).toBe(RATEL_EXPERIMENT_UNIT);
  });

  it("keeps the borrowed gen_ai.* interop keys under gen_ai.*, never renamed into ratel.*", () => {
    expect(GEN_AI_OPERATION_NAME).toBe("gen_ai.operation.name");
    expect(GEN_AI_TOOL_NAME).toBe("gen_ai.tool.name");
    expect(GEN_AI_TOOL_CALL_ID).toBe("gen_ai.tool.call.id");
    expect(GEN_AI_TOOL_CALL_ARGUMENTS).toBe("gen_ai.tool.call.arguments");
    expect(GEN_AI_TOOL_CALL_RESULT).toBe("gen_ai.tool.call.result");
    for (const key of [
      GEN_AI_OPERATION_NAME,
      GEN_AI_TOOL_NAME,
      GEN_AI_TOOL_CALL_ID,
      GEN_AI_TOOL_CALL_ARGUMENTS,
      GEN_AI_TOOL_CALL_RESULT,
    ]) {
      expect(key.startsWith("gen_ai.")).toBe(true);
      expect(key.startsWith("ratel.")).toBe(false);
    }
  });

  it("namespaces every ratel.* attribute key under ratel.*", () => {
    for (const key of [
      RATEL_ORIGIN,
      RATEL_SEARCH_TARGET,
      RATEL_SEARCH_TOP_K,
      RATEL_SEARCH_HIT_COUNT,
      RATEL_SEARCH_QUERY,
      RATEL_TOOL_ARGS_SIZE_BYTES,
      RATEL_UPSTREAM_SERVER,
      RATEL_UPSTREAM_TRANSPORT,
      RATEL_UPSTREAM_TOOL_COUNT,
      RATEL_SKILL_ID,
      RATEL_AUTH_OUTCOME,
      RATEL_EXPERIMENT_ID,
      RATEL_EXPERIMENT_SELECTION_ID,
      RATEL_EXPERIMENT_ARM,
      RATEL_EXPERIMENT_ROLE,
      RATEL_EXPERIMENT_UNIT,
      RATEL_EXPERIMENT_COLD,
      RATEL_EXPERIMENT_OUTCOME,
      RATEL_EXPERIMENT_DURATION_MS,
      RATEL_EXPERIMENT_HIT_COUNT,
      RATEL_EXPERIMENT_RANKING_ERROR,
      RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
      RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
      RATEL_EXPERIMENT_RESULT_IDS,
      RATEL_EXPERIMENT_RESULT_SCORES,
      RATEL_EXPERIMENT_RESULT_ATTRS,
      RATEL_EXPERIMENT_SERVED_ARM,
      RATEL_EXPERIMENT_SERVED_OUTCOME,
      RATEL_EXPERIMENT_SERVED_DURATION_MS,
      RATEL_EXPERIMENT_SERVED_HIT_COUNT,
      RATEL_EXPERIMENT_SHADOW_ARM,
      RATEL_EXPERIMENT_SHADOW_OUTCOME,
      RATEL_EXPERIMENT_SHADOW_DURATION_MS,
      RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
      RATEL_EXPERIMENT_AGREEMENT_TOP1,
      RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
      RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
      RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
      RATEL_EXPERIMENT_AGREEMENT_K,
      RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
      RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
      RATEL_EXPERIMENT_EFFECTIVE_ARM,
      RATEL_EXPERIMENT_SKIP_ARM,
      RATEL_EXPERIMENT_SKIP_CONCURRENCY,
      RATEL_EXPERIMENT_SKIP_REASON,
      RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
      RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
      RATEL_EXPERIMENT_DROP_REASON,
      RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
      RATEL_EXPERIMENT_INVOCATION_RANK,
      RATEL_EXPERIMENT_INVOCATION_AGE_MS,
      RATEL_EXPERIMENT_TURN,
      RATEL_EXPERIMENT_OUTCOME_LABEL,
      RATEL_EXPERIMENT_OUTCOME_SCORE,
    ]) {
      expect(key.startsWith("ratel.")).toBe(true);
    }
  });

  it("keeps every attribute key unique (no copy-paste dup shares a wire key)", () => {
    const keys = [
      RATEL_ORIGIN,
      RATEL_SEARCH_TARGET,
      RATEL_SEARCH_TOP_K,
      RATEL_SEARCH_HIT_COUNT,
      RATEL_SEARCH_QUERY,
      RATEL_TOOL_ARGS_SIZE_BYTES,
      RATEL_UPSTREAM_SERVER,
      RATEL_UPSTREAM_TRANSPORT,
      RATEL_UPSTREAM_TOOL_COUNT,
      RATEL_SKILL_ID,
      RATEL_AUTH_OUTCOME,
      RATEL_EXPERIMENT_ID,
      RATEL_EXPERIMENT_SELECTION_ID,
      RATEL_EXPERIMENT_ARM,
      RATEL_EXPERIMENT_ROLE,
      RATEL_EXPERIMENT_UNIT,
      RATEL_EXPERIMENT_COLD,
      RATEL_EXPERIMENT_OUTCOME,
      RATEL_EXPERIMENT_DURATION_MS,
      RATEL_EXPERIMENT_HIT_COUNT,
      RATEL_EXPERIMENT_RANKING_ERROR,
      RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
      RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
      RATEL_EXPERIMENT_RESULT_IDS,
      RATEL_EXPERIMENT_RESULT_SCORES,
      RATEL_EXPERIMENT_RESULT_ATTRS,
      RATEL_EXPERIMENT_SERVED_ARM,
      RATEL_EXPERIMENT_SERVED_OUTCOME,
      RATEL_EXPERIMENT_SERVED_DURATION_MS,
      RATEL_EXPERIMENT_SERVED_HIT_COUNT,
      RATEL_EXPERIMENT_SHADOW_ARM,
      RATEL_EXPERIMENT_SHADOW_OUTCOME,
      RATEL_EXPERIMENT_SHADOW_DURATION_MS,
      RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
      RATEL_EXPERIMENT_AGREEMENT_TOP1,
      RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
      RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
      RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
      RATEL_EXPERIMENT_AGREEMENT_K,
      RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
      RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
      RATEL_EXPERIMENT_EFFECTIVE_ARM,
      RATEL_EXPERIMENT_SKIP_ARM,
      RATEL_EXPERIMENT_SKIP_CONCURRENCY,
      RATEL_EXPERIMENT_SKIP_REASON,
      RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
      RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
      RATEL_EXPERIMENT_DROP_REASON,
      RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
      RATEL_EXPERIMENT_INVOCATION_RANK,
      RATEL_EXPERIMENT_INVOCATION_AGE_MS,
      RATEL_EXPERIMENT_TURN,
      RATEL_EXPERIMENT_OUTCOME_LABEL,
      RATEL_EXPERIMENT_OUTCOME_SCORE,
      GEN_AI_OPERATION_NAME,
      GEN_AI_TOOL_NAME,
      GEN_AI_TOOL_CALL_ID,
      GEN_AI_TOOL_CALL_ARGUMENTS,
      GEN_AI_TOOL_CALL_RESULT,
    ];
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("keeps every EventRecord name unique", () => {
    const names = [
      RATEL_SEARCH_RESULTS,
      RATEL_TOOL_EXECUTION_DETAILS,
      RATEL_EXPERIMENT_RESULTS,
      RATEL_EXPERIMENT_COMPARISON,
      RATEL_EXPERIMENT_SKIP,
      RATEL_EXPERIMENT_FALLBACK,
      RATEL_EXPERIMENT_DROP,
      RATEL_EXPERIMENT_INVOCATION,
      RATEL_EXPERIMENT_OUTCOME,
      GEN_AI_INFERENCE_DETAILS,
    ];
    expect(new Set(names).size).toBe(names.length);
  });

  it("maps ratel.origin to its wire strings", () => {
    expect(Origin.Direct).toBe("direct");
    expect(Origin.Agent).toBe("agent");
    expect(Origin.Baseline).toBe("baseline");
  });

  it("maps ratel.search.target to its wire strings", () => {
    expect(SearchTarget.Tool).toBe("tool");
    expect(SearchTarget.Skill).toBe("skill");
    expect(SearchTarget.Fact).toBe("fact");
  });

  it("maps ratel.auth.outcome to its wire strings", () => {
    expect(AuthOutcome.Ok).toBe("ok");
    expect(AuthOutcome.Refreshed).toBe("refreshed");
    expect(AuthOutcome.NeedsAuth).toBe("needs_auth");
    expect(AuthOutcome.Failed).toBe("failed");
  });

  it("maps experiment arm roles to their wire strings", () => {
    expect(ExperimentArmRole.Serving).toBe("serving");
    expect(ExperimentArmRole.Shadow).toBe("shadow");
  });

  it("maps experiment arm outcomes to their wire strings", () => {
    expect(ExperimentArmOutcome.Ok).toBe("ok");
    expect(ExperimentArmOutcome.Empty).toBe("empty");
    expect(ExperimentArmOutcome.Timeout).toBe("timeout");
    expect(ExperimentArmOutcome.Error).toBe("error");
  });

  it("maps experiment skip reasons to their wire strings", () => {
    expect(ExperimentSkipReason.Capacity).toBe("capacity");
  });

  it("maps experiment drop reasons to their wire strings", () => {
    expect(ExperimentDropReason.ArmFailed).toBe("arm-failed");
    expect(ExperimentDropReason.FallbackConsumed).toBe("fallback-consumed");
    expect(ExperimentDropReason.SelectionFailed).toBe("selection-failed");
    expect(ExperimentDropReason.ServedRankingFailed).toBe("served-ranking-failed");
    expect(ExperimentDropReason.ComparisonFailed).toBe("comparison-failed");
  });
});
