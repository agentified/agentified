import { describe, expect, it } from "vitest";
import {
  AuthOutcome,
  CAPTURE_CONTENT_ENV,
  EXECUTE_TOOL,
  GEN_AI_INFERENCE_DETAILS,
  GEN_AI_OPERATION_NAME,
  GEN_AI_TOOL_CALL_ARGUMENTS,
  GEN_AI_TOOL_CALL_ID,
  GEN_AI_TOOL_CALL_RESULT,
  GEN_AI_TOOL_NAME,
  Origin,
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
  SearchTarget,
} from "./index.js";

describe("ratel telemetry vocabulary", () => {
  it("pins the OTel gen_ai semconv version", () => {
    expect(SEMCONV_VERSION).toBe("1.42.0");
  });

  it("gates content capture on the ecosystem instrumentation env var", () => {
    expect(CAPTURE_CONTENT_ENV).toBe("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT");
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
    expect(RATEL_SKILL_LOAD).toBe("ratel.skill.load");
    expect(RATEL_UPSTREAM_REGISTER).toBe("ratel.upstream.register");
    expect(RATEL_AUTH_FLOW).toBe("ratel.auth.flow");
  });

  it("names the EventRecords per the pin", () => {
    expect(RATEL_SEARCH_RESULTS).toBe("ratel.search.results");
    expect(RATEL_TOOL_EXECUTION_DETAILS).toBe("ratel.tool.execution.details");
    expect(GEN_AI_INFERENCE_DETAILS).toBe("gen_ai.client.inference.operation.details");
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
      GEN_AI_OPERATION_NAME,
      GEN_AI_TOOL_NAME,
      GEN_AI_TOOL_CALL_ID,
      GEN_AI_TOOL_CALL_ARGUMENTS,
      GEN_AI_TOOL_CALL_RESULT,
    ];
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("maps ratel.origin to its wire strings", () => {
    expect(Origin.Direct).toBe("direct");
    expect(Origin.Agent).toBe("agent");
  });

  it("maps ratel.search.target to its wire strings", () => {
    expect(SearchTarget.Tool).toBe("tool");
    expect(SearchTarget.Skill).toBe("skill");
  });

  it("maps ratel.auth.outcome to its wire strings", () => {
    expect(AuthOutcome.Ok).toBe("ok");
    expect(AuthOutcome.Refreshed).toBe("refreshed");
    expect(AuthOutcome.NeedsAuth).toBe("needs_auth");
    expect(AuthOutcome.Failed).toBe("failed");
  });
});
