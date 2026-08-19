/**
 * OpenTelemetry emission for the SDK's `ratel.*` / `gen_ai.*` funnel (ADR-0007).
 * The catalog, capability-tool, skill, and MCP paths call these helpers to
 * open a span around each operation and, when selected by content capture, emit a
 * structured Logs EventRecord. Names and attribute keys come from the OTel-free
 * `@ratel-ai/telemetry` vocabulary.
 *
 * Emission is **transparent**: records go to whichever OpenTelemetry tracer and logger
 * providers are registered globally. The SDK never registers one — that is the host's
 * job (`new NodeSDK({ spanProcessors })`). Until the host wires providers, every span is
 * a no-op `NonRecordingSpan`, so instrumentation is effectively free and the local trace
 * stream (`recordEvent`) is untouched. This mirrors how the Vercel AI SDK instruments:
 * the library emits; the app decides where it goes.
 *
 * Message/tool content (`ratel.search.query`, tool args/result) follows the ecosystem
 * capture gate's span-attribute and Logs EventRecord channels (default off), per ADR-0007.
 */

import { createHash } from "node:crypto";
import {
  context,
  type Context as OtelContext,
  propagation,
  type Span,
  SpanKind,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";
import { type AnyValue, type AnyValueMap, logs } from "@opentelemetry/api-logs";
import {
  AuthOutcome,
  ContentCapture,
  clearContentCapture,
  contentCaptureMode,
  EXECUTE_TOOL,
  EXPERIMENTAL_CATALOG_DEFINITIONS_ENV,
  GEN_AI_OPERATION_NAME,
  GEN_AI_TOOL_CALL_ARGUMENTS,
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
  RATEL_CATALOG_SCHEMA_OMITTED,
  RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
  RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
  RATEL_CATALOG_TAGS,
  RATEL_CATALOG_USE_DEFINITION_OVERRIDES,
  RATEL_EVENT_ID,
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
  type SearchTarget,
  setContentCapture,
} from "@ratel-ai/telemetry";
import canonicalize from "canonicalize";
import { isAsyncIterable, isPromiseLike } from "./async.js";
import type { SearchOrigin } from "./catalog.js";
import type {
  ExperimentArmCompletionEvaluation,
  ExperimentArmEvaluation,
  ExperimentArmEvaluationHandle,
  ExperimentEvaluationSink,
} from "./experiment-sink.js";
import type { ExperimentRankedItem } from "./experiment-types.js";
import { newRuntimeEventId } from "./runtime-events.js";

const TRACER_NAME = "@ratel-ai/sdk";
const LOGGER_NAME = "@ratel-ai/sdk";
const ERROR_TYPE = "error.type";
const CATALOG_SCHEMA_MAX_ATTRIBUTE_BYTES = 64 * 1_024;
const RESERVED_EXPERIMENT_ATTRIBUTES = new Set([
  ERROR_TYPE,
  GEN_AI_TOOL_NAME,
  RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
  RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
  RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
  RATEL_EXPERIMENT_AGREEMENT_K,
  RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
  RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
  RATEL_EXPERIMENT_AGREEMENT_TOP1,
  RATEL_EXPERIMENT_ARM,
  RATEL_EXPERIMENT_COLD,
  RATEL_EXPERIMENT_DROP_REASON,
  RATEL_EXPERIMENT_DURATION_MS,
  RATEL_EXPERIMENT_EFFECTIVE_ARM,
  RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
  RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
  RATEL_EXPERIMENT_HIT_COUNT,
  RATEL_EXPERIMENT_ID,
  RATEL_EXPERIMENT_INVOCATION_AGE_MS,
  RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
  RATEL_EXPERIMENT_INVOCATION_RANK,
  RATEL_EXPERIMENT_OUTCOME,
  RATEL_EXPERIMENT_OUTCOME_LABEL,
  RATEL_EXPERIMENT_OUTCOME_SCORE,
  RATEL_EXPERIMENT_RANKING_ERROR,
  RATEL_EXPERIMENT_RESULT_ATTRS,
  RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
  RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
  RATEL_EXPERIMENT_RESULT_IDS,
  RATEL_EXPERIMENT_RESULT_SCORES,
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
  RATEL_EXPERIMENT_SKIP_ARM,
  RATEL_EXPERIMENT_SKIP_CONCURRENCY,
  RATEL_EXPERIMENT_SKIP_REASON,
  RATEL_EXPERIMENT_TURN,
  RATEL_EXPERIMENT_UNIT,
]);

/** @internal Identity shared by one runtime envelope and its OTel projection. */
export interface RuntimeEventProjection {
  eventId: string;
  invocationId?: string;
  traceId?: string;
  spanId?: string;
}

/** @internal Definition fields shared by tool, skill, and fact registrations. */
export interface CatalogDefinitionInput {
  id: string;
  name: string;
  description: string;
  experimentalSearchableDescription?: string;
  tags?: string[];
  inputSchema?: unknown;
  outputSchema?: unknown;
}

/**
 * Project changed catalog definitions into the opt-in Logs channel.
 * `emittedHashes` scopes deduplication to one registry/session.
 * @internal
 */
export function recordCatalogDefinitions(
  kind: "tool" | "skill" | "fact",
  definitions: readonly CatalogDefinitionInput[],
  emittedHashes: Map<string, string>,
  useCloudDefinitions = false,
): void {
  if (
    process.env[EXPERIMENTAL_CATALOG_DEFINITIONS_ENV]?.toLowerCase() !== "true" ||
    !captureContentOnEvent()
  ) {
    return;
  }
  for (const definition of definitions) {
    let attributes: Record<string, AnyValue> | undefined;
    try {
      attributes = catalogDefinitionAttributes(kind, definition);
    } catch {
      continue;
    }
    if (attributes === undefined) continue;
    const contentHash = attributes[RATEL_CATALOG_CONTENT_HASH] as string;
    const dedupeHash = useCloudDefinitions ? `${contentHash}:overrides` : contentHash;
    if (emittedHashes.get(definition.id) === dedupeHash) continue;
    getLogger().emit({
      eventName: RATEL_CATALOG_DEFINITION,
      attributes: {
        ...attributes,
        ...(useCloudDefinitions ? { [RATEL_CATALOG_USE_DEFINITION_OVERRIDES]: true } : {}),
      },
    });
    emittedHashes.set(definition.id, dedupeHash);
  }
}

function catalogDefinitionAttributes(
  kind: "tool" | "skill" | "fact",
  definition: CatalogDefinitionInput,
): Record<string, AnyValue> | undefined {
  const tags = definition.tags ?? [];
  const searchableDescription =
    definition.experimentalSearchableDescription ?? definition.description;
  const searchableDescriptionOverridden =
    definition.experimentalSearchableDescription !== undefined;
  const inputSchema = kind === "tool" ? definition.inputSchema : null;
  const outputSchema = kind === "tool" ? definition.outputSchema : null;
  const content = {
    kind,
    id: definition.id,
    name: definition.name,
    description: definition.description,
    tags,
    input_schema: inputSchema,
    output_schema: outputSchema,
    searchable_description: searchableDescription,
    searchable_description_overridden: searchableDescriptionOverridden,
  };
  if (hasUnsafeInteger(content)) return undefined;
  const contentHash = createHash("sha256").update(canonicalJson(content), "utf8").digest("hex");
  const canonicalInputSchema = kind === "tool" ? canonicalJson(inputSchema) : undefined;
  const canonicalOutputSchema = kind === "tool" ? canonicalJson(outputSchema) : undefined;
  const inputSchemaOmitted = exceedsCatalogSchemaAttributeLimit(canonicalInputSchema);
  const outputSchemaOmitted = exceedsCatalogSchemaAttributeLimit(canonicalOutputSchema);
  return {
    [RATEL_CATALOG_KIND]: kind,
    [RATEL_CATALOG_ID]: definition.id,
    [RATEL_CATALOG_NAME]: definition.name,
    [RATEL_CATALOG_DESCRIPTION]: definition.description,
    [RATEL_CATALOG_TAGS]: tags,
    ...(kind === "tool"
      ? {
          ...(inputSchemaOmitted
            ? {}
            : { [RATEL_CATALOG_INPUT_SCHEMA]: canonicalInputSchema as string }),
          ...(outputSchemaOmitted
            ? {}
            : { [RATEL_CATALOG_OUTPUT_SCHEMA]: canonicalOutputSchema as string }),
          ...(inputSchemaOmitted || outputSchemaOmitted
            ? { [RATEL_CATALOG_SCHEMA_OMITTED]: true }
            : {}),
        }
      : {}),
    [RATEL_CATALOG_SEARCHABLE_DESCRIPTION]: searchableDescription,
    [RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN]: searchableDescriptionOverridden,
    [RATEL_CATALOG_CONTENT_HASH]: contentHash,
  };
}

function exceedsCatalogSchemaAttributeLimit(value: string | undefined): boolean {
  return (
    value !== undefined && Buffer.byteLength(value, "utf8") > CATALOG_SCHEMA_MAX_ATTRIBUTE_BYTES
  );
}

function hasUnsafeInteger(value: unknown): boolean {
  if (typeof value === "number") return Number.isInteger(value) && !Number.isSafeInteger(value);
  if (Array.isArray(value)) return value.some(hasUnsafeInteger);
  if (value !== null && typeof value === "object") {
    return Object.values(value).some(hasUnsafeInteger);
  }
  return false;
}

function canonicalJson(value: unknown): string {
  const canonical = canonicalize(value);
  if (canonical === undefined) throw new TypeError("catalog definition is not JSON serializable");
  return canonical;
}

function eventProjection(span: Span, invocationId?: string): RuntimeEventProjection {
  const eventId = newRuntimeEventId();
  const spanContext = span.spanContext();
  span.setAttribute(RATEL_EVENT_ID, eventId);
  return {
    eventId,
    ...(invocationId === undefined ? {} : { invocationId }),
    ...(spanContext.isRemote || /^0+$/.test(spanContext.traceId)
      ? {}
      : { traceId: spanContext.traceId, spanId: spanContext.spanId }),
  };
}

function getTracer() {
  return trace.getTracer(TRACER_NAME);
}

function getLogger() {
  return logs.getLogger(LOGGER_NAME);
}

/** Validate and snapshot caller-supplied scalar experiment telemetry. */
export function validateExperimentAttributes(
  attributes: Record<string, string | number | boolean> | undefined,
): Record<string, string | number | boolean> | undefined {
  if (attributes === undefined) {
    return undefined;
  }
  for (const key of Object.keys(attributes)) {
    if (RESERVED_EXPERIMENT_ATTRIBUTES.has(key)) {
      throw new Error(
        `experimentalDefineExperiment.select: "${key}" is a reserved telemetry attribute`,
      );
    }
  }
  return { ...attributes };
}

/** Build the SDK-owned OpenTelemetry sink used by the public experiment factory. */
export function createExperimentTelemetrySink<Arm extends string>(): ExperimentEvaluationSink<Arm> {
  return {
    arm(evaluation) {
      return startExperimentArm(evaluation);
    },
    comparison(evaluation, arm) {
      arm?.event(
        RATEL_EXPERIMENT_COMPARISON,
        {
          [RATEL_EXPERIMENT_SERVED_ARM]: evaluation.served.arm,
          [RATEL_EXPERIMENT_SERVED_OUTCOME]: evaluation.served.outcome,
          [RATEL_EXPERIMENT_SERVED_DURATION_MS]: evaluation.served.durationMs,
          [RATEL_EXPERIMENT_SERVED_HIT_COUNT]: evaluation.served.hitCount,
          [RATEL_EXPERIMENT_SHADOW_ARM]: evaluation.shadow.arm,
          [RATEL_EXPERIMENT_SHADOW_OUTCOME]: evaluation.shadow.outcome,
          [RATEL_EXPERIMENT_SHADOW_DURATION_MS]: evaluation.shadow.durationMs,
          [RATEL_EXPERIMENT_SHADOW_HIT_COUNT]: evaluation.shadow.hitCount,
          [RATEL_EXPERIMENT_AGREEMENT_TOP1]: evaluation.agreement.top1,
          [RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER]: evaluation.agreement.exactOrder,
          [RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT]: evaluation.agreement.overlapCount,
          [RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K]: evaluation.agreement.jaccardAtK,
          [RATEL_EXPERIMENT_AGREEMENT_K]: evaluation.agreement.k,
          ...(evaluation.agreement.itemAttrs === undefined
            ? {}
            : { [RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS]: evaluation.agreement.itemAttrs }),
          ...(evaluation.agreement.resultAttrs === undefined
            ? {}
            : { [RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS]: evaluation.agreement.resultAttrs }),
        },
        evaluation.eventId,
      );
    },
    skip(evaluation, arm) {
      arm?.event(
        RATEL_EXPERIMENT_SKIP,
        {
          [RATEL_EXPERIMENT_SKIP_ARM]: evaluation.skippedArm,
          [RATEL_EXPERIMENT_SKIP_CONCURRENCY]: evaluation.concurrency,
          [RATEL_EXPERIMENT_SKIP_REASON]: "capacity",
        },
        evaluation.eventId,
      );
    },
    fallback(evaluation, arm) {
      arm?.event(
        RATEL_EXPERIMENT_FALLBACK,
        {
          [RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM]: evaluation.effectiveArm,
          [RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW]: evaluation.reusedShadow,
        },
        evaluation.eventId,
      );
    },
    drop(evaluation, arm) {
      arm?.event(
        RATEL_EXPERIMENT_DROP,
        {
          [RATEL_EXPERIMENT_DROP_REASON]: evaluation.reason,
        },
        evaluation.eventId,
      );
    },
    invocation(evaluation) {
      const attribution = evaluation.attribution;
      getLogger().emit({
        eventName: RATEL_EXPERIMENT_INVOCATION,
        attributes: {
          [RATEL_EVENT_ID]: evaluation.eventId ?? newRuntimeEventId(),
          [RATEL_EXPERIMENT_ID]: evaluation.experimentId,
          [RATEL_EXPERIMENT_UNIT]: evaluation.unitHash,
          [GEN_AI_TOOL_NAME]: evaluation.toolId,
          [RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED]: attribution.attributed,
          ...(evaluation.turn === undefined ? {} : { [RATEL_EXPERIMENT_TURN]: evaluation.turn }),
          ...(attribution.attributed
            ? {
                [RATEL_EXPERIMENT_SELECTION_ID]: attribution.selectionId,
                [RATEL_EXPERIMENT_EFFECTIVE_ARM]: attribution.effectiveArm,
                [RATEL_EXPERIMENT_INVOCATION_RANK]: attribution.rank,
                [RATEL_EXPERIMENT_INVOCATION_AGE_MS]: attribution.ageMs,
              }
            : {}),
        },
        context: context.active(),
      });
    },
    outcome(evaluation) {
      getLogger().emit({
        eventName: RATEL_EXPERIMENT_OUTCOME,
        attributes: {
          [RATEL_EVENT_ID]: evaluation.eventId ?? newRuntimeEventId(),
          [RATEL_EXPERIMENT_ID]: evaluation.experimentId,
          [RATEL_EXPERIMENT_SELECTION_ID]: evaluation.selectionId,
          ...(evaluation.label === undefined
            ? {}
            : { [RATEL_EXPERIMENT_OUTCOME_LABEL]: evaluation.label }),
          ...(evaluation.score === undefined
            ? {}
            : { [RATEL_EXPERIMENT_OUTCOME_SCORE]: evaluation.score }),
        },
        context: context.active(),
      });
    },
  };
}

function startExperimentArm<Arm extends string>(
  evaluation: ExperimentArmEvaluation<Arm>,
): ExperimentArmEvaluationHandle {
  const stamp = {
    [RATEL_EXPERIMENT_ID]: evaluation.experimentId,
    [RATEL_EXPERIMENT_SELECTION_ID]: evaluation.selectionId,
    [RATEL_EXPERIMENT_ARM]: evaluation.arm,
    [RATEL_EXPERIMENT_ROLE]: evaluation.role,
    [RATEL_EXPERIMENT_UNIT]: evaluation.unitHash,
  };
  const attributes = { ...evaluation.attributes, ...stamp };
  const parentContext = context.active();
  let baggage = propagation.getBaggage(parentContext) ?? propagation.createBaggage();
  baggage = baggage
    .setEntry(RATEL_EXPERIMENT_ID_BAGGAGE_KEY, { value: evaluation.experimentId })
    .setEntry(RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY, { value: evaluation.selectionId })
    .setEntry(RATEL_EXPERIMENT_ARM_BAGGAGE_KEY, { value: evaluation.arm })
    .setEntry(RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY, { value: evaluation.role })
    .setEntry(RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY, { value: evaluation.unitHash });
  const baggageContext = propagation.setBaggage(parentContext, baggage);
  const span = getTracer().startSpan(
    RATEL_EXPERIMENT_ARM,
    {
      kind: SpanKind.INTERNAL,
      attributes: {
        ...attributes,
        [RATEL_EVENT_ID]: evaluation.eventId ?? newRuntimeEventId(),
        [RATEL_EXPERIMENT_COLD]: evaluation.cold,
      },
    },
    baggageContext,
  );
  const armContext = trace.setSpan(baggageContext, span);

  return {
    run: (callback) => context.with(armContext, callback),
    complete: (completion) => completeExperimentArm(span, armContext, attributes, completion),
    event: (eventName, eventAttributes, eventId) =>
      addExperimentEvent(eventName, attributes, eventAttributes, armContext, eventId),
  };
}

function addExperimentEvent(
  eventName: string,
  armAttributes: Record<string, string | number | boolean>,
  eventAttributes: Record<string, unknown>,
  armContext: OtelContext,
  eventId?: string,
): void {
  getLogger().emit({
    eventName,
    attributes: {
      ...armAttributes,
      ...eventAttributes,
      [RATEL_EVENT_ID]: eventId ?? newRuntimeEventId(),
    } as AnyValueMap,
    context: armContext,
  });
}

function completeExperimentArm(
  span: Span,
  armContext: OtelContext,
  attributes: Record<string, string | number | boolean>,
  completion: ExperimentArmCompletionEvaluation,
): void {
  try {
    span.setAttribute(RATEL_EXPERIMENT_OUTCOME, completion.outcome);
    span.setAttribute(RATEL_EXPERIMENT_DURATION_MS, completion.durationMs);
    if (completion.hitCount !== undefined) {
      span.setAttribute(RATEL_EXPERIMENT_HIT_COUNT, completion.hitCount);
    }
    if (completion.failure !== undefined) {
      span.setAttribute(ERROR_TYPE, errorType(completion.failure.error));
      if (completion.failure.error instanceof Error) {
        span.recordException(completion.failure.error);
      }
    }
    if (completion.rankingFailure !== undefined) {
      span.setAttribute(RATEL_EXPERIMENT_RANKING_ERROR, errorType(completion.rankingFailure.error));
    }
    if (completion.resultAttributesFailure !== undefined) {
      span.setAttribute(
        RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
        errorType(completion.resultAttributesFailure.error),
      );
      if (completion.resultAttributesFailure.error instanceof Error) {
        span.recordException(completion.resultAttributesFailure.error);
      }
    }
    span.setStatus({
      code:
        completion.outcome === "ok" || completion.outcome === "empty"
          ? SpanStatusCode.OK
          : SpanStatusCode.ERROR,
      ...(completion.failure === undefined
        ? {}
        : { message: errorMessage(completion.failure.error) }),
    });
    if (completion.ranking !== undefined) {
      const itemAttributes = encodeExperimentResultAttributes(span, completion.ranking);
      addExperimentResultsEvent(
        completion.ranking,
        attributes,
        armContext,
        itemAttributes?.eventValue,
        completion.eventId,
      );
    }
  } finally {
    span.end();
  }
}

function addExperimentResultsEvent(
  ranking: readonly ExperimentRankedItem[],
  attributes: Record<string, string | number | boolean>,
  armContext: OtelContext,
  resultAttributes?: AnyValue,
  eventId?: string,
): void {
  const scores = ranking.map((item) => item.score);
  getLogger().emit({
    eventName: RATEL_EXPERIMENT_RESULTS,
    attributes: {
      ...attributes,
      [RATEL_EVENT_ID]: eventId ?? newRuntimeEventId(),
      [RATEL_EXPERIMENT_RESULT_IDS]: ranking.map((item) => item.id),
      ...(scores.every(
        (score): score is number => typeof score === "number" && Number.isFinite(score),
      )
        ? { [RATEL_EXPERIMENT_RESULT_SCORES]: scores }
        : {}),
      ...(resultAttributes === undefined
        ? {}
        : { [RATEL_EXPERIMENT_RESULT_ATTRS]: resultAttributes }),
    },
    context: armContext,
  });
}

function encodeExperimentResultAttributes(
  span: Span,
  ranking: readonly ExperimentRankedItem[],
): { eventValue?: AnyValue } | undefined {
  if (!captureContentOnSpan() && !captureContentOnEvent()) {
    return undefined;
  }
  try {
    const encoded = JSON.stringify(ranking.map((item) => item.attrs ?? null));
    if (encoded === undefined) {
      throw new Error("experiment result attributes are not JSON-encodable");
    }
    if (captureContentOnSpan()) {
      span.setAttribute(RATEL_EXPERIMENT_RESULT_ATTRS, encoded);
    }
    return {
      ...(captureContentOnEvent() ? { eventValue: JSON.parse(encoded) as AnyValue } : {}),
    };
  } catch (error) {
    span.setAttribute(RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR, errorType(error));
    if (error instanceof Error) {
      span.recordException(error);
    }
    return undefined;
  }
}

function errorType(error: unknown): string {
  if ((typeof error === "object" && error !== null) || typeof error === "function") {
    try {
      const name = (error as { name?: unknown }).name;
      if (typeof name === "string" && name.length > 0) {
        return name;
      }
    } catch {
      // Fall through to the stable non-Error classification.
    }
  }
  return error instanceof Error ? error.name : typeof error;
}

/** Content rides span attributes only when the capture gate selects a span mode. */
function captureContentOnSpan(): boolean {
  const mode = contentCaptureMode();
  return mode === ContentCapture.SpanOnly || mode === ContentCapture.SpanAndEvent;
}

/** Content rides Logs EventRecords when the capture gate selects an event mode. */
function captureContentOnEvent(): boolean {
  const mode = contentCaptureMode();
  return mode === ContentCapture.EventOnly || mode === ContentCapture.SpanAndEvent;
}

/**
 * Emit the Opt-In tool execution EventRecord with structured arguments and,
 * on success, a structured result.
 */
function addToolContentEvent(
  toolId: string,
  args: unknown,
  eventContext: OtelContext,
  eventId: string,
  result?: { value: unknown },
): void {
  const attributes = {
    [GEN_AI_OPERATION_NAME]: EXECUTE_TOOL,
    [GEN_AI_TOOL_NAME]: toolId,
    [RATEL_EVENT_ID]: eventId,
    [GEN_AI_TOOL_CALL_ARGUMENTS]: toLogValue(args),
    ...(result ? { [GEN_AI_TOOL_CALL_RESULT]: toLogValue(result.value) } : {}),
  };
  getLogger().emit({
    eventName: RATEL_TOOL_EXECUTION_DETAILS,
    attributes,
    context: eventContext,
  });
}

/**
 * Emit the Opt-In `ratel.search.results` EventRecord carrying the search text.
 * Hit ids/scores/BM25 timing are local-stream only.
 */
function addSearchResultsEvent(query: string, eventContext: OtelContext, eventId: string): void {
  getLogger().emit({
    eventName: RATEL_SEARCH_RESULTS,
    attributes: { [RATEL_SEARCH_QUERY]: query, [RATEL_EVENT_ID]: eventId },
    context: eventContext,
  });
}

/** UTF-8 byte size of the JSON-encoded args (0 if not encodable). */
export function argsSizeBytes(args: unknown): number {
  try {
    return new TextEncoder().encode(JSON.stringify(args) ?? "").length;
  } catch {
    return 0;
  }
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return "";
  }
}

function toLogValue(value: unknown): AnyValue {
  const encoded = safeJson(value);
  if (encoded === "") return null;
  return JSON.parse(encoded) as AnyValue;
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * The upstream MCP server backing a tool, derived from the `<server>__<tool>`
 * id convention. `undefined` for a plain (non-proxied) tool id.
 */
export function upstreamFromToolId(toolId: string): string | undefined {
  const idx = toolId.indexOf("__");
  if (idx <= 0) return undefined;
  return toolId.slice(0, idx);
}

/** Close a span in the failure path: record the exception + ERROR status. */
function fail(span: Span, err: unknown): void {
  if (err instanceof Error) span.recordException(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message: errorMessage(err) });
}

/**
 * Wrap a tool invocation in a standard `execute_tool` span (`gen_ai.operation.name
 * = execute_tool`, enriched with `ratel.*`). Deliberately the OTel gen_ai tool
 * operation, not a bespoke span, so a generic backend understands it
 * (ADR-0007). Preserves the executor's immediate return shape and keeps the
 * span open through `AsyncIterable` completion, cancellation, or failure.
 */
export function traceExecuteTool<T>(
  toolId: string,
  args: unknown,
  run: (projection: RuntimeEventProjection) => T,
): T {
  return getTracer().startActiveSpan(
    `${EXECUTE_TOOL} ${toolId}`,
    { kind: SpanKind.INTERNAL },
    (span) => {
      const activeContext = trace.setSpan(context.active(), span);
      const projection = eventProjection(span, newRuntimeEventId());
      span.setAttribute(GEN_AI_OPERATION_NAME, EXECUTE_TOOL);
      span.setAttribute(GEN_AI_TOOL_NAME, toolId);
      const upstream = upstreamFromToolId(toolId);
      if (upstream) span.setAttribute(RATEL_UPSTREAM_SERVER, upstream);
      span.setAttribute(RATEL_TOOL_ARGS_SIZE_BYTES, argsSizeBytes(args));
      if (captureContentOnSpan()) span.setAttribute(GEN_AI_TOOL_CALL_ARGUMENTS, safeJson(args));

      const succeed = (result: unknown): void => {
        if (captureContentOnSpan()) span.setAttribute(GEN_AI_TOOL_CALL_RESULT, safeJson(result));
        if (captureContentOnEvent()) {
          addToolContentEvent(toolId, args, activeContext, projection.eventId, { value: result });
        }
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
      };
      const reject = (err: unknown): void => {
        if (captureContentOnEvent()) {
          addToolContentEvent(toolId, args, activeContext, projection.eventId);
        }
        fail(span, err);
        span.end();
      };

      try {
        return observeExecutionResult(run(projection), succeed, reject, activeContext) as T;
      } catch (err) {
        reject(err);
        throw err;
      }
    },
  );
}

function observeExecutionResult(
  result: unknown,
  onSuccess: (result: unknown) => void,
  onError: (error: unknown) => void,
  activeContext: OtelContext,
): unknown {
  if (isAsyncIterable(result)) {
    return observeAsyncIterable(result, onSuccess, onError, activeContext);
  }
  if (isPromiseLike(result)) {
    return Promise.resolve(result).then(
      (value) => {
        onSuccess(value);
        return value;
      },
      (error) => {
        onError(error);
        throw error;
      },
    );
  }
  onSuccess(result);
  return result;
}

async function* observeAsyncIterable(
  iterable: AsyncIterable<unknown>,
  onSuccess: (result: unknown) => void,
  onError: (error: unknown) => void,
  activeContext: OtelContext,
): AsyncGenerator<unknown> {
  const iterator = iterable[Symbol.asyncIterator]();
  let completed = false;
  let failed = false;
  let lastValue: unknown;
  try {
    while (true) {
      const next = await context.with(activeContext, () => iterator.next());
      if (next.done) {
        completed = true;
        break;
      }
      lastValue = next.value;
      yield next.value;
    }
  } catch (error) {
    failed = true;
    onError(error);
    throw error;
  } finally {
    if (!completed && !failed && iterator.return) {
      await closeAsyncIterator(iterator, activeContext, (error) => {
        failed = true;
        onError(error);
      });
    }
    if (!failed) onSuccess(lastValue);
  }
}

async function closeAsyncIterator(
  iterator: AsyncIterator<unknown>,
  activeContext: OtelContext,
  onError: (error: unknown) => void,
): Promise<void> {
  try {
    await context.with(activeContext, () => iterator.return?.());
  } catch (error) {
    onError(error);
    throw error;
  }
}

/**
 * Wrap a capability search (tool or skill) in a `ratel.search` span. Synchronous:
 * the native BM25 search returns inline. `run` returns the hit array; its length
 * becomes `ratel.search.hit_count`.
 */
export function traceSearch<T extends { length: number }>(
  target: SearchTarget,
  query: string,
  topK: number,
  origin: SearchOrigin,
  run: (projection: RuntimeEventProjection) => T,
): T {
  return getTracer().startActiveSpan(RATEL_SEARCH, { kind: SpanKind.INTERNAL }, (span) => {
    const eventContext = trace.setSpan(context.active(), span);
    const projection = eventProjection(span);
    span.setAttribute(RATEL_SEARCH_TARGET, target);
    span.setAttribute(RATEL_SEARCH_TOP_K, topK);
    span.setAttribute(RATEL_ORIGIN, origin);
    if (captureContentOnSpan()) span.setAttribute(RATEL_SEARCH_QUERY, query);
    try {
      const hits = run(projection);
      span.setAttribute(RATEL_SEARCH_HIT_COUNT, hits.length);
      if (captureContentOnEvent()) addSearchResultsEvent(query, eventContext, projection.eventId);
      span.setStatus({ code: SpanStatusCode.OK });
      return hits;
    } catch (err) {
      fail(span, err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/** Wrap an asynchronous capability search in a `ratel.search` span. */
export function traceSearchAsync<T extends { length: number }>(
  target: SearchTarget,
  query: string,
  topK: number,
  origin: SearchOrigin,
  run: (projection: RuntimeEventProjection) => Promise<T>,
): Promise<T> {
  return getTracer().startActiveSpan(RATEL_SEARCH, { kind: SpanKind.INTERNAL }, async (span) => {
    const eventContext = trace.setSpan(context.active(), span);
    const projection = eventProjection(span);
    span.setAttribute(RATEL_SEARCH_TARGET, target);
    span.setAttribute(RATEL_SEARCH_TOP_K, topK);
    span.setAttribute(RATEL_ORIGIN, origin);
    if (captureContentOnSpan()) span.setAttribute(RATEL_SEARCH_QUERY, query);
    try {
      const hits = await run(projection);
      span.setAttribute(RATEL_SEARCH_HIT_COUNT, hits.length);
      if (captureContentOnEvent()) addSearchResultsEvent(query, eventContext, projection.eventId);
      span.setStatus({ code: SpanStatusCode.OK });
      return hits;
    } catch (err) {
      fail(span, err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/** Wrap a skill-content load in a `ratel.skill.load` span. */
export function traceSkillLoad<T>(
  skillId: string,
  run: (projection: RuntimeEventProjection) => T,
): T {
  return getTracer().startActiveSpan(RATEL_SKILL_LOAD, { kind: SpanKind.INTERNAL }, (span) => {
    const projection = eventProjection(span);
    span.setAttribute(RATEL_SKILL_ID, skillId);
    try {
      const body = run(projection);
      span.setStatus({ code: SpanStatusCode.OK });
      return body;
    } catch (err) {
      fail(span, err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/**
 * Wrap an upstream-MCP registration in a `ratel.upstream.register` span. `run`
 * receives a `reportToolCount` callback to set `ratel.upstream.tool_count` once
 * the tool list is known.
 */
export function traceUpstreamRegister<T>(
  server: string,
  transport: string,
  run: (reportToolCount: (n: number) => void, projection: RuntimeEventProjection) => Promise<T>,
): Promise<T> {
  return getTracer().startActiveSpan(
    RATEL_UPSTREAM_REGISTER,
    { kind: SpanKind.INTERNAL },
    async (span) => {
      const projection = eventProjection(span);
      span.setAttribute(RATEL_UPSTREAM_SERVER, server);
      span.setAttribute(RATEL_UPSTREAM_TRANSPORT, transport);
      try {
        const result = await run(
          (n) => span.setAttribute(RATEL_UPSTREAM_TOOL_COUNT, n),
          projection,
        );
        span.setStatus({ code: SpanStatusCode.OK });
        return result;
      } catch (err) {
        fail(span, err);
        throw err;
      } finally {
        span.end();
      }
    },
  );
}

/**
 * Mark an upstream tool call that failed with a 401 / needs-reauthorization: a
 * short `ratel.auth.flow` span carrying `ratel.auth.outcome = needs_auth`.
 */
export function recordAuthNeeded(server?: string): RuntimeEventProjection {
  const span = getTracer().startSpan(RATEL_AUTH_FLOW, { kind: SpanKind.INTERNAL });
  const projection = eventProjection(span);
  if (server) span.setAttribute(RATEL_UPSTREAM_SERVER, server);
  span.setAttribute(RATEL_AUTH_OUTCOME, AuthOutcome.NeedsAuth);
  span.end();
  return projection;
}

// Re-exported so hosts configuring capture don't need a second import from
// @ratel-ai/telemetry.
export { ContentCapture, clearContentCapture, setContentCapture };
