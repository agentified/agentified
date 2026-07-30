import { context, propagation } from "@opentelemetry/api";
import {
  RATEL_EXPERIMENT_ARM,
  RATEL_EXPERIMENT_ARM_BAGGAGE_KEY,
  RATEL_EXPERIMENT_ID,
  RATEL_EXPERIMENT_ID_BAGGAGE_KEY,
  RATEL_EXPERIMENT_ROLE,
  RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY,
  RATEL_EXPERIMENT_SELECTION_ID,
  RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY,
  RATEL_EXPERIMENT_UNIT,
  RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY,
} from "@ratel-ai/telemetry";

/** The mutable Mastra span surface used by its span-output processor contract. */
export interface RatelMastraSpan {
  attributes?: unknown;
}

/** A structural Mastra span-output processor, compatible across Mastra 1.x. */
export interface RatelExperimentSpanOutputProcessor {
  readonly name: string;
  process<T extends RatelMastraSpan | undefined = undefined>(span?: T): T;
  shutdown(): Promise<void>;
}

const PROCESSOR_NAME = "ratel-experiment-context";
const EXPERIMENT_JOIN_FIELDS = [
  [RATEL_EXPERIMENT_ID_BAGGAGE_KEY, RATEL_EXPERIMENT_ID],
  [RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY, RATEL_EXPERIMENT_SELECTION_ID],
  [RATEL_EXPERIMENT_ARM_BAGGAGE_KEY, RATEL_EXPERIMENT_ARM],
  [RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY, RATEL_EXPERIMENT_ROLE],
  [RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY, RATEL_EXPERIMENT_UNIT],
] as const;

/**
 * Create a Mastra span-output processor that stamps the active retrieval
 * experiment's five controlled join fields onto Mastra's private spans.
 */
export function experimentalRatelSpanOutputProcessor(): RatelExperimentSpanOutputProcessor {
  const joinsBySpan = new WeakMap<RatelMastraSpan, Record<string, string> | null>();
  return {
    name: PROCESSOR_NAME,
    process<T extends RatelMastraSpan | undefined = undefined>(span?: T): T {
      if (span === undefined) {
        return span as T;
      }
      let join = joinsBySpan.get(span);
      if (!joinsBySpan.has(span)) {
        join = activeExperimentJoin() ?? null;
        joinsBySpan.set(span, join);
      }
      if (join !== null && join !== undefined) {
        span.attributes = { ...attributesOf(span), ...join };
      }
      return span;
    },
    shutdown: () => Promise.resolve(),
  };
}

function activeExperimentJoin(): Record<string, string> | undefined {
  const baggage = propagation.getBaggage(context.active());
  if (baggage === undefined) {
    return undefined;
  }
  const attributes: Record<string, string> = {};
  for (const [baggageKey, attributeKey] of EXPERIMENT_JOIN_FIELDS) {
    const entry = baggage.getEntry(baggageKey);
    if (entry !== undefined) {
      attributes[attributeKey] = entry.value;
    }
  }
  return Object.keys(attributes).length === 0 ? undefined : attributes;
}

function attributesOf(span: RatelMastraSpan): Record<string, unknown> {
  return span.attributes !== null &&
    typeof span.attributes === "object" &&
    !Array.isArray(span.attributes)
    ? (span.attributes as Record<string, unknown>)
    : {};
}
