import { randomBytes, randomUUID } from "node:crypto";
import type { NativeEventSubscription } from "../native/index.cjs";
import type { ToolDefinition } from "./catalog.js";
import type { SkillDefinition } from "./skill-catalog.js";

const DEFAULT_SOURCE_ID = "ratel";
const REQUIRED_ENVELOPE_FIELDS = new Set([
  "v",
  "event_id",
  "ts",
  "session_id",
  "source_id",
  "type",
]);

/** Frozen remotely publishable v1 event names from ADR-0019. */
export const RUNTIME_EVENT_TYPES = [
  "search",
  "skill_search",
  "gateway_search",
  "invoke_start",
  "invoke_end",
  "invoke_error",
  "gateway_invoke",
  "gateway_error",
  "skill_invoke",
  "index_churn",
  "skill_churn",
  "upstream_register",
  "upstream_invoke",
  "upstream_error",
  "auth_refresh",
  "auth_needs",
  "auth_flow_start",
  "auth_flow_end",
  "experiment_selection",
  "experiment_results",
  "experiment_comparison",
  "experiment_skip",
  "experiment_fallback",
  "experiment_drop",
  "experiment_invocation",
  "experiment_outcome",
  "events_dropped",
] as const;

/** Maximum serialized size of one public event envelope. */
export const RUNTIME_EVENT_MAX_PAYLOAD_BYTES = 64 * 1_024;
/** Maximum UTF-8 byte size of a public search query. */
export const RUNTIME_EVENT_MAX_QUERY_BYTES = 4 * 1_024;
/** Maximum ranked hits carried by one public search event. */
export const RUNTIME_EVENT_MAX_HITS = 100;

/** Stable envelope shared by every public runtime event (ADR-0019). */
export interface RuntimeEvent {
  /** Envelope schema version. */
  readonly v: 2;
  /** Canonical client ULID used to deduplicate and join OTel. */
  readonly event_id: string;
  /** Unix timestamp in milliseconds. */
  readonly ts: number;
  /** Process/session identity shared by the merged stream. */
  readonly session_id: string;
  /** Stable deployment source identity. */
  readonly source_id: string;
  /** Additive event discriminator. */
  readonly type: string;
  /** Lifecycle group for invocation start/end/error events. */
  readonly invocation_id?: string;
  /** Catalog generation when supplied by the producer. */
  readonly catalog_version?: string;
  /** Optional deployment environment. */
  readonly environment?: string;
  /** Optional pseudonymous application user identity. */
  readonly end_user_id?: string;
  /** Active OTel trace identity when a recording span exists. */
  readonly trace_id?: string;
  /** Active OTel span identity when a recording span exists. */
  readonly span_id?: string;
  readonly [field: string]: unknown;
}

/** Identity and bounded-delivery options for {@link RuntimeEvents}. */
export interface RuntimeEventsOptions {
  /** Stable identity for this process/session. Defaults to a fresh UUID. */
  sessionId?: string;
  /** Stable deployment source. Defaults to OTel `service.name`, then `"ratel"`. */
  sourceId?: string;
  /** Per-registry subscriber queue capacity. Defaults to the native bridge default. */
  queueCapacity?: number;
  /** Maximum envelopes per callback batch. Defaults to the native bridge default. */
  batchSize?: number;
}

/** One runtime-events subscription. */
export interface RuntimeEventSubscription {
  /** Stop accepting new events; already-queued native envelopes still drain. */
  unsubscribe(): void;
  /** Wait until accepted registry events reach the handler and async work settles. */
  flush(): Promise<void>;
  /** Envelopes dropped from this subscriber's bounded native queues. */
  readonly droppedCount: number;
}

/** Complete executor-free catalog state published separately from runtime events. */
export interface CatalogSnapshot {
  /** Stable deployment source shared with runtime envelopes. */
  readonly source_id: string;
  /** Complete serializable tool definitions, sorted by id. */
  readonly tools: readonly ToolDefinition[];
  /** Complete public skill metadata, sorted by id and excluding bodies. */
  readonly skills: readonly SkillDefinition[];
}

/** Public catalog-state seam. */
export interface RuntimeCatalog {
  /** Return a current, serializable full replacement snapshot. */
  snapshot(): CatalogSnapshot;
}

/** Asynchronous, fail-open consumer of one public event batch. */
export type RuntimeEventHandler = (batch: readonly RuntimeEvent[]) => void | PromiseLike<void>;

interface SdkSubscriber {
  readonly handler: RuntimeEventHandler;
  readonly pending: Set<Promise<void>>;
}

interface EventSource {
  subscribeEvents(
    handler: (batch: RuntimeEvent[]) => void,
    options: Required<RuntimeEventsOptions>,
  ): NativeEventSubscription;
}

/** Public merged push stream over tool, skill, and SDK-owned runtime facts. */
export class RuntimeEvents {
  /** Process/session identity stamped onto every delivered envelope. */
  readonly sessionId: string;
  /** Stable deployment identity stamped onto events and catalog snapshots. */
  readonly sourceId: string;
  readonly #options: Required<RuntimeEventsOptions>;
  readonly #sources: readonly EventSource[];
  readonly #sdkSubscribers = new Set<SdkSubscriber>();

  /** @internal Constructed by {@link ratel}; consumers use `runtime.events`. */
  constructor(sources: readonly EventSource[], options: RuntimeEventsOptions = {}) {
    this.sessionId = options.sessionId ?? randomUUID();
    this.sourceId = options.sourceId ?? defaultSourceId();
    this.#options = {
      sessionId: this.sessionId,
      sourceId: this.sourceId,
      queueCapacity: options.queueCapacity ?? 1_024,
      batchSize: options.batchSize ?? 64,
    };
    this.#sources = sources;
  }

  /**
   * Subscribe to merged runtime-event batches. Delivery is asynchronous and
   * bounded; handler work never blocks the emitting registry operation.
   */
  subscribe(handler: RuntimeEventHandler): RuntimeEventSubscription {
    const sdkSubscriber: SdkSubscriber = { handler, pending: new Set() };
    const deliver = (batch: RuntimeEvent[]): void => {
      trackHandlerWork(sdkSubscriber, batch.map(normalizeRuntimeEvent));
    };
    const subscriptions = this.#sources.map((source) =>
      source.subscribeEvents(deliver, this.#options),
    );
    this.#sdkSubscribers.add(sdkSubscriber);
    let active = true;
    return {
      unsubscribe: () => {
        if (!active) return;
        active = false;
        this.#sdkSubscribers.delete(sdkSubscriber);
        for (const subscription of subscriptions) subscription.unsubscribe();
      },
      flush: async () => {
        await Promise.all(subscriptions.map((subscription) => subscription.flush()));
        await Promise.all([...sdkSubscriber.pending]);
      },
      get droppedCount() {
        return subscriptions.reduce((total, subscription) => total + subscription.droppedCount, 0);
      },
    };
  }

  /** @internal Merge an SDK-owned fact (experiments) into every public subscriber. */
  emit(event: Record<string, unknown>): RuntimeEvent {
    const eventId = typeof event.event_id === "string" ? event.event_id : newRuntimeEventId();
    const envelope = normalizeRuntimeEvent({
      ...event,
      v: 2,
      event_id: eventId,
      ts: Date.now(),
      session_id: this.sessionId,
      source_id: this.sourceId,
      type: String(event.type ?? "unknown"),
    });
    for (const subscriber of this.#sdkSubscribers) {
      const pending = new Promise<void>((resolve) => {
        queueMicrotask(() => {
          try {
            void Promise.resolve(subscriber.handler([envelope])).then(resolve, resolve);
          } catch {
            resolve();
          }
        });
      });
      subscriber.pending.add(pending);
      void pending.finally(() => subscriber.pending.delete(pending));
    }
    return envelope;
  }
}

function trackHandlerWork(subscriber: SdkSubscriber, batch: readonly RuntimeEvent[]): void {
  let pending: Promise<void>;
  try {
    pending = Promise.resolve(subscriber.handler(batch)).then(
      () => {},
      () => {},
    );
  } catch {
    // Runtime-event consumers are observational and fail open.
    return;
  }
  subscriber.pending.add(pending);
  void pending.then(() => subscriber.pending.delete(pending));
}

function defaultSourceId(): string {
  if (process.env.OTEL_SERVICE_NAME) return process.env.OTEL_SERVICE_NAME;
  const serviceName = process.env.OTEL_RESOURCE_ATTRIBUTES?.split(",")
    .map((entry) => entry.trim().split("=", 2))
    .find(([key]) => key === "service.name")?.[1];
  return serviceName || DEFAULT_SOURCE_ID;
}

/** Minimal monotonicity-free ULID generator: event uniqueness, time sorting, wire alphabet. */
/** @internal Mint the canonical client event id before stream + OTel projection. */
export function newRuntimeEventId(now = Date.now()): string {
  const alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ";
  let time = now;
  let encodedTime = "";
  for (let index = 0; index < 10; index += 1) {
    encodedTime = alphabet[time % 32] + encodedTime;
    time = Math.floor(time / 32);
  }
  const entropy = randomBytes(10);
  let buffer = 0;
  let bits = 0;
  let encodedEntropy = "";
  for (const byte of entropy) {
    buffer = (buffer << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      encodedEntropy += alphabet[(buffer >> bits) & 31];
    }
  }
  return encodedTime + encodedEntropy;
}

function normalizeRuntimeEvent(input: Record<string, unknown>): RuntimeEvent {
  const normalized = sanitizeValue(input, "") as Record<string, unknown>;
  if (serializedSize(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) {
    return normalized as unknown as RuntimeEvent;
  }

  for (const key of Object.keys(normalized)) {
    if (!REQUIRED_ENVELOPE_FIELDS.has(key) && !isProductFactField(key)) {
      delete normalized[key];
      normalized.payload_truncated = true;
      if (serializedSize(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) break;
    }
  }
  if (serializedSize(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) {
    return normalized as unknown as RuntimeEvent;
  }

  const bounded = Object.fromEntries(
    Object.entries(normalized).filter(([key]) => REQUIRED_ENVELOPE_FIELDS.has(key)),
  );
  bounded.payload_truncated = true;
  for (const [key, value] of Object.entries(normalized)) {
    if (REQUIRED_ENVELOPE_FIELDS.has(key) || !isProductFactField(key)) continue;
    bounded[key] = sanitizeBoundedValue(value);
    if (serializedSize(bounded) > RUNTIME_EVENT_MAX_PAYLOAD_BYTES) delete bounded[key];
  }
  return bounded as unknown as RuntimeEvent;
}

function sanitizeValue(value: unknown, key: string): unknown {
  if (typeof value === "string") {
    return truncateUtf8(value, key === "query" ? RUNTIME_EVENT_MAX_QUERY_BYTES : 4_096);
  }
  if (Array.isArray(value)) {
    return value.slice(0, RUNTIME_EVENT_MAX_HITS).map((item) => sanitizeValue(item, ""));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 100)
        .map(([childKey, childValue]) => [childKey, sanitizeValue(childValue, childKey)]),
    );
  }
  return value;
}

function sanitizeBoundedValue(value: unknown): unknown {
  if (typeof value === "string") return truncateUtf8(value, 512);
  if (Array.isArray(value)) {
    return value.slice(0, RUNTIME_EVENT_MAX_HITS).map((item) => sanitizeBoundedValue(item));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .slice(0, 32)
        .map(([key, child]) => [key, sanitizeBoundedValue(child)]),
    );
  }
  return value;
}

function truncateUtf8(value: string, maxBytes: number): string {
  if (Buffer.byteLength(value, "utf8") <= maxBytes) return value;
  let result = Buffer.from(value, "utf8").subarray(0, maxBytes).toString("utf8");
  while (Buffer.byteLength(result, "utf8") > maxBytes) result = result.slice(0, -1);
  return result;
}

function serializedSize(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value), "utf8");
}

function isProductFactField(key: string): boolean {
  return (
    key.endsWith("_id") ||
    key.endsWith("_ids") ||
    key.endsWith("_ms") ||
    key.endsWith("_count") ||
    key.endsWith("_score") ||
    key.endsWith("_scores") ||
    [
      "query",
      "target",
      "origin",
      "top_k",
      "hits",
      "outcome",
      "error",
      "error_class",
      "transport",
      "role",
      "cold",
      "agreement",
      "reason",
      "label",
      "score",
      "attributed",
      "rank",
      "turn",
      "action",
    ].includes(key)
  );
}
