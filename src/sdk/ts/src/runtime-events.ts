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
const CATALOG_CRITICAL_FIELDS = ["kind", "id", "name", "content_hash"] as const;
const CATALOG_SCHEMA_FIELDS = ["input_schema", "output_schema"] as const;
const CATALOG_DEFINITION_FIELDS = new Set([
  ...CATALOG_CRITICAL_FIELDS,
  ...CATALOG_SCHEMA_FIELDS,
  "description",
  "tags",
  "searchable_description",
  "searchable_description_overridden",
]);

/** Frozen remotely publishable v1 event names from ADR-0020. */
export const RUNTIME_EVENT_TYPES = [
  "catalog_definition",
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

/** Stable envelope shared by every public runtime event (ADR-0020). */
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
  /**
   * Stable deployment source. Defaults to the env-var-configured OTel `service.name`
   * (`OTEL_SERVICE_NAME`, then `service.name` in `OTEL_RESOURCE_ATTRIBUTES`), then
   * `"ratel"`. A programmatically configured OTel resource is not read; pass this
   * option explicitly in that case.
   */
  sourceId?: string;
  /** Per-registry subscriber queue capacity. Defaults to the native bridge default. */
  queueCapacity?: number;
  /** Maximum envelopes per callback batch. Defaults to the native bridge default. */
  batchSize?: number;
  /** ⚠️ Experimental. Publish complete catalog-definition events. Defaults to false. */
  experimentalCatalogDefinitions?: boolean;
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

/** One externally-authored retrieval-description override for a local catalog entry. */
export interface DefinitionOverride {
  /** Catalog containing the entry. */
  readonly kind: "tool" | "skill" | "fact";
  /** Stable local catalog entry id. */
  readonly entryId: string;
  /** Externally-authored replacement for the description component used by retrieval. */
  readonly searchableDescription: string;
}

/** Successful payload from the runtime-catalog override endpoint. */
export interface DefinitionOverlay {
  /** Complete override set, sorted by the source's wire contract. */
  readonly overrides: readonly DefinitionOverride[];
}

/** Fresh complete overlay and its strong entity tag. */
export interface DefinitionOverlayUpdated {
  /** Successful conditional request. */
  readonly status: 200;
  /** Strong entity tag to send on the next refresh. */
  readonly etag: string;
  /** Complete overlay payload. */
  readonly body: DefinitionOverlay;
}

/** Response indicating that the previously applied overlay is still current. */
export interface DefinitionOverlayNotModified {
  /** Not-modified conditional response. */
  readonly status: 304;
}

/** Conditional overlay response surfaced by an injected overlay source. */
export type DefinitionOverlayResponse = DefinitionOverlayUpdated | DefinitionOverlayNotModified;

/**
 * Pluggable boundary implemented by whichever client owns the overlay; the Ratel
 * Cloud SDK's runtime client is the first implementation, not the only one.
 */
export interface DefinitionOverlaySource {
  /** Pull the overlay, conditionally when a previous strong entity tag is available. */
  fetch(ifNoneMatch?: string): Promise<DefinitionOverlayResponse>;
}

/**
 * Opt-in options for letting an overlay source own Retrieval descriptions.
 * Opting in is one-way for the life of the process: the flag is set-only, and a
 * later attach without it neither clears the opt-in nor restores local text.
 */
export type DefinitionOverridesAttachOptions =
  | {
      /** Give the overlay source ownership of Retrieval descriptions for covered entries. */
      readonly useDefinitionOverrides: true;
      /** Source supplied by the attaching client. */
      readonly source: DefinitionOverlaySource;
    }
  | {
      /** Keep local Retrieval descriptions authoritative. */
      readonly useDefinitionOverrides?: false;
      /** Ignored while override ownership is disabled. */
      readonly source?: DefinitionOverlaySource;
    };

/** Active definition-overrides attachment. */
export interface DefinitionOverridesAttachment {
  /** Pull and apply a changed complete overlay; returns false for 304 or disabled attach. */
  refresh(): Promise<boolean>;
}

/** Public catalog-state seam. */
export interface RuntimeCatalog {
  /** Return a current, serializable full replacement snapshot. */
  snapshot(): CatalogSnapshot;
}

/** Runtime catalog implemented by current SDK runtimes for definition-overlay attach. */
export interface DefinitionOverridesRuntimeCatalog extends RuntimeCatalog {
  /** Attach an injected definition-overlay source and complete the initial pull. */
  attachDefinitionOverrides(
    options: DefinitionOverridesAttachOptions,
  ): Promise<DefinitionOverridesAttachment>;
}

/**
 * @internal Implementation-side catalog. `applyDefinitionOverrides` bypasses the
 * conditional-request protocol that keeps an attachment's entity tag honest, so it
 * stays off the public interface; the attach path and tests reach it through here.
 */
export interface InternalDefinitionOverridesRuntimeCatalog
  extends DefinitionOverridesRuntimeCatalog {
  /** Apply the latest complete override set to live local registrations. */
  applyDefinitionOverrides(overrides: readonly DefinitionOverride[]): Promise<void>;
}

/** Asynchronous, fail-open consumer of one public event batch. */
export type RuntimeEventHandler = (batch: readonly RuntimeEvent[]) => void | PromiseLike<void>;

interface SdkSubscriber {
  readonly handler: RuntimeEventHandler;
  readonly pending: Set<Promise<void>>;
}

interface EventSource {
  experimentalEnableCatalogDefinitions(): void;
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
      experimentalCatalogDefinitions: options.experimentalCatalogDefinitions ?? false,
    };
    this.#sources = sources;
    if (this.#options.experimentalCatalogDefinitions) {
      for (const source of sources) source.experimentalEnableCatalogDefinitions();
    }
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
    const subscriptions: NativeEventSubscription[] = [];
    try {
      for (const source of this.#sources) {
        subscriptions.push(source.subscribeEvents(deliver, this.#options));
      }
    } catch (error) {
      for (const subscription of subscriptions.reverse()) subscription.unsubscribe();
      throw error;
    }
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
  if (serializedSizeUpperBound(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) {
    return normalized as unknown as RuntimeEvent;
  }

  // The public subscribe API still delivers objects. Passing pre-serialized envelopes through so
  // cloud-sdk can reuse them in its batcher is a deferred cross-repo design change.
  let normalizedSize = serializedSize(normalized);
  if (normalizedSize <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) {
    return normalized as unknown as RuntimeEvent;
  }

  normalizedSize = trimCatalogSchemas(normalized, normalizedSize);
  if (normalizedSize <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) {
    return normalized as unknown as RuntimeEvent;
  }

  for (const key of Object.keys(normalized)) {
    if (!REQUIRED_ENVELOPE_FIELDS.has(key) && !isProductFactField(key)) {
      normalizedSize = sizeAfterDeletingProperty(normalized, key, normalizedSize);
      delete normalized[key];
      normalizedSize = sizeAfterSettingProperty(
        normalized,
        "payload_truncated",
        true,
        normalizedSize,
      );
      normalized.payload_truncated = true;
      if (normalizedSize <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) break;
    }
  }
  if (normalizedSize <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) {
    return normalized as unknown as RuntimeEvent;
  }

  const bounded = Object.fromEntries(
    Object.entries(normalized).filter(([key]) => REQUIRED_ENVELOPE_FIELDS.has(key)),
  );
  bounded.payload_truncated = true;
  let boundedSize = serializedSize(bounded);
  for (const [key, value] of prioritizedProductFactEntries(normalized)) {
    if (REQUIRED_ENVELOPE_FIELDS.has(key) || !isProductFactField(key)) continue;
    const boundedValue = sanitizeBoundedValue(value);
    const candidateSize = sizeAfterSettingProperty(bounded, key, boundedValue, boundedSize);
    if (candidateSize > RUNTIME_EVENT_MAX_PAYLOAD_BYTES) continue;
    bounded[key] = boundedValue;
    boundedSize = candidateSize;
  }
  return bounded as unknown as RuntimeEvent;
}

function trimCatalogSchemas(value: Record<string, unknown>, currentSize: number): number {
  if (value.type !== "catalog_definition") return currentSize;
  const fields = CATALOG_SCHEMA_FIELDS.filter((key) => Object.hasOwn(value, key)).sort(
    (left, right) =>
      serializedPropertySize(right, value[right]) - serializedPropertySize(left, value[left]),
  );
  for (const key of fields) {
    currentSize = sizeAfterDeletingProperty(value, key, currentSize);
    delete value[key];
    currentSize = sizeAfterSettingProperty(value, "payload_truncated", true, currentSize);
    value.payload_truncated = true;
    if (currentSize <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES) break;
  }
  return currentSize;
}

function prioritizedProductFactEntries(value: Record<string, unknown>): [string, unknown][] {
  const entries = Object.entries(value).filter(
    ([key]) => !REQUIRED_ENVELOPE_FIELDS.has(key) && isProductFactField(key),
  );
  if (value.type !== "catalog_definition") return entries;
  return [
    ...CATALOG_CRITICAL_FIELDS.flatMap((key) =>
      Object.hasOwn(value, key) ? ([[key, value[key]]] as [string, unknown][]) : [],
    ),
    ...entries.filter(([key]) => !(CATALOG_CRITICAL_FIELDS as readonly string[]).includes(key)),
  ];
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

function serializedSizeUpperBound(value: unknown, limit = RUNTIME_EVENT_MAX_PAYLOAD_BYTES): number {
  if (value === null) return 4;
  if (typeof value === "string") return Math.min(6 * value.length + 2, limit + 1);
  if (typeof value === "number") return 32;
  if (typeof value === "boolean") return 5;
  if (typeof value !== "object") return limit + 1;

  if (Array.isArray(value)) {
    let size = 2;
    for (const [index, child] of value.entries()) {
      size += serializedSizeUpperBound(child, limit - size) + (index === 0 ? 0 : 1);
      if (size > limit) return limit + 1;
    }
    return size;
  }

  let size = 2;
  for (const [index, [key, child]] of Object.entries(value).entries()) {
    size +=
      6 * key.length + 3 + serializedSizeUpperBound(child, limit - size) + (index === 0 ? 0 : 1);
    if (size > limit) return limit + 1;
  }
  return size;
}

function sizeAfterDeletingProperty(
  value: Record<string, unknown>,
  key: string,
  currentSize: number,
): number {
  const propertySize = serializedPropertySize(key, value[key]);
  if (propertySize === 0) return currentSize;
  const commaSize = currentSize > propertySize + 2 ? 1 : 0;
  return currentSize - propertySize - commaSize;
}

function sizeAfterSettingProperty(
  value: Record<string, unknown>,
  key: string,
  nextValue: unknown,
  currentSize: number,
): number {
  const nextPropertySize = serializedPropertySize(key, nextValue);
  const currentPropertySize = Object.hasOwn(value, key)
    ? serializedPropertySize(key, value[key])
    : 0;
  if (currentPropertySize > 0 && nextPropertySize > 0) {
    return currentSize - currentPropertySize + nextPropertySize;
  }
  if (currentPropertySize > 0) {
    return currentSize - currentPropertySize - (currentSize > currentPropertySize + 2 ? 1 : 0);
  }
  return currentSize + nextPropertySize + (currentSize > 2 && nextPropertySize > 0 ? 1 : 0);
}

function serializedPropertySize(key: string, value: unknown): number {
  return serializedSize({ [key]: value }) - 2;
}

function isProductFactField(key: string): boolean {
  return (
    CATALOG_DEFINITION_FIELDS.has(key) ||
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
