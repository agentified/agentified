import { randomBytes, randomUUID } from "node:crypto";
import type { NativeEventSubscription } from "../native/index.cjs";
import type { ToolDefinition } from "./catalog.js";
import type { SkillDefinition } from "./skill-catalog.js";

const DEFAULT_SOURCE_ID = "ratel";

/** Stable envelope shared by every public runtime event (ADR-0019). */
export interface RuntimeEvent {
  readonly v: 2;
  readonly event_id: string;
  readonly ts: number;
  readonly session_id: string;
  readonly source_id: string;
  readonly type: string;
  readonly invocation_id?: string;
  readonly catalog_version?: string;
  readonly environment?: string;
  readonly end_user_id?: string;
  readonly trace_id?: string;
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
  /** Stop accepting future events for this subscriber. */
  unsubscribe(): void;
  /** Wait until work already accepted by both registry streams reaches the handler. */
  flush(): Promise<void>;
  /** Envelopes dropped from this subscriber's bounded native queues. */
  readonly droppedCount: number;
}

/** Complete executor-free catalog state published separately from runtime events. */
export interface CatalogSnapshot {
  readonly sourceId: string;
  readonly tools: readonly ToolDefinition[];
  readonly skills: readonly SkillDefinition[];
}

/** Public catalog-state seam. */
export interface RuntimeCatalog {
  /** Return a current, serializable full replacement snapshot. */
  snapshot(): CatalogSnapshot;
}

type RuntimeEventHandler = (
  batch: readonly RuntimeEvent[],
) => void | PromiseLike<void>;

interface EventSource {
  subscribeEvents(
    handler: (batch: RuntimeEvent[]) => void,
    options: Required<RuntimeEventsOptions>,
  ): NativeEventSubscription;
}

/** Public merged push stream over tool, skill, and SDK-owned runtime facts. */
export class RuntimeEvents {
  readonly sessionId: string;
  readonly sourceId: string;
  readonly #options: Required<RuntimeEventsOptions>;
  readonly #sources: readonly EventSource[];
  readonly #sdkSubscribers = new Set<RuntimeEventHandler>();

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
    const deliver = (batch: RuntimeEvent[]): void => {
      try {
        void Promise.resolve(handler(batch)).catch(() => {});
      } catch {
        // Runtime-event consumers are observational and fail open.
      }
    };
    const subscriptions = this.#sources.map((source) =>
      source.subscribeEvents(deliver, this.#options),
    );
    this.#sdkSubscribers.add(handler);
    let active = true;
    return {
      unsubscribe: () => {
        if (!active) return;
        active = false;
        this.#sdkSubscribers.delete(handler);
        for (const subscription of subscriptions) subscription.unsubscribe();
      },
      flush: async () => {
        await Promise.all(subscriptions.map((subscription) => subscription.flush()));
      },
      get droppedCount() {
        return subscriptions.reduce((total, subscription) => total + subscription.droppedCount, 0);
      },
    };
  }

  /** @internal Merge an SDK-owned fact (experiments) into every public subscriber. */
  emit(event: Record<string, unknown>): RuntimeEvent {
    const envelope: RuntimeEvent = {
      v: 2,
      event_id: newUlid(),
      ts: Date.now(),
      session_id: this.sessionId,
      source_id: this.sourceId,
      type: String(event.type ?? "unknown"),
      ...event,
    };
    for (const handler of this.#sdkSubscribers) {
      queueMicrotask(() => {
        try {
          void Promise.resolve(handler([envelope])).catch(() => {});
        } catch {
          // Runtime-event consumers are observational and fail open.
        }
      });
    }
    return envelope;
  }
}

function defaultSourceId(): string {
  if (process.env.OTEL_SERVICE_NAME) return process.env.OTEL_SERVICE_NAME;
  const serviceName = process.env.OTEL_RESOURCE_ATTRIBUTES?.split(",")
    .map((entry) => entry.trim().split("=", 2))
    .find(([key]) => key === "service.name")?.[1];
  return serviceName || DEFAULT_SOURCE_ID;
}

/** Minimal monotonicity-free ULID generator: event uniqueness, time sorting, wire alphabet. */
function newUlid(now = Date.now()): string {
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
