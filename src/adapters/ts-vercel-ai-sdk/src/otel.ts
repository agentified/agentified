import { type EnrichSpan, OpenTelemetry, type OpenTelemetryOptions } from "@ai-sdk/otel";
import { Origin, RATEL_ORIGIN } from "@ratel-ai/telemetry";

/**
 * {@link RatelOtelIntegration} options: every `@ai-sdk/otel` knob except
 * `enrichSpan`, which the integration owns — stamping the `ratel.*` overlay is
 * the whole reason to pick it over the bare emitter.
 */
export type RatelOtelIntegrationOptions = Omit<OpenTelemetryOptions, "enrichSpan">;

/**
 * `ai@7`'s `Telemetry` contract, taken from the emitter we delegate to rather
 * than imported from `ai` directly. The adapter builds against `ai@5`–`ai@7`
 * and only `ai@7` exports `Telemetry`, so importing it here would break the
 * older majors for everyone — including hosts that never touch telemetry.
 * `@ai-sdk/otel` pins its own `ai@7`, and TypeScript is structural, so a class
 * satisfying this is still assignable wherever `registerTelemetry` wants a
 * `Telemetry`.
 */
type EmittingTelemetry = Pick<
  OpenTelemetry,
  | "executeTool"
  | "executeLanguageModelCall"
  | "onStart"
  | "onStepStart"
  | "onLanguageModelCallStart"
  | "onLanguageModelCallEnd"
  | "onToolExecutionStart"
  | "onToolExecutionEnd"
  | "onStepEnd"
  | "onStepFinish"
  | "onObjectStepStart"
  | "onObjectStepEnd"
  | "onEnd"
  | "onEmbedStart"
  | "onEmbedEnd"
  | "onRerankStart"
  | "onRerankEnd"
  | "onAbort"
  | "onError"
>;

/**
 * The `ratel.*` overlay stamped on every AI SDK span. `enrichSpan` runs at span
 * creation and returns attributes rather than mutating the span; it is handed
 * the `spanType`, `operationId`, `callId`, and `runtimeContext`, so per-span-kind
 * and per-call attributes are reachable here when there is a vocabulary for them.
 * Origin is the deliberate baseline.
 */
const ratelOverlay: EnrichSpan = () => ({ [RATEL_ORIGIN]: Origin.Agent });

/**
 * An `ai@7` telemetry integration that emits the AI SDK's standard `gen_ai.*`
 * spans and adds Ratel's `ratel.*` overlay.
 *
 * It **creates** spans onto whatever OTel provider the host has registered; it
 * never registers a provider and never exports. Delivery is the host's: any
 * processor on that provider — Langfuse, a generic OTLP exporter, Ratel Cloud —
 * receives these spans. Wire it once, on a provider you own:
 *
 * ```ts
 * const sdk = new NodeSDK({ spanProcessors: [new LangfuseSpanProcessor()] });
 * sdk.start();
 * registerTelemetry(new RatelOtelIntegration());
 * ```
 *
 * **Register exactly one emitting integration.** This one, Langfuse's, and the
 * bare `OpenTelemetry` from `@ai-sdk/otel` all embed the same emitter, so
 * registering two duplicates every `gen_ai.*` span. Ratel-specific enrichment of
 * AI SDK spans needs this one; every processor on the shared provider sees the
 * spans either way.
 *
 * **This targets the `ai@7` seam only.** `ai@5` has none; `ai@6` (from `6.0.150`)
 * has an earlier, different one — `registerTelemetryIntegration` with a
 * six-method `TelemetryIntegration` interface — that this class does not
 * implement. On either, pass `experimental_telemetry: { isEnabled: true }` per
 * call; the SDK's own `ratel.*` spans are unaffected and need no wiring at all.
 */
export class RatelOtelIntegration implements EmittingTelemetry {
  readonly #emitter: OpenTelemetry;

  constructor(options: RatelOtelIntegrationOptions = {}) {
    this.#emitter = new OpenTelemetry({ ...options, enrichSpan: ratelOverlay });
  }

  // --- context wrappers -----------------------------------------------------
  // Not event callbacks: these run the AI SDK's work inside the active span's
  // context. Dropping one doesn't lose an event, it silently unparents every
  // span underneath it.

  executeTool<T>(options: {
    callId: string;
    toolCallId: string;
    execute: () => PromiseLike<T>;
  }): PromiseLike<T> {
    return this.#emitter.executeTool(options);
  }

  executeLanguageModelCall<T>(options: {
    callId: string;
    execute: () => PromiseLike<T>;
  }): PromiseLike<T> {
    return this.#emitter.executeLanguageModelCall(options);
  }

  // --- lifecycle events -----------------------------------------------------

  onStart(...args: Parameters<OpenTelemetry["onStart"]>): void {
    this.#emitter.onStart(...args);
  }

  onStepStart(...args: Parameters<OpenTelemetry["onStepStart"]>): void {
    this.#emitter.onStepStart(...args);
  }

  onLanguageModelCallStart(...args: Parameters<OpenTelemetry["onLanguageModelCallStart"]>): void {
    this.#emitter.onLanguageModelCallStart(...args);
  }

  onLanguageModelCallEnd(...args: Parameters<OpenTelemetry["onLanguageModelCallEnd"]>): void {
    this.#emitter.onLanguageModelCallEnd(...args);
  }

  onToolExecutionStart(...args: Parameters<OpenTelemetry["onToolExecutionStart"]>): void {
    this.#emitter.onToolExecutionStart(...args);
  }

  onToolExecutionEnd(...args: Parameters<OpenTelemetry["onToolExecutionEnd"]>): void {
    this.#emitter.onToolExecutionEnd(...args);
  }

  onStepEnd(...args: Parameters<OpenTelemetry["onStepEnd"]>): void {
    this.#emitter.onStepEnd(...args);
  }

  onEnd(...args: Parameters<OpenTelemetry["onEnd"]>): void {
    this.#emitter.onEnd(...args);
  }

  onEmbedStart(...args: Parameters<OpenTelemetry["onEmbedStart"]>): void {
    this.#emitter.onEmbedStart(...args);
  }

  onEmbedEnd(...args: Parameters<OpenTelemetry["onEmbedEnd"]>): void {
    this.#emitter.onEmbedEnd(...args);
  }

  onRerankStart(...args: Parameters<OpenTelemetry["onRerankStart"]>): void {
    this.#emitter.onRerankStart(...args);
  }

  onRerankEnd(...args: Parameters<OpenTelemetry["onRerankEnd"]>): void {
    this.#emitter.onRerankEnd(...args);
  }

  onAbort(...args: Parameters<OpenTelemetry["onAbort"]>): void {
    this.#emitter.onAbort(...args);
  }

  onError(...args: Parameters<OpenTelemetry["onError"]>): void {
    this.#emitter.onError(...args);
  }

  // --- deprecated upstream, forwarded anyway --------------------------------
  // `ai@7` still calls these on integrations that expose them; a host on an
  // older `ai@7.x` minor would silently lose spans if we dropped them.

  /** @deprecated The AI SDK calls {@link RatelOtelIntegration.onStepEnd} instead. */
  onStepFinish(...args: Parameters<OpenTelemetry["onStepFinish"]>): void {
    this.#emitter.onStepFinish(...args);
  }

  /** @deprecated Superseded upstream by the generic operation-start events. */
  onObjectStepStart(...args: Parameters<OpenTelemetry["onObjectStepStart"]>): void {
    this.#emitter.onObjectStepStart(...args);
  }

  /** @deprecated Superseded upstream by the generic operation-end events. */
  onObjectStepEnd(...args: Parameters<OpenTelemetry["onObjectStepEnd"]>): void {
    this.#emitter.onObjectStepEnd(...args);
  }
}
