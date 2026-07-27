import { type EnrichSpan, OpenTelemetry, type OpenTelemetryOptions } from "@ai-sdk/otel";
import { Origin, RATEL_ORIGIN } from "@ratel-ai/telemetry";

/**
 * Re-exported so `origin` is spellable without depending on `@ratel-ai/telemetry`
 * directly: it is *this* package's runtime dependency, and an isolated
 * `node_modules` (pnpm's default, Yarn PnP) keeps a host off its resolution path.
 * The bare `"agent"` / `"direct"` literals are assignable too.
 */
export { Origin };

/**
 * {@link RatelOtelIntegration} options: every `@ai-sdk/otel` knob, plus the
 * `ratel.origin` the overlay stamps.
 *
 * `enrichSpan` is composed with, not taken over. Owning it outright would cost
 * a host every attribute its own hook adds, silently and unrecoverably — the
 * embedded emitter is private, and registering a second integration to get the
 * hook back duplicates every `gen_ai.*` span. A host hook that throws loses its
 * own attributes and nothing else (see {@link hostAttributes}).
 *
 * `origin` defaults to `agent`, which is right for a `generateText` loop and
 * wrong for `embed` / `embedMany` / `rerank` driven straight from host code. It
 * is fixed per instance, not per span, so a host that makes both kinds of call
 * varies it with a second integration passed per call through
 * `telemetry: { integrations: [...] }` — not by re-registering globally.
 */
export type RatelOtelIntegrationOptions = OpenTelemetryOptions & { origin?: Origin };

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
 * An `ai@7` telemetry integration that emits the AI SDK's standard `gen_ai.*`
 * spans and adds Ratel's `ratel.*` overlay.
 *
 * It **creates** spans onto whatever OTel provider the host has registered; it
 * never registers a provider and never exports. Delivery is the host's: every
 * processor on that provider — Langfuse, a generic OTLP exporter, anything else —
 * receives these spans. Wire it once, on a provider you own:
 *
 * ```ts
 * const sdk = new NodeSDK({
 *   spanProcessors: [
 *     new LangfuseSpanProcessor({
 *       // Langfuse's default filter drops the SDK's ratel.* spans after they
 *       // arrive here, so widen it — by scope, since `execute_tool <tool>` is
 *       // emitted under both scopes and its name says nothing about the source.
 *       shouldExportSpan: ({ otelSpan }) =>
 *         isDefaultExportSpan(otelSpan) ||
 *         otelSpan.instrumentationScope.name === "@ratel-ai/sdk",
 *     }),
 *   ],
 * });
 * sdk.start();
 * registerTelemetry(new RatelOtelIntegration());
 * ```
 *
 * **Enrichment is per-emitter.** `enrichSpan` reaches only the spans this
 * integration's embedded emitter creates (scope `gen_ai`). The SDK's own
 * `ratel.*` and `execute_tool <tool>` spans come from a different tracer and are
 * never enriched here, `gen_ai.*` attributes on them notwithstanding.
 *
 * **Register exactly one emitting integration.** This one, Langfuse's, and the
 * bare `OpenTelemetry` from `@ai-sdk/otel` all embed the same emitter, so
 * registering two duplicates every `gen_ai.*` span. Ratel-specific enrichment of
 * AI SDK spans needs this one; every processor on the shared provider sees the
 * spans either way.
 *
 * **This targets the `ai@7` seam only.** `ai@5` has none; `ai@6` (from `6.0.108`)
 * has an earlier, different one — `registerTelemetryIntegration` with a
 * six-method `TelemetryIntegration` interface — that this class does not
 * implement. On either, pass `experimental_telemetry: { isEnabled: true }` per
 * call; the SDK's own `ratel.*` spans are unaffected and need no wiring at all.
 */
export class RatelOtelIntegration implements EmittingTelemetry {
  readonly #emitter: OpenTelemetry;

  constructor({ origin = Origin.Agent, enrichSpan, ...options }: RatelOtelIntegrationOptions = {}) {
    this.#emitter = new OpenTelemetry({
      ...options,
      // Ratel's keys land last: `ratel.origin` is Ratel vocabulary, so a host
      // hook that writes it too must not decide what the overlay reports.
      enrichSpan: (event) => ({ ...hostAttributes(enrichSpan, event), [RATEL_ORIGIN]: origin }),
    });
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

/**
 * A host `enrichSpan`'s attributes, or none when it throws. The emitter guards
 * the hook already, but its `catch` drops the *whole* return value — so without
 * this one a host bug would take `ratel.origin` down with it, on every span,
 * silently. Composing means owning that blast radius: the host loses only what
 * the host contributed.
 */
function hostAttributes(enrichSpan: EnrichSpan | undefined, event: Parameters<EnrichSpan>[0]) {
  try {
    return enrichSpan?.(event);
  } catch {
    return undefined;
  }
}
