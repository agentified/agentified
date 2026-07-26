/**
 * The content-capture gate for the `ratel.*` telemetry vocabulary. No OpenTelemetry
 * SDK import (ADR-0007): this only parses the capture env var and holds the
 * programmatic override, so it stays weight-free for the three consumers that need
 * the vocabulary without an exporter — the SDK (emit side), the server (read side),
 * and edge/serverless emitters. Endpoint and auth resolution is the host's job: it
 * owns the provider, so it configures its own OTLP exporters.
 */

import { CAPTURE_CONTENT_ENV } from "./index.js";

/**
 * Message/tool content capture modes for
 * `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (CONVENTIONS.md § Capture
 * gating). Default off.
 */
export const ContentCapture = {
  NoContent: "NO_CONTENT",
  SpanOnly: "SPAN_ONLY",
  EventOnly: "EVENT_ONLY",
  SpanAndEvent: "SPAN_AND_EVENT",
} as const;
export type ContentCapture = (typeof ContentCapture)[keyof typeof ContentCapture];

/** Module-level programmatic override; `null` means unset (env-driven). */
let contentCaptureOverride: ContentCapture | null = null;

/**
 * Monotonically increasing token identifying the most recent {@link setContentCapture}
 * call, so a stale holder (e.g. an old telemetry handle's shutdown) cannot clear an
 * override a newer caller owns via {@link clearContentCapture}.
 */
let contentCaptureGeneration = 0;

/**
 * The single normalizer for capture-mode strings, shared by the env parser and the
 * programmatic setter: trim + case-insensitive, with the legacy boolean forms
 * (`true`/`1` -> full capture, `false`/`0` -> none). `undefined` when unrecognized —
 * the two callers diverge there ({@link contentCaptureMode} defaults, {@link
 * setContentCapture} throws).
 */
function parseContentCapture(raw: string): ContentCapture | undefined {
  switch (raw.trim().toUpperCase()) {
    case "NO_CONTENT":
      return ContentCapture.NoContent;
    case "SPAN_ONLY":
      return ContentCapture.SpanOnly;
    case "EVENT_ONLY":
      return ContentCapture.EventOnly;
    case "SPAN_AND_EVENT":
      return ContentCapture.SpanAndEvent;
    case "TRUE":
    case "1":
      return ContentCapture.SpanAndEvent;
    case "FALSE":
    case "0":
      return ContentCapture.NoContent;
    default:
      return undefined;
  }
}

/**
 * Programmatically set the content-capture mode. While set,
 * {@link contentCaptureMode} returns this mode regardless of
 * `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` — programmatic config wins
 * over the environment, matching how OpenTelemetry treats env vars as the
 * fallback for code-level configuration. Pass `null`/`undefined` to clear the
 * override unconditionally and return to env-driven parsing.
 *
 * The mode is validated like the env var (case-insensitive, legacy `true`/`false`/
 * `1`/`0` accepted) and **throws a `TypeError`** on anything unrecognized — failing
 * loud at config time instead of storing a value that would both disable capture
 * and mask the env var.
 *
 * Returns a generation token identifying this call as the current owner of the
 * override; pass it to {@link clearContentCapture} to clear only if no newer
 * set has happened since (the safe form for shutdown/teardown hooks).
 */
export function setContentCapture(mode: ContentCapture | null | undefined): number {
  if (mode == null) {
    contentCaptureOverride = null;
    return ++contentCaptureGeneration;
  }
  const parsed = parseContentCapture(mode);
  if (parsed === undefined) {
    throw new TypeError(
      `setContentCapture: unrecognized mode ${JSON.stringify(mode)}. Valid values: ` +
        `${Object.values(ContentCapture).join(", ")} (case-insensitive; legacy ` +
        "true/false/1/0 also accepted), or null/undefined to clear.",
    );
  }
  contentCaptureOverride = parsed;
  return ++contentCaptureGeneration;
}

/**
 * Clear the programmatic content-capture override, but only when `generation` —
 * the token returned by {@link setContentCapture} — still identifies the most
 * recent set. A stale token no-ops, so an old handle shutting down late cannot
 * clobber an override a newer caller installed (and silently re-enable, or
 * disable, capture via the env fallback). For an unconditional clear, use
 * `setContentCapture(null)`.
 */
export function clearContentCapture(generation: number): void {
  if (generation !== contentCaptureGeneration) return; // a newer set owns the slot
  contentCaptureOverride = null;
}

/**
 * Parse the ecosystem content-capture gate. A mode set via
 * {@link setContentCapture} wins outright (env is the fallback, as in OTel);
 * otherwise default {@link ContentCapture.NoContent} when unset/empty/unrecognized.
 * The legacy boolean form maps `true` to full capture
 * ({@link ContentCapture.SpanAndEvent}) and `false` to none.
 */
export function contentCaptureMode(
  env: Record<string, string | undefined> = process.env,
): ContentCapture {
  if (contentCaptureOverride !== null) {
    return contentCaptureOverride;
  }
  const raw = env[CAPTURE_CONTENT_ENV];
  if (raw == null || raw.trim() === "") {
    return ContentCapture.NoContent;
  }
  return parseContentCapture(raw) ?? ContentCapture.NoContent;
}
