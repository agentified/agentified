/**
 * Pure OTLP config resolution + the content-capture gate for the `ratel.*`
 * telemetry vocabulary. No OpenTelemetry SDK import (ADR-0007): these helpers only
 * resolve endpoint/auth precedence and parse the capture env var, so they stay
 * weight-free for the three consumers that need the vocabulary without the
 * exporter — the SDK (emit side), the server (read side), and edge/serverless
 * emitters. The host owns the exporter and feeds it what `resolveOtlpConfig()`
 * returns.
 */

import { CAPTURE_CONTENT_ENV } from "./index.js";

/**
 * Env var whose value is the OTLP traces endpoint.
 */
export const OTLP_ENDPOINT_ENV = "RATEL_OTLP_ENDPOINT";

/** Env var whose value is the default Ratel API key. */
export const API_KEY_ENV = "RATEL_API_KEY";

/** `service.name` used when the caller does not pass one. */
export const DEFAULT_SERVICE_NAME = "ratel";

/**
 * {@link resolveOtlpConfig} resolves the endpoint from
 * `RATEL_OTLP_ENDPOINT` and auth from `RATEL_API_KEY`, with explicit
 * `{ endpoint, apiKey }` values taking precedence over the environment. An
 * explicit `apiKey` sets the Bearer header; the
 * `RATEL_API_KEY` fallback only applies when neither `apiKey` nor an explicit
 * `Authorization` header is given, so ambient env never overrides an auth header
 * the caller passed on purpose.
 */
export interface InitOptions {
  /** `service.name` resource attribute. Defaults to {@link DEFAULT_SERVICE_NAME}. */
  serviceName?: string;
  /** Ratel API key; sent as `Authorization: Bearer <apiKey>`. Defaults to `RATEL_API_KEY`. */
  apiKey?: string;
  /** Full OTLP traces URL (incl. `/v1/traces`). Defaults to `RATEL_OTLP_ENDPOINT`. */
  endpoint?: string;
  /** Full OTLP logs URL. Defaults to the sibling `/v1/logs` URL derived from {@link endpoint}. */
  logsEndpoint?: string;
  /** Extra headers merged onto the request. An explicit `Authorization` here is kept over the `RATEL_API_KEY` env fallback. */
  headers?: Record<string, string>;
}

/** Resolved exporter configuration; the pure core of the OTLP exporter, exposed for testing. */
export interface ResolvedOtlpConfig {
  url: string;
  logsUrl: string;
  headers: Record<string, string>;
  serviceName: string;
}

/**
 * Resolve {@link InitOptions} into concrete exporter config. Pure and
 * env-injectable so the endpoint/auth precedence is testable without a network.
 */
export function resolveOtlpConfig(
  opts: InitOptions = {},
  env: Record<string, string | undefined> = process.env,
): ResolvedOtlpConfig {
  const url = opts.endpoint ?? env[OTLP_ENDPOINT_ENV];
  if (!url) {
    throw new Error(
      `ratel telemetry init: no endpoint. Pass { endpoint } or set ${OTLP_ENDPOINT_ENV} ` +
        `(use { apiKey } or ${API_KEY_ENV} for Bearer auth).`,
    );
  }
  const headers: Record<string, string> = { ...opts.headers };
  // An explicit apiKey sets the Bearer header (code-level config wins). The RATEL_API_KEY
  // env fallback applies only when the caller passed neither apiKey nor an Authorization
  // header, so ambient env never clobbers auth the caller set on purpose.
  if (opts.apiKey) {
    headers.Authorization = `Bearer ${opts.apiKey}`;
  } else if (env[API_KEY_ENV] && !hasAuthorizationHeader(headers)) {
    headers.Authorization = `Bearer ${env[API_KEY_ENV]}`;
  }
  return {
    url,
    logsUrl: opts.logsEndpoint ?? deriveLogsUrl(url),
    headers,
    serviceName: opts.serviceName ?? DEFAULT_SERVICE_NAME,
  };
}

function deriveLogsUrl(tracesUrl: string): string {
  return tracesUrl.replace(/\/v1\/traces(?=\/?(?:[?#]|$))/, "/v1/logs");
}

/** Whether the caller already supplied an `Authorization` header (any casing). */
function hasAuthorizationHeader(headers: Record<string, string>): boolean {
  return Object.keys(headers).some((key) => key.toLowerCase() === "authorization");
}

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
