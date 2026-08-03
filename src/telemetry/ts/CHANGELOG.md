# Changelog

All notable changes to `@ratel-ai/telemetry` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Origin.Baseline` / `BASELINE` (`baseline`) mirrors the new local-trace origin for queries observed while Ratel is not serving retrieval. `ratel.origin` now takes `direct | agent | baseline`.

## [0.2.0] - 2026-07-26

### Removed

- **BREAKING:** the OTLP exporter configuration — `resolveOtlpConfig()` with its `InitOptions` / `ResolvedOtlpConfig` types, and the `OTLP_ENDPOINT_ENV`, `API_KEY_ENV`, and `DEFAULT_SERVICE_NAME` constants. This package is now the `ratel.*` vocabulary plus the content-capture gate, nothing else. A host that owns the OpenTelemetry provider already owns the endpoint and auth that feed its exporters, so resolving them here was a second, Ratel-specific config path beside the standard `OTEL_EXPORTER_OTLP_*` one. To migrate, read your own endpoint, derive the Logs URL by replacing the traces URL's terminal `/v1/traces` with `/v1/logs`, and set `Authorization: Bearer <api key>` — see [`examples/telemetry-ts`](../../../examples/telemetry-ts/README.md) for the whole of it.

### Changed

- The companion `@ratel-ai/telemetry-otlp` package is gone: TypeScript hosts own the OpenTelemetry provider and build the OTLP exporters and span/log-record processors themselves. The `ratel.*` constants, the value enums, `SEMCONV_VERSION`, and the content-capture gate are unchanged.
- `ratel.origin` is now specified for third-party `gen_ai.*` spans that a framework adapter overlays, not just Ratel's own search/invoke spans. The key, its `direct | agent` values, and `SEMCONV_VERSION` are unchanged; only the vocabulary spec widened (`CONVENTIONS.md`), so on the overlay path the value is host-selectable and defaults to `agent`.

## [0.1.3] - 2026-07-24

### Added

- `GEN_AI_SYSTEM_INSTRUCTIONS`, `GEN_AI_INPUT_MESSAGES`, `GEN_AI_OUTPUT_MESSAGES`,
  `RATEL_TOOL_EXECUTION_DETAILS`, and a distinct Logs endpoint in resolved OTLP configuration.

### Changed

- Define content events as structured OpenTelemetry Logs `EventRecord`s and reserve inference output messages for model outputs with `finish_reason`.

## [0.1.2] - 2026-07-11

### Added

- `API_KEY_ENV` (`RATEL_API_KEY`) and API-key environment fallback in `resolveOtlpConfig`. An explicit `apiKey` remains authoritative; the env fallback applies only when neither `apiKey` nor an explicit `Authorization` header is given, so ambient `RATEL_API_KEY` never clobbers a caller-supplied auth header.

## [0.1.1] - 2026-07-10

### Added

- `setContentCapture(mode)`: programmatic override of the content-capture gate. While set, `contentCaptureMode()` returns the given mode regardless of `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` — code-level config wins over the environment, matching how OpenTelemetry treats env vars as the fallback for programmatic configuration. The mode is validated exactly like the env var (case-insensitive, trimmed, legacy `true`/`false`/`1`/`0` forms accepted) and throws a `TypeError` naming the valid values on anything unrecognized — failing loud at config time instead of storing a value that would both disable capture and mask the env var. Pass `null`/`undefined` to clear unconditionally. Returns a generation token identifying the call as the current owner of the override.
- `clearContentCapture(generation)`: clears the override only when `generation` (the token returned by `setContentCapture`) still identifies the most recent set. A stale token no-ops, so an old telemetry handle shutting down late cannot clobber an override a newer caller installed and silently flip capture back to the env value.

## [0.1.0] - 2026-07-06

### Added

- The telemetry helper (ADR-0015): the full `ratel.*` vocabulary as constants (attribute keys, span/event names, `gen_ai.*` interop keys, and the `Origin`/`SearchTarget`/`AuthOutcome` value enums) pinned to OpenTelemetry semconv `gen_ai` v1.42.0.
- Shared contract-against-the-pin conformance suite (`../conformance/fixtures.json`): spans built from the constants through the real SDK must emit the exact pinned keys.
- Usage example in the README (runnable end-to-end in `examples/telemetry-ts`).
- A regression guard that no `@opentelemetry/*` runtime dependency or shipped-source import can creep back into the vocabulary package.

### Changed

- `init()` lives in `@ratel-ai/telemetry-otlp`, not this package: importing `@ratel-ai/telemetry` pulls no OpenTelemetry SDK (ADR-0015), so the SDK (emit), the server (read), and edge/serverless emitters take the `ratel.*` vocabulary weight-free. This package keeps the constants plus the pure `resolveOtlpConfig` / `contentCaptureMode`; callers of `init()` install `@ratel-ai/telemetry-otlp` and import it from there.
- Released as an independent npm unit under the `telemetry-ts-v*` tag prefix.
