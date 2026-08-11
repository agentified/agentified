# Changelog

All notable changes to `ratel-ai-telemetry` (the Python telemetry helper) are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0-rc.0] - 2026-08-11

### Added

- `SearchTarget.FACT` (`"fact"`) — the third `ratel.search.target` value, beside `TOOL` and `SKILL`, for the fact retrieval the SDKs' grounding path emits. Additive: the existing members, their values, and `SEMCONV_VERSION` are unchanged.

## [0.1.3] - 2026-07-26

### Added

- `GEN_AI_SYSTEM_INSTRUCTIONS`, `GEN_AI_INPUT_MESSAGES`, `GEN_AI_OUTPUT_MESSAGES`,
  `RATEL_TOOL_EXECUTION_DETAILS`, distinct Logs endpoint resolution, and composable
  `ratel_log_record_processor`, `ratel_log_exporter`, and `ratel_event_filter` helpers.
- `OTLP_ENDPOINT_ENV` (`RATEL_OTLP_ENDPOINT`), the dedicated OTLP traces endpoint variable, and `LEGACY_ENDPOINT_ENV` (`RATEL_URL`) naming the superseded one. `RATEL_URL` also selects the SDK's catalog source (ADR-0003), so overloading it as the OTLP destination conflated two unrelated destinations; TypeScript made the same move in `@ratel-ai/telemetry` 0.2.0.

### Changed

- `init()` now owns and shuts down matched tracer and logger providers.
- Define content events as structured OpenTelemetry Logs `EventRecord`s and reserve inference output messages for model outputs with `finish_reason`.
- `resolve_otlp_config()` resolves the endpoint as explicit `endpoint=` > `RATEL_OTLP_ENDPOINT` > `RATEL_URL`, and the missing-endpoint error names the new variable. Nothing breaks here: a `RATEL_URL`-only install keeps working and raises a `DeprecationWarning` naming the replacement. `ENDPOINT_ENV` stays exported and now equals `OTLP_ENDPOINT_ENV`, so code that sets `os.environ[ENDPOINT_ENV]` targets the new variable automatically. The `RATEL_URL` fallback is scheduled for removal in the next minor.
- `RATEL_ORIGIN` is now specified for third-party `gen_ai.*` spans a framework adapter overlays, not just Ratel's own search/invoke spans. The constant, its `direct | agent` values, and `SEMCONV_VERSION` are unchanged; only the shared vocabulary spec widened (`CONVENTIONS.md`).

## [0.1.2] - 2026-07-11

### Added

- `API_KEY_ENV` (`RATEL_API_KEY`) and API-key environment fallback. Explicit `api_key=` remains authoritative; the env fallback applies only when neither `api_key=` nor an explicit `Authorization` header is given, so ambient `RATEL_API_KEY` never clobbers a caller-supplied auth header.
- On first setup, `init(enabled=False)` returns an OTel-free no-op handle without endpoint configuration or the `[otlp]` extra; once Ratel owns the provider, repeated calls return it. `ratel_span_processor(enabled=False)` always returns a no-op processor.
- `init(span_filter=...)` filters the turnkey provider, and repeated/module-reloaded calls return the exact Ratel-owned handle while foreign providers still raise. If another `init()` wins the registration race, the loser now returns that Ratel-owned handle instead of raising.
- Public `TelemetryHandle` protocol describing `init()`'s return (`shutdown()` / `force_flush()`).

### Changed

- `init()` is now typed to return a `TelemetryHandle` (shutdown handle), not a `TracerProvider` — the disabled/no-op path no longer masquerades as a full provider. Emit spans through the global OTel API (`opentelemetry.trace.get_tracer(...)`).
- `init()` shutdown is terminal: OTel's global provider is set once per process, so after the handle's `shutdown()` a later `init()` raises instead of returning a provider whose exporter is dead. A shared handle's `shutdown()` stops export for all callers.

## [0.1.1] - 2026-07-10

### Added

- `set_content_capture(mode)`: programmatic override of the content-capture gate. While set, `content_capture_mode()` returns the given mode regardless of `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` — code-level config wins over the environment, matching how OpenTelemetry treats env vars as the fallback for programmatic configuration. The mode is validated exactly like the env var (case-insensitive, trimmed, legacy `true`/`false`/`1`/`0` forms accepted) and raises a `ValueError` naming the valid values on anything unrecognized — failing loud at config time instead of storing a value that would both disable capture and mask the env var. Pass `None` to clear unconditionally. Returns a generation token identifying the call as the current owner of the override.
- `clear_content_capture(generation)`: clears the override only when `generation` (the token returned by `set_content_capture`) still identifies the most recent set. A stale token no-ops, so an old telemetry handle shutting down late cannot clobber an override a newer caller installed and silently flip capture back to the env value.

## [0.1.0] - 2026-07-06

### Added

- The telemetry helper (ADR-0015): the full `ratel.*` vocabulary as constants (attribute keys, span/event names, `gen_ai.*` interop keys, and the `Origin`/`SearchTarget`/`AuthOutcome` value enums) pinned to OpenTelemetry semconv `gen_ai` v1.42.0.
- `init()` sugar over the OpenTelemetry Python SDK: wires an OTLP `http/protobuf` exporter to `RATEL_URL` (or `endpoint=`/`headers=`) with a `service.name` resource and batch processor, and returns the provider as a shutdown handle. `content_capture_mode()` reads the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` gate (default off). OTel deps pinned below 1.42 to keep Python 3.9 support.
- Shared contract-against-the-pin conformance suite (`../conformance/fixtures.json`): spans built from the constants through the real SDK must emit the exact pinned keys.
- Usage example in the README (runnable end-to-end in `examples/telemetry-python`).
- `ratel_span_processor()` / `ratel_span_exporter()` + the default `ratel_signal_filter`: a composable OTLP span-processor for multi-provider coexistence. OpenTelemetry's model is one provider with many span-processors, so a partner already running one (e.g. Langfuse + the Vercel AI SDK) calls `provider.add_span_processor(ratel_span_processor(...))` to dual-export to Ratel — forwarding only the `gen_ai.*` / `ratel.*` signal (overridable via `span_filter`), so the framework's `ai.*` wrapper noise stays out of Ratel. Resolvable top-level via the lazy accessor (a plain `import ratel_ai_telemetry` still pulls no OTel).
- A regression guard that a plain `import ratel_ai_telemetry` pulls no OpenTelemetry SDK.

### Changed

- The OpenTelemetry SDK is an optional `[otlp]` extra: importing `ratel_ai_telemetry` pulls no OTel SDK (ADR-0015), so the SDK (emit), the server (read), and edge/serverless emitters take the `ratel.*` vocabulary weight-free. `init()` lives in the `ratel_ai_telemetry.otlp` submodule (behind the extra) and raises a clear "install `ratel-ai-telemetry[otlp]`" error when it is absent; a lazy top-level accessor keeps `from ratel_ai_telemetry import init` working.
- `init()` refactored onto `ratel_span_processor` (still exports every span — it owns the provider) and now raises, pointing at `ratel_span_processor`, when a `TracerProvider` is already registered globally, instead of silently no-op'ing.
- Released as an independent PyPI unit under the `telemetry-py-v*` tag prefix.
