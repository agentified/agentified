# `examples/telemetry-ts` — emit `ratel.*` telemetry with OpenTelemetry (TypeScript)

Shows how to emit Ratel's telemetry vocabulary through the standard [OpenTelemetry JS SDK](https://opentelemetry.io/docs/languages/js/) using [`@ratel-ai/telemetry`](../../src/telemetry/ts/README.md) for constants and OTLP config resolution. Ratel telemetry *is* OpenTelemetry ([ADR-0007](../../docs/adr/0007-telemetry-two-streams.md)): the vocabulary package provides the `ratel.*` constants and value enums, and ships no transport and no bootstrap — the host builds and owns the providers.

The trace-only offline demo emits one realistic trace — a `ratel.search` span followed by an `execute_tool` span under a root agent-turn span — and prints it with a `ConsoleSpanExporter`. The production path adds a `LoggerProvider` so the content-bearing Logs EventRecords go out too.

## Setup

```bash
pnpm install
pnpm -F @ratel-ai/example-telemetry start
```

`start` builds `@ratel-ai/telemetry` and runs `src/index.ts` with [tsx](https://tsx.is/). It prints the trace (the agent-turn root plus the two Ratel spans, with their `ratel.*` / `gen_ai.*` attributes) and shows how exporter setup resolves its endpoint + auth.

To export a real trace instead of printing, set the endpoint and run again:

```bash
export RATEL_OTLP_ENDPOINT=https://cloud.ratel.sh/v1/traces
export RATEL_API_KEY=sk-...          # optional; sent as Authorization: Bearer
pnpm -F @ratel-ai/example-telemetry start
```

## What it illustrates

- **The vocabulary is just constants.** `RATEL_SEARCH`, `EXECUTE_TOOL`, `RATEL_ORIGIN`, `GEN_AI_TOOL_NAME`, … are `import`ed from `@ratel-ai/telemetry` and set as attributes on stock OTel spans. The `Origin` / `SearchTarget` value enums carry the exact wire strings.
- **Tool calls are standard `gen_ai` spans.** The invocation is an `execute_tool` span (so any OTel backend understands it), enriched with `ratel.*` attributes — not a bespoke Ratel span.
- **The host owns the providers.** `resolveOtlpConfig()` (pure, shown in the output) resolves trace and Logs URLs plus auth; the example feeds those into its own `NodeTracerProvider` + `LoggerProvider`, built exactly like the offline one with OTLP batch processors swapped in for the console exporter. It builds them only when `RATEL_OTLP_ENDPOINT` is set; the offline demo above needs neither.
- **A real host registers those providers globally.** The example threads them into its emitter instead, to stay side-effect-free. Call `tracerProvider.register()` and `logs.setGlobalLoggerProvider(loggerProvider)` in your own app: `@ratel-ai/sdk` emits into the global providers.
- **Content capture is gated.** `contentCaptureMode()` reads `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (default `NO_CONTENT`).

## Layout

```
src/index.ts   emitRatelTrace() — builds the trace (root + the two Ratel spans) and the ratel.search.results EventRecord from the constants; main() wires the providers and prints
```

## Why it's a separate workspace package

Examples don't ship in the telemetry package. The OTel trace, Logs and OTLP exporter dependencies the demo needs stay here; `@ratel-ai/telemetry` itself is OTel-free.
