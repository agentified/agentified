# `@ratel-ai/telemetry`

The `ratel.*` telemetry vocabulary for TypeScript: the constants that codify the Tier 2
overlay of [`../CONVENTIONS.md`](../CONVENTIONS.md) (attribute keys, span/EventRecord names,
the `Origin`/`SearchTarget`/`AuthOutcome` value enums, the pinned semconv version), plus the
content-capture gate (`contentCaptureMode`). **This package is OTel-free** — importing it
pulls no OpenTelemetry SDK, so the SDK (emit side), the server (read side), and
edge/serverless emitters take the vocabulary weight-free
([ADR-0007](../../../docs/adr/0007-telemetry-two-streams.md)).

The vocabulary and that gate are the whole surface: there is no exporter configuration here.
The host owns the OpenTelemetry provider, so it also owns the endpoint, the auth headers, and
the exporters it builds from them. This package never registers a provider and never reads an
endpoint.

## Usage

```ts
import { trace } from "@opentelemetry/api";
import { EXECUTE_TOOL, GEN_AI_OPERATION_NAME, GEN_AI_TOOL_NAME, Origin, RATEL_ORIGIN } from "@ratel-ai/telemetry";

// Emit a standard gen_ai `execute_tool` span enriched with the ratel.* overlay,
// on your own OTel provider — the vocabulary adds no transport.
const span = trace.getTracer("my-agent").startSpan(EXECUTE_TOOL, {
  attributes: {
    [GEN_AI_OPERATION_NAME]: EXECUTE_TOOL,
    [GEN_AI_TOOL_NAME]: "send_email",
    [RATEL_ORIGIN]: Origin.Agent,
  },
});
span.end();
```

Exporting to Ratel: point your own OTLP exporters at the Ratel traces URL and its sibling
`/v1/logs`, with `Authorization: Bearer <api key>`, on the tracer and logger providers you
build and register. A complete, offline-runnable host wiring (console exporter + a
`ratel.search` → `execute_tool` trace) is in
[`examples/telemetry-ts`](../../../examples/telemetry-ts/README.md).

## Package shape

- Package name: `@ratel-ai/telemetry`
- Pure TypeScript (no native binding), **zero runtime dependencies** (OTel-free)
- Released under the `telemetry-ts-v*` tag prefix ([ADR-0008](../../../docs/adr/0008-release-engineering.md))
- MIT ([ADR-0009](../../../docs/adr/0009-licensing.md)); member of the pnpm workspace

## Build & test

From the repo root:

```bash
pnpm --filter @ratel-ai/telemetry build
pnpm --filter @ratel-ai/telemetry typecheck
pnpm --filter @ratel-ai/telemetry lint
pnpm --filter @ratel-ai/telemetry test
```

The tests cover the vocabulary (each constant asserted against the pin), the content-capture
gate, a purity guard that no OTel dependency or import creeps back in, and the shared
contract-against-the-pin conformance in
[`../conformance/`](../conformance/README.md) (spans and EventRecords built from these
constants through the real SDK must emit the exact pinned keys).
