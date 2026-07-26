# `@ratel-ai/telemetry`

The `ratel.*` telemetry vocabulary for TypeScript: the constants that codify the Tier 2
overlay of [`../CONVENTIONS.md`](../CONVENTIONS.md) (attribute keys, span/EventRecord names,
the `Origin`/`SearchTarget`/`AuthOutcome` value enums, the pinned semconv version), plus
the pure OTLP config resolver (`resolveOtlpConfig`) and the content-capture gate
(`contentCaptureMode`). **This package is OTel-free** — importing it pulls no OpenTelemetry
SDK, so the SDK (emit side), the server (read side), and edge/serverless emitters take the
vocabulary weight-free ([ADR-0007](../../../docs/adr/0007-telemetry-two-streams.md)).
Wiring an OpenTelemetry provider is the host's job; this package never registers one.

`resolveOtlpConfig()` reads the OTLP traces endpoint from `RATEL_OTLP_ENDPOINT` and auth from
`RATEL_API_KEY`; explicit
`endpoint` / `logsEndpoint` / `apiKey` values win over the environment. `endpoint` is the full
traces URL. The Logs URL is derived by replacing its terminal `/v1/traces` with `/v1/logs`, or can
be set explicitly with `logsEndpoint`. The package exports the environment names as
`OTLP_ENDPOINT_ENV` and `API_KEY_ENV`.

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

Exporting to Ratel: feed the endpoints and headers `resolveOtlpConfig()` returns into your
own OTLP exporters, on the tracer and logger providers you build and register. A complete,
offline-runnable host wiring (console exporter + a
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

The tests cover the vocabulary (each constant asserted against the pin), `resolveOtlpConfig`'s
endpoint/auth resolution and precedence, the content-capture gate, a purity guard that no OTel dependency
or import creeps back in, and the shared contract-against-the-pin conformance in
[`../conformance/`](../conformance/README.md) (spans and EventRecords built from these
constants through the real SDK must emit the exact pinned keys).
