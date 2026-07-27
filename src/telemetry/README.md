# `src/telemetry/`

Ratel's **remote** telemetry: OpenTelemetry conventions plus thin helper packages. Ratel telemetry *is* OpenTelemetry: LLM calls are `gen_ai.*` spans, content-bearing details are Logs `EventRecord`s, the capability/skill funnel is a `ratel.*` overlay, and ingest is stock OTLP. No custom transport or FFI.

Distinct from the local JSONL trace stream (ADR-0007, in [`../core/`](../core/README.md)), which stays as-is; only the remote path lives here.

The wire contract is [`CONVENTIONS.md`](CONVENTIONS.md): the `gen_ai.*` mapping and the `ratel.*` vocabulary every consumer reads against. The helpers below codify the `ratel.*` half as constants.

## Layout

```
CONVENTIONS.md   the telemetry wire contract (gen_ai.* mapping + ratel.* vocabulary)
conformance/     shared contract-against-the-pin fixtures every language helper asserts against
core/            ratel-ai-telemetry (crates.io): the ratel.* constants (shared vocabulary)
ts/              @ratel-ai/telemetry (npm): the ratel.* constants + capture gate, OTel-free
python/          ratel-ai-telemetry (PyPI): the ratel.* constants; init() behind the [otlp] extra
```

The vocabulary is kept OTel-free so the SDK (emit side), the cloud (read side), and edge/serverless emitters take it weight-free (ADR-0007): importing `@ratel-ai/telemetry` or `ratel_ai_telemetry` pulls no OpenTelemetry SDK. In TypeScript the host owns the OpenTelemetry provider outright: `@ratel-ai/telemetry` is vocabulary plus the content-capture gate and carries no exporter configuration at all, so the host resolves its own endpoint and auth and builds the exporters and span/log-record processors on the providers it registers, owning their flush and shutdown. Python keeps turnkey exporter sugar as `init()` behind the optional `[otlp]` extra (`ratel_ai_telemetry.otlp`, reading `RATEL_OTLP_ENDPOINT`, with the superseded `RATEL_URL` still honoured with a warning); the asymmetry is deliberate. The `core/` crate carries the same `ratel.*` constants as the shared source of truth for in-process Rust consumers. The TS and Python helpers' conformance tests build spans and EventRecords from their own constants and assert them against the single shared fixture set in [`conformance/`](conformance/README.md), so those two cannot drift; the `core/` crate stays dependency-free and pins the same constants by literal-equality unit tests.

Rationale and the two-tier design are documented in [ADR 0007](../../docs/adr/0007-telemetry-two-streams.md). Each package is an independent release unit per [ADR 0008](../../docs/adr/0008-release-engineering.md): `core/` ships on `telemetry-core-v*`, `ts/` on `telemetry-ts-v*`, and `python/` on `telemetry-py-v*`.

## What the SDK emits

Emission needs no wiring and no configuration. `@ratel-ai/sdk` writes to whatever OpenTelemetry
providers are registered globally and registers none itself, so with no provider wired every
span is a no-op — there is nothing to switch on, and nothing to switch off. Five span shapes,
all `SpanKind.INTERNAL`, all on instrumentation scope **`@ratel-ai/sdk`** (`ratel-ai` in
Python):

| emitted span name | attributes (gated ones marked) |
| --- | --- |
| `ratel.search` | `ratel.search.target` / `.top_k` / `.hit_count`, `ratel.origin`; gated `ratel.search.query` |
| `execute_tool <tool name>` | `gen_ai.operation.name`, `gen_ai.tool.name`, `ratel.tool.args_size_bytes`, `ratel.upstream.server` (only for a `<server>__<tool>` id); gated `gen_ai.tool.call.arguments` / `.result` |
| `ratel.skill.load` | `ratel.skill.id` |
| `ratel.upstream.register` | `ratel.upstream.server` / `.transport` / `.tool_count` |
| `ratel.auth.flow` | `ratel.upstream.server`, `ratel.auth.outcome` |

The tool span's name carries the tool, so it reads `execute_tool send_email` on the wire; bare
`execute_tool` is the `gen_ai.operation.name` *value*, not the name. The asymmetry in that table
is the load-bearing detail: `execute_tool <…>` is the only *mixed* shape, carrying `gen_ai.*`
keys alongside its `ratel.*` ones while the other four are purely `ratel.*`. That one difference
decides on its own whether a backend shows you the other four (see below).

One known gap: [`CONVENTIONS.md`](CONVENTIONS.md) lists `ratel.origin` on the tool-invocation
span, but neither SDK emits it there — both set it on `ratel.search` only, and the AI SDK
integration's overlay lands on that emitter's own spans rather than on this one. The table above
reports what is emitted; closing the drift is a follow-up on the emit side, not a docs edit.

The content-bearing half rides the Logs data model as two `EventRecord`s —
`ratel.search.results` (the query text) and `ratel.tool.execution.details` (tool arguments, plus
the result on success). Both are off by default and gated by
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` / `setContentCapture()`. Both are
`ratel.*`-named: a filter keyed on `gen_ai.*` event names drops all of them.

## Emission is not delivery

A provider fans every span out to every processor on it, and no processor can starve another.
What each *destination* then keeps is that destination's own decision, made after the span has
already arrived — which means a backend can show you nothing while the wiring is perfectly
correct, with no error anywhere to say so.

The concrete case, pinned by
[`coexistence.test.ts`](../adapters/ts-vercel-ai-sdk/src/coexistence.test.ts): a stock
`new LangfuseSpanProcessor()` keeps a span only if it carries a `gen_ai.*` attribute or comes
from a scope on Langfuse's known-instrumentor list. `@ratel-ai/sdk` is on neither list, so
`execute_tool <…>` survives on the strength of its `gen_ai.tool.name` while `ratel.search`,
`ratel.skill.load`, `ratel.upstream.register` and `ratel.auth.flow` are dropped before export.
A host wires it up, sees its `gen_ai.*` traces and its tool calls land, concludes it works — and
never learns that every retrieval span, the thing Ratel exists to show them, was discarded.

Opting Ratel in is one line, and it belongs in the host's config rather than in Ratel's code,
because the predicate is the vendor's own option. Key it on **scope**, not on a `ratel.` name
prefix: `execute_tool send_email` says nothing about who emitted it, and both Ratel and
`@ai-sdk/otel` emit a span by that name.

```ts
import { isDefaultExportSpan, LangfuseSpanProcessor } from "@langfuse/otel";

new LangfuseSpanProcessor({
  shouldExportSpan: ({ otelSpan }) =>
    isDefaultExportSpan(otelSpan) || otelSpan.instrumentationScope.name === "@ratel-ai/sdk",
});
```

Two details that bite. `shouldExportSpan` receives a wrapper (`{ otelSpan }`) while
`isDefaultExportSpan` takes the bare span, so the two shapes are easy to swap. And Langfuse may
call the predicate at span *start* as well as at end, so it must be side-effect-free and must
tolerate attributes that aren't set yet: a scope-name check is safe under that rule, a check on
attribute keys is not obviously. The scope to match is `@ratel-ai/sdk` in TypeScript and
`ratel-ai` in Python; the two SDKs do not share one.

`otel.scope.name` is in general the only thing that identifies an emitter. Span names collide
across emitters and `gen_ai.*` is shared vocabulary, so neither tells you the source.

## Host wiring

The host always owns the provider. Every scenario is `new NodeSDK({ … })` plus `sdk.start()`,
and flush and shutdown belong to whoever built it. Ratel ships no provider bootstrap.

**1. Ratel on its own, to any OTel backend.** No Ratel telemetry package is involved: the SDK
emits standard OpenTelemetry, so any provider and exporter receive it.

```ts
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-proto";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { ratel } from "@ratel-ai/sdk";

const sdk = new NodeSDK({
  spanProcessors: [
    new BatchSpanProcessor(new OTLPTraceExporter({ url: "https://<your-backend>/v1/traces" })),
    // ...or any vendor processor, side by side — see the predicate above for Langfuse's
  ],
  logRecordProcessors: [
    new BatchLogRecordProcessor({
      exporter: new OTLPLogExporter({ url: "https://<your-backend>/v1/logs" }),
    }),
  ],
});
sdk.start();

const r = ratel({ /* catalogs */ }); // ratel.* spans ride the global tracer from here on
```

Both lists matter. With `spanProcessors` alone, `NodeSDK` builds the logger provider from the
environment and the `EventRecord`s go to the default OTLP endpoint rather than the URL above.
Mind the constructor asymmetry too: `BatchSpanProcessor` takes the exporter positionally while
`BatchLogRecordProcessor` takes `{ exporter }`. TypeScript catches the mix-up (`exporter` is
required, so a positional exporter is a `TS2345`); plain JavaScript does not, and the processor
then silently drops every record.

**2. Add the AI SDK's `gen_ai.*` spans (`ai@7`).** One integration registration, onto the same
host-owned provider; every processor on it receives both families. `RatelOtelIntegration` and
its register-once rule are documented in
[`@ratel-ai/vercel-ai-sdk`](../adapters/ts-vercel-ai-sdk/README.md#telemetry-ratelotelintegration-ai7)
(`0.2.0` or newer, for the `/otel` entrypoint).

```ts
import { registerTelemetry } from "ai";
import { RatelOtelIntegration } from "@ratel-ai/vercel-ai-sdk/otel";
import { ratel } from "@ratel-ai/sdk";
import { aiSdk } from "@ratel-ai/vercel-ai-sdk";

// Scenario 1's NodeSDK, unchanged, with the widened Langfuse predicate added to
// its processor list. This scenario adds only the registration below.

registerTelemetry(new RatelOtelIntegration());

const r = ratel({ /* catalogs */ }).adaptTo(aiSdk());
```

The integration's `enrichSpan` hook decorates only the spans the AI SDK's emitter creates
(scope `gen_ai`). The SDK's own `ratel.*` and `execute_tool <…>` spans come from a different
emitter and carry no host enrichment — which is the second reason the filter above keys on
scope. `RatelOtelIntegration` targets `ai@7`'s seam only: `ai@5` has none, and `ai@6` (from
`6.0.108`) has an earlier, different one this class does not implement. On either, pass
`experimental_telemetry: { isEnabled: true }` per call instead.

**3. Mastra is the exception.** Its AI tracing is a private pipeline that never registers a
global provider, so it coexists with the host's `NodeSDK` without a fight — but its spans never
pass through the host's processors either. Delivery is two parallel egress paths rather than one
shared stream: see
[`@ratel-ai/mastra`](../adapters/ts-mastra/README.md#telemetry).

A complete, offline-runnable emitter built from the vocabulary constants is in
[`examples/telemetry-ts`](../../examples/telemetry-ts/README.md).
