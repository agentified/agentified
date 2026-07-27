// Compile-only assertions for the host-wiring snippet the telemetry docs tell
// readers to copy — `src/telemetry/README.md` and the `## Telemetry` section of
// this package's README both carry it verbatim.
//
// `coexistence.test.ts` proves the *behaviour* (which spans reach which
// destination). What it cannot prove is that the documented snippet still
// compiles, because it writes a scope-only predicate rather than the widened
// `isDefaultExportSpan(...) || ...` form a real host wants. Two shapes in that
// one line are easy to get backwards and neither is checked anywhere else:
// `shouldExportSpan` receives a WRAPPER (`{ otelSpan }`) while
// `isDefaultExportSpan` takes the BARE span, and the scope lives on
// `instrumentationScope`, not the long-deprecated `instrumentationLibrary`.
//
// This package is the only workspace member that resolves both halves — the
// Ratel adapter and a real vendor processor — so the check lives here even
// though the snippet is documented one level up.
import { isDefaultExportSpan, LangfuseSpanProcessor } from "@langfuse/otel";
import type { SpanProcessor } from "@opentelemetry/sdk-trace-base";
import { RatelOtelIntegration } from "./otel.js";

// The documented predicate. Keyed on scope, because `execute_tool <tool name>`
// is emitted under both `@ratel-ai/sdk` and `gen_ai` and its name says nothing
// about the source. Both scope literals are pinned at runtime by
// `coexistence.test.ts`; this only pins that the expression typechecks.
const processor: SpanProcessor = new LangfuseSpanProcessor({
  publicKey: "pk",
  secretKey: "sk",
  shouldExportSpan: ({ otelSpan }) =>
    isDefaultExportSpan(otelSpan) || otelSpan.instrumentationScope.name === "@ratel-ai/sdk",
});

// A `LangfuseSpanProcessor` is assignable to the `spanProcessors` element type a
// host-owned provider takes, which is the whole claim behind "any processor you
// already run receives them". Asserted against `SpanProcessor` rather than
// `new NodeSDK({...})` so the check needs no `@opentelemetry/sdk-node` devDep;
// `NodeSDK`'s `spanProcessors` is `SpanProcessor[]`.
void processor;

// `registerTelemetry(new RatelOtelIntegration())` is typechecked against the
// real `ai@7` `Telemetry` interface by the packed consumer in
// `.github/workflows/ts.yml`. Here we only pin the zero-argument construction
// the snippet uses, and that `origin` accepts the bare literal the README says
// it does (`Origin` is an `as const` map, not an enum).
void new RatelOtelIntegration();
void new RatelOtelIntegration({ origin: "direct" });
