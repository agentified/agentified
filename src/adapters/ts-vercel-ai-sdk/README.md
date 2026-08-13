# `@ratel-ai/vercel-ai-sdk`

The [Vercel AI SDK](https://sdk.vercel.ai) (`ai@^5 || ^6 || ^7`) adapter for [Ratel](https://github.com/ratel-ai/ratel). `ratel(config).adaptTo(aiSdk())` layers a framework-shaped view over the framework-neutral core (ADR-0013), so an AI SDK agent registers its own `tool()`s, hands the model Ratel's capability funnel, and gets per-turn recall — all in the SDK's native `Tool` and `ModelMessage` shapes, with no glue in app code.

Ratel keeps the model's tool list small and stable: instead of advertising every tool, it exposes three capability tools (`search_capabilities` / `invoke_tool` / `get_skill_content`) and injects a ranked, per-turn `search_capabilities` result for the current user message. The core owns all state and every guard (reserved ids, top-K clamp, first-registration-wins, recall-id counter); the adapter has three required codecs, the experimental passthrough exposure hook, and two recall idioms.

## Install

Install the compatible GA pair:

```bash
pnpm add @ratel-ai/sdk@^0.6.0 @ratel-ai/vercel-ai-sdk@^0.3.0 ai@^7
```

## Usage

```ts
import { anthropic } from "@ai-sdk/anthropic";
import { aiSdk } from "@ratel-ai/vercel-ai-sdk";
import { ratel } from "@ratel-ai/sdk";
import { type ModelMessage, streamText, tool } from "ai";
import { z } from "zod";

const r = ratel({ method: "hybrid", recallTopK: 5 }).adaptTo(aiSdk());

// Executable tools may register after modelTools(): they live behind the stable
// capability set. Passthrough tools need a fresh modelTools() snapshot; see Limitations.
await r.tools.register({
  weather: tool({
    description: "Get the weather in a location",
    inputSchema: z.object({ location: z.string() }),
    execute: async ({ location }) => ({ location, tempF: 72 }),
  }),
});

// Take the model-facing set once after registering passthrough tools, then reuse
// it: the three capability tools never change across turns, so the prompt cache survives.
const tools = r.modelTools();

const messages: ModelMessage[] = [{ role: "user", content: "what's the weather in Paris?" }];

const result = streamText({
  model: anthropic("claude-haiku-4-5"),
  tools,
  // Rank the catalog for this turn and splice in the synthetic search_capabilities
  // pair (recall mode).
  messages: await r.appendRecall(messages),
});
for await (const delta of result.textStream) process.stdout.write(delta);

// Persist the real tool-loop traffic so next turn sees this turn's calls/results.
// (`responseMessages` is ai@7's accessor; on ai@5/ai@6 push `(await result.response).messages`.)
messages.push(...(await result.responseMessages));
```

`prepareStep` is the alternative injection point — drop it straight into `generateText` / `streamText` / `ToolLoopAgent` and skip the manual `appendRecall` call:

```ts
const result = streamText({
  model: anthropic("claude-haiku-4-5"),
  tools: r.modelTools(),
  messages, // your own history, untouched
  prepareStep: r.prepareStep, // injects the recall pair on step 0
});
```

Standalone (framework-free) use of the same core is also fine — `r` is `ratel(config)` before `.adaptTo`, exposing native `ExecutableTool`s. See [`@ratel-ai/sdk`](../../sdk/ts/README.md).

## Two ways to recall: `appendRecall` vs `prepareStep`

Both inject the same synthetic `search_capabilities` call/result pair; they differ in **persistence**, which is what drives prompt-cache behaviour across turns.

- **`appendRecall(messages)`** mutates your message array in place, appends the pair at the **suffix**, and returns the same array. You persist it into your durable history (right alongside `result.responseMessages`). Because a suffix append only *extends* the transcript prefix your host replays next turn, it grows the cached prefix instead of busting it, and each turn's recall stacks after the last. Cost: your stored history carries one recall pair per turn, and persisting it is your responsibility.

- **`prepareStep`** injects the pair via a step-0 `messages` override on a single `generateText` / `streamText` call, as a fresh array that never touches your stored history. Within a multi-step tool loop the pair rides in **every step's prompt** of that call: `ai@7` carries the step-0 override forward on its own, while `ai@5`/`ai@6` rebuild the prompt per step, so the adapter reinserts the cached pair at its original boundary (per-run state; never a duplicate, never a second recall). Either way it is discarded once the call returns and never enters your durable transcript. Nothing to persist; your history stays clean. Cost: the pair is rebuilt on each call and re-sent on each step of a multi-step call, and — living outside the history your host replays — it accrues no cross-turn cache credit the way a persisted append does.

Rule of thumb: for a long-lived multi-turn agent that already persists `responseMessages`, `appendRecall` keeps recalls inside the cached prefix across turns. For a one-shot or stateless call — or when you'd rather not store recall pairs in your history — `prepareStep` is the lighter drop-in. Both are shipped so a host can measure `cachedInputTokens` on its own traffic and pick.

## Telemetry: `RatelOtelIntegration` (`ai@7`)

Emission and delivery are separate concerns. `@ratel-ai/sdk` already emits its own
`ratel.*` spans onto the global OTel tracer with no wiring at all. This integration adds
the *other* half on `ai@7`: the AI SDK's standard `gen_ai.*` spans, with Ratel's `ratel.*`
overlay stamped on each one. The entrypoint requires `@ratel-ai/vercel-ai-sdk@0.2.0` or newer;
`0.1.0` and its release candidates ship no `./otel`.

It only **creates** spans, onto a provider **you** own. It never registers a provider and
never exports, so whatever processors you already run receive them:

```ts
import { isDefaultExportSpan, LangfuseSpanProcessor } from "@langfuse/otel";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { RatelOtelIntegration } from "@ratel-ai/vercel-ai-sdk/otel";
import { registerTelemetry } from "ai";

const sdk = new NodeSDK({
  spanProcessors: [
    new LangfuseSpanProcessor({
      // Required, not polish. Langfuse's default filter keeps a span only when it carries a
      // gen_ai.* attribute or comes from a scope it already knows; "@ratel-ai/sdk" is on
      // neither list, so the SDK's ratel.* spans are dropped *after* reaching this processor.
      shouldExportSpan: ({ otelSpan }) =>
        isDefaultExportSpan(otelSpan) || otelSpan.instrumentationScope.name === "@ratel-ai/sdk",
    }),
  ],
});
sdk.start();

registerTelemetry(new RatelOtelIntegration());
```

The provider fans spans out; the active host context correlates them. AI SDK runs
`prepareStep` before activating its own step span, so retrieval there inherits the
surrounding request or agent-operation span. HTTP auto-instrumentation normally supplies one.
For a job or other uninstrumented entrypoint, create it explicitly:

```ts
import { trace } from "@opentelemetry/api";

await trace.getTracer("my-app").startActiveSpan("agent turn", async (span) => {
  try {
    await generateText({ model, tools: r.modelTools(), prompt, prepareStep: r.prepareStep });
  } finally {
    span.end();
  }
});
```

Install `@ai-sdk/otel` alongside it — it's an optional peer, and the integration embeds its
`OpenTelemetry` emitter as a private delegate, stamping `ratel.*` through that emitter's
`enrichSpan` hook. An `enrichSpan` of your own still runs (see below). Flush and shutdown
belong to the host's `NodeSDK`.

That predicate is the only non-obvious line, and it is about *delivery*, not emission. The
provider hands every span to every processor; each destination then decides what to keep, after
the span has arrived. Without the widening a host sees its `gen_ai.*` traces and its tool calls
land in Langfuse and reasonably concludes the wiring works, while `ratel.search`,
`ratel.skill.load`, `ratel.upstream.register` and `ratel.auth.flow` are discarded with no error
anywhere. Key it on **scope**, never on a `ratel.` name prefix: this integration's emitter and
`@ratel-ai/sdk` both produce a span named `execute_tool <tool name>`, and `gen_ai.*` attributes
appear on both, so `otel.scope.name` is the only thing that says who emitted what — `gen_ai`
for the AI SDK's emitter, `@ratel-ai/sdk` for the SDK's own. The behaviour is pinned in
[`src/coexistence.test.ts`](src/coexistence.test.ts); the span inventory and the other host
wirings are in [`src/telemetry/`](../../telemetry/README.md#emission-is-not-delivery).

**Match the `@ai-sdk/otel` patch to your `ai@7.0.N`.** `@ai-sdk/otel` doesn't *peer* `ai`, it
*depends* on one exact release: every published `1.0.N` pins `ai@7.0.N`, 1:1 across every stable
`1.0.x`, no exceptions. Install `@ai-sdk/otel@1.0.37` next to `ai@7.0.12` and the resolver nests
a second `ai@7.0.37` under it — the same two-copies-of-`ai` type graph described below. Your
build then fails on types that look identical: `TS2345` on the `registerTelemetry` argument
under the usual `skipLibCheck`, or a pile including `TS2403` without it. The peer here stays
`^1.0.0` on purpose: the release you need is a function of the `ai` *you* pinned, and no static
range declared in this package can see that.

- **Register exactly one emitting integration.** `RatelOtelIntegration`,
  `LangfuseVercelAiSdkIntegration` (from `@langfuse/vercel-ai-sdk` — not the `@langfuse/otel`
  processor in the snippet above), and the bare `OpenTelemetry` from `@ai-sdk/otel` all
  embed the same emitter, so registering two duplicates every `gen_ai.*` span. Every
  processor on the shared provider sees the spans regardless of which one you pick — but
  only this one adds the `ratel.*` overlay. Per-call `telemetry: { integrations: [...] }`
  overrides the global registration for that call.
- **It's a separate entrypoint on purpose.** `@ai-sdk/otel` depends on an exact `ai@7`, so
  exporting it from the package root would pull a second `ai` into every `ai@5`/`ai@6`
  host's type graph — where two copies redeclare `AI_SDK_DEFAULT_PROVIDER` and the host's
  build fails without it ever importing the integration. Importing
  `@ratel-ai/vercel-ai-sdk` costs you nothing if you don't ask for `/otel`.
- **`RatelOtelIntegration` targets the `ai@7` seam only.** `ai@5` has none at all; `ai@6`
  (from `6.0.108`) has an *earlier, different* one — `registerTelemetryIntegration` with a
  six-method `TelemetryIntegration` interface, not `registerTelemetry`/`Telemetry` — which
  this class does not implement. On either major, pass
  `experimental_telemetry: { isEnabled: true }` per call instead. The SDK's own `ratel.*`
  spans are unaffected and still need no wiring.
- **Enrichment is per-emitter, not per-span.** `enrichSpan` is a hook on the embedded emitter,
  so it reaches exactly the spans that emitter creates and nothing else. The SDK's own
  `ratel.*` spans and its `execute_tool <tool name>` span come from a different tracer and are
  never touched by it — including when they carry `gen_ai.*` attributes of their own, which
  `execute_tool` does. Selecting spans by attribute prefix and expecting the overlay on them
  is asserting something untrue; select by scope.
- **Every span gets `ratel.origin`; active retrieval experiments add their full join.**
  When AI SDK work starts inside an `experimentalDefineExperiment` arm callback, the integration
  copies the five controlled baggage fields — experiment id, selection id, arm, role, and
  pseudonymous unit hash — onto every emitted `gen_ai.*` span. The selection id is the exact join
  to the value returned by `select()`. Register a ContextManager unconditionally: without one,
  the experiment arm's direct attributes survive but descendant baggage does not. Starting
  `generateText()` only after `select()` returns is also too late; the arm context has been
  restored by then. Ratel-owned values land after a host `enrichSpan`, so that hook cannot
  counterfeit them.
- **`origin` is selectable, and your `enrichSpan` composes.**
  `new RatelOtelIntegration({ origin: Origin.Direct })` overrides the `agent` default. `Origin`
  is re-exported from this entrypoint, so you don't need `@ratel-ai/telemetry` on your own
  resolution path; the bare `"direct"` / `"agent"` literals work too. `agent` is the honest
  value for the tool-loop spans that dominate a Ratel agent and the wrong one for `embed` /
  `embedMany` / `rerank`, which the host calls directly instead of the model synthesizing them
  mid-loop — that split is why the value is selectable rather than hardcoded. It is fixed per
  instance, so a host doing both passes a second integration per call via
  `telemetry: { integrations: [...] }` rather than re-registering globally.
  Every other option passes through to the embedded emitter, including an `enrichSpan` of your
  own: it is called and its attributes merged *under* Ratel's. `ratel.origin` is always a
  controlled overlay; during an active experiment the five experiment id, selection id, arm,
  role, and unit keys are controlled too. Host attributes on every other key land unchanged. If
  your hook throws, you lose your own attributes for that span and nothing else — Ratel's
  controlled overlays still land.

## Limitations

- **Persist the response messages** (`await result.responseMessages` on `ai@7`; `(await result.response).messages` on `ai@5`/`ai@6`, which have no `responseMessages`). Recall fires only when the last message is the user's turn. If you drop the accumulated response messages between turns, turn *N+1* loses turn *N*'s tool calls and results — standard AI SDK message hygiene, load-bearing here.
- **`modelTools()` snapshots passthrough tools.** Plain function tools enter the shared catalog and may register after a snapshot because the model still reaches them through the stable capability tools. Provider-defined/dynamic tools, tools without an `execute`, and tools with AI SDK-only model metadata or lifecycle behavior (`contextSchema`, approval/input hooks, `toModelOutput`, provider options/metadata, strict mode, input examples, or title) stay native. At each snapshot, a client-executed passthrough gets a descriptor-preserving execution wrapper that leaves its lifecycle hooks, metadata, registered-tool `this`, options, and scalar/promise/stream return shape intact while sending the call through Ratel's `execute_tool` funnel; the registered tool object is never mutated. A passthrough with no client-side `execute` is provider- or host-executed, so Ratel cannot observe its invocation and emits no SDK tool span for it. Register passthroughs before taking the snapshot, or call `modelTools()` again and replace the model-facing set.
- **Cataloged executable schemas must resolve synchronously.** Registration synchronously rejects a cataloged executable tool whose `inputSchema` or `outputSchema` converts to a Promise. The whole registration batch remains unchanged. Native passthrough tools never enter this conversion path. Use a synchronous Zod schema or static JSON Schema wrapper for cataloged tools.
- **Live execution options thread through `invoke_tool`; direct catalog calls fall back.** When the model runs a cataloged tool through `invoke_tool`, the adapter forwards the AI SDK's complete live execution options unchanged — `toolCallId`, `messages`, `abortSignal`, and the outer capability's context field (`experimental_context` on `ai@6`/late `ai@5`, `context` on `ai@7`). A tool declaring its own `contextSchema` stays native, so the host validates and routes its named context normally. The driver-level escape hatch `r.tools.catalog.invoke(id, args)` has no AI SDK invocation to thread, so it validates the original input schema and uses a fabricated fallback (`toolCallId: "ratel_<id>"`, `messages: []`, both context spellings `undefined`). Live-option forwarding spans this adapter and `@ratel-ai/sdk` — upgrade their RCs together; an older SDK (before `0.5.1-rc.1`) drops the opaque context before catalog execution.
- **`appendRecall` is async.** Core recall is asynchronous (unlike the sync prototype this was extracted from) — `await` it.
- **Dynamic tool descriptions resolve once, at ingest.** Retrieval ranks on the description at registration time, so a function `description` is called once with a null context (`{ context: undefined }`) when the tool is registered. A description that depends on live tool context won't reflect it in ranking.

## Compatibility

Peer range: **`ai@^5.0.0 || ^6.0.0 || ^7.0.0`** — one shared code path, no per-major builds. The differences the adapter absorbs: provider-defined tools use `type: "provider-defined"` in `ai@5` vs `type: "provider"` in `ai@6`/`ai@7`; tool executors get `experimental_context` in `ai@6` and later `ai@5` releases (the `5.0.0` floor predates any context field) vs `context` in `ai@7` — the adapter forwards whichever spelling the host set live through `invoke_tool`, and fabricates both only for the direct-call fallback; `prepareStep`'s step-0 override is carried forward by `ai@7` but rebuilt per step by `ai@5`/`ai@6` (the adapter reinserts). One difference stays host-side: the persisted-history accessor is `result.responseMessages` on `ai@7` vs `(await result.response).messages` on `ai@5`/`ai@6`.

Approval (`needsApproval`) is available on AI SDK 6+, while per-tool `contextSchema` is AI
SDK 7-only. When present, both stay on the native passthrough path.

Experiment joins degrade explicitly on older majors: `ai@5` has no telemetry-integration seam,
and `ai@6` has a different legacy seam that `RatelOtelIntegration` does not implement. Their
framework-emitted `gen_ai.*` spans therefore receive no direct Ratel experiment overlay. The
adapter still routes every client-executed passthrough through the SDK's own `execute_tool` span
(cataloged tools already use it), preserving active baggage for a host
`BaggageSpanProcessor`; passthroughs without `execute` remain invisible as described above.

Each supported major is verified in CI at two exact releases — its floor and its latest verified release — as `ai@5.0.0`, `5.0.217`, `6.0.0`, `6.0.232`, `7.0.0`, `7.0.37` (the `ai-sdk-compat` matrix in `.github/workflows/ci.yml`): every row builds, typechecks, tests, packs, and typechecks a packed-tarball consumer against that exact `ai`. Every row also asserts that consumer resolved **no** `@ai-sdk/otel`; the v7 rows additionally install it and typecheck `RatelOtelIntegration` against the real `ai@7` `Telemetry` interface. Releases between floor and latest are covered by the range, not row-verified.

- **`ai@4` is excluded.** The v5 release reshaped the tool and message surface the adapter speaks (`inputSchema`/`ModelMessage`-era shapes); `ai@4` predates it and would need a different adapter, not a wider range.
- **Breaking-change policy:** narrowing the supported-majors peer range (dropping a major) is a breaking change of this adapter and ships as a major (post-1.0) with a changelog callout — never a patch or minor. Widening the range to a new `ai` major is additive.

## Package shape

- Package name: `@ratel-ai/vercel-ai-sdk`
- Two entrypoints: `.` (the adapter) and `./otel` (the `ai@7` telemetry integration).
- Pure TypeScript, and the adapter itself is still glue: `ai@^5.0.0 || ^6.0.0 || ^7.0.0` and `@ratel-ai/sdk` are peers the host already installs, and `@ai-sdk/otel` / `@opentelemetry/api` are *optional* peers only `./otel` needs. Its one runtime dependency is `@ratel-ai/telemetry`, the zero-dependency `ratel.*` constants package that `./otel` imports unconditionally — a peer there could go unmet or unhoisted and break the import at load time, which no typecheck would catch.
- MIT ([ADR-0009](../../../docs/adr/0009-licensing.md)); member of the pnpm workspace; `publishConfig` provenance on.

## Build & test

From the repo root (the SDK is built first by `pnpm -r build`, which the tests import):

```bash
pnpm --filter @ratel-ai/vercel-ai-sdk build
pnpm --filter @ratel-ai/vercel-ai-sdk typecheck
pnpm --filter @ratel-ai/vercel-ai-sdk lint
pnpm --filter @ratel-ai/vercel-ai-sdk test
```

The suite covers the three codecs, both recall helpers (including id economy on the no-op paths), mock-model integration tests that drive real two-step `generateText` / `streamText` loops, a compile-only type-test locking the `ai` surface, and the `@ratel-ai/sdk/testkit` conformance battery (22 cases, 0 skipped). The dev `ai` is pinned to the exact release the adapter was last live-verified on; the CI matrix re-pins it per row (see [Compatibility](#compatibility)).
