# `@ratel-ai/mastra`

The [Mastra](https://mastra.ai) (`@mastra/core`) adapter for [Ratel](https://github.com/ratel-ai/ratel). `ratel(config).adaptTo(mastra())` layers a framework-shaped view over the framework-neutral core (ADR-0013), so a Mastra agent registers its own `createTool()`s, hands the model Ratel's capability funnel, and gets per-turn recall — all in Mastra's native `Tool` and `MastraDBMessage` shapes, with no glue in app code.

Ratel keeps the model's tool list small and stable: instead of advertising every tool, it exposes three capability tools (`search_capabilities` / `invoke_tool` / `get_skill_content`) and injects a ranked, per-turn `search_capabilities` result for the current user message. The core owns all state and every guard (reserved ids, top-K clamp, first-registration-wins, recall-id counter); the adapter is just three codecs plus one recall idiom.

## Usage

```ts
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { mastra } from "@ratel-ai/mastra";
import { ratel } from "@ratel-ai/sdk";
import { z } from "zod";

const r = ratel({ method: "hybrid", recallTopK: 5 }).adaptTo(mastra());

// Register the app's Mastra tools into the shared catalog (any time, also after
// modelTools()). Tools without an `execute` (client/provider-executed) pass through eagerly.
r.tools.register({
  weather: createTool({
    id: "weather",
    description: "Get the weather in a location",
    inputSchema: z.object({ location: z.string() }),
    execute: async ({ location }) => ({ location, tempF: 72 }),
  }),
});

const agent = new Agent({
  id: "assistant",
  name: "assistant",
  instructions: "Help the user with their tasks.",
  model: "openai/gpt-4o-mini",
  // The three capability tools in Mastra shape. Take the set ONCE per agent and
  // reuse it: it never changes across turns, so the prompt cache survives.
  tools: r.modelTools(),
  // Rank the catalog for each user turn and inject the synthetic search_capabilities
  // call+result before the model runs (recall mode).
  inputProcessors: [r.recallProcessor()],
});

const result = await agent.generate("what's the weather in Paris?");
console.log(result.text);
```

Standalone (framework-free) use of the same core is also fine — `r` is `ratel(config)` before `.adaptTo`, exposing native `ExecutableTool`s. See [`@ratel-ai/sdk`](../../sdk/ts/README.md).

## The `recallProcessor()` idiom

`r.recallProcessor()` returns a fresh Mastra [`Processor`](https://mastra.ai/en/docs/agents/input-processors) each call (so several agents each get their own). It implements `processInput`, which Mastra runs **once at the start of every generation** — i.e. once per user turn. On each turn it:

1. reads the last message's text iff that message is the user's turn (multi-part text joins with newlines);
2. ranks the catalog with the core's `recall(query)`;
3. if there are hits, appends the synthetic `search_capabilities` call+result to the messages the model sees.

It is a no-op — spending no recall-id — when the last message is not a user turn, the user text is empty, or nothing matched. Because `processInput` runs once per generation (not per step), the pair is injected once and is **not** re-injected during the agent's tool-call loop.

## Limitations

- **Single-message recall encoding.** A `MastraDBMessage` has no `tool` role: a completed call+result is one assistant message with `content.format: 2` and a single resolved `tool-invocation` part. The recall pair is therefore encoded as **one** assistant message (Mastra renders it to the model as an assistant tool-call followed by a tool result).
- **Direct catalog invocation has no Mastra context.** The normal model path through `invoke_tool` forwards Mastra's complete live `ToolExecutionContext` unchanged, including `requestContext`, workspace, agent thread/resource metadata, `mastra`, and `abortSignal`; `requestContextSchema` therefore validates against the caller's real values. The driver-level escape hatch `r.tools.catalog.invoke(id, args)` bypasses a Mastra invocation, so it retains a minimal fallback context (`{ observe }` no-op, a fresh empty `requestContext`, other live fields absent). Pass tools through `modelTools()` when they depend on request-scoped context.
- **Any Mastra tool schema works.** `ingest` reads Mastra's *already-normalized* input schema, so tools built with zod 3, zod 4, or a raw JSON Schema all catalog correctly — the adapter never re-converts schemas itself. (`zod` is a peer only because the exposed capability tools carry hand-written zod schemas.)
- **Persist the conversation across turns.** Recall fires only when the last message is the user's turn. Standard Mastra memory hygiene applies; if you rebuild the message history per call, keep the user turn last so recall can find it.

## Telemetry

Two streams, and — unlike every other host — they do **not** share a pipeline.

Ratel's half needs no wiring. `@ratel-ai/sdk` emits its `ratel.*` spans onto whatever
OpenTelemetry provider is registered globally, so a host that runs a `NodeSDK` at all gets the
retrieval and tool-execution spans for free; with no provider registered they are no-ops. That
is the same everywhere and is documented in [`src/telemetry/`](../../telemetry/README.md).

Mastra's half is a **private pipeline**. Its AI tracing never registers a global provider — so
there is no fight with the host's `NodeSDK` — but the flip side is that Mastra's spans never
pass through the host's span processors either. They leave over the exporter's own socket. There
are two parallel egress paths, not one shared stream, and they do not share trace ids; joining
the two trees needs Mastra's `bridge` seam (`@mastra/otel-bridge`), not configuration.

Nothing is ambient. An un-instrumented Mastra emits nothing to OpenTelemetry — the `gen_ai.*`
strings in its bundle are a vendored AI SDK copy that stays off unless asked — so Mastra tracing
requires wiring `@mastra/observability` explicitly:

```ts
// pnpm add @mastra/observability @mastra/otel-exporter @opentelemetry/exporter-trace-otlp-proto
import { Mastra } from "@mastra/core";
import { Observability } from "@mastra/observability";
import { OtelExporter } from "@mastra/otel-exporter";
import { NodeSDK } from "@opentelemetry/sdk-node";

// The host's global provider, carrying Ratel's ratel.* spans (Mastra registers none).
new NodeSDK({ spanProcessors: [/* your processors */] }).start();

export const app = new Mastra({
  agents: { assistant },
  observability: new Observability({
    configs: {
      default: {
        serviceName: "my-agent",
        exporters: [
          new OtelExporter({
            provider: {
              custom: {
                endpoint: "https://<your-backend>", // base URL; the per-signal path is appended
                protocol: "http/protobuf",
                headers: { authorization: `Bearer ${process.env.BACKEND_API_KEY}` },
              },
            },
            signals: { logs: false }, // unless your backend serves /v1/logs
          }),
        ],
      },
    },
  }),
});
```

Four things that fail quietly if you improvise:

- **`Observability` is a class from `@mastra/observability`**, a separate package — not
  `@mastra/core`, and not the `@mastra/core/observability` subpath. Passing a plain config
  object instead of an instance does not throw: Mastra warns once and installs
  `NoOpObservability`, and you get no spans at all.
- **The OTLP exporter is a peer of `@mastra/otel-exporter`, not a dependency.**
  `protocol: "http/protobuf"` needs `@opentelemetry/exporter-trace-otlp-proto` on your own
  resolution path; without it the exporter debug-logs and exports nothing. `custom` also
  defaults to `http/json`, so the explicit `protocol` is load-bearing.
- **`endpoint` is a base URL, and only a base URL is unambiguous.** The exporter normalizes what
  you give it — trailing slashes and a trailing `/v1/traces` or `/v1/logs` are stripped, then the
  path for the signal being exported is appended — so a full traces URL resolves correctly for
  both signals too. Passing the base is still the form to document, because it is the only one
  that reads as what it is.
- **Telemetry raises this adapter's Mastra floor to `@mastra/core >= 1.16`** —
  `@mastra/observability` and `@mastra/otel-exporter` both peer that, five minors above the
  `>= 1.11.0` the adapter itself supports. On 1.11–1.15 the adapter works; this wiring does not
  install.

Verified against `@mastra/core@1.51.0`, `@mastra/observability@1.16.2`,
`@mastra/otel-exporter@1.3.5`.

Mastra-side enrichment is deferred. Mastra's `spanOutputProcessors` seam is where a processor
would stamp Ratel's overlay onto Mastra's own spans; Ratel ships none yet, so today the two
streams stay separate and only Ratel's carries `ratel.*`. There is no `./otel` entrypoint on
this package — the `ai@7` integration is specific to the Vercel AI SDK.

## Package shape

- Package name: `@ratel-ai/mastra`
- Pure TypeScript, **zero runtime dependencies** — the adapter is glue. `@mastra/core@>=1.11.0 <2`, `zod@^3.25.0 || ^4.0.0` (matching Mastra's own zod peer), and `@ratel-ai/sdk` are peers the host already installs.
- Live context forwarding spans this adapter and `@ratel-ai/sdk`; upgrade their RCs together. An older SDK silently drops the adapter's opaque context before catalog execution.
- Requires Node.js 22.13 or newer, matching Mastra's own requirement.
- MIT ([ADR-0009](../../../docs/adr/0009-licensing.md)); member of the pnpm workspace; `publishConfig` provenance on.

## Mastra compatibility

The supported range is `@mastra/core@>=1.11.0 <2`. Version 1.11 is the floor because it is the first 1.x release where `createTool()` normalizes zod and raw JSON schemas to the Standard Schema surface that `ingest` reads.

There is no runtime version detection. The adapter stays on the common public tool, message, processor, and `ToolExecutionContext` shapes and owns two tiny compatibility details locally: the direct-call no-op observer fallback and the structural validation-error check. This avoids imports that Mastra only exported later (`isValidationError` in 1.18 and `noopObserve` in 1.37) while preserving their behavior. CI runs the adapter build, suite, and type tests against exact 1.11.0, 1.31.0, and 1.51.0; the worked Mastra example also drives a real 1.51 Agent loop.

## Build & test

From the repo root (the SDK is built first by `pnpm -r build`, which the tests import):

```bash
pnpm --filter @ratel-ai/mastra build
pnpm --filter @ratel-ai/mastra typecheck
pnpm --filter @ratel-ai/mastra lint
pnpm --filter @ratel-ai/mastra test
```

The suite covers the three codecs, the recall processor (including id economy on the no-op paths), a mock-model integration test that drives the real Mastra `Agent` loop, a compile-only type-test locking the supported `@mastra/core` surface, and the `@ratel-ai/sdk/testkit` conformance battery (22 cases, 0 skipped). CI repeats it against the minimum Mastra release.
