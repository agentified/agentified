<div align="center">
  <h1>@ratel-ai/sdk</h1>
  <p>Context engineering for TypeScript and Node.js agents.</p>

  <p>
    <a href="https://docs.ratel.sh">Docs</a> •
    <a href="https://github.com/ratel-ai/ratel">GitHub</a> •
    <a href="https://discord.gg/75vAPdjYqT">Discord</a>
  </p>

  <p>
    <a href="https://www.npmjs.com/package/@ratel-ai/sdk"><img src="https://img.shields.io/npm/v/@ratel-ai/sdk?label=npm&color=cb3837" alt="npm" /></a>
    <a href="https://github.com/ratel-ai/ratel/stargazers"><img src="https://img.shields.io/github/stars/ratel-ai/ratel?style=social" alt="GitHub stars" /></a>
    <a href="https://github.com/ratel-ai/ratel/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license" /></a>
  </p>
</div>

`@ratel-ai/sdk` retrieves the tools and skills relevant to each agent turn instead of sending the full catalog to the model. It bundles Ratel's Rust engine in-process: BM25 by default, with local semantic and hybrid retrieval available when needed. No API key, vector database, or service is required. Installing a published package on a supported prebuilt target also requires no Rust toolchain.

Use `ToolCatalog` for ranked tools with executable handlers and `SkillCatalog` for ranked playbooks loaded on demand. Expose `searchCapabilitiesTool`, `invokeToolTool`, and `getSkillContentTool` so an agent can discover tools and skills, invoke tools, and load full skill instructions. Tools from existing MCP servers can be ingested into the tool catalog.

Semantic and hybrid retrieval use a configurable embedding model ([ADR 0012](../../../docs/adr/0012-configurable-embedding-models.md)), set per catalog via the `embedding` option: the built-in default, a HuggingFace repo or local directory (in-process), or an OpenAI-compatible endpoint (OpenAI, Ollama, TEI, vLLM).

For semantic or hybrid retrieval, `register()` folds embedding in: it accepts one tool or a whole array and embeds on a libuv worker, so model loading, HTTP, and inference never block Node's event loop — and embedding errors surface right at `register()`:

```ts
const catalog = new ToolCatalog({ method: "semantic", embedding: { ollama: "nomic-embed-text" } });
await catalog.register(tools);                              // embeds the batch here
const hits = await catalog.searchAsync("deploy the service", 5);
```

`register()` returns a promise for every method (BM25 too); `search()` stays synchronous for BM25 only, and `searchAsync()` covers all three. To change the endpoint's model or vector dimension, construct a new catalog and re-register.

A `SkillCatalog` also takes a whole reloaded catalog at once with `replaceAll()`, for a source that fetches the full set rather than individual changes ([ADR 0015](../../../docs/adr/0015-whole-catalog-skill-reload.md)). The batch *is* the catalog: ids missing from it are removed, including ones registered in-process, so a host that mixes local and remote skills composes the batch itself. It mutates in place, so `r.skills`, every adapted view, and the capability tools already handed to the model all see the reload without being rebuilt.

```ts
const outcome = await r.skills.replaceAll([...localSkills, ...(await fetchRemoteSkills())]);
console.log(`reload: +${outcome.added} -${outcome.removed} ~${outcome.updated}`);
```

Only new and re-worded skills are embedded — reloading an unchanged catalog costs no embedding calls — and a reload that races a dense operation throws rather than applying half of itself.

Embedding failures from `register()`/`searchAsync()` are typed `EmbedderError`s (with a stable `.code` such as `"Load"`, `"NotCached"`, or `"DimensionMismatch"`); a dimension mismatch is a `DimensionMismatchError` subclass — the parity of Python's `EmbedderError`/`DimensionMismatchError`. Invalid embedding config still throws at construction.

```ts
import { EmbedderError, DimensionMismatchError } from "@ratel-ai/sdk";

try {
  await catalog.register(tools);
} catch (err) {
  if (err instanceof DimensionMismatchError) {
    // the model changed under the corpus — rebuild with a fresh catalog
  } else if (err instanceof EmbedderError) {
    console.error(`embedding failed (${err.code}): ${err.message}`);
  }
}
```

## Install

```bash
pnpm add @ratel-ai/sdk
```

## Quickstart

Save as `quickstart.mjs`, then run `node quickstart.mjs`:

```js
import { ToolCatalog } from "@ratel-ai/sdk";

const catalog = new ToolCatalog();
await catalog.register({
  id: "get_weather",
  name: "get_weather",
  description: "Get the current weather for a city.",
  inputSchema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  outputSchema: {
    type: "object",
    properties: { forecast: { type: "string" } },
  },
  execute: ({ city }) => ({ forecast: `Sunny in ${city}` }),
});

const [hit] = catalog.search("What is the weather in Rome?", 1);
console.log(await catalog.invoke(hit.toolId, { city: "Rome" }));
```

## Framework adapters

To work in a host framework's native tool and message shapes, adapt the core with a
`RatelAdapter` from a framework package instead of wiring the capability tools by hand:

```js
import { ratel } from "@ratel-ai/sdk";
import { aiSdk } from "@ratel-ai/vercel-ai-sdk"; // ships separately

const r = ratel({ recallTopK: 5 }).adaptTo(aiSdk());
await r.tools.register(myTools);              // async; callable any time, also after modelTools()
const tools = r.modelTools();                 // stable capability set — take once, reuse
const messages = await r.appendRecall(history); // per-turn recall (AI SDK idiom)
```

`r.tools` is a handle over the core's one shared catalog — registration and exposure are separate
acts, and tools registered after `modelTools()` are still discoverable because the capability tools
search the live catalog. `register(...)` is async: it validates synchronously (a bad tool throws at
the call site) and its promise resolves once the tools are indexed and, on a semantic/hybrid core,
embedded — `await` it so embedding errors surface at registration. The core also works standalone,
without any adapter: `ratel().tools.register(...)` takes native `ExecutableTool`s, `modelTools()`
returns the three capability tools in native shape, and `recall(query)` is a pure query returning
the canonical `search_capabilities` result.

`ratel(config)` owns one `ToolCatalog` + `SkillCatalog` + recall-id counter and every
framework-independent guard (reserved capability-tool ids, top-K clamp, first-registration-wins
on the adapted path, passthrough of provider-run tools); an adapter is just three codecs
(`ingest` / `expose` / `recallMessages`) plus its framework idioms. `adaptTo` infers the
framework's tool and message types, so app code needs no casts. A framework tool registered on
the un-adapted core throws an error pointing at the adapter package to install. See ADR-0013.

Continue with the [TypeScript guide](https://docs.ratel.sh/docs/sdks/typescript), [capability tools](https://docs.ratel.sh/docs/capability-tools), [API reference](https://docs.ratel.sh/docs/api/sdk-typescript), or the [Vercel AI SDK example](https://github.com/ratel-ai/ratel/tree/main/examples/ai-sdk).

## Adapter conformance testkit

Building a `RatelAdapter` for another framework? `@ratel-ai/sdk/testkit` ships a runner-agnostic
battery that pins the SPI contract — ingest/expose round-trip, the reserved-id guard, recall
top-K clamping, passthrough semantics, and recall-pair shape. Teach it your framework's tool and
message shapes once, then run the whole battery under your test runner:

```ts
import { describe, it } from "vitest";
import { describeAdapterConformance } from "@ratel-ai/sdk/testkit";
import { myConformanceOptions } from "./conformance-options.js";

describeAdapterConformance(myConformanceOptions(), { describe, it });
```

Assertions use `node:assert`, so no test runner leaks into your published types;
`referenceConformanceOptions` is a worked example to copy. Prefer full control? `adapterConformanceCases(options)` returns the named cases to run yourself.

## Telemetry

Telemetry is emit-only and always on: the SDK writes `ratel.*` / `gen_ai.*` spans and EventRecords to whatever OpenTelemetry providers are registered globally, and registers none itself — with no provider wired, every span is a no-op, so there is nothing to configure or switch off. Delivery is yours:

```ts
import { NodeSDK } from "@opentelemetry/sdk-node";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { BatchLogRecordProcessor } from "@opentelemetry/sdk-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-proto";

new NodeSDK({
  spanProcessors: [new BatchSpanProcessor(new OTLPTraceExporter({ url: "https://<your-backend>/v1/traces" }))],
  logRecordProcessors: [new BatchLogRecordProcessor({ exporter: new OTLPLogExporter({ url: "https://<your-backend>/v1/logs" }) })],
}).start();
```

Both lists matter: with `spanProcessors` alone, `NodeSDK` builds the logger provider from the environment and the EventRecords land on the default OTLP endpoint, not the URL above. Several processors can sit side by side, and flush/shutdown stay with the host that owns the provider.

**A vendor processor may drop most of it, silently.** Emission and delivery are separate: the provider hands every span to every processor, and each destination then applies its own filter. A stock `new LangfuseSpanProcessor()` keeps a span only if it carries a `gen_ai.*` attribute or comes from a scope it already knows, and `@ratel-ai/sdk` is on neither list — so `execute_tool <tool>` survives while `ratel.search`, `ratel.skill.load`, `ratel.upstream.register` and `ratel.auth.flow` do not. Nothing errors; the retrieval spans are simply absent. Widening it is one line of the vendor's own config, keyed on scope — see [`src/telemetry/`](../../telemetry/README.md#emission-is-not-delivery) for the predicate, the full span inventory, and the `ai@7` and Mastra wirings.

Message and tool content is off by default; opt in with the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var or `setContentCapture()` (see the [telemetry guide](https://docs.ratel.sh/docs/telemetry) for the capture modes and their privacy implications). The `ratel.*` constants themselves live in [`@ratel-ai/telemetry`](../../telemetry/ts/README.md); this package re-exports only the content-capture gate.

## Package layout

`src/` is the TypeScript surface, `native/` contains the NAPI binding, `npm/` holds platform packages, and tests live beside their source. From the repository root, run `pnpm --filter @ratel-ai/sdk... build` and `pnpm --filter @ratel-ai/sdk test`.
