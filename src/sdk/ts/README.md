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

Use `ToolCatalog` for ranked tools with executable handlers and `SkillCatalog` for ranked playbooks loaded on demand. Expose `searchCapabilitiesTool`, `invokeToolTool`, and `getSkillContentTool` so an agent can discover tools and skills, invoke tools, and load full skill instructions. Tools from existing MCP servers can be ingested into the tool catalog. **Experimental — facts:** the `experimental` namespace adds `experimental.FactCatalog` for constant grounding content (a shop's address, a brand's voice). See [Facts](#facts-experimental) below. This API may change without a major-version bump.

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

The corpus swap is the synchronous half of that call, so the counts are already final when it returns — read them without awaiting and a reload whose embedding pass fails still reports what it changed:

```ts
const reload = r.skills.replaceAll(batch); // corpus is live; counts are final
try {
  await reload; // drives the embedding pass
} catch {
  console.warn(`applied +${reload.added} -${reload.removed}, embeddings pending`);
}
```

Only new and re-worded skills are embedded — reloading an unchanged catalog costs no embedding calls — and a reload that races an in-flight operation — dense work, but also an ordinary BM25 `searchAsync` holding the read lock — throws rather than applying half of itself.

Build-time embedding artifacts ([ADR 0018](../../../docs/adr/0018-build-time-embedding-artifacts.md), experimental) avoid corpus/document embedding inference for covered entries on cold start: `experimentalBuildEmbeddingArtifact` writes a mixed Tool+Skill RAT1 (halves merged internally; no public merge API), and catalogs accept `experimentalEmbeddingArtifact: { path } | { bytes }` (default `onMiss: "error"`) to warm the dense cache on `register` / `replaceAll` — each call re-resolves the artifact source and re-warms the whole current corpus. `ToolRegistry` / `SkillRegistry` also expose `experimentalBuildEmbeddingArtifact` and `experimentalWarmEmbeddingsFromArtifact`. With default `onMiss: "error"`, every id in each non-empty registering corpus must be covered; a tool-only artifact is valid while Skill stays empty (and vice versa); when both sides register, use a mixed artifact or `onMiss: "embed"`. Semantic/hybrid search still requires query embedding through the configured backend; Local/HF paths may still initialize/load the model, and endpoint performs its normal remote query embedding. `ArtifactWarmError` covers warm failures (`.code`, `.missing` where applicable); `ArtifactError` covers non-embedder artifact construction failures (embedding/backend failures remain `EmbedderError`); `IncompatibleMergeError` extends `ArtifactError` and may surface from the public mixed builder's internal Tool+Skill merge.

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
on the adapted path, passthrough of provider-run tools); an adapter has three required codecs
(`ingest` / `expose` / `recallMessages`), an optional `experimentalExposePassthrough` hook, plus
its framework idioms. The experimental hook receives a core-owned invocation wrapper, so a
framework can preserve native tool semantics while client-side passthrough execution still enters
the standard OTel/local trace funnel. `adaptTo` infers the framework's tool and message types, so
app code needs no casts. A framework tool registered on the un-adapted core throws an error
pointing at the adapter package to install. See ADR-0013.

Continue with the [TypeScript guide](https://docs.ratel.sh/docs/sdks/typescript), [capability tools](https://docs.ratel.sh/docs/capability-tools), [API reference](https://docs.ratel.sh/docs/api/sdk-typescript), or the [Vercel AI SDK example](https://github.com/ratel-ai/ratel/tree/main/examples/ai-sdk).

## Facts (experimental)

Tools and skills are **pulled** — a query ranks them and only the winners reach the model. Facts are the opposite: constant content the agent should always work from (a shop's address, hours, a brand's voice), **pushed** into the context and deduplicated so it is injected once rather than every turn.

Facts live behind the `experimental` namespace and may change without a major-version bump. Registering one is like a skill, plus a `pin` tier:

```ts
import { ratel, experimental } from "@ratel-ai/sdk";

const r = ratel();
await r.facts.register([
  {
    id: "shop-address",
    name: "shop address & hours",
    description: "where the shop is and when it's open",
    body: "Fade & Blade — 12 Baker Street, London. Open Mon–Sat 9am–7pm.",
    pin: experimental.Pin.Always,     // every turn, regardless of the query
  },
  {
    id: "cancellation",
    name: "cancellation policy",
    description: "cancelling or rescheduling a booking, and refunds",
    body: "Cancel at least 24h ahead for a full refund; same-day is a 50% fee.",
    pin: experimental.Pin.Retrieved,  // only when the turn's query ranks it in (default)
  },
]);
```

Then pick **one** of two injection modes per turn — the same persist-vs-per-call split as `appendRecall` vs `prepareStep` for recall.

**`ground()` — persist into your stored history.** Returns only the facts not already present; render each `body` verbatim and keep it in the messages you save. It takes **per-message text**, so flatten your history first — the AI SDK's `ModelMessage.content` is `string | Array<Part>`, and handing it an array of parts makes the presence check element equality, which never matches (every fact would then re-inject every turn):

```ts
const textOf = (m: ModelMessage): string =>
  typeof m.content === "string"
    ? m.content
    : m.content.map((p) => ("text" in p ? p.text : "")).join("\n");

const { inject } = await r.ground(userText, messages.map(textOf));
for (const f of inject) {
  messages.push({ role: "system", content: f.body }); // verbatim — presence is the dedupe
}
```

Turn 1 injects the address; turn 2 sees it in the transcript and injects nothing. It re-injects only when the body is gone (compaction) or was edited — `f.reason` is `never` / `evicted` / `mutated`.

**`groundSnapshot()` — per call, nothing stored.** Returns the full applicable set every time; put it in the request you're about to send and discard it:

```ts
const facts = await r.groundSnapshot(userText);
const result = await generateText({
  model,
  messages: [...facts.map((f) => ({ role: "system", content: f.body })), ...messages],
});
```

Use `ground()` for a long-lived agent whose messages you persist (pays once, stays in the cached prefix); `groundSnapshot()` for one-shot or stateless calls, or to keep injected content out of your stored history.

Facts are **host-driven, on their own path**: `ground()`/`groundSnapshot()` are the only ways facts reach the context. `modelTools()`, the model-facing `search_capabilities` tool, and `recall()` are all unchanged and never return facts — the model doesn't discover facts by calling a tool, and there is no second place to look. You decide what is true and inject it. Every decision is traced (`fact_inject` with its reason, `fact_inject_skip`, `fact_snapshot`), so the skip rate — the tokens you saved — is measurable. See [ADR-0017](../../../docs/adr/0017-facts-and-injection-freshness.md).

## Retrieval experiments (experimental)

`experimentalDefineExperiment` wraps host-supplied async selectors with deterministic A/B
assignment and bounded shadow evaluation. It is opt-in and TypeScript-only. It does not change
`ToolCatalog` or `SkillCatalog` ranking, and there is deliberately no stable `defineExperiment`
alias yet.

This complete example assigns a stable unit, serves one arm, runs the other as detached shadow
work, compares their rankings, attributes a later invocation, and reports an outcome:

```ts
import { experimentalDefineExperiment } from "@ratel-ai/sdk";

type SearchParams = { query: string };
type SearchHit = { id: string; score: number };
type SearchArm = "control" | "candidate";

const experiment = experimentalDefineExperiment<SearchParams, SearchHit[], SearchArm>({
  id: "retrieval-v2",
  arms: {
    control: {
      select: async () => [
        { id: "inspect-ci", score: 1 },
        { id: "search-logs", score: 0.8 },
      ],
    },
    candidate: {
      select: async () => [
        { id: "search-logs", score: 0.9 },
        { id: "inspect-ci", score: 0.7 },
      ],
    },
  },
  split: [
    { arm: "control", weight: 90 },
    { arm: "candidate", weight: 10 },
  ],
  ranking: (hits) => hits.map(({ id, score }) => ({ id, score })),
  evaluation: {
    k: 2,
    references: [
      "peer-selection",
      { kind: "invocation", window: { turns: 5, maxAgeMs: 300_000 } },
    ],
    outcome: true,
  },
  fallbackArm: "control",
  shadowPolicy: { concurrency: 1 },
});

const selection = await experiment.select(
  { query: "build failure" },
  { unitId: "account-42", shadow: true, k: 2 },
);

const [chosen] = selection.result;
if (chosen !== undefined) {
  experiment.reportInvocation({ unitId: "account-42", toolId: chosen.id });
}
experiment.reportOutcome({
  selectionId: selection.selectionId,
  label: "accepted",
  score: 1,
});

console.log(selection.assignedArm, selection.effectiveArm, selection.result);
await experiment.drain();
```

The public instance has five operations:

| operation | contract |
| --- | --- |
| `select(params, options)` | Returns the transformed effective result plus `selectionId`, `assignedArm`, `effectiveArm`, and arm-callback `durationMs`. An explicit `arm` overrides `split`. |
| `warm()` | Starts unresolved arm warmups concurrently. It never rejects; failed warmups remain cold and retry on the next call. Selection never waits for warmup. |
| `drain()` | Waits for a snapshot of detached shadow and comparison work, always all-settled. It is repeatable, does not close the experiment, and does not include work started after the call. |
| `reportInvocation({ unitId, toolId, turn? })` | Attributes a tool to the configured in-process invocation window. It is a no-op unless an invocation reference is configured. |
| `reportOutcome({ selectionId, label?, score? })` | Appends a delayed outcome when `evaluation.outcome` is true. At least one non-empty label or finite score is required; repeated reports remain distinct. |

Assignment hashes `JSON.stringify([experimentId, unitId])` with SHA-256 into the ordered integer
weights. Pass an opaque stable unit id; telemetry separately records the first 16 lowercase hex
characters of `SHA-256(unitId)`. `shadow: true` attempts every non-assigned arm in declaration
order. Capacity is per experiment instance and skips instead of queueing. Successful selections
do not await detached shadows; after an assigned selector rejects, fallback may reuse and await an
already-running fallback-arm shadow. Fallback does not run for an empty ranking or a projection
failure.

`transform` runs synchronously after the arm callback and before ranking, result/comparison
telemetry, and return. Arm callbacks own their deadlines and can choose one from `context.role`; a
rejection whose `name` is `TimeoutError` is classified as a timeout. Rank comparison uses ids
(top-1, exact order, overlap, and Jaccard@K), never cross-arm score deltas. Request `k` overrides
the configured value, whose default is 10, without truncating the returned result. See
[ADR 0019](../../../docs/adr/0019-retrieval-experiments.md) for every validation and edge-case
rule.

Experiment telemetry needs an OpenTelemetry `ContextManager` even when no exporter is enabled,
plus both span and Logs processors when exporting. Start descendant model/tool work inside the
arm callback so it inherits the five-field experiment baggage join. See the
[retrieval-experiment telemetry scenario](../../telemetry/README.md#retrieval-experiment-telemetry).

### Migrate a KP-5 vendored stub

Use this mechanical cutover for a host carrying the KP-5 reference implementation:

1. Upgrade to the first published `@ratel-ai/sdk` version that contains
   `experimentalDefineExperiment`, then update the lockfile.
2. Replace the vendored import and public names:

   ```diff
   -import { defineExperiment, type ArmOutcome, type ArmRole, type RankedItem } from "./ratel-ai-sdk.js";
   +import {
   +  experimentalDefineExperiment,
   +  type Experiment,
   +  type ExperimentArmOutcome,
   +  type ExperimentArmRole,
   +  type ExperimentRankedItem,
   +} from "@ratel-ai/sdk";
   ```

   Add the declared arm union as the third generic, for example
   `Experiment<Params, Result, "legacy" | "ratel">` and
   `experimentalDefineExperiment<Params, Result, "legacy" | "ratel">(...)`. Omitting all explicit
   generics and relying on inference is also valid.
3. Rename only the define-time capacity config:
   `shadow: { concurrency: 1 }` becomes `shadowPolicy: { concurrency: 1 }`. Keep the per-call
   `select({ shadow })` boolean.
4. Keep `select({ k })`, `evaluation.attributes`, `fallbackArm`, `transform`, and non-reserved
   caller attributes. Those KP-5 amendments are part of the package contract.
5. Delete the vendored implementation and its private helpers. Do not replace
   `withArmContext`, `hashUnitId`, `compareRankings`, or `drainExperimentForTests` imports; they
   are intentionally not public. Pass the raw unit id and use `experiment.drain()` in shutdown
   and tests.
6. Add a Logs provider/processor beside the existing span processor. Keep saved filters on
   `otel.scope.name = "@ratel-ai/sdk"`, but move experiment event queries from SpanEvents to Logs
   EventRecords and rename the old `ratel.search.results` experiment event to
   `ratel.experiment.results`.
7. Search for stale stub references, then run the host's typecheck, lint, tests, and an in-memory
   trace-plus-log integration:

   ```bash
   rg -n 'ratel-ai-sdk|\bdefineExperiment\b|\bArmOutcome\b|\bArmRole\b|\bRankedItem\b|drainExperimentForTests|withArmContext|hashUnitId|shadow:\s*\{' src
   ```

   Remaining `shadow:` keys should be per-call booleans. Verify one served-plus-shadow request,
   fallback, capacity skip, active host-parent correlation, and shutdown drain.

Account for these contract differences when updating tests and dashboards:

| contract | KP-5 vendored stub | published SDK contract |
| --- | --- | --- |
| Factory and types | `defineExperiment`, `ArmRole`, `ArmOutcome`, `RankedItem` | `experimentalDefineExperiment`, `ExperimentArmRole`, `ExperimentArmOutcome`, `ExperimentRankedItem`; no stable alias |
| Assignment and return | Explicit arm only; result/effective arm/duration | Optional ordered integer `split`; return also includes UUID `selectionId` and original `assignedArm` |
| Lifecycle | Private drain/test helper | Public repeatable `warm()` and snapshot/all-settled `drain()` |
| Invocation and outcome | Invocation could emit without a reference; outcome unsupported | Invocation is a no-op until configured; `evaluation.outcome: true` enables append-only `reportOutcome()` |
| Capacity and fallback | Shadow capacity covered the full detached continuation; fallback broadly dropped peer comparison | Capacity releases when the arm callback settles; other shadows compare against the effective fallback |
| Correlation stamp | Experiment id, arm, role, and unit | Adds the exact `selection_id`; direct arm attributes survive without context, descendant baggage does not |
| Event carrier | Experiment SpanEvents and an invocation span | Seven Logs EventRecords, never SpanEvents; requires a log-record processor |
| Results and agreement | `ratel.search.results`; dynamic `agreement.attr.<key>` fields | `ratel.experiment.results`; structured `agreement.item_attrs` and `agreement.result_attrs` plus served/shadow facts |
| Content boundary | Stub-specific | Ranked ids/scores are always measurements; item attrs are capture-gated; result values are not emitted, only agreement booleans |

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

## Runtime events and catalog snapshots

`ratel()` exposes a merged, asynchronous facts stream over its tool and skill registries. Every
envelope has one session/source stamp and a client ULID. The matching OTel projection carries the
same ID as `ratel.event.id`; OTel remains a parallel observability channel.

```ts
const r = ratel({ events: { sourceId: "checkout-api" } });
const subscription = r.events.subscribe(async (batch) => publish(batch));

// Full serializable state; tool executors and skill bodies are always omitted.
const snapshot = r.catalog.snapshot();

await subscription.flush();
subscription.unsubscribe();
```

Delivery is best effort, bounded, and fail-open: subscriber work never blocks catalog operations.
`flush()` drains work already accepted by this process. The stream includes search, invocation,
catalog churn, upstream/auth, experiment, and observable delivery-loss facts described by
[ADR 0019](../../../docs/adr/0019-runtime-events-lane.md).

Experiments are SDK-owned facts. Pass the runtime stream as the second argument so their OTel and
runtime projections share event IDs:

```ts
const experiment = experimentalDefineExperiment(config, r.events);
```

`catalog.snapshot()` is deliberately separate from the event stream: consumers publish it as an
atomic full replacement under `snapshot.source_id`.

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

**A vendor processor may drop most of it, silently.** Emission and delivery are separate: the provider hands every span to every processor, and each destination then applies its own filter. A stock `new LangfuseSpanProcessor()` keeps a span only if it carries a `gen_ai.*` attribute or comes from a scope it already knows, and `@ratel-ai/sdk` is on neither list — so `execute_tool <tool>` survives while `ratel.search`, `ratel.skill.load`, `ratel.upstream.register`, `ratel.auth.flow`, and `ratel.experiment.arm` do not. Nothing errors; the retrieval spans are simply absent. Widening it is one line of the vendor's own config, keyed on scope — see [`src/telemetry/`](../../telemetry/README.md#emission-is-not-delivery) for the predicate, the full span inventory, and the `ai@7` and Mastra wirings.

Message and tool content is off by default; opt in with the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` env var or `setContentCapture()` (see the [telemetry guide](https://docs.ratel.sh/docs/telemetry) for the capture modes and their privacy implications). The `ratel.*` constants themselves live in [`@ratel-ai/telemetry`](../../telemetry/ts/README.md); this package re-exports only the content-capture gate.

## Package layout

`src/` is the TypeScript surface (including `embedding-artifact.ts` for build/warm helpers), `native/` contains the NAPI binding, `npm/` holds platform packages, and tests live beside their source. From the repository root, run `pnpm --filter @ratel-ai/sdk... build` and `pnpm --filter @ratel-ai/sdk test`.
