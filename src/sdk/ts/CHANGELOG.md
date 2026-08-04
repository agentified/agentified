# Changelog

All notable changes to `@ratel-ai/sdk` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `experimentalEnableAdaptiveRanking` accepts the same `origins` / `confirmation` / `provenance` options as `experimentalInitializeIntentGraph`, so what counts as evidence no longer depends on which path produced the graph. Defaults are unchanged, and the policy survives a `setTraceSink`.
- `experimentalInitializeIntentGraph(jsonl, options?)` on `ToolCatalog` / `ToolRegistry` builds an intent graph from a JSONL trace log, and `experimentalRecordBaselineQuery(query)` records a turn observed while Ratel serves no retrieval. Together they are the seed-first path: capture what an agent invokes on its own, build a graph offline, inspect it, then enable ranking. Every distinct query is embedded up front so clusters form densely; the returned graph is detached, so enabling stays explicit. Policy options (`origins` / `confirmation` / `provenance`) default to live behavior and reject unknown values.
- `"baseline"` is now a valid `SearchOrigin`, for recording a query while Ratel observes but does not serve retrieval. Unknown origin strings still degrade to `"direct"` rather than failing a search.

## [0.6.0] - 2026-07-28

> **Coming from `0.6.0-rc.0`?** That RC was tagged off a branch that predated 0.5.3, so it
> still shipped `configureTelemetry()` and pinned `@ratel-ai/telemetry@^0.1.2`. Read the
> 0.5.3 entry below as part of this upgrade — its breaking changes land for you here.
> Upgrades from 0.5.3 are unaffected: for them this release is purely additive.

### Added

- **Experimental adaptive usage ranking (ADR-0014).** `IntentGraph` plus `experimentalEnableAdaptiveRanking`, `experimentalRebuildIntentGraph`, `experimentalDisableAdaptiveRanking`, and `experimentalAdaptiveRankingStatus` on `ToolCatalog` / `SkillCatalog` (and the underlying registries). The catalog learns from each search-then-invoke and boosts future rankings; persist and reload the learning via `IntentGraph.toJson` / `fromJson`, and track writes via `graph.rev`. Shipped behind an `experimental` prefix — the API may change until it graduates.
- `rank` and `fused` on search hits: order on `rank`, and branch on `fused` to know whether the usage arm changed the ranking.
- Opt-in recovery after an embedding-model change: `experimentalEnableAdaptiveRanking(graph, { rebuildOnModelChange: true })` re-embeds a paused graph on the next dense search. Default off; explicit `experimentalRebuildIntentGraph()` otherwise. `experimentalAdaptiveRankingStatus` reports the paused/active state and the mismatched model detail.

## [0.5.3] - 2026-07-26

> **Read this before upgrading from 0.5.2.** Despite the patch version, this release
> **removes** `configureTelemetry()`. The number is patch because 0.5.3 was already cut
> in the repo before the removal landed and the two shipped together; treat the upgrade
> as breaking if you called that function. Everything else is additive.

### Added

- Content capture emits structured OpenTelemetry Logs `EventRecord`s. `EVENT_ONLY` and `SPAN_AND_EVENT` now produce real records (in 0.5.2 the event half of those modes emitted nothing), carrying `gen_ai.system_instructions`, `gen_ai.input_messages`, `gen_ai.output_messages`, and `ratel.tool.execution_details`. Tool results stay out of inference-output messages, which are reserved for model output with a `finish_reason`.

### Changed

- **BREAKING:** the telemetry bootstrap `configureTelemetry()` is gone, with its `TelemetryHandle`, `InitOptions`, and `ConfigureTelemetryOptions` types. The SDK ships no OpenTelemetry provider wiring — it emits `ratel.*`/`gen_ai.*` to whatever providers the host has registered, and the host owns the provider (`new NodeSDK({ spanProcessors })`), its composition, and its flush/shutdown. The content-capture gate (`ContentCapture`, `setContentCapture`, `clearContentCapture`) and the emitted spans and `EventRecord`s are unchanged. To migrate, build the provider yourself and register it before importing the SDK; see the [package README](README.md).
- **BREAKING:** the optional `@ratel-ai/telemetry-otlp` peer dependency is gone, along with the `require()`-an-ES-module machinery that loaded it. `@ratel-ai/telemetry-otlp` is discontinued, and 0.4.0's `ratelSpanProcessor` recipe with it: build the OTLP exporter onto the provider you register.
- `engines.node` is now `>=20.6.0` (0.5.2 declared `>=20.0.0`), matching the `@ratel-ai/telemetry` runtime dependency.
- New runtime dependency `@opentelemetry/api-logs` (`^0.220.0`), the API-only Logs surface the `EventRecord` emission needs. Like `@opentelemetry/api` it is inert until the host registers a provider.
- The `@ratel-ai/telemetry` dependency floor moves to `^0.2.0`, the vocabulary version that defines the `EventRecord` contract.

## [0.5.2] - 2026-07-24

### Changed

- Ship the native addon built with symbol stripping and thin LTO (`[profile.release]`), shrinking the `.node` binary ~26% (9.90 MB → 7.36 MB on the sdk-ts cdylib). No API or behavior change.

## [0.5.1] - 2026-07-23

### Added

- **Framework-adapter SPI + `ratel()` factory (ADR-0013).** `ratel(config)` is a standalone, framework-free core: `r.tools` is a handle over its one shared `ToolCatalog` (register native `ExecutableTool`s any time — also after exposure, since the capability tools search the live catalog), `r.skills` the shared `SkillCatalog`, `modelTools()` returns the three capability tools (always all three, so the set never depends on registration order), and `recall(query)` is an async pure query resolving to the canonical `search_capabilities` result or `null`. `adaptTo(adapter)` layers a framework-shaped view over the same state: `tools.register(...)` ingests framework tools (first registration of an id wins across views; provider-run tools pass through per view), `modelTools()` returns the model-facing set in framework shape, `recall(query)` resolves to the synthetic message pair with a call id from the core's private counter (never a transcript position). `RatelConfig` forwards `method` and `embedding` to both catalogs, so a `"semantic"`/`"hybrid"` core is fully configurable through the factory (ADR-0012's models). `r.tools.register(...)` is async: it validates synchronously (a missing `execute`, a reserved id, or a framework-shaped tool throws at the call site) and returns a promise that resolves once the batch is indexed and — on a semantic/hybrid core — embedded, rejecting if embedding fails, so errors surface at registration (`await r.tools.register(...)` before searching a dense core). The handle's `search(...)` is synchronous BM25-only (a dense method points to `searchAsync` instead of leaking the native error); `searchAsync(...)` ranks any method off the event loop. Types are inferred from the adapter (`AdaptedRatel<A>`), so app code needs no casts. A `RatelAdapter` is three codecs — `ingest` (framework tool → catalog registration, or `"passthrough"`), `expose` (capability tool → framework tool), `recallMessages` (synthetic `search_capabilities` pair) — plus an optional `extend` for framework idioms; the framework packages (`@ratel-ai/vercel-ai-sdk`, `@ratel-ai/mastra`) ship separately. Guards are core-owned: reserved capability-tool ids throw on registration, recall top-K is capped at 50 (invalid values fall back to the default 5), and a framework-shaped tool on the native path throws an actionable install-the-adapter error, probing known frameworks via `isPeerInstalled` (message only). The existing piecemeal API (`ToolCatalog`, capability-tool builders) is unchanged, except one additive option: `SearchCapabilitiesOptions.advertiseSkills` pins the skills clause of the `search_capabilities` description on or off (the size-gated default is untouched); `modelTools()` uses it so the exposed payload is byte-identical whether skills register before or after it is taken.
- `runCapabilitiesSearch(toolCatalog, query, opts)` — the exported single source of truth for the `search_capabilities` result shape, shared by `searchCapabilitiesTool` (origin `agent`) and the host-driven recall path (origin `direct`). Async, matching the catalog's `searchAsync` retrieval. `JSONSchema7` is re-exported as the SDK's public JSON-Schema spelling so adapters type their registrations without casts.
- **Adapter conformance testkit (`@ratel-ai/sdk/testkit`).** A runner-agnostic battery every framework adapter must pass, pinning the whole SPI contract: ingest/expose round-trip, the reserved-id guard, recall top-K clamp, passthrough semantics, and recall-pair shape (validated through framework-supplied hooks). `adapterConformanceCases(options)` returns named cases (assertions via `node:assert`, so no test-runner dependency leaks into shipped types); `describeAdapterConformance(options, { describe, it })` registers them as first-class tests under Vitest/Jest/`node:test`. Ships `referenceAdapter`/`referenceConformanceOptions` as the worked example a real adapter's options copy. Reached via a new `exports` map whose `.` entry is byte-identical to today's `main`/`types`; the map also seals hypothetical deep imports of package internals.
- Framework adapters can attach an `InputValidator` to `CatalogRegistration` / `ExecutableTool`; the shared `ToolCatalog` keeps that parser authoritative across adapted views and native hot-swaps. `validateInput()` exposes the live parser, and `invokeValidatedRaw()` preserves a prevalidated executor's immediate scalar, promise, or `AsyncIterable` shape. `invokeRaw()` provides the same preservation after synchronous validation, while `invoke()` remains the Promise-based public convenience path.
- `invoke_tool` target failures remain structured for generic hosts and now carry their original cause under a non-enumerable symbol for framework adapters.

### Changed

- Add an optional opaque invocation context to `Executor`, `CatalogRegistration.execute`, and `ToolCatalog.invoke`, and forward it unchanged through `invokeToolTool`. Framework adapters can now preserve request-scoped execution state without the core inspecting, storing, or tracing it; existing one-argument executors retain their source compatibility and runtime call arity.

### Fixed

- `invoke_tool` no longer collapses streamed tool results into an opaque object; local trace events and `execute_tool` spans now settle when iteration completes, is cancelled, or fails, including cancellation-cleanup failures.

## [0.5.0] - 2026-07-20

### Added

- `register()` accepts a single item or an array across tool/skill registries and catalogs.
- Configurable default, HuggingFace, local Candle, Ollama, and OpenAI-compatible
  endpoint embedding sources, with public `EmbeddingSpec` and
  `EmbeddingModelConfig` types.
- Typed embedding errors: `EmbedderError` (with a stable `code`) and its `DimensionMismatchError` subclass are thrown from `register()`/`searchAsync()` on a semantic/hybrid catalog, so callers can branch on `instanceof`/`code` instead of matching message text — parity with the Python SDK. Invalid embedding config still throws at construction.

### Changed

- **BREAKING:** `register()` now returns a promise and accepts a single tool/skill **or an array of them**, and folds embedding in: on a `"semantic"`/`"hybrid"` catalog it embeds the batch on a libuv worker (never blocking the event loop), so embedding errors (model load / endpoint / auth / dimension) surface from `await register(...)`. A `"bm25"` catalog registers metadata only and never loads a model. `search()` stays synchronous BM25-only; `searchAsync()` covers BM25/semantic/hybrid. There is **no** `registerMany()`, `buildEmbeddings()`, or `rebuildEmbeddings()` — `register()` embeds, and recovery from a model/dimension change is to construct a new catalog and re-register.
- Capability tools await async retrieval; MCP ingestion embeds ingested tools during `register`.
- Embedding configuration is validated and retained on BM25-default catalogs for later async semantic/hybrid overrides; source unions are mutually exclusive.

### Fixed

- A `"semantic"`/`"hybrid"` `searchAsync()` whose corpus was never embedded (the signature of a forgotten `await register(...)`) now reports an actionable "did you await register()?" hint, not just the bare "embeddings not computed" message.

## [0.4.1] - 2026-07-10

### Added

- `configureTelemetry` opts into message/tool content capture programmatically: `captureContent` sets the exact `ContentCapture` mode (validated like the env var — case-insensitive, legacy boolean forms accepted — throwing a `TypeError` on garbage before any exporter is wired), and `includeSpanAndEvents` is boolean sugar (`true` → `SPAN_AND_EVENT`, `false` → `NO_CONTENT`). `captureContent` wins over `includeSpanAndEvents`; when neither is provided, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` keeps ruling (a provided option beats the env var, as in OTel code-over-env precedence). The handle's `shutdown()` restores env-driven behavior via a generation-scoped clear (`clearContentCapture`), so a stale handle shutting down late never clobbers an override a newer `configureTelemetry` installed. `ContentCapture`, `setContentCapture`, and `clearContentCapture` are re-exported from `@ratel-ai/sdk` (new `ConfigureTelemetryOptions` type).

## [0.4.0] - 2026-07-07

### Added

- **OpenTelemetry emission.** The SDK now opens an OTel span at each funnel boundary — `execute_tool` (`gen_ai.operation.name`, `gen_ai.tool.name`, `ratel.tool.args_size_bytes`, plus `ratel.upstream.server` for MCP-proxied tools), `ratel.search` (target, `top_k`, origin, `hit_count`), `ratel.skill.load`, `ratel.upstream.register`, and `ratel.auth.flow` — alongside the existing local `recordEvent` stream, which is unchanged. Emission is transparent and free by default: spans flow to whatever OpenTelemetry provider is registered and are a no-op until one is, so a host already running OTel sees Ratel's funnel on its traces with no setup. Built on `@opentelemetry/api` + the OTel-free `@ratel-ai/telemetry` vocabulary, so the base install stays OTel-SDK-free.
- `configureTelemetry({ apiKey })` convenience wiring (with `TelemetryHandle` / `InitOptions`): lazily loads the optional `@ratel-ai/telemetry-otlp` peer to ship the SDK's spans to Ratel Cloud (or any OTLP endpoint). Hosts already running OpenTelemetry skip it and add `ratelSpanProcessor` from `@ratel-ai/telemetry-otlp` instead.
- Message/tool content (`ratel.search.query`, `gen_ai.tool.call.arguments` / `.result`) rides span attributes only when `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` selects a span mode (`SPAN_ONLY` / `SPAN_AND_EVENT`); default off. `ratel.tool.args_size_bytes` is measured in UTF-8 bytes.

## [0.3.0] - 2026-07-06

### Added

- `ToolCatalog` / `SkillCatalog` accept a default `method` (`"bm25"` | `"semantic"` | `"hybrid"`) and `search(query, topK, origin?, method?)` takes a per-call override. `"bm25"` (default) is unchanged and model-free; `"semantic"` / `"hybrid"` load a local embedding model and throw if it fails to load. Exposed via the native `searchWithMethod` binding and the `SearchMethod` type.
- A `"semantic"`/`"hybrid"` catalog embeds each tool/skill **eagerly at `register`** (incrementally), so searches never pay the corpus-embedding cost. New `catalog.buildEmbeddings()` pre-computes embeddings on demand (e.g. after a bulk register). BM25 catalogs do neither. A semantic/hybrid search on a catalog with no embeddings built throws (embeddings not computed) rather than embedding during the search.

## [0.2.1-rc.1] - 2026-07-04

### Changed

- First release cut under the per-package release scheme (ADR-0008): `@ratel-ai/sdk` (loader + platform packages) now versions and ships independently, tagged `sdk-js-v*`. No API changes since 0.2.0.

## [0.2.0] - 2026-06-16

### Changed

- **BREAKING:** the discovery gateway tool is renamed `search_tools` → `search_capabilities`. It now returns two independently-ranked, separately-budgeted buckets — `{ tools, skills }` — so a relevant skill is never crowded out by matching tools. New surface: `searchCapabilitiesTool`/`SEARCH_CAPABILITIES_ID`. The old `searchToolsTool`/`SEARCH_TOOLS_ID` are kept as deprecated aliases (see _Deprecated_), so `0.1.x` code keeps working after upgrading.

### Added

- First-class **skills**: `SkillCatalog`, `getSkillContentTool` (`get_skill_content`), and `Skill`/`SkillHit`/`SkillRegistry`. Skills are reusable playbooks ranked by a separate BM25 corpus and loaded on demand.
- Skill–tool coupling: a `Skill` can declare a `tools` list, and `search_capabilities` pulls a matched skill's declared tools into the `tools` bucket — additively (beyond `topKTools`) and deduped against query hits — so the agent gets the playbook and the tools it needs in one turn.

### Deprecated

- `searchToolsTool`, `SEARCH_TOOLS_ID`, and the `SearchToolHit`/`SearchToolsGroup`/`SearchToolsResult`/`SearchToolsToolOptions` types. They retain their pre-0.2.0 behaviour — a tools-only `{ groups }` result and the `search_tools` id — so upgrading from `0.1.x` does not break existing callers. Migrate to `searchCapabilitiesTool`; the aliases will be removed in a future release.

### Fixed

- Gateway error payloads (`invoke_tool`, `get_skill_content`) carry `isError: true`, so a host can flag a failed call rather than read it as content.
- `invoke_tool` rejects a non-object `args` instead of forwarding stray top-level keys.
- `search_capabilities` validates `topKTools`/`topKSkills` (declared `integer`, positive): `0`, negative, and fractional values fall back to the default instead of returning zero results — or, via a negative wrapping to `u32` in the native layer, an unbounded set. TypeScript and Python behave identically.
- `search_capabilities` advertises the `skills` bucket and `get_skill_content` in its description only when a non-empty skill catalog is wired in.
- `Skill.tags` and `Skill.body` are optional (default `[]`/`""`), matching the Python SDK — a minimal `{ id, name, description }` skill is valid.

## [0.1.6] - 2026-06-10

### Fixed

- TypeScript typehint for JSON-schema tool input/output ([#54](https://github.com/ratel-ai/ratel/pull/54)).

## [0.1.5] - 2026-05-10

### Added

- Initial release on the v1 (revamp) line. TypeScript SDK over the Rust core: BM25 tool retrieval, MCP ingestion, framework-neutral gateway tools. See the [package README](README.md) for the full surface.
- `ToolCatalog` accepts `{ trace }` config in its constructor — `noop` (default), `memory`, or `jsonl`. Captured events flow through the Rust core sink ([ADR-0007](../../../docs/adr/0007-telemetry-two-streams.md)). New `recordEvent`, `drainTraceEvents`, and an optional third `origin` argument on `search`.
- `searchToolsTool` emits `gateway_search` with `origin: "agent"`. `invokeToolTool` emits `gateway_invoke` on success and `gateway_error` for unknown ids, `needs_auth`, and underlying throws.
- `ToolCatalog.invoke` emits `invoke_start` / `invoke_end` / `invoke_error` around the executor with `args_size_bytes` and `took_ms`.
- `registerMcpServer` emits `upstream_register` on connect and `upstream_invoke` / `upstream_error` per upstream call. New `searchWithOrigin` and trace plumbing on the underlying NAPI `ToolRegistry`. `SearchOrigin` is now `"direct" | "agent"` (was `"user" | "agent"`); the union is exported from the package entry point.

### Changed

- Rewrote `search_tools` description to nudge agents toward discovery before falling back to broad listing.
