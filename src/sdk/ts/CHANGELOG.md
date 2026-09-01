# Changelog

All notable changes to `@ratel-ai/sdk` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- A tool call that reports failure in its result (`isError`) now closes the `execute_tool` span as `ERROR` and emits `invoke_error`; it previously recorded as a success.

## [0.12.0] - 2026-08-21

### Added

- Experimental catalog-definition telemetry. `events.experimentalCatalogDefinitions: true` publishes change-sensitive `catalog_definition` runtime events; OpenTelemetry export additionally requires `RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS=true` and EventRecord content capture. Definitions use canonical hashes, omit unsafe numeric schemas, and preserve critical identity fields when payloads are bounded.
- Experimental definition overrides through runtime attach: `catalog.experimentalAttachDefinitionOverrides({ source })` takes any `ExperimentalDefinitionOverlaySource` — `@ratel-ai/cloud-sdk` is the first one — which performs the initial pull and serialized, ETag-aware refreshes. Responses are runtime-validated with bounded entry fields and named `DefinitionOverlayError` failures; tool, skill, and fact updates roll back together on failure without advancing the ETag or override-ownership latch. Complete overlays restore local values on clear and warn when an override shadows an explicit local value. Calling it is one-way opt-in for the life of the process.
- `experimentalSearchableDescription` on tool, skill, and fact registrations, and on `CatalogRegistration` (ADR-0021): an override for the description component BM25 and embeddings actually rank, so retrieval text can be tuned without changing the `description` the model reads or the `body` it receives. Names stay indexed, skill and fact tags stay indexed, and for a tool the override additionally opts that entry out of schema indexing — `inputSchema` / `outputSchema` stay model-facing but stop contributing tokens. Leave it out and nothing changes: tools still rank description plus schema tokens, skills and facts still rank their authored description. Optional everywhere it appears, so existing registrations compile and rank exactly as before. Experimental, so it may change or be removed without a major-version bump.

## [0.11.0] - 2026-08-17

### Added

- `experimentalRecordBaselineTurn({ query, invoked, invokedSkills })` records a whole baseline turn in one call, for hosts that cannot hold a turn open while it happens — a process-per-request server where the search and the invocation that follows are different requests, on possibly different machines. Reassemble the turn from your own storage, then hand it over whole. Beyond ergonomics: the chained `experimentalBaselineTurn` builder lets you `await` between `invoked()` calls, so concurrent turns can interleave their events in one sink and break the search-then-invoke adjacency the graph pairs on, and splitting one search with three invocations into three recorded turns counts the query three times — inflating the support that scales the boost and gates the flip. The builder is unchanged.
- A `"callback"` trace sink: `{ kind: "callback", sessionId, onEvent }` hands each envelope to `onEvent` as a JSON line rather than writing it, for hosts whose destination the SDK cannot own. Each line is the same wire form `"jsonl"` would have written, field for field, bar the per-record `ts` and `event_id` every envelope-aware sink mints for itself (neither of which replay reads), so lines collected across processes can be joined with newlines and passed straight to `experimentalBuildIntentGraph`. Delivery is **asynchronous** — recording queues the line and returns, and the callback runs on a later turn of the event loop with no ordering guarantee against your own microtasks — and **lossy** under backpressure, per ADR-0007. `sessionId` is a default rather than an identity: replay pairs searches with invokes per session, so a host reassembling turns should restamp each line with an id unique to each concurrent turn.
- Building a graph from a log now defaults to seeding — `origins: baseline`, `provenance: seeded` — since that is what an offline build is. The common call passes nothing. Enabling live learning still defaults to `any` / `live`; changing that would alter existing behavior. Pass `origins: agent`, `provenance: live` to re-derive a graph from a period when Ratel was already serving.
- `origins` accepts `any` / `agent` / `baseline`. `direct` is a valid search origin but not a filter — learning only from searches your own code made means learning from your plumbing.
- Policy options are now a closed set of literal types (`OriginFilterOption`, `ProvenanceOption`) declared by the SDK rather than the native binding's generated `string` fields, so `{ origins: "baselien" }` is a compile error with completion on the legal values. Runtime validation is unchanged for callers without types.
- `experimentalEnableAdaptiveRanking` accepts the same `origins` / `provenance` options as `experimentalBuildIntentGraph`, so what counts as evidence no longer depends on which path produced the graph. Defaults are unchanged, and the policy survives a `setTraceSink`.
- `experimentalBuildIntentGraph(jsonl, options?)` on `ToolCatalog` / `ToolRegistry` builds an intent graph from a JSONL trace log, and `experimentalBaselineTurn(query)` records a turn observed while Ratel serves no retrieval — name the query, name what the agent invoked, `record()`. The turn is buffered, so a turn that fails your own quality gate is never written. Together they are the seed-first path: capture what an agent invokes on its own, build a graph offline, inspect it, then enable ranking. Every distinct query is embedded up front so clusters form densely; the returned graph is detached, so enabling stays explicit. Policy options (`origins` / `provenance`) default to live behavior and reject unknown values.
- `"baseline"` is now a valid `SearchOrigin`, for recording a query while Ratel observes but does not serve retrieval. Unknown origin strings still degrade to `"direct"` rather than failing a search.

### Changed

- Corrected the account of session ids carried in the shipped JSDoc. Recording a turn whole emits its search and its invoke adjacently, and replay tracks one pending query per session in log order, so a shared session id pairs correctly. A per-turn id is what protects you once lines from several producers are merged without preserving each turn's adjacency, not something the pairing depends on in the first place.
- Internal: every path that installs a trace sink now shares one learner-wrapping helper, so a new install site cannot forget that adaptive ranking learns by decorating the sink.

## [0.10.0] - 2026-08-15

### Added

- **Runtime events stream (ADR-0020).** `ratel.events` exposes a first-class, subscribable runtime-events stream: `subscribe(handler)` delivers envelope-v2 batches (`v`, `event_id`, `ts`, `session_id`, `source_id`, `type`, plus trace/span ids when a span is active) for the frozen remotely publishable v1 set exported as `RUNTIME_EVENT_TYPES` — searches, invocation lifecycle, catalog churn, upstream MCP, auth, the experiment lifecycle, and the `events_dropped` backpressure meta-event. Event ids are stable and shared with the OTel projection (`ratel.event.id`), so consumers can cross-link facts with spans. Subscriptions ride a bounded per-subscriber queue (drop-oldest; losses surface as `events_dropped`), while pre-existing local sinks keep recording synchronously and losslessly. `RuntimeEventSubscription` exposes `flush()` and `unsubscribe()` (stops intake; already-queued envelopes drain). `sourceId` defaults to the env-configured OTel `service.name` (`OTEL_SERVICE_NAME`, then `service.name` in `OTEL_RESOURCE_ATTRIBUTES`), then `"ratel"`. Exported: `RuntimeEvents`, `RuntimeEvent`, `RuntimeEventHandler`, `RuntimeEventsOptions`, `RuntimeEventSubscription`, `RuntimeCatalog`, `CatalogSnapshot`, `newRuntimeEventId`, and the `RUNTIME_EVENT_MAX_*` caps. Primary consumer: `@ratel-ai/cloud-sdk`'s `attach()`.

## [0.9.1] - 2026-08-14

### Added

- **Retrieval experiments: A/B and shadow evaluation (ADR-0019).** `experimentalDefineExperiment` declares an experiment over retrieval configurations and assigns each unit deterministically — the same unit id always lands in the same arm, so assignment needs no shared state and survives restarts. Two modes: A/B, where the selected arm's ranking is what the caller receives; and shadow, where a candidate arm is scored alongside the control but never served, bounding the blast radius of an unproven configuration to telemetry only. Both emit the `ratel.experiment.*` vocabulary from `@ratel-ai/telemetry` — per-arm outcomes, result payloads, and served-vs-shadow comparison with top-1 agreement — so arms can be compared without the host writing its own metrics. A failure in a shadow arm is contained and reported as a drop; a failure in the serving path reaches the caller rather than being silently swallowed. `Experiment`, `ExperimentConfig`, `ExperimentSelection`, `ExperimentSplit`, `ExperimentArmRole`, `ExperimentArmOutcome`, `ExperimentRankedItem`, `ExperimentReportedOutcome`, `ExperimentEvaluationReference`, and `ExperimentSelectOptions` are exported. Shipped behind an `experimental` prefix — the API may change until it graduates.
- `RatelAdapter.experimentalExposePassthrough`, an optional experimental exposure-time hook for framework-native tools. Its core-owned invocation wrapper preserves scalar, promise, and `AsyncIterable` return shapes while emitting the standard `execute_tool` span and local invoke events; adapters can instrument client-executed passthroughs without mutating caller-owned tools.

## [0.9.0] - 2026-08-13

### Added

- **Experimental build-time embedding artifacts (ADR-0018).** `experimentalBuildEmbeddingArtifact` builds a mixed Tool+Skill RAT1 (Tool and Skill halves merged internally; no public merge API). Hosts own artifact persistence; the core artifact APIs accept/return bytes and perform no artifact filesystem I/O. Catalogs accept `experimentalEmbeddingArtifact: { path } | { bytes }` (default `onMiss: "error"`) and warm on `register` / `replaceAll` for any search method before eager document embedding on semantic/hybrid. `ToolRegistry` / `SkillRegistry` expose `experimentalBuildEmbeddingArtifact` and `experimentalWarmEmbeddingsFromArtifact`. Failures: `ArtifactWarmError` (`.code`, `.missing` for `"Incomplete"`), `ArtifactError`, and `IncompatibleMergeError` (from incompatible Tool/Skill halves during mixed build — not a public merge function). Shipped behind an `experimental` prefix (the API may change until it graduates).

## [0.8.0] - 2026-08-11

### Added

- **⚠️ Experimental: facts and grounding (ADR-0017).** Constant content the agent should always have on hand, registered and monitored like skills, reaching the model on its own host-driven path. Register with `experimental.FactCatalog` (or `r.facts`), then pick one of two injection modes per turn: `r.ground(query, transcript)` persists into your stored history behind a re-injection freshness gate, and `r.groundSnapshot(query)` is the stateless per-call twin for one-shot calls or hosts that keep synthetic content out of their history. `Pin.Always` facts ride every applicable turn; `Pin.Retrieved` (the default) surface only when the turn's query ranks them in, budgeted by `RatelConfig.factsTopK` (default 3).
- The freshness gate re-injects a fact only when its body is **absent** from the window (`never` / `evicted`) or has been **edited** (`mutated`) — never merely because a turn elapsed. Presence is the fact's own body text scanned per message: no markers, no tags, no extra tokens, and a stable transcript prefix that improves prompt-cache hit rates rather than churning them. `experimental.planInjection` is the pure, framework-agnostic decision function behind it. Every decision is traced (`fact_inject` with its reason, `fact_inject_skip`, `fact_snapshot`), so the skip rate is measurable.
- The whole surface is quarantined behind the `experimental` namespace (`import { experimental } from "@ratel-ai/sdk"`), never the root export, and constructing a catalog logs a one-time warning (silence with `RATEL_EXPERIMENTAL_SILENCE=1`). The `ratel()` touchpoints that can't move off the stable object — `r.facts`, `r.ground`, `r.groundSnapshot`, `RatelConfig.factsTopK` — are marked experimental in their docs, and `r.facts` is built lazily and non-enumerably, so a host that never touches facts never constructs one and never sees the warning.

### Changed

- Nothing on the stable path. `recall()`, `modelTools()`, and the model-facing `search_capabilities` tool are byte-identical to 0.7.0 — the tool result carries no `facts` key, so the model's contract is unchanged and facts are never discovered by the model calling a tool.

## [0.7.0] - 2026-08-07

> **On npm's `rc` channel?** The `0.6.1-rc.*` and `0.7.0-rc.0` prereleases on npm were cut
> off experimental branches that never landed, so they are **not** ancestors of this
> release. `0.7.0` does not contain their seed-first intent-graph APIs
> (`experimentalBuildIntentGraph`, `experimentalBaselineTurn`, the `"callback"` trace sink)
> or `experimentalExposePassthrough`; moving from `rc` to `latest` drops them. Upgrades
> from `0.6.0` are unaffected — for them this release is purely additive.

### Added

- **Whole-catalog skill reload: `SkillCatalog.replaceAll` (ADR-0015).** For a source that fetches the full skill catalog rather than individual changes — the batch *is* the catalog, so ids missing from it are removed, including ones registered in-process (a host mixing local and remote skills composes the batch itself). It mutates in place, so the one `SkillCatalog` behind `r.skills`, every adapted view, and every capability tool from `modelTools()` all see the reload. Two-phase like `register`: the corpus swap commits synchronously and the embedding pass is the awaitable, so a reload whose embedding pass fails still reports what the swap changed — `replaceAll` returns a `PendingReplace` carrying the `ReplaceOutcome` counts (added / removed / updated / unchanged) and awaitable separately. On embedding failure the new corpus is live and BM25 ranks it while semantic search reports `EmbeddingsNotBuilt` until a later pass succeeds; a reload started while a dense operation owns the registry is rejected rather than blended. Reloading an unchanged catalog costs zero embeddings, and `advertiseSkills` already pins the `search_capabilities` description, so a reload can't bust the prompt cache. `PendingReplace` and `ReplaceOutcome` are exported.
- **`registerMcpServer` follows every `tools/list` page.** Ingestion previously read only the first page, silently dropping every tool past it on a paginated server. It now walks `nextCursor` to exhaustion (treating `""` as a valid cursor, per MCP — only an absent `nextCursor` ends pagination) across both `mcp` 1.x and 2.x clients, capped at 64 pages. `McpToolsListError` (with a stable `code` of `"RepeatedCursor"` or `"PaginationExceeded"`) and the `McpToolsListErrorCode` type are exported, so a cursor loop or a runaway server is a typed failure rather than a hang. `McpServerHandle.toolIds` now spans all pages in upstream list order.

### Fixed

- `registerMcpServer` closes the MCP client when connecting, listing, or catalog registration throws. It previously leaked the live connection on any failure after `connect`.

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
