# Changelog

All notable changes to `@ratel-ai/vercel-ai-sdk` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.0] - 2026-08-21

### Added

- Preserve a tool's experimental searchable-description projection while ingesting Vercel AI SDK definitions, allowing protocol-v2 retrieval text to differ from the model-facing description.

## [0.4.0] - 2026-08-17

### Added

- `RatelOtelIntegration` copies the active retrieval experiment's five-field baggage stamp onto every AI SDK `gen_ai.*` span, giving generic and vendor destinations the exact experiment/selection join plus arm, role, and unit context.
- Client-executed passthrough tools now enter Ratel's `execute_tool` funnel through a descriptor-preserving exposure wrapper. Native lifecycle hooks, metadata, execution options, and scalar/promise/stream return shapes remain intact; inherited accessors and methods run against the original instance, so a class-backed tool reading private (`#`) state works through the wrapper. Provider/host-executed tools without `execute` remain unobservable.

### Fixed

- The published `@ratel-ai/sdk` peer is now the floor range `>=0.11.0 <1.0.0` instead of a caret of the in-repo SDK version.

## [0.3.0] - 2026-07-28

### Changed

- The `@ratel-ai/sdk` peer range moves to `^0.6.0`, tracking the SDK release that adds experimental adaptive usage ranking. The adapter's own API is unchanged, and 0.6.0 is additive over 0.5.3 — the range moves because at 0.x a caret does not span minors, so `^0.5.3` excludes `0.6.x`. npm resolves peers automatically, which makes that mismatch an install-time `ERESOLVE` rather than a warning: `@ratel-ai/vercel-ai-sdk@0.2.0` cannot be installed alongside `@ratel-ai/sdk@0.6.x`, and this release cannot be installed alongside `@ratel-ai/sdk@0.5.x`. Staying on the 0.5 SDK means staying on adapter 0.2.0.

## [0.2.0] - 2026-07-26

### Added

- `RatelOtelIntegration`, an `ai@7` telemetry integration, at the new `@ratel-ai/vercel-ai-sdk/otel` entrypoint. It embeds `@ai-sdk/otel`'s `OpenTelemetry` emitter as a private delegate and stamps Ratel's `ratel.origin` overlay on every span through that emitter's `enrichSpan` hook, so hosts get the AI SDK's standard `gen_ai.*` spans plus the `ratel.*` overlay. It only _creates_ spans, onto a provider the host owns — it never registers a provider and never exports, so any processor already on that provider (Langfuse, a generic OTLP exporter, anything else) receives them. Its options are `@ai-sdk/otel`'s `OpenTelemetryOptions` plus `origin`, which picks the `ratel.origin` value and defaults to `agent` — right for the tool-loop spans an agent synthesizes, wrong for host-driven `embed` / `embedMany` / `rerank`, so the value is the host's to set. An `enrichSpan` of the host's own is called too and its attributes merged _under_ Ratel's: every host attribute lands except `ratel.origin` itself, which the overlay keeps. A host hook that throws costs the host its own attributes for that span and nothing more — the emitter's own guard discards the whole return value, so the integration guards the host call separately to keep `ratel.origin` unconditional. `Origin` is re-exported from the entrypoint, since `@ratel-ai/telemetry` is the adapter's dependency rather than the host's. Register exactly one emitting integration: this one, Langfuse's, and the bare `OpenTelemetry` all embed the same emitter, so two would duplicate every `gen_ai.*` span. On `ai@5`/`ai@6` there is no integration seam — pass `experimental_telemetry: { isEnabled: true }` per call instead.
- `@ai-sdk/otel` and `@opentelemetry/api` as **optional** peers, needed only by `./otel`. Optional and off-root are both load-bearing, not stylistic: `@ai-sdk/otel` depends on an exact `ai@7`, so a required peer or a root re-export drags a second `ai` into an `ai@5`/`ai@6` host's type graph, where the two copies redeclare `AI_SDK_DEFAULT_PROVIDER` and break the host's build (TS2403) without it ever importing the integration. The compat matrix now asserts a packed consumer resolves no `@ai-sdk/otel`; its v7 rows typecheck `RatelOtelIntegration` against the real `ai@7` `Telemetry` interface and then actually import and construct it.

### Changed

- The adapter takes its first runtime dependency, `@ratel-ai/telemetry` — the zero-dependency `ratel.*` constants package that `dist/otel.js` imports unconditionally. A peer could go unmet or unhoisted and break that import at load time with `ERR_MODULE_NOT_FOUND`, which no typecheck would catch: the `ratel.*` constants never reach `otel.d.ts`. It carries no `ai` of its own, so none of the duplicate-`ai` hazard that keeps the other two peers optional applies.
- Dev-pinned `ai` moves to `7.0.37` (matching `@ai-sdk/otel@1.0.37`, which depends on that exact release), and the compat matrix's latest v7 row moves with it. The peer range is unchanged.
- The `@ratel-ai/sdk` peer range moves to `^0.5.3`, the release that removed `configureTelemetry()` and moved telemetry provider ownership to the host. The adapter's own API is unchanged; the floor moves because the `./otel` entrypoint assumes a host-owned provider.

## [0.1.0] - 2026-07-23

### Added

- Initial release: `@ratel-ai/vercel-ai-sdk`, the Vercel AI SDK adapter for Ratel. `ratel(config).adaptTo(aiSdk())` speaks the AI SDK's native `Tool` / `ModelMessage` shapes through the framework-adapter SPI (ADR-0013): the `ingest` / `expose` / `recallMessages` codecs plus two per-turn recall idioms — `appendRecall` (mutate-and-suffix-append, cache-preserving) and `prepareStep` (step-0 fresh-array override for `generateText` / `streamText` / `ToolLoopAgent`). Peers `@ratel-ai/sdk` with zero runtime dependencies; passes the `@ratel-ai/sdk/testkit` conformance battery (21 cases). Extracted from the live-verified `bratislava` prototype. (Supersedes the pre-release `@ratel-ai/ai-sdk-adapter@0.1.0-rc.1`, published under the old name before the rename.)
- AI SDK v5, v6, and v7 support: the `ai` peer range is `^5.0.0 || ^6.0.0 || ^7.0.0` (`ai@4` predates the v5 tool/message reshape). One shared code path absorbs the per-major differences: provider-defined tools pass through under both discriminators (`provider-defined` in `ai@5`, `provider` in `ai@6`/`ai@7`), catalog executors receive both context spellings (`experimental_context` and `context`), and a Promise-like JSON Schema is rejected synchronously before the registration batch commits. CI pins an exact-version compatibility matrix (`ai@5.0.0`, `5.0.217`, `6.0.0`, `6.0.232`, `7.0.0`, `7.0.33`), each of which builds, typechecks, tests, packs, and typechecks a packed-tarball consumer. Narrowing the supported-majors peer range is a breaking change of the adapter (see the README's Compatibility section).
- Live execution-context forwarding: when the model runs one of your tools through `invoke_tool`, the adapter forwards the AI SDK's complete live execution options (`toolCallId`, `messages`, `abortSignal`, and the version's `experimental_context` / `context`) unchanged to the tool, instead of a fabricated options object. The whole context rides through the catalog as an opaque value under a private, package-stable symbol tag (ADR-0013), so a sibling framework view over the same catalog can never have its context mistaken for AI SDK options. The driver-level `r.tools.catalog.invoke(id, args)` escape hatch — which has no AI SDK invocation to thread — keeps the fabricated fallback. Requires `@ratel-ai/sdk@^0.5.1`.

### Fixed

- `prepareStep` preserves the injected recall pair across the steps of one `generateText` / `streamText` / `ToolLoopAgent` run on `ai@5`/`ai@6`, which rebuild the prompt per step (the pair is reinserted at its original boundary from per-run state); on `ai@7`, which carries the step-0 override forward itself, the duplicate check makes reinsertion a no-op.
- Preserve AI SDK tool semantics through the capability funnel: nested input schemas now validate and apply defaults/transforms, streamed executors retain preliminary/final outputs, and target exceptions surface as native `tool-error` results. Tools with AI SDK-only lifecycle or model metadata stay eagerly exposed in their original shape, preserving approval, per-tool context routing, input hooks, and `toModelOutput`.
