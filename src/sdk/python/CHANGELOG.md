# Changelog

All notable changes to `ratel-ai` (the Python SDK) are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Policy keywords are now `Literal` types (`OriginFilterOption`, `ConfirmationOption`, `ProvenanceOption`) rather than `str`, so `origins="baselien"` is a type error rather than a runtime one. Runtime validation is unchanged for callers without a type checker.
- `experimental_enable_adaptive_ranking` accepts the same `origins` / `confirmation` / `provenance` keywords as `experimental_build_intent_graph`, so what counts as evidence no longer depends on which path produced the graph. Defaults are unchanged, and the policy survives a `set_trace_sink`.
- `experimental_build_intent_graph(jsonl, *, origins, confirmation, provenance)` on `ToolCatalog` / `ToolRegistry` builds an intent graph from a JSONL trace log, and `experimental_record_baseline_query(query)` records a turn observed while Ratel serves no retrieval. Together they are the seed-first path: capture what an agent invokes on its own, build a graph offline, inspect it, then enable ranking. Every distinct query is embedded up front so clusters form densely; the returned graph is detached, so enabling stays explicit. Policy keywords default to live behavior and reject unknown values.
- `"baseline"` is now a valid `SearchOrigin`, for recording a query while Ratel observes but does not serve retrieval. Unknown origin strings still degrade to `"direct"` rather than failing a search.

## [0.6.0] - 2026-07-28

> **Coming from `0.6.0rc0`?** That RC was tagged off a branch that predated 0.5.2, so it
> shipped neither the `EventRecord` content-capture fixes nor the base `ratel-ai-telemetry`
> runtime dependency. Read the 0.5.2 entry below as part of this upgrade. Upgrades from
> 0.5.2 are unaffected: for them this release is purely additive.

### Added

- **Experimental adaptive usage ranking (ADR-0014).** `IntentGraph` plus `experimental_enable_adaptive_ranking`, `experimental_rebuild_intent_graph`, `experimental_disable_adaptive_ranking`, and `experimental_adaptive_ranking_status` on `ToolCatalog` / `SkillCatalog`. The catalog learns from each search-then-invoke and boosts future rankings; persist and reload via `IntentGraph.to_json` / `from_json`, and track writes via `graph.rev`. Shipped behind an `experimental_` prefix — the API may change until it graduates.
- `rank` and `fused` on search hits: order on `rank`, and branch on `fused` to know whether the usage arm changed the ranking.
- Opt-in recovery after an embedding-model change: `experimental_enable_adaptive_ranking(graph, rebuild_on_model_change=True)` re-embeds a paused graph on the next dense search. Default off; explicit `experimental_rebuild_intent_graph()` otherwise. `experimental_adaptive_ranking_status` returns an `AdaptiveRankingStatus` that carries the paused/active state and the mismatched-model detail.

## [0.5.2] - 2026-07-26

### Changed

- `configure_telemetry(endpoint=...)` now defaults to `RATEL_OTLP_ENDPOINT`, falling back to the superseded `RATEL_URL` with a `DeprecationWarning`. `RATEL_URL` also selects the catalog source (ADR-0003), so it no longer doubles as the OTLP destination. Resolution lives in `ratel-ai-telemetry>=0.1.3`; nothing breaks here, a `RATEL_URL`-only install keeps exporting to the same endpoint.

### Fixed

- Emit content capture as structured OpenTelemetry Logs `EventRecord`s, keep tool results out of inference-output messages, and export those records through `configure_telemetry()`.
- Serialize real MCP `CallToolResult` values into telemetry instead of recording an empty string.
- Preserve heterogeneous JSON arrays losslessly despite OpenTelemetry Python 1.41's
  homogeneous-array limitation.
- Ship and require the OTel-free telemetry vocabulary version that defines the EventRecord
  contract, so base installs emit into host-owned OpenTelemetry providers.

## [0.5.1] - 2026-07-24

### Changed

- Ship the native extension built with symbol stripping and thin LTO (`[profile.release]`), materially shrinking the wheel's compiled binary. No API or behavior change.

## [0.5.0] - 2026-07-20

### Added

- `register()` accepts a single item or an iterable across tool/skill registries and catalogs.
- Configurable default, HuggingFace, local Candle, Ollama, and OpenAI-compatible
  endpoint embedding sources, with public `EmbeddingSpec`,
  `EmbeddingModelConfig`, and source-specific `TypedDict` variants.

### Changed

- **BREAKING:** `register()` now accepts a single tool/skill **or an iterable of them** and folds embedding in, returning an awaitable (`Awaitable[None]`): on a `"semantic"`/`"hybrid"` catalog it embeds the batch off the asyncio loop (GIL released), so embedding errors (model load / endpoint / auth / dimension) surface from `await register(...)`. A `"bm25"` catalog registers metadata only and never loads a model. `search()` stays synchronous BM25-only; `search_async()` covers BM25/semantic/hybrid. There is **no** `register_many()`, `build_embeddings()`, or `rebuild_embeddings()` — `register()` embeds, and recovery from a model/dimension change is to construct a new catalog and re-register.
- Capability tools await async retrieval; MCP ingestion embeds ingested tools during `register`.
- Embedding configuration is validated and retained on BM25-default catalogs for later async semantic/hybrid overrides; typed config variants are mutually exclusive.

### Fixed

- A forgotten `await` on `register()` no longer silently drops the corpus: an un-awaited call still registers the tools/skills (BM25 keeps working), and a `"semantic"`/`"hybrid"` `search_async` after an un-awaited `register()` raises an actionable "not awaited" error instead of ranking an empty corpus.

## [0.4.2] - 2026-07-11

### Changed

- `configure_telemetry()` returns a per-call shutdown handle (`handle.shutdown()` / `handle.force_flush()`) on every path — the no-override path no longer leaks `init()`'s shared provider directly. Because that provider is shared across callers, shutting one handle down stops export for all of them.

### Fixed

- `configure_telemetry()` no longer mutates a shared provider's `shutdown` method. When idempotent telemetry initialization reuses one provider, a stale handle can no longer clear a newer caller's generation-scoped content-capture override.

## [0.4.1] - 2026-07-10

### Added

- `configure_telemetry` opts into message/tool content capture programmatically: `capture_content` sets the exact `ContentCapture` mode (validated like the env var — case-insensitive, legacy boolean forms accepted — raising a `ValueError` on garbage before any exporter is wired), and `include_span_and_events` is boolean sugar (`True` → `SPAN_AND_EVENT`, `False` → `NO_CONTENT`). `capture_content` wins over `include_span_and_events`; when neither is provided, `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` keeps ruling (a provided option beats the env var, as in OTel code-over-env precedence). The returned provider's `shutdown()` restores env-driven behavior via a generation-scoped clear (`clear_content_capture`), so a stale handle shutting down late never clobbers an override a newer `configure_telemetry` installed. The `set_content_capture` / `clear_content_capture` / `ContentCapture` primitives live in `ratel_ai_telemetry`.

## [0.4.0] - 2026-07-07

### Added

- **OpenTelemetry emission.** The SDK now opens an OTel span at each funnel boundary — `execute_tool` (`gen_ai.operation.name`, `gen_ai.tool.name`, `ratel.tool.args_size_bytes`), `ratel.search` (target, `top_k`, origin, `hit_count`), `ratel.skill.load`, `ratel.upstream.register`, and `ratel.auth.flow` — alongside the existing local `record_event` stream, which is unchanged. Emission is transparent and free by default: the `opentelemetry` API and the vocabulary are imported lazily, so the base (dependency-free) install is a pure pass-through no-op, and when OpenTelemetry is present the spans flow to whatever provider is registered. Built on the OTel-free `ratel_ai_telemetry` vocabulary, so the base install stays dependency-free.
- `configure_telemetry(api_key=..., endpoint=..., headers=..., service_name=...)` convenience wiring, exported from `ratel_ai`: installs a Ratel-owned OTLP exporter (via the new `[otlp]` extra, `pip install 'ratel-ai[otlp]'`) that ships the SDK's spans to Ratel Cloud (or any OTLP endpoint) and returns the provider as a shutdown handle. Hosts already running OpenTelemetry skip it and add `ratel_span_processor` from `ratel_ai_telemetry` instead.
- Message/tool content (`ratel.search.query`, `gen_ai.tool.call.arguments` / `.result`) rides span attributes only when `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` selects a span mode (`SPAN_ONLY` / `SPAN_AND_EVENT`); default off.

## [0.3.0] - 2026-07-06

### Added

- `ToolCatalog` / `SkillCatalog` accept a default `method` (`"bm25"` | `"semantic"` | `"hybrid"`) and `search(query, top_k, origin=..., method=...)` takes a per-call override. `"bm25"` (default) is unchanged and model-free; `"semantic"` / `"hybrid"` load a local embedding model and raise `RuntimeError` if it fails to load. Exposed via the native `search_with_method` binding and the `SearchMethod` type.
- A `"semantic"`/`"hybrid"` catalog embeds each tool/skill **eagerly at `register`** (incrementally), so searches never pay the corpus-embedding cost. New `catalog.build_embeddings()` pre-computes embeddings on demand (e.g. after a bulk register). BM25 catalogs do neither. A semantic/hybrid search on a catalog with no embeddings built raises `RuntimeError` (embeddings not computed) rather than embedding during the search.

## [0.2.1-rc.1] - 2026-07-04

### Changed

- First release cut under the per-package release scheme (ADR-0008): `ratel-ai` now versions and ships independently of the core crate and JS SDK, tagged `sdk-py-v*`. No API changes since 0.2.0.

## [0.2.0] - 2026-06-16

### Changed

- **BREAKING:** `search_tools_tool` → `search_capabilities_tool` (`SEARCH_TOOLS_ID` → `SEARCH_CAPABILITIES_ID`). It now returns two independently-ranked buckets — `{ tools, skills }`. Brings the Python SDK to parity with the TypeScript SDK. The old `search_tools_tool`/`SEARCH_TOOLS_ID` are kept as deprecated aliases (see _Deprecated_), so `0.1.x` code keeps working after upgrading.

### Added

- First-class **skills**: `SkillCatalog`, `Skill`, `get_skill_content_tool` (`get_skill_content`), and the native `SkillRegistry`/`SkillHit` — the on-demand skill analogue of the tool catalog, ranked by a separate BM25 corpus.
- Skill–tool coupling: a `Skill` can declare a `tools` list, and `search_capabilities` pulls a matched skill's declared tools into the `tools` bucket — additively (beyond `topKTools`) and deduped against query hits — so the agent gets the playbook and the tools it needs in one turn.

### Deprecated

- `search_tools_tool` and `SEARCH_TOOLS_ID`. They retain their pre-0.2.0 behaviour — a tools-only `{groups}` result and the `search_tools` id — so upgrading from `0.1.x` does not break existing callers. Migrate to `search_capabilities_tool`; the aliases will be removed in a future release.

### Fixed

- Gateway error payloads carry `isError: True`; `invoke_tool` rejects a non-object `args` instead of forwarding stray top-level keys.
- `search_capabilities_tool` validates `topKTools`/`topKSkills` (declared `integer`, positive): `0`, negative, `bool`, and `float` fall back to the default, matching the TypeScript SDK exactly.
- `search_capabilities_tool` advertises the `skills` bucket and `get_skill_content` in its description only when a non-empty `SkillCatalog` is provided.

## [0.1.6] - 2026-06-10

### Added

- Initial release of the Python SDK. Binds the Rust core (`ratel-ai-core`) via PyO3, distributed as prebuilt `abi3` wheels for darwin-arm64, darwin-x64, linux-x64-gnu, linux-arm64-gnu, and win32-x64-msvc — no Rust toolchain required to install. (`v0.1.5` shipped TS-only on 2026-05-10; the first release carrying Python is the next version bump.) Binding strategy locked in [ADR-0006](../../../docs/adr/0006-native-ffi-bindings.md).
- Full feature parity with the TypeScript SDK (`@ratel-ai/sdk`):
  - `ToolRegistry` / `SearchHit` — metadata-only BM25 index (native).
  - `ToolCatalog` accepts a `trace` config (`noop` default, `memory`, or `jsonl`); captured events flow through the Rust core sink ([ADR-0007](../../../docs/adr/0007-telemetry-two-streams.md)). Exposes `record_event`, `drain_trace_events`, and an `origin` argument on `search`. `invoke` emits `invoke_start` / `invoke_end` / `invoke_error` with `args_size_bytes` and `took_ms`, and awaits coroutine executors.
  - `search_tools_tool` / `invoke_tool_tool` gateway factories with verbatim descriptions and JSON schemas from the TS SDK. `search_tools_tool` emits `gateway_search` with `origin: "agent"`; `invoke_tool_tool` emits `gateway_invoke` / `gateway_error` and handles the `needs_auth` / `on_unauthorized` path.
  - `register_mcp_server` ingests an upstream MCP `ClientSession` (optional `mcp` extra, `pip install 'ratel-ai[mcp]'`), namespacing tool ids as `<server>__<tool>` and emitting `upstream_register` / `upstream_invoke` / `upstream_error`.
- Ships type stubs (`_native.pyi`, `py.typed`) for a fully typed install.
