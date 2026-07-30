# Ratel telemetry conventions

The wire contract for Ratel's **remote** telemetry. Ratel telemetry *is* OpenTelemetry:
LLM calls are `gen_ai.*` spans with content-bearing Logs EventRecords, Ratel's
capability/skill funnel is a `ratel.*` overlay on the same traces, and ingest is stock OTLP.
This document is what every consumer (Ratel Cloud,
dashboards, a self-hosted receiver) reads against; the per-language helpers under
`core/`, `ts/`, `python/` codify the `ratel.*` half as constants.

Decision of record: [ADR-0007, Telemetry: core-owned local trace stream, OTel remote conventions](../../docs/adr/0007-telemetry-two-streams.md).
This spec is the concrete mapping that ADR locks; it does not re-decide anything the ADR decided.

Scope is the **remote** stream only. The local JSONL trace stream (ADR-0007: `src/core/src/trace/`,
consumed by the statusline / savings report) is untouched and is **not** part of
this contract. Local and remote are two streams on purpose.

## The pin

Ratel adopts **OpenTelemetry semantic conventions v1.42.0, `gen_ai` group**, and tracks it explicitly.
The pin is the contract; consumers read against the pinned version, not "latest".

Two facts about this baseline the pin maintainer must know:

- The `gen_ai.*` group is **`Development`** (not Stable). It will churn. Absorbing a `gen_ai.*` rename
  is a deliberate, reviewed bump of this baseline, never ambient drift.
- At the **v1.42.0** tag the `gen_ai` group was **relocated** out of `open-telemetry/semantic-conventions`
  (into the still-untagged `semantic-conventions-genai` repo) and left behind as a frozen snapshot under
  `model/gen-ai/deprecated/`. "Deprecated" here means *moved*, not *withdrawn*: the v1.42.0 definitions are
  that frozen YAML. The keys below were read from it and cross-checked against the last live rendered prose (v1.41.0).

**Bump process.** Changing the pin is a reviewed change with its own PR: diff the new baseline's `gen_ai.*`
registry against this table, update the mapping and the `ratel.*`-adjacent notes, bump the constant in each
helper, and note the move in a superseding ADR if the shape (not just keys) changed.

## Two tiers

| Tier | Namespace | Owner | Carries |
|---|---|---|---|
| Base | `gen_ai.*` | OpenTelemetry (pinned v1.42.0) | the LLM call: operation, provider, model, params, usage, finish; inference content on the details EventRecord |
| Overlay | `ratel.*` | this repo | the capability/skill funnel as spans, attributes, and content EventRecords correlated through the active host context |

`gen_ai.*` is adopted **verbatim**, not one key renamed or re-nested. `ratel.*` is the only vocabulary
Ratel designs and versions. A Ratel-instrumented agent and a plain-`gen_ai.*` agent running under the
same active host span land in one trace, told apart by namespace and joined on trace/span id.

`ratel.*` follows ADR-0007's schema discipline: **adding** a span or attribute is non-breaking; **renaming or
removing** one is breaking and needs a superseding note.

---

## Tier 1: the LLM call (`gen_ai.*`)

An LLM call is a `gen_ai` client span. Span kind `CLIENT` (`INTERNAL` allowed for in-process models).
Span name is `{gen_ai.operation.name} {gen_ai.request.model}` (e.g. `chat gpt-5.5`), falling back to
`{gen_ai.operation.name}` when the model is unknown.

### Legacy inventory to `gen_ai.*`

The `src/cloud/` schema at `961985d` (pre-compaction ADR-0013, deleted, never published; in git
history) is the concept inventory. Every field re-expresses in a standard v1.42.0 key, including
cached and reasoning tokens, which the earlier assumption held were missing:

| Legacy field | `gen_ai.*` key (v1.42.0) | Notes |
|---|---|---|
| `provider` (resolved) | `gen_ai.provider.name` | Well-known enum, open to custom values. Replaces the deprecated `gen_ai.system`. Enum incl. `openai`, `anthropic`, `aws.bedrock`, `gcp.vertex_ai`, `azure.ai.openai`, `mistral_ai`, `x_ai`, ... |
| `model` (resolved) | `gen_ai.request.model` + `gen_ai.response.model` | request = asked, response = served |
| `ts` | span **start time** | not an attribute |
| `latency_ms` | span **duration** | also the `gen_ai.client.operation.duration` metric |
| `stream` | `gen_ai.request.stream` | boolean; cond. required iff streaming |
| `system` | `gen_ai.system_instructions` | on the details **event** (content), see Tier 1 content |
| `tools` (offered defs) | `gen_ai.tool.definitions` | Opt-In; list of JSON-schema-shaped defs |
| `messages` | `gen_ai.input.messages` / `gen_ai.output.messages` | on the details **event**, see Tier 1 content |
| `params.temperature` | `gen_ai.request.temperature` | double |
| `params.top_p` | `gen_ai.request.top_p` | double |
| `params.max_tokens` | `gen_ai.request.max_tokens` | int |
| `params.stop` | `gen_ai.request.stop_sequences` | string[] |
| `usage.input_tokens` | `gen_ai.usage.input_tokens` | **includes** cached tokens |
| `usage.output_tokens` | `gen_ai.usage.output_tokens` | **includes** reasoning tokens |
| `usage.cached_tokens` | `gen_ai.usage.cache_read.input_tokens` | subset of `input_tokens`. (`cache_creation.input_tokens` also exists for cache writes.) |
| `usage.reasoning_tokens` | `gen_ai.usage.reasoning.output_tokens` | subset of `output_tokens`; "when applicable" |
| `finish_reason` | `gen_ai.response.finish_reasons` | **array** (string[]), one per generation |

Additional v1.42.0 keys worth emitting when available: `gen_ai.response.id`, `gen_ai.conversation.id`,
`gen_ai.request.seed`, `gen_ai.request.top_k` (double), `gen_ai.request.frequency_penalty`,
`gen_ai.request.presence_penalty`, `gen_ai.request.choice.count`, `gen_ai.output.type`,
`server.address` / `server.port`, `error.type`.

**`finish_reason` value note.** The legacy enum was `stop | length | tool_call | content_filter | refusal`.
The v1.42.0 normative **output-message** schema (`gen-ai-output-messages.json`, the per-message
`finish_reason` field) is `stop | length | content_filter | tool_call | error`, with no `refusal`. Emit the
singular `tool_call` from that schema; do **not** emit `tool_calls` (plural), which is the value from the
*deprecated* `gen_ai.choice` event, not the message-part schema. The span-level
`gen_ai.response.finish_reasons` array is an open `string[]`, so emit `refusal` verbatim there rather than
lossily folding it into `content_filter`.

**Do not spec these stale keys:** `gen_ai.system` (to `provider.name`), `gen_ai.usage.prompt_tokens`
(to `input_tokens`), `gen_ai.usage.completion_tokens` (to `output_tokens`), `gen_ai.prompt` / `gen_ai.completion`
(to messages), `gen_ai.openai.request.seed` (to `request.seed`).

### Tier 1 inference content: on OpenTelemetry EventRecords

Message text and tool-call arguments ride the **`gen_ai.client.inference.operation.details`** event
(`gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`) for an LLM inference
operation, never span attributes. Tool-execution and search content follow the separate two-channel
capture table below. This is an OpenTelemetry Event in the **Logs data model**, not a
SpanEvent. Maps and arrays MUST be recorded as structured AnyValue values, not JSON strings. Locked by
ADR-0007: span attributes are size-bounded and message content is not.

Message shape (v1.42.0): `{ role, parts[], name? }`; output messages also carry `finish_reason`.
Roles: `system | user | assistant | tool` (open). `system_instructions` is a bare `parts[]` (no role wrapper).

Legacy content blocks to v1.42.0 message parts:

| Legacy block | v1.42.0 part `type` | Notes |
|---|---|---|
| `Text { text }` | `text` (`content`) | |
| `ToolCall { id, name, arguments }` | `tool_call` (`id?, name, arguments`) | on an **assistant** message; `arguments` a parsed object |
| tool result (`Message::Tool { tool_call_id, content }`) | `tool_call_response` (`id?, response`) | on a **tool** message; normative field is `response` (the registry example shows `result`, a known upstream schema/example mismatch) |
| `Image { source, media_type }` (inline) | `blob` (`modality: image`, `mime_type`, `content` = base64) | |
| `Image { url, media_type }` | `uri` (`modality: image`, `uri`) | |
| `File { source, media_type }` | `blob` (`modality` per mime) or `file` (`file_id`) | |
| `File { url, media_type }` | `uri` | |
| (reasoning / thinking text) | `reasoning` (`content`) | new in v1.42.0; the legacy schema flagged this as the most sensitive surface |

Other v1.42.0 parts available but not in the legacy inventory: `server_tool_call` /
`server_tool_call_response` (provider-executed tools), `generic` (extensibility escape hatch).

**Capture gating.** Content is Opt-In, **default off**. The gate is the ecosystem instrumentation
convention `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`. Note this is an *instrumentation-level*
convention, not a semconv-v1.42.0 attribute. Honor it rather than inventing a Ratel flag. Values:
legacy boolean, or the enum `NO_CONTENT` (default) | `SPAN_ONLY` | `EVENT_ONLY` | `SPAN_AND_EVENT`.

The four values select **two independent channels** — span attributes and Logs EventRecords — and every
content-bearing helper honors both, so a mode is truthful (no mode silently drops content it names):

| Value | Span attributes | OTel Logs EventRecords |
|---|---|---|
| `NO_CONTENT` (default) | — | — |
| `SPAN_ONLY` | yes | — |
| `EVENT_ONLY` | — | yes |
| `SPAN_AND_EVENT` | yes | yes (both) |

- **Span-attribute channel** (`SPAN_ONLY`/`SPAN_AND_EVENT`): tool content on `gen_ai.tool.call.arguments`
  / `gen_ai.tool.call.result`; search text on `ratel.search.query`.
- **EventRecord channel** (`EVENT_ONLY`/`SPAN_AND_EVENT`): tool content on a
  `ratel.tool.execution.details` event carrying structured `gen_ai.tool.call.arguments` and, on
  success, `gen_ai.tool.call.result`, plus `gen_ai.operation.name = execute_tool` and
  `gen_ai.tool.name`; search text on a `ratel.search.results` event carrying
  `ratel.search.query`. The search event carries only the query — hit ids/scores/BM25 timing live
  on the local trace stream, not the OTLP glue.

`gen_ai.output.messages` is reserved for model-generated outputs; every output message requires
`finish_reason`. A tool execution result is therefore never encoded as an output message.

**Python 3.9 compatibility.** OpenTelemetry Python 1.41 rejects heterogeneous AnyValue arrays.
Rather than silently replace such content with `null`, the Python SDK encodes only those arrays
as `{"ratel.type": "array", "ratel.items": {"0": ..., "1": ...}}`. The indexed map is
lossless and order-preserving; homogeneous arrays retain their normal array shape.

---

## Tier 2: the Ratel funnel (`ratel.*`)

The local trace event set (ADR-0007) plus the skill events (ADR-0005) are the mapping source: search, invoke (start/end/error), skill search/invoke,
upstream-MCP ingest, auth / `needs_auth`. Each becomes a span, attributes on a `gen_ai` span,
or a content-bearing Logs EventRecord under `ratel.*`.

**Errors** use standard OTel span status (`ERROR`) + `error.type` and the exception event, not a bespoke
`ratel.*.error` attribute. **Origin** (agent-synthesized vs direct library call) is a shared attribute:

| Attribute | Type | On | Values |
|---|---|---|---|
| `ratel.origin` | enum | search, invoke, and third-party `gen_ai.*` spans a framework adapter overlays | `direct \| agent` |

On Ratel's own spans the emitter knows which case it is. On the overlay case it cannot: a
framework adapter (`@ratel-ai/vercel-ai-sdk/otel`) stamps `ratel.origin` onto `gen_ai.*` spans
another library created, so the value is host-selectable there and defaults to `agent` — right
for the tool-loop spans an agent synthesizes, wrong for host-driven `embed` / `rerank` calls.
It is selected per adapter instance, not per span, so a host making both kinds of call marks
the direct ones by passing a second, `direct`-configured instance on those calls.

### `ratel.search`: capability search (unifies `search`, `gateway_search`, `skill_search`)

| Attribute | Type | Notes |
|---|---|---|
| `ratel.search.target` | enum | `tool \| skill` (folds tool-search and skill-search into one span shape) |
| `ratel.search.top_k` | int | requested result count |
| `ratel.search.hit_count` | int | results returned |
| `ratel.search.query` | string | **content, gated** like message content; may hold user/agent text |
| `ratel.origin` | enum | `direct \| agent` |

The search text rides an Opt-In Logs EventRecord **`ratel.search.results`** (as
`ratel.search.query`), gated under the event channel. The span's non-content fields carry counts;
`SPAN_ONLY` and `SPAN_AND_EVENT` additionally put the gated query on the span. Hit ids + scores +
per-stage BM25 timing live on the local trace stream (the SDK's OTLP glue does not have them at the
span boundary), not this remote event.

### tool invocation: `execute_tool` span + `ratel.*`

An `invoke_tool` call (unifying `invoke_start/end/error`, `gateway_invoke/error`, `upstream_invoke/error`)
is modelled as a standard **`gen_ai.operation.name = execute_tool`** span (interop: a generic OTel backend
already understands it) enriched with `ratel.*`:

| Attribute | Type | Notes |
|---|---|---|
| `gen_ai.tool.name` | string | the capability tool id |
| `gen_ai.tool.call.id` | string | when available |
| `ratel.tool.args_size_bytes` | int | argument payload size (from `invoke_start`) |
| `ratel.upstream.server` | string | upstream MCP server backing the tool, when the invoke proxies one |
| `ratel.origin` | enum | `direct \| agent` |

Span duration is the invoke latency; failure sets span status `ERROR`. Tool arguments/results are Opt-In
content: on the span attributes `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` under the
span-attribute channel, and/or as structured attributes on the
`ratel.tool.execution.details` Logs EventRecord under the EventRecord channel — gated like messages,
per the two-channel table in § Tier 1 content.

> **Decided (2026-07-05):** invoke is modelled as an `execute_tool` span enriched with `ratel.*`, for
> OTel-backend interop, not a pure `ratel.invoke` span. The considered alternative (a pure `ratel.invoke`
> span, full tier separation, no interop) was rejected. Revisit only via a superseding note.

### `ratel.skill.load`: skill content load (`skill_invoke` / `get_skill_content`)

| Attribute | Type |
|---|---|
| `ratel.skill.id` | string |

### `ratel.upstream.register`: upstream-MCP ingest (`upstream_register`)

| Attribute | Type | Notes |
|---|---|---|
| `ratel.upstream.server` | string | |
| `ratel.upstream.transport` | string | `stdio \| http \| sse \| ...` |
| `ratel.upstream.tool_count` | int | tools ingested |

### `ratel.auth.flow`: MCP auth (`auth_refresh`, `auth_needs`, `auth_flow_start/end`)

| Attribute | Type | Notes |
|---|---|---|
| `ratel.upstream.server` | string | |
| `ratel.auth.outcome` | enum | `ok \| refreshed \| needs_auth \| failed` (`needs_auth` = the 401-driven `AuthNeeds`) |

### `ratel.experiment.arm`: retrieval experiment dispatch

Every serving or shadow arm run is an `INTERNAL` span named
**`ratel.experiment.arm`**. The TypeScript SDK uses instrumentation scope
`@ratel-ai/sdk`; `ratel.search` remains capability search only.

Five controlled strings form the arm stamp. Emitters set them directly on the span and use the
same keys in OTel baggage for descendant correlation:

| Attribute / baggage key | Type | Notes |
|---|---|---|
| `ratel.experiment.id` | string | configured experiment id |
| `ratel.experiment.selection_id` | string | opaque selection correlation id |
| `ratel.experiment.arm` | string | declared arm name; this key literal also names the arm span |
| `ratel.experiment.role` | enum | exactly `serving \| shadow`; never `served` |
| `ratel.experiment.unit` | string | first 16 lowercase hex characters of SHA-256(unit id) |

The remaining arm-span attributes describe dispatch and completion:

| Attribute | Type and presence |
|---|---|
| `ratel.experiment.cold` | required boolean at dispatch |
| `ratel.experiment.outcome` | required `ok \| empty \| timeout \| error` at completion; also the reported-outcome EventRecord name |
| `ratel.experiment.duration_ms` | required non-negative arm-callback duration |
| `ratel.experiment.hit_count` | non-negative integer when ranking succeeds |
| `ratel.experiment.ranking_error` | error type when ranking fails |
| `ratel.experiment.result_attributes_error` | error type when the result-level projector fails |
| `ratel.experiment.result_attrs_encoding_error` | error type when gated item attrs cannot be encoded |

Experiment lifecycle and evaluation use seven Logs **EventRecords**, never SpanEvents:

| EventRecord | Fixed experiment-specific attributes |
|---|---|
| `ratel.experiment.results` | full arm stamp; `ratel.experiment.result_ids`, `.result_scores`, `.result_attrs` |
| `ratel.experiment.comparison` | full shadow stamp; `ratel.experiment.served.{arm,outcome,duration_ms,hit_count}`, `ratel.experiment.shadow.{arm,outcome,duration_ms,hit_count}`, `ratel.experiment.agreement.{top1,exact_order,overlap_count,jaccard_at_k,k,item_attrs,result_attrs}` |
| `ratel.experiment.skip` | assigned-arm stamp; `ratel.experiment.skip.{arm,concurrency,reason}` |
| `ratel.experiment.fallback` | failed assigned-arm stamp; `ratel.experiment.fallback.{effective_arm,reused_shadow}` |
| `ratel.experiment.drop` | full shadow stamp; `ratel.experiment.drop.reason` |
| `ratel.experiment.invocation` | experiment id + unit + `ratel.experiment.invocation.attributed`; when attributed, `.selection_id`, `.effective_arm`, `.invocation.{rank,age_ms}`; optional `.turn`; borrowed `gen_ai.tool.name` |
| `ratel.experiment.outcome` | experiment id + selection id; `ratel.experiment.outcome.{label,score}` |

`ratel.experiment.result_ids` is an ordered `string[]` and is always a measurement, not
content. `result_scores` is present only when every item has a finite score.
`ratel.experiment.result_attrs` is the only experiment field controlled by the content-capture
gate: `SPAN_ONLY` / `SPAN_AND_EVENT` put its JSON encoding on the arm span; `EVENT_ONLY` /
`SPAN_AND_EVENT` put the structured, id-aligned array on `ratel.experiment.results`.
`NO_CONTENT` emits neither copy. Result-level projector values are never emitted; only
`agreement.result_attrs` booleans are.

Closed lifecycle values are:

- skip reason: `capacity`;
- drop reason:
  `arm-failed | fallback-consumed | selection-failed | served-ranking-failed | comparison-failed`.

`ratel.experiment.effective_arm` is selection-level and is not added to an already-completed arm
span. Standard `error.type` remains the error key; it is not mirrored under `ratel.*`.
Experiment baggage propagates only when the host registers a `ContextManager`, which is required
independently of exporter setup. The SDK registers neither providers nor exporters; hosts need
both span and log-record processors to deliver the complete experiment signal.

### Out of the remote tier

`index_churn` / `skill_churn` are internal catalog-maintenance events with no consumer in this
mapping source. They stay **local-only** (the ADR-0007 JSONL stream) and are not expressed in `ratel.*`.

---

## Ingest bounds (informative, server-side)

The legacy schema's abuse/`int4` bounds (about 2 MB per text field, about 20 MB per blob, `int4` token ceilings,
at most 10k messages, at most 2k tool defs, cache <= input, reasoning <= output) are **enforced at ingest**
(Ratel Cloud), not re-implemented in the helpers. They are recorded here so the mapping is complete; a helper
does not reject an oversized span, the ingest endpoint does.

---

## Conformance

The OTel re-founding (ADR-0007) retired the legacy schema's three-way golden-JSON round-trip. That
machinery existed to stop three hand-mirrored schemas from drifting; with one borrowed schema
(`gen_ai.*`) and one owned overlay (`ratel.*`), that reason is gone.

**Decided (2026-07-05):** keep a conformance suite but re-scope it as below. This resolves the phrase
"rebuild the conformance-vector pattern" carried over from the task brief, which predates that
retirement of the cross-mirror fixtures.

Conformance is re-scoped to **contract-against-the-pin**: a shared fixture set of
`(known input -> expected emitted keys/values)`, asserted per language against in-memory span and
Logs EventRecord exporters. Each helper, given a fixture, must emit the exact `gen_ai.*` keys this spec pins and the `ratel.*`
keys it owns, at the pinned semconv version. This tests "does the helper emit the convention correctly",
not "do three schemas agree". The `ratel.*` constants are the unit under test; `gen_ai.*` keys are asserted
against the v1.42.0 table above.

## Exporter initialization surface (recorded; implemented in the Python helper)

Python `init()` is sugar over the standard OTel SDK plus the `ratel.*` constants: no transport, no
FFI, no schema crate. TypeScript has no counterpart and no exporter configuration either: the host
owns the OTel providers, so it resolves its own endpoint and auth, and `@ratel-ai/telemetry` stays
vocabulary plus the content-capture gate. The asymmetry is deliberate (ADR-0007).

Turnkey initialization:

- Resolves the traces endpoint from `RATEL_OTLP_ENDPOINT`, falling back to the superseded
  `RATEL_URL` with a `DeprecationWarning` (that var also selects the SDK's catalog source per
  ADR-0003, so it no longer doubles as the OTLP destination); an explicit `endpoint=` wins over
  both. Resolves auth from `RATEL_API_KEY`; an explicit `api_key=` wins. Custom `headers` compose with
  either form. An explicit API key sets `Authorization: Bearer ...`; the `RATEL_API_KEY` fallback
  applies only when neither an explicit API key nor an explicit `Authorization` header is given, so
  ambient env never clobbers auth the caller set on purpose. The traces endpoint remains the full
  `/v1/traces` URL; the Logs exporter derives its sibling `/v1/logs` URL. `logs_endpoint` overrides
  that derivation.
- On first setup, accepts `enabled=False` before resolving configuration or registering a provider,
  returning a no-op handle without importing the OTel SDK at all. The composable span and
  log-record processors have the same switch. If Ratel already owns the global providers,
  idempotence wins and every later initialization call returns the original handle regardless of options.
- Exports every span and EventRecord by default on the turnkey path; `span_filter` and `log_filter`
  narrow those sets without requiring callers to construct providers.
- Is idempotent to itself: while the Ratel-owned tracer and logger providers are active, repeated calls (including
  module reloads) return the exact original handle and the first call's configuration remains
  authoritative; because that handle is shared, shutting it down stops export for every caller. A
  foreign global tracer or logger provider still raises with processor-based coexistence guidance.
- Shutdown is terminal: OTel's global providers register once per process, so after the handle's
  `shutdown()` a later initialization raises rather than return a dead handle; Python has no
  provider-reset escape hatch.
- Wires OTLP **`http/protobuf`** trace and Logs exporters with sane batching + shared resource defaults; everything else is the
  untouched OTel SDK the caller can configure directly.

Both the Python and TypeScript helpers expose the `ratel.*` attribute/span constants so callers emit
the vocabulary without stringly-typed keys, and honor
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` for content capture (default off).

A Python caller who already runs the OTel SDK skips turnkey initialization and adds
`ratel_span_processor()` plus `ratel_log_record_processor()` to the host tracer and logger
providers. Both default to the `gen_ai.*` / `ratel.*` signal filter and can be overridden. The
`[otlp]` extra supplies the complete exporter/SDK implementation.
