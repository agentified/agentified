# 20. Runtime events lane: subscribable facts and catalog snapshots

Date: 2026-08-13

## Status

Accepted

## Context

Ratel's core trace stream already records the product facts that power inspection, adaptive
ranking, suggestions, and catalog views. Ratel Cloud can infer some of those facts from the
parallel OpenTelemetry projection, but sampling, processors, and the content-capture gate make
OTel an observation channel rather than a census. Catalog definitions are state, not telemetry,
and cannot be reconstructed from lossy churn events.

The SDK therefore needs one public, cross-language event seam that local consumers and an
optional Cloud adapter can subscribe to without changing the existing OTel wrappers. It also
needs a separate snapshot seam for publishing current catalog state.

## Decision

### Contract and evolution

The core trace producer becomes a public **runtime events** stream. Rust owns the schema; the
TypeScript and Python SDKs expose push subscriptions to the same flattened JSON vocabulary.
SDK-only experiment events join that vocabulary before delivery. Additive fields and event
types are non-breaking; renames and removals are breaking. Consumers MUST ignore unknown fields
and accept unknown event types. Shared fixtures under `src/telemetry/conformance/` pin the wire
names and values across languages.

The remotely publishable v1 event set is:

| Family | Event types | Required product facts |
|---|---|---|
| Search | `search`, `skill_search`, `gateway_search` | query, target/origin, `top_k`, duration, and ordered `hits[]` of target id and score |
| Tool invocation | `invoke_start`, `invoke_end`, `invoke_error`, `gateway_invoke`, `gateway_error` | tool id, `invocation_id`, outcome/error class, and duration where known |
| Skill use | `skill_invoke` | skill id, outcome, and duration |
| Catalog churn | `index_churn`, `skill_churn` | add/remove, target id, and catalog version where known |
| Upstream MCP | `upstream_register`, `upstream_invoke`, `upstream_error` | server, transport/tool count or tool id, outcome/error class, and duration where known |
| Auth | `auth_refresh`, `auth_needs`, `auth_flow_start`, `auth_flow_end` | upstream id and outcome; never credentials |
| Experiments | `experiment_selection`, `experiment_results`, `experiment_comparison`, `experiment_skip`, `experiment_fallback`, `experiment_drop`, `experiment_invocation`, `experiment_outcome` | `selection_id`; served/shadow arm data; agreement metrics; result ids/scores; attribution, drop/fallback reason, and labelled outcome as applicable |
| Delivery | `events_dropped` | dropped count, reason, and observation window |

For search events, the envelope `event_id` identifies the search. A hit's zero-based rank is its
position in the ordered `hits[]` array rather than a repeated field on each hit.

Newer SDKs may emit additive types before every receiver understands them; the receiver stores
the unknown envelope rather than rejecting the batch. Core diagnostic variants that are not in
the table remain local until deliberately added to the remotely publishable set.

Search query text and hit ids/scores are part of the facts contract. A query is at most 4 KiB
and `hits[]` at most 100 entries. Attaching a remote publisher is explicit consent to send those
fields and uses a gate independent of `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
Inference messages, tool arguments/results, executors, tokens, cost, and model details do not
enter this lane; they remain absent or OTel-only as applicable.

### Envelope v2 and identity

Every event is flattened into this v2 envelope:

| Field | Contract |
|---|---|
| `v` | integer `2` |
| `event_id` | client-generated ULID, minted once for this event |
| `ts` | producer time, Unix milliseconds |
| `session_id` | stable for one agent session |
| `source_id` | stable identity shared by events and snapshots |
| `invocation_id` | present on invocation lifecycle events; one opaque id shared by start/end/error |
| `catalog_version` | optional catalog revision known at emission |
| `environment` | optional deployment environment |
| `end_user_id` | optional application-provided subject id |
| `trace_id`, `span_id` | optional active OTel correlation ids |
| `type` and payload | flattened event tag and fields |

`event_id` is the canonical deduplication and join key. The same value survives fan-out,
batching, and retries and is stamped as `ratel.event.id` on the corresponding OTel span or Logs
EventRecord. A receiver derives its storage id deterministically as UUIDv8 over
`(project_id, event_id)`; replaying a batch is therefore idempotent without changing the public
ULID. Each event in an invocation has its own `event_id`; `invocation_id` groups the lifecycle.

`source_id` is an explicit attach option, defaulting to the env-var-configured OTel
`service.name` (`OTEL_SERVICE_NAME`, then `service.name` in `OTEL_RESOURCE_ATTRIBUTES`),
falling back to `ratel`. A programmatically configured OTel resource — including a
service name passed to the SDK's own `configure_telemetry` — is not read; pass `source_id`
explicitly in that case. It MUST remain stable: renaming it starts a new source era and a
distinct catalog snapshot.

### Delivery and bounds

Emission fans out to multiple independent subscribers. Each push subscriber has a bounded
queue; enqueue is synchronous and cheap, delivery is batched and asynchronous, and subscriber
code can never block the emitting operation. On overflow the queue drops its oldest event and
reports loss through a later `events_dropped` meta-event; monotonic SDK drop counts remain
available for local accounting. An OTel counter projection is deferred. The test-only memory
sink may remain unbounded.

Delivery is deliberately best-effort: in-memory queue plus explicit `flush()`, with no disk
spool, ordering guarantee across producers, or synchronous durability. `flush()` drains work
already accepted by the process; it cannot recover events lost to overflow or abrupt process
termination. The TypeScript bridge uses a napi threadsafe callback and Python marshals callback
delivery safely off the Rust worker thread/GIL-sensitive hot path.

The Cloud receiver exposes `POST /api/v1/events` with the same `rtl_` Bearer-key model as OTLP.
Wire limits are 64 KB (65,536 bytes) per serialized event and 4,000,000 bytes or 5,000
events per batch, whichever is reached first; the first-party Cloud SDK batches below
3,900,000 bytes to stay clear of the receiver cap. It accepts valid events with OTLP-style partial
success, returning rejected event ids and reasons, and rate-limits each key to 120 batch
requests/minute with burst 240 (`429` plus `Retry-After`). Runtime facts are quota-exempt for
now. Reject storage is metadata-only with roughly seven-day retention; raw accepted envelopes
retain for 30 days by server receipt time and feed frontier read models at ingest.

A Cloud publisher is fail-open, owns retry/backoff and lifecycle, and can be disabled with
`RATEL_CLOUD_EVENTS=off`. Cloud transport itself is not part of this repository.

### Parallel OTel projection and per-source supersede

OpenTelemetry emission remains unchanged and is never rebuilt as a subscriber to this stream.
The two projections share `event_id` and optional trace/span ids, but have different authority:
runtime events are the census for Ratel-owned product facts; OTel remains the observability and
trace-context path and the sole path for LLM tokens, cost, model data, and trace structure.

Cloud supersedes OTel-derived product facts **per source**, not per project. The first direct
facts event for a `source_id` starts that source's facts era; from that boundary Cloud stops
deriving the overlapping product fact from OTel for that source. It continues to ingest OTel,
deduplicates overlap by `ratel.event.id`, and uses trace/span correlation for enrichment and
drill-down. Other sources in the project may continue on OTel-derived facts.

### Catalog snapshot publication carve-out

Catalog definitions do not ride runtime events. `catalog.snapshot()` returns the complete
serializable definition set (ids, names, descriptions, schemas, and public metadata), never an
executor or secret material. The Cloud adapter publishes that state separately under the same
`source_id`, using a canonical content hash and atomic full replacement so removals work and
unchanged snapshots are skipped. Churn events may debounce/trigger publication, but are not an
oplog from which catalog state is rebuilt.

This upward path is not catalog authoring or bidirectional source sync. It publishes the
running SDK's observed state and append-only facts. Event and snapshot shapes structurally omit
API keys, OAuth tokens, tool arguments/results, executors, and other secret-bearing fields;
secrets-never-sync remains invariant.

The Cloud adapter lives in `@ratel-ai/cloud-sdk/runtime`. Core runtime events and snapshots have
TypeScript/Python parity; one-line Cloud attach is TypeScript-only while no Python Cloud adapter
exists.

## Consequences

- Ratel Cloud product facts no longer depend on OTel sampling or content capture, while OTel
  interoperability and host-owned provider configuration remain intact.
- The runtime vocabulary is now a frozen cross-language contract with a conformance burden.
- Slow or failed subscribers cannot stall an agent, at the accepted cost of observable loss.
- Catalog removals converge through full snapshots even when churn events drop.
- Stable source identity becomes operational configuration: changing it forks the facts era
  and catalog state.

## Rejected

- **OTel-only product facts:** sampling and processors make counts incomplete; the content gate
  can remove fields Cloud features require.
- **Rebase OTel on runtime events:** point events cannot preserve wrapper span lifecycle and
  ambient context without redesigning working instrumentation.
- **Catalog definitions as events:** a lossy stream cannot safely reconstruct current state or
  removals.
- **Durable delivery in the SDK:** a spool/ack protocol is a different reliability tier; this
  lane remains fail-open and in-memory.
- **Bidirectional catalog sync:** creates offline merge and secret-leak classes. Source pull and
  source-scoped snapshot publication remain distinct one-way operations.
- **Cloud transport in core:** product-specific auth, retries, and lifecycle belong to the Cloud
  adapter, not the language-neutral event producer.
