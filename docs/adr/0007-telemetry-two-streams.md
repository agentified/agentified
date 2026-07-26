# 7. Telemetry: core-owned local trace stream, OTel remote conventions

Date: 2026-07-05

## Status

Accepted

Compacted 2026-07 from pre-compaction ADR-0009 (trace events, 2026-05-08) and ADR-0015 (OTel
re-founding, 2026-07-04). Pre-compaction ADR-0013 (a bespoke cloud telemetry schema, built at
`961985d` but never published) was fully superseded by 0015 and is dropped; that SHA remains
the concept-inventory reference.

Amended 2026-07-24 to align content capture with the OpenTelemetry Logs Event API and its
four capture modes.

Amended 2026-07-26 to drop the `@ratel-ai/telemetry-otlp` helper, and the TS OTLP config
resolver with it, now that the TS host owns the OTel provider and its configuration.

## Context

Two telemetry surfaces exist for different consumers. A **local** stream feeds the offline
inspector, statusline / savings reporting, and future rerankers and suggestion analyzers. A
**remote** path feeds Ratel Cloud and any observability backend the customer already runs.
The industry standardized the remote payload (OpenTelemetry's `gen_ai.*` semantic
conventions) after Ratel's first bespoke design; building our own schema and transport would
make Ratel Cloud an island.

## Decision

### Local stream: core-owned schema, query-log semantics

- The trace event data model and the sink trait live in `ratel-ai-core`. Every host language
  emits the same shapes; every consumer reads the same shapes. Event set: search, invoke
  start/end/error, gateway-tool calls, upstream-MCP ingest, auth / `needs_auth`, plus the
  skill events ([ADR-0005](0005-first-class-skills.md)). Additions are non-breaking;
  renames/removals are not.
- **One tagged stream, filtering at the consumer**: rerankers, suggestion analysis, and
  inspection subscribe to different cuts of the same producer; no parallel pipes to drift.
- **Query-log semantics, not oplog semantics**: trace events are observations of usage.
  Best-effort, sampleable, lossy on backpressure, loosely ordered, no synchronous durability
  on the hot path (ring buffer, periodic flush). Losing an event is acceptable; corrupting a
  catalog is not.
- The SDK records events into the core sink via a synchronous FFI call per event (negligible
  against the LLM call bracketing each invocation). The JSONL sink writes per-project buckets
  under `~/.ratel/telemetry/<project-slug>/`; the slug convention mirrors Claude Code's
  project directories and is owned by the consuming shell (today ratel-local), while the
  core sink accepts any path.

### Remote path: OpenTelemetry, pinned, two tiers

- Remote telemetry **is** OpenTelemetry: LLM calls are `gen_ai.*` spans per a **pinned**
  semconv baseline (v1.42.0, `gen_ai` group; the group is still `Development`, so the pin is
  the contract and bumps are reviewed changes).
- **Two tiers, layered not forked**: `gen_ai.*` adopted verbatim (never renamed or re-nested)
  plus `ratel.*`, the vocabulary we own: the gateway / skill funnel expressed as OTel
  spans and attributes, joinable with any `gen_ai.*` trace by trace/span id.
- **Content capture uses the OTel-defined span/EventRecord channels and stays default-off**:
  inference messages ride the standard `gen_ai.client.inference.operation.details` EventRecord;
  tool arguments/results ride `gen_ai.tool.call.*` span attributes in span modes and the
  structured `ratel.tool.execution.details` EventRecord in event modes; search text follows the
  same split. These are Events in the Logs data model, not SpanEvents.
- **Ratel Cloud ingests stock OTLP** (`http/protobuf` + `Bearer`). No custom wire format or
  auth: a customer who already runs OTel dual-exports to Ratel by adding a second exporter.
- **Three thin vocabulary helpers**, with stock OTel transport: `ratel-ai-telemetry`
  (crates.io; the `ratel.*` constants), `@ratel-ai/telemetry` (npm; constants + the
  content-capture gate, OTel-free so importing the vocabulary pulls no OTel SDK),
  `ratel-ai-telemetry` (PyPI; constants, with `init()` behind the `[otlp]` extra). The
  asymmetry is deliberate: in TS the host owns the OTel provider *and* its endpoint/auth
  configuration, while Python keeps `init()`. Shared conformance fixtures assert every helper
  against the pin so the languages cannot drift.

### Two streams by design

Local and remote stay separate producers: the local stream is offline-first with its own
reliability profile; the remote path is an OTLP export. Converging them is a future decision,
not an accident of this one.

## Consequences

- Interoperable by construction: Ratel-emitted traces land in any OTel backend; existing
  traces reach Ratel Cloud as a config change.
- The pin is a maintenance obligation (the `gen_ai.*` group churns); `ratel.*` is the only
  vocabulary we design and version, with the same care as the local event schema.
- Cross-language reuse is built in: TS- and Python-emitted remote spans and EventRecords share
  one contract, while their local events continue to share the core-owned schema.
- Rejected: a bespoke unified schema and per-language clients (duplicates a ratified
  standard; the pre-compaction 0013 built exactly this and it was deleted unpublished);
  inference-message content on span attributes (attribute limits reject unbounded text);
  forking `gen_ai.*`
  into a Ratel namespace (re-breaks the interop the standard buys); tracking semconv
  `latest` (unreviewed breaks); converging local and remote now (opposite reliability and
  offline constraints).
