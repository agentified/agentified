# Ratel catalog-source contract — v1

The v1 surface is deliberately small: **catalog pull-sync + authentication.** A conforming
catalog source is the authoritative *source* of a project's published catalog; a client pulls
it and runs retrieval (`search_capabilities` / `invoke_tool` / `get_skill_content`)
**locally** over that replica. There is no remote search/invoke on the wire — see
[ADR-0003](../../docs/adr/0003-catalog-source-interface.md) for the embedded-replica model,
authentication, scope, and sync semantics.

Transport is HTTP/1.1+ with JSON bodies. Versioned paths are under `/v1`; `/healthz` is the
only unversioned endpoint. The in-process SDKs in this repository do not speak this protocol;
the contract is for **networked** sources and their loaders.

## Files

- [`schema/`](schema/) — JSON Schemas for the wire shapes (`CatalogSkillWire`, the catalog
  response, the error body) and for the [intent graph](#intent-graph) (`IntentGraph`).
- [`conformance/`](conformance/) — the executable conformance vectors and the reference
  verifier that every source and loader MUST pass.

## Authentication

Every `/v1` request to a networked source carries `Authorization: Bearer <key>`. The source
takes `sha256(key)` (hex) and looks it up against active keys (`key_hash = $ AND revoked_at IS
NULL`). Keys are stored **hash-only** — plaintext is shown once at creation and never
persisted (ADR-0003).

- missing or malformed header → `401`
- unknown or revoked key → `401`
- key store unreachable → `503`

A networked source MUST be served over TLS; a bearer key is a replayable static secret. Every
`/v1` request to a networked source carries a key — v1 has no loopback exception (see
non-goals).

## Catalog pull-sync

### `GET /v1/catalog`

Returns the project's published catalog. Conditional-GET with an `ETag`; the client caches
the last body and revalidates.

**Query parameters**

- `scope` *(optional)* — an opaque subject id (the SDK's end-user identifier). Selects the
  subject layer **overlaid on the global layer**: a skill in both layers is taken from the
  subject layer (**subject wins on name collision**). Absent — or naming a subject the source
  does not know — ⇒ the global layer only, byte-compatible with a source that has no notion of
  subjects. The scope *model* (`tenant → project → subject`, authorization, confidential
  isolation) is [ADR-0010](../../docs/adr/0010-catalog-scope-model.md); only the wire mechanics
  of `?scope=` are frozen here.

**Request headers**

- `If-None-Match: <etag>` *(optional)* — the client's cached ETag.

**Responses**

- `200 OK` — body below, plus `ETag: "<hash>"` and `Cache-Control: no-cache`.
- `304 Not Modified` — the client's `If-None-Match` matches the current ETag for the requested
  scope; no body. The ETag is a content hash of the resolved set for that scope, so a tag
  issued for one scope matches another only when both scopes currently hash to the same bytes.
- `401` / `503` — per Authentication.

**200 body**

```
{
  "catalogVersion": "<etag-string>",
  "skills": [ CatalogSkillWire, ... ]
}
```

`CatalogSkillWire` mirrors the engine `Skill` struct field-for-field, so a client hydrates its
`SkillCatalog` with no remapping (JSON Schema:
[`schema/catalog-skill.schema.json`](schema/catalog-skill.schema.json)):

```
{
  "id":          string,
  "name":        string,               // kebab, unique within the project's set
  "description": string,
  "tags":        string[],
  "tools":       string[],             // skill's referenced tool ids
  "metadata":    { [key: string]: string[] },
  "body":        string                // SKILL.md payload
}
```

v1 serves **skills only**. Tool definitions and authoring/write operations are not part of
this contract.

### Tool resolution (client-owned registry)

A synced skill's `tools` are tool **ids**, not definitions. The **tool registry is
client-owned** — tools are registered in-process by the SDK (e.g. from the client's own
upstream MCP servers). A referenced id resolves against that local registry; an id with no
local definition behaves as unknown-tool at invoke time, though the skill stays discoverable by
search. A source-published skill is therefore a coupling contract: it should reference only
tools the consuming client registers locally.

### ETag algorithm (frozen at v1)

The ETag is a SHA-256 over a **canonical serialization of the resolved published set for the
request's scope**. The steps are byte-exact so every conforming implementation produces the
identical ETag:

1. **Resolve** the set for the scope (see `scope` above): the global layer, or the subject
   layer overlaid on it.
2. **Project** each skill to exactly `{id, name, description, tags, tools, metadata, body}`.
   Every other field — timestamps, status, version, anything unrecognised — is dropped, so a
   byte-identical republish keeps the ETag stable.
3. **Canonicalize** each projected skill to JSON: the seven keys in the order above;
   `metadata` keys sorted ascending by UTF-8 byte order; `tags`, `tools`, and every `metadata`
   value array in **authored order** (order is significant); minimal JSON string escaping;
   non-ASCII emitted as raw UTF-8 (never `\u`-escaped); no insignificant whitespace.
4. **Sort** the projected skills by `id`, ascending by UTF-8 byte order, and join them into a
   compact JSON array (`[skill,skill,…]`, no whitespace).
5. `etag_hex = lowercase_hex(sha256(utf8_bytes(that array)))`. The `ETag` header is the strong
   tag `"<etag_hex>"`; the body's `catalogVersion` is the bare `<etag_hex>`.

An empty catalog hashes the two bytes `[]`. `If-None-Match` uses **weak comparison**
(RFC 7232 §3.2): tolerate a `W/` prefix, surrounding quotes, comma-lists, and `*` (which
matches any current representation).

Because the ETag is part of the compatibility surface, **this projection and serialization are
frozen at v1**: changing which fields are hashed, their order, the sort, or the escaping
invalidates every client's cache and is a breaking (v2) change. The vectors in
[`conformance/`](conformance/) pin the exact bytes.

## Operational endpoints

- `GET /healthz` — unauthenticated liveness probe for a networked source. Not under `/v1`,
  not versioned, returns `200` with no meaningful body.

## Error model

Every non-2xx `/v1` response carries a uniform body (JSON Schema:
[`schema/error.schema.json`](schema/error.schema.json)):

```
{ "error": { "code": string, "message": string, "details"?: object } }
```

v1 codes: `unauthorized` (401), `not_found` (404), `invalid_request` (400), and
`unavailable` (503).

## Conformance

The conformance vectors in [`conformance/vectors.json`](conformance/vectors.json) are part of
this contract: fixture catalogs paired with their expected ETag and resolved id-set, the
canonicalization invariants (field-order / metadata-key-sort / array-order / projection),
scope-overlay cases, `If-None-Match` / 304 semantics, and the secrets-never-sync field rule.
Every source and loader MUST reproduce them, so the contract — not any single implementation
— stays normative.
[`conformance/verify.mjs`](conformance/verify.mjs) is the reference implementation of the
algorithm and the vector runner (`node verify.mjs`); the JSON Schemas in [`schema/`](schema/)
pin the wire shapes.

## Versioning

- Major version is the URL prefix `/v1`. Additive changes (new optional fields, new
  endpoints) stay within `/v1`; clients MUST ignore unknown fields.
- The frozen v1 surface: the `Authorization: Bearer` scheme and 401/503 contract, the
  `GET /v1/catalog` shape, the `scope` selector semantics, `CatalogSkillWire`, and the ETag
  content projection. A break in any of these is a `/v2`.

## Explicit non-goals for v1

- **No remote search / invoke / get_skill over the wire.** The client runs the engine's
  retrieval and the capability tools locally over the pulled replica (ADR-0003).
- **No authoring / CRUD / publish / archive.** Mutation verbs are outside the v1 read
  contract.
- **No telemetry ingest.** Remote telemetry is stock OTLP `http/protobuf` + Bearer into a
  separate receiver ([ADR-0007](../../docs/adr/0007-telemetry-two-streams.md)), never
  this contract.
- **Secrets never appear on the wire.** OAuth tokens and API keys stay in the local secret
  stores; no field of any v1 shape can carry one (ADR-0003).
- **No loopback auth exception in v1.** Every `/v1` request to a networked source requires
  Bearer authentication; `/healthz` remains unauthenticated (ADR-0003).
- **No suggestion / analytics / ranking *endpoints*.** Those APIs are outside this contract;
  a conforming source is not expected to implement them (ADR-0003). The
  [intent graph](#intent-graph) is the one ranking *shape* published here, and it is a
  producer contract only — no endpoint serves it in v1.

## Intent graph

[`schema/intent-graph.schema.json`](schema/intent-graph.schema.json) specifies the
**usage-ranking read model** ([ADR-0014](../../docs/adr/0014-adaptive-usage-ranking.md)):
clusters of past queries, each carrying weighted edges to the capabilities users actually
invoked after them. A retrieval engine consumes it as an extra ranking arm beside BM25 and
dense retrieval.

It is specified here for one reason: **two independent producers must agree on it.** The
in-process local learner in `ratel-ai-core` grows one from the local trace stream; Ratel
Cloud builds one from hosted traces. Publishing the shape is what keeps either from
silently becoming the definition — the same reasoning the conformance vectors apply to the
catalog itself (ADR-0003).

It is **not** a synced wire artifact in v1:

- no endpoint serves it, and it has no ETag / conditional-GET semantics;
- it carries no secrets, and — like every v1 shape — no field of it can;
- `members` holds user query text, so a producer's transport and storage choices are
  governed by that, not by this contract.

Two properties matter to implementers. `members` is the **match key**, so a graph clustered
lexically (no `centroid`) is fully usable by a consumer that has an embedder, and vice
versa. And edge weights come from **invocations**, never from what retrieval returned —
recording a ranker's own output would reinforce its mistakes.

### Persistence & compatibility contract

The graph is held in memory; storing it is the caller's job. Three guarantees make that safe
to delegate:

- **Change-tracking.** The optional `rev` field is a monotonic write counter — bumped once per
  mutation, never read during ranking. A caller persists only when `rev` changed since the last
  save (**save-when-changed**), and before overwriting a stored graph compares its `rev` to the
  base it loaded — a higher value means another writer got there first (**stale-base
  detection**). Single-writer is the supported model; `rev` makes a clobber detectable, not
  merged. Absent `rev` is treated as 0.
- **Additive evolution.** Within `v: 1`, new fields are optional and a consumer **MUST ignore
  fields it does not recognize** (the schema is `additionalProperties: true`). An older
  consumer reads a newer graph; a graph missing a newer field (`model`, `last_ts`, `rev`) loads
  with a safe default.
- **Version safety.** A consumer **MUST reject an unrecognized `v`** with a clean error, never a
  crash or a silent degrade.

Structural fixtures live in [`conformance/vectors.json`](conformance/vectors.json) under
`graph`, with the consumer rules a conforming implementation MUST follow; `node
conformance/verify.mjs` checks them — including a graph carrying `rev`, a graph with unknown
extra fields (which MUST validate), and an unknown version (which MUST be rejected).
