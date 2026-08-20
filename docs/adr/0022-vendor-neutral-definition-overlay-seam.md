# 22. Vendor-neutral definition-overlay seam

Date: 2026-08-19

## Status

Accepted — experimental rollout

Extends ADR-0021. The searchable-description projection is unchanged; this ADR decides who may
supply that description at runtime and how.

## Context

ADR-0021 made the retrieval description an independent, tunable field. Tuning it is only useful
if it can be changed without redeploying the host, which means some component outside the process
must be able to supply it. Ratel Cloud is the first such component, and the first implementation
of this seam was built and named for it: `attachCloudDefinitions`, `CloudDefinitions*` types, a
`useCloudDefinitions` flag, and the wire attribute `ratel.catalog.use_cloud_definitions`.

The mechanism itself never depended on Cloud. It is an injected `fetch(ifNoneMatch)` source with a
neutral payload; the SDK holds no URL, no credential, and no vendor-specific behavior. Only the
vocabulary was branded. Telemetry attribute names are public wire vocabulary consumed by any OTLP
backend, so that branding would have been unrenamable once the SDK reached GA.

## Decision

The overlay seam is vendor-neutral in name as well as in mechanism.

- The SDK exposes `ExperimentalDefinitionOverlaySource` — one method, `fetch(ifNoneMatch?)`, returning a `200`
  with a strong ETag and a complete `{ overrides: [...] }` body, or a `304`. Any implementation
  qualifies. `@ratel-ai/cloud-sdk` is the first one, not a privileged one; it keeps its own Cloud
  branding on its own facade.
- The SDK treats source responses as untrusted. It validates status, ETag, body, entry shape, and
  bounded string sizes before catalog mutation. Invalid input raises `DefinitionOverlayError`.
- One refresh has cross-catalog rollback. Tool, skill, and fact applications all settle before
  failure restores the last accepted maps; the ETag and one-way ownership latch advance only after
  all three succeed. Concurrent callers share one in-flight refresh.
- Opting in is the explicitly experimental
  `catalog.experimentalAttachDefinitionOverrides({ source })`. It is one-way for the life of the
  process; reverting means restarting the runtime.
- `applyDefinitionOverrides` is internal. It writes an override set straight to the live catalogs
  and so bypasses the conditional-request protocol that keeps an attachment's ETag honest; only
  the attach path and tests call it.
- Adoption is observable as `ratel.catalog.use_definition_overrides` on `ratel.catalog.definition`
  events, across all three telemetry packages. The attribute describes the mechanism, not its
  vendor.
- Overrides replace only the effective searchable description. Local registration remains the sole
  owner of the model-facing description, schemas, bodies, tags, and executors, and
  `catalog.snapshot()` stays local-only. One overlay is attached at a time; there is no merge order
  to define.

## Consequences

- A self-hosted, in-house, or third-party override service is a first-class citizen: implement the
  interface, no vendor emulation, no fork.
- The wire vocabulary is renamed before GA rather than after. Downstream consumers of the old
  attribute — the Cloud ingest key and its fixtures — move in lockstep with this change; there is
  no compatibility window, which is only affordable now because the SDK has not shipped it.
- A host cannot programmatically leave override ownership. This is deliberate: unwinding an
  overlay mid-process would mean restoring per-entry text the runtime no longer treats as
  authoritative, and a restart is the honest boundary.
- Rejected: keeping the Cloud vocabulary and documenting the seam as neutral. The names are what
  integrators read; branded names make an open seam look like a vendor hook, and the telemetry
  attribute would have been permanent.
- Rejected: multiple stacked overlays with precedence rules. No source needed it, and precedence
  is far harder to remove later than to add.
