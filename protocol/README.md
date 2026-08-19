# protocol/

`protocol/` defines the Ratel **catalog-source wire contract** design: the compatibility
surface a *catalog source* implements and an SDK *loader* consumes. A source serves a
published catalog; a loader pulls it and runs retrieval locally
([ADR-0003](../docs/adr/0003-catalog-source-interface.md)). It is a product surface, not a
code module, so it lives at the repo top level beside `src/` (per
[ADR-0002](../docs/adr/0002-product-split-engine-local-cloud.md)).

This directory is a language-agnostic specification, not an implementation. Its shapes, JSON
Schemas, and executable conformance vectors define how catalog sources and loaders interoperate;
loader and source implementations live outside `protocol/`.

## Layout

- `v1/` — the current contract: auth, catalog pull-sync, seven-field skill wire/ETag projection,
  error model, and intent-graph producer shape. It remains available unchanged.
- `v2/` — experimental preview: v1 plus optional `searchableDescription` and its
  eight-field ETag projection with a canonical `null` unset representation. It includes its
  own schemas and executable conformance vectors.

Both versions publish the `IntentGraph` usage-ranking shape — a producer contract shared by
the local learner and Ratel Cloud, not a synced endpoint
([ADR-0014](../docs/adr/0014-adaptive-usage-ranking.md)).

## Versioning

Each **major** version is a subfolder. Preview versions may change before graduation. Within a
stable version, additive changes — new optional fields, new endpoints,
new enum members — land **within** that major; a conforming client MUST ignore fields
it does not recognise. A change that alters an existing shape's meaning (including the ETag
content projection) is **breaking** and requires a new major. The frozen pieces of each major
are listed in that major's README.

The contract versions independently of any implementation, so a bump is a compatibility event
for every source and loader, not a release of a single package.
