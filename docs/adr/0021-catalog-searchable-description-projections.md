# 21. Catalog searchable-description projections and tool selection

Date: 2026-08-15

## Status

Accepted — experimental rollout

Extends ADR-0004. Its schema-inclusive tool projection remains the stable default.

## Context

Tool, skill, and fact retrieval originally coupled ranking text to the model-facing catalog
definition. In particular, tools indexed semantic tokens extracted from input and output
schemas. This made recall depend on schema wording and prevented a catalog operator from tuning
retrieval without changing what the model sees. Skills and facts already excluded their payloads,
but likewise had no independent ranking description.

The ranking projection is shared by BM25 and dense retrieval. Any override must therefore live at
that seam, preserve stable identity signals, and participate in embedding-cache invalidation.

## Decision

Each `Tool`, `Skill`, and `Fact` carries
`experimental_searchable_description: Option<String>`. Supplying it opts that entry into the
experimental projection, ordered as follows:

1. name, both whole and identifier-split;
2. the override verbatim;
3. tags, whole and identifier-split, for skills and facts.

An override replaces only the description component. Name and tags remain indexed, including
when the override is empty. For an opted-in tool, input and output schemas are model-facing and
are not indexed by either BM25 or dense retrieval. A tool without an override retains ADR-0004's
stable schema-inclusive projection; skills and facts continue to use their authored description.

The private per-catalog `searchable_text()` functions remain the single projection seam used by
BM25 documents and `Embeddable::embed_text`. The artifact `projection_version` hashes all three
projection source files, while each cached entry keeps a SHA-256 hash of its projected text.
Consequently, an override edit is catalog churn and invalidates that entry when its effective
projection changes; byte-identical entries retain their vectors.

Tool selection remains as decided in ADR-0004: `replace` is the default and `suggest` is opt-in.

## Consequences

- Retrieval can be tuned independently of the LLM-facing description, schemas, and payload.
- Existing callers retain schema-inclusive tool ranking. Only entries carrying the experimental
  override change projection. BFCL and SR-Agents impact is measured separately in `ratel-bench`
  before graduation.
- The experimental field may change or be removed without a major-version bump. Graduation adds
  the stable field while retaining the experimental spelling as a deprecated compatibility shim.
- Existing embedding artifacts self-invalidate when any catalog projection implementation changes.
- Rejected: overriding the entire projection. That would let optimization erase durable name and
  tag signals and make catalog entries undiscoverable by their identifiers or labels.
