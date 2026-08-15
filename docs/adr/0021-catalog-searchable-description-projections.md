# 21. Catalog searchable-description projections and tool selection

Date: 2026-08-15

## Status

Accepted

Supersedes ADR-0004. Its tool-selection decision is retained; its schema-inclusive tool
projection is replaced.

## Context

Tool, skill, and fact retrieval originally coupled ranking text to the model-facing catalog
definition. In particular, tools indexed semantic tokens extracted from input and output
schemas. This made recall depend on schema wording and prevented a catalog operator from tuning
retrieval without changing what the model sees. Skills and facts already excluded their payloads,
but likewise had no independent ranking description.

The ranking projection is shared by BM25 and dense retrieval. Any override must therefore live at
that seam, preserve stable identity signals, and participate in embedding-cache invalidation.

## Decision

Each `Tool`, `Skill`, and `Fact` carries `searchable_description: Option<String>`. Its retrieval
projection is deterministic and ordered as follows:

1. name, both whole and identifier-split;
2. effective searchable description — `description` when the override is `None`, otherwise the
   override verbatim;
3. tags, whole and identifier-split, for skills and facts.

An override replaces only the description component. Name and tags remain indexed, including
when the override is empty. Tool input and output schemas are model-facing and are no longer
indexed by either BM25 or dense retrieval.

The private per-catalog `searchable_text()` functions remain the single projection seam used by
BM25 documents and `Embeddable::embed_text`. The artifact `projection_version` hashes all three
projection source files, while each cached entry keeps a SHA-256 hash of its projected text.
Consequently, an override edit is catalog churn and invalidates that entry when its effective
projection changes; byte-identical entries retain their vectors.

Tool selection remains as decided in ADR-0004: `replace` is the default and `suggest` is opt-in.

## Consequences

- Retrieval can be tuned independently of the LLM-facing description, schemas, and payload.
- The default tool ranking behavior changes because schema-only terms stop matching. This is a
  breaking ranking change at 0.x and requires a minor core/SDK release. BFCL and SR-Agents impact
  is measured separately in `ratel-bench`; numbers follow this decision rather than gate it.
- Existing embedding artifacts self-invalidate when any catalog projection implementation changes.
- Rejected: overriding the entire projection. That would let optimization erase durable name and
  tag signals and make catalog entries undiscoverable by their identifiers or labels.
