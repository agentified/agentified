# 4. Retrieval: tool indexing and replace-vs-suggest selection

Date: 2026-07-05

## Status

Accepted

Compacted 2026-07 from pre-compaction ADR-0003 (tool selection, 2026-04-30) and ADR-0004
(tool indexing, 2026-04-30).

## Context

The engine ranks an agent's registered tools per query and returns the top-K. Two questions
stack: what text a tool contributes to the index (indexing raw schema JSON would drown the
meaningful tokens in structural noise), and how the ranked result reaches the model (replace
the tool list, or suggest alongside it).

## Decision

### Indexing

The durable decision is **what text represents a tool**, independent of the scorer that ranks
it:

- A private `searchable_text(&Tool) -> String` flattener emits **only semantic tokens**.
  Structural tokens (`type`, `required`, `$ref`, braces, quotes) are skipped, and
  schema-defined order is preserved via `serde_json`'s `preserve_order`, so the projection is
  deterministic.
  > **Superseded by [ADR-0021](0021-searchable-text-indexes-names-not-schema-prose.md).** This
  > ADR walked both schemas and emitted each property's key, **description** and **enum**
  > values. Those inflated parameter-heavy write operations past read operations that answered
  > the query; the projection now emits tool name, description, and input-schema property
  > **names** only.
- This flattened projection is the **contract**: telemetry, suggestions, and every retrieval
  layer build on it. Changing the flattening algorithm is a breaking change and warrants
  supersession — as ADR-0021 does.

The scorers over that projection are selectable while the projection stays stable:

- Current retrieval uses the [`bm25`](https://crates.io/crates/bm25) crate with its default
  English tokenizer, tuned `k1 = 0.9`, `b = 0.75` since ADR-0021 (`b` was 0.4 while the
  projection folded in whole schemas — below the crate defaults, because tool
  descriptions are short: term frequency saturates faster and length normalization matters
  less).
- Semantic retrieval embeds the same projection with a configurable in-process model or
  OpenAI-compatible endpoint; hybrid retrieval merges its ranking with BM25 through reciprocal
  rank fusion. See ADR-0011 and ADR-0012 for retrieval, cache, and async SDK semantics. The
  `searchable_text` contract above stays stable across scorer changes; rankers and their
  parameters are current tuning, not a frozen surface.

### Selection

- **`replace` is the default**: Ratel intercepts the tool list before the model call and
  replaces it with the ranked top-K. The token reduction is direct and attributable.
- **`suggest` is opt-in**: a ranked subset emitted alongside the existing tool list, for
  frameworks that own the list and offer no replace seam.
- Both modes are a configurable enum on the public API; each framework integration wires the
  chosen mode to the framework's actual tool-injection point.

## Consequences

- The tool author's vocabulary flows directly into recall: documentation quality is the
  retrieval lever, and the integrator's knob. Which parts — see
  [ADR-0021](0021-searchable-text-indexes-names-not-schema-prose.md); enum values and property
  descriptions no longer count, and `b` is 0.75.
- The current `k1`/`b` are fixed tuning, reproducible for today's benchmarks; they are not a
  public knob. Revisiting them or any dense-ranking detail does not disturb the
  `searchable_text` contract.
- Rejected: rolling our own lexical scorer (maintenance, no gain at this scope); indexing raw
  JSON (flattens term distribution, hurts IDF); a typed schema struct instead of `Value`
  (locks the public `Tool` shape at every framework boundary); `suggest` as default (ranking
  competes with framework-appended tools, savings become unattributable).
