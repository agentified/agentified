# 23. The searchable-text projection indexes names, not schema prose

Date: 2026-08-21

## Status

**Rejected — 2026-08-31.** Accepted, implemented, and reverted before it shipped in any release.

Measured on the 50-turn fixture, dropping property descriptions and enum values changed
**nothing**: top-1 against the invoked tool stayed 12 of 47 and read-queries-served-a-write-op
stayed 8 of 25 at `b = 0.4`, with the schema prose in or out. The fixture can see the change —
all 26 tools carry property descriptions and 9 carry enums — so this is a negative result, not an
absent test. End to end through all three arms it moved one query of 25 in the right direction,
which is inside the noise of a fixture this size.

The only evidence for it is experiment E2 in the Kestral misranking investigation: on *their*
catalog the create family rode 3–6 ranks on schema tokens alone. Real, but not reproducible here,
and not enough to justify changing the projection **for every catalog** — a breaking change that
alters ranking for anyone who upgrades.

[ADR-0021](0021-catalog-searchable-description-projections.md), which landed independently,
already addresses the same problem the other way: a single entry sets
`experimental_searchable_description` and opts out of schema indexing, per entry, opt-in, breaking
nobody. That is the better shape for a fix whose evidence comes from one catalog.

`b` was raised 0.4 → 0.75 alongside this and **was not reverted with it**, though its stated
justification was this decision. On the same fixture 0.75 leaves accuracy unchanged and raises
read-queries-served-a-write-op from 8 of 25 to 11. It stays only because 0.75 is the field
standard. See `harness-results.md`'s `b` sweep, and RS-95 for real corpora.

RS-95's SR-Agents run (`0.4.0` → `rc.2`, +0.082 recall@5) contains this `b` change but does not
isolate it — the same delta also contains score fusion
([ADR-0024](0024-hybrid-fuses-on-scores.md)). An `rc.1` run on that slice would separate them,
since rc.1 carries `b = 0.75` without score fusion. Until then `b` rests on the field standard
and a fixture that argues against it.

The record below is kept as written, so a future attempt inherits the argument and the
measurement that refused it rather than rediscovering both.

## Context

`searchable_text` flattened a tool into one bag: name, description, then — for **both** input
and output schema, recursively — every property name, every property **description**, and every
**enum** value. All weighted alike.

That makes a tool's argument list compete with its own description for what the tool *means*,
and the imbalance is structural rather than incidental. A create operation carries fifteen
parameters, a search operation two or three, so the write tool arrives at the index with a far
larger document composed mostly of words describing *arguments*.

BM25 penalises length, so each match in a big document is worth less. But fifteen extra tokens
buy fifteen extra chances to match *at all*, and a query word absent from a short description
often appears in a parameter name. With `b = 0.4` the discount did not offset the extra chances.

Experiment E2 of the 2026-08-11 misranking investigation measured it: emptying every input
schema dropped the create family **3–6 ranks** across BM25 and hybrid, while the search family
moved by at most +2. Its conclusion — *"schema tokens explain the create-family inflation,
descriptions explain the rest"* — names this as a distinct cause from the catalog's wording.

## Decision

The projection indexes **tool name, tool description, and input-schema property names**,
recursing through nested objects and array `items` for names only.

Dropped, and why each:

- **Property descriptions.** Prose written to help a model fill an argument in — "Which column
  to move the task to" — routinely longer than the tool's own description. It describes an
  argument, not a purpose.
- **Enum values.** Data, not meaning. `"done"` says nothing about what an operation is for, and
  a workflow tool winning a query on that token is an accident rather than a match.
- **The output schema, entirely.** It describes what comes *back*. A caller asking for something
  is not describing the shape of the answer.

Property **names** stay. `branch` genuinely helps match "tasks for a branch", and a name is the
one part of a schema written to be read as a word rather than as documentation.

`b` returns to the standard **0.75**. The 0.4 was chosen when a document was a description plus
its whole schema, where length mostly reflected how many arguments a tool took and penalising
that would have penalised the wrong thing. A document is now close to the description itself, so
its length carries information again.

## Consequences

- **This is what an integrator's vocabulary reaches recall through**, replacing ADR-0004's
  "description, parameter names, enum values": the description and the parameter names. Writing
  richer parameter docs no longer changes retrieval — the tool description is the lever.
- **The dense arm changes too, and by more than the lexical one is easy to reason about.**
  `Embeddable for Tool` embeds this same projection, so schema prose was in the vectors. E2 only
  measured BM25 and hybrid; it never isolated this. A tool with a terse description and a rich
  schema moves meaningfully in vector space.
- **Every prebuilt embedding artifact is stale.** `projection_version()` hashes this module's
  source, so it bumps automatically — but the warm path previously discarded it and reported the
  change as every entry missing, or silently re-embedded the catalog. It now fails with a
  `WarmError` naming the projection and the remedy.
- Skills and Facts are untouched: they carry no schema, and only share `push_identifier`.
- Measured on the 50-turn harness fixture, whose catalog gained realistic schemas for this work:
  read-phrased queries served a **write** operation at top-1 fell from **9 of 29 to 6 of 29**,
  and 34 of 47 served top-3 lists changed. Raising `b` afterwards moved 10 more lists without
  changing that count — kept because the justification for the old value no longer holds, not
  because it was measured to help.
- Rejected: **removing the schema entirely**, which E2 tested — it discards property names, the
  part that carries real signal, and would break matching a tool by an argument it takes.
  Rejected: **field weighting (BM25F)**, scoring name, description and parameters with separate
  weights. It is the principled answer and remains open, but scoring is delegated to the `bm25`
  crate over one flat string, and per-field length normalisation means owning the scorer.
