# 23. The searchable-text projection indexes names, not schema prose

Date: 2026-08-21

## Status

Accepted

Supersedes the **projection** decision in
[ADR-0004](0004-retrieval-and-tool-selection.md); the rest of that ADR — the choice of BM25,
`replace` vs `suggest`, and `k1`/`b` as fixed tuning rather than a public knob — stands.
ADR-0004 asked for this: *"Changing the flattening algorithm is a breaking change and warrants
supersession."*

Composes with [ADR-0021](0021-catalog-searchable-description-projections.md), which landed
independently. That decision lets a single entry override the description component and, for a
tool, opt out of schema indexing entirely. This one changes what the **stable** projection
indexes for every entry that sets no override. The two do not overlap: after this, the default
already drops schema prose, so the override's remaining effect on a tool is to drop property
*names* as well and replace the description outright. An entry that sets the override is
unaffected by anything below.

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
