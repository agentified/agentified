# `docs/`

Project documentation that doesn't belong in a code folder.

## Layout

```
adr/                              Architecture decision records
assets/                           Images and other static assets
baseline-capture-distributed.md   Collecting seeding evidence from a multi-instance host
```

## Guides

[Baseline capture in a distributed host](baseline-capture-distributed.md) — how to collect the evidence for [ADR-0014](adr/0014-adaptive-usage-ranking.md) when the search and the invocation that follows land in different processes. The single-process path is [`examples/configurable-adaptive-ranking-ts`](../examples/configurable-adaptive-ranking-ts/README.md).

## `adr/` — Architecture decision records

The record of cross-cutting choices, kept **minimal and current**. Nygard format (`Status` / `Context` / `Decision` / `Consequences`), numbered sequentially (`NNNN-kebab-title.md`).

Amend in place for small drift; write a superseding ADR for real decision reversals; compact periodically — git history is the archive.

[ADR 0001](adr/0001-record-architecture-decisions.md) is the meta-ADR that carries the full convention. The `.adr-dir` file at the repo root points [adr-tools](https://github.com/npryce/adr-tools) here.
