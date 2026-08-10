# 17. Build-time embedding artifacts

Date: 2026-08-10

## Status

Accepted

Addresses the persistent on-disk embedding-cache follow-up from
[ADR-0012](0012-configurable-embedding-models.md).

## Context

Dense document embeddings normally live only in-process. Re-embedding the full
corpus on every cold start is cheap locally but costly over an endpoint. We need
an immutable build-time representation that can warm the runtime dense cache
when ids and searchable projections still match.

## Decision

Ship a build-time binary embedding artifact (RAT1). Hosts own persistence;
`ratel-ai-core` consumes and produces `Vec<u8>` and remains filesystem/I/O-free
on this surface. Embedding artifacts and model-weight caches are separate
lifecycles.

**Format.** Magic `RAT1`, `format_version` (`1`), `payload_len`, SHA-256 of
payload, then payload. Header: `projection_version`, `dim`,
`model_fingerprint`. Each entry: `kind` (Tool or Skill), stable `id`,
`projection_hash` (SHA-256 of projection text — text never stored),
L2-normalized vector. One RAT1 v1 file may mix Tool and Skill entries. Unknown
kind bytes are corrupt. Descriptions, bodies, executors, and credentials are
not stored.

**Warm.** A Tool registry reuses only Tool entries; a Skill registry only Skill
entries. Matching requires kind + id + `projection_hash == sha256(embed_text())`.
Fingerprint mismatch fails closed (zero commits). Artifact-only entries are
ignored, so a larger artifact can warm a subset corpus. Id+hash matching runs
before embedder resolve; an empty reuse set never loads a model.

**Build / merge.** Each registry builds a single-kind artifact (empty corpus →
valid empty RAT1, no embedder). `merge_embedding_artifacts` combines compatible
parts into one mixed RAT1 (shared format/projection version, fingerprint, and
dim; no duplicate `(kind, id)`). Malformed input is corruption;
valid-but-incompatible parts are `IncompatibleMerge`.

**Miss policy.** `OnArtifactMiss::Error` → `Incomplete { missing }`;
`Embed` → embed only uncached ids. Bytes are immutable build output (no
runtime write-through).

**TypeScript.** `experimentalBuildEmbeddingArtifact` builds BM25 metadata
registries, embeds once per side, merges, and writes the file. Runtime
`experimentalEmbeddingArtifact: { path } | { bytes }` (default
`onMiss: "error"`) is warmed on registration **before** eager document
embedding.

**Python.** `experimental_build_embedding_artifact` mirrors that build helper
(snake_case). Runtime `experimental_embedding_artifact` on registry/catalog
constructors warms before eager document embedding; default `on_miss` is
`"error"`. PyO3 maps core artifact errors directly (no NAPI JSON envelope).

## Consequences

Unchanged catalog entries can skip document inference at cold start.
Changed or missing projections follow the configured miss policy. Semantic
search still requires query embedding against the warmed or built dense cache.
