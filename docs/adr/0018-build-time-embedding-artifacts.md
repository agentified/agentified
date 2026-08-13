# 18. Build-time embedding artifacts

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

Ship a build-time binary embedding artifact (RAT1). Artifact persistence
remains host-owned; the core artifact APIs accept/return bytes and perform no
artifact filesystem I/O. Embedding artifacts and model-weight caches are
separate lifecycles.

**Format.** Magic `RAT1`, `format_version` (`1`), `payload_len`, SHA-256 of
payload, then payload. The checksum provides integrity/corruption detection
only — not authentication, signing, or semantic validity by itself. After
checksum verification, the core applies canonical semantic validation before
merge or cache mutation: a non-empty artifact requires `dim > 0`; vector widths
must match `dim`; values must be finite; vectors must be unit-normalized using
the same squared-L2-norm tolerance as the dense cache (not a tolerance on the
vector length itself). Malformed checksum-valid bytes are rejected. Canonical
builder output for an empty artifact: `entry_count = 0`, `dim = 0`. Decoding
does not newly reject an otherwise-valid empty RAT1 solely because
`entry_count = 0` and `dim > 0`. Builders enforce the same vector semantic
invariants before serialization. Header: `projection_version`, `dim`,
`model_fingerprint`. Each entry: `kind` (Tool or Skill), stable `id`,
`projection_hash` (SHA-256 of projection text — text never stored),
L2-normalized vector. One RAT1 v1 file may mix Tool and Skill entries.
Unknown kind bytes are corrupt. Descriptions, bodies, executors, and credentials are not stored.

**Local model identity (artifact vs runtime).** RAT1 header `model_fingerprint`
for `Local` uses the artifact compatibility digest (see ADR-0012); runtime
dense-cache identity for `Local` remains path-based. Hashing happens lazily
only when RAT1 build or warm requires it; no persistent digest store;
`(len, mtime)` memo is a process-local accelerator only. Once a digest has
been established for the resident model, stamp drift triggers digest
recomputation and comparison rather than failing on metadata drift alone. An
equal digest is accepted and refreshes the stamps; different content, or stamp
drift before any digest was established, remains fail-closed.

**Warm.** A Tool registry reuses only Tool entries; a Skill registry only Skill
entries. Matching requires kind + id + `projection_hash == sha256(embed_text())`.
Fingerprint mismatch fails closed (zero commits). Artifact-only entries are
ignored, so a larger artifact can warm a subset corpus. Other-kind entries do
not satisfy missing entries for the active kind. The same textual id across Tool
and Skill kinds is allowed. Id+hash matching runs before embedder resolution.
If no entries are reusable, the matching phase returns without resolving the
embedder. With the fail-closed `Error` miss policy this becomes `Incomplete`;
with `Embed`, registry-level follow-up embedding still uses the configured
embedder for missing entries. When at least one entry is reusable, warm resolves
the configured embedder to compare model identity. For `Local`, this includes
model initialization/loading during registration. RAT1 removes corpus/document
inference for covered entries, not compatibility resolution or fallback
embedding costs.

**Build / merge.** Each registry builds a single-kind artifact (empty corpus →
valid empty RAT1, no embedder). `merge_embedding_artifacts` is a public Rust
core bytes-only primitive that combines compatible parts into one mixed RAT1
(shared format/projection version, fingerprint, and dim; no duplicate
`(kind, id)`). SDK high-level mixed builders (`experimentalBuildEmbeddingArtifact`
/ `experimental_build_embedding_artifact`) build a Tool half and a Skill half
and merge internally — merge is not a public SDK API. Malformed input is
corruption; valid-but-incompatible parts are `IncompatibleMerge`.

**Coverage / miss policy.** Default `onMiss` / `on_miss` is `"error"`. With
that default, every id in each non-empty registering corpus must be covered. A
tool-only artifact is valid while the Skill corpus stays empty (and vice versa).
When both sides register, use a mixed artifact or `onMiss` / `on_miss`
`"embed"` to fill uncovered current entries. `OnArtifactMiss::Error` →
`Incomplete { missing }`; `Embed` → embed only uncached ids. Bytes are immutable
build output (no runtime write-through).

**Lifecycle.** Every `register` / `replaceAll` / `replace_all` re-resolves the
configured artifact source (path-backed config re-reads current bytes),
re-warms against the whole current corpus, and applies the current miss policy.
No resolve-once memoization, path-byte cache, or already-warmed flag.

**TypeScript.** `experimentalBuildEmbeddingArtifact` builds BM25 metadata
registries, embeds once per side, merges internally, and writes the file.
`ToolRegistry` / `SkillRegistry` expose
`experimentalBuildEmbeddingArtifact` and
`experimentalWarmEmbeddingsFromArtifact`. Runtime
`experimentalEmbeddingArtifact: { path } | { bytes }` (default
`onMiss: "error"`) is warmed on registration **before** eager document
embedding. Public errors include `ArtifactError`, `IncompatibleMergeError`
(may surface from the high-level mixed builder's internal Tool+Skill
composition), and `ArtifactWarmError`.

**Python.** `experimental_build_embedding_artifact` mirrors that build helper
(snake_case). Registries expose `experimental_build_embedding_artifact` and
`experimental_warm_embeddings_from_artifact`. Runtime
`experimental_embedding_artifact` on registry/catalog constructors warms before
eager document embedding; default `on_miss` is `"error"`. PyO3 maps core
artifact errors directly (no NAPI JSON envelope). Public errors include
`ArtifactError`, `IncompatibleMergeError`, and `ArtifactWarmError` (`.code`,
`.missing`; no JS-style `.name` — use `type(err).__name__`).

## Consequences

Covered artifact entries avoid corpus/document embedding inference at cold
start. When warm reuses at least one entry, it resolves the configured embedder
during registration to compare model identity (for `Local`, including model
initialization/loading). Changed or missing projections follow the configured
miss policy. Semantic/hybrid search still requires query embedding through the
configured embedding backend against the warmed or built dense cache; Local/HF
backends may therefore still perform query-time model use, and endpoint
continues to perform its normal remote query-embedding operation. RAT1 removes
corpus/document inference; it does not eliminate runtime query embedding or the
compatibility resolution performed by warm.
