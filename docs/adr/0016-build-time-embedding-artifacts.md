# 16. Build-time embedding artifacts

Date: 2026-08-10

## Status

Accepted

Addresses the persistent on-disk embedding-cache follow-up from
[ADR-0012](0012-configurable-embedding-models.md).

## Context

ADR-0012 stamps a resolved `model_fingerprint` on dense-cache vectors so model
drift is a hard error, but the cache is in-process: every cold start re-embeds
the catalog — cheap locally, costly over an endpoint.

## Decision

Ship a build-time binary artifact (CI embeds once; runtime warms the cache when
ids and projection text still match). Hosts own persistence; the core speaks
only `Vec<u8>`.

### Format (RAT1)

Magic `RAT1`, `format_version`, `payload_len`, SHA-256 of payload, then payload.
Header: `projection_version`, `dim`, `model_fingerprint`. Entry: `kind`, `id`,
`projection_hash` (SHA-256 of projection text — text never stored), L2-normalized
vector. One kind per file; no descriptions, bodies, or credentials.

### Invalidation

Reuse only on matching `id`, expected `kind`, and
`projection_hash == sha256(embed_text())`. Fingerprint mismatch or unexpected
kind fails closed (zero commits). Ids absent or hash-mismatched are `missing`;
artifact-only ids are ignored. Id+hash matching runs before any embedder resolve;
an empty reuse set never loads a model.

Endpoint: compare the static client fingerprint first; probe with one input only
on mismatch (Local/HF identity is already resolved by load).

### API and layering

- `build_embedding_artifact() -> Result<Vec<u8>, ArtifactError>` — no
  `operation_lock` (would stall concurrent semantic search for the whole embed). Empty corpus skips embedder resolve.
- `warm_embeddings_from_artifact(bytes, OnArtifactMiss::{Error, Embed})` —
  `Error` → `Incomplete { missing }`; `Embed` → `build_embeddings()` (extends
  only uncached ids).
- Embedder resolution stays inside `DenseCache`. Wire format:
  `embedding_artifact.rs`. Cross-registry policy/errors: `artifact_warm.rs`
  (same role as `method.rs` for `SearchMethod`).

## Consequences

Cold starts skip inference for unchanged ids. Projection or text changes force
re-embed under `Embed`. Core stays I/O-free on this surface.

## Rejected

- **Post-registration import seam** — would duplicate `extend`'s id-keyed commit.
- **Write-through mutable artifact** — bytes are immutable build output.
- **Remote-storage adapters in core** — host concern.
- **Weights + vectors in one file** — unrelated lifecycles; identity is already
  content-derived (ADR-0012).
