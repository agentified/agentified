# 12. Configurable embedding models

Date: 2026-07-10

## Status

Accepted

Extends ADR-0011 (selectable retrieval methods), which fixed the semantic/hybrid
arms to a single hardcoded model. Builds on ADR-0006 (native FFI bindings) and
ADR-0007 (telemetry schema) for how the choice surfaces through the SDKs and traces.

## Context

ADR-0011 shipped semantic/hybrid retrieval against one pinned model,
`BAAI/bge-small-en-v1.5`, as two `const`s loaded through a process-global
singleton. Users cannot bring their own: a larger or multilingual HuggingFace
model, an air-gapped local checkpoint, or a model served over HTTP (OpenAI,
Ollama, TEI, vLLM). BM25 must stay model-free and the zero-config default must
stay byte-for-byte identical.

## Decision

**The embedding model is configurable per catalog, declared once at
construction, via an `EmbeddingModel` enum resolved from a cross-SDK spec.** The
model is used for _both_ document and query embedding, so the two sides can never
land in different vector spaces.

- **Four sources.** `Default` (the pinned bge-small, unchanged); `HuggingFace {
repo, revision }` and `Local { path }` — BERT-family models run in-process via
  Candle; `Endpoint { url, model, api_key_env }` — an OpenAI-compatible
  `/embeddings` HTTP call (OpenAI, **Ollama**, TEI, vLLM), the only path that
  supports non-BERT models. `revision` is optional (defaults to `main`).
- **Explicit source, no `kind` field.** A bare string is a **local model
  directory path**; every other source is an object whose _key_ names it —
  `{huggingface}` / `{local}` / `{ollama}` / `{url, model}` — symmetric across
  the board (a repo-id-looking or URL string is rejected with a pointer to the
  right object form, rather than guessed). Resolution and all validation live in
  **core** (`EmbeddingModel::resolve`), shared by both SDKs rather than
  duplicated; config errors surface at construction. Windows drive paths (`C:\…`)
  are never mistaken for URLs (the URL rule requires `://`).
- **Per-catalog, deduplicated by client identity.** The global one-model singleton
  becomes a process-wide cache keyed by model configuration. Endpoint client keys
  include the `api_key_env` _name_, never its secret value, preventing credential
  cross-talk. Vector identity is separate: URL, resolved response model, pooling,
  and prefixes define the vector space. A failed load is not cached (retries),
  preserving ADR-0011's non-poisoning contract.
- **Three correctness guards** against silently-wrong cosine results — the danger
  when vectors from different models/dimensions are ranked together, which does
  not error on its own. (1) **Normalize at the embedder boundary**: every vector
  from any source is L2-normalized before it reaches the cache or `dense_search`,
  so an endpoint returning un-normalized vectors can't break the cosine==dot
  assumption. (2) **Dimension mismatch is a hard `EmbedderError`** (the cache
  stamps its width; a query or doc of another width is rejected, never
  zip-truncated). (3) **Model-identity drift is a hard `ModelMismatch` error**:
  the cache stamps the fingerprint that built it and never mixes or queries a
  different vector space. The remediation is to re-embed the corpus against the new
  model — the core `rebuild_embeddings` primitive atomically replaces the full cache;
  an SDK caller constructs a fresh catalog and re-registers.
- **Endpoint specifics.** `api_key_env` names the env var holding the key (read
  at call time), keeping secrets out of code and serialized config; a
  named-but-unset var is a clear error, not a downstream 401. Document requests
  are chunked into at most 64 inputs and each response is capped at 64 MiB. The
  response indices must be the exact `0..n` permutation, and vectors must be
  present, finite, non-zero, and one dimension. Chunks preserve global order and
  commit only after all succeed. An optional response `model` becomes the resolved
  vector identity; drift is `ModelMismatch`. A 404 from localhost Ollama hints
  `ollama pull <model>`. The sync `ureq` client has a timeout and runs only on an
  SDK worker thread.
- **SDK dense work is asynchronous and folded into `register`.** A semantic/hybrid
  `register` embeds the batch off the host runtime (TypeScript via NAPI async tasks;
  Python via GIL-releasing native calls through `asyncio.to_thread`), so a model /
  endpoint failure surfaces from `register`; a BM25 catalog registers metadata only.
  Dense operations serialize per catalog, and a register during an active or queued
  dense operation fails promptly instead of blocking the event loop or GIL. Synchronous
  SDK search remains BM25-only.
- **Ratel auto-downloads only the built-in default.** An explicitly-configured
  HuggingFace model is **cache-only**: it must already be in the local HF cache,
  or the caller opts in with `download=true`. A missing one errors as
  `EmbedderError::NotCached` with a `huggingface-cli download …` hint — symmetric
  with Ollama's "not pulled", so a `register()` never silently pulls a multi-GB
  model on the user's behalf. When a download _does_ happen (the default, or an
  opt-in), a cold fetch emits a `TraceEvent::EmbedderDownload` with the **actual
  byte size** plus a stderr notice, so it is never silent.
- **Non-BERT in an in-process source** (`local`/`huggingface`) fails
  `BertModel::load` with a typed error that **signposts the endpoint/Ollama
  route** — turning a dead end into a pointer.

## Consequences

- The default path is unchanged and stays reproducible (SHA-pinned); BM25 remains
  model-free. A BM25-default catalog still validates its `embedding` config at
  construction (a malformed config errors immediately) even though it stays
  model-free. One new direct dependency, `ureq` (already in-tree via hf-hub, rustls
  — no new transitive cost, keeps ADR-0011's clean cross-platform wheels).
- BERT-family models run in-process (Candle's `bert`), pooled the way they were
  trained: **pooling (CLS/mean) is auto-detected** from the repo's
  `1_Pooling/config.json`, with a `pooling` override and a warn-then-assume-Mean
  fallback when a model ships no pooling metadata — so compatible BERT-family
  sentence-transformers models (bge, e5, gte, MiniLM) rank correctly, not just
  the CLS-pooled bge family. Asymmetric models are supported on both sides
  (`query_prefix` + `doc_prefix`), and weights load from `model.safetensors` or a
  `pytorch_model.bin` fallback. Every non-BERT model — nomic, Qwen-embed,
  GGUF-only — still runs via a local or hosted **endpoint**. We accept this rather
  than reintroduce an ONNX/C++ runtime, which would reverse ADR-0011's
  clean-wheels decision. MPNet-specific architecture/pooling corrections remain
  deferred; use an endpoint for MPNet models in the meantime.
- **Local model identity is content-derived.** A `Local` fingerprint hashes the
  files Candle loads (`config.json`, `tokenizer.json`, weights, and optional
  `1_Pooling/config.json`), not the directory path. Path-keyed identity breaks
  when the same weights are remounted at a different path (typical
  build-time → runtime / Docker); content identity is what a future on-disk
  embedding artifact needs to validate the model after that remount. Digests are
  memoized against file `(len, mtime)` so warm process-cache lookups do not
  re-read weight bytes.
- **Known limitation, addressed in [ADR-0017](0017-build-time-embedding-artifacts.md):** the embedding cache is in-process only — every process start re-embeds the corpus, cheap locally but costly over an endpoint. ADR-0016 adds a persistent build-time artifact keyed by the fingerprint already stamped on the cache. Also deferred: non-OpenAI endpoint request shapes and in-process GGUF/ONNX.
