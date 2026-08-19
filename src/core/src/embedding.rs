//! Dense embedders behind the semantic/hybrid engines. The model is configurable
//! per catalog (see [`crate::embedding_config`] and ADR-0012); this module owns
//! the two backends and the process-wide, identity-keyed embedder cache.
//!
//! - [`CandleEmbedder`] runs a **BERT-family** model in-process via Candle — the
//!   built-in default (`bge-small-en-v1.5`), any HuggingFace repo, or an on-disk
//!   directory. Pure-Rust inference (no C++/ONNX native dep) keeps the SDK
//!   wheels/addons clean cross-platform. Pooling (CLS or mean) is auto-detected
//!   from the model's `1_Pooling/config.json` (overridable, warn-then-assume-mean
//!   when absent), then we L2-normalize, so cosine similarity is a dot product
//!   (see [`crate::dense_search`]). It supports asymmetric models on both sides —
//!   an optional query prefix and doc prefix. Weights load from
//!   `model.safetensors` or a `pytorch_model.bin` fallback.
//! - [`EndpointEmbedder`] calls an OpenAI-compatible `/embeddings` HTTP endpoint
//!   (OpenAI, Ollama, TEI, vLLM…) — any model, including non-BERT ones. Returned
//!   vectors are **re-normalized** on ingestion (an arbitrary endpoint may not
//!   normalize), so `dense_search`'s unit-vector assumption always holds.
//!
//! In-process weights are **not** bundled and live in the shared HuggingFace
//! cache (`~/.cache/huggingface`, `HF_HOME`-overridable). Ratel **auto-downloads
//! only the built-in default**; an explicit HuggingFace model is **cache-only**
//! (it must already be present, or opt in with `download=true`) — a missing one
//! errors as [`EmbedderError::NotCached`], symmetric with Ollama's "not pulled",
//! so an explicit embedding build never silently pulls a multi-GB model. Once cached, load is
//! offline and deterministic when the revision is pinned. Two catalogs on the
//! same model share one resident embedder (keyed by model fingerprint); a cold
//! download emits a [`TraceEvent::EmbedderDownload`].
//!
//! **Footprint & failure modes.** A resident BERT model is ~130 MB+ of f32
//! weights plus runtime buffers, and inference is CPU-only, so a constrained
//! machine may load or embed slowly — surfaced as a [`TraceEvent::EmbedderLoad`]
//! with status `slow` — or, if it runs out of memory, be killed by the OS (an
//! uncatchable SIGKILL, nothing we can flag). Load and inference are otherwise
//! **fallible**: a failure returns a typed [`EmbedderError`] (network, unwritable
//! cache, corrupt weights, inference, dimension mismatch, config) rather than
//! aborting the process, and a failed load is **not cached**, so a later call
//! retries once the cause clears.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::embedding_config::{
    DEFAULT_REPO, DEFAULT_REVISION, EmbeddingModel, LocalContentIdentity, OLLAMA_DEFAULT_URL,
    Pooling, endpoint_fingerprint, fingerprint_suffix, huggingface_fingerprint,
    local_content_fingerprint, local_fingerprint, parse_pooling_config, resolve_local_model_files,
    stamp_local_hash_paths,
};
use crate::trace::{EmbedderLoadStatus, TraceEvent, TraceSink};

/// HTTP timeout for a single endpoint embedding request, so a stalled endpoint
/// can't hang an embedding build/query forever.
const ENDPOINT_TIMEOUT_SECS: u64 = 30;

/// Maximum number of inputs sent in one OpenAI-compatible embeddings request.
const ENDPOINT_BATCH_SIZE: usize = 64;

/// Maximum response body accepted for one endpoint chunk.
const ENDPOINT_RESPONSE_LIMIT_BYTES: u64 = 64 * 1024 * 1024;

/// Default cold-load latency (ms) above which the load is flagged `slow`, a hint
/// that the machine may be underpowered. Override with `RATEL_EMBED_SLOW_MS`.
const DEFAULT_SLOW_LOAD_MS: u64 = 5_000;

/// Human-readable reason attached to a `slow` load event.
const SLOW_LOAD_REASON: &str = "embedding model load was slow — this machine may be underpowered \
     for in-process CPU inference; expect slow embedding builds and queries";

/// A recoverable embedder failure. Returned instead of panicking so a load or
/// inference problem surfaces to the SDK as a **catchable** error (with a
/// remediation hint in `Display`) rather than aborting the host process.
#[derive(Debug, Clone)]
pub enum EmbedderError {
    /// A configured embedding source could not be reached: offline, DNS/TLS,
    /// timeout, or an endpoint/model request returned a 4xx.
    Download {
        /// The configured model/source display name.
        model: String,
        /// The underlying fetch error.
        source: String,
    },
    /// The HuggingFace cache could not be written: permissions, disk full, or a
    /// read-only filesystem.
    CacheUnwritable {
        /// The underlying filesystem error.
        source: String,
    },
    /// Model files are unusable: missing or corrupt weights, or a
    /// config/tokenizer that failed to parse.
    Load {
        /// The configured model/source display name.
        model: String,
        /// The underlying load error.
        source: String,
    },
    /// Embedding a specific text failed (tokenization or the forward pass).
    Inference {
        /// The underlying tokenizer/inference error.
        source: String,
    },
    /// A semantic/hybrid search was requested but the embedding cache is not
    /// built for the current corpus — `build_embeddings` was never run. No model
    /// is loaded; the caller must build the embeddings first.
    EmbeddingsNotBuilt,
    /// A vector's dimension does not match the embedding cache's. Cosine over
    /// mismatched dimensions is silently wrong, so this is a hard error — raised
    /// when a query (or an endpoint's response) has a different width than the
    /// vectors the cache was built with.
    DimensionMismatch {
        /// Vector width the cache was built with.
        expected: usize,
        /// Vector width actually seen.
        got: usize,
        /// The active model, named for diagnosis.
        model: String,
    },
    /// A vector was produced by a different resolved model than the vectors
    /// already in the cache. Mixing vector spaces is never safe; callers must
    /// explicitly rebuild the full corpus to adopt the active model.
    ModelMismatch {
        /// Resolved model identity that built the cache.
        built: String,
        /// Resolved model identity returned by the active embedder.
        active: String,
    },
    /// The embedding configuration is invalid — a bad source combination, a
    /// missing required field, or a named `api_key_env` that is not set. Surfaced
    /// at catalog construction or on first use.
    Config {
        /// What is wrong with the configuration.
        message: String,
    },
    /// An explicitly-configured HuggingFace model is not in the local cache, and
    /// Ratel auto-downloads only the built-in default. The user must fetch it
    /// first — symmetric with Ollama's "model not pulled".
    NotCached {
        /// The repo id that is not cached.
        model: String,
        /// The requested revision, if pinned.
        revision: Option<String>,
    },
}

impl EmbedderError {
    /// One-line remediation hint, embedded in the `Display` message.
    fn hint(&self) -> &'static str {
        match self {
            EmbedderError::Download { .. } => {
                "check source availability, connectivity, and the configured model identifier"
            }
            EmbedderError::CacheUnwritable { .. } => {
                "check ~/.cache/huggingface permissions and free disk space (or set HF_HOME)"
            }
            EmbedderError::Load { .. } => {
                "check that model files and configuration are present, readable, and compatible"
            }
            EmbedderError::Inference { .. } => {
                "check model input/configuration or the endpoint response and retry"
            }
            EmbedderError::EmbeddingsNotBuilt => {
                "embed the corpus before running a semantic/hybrid search"
            }
            EmbedderError::DimensionMismatch { .. } => {
                "the configured model changed; re-embed the corpus with the new model"
            }
            EmbedderError::ModelMismatch { .. } => {
                "the configured model changed; re-embed the corpus with the new model"
            }
            EmbedderError::Config { .. } => {
                "give exactly one embedding source (a model id / path / url) with its required fields"
            }
            EmbedderError::NotCached { .. } => {
                "Ratel auto-downloads only the default model; pre-download this one, pass download=true, or use a local path / endpoint"
            }
        }
    }
}

impl std::fmt::Display for EmbedderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hint = self.hint();
        match self {
            EmbedderError::Download { model, source } => write!(
                f,
                "failed to download embedding model {model}: {source} (hint: {hint})"
            ),
            EmbedderError::CacheUnwritable { source } => write!(
                f,
                "embedding model cache is not writable: {source} (hint: {hint})"
            ),
            EmbedderError::Load { model, source } => write!(
                f,
                "failed to load embedding model {model}: {source} (hint: {hint})"
            ),
            EmbedderError::Inference { source } => {
                write!(f, "embedding failed: {source} (hint: {hint})")
            }
            EmbedderError::DimensionMismatch {
                expected,
                got,
                model,
            } => write!(
                f,
                "embedding dimension mismatch for {model}: expected {expected}, got {got} (hint: {hint})"
            ),
            EmbedderError::ModelMismatch { built, active } => write!(
                f,
                "embedding model mismatch: cache was built with {built}, active model is {active} (hint: {hint})"
            ),
            EmbedderError::Config { message } => write!(f, "{message} (hint: {hint})"),
            EmbedderError::NotCached { model, revision } => {
                let rev = revision
                    .as_deref()
                    .map(|r| format!(" --revision {r}"))
                    .unwrap_or_default();
                write!(
                    f,
                    "embedding model {model} is not in the local HuggingFace cache — download it \
                     first: `huggingface-cli download {model}{rev}` (hint: {hint})"
                )
            }
            EmbedderError::EmbeddingsNotBuilt => {
                write!(
                    f,
                    "embeddings are not computed for semantic search (hint: {hint})"
                )
            }
        }
    }
}

impl std::error::Error for EmbedderError {}

/// An embedding value paired with the resolved identity of the model that
/// produced it. Carrying identity in the result avoids races when one process-
/// cached endpoint embedder is shared by concurrent catalogs.
pub(crate) struct Embedded<T> {
    pub(crate) value: T,
    pub(crate) fingerprint: String,
}

/// Maps a tool's searchable text (and a query) to an L2-normalized vector.
/// A trait so the model is swappable — a HuggingFace/local BERT model or a
/// remote endpoint can back the same registries without touching them.
pub(crate) trait Embedder: Send + Sync {
    fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;

    /// Embed a batch of documents. The default loops `embed_doc` (fine for an
    /// in-process model); an endpoint embedder overrides it with ordered HTTP
    /// chunks, since per-document round-trips would be pathological.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        texts.iter().map(|t| self.embed_doc(t)).collect()
    }

    /// Embed a query and return the vector with its resolved model identity.
    /// Fixed-identity embedders use [`Self::fingerprint`]; endpoint embedders
    /// override this to carry the response's optional resolved `model`.
    fn embed_query_with_identity(&self, text: &str) -> Result<Embedded<Vec<f32>>, EmbedderError> {
        Ok(Embedded {
            value: self.embed_query(text)?,
            fingerprint: self.fingerprint(),
        })
    }

    /// Embed documents and return the complete batch with its resolved model
    /// identity. The cache validates and commits this result atomically.
    fn embed_batch_with_identity(
        &self,
        texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        Ok(Embedded {
            value: self.embed_batch(texts)?,
            fingerprint: self.fingerprint(),
        })
    }

    /// Embed documents and return the batch with the identity stamped into a
    /// RAT1 artifact header. The default preserves [`Self::embed_batch_with_identity`]
    /// (Endpoint response-resolved model, HF/Default runtime fingerprint). Local
    /// Candle overrides this to substitute the portable content-derived identity.
    fn embed_batch_with_artifact_identity(
        &self,
        texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        self.embed_batch_with_identity(texts)
    }

    /// Runtime model identity (concrete HF revision/SHA, local path, or endpoint
    /// URL + model), encoded as collision-proof length-delimited fields. It is
    /// stamped on the dense cache so a later model swap over an existing
    /// embedding set is detectable. Test stubs use the default.
    fn fingerprint(&self) -> String {
        "unknown".to_string()
    }

    /// RAT1 artifact / warm-compare identity. Default equals [`Self::fingerprint`].
    /// Local Candle computes a content digest lazily (only when RAT1 needs it).
    fn artifact_identity(&self) -> Result<String, EmbedderError> {
        Ok(self.fingerprint())
    }
}

/// A one-time notice that the embedding model was actually downloaded (cold HF
/// cache), carrying the real byte size — surfaced so a multi-second first-run
/// fetch is never a silent surprise.
struct DownloadNotice {
    model: String,
    bytes: u64,
}

/// One-time notices produced by a cold load, surfaced by the telemetry layer.
#[derive(Default)]
struct LoadNotices {
    /// Set when a cold HF fetch actually downloaded weights.
    download: Option<DownloadNotice>,
    /// The model name, set when pooling could not be detected and Mean was
    /// assumed — so the guess is never silent (the user can set `pooling`).
    pooling_assumed: Option<String>,
}

/// Process-wide embedder cache, **keyed by model identity**, loaded once per
/// model on first use. Two catalogs on the same model share one resident
/// embedder; different models coexist. A failed load is **not** cached (the key
/// stays empty), so a later call retries — a transient network blip must not
/// poison a model for the whole process. Returns the embedder, the cold-load
/// latency (`Some` only on the loading call), and a download notice (`Some` only
/// when a cold HF fetch actually downloaded).
/// A resolved embedder plus one-time cold-load telemetry: the load latency
/// (`Some` on the loading call) and any cold-load notices (download / assumed
/// pooling).
type ResolvedEmbedder = (Arc<dyn Embedder>, Option<u64>, LoadNotices);

fn embedder_for(model: &EmbeddingModel) -> Result<ResolvedEmbedder, EmbedderError> {
    static CELL: OnceLock<Mutex<HashMap<String, LoadSlot<dyn Embedder>>>> = OnceLock::new();
    let cache = CELL.get_or_init(|| Mutex::new(HashMap::new()));
    let mut notices = LoadNotices::default();
    let (emb, load_ms) = get_or_load_keyed(cache, &model.embedder_cache_key(), || {
        let (emb, n) = build_embedder(model)?;
        notices = n;
        Ok(emb)
    })?;
    Ok((emb, load_ms, notices))
}

/// Construct the embedder for a model (may hit the network/disk). Threads the
/// query/doc prefixes and pooling override in, and returns cold-load notices.
fn build_embedder(
    model: &EmbeddingModel,
) -> Result<(Arc<dyn Embedder>, LoadNotices), EmbedderError> {
    let query_prefix = model.query_prefix();
    let doc_prefix = model.doc_prefix();
    let pooling = model.pooling_override();
    match model {
        EmbeddingModel::Default => {
            // The built-in default is the one model Ratel always auto-downloads.
            let (e, n) = CandleEmbedder::load_hf(
                DEFAULT_REPO,
                DEFAULT_REVISION,
                query_prefix,
                doc_prefix,
                pooling,
                true,
            )?;
            Ok((Arc::new(e), n))
        }
        EmbeddingModel::HuggingFace {
            repo,
            revision,
            download,
            ..
        } => {
            let (e, n) = CandleEmbedder::load_hf(
                repo,
                revision.as_deref().unwrap_or("main"),
                query_prefix,
                doc_prefix,
                pooling,
                *download,
            )?;
            Ok((Arc::new(e), n))
        }
        EmbeddingModel::Local { path, .. } => {
            let (e, n) = CandleEmbedder::load_path(path, query_prefix, doc_prefix, pooling)?;
            Ok((Arc::new(e), n))
        }
        EmbeddingModel::Endpoint {
            url,
            model,
            api_key_env,
            ..
        } => {
            let e = EndpointEmbedder::new(
                url.clone(),
                model.clone(),
                api_key_env.clone(),
                query_prefix.into(),
                doc_prefix.into(),
            )?;
            Ok((Arc::new(e), LoadNotices::default()))
        }
    }
}

/// A per-key load slot in a keyed cache: `None` until the value is loaded, then
/// `Some`. Guarded by its own mutex so a `load` serializes callers of *that* key
/// without holding the whole-map lock.
type LoadSlot<T> = Arc<Mutex<Option<Arc<T>>>>;

/// Get-or-load into a keyed cache with **no failure caching**: on `Err` the key's
/// slot stays empty so a later call retries. Returns the value plus `Some(load_ms)`
/// on the call that performed the load, `None` on warm reuse. Generic so the
/// non-poisoning + once contract is unit-tested without touching the network.
///
/// The map lock is held only long enough to get/create the key's slot; `load`
/// runs against that per-key slot mutex, never the map. So **different keys load
/// concurrently** (a cold load of one model no longer blocks another), while
/// **same-key loads stay single-flight** (concurrent cold misses share the slot,
/// so `load` runs once and exactly one caller reports `Some(load_ms)`).
fn get_or_load_keyed<T: ?Sized>(
    cache: &Mutex<HashMap<String, LoadSlot<T>>>,
    key: &str,
    load: impl FnOnce() -> Result<Arc<T>, EmbedderError>,
) -> Result<(Arc<T>, Option<u64>), EmbedderError> {
    // Hold the map lock only to get/create this key's slot, then release it.
    let slot = {
        let mut guard = cache.lock().expect("embedder cache mutex poisoned");
        Arc::clone(guard.entry(key.to_string()).or_default())
    };
    // Serialize per key on the slot — the map lock is NOT held across `load`.
    let mut slot = slot.lock().expect("embedder cache slot mutex poisoned");
    if let Some(existing) = slot.as_ref() {
        return Ok((existing.clone(), None));
    }
    let started = Instant::now();
    let loaded = load()?; // Err leaves the slot `None`, so a later call retries.
    let took_ms = started.elapsed().as_millis() as u64;
    *slot = Some(loaded.clone());
    Ok((loaded, Some(took_ms)))
}

/// Resolve the process embedder and record the one-time load-telemetry event on
/// `sink` (a slow/failed load is also logged to stderr). Registries call this so
/// the [`TraceEvent::EmbedderLoad`] flag is emitted from the layer that owns a
/// sink; the embedder itself stays sink-agnostic.
pub(crate) fn embedder_with_telemetry(
    model: &EmbeddingModel,
    sink: &dyn TraceSink,
) -> Result<Arc<dyn Embedder>, EmbedderError> {
    let display = model.display_name();
    let (result, load_ms, notices) = match embedder_for(model) {
        Ok((emb, ms, notices)) => (Ok(emb), ms, notices),
        Err(e) => (Err(e), None, LoadNotices::default()),
    };
    if let Some(DownloadNotice { model, bytes }) = notices.download {
        let mb = bytes as f64 / 1_048_576.0;
        eprintln!("ratel: downloaded embedding model {model} ({mb:.0} MB, one-time)");
        sink.record(TraceEvent::EmbedderDownload { model, bytes });
    }
    if let Some(model) = notices.pooling_assumed {
        eprintln!(
            "ratel: pooling not detected for {model}, assuming mean; \
             set pooling=\"cls\"|\"mean\" to override"
        );
        sink.record(TraceEvent::EmbedderPoolingAssumed {
            model,
            pooling: "mean".to_string(),
        });
    }
    if let Some(event) = embedder_load_event(&display, load_ms, result.as_ref().err()) {
        if let TraceEvent::EmbedderLoad {
            status,
            took_ms,
            reason,
            ..
        } = &event
            && !matches!(status, EmbedderLoadStatus::Ok)
        {
            eprintln!(
                "ratel: embedding model load {status:?} ({took_ms}ms): {}",
                reason.as_deref().unwrap_or("")
            );
        }
        sink.record(event);
    }
    result
}

/// Decide the load-telemetry event for a cold-load outcome. `None` on warm reuse
/// (no `load_ms`, no error). Pure, so the slow/failed thresholding is unit-tested
/// without the network.
fn embedder_load_event(
    model: &str,
    load_ms: Option<u64>,
    error: Option<&EmbedderError>,
) -> Option<TraceEvent> {
    match (load_ms, error) {
        (_, Some(err)) => Some(TraceEvent::EmbedderLoad {
            model: model.to_string(),
            status: EmbedderLoadStatus::Failed,
            took_ms: load_ms.unwrap_or(0),
            reason: Some(err.to_string()),
        }),
        (Some(ms), None) => {
            let slow = ms > slow_load_ms();
            Some(TraceEvent::EmbedderLoad {
                model: model.to_string(),
                status: if slow {
                    EmbedderLoadStatus::Slow
                } else {
                    EmbedderLoadStatus::Ok
                },
                took_ms: ms,
                reason: slow.then(|| SLOW_LOAD_REASON.to_string()),
            })
        }
        (None, None) => None,
    }
}

fn slow_load_ms() -> u64 {
    std::env::var("RATEL_EMBED_SLOW_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SLOW_LOAD_MS)
}

/// A BERT-family embedding model run in-process via Candle — the backend for the
/// built-in default, any HuggingFace repo, and any on-disk model directory.
/// Carries its pooling mode, asymmetric prefixes, and a resolved fingerprint.
/// Documents per padded forward pass in [`CandleEmbedder::embed_batch_inner`]. A
/// whole-corpus `rebuild` hands the full slice in at once, so this bounds peak
/// activation memory; it does not affect the produced vectors (each row's output is
/// independent of its chunk-mates).
const EMBED_BATCH_CHUNK: usize = 32;

pub(crate) struct CandleEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    pooling: Pooling,
    query_prefix: String,
    doc_prefix: String,
    fingerprint: String,
    local_source: Option<LocalArtifactSource>,
}

/// Paths retained at load so RAT1 can hash later without digesting on ordinary dense use
struct LocalArtifactSource {
    dir: PathBuf,
    hash_paths: [PathBuf; 3],
    identity: LocalContentIdentity,
}

/// The resolved files + pooling a build needs.
struct Loaded {
    config: PathBuf,
    tokenizer: PathBuf,
    weights: PathBuf,
    pooling: Pooling,
}

impl CandleEmbedder {
    /// Load a BERT-family model from a HuggingFace repo. When `allow_download` is
    /// set (only the built-in default, or an explicit opt-in) missing files are
    /// fetched into the shared HF cache — `from_env` honors `HF_HOME` /
    /// `HF_ENDPOINT`. Otherwise it is **cache-only**: a model not already present
    /// errors as [`EmbedderError::NotCached`] (Ratel doesn't silently download
    /// non-default models — symmetric with Ollama's "not pulled"). Weights are
    /// `model.safetensors`, falling back to `pytorch_model.bin`.
    fn load_hf(
        repo_id: &str,
        revision: &str,
        query_prefix: &str,
        doc_prefix: &str,
        pooling_override: Option<Pooling>,
        allow_download: bool,
    ) -> Result<(Self, LoadNotices), EmbedderError> {
        let device = Device::Cpu;
        let repo_spec =
            Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
        let cache_repo = hf_hub::Cache::from_env().repo(repo_spec.clone());

        let (config_path, tokenizer_path, weights_path, pooling_file, download) = if allow_download
        {
            // Cold-cache detection *before* fetching, so we only announce a real
            // download — and give a heads-up before the (blocking) fetch, since a
            // multi-second/GB download with no message reads as a hang.
            let was_cached = cache_repo.get("model.safetensors").is_some()
                || cache_repo.get("pytorch_model.bin").is_some();
            if !was_cached {
                eprintln!(
                    "ratel: downloading embedding model {repo_id} (one-time; this may take a moment)…"
                );
            }
            let api = ApiBuilder::from_env()
                .build()
                .map_err(|e| EmbedderError::Download {
                    model: repo_id.to_string(),
                    source: e.to_string(),
                })?;
            let repo = api.repo(repo_spec);
            let config = fetch_cached(&repo, "config.json", repo_id)?;
            let tokenizer = fetch_cached(&repo, "tokenizer.json", repo_id)?;
            // Prefer safetensors; fall back to a pickled `pytorch_model.bin`.
            let weights = match fetch_cached(&repo, "model.safetensors", repo_id) {
                Ok(p) => p,
                Err(EmbedderError::Download { source, .. }) if is_not_found(&source) => {
                    fetch_cached(&repo, "pytorch_model.bin", repo_id)?
                }
                Err(e) => return Err(e),
            };
            let pooling_file = fetch_optional(&repo, "1_Pooling/config.json");
            let notice = (!was_cached).then(|| DownloadNotice {
                model: repo_id.to_string(),
                bytes: [&config, &tokenizer, &weights]
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                    .sum(),
            });
            (config, tokenizer, weights, pooling_file, notice)
        } else {
            // Cache-only: never touch the network. A file missing from the cache
            // means the model was never downloaded → NotCached.
            let not_cached = || EmbedderError::NotCached {
                model: repo_id.to_string(),
                revision: (revision != "main").then(|| revision.to_string()),
            };
            let config = cache_repo.get("config.json").ok_or_else(not_cached)?;
            let tokenizer = cache_repo.get("tokenizer.json").ok_or_else(not_cached)?;
            let weights = cache_repo
                .get("model.safetensors")
                .or_else(|| cache_repo.get("pytorch_model.bin"))
                .ok_or_else(not_cached)?;
            let pooling_file = cache_repo.get("1_Pooling/config.json");
            (config, tokenizer, weights, pooling_file, None)
        };

        // Pooling: override wins, else the repo's `1_Pooling/config.json`, else Mean.
        let detected =
            pooling_override.or_else(|| pooling_file.and_then(|p| detect_pooling_file(&p)));
        let (pooling, pooling_assumed) = resolve_pooling(detected);
        let notices = LoadNotices {
            download,
            pooling_assumed: pooling_assumed.then(|| repo_id.to_string()),
        };

        // Resolve `main` (or any ref) to the concrete commit so the fingerprint
        // pins a real snapshot, not a moving label.
        let sha = snapshot_sha(&weights_path).unwrap_or_else(|| revision.to_string());
        let loaded = Loaded {
            config: config_path,
            tokenizer: tokenizer_path,
            weights: weights_path,
            pooling,
        };
        let embedder = Self::build(
            device,
            &loaded,
            query_prefix,
            doc_prefix,
            huggingface_fingerprint(repo_id, &sha),
            repo_id,
            None,
        )?;
        Ok((embedder, notices))
    }

    /// Load a BERT-family model directly from a directory of files (no hf-hub, no
    /// network) — the air-gapped / bring-your-own-checkpoint path.
    fn load_path(
        dir: &Path,
        query_prefix: &str,
        doc_prefix: &str,
        pooling_override: Option<Pooling>,
    ) -> Result<(Self, LoadNotices), EmbedderError> {
        let device = Device::Cpu;
        let name = dir.display().to_string();
        let files = resolve_local_model_files(dir)?;
        let load_stamps = stamp_local_hash_paths(&files, &name)?;
        let hash_paths = [
            files.config.clone(),
            files.tokenizer.clone(),
            files.weights.clone(),
        ];

        let detected = pooling_override.or_else(|| {
            files
                .pooling_config
                .as_ref()
                .and_then(|p| detect_pooling_file(p))
        });
        let (pooling, pooling_assumed) = resolve_pooling(detected);
        let notices = LoadNotices {
            download: None,
            pooling_assumed: pooling_assumed.then(|| name.clone()),
        };

        let loaded = Loaded {
            config: files.config,
            tokenizer: files.tokenizer,
            weights: files.weights,
            pooling,
        };
        let local_source = Some(LocalArtifactSource {
            dir: dir.to_path_buf(),
            hash_paths,
            identity: LocalContentIdentity::new(load_stamps),
        });
        let embedder = Self::build(
            device,
            &loaded,
            query_prefix,
            doc_prefix,
            local_fingerprint(&name),
            &name,
            local_source,
        )?;
        Ok((embedder, notices))
    }

    /// Shared file→model build. A non-BERT checkpoint fails `BertModel::load`;
    /// the error signposts the endpoint/Ollama route (any model can run there).
    fn build(
        device: Device,
        loaded: &Loaded,
        query_prefix: &str,
        doc_prefix: &str,
        base_fingerprint: String,
        model_name: &str,
        local_source: Option<LocalArtifactSource>,
    ) -> Result<Self, EmbedderError> {
        let load_err = |source: String| EmbedderError::Load {
            model: model_name.to_string(),
            source,
        };

        let config_bytes = std::fs::read(&loaded.config).map_err(|e| load_err(e.to_string()))?;
        let config: Config =
            serde_json::from_slice(&config_bytes).map_err(|e| load_err(e.to_string()))?;

        let mut tokenizer =
            Tokenizer::from_file(&loaded.tokenizer).map_err(|e| load_err(e.to_string()))?;
        // Cap at the model's positional limit so long tool text can't index past
        // the position embeddings.
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_position_embeddings,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| load_err(e.to_string()))?;

        // Upstream weights are f32; load them directly for reproducible CPU math.
        // safetensors is mmap'd; a `.bin`/`.pth` checkpoint is loaded via pickle.
        let is_safetensors =
            loaded.weights.extension().and_then(|e| e.to_str()) == Some("safetensors");
        let vb = if is_safetensors {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&loaded.weights], DType::F32, &device)
                    .map_err(|e| load_err(e.to_string()))?
            }
        } else {
            VarBuilder::from_pth(&loaded.weights, DType::F32, &device)
                .map_err(|e| load_err(e.to_string()))?
        };
        let model = BertModel::load(vb, &config).map_err(|e| EmbedderError::Load {
            model: model_name.to_string(),
            source: format!(
                "{e} — if this is not a BERT-family model it can't run in-process; \
                 serve it in a local model server and use {{\"ollama\": \"…\"}} or \
                 {{\"url\", \"model\"}} (e.g. Ollama at {OLLAMA_DEFAULT_URL})"
            ),
        })?;

        // Pooling + prefixes change the vectors, so they are part of the identity.
        let fingerprint = format!(
            "{base_fingerprint}{}",
            fingerprint_suffix(Some(loaded.pooling), query_prefix, doc_prefix)
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            pooling: loaded.pooling,
            query_prefix: query_prefix.to_string(),
            doc_prefix: doc_prefix.to_string(),
            fingerprint,
            local_source,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.embed_inner(text)
            .map_err(|e| EmbedderError::Inference {
                source: e.to_string(),
            })
    }

    fn embed_inner(&self, text: &str) -> candle_core::Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let ids = encoding.get_ids();
        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?; // (1, seq)
        let token_type_ids = input_ids.zeros_like()?;
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let attention_mask = Tensor::new(mask.as_slice(), &self.device)?.unsqueeze(0)?;

        // (1, seq, hidden)
        let sequence_output =
            self.model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
        let pooled = match self.pooling {
            // CLS pooling = the first token's hidden state.
            Pooling::Cls => sequence_output.i((0, 0))?, // (hidden,)
            // Mean pooling = masked average over the real (non-pad) tokens.
            Pooling::Mean => mean_pool(&sequence_output, &attention_mask)?,
        };
        let vec = pooled.to_vec1::<f32>()?;
        Ok(l2_normalize(vec))
    }

    /// Batched twin of [`Self::embed_inner`]: one padded forward pass per chunk of
    /// documents instead of one per document. A padded (masked) key contributes
    /// exactly zero to every real token's attention (candle adds `f32::MIN` before
    /// softmax), and right-padding never leaks into real tokens, so each vector is
    /// **bit-for-bit identical** to the per-document path — chunking only bounds the
    /// activation memory a whole-corpus `rebuild` would otherwise allocate at once.
    fn embed_batch_inner(&self, texts: &[String]) -> candle_core::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(EMBED_BATCH_CHUNK) {
            // Documents get the doc-side prefix, mirroring `embed_doc`.
            let inputs: Vec<String> = if self.doc_prefix.is_empty() {
                chunk.to_vec()
            } else {
                chunk
                    .iter()
                    .map(|t| format!("{}{}", self.doc_prefix, t))
                    .collect()
            };
            let encodings = self
                .tokenizer
                .encode_batch(inputs, true)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let n = encodings.len();
            let max_len = encodings
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0);
            // Right-pad ids (pad id 0 — masked out, so the value is irrelevant) and
            // the attention mask into rectangular `(n, max_len)` buffers.
            let mut ids = vec![0u32; n * max_len];
            let mut mask = vec![0u32; n * max_len];
            for (row, enc) in encodings.iter().enumerate() {
                let e_ids = enc.get_ids();
                let e_mask = enc.get_attention_mask();
                let base = row * max_len;
                ids[base..base + e_ids.len()].copy_from_slice(e_ids);
                mask[base..base + e_mask.len()].copy_from_slice(e_mask);
            }
            let input_ids = Tensor::from_vec(ids, (n, max_len), &self.device)?;
            let attention_mask = Tensor::from_vec(mask, (n, max_len), &self.device)?;
            let token_type_ids = input_ids.zeros_like()?;

            // (n, max_len, hidden)
            let sequence_output =
                self.model
                    .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
            let pooled = match self.pooling {
                // CLS pooling = each row's first token.
                Pooling::Cls => sequence_output.narrow(1, 0, 1)?.squeeze(1)?, // (n, hidden)
                // Mean pooling = masked average over the real tokens, per row.
                Pooling::Mean => mean_pool_batch(&sequence_output, &attention_mask)?,
            };
            for row in pooled.to_vec2::<f32>()? {
                out.push(l2_normalize(row));
            }
        }
        Ok(out)
    }
}

/// Masked mean over tokens: `Σ hidden[t]·mask[t] / Σ mask[t]`. `sequence_output`
/// is `(1, seq, hidden)`, `attention_mask` is `(1, seq)`; returns `(hidden,)`.
fn mean_pool(sequence_output: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?; // (1, seq, 1)
    let summed = sequence_output.broadcast_mul(&mask)?.sum(1)?; // (1, hidden)
    let counts = mask.sum(1)?; // (1, 1)
    summed.broadcast_div(&counts)?.i(0) // (hidden,)
}

/// Masked mean over tokens for a whole batch — the row-wise twin of [`mean_pool`].
/// `sequence_output` is `(n, seq, hidden)`, `attention_mask` is `(n, seq)`; returns
/// `(n, hidden)`. Padded tokens (mask 0) contribute nothing and don't count.
fn mean_pool_batch(
    sequence_output: &Tensor,
    attention_mask: &Tensor,
) -> candle_core::Result<Tensor> {
    let mask = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?; // (n, seq, 1)
    let summed = sequence_output.broadcast_mul(&mask)?.sum(1)?; // (n, hidden)
    let counts = mask.sum(1)?; // (n, 1)
    summed.broadcast_div(&counts) // (n, hidden)
}

impl Embedder for CandleEmbedder {
    fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        if self.doc_prefix.is_empty() {
            self.embed(text)
        } else {
            self.embed(&format!("{}{}", self.doc_prefix, text))
        }
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        if self.query_prefix.is_empty() {
            self.embed(text)
        } else {
            self.embed(&format!("{}{}", self.query_prefix, text))
        }
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.embed_batch_inner(texts)
            .map_err(|e| EmbedderError::Inference {
                source: e.to_string(),
            })
    }

    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }

    fn artifact_identity(&self) -> Result<String, EmbedderError> {
        let Some(source) = &self.local_source else {
            return Ok(self.fingerprint.clone());
        };
        let paths: [&Path; 3] = [
            &source.hash_paths[0],
            &source.hash_paths[1],
            &source.hash_paths[2],
        ];
        let content_id = source.identity.content_id(&source.dir, &paths)?;
        Ok(format!(
            "{}{}",
            local_content_fingerprint(&content_id),
            fingerprint_suffix(Some(self.pooling), &self.query_prefix, &self.doc_prefix)
        ))
    }

    fn embed_batch_with_artifact_identity(
        &self,
        texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        if self.local_source.is_none() {
            return self.embed_batch_with_identity(texts);
        }
        Ok(Embedded {
            value: self.embed_batch(texts)?,
            fingerprint: self.artifact_identity()?,
        })
    }
}

/// Read a `1_Pooling/config.json` file into a [`Pooling`], or `None` if absent /
/// unparseable / neither cls-nor-mean.
fn detect_pooling_file(path: &Path) -> Option<Pooling> {
    let bytes = std::fs::read(path).ok()?;
    parse_pooling_config(&bytes)
}

/// Resolve pooling: a detected/overridden mode, else assume Mean (and flag it so
/// the assumption is surfaced, never silent).
fn resolve_pooling(detected: Option<Pooling>) -> (Pooling, bool) {
    match detected {
        Some(p) => (p, false),
        None => (Pooling::Mean, true),
    }
}

/// Best-effort optional fetch of a small side file (pooling config, alt weights):
/// `None` on any error, so a missing file never fails the load.
fn fetch_optional(repo: &ApiRepo, file: &str) -> Option<PathBuf> {
    repo.get(file).ok()
}

/// Whether an hf-hub fetch error is a "file not in repo" (so we try a fallback
/// file) rather than a real network/cache failure.
fn is_not_found(source: &str) -> bool {
    let l = source.to_lowercase();
    l.contains("404") || l.contains("not found") || l.contains("entry not found")
}

/// The concrete commit SHA a HuggingFace fetch resolved to, read from the cached
/// snapshot path (`…/snapshots/<sha>/<file>`). `None` if the path isn't in that
/// layout — the caller falls back to the requested revision string.
fn snapshot_sha(weights_path: &Path) -> Option<String> {
    let name = weights_path.parent()?.file_name()?.to_str()?;
    (name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit())).then(|| name.to_string())
}

/// OpenAI-compatible HTTP embedding endpoint (OpenAI, Ollama, TEI, vLLM…). Any
/// model can back it, including non-BERT ones the in-process path can't run.
/// Vectors are **re-normalized on ingestion** — an arbitrary endpoint may return
/// un-normalized embeddings, and `dense_search` assumes unit vectors.
pub(crate) struct EndpointEmbedder {
    url: String,
    model: String,
    api_key_env: Option<String>,
    query_prefix: String,
    doc_prefix: String,
    agent: ureq::Agent,
    fingerprint: String,
}

/// OpenAI `/embeddings` response shape: `{ "data": [{ "embedding": [...], "index": n }] }`.
#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

struct ParsedEmbeddings {
    vectors: Vec<Vec<f32>>,
    model: Option<String>,
}

impl EndpointEmbedder {
    fn new(
        url: String,
        model: String,
        api_key_env: Option<String>,
        query_prefix: String,
        doc_prefix: String,
    ) -> Result<Self, EmbedderError> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(ENDPOINT_TIMEOUT_SECS)))
            .build()
            .into();
        // Prefixes are part of the identity (they change the vectors).
        let fingerprint = format!(
            "{}{}",
            endpoint_fingerprint(&url, &model),
            fingerprint_suffix(None, &query_prefix, &doc_prefix)
        );
        Ok(Self {
            url,
            model,
            api_key_env,
            query_prefix,
            doc_prefix,
            agent,
            fingerprint,
        })
    }

    /// Read the API key from the named env var (at call time, so it can be set
    /// after construction). A named-but-unset var is a clear `Config` error, not
    /// a downstream 401.
    fn api_key(&self) -> Result<Option<String>, EmbedderError> {
        match &self.api_key_env {
            None => Ok(None),
            Some(var) => std::env::var(var)
                .map(Some)
                .map_err(|_| EmbedderError::Config {
                    message: format!(
                        "api_key_env=\"{var}\" but that environment variable is not set"
                    ),
                }),
        }
    }

    fn request_chunk(&self, inputs: &[String]) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        let key = self.api_key()?;
        let body = serde_json::json!({ "model": self.model, "input": inputs });
        let mut req = self
            .agent
            .post(&self.url)
            .header("content-type", "application/json");
        if let Some(k) = key {
            req = req.header("authorization", &format!("Bearer {k}"));
        }
        let mut resp = req.send_json(&body).map_err(|e| self.classify(e))?;
        let parsed: EmbeddingsResponse = resp
            .body_mut()
            .with_config()
            .limit(ENDPOINT_RESPONSE_LIMIT_BYTES)
            .read_json()
            .map_err(|e| EmbedderError::Inference {
                source: format!("malformed or oversized endpoint response: {e}"),
            })?;
        let parsed = parse_embeddings(parsed, inputs.len())?;
        let resolved_model = parsed.model.as_deref().unwrap_or(&self.model);
        Ok(Embedded {
            value: parsed.vectors,
            fingerprint: self.fingerprint_for_model(resolved_model),
        })
    }

    fn request(&self, inputs: &[String]) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        if inputs.is_empty() {
            return Ok(Embedded {
                value: Vec::new(),
                fingerprint: self.fingerprint.clone(),
            });
        }

        let mut vectors = Vec::with_capacity(inputs.len());
        let mut fingerprint: Option<String> = None;
        let mut dimension = None;
        for chunk in inputs.chunks(ENDPOINT_BATCH_SIZE) {
            let embedded = self.request_chunk(chunk)?;
            if let Some(first) = &fingerprint {
                if first != &embedded.fingerprint {
                    return Err(EmbedderError::ModelMismatch {
                        built: first.clone(),
                        active: embedded.fingerprint,
                    });
                }
            } else {
                fingerprint = Some(embedded.fingerprint.clone());
            }
            let chunk_dimension = embedded
                .value
                .first()
                .expect("non-empty request chunk has a non-empty response")
                .len();
            if let Some(expected) = dimension {
                if expected != chunk_dimension {
                    return Err(EmbedderError::Inference {
                        source: format!(
                            "endpoint returned mixed embedding dimensions across chunks: expected {expected}, got {chunk_dimension}"
                        ),
                    });
                }
            } else {
                dimension = Some(chunk_dimension);
            }
            vectors.extend(embedded.value);
        }
        Ok(Embedded {
            value: vectors,
            fingerprint: fingerprint.expect("non-empty input produced at least one chunk"),
        })
    }

    fn fingerprint_for_model(&self, model: &str) -> String {
        format!(
            "{}{}",
            endpoint_fingerprint(&self.url, model),
            fingerprint_suffix(None, &self.query_prefix, &self.doc_prefix)
        )
    }

    /// Map an endpoint transport/HTTP error to a typed `EmbedderError`, with an
    /// `ollama pull` hint on a 404 from a local Ollama.
    fn classify(&self, e: ureq::Error) -> EmbedderError {
        let status = match &e {
            ureq::Error::StatusCode(code) => Some(*code),
            _ => None,
        };
        let is_local_ollama =
            self.url.contains("localhost:11434") || self.url.contains("127.0.0.1:11434");
        match status {
            Some(401) | Some(403) => EmbedderError::Config {
                message: format!(
                    "endpoint rejected the request ({}); check api_key_env / the key",
                    status.unwrap()
                ),
            },
            // A model that isn't served yet: on a local Ollama, tell them to pull it.
            Some(404) => {
                let hint = if is_local_ollama {
                    format!(" — run: ollama pull {}", self.model)
                } else {
                    String::new()
                };
                EmbedderError::Download {
                    model: self.model.clone(),
                    source: format!("endpoint returned 404 for model '{}'{hint}", self.model),
                }
            }
            // A transport error (connection refused, timeout, DNS): on a local
            // Ollama, the server most likely isn't running — say how to start it.
            _ if is_local_ollama => EmbedderError::Download {
                model: self.model.clone(),
                source: format!(
                    "could not reach Ollama at {} ({e}) — is it running? start it with \
                     `ollama serve`, then `ollama pull {}`",
                    self.url, self.model
                ),
            },
            _ => EmbedderError::Download {
                model: self.model.clone(),
                source: e.to_string(),
            },
        }
    }
}

/// Turn a parsed endpoint response into ordered, L2-normalized vectors. Pure, so
/// the ordering + normalization guard is unit-tested without the network.
fn parse_embeddings(
    resp: EmbeddingsResponse,
    expected_len: usize,
) -> Result<ParsedEmbeddings, EmbedderError> {
    if resp.data.len() != expected_len {
        return Err(EmbedderError::Inference {
            source: format!(
                "endpoint returned {} embeddings for {expected_len} inputs",
                resp.data.len()
            ),
        });
    }
    if resp
        .model
        .as_deref()
        .is_some_and(|model| model.trim().is_empty())
    {
        return Err(EmbedderError::Inference {
            source: "endpoint returned a blank model identity".into(),
        });
    }

    let mut ordered: Vec<Option<Vec<f32>>> = (0..expected_len).map(|_| None).collect();
    let mut dimension = None;
    for data in resp.data {
        if data.index >= expected_len {
            return Err(EmbedderError::Inference {
                source: format!(
                    "endpoint returned out-of-range embedding index {} for {expected_len} inputs",
                    data.index
                ),
            });
        }
        if ordered[data.index].is_some() {
            return Err(EmbedderError::Inference {
                source: format!("endpoint returned duplicate embedding index {}", data.index),
            });
        }
        let vector = normalize_endpoint_vector(data.embedding, &mut dimension)?;
        ordered[data.index] = Some(vector);
    }

    let vectors = ordered
        .into_iter()
        .enumerate()
        .map(|(index, vector)| {
            vector.ok_or_else(|| EmbedderError::Inference {
                source: format!("endpoint response is missing embedding index {index}"),
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(ParsedEmbeddings {
        vectors,
        model: resp.model,
    })
}

fn normalize_endpoint_vector(
    mut vector: Vec<f32>,
    dimension: &mut Option<usize>,
) -> Result<Vec<f32>, EmbedderError> {
    if vector.is_empty() {
        return Err(EmbedderError::Inference {
            source: "endpoint returned an empty embedding vector".into(),
        });
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(EmbedderError::Inference {
            source: "endpoint returned a non-finite embedding value".into(),
        });
    }
    match *dimension {
        Some(expected) if vector.len() != expected => {
            return Err(EmbedderError::Inference {
                source: format!(
                    "endpoint returned mixed embedding dimensions: expected {expected}, got {}",
                    vector.len()
                ),
            });
        }
        None => *dimension = Some(vector.len()),
        Some(_) => {}
    }

    let norm = vector
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(EmbedderError::Inference {
            source: "endpoint returned a zero or non-normalizable embedding vector".into(),
        });
    }
    for value in &mut vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(vector)
}

impl Embedder for EndpointEmbedder {
    fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.embed_batch_with_identity(std::slice::from_ref(&text.to_string()))?
            .value
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::Inference {
                source: "endpoint returned no embedding".into(),
            })
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.embed_query_with_identity(text)?.value)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        Ok(self.embed_batch_with_identity(texts)?.value)
    }

    fn embed_query_with_identity(&self, text: &str) -> Result<Embedded<Vec<f32>>, EmbedderError> {
        let q = if self.query_prefix.is_empty() {
            text.to_string()
        } else {
            format!("{}{}", self.query_prefix, text)
        };
        let embedded = self.request(&[q])?;
        let fingerprint = embedded.fingerprint;
        let value = embedded
            .value
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::Inference {
                source: "endpoint returned no embedding".into(),
            })?;
        Ok(Embedded { value, fingerprint })
    }

    fn embed_batch_with_identity(
        &self,
        texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        if self.doc_prefix.is_empty() {
            self.request(texts)
        } else {
            let prefixed: Vec<String> = texts
                .iter()
                .map(|t| format!("{}{}", self.doc_prefix, t))
                .collect();
            self.request(&prefixed)
        }
    }

    fn fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

/// Resolve one model file from the HF cache, tolerating the cross-process
/// download race on a cold cache. hf-hub guards each blob with a *non-blocking*
/// `flock` and gives up after ~5s (5 × 1s); a first fetch of the ~130 MB weights
/// takes longer, so when several processes load the embedder at once on a cold
/// cache — parallel test workers, a web server's worker pool cold-starting,
/// `multiprocessing` — every process but the lock holder gets `LockAcquisition`
/// and would fail. Retry with backoff: the losers wait for the winner's download
/// to land, then `get()` returns the now-cached blob without locking (hf-hub
/// checks the cache before it locks). Any other failure is classified and
/// returned immediately. See ADR-0011.
fn fetch_cached(repo: &ApiRepo, file: &str, model: &str) -> Result<PathBuf, EmbedderError> {
    // ~30 × (up to hf-hub's own ~5s lock wait + 1s backoff) comfortably outlasts
    // a single cold-cache download; the loser normally succeeds within a few.
    const MAX_ATTEMPTS: u32 = 30;
    const BACKOFF: Duration = Duration::from_secs(1);

    let mut attempt = 1;
    loop {
        match repo.get(file) {
            Ok(path) => return Ok(path),
            Err(e) => {
                let msg = e.to_string();
                if attempt < MAX_ATTEMPTS && is_lock_contention(&msg) {
                    attempt += 1;
                    std::thread::sleep(BACKOFF);
                    continue;
                }
                return Err(classify_fetch_error(model, &msg));
            }
        }
    }
}

/// True only when an hf-hub fetch failed because another process holds the
/// download lock — the one error worth retrying, since the blob appears once the
/// winner finishes. Every other failure is terminal and classified below.
fn is_lock_contention(err: &str) -> bool {
    err.contains("Lock acquisition failed")
}

/// Map an hf-hub fetch error string to a typed [`EmbedderError`]. A cache
/// permission/space problem is distinct (and actionable) from a network/model
/// problem; everything else is treated as a download failure.
fn classify_fetch_error(model: &str, msg: &str) -> EmbedderError {
    let lower = msg.to_lowercase();
    let unwritable = lower.contains("permission denied")
        || lower.contains("read-only")
        || lower.contains("no space")
        || lower.contains("os error 13") // EACCES
        || lower.contains("os error 28") // ENOSPC
        || lower.contains("os error 30"); // EROFS
    if unwritable {
        EmbedderError::CacheUnwritable {
            source: msg.to_string(),
        }
    } else {
        EmbedderError::Download {
            model: model.to_string(),
            source: msg.to_string(),
        }
    }
}

/// Scale to unit L2 norm so downstream cosine similarity is a plain dot product.
/// A zero vector is returned unchanged (no NaNs).
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    use super::*;
    use crate::{Origin, SearchMethod, Tool, ToolRegistry};

    #[test]
    fn only_lock_contention_is_retried() {
        // The cold-cache download race: retry and wait for the winner.
        assert!(is_lock_contention(
            "Lock acquisition failed: /home/u/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs/abc.lock"
        ));
        // Everything else is terminal — classify, don't spin.
        assert!(!is_lock_contention("request error: connection refused"));
        assert!(!is_lock_contention("Http(reqwest::Error { status: 404 })"));
        assert!(!is_lock_contention(
            "No such file or directory (os error 2)"
        ));
    }

    #[test]
    fn classifies_cache_permission_and_space_as_unwritable() {
        assert!(matches!(
            classify_fetch_error("m", "Permission denied (os error 13)"),
            EmbedderError::CacheUnwritable { .. }
        ));
        assert!(matches!(
            classify_fetch_error("m", "No space left on device (os error 28)"),
            EmbedderError::CacheUnwritable { .. }
        ));
        assert!(matches!(
            classify_fetch_error("m", "Read-only file system (os error 30)"),
            EmbedderError::CacheUnwritable { .. }
        ));
    }

    #[test]
    fn classifies_network_and_http_as_download() {
        assert!(matches!(
            classify_fetch_error("m", "error sending request: dns error: failed to lookup"),
            EmbedderError::Download { .. }
        ));
        assert!(matches!(
            classify_fetch_error("m", "Http status client error (404 Not Found)"),
            EmbedderError::Download { .. }
        ));
    }

    #[test]
    fn error_display_carries_source_and_hint() {
        let s = EmbedderError::Download {
            model: "embed-v1 @ https://embeddings.example.test".into(),
            source: "connection refused".into(),
        }
        .to_string();
        assert!(s.contains("connection refused"), "got: {s}");
        assert!(s.contains("hint:"), "got: {s}");
        assert!(!s.contains("revision"), "got: {s}");

        let load = EmbedderError::Load {
            model: "/models/embed".into(),
            source: "missing config.json".into(),
        }
        .to_string();
        assert!(!load.contains("re-download"), "got: {load}");

        let inference = EmbedderError::Inference {
            source: "endpoint returned duplicate index 0".into(),
        }
        .to_string();
        assert!(!inference.contains("underpowered"), "got: {inference}");
    }

    #[test]
    fn get_or_load_keyed_does_not_cache_failure_and_reports_latency_once() {
        let cache: Mutex<HashMap<String, LoadSlot<i32>>> = Mutex::new(HashMap::new());
        let boom = || {
            Err::<Arc<i32>, _>(EmbedderError::Inference {
                source: "boom".into(),
            })
        };
        // A failed load must NOT be cached.
        assert!(get_or_load_keyed(&cache, "k", boom).is_err());
        // The next call retries and loads; it reports the load latency.
        let (v, ms) =
            get_or_load_keyed(&cache, "k", || Ok::<_, EmbedderError>(Arc::new(7))).unwrap();
        assert_eq!(*v, 7);
        assert!(ms.is_some(), "the loading call reports latency");
        // Warm reuse keeps the first value and reports no latency.
        let (v2, ms2) =
            get_or_load_keyed(&cache, "k", || Ok::<_, EmbedderError>(Arc::new(999))).unwrap();
        assert_eq!(*v2, 7);
        assert!(ms2.is_none(), "warm reuse reports no load latency");
    }

    #[test]
    fn distinct_key_loads_run_concurrently() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        // Two loads for *different* keys must be able to run at the same time.
        // The barrier only releases once BOTH loads have entered — so if the
        // cache serializes distinct-key loads (map lock held across `load`), the
        // second thread can never start and the receive below times out.
        let cache: Arc<Mutex<HashMap<String, LoadSlot<i32>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let barrier = Arc::new(Barrier::new(2));
        let inflight = Arc::new(AtomicUsize::new(0));
        let (done_tx, done_rx) = mpsc::channel();

        for (i, key) in ["a", "b"].into_iter().enumerate() {
            let cache = Arc::clone(&cache);
            let barrier = Arc::clone(&barrier);
            let inflight = Arc::clone(&inflight);
            let done_tx = done_tx.clone();
            thread::spawn(move || {
                let result = get_or_load_keyed(&cache, key, || {
                    inflight.fetch_add(1, Ordering::SeqCst);
                    barrier.wait(); // both distinct-key loads must be in-flight here
                    Ok::<_, EmbedderError>(Arc::new(i as i32))
                });
                let _ = done_tx.send(result.map(|(v, _)| *v));
            });
        }
        drop(done_tx);

        let mut got = Vec::new();
        for _ in 0..2 {
            let value = done_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("distinct-key loads did not run concurrently (map lock held across load)");
            got.push(value.unwrap());
        }
        got.sort_unstable();
        assert_eq!(got, vec![0, 1]);
        assert_eq!(inflight.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn same_key_load_is_single_flight() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        // Many threads race the SAME key: `load` must run exactly once, and
        // exactly one caller reports a load latency; the rest see warm reuse.
        const THREADS: usize = 8;
        let cache: Arc<Mutex<HashMap<String, LoadSlot<i32>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let loads = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(THREADS));
        let (tx, rx) = mpsc::channel();

        for _ in 0..THREADS {
            let cache = Arc::clone(&cache);
            let loads = Arc::clone(&loads);
            let start = Arc::clone(&start);
            let tx = tx.clone();
            thread::spawn(move || {
                start.wait(); // maximize the race on the cold miss
                let result = get_or_load_keyed(&cache, "shared", || {
                    loads.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, EmbedderError>(Arc::new(42))
                });
                let (value, ms) = result.unwrap();
                let _ = tx.send((*value, ms.is_some()));
            });
        }
        drop(tx);

        let mut reported = 0;
        for _ in 0..THREADS {
            let (value, loaded) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(value, 42);
            if loaded {
                reported += 1;
            }
        }
        assert_eq!(
            loads.load(Ordering::SeqCst),
            1,
            "load must run exactly once per key"
        );
        assert_eq!(reported, 1, "exactly one caller reports the load latency");
    }

    #[test]
    fn load_event_flags_slow_ok_failed_and_warm() {
        // Above the 5s default → slow (underpowered-machine flag).
        assert!(matches!(
            embedder_load_event("m", Some(10_000), None),
            Some(TraceEvent::EmbedderLoad {
                status: EmbedderLoadStatus::Slow,
                took_ms: 10_000,
                ..
            })
        ));
        // Comfortably under → ok.
        assert!(matches!(
            embedder_load_event("m", Some(5), None),
            Some(TraceEvent::EmbedderLoad {
                status: EmbedderLoadStatus::Ok,
                ..
            })
        ));
        // A load error → failed, carrying the reason.
        let err = EmbedderError::Inference { source: "x".into() };
        assert!(matches!(
            embedder_load_event("m", None, Some(&err)),
            Some(TraceEvent::EmbedderLoad {
                status: EmbedderLoadStatus::Failed,
                ..
            })
        ));
        // Warm reuse → no event.
        assert!(embedder_load_event("m", None, None).is_none());
    }

    #[test]
    fn endpoint_embeddings_are_normalized_and_ordered_by_index() {
        // An endpoint may return un-normalized vectors in any order; ingestion
        // must L2-normalize (so cosine==dot holds) and restore index order.
        let resp = EmbeddingsResponse {
            model: Some("resolved-model".into()),
            data: vec![
                EmbeddingData {
                    embedding: vec![0.0, 3.0], // index 1, un-normalized
                    index: 1,
                },
                EmbeddingData {
                    embedding: vec![4.0, 0.0], // index 0, un-normalized
                    index: 0,
                },
            ],
        };
        let out = parse_embeddings(resp, 2).expect("parse");
        assert_eq!(out.vectors[0], vec![1.0, 0.0], "index 0 first, normalized");
        assert_eq!(out.vectors[1], vec![0.0, 1.0], "index 1 second, normalized");
        assert_eq!(out.model.as_deref(), Some("resolved-model"));
    }

    #[test]
    fn endpoint_response_count_mismatch_errors() {
        let resp = EmbeddingsResponse {
            model: None,
            data: vec![EmbeddingData {
                embedding: vec![1.0],
                index: 0,
            }],
        };
        assert!(matches!(
            parse_embeddings(resp, 2),
            Err(EmbedderError::Inference { .. })
        ));
    }

    fn response(vectors: &[(usize, Vec<f32>)]) -> EmbeddingsResponse {
        EmbeddingsResponse {
            model: None,
            data: vectors
                .iter()
                .cloned()
                .map(|(index, embedding)| EmbeddingData { embedding, index })
                .collect(),
        }
    }

    #[test]
    fn endpoint_response_requires_an_exact_index_permutation() {
        assert!(
            serde_json::from_value::<EmbeddingsResponse>(serde_json::json!({
                "data": [{ "embedding": [1.0] }]
            }))
            .is_err(),
            "index is required"
        );
        for malformed in [
            response(&[(0, vec![1.0]), (0, vec![1.0])]),
            response(&[(0, vec![1.0]), (2, vec![1.0])]),
        ] {
            assert!(matches!(
                parse_embeddings(malformed, 2),
                Err(EmbedderError::Inference { .. })
            ));
        }
    }

    #[test]
    fn endpoint_response_rejects_invalid_vectors() {
        for malformed in [
            response(&[(0, vec![])]),
            response(&[(0, vec![0.0, 0.0])]),
            response(&[(0, vec![f32::NAN, 1.0])]),
            response(&[(0, vec![f32::INFINITY, 1.0])]),
            response(&[(0, vec![1.0, 0.0]), (1, vec![1.0])]),
        ] {
            let expected_len = malformed.data.len();
            assert!(matches!(
                parse_embeddings(malformed, expected_len),
                Err(EmbedderError::Inference { .. })
            ));
        }
    }

    #[test]
    fn endpoint_rejects_a_response_over_64_mib() {
        let (url, requests_rx, server) = mock_endpoint(vec![MockReply::Oversized]);
        let embedder = EndpointEmbedder::new(
            url,
            "requested-model".into(),
            None,
            String::new(),
            String::new(),
        )
        .unwrap();

        let err = embedder
            .embed_batch(&["one".to_string()])
            .expect_err("oversized response must fail");
        let requests = requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        server.join().unwrap();

        assert!(err.to_string().contains("oversized"), "got: {err}");
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn endpoint_batches_65_inputs_as_64_plus_1_and_preserves_global_order() {
        let (url, requests_rx, server) = mock_endpoint(vec![
            MockReply::Embeddings("resolved-model"),
            MockReply::Embeddings("resolved-model"),
        ]);

        let embedder = EndpointEmbedder::new(
            url,
            "requested-model".into(),
            None,
            String::new(),
            String::new(),
        )
        .unwrap();
        let inputs = (0..65).map(|index| index.to_string()).collect::<Vec<_>>();
        let embedded = embedder.embed_batch_with_identity(&inputs).unwrap();
        let requests = requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        server.join().unwrap();

        assert_eq!(
            requests
                .iter()
                .map(|request| request.inputs.len())
                .collect::<Vec<_>>(),
            vec![64, 1]
        );
        assert_eq!(
            requests
                .into_iter()
                .flat_map(|request| request.inputs)
                .collect::<Vec<_>>(),
            inputs
        );
        assert_eq!(embedded.value.len(), 65);
        assert!(embedded.fingerprint.contains("resolved-model"));
    }

    #[test]
    fn second_chunk_failure_commits_neither_chunk_and_retry_sends_all_inputs() {
        let (url, requests_rx, server) = mock_endpoint(vec![
            MockReply::Embeddings("resolved-model"),
            MockReply::Status(500),
            MockReply::Embeddings("resolved-model"),
            MockReply::Embeddings("resolved-model"),
        ]);
        let mut registry = ToolRegistry::with_embedding(EmbeddingModel::Endpoint {
            url,
            model: "requested-model".into(),
            api_key_env: None,
            query_prefix: None,
            doc_prefix: None,
        });
        for index in 0..65 {
            registry.register(tool_for_endpoint(index));
        }

        assert!(registry.build_embeddings().is_err());
        registry.build_embeddings().unwrap();
        let requests = requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        server.join().unwrap();

        assert_eq!(
            requests
                .iter()
                .map(|request| request.inputs.len())
                .collect::<Vec<_>>(),
            vec![64, 1, 64, 1]
        );
    }

    #[test]
    fn endpoint_cache_separates_api_key_env_names_and_sends_each_bearer_token() {
        const KEY_A: &str = "RATEL_CORE_ENDPOINT_TEST_KEY_A";
        const KEY_B: &str = "RATEL_CORE_ENDPOINT_TEST_KEY_B";
        // Unique test-only names are not read by any other thread in the process.
        unsafe {
            std::env::set_var(KEY_A, "alpha-token");
            std::env::set_var(KEY_B, "beta-token");
        }
        let (url, requests_rx, server) = mock_endpoint(vec![
            MockReply::Embeddings("resolved-model"),
            MockReply::Embeddings("resolved-model"),
        ]);
        for (id, env_name) in [("a", KEY_A), ("b", KEY_B)] {
            let mut registry = ToolRegistry::with_embedding(EmbeddingModel::Endpoint {
                url: url.clone(),
                model: "requested-model".into(),
                api_key_env: Some(env_name.into()),
                query_prefix: None,
                doc_prefix: None,
            });
            registry.register(Tool {
                id: id.into(),
                name: id.into(),
                description: "endpoint auth test".into(),
                experimental_searchable_description: None,
                input_schema: serde_json::json!({}),
                output_schema: serde_json::json!({}),
            });
            registry.build_embeddings().unwrap();
        }
        let requests = requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        server.join().unwrap();
        unsafe {
            std::env::remove_var(KEY_A);
            std::env::remove_var(KEY_B);
        }

        assert_eq!(
            requests
                .into_iter()
                .map(|request| request.authorization)
                .collect::<Vec<_>>(),
            vec![
                Some("Bearer alpha-token".into()),
                Some("Bearer beta-token".into())
            ]
        );
    }

    #[test]
    fn response_model_drift_is_hard_and_rebuild_adopts_the_new_identity() {
        let (url, requests_rx, server) = mock_endpoint(vec![
            MockReply::Embeddings("model-a"),
            MockReply::Embeddings("model-b"),
            MockReply::Embeddings("model-b"),
            MockReply::Embeddings("model-b"),
        ]);
        let mut registry = ToolRegistry::with_embedding(EmbeddingModel::Endpoint {
            url,
            model: "requested-model".into(),
            api_key_env: None,
            query_prefix: None,
            doc_prefix: None,
        });
        registry.register(tool_for_endpoint(0));
        registry.build_embeddings().unwrap();

        assert!(matches!(
            registry.search_with_method("tool", 1, Origin::Direct, SearchMethod::Semantic),
            Err(EmbedderError::ModelMismatch { .. })
        ));
        registry.rebuild_embeddings().unwrap();
        assert_eq!(
            registry
                .search_with_method("tool", 1, Origin::Direct, SearchMethod::Semantic)
                .unwrap()
                .len(),
            1
        );
        let requests = requests_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        server.join().unwrap();
        assert_eq!(requests.len(), 4);
    }

    fn tool_for_endpoint(index: usize) -> Tool {
        Tool {
            id: format!("tool-{index}"),
            name: format!("tool-{index}"),
            description: format!("endpoint tool {index}"),
            experimental_searchable_description: None,
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
        }
    }

    #[derive(Clone, Copy)]
    enum MockReply {
        Embeddings(&'static str),
        Status(u16),
        Oversized,
    }

    struct MockRequest {
        inputs: Vec<String>,
        authorization: Option<String>,
    }

    type MockServer = (
        String,
        mpsc::Receiver<Vec<MockRequest>>,
        std::thread::JoinHandle<()>,
    );

    fn mock_endpoint(replies: Vec<MockReply>) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
        let (requests_tx, requests_rx) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut replies = std::collections::VecDeque::from(replies);
            let mut requests = Vec::new();
            while let Some(reply) = replies.front().copied() {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        let (body, authorization) = read_http_request(&mut stream);
                        let inputs = body["input"]
                            .as_array()
                            .expect("input array")
                            .iter()
                            .map(|value| value.as_str().expect("string input").to_string())
                            .collect::<Vec<_>>();
                        write_mock_response(&mut stream, reply, inputs.len());
                        requests.push(MockRequest {
                            inputs,
                            authorization,
                        });
                        replies.pop_front();
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("accept failed: {error}"),
                }
            }
            requests_tx.send(requests).unwrap();
        });
        (url, requests_rx, server)
    }

    fn write_mock_response(stream: &mut std::net::TcpStream, reply: MockReply, input_len: usize) {
        let (status, response) = match reply {
            MockReply::Embeddings(model) => {
                let data = (0..input_len)
                    .map(|index| {
                        serde_json::json!({
                            "index": index,
                            "embedding": [1.0, 0.0]
                        })
                    })
                    .collect::<Vec<_>>();
                (
                    "200 OK",
                    serde_json::json!({ "data": data, "model": model }).to_string(),
                )
            }
            MockReply::Status(code) => {
                ("500 Internal Server Error", format!("{{\"code\":{code}}}"))
            }
            MockReply::Oversized => {
                write_oversized_response(stream);
                return;
            }
        };
        write!(
            stream,
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            response.len(),
            response
        )
        .unwrap();
    }

    fn write_oversized_response(stream: &mut std::net::TcpStream) {
        const PREFIX: &[u8] = b"{\"data\":[],\"padding\":\"";
        const SUFFIX: &[u8] = b"\"}";
        const CHUNK: &[u8] = &[b'x'; 64 * 1024];
        let padding_len = ENDPOINT_RESPONSE_LIMIT_BYTES;
        let content_len = PREFIX.len() as u64 + padding_len + SUFFIX.len() as u64;
        write!(
            stream,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {content_len}\r\nconnection: close\r\n\r\n"
        )
        .unwrap();
        if stream.write_all(PREFIX).is_err() {
            return;
        }
        let mut remaining = padding_len;
        while remaining > 0 {
            let len = remaining.min(CHUNK.len() as u64) as usize;
            if stream.write_all(&CHUNK[..len]).is_err() {
                return;
            }
            remaining -= len as u64;
        }
        let _ = stream.write_all(SUFFIX);
    }

    fn read_http_request(stream: &mut std::net::TcpStream) -> (serde_json::Value, Option<String>) {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).unwrap();
            assert!(read > 0, "connection closed before request body");
            request.extend_from_slice(&buffer[..read]);
            if let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                let body_start = header_end + 4;
                let headers = std::str::from_utf8(&request[..header_end]).unwrap();
                let content_len = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .expect("content-length");
                if request.len() >= body_start + content_len {
                    let authorization = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("authorization")
                            .then(|| value.trim().to_string())
                    });
                    let body =
                        serde_json::from_slice(&request[body_start..body_start + content_len])
                            .unwrap();
                    return (body, authorization);
                }
            }
        }
    }

    #[test]
    fn mean_pool_averages_only_unmasked_tokens() {
        let dev = Device::Cpu;
        // (1, 2, 2): two tokens, hidden = 2.
        let seq = Tensor::new(&[[[1.0f32, 2.0], [3.0, 4.0]]], &dev).unwrap();
        // Both tokens count → column means [2, 3].
        let all = Tensor::new(&[[1u32, 1]], &dev).unwrap();
        assert_eq!(
            mean_pool(&seq, &all).unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0, 3.0]
        );
        // Only the first token counts (second is padding) → [1, 2].
        let first = Tensor::new(&[[1u32, 0]], &dev).unwrap();
        assert_eq!(
            mean_pool(&seq, &first).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn mean_pool_batch_averages_each_row_over_its_own_unmasked_tokens() {
        let dev = Device::Cpu;
        // (2, 2, 2): two rows, two tokens, hidden = 2.
        let seq = Tensor::new(
            &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            &dev,
        )
        .unwrap();
        // Row 0: both tokens count → [2, 3]. Row 1: second token padded → [5, 6].
        let mask = Tensor::new(&[[1u32, 1], [1, 0]], &dev).unwrap();
        assert_eq!(
            mean_pool_batch(&seq, &mask)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![2.0, 3.0], vec![5.0, 6.0]]
        );
    }

    #[test]
    fn parse_pooling_config_maps_cls_mean_and_none() {
        assert_eq!(
            parse_pooling_config(br#"{"pooling_mode_cls_token": true}"#),
            Some(Pooling::Cls)
        );
        assert_eq!(
            parse_pooling_config(br#"{"pooling_mode_mean_tokens": true}"#),
            Some(Pooling::Mean)
        );
        // A mode we don't support (max) → None → caller assumes Mean.
        assert_eq!(
            parse_pooling_config(br#"{"pooling_mode_max_tokens": true}"#),
            None
        );
        assert_eq!(parse_pooling_config(b"not json"), None);
    }

    #[test]
    fn resolve_pooling_assumes_mean_and_flags_it() {
        assert_eq!(resolve_pooling(Some(Pooling::Cls)), (Pooling::Cls, false));
        assert_eq!(resolve_pooling(Some(Pooling::Mean)), (Pooling::Mean, false));
        assert_eq!(resolve_pooling(None), (Pooling::Mean, true));
    }

    #[test]
    fn local_load_path_does_not_content_hash() {
        use crate::embedding_config::{test_content_hash_calls, test_reset_content_hash_calls};

        let dir = tempfile::tempdir().unwrap();
        // Resolve/stamp succeed; BERT config parse fails inside `build` — so we
        // exercise ordinary Local load past identity stamping without needing a
        // real model, and without ever reaching artifact hashing.
        std::fs::write(dir.path().join("config.json"), b"not-a-bert-config").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();

        test_reset_content_hash_calls();
        let result = CandleEmbedder::load_path(dir.path(), "", "", None);
        assert!(
            matches!(result, Err(EmbedderError::Load { .. })),
            "expected load failure on invalid config"
        );
        assert_eq!(
            test_content_hash_calls(),
            0,
            "CandleEmbedder::load_path must not digest Local model bytes"
        );
    }

    #[test]
    fn is_not_found_distinguishes_missing_file_from_network_error() {
        assert!(is_not_found("Http status client error (404 Not Found)"));
        assert!(is_not_found("Entry Not Found"));
        assert!(!is_not_found("error sending request: connection refused"));
    }

    #[test]
    fn endpoint_missing_api_key_env_is_a_config_error() {
        let e = EndpointEmbedder::new(
            "http://localhost:11434/v1/embeddings".into(),
            "nomic".into(),
            Some("RATEL_TEST_DEFINITELY_UNSET_KEY".into()),
            String::new(),
            String::new(),
        )
        .unwrap();
        let err = e.api_key().unwrap_err();
        assert!(matches!(err, EmbedderError::Config { .. }));
        assert!(err.to_string().contains("RATEL_TEST_DEFINITELY_UNSET_KEY"));
    }

    #[test]
    #[ignore = "downloads the ~130 MB bge model; run with `cargo test -- --ignored`"]
    fn embeds_to_unit_norm_384_vectors_deterministically() {
        let e = embedder_for(&EmbeddingModel::Default)
            .expect("load embedder")
            .0;
        let a = e.embed_doc("read a file from disk").expect("embed");
        let b = e.embed_doc("read a file from disk").expect("embed");
        assert_eq!(a.len(), 384, "bge-small is 384-dim");
        assert_eq!(a, b, "same text must embed identically (determinism)");
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
    }

    #[test]
    #[ignore = "downloads the ~130 MB bge model; run with `cargo test -- --ignored`"]
    fn embed_batch_matches_looping_embed_doc_cls() {
        // The batched forward must be bit-for-bit identical to embedding each doc
        // alone (CLS pooling, built-in bge-small). Differing lengths exercise padding;
        // > EMBED_BATCH_CHUNK items span a chunk boundary.
        let e = embedder_for(&EmbeddingModel::Default)
            .expect("load embedder")
            .0;
        let docs: Vec<String> = (0..EMBED_BATCH_CHUNK + 5)
            .map(|i| format!("{}read a file from disk", "word ".repeat(i % 7)))
            .collect();
        let batched = e.embed_batch(&docs).expect("embed_batch");
        let looped: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| e.embed_doc(d).expect("embed_doc"))
            .collect();
        assert_eq!(
            batched, looped,
            "batched embedding must be bit-for-bit identical to the per-doc path"
        );
    }

    #[test]
    #[ignore = "downloads a mean-pooled model (gte-small); run with `cargo test -- --ignored`"]
    fn embed_batch_matches_looping_embed_doc_mean() {
        // Same exact-equality contract on the mean-pooled path (gte-small).
        let model = EmbeddingModel::HuggingFace {
            repo: "thenlper/gte-small".into(),
            revision: None,
            query_prefix: None,
            doc_prefix: None,
            pooling: None,  // auto-detected → Mean
            download: true, // ignored test: allow the fetch
        };
        let e = embedder_for(&model).expect("load gte-small").0;
        let docs: Vec<String> = (0..EMBED_BATCH_CHUNK + 3)
            .map(|i| format!("{}deploy the service", "x ".repeat(i % 5)))
            .collect();
        let batched = e.embed_batch(&docs).expect("embed_batch");
        let looped: Vec<Vec<f32>> = docs
            .iter()
            .map(|d| e.embed_doc(d).expect("embed_doc"))
            .collect();
        assert_eq!(
            batched, looped,
            "mean-pooled batched embedding must be bit-for-bit identical to the per-doc path"
        );
    }

    #[test]
    #[ignore = "downloads the ~130 MB bge model; run with `cargo test -- --ignored`"]
    fn query_prefix_changes_the_embedding() {
        let e = embedder_for(&EmbeddingModel::Default)
            .expect("load embedder")
            .0;
        let doc = e.embed_doc("delete a file").expect("embed");
        let query = e.embed_query("delete a file").expect("embed");
        assert_ne!(doc, query, "query instruction prefix must shift the vector");
    }

    #[test]
    #[ignore = "downloads the ~130 MB bge model; run with `cargo test -- --ignored`"]
    fn ranks_synonyms_above_lexically_unrelated_text() {
        // The "missing gold" case BM25 can't see: query and doc share no words.
        let e = embedder_for(&EmbeddingModel::Default)
            .expect("load embedder")
            .0;
        let q = e.embed_query("remove a file").expect("embed");
        let delete = e
            .embed_doc("delete a path from the filesystem")
            .expect("embed");
        let weather = e
            .embed_doc("get the current weather forecast")
            .expect("embed");
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &delete) > dot(&q, &weather),
            "semantic match should beat an unrelated tool"
        );
    }

    #[test]
    #[ignore = "downloads a mean-pooled model (gte-small); run with `cargo test -- --ignored`"]
    fn mean_pooled_model_ranks_synonyms_correctly() {
        // gte-small is *mean*-pooled and ships `1_Pooling/config.json`, so pooling
        // must auto-detect Mean — CLS-on-a-mean-model is the silent-quality bug this
        // guards against. Assert a synonym query still ranks the related doc first.
        let model = EmbeddingModel::HuggingFace {
            repo: "thenlper/gte-small".into(),
            revision: None,
            query_prefix: None,
            doc_prefix: None,
            pooling: None,  // auto-detected → Mean
            download: true, // ignored test: allow the fetch
        };
        let e = embedder_for(&model).expect("load gte-small").0;
        let q = e.embed_query("remove a file").expect("embed");
        let delete = e
            .embed_doc("delete a path from the filesystem")
            .expect("embed");
        let weather = e
            .embed_doc("get the current weather forecast")
            .expect("embed");
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(
            dot(&q, &delete) > dot(&q, &weather),
            "mean-pooled semantic match should beat an unrelated tool"
        );
    }
}
