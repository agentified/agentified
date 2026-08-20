//! Which embedding model backs a catalog's semantic/hybrid retrieval.
//!
//! The model is chosen per catalog and declared once, then used for both
//! document and query embedding so the two sides can never land in different
//! vector spaces. Four sources: the built-in default (`bge-small`), any
//! BERT-family HuggingFace repo or on-disk directory (loaded in-process via
//! Candle), and an OpenAI-compatible HTTP endpoint (any model, incl. Ollama).
//!
//! [`EmbeddingModel::resolve`] turns the cross-SDK [`EmbeddingSpec`] DTO into a
//! validated model. The source is named explicitly: a bare string is a **local
//! directory path**, and every other source is a keyed object
//! (`{huggingface}` / `{local}` / `{ollama}` / `{url, model}`), symmetric across
//! the board. Resolution/validation **lives here** (in the core) so both SDKs
//! share one implementation instead of two that could drift. See ADR-0012.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use crate::embedding::EmbedderError;

/// The built-in default: bge-small, pinned to a commit so embeddings are
/// reproducible. These are the canonical identity of the zero-config model;
/// `embedding.rs` loads against them.
pub(crate) const DEFAULT_REPO: &str = "BAAI/bge-small-en-v1.5";
pub(crate) const DEFAULT_REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";
/// bge asymmetric-retrieval query prefix; only the query side gets it.
pub(crate) const DEFAULT_QUERY_INSTRUCTION: &str =
    "Represent this sentence for searching relevant passages: ";

/// Default Ollama OpenAI-compatible embeddings route. The `{ollama: model}`
/// shortcut expands to this; a non-default host uses the full `{url, model}` form.
pub(crate) const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434/v1/embeddings";

/// How a BERT model's per-token outputs are collapsed into one sentence vector.
/// A model is *trained* with one mode — using the other silently degrades ranking
/// — so it is auto-detected from the repo's `1_Pooling/config.json`, with this as
/// an explicit override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// The `[CLS]` (first) token's vector. bge is CLS-pooled.
    Cls,
    /// Masked average of all token vectors. e5/gte/MiniLM/mpnet are mean-pooled.
    Mean,
}

impl Pooling {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Pooling::Cls => "cls",
            Pooling::Mean => "mean",
        }
    }
}

impl std::str::FromStr for Pooling {
    type Err = EmbedderError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cls" => Ok(Pooling::Cls),
            "mean" => Ok(Pooling::Mean),
            other => Err(cfg(format!(
                "unknown pooling '{other}'; expected 'cls' or 'mean'"
            ))),
        }
    }
}

/// Model-identity suffix for the fingerprint / process-cache key. Pooling and the
/// asymmetric prefixes change the produced vectors, so two configs differing only
/// in these must not share a cached embedder or pass the drift check.
pub(crate) fn fingerprint_suffix(
    pooling: Option<Pooling>,
    query_prefix: &str,
    doc_prefix: &str,
) -> String {
    let mut s = String::new();
    if let Some(p) = pooling {
        push_fingerprint_field(&mut s, "pool", p.as_str());
    }
    if !query_prefix.is_empty() {
        push_fingerprint_field(&mut s, "q", query_prefix);
    }
    if !doc_prefix.is_empty() {
        push_fingerprint_field(&mut s, "d", doc_prefix);
    }
    s
}

pub(crate) fn huggingface_fingerprint(repo: &str, revision: &str) -> String {
    fingerprint("hf", &[("repo", repo), ("revision", revision)])
}

/// Runtime / process-cache Local identity: the configured directory path.
/// Pre-PR spelling — IntentGraph and dense-cache runtime stamps depend on it.
pub(crate) fn local_fingerprint(path: &str) -> String {
    fingerprint("local", &[("path", path)])
}

/// RAT1 artifact Local identity: content digest of the effective model inputs.
pub(crate) fn local_content_fingerprint(content_id: &str) -> String {
    fingerprint("local", &[("content", content_id)])
}

pub(crate) fn endpoint_fingerprint(url: &str, model: &str) -> String {
    fingerprint("endpoint", &[("url", url), ("model", model)])
}

/// `(len, mtime)` stamp for Local artifact coherence checks / content-id memo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileStamp {
    pub(crate) len: u64,
    pub(crate) modified: Option<SystemTime>,
}

struct ContentIdMemo {
    stamps: Vec<FileStamp>,
    content_id: String,
}

/// Files Candle opens for a local model. Shared by load and RAT1 artifact hashing
/// so the identity set cannot drift from what is actually read.
#[derive(Clone, Debug)]
pub(crate) struct LocalModelFiles {
    pub(crate) config: PathBuf,
    pub(crate) tokenizer: PathBuf,
    pub(crate) weights: PathBuf,
    /// Present when `1_Pooling/config.json` exists — used for autodetection only;
    /// never part of the content digest (effective pooling is the resolved mode).
    pub(crate) pooling_config: Option<PathBuf>,
}

impl LocalModelFiles {
    /// Ordered paths whose bytes define the Local artifact content identity.
    pub(crate) fn content_hash_paths(&self) -> [&Path; 3] {
        [&self.config, &self.tokenizer, &self.weights]
    }
}

/// Resolve the Local model files Candle loads from `dir`.
pub(crate) fn resolve_local_model_files(dir: &Path) -> Result<LocalModelFiles, EmbedderError> {
    let name = dir.display().to_string();
    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    let weights = [dir.join("model.safetensors"), dir.join("pytorch_model.bin")]
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| EmbedderError::Load {
            model: name.clone(),
            source: format!("missing model.safetensors / pytorch_model.bin in {name}"),
        })?;
    for (p, f) in [(&config, "config.json"), (&tokenizer, "tokenizer.json")] {
        if !p.exists() {
            return Err(EmbedderError::Load {
                model: name.clone(),
                source: format!(
                    "missing {f} in {name} — a fast tokenizer.json is required; run \
                     tokenizer.save_pretrained() upstream, or serve the model via an endpoint"
                ),
            });
        }
    }
    let pooling = dir.join("1_Pooling/config.json");
    let pooling_config = pooling.exists().then_some(pooling);
    Ok(LocalModelFiles {
        config,
        tokenizer,
        weights,
        pooling_config,
    })
}

/// Stat the three content-hash paths at load time (metadata only — not a digest).
pub(crate) fn stamp_local_hash_paths(
    files: &LocalModelFiles,
    model: &str,
) -> Result<[FileStamp; 3], EmbedderError> {
    let paths = files.content_hash_paths();
    Ok([
        stamp_file(paths[0], model)?,
        stamp_file(paths[1], model)?,
        stamp_file(paths[2], model)?,
    ])
}

pub(crate) fn stamp_file(path: &Path, model: &str) -> Result<FileStamp, EmbedderError> {
    let meta = std::fs::metadata(path).map_err(|e| EmbedderError::Load {
        model: model.to_string(),
        source: format!("stat {}: {e}", path.display()),
    })?;
    Ok(FileStamp {
        len: meta.len(),
        modified: meta.modified().ok(),
    })
}

/// Content digest of Local model inputs that produce vectors, memoized against
/// `(len, mtime)`. Used only for RAT1 artifact identity — never for runtime
/// cache keys or ordinary dense load. Load-stamp coherence is
/// [`LocalContentIdentity`].
pub(crate) fn local_model_content_id_from_paths(
    dir: &Path,
    paths: &[&Path],
) -> Result<String, EmbedderError> {
    let name = dir.display().to_string();
    let stamps: Vec<FileStamp> = paths
        .iter()
        .map(|p| stamp_file(p, &name))
        .collect::<Result<_, _>>()?;

    let memo_key = dir
        .canonicalize()
        .unwrap_or_else(|_| dir.to_path_buf())
        .display()
        .to_string();

    {
        let cache = content_id_memo();
        let guard = cache.lock().expect("local content-id memo poisoned");
        if let Some(entry) = guard.get(&memo_key)
            && entry.stamps == stamps
        {
            return Ok(entry.content_id.clone());
        }
    }

    let path_bufs: Vec<PathBuf> = paths.iter().map(|p| p.to_path_buf()).collect();
    let content_id = hash_local_identity_files(&path_bufs, &name)?;
    let mut guard = content_id_memo()
        .lock()
        .expect("local content-id memo poisoned");
    guard.insert(
        memo_key,
        ContentIdMemo {
            stamps,
            content_id: content_id.clone(),
        },
    );
    Ok(content_id)
}

pub(crate) struct LocalContentIdentity {
    state: Mutex<LocalIdentityState>,
}

#[derive(Clone)]
struct LocalIdentityState {
    stamps: Vec<FileStamp>,
    established: Option<String>,
}

impl LocalContentIdentity {
    pub(crate) fn new(load_stamps: [FileStamp; 3]) -> Self {
        Self {
            state: Mutex::new(LocalIdentityState {
                stamps: load_stamps.to_vec(),
                established: None,
            }),
        }
    }

    /// Hash and restat with the mutex released; commit only if stamps are unchanged.
    pub(crate) fn content_id(&self, dir: &Path, paths: &[&Path]) -> Result<String, EmbedderError> {
        loop {
            let now = stamp_paths(dir, paths)?;
            let snapshot = {
                let guard = self.state.lock().expect("local content identity poisoned");
                guard.clone()
            };

            if now == snapshot.stamps {
                if let Some(id) = &snapshot.established {
                    return Ok(id.clone());
                }
            } else if snapshot.established.is_none() {
                return Err(drift_before_established(dir));
            }

            let id = local_model_content_id_from_paths(dir, paths)?;
            #[cfg(test)]
            after_hash_hook::take_and_run();
            let after = stamp_paths(dir, paths)?;
            if after != now {
                continue;
            }

            let mut guard = self.state.lock().expect("local content identity poisoned");
            if guard.stamps != snapshot.stamps || guard.established != snapshot.established {
                continue;
            }
            match &snapshot.established {
                None => {
                    guard.stamps = now;
                    guard.established = Some(id.clone());
                    return Ok(id);
                }
                Some(known) if known == &id => {
                    guard.stamps = now;
                    return Ok(known.clone());
                }
                Some(_) => return Err(contents_differ(dir)),
            }
        }
    }
}

fn stamp_paths(dir: &Path, paths: &[&Path]) -> Result<Vec<FileStamp>, EmbedderError> {
    let model = dir.display().to_string();
    paths.iter().map(|p| stamp_file(p, &model)).collect()
}

fn drift_before_established(dir: &Path) -> EmbedderError {
    let model = dir.display().to_string();
    EmbedderError::Load {
        model: model.clone(),
        source: format!(
            "local model files under {model} changed after the model was loaded and before this \
             process established its artifact identity — the resident model may no longer match \
             the files on disk; start a new process to build or warm an embedding artifact from \
             the current files"
        ),
    }
}

fn contents_differ(dir: &Path) -> EmbedderError {
    let model = dir.display().to_string();
    EmbedderError::Load {
        model: model.clone(),
        source: format!(
            "local model files under {model} changed since the model was loaded (contents differ, \
             not just timestamps) — the resident model still holds the previous weights; start a \
             new process to build or warm an embedding artifact from the current files"
        ),
    }
}

/// Compose the portable Local RAT1 artifact identity for a directory, using the
/// same effective pooling resolution as load (`override` → pooling file → Mean).
/// Test/helper surface — production Local artifact identity goes through
/// [`crate::embedding::Embedder::artifact_identity`] on a loaded Candle embedder.
#[cfg(test)]
pub(crate) fn local_artifact_fingerprint(
    dir: &Path,
    pooling_override: Option<Pooling>,
    query_prefix: &str,
    doc_prefix: &str,
) -> Result<String, EmbedderError> {
    let files = resolve_local_model_files(dir)?;
    let detected = pooling_override.or_else(|| {
        files
            .pooling_config
            .as_ref()
            .and_then(|p| detect_pooling_config_file(p))
    });
    // Same fallback as `resolve_pooling` in embedding.rs (Mean when undetected).
    let pooling = detected.unwrap_or(Pooling::Mean);
    let paths = files.content_hash_paths();
    let content_id = local_model_content_id_from_paths(dir, &paths)?;
    Ok(format!(
        "{}{}",
        local_content_fingerprint(&content_id),
        fingerprint_suffix(Some(pooling), query_prefix, doc_prefix)
    ))
}

/// Pure `1_Pooling/config.json` → [`Pooling`] mapping (shared with the loader).
pub(crate) fn parse_pooling_config(bytes: &[u8]) -> Option<Pooling> {
    #[derive(serde::Deserialize)]
    struct PoolingConfig {
        #[serde(default)]
        pooling_mode_cls_token: bool,
        #[serde(default)]
        pooling_mode_mean_tokens: bool,
    }
    let c: PoolingConfig = serde_json::from_slice(bytes).ok()?;
    if c.pooling_mode_cls_token {
        Some(Pooling::Cls)
    } else if c.pooling_mode_mean_tokens {
        Some(Pooling::Mean)
    } else {
        None
    }
}

#[cfg(test)]
fn detect_pooling_config_file(path: &Path) -> Option<Pooling> {
    let bytes = std::fs::read(path).ok()?;
    parse_pooling_config(&bytes)
}

fn content_id_memo() -> &'static Mutex<HashMap<String, ContentIdMemo>> {
    static CELL: OnceLock<Mutex<HashMap<String, ContentIdMemo>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
mod content_hash_probe {
    use std::cell::Cell;

    thread_local! {
        static CALLS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn note_call() {
        CALLS.with(|c| c.set(c.get() + 1));
    }

    pub(crate) fn reset() {
        CALLS.with(|c| c.set(0));
    }

    pub(crate) fn count() -> usize {
        CALLS.with(Cell::get)
    }
}

#[cfg(test)]
mod after_hash_hook {
    use std::cell::RefCell;

    thread_local! {
        static HOOK: RefCell<Option<Box<dyn FnOnce()>>> = RefCell::new(None);
    }

    pub(super) fn set(hook: impl FnOnce() + 'static) {
        HOOK.with(|h| *h.borrow_mut() = Some(Box::new(hook)));
    }

    pub(super) fn take_and_run() {
        if let Some(hook) = HOOK.with(|h| h.borrow_mut().take()) {
            hook();
        }
    }
}

#[cfg(test)]
pub(crate) fn test_reset_content_hash_calls() {
    content_hash_probe::reset();
}

#[cfg(test)]
pub(crate) fn test_content_hash_calls() -> usize {
    content_hash_probe::count()
}

fn hash_local_identity_files(files: &[PathBuf], model: &str) -> Result<String, EmbedderError> {
    #[cfg(test)]
    content_hash_probe::note_call();
    let mut hasher = Sha256::new();
    for path in files {
        // Length-prefix each file so concatenation cannot collide across a
        // boundary (e.g. "ab"+"c" vs "a"+"bc").
        let mut file = File::open(path).map_err(|e| EmbedderError::Load {
            model: model.to_string(),
            source: format!("open {}: {e}", path.display()),
        })?;
        let len = file
            .metadata()
            .map_err(|e| EmbedderError::Load {
                model: model.to_string(),
                source: format!("stat {}: {e}", path.display()),
            })?
            .len();
        hasher.update(len.to_le_bytes());
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf).map_err(|e| EmbedderError::Load {
                model: model.to_string(),
                source: format!("read {}: {e}", path.display()),
            })?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    Ok(hex_lower(hasher.finalize()))
}

fn hex_lower(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// The embedding model backing a catalog's semantic/hybrid engines.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingModel {
    /// Built-in `bge-small-en-v1.5`, pinned. The zero-config default.
    Default,
    /// A BERT-family HuggingFace repo, loaded in-process via Candle. `revision`
    /// defaults to `main`; `pooling` is auto-detected when `None`. `download`
    /// (default `false`) must be opted into for Ratel to fetch it — otherwise it
    /// must already be in the local cache (Ratel auto-downloads only the default).
    HuggingFace {
        /// HuggingFace repo id (e.g. `intfloat/e5-small-v2`).
        repo: String,
        /// Git revision to pin; `None` → `main`.
        revision: Option<String>,
        /// Query-side prefix for asymmetric models.
        query_prefix: Option<String>,
        /// Document-side prefix for asymmetric models.
        doc_prefix: Option<String>,
        /// Pooling override; `None` auto-detects.
        pooling: Option<Pooling>,
        /// Opt in to downloading if not already cached.
        download: bool,
    },
    /// A BERT-family model directory on disk (`config.json` / `tokenizer.json` /
    /// `model.safetensors`), loaded in-process via Candle.
    Local {
        /// Path to the model directory.
        path: PathBuf,
        /// Query-side prefix for asymmetric models.
        query_prefix: Option<String>,
        /// Document-side prefix for asymmetric models.
        doc_prefix: Option<String>,
        /// Pooling override; `None` auto-detects.
        pooling: Option<Pooling>,
    },
    /// An OpenAI-compatible `/embeddings` HTTP endpoint (OpenAI, Ollama, TEI,
    /// vLLM…). `api_key_env` names the env var holding the key (read at call time).
    /// Pooling lives server-side, so there is no `pooling` here.
    Endpoint {
        /// Full endpoint URL.
        url: String,
        /// Model name sent in the request body.
        model: String,
        /// Env var holding the bearer key; `None` for no auth.
        api_key_env: Option<String>,
        /// Query-side prefix for asymmetric models.
        query_prefix: Option<String>,
        /// Document-side prefix for asymmetric models.
        doc_prefix: Option<String>,
    },
}

/// Normalized, cross-SDK embedding config as forwarded by the native bindings.
/// Exactly one *primary* source must be set: either `spec` (the raw string
/// shortcut) or one of `huggingface` / `local` / `ollama` / `url`. The rest are
/// modifiers.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingSpec {
    /// Raw string shortcut — a **local model directory path** only. A repo-id or
    /// URL string is rejected in favor of the explicit `huggingface`/`url` keys.
    pub spec: Option<String>,
    /// Primary source: a HuggingFace repo id.
    pub huggingface: Option<String>,
    /// Primary source: a local model directory path.
    pub local: Option<String>,
    /// Primary source: an Ollama model name (served via the local Ollama endpoint).
    pub ollama: Option<String>,
    /// Primary source: a full OpenAI-compatible endpoint URL.
    pub url: Option<String>,
    /// Model name for an endpoint source.
    pub model: Option<String>,
    /// Git revision for a HuggingFace source.
    pub revision: Option<String>,
    /// Env var holding the endpoint bearer key.
    pub api_key_env: Option<String>,
    /// Query-side prefix for asymmetric models.
    pub query_prefix: Option<String>,
    /// Document-side prefix for asymmetric models (e.g. e5's `"passage: "`).
    pub doc_prefix: Option<String>,
    /// `"cls"` | `"mean"` — overrides auto-detection for an in-process model.
    pub pooling: Option<String>,
    /// Opt in to letting Ratel download a HuggingFace model that is not yet
    /// cached (default `false`; the built-in default always downloads).
    pub download: Option<bool>,
}

fn cfg(message: impl Into<String>) -> EmbedderError {
    EmbedderError::Config {
        message: message.into(),
    }
}

impl EmbeddingModel {
    /// Validate a concrete Rust model value. SDK configs normally enter through
    /// [`Self::resolve`], but Rust callers can construct public enum variants
    /// directly; this keeps that path subject to the same nonblank-field rules.
    ///
    /// # Errors
    ///
    /// [`EmbedderError::Config`] when a required source, model, URL, or env-var
    /// name is blank.
    pub fn validate(&self) -> Result<(), EmbedderError> {
        match self {
            EmbeddingModel::Default => Ok(()),
            EmbeddingModel::HuggingFace { repo, .. } => validate_nonblank("huggingface", repo),
            EmbeddingModel::Local { path, .. } => {
                validate_nonblank("local", &path.to_string_lossy())
            }
            EmbeddingModel::Endpoint {
                url,
                model,
                api_key_env,
                ..
            } => {
                validate_nonblank("url", url)?;
                validate_nonblank("model", model)?;
                if let Some(api_key_env) = api_key_env {
                    validate_nonblank("api_key_env", api_key_env)?;
                }
                Ok(())
            }
        }
    }

    /// Validate and resolve a spec into a concrete model. Runs at catalog
    /// construction, so config mistakes surface immediately (not at first search).
    pub fn resolve(spec: EmbeddingSpec) -> Result<EmbeddingModel, EmbedderError> {
        for (name, value) in [
            ("spec", spec.spec.as_deref()),
            ("huggingface", spec.huggingface.as_deref()),
            ("local", spec.local.as_deref()),
            ("ollama", spec.ollama.as_deref()),
            ("url", spec.url.as_deref()),
            ("model", spec.model.as_deref()),
            ("api_key_env", spec.api_key_env.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(cfg(format!("embedding '{name}' must not be blank")));
            }
        }

        let primaries = [
            ("spec", spec.spec.is_some()),
            ("huggingface", spec.huggingface.is_some()),
            ("local", spec.local.is_some()),
            ("ollama", spec.ollama.is_some()),
            ("url", spec.url.is_some()),
        ];
        let set: Vec<&str> = primaries
            .iter()
            .filter(|(_, present)| *present)
            .map(|(key, _)| *key)
            .collect();
        match set.len() {
            0 => {
                return Err(cfg(
                    "no embedding source given; pass a local directory path, or one of \
                     huggingface/local/ollama/url",
                ));
            }
            1 => {}
            _ => {
                return Err(cfg(format!(
                    "conflicting embedding keys {set:?}; give exactly one of \
                     spec/huggingface/local/ollama/url",
                )));
            }
        }

        // Parse the pooling override once (validates the string); only in-process
        // sources may carry it.
        let pooling = spec
            .pooling
            .as_deref()
            .map(str::parse::<Pooling>)
            .transpose()?;

        // `download` is a HuggingFace-only fetch policy.
        if spec.download.is_some() && set[0] != "huggingface" {
            return Err(cfg("'download' is only valid for a HuggingFace repo"));
        }

        let model = match set[0] {
            "spec" => infer_from_string(spec.spec.as_deref().unwrap(), &spec, pooling),
            "huggingface" => {
                reject_endpoint_only(&spec, "a HuggingFace repo")?;
                Ok(EmbeddingModel::HuggingFace {
                    repo: spec.huggingface.unwrap(),
                    revision: spec.revision,
                    query_prefix: spec.query_prefix,
                    doc_prefix: spec.doc_prefix,
                    pooling,
                    download: spec.download.unwrap_or(false),
                })
            }
            "local" => {
                reject_endpoint_only(&spec, "a local model")?;
                if spec.revision.is_some() {
                    return Err(cfg("'revision' is only valid for a HuggingFace repo"));
                }
                Ok(EmbeddingModel::Local {
                    path: PathBuf::from(spec.local.unwrap()),
                    query_prefix: spec.query_prefix,
                    doc_prefix: spec.doc_prefix,
                    pooling,
                })
            }
            "ollama" => {
                if spec.model.is_some() {
                    return Err(cfg(
                        "'model' is redundant with 'ollama' (the ollama value is the model name)",
                    ));
                }
                if spec.api_key_env.is_some() {
                    return Err(cfg(
                        "'api_key_env' is not valid with the Ollama shortcut; use a full endpoint 'url'",
                    ));
                }
                reject_in_process_only(&spec, pooling)?;
                Ok(EmbeddingModel::Endpoint {
                    url: OLLAMA_DEFAULT_URL.to_string(),
                    model: spec.ollama.unwrap(),
                    api_key_env: None,
                    query_prefix: spec.query_prefix,
                    doc_prefix: spec.doc_prefix,
                })
            }
            "url" => {
                reject_in_process_only(&spec, pooling)?;
                let model = spec
                    .model
                    .ok_or_else(|| cfg("endpoint embedding requires both 'url' and 'model'"))?;
                Ok(EmbeddingModel::Endpoint {
                    url: spec.url.unwrap(),
                    model,
                    api_key_env: spec.api_key_env,
                    query_prefix: spec.query_prefix,
                    doc_prefix: spec.doc_prefix,
                })
            }
            _ => unreachable!("primary key set is closed"),
        }?;
        model.validate()?;
        Ok(model)
    }

    /// The query-side instruction prefix (bge is asymmetric). Empty unless the
    /// model sets one; the built-in default carries bge's instruction.
    pub(crate) fn query_prefix(&self) -> &str {
        match self {
            EmbeddingModel::Default => DEFAULT_QUERY_INSTRUCTION,
            EmbeddingModel::HuggingFace { query_prefix, .. }
            | EmbeddingModel::Local { query_prefix, .. }
            | EmbeddingModel::Endpoint { query_prefix, .. } => {
                query_prefix.as_deref().unwrap_or("")
            }
        }
    }

    /// The document-side prefix (asymmetric models like e5 use `"passage: "`).
    /// Empty unless the model sets one.
    pub(crate) fn doc_prefix(&self) -> &str {
        match self {
            EmbeddingModel::Default => "",
            EmbeddingModel::HuggingFace { doc_prefix, .. }
            | EmbeddingModel::Local { doc_prefix, .. }
            | EmbeddingModel::Endpoint { doc_prefix, .. } => doc_prefix.as_deref().unwrap_or(""),
        }
    }

    /// The explicit pooling override, if any. The built-in default is pinned to
    /// CLS (so it never needs the pooling-config fetch); an in-process model uses
    /// its field (auto-detected when `None`); an endpoint pools server-side.
    pub(crate) fn pooling_override(&self) -> Option<Pooling> {
        match self {
            EmbeddingModel::Default => Some(Pooling::Cls),
            EmbeddingModel::HuggingFace { pooling, .. } | EmbeddingModel::Local { pooling, .. } => {
                *pooling
            }
            EmbeddingModel::Endpoint { .. } => None,
        }
    }

    /// Human-readable model name for telemetry (the `model` field of load
    /// events). Friendlier than the fingerprint.
    pub(crate) fn display_name(&self) -> String {
        match self {
            EmbeddingModel::Default => DEFAULT_REPO.to_string(),
            EmbeddingModel::HuggingFace { repo, .. } => repo.clone(),
            EmbeddingModel::Local { path, .. } => path.display().to_string(),
            EmbeddingModel::Endpoint { url, model, .. } => format!("{model} @ {url}"),
        }
    }

    /// Pre-load identity used to **key the process model cache** so two catalogs
    /// on the same model load it once. HF `revision` may still be `main` here;
    /// the resolved-with-SHA form is [`crate::embedding::Embedder::fingerprint`],
    /// stamped on the dense cache after load. Local uses the path-derived runtime
    /// spelling (not the RAT1 content digest).
    pub(crate) fn configured_fingerprint(&self) -> String {
        let base = match self {
            EmbeddingModel::Default => huggingface_fingerprint(DEFAULT_REPO, DEFAULT_REVISION),
            EmbeddingModel::HuggingFace { repo, revision, .. } => {
                huggingface_fingerprint(repo, revision.as_deref().unwrap_or("main"))
            }
            EmbeddingModel::Local { path, .. } => local_fingerprint(&path.display().to_string()),
            EmbeddingModel::Endpoint { url, model, .. } => endpoint_fingerprint(url, model),
        };
        // Pooling + prefixes change the vectors, so they are part of the identity.
        format!(
            "{base}{}",
            fingerprint_suffix(
                self.pooling_override(),
                self.query_prefix(),
                self.doc_prefix()
            )
        )
    }

    /// Process-cache identity for the client/embedder instance. Credentials do
    /// not change the vector space, so [`Self::configured_fingerprint`] excludes
    /// them; the cached endpoint client does capture the *name* of the env var it
    /// reads, however, so that non-secret name must distinguish client instances.
    pub(crate) fn embedder_cache_key(&self) -> String {
        let vector_identity = self.configured_fingerprint();
        match self {
            EmbeddingModel::Endpoint { api_key_env, .. } => match api_key_env {
                Some(name) => {
                    let mut key = vector_identity;
                    push_fingerprint_field(&mut key, "api_key_env", name);
                    key
                }
                None => format!("{vector_identity}|api_key_env=none"),
            },
            _ => vector_identity,
        }
    }
}

fn fingerprint(kind: &str, fields: &[(&str, &str)]) -> String {
    let mut fingerprint = kind.to_string();
    for (name, value) in fields {
        push_fingerprint_field(&mut fingerprint, name, value);
    }
    fingerprint
}

fn push_fingerprint_field(fingerprint: &mut String, name: &str, value: &str) {
    fingerprint.push_str(&format!("|{name}={}:{}", value.len(), value));
}

fn validate_nonblank(name: &str, value: &str) -> Result<(), EmbedderError> {
    if value.trim().is_empty() {
        return Err(cfg(format!("embedding '{name}' must not be blank")));
    }
    Ok(())
}

/// Reject endpoint-only modifiers on an in-process (HF/local) source.
fn reject_endpoint_only(spec: &EmbeddingSpec, what: &str) -> Result<(), EmbedderError> {
    if spec.model.is_some() {
        return Err(cfg(format!(
            "'model' is only valid with an endpoint 'url', not {what}"
        )));
    }
    if spec.api_key_env.is_some() {
        return Err(cfg(format!(
            "'api_key_env' is only valid with an endpoint 'url', not {what}"
        )));
    }
    Ok(())
}

/// Reject in-process-only modifiers on an endpoint source: `revision` (no HF
/// fetch) and `pooling` (the server pools).
fn reject_in_process_only(
    spec: &EmbeddingSpec,
    pooling: Option<Pooling>,
) -> Result<(), EmbedderError> {
    if spec.revision.is_some() {
        return Err(cfg("'revision' is only valid for a HuggingFace repo"));
    }
    if pooling.is_some() {
        return Err(cfg(
            "'pooling' is only valid for an in-process model (huggingface/local); \
             an endpoint pools server-side",
        ));
    }
    Ok(())
}

/// Interpret the raw string shortcut, which is a **local model directory path
/// only** — every non-path source (HuggingFace, endpoint) uses the explicit
/// keyed object, symmetric with `{ollama}`/`{url}`. A URL or a repo-id-looking
/// string is rejected with a pointer to the right object form, so the source is
/// never guessed from an ambiguous string.
fn infer_from_string(
    s: &str,
    spec: &EmbeddingSpec,
    pooling: Option<Pooling>,
) -> Result<EmbeddingModel, EmbedderError> {
    if spec.model.is_some() || spec.api_key_env.is_some() {
        return Err(cfg(
            "'model'/'api_key_env' are only valid with an endpoint 'url'; a bare string \
             is only a local model directory path",
        ));
    }
    if spec.revision.is_some() {
        return Err(cfg("'revision' is only valid for a HuggingFace repo; use \
             {\"huggingface\": \"…\", \"revision\": \"…\"}"));
    }
    if looks_like_url(s) {
        return Err(cfg(format!(
            "'{s}' looks like an endpoint URL but has no model name; use \
             {{\"url\": \"{s}\", \"model\": \"…\"}}"
        )));
    }
    if looks_like_path(s) || Path::new(s).is_dir() {
        return Ok(EmbeddingModel::Local {
            path: PathBuf::from(s),
            query_prefix: spec.query_prefix.clone(),
            doc_prefix: spec.doc_prefix.clone(),
            pooling,
        });
    }
    // Not a path → most likely a HuggingFace repo id. The bare-string form is
    // local-only, so name the explicit object rather than guess the source.
    Err(cfg(format!(
        "'{s}' is not a local directory path; to use a HuggingFace repo pass \
         {{\"huggingface\": \"{s}\"}}, or give an absolute/relative directory \
         path for a local model"
    )))
}

/// A `scheme://…` URL. Requires `://`, so a Windows `C:\…` path never matches.
fn looks_like_url(s: &str) -> bool {
    match s.find("://") {
        Some(idx) if idx > 0 => {
            let scheme = &s[..idx];
            scheme
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
        }
        _ => false,
    }
}

/// An unambiguous local-path intent — even if the path doesn't exist yet, so a
/// mistyped local path errors as "not found" rather than a phantom HF repo.
fn looks_like_path(s: &str) -> bool {
    s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("../")
        || s.starts_with('~')
        || s.starts_with(r"\\") // UNC
        || s.starts_with(r".\")
        || s.starts_with(r"..\")
        || is_windows_drive(s)
}

/// `C:\` or `C:/` — a drive letter, colon, separator.
fn is_windows_drive(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_str(s: &str) -> Result<EmbeddingModel, EmbedderError> {
        EmbeddingModel::resolve(EmbeddingSpec {
            spec: Some(s.to_string()),
            ..Default::default()
        })
    }

    #[test]
    fn bare_repo_id_string_is_rejected_pointing_to_huggingface() {
        // A repo-id-looking string is not a path → rejected with a pointer to the
        // explicit object form, so the source is never guessed.
        let err = from_str("BAAI/bge-base-en-v1.5").unwrap_err();
        assert!(matches!(err, EmbedderError::Config { .. }));
        assert!(err.to_string().contains("huggingface"), "got: {err}");
    }

    #[test]
    fn huggingface_object_infers_default_revision() {
        assert_eq!(
            EmbeddingModel::resolve(EmbeddingSpec {
                huggingface: Some("BAAI/bge-base-en-v1.5".into()),
                ..Default::default()
            })
            .unwrap(),
            EmbeddingModel::HuggingFace {
                repo: "BAAI/bge-base-en-v1.5".into(),
                revision: None,
                query_prefix: None,
                doc_prefix: None,
                pooling: None,
                download: false,
            }
        );
    }

    #[test]
    fn absolute_and_relative_paths_infer_local_even_when_absent() {
        for p in [
            "/opt/models/x",
            "./models/x",
            "../x",
            "~/models/x",
            r"\\host\share\x",
        ] {
            assert!(
                matches!(from_str(p).unwrap(), EmbeddingModel::Local { .. }),
                "{p} should infer Local"
            );
        }
    }

    #[test]
    fn windows_drive_path_is_local_not_url() {
        for p in [r"C:\models\bge", "C:/models/bge"] {
            assert!(
                matches!(from_str(p).unwrap(), EmbeddingModel::Local { .. }),
                "{p} should infer Local, not be mistaken for a URL"
            );
        }
    }

    #[test]
    fn existing_directory_infers_local() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        assert!(matches!(
            from_str(path).unwrap(),
            EmbeddingModel::Local { .. }
        ));
    }

    #[test]
    fn bare_url_string_is_rejected_needs_model() {
        let err = from_str("https://api.openai.com/v1/embeddings").unwrap_err();
        assert!(matches!(err, EmbedderError::Config { .. }));
        assert!(err.to_string().contains("model"), "got: {err}");
    }

    #[test]
    fn ollama_object_expands_to_localhost_endpoint() {
        let m = EmbeddingModel::resolve(EmbeddingSpec {
            ollama: Some("nomic-embed-text".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            m,
            EmbeddingModel::Endpoint {
                url: OLLAMA_DEFAULT_URL.into(),
                model: "nomic-embed-text".into(),
                api_key_env: None,
                query_prefix: None,
                doc_prefix: None,
            }
        );
    }

    #[test]
    fn endpoint_object_requires_model() {
        let err = EmbeddingModel::resolve(EmbeddingSpec {
            url: Some("https://api.openai.com/v1/embeddings".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("'url' and 'model'"), "got: {err}");
    }

    #[test]
    fn ollama_and_url_together_conflict() {
        let err = EmbeddingModel::resolve(EmbeddingSpec {
            ollama: Some("nomic".into()),
            url: Some("http://host:11434/v1/embeddings".into()),
            model: Some("nomic".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("conflicting"), "got: {err}");
    }

    #[test]
    fn huggingface_object_with_revision() {
        let m = EmbeddingModel::resolve(EmbeddingSpec {
            huggingface: Some("BAAI/bge-base-en-v1.5".into()),
            revision: Some("abc123".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            m,
            EmbeddingModel::HuggingFace {
                repo: "BAAI/bge-base-en-v1.5".into(),
                revision: Some("abc123".into()),
                query_prefix: None,
                doc_prefix: None,
                pooling: None,
                download: false,
            }
        );
    }

    #[test]
    fn empty_spec_is_rejected() {
        assert!(EmbeddingModel::resolve(EmbeddingSpec::default()).is_err());
    }

    #[test]
    fn blank_source_and_endpoint_fields_are_rejected() {
        for spec in [
            EmbeddingSpec {
                huggingface: Some("   ".into()),
                ..Default::default()
            },
            EmbeddingSpec {
                local: Some("\t".into()),
                ..Default::default()
            },
            EmbeddingSpec {
                ollama: Some("\n".into()),
                ..Default::default()
            },
            EmbeddingSpec {
                url: Some(" ".into()),
                model: Some("model".into()),
                ..Default::default()
            },
            EmbeddingSpec {
                url: Some("http://localhost/v1/embeddings".into()),
                model: Some(" ".into()),
                ..Default::default()
            },
            EmbeddingSpec {
                url: Some("http://localhost/v1/embeddings".into()),
                model: Some("model".into()),
                api_key_env: Some(" ".into()),
                ..Default::default()
            },
        ] {
            assert!(matches!(
                EmbeddingModel::resolve(spec),
                Err(EmbedderError::Config { .. })
            ));
        }
    }

    #[test]
    fn ollama_rejects_api_key_env_instead_of_ignoring_it() {
        let err = EmbeddingModel::resolve(EmbeddingSpec {
            ollama: Some("nomic-embed-text".into()),
            api_key_env: Some("OLLAMA_KEY".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(matches!(err, EmbedderError::Config { .. }));
        assert!(err.to_string().contains("api_key_env"));
    }

    #[test]
    fn download_defaults_false_and_is_huggingface_only() {
        // Default off — explicit HF models are cache-only unless opted in.
        assert!(matches!(
            EmbeddingModel::resolve(EmbeddingSpec {
                huggingface: Some("org/m".into()),
                ..Default::default()
            })
            .unwrap(),
            EmbeddingModel::HuggingFace {
                download: false,
                ..
            }
        ));
        // Opt in.
        assert!(matches!(
            EmbeddingModel::resolve(EmbeddingSpec {
                huggingface: Some("org/m".into()),
                download: Some(true),
                ..Default::default()
            })
            .unwrap(),
            EmbeddingModel::HuggingFace { download: true, .. }
        ));
        // Meaningless on a non-HF source.
        let err = EmbeddingModel::resolve(EmbeddingSpec {
            ollama: Some("nomic".into()),
            download: Some(true),
            ..Default::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("download"), "got: {err}");
    }

    #[test]
    fn default_query_prefix_is_bge_instruction() {
        assert_eq!(
            EmbeddingModel::Default.query_prefix(),
            DEFAULT_QUERY_INSTRUCTION
        );
        assert_eq!(
            EmbeddingModel::Endpoint {
                url: "u".into(),
                model: "m".into(),
                api_key_env: None,
                query_prefix: None,
                doc_prefix: None,
            }
            .query_prefix(),
            ""
        );
    }

    #[test]
    fn fingerprints_are_distinct_per_source() {
        // The default carries its pinned CLS pooling + bge query prefix.
        assert_eq!(
            EmbeddingModel::Default.configured_fingerprint(),
            format!(
                "hf|repo={}:{}|revision={}:{}|pool=3:cls|q={}:{}",
                DEFAULT_REPO.len(),
                DEFAULT_REPO,
                DEFAULT_REVISION.len(),
                DEFAULT_REVISION,
                DEFAULT_QUERY_INSTRUCTION.len(),
                DEFAULT_QUERY_INSTRUCTION
            )
        );
        assert_eq!(
            EmbeddingModel::HuggingFace {
                repo: "r".into(),
                revision: None,
                query_prefix: None,
                doc_prefix: None,
                pooling: None,
                download: false,
            }
            .configured_fingerprint(),
            "hf|repo=1:r|revision=4:main"
        );
        assert_eq!(
            EmbeddingModel::Endpoint {
                url: "u".into(),
                model: "m".into(),
                api_key_env: None,
                query_prefix: None,
                doc_prefix: None,
            }
            .configured_fingerprint(),
            "endpoint|url=1:u|model=1:m"
        );
    }

    #[test]
    fn huggingface_and_endpoint_fingerprints_are_unchanged() {
        assert_eq!(
            huggingface_fingerprint("org/m", "abc"),
            "hf|repo=5:org/m|revision=3:abc"
        );
        assert_eq!(
            endpoint_fingerprint("http://x/v1/embeddings", "nomic"),
            "endpoint|url=22:http://x/v1/embeddings|model=5:nomic"
        );
    }

    #[test]
    fn local_runtime_fingerprint_matches_pre_pr_path_spelling() {
        let model = EmbeddingModel::Local {
            path: PathBuf::from("/models/foo"),
            query_prefix: None,
            doc_prefix: None,
            pooling: None,
        };
        assert_eq!(model.configured_fingerprint(), "local|path=11:/models/foo");
        assert_eq!(
            local_fingerprint("/models/foo"),
            "local|path=11:/models/foo"
        );

        let with_pool = EmbeddingModel::Local {
            path: PathBuf::from("/models/foo"),
            query_prefix: Some("q: ".into()),
            doc_prefix: Some("d: ".into()),
            pooling: Some(Pooling::Cls),
        };
        assert_eq!(
            with_pool.configured_fingerprint(),
            "local|path=11:/models/foo|pool=3:cls|q=3:q: |d=3:d: ".to_string()
        );
    }

    #[test]
    fn local_configured_fingerprint_is_infallible_without_model_files() {
        let model = EmbeddingModel::Local {
            path: PathBuf::from("/nonexistent/local-model"),
            query_prefix: None,
            doc_prefix: None,
            pooling: None,
        };
        assert_eq!(
            model.configured_fingerprint(),
            "local|path=24:/nonexistent/local-model"
        );
        assert_eq!(model.embedder_cache_key(), model.configured_fingerprint());
    }

    fn write_local_model(dir: &Path, config: &[u8], tokenizer: &[u8], weights: &[u8]) {
        std::fs::write(dir.join("config.json"), config).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
        std::fs::write(dir.join("model.safetensors"), weights).unwrap();
    }

    fn touch_mtime(path: &Path) {
        use std::fs::FileTimes;
        use std::time::Duration;

        let file = std::fs::File::options().write(true).open(path).unwrap();
        let current = file.metadata().unwrap().modified().unwrap();
        let times = FileTimes::new().set_modified(current + Duration::from_secs(2));
        file.set_times(times).unwrap();
    }

    fn identity_from_dir(dir: &Path) -> (LocalContentIdentity, [PathBuf; 3]) {
        let files = resolve_local_model_files(dir).unwrap();
        let stamps = stamp_local_hash_paths(&files, "m").unwrap();
        (
            LocalContentIdentity::new(stamps),
            [
                files.config.clone(),
                files.tokenizer.clone(),
                files.weights.clone(),
            ],
        )
    }

    fn path_refs(paths: &[PathBuf; 3]) -> [&Path; 3] {
        [&paths[0], &paths[1], &paths[2]]
    }

    fn write_pooling(dir: &Path, cls: bool, mean: bool) {
        let pooling_dir = dir.join("1_Pooling");
        std::fs::create_dir_all(&pooling_dir).unwrap();
        std::fs::write(
            pooling_dir.join("config.json"),
            format!(r#"{{"pooling_mode_cls_token":{cls},"pooling_mode_mean_tokens":{mean}}}"#),
        )
        .unwrap();
    }

    #[test]
    fn ordinary_local_cache_key_does_not_content_hash() {
        test_reset_content_hash_calls();
        let model = EmbeddingModel::Local {
            path: PathBuf::from("/models/foo"),
            query_prefix: None,
            doc_prefix: None,
            pooling: None,
        };
        for _ in 0..5 {
            let _ = model.configured_fingerprint();
            let _ = model.embedder_cache_key();
        }
        assert_eq!(
            test_content_hash_calls(),
            0,
            "runtime Local identity must not stream model bytes"
        );
    }

    #[test]
    fn local_artifact_identity_does_content_hash() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        test_reset_content_hash_calls();
        let _ = local_artifact_fingerprint(dir.path(), None, "", "").unwrap();
        assert!(
            test_content_hash_calls() >= 1,
            "artifact identity must digest model inputs"
        );
    }

    #[test]
    fn local_artifact_identity_ignores_mount_path() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        write_local_model(a.path(), b"cfg", b"tok", b"w");
        write_local_model(b.path(), b"cfg", b"tok", b"w");

        let id_a = local_artifact_fingerprint(a.path(), None, "", "").unwrap();
        let id_b = local_artifact_fingerprint(b.path(), None, "", "").unwrap();
        assert_eq!(id_a, id_b, "artifact identity must ignore the mount path");
        assert!(
            id_a.starts_with("local|content="),
            "artifact identity is content-keyed; got {id_a}"
        );
        // Runtime identity remains path-keyed and therefore differs across mounts.
        let model_a = EmbeddingModel::Local {
            path: a.path().to_path_buf(),
            query_prefix: None,
            doc_prefix: None,
            pooling: None,
        };
        let model_b = EmbeddingModel::Local {
            path: b.path().to_path_buf(),
            query_prefix: None,
            doc_prefix: None,
            pooling: None,
        };
        assert_ne!(
            model_a.configured_fingerprint(),
            model_b.configured_fingerprint()
        );
    }

    #[test]
    fn local_artifact_identity_changes_with_weights_config_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"weights-v1");
        let base = local_artifact_fingerprint(dir.path(), None, "", "").unwrap();

        std::fs::write(dir.path().join("model.safetensors"), b"weights-v2-longer").unwrap();
        assert_ne!(
            base,
            local_artifact_fingerprint(dir.path(), None, "", "").unwrap()
        );

        write_local_model(dir.path(), b"cfg-changed", b"tok", b"weights-v2-longer");
        let after_cfg = local_artifact_fingerprint(dir.path(), None, "", "").unwrap();
        assert_ne!(base, after_cfg);

        write_local_model(
            dir.path(),
            b"cfg-changed",
            b"tok-changed",
            b"weights-v2-longer",
        );
        assert_ne!(
            after_cfg,
            local_artifact_fingerprint(dir.path(), None, "", "").unwrap()
        );
    }

    #[test]
    fn local_artifact_identity_uses_resolved_pooling_not_irrelevant_file() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        write_pooling(dir.path(), false, true); // mean in file

        let with_override =
            local_artifact_fingerprint(dir.path(), Some(Pooling::Cls), "", "").unwrap();
        assert!(
            with_override.contains("|pool=3:cls"),
            "override must win; got {with_override}"
        );

        // Mutating an unused pooling file must not change artifact identity.
        write_pooling(dir.path(), true, false);
        let after_file_change =
            local_artifact_fingerprint(dir.path(), Some(Pooling::Cls), "", "").unwrap();
        assert_eq!(with_override, after_file_change);

        // Without override, resolved pooling from the file must affect identity.
        let cls = local_artifact_fingerprint(dir.path(), None, "", "").unwrap();
        write_pooling(dir.path(), false, true);
        let mean = local_artifact_fingerprint(dir.path(), None, "", "").unwrap();
        assert_ne!(cls, mean);
        assert!(cls.contains("|pool=3:cls"));
        assert!(mean.contains("|pool=4:mean"));
    }

    #[test]
    fn local_artifact_identity_includes_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        let plain = local_artifact_fingerprint(dir.path(), None, "", "").unwrap();
        let with_q = local_artifact_fingerprint(dir.path(), None, "query: ", "").unwrap();
        let with_qd = local_artifact_fingerprint(dir.path(), None, "query: ", "passage: ").unwrap();
        assert_ne!(plain, with_q);
        assert_ne!(with_q, with_qd);
        assert!(with_qd.contains("|q=7:query: "));
        assert!(with_qd.contains("|d=9:passage: "));
    }

    #[test]
    fn local_artifact_identity_same_length_weight_change_still_mismatches() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"weights-AAAA");
        let files = resolve_local_model_files(dir.path()).unwrap();
        let paths: Vec<PathBuf> = files
            .content_hash_paths()
            .iter()
            .map(|p| (*p).to_path_buf())
            .collect();

        // Hash bytes directly — not via the (len,mtime) memo — so same-length
        // overwrites cannot hide behind coarse filesystem timestamps.
        let before = hash_local_identity_files(&paths, "m").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"weights-BBBB").unwrap();
        let after = hash_local_identity_files(&paths, "m").unwrap();

        assert_ne!(
            before, after,
            "same-length weight bytes must still change the content digest"
        );
        assert_ne!(
            local_content_fingerprint(&before),
            local_content_fingerprint(&after)
        );
    }

    #[test]
    fn endpoint_client_cache_key_includes_env_name_but_vector_identity_does_not() {
        let endpoint = |api_key_env: &str| EmbeddingModel::Endpoint {
            url: "https://example.test/v1/embeddings".into(),
            model: "embed-v1".into(),
            api_key_env: Some(api_key_env.into()),
            query_prefix: None,
            doc_prefix: None,
        };
        let a = endpoint("KEY_A");
        let b = endpoint("KEY_B");

        assert_eq!(a.configured_fingerprint(), b.configured_fingerprint());
        assert_ne!(a.embedder_cache_key(), b.embedder_cache_key());
        assert!(a.embedder_cache_key().contains("KEY_A"));
    }

    #[test]
    fn fingerprint_fields_cannot_collide_through_delimiters() {
        let endpoint = |url: &str, model: &str, query_prefix: &str, doc_prefix: Option<&str>| {
            EmbeddingModel::Endpoint {
                url: url.into(),
                model: model.into(),
                api_key_env: None,
                query_prefix: Some(query_prefix.into()),
                doc_prefix: doc_prefix.map(str::to_string),
            }
        };

        assert_ne!(
            endpoint("https://example.test#a", "b", "", None).configured_fingerprint(),
            endpoint("https://example.test", "a#b", "", None).configured_fingerprint()
        );
        assert_ne!(
            endpoint("u", "m", "x|d=y", None).configured_fingerprint(),
            endpoint("u", "m", "x", Some("y")).configured_fingerprint()
        );
    }

    #[test]
    fn endpoint_cache_key_distinguishes_no_key_from_literal_sentinel_name() {
        let endpoint = |api_key_env| EmbeddingModel::Endpoint {
            url: "u".into(),
            model: "m".into(),
            api_key_env,
            query_prefix: None,
            doc_prefix: None,
        };

        assert_ne!(
            endpoint(None).embedder_cache_key(),
            endpoint(Some("<none>".into())).embedder_cache_key()
        );
    }

    #[test]
    fn pooling_override_parses_and_is_rejected_on_endpoint() {
        // Valid on huggingface.
        assert!(matches!(
            EmbeddingModel::resolve(EmbeddingSpec {
                huggingface: Some("org/m".into()),
                pooling: Some("mean".into()),
                ..Default::default()
            })
            .unwrap(),
            EmbeddingModel::HuggingFace {
                pooling: Some(Pooling::Mean),
                ..
            }
        ));
        // A bad value is a Config error.
        assert!(
            EmbeddingModel::resolve(EmbeddingSpec {
                huggingface: Some("org/m".into()),
                pooling: Some("median".into()),
                ..Default::default()
            })
            .is_err()
        );
        // Meaningless on an endpoint (the server pools).
        let err = EmbeddingModel::resolve(EmbeddingSpec {
            ollama: Some("nomic".into()),
            pooling: Some("mean".into()),
            ..Default::default()
        })
        .unwrap_err();
        assert!(err.to_string().contains("pooling"), "got: {err}");
    }

    #[test]
    fn doc_prefix_threads_through_and_affects_fingerprint() {
        let m = EmbeddingModel::resolve(EmbeddingSpec {
            huggingface: Some("intfloat/e5-small-v2".into()),
            query_prefix: Some("query: ".into()),
            doc_prefix: Some("passage: ".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(m.doc_prefix(), "passage: ");
        assert!(m.configured_fingerprint().contains("|d=9:passage: "));
        // Pooling is part of identity: same repo, different pooling → different key.
        let cls = EmbeddingModel::resolve(EmbeddingSpec {
            huggingface: Some("org/m".into()),
            pooling: Some("cls".into()),
            ..Default::default()
        })
        .unwrap();
        let mean = EmbeddingModel::resolve(EmbeddingSpec {
            huggingface: Some("org/m".into()),
            pooling: Some("mean".into()),
            ..Default::default()
        })
        .unwrap();
        assert_ne!(cls.configured_fingerprint(), mean.configured_fingerprint());
    }

    #[test]
    fn missing_local_tokenizer_error_message_is_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"w").unwrap();
        let err = resolve_local_model_files(dir.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("missing tokenizer.json")
                && msg.contains("fast tokenizer.json is required"),
            "got: {msg}"
        );
    }

    #[test]
    fn resolve_and_stamp_local_model_files_does_not_content_hash() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        test_reset_content_hash_calls();
        let files = resolve_local_model_files(dir.path()).unwrap();
        let _ = stamp_local_hash_paths(&files, "m").unwrap();
        assert_eq!(
            test_content_hash_calls(),
            0,
            "resolve + stamp may touch metadata but must not digest weights"
        );
    }

    #[test]
    fn metadata_only_touch_keeps_local_artifact_identity_stable() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        let (identity, paths) = identity_from_dir(dir.path());
        let refs = path_refs(&paths);

        let established = identity.content_id(dir.path(), &refs).unwrap();
        touch_mtime(&dir.path().join("config.json"));
        let after = identity.content_id(dir.path(), &refs).unwrap();
        assert_eq!(
            established, after,
            "byte-identical mtime touch must keep the established identity"
        );
    }

    #[test]
    fn metadata_only_touch_is_recoverable_and_stable_across_repeats() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        let (identity, paths) = identity_from_dir(dir.path());
        let refs = path_refs(&paths);

        test_reset_content_hash_calls();
        let established = identity.content_id(dir.path(), &refs).unwrap();
        assert_eq!(
            test_content_hash_calls(),
            1,
            "first establishment hashes once"
        );

        for _ in 0..2 {
            touch_mtime(&dir.path().join("config.json"));
            let before = test_content_hash_calls();
            let after_touch = identity.content_id(dir.path(), &refs).unwrap();
            assert_eq!(after_touch, established);
            assert_eq!(
                test_content_hash_calls(),
                before + 1,
                "each metadata-only touch recomputes the digest once"
            );
            let repeat = identity.content_id(dir.path(), &refs).unwrap();
            assert_eq!(repeat, established);
            assert_eq!(
                test_content_hash_calls(),
                before + 1,
                "unchanged stamps after an accepted touch must take the fast path"
            );
        }
    }

    #[test]
    fn content_drift_after_established_identity_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"weights-v1");
        let (identity, paths) = identity_from_dir(dir.path());
        let refs = path_refs(&paths);

        identity.content_id(dir.path(), &refs).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"weights-v2-longer").unwrap();
        let err = identity.content_id(dir.path(), &refs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("contents differ") && msg.contains("start a new process"),
            "content drift after establishment must name contents and a new process; got {msg}"
        );
        assert!(
            !msg.contains("reload the embedder"),
            "error must not suggest an impossible reload; got {msg}"
        );
    }

    #[test]
    fn stamp_drift_before_established_identity_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        let (identity, paths) = identity_from_dir(dir.path());
        let refs = path_refs(&paths);

        touch_mtime(&dir.path().join("config.json"));
        let err = identity.content_id(dir.path(), &refs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("before this process established its artifact identity")
                && msg.contains("start a new process"),
            "stamp drift before establishment must fail closed with a new-process remedy; got {msg}"
        );
        assert!(
            !msg.contains("reload the embedder"),
            "error must not suggest an impossible reload; got {msg}"
        );
    }

    #[test]
    fn same_length_content_swap_after_established_identity_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"weights-v1");
        let (identity, paths) = identity_from_dir(dir.path());
        let refs = path_refs(&paths);

        identity.content_id(dir.path(), &refs).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), b"weights-v2").unwrap();
        let err = identity.content_id(dir.path(), &refs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("contents differ"),
            "same-length weight swap must fail closed; got {msg}"
        );
    }

    #[test]
    fn content_id_does_not_commit_digest_when_stamps_change_during_hash() {
        let dir = tempfile::tempdir().unwrap();
        write_local_model(dir.path(), b"cfg", b"tok", b"w");
        let (identity, paths) = identity_from_dir(dir.path());
        let refs = path_refs(&paths);
        let config = dir.path().join("config.json");

        super::after_hash_hook::set(move || touch_mtime(&config));
        test_reset_content_hash_calls();
        let err = identity.content_id(dir.path(), &refs).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("before this process established its artifact identity")
                && msg.contains("start a new process"),
            "TOCTOU during first hash must not establish identity; got {msg}"
        );
        assert!(
            test_content_hash_calls() >= 1,
            "the interrupted establishment must have hashed"
        );

        let err = identity.content_id(dir.path(), &refs).unwrap_err();
        assert!(
            err.to_string()
                .contains("before this process established its artifact identity"),
            "identity must remain unestablished after a discarded in-flight digest; got {err}"
        );
    }
}
