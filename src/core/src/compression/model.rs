//! The LLMLingua-2 token classifier, run in-process via Candle.
//!
//! `candle-transformers` ships `BertModel` — the encoder — and no
//! `*ForTokenClassification` head. The delta is a single `candle_nn::Linear` over
//! `classifier.weight` / `classifier.bias`, which is why prompt compression stays
//! inside the crate's pure-Rust, encoder-only envelope rather than reversing
//! ADR-0011 for an ONNX runtime.
//!
//! Two loading details are load-bearing and easy to get wrong:
//!
//! - **The root `VarBuilder`, not `vb.pp("bert")`.** `BertModel::load` tries
//!   `embeddings`/`encoder` and falls back to `{config.model_type}.embeddings` —
//!   which resolves the `bert.*` prefix a `*ForTokenClassification` checkpoint
//!   uses. Prefixing manually would break a bare encoder checkpoint that the
//!   fallback already handles.
//! - **Truncation is disabled.** Chunking is ours (see [`super::tokens`]); a
//!   `tokenizer.json` shipping a truncation block would silently drop the tail of
//!   every prompt.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

use super::config::{CompressionModel, DEFAULT_REPO, DEFAULT_REVISION};
use super::error::CompressorError;
use super::tokens::TokenSpan;
use crate::model_files::{
    LoadSlot, fetch_cached_with, get_or_load_keyed, is_cache_unwritable, is_not_found,
};
use crate::trace::{EmbedderLoadStatus, TraceEvent, TraceSink};

/// The classifier's two labels are `EXCLUDE` (0) and `INCLUDE` (1); the INCLUDE
/// probability is a token's importance.
const KEEP_LABEL_INDEX: usize = 1;
const NUM_LABELS: usize = 2;

/// Chunks per padded forward pass. Smaller than `EMBED_BATCH_CHUNK` (32) because
/// a compression row is a full 512 positions where tool text is short — this
/// bounds peak activation memory while still doing the common 1–4-chunk document
/// in a single pass.
const BATCH_CHUNK: usize = 4;

/// Default cold-load latency (ms) above which a load is flagged `slow`. Override
/// with `RATEL_COMPRESS_SLOW_MS`.
const DEFAULT_SLOW_LOAD_MS: u64 = 5_000;

const SLOW_LOAD_REASON: &str = "compression model load was slow — this machine may be underpowered for in-process CPU \
     inference; expect slow compression";

/// Set when a cold HF fetch actually downloaded weights.
pub(crate) struct DownloadNotice {
    pub model: String,
    pub bytes: u64,
}

pub(crate) struct BertTokenClassifier {
    bert: BertModel,
    classifier: Linear,
    tokenizer: Tokenizer,
    device: Device,
    /// `max_position_embeddings - 2`, leaving room for `[CLS]` and `[SEP]`.
    max_body_tokens: usize,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
}

/// Resolved model files.
struct Loaded {
    config: std::path::PathBuf,
    tokenizer: std::path::PathBuf,
    weights: std::path::PathBuf,
}

impl BertTokenClassifier {
    /// Body tokens one chunk may hold.
    pub(crate) fn max_body_tokens(&self) -> usize {
        self.max_body_tokens
    }

    /// Encode the whole prompt once, with no special tokens and no truncation.
    ///
    /// The returned spans are the **only** token stream compression works from:
    /// chunks are index ranges into it, so offsets are global by construction and
    /// no substring is ever re-tokenized.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<TokenSpan>, CompressorError> {
        let encoding =
            self.tokenizer
                .encode(text, false)
                .map_err(|e| CompressorError::Inference {
                    source: e.to_string(),
                })?;
        let ids = encoding.get_ids();
        let words = encoding.get_word_ids();
        let offsets = encoding.get_offsets();
        let special = encoding.get_special_tokens_mask();

        let mut spans = Vec::with_capacity(ids.len());
        for i in 0..ids.len() {
            // Specials carry no text; a `None` word id means the same thing for a
            // tokenizer that doesn't set the mask.
            if special[i] == 1 {
                continue;
            }
            let Some(word) = words[i] else { continue };
            let (start, end) = offsets[i];
            if start >= end {
                continue; // an empty span can't be kept or dropped
            }
            spans.push(TokenSpan {
                id: ids[i],
                word,
                start,
                end,
            });
        }
        Ok(spans)
    }

    /// `P(INCLUDE)` for every token in `rows`, one padded forward pass per batch.
    ///
    /// Padding is masked out, so a row's probabilities do not depend on its
    /// batch-mates — batching only bounds activation memory.
    pub(crate) fn keep_probs(&self, rows: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, CompressorError> {
        self.keep_probs_inner(rows)
            .map_err(|e| CompressorError::Inference {
                source: e.to_string(),
            })
    }

    fn keep_probs_inner(&self, rows: &[Vec<u32>]) -> candle_core::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(rows.len());
        for batch in equal_length_batches(rows) {
            let n = batch.len();
            // `[CLS] body [SEP]`. Batches are equal-length, so this pads nothing.
            let max_len = batch.iter().map(|r| r.len() + 2).max().unwrap_or(0);
            let mut ids = vec![self.pad_id; n * max_len];
            let mut mask = vec![0u32; n * max_len];
            for (row, body) in batch.iter().enumerate() {
                let base = row * max_len;
                ids[base] = self.cls_id;
                ids[base + 1..base + 1 + body.len()].copy_from_slice(body);
                ids[base + 1 + body.len()] = self.sep_id;
                for m in &mut mask[base..base + body.len() + 2] {
                    *m = 1;
                }
            }
            let input_ids = Tensor::from_vec(ids, (n, max_len), &self.device)?;
            let attention_mask = Tensor::from_vec(mask, (n, max_len), &self.device)?;
            let token_type_ids = input_ids.zeros_like()?;

            let hidden = self
                .bert
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
            let logits = self.classifier.forward(&hidden)?; // (n, seq, 2)
            let keep = softmax_last_dim(&logits)?
                .narrow(2, KEEP_LABEL_INDEX, 1)?
                .squeeze(2)?; // (n, seq)

            for (row, body) in keep.to_vec2::<f32>()?.into_iter().zip(batch) {
                // Drop the [CLS] prefix and everything past the body.
                out.push(row[1..1 + body.len()].to_vec());
            }
        }
        Ok(out)
    }

    /// Load from a HuggingFace repo. `allow_download` gates network access: the
    /// built-in default sets it, an explicit repo must opt in, and otherwise a
    /// missing model is [`CompressorError::NotCached`] rather than a silent
    /// multi-hundred-megabyte fetch (ADR-0012's policy, unchanged).
    fn load_hf(
        repo_id: &str,
        revision: &str,
        allow_download: bool,
    ) -> Result<(Self, Option<DownloadNotice>), CompressorError> {
        let repo_spec =
            Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
        let cache_repo = hf_hub::Cache::from_env().repo(repo_spec.clone());

        let (loaded, download) = if allow_download {
            // Detect the cold cache *before* fetching, so we only announce a real
            // download — and warn before the blocking fetch, since ~700 MB with no
            // message reads as a hang.
            let was_cached = cache_repo.get("model.safetensors").is_some()
                || cache_repo.get("pytorch_model.bin").is_some();
            if !was_cached {
                eprintln!(
                    "ratel: downloading compression model {repo_id} (~700 MB, one-time; this will \
                     take a while)…"
                );
            }
            let api = ApiBuilder::from_env()
                .build()
                .map_err(|e| CompressorError::Download {
                    model: repo_id.to_string(),
                    source: e.to_string(),
                })?;
            let repo = api.repo(repo_spec);
            let fetch = |file: &str| {
                fetch_cached_with(&repo, file, |msg| classify_fetch_error(repo_id, msg))
            };
            let config = fetch("config.json")?;
            let tokenizer = fetch("tokenizer.json")?;
            let weights = match fetch("model.safetensors") {
                Ok(p) => p,
                Err(CompressorError::Download { source, .. }) if is_not_found(&source) => {
                    fetch("pytorch_model.bin")?
                }
                Err(e) => return Err(e),
            };
            let notice = (!was_cached).then(|| DownloadNotice {
                model: repo_id.to_string(),
                bytes: [&config, &tokenizer, &weights]
                    .iter()
                    .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                    .sum(),
            });
            (
                Loaded {
                    config,
                    tokenizer,
                    weights,
                },
                notice,
            )
        } else {
            let not_cached = || CompressorError::NotCached {
                model: repo_id.to_string(),
                revision: (revision != "main").then(|| revision.to_string()),
            };
            let config = cache_repo.get("config.json").ok_or_else(not_cached)?;
            let tokenizer = cache_repo.get("tokenizer.json").ok_or_else(not_cached)?;
            let weights = cache_repo
                .get("model.safetensors")
                .or_else(|| cache_repo.get("pytorch_model.bin"))
                .ok_or_else(not_cached)?;
            (
                Loaded {
                    config,
                    tokenizer,
                    weights,
                },
                None,
            )
        };

        Ok((Self::build(&loaded, repo_id)?, download))
    }

    /// Load from a directory — the air-gapped / bring-your-own-checkpoint path.
    fn load_path(dir: &std::path::Path) -> Result<Self, CompressorError> {
        let name = dir.display().to_string();
        let weights = [dir.join("model.safetensors"), dir.join("pytorch_model.bin")]
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| CompressorError::Load {
                model: name.clone(),
                source: format!("missing model.safetensors / pytorch_model.bin in {name}"),
            })?;
        let config = dir.join("config.json");
        let tokenizer = dir.join("tokenizer.json");
        for (p, f) in [(&config, "config.json"), (&tokenizer, "tokenizer.json")] {
            if !p.exists() {
                return Err(CompressorError::Load {
                    model: name.clone(),
                    source: format!(
                        "missing {f} in {name} — a fast tokenizer.json is required; run \
                         tokenizer.save_pretrained() upstream"
                    ),
                });
            }
        }
        Self::build(
            &Loaded {
                config,
                tokenizer,
                weights,
            },
            &name,
        )
    }

    /// Shared file→model build.
    fn build(loaded: &Loaded, model_name: &str) -> Result<Self, CompressorError> {
        let device = Device::Cpu;
        let load_err = |source: String| CompressorError::Load {
            model: model_name.to_string(),
            source,
        };

        let config_bytes = std::fs::read(&loaded.config).map_err(|e| load_err(e.to_string()))?;
        let config: Config =
            serde_json::from_slice(&config_bytes).map_err(|e| load_err(e.to_string()))?;

        // safetensors is mmap'd; a `.bin`/`.pth` checkpoint goes through pickle.
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

        // Check the head **before** anything else about the files, because
        // "this is the wrong kind of model" is the more fundamental complaint —
        // and pointing at an embedding model is the likely mistake. Deep candle
        // tensor-not-found errors and tokenizer parse errors both read as
        // something being broken, when the checkpoint is simply not a classifier.
        if !vb.contains_tensor("classifier.weight") {
            return Err(load_err(
                "no `classifier.weight` in the checkpoint — this is an encoder-only BERT; prompt \
                 compression needs a BertForTokenClassification model such as \
                 microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
                    .into(),
            ));
        }

        let bert = BertModel::load(vb.clone(), &config).map_err(|e| {
            load_err(format!(
                "{e} — if the checkpoint's tensors are prefixed (`bert.encoder.…`), its \
                 config.json must carry \"model_type\": \"bert\" for the prefix to resolve"
            ))
        })?;
        // The default checkpoint's config.json omits `num_labels`/`id2label`, so
        // the head width comes from the tensor itself: a shape other than
        // (2, hidden) fails here rather than scoring against the wrong class.
        let classifier = candle_nn::linear(config.hidden_size, NUM_LABELS, vb.pp("classifier"))
            .map_err(|e| {
                load_err(format!(
                    "{e} — expected a 2-label (EXCLUDE/INCLUDE) token-classification head"
                ))
            })?;

        let mut tokenizer =
            Tokenizer::from_file(&loaded.tokenizer).map_err(|e| load_err(e.to_string()))?;
        // Chunking is ours — a truncation block here would silently eat the tail
        // of every prompt.
        tokenizer
            .with_truncation(None)
            .map_err(|e| load_err(e.to_string()))?;
        tokenizer.with_padding(None);

        let id = |t: &str| {
            tokenizer
                .token_to_id(t)
                .ok_or_else(|| load_err(format!("tokenizer has no {t} token")))
        };
        Ok(Self {
            max_body_tokens: config.max_position_embeddings.saturating_sub(2).max(1),
            cls_id: id("[CLS]")?,
            sep_id: id("[SEP]")?,
            pad_id: tokenizer.token_to_id("[PAD]").unwrap_or(0),
            bert,
            classifier,
            tokenizer,
            device,
        })
    }
}

/// Group rows into batches of **equal length**, up to [`BATCH_CHUNK`].
///
/// Padding a batch to its longest row costs a full forward pass per padded
/// position: attention is quadratic in sequence length and the feed-forward is
/// linear, so batching a 512-token chunk with a 115-token tail computes 1024
/// positions to score 627. Chunk planning makes every chunk but the last exactly
/// `max_body` long, so grouping by equal length pads **nothing** in the common
/// case while still batching the uniform chunks of a long document.
///
/// Rows are never reordered, so results stay aligned with the caller's chunks.
fn equal_length_batches(rows: &[Vec<u32>]) -> Vec<&[Vec<u32>]> {
    let mut batches = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let len = rows[start].len();
        let mut end = start + 1;
        while end < rows.len() && rows[end].len() == len && end - start < BATCH_CHUNK {
            end += 1;
        }
        batches.push(&rows[start..end]);
        start = end;
    }
    batches
}

fn classify_fetch_error(model: &str, msg: &str) -> CompressorError {
    if is_cache_unwritable(msg) {
        CompressorError::CacheUnwritable {
            source: msg.to_string(),
        }
    } else {
        CompressorError::Download {
            model: model.to_string(),
            source: msg.to_string(),
        }
    }
}

/// A resolved classifier plus one-time cold-load telemetry: the load latency
/// (`Some` on the loading call) and a download notice (`Some` only when a cold
/// HF fetch actually downloaded).
type ResolvedClassifier = (
    Arc<BertTokenClassifier>,
    Option<u64>,
    Option<DownloadNotice>,
);

/// Process-wide classifier cache, keyed by configured model identity. Two
/// compressors on the same model share one resident copy — which matters more
/// here than for embeddings, at ~700 MB each. A failed load is **not** cached, so
/// a transient blip doesn't poison the model for the process.
fn classifier_for(model: &CompressionModel) -> Result<ResolvedClassifier, CompressorError> {
    static CELL: OnceLock<Mutex<HashMap<String, LoadSlot<BertTokenClassifier>>>> = OnceLock::new();
    let cache = CELL.get_or_init(|| Mutex::new(HashMap::new()));
    let mut notice = None;
    let (model, load_ms) = get_or_load_keyed(cache, &model.configured_fingerprint(), || {
        let (built, n) = build_classifier(model)?;
        notice = n;
        Ok(Arc::new(built))
    })?;
    Ok((model, load_ms, notice))
}

fn build_classifier(
    model: &CompressionModel,
) -> Result<(BertTokenClassifier, Option<DownloadNotice>), CompressorError> {
    match model {
        // The built-in default is the one model Ratel auto-downloads.
        CompressionModel::Default => {
            BertTokenClassifier::load_hf(DEFAULT_REPO, DEFAULT_REVISION, true)
        }
        CompressionModel::HuggingFace {
            repo,
            revision,
            download,
        } => BertTokenClassifier::load_hf(repo, revision.as_deref().unwrap_or("main"), *download),
        CompressionModel::Local { path } => BertTokenClassifier::load_path(path).map(|m| (m, None)),
    }
}

/// Resolve the process classifier and record one-time load telemetry on `sink`.
/// Mirrors `embedding::embedder_with_telemetry` so both model paths report the
/// same way.
pub(crate) fn classifier_with_telemetry(
    model: &CompressionModel,
    sink: &dyn TraceSink,
) -> Result<Arc<BertTokenClassifier>, CompressorError> {
    let display = model.display_name();
    let (result, load_ms, notice) = match classifier_for(model) {
        Ok((m, ms, notice)) => (Ok(m), ms, notice),
        Err(e) => (Err(e), None, None),
    };
    if let Some(DownloadNotice { model, bytes }) = notice {
        let mb = bytes as f64 / 1_048_576.0;
        eprintln!("ratel: downloaded compression model {model} ({mb:.0} MB, one-time)");
        sink.record(TraceEvent::CompressorDownload { model, bytes });
    }
    if let Some(event) = load_event(&display, load_ms, result.as_ref().err()) {
        if let TraceEvent::CompressorLoad {
            status,
            took_ms,
            reason,
            ..
        } = &event
            && !matches!(status, EmbedderLoadStatus::Ok)
        {
            eprintln!(
                "ratel: compression model load {status:?} ({took_ms}ms): {}",
                reason.as_deref().unwrap_or("")
            );
        }
        sink.record(event);
    }
    result
}

/// Decide the load-telemetry event for a cold-load outcome. `None` on warm reuse.
/// Pure, so the slow/failed thresholding is unit-tested without the network.
fn load_event(
    model: &str,
    load_ms: Option<u64>,
    error: Option<&CompressorError>,
) -> Option<TraceEvent> {
    match (load_ms, error) {
        (_, Some(err)) => Some(TraceEvent::CompressorLoad {
            model: model.to_string(),
            status: EmbedderLoadStatus::Failed,
            took_ms: load_ms.unwrap_or(0),
            reason: Some(err.to_string()),
        }),
        (Some(ms), None) => {
            let slow = ms > slow_load_ms();
            Some(TraceEvent::CompressorLoad {
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
    std::env::var("RATEL_COMPRESS_SLOW_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SLOW_LOAD_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_group_equal_lengths_and_never_reorder_rows() {
        let rows: Vec<Vec<u32>> = [510, 510, 510, 115]
            .iter()
            .map(|n| vec![7u32; *n])
            .collect();
        let batches = equal_length_batches(&rows);
        // The uniform chunks batch together; the short tail runs alone rather
        // than dragging 395 padding positions through a forward pass with them.
        assert_eq!(batches.iter().map(|b| b.len()).collect::<Vec<_>>(), [3, 1]);
        assert_eq!(
            batches.iter().flat_map(|b| b.iter()).count(),
            rows.len(),
            "every row must appear exactly once, in order"
        );
    }

    #[test]
    fn batches_respect_the_size_cap() {
        let rows: Vec<Vec<u32>> = (0..10).map(|_| vec![7u32; 510]).collect();
        let batches = equal_length_batches(&rows);
        assert!(batches.iter().all(|b| b.len() <= BATCH_CHUNK));
        assert_eq!(batches.iter().map(|b| b.len()).sum::<usize>(), 10);
    }

    #[test]
    fn no_rows_means_no_batches() {
        assert!(equal_length_batches(&[]).is_empty());
    }

    #[test]
    fn warm_reuse_records_no_load_event() {
        assert!(load_event("m", None, None).is_none());
    }

    #[test]
    fn a_slow_cold_load_is_flagged_with_a_reason() {
        let event = load_event("m", Some(DEFAULT_SLOW_LOAD_MS + 1), None).unwrap();
        match event {
            TraceEvent::CompressorLoad { status, reason, .. } => {
                assert!(matches!(status, EmbedderLoadStatus::Slow));
                assert!(reason.unwrap().contains("underpowered"));
            }
            other => panic!("expected CompressorLoad, got {other:?}"),
        }
    }

    #[test]
    fn a_fast_cold_load_is_ok_with_no_reason() {
        match load_event("m", Some(1), None).unwrap() {
            TraceEvent::CompressorLoad { status, reason, .. } => {
                assert!(matches!(status, EmbedderLoadStatus::Ok));
                assert!(reason.is_none());
            }
            other => panic!("expected CompressorLoad, got {other:?}"),
        }
    }

    #[test]
    fn a_failed_load_reports_the_error_even_with_no_latency() {
        let err = CompressorError::NotCached {
            model: "org/repo".into(),
            revision: None,
        };
        match load_event("org/repo", None, Some(&err)).unwrap() {
            TraceEvent::CompressorLoad {
                status,
                took_ms,
                reason,
                ..
            } => {
                assert!(matches!(status, EmbedderLoadStatus::Failed));
                assert_eq!(took_ms, 0);
                assert!(
                    reason
                        .unwrap()
                        .contains("not in the local HuggingFace cache")
                );
            }
            other => panic!("expected CompressorLoad, got {other:?}"),
        }
    }

    #[test]
    fn fetch_errors_split_cache_problems_from_network_problems() {
        assert!(matches!(
            classify_fetch_error("m", "Permission denied (os error 13)"),
            CompressorError::CacheUnwritable { .. }
        ));
        assert!(matches!(
            classify_fetch_error("m", "error sending request: connection refused"),
            CompressorError::Download { .. }
        ));
    }
}
