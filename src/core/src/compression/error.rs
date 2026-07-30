//! The typed failure surface for prompt compression.
//!
//! Mirrors [`crate::EmbedderError`]'s idiom — a flat enum, a one-line
//! remediation `hint()` folded into `Display` — but stays a **separate type**:
//! every `EmbedderError` message says "embedding model", and that enum is public
//! API we must not repurpose. Compression is also stateless per call, so it has
//! no analogue of `EmbeddingsNotBuilt`, `DimensionMismatch`, or `ModelMismatch`.

/// A recoverable prompt-compression failure. Returned instead of panicking so a
/// load or inference problem surfaces to the SDK as a **catchable** error (with a
/// remediation hint in `Display`) rather than aborting the host process.
#[derive(Debug, Clone)]
pub enum CompressorError {
    /// The compression model could not be fetched: offline, DNS/TLS, timeout, or
    /// the repo/revision does not exist.
    Download {
        /// The configured model display name.
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
    /// Model files are unusable: missing or corrupt weights, a config/tokenizer
    /// that failed to parse, or a checkpoint that is not a binary token
    /// classifier.
    Load {
        /// The configured model display name.
        model: String,
        /// The underlying load error.
        source: String,
    },
    /// Compressing a specific text failed (tokenization or the forward pass).
    Inference {
        /// The underlying tokenizer/inference error.
        source: String,
    },
    /// The compression configuration is invalid — a bad source combination, a
    /// blank required field, or an unparseable protect pattern.
    Config {
        /// What is wrong with the configuration.
        message: String,
    },
    /// An explicitly-configured HuggingFace model is not in the local cache, and
    /// Ratel auto-downloads only the built-in default.
    NotCached {
        /// The repo id that is not cached.
        model: String,
        /// The requested revision, if pinned.
        revision: Option<String>,
    },
    /// The input needs more chunks than `max_chunks` allows. Compression cost is
    /// linear in chunks and each is a full encoder pass, so an unbounded input is
    /// refused rather than silently truncated (or silently expensive).
    InputTooLong {
        /// Model tokens the input encoded to.
        tokens: usize,
        /// Chunks that would be required.
        chunks: usize,
        /// The configured ceiling.
        max_chunks: usize,
    },
}

impl CompressorError {
    /// One-line remediation hint, embedded in the `Display` message.
    fn hint(&self) -> &'static str {
        match self {
            CompressorError::Download { .. } => {
                "check connectivity and the configured model identifier"
            }
            CompressorError::CacheUnwritable { .. } => {
                "check ~/.cache/huggingface permissions and free disk space (or set HF_HOME)"
            }
            CompressorError::Load { .. } => {
                "prompt compression needs a BERT token-classification checkpoint (an LLMLingua-2 \
                 model); check the files are present, readable, and compatible"
            }
            CompressorError::Inference { .. } => "check the model input/configuration and retry",
            CompressorError::Config { .. } => {
                "give exactly one compression source (a model id / path) with its required fields"
            }
            CompressorError::NotCached { .. } => {
                "Ratel auto-downloads only the default model; pre-download this one, pass download=true, or use a local path"
            }
            CompressorError::InputTooLong { .. } => {
                "split the prompt and compress the parts, or raise max_chunks — cost is linear in chunks"
            }
        }
    }
}

impl std::fmt::Display for CompressorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hint = self.hint();
        match self {
            CompressorError::Download { model, source } => write!(
                f,
                "failed to download compression model {model}: {source} (hint: {hint})"
            ),
            CompressorError::CacheUnwritable { source } => write!(
                f,
                "compression model cache is not writable: {source} (hint: {hint})"
            ),
            CompressorError::Load { model, source } => write!(
                f,
                "failed to load compression model {model}: {source} (hint: {hint})"
            ),
            CompressorError::Inference { source } => {
                write!(f, "compression failed: {source} (hint: {hint})")
            }
            CompressorError::Config { message } => write!(f, "{message} (hint: {hint})"),
            CompressorError::NotCached { model, revision } => {
                let rev = revision
                    .as_deref()
                    .map(|r| format!(" --revision {r}"))
                    .unwrap_or_default();
                write!(
                    f,
                    "compression model {model} is not in the local HuggingFace cache — download it \
                     first: `huggingface-cli download {model}{rev}` (hint: {hint})"
                )
            }
            CompressorError::InputTooLong {
                tokens,
                chunks,
                max_chunks,
            } => write!(
                f,
                "prompt is too long to compress: {tokens} model tokens need {chunks} chunks, \
                 max_chunks is {max_chunks} (hint: {hint})"
            ),
        }
    }
}

impl std::error::Error for CompressorError {}

/// Shorthand for a configuration error.
pub(crate) fn cfg(message: impl Into<String>) -> CompressorError {
    CompressorError::Config {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_displays_its_source_and_a_hint() {
        let cases = [
            CompressorError::Download {
                model: "org/repo".into(),
                source: "connection refused".into(),
            },
            CompressorError::CacheUnwritable {
                source: "Permission denied (os error 13)".into(),
            },
            CompressorError::Load {
                model: "/models/x".into(),
                source: "missing config.json".into(),
            },
            CompressorError::Inference {
                source: "shape mismatch".into(),
            },
            cfg("no compression source given"),
            CompressorError::NotCached {
                model: "org/repo".into(),
                revision: None,
            },
            CompressorError::InputTooLong {
                tokens: 20_000,
                chunks: 40,
                max_chunks: 16,
            },
        ];
        for e in cases {
            let s = e.to_string();
            assert!(s.contains("hint:"), "no hint in: {s}");
            assert!(!s.contains("embedding"), "borrowed embedding wording: {s}");
        }
    }

    #[test]
    fn actionable_nouns_appear_in_the_messages() {
        // Each of these names the exact thing the reader has to go do.
        let not_cached = CompressorError::NotCached {
            model: "microsoft/llmlingua-2".into(),
            revision: Some("abc123".into()),
        }
        .to_string();
        assert!(
            not_cached.contains("huggingface-cli download microsoft/llmlingua-2 --revision abc123")
        );

        let too_long = CompressorError::InputTooLong {
            tokens: 9_000,
            chunks: 18,
            max_chunks: 16,
        }
        .to_string();
        assert!(too_long.contains("max_chunks"), "got: {too_long}");

        let load = CompressorError::Load {
            model: "bge-small".into(),
            source: "cannot find tensor classifier.weight".into(),
        }
        .to_string();
        assert!(
            load.contains("token-classification"),
            "the Load hint must name what kind of checkpoint is required: {load}"
        );
    }

    #[test]
    fn not_cached_omits_the_revision_flag_when_unpinned() {
        let s = CompressorError::NotCached {
            model: "org/repo".into(),
            revision: None,
        }
        .to_string();
        assert!(!s.contains("--revision"), "got: {s}");
    }
}
