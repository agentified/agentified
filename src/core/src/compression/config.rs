//! Which model backs prompt compression.
//!
//! A direct transposition of [`crate::embedding_config`] (ADR-0012) down to three
//! variants: the pinned built-in default, any HuggingFace repo, or an on-disk
//! directory. The source is named explicitly — a bare string is a **local
//! directory path**, everything else is a keyed object — and resolution lives
//! here in the core so both SDKs share one implementation.
//!
//! **No endpoint variant.** An OpenAI-compatible `/embeddings` route returns
//! vectors; nothing standard returns per-token classification logits, so an
//! endpoint source would be a shape with no protocol behind it. See ADR-0016.

use std::path::{Path, PathBuf};

use super::error::{CompressorError, cfg};
use crate::embedding_config::{fingerprint, looks_like_path, looks_like_url};

/// The built-in default: LLMLingua-2's BERT-base multilingual token classifier,
/// pinned to a commit so compression is reproducible. `compression/model.rs`
/// loads against these.
pub(crate) const DEFAULT_REPO: &str =
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank";
pub(crate) const DEFAULT_REVISION: &str = "5f0c82792b7ea14c6484e015b6a072009496b7f2";

/// The model backing [`super::PromptCompressor`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CompressionModel {
    /// The pinned built-in LLMLingua-2 checkpoint. The zero-config default, and
    /// the one model Ratel auto-downloads.
    #[default]
    Default,
    /// A BERT token-classification HuggingFace repo, loaded in-process via
    /// Candle. `revision` defaults to `main`; `download` (default `false`) must
    /// be opted into for Ratel to fetch it — otherwise it must already be in the
    /// local cache.
    HuggingFace {
        /// HuggingFace repo id.
        repo: String,
        /// Git revision; `main` when absent.
        revision: Option<String>,
        /// Opt in to fetching this model if it isn't cached.
        download: bool,
    },
    /// An on-disk model directory — the air-gapped / bring-your-own-checkpoint
    /// path. Never touches the network.
    Local {
        /// Directory holding `config.json`, `tokenizer.json`, and weights.
        path: PathBuf,
    },
}

/// The cross-SDK compression-model DTO, resolved by
/// [`CompressionModel::resolve`]. Exactly one primary source may be set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompressionSpec {
    /// Raw string shortcut — a **local model directory path** only. A repo-id or
    /// URL string is rejected in favor of the explicit `huggingface` key.
    pub spec: Option<String>,
    /// Primary source: a HuggingFace repo id.
    pub huggingface: Option<String>,
    /// Primary source: a local model directory path.
    pub local: Option<String>,
    /// Git revision for a HuggingFace source.
    pub revision: Option<String>,
    /// Opt in to downloading a HuggingFace source that isn't cached.
    pub download: Option<bool>,
}

impl CompressionModel {
    /// Validate a concrete Rust model value. SDK configs normally enter through
    /// [`Self::resolve`], but Rust callers can construct the enum directly; this
    /// keeps that path subject to the same nonblank-field rules.
    ///
    /// # Errors
    ///
    /// [`CompressorError::Config`] when a required repo or path is blank.
    pub fn validate(&self) -> Result<(), CompressorError> {
        match self {
            CompressionModel::Default => Ok(()),
            CompressionModel::HuggingFace { repo, .. } => validate_nonblank("huggingface", repo),
            CompressionModel::Local { path } => validate_nonblank("local", &path.to_string_lossy()),
        }
    }

    /// Resolve the cross-SDK [`CompressionSpec`] into a validated model.
    ///
    /// # Errors
    ///
    /// [`CompressorError::Config`] when no source, more than one source, a blank
    /// field, or a modifier on the wrong source is given.
    pub fn resolve(spec: CompressionSpec) -> Result<CompressionModel, CompressorError> {
        for (name, value) in [
            ("spec", spec.spec.as_deref()),
            ("huggingface", spec.huggingface.as_deref()),
            ("local", spec.local.as_deref()),
            ("revision", spec.revision.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(cfg(format!("compression '{name}' must not be blank")));
            }
        }

        let primaries = [
            ("spec", spec.spec.is_some()),
            ("huggingface", spec.huggingface.is_some()),
            ("local", spec.local.is_some()),
        ];
        let set: Vec<&str> = primaries
            .iter()
            .filter(|(_, present)| *present)
            .map(|(key, _)| *key)
            .collect();
        match set.len() {
            0 => {
                return Err(cfg(
                    "no compression source given; pass a local directory path, or one of \
                     huggingface/local",
                ));
            }
            1 => {}
            _ => {
                return Err(cfg(format!(
                    "conflicting compression keys {set:?}; give exactly one of \
                     spec/huggingface/local",
                )));
            }
        }

        // `download` and `revision` are HuggingFace-only.
        if spec.download.is_some() && set[0] != "huggingface" {
            return Err(cfg("'download' is only valid for a HuggingFace repo"));
        }
        if spec.revision.is_some() && set[0] != "huggingface" {
            return Err(cfg("'revision' is only valid for a HuggingFace repo"));
        }

        let model = match set[0] {
            "spec" => infer_from_string(spec.spec.as_deref().unwrap())?,
            "huggingface" => CompressionModel::HuggingFace {
                repo: spec.huggingface.unwrap(),
                revision: spec.revision,
                download: spec.download.unwrap_or(false),
            },
            "local" => CompressionModel::Local {
                path: PathBuf::from(spec.local.unwrap()),
            },
            _ => unreachable!("primary key set is closed"),
        };
        model.validate()?;
        Ok(model)
    }

    /// Human-readable name for errors and trace events.
    pub(crate) fn display_name(&self) -> String {
        match self {
            CompressionModel::Default => DEFAULT_REPO.to_string(),
            CompressionModel::HuggingFace { repo, revision, .. } => match revision {
                Some(rev) => format!("{repo}@{rev}"),
                None => repo.clone(),
            },
            CompressionModel::Local { path } => path.display().to_string(),
        }
    }

    /// Identity as configured, before load. Keys the process-wide model cache, so
    /// two compressors on the same model share one resident copy. Uses the
    /// length-delimited field encoding from `embedding_config`, so no combination
    /// of values can collide by concatenation.
    pub(crate) fn configured_fingerprint(&self) -> String {
        match self {
            CompressionModel::Default => fingerprint(
                "hf",
                &[("repo", DEFAULT_REPO), ("revision", DEFAULT_REVISION)],
            ),
            CompressionModel::HuggingFace { repo, revision, .. } => fingerprint(
                "hf",
                &[
                    ("repo", repo.as_str()),
                    ("revision", revision.as_deref().unwrap_or("main")),
                ],
            ),
            CompressionModel::Local { path } => {
                fingerprint("local", &[("path", &path.display().to_string())])
            }
        }
    }
}

fn validate_nonblank(name: &str, value: &str) -> Result<(), CompressorError> {
    if value.trim().is_empty() {
        return Err(cfg(format!("compression '{name}' must not be blank")));
    }
    Ok(())
}

/// Interpret the raw string shortcut, which is a **local model directory path
/// only**. A URL or a repo-id-looking string is rejected with a pointer to the
/// right object form, so the source is never guessed from an ambiguous string.
fn infer_from_string(s: &str) -> Result<CompressionModel, CompressorError> {
    if looks_like_url(s) {
        return Err(cfg(format!(
            "'{s}' looks like a URL; prompt compression runs in-process only — pass a \
             local directory path or {{\"huggingface\": \"…\"}}"
        )));
    }
    if looks_like_path(s) || Path::new(s).is_dir() {
        return Ok(CompressionModel::Local {
            path: PathBuf::from(s),
        });
    }
    Err(cfg(format!(
        "'{s}' is not a local directory path; to use a HuggingFace repo pass \
         {{\"huggingface\": \"{s}\"}}, or give an absolute/relative directory \
         path for a local model"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_str(s: &str) -> Result<CompressionModel, CompressorError> {
        CompressionModel::resolve(CompressionSpec {
            spec: Some(s.to_string()),
            ..Default::default()
        })
    }

    #[test]
    fn a_bare_string_is_a_local_directory_path_only() {
        assert_eq!(
            from_str("/models/llmlingua").unwrap(),
            CompressionModel::Local {
                path: PathBuf::from("/models/llmlingua")
            }
        );
        assert_eq!(
            from_str("./llmlingua").unwrap(),
            CompressionModel::Local {
                path: PathBuf::from("./llmlingua")
            }
        );
    }

    #[test]
    fn a_repo_id_string_points_at_the_explicit_key_instead_of_being_guessed() {
        let e = from_str("microsoft/llmlingua-2").unwrap_err().to_string();
        assert!(
            e.contains("{\"huggingface\": \"microsoft/llmlingua-2\"}"),
            "got: {e}"
        );
    }

    #[test]
    fn a_url_string_is_rejected_because_compression_is_in_process_only() {
        let e = from_str("http://localhost:11434/v1/embeddings")
            .unwrap_err()
            .to_string();
        assert!(e.contains("in-process only"), "got: {e}");
    }

    #[test]
    fn exactly_one_source_is_required() {
        let none = CompressionModel::resolve(CompressionSpec::default())
            .unwrap_err()
            .to_string();
        assert!(none.contains("no compression source given"), "got: {none}");

        let both = CompressionModel::resolve(CompressionSpec {
            huggingface: Some("a/b".into()),
            local: Some("/m".into()),
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
        assert!(both.contains("conflicting compression keys"), "got: {both}");
    }

    #[test]
    fn blank_fields_are_rejected() {
        for spec in [
            CompressionSpec {
                huggingface: Some("   ".into()),
                ..Default::default()
            },
            CompressionSpec {
                local: Some("".into()),
                ..Default::default()
            },
            CompressionSpec {
                huggingface: Some("a/b".into()),
                revision: Some(" ".into()),
                ..Default::default()
            },
        ] {
            assert!(
                CompressionModel::resolve(spec)
                    .unwrap_err()
                    .to_string()
                    .contains("must not be blank")
            );
        }
    }

    #[test]
    fn download_and_revision_are_huggingface_only() {
        let dl = CompressionModel::resolve(CompressionSpec {
            local: Some("/m".into()),
            download: Some(true),
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
        assert!(
            dl.contains("'download' is only valid for a HuggingFace repo"),
            "got: {dl}"
        );

        let rev = CompressionModel::resolve(CompressionSpec {
            local: Some("/m".into()),
            revision: Some("abc".into()),
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
        assert!(
            rev.contains("'revision' is only valid for a HuggingFace repo"),
            "got: {rev}"
        );
    }

    #[test]
    fn huggingface_defaults_to_no_download_and_main() {
        assert_eq!(
            CompressionModel::resolve(CompressionSpec {
                huggingface: Some("a/b".into()),
                ..Default::default()
            })
            .unwrap(),
            CompressionModel::HuggingFace {
                repo: "a/b".into(),
                revision: None,
                download: false,
            }
        );
    }

    #[test]
    fn fingerprints_distinguish_every_source_and_cannot_collide() {
        let default = CompressionModel::Default.configured_fingerprint();
        let same_repo_main = CompressionModel::HuggingFace {
            repo: DEFAULT_REPO.into(),
            revision: None,
            download: false,
        }
        .configured_fingerprint();
        let same_repo_pinned = CompressionModel::HuggingFace {
            repo: DEFAULT_REPO.into(),
            revision: Some(DEFAULT_REVISION.into()),
            download: false,
        }
        .configured_fingerprint();
        let local = CompressionModel::Local {
            path: PathBuf::from("/models/x"),
        }
        .configured_fingerprint();

        // `main` is a moving label; the pinned default is not the same identity.
        assert_ne!(default, same_repo_main);
        // Pinning to the default's own revision *is* the same identity.
        assert_eq!(default, same_repo_pinned);
        assert_ne!(default, local);

        // Length-delimited fields: a value containing the separator can't forge
        // another field's boundary.
        let sneaky = CompressionModel::HuggingFace {
            repo: "a|revision=3:xyz".into(),
            revision: None,
            download: false,
        }
        .configured_fingerprint();
        assert_ne!(sneaky, same_repo_main);
    }

    #[test]
    fn download_is_not_part_of_identity() {
        // It's a fetch policy, not a property of the weights — two catalogs that
        // differ only there must share one resident model.
        let a = CompressionModel::HuggingFace {
            repo: "a/b".into(),
            revision: None,
            download: true,
        };
        let b = CompressionModel::HuggingFace {
            repo: "a/b".into(),
            revision: None,
            download: false,
        };
        assert_eq!(a.configured_fingerprint(), b.configured_fingerprint());
    }

    #[test]
    fn direct_enum_construction_is_validated_too() {
        assert!(
            CompressionModel::HuggingFace {
                repo: " ".into(),
                revision: None,
                download: false,
            }
            .validate()
            .is_err()
        );
        assert!(CompressionModel::Default.validate().is_ok());
    }
}
