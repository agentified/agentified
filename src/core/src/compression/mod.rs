//! Prompt compression: LLMLingua-2 token importance plus a policy layer that
//! makes the loss bounded, visible, and steerable. See ADR-0016.
//!
//! Every other capability in this crate **selects** — it chooses which existing
//! entries enter the context. This one **transforms**: it returns a string that
//! did not exist before. That difference is why the policy below exists at all.
//!
//! # What it does
//!
//! An LLMLingua-2 checkpoint is a BERT token classifier over two labels,
//! `EXCLUDE` and `INCLUDE`. `P(INCLUDE)` is a per-token importance score, and a
//! threshold on it is a compressor. Thresholding alone loses too much, so:
//!
//! - Decisions are made on **atoms** — runs of tokens with no bytes between them
//!   — so `doesn't`, `8,400`, and `pg_upgrade` are never half-kept.
//! - **Protected** spans — caller patterns and negations — survive at any rate,
//!   and their cost is charged to the budget first.
//! - The budget is spent in **model tokens**, highest importance first.
//! - Output is rebuilt from the **original bytes**, never detokenized.
//! - Prompts below a minimum length are returned **verbatim**, with
//!   [`CompressionStats::gate`] saying so — short text has no redundancy to
//!   spend, and raising the rate does not recover it.
//!
//! # What it does not do
//!
//! Compression is lossy. Nothing here promises a compressed prompt yields the
//! same model output; the gate, the protections, and the per-atom
//! [`CompressedPrompt::dropped`] scores make the loss visible and steerable, they
//! do not make it safe. Evaluate it on your own data.
//!
//! Compress **context**, not instructions: the established usage is to compress
//! retrieved passages, transcripts, and documents while leaving the system
//! prompt and the question verbatim.
//!
//! # Cost
//!
//! It is not cheap. The default checkpoint is ~700 MB and a 623-token document
//! takes **~2–2.6 s** on an 8-core laptop CPU — several times an ONNX runtime on
//! the same weights, which is the price of staying pure-Rust (ADR-0011). Call
//! [`PromptCompressor::preload`] at startup so no request pays the cold load,
//! and pin `RAYON_NUM_THREADS` in a request-per-thread server so concurrent
//! calls don't oversubscribe the CPU.
//!
//! # Example
//!
//! ```no_run
//! use ratel_ai_core::{CompressionOptions, PromptCompressor, ProtectPattern};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let long_context = "";
//! let compressor = PromptCompressor::new();
//! compressor.preload()?; // pay the ~700 MB load on purpose, not in a request
//!
//! let options = CompressionOptions {
//!     rate: 0.4,
//!     protect: vec![ProtectPattern::Regex(r"\$[\d,]+".into())],
//!     ..Default::default()
//! };
//! let out = compressor.compress(long_context, Some(&options))?;
//!
//! println!("{} -> {} tokens", out.stats.model_tokens_in, out.stats.model_tokens_out);
//! for unit in out.dropped.iter().take(5) {
//!     println!("dropped {:?} ({:.2})", unit.text, unit.importance);
//! }
//! # Ok(())
//! # }
//! ```

mod config;
mod error;
mod model;
mod policy;
mod render;
mod tokens;

use std::sync::Arc;
use std::time::Instant;

pub use config::{CompressionModel, CompressionSpec};
pub use error::CompressorError;
pub use policy::ProtectPattern;

use crate::trace::{NoopSink, TraceEvent, TraceSink};

/// Default keep-ratio. The prototype's best configuration: at `0.4` a 566-token
/// transcript came back at 234 tokens with all six of its hand-checked critical
/// facts intact. Unswept — see ADR-0016.
const DEFAULT_RATE: f32 = 0.40;

/// Below this many whitespace-separated words, compression is skipped **before
/// the model is touched**, so a short prompt never pays a 709 MB load.
const DEFAULT_MIN_WORDS: usize = 40;

/// Below this many model tokens, the input is returned verbatim. The prototype
/// established that short prompts are out of domain, not under-tuned:
/// `"translate this to french"` degrades to `"translate to"` at every rate up to
/// 0.9.
const DEFAULT_MIN_TOKENS: usize = 50;

/// Chunk ceiling (~8k model tokens). Each chunk is a full encoder pass, so an
/// unbounded input is refused rather than silently expensive.
const DEFAULT_MAX_CHUNKS: usize = 16;

/// How to compress one prompt. A plain `Default` struct, so callers write
/// `CompressionOptions { rate: 0.3, ..Default::default() }`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionOptions {
    /// Approximate keep-ratio in the compression model's own tokens, `0.0` to
    /// `1.0`. Approximate because protection is a hard promise and can overrun
    /// it; [`CompressionStats`] reports the exact counts.
    pub rate: f32,
    /// Word count below which the input is returned verbatim, checked before any
    /// model load.
    pub min_words: usize,
    /// Model-token count below which the input is returned verbatim.
    pub min_tokens: usize,
    /// Maximum chunks; beyond it, [`CompressorError::InputTooLong`].
    pub max_chunks: usize,
    /// Spans that must survive at any rate.
    pub protect: Vec<ProtectPattern>,
    /// Protect every atom containing a digit. Default **`false`**.
    ///
    /// Off by default because it is indiscriminate and measurably harmful: on
    /// the reference transcript it protects 23 units — `Q3`, `12 terabytes`,
    /// `November 2024`, `four hours` — and the budget they consume starves prose
    /// that matters more, costing *more* critical facts than it saves at every
    /// rate below 0.4. The prototype's number-splitting failure (`8,400` kept as
    /// `8` + `400`) is fixed by atoms, not by this. Turn it on for input where
    /// every figure is load-bearing, or better, name the ones that are with
    /// [`Self::protect`]. See ADR-0016.
    pub protect_numbers: bool,
    /// Protect negations, whose loss inverts a claim rather than shortening it.
    /// Default `true` — it costs ~3 units on a page of prose and is what keeps
    /// `doesn't support` from becoming `support` under an aggressive rate.
    pub protect_negations: bool,
    /// Replace the built-in (English) negation list. Compared against atom text
    /// with punctuation and case removed, so `doesnt` matches `doesn't`.
    pub negation_terms: Option<Vec<String>>,
    /// Keep blank lines as blank lines. Default `true`.
    pub preserve_paragraphs: bool,
    /// Populate [`CompressedPrompt::kept`] / [`CompressedPrompt::dropped`].
    /// Default `true`; the only allocation proportional to input size beyond the
    /// output itself.
    pub explain: bool,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            rate: DEFAULT_RATE,
            min_words: DEFAULT_MIN_WORDS,
            min_tokens: DEFAULT_MIN_TOKENS,
            max_chunks: DEFAULT_MAX_CHUNKS,
            protect: Vec::new(),
            protect_numbers: false,
            protect_negations: true,
            negation_terms: None,
            preserve_paragraphs: true,
            explain: true,
        }
    }
}

/// Why an input was returned verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionGate {
    /// Fewer words than `min_words` — checked before the model is loaded.
    TooShortWords,
    /// Fewer model tokens than `min_tokens`.
    TooShortTokens,
    /// `rate >= 1.0`, so there is nothing to remove.
    RateOne,
}

/// One scored unit of the input — an atom, which is a word in the intuitive
/// sense (`doesn't` and `8,400` are each one).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct WordScore {
    /// The unit's text, exactly as it appears in the input.
    pub text: String,
    /// Byte offset of the unit in the input.
    pub start: usize,
    /// Byte offset one past the unit in the input.
    pub end: usize,
    /// `P(INCLUDE)` averaged over the unit's model tokens; `1.0` when protected.
    pub importance: f32,
    /// Model tokens the unit costs.
    pub tokens: u32,
    /// Whether the unit was protected (and so kept regardless of importance).
    pub protected: bool,
}

/// What compression cost and produced.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CompressionStats {
    /// Model tokens in the input.
    pub model_tokens_in: u32,
    /// Model tokens in the output.
    pub model_tokens_out: u32,
    /// Scored units in the input.
    pub words_in: u32,
    /// Scored units kept.
    pub words_out: u32,
    /// Encoder passes performed.
    pub chunks: u32,
    /// Units protected from removal.
    pub protected_units: u32,
    /// The keep-ratio that was requested.
    pub rate: f32,
    /// `Some` when the input was returned verbatim; nothing was compressed.
    pub gate: Option<CompressionGate>,
    /// Protection alone exceeded the budget, so `rate` was overrun.
    pub budget_exceeded: bool,
    /// Wall-clock milliseconds, excluding a cold model load.
    pub took_ms: u64,
}

/// A compressed prompt plus the evidence for every decision made.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CompressedPrompt {
    /// The compressed text, built from slices of the input.
    pub text: String,
    /// Units that survived. Empty when `explain` is off.
    pub kept: Vec<WordScore>,
    /// Units that were removed, with the scores that removed them. Empty when
    /// `explain` is off.
    pub dropped: Vec<WordScore>,
    /// Counts and gating for this call.
    pub stats: CompressionStats,
}

/// Compresses prompts with an LLMLingua-2 token classifier.
///
/// Stateless per call: nothing is cached between calls, so a compressor is
/// cheap to hold and safe to share. The model itself is process-wide and keyed
/// by identity, so two compressors on the same model share one resident copy.
///
/// The model is loaded **lazily**, on the first call that gets past the gate —
/// constructing a `PromptCompressor` touches no disk and no network.
pub struct PromptCompressor {
    model: CompressionModel,
    defaults: CompressionOptions,
    sink: Arc<dyn TraceSink>,
}

impl Default for PromptCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptCompressor {
    /// The built-in default model, with tracing off.
    pub fn new() -> Self {
        Self {
            model: CompressionModel::Default,
            defaults: CompressionOptions::default(),
            sink: Arc::new(NoopSink),
        }
    }

    /// An explicit model, validated eagerly so a bad configuration surfaces at
    /// construction rather than on the first compression.
    ///
    /// # Errors
    ///
    /// [`CompressorError::Config`] when the model is invalid.
    pub fn with_model(model: CompressionModel) -> Result<Self, CompressorError> {
        model.validate()?;
        Ok(Self {
            model,
            defaults: CompressionOptions::default(),
            sink: Arc::new(NoopSink),
        })
    }

    /// Attach a trace sink, which receives [`TraceEvent::Compress`] per call and
    /// the one-time model-load events.
    pub fn set_trace_sink(&mut self, sink: Arc<dyn TraceSink>) {
        self.sink = sink;
    }

    /// Set the options used when [`Self::compress`] is called with `None`.
    pub fn set_defaults(&mut self, options: CompressionOptions) {
        self.defaults = options;
    }

    /// Load (and, for the default model, download) the weights now, so the first
    /// real call is not charged the cold load. Idempotent and safe to race.
    ///
    /// Worth calling at startup in a server: the default model is ~700 MB, and a
    /// request that pays for that inline stalls for seconds.
    ///
    /// # Errors
    ///
    /// Any [`CompressorError`] the load can raise.
    pub fn preload(&self) -> Result<(), CompressorError> {
        model::classifier_with_telemetry(&self.model, self.sink.as_ref())?;
        Ok(())
    }

    /// Per-unit importance with **no policy applied** — the raw signal, for
    /// callers who want to inspect a compression or write their own selection
    /// rule.
    ///
    /// # Errors
    ///
    /// Any [`CompressorError`] the model load or the forward pass can raise.
    pub fn score(&self, text: &str) -> Result<Vec<WordScore>, CompressorError> {
        let options = CompressionOptions {
            max_chunks: self.defaults.max_chunks,
            ..Default::default()
        };
        let classifier = model::classifier_with_telemetry(&self.model, self.sink.as_ref())?;
        let (atoms, _) = self.scored_atoms(&classifier, text, &options)?;
        Ok(word_scores(text, &atoms, |_| true))
    }

    /// Compress `text`, spending roughly `rate` of its tokens.
    ///
    /// Returns the input **verbatim** with [`CompressionStats::gate`] set when it
    /// is too short to compress — and does so before loading a model, so a short
    /// prompt costs nothing.
    ///
    /// # Errors
    ///
    /// [`CompressorError::Config`] for an invalid protect pattern,
    /// [`CompressorError::InputTooLong`] beyond `max_chunks`, and any load or
    /// inference failure.
    pub fn compress(
        &self,
        text: &str,
        options: Option<&CompressionOptions>,
    ) -> Result<CompressedPrompt, CompressorError> {
        let options = options.unwrap_or(&self.defaults);
        let started = Instant::now();

        // Pre-gate, before any model work. `split_whitespace` is a cheap proxy
        // that only has to be right about "obviously too short".
        if options.rate >= 1.0 {
            return Ok(self.gated(text, options, CompressionGate::RateOne, started));
        }
        if text.split_whitespace().count() < options.min_words {
            return Ok(self.gated(text, options, CompressionGate::TooShortWords, started));
        }

        // Compile protect patterns before loading ~700 MB of weights, so a typo
        // in a regex is a fast error rather than a slow one.
        let protectors = policy::compile_protectors(&options.protect)?;

        let classifier = model::classifier_with_telemetry(&self.model, self.sink.as_ref())?;
        let (mut atoms, chunks) = self.scored_atoms(&classifier, text, options)?;

        let tokens_in: u32 = atoms.iter().map(tokens::Atom::token_count).sum();
        if (tokens_in as usize) < options.min_tokens {
            return Ok(self.gated(text, options, CompressionGate::TooShortTokens, started));
        }

        let protected_units = policy::apply_protection(
            text,
            &mut atoms,
            &policy::Protection {
                patterns: &protectors,
                numbers: options.protect_numbers,
                negations: options.protect_negations,
                negation_terms: options.negation_terms.as_deref(),
            },
        );
        let selection = policy::select(&atoms, options.rate);
        let out = render::render(text, &atoms, &selection.keep, options.preserve_paragraphs);

        let stats = CompressionStats {
            model_tokens_in: tokens_in,
            model_tokens_out: selection.tokens_out,
            words_in: atoms.len() as u32,
            words_out: selection.keep.iter().filter(|k| **k).count() as u32,
            chunks: chunks as u32,
            protected_units,
            rate: options.rate,
            gate: None,
            budget_exceeded: selection.budget_exceeded,
            took_ms: started.elapsed().as_millis() as u64,
        };
        self.record(&stats);

        let (kept, dropped) = if options.explain {
            (
                word_scores(text, &atoms, |i| selection.keep[i]),
                word_scores(text, &atoms, |i| !selection.keep[i]),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(CompressedPrompt {
            text: out,
            kept,
            dropped,
            stats,
        })
    }

    /// Encode once, chunk, run the forward passes, and fill in importances.
    fn scored_atoms(
        &self,
        classifier: &model::BertTokenClassifier,
        text: &str,
        options: &CompressionOptions,
    ) -> Result<(Vec<tokens::Atom>, usize), CompressorError> {
        let spans = classifier.encode(text)?;
        let mut atoms = tokens::atoms_from_tokens(text, &spans);
        let plan = tokens::plan_chunks(spans.len(), &atoms, classifier.max_body_tokens());
        if plan.len() > options.max_chunks {
            return Err(CompressorError::InputTooLong {
                tokens: spans.len(),
                chunks: plan.len(),
                max_chunks: options.max_chunks,
            });
        }

        let rows: Vec<Vec<u32>> = plan
            .iter()
            .map(|r| spans[r.clone()].iter().map(|t| t.id).collect())
            .collect();
        let probs: Vec<f32> = classifier.keep_probs(&rows)?.concat();
        debug_assert_eq!(probs.len(), spans.len(), "one probability per token");
        tokens::apply_probs(&mut atoms, &probs);
        Ok((atoms, plan.len()))
    }

    /// Return the input untouched, with the reason recorded.
    fn gated(
        &self,
        text: &str,
        options: &CompressionOptions,
        gate: CompressionGate,
        started: Instant,
    ) -> CompressedPrompt {
        let stats = CompressionStats {
            model_tokens_in: 0,
            model_tokens_out: 0,
            words_in: 0,
            words_out: 0,
            chunks: 0,
            protected_units: 0,
            rate: options.rate,
            gate: Some(gate),
            budget_exceeded: false,
            took_ms: started.elapsed().as_millis() as u64,
        };
        self.record(&stats);
        CompressedPrompt {
            text: text.to_string(),
            kept: Vec::new(),
            dropped: Vec::new(),
            stats,
        }
    }

    fn record(&self, stats: &CompressionStats) {
        self.sink.record(TraceEvent::Compress {
            model_tokens_in: stats.model_tokens_in,
            model_tokens_out: stats.model_tokens_out,
            words_in: stats.words_in,
            words_out: stats.words_out,
            rate: stats.rate,
            chunks: stats.chunks,
            protected_units: stats.protected_units,
            gated: stats.gate.is_some(),
            took_ms: stats.took_ms,
        });
    }
}

/// Project the atoms matching `include` into the public scored form.
fn word_scores(
    text: &str,
    atoms: &[tokens::Atom],
    include: impl Fn(usize) -> bool,
) -> Vec<WordScore> {
    atoms
        .iter()
        .enumerate()
        .filter(|(i, _)| include(*i))
        .map(|(_, a)| WordScore {
            text: text[a.start..a.end].to_string(),
            start: a.start,
            end: a.end,
            importance: a.importance,
            tokens: a.token_count(),
            protected: a.protected,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_documented_configuration() {
        // Pinned so a silent retune shows up as a failing test rather than as a
        // quiet change in everyone's output.
        let d = CompressionOptions::default();
        assert_eq!(d.rate, 0.40);
        assert_eq!(d.min_words, 40);
        assert_eq!(d.min_tokens, 50);
        assert_eq!(d.max_chunks, 16);
        assert!(d.protect.is_empty());
        assert!(
            !d.protect_numbers,
            "blanket digit protection costs more critical facts than it saves (ADR-0016)"
        );
        assert!(d.protect_negations);
        assert!(d.negation_terms.is_none());
        assert!(d.preserve_paragraphs);
        assert!(d.explain);
    }
}
