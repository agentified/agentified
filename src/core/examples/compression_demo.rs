//! Prompt compression end to end, and the latency harness behind ADR-0016's
//! numbers.
//!
//! ```text
//! cargo run -p ratel-ai-core --release --example compression_demo
//! ```
//!
//! Downloads the default model on first run (~700 MB, one-time).

use std::time::Instant;

use ratel_ai_core::{CompressionOptions, PromptCompressor};

const TRANSCRIPT: &str = include_str!("../tests/fixtures/transcript.txt");

/// The documented usage contract: compress the **context**, leave the
/// instruction and the question verbatim.
const INSTRUCTION: &str = "You are given notes from an engineering meeting. Answer the question \
                           using only these notes. If the notes do not contain the answer, say so.";
const QUESTION: &str = "What is the plan for the Postgres migration, and what is the main risk?";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressor = PromptCompressor::new();

    let cold = Instant::now();
    compressor.preload()?;
    println!("model load: {} ms\n", cold.elapsed().as_millis());

    // --- the headline configuration ------------------------------------
    let out = compressor.compress(TRANSCRIPT, None)?;
    println!("=== compressed context (rate 0.40) ===\n{}\n", out.text);
    println!(
        "context: {} -> {} model tokens ({:.2}x), {} -> {} units, {} protected, {} chunk(s)",
        out.stats.model_tokens_in,
        out.stats.model_tokens_out,
        out.stats.model_tokens_in as f32 / out.stats.model_tokens_out.max(1) as f32,
        out.stats.words_in,
        out.stats.words_out,
        out.stats.protected_units,
        out.stats.chunks,
    );

    let mut worst: Vec<_> = out.dropped.iter().collect();
    worst.sort_by(|a, b| b.importance.total_cmp(&a.importance));
    println!(
        "highest-scoring dropped: {}",
        worst
            .iter()
            .take(8)
            .map(|w| format!("{}({:.2})", w.text, w.importance))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // --- whole-prompt economics ----------------------------------------
    let scaffolding = INSTRUCTION.len() + QUESTION.len();
    println!(
        "\nwhole prompt: instruction + question ({scaffolding} bytes) stay verbatim; only the \
         context is compressed"
    );

    // --- latency, median of N warm calls --------------------------------
    const REPS: usize = 7;
    let mut timings: Vec<u128> = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        compressor.compress(TRANSCRIPT, None)?;
        timings.push(t.elapsed().as_millis());
    }
    timings.sort_unstable();
    println!(
        "\nlatency over {} model tokens, n={REPS}: median {} ms (min {}, max {})",
        out.stats.model_tokens_in,
        timings[REPS / 2],
        timings[0],
        timings[REPS - 1],
    );

    // --- the rate ladder -------------------------------------------------
    println!("\nbar    tokens   ratio   units   protected");
    for bar in [0.9, 0.7, 0.5, 0.3, 0.1] {
        let options = CompressionOptions {
            min_importance: bar,
            ..Default::default()
        };
        let r = compressor.compress(TRANSCRIPT, Some(&options))?;
        println!(
            "{bar:<6.2} {:<8} {:<7.2} {:<7} {}",
            r.stats.model_tokens_out,
            r.stats.model_tokens_in as f32 / r.stats.model_tokens_out.max(1) as f32,
            r.stats.words_out,
            r.stats.protected_units,
        );
    }

    // --- the gate --------------------------------------------------------
    let short = compressor.compress("translate this to french", None)?;
    println!(
        "\nshort prompt: returned verbatim ({:?}) — {:?}",
        short.stats.gate, short.text
    );

    Ok(())
}
