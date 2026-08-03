//! Model-backed regression suite for prompt compression.
//!
//! `#[ignore]`d because it needs the ~700 MB default checkpoint on disk. CI runs
//! the pure layer and `compression.rs`; this is what a human runs when the policy
//! changes:
//!
//! ```text
//! huggingface-cli download microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
//!   --revision 5f0c82792b7ea14c6484e015b6a072009496b7f2
//! cargo test -p ratel-ai-core --test compression_golden --release -- --ignored --nocapture
//! ```
//!
//! The fixture and its six critical facts are ported verbatim from the
//! prototype (`prompt-compression/src/shared.ts`), so the headline result there —
//! 6/6 facts retained at `rate=0.4` with a ~2.4x token reduction — is a test
//! rather than a claim in a document.

use std::sync::Arc;

use ratel_ai_core::{
    CompressionOptions, CompressorError, MemorySink, PromptCompressor, ProtectPattern, TraceEvent,
};

const TRANSCRIPT: &str = include_str!("fixtures/transcript.txt");

/// The facts a correct answer to "What is the plan for the Postgres migration,
/// and what is the main risk?" depends on. Each is a substring test over the
/// compressed output.
const CRITICAL: [(&str, &str); 6] = [
    ("logical replication (the plan)", "logical replication"),
    ("negation on the TimescaleDB risk", "doesn't support"),
    ("Timescale 2.8 -> 2.14", "2.14"),
    ("cutover 90 seconds", "90 seconds"),
    ("tested in staging", "staging"),
    ("cost 8,400", "8,400"),
];

fn compressor() -> PromptCompressor {
    PromptCompressor::new()
}

fn surviving(text: &str) -> Vec<&'static str> {
    CRITICAL
        .iter()
        .filter(|(_, needle)| text.contains(needle))
        .map(|(name, _)| *name)
        .collect()
}

fn missing(text: &str) -> Vec<&'static str> {
    CRITICAL
        .iter()
        .filter(|(_, needle)| !text.contains(needle))
        .map(|(name, _)| *name)
        .collect()
}

#[test]
#[ignore = "needs the compression model cached"]
fn critical_facts_survive_at_the_default_rate() {
    // The prototype's headline metric, as a regression test.
    let out = compressor().compress(TRANSCRIPT, None).unwrap();
    println!(
        "rate 0.40: {} -> {} model tokens ({:.2}x), {} words -> {}, {} protected, {} chunks, {} ms",
        out.stats.model_tokens_in,
        out.stats.model_tokens_out,
        out.stats.model_tokens_in as f32 / out.stats.model_tokens_out.max(1) as f32,
        out.stats.words_in,
        out.stats.words_out,
        out.stats.protected_units,
        out.stats.chunks,
        out.stats.took_ms,
    );
    println!("---\n{}\n---", out.text);

    assert!(
        out.stats.gate.is_none(),
        "the transcript must be compressed"
    );
    assert_eq!(
        surviving(&out.text).len(),
        CRITICAL.len(),
        "lost: {:?}",
        missing(&out.text)
    );
}

#[test]
#[ignore = "needs the compression model cached"]
fn the_transcript_compresses_at_least_two_fold() {
    // The prototype measured 566 -> 234 o200k tokens (2.42x). We assert on our
    // own (WordPiece) counts with margin, since mBERT fragments differently.
    let out = compressor().compress(TRANSCRIPT, None).unwrap();
    let ratio = out.stats.model_tokens_in as f32 / out.stats.model_tokens_out.max(1) as f32;
    assert!(ratio >= 2.0, "only {ratio:.2}x: {:?}", out.stats);
    assert!(
        out.text.len() < TRANSCRIPT.len(),
        "output must be shorter than the input"
    );
}

#[test]
#[ignore = "needs the compression model cached"]
fn negation_protection_keeps_a_negation_the_bar_would_drop() {
    // Raise the bar until ordinary prose falls away, and check the negation and
    // what it governs still survive. The failure this guards against is not a
    // shortened claim but an inverted one: `without any downtime` compressing to
    // `without cutover` says the opposite of the input.
    let aggressive = |negations: bool| CompressionOptions {
        min_importance: 0.95,
        protect_negations: negations,
        ..Default::default()
    };
    let with = compressor()
        .compress(TRANSCRIPT, Some(&aggressive(true)))
        .unwrap();
    println!(
        "bar 0.95 — with negations lost {:?}; without, lost {:?}",
        missing(&with.text),
        missing(
            &compressor()
                .compress(TRANSCRIPT, Some(&aggressive(false)))
                .unwrap()
                .text
        )
    );
    assert!(
        with.text.contains("doesn't support"),
        "negation protection must keep the negation and its verb: {:?}",
        with.text
    );
}

#[test]
#[ignore = "needs the compression model cached"]
fn protection_can_only_add_now_that_there_is_no_budget() {
    // Under a budget, protection competed: marking 23 digit-bearing units stole
    // room from prose and cost more critical facts than it saved, which is why
    // `protect_numbers` defaults off. Selection is now `protected || score >=
    // bar`, so protection is purely additive — it can add tokens, never remove
    // content. If this ever fails, the budget has crept back in somewhere.
    let off = CompressionOptions {
        min_importance: 0.8,
        ..Default::default()
    };
    let on = CompressionOptions {
        protect_numbers: true,
        ..off.clone()
    };
    let a = compressor().compress(TRANSCRIPT, Some(&off)).unwrap();
    let b = compressor().compress(TRANSCRIPT, Some(&on)).unwrap();

    assert!(
        b.stats.model_tokens_out >= a.stats.model_tokens_out,
        "protection may grow the output, never shrink it"
    );
    for unit in &a.kept {
        assert!(
            b.kept.iter().any(|k| k.start == unit.start),
            "{:?} survived without protection but not with it",
            unit.text
        );
    }
    println!(
        "bar 0.8: numbers off {} tokens, on {} tokens ({} protected)",
        a.stats.model_tokens_out, b.stats.model_tokens_out, b.stats.protected_units
    );
}

#[test]
#[ignore = "needs the compression model cached"]
fn output_is_built_from_the_input_and_never_mangled() {
    // The prototype's detokenizer produced `pg _ upgrade`, `doesn ' t`, `4 , 200`.
    // Rendering from byte offsets makes those impossible; assert it directly.
    let out = compressor().compress(TRANSCRIPT, None).unwrap();
    for artifact in ["pg _ upgrade", "doesn ' t", "8 , 400", "2. 14", "db. r6g"] {
        assert!(
            !out.text.contains(artifact),
            "detokenizer artifact {artifact:?} in output"
        );
    }
    // Every kept unit appears in the output, in order, byte-identical.
    let mut cursor = 0usize;
    for unit in &out.kept {
        let found = out.text[cursor..]
            .find(&unit.text)
            .unwrap_or_else(|| panic!("{:?} missing from output", unit.text));
        cursor += found + unit.text.len();
        assert_eq!(
            &TRANSCRIPT[unit.start..unit.end],
            unit.text,
            "offsets must be exact"
        );
    }
}

#[test]
#[ignore = "needs the compression model cached"]
fn the_explain_channel_accounts_for_every_unit() {
    let out = compressor().compress(TRANSCRIPT, None).unwrap();
    assert_eq!(
        (out.kept.len() + out.dropped.len()) as u32,
        out.stats.words_in,
        "kept + dropped must cover the input exactly"
    );
    assert_eq!(out.kept.len() as u32, out.stats.words_out);
    assert!(out.kept.iter().all(|w| (0.0..=1.0).contains(&w.importance)));
    assert!(
        out.kept.iter().filter(|w| w.protected).count() as u32 == out.stats.protected_units,
        "every protected unit must have been kept"
    );

    let mut worst_dropped: Vec<_> = out.dropped.iter().collect();
    worst_dropped.sort_by(|a, b| b.importance.total_cmp(&a.importance));
    println!(
        "highest-scoring dropped units: {:?}",
        worst_dropped
            .iter()
            .take(8)
            .map(|w| format!("{}({:.2})", w.text, w.importance))
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "needs the compression model cached"]
fn explain_off_returns_no_scores_but_the_same_text() {
    let quiet = CompressionOptions {
        explain: false,
        ..Default::default()
    };
    let with = compressor().compress(TRANSCRIPT, None).unwrap();
    let without = compressor().compress(TRANSCRIPT, Some(&quiet)).unwrap();
    assert_eq!(with.text, without.text);
    assert!(without.kept.is_empty() && without.dropped.is_empty());
}

#[test]
#[ignore = "needs the compression model cached"]
fn a_user_pattern_survives_an_aggressive_rate() {
    let options = CompressionOptions {
        min_importance: 0.9,
        protect: vec![ProtectPattern::Regex(r"\d[\d,\.]*".into())],
        protect_numbers: false,
        protect_negations: false,
        ..Default::default()
    };
    let out = compressor().compress(TRANSCRIPT, Some(&options)).unwrap();
    for number in ["8,400", "2.14", "90"] {
        assert!(out.text.contains(number), "{number} lost at rate 0.15");
    }
    println!(
        "bar 0.9: {} -> {} tokens",
        out.stats.model_tokens_in, out.stats.model_tokens_out
    );
}

#[test]
#[ignore = "needs the compression model cached"]
fn output_is_byte_stable_across_runs() {
    let a = compressor().compress(TRANSCRIPT, None).unwrap();
    let b = compressor().compress(TRANSCRIPT, None).unwrap();
    assert_eq!(a.text, b.text, "selection must be deterministic");
    assert_eq!(a.stats.model_tokens_out, b.stats.model_tokens_out);
}

#[test]
#[ignore = "needs the compression model cached"]
fn a_multi_chunk_document_has_no_seam_artifacts() {
    // Three transcripts back to back exceed one 510-token chunk, exercising the
    // chunk boundaries. Rendering from original bytes means a seam cannot inject
    // whitespace the input did not have.
    let long = [TRANSCRIPT, TRANSCRIPT, TRANSCRIPT].join("\n\n");
    let out = compressor().compress(&long, None).unwrap();
    assert!(out.stats.chunks > 1, "expected multiple chunks");
    assert_eq!(surviving(&out.text).len(), CRITICAL.len());
    let mut cursor = 0usize;
    for unit in &out.kept {
        let found = out.text[cursor..].find(&unit.text).expect("unit in output");
        cursor += found + unit.text.len();
    }
}

#[test]
#[ignore = "needs the compression model cached"]
fn beyond_max_chunks_is_refused_rather_than_silently_expensive() {
    let options = CompressionOptions {
        max_chunks: 1,
        ..Default::default()
    };
    let long = [TRANSCRIPT, TRANSCRIPT, TRANSCRIPT].join("\n\n");
    let err = compressor().compress(&long, Some(&options)).unwrap_err();
    assert!(
        matches!(err, CompressorError::InputTooLong { .. }),
        "got {err:?}"
    );
    assert!(err.to_string().contains("max_chunks"));
}

#[test]
#[ignore = "needs the compression model cached"]
fn score_returns_one_entry_per_unit_with_exact_offsets() {
    let scores = compressor().score(TRANSCRIPT).unwrap();
    assert!(!scores.is_empty());
    for s in &scores {
        assert_eq!(&TRANSCRIPT[s.start..s.end], s.text);
        assert!((0.0..=1.0).contains(&s.importance));
        assert!(!s.protected, "score() applies no policy");
        assert!(s.tokens >= 1);
    }
    // Ascending, non-overlapping.
    assert!(scores.windows(2).all(|w| w[0].end <= w[1].start));
}

#[test]
#[ignore = "needs the compression model cached"]
fn preload_pays_the_load_once_and_compress_reports_no_second_load() {
    let sink = Arc::new(MemorySink::new("golden"));
    let mut c = PromptCompressor::new();
    c.set_trace_sink(sink.clone());

    c.preload().unwrap();
    sink.drain(); // discard whatever the (possibly cold) load emitted

    c.compress(TRANSCRIPT, None).unwrap();
    let events: Vec<TraceEvent> = sink.drain().into_iter().map(|e| e.event).collect();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, TraceEvent::CompressorLoad { .. })),
        "a warm compress must not reload the model: {events:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, TraceEvent::Compress { .. }))
    );
}
