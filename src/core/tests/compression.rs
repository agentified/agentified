//! Public-API tests for prompt compression that need **no model on disk**.
//!
//! The load-bearing ones are negative: that a short prompt is returned without
//! ever touching a model, that a misconfiguration fails before anything is
//! downloaded, and that linking the module changes nothing about tool search.
//! Model-backed behavior lives in `compression_golden.rs`, which is `#[ignore]`d.

use std::io::Write;
use std::sync::Arc;

use ratel_ai_core::{
    CompressionGate, CompressionModel, CompressionOptions, CompressionSpec, CompressorError,
    MemorySink, PromptCompressor, Tool, ToolRegistry, TraceEvent,
};

/// A model path that cannot exist, so any attempt to load fails loudly. Using it
/// is how a test proves a code path never reached the model.
fn unloadable() -> CompressionModel {
    CompressionModel::Local {
        path: "/definitely/not/here/ratel-compression".into(),
    }
}

fn compressor() -> PromptCompressor {
    PromptCompressor::with_model(unloadable()).expect("a path model is valid config")
}

#[test]
fn a_short_prompt_is_returned_verbatim_without_loading_a_model() {
    // The prototype established this is out of domain, not under-tuned:
    // "translate this to french" degrades to "translate to" at every rate up to
    // 0.9. Returning it untouched is the only correct answer.
    //
    // The model here is unloadable, so a passing test is the proof that the gate
    // runs *before* the load — an ordering guarantee that is otherwise untestable
    // without a 709 MB download in CI.
    let input = "translate this to french";
    let out = compressor().compress(input, None).unwrap();

    assert_eq!(out.text, input);
    assert_eq!(out.stats.gate, Some(CompressionGate::TooShortWords));
    assert_eq!(out.stats.model_tokens_in, 0, "no model ran");
    assert!(out.kept.is_empty() && out.dropped.is_empty());
}

#[test]
fn a_zero_bar_is_returned_verbatim_without_loading_a_model() {
    let input = "a b c ".repeat(200);
    let options = CompressionOptions {
        min_importance: 0.0,
        ..Default::default()
    };
    let out = compressor().compress(&input, Some(&options)).unwrap();
    assert_eq!(out.text, input);
    assert_eq!(out.stats.gate, Some(CompressionGate::KeepEverything));
}

#[test]
fn an_invalid_protect_pattern_fails_before_the_model_is_loaded() {
    // A typo in a regex must be a fast error, not one paid for after a
    // multi-hundred-megabyte load.
    let options = CompressionOptions {
        protect: vec![ratel_ai_core::ProtectPattern::Regex("a(".into())],
        ..Default::default()
    };
    let long = "word ".repeat(100);
    let err = compressor().compress(&long, Some(&options)).unwrap_err();
    assert!(matches!(err, CompressorError::Config { .. }), "got {err:?}");
    assert!(err.to_string().contains("invalid protect pattern"));
}

#[test]
fn an_uncached_huggingface_model_errors_not_cached_without_touching_the_network() {
    let model = CompressionModel::HuggingFace {
        repo: "ratel-test/definitely-not-a-real-repo".into(),
        revision: None,
        download: false, // cache-only: ADR-0012's policy, unchanged
    };
    let long = "word ".repeat(100);
    let err = PromptCompressor::with_model(model)
        .unwrap()
        .compress(&long, None)
        .unwrap_err();

    assert!(
        matches!(err, CompressorError::NotCached { .. }),
        "got {err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("huggingface-cli download"), "got: {msg}");
}

#[test]
fn a_missing_local_directory_errors_load_with_a_readable_message() {
    let long = "word ".repeat(100);
    let err = compressor().compress(&long, None).unwrap_err();
    assert!(matches!(err, CompressorError::Load { .. }), "got {err:?}");
    assert!(
        err.to_string().contains("model.safetensors"),
        "the message must name what is missing: {err}"
    );
}

#[test]
fn an_encoder_only_checkpoint_is_rejected_with_a_token_classification_hint() {
    // A plain BERT (or an embedding model) has no classifier head. That must say
    // what kind of checkpoint is required, not fail deep inside candle with
    // "cannot find tensor classifier.weight".
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"vocab_size":10,"hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,
            "intermediate_size":4,"hidden_act":"gelu","hidden_dropout_prob":0.1,
            "max_position_embeddings":512,"type_vocab_size":2,"initializer_range":0.02,
            "layer_norm_eps":1e-12,"pad_token_id":0,"model_type":"bert","classifier_dropout":null}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    write_safetensors_without_a_classifier(&dir.path().join("model.safetensors"));

    let err = PromptCompressor::with_model(CompressionModel::Local {
        path: dir.path().into(),
    })
    .unwrap()
    .compress(&"word ".repeat(100), None)
    .unwrap_err();

    let msg = err.to_string();
    assert!(matches!(err, CompressorError::Load { .. }), "got {err:?}");
    assert!(msg.contains("encoder-only"), "got: {msg}");
    assert!(msg.contains("BertForTokenClassification"), "got: {msg}");
}

/// A minimal, structurally valid safetensors file holding one unrelated tensor —
/// enough for the `classifier.weight` pre-check to run and reject it. Written by
/// hand (8-byte LE header length, JSON header, then data) so the test needs no
/// dependency beyond what the crate already has.
fn write_safetensors_without_a_classifier(path: &std::path::Path) {
    let header = br#"{"unrelated":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header).unwrap();
    f.write_all(&0f32.to_le_bytes()).unwrap();
}

#[test]
fn spec_resolution_names_the_source_rather_than_guessing_it() {
    let local = CompressionModel::resolve(CompressionSpec {
        spec: Some("/models/llmlingua".into()),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        local,
        CompressionModel::Local {
            path: "/models/llmlingua".into()
        }
    );

    // A repo-id-looking bare string points at the explicit key instead of being
    // silently treated as a HuggingFace repo.
    let err = CompressionModel::resolve(CompressionSpec {
        spec: Some("microsoft/llmlingua-2".into()),
        ..Default::default()
    })
    .unwrap_err();
    assert!(err.to_string().contains("\"huggingface\""), "got: {err}");
}

#[test]
fn a_gated_call_records_a_compress_event_and_no_model_load() {
    let sink = Arc::new(MemorySink::new("s1"));
    let mut compressor = compressor();
    compressor.set_trace_sink(sink.clone());
    compressor.compress("too short to bother", None).unwrap();

    let events: Vec<TraceEvent> = sink.drain().into_iter().map(|e| e.event).collect();
    assert_eq!(events.len(), 1, "got {events:?}");
    match &events[0] {
        TraceEvent::Compress {
            gated,
            model_tokens_in,
            chunks,
            ..
        } => {
            assert!(gated);
            assert_eq!(*model_tokens_in, 0);
            assert_eq!(*chunks, 0);
        }
        other => panic!("expected Compress, got {other:?}"),
    }
}

#[test]
fn compress_events_carry_counts_and_never_the_prompt_text() {
    // Traces land in a local JSONL file; a compression event holding the full
    // prompt would be the largest PII surface in the crate.
    let sink = Arc::new(MemorySink::new("s1"));
    let mut compressor = compressor();
    compressor.set_trace_sink(sink.clone());
    let secret = "hunter2-swordfish-correcthorse";
    compressor.compress(secret, None).unwrap();

    let json = serde_json::to_string(&sink.drain()).unwrap();
    assert!(
        !json.contains(secret),
        "prompt text leaked into the trace: {json}"
    );
}

#[test]
fn unused_compression_leaves_tool_search_byte_identical() {
    // The zero-impact guarantee: linking the module changes no existing ranking.
    // Framed as a negative test, in the style of `usage_ranking.rs`.
    let mut registry = ToolRegistry::new();
    for (id, description) in [
        ("gh_run_list", "list recent GitHub Actions workflow runs"),
        ("docker_build", "build a docker image from a Dockerfile"),
        ("pg_dump", "dump a Postgres database to a file"),
    ] {
        registry.register(Tool {
            id: id.into(),
            name: id.into(),
            description: description.into(),
            input_schema: serde_json::json!({}),
            output_schema: serde_json::json!({}),
        });
    }

    let ranked = |r: &ToolRegistry| -> Vec<(String, f32)> {
        r.search("why is the build broken", 10)
            .into_iter()
            .map(|h| (h.tool_id, h.score))
            .collect()
    };
    let before = ranked(&registry);

    // Construct and run a compressor against the same process.
    let out = compressor()
        .compress("translate this to french", None)
        .unwrap();
    assert_eq!(out.stats.gate, Some(CompressionGate::TooShortWords));

    assert_eq!(
        before,
        ranked(&registry),
        "compression must not perturb retrieval"
    );
    assert!(
        !before.is_empty(),
        "the fixture must actually match something"
    );
}

#[test]
fn options_defaults_are_the_documented_configuration() {
    // Pinned here as well as in the unit tests, because these values are the
    // public contract: a silent retune changes everyone's output.
    let d = CompressionOptions::default();
    assert_eq!(d.min_importance, 0.50);
    assert_eq!(d.min_words, 40);
    assert_eq!(d.min_tokens, 50);
    assert_eq!(d.max_chunks, 16);
    assert!(
        !d.protect_numbers,
        "blanket digit protection is measurably harmful; see ADR-0016"
    );
    assert!(d.protect_negations);
    assert!(d.preserve_paragraphs && d.explain);
}
