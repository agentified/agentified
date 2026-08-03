//! Compression on ten coding-task queries: before, after, and latency.
//!
//! ```text
//! cargo run -p ratel-ai-core --release --example compression_queries
//! ```
//!
//! The min-length gate is disabled here so every query actually compresses —
//! by default half of these are short enough that Ratel returns them verbatim.

use std::time::Instant;

use ratel_ai_core::{CompressionOptions, PromptCompressor};

const QUERIES: [(&str, &str); 10] = [
    ("terse-imperative", "fix the failing test"),
    (
        "short-howto",
        "How do I reset a git branch to match origin/main exactly?",
    ),
    (
        "short-why",
        "Why is my React component re-rendering on every keystroke?",
    ),
    (
        "negated-constraint",
        "Refactor this handler to use async/await instead of promise chains, but do not \
         change the public function signature and do not add any new dependencies.",
    ),
    (
        "versioned-upgrade",
        "We're on TypeScript 5.2 and want to move to 5.7. Our build uses ts-node with \
         CommonJS output and we have a few files using experimental decorators. What \
         breaks, and in what order should we do the upgrade?",
    ),
    (
        "stack-trace-debug",
        "I'm getting `TypeError: Cannot read properties of undefined (reading 'map')` in \
         src/components/UserList.tsx line 42, but only in production builds, never in dev. \
         The component fetches from /api/users which returns 200 with a valid JSON array \
         when I curl it. What should I check first?",
    ),
    (
        "perf-with-numbers",
        "Our Postgres query that joins orders to line_items takes 4.2 seconds on a table \
         with 12 million rows. There's a btree index on orders.created_at but the planner \
         is doing a sequential scan. EXPLAIN ANALYZE shows the filter removing 11.8 million \
         rows. How do I get it under 200ms?",
    ),
    (
        "multi-constraint-refactor",
        "Split this 800-line module into smaller files. Keep the public exports identical \
         so no caller has to change, don't introduce a barrel file, put each React hook in \
         its own file under hooks/, and leave the existing tests passing without editing them.",
    ),
    (
        "build-failure",
        "CI fails with `error: linking with cc failed: exit status: 1` and `undefined \
         reference to pthread_create` when cross-compiling to aarch64-unknown-linux-gnu. \
         It builds fine on x86_64 and on macOS locally. The crate depends on openssl-sys \
         0.9. How do I fix the cross build?",
    ),
    (
        "review-request",
        "Review this pull request for correctness and security. It adds a new endpoint \
         POST /api/admin/users that creates users, reads the role from the request body, \
         and does not currently check whether the caller is an admin. Tell me what's wrong \
         and how you'd fix it.",
    ),
];

/// Words too common to carry topic. Dropping these is compression working, so
/// the content-only column excludes them.
const STOPWORDS: [&str; 46] = [
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with", "as",
    "is", "are", "was", "were", "be", "it", "its", "this", "that", "i", "we", "you", "my", "our",
    "so", "do", "does", "did", "has", "have", "how", "what", "when", "should", "would", "can",
    "from", "by", "into", "there", "me",
];

/// Normalize a unit to a comparable word; `None` for punctuation-only.
fn word(raw: &str) -> Option<String> {
    let w: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '-' | '/'))
        .flat_map(char::to_lowercase)
        .collect();
    let w = w.trim_matches(|c: char| matches!(c, '.' | '-' | '_' | '/'));
    (!w.is_empty()).then(|| w.to_string())
}

/// Jaccard similarity between the input's and the output's word sets.
///
/// Output is a subsequence of the input, so `set(out)` is always a subset of
/// `set(in)` — the intersection *is* the output set and the union *is* the input
/// set. Jaccard therefore reduces to `|out| / |in|`, i.e. word recall.
///
/// Sets are distinct words, so dropping a repeated word costs nothing — which is
/// the point: repetition is the redundancy compression is meant to remove.
fn jaccard(kept: &[String], dropped: &[String], content_only: bool) -> f32 {
    let keep = |w: &String| !content_only || !STOPWORDS.contains(&w.as_str());
    let mut all: Vec<&String> = Vec::new();
    let mut surviving: Vec<&String> = Vec::new();
    for w in kept.iter().chain(dropped.iter()).filter(|w| keep(w)) {
        if !all.contains(&w) {
            all.push(w);
        }
    }
    for w in kept.iter().filter(|w| keep(w)) {
        if !surviving.contains(&w) {
            surviving.push(w);
        }
    }
    if all.is_empty() {
        return 1.0;
    }
    surviving.len() as f32 / all.len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compressor = PromptCompressor::new();

    let cold = Instant::now();
    compressor.preload()?;
    println!("model load: {} ms\n", cold.elapsed().as_millis());

    // Disable the gate so every query compresses; by default the short ones are
    // returned verbatim.
    let options = CompressionOptions {
        min_importance: 0.5,
        min_words: 0,
        min_tokens: 0,
        ..Default::default()
    };

    // One warm-up so the first query isn't charged for lazily-faulted pages.
    compressor.compress(QUERIES[0].1, Some(&options))?;

    let mut rows = Vec::new();
    for (name, query) in QUERIES {
        // Median of 5 so a single scheduling hiccup doesn't set the number.
        let mut timings = Vec::new();
        let mut result = None;
        for _ in 0..5 {
            let started = Instant::now();
            let out = compressor.compress(query, Some(&options))?;
            timings.push(started.elapsed().as_millis());
            result = Some(out);
        }
        timings.sort_unstable();
        let out = result.expect("at least one run");

        println!("── {name} ─────────────────────────────────────────────");
        println!(
            "before ({} tokens):\n  {}",
            out.stats.model_tokens_in, query
        );
        println!(
            "after  ({} tokens):\n  {}",
            out.stats.model_tokens_out, out.text
        );
        let kept: Vec<String> = out.kept.iter().filter_map(|w| word(&w.text)).collect();
        let dropped: Vec<String> = out.dropped.iter().filter_map(|w| word(&w.text)).collect();
        let j_all = jaccard(&kept, &dropped, false);
        let j_content = jaccard(&kept, &dropped, true);

        // score() applies no policy, so these are the raw model opinions. A
        // high-importance unit marked "no" was lost to the budget, not to
        // the model's judgement.
        let scored = compressor.score(query)?;
        let kept_starts: Vec<usize> = out.kept.iter().map(|u| u.start).collect();
        println!("\n{:<30} {:>7} {:>7}  kept", "unit", "tokens", "imp");
        for u in &scored {
            let text: String = u.text.chars().take(30).collect();
            println!(
                "{:<30} {:>7} {:>7.3}  {}",
                text,
                u.tokens,
                u.importance,
                if kept_starts.contains(&u.start) {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        println!();

        println!(
            "ratio {:.2}x   jaccard {:.2} (content {:.2})   latency {} ms (min {}, max {})\n",
            out.stats.model_tokens_in as f32 / out.stats.model_tokens_out.max(1) as f32,
            j_all,
            j_content,
            timings[timings.len() / 2],
            timings[0],
            timings[timings.len() - 1],
        );

        rows.push((
            name,
            out.stats.model_tokens_in,
            out.stats.model_tokens_out,
            timings[timings.len() / 2],
            j_all,
            j_content,
        ));
    }

    println!(
        "{:<28} {:>7} {:>7} {:>8} {:>9} {:>9} {:>10}",
        "query", "before", "after", "ratio", "jaccard", "content", "median ms"
    );
    println!("{}", "─".repeat(84));
    let (mut total_in, mut total_out, mut total_ms) = (0u32, 0u32, 0u128);
    let (mut sum_j, mut sum_jc) = (0f32, 0f32);
    for (name, before, after, ms, j_all, j_content) in &rows {
        println!(
            "{name:<28} {before:>7} {after:>7} {:>7.2}x {j_all:>9.2} {j_content:>9.2} {ms:>10}",
            *before as f32 / (*after).max(1) as f32
        );
        total_in += before;
        total_out += after;
        total_ms += ms;
        sum_j += j_all;
        sum_jc += j_content;
    }
    let n = rows.len() as f32;
    println!("{}", "─".repeat(84));
    println!(
        "{:<28} {total_in:>7} {total_out:>7} {:>7.2}x {:>9.2} {:>9.2} {total_ms:>10}",
        "TOTAL / mean",
        total_in as f32 / total_out.max(1) as f32,
        sum_j / n,
        sum_jc / n,
    );

    Ok(())
}
