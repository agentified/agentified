//! Compression against **coding queries** — the shape of prompt Ratel's users
//! actually send, as opposed to the long meeting transcript in
//! `compression_golden.rs`.
//!
//! `#[ignore]`d because it needs the ~700 MB default checkpoint on disk:
//!
//! ```text
//! cargo test -p ratel-ai-core --release --test compression_queries -- --ignored --nocapture
//! ```
//!
//! The question this suite answers is not "how much can we compress" but **"is
//! compression the right tool for this input at all"**. A coding query carries
//! identifiers, versions, and negations in a few dozen tokens, with almost no
//! redundancy to spend — the opposite of a transcript. The suite is written so a
//! *negative* result is a first-class outcome rather than a failure.

use ratel_ai_core::{CompressionOptions, PromptCompressor};

/// One query plus the tokens an answer depends on. A compressed query that
/// loses any of these is worse than useless: it silently redirects the agent.
struct Query {
    name: &'static str,
    text: &'static str,
    critical: &'static [&'static str],
}

/// Ten coding queries spanning the realistic range, from a one-line ask to a
/// paste-a-stack-trace debugging prompt.
const QUERIES: &[Query] = &[
    Query {
        name: "terse-imperative",
        text: "fix the failing test",
        critical: &["fix", "failing", "test"],
    },
    Query {
        name: "short-howto",
        text: "How do I reset a git branch to match origin/main exactly?",
        critical: &["reset", "git", "origin/main"],
    },
    Query {
        name: "short-why",
        text: "Why is my React component re-rendering on every keystroke?",
        critical: &["React", "re-rendering", "keystroke"],
    },
    Query {
        name: "negated-constraint",
        text: "Refactor this handler to use async/await instead of promise chains, \
               but do not change the public function signature and do not add any \
               new dependencies.",
        critical: &["async/await", "not change", "signature", "not add"],
    },
    Query {
        name: "versioned-upgrade",
        text: "We're on TypeScript 5.2 and want to move to 5.7. Our build uses ts-node \
               with CommonJS output and we have a few files using experimental decorators. \
               What breaks, and in what order should we do the upgrade?",
        critical: &[
            "TypeScript",
            "5.2",
            "5.7",
            "ts-node",
            "CommonJS",
            "decorators",
        ],
    },
    Query {
        name: "stack-trace-debug",
        text: "I'm getting `TypeError: Cannot read properties of undefined (reading 'map')` \
               in src/components/UserList.tsx line 42, but only in production builds, never \
               in dev. The component fetches from /api/users which returns 200 with a valid \
               JSON array when I curl it. What should I check first?",
        critical: &[
            "TypeError",
            "undefined",
            "UserList.tsx",
            "42",
            "production",
            "/api/users",
            "200",
        ],
    },
    Query {
        name: "perf-with-numbers",
        text: "Our Postgres query that joins orders to line_items takes 4.2 seconds on a \
               table with 12 million rows. There's a btree index on orders.created_at but \
               the planner is doing a sequential scan. EXPLAIN ANALYZE shows the filter \
               removing 11.8 million rows. How do I get it under 200ms?",
        critical: &[
            "Postgres",
            "line_items",
            "4.2",
            "12 million",
            "btree",
            "created_at",
            "sequential scan",
            "200ms",
        ],
    },
    Query {
        name: "multi-constraint-refactor",
        text: "Split this 800-line module into smaller files. Keep the public exports \
               identical so no caller has to change, don't introduce a barrel file, put \
               each React hook in its own file under hooks/, and leave the existing tests \
               passing without editing them.",
        critical: &[
            "public exports",
            "identical",
            "don't introduce",
            "hooks/",
            "without editing",
        ],
    },
    Query {
        name: "build-failure",
        text: "CI fails with `error: linking with cc failed: exit status: 1` and \
               `undefined reference to pthread_create` when cross-compiling to \
               aarch64-unknown-linux-gnu. It builds fine on x86_64 and on macOS locally. \
               The crate depends on openssl-sys 0.9. How do I fix the cross build?",
        critical: &[
            "linking",
            "pthread_create",
            "aarch64-unknown-linux-gnu",
            "x86_64",
            "openssl-sys",
        ],
    },
    Query {
        name: "review-request",
        text: "Review this pull request for correctness and security. It adds a new \
               endpoint POST /api/admin/users that creates users, reads the role from the \
               request body, and does not currently check whether the caller is an admin. \
               Tell me what's wrong and how you'd fix it.",
        critical: &["POST /api/admin/users", "role", "does not", "admin"],
    },
];

/// Which of a query's load-bearing tokens did **not** survive compression.
fn lost<'a>(text: &str, critical: &[&'a str]) -> Vec<&'a str> {
    critical
        .iter()
        .filter(|needle| !text.contains(**needle))
        .copied()
        .collect()
}

/// The headline: run every query at the default configuration and print what
/// happened. The gate firing is a **correct** outcome, not a skipped test.
#[test]
#[ignore = "needs the compression model cached"]
fn coding_queries_at_the_default_configuration() {
    let compressor = PromptCompressor::new();
    compressor.preload().expect("model");

    println!(
        "\n{:<26} {:>6} {:>7} {:>7}  {:<16} lost",
        "query", "tokens", "out", "ratio", "gate",
    );
    println!("{}", "-".repeat(96));

    let mut compressed = 0usize;
    let mut lossy = 0usize;

    for q in QUERIES {
        let out = compressor.compress(q.text, None).expect("compress");
        let missing = lost(&out.text, q.critical);
        let gate = out
            .stats
            .gate
            .map(|g| format!("{g:?}"))
            .unwrap_or_else(|| "-".to_string());
        if out.stats.gate.is_none() {
            compressed += 1;
            if !missing.is_empty() {
                lossy += 1;
            }
        }
        println!(
            "{:<26} {:>6} {:>7} {:>7}  {:<16} {}",
            q.name,
            out.stats.model_tokens_in,
            out.stats.model_tokens_out,
            if out.stats.model_tokens_out > 0 {
                format!(
                    "{:.2}x",
                    out.stats.model_tokens_in as f32 / out.stats.model_tokens_out as f32
                )
            } else {
                "-".to_string()
            },
            gate,
            if missing.is_empty() {
                "-".to_string()
            } else {
                format!("{missing:?}")
            },
        );
        if out.stats.gate.is_none() {
            println!("      -> {}", out.text);
        }
    }

    println!(
        "\n{} of {} queries were compressed at all; {} of those lost a critical token.",
        compressed,
        QUERIES.len(),
        lossy
    );
}

/// A gated query must come back **byte-identical**. This is the load-bearing
/// guarantee for the query side: if compression is going to be a no-op, it must
/// be an exact no-op, not a nearly-exact one.
#[test]
#[ignore = "needs the compression model cached"]
fn gated_queries_are_returned_byte_identical() {
    let compressor = PromptCompressor::new();
    for q in QUERIES {
        let out = compressor.compress(q.text, None).expect("compress");
        if out.stats.gate.is_some() {
            assert_eq!(out.text, q.text, "{} was altered while gated", q.name);
        }
    }
}

/// Force compression on every query by disabling the gate, and measure what it
/// costs. This is the experiment that says whether the gate's threshold is
/// defensible — if forcing compression were harmless, the gate would be
/// over-cautious and should be lowered.
#[test]
#[ignore = "needs the compression model cached"]
fn forcing_compression_past_the_gate_shows_why_the_gate_exists() {
    let compressor = PromptCompressor::new();
    compressor.preload().expect("model");

    let forced = CompressionOptions {
        min_words: 0,
        min_tokens: 0,
        ..Default::default()
    };

    println!("\nforcing every query past the gate at rate 0.40:\n");
    let mut damaged = Vec::new();
    for q in QUERIES {
        let out = compressor
            .compress(q.text, Some(&forced))
            .expect("compress");
        let missing = lost(&out.text, q.critical);
        println!(
            "{:<26} {:>3} -> {:<3} tokens   lost {:?}",
            q.name, out.stats.model_tokens_in, out.stats.model_tokens_out, missing
        );
        println!("      -> {}", out.text);
        if !missing.is_empty() {
            damaged.push(q.name);
        }
    }

    println!(
        "\n{} of {} queries lost a critical token when forced: {damaged:?}",
        damaged.len(),
        QUERIES.len()
    );
    assert!(
        !damaged.is_empty(),
        "if forcing compression on short queries were lossless, the min-length gate \
         would be over-cautious and its threshold should be lowered"
    );
}

/// The whole batch, as a single budget. Even summed, coding queries are a poor
/// target: the total saving is small in absolute tokens, and every token saved
/// is one the agent might have needed.
#[test]
#[ignore = "needs the compression model cached"]
fn the_whole_batch_saves_little_in_absolute_terms() {
    let compressor = PromptCompressor::new();
    compressor.preload().expect("model");

    let (mut total_in, mut total_out) = (0u32, 0u32);
    for q in QUERIES {
        let out = compressor.compress(q.text, None).expect("compress");
        // A gated call reports 0 tokens (no model ran), so count the text itself.
        let (i, o) = match out.stats.gate {
            Some(_) => {
                let n = compressor
                    .score(q.text)
                    .map(|s| s.iter().map(|w| w.tokens).sum::<u32>());
                let n = n.unwrap_or(0);
                (n, n)
            }
            None => (out.stats.model_tokens_in, out.stats.model_tokens_out),
        };
        total_in += i;
        total_out += o;
    }
    println!(
        "\nbatch of {} coding queries: {} -> {} model tokens ({} saved)",
        QUERIES.len(),
        total_in,
        total_out,
        total_in - total_out
    );
}
