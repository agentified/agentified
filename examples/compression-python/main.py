"""Compression on ten coding-task queries: before, after, and latency.

    uv run main.py

Downloads the ~700 MB compression model on first run.

The min-length gate is disabled here so every query actually compresses — by
default half of these are short enough that Ratel returns them verbatim.
"""

import asyncio
import time

from ratel_ai import ExperimentalCompression

QUERIES: list[tuple[str, str]] = [
    ("terse-imperative", "fix the failing test"),
    ("short-howto", "How do I reset a git branch to match origin/main exactly?"),
    ("short-why", "Why is my React component re-rendering on every keystroke?"),
    (
        "negated-constraint",
        "Refactor this handler to use async/await instead of promise chains, but do not "
        "change the public function signature and do not add any new dependencies.",
    ),
    (
        "versioned-upgrade",
        "We're on TypeScript 5.2 and want to move to 5.7. Our build uses ts-node with "
        "CommonJS output and we have a few files using experimental decorators. What "
        "breaks, and in what order should we do the upgrade?",
    ),
    (
        "stack-trace-debug",
        "I'm getting `TypeError: Cannot read properties of undefined (reading 'map')` in "
        "src/components/UserList.tsx line 42, but only in production builds, never in dev. "
        "The component fetches from /api/users which returns 200 with a valid JSON array "
        "when I curl it. What should I check first?",
    ),
    (
        "perf-with-numbers",
        "Our Postgres query that joins orders to line_items takes 4.2 seconds on a table "
        "with 12 million rows. There's a btree index on orders.created_at but the planner "
        "is doing a sequential scan. EXPLAIN ANALYZE shows the filter removing 11.8 "
        "million rows. How do I get it under 200ms?",
    ),
    (
        "multi-constraint-refactor",
        "Split this 800-line module into smaller files. Keep the public exports identical "
        "so no caller has to change, don't introduce a barrel file, put each React hook in "
        "its own file under hooks/, and leave the existing tests passing without editing "
        "them.",
    ),
    (
        "build-failure",
        "CI fails with `error: linking with cc failed: exit status: 1` and `undefined "
        "reference to pthread_create` when cross-compiling to aarch64-unknown-linux-gnu. "
        "It builds fine on x86_64 and on macOS locally. The crate depends on openssl-sys "
        "0.9. How do I fix the cross build?",
    ),
    (
        "review-request",
        "Review this pull request for correctness and security. It adds a new endpoint "
        "POST /api/admin/users that creates users, reads the role from the request body, "
        "and does not currently check whether the caller is an admin. Tell me what's wrong "
        "and how you'd fix it.",
    ),
]


# Words too common to carry topic. Dropping these is compression working, so the
# content-only column excludes them.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with", "as",
    "is", "are", "was", "were", "be", "it", "its", "this", "that", "i", "we", "you", "my", "our",
    "so", "do", "does", "did", "has", "have", "how", "what", "when", "should", "would", "can",
    "from", "by", "into", "there", "me",
}


def _word(raw: str) -> str | None:
    """Normalize a unit to a comparable word; None for punctuation-only."""
    w = "".join(c for c in raw.lower() if c.isalnum() or c in "_.-/").strip("._-/")
    return w or None


def jaccard(kept: list[str], dropped: list[str], *, content_only: bool) -> float:
    """Jaccard similarity between the input's and the output's word sets.

    Output is a subsequence of the input, so set(out) is always a subset of
    set(in) — the intersection *is* the output set and the union *is* the input
    set. Jaccard therefore reduces to |out| / |in|, i.e. word recall.

    Sets are distinct words, so dropping a repeated word costs nothing — which is
    the point: repetition is the redundancy compression is meant to remove.
    """
    keep = (lambda w: w not in STOPWORDS) if content_only else (lambda w: True)
    all_words = {w for w in kept + dropped if keep(w)}
    surviving = {w for w in kept if keep(w)}
    return 1.0 if not all_words else len(surviving) / len(all_words)


#: Importance at or above which the model is treating a token as load-bearing.
IMPORTANT = 0.7


def density(scored: list) -> float:
    """Fraction of tokens the model rates as important.

    This is the model's own view of how much the text can afford to lose. Read it
    against `rate`: compressing to 0.4 when density is 0.7 means cutting 30 points
    of material the model considers important — damage you can predict before you
    do it.
    """
    total = sum(u.tokens for u in scored)
    if total == 0:
        return 0.0
    return sum(u.tokens for u in scored if u.importance >= IMPORTANT) / total


async def main() -> None:
    compressor = ExperimentalCompression()

    print("loading model…")
    started = time.perf_counter()
    await compressor.preload()
    print(f"ready in {int((time.perf_counter() - started) * 1000)} ms\n")

    # min_words/min_tokens at 0 turns the gate off so every query compresses.
    options = {"min_importance": 0.5, "min_words": 0, "min_tokens": 0}

    # One warm-up so the first query isn't charged for lazily-faulted pages.
    await compressor.compress(QUERIES[0][1], **options)  # type: ignore[arg-type]

    rows = []
    for name, query in QUERIES:
        # Median of 5 so a single scheduling hiccup doesn't set the number.
        timings = []
        result = None
        for _ in range(5):
            t0 = time.perf_counter()
            result = await compressor.compress(query, **options)  # type: ignore[arg-type]
            timings.append((time.perf_counter() - t0) * 1000)
        timings.sort()
        assert result is not None
        stats = result.stats

        print(f"── {name} " + "─" * max(0, 45 - len(name)))
        print(f"before ({stats.model_tokens_in} tokens):\n  {query}")
        print(f"after  ({stats.model_tokens_out} tokens):\n  {result.text}")
        # score() applies no policy, so these are the raw model opinions —
        # protection would otherwise pin some to 1.0 and inflate density.
        scored = await compressor.score(query)
        dens = density(scored)
        # With a bar rather than a budget, the gate question is simply whether
        # most of the text clears it — nothing is being traded away.
        would_gate = dens > 0.5

        # Per-unit detail: what the model thought, and what selection did with it.
        # A high-importance unit marked "no" was lost to the budget, not to the
        # model's judgement.
        kept_starts = {u.start for u in result.kept}
        print(f"\n{'unit':<30} {'tokens':>7} {'imp':>7}  kept")
        for u in scored:
            print(
                f"{u.text[:30]:<30} {u.tokens:>7} {u.importance:>7.3f}  "
                f"{'yes' if u.start in kept_starts else 'no'}"
            )
        print()

        kept = [w for w in (_word(u.text) for u in result.kept) if w]
        dropped = [w for w in (_word(u.text) for u in result.dropped) if w]
        j_all = jaccard(kept, dropped, content_only=False)
        j_content = jaccard(kept, dropped, content_only=True)

        print(
            f"ratio {stats.model_tokens_in / max(stats.model_tokens_out, 1):.2f}x   "
            f"jaccard {j_all:.2f} (content {j_content:.2f})   "
            f"density {dens:.2f}{' GATE' if would_gate else ''}   "
            f"latency {timings[len(timings) // 2]:.0f} ms "
            f"(min {timings[0]:.0f}, max {timings[-1]:.0f})\n"
        )
        rows.append(
            (
                name,
                stats.model_tokens_in,
                stats.model_tokens_out,
                timings[len(timings) // 2],
                j_all,
                j_content,
                dens,
                would_gate,
            )
        )

    print(
        f"{'query':<28} {'before':>7} {'after':>7} {'ratio':>8} "
        f"{'jaccard':>9} {'content':>9} {'density':>9} {'gate':>6} {'median ms':>10}"
    )
    print("─" * 101)
    total_in = total_out = 0
    total_ms = sum_j = sum_jc = 0.0
    for name, before, after, ms, j_all, j_content, dens, would_gate in rows:
        print(
            f"{name:<28} {before:>7} {after:>7} {before / max(after, 1):>7.2f}x "
            f"{j_all:>9.2f} {j_content:>9.2f} {dens:>9.2f} "
            f"{'REFUSE' if would_gate else '-':>6} {ms:>10.0f}"
        )
        total_in += before
        total_out += after
        total_ms += ms
        sum_j += j_all
        sum_jc += j_content
    n = len(rows)
    refused = sum(1 for r in rows if r[7])
    print("─" * 101)
    print(
        f"{'TOTAL / mean':<28} {total_in:>7} {total_out:>7} "
        f"{total_in / max(total_out, 1):>7.2f}x "
        f"{sum_j / n:>9.2f} {sum_jc / n:>9.2f} "
        f"{sum(r[6] for r in rows) / n:>9.2f} {f'{refused}/{n}':>6} {total_ms:>10.0f}"
    )
    print(
        f"\ndensity = fraction of tokens the model rates >= {IMPORTANT}, i.e. how much of\n"
        f"the text carries meaning. Selection keeps every unit at or above\n"
        f"min_importance={options['min_importance']}, so the ratio is an *output*: dense text\n"
        "barely compresses, redundant text compresses hard, and nothing above the bar\n"
        "is ever traded away to hit a size."
    )


if __name__ == "__main__":
    asyncio.run(main())
