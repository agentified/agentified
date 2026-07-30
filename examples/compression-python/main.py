"""Experimental prompt compression: compress the context, keep the scaffolding.

    uv run main.py

Downloads the ~700 MB compression model on first run.
"""

import asyncio
import re
import time

from ratel_ai import ExperimentalCompression

# Notes an agent is carrying — the part worth compressing.
NOTES = """
Speaker 1: Let's get the Q3 infrastructure review started. The first item is the
database migration we've been putting off since May. We're running Postgres 12 in
production, which reached end of life in November 2024, so we are unsupported and
security patches are no longer issued.

Speaker 2: We tried this migration once already, in April, and it failed. The
pg_upgrade step took about four hours on the primary and our maintenance window
was only two hours, so we rolled back halfway through. That rollback took another
hour and we ended up with roughly three hours of degraded read performance.

Speaker 1: What's different this time?

Speaker 2: We're using logical replication instead of pg_upgrade, so we can build
the Postgres 16 replica alongside the existing primary and let it catch up over
several days with no downtime. The cutover is a DNS switch plus a connection pool
drain, somewhere between 90 seconds and 4 minutes. We've tested this exact
procedure twice in staging with a full production data snapshot.

Speaker 3: What does running both clusters cost?

Speaker 2: About 8,400 dollars for the two week overlap. TimescaleDB is the risky
one: the version we're on, 2.8, doesn't support Postgres 16 at all, so we need to
bump Timescale to 2.14 first in a separate window.
""".strip()

# The instruction and the question are never compressed — losing a word in
# either changes the task, not the background.
INSTRUCTION = "You are given notes from an engineering meeting. Answer using only these notes."
QUESTION = "What is the plan for the Postgres migration, and what is the main risk?"


async def main() -> None:
    compressor = ExperimentalCompression()

    # Pay the ~700 MB load deliberately rather than inside the first request.
    print("loading model…")
    started = time.perf_counter()
    await compressor.preload()
    print(f"ready in {int((time.perf_counter() - started) * 1000)} ms\n")

    result = await compressor.compress(NOTES, rate=0.4)

    print("=== compressed notes ===")
    print(result.text)
    print("\n=== what it cost ===")
    stats = result.stats
    ratio = stats.model_tokens_in / max(stats.model_tokens_out, 1)
    print(
        f"{stats.model_tokens_in} -> {stats.model_tokens_out} model tokens ({ratio:.2f}x), "
        f"{stats.words_in} -> {stats.words_out} units, "
        f"{stats.chunks} encoder pass(es), {stats.took_ms} ms"
    )

    # The explainability channel: why each unit went, not just that it did.
    worst = sorted(result.dropped, key=lambda w: -w.importance)[:6]
    print(
        "closest calls (highest-scoring dropped):",
        ", ".join(f"{w.text}({w.importance:.2f})" for w in worst),
    )

    # Assemble the real prompt: only the middle was rewritten.
    prompt = f"{INSTRUCTION}\n\n<notes>\n{result.text}\n</notes>\n\nQuestion: {QUESTION}"
    print(f"\nprompt assembled: {len(prompt)} chars (notes were {len(NOTES)})")

    # Short prompts are out of domain — they come back untouched, and say so.
    short = await compressor.compress("translate this to french")
    print(
        f"\nshort input -> gate={short.stats.gate}, "
        f"text unchanged: {short.text == 'translate this to french'}"
    )

    # Protect what must survive at any rate. Naming the figures is cheaper and
    # more precise than protecting every digit.
    aggressive = await compressor.compress(
        NOTES, rate=0.15, protect=[re.compile(r"8,400"), re.compile(r"2\.14")]
    )
    print(
        f"\nat rate 0.15 with protection: {aggressive.stats.model_tokens_out} tokens, "
        f"8,400 kept={'8,400' in aggressive.text}, 2.14 kept={'2.14' in aggressive.text}, "
        f"budget_exceeded={aggressive.stats.budget_exceeded}"
    )


if __name__ == "__main__":
    asyncio.run(main())
