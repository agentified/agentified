"""Configurable adaptive ranking: seed an intent graph from a baseline capture.

    uv run main.py

No model download, no API key.

The problem this solves: adaptive ranking learns from what the agent invokes,
but when Ratel is already ranking, what it invokes is partly Ratel's own doing.
Capturing first — while Ratel serves nothing — gives evidence the ranker had no
hand in, and a graph that is useful on day one instead of empty.

    A. collect     record each turn + the tools the agent chose, to a JSONL log
    B. initialize  build a graph from that log, offline
    C. inspect     decide whether it is worth switching on
    D. serve       attach it, and keep learning live from there

The TypeScript mirror is ``examples/configurable-adaptive-ranking-ts/src/index.ts``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from ratel_ai import TraceSinkConfig

from tools import BASELINE_TURNS, build_catalog, top_ids

QUERY = "why is the build broken"


async def main() -> None:
    log_path = Path(tempfile.mkdtemp(prefix="ratel-baseline-")) / "telemetry.jsonl"

    # -----------------------------------------------------------------------
    # A. Collect — the agent runs on its own full tool list; Ratel only records.
    # -----------------------------------------------------------------------
    capture = await build_catalog(
        TraceSinkConfig(kind="jsonl", session_id="session-1", path=str(log_path))
    )

    print(f'query: "{QUERY}"')
    print(f"  cold (BM25 only) : {' > '.join(top_ids(capture, QUERY))}")
    print(f"  ranking status   : {capture.experimental_adaptive_ranking_status}")

    for entry in BASELINE_TURNS:
        # The quality gate. Emission is per turn and opt-in, so a turn you would
        # not want the graph to learn from simply never enters it.
        if not entry["ok"]:
            continue

        # The query first: invocations attribute to the session's most recent one.
        capture.experimental_record_baseline_query(str(entry["turn"]))
        capture.record_event(
            {"type": "invoke_start", "tool_id": entry["invoked"], "args_size_bytes": 0}
        )

    kept = sum(1 for entry in BASELINE_TURNS if entry["ok"])
    print(f"\ncaptured {kept} turns -> {log_path}")

    # -----------------------------------------------------------------------
    # B. Initialize — build the graph from the log, offline.
    # -----------------------------------------------------------------------
    serving = await build_catalog()
    graph = await serving.experimental_initialize_intent_graph(
        log_path.read_text(),
        origins="baseline",  # only observed turns count, not Ratel's own searches
        provenance="seeded",  # record that this came from a capture, not live traffic
    )

    # -----------------------------------------------------------------------
    # C. Inspect — before switching anything on.
    # -----------------------------------------------------------------------
    inspected = json.loads(graph.to_json())
    print("\nbuilt graph:")
    print(f"  clusters        : {graph.cluster_count}")
    for intent in inspected["intents"]:
        # Edge weights are invocation counts, carried as JSON numbers — print
        # them as the integers they are rather than as "x3.0".
        edges = ", ".join(f"{tool} x{weight:g}" for tool, weight in intent["tools"].items())
        print(f'  "{intent["label"]}"')
        print(
            f"    observations  : {intent['support']} "
            f"({intent.get('seeded_support', 0)} seeded)"
        )
        print(f"    invoked       : {edges}")
        print(f"    phrasings     : {len(intent['members'])}")
    # Still detached: building a graph never switches ranking on.
    print(f"  ranking status  : {serving.experimental_adaptive_ranking_status}")

    # -----------------------------------------------------------------------
    # D. Serve — attach, and rank on what the agent actually did.
    # -----------------------------------------------------------------------
    serving.experimental_enable_adaptive_ranking(graph)
    print(f"\n  after seeding   : {' > '.join(top_ids(serving, QUERY))}")
    print(f"  ranking status  : {serving.experimental_adaptive_ranking_status}")

    # From here the live learner keeps adding to the same graph. `support` grows
    # while `seeded_support` stays put, so the gap tells you how much of each
    # cluster still rests on the baseline versus what live traffic has confirmed.
    print(f"\npersist with graph.to_json() — rev={graph.rev} marks what to save.")


if __name__ == "__main__":
    asyncio.run(main())
