"""Configurable adaptive ranking: seed an intent graph from a baseline capture.

    uv run main.py

No model download, no API key.

The problem this solves: adaptive ranking learns from what the agent invokes,
but when Ratel is already ranking, what it invokes is partly Ratel's own doing.
Capturing first — while Ratel serves nothing — gives evidence the ranker had no
hand in, and a graph that is useful on day one instead of empty.

    A. collect     record each turn + the tools the agent chose, to a JSONL log,
                   scoring the graph-so-far after each one
    B. inspect     read the finished graph before switching anything on
    C. serve       attach it, and keep learning live from there

The TypeScript mirror is ``examples/configurable-adaptive-ranking-ts/src/index.ts``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ratel_ai import IntentGraph, TraceSinkConfig

from tools import BASELINE_TURNS, build_catalog, top_ids

QUERY = "why is the build broken"

SUPPORT_FULL = 3
"""Ratel's full-confidence threshold (``SUPPORT_FULL`` in core). Below it the
arm's weight is ramped down proportionally. Not yet exposed by the SDK — see the
README's "rough edges"."""


@dataclass(frozen=True)
class Readiness:
    """What the graph-so-far looks like."""

    clusters: int
    support: list[int]
    """Each cluster's observation count, strongest first. Printed as ``n/3``
    because 3 is where the boost reaches full strength."""
    observations: int
    from_baseline: int


def readiness(graph: IntentGraph) -> Readiness:
    """Score a candidate graph. Reads the graph only — nothing is attached."""
    intents = json.loads(graph.to_json())["intents"]
    return Readiness(
        clusters=graph.cluster_count,
        support=sorted((it["support"] for it in intents), reverse=True),
        observations=sum(it["support"] for it in intents),
        from_baseline=sum(it.get("seeded_support", 0) for it in intents),
    )


async def main() -> None:
    log_path = Path(tempfile.mkdtemp(prefix="ratel-baseline-")) / "telemetry.jsonl"

    capture = await build_catalog(
        TraceSinkConfig(kind="jsonl", session_id="session-1", path=str(log_path))
    )
    serving = await build_catalog()

    print(f'query: "{QUERY}"')
    print(f"  cold (BM25 only) : {' > '.join(top_ids(capture, QUERY))}")
    print(f"  ranking status   : {capture.experimental_adaptive_ranking_status}")

    # -----------------------------------------------------------------------
    # A. Collect — the agent runs on its own full tool list; Ratel only records.
    #
    #    After each turn we rebuild the graph from the log SO FAR and score it.
    #    `experimental_initialize_intent_graph` is a pure function of (log,
    #    policy) returning a DETACHED graph, so polling mid-capture is safe —
    #    nothing being served is touched.
    # -----------------------------------------------------------------------
    print("\nA. collecting — Ratel records every invocation; the graph is scored after each\n")

    for i, (turn, invoked) in enumerate(BASELINE_TURNS, start=1):
        # Every invocation is evidence. Nothing in a trace says whether a turn
        # went well, so none of them is filtered — including the wrong one.
        # The query first: invocations attribute to the session's most recent.
        capture.experimental_record_baseline_query(turn)
        capture.record_event(
            {"type": "invoke_start", "tool_id": invoked, "args_size_bytes": 0}
        )

        graph = await serving.experimental_initialize_intent_graph(
            log_path.read_text(), origins="baseline", provenance="seeded"
        )
        r = readiness(graph)
        support = ", ".join(f"{n}/{SUPPORT_FULL}" for n in r.support)
        print(
            f"  turn {i}  {invoked:<13} clusters={r.clusters} support={support} "
            f"obs={r.observations} from_baseline={r.from_baseline}"
        )

    print(f"\n  log -> {log_path}")

    # -----------------------------------------------------------------------
    # B. Inspect — the finished graph, before switching anything on.
    # -----------------------------------------------------------------------
    graph = await serving.experimental_initialize_intent_graph(
        log_path.read_text(),
        origins="baseline",  # only observed turns count, not Ratel's own searches
        provenance="seeded",  # record that this came from a capture, not live traffic
    )
    print("\nB. built graph:")
    for intent in json.loads(graph.to_json())["intents"]:
        edges = ", ".join(f"{tool} x{weight:g}" for tool, weight in intent["tools"].items())
        print(f'  "{intent["label"]}"')
        print(
            f"    observations  : {intent['support']} "
            f"({intent.get('seeded_support', 0)} from this capture)"
        )
        print(f"    invoked       : {edges}")
        print(f"    phrasings     : {len(intent['members'])}")

    # Still detached: building and scoring never switch ranking on.
    print(f"  ranking status  : {serving.experimental_adaptive_ranking_status}")

    # -----------------------------------------------------------------------
    # C. Serve — attach, and rank on what the agent actually did.
    # -----------------------------------------------------------------------
    serving.experimental_enable_adaptive_ranking(graph)
    print(f"\nC. after seeding   : {' > '.join(top_ids(serving, QUERY))}")
    print(f"   ranking status  : {serving.experimental_adaptive_ranking_status}")

    # From here the live learner keeps adding to the same graph. `support` grows
    # while `seeded_support` stays put, so the gap tells you how much of each
    # cluster still rests on the baseline versus what live traffic has confirmed.
    print(f"\npersist with graph.to_json() — rev={graph.rev} marks what to save.")
    print("""
Reading the collection columns:
  clusters        distinct intents found so far
  support         each cluster's observations, out of the 3 that reach full
                  strength — below that the boost is scaled down proportionally
  obs             confirmed observations across every cluster
  from_baseline   how many of those came from this capture rather than live
                  traffic; after the flip it stays put while obs keeps growing

Treat these as a report for a person to read, not an auto-trigger.""")


if __name__ == "__main__":
    asyncio.run(main())
