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

from tools import BASELINE_TURNS, HELD_OUT, build_catalog, top_ids

QUERY = "why is the build broken"

SUPPORT_FULL = 3
"""Ratel's full-confidence threshold (``SUPPORT_FULL`` in core). Below it the
arm's weight is ramped down proportionally. Not yet exposed by the SDK — see the
README's "rough edges"."""


@dataclass(frozen=True)
class Readiness:
    """What the graph-so-far looks like."""

    clusters: int
    support: int
    """Observations behind the cluster THIS turn landed in. Printed as ``n/3``
    because 3 is where the boost reaches full strength — reporting one cluster
    per line, since a list of every cluster's support says nothing about which
    one the turn just changed."""
    observations: int
    from_baseline: int
    coverage_hits: int
    coverage_probed: int
    """Held-out queries that matched a cluster, over the number probed. The only
    column measured against questions the graph has NOT seen — the others rise
    whether or not it generalises."""


async def readiness(graph: IntentGraph, turn: str) -> Readiness:
    """Score a candidate graph without attaching it to anything you serve.

    The coverage probe runs on a throwaway catalog, so the graph under test never
    touches live ranking.
    """
    intents = json.loads(graph.to_json())["intents"]
    landed = next((it for it in intents if turn in it["members"]), None)

    probe = await build_catalog(TraceSinkConfig(kind="memory", session_id="probe"))
    probe.experimental_enable_adaptive_ranking(graph)
    for query in HELD_OUT:
        probe.search(query, 5)
    boosts = [e for e in probe.drain_trace_events() if e["type"] == "usage_boost"]

    return Readiness(
        clusters=graph.cluster_count,
        support=landed["support"] if landed else 0,
        observations=sum(it["support"] for it in intents),
        from_baseline=sum(it.get("seeded_support", 0) for it in intents),
        coverage_hits=sum(1 for e in boosts if e["intent"] is not None),
        coverage_probed=len(boosts),
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
    probes = ", ".join(f'"{q}"' for q in HELD_OUT)
    print(f"\nA. collecting — scoring after each turn against held-out: {probes}\n")

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
        r = await readiness(graph, turn)
        # Past the threshold "5/3" reads like a bug, so say what it means.
        support = (
            f"{r.support} (full)" if r.support >= SUPPORT_FULL else f"{r.support}/{SUPPORT_FULL}"
        )
        print(
            f"  turn {i:>2}  {invoked:<13} clusters={r.clusters} "
            f"support={support:<9} "
            f"obs={r.observations:<2} from_baseline={r.from_baseline} "
            f"coverage={r.coverage_hits}/{r.coverage_probed}"
        )

    print(f"\n  log -> {log_path}")
    print(
        '\n  coverage stops at 3/4: "is CI green on my branch" never matches. It'
        "\n  means the same as a captured turn but shares no words with one, and"
        "\n  this demo runs on BM25. Bridging that is what the dense tier is for —"
        '\n  a real deployment wants method="semantic".'
    )

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
  support         observations behind the cluster THIS turn landed in, out of
                  the 3 that reach full strength — below that the boost is
                  scaled down proportionally
  obs             confirmed observations across every cluster
  from_baseline   how many of those came from this capture rather than live
                  traffic; after the flip it stays put while obs keeps growing
  coverage        held-out queries that matched a cluster — none of them a
                  captured turn. THE ONE TO GATE ON: the others rise whether or
                  not the graph generalises, so a healthy-looking graph can
                  still fire on none of your traffic

Treat these as a report for a person to read, not an auto-trigger.""")


if __name__ == "__main__":
    asyncio.run(main())
