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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ratel_ai import IntentGraph, ToolCatalog, TraceSinkConfig

from tools import BASELINE_TURNS, HELD_OUT, build_catalog, top_ids

QUERY = "why is the build broken"

SUPPORT_FULL = 3
"""Ratel's full-confidence threshold (``SUPPORT_FULL`` in core). Below it the
arm's weight is ramped down proportionally. Not yet exposed by the SDK — see the
README's "rough edges"."""


@dataclass(frozen=True)
class Readiness:
    """What a maturity check says about a candidate graph."""

    clusters: int
    mature: int
    observations: int
    seeded: int
    ghosts: list[str]
    coverage_hits: int
    coverage_probed: int


async def readiness(
    graph: IntentGraph,
    serving: ToolCatalog,
    known: Callable[[str], bool] | None = None,
) -> Readiness:
    """Score a candidate graph without attaching it to anything you serve.

    The coverage probe runs on a throwaway catalog, so the graph under test
    never touches live ranking. ``known`` overrides which tool ids count as
    defined — used below to simulate catalog drift.
    """
    intents = json.loads(graph.to_json())["intents"]
    defines = known or serving.has

    probe = await build_catalog(TraceSinkConfig(kind="memory", session_id="probe"))
    probe.experimental_enable_adaptive_ranking(graph)
    for query in HELD_OUT:
        probe.search(query, 5)
    boosts = [e for e in probe.drain_trace_events() if e["type"] == "usage_boost"]

    return Readiness(
        clusters=graph.cluster_count,
        mature=sum(1 for it in intents if it["support"] >= SUPPORT_FULL),
        observations=sum(it["support"] for it in intents),
        seeded=sum(it.get("seeded_support", 0) for it in intents),
        ghosts=[tool for it in intents for tool in it["tools"] if not defines(tool)],
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
    print(f"\nA. collecting — scoring against held-out queries: {probes}\n")

    for i, entry in enumerate(BASELINE_TURNS, start=1):
        # The quality gate. Emission is per turn and opt-in, so a turn you would
        # not want the graph to learn from simply never enters it.
        if not entry["ok"]:
            print(f'  turn {i}  "{entry["turn"]}" -> {entry["invoked"]}   SKIPPED (unsuccessful)')
            continue

        # The query first: invocations attribute to the session's most recent one.
        capture.experimental_record_baseline_query(str(entry["turn"]))
        capture.record_event(
            {"type": "invoke_start", "tool_id": entry["invoked"], "args_size_bytes": 0}
        )

        graph = await serving.experimental_initialize_intent_graph(
            log_path.read_text(), origins="baseline", provenance="seeded"
        )
        r = await readiness(graph, serving)
        note = ""
        if r.mature == 1 and r.clusters == 1 and r.observations == SUPPORT_FULL:
            note = "   <- support hit 3: full arm weight"
        elif r.coverage_hits == r.coverage_probed:
            note = "   <- every probe covered"
        print(
            f"  turn {i}  clusters={r.clusters} mature={r.mature} obs={r.observations} "
            f"seeded={r.seeded} ghosts={len(r.ghosts)} "
            f"coverage={r.coverage_hits}/{r.coverage_probed}{note}"
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
            f"({intent.get('seeded_support', 0)} seeded)"
        )
        print(f"    invoked       : {edges}")
        print(f"    phrasings     : {len(intent['members'])}")

    # What catalog drift looks like. A graph outlives the catalog it was built
    # against — tools get renamed or removed. Those edges are dropped silently
    # at rank time, so a graph can look populated and boost nothing.
    drifted = await readiness(
        graph, serving, known=lambda tool: tool != "gh_run_list" and serving.has(tool)
    )
    print(
        f"\n  if gh_run_list left the catalog: ghosts={len(drifted.ghosts)} "
        f"({', '.join(drifted.ghosts)}) — those edges would rank nothing"
    )
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
  clusters   distinct intents found so far
  mature     clusters at support >= 3 — below that the boost is ramped down
  obs        confirmed observations; seeded says how many came from capture
  ghosts     edges naming tools the serving catalog no longer has (want 0)
  coverage   held-out queries that matched a cluster — the generalisation check

Only coverage is measured against queries the graph has not seen. Clusters and
support both rise whether or not it generalises, so gate on coverage and treat
the rest as context — as a report for a person to read, not an auto-trigger.""")


if __name__ == "__main__":
    asyncio.run(main())
