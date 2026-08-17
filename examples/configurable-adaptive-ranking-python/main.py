from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ratel_ai import IntentGraph, TraceSinkConfig

from show_graph import show
from tools import BASELINE_TURNS, HELD_OUT, build_catalog, top_ids

QUERY = "why is the build broken"
LIVE_QUERY = "is the build broken today"

SUPPORT_FULL = 3


@dataclass(frozen=True)
class Readiness:
    clusters: int
    support: int
    observations: int
    from_baseline: int
    coverage_hits: int
    coverage_probed: int



async def readiness(graph: IntentGraph, turn: str) -> Readiness:
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
    log_path = Path(tempfile.mkdtemp(prefix="ratel-baseline-")) / "trace.jsonl"

    capture = await build_catalog(
        TraceSinkConfig(kind="jsonl", session_id="session-1", path=str(log_path))
    )
    serving = await build_catalog()

    print(f'query: "{QUERY}"')
    print(f"  cold (BM25 only) : {' > '.join(top_ids(capture, QUERY))}")

    held_out = ", ".join(f'"{q}"' for q in HELD_OUT)
    print(f"\nA. collect — scoring after each turn against held-out: {held_out}\n")

    for i, (turn, invoked) in enumerate(BASELINE_TURNS, start=1):
        capture.experimental_baseline_turn(turn).invoked(invoked).record()

        graph = await serving.experimental_build_intent_graph(log_path.read_text())
        r = await readiness(graph, turn)
        support = (
            f"{r.support} (full)" if r.support >= SUPPORT_FULL else f"{r.support}/{SUPPORT_FULL}"
        )
        label = f'"{turn}"'
        print(
            f"  turn {i:>2}  {label:<30} {invoked:<13} clusters={r.clusters} "
            f"support={support:<9} "
            f"obs={r.observations:<2} from_baseline={r.from_baseline} "
            f"coverage={r.coverage_hits}/{r.coverage_probed}"
        )

    print(f"\n  log -> {log_path}")

    graph = await serving.experimental_build_intent_graph(log_path.read_text())
    print(
        f"\nB. build — {graph.cluster_count} clusters from the log, "
        "detached (ranking still off)"
    )

    print("\nC. inspect")
    show(json.loads(graph.to_json()))

    serving.experimental_enable_adaptive_ranking(graph, origins="agent")
    print(f"\nD. serve — after seeding : {' > '.join(top_ids(serving, QUERY))}")

    print("   live learning, from agent searches only")
    for origin in ("direct", "agent"):
        serving.search(LIVE_QUERY, 5, origin=origin)
        await serving.invoke("gh_run_list", {})
        intents = json.loads(graph.to_json())["intents"]
        obs = sum(it["support"] for it in intents)
        seeded = sum(it.get("seeded_support", 0) for it in intents)
        # Quoted separately: a nested f-string carrying backslash escapes needs
        # PEP 701 (3.12+), and this example declares `requires-python = ">=3.10"`.
        label = f'"{LIVE_QUERY}"'
        print(
            f"     {origin:<8} search  {label:<30} gh_run_list   "
            f"obs={obs:<3} from_baseline={seeded}"
        )


if __name__ == "__main__":
    asyncio.run(main())
