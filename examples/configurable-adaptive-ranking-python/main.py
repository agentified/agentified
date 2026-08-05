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


def render(graph: IntentGraph) -> str:
    """The graph's wire form, with centroids elided — they are 384 floats each."""
    wire = json.loads(graph.to_json())
    for intent in wire["intents"]:
        if "centroid" in intent:
            intent["centroid"] = f"<{len(intent['centroid'])} floats>"
    return json.dumps(wire, indent=2)


async def main() -> None:
    log_path = Path(tempfile.mkdtemp(prefix="ratel-baseline-")) / "telemetry.jsonl"

    capture = await build_catalog(
        TraceSinkConfig(kind="jsonl", session_id="session-1", path=str(log_path))
    )
    serving = await build_catalog()

    print(f'query: "{QUERY}"')
    print(f"  cold (BM25 only) : {' > '.join(top_ids(capture, QUERY))}")

    probes = ", ".join(f'"{q}"' for q in HELD_OUT)
    print(f"\nA. collecting — scoring after each turn against held-out: {probes}\n")

    for i, (turn, invoked) in enumerate(BASELINE_TURNS, start=1):

        capture.experimental_record_baseline_query(turn)
        capture.record_event(
            {"type": "invoke_start", "tool_id": invoked, "args_size_bytes": 0}
        )

        graph = await serving.experimental_build_intent_graph(log_path.read_text())
        r = await readiness(graph, turn)
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

    graph = await serving.experimental_build_intent_graph(log_path.read_text())
    print("\nB. graph\n")
    print(render(graph))

    serving.experimental_enable_adaptive_ranking(graph, origins="agent")
    print(f"\nC. after seeding : {' > '.join(top_ids(serving, QUERY))}")



if __name__ == "__main__":
    asyncio.run(main())
