"""Watching a graph mature while capture is still running.

    uv run monitor.py

``experimental_initialize_intent_graph`` is a pure function of (log, policy) and
returns a DETACHED graph — so you can rebuild from the log so far as often as you
like, score it, and decide when it is worth switching on. Nothing you serve is
touched until you call ``experimental_enable_adaptive_ranking``.

Run this on a cron against a growing log. Report the numbers; let a person read
them. A threshold that auto-enables would flip on coverage that looks healthy
while the clusters behind it are wrong.

The TypeScript mirror is ``examples/configurable-adaptive-ranking-ts/src/monitor.ts``.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ratel_ai import TraceSinkConfig

from tools import HELD_OUT, build_catalog, readiness

# Two intents, so the second appears partway through and coverage climbs.
TURNS = [
    ("why is the build broken", "gh_run_list"),
    ("is the build broken again", "gh_run_list"),
    ("the build broken on main", "gh_run_list"),
    ("rotate the signing key", "vault_rotate"),
]


async def main() -> None:
    log_path = Path(tempfile.mkdtemp(prefix="ratel-monitor-")) / "telemetry.jsonl"
    capture = await build_catalog(
        TraceSinkConfig(kind="jsonl", session_id="session-1", path=str(log_path))
    )
    serving = await build_catalog()

    probes = ", ".join(f'"{q}"' for q in HELD_OUT)
    print(f"probing with held-out queries: {probes}\n")

    for i, (turn, invoked) in enumerate(TURNS):
        capture.experimental_record_baseline_query(turn)
        capture.record_event(
            {"type": "invoke_start", "tool_id": invoked, "args_size_bytes": 0}
        )

        # Rebuild from the log SO FAR. Detached, so this is safe mid-capture.
        graph = await serving.experimental_initialize_intent_graph(
            log_path.read_text(), origins="baseline", provenance="seeded"
        )
        r = await readiness(graph, serving)

        note = ""
        if r.mature > 0 and i > 0 and r.coverage_hits == r.coverage_probed:
            note = "   <- every probe covered"
        elif r.mature == 1 and i == 2:
            note = "   <- support hit 3: full arm weight"
        print(
            f"turn {i + 1}: clusters={r.clusters} mature={r.mature} "
            f"obs={r.observations} seeded={r.seeded} ghosts={len(r.ghosts)} "
            f"coverage={r.coverage_hits}/{r.coverage_probed}{note}"
        )

    # What catalog drift looks like. A graph outlives the catalog it was built
    # against — tools get renamed or removed. Those edges are dropped silently at
    # rank time, so a graph can look populated and boost nothing.
    final_graph = await serving.experimental_initialize_intent_graph(
        log_path.read_text(), origins="baseline", provenance="seeded"
    )
    drifted = await readiness(
        final_graph, serving, known=lambda tool: tool != "gh_run_list" and serving.has(tool)
    )
    print(
        f"\nif gh_run_list left the catalog: ghosts={len(drifted.ghosts)} "
        f"({', '.join(drifted.ghosts)}) — those edges rank nothing"
    )

    print("""
Reading the columns:
  clusters   distinct intents found so far
  mature     clusters at support >= 3 — below that the boost is ramped down
  obs        confirmed observations; seeded says how many came from capture
  ghosts     edges naming tools the serving catalog no longer has (want 0)
  coverage   held-out queries that matched a cluster — the generalisation check

Only coverage is measured against queries the graph has not seen. Clusters and
support both rise whether or not it generalises, so gate on coverage and treat
the rest as context.""")


if __name__ == "__main__":
    asyncio.run(main())
