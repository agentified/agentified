"""Catalog and baseline turns for the seeding demo.

The Python mirror of ``examples/configurable-adaptive-ranking-ts/src/tools.ts``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ratel_ai import ExecutableTool, IntentGraph, ToolCatalog, TraceSinkConfig


async def build_catalog(trace: TraceSinkConfig | None = None) -> ToolCatalog:
    """A catalog where lexical retrieval is confidently wrong.

    "why is the build broken" hits ``docker_build`` on the token *build*, while
    the tool people actually reach for is ``gh_run_list``. That gap is what
    usage history closes.
    """
    catalog = ToolCatalog(trace=trace) if trace else ToolCatalog()
    await catalog.register(
        [
            ExecutableTool(
                id="docker_build",
                name="docker_build",
                description="Build a Docker image from a Dockerfile",
                execute=lambda _args: "built",
            ),
            ExecutableTool(
                id="gh_run_list",
                name="gh_run_list",
                description="List CI workflow runs and whether the build passed",
                execute=lambda _args: "listed",
            ),
            ExecutableTool(
                id="vault_rotate",
                name="vault_rotate",
                description="Rotate a signing key in the vault",
                execute=lambda _args: "rotated",
            ),
            ExecutableTool(
                id="read_file",
                name="read_file",
                description="Read a file from disk",
                execute=lambda _args: "read",
            ),
        ]
    )
    return catalog


BASELINE_TURNS = [
    {"turn": "why is the build broken", "invoked": "gh_run_list", "ok": True},
    {"turn": "is the build broken again", "invoked": "gh_run_list", "ok": True},
    {"turn": "the build broken on main", "invoked": "gh_run_list", "ok": True},
    {"turn": "why is the build broken", "invoked": "docker_build", "ok": False},
]
"""What the customer's agent did on its own, before Ratel ranked anything.

``ok`` is their success signal — an eval verdict, a thumbs-up, a completed
workflow. Only successful turns are seeded, which is the main defence against
teaching the graph a mistake.
"""


def top_ids(catalog: ToolCatalog, query: str) -> list[str]:
    return [hit.tool_id for hit in catalog.search(query, 3)]


HELD_OUT = ["why is the build broken", "rotate the signing key"]
"""Queries the graph has never been trained on — the coverage probe."""

SUPPORT_FULL = 3
"""Ratel's full-confidence threshold (``SUPPORT_FULL`` in core).

Not yet exposed by the SDK — see the README's "rough edges".
"""


@dataclass(frozen=True)
class Readiness:
    """What a maturity check reports about a candidate graph."""

    clusters: int
    mature: int
    """Clusters whose arm carries FULL weight. Below this the boost is ramped
    down proportionally, so a graph of support-1 clusters barely moves ranking."""
    observations: int
    seeded: int
    ghosts: list[str]
    """Edges naming ids the serving catalog no longer defines. These are dropped
    silently at rank time, so a non-zero count means the graph looks populated
    but will boost less than it appears to — or nothing at all."""
    coverage_hits: int
    coverage_probed: int
    """Held-out queries that matched some cluster, over the number probed. The
    only metric measured against queries the graph has NOT seen: cluster count
    and support both rise whether or not it generalises."""


async def readiness(
    graph: IntentGraph,
    serving: ToolCatalog,
    held_out: Sequence[str] = tuple(HELD_OUT),
    known: Callable[[str], bool] | None = None,
) -> Readiness:
    """Score a candidate graph without attaching it to anything you serve.

    ``serving`` supplies the tool ids to check edges against (override with
    ``known`` to simulate drift); the coverage probe runs on a throwaway catalog
    so the graph under test never touches live ranking.
    """
    intents = json.loads(graph.to_json())["intents"]
    defines = known or serving.has

    probe = await build_catalog(TraceSinkConfig(kind="memory", session_id="readiness-probe"))
    probe.experimental_enable_adaptive_ranking(graph)
    for query in held_out:
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
