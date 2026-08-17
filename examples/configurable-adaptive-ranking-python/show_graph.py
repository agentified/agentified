"""Pretty-print an intent graph.

    uv run show_graph.py [path]          # defaults to ./intent_graph.json

Reads the `protocol/v1` wire form, so it renders a graph written by either the
Python or the TypeScript demo. Centroids are summarized rather than printed —
384 floats per cluster is what makes the raw JSON unreadable.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORT_FULL = 3  # core's ramp: arm weight is capped once support reaches this

DIM = "\x1b[2m"
BOLD = "\x1b[1m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
OFF = "\x1b[0m"


def ts(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def bar(fraction: float, width: int = 12) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "·" * (width - filled)


def show(graph: dict[str, Any], source: str = "") -> None:
    """Print a graph already parsed from its wire form."""
    intents = graph.get("intents", [])
    print()
    print(f"{BOLD}intent graph{OFF}  {DIM}{source}{OFF}")
    print(
        f"  schema v{graph.get('v')}   rev {graph.get('rev', 0)}   "
        f"clusters {len(intents)}   built {ts(graph.get('built_from_ts'))}"
    )
    if graph.get("model"):
        # The fingerprint is a long pinned identity (repo, revision, pooling,
        # query prefix). Show the repo; the rest only matters on a mismatch.
        repo = next(
            (p.split("=", 1)[1] for p in graph["model"].split("|") if p.startswith("repo=")),
            graph["model"],
        )
        print(f"  model  {DIM}{repo.split(':', 1)[-1]}{OFF}")
    if not intents:
        print(f"\n  {DIM}(cold — nothing learned yet){OFF}\n")
        return

    for intent in intents:
        support = intent.get("support", 0)
        seeded = intent.get("seeded_support", 0)
        tools = intent.get("tools", {})
        skills = intent.get("skills", {})
        members = intent.get("members", [])
        centroid = intent.get("centroid")

        print()
        print(f"{CYAN}┌ {intent.get('id')}{OFF}  {BOLD}{intent.get('label')}{OFF}")
        ramp = min(1.0, support / SUPPORT_FULL)
        note = "full weight" if ramp >= 1.0 else f"{ramp:.0%} of full weight"
        origin = f", {seeded} from a capture" if seeded else ""
        print(
            f"{CYAN}│{OFF} support   {GREEN}{bar(ramp)}{OFF} {support}  "
            f"{DIM}({note}{origin}){OFF}"
        )

        if intent.get("terms"):
            print(f"{CYAN}│{OFF} terms     {DIM}{', '.join(intent['terms'])}{OFF}")

        edges = {**tools, **skills}
        if edges:
            top = max(edges.values())
            print(f"{CYAN}│{OFF} edges")
            for name, weight in sorted(edges.items(), key=lambda kv: -kv[1]):
                kind = "skill" if name in skills else "tool "
                print(
                    f"{CYAN}│{OFF}   {YELLOW}{bar(weight / top, 8)}{OFF} "
                    f"{weight:>5.1f}  {DIM}{kind}{OFF} {name}"
                )

        plural = "query" if len(members) == 1 else "queries"
        print(f"{CYAN}│{OFF} members   {DIM}({len(members)} {plural}){OFF}")
        for member in members:
            marker = "*" if member == intent.get("label") else " "
            print(f"{CYAN}│{OFF}   {marker} {member}")

        tail = f"{len(centroid)} dims" if centroid else f"{DIM}none — lexical cluster{OFF}"
        print(f"{CYAN}│{OFF} centroid  {tail}")
        print(f"{CYAN}└{OFF} last seen {ts(intent.get('last_ts'))}")

    print(f"\n{DIM}* = cluster label (the most central member){OFF}")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "intent_graph.json")
    if not path.exists():
        sys.exit(f"no graph at {path} — run the demo first")
    show(json.loads(path.read_text()), str(path))


if __name__ == "__main__":
    main()
