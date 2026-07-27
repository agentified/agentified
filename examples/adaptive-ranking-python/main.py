"""Adaptive usage ranking, end to end — no model or API key required.

    uv run main.py

BM25 alone ranks ``docker_build`` first for a build question; after learning
from real invocations ``gh_run_list`` climbs, and the learning survives a
save/reload through the graph's JSON wire form. ``rev`` is the primitive that
tells you when to persist. The TypeScript mirror is
``examples/adaptive-ranking-ts/src/index.ts``.
"""

from __future__ import annotations

import asyncio

from ratel_ai import IntentGraph

from tools import SESSION, build_catalog, learn, top_ids

QUERY = "why is the build broken"


async def main() -> None:
    # 1. Cold catalog: BM25 alone is confidently wrong (docker_build wins on "build").
    catalog = await build_catalog()
    print(f'query: "{QUERY}"')
    print(f"  before learning : {' > '.join(top_ids(catalog, QUERY))}")

    # 2. Attach a graph and learn from the session's search -> invoke pairs.
    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)
    for query, invoked in SESSION:
        await learn(catalog, query, invoked)
    print(f"  after learning  : {' > '.join(top_ids(catalog, QUERY))}   (rev={graph.rev})")

    # 3. Persist. The graph lives in memory, so `to_json` is how learning outlives
    #    the process — write these bytes wherever you keep state (file, DB, blob).
    saved = graph.to_json()
    saved_rev = graph.rev

    # 4. Reload into a fresh catalog — a restart keeps what earlier runs discovered.
    restored_catalog = await build_catalog()
    restored = IntentGraph.from_json(saved)
    restored_catalog.experimental_enable_adaptive_ranking(restored)
    print(f"  after reload    : {' > '.join(top_ids(restored_catalog, QUERY))}   (rev={restored.rev})")

    # 5. `rev` drives save-when-changed: learn once more, then persist only because
    #    the counter moved. Had nothing changed, you would skip the write entirely.
    await learn(restored_catalog, "why did CI fail", "gh_run_list")
    if restored.rev != saved_rev:
        # In a real app: save(restored.to_json()). Here we just show the decision.
        print(f"\nrev {saved_rev} -> {restored.rev}: changed, so persist.")
    else:
        print(f"\nrev unchanged ({saved_rev}): skip the write.")


if __name__ == "__main__":
    asyncio.run(main())
