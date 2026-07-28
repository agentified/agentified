"""Recovering adaptive ranking after an embedding-model swap.

    uv run model_swap.py

Unlike ``main.py`` (BM25, no model), this uses a **semantic** catalog, so it
needs the default embedding model (bge-small) available locally; it prints a
skip notice and exits cleanly if the model can't be loaded.

Centroids are tied to the model that built them, so a persisted graph loaded
under a *different* model can't be cosine-compared — the boost **pauses** instead
of ranking across incompatible vector spaces. ``experimental_rebuild_intent_graph()`` re-embeds
the graph under the current model; ``rebuild_on_model_change=True`` does that
automatically on the next dense search. The TypeScript mirror is
``examples/adaptive-ranking-ts/src/model-swap.ts``.
"""

from __future__ import annotations

import asyncio
import json

from ratel_ai import EmbedderError, IntentGraph, ToolCatalog

from tools import SESSION, TOOLS

QUERY = "why is the build broken"


async def semantic_catalog() -> ToolCatalog:
    catalog = ToolCatalog(method="semantic")  # default model: bge-small
    await catalog.register(TOOLS)
    return catalog


async def learn(catalog: ToolCatalog, query: str, invoked: str) -> None:
    await catalog.search_async(query, 5, method="semantic")
    await catalog.invoke(invoked, {})


def swap_model(saved: str) -> str:
    """Simulate a model swap: rewrite the fingerprint the graph was built under,
    as if a different (or older) embedding model had produced its centroids."""
    doc = json.loads(saved)
    doc["model"] = "some-other-embedding-model"
    return json.dumps(doc)


async def main() -> None:
    try:
        catalog = await semantic_catalog()
    except EmbedderError as exc:
        print(f"skipping: this example needs the bge-small model locally ({exc})")
        return

    # 1. Learn on the current model, then persist — and pretend a different model
    #    produced it, the state you'd reload after upgrading your embedder.
    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)
    for query, invoked in SESSION:
        await learn(catalog, query, invoked)
    saved = swap_model(graph.to_json())

    # 2. Reload under the current model: the stored centroids no longer match, so
    #    the arm pauses (base ranking is untouched) rather than boost on garbage.
    catalog = await semantic_catalog()
    stale = IntentGraph.from_json(saved)
    catalog.experimental_enable_adaptive_ranking(stale, warn_on_model_mismatch=False)
    print(f"after a model swap  : {catalog.experimental_adaptive_ranking_status}")

    # 3a. Manual recovery: re-embed every cluster under the current model.
    await catalog.experimental_rebuild_intent_graph()
    print(f"after rebuild       : {catalog.experimental_adaptive_ranking_status}")

    # 3b. Or opt in and let the next dense search recover for you. Recovery is
    #     lazy (enable is sync, rebuild is async), so status stays paused until
    #     that first search.
    catalog = await semantic_catalog()
    stale = IntentGraph.from_json(saved)
    catalog.experimental_enable_adaptive_ranking(
        stale, warn_on_model_mismatch=False, rebuild_on_model_change=True
    )
    print(f"auto, before search : {catalog.experimental_adaptive_ranking_status}")
    await catalog.search_async(QUERY, 5, method="semantic")
    print(f"auto, after search  : {catalog.experimental_adaptive_ranking_status}")


if __name__ == "__main__":
    asyncio.run(main())
