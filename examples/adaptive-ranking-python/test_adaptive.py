"""Model-free test of the adaptive-ranking wiring (the Python mirror of
examples/adaptive-ranking-ts/test/adaptive.test.ts). No API key, no network — it
exercises the whole loop: BM25 ranks the decoy first, learning promotes the real
tool, `rev` advances and survives the wire form, and a reload keeps the learning.
"""

from __future__ import annotations

from ratel_ai import IntentGraph

from tools import SESSION, build_catalog, learn, top_ids

QUERY = "why is the build broken"


async def test_learning_beats_bm25_and_survives_reload() -> None:
    catalog = await build_catalog()
    # BM25 alone is confidently wrong: docker_build wins on the token "build".
    assert top_ids(catalog, QUERY)[0] == "docker_build"

    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)
    assert graph.rev == 0
    for query, invoked in SESSION:
        await learn(catalog, query, invoked)

    # After learning, the tool people actually invoke climbs above the decoy.
    after = top_ids(catalog, QUERY)
    assert after.index("gh_run_list") < after.index("docker_build"), after
    assert graph.rev > 0

    # Persistence round-trips: reload into a fresh catalog, learning survives.
    restored = IntentGraph.from_json(graph.to_json())
    assert restored.rev == graph.rev
    fresh = await build_catalog()
    fresh.experimental_enable_adaptive_ranking(restored)
    after_reload = top_ids(fresh, QUERY)
    assert after_reload.index("gh_run_list") < after_reload.index("docker_build"), after_reload
