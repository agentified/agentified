import asyncio

from ratel_ai import ExecutableTool, IntentGraph, ToolCatalog

TOOLS = [
    ("docker_build", "Build a Docker image from a Dockerfile"),
    ("gh_run_list", "List CI workflow runs and whether the build passed"),
]
query = "why is the build broken"


async def main():
    catalog = ToolCatalog(method="semantic")  # define the tool catalog

    await catalog.register([ExecutableTool(id=i, name=i, description=d, execute=lambda _a: "ok") for i, d in TOOLS])  # tool registration

    graph = IntentGraph()  # define the intent graph

    catalog.experimental_enable_adaptive_ranking(graph)  # attach it: learn from usage and boost with it

    hits = await catalog.search_async(query, 5, method="semantic")  # semantic search, top-5 (dense is async)

    print("rank:", hits[0].rank)    # 0-based position — order on this, not score
    print("fused:", hits[0].fused)  # True once the usage arm boosted the result

    await catalog.invoke("gh_run_list", {})  # invoke a tool: search + invoke = one observation

    print("clusters:", graph.cluster_count)  # clusters learned
    print("rev:", graph.rev)                 # write counter — persist only when it changes

    saved = graph.to_json()               # serialize the in-memory graph
    graph = IntentGraph.from_json(saved)  # reload it (invalid graphs are rejected)

    print("status:", catalog.experimental_adaptive_ranking_status)  # active | inactive | unknown | paused: model mismatch

    await catalog.experimental_rebuild_intent_graph()  # re-embed under the current model (recover after a model swap)

    catalog.experimental_enable_adaptive_ranking(graph, rebuild_on_model_change=True)  # default False; True auto-recovers on next search

    catalog.experimental_disable_adaptive_ranking()  # turn off; the graph keeps what it learned


if __name__ == "__main__":
    asyncio.run(main())
