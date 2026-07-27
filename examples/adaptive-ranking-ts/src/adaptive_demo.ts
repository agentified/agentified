import { type ExecutableTool, IntentGraph, ToolCatalog } from "@ratel-ai/sdk";

const TOOLS: [string, string][] = [
  ["docker_build", "Build a Docker image from a Dockerfile"],
  ["gh_run_list", "List CI workflow runs and whether the build passed"],
];
const query = "why is the build broken";

const catalog = new ToolCatalog({ method: "semantic" }); // define the tool catalog

await catalog.register(TOOLS.map(([id, description]): ExecutableTool => ({ id, name: id, description, inputSchema: {}, outputSchema: {}, execute: async () => "ok" }))); // tool registration

const graph = new IntentGraph(); // define the intent graph

catalog.enableAdaptiveRanking(graph); // attach it: learn from usage and boost with it

const hits = await catalog.searchAsync(query, 5, "direct", "semantic"); // semantic search, top-5 (dense is async)

console.log("rank:", hits[0].rank); // 0-based position — order on this, not score
console.log("fused:", hits[0].fused); // true once the usage arm boosted the result

await catalog.invoke("gh_run_list", {}); // invoke a tool: search + invoke = one observation

console.log("clusters:", graph.clusterCount); // clusters learned
console.log("rev:", graph.rev); // write counter — persist only when it changes

const saved = graph.toJson(); // serialize the in-memory graph
const graph2 = IntentGraph.fromJson(saved); // reload it (invalid graphs are rejected)

console.log("status:", catalog.adaptiveRankingStatus.status); // active | inactive | unknown | paused: model mismatch

await catalog.rebuildIntentGraph(); // re-embed under the current model (recover after a model swap)

catalog.enableAdaptiveRanking(graph2, { rebuildOnModelChange: true }); // default false; true auto-recovers on next search

catalog.disableAdaptiveRanking(); // turn off; the graph keeps what it learned
