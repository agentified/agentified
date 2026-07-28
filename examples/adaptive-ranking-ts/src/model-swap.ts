// Recovering adaptive ranking after an embedding-model swap.
//
// Unlike `index.ts` (BM25, no model), this uses a SEMANTIC catalog, so it needs
// the default embedding model (bge-small) available locally; it prints a skip
// notice and exits cleanly if the model can't be loaded.
//
// Centroids are tied to the model that built them, so a graph reloaded under a
// different model can't be cosine-compared — the boost PAUSES instead of ranking
// across incompatible vector spaces. `experimentalRebuildIntentGraph()` re-embeds it under
// the current model; `{ rebuildOnModelChange: true }` does that automatically on
// the next dense search.
import { EmbedderError, IntentGraph, ToolCatalog } from "@ratel-ai/sdk";
import { SESSION, TOOLS } from "./tools.js";

const QUERY = "why is the build broken";

async function semanticCatalog(): Promise<ToolCatalog> {
  const catalog = new ToolCatalog({ method: "semantic" }); // default model: bge-small
  await catalog.register(TOOLS);
  return catalog;
}

async function learn(catalog: ToolCatalog, query: string, invoked: string): Promise<void> {
  await catalog.searchAsync(query, 5, "direct", "semantic");
  await catalog.invoke(invoked, {});
}

// Simulate a model swap: rewrite the fingerprint the graph was built under, as
// if a different (or older) embedding model had produced its centroids.
function swapModel(saved: string): string {
  const doc = JSON.parse(saved);
  doc.model = "some-other-embedding-model";
  return JSON.stringify(doc);
}

async function main(): Promise<void> {
  let catalog: ToolCatalog;
  try {
    catalog = await semanticCatalog();
  } catch (error) {
    if (error instanceof EmbedderError) {
      console.log(`skipping: this example needs the bge-small model locally (${error.message})`);
      return;
    }
    throw error;
  }

  // 1. Learn on the current model, persist, and pretend a different model made it.
  const graph = new IntentGraph();
  catalog.experimentalEnableAdaptiveRanking(graph);
  for (const { query, invoked } of SESSION) await learn(catalog, query, invoked);
  const saved = swapModel(graph.toJson());

  // 2. Reload under the current model: stored centroids no longer match → paused.
  catalog = await semanticCatalog();
  let stale = IntentGraph.fromJson(saved);
  catalog.experimentalEnableAdaptiveRanking(stale, { warnOnModelMismatch: false });
  console.log(`after a model swap  : ${catalog.experimentalAdaptiveRankingStatus.status}`);

  // 3a. Manual recovery: re-embed every cluster under the current model.
  await catalog.experimentalRebuildIntentGraph();
  console.log(`after rebuild       : ${catalog.experimentalAdaptiveRankingStatus.status}`);

  // 3b. Or opt in — the next dense search recovers for you. Recovery is lazy
  //     (enable is sync, rebuild is async), so status stays paused until then.
  catalog = await semanticCatalog();
  stale = IntentGraph.fromJson(saved);
  catalog.experimentalEnableAdaptiveRanking(stale, { warnOnModelMismatch: false, rebuildOnModelChange: true });
  console.log(`auto, before search : ${catalog.experimentalAdaptiveRankingStatus.status}`);
  await catalog.searchAsync(QUERY, 5, "direct", "semantic");
  console.log(`auto, after search  : ${catalog.experimentalAdaptiveRankingStatus.status}`);
}

await main();
