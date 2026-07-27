// Adaptive usage ranking, end to end — no model or API key required.
//
// BM25 alone ranks `docker_build` first for a build question; after learning
// from real invocations `gh_run_list` climbs, and the learning survives a
// save/reload through the graph's JSON wire form. `rev` is the primitive that
// tells you when to persist.
import { IntentGraph } from "@ratel-ai/sdk";
import { buildCatalog, learn, SESSION, topIds } from "./tools.js";

const QUERY = "why is the build broken";

// 1. Cold catalog: BM25 alone is confidently wrong (docker_build wins on "build").
const catalog = await buildCatalog();
console.log(`query: "${QUERY}"`);
console.log(`  before learning : ${topIds(catalog, QUERY).join(" > ")}`);

// 2. Attach a graph and learn from the session's search -> invoke pairs.
const graph = new IntentGraph();
catalog.experimentalEnableAdaptiveRanking(graph);
for (const { query, invoked } of SESSION) await learn(catalog, query, invoked);
console.log(`  after learning  : ${topIds(catalog, QUERY).join(" > ")}   (rev=${graph.rev})`);

// 3. Persist. The graph lives in memory, so `toJson` is how learning outlives
//    the process — write these bytes wherever you keep state (file, DB, blob).
const saved = graph.toJson();
let savedRev = graph.rev;

// 4. Reload into a fresh catalog — a restart keeps what earlier runs discovered.
const restoredCatalog = await buildCatalog();
const restored = IntentGraph.fromJson(saved);
restoredCatalog.experimentalEnableAdaptiveRanking(restored);
console.log(`  after reload    : ${topIds(restoredCatalog, QUERY).join(" > ")}   (rev=${restored.rev})`);

// 5. `rev` drives save-when-changed: learn once more, then persist only because
//    the counter moved. Had nothing changed, you would skip the write entirely.
await learn(restoredCatalog, "why did CI fail", "gh_run_list");
if (restored.rev !== savedRev) {
  // In a real app: save(restored.toJson()). Here we just show the decision.
  console.log(`\nrev ${savedRev} -> ${restored.rev}: changed, so persist.`);
  savedRev = restored.rev;
} else {
  console.log(`\nrev unchanged (${savedRev}): skip the write.`);
}
