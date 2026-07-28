// Model-free test of the adaptive-ranking wiring (the TS analogue of
// examples/adaptive-ranking-python/test_adaptive.py). No API key, no network —
// it exercises the whole loop that the SDK-level e2e can't frame as a story:
//   - BM25 alone ranks the decoy first
//   - learning from search->invoke pairs promotes the real tool
//   - `rev` advances on learning and survives the wire form
//   - a reload into a fresh catalog keeps the learning
//
// Run: `tsx test/adaptive.test.ts` (the `example` CI job builds @ratel-ai/sdk first).
import assert from "node:assert/strict";

import { IntentGraph } from "@ratel-ai/sdk";
import { buildCatalog, learn, SESSION, topIds } from "../src/tools.js";

const QUERY = "why is the build broken";

async function main() {
  const catalog = await buildCatalog();
  // BM25 alone is confidently wrong: docker_build wins on the token "build".
  assert.equal(topIds(catalog, QUERY)[0], "docker_build", "expected BM25 to rank docker_build first");

  const graph = new IntentGraph();
  catalog.experimentalEnableAdaptiveRanking(graph);
  assert.equal(graph.rev, 0, "a fresh graph has rev 0");
  for (const { query, invoked } of SESSION) await learn(catalog, query, invoked);

  // After learning, the tool people actually invoke climbs above the decoy.
  const after = topIds(catalog, QUERY);
  assert.ok(
    after.indexOf("gh_run_list") < after.indexOf("docker_build"),
    `expected gh_run_list above docker_build, got ${after.join(" > ")}`,
  );
  assert.ok(graph.rev > 0, "learning must advance rev");

  // Persistence round-trips: reload into a fresh catalog, learning survives.
  const restored = IntentGraph.fromJson(graph.toJson());
  assert.equal(restored.rev, graph.rev, "rev must survive the wire form");
  const fresh = await buildCatalog();
  fresh.experimentalEnableAdaptiveRanking(restored);
  const afterReload = topIds(fresh, QUERY);
  assert.ok(
    afterReload.indexOf("gh_run_list") < afterReload.indexOf("docker_build"),
    `reload lost the learning: ${afterReload.join(" > ")}`,
  );

  console.log(
    `PASS (adaptive-ranking example): before=docker_build, after=[${after.join(" > ")}], rev=${graph.rev}`,
  );
}

main().catch((err) => {
  console.error("FAIL (adaptive-ranking example):", err);
  process.exit(1);
});
