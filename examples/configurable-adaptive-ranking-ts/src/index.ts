// Configurable adaptive ranking: seed an intent graph from a baseline capture,
// then switch Ratel on. No model download, no API key.
//
// The problem this solves: adaptive ranking learns from what the agent invokes,
// but when Ratel is already ranking, what it invokes is partly Ratel's own
// doing. Capturing first — while Ratel serves nothing — gives evidence the
// ranker had no hand in, and a graph that is useful on day one instead of
// empty.
//
//   A. collect     record each turn + the tools the agent chose, to a JSONL log
//   B. initialize  build a graph from that log, offline
//   C. inspect     decide whether it is worth switching on
//   D. serve       attach it, and keep learning live from there
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BASELINE_TURNS, buildCatalog, topIds } from "./tools.js";

const QUERY = "why is the build broken";
const logPath = join(mkdtempSync(join(tmpdir(), "ratel-baseline-")), "telemetry.jsonl");

// ---------------------------------------------------------------------------
// A. Collect — the agent runs on its own full tool list; Ratel only records.
// ---------------------------------------------------------------------------
const capture = await buildCatalog({ kind: "jsonl", sessionId: "session-1", path: logPath });

console.log(`query: "${QUERY}"`);
console.log(`  cold (BM25 only) : ${topIds(capture, QUERY).join(" > ")}`);
console.log(`  ranking status   : ${capture.experimentalAdaptiveRankingStatus.status}`);

for (const { turn, invoked, ok } of BASELINE_TURNS) {
  // The quality gate. Emission is per turn and opt-in, so a turn you would not
  // want the graph to learn from simply never enters it.
  if (!ok) continue;

  // The query first: invocations attribute to the session's most recent one.
  capture.experimentalRecordBaselineQuery(turn);
  capture.recordEvent({ type: "invoke_start", tool_id: invoked, args_size_bytes: 0 });
}
console.log(`\ncaptured ${BASELINE_TURNS.filter((t) => t.ok).length} turns -> ${logPath}`);

// ---------------------------------------------------------------------------
// B. Initialize — build the graph from the log, offline.
// ---------------------------------------------------------------------------
const serving = await buildCatalog();
const graph = await serving.experimentalInitializeIntentGraph(readFileSync(logPath, "utf8"), {
  origins: "baseline", // only observed turns count, not Ratel's own searches
  provenance: "seeded", // record that this came from a capture, not live traffic
});

// ---------------------------------------------------------------------------
// C. Inspect — before switching anything on.
// ---------------------------------------------------------------------------
const inspected = JSON.parse(graph.toJson());
console.log("\nbuilt graph:");
console.log(`  clusters        : ${graph.clusterCount}`);
for (const intent of inspected.intents) {
  const edges = Object.entries(intent.tools as Record<string, number>)
    .map(([id, weight]) => `${id} x${weight}`)
    .join(", ");
  console.log(`  "${intent.label}"`);
  console.log(`    observations  : ${intent.support} (${intent.seeded_support ?? 0} seeded)`);
  console.log(`    invoked       : ${edges}`);
  console.log(`    phrasings     : ${intent.members.length}`);
}
// Still detached: building a graph never switches ranking on.
console.log(`  ranking status  : ${serving.experimentalAdaptiveRankingStatus.status}`);

// ---------------------------------------------------------------------------
// D. Serve — attach, and rank on what the agent actually did.
// ---------------------------------------------------------------------------
serving.experimentalEnableAdaptiveRanking(graph);
console.log(`\n  after seeding   : ${topIds(serving, QUERY).join(" > ")}`);
console.log(`  ranking status  : ${serving.experimentalAdaptiveRankingStatus.status}`);

// From here the live learner keeps adding to the same graph. `support` grows
// while `seeded_support` stays put, so the gap tells you how much of each
// cluster still rests on the baseline versus what live traffic has confirmed.
console.log(`\npersist with graph.toJson() — rev=${graph.rev} marks what to save.`);
