// Configurable adaptive ranking: seed an intent graph from a baseline capture,
// then switch Ratel on. No model download, no API key.
//
// The problem this solves: adaptive ranking learns from what the agent invokes,
// but when Ratel is already ranking, what it invokes is partly Ratel's own
// doing. Capturing first — while Ratel serves nothing — gives evidence the
// ranker had no hand in, and a graph that is useful on day one instead of
// empty.
//
//   A. collect   record each turn + the tools the agent chose, to a JSONL log,
//                scoring the graph-so-far after each one
//   B. inspect   read the finished graph before switching anything on
//   C. serve     attach it, and keep learning live from there
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { IntentGraph } from "@ratel-ai/sdk";
import { BASELINE_TURNS, buildCatalog, topIds } from "./tools.js";

const QUERY = "why is the build broken";
const logPath = join(mkdtempSync(join(tmpdir(), "ratel-baseline-")), "telemetry.jsonl");

/** Ratel's full-confidence threshold (`SUPPORT_FULL` in core). Below it the arm's
 *  weight is ramped down proportionally. Not yet exposed by the SDK — see the
 *  README's "rough edges". */
const SUPPORT_FULL = 3;

interface Readiness {
  clusters: number;
  /** Observations behind the cluster THIS turn landed in. Printed as `n/3`
   *  because 3 is where the boost reaches full strength — reporting one cluster
   *  per line, since a list of every cluster's support says nothing about which
   *  one the turn just changed. */
  support: number;
  observations: number;
  fromBaseline: number;
}

/** Score a candidate graph. Reads the graph only — nothing is attached. */
function readiness(graph: IntentGraph, turn: string): Readiness {
  const intents: { support: number; seeded_support?: number; members: string[] }[] = JSON.parse(
    graph.toJson(),
  ).intents;
  const landed = intents.find((it) => it.members.includes(turn));
  return {
    clusters: graph.clusterCount,
    support: landed?.support ?? 0,
    observations: intents.reduce((n, it) => n + it.support, 0),
    fromBaseline: intents.reduce((n, it) => n + (it.seeded_support ?? 0), 0),
  };
}

const capture = await buildCatalog({ kind: "jsonl", sessionId: "session-1", path: logPath });
const serving = await buildCatalog();

console.log(`query: "${QUERY}"`);
console.log(`  cold (BM25 only) : ${topIds(capture, QUERY).join(" > ")}`);
console.log(`  ranking status   : ${capture.experimentalAdaptiveRankingStatus.status}`);

// ---------------------------------------------------------------------------
// A. Collect — the agent runs on its own full tool list; Ratel only records.
//
//    After each turn we rebuild the graph from the log SO FAR and score it.
//    `experimentalInitializeIntentGraph` is a pure function of (log, policy)
//    returning a DETACHED graph, so polling mid-capture is safe — nothing being
//    served is touched.
// ---------------------------------------------------------------------------
console.log("\nA. collecting — Ratel records every invocation; the graph is scored after each\n");

for (const [i, [turn, invoked]] of BASELINE_TURNS.entries()) {
  // Every invocation is evidence. Nothing in a trace says whether a turn went
  // well, so none of them is filtered.
  // The query first: invocations attribute to the session's most recent.
  capture.experimentalRecordBaselineQuery(turn);
  capture.recordEvent({ type: "invoke_start", tool_id: invoked, args_size_bytes: 0 });

  const soFar = await serving.experimentalInitializeIntentGraph(readFileSync(logPath, "utf8"), {
    origins: "baseline",
    provenance: "seeded",
  });
  const r = readiness(soFar, turn);
  console.log(
    `  turn ${i + 1}  ${invoked.padEnd(13)} clusters=${r.clusters} ` +
      `support=${r.support}/${SUPPORT_FULL} ` +
      `obs=${r.observations} fromBaseline=${r.fromBaseline}`,
  );
}

console.log(`\n  log -> ${logPath}`);

// ---------------------------------------------------------------------------
// B. Inspect — the finished graph, before switching anything on.
// ---------------------------------------------------------------------------
const graph = await serving.experimentalInitializeIntentGraph(readFileSync(logPath, "utf8"), {
  origins: "baseline", // only observed turns count, not Ratel's own searches
  provenance: "seeded", // record that this came from a capture, not live traffic
});
console.log("\nB. built graph:");
for (const intent of JSON.parse(graph.toJson()).intents) {
  const edges = Object.entries(intent.tools as Record<string, number>)
    .map(([id, weight]) => `${id} x${weight}`)
    .join(", ");
  console.log(`  "${intent.label}"`);
  console.log(
    `    observations  : ${intent.support} (${intent.seeded_support ?? 0} from this capture)`,
  );
  console.log(`    invoked       : ${edges}`);
  console.log(`    phrasings     : ${intent.members.length}`);
}

// Still detached: building and scoring never switch ranking on.
console.log(`  ranking status  : ${serving.experimentalAdaptiveRankingStatus.status}`);

// ---------------------------------------------------------------------------
// C. Serve — attach, and rank on what the agent actually did.
// ---------------------------------------------------------------------------
serving.experimentalEnableAdaptiveRanking(graph);
console.log(`\nC. after seeding   : ${topIds(serving, QUERY).join(" > ")}`);
console.log(`   ranking status  : ${serving.experimentalAdaptiveRankingStatus.status}`);

// From here the live learner keeps adding to the same graph. `support` grows
// while `seeded_support` stays put, so the gap tells you how much of each
// cluster still rests on the baseline versus what live traffic has confirmed.
console.log(`\npersist with graph.toJson() — rev=${graph.rev} marks what to save.`);
console.log(`
Reading the collection columns:
  clusters       distinct intents found so far
  support        observations behind the cluster THIS turn landed in, out of
                 the 3 that reach full strength — below that the boost is
                 scaled down proportionally
  obs            confirmed observations across every cluster
  fromBaseline   how many of those came from this capture rather than live
                 traffic; after the flip it stays put while obs keeps growing

Treat these as a report for a person to read, not an auto-trigger.`);
