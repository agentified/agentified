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
import type { IntentGraph, ToolCatalog } from "@ratel-ai/sdk";
import { BASELINE_TURNS, buildCatalog, HELD_OUT, topIds } from "./tools.js";

const QUERY = "why is the build broken";
const logPath = join(mkdtempSync(join(tmpdir(), "ratel-baseline-")), "telemetry.jsonl");

/** Ratel's full-confidence threshold (`SUPPORT_FULL` in core). Below it the arm's
 *  weight is ramped down proportionally. Not yet exposed by the SDK — see the
 *  README's "rough edges". */
const SUPPORT_FULL = 3;

interface Readiness {
  clusters: number;
  mature: number;
  observations: number;
  seeded: number;
  ghosts: string[];
  coverage: { hits: number; probed: number };
}

/**
 * Score a candidate graph without attaching it to anything you serve.
 *
 * The coverage probe runs on a throwaway catalog, so the graph under test never
 * touches live ranking. `known` overrides which tool ids count as defined —
 * used below to simulate catalog drift.
 */
async function readiness(
  graph: IntentGraph,
  serving: ToolCatalog,
  known?: (id: string) => boolean,
): Promise<Readiness> {
  const intents: { support: number; seeded_support?: number; tools: Record<string, number> }[] =
    JSON.parse(graph.toJson()).intents;
  const defines = known ?? ((id: string) => serving.has(id));

  const probe = await buildCatalog({ kind: "memory", sessionId: "probe" });
  probe.experimentalEnableAdaptiveRanking(graph);
  for (const query of HELD_OUT) probe.search(query, 5);
  const boosts = probe
    .drainTraceEvents()
    .filter((e): e is { type: string; intent: string | null } => (e as any).type === "usage_boost");

  return {
    clusters: graph.clusterCount,
    mature: intents.filter((it) => it.support >= SUPPORT_FULL).length,
    observations: intents.reduce((n, it) => n + it.support, 0),
    seeded: intents.reduce((n, it) => n + (it.seeded_support ?? 0), 0),
    ghosts: intents.flatMap((it) => Object.keys(it.tools).filter((id) => !defines(id))),
    coverage: { hits: boosts.filter((e) => e.intent !== null).length, probed: boosts.length },
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
console.log(
  `\nA. collecting — scoring against held-out queries: ${HELD_OUT.map((q) => `"${q}"`).join(", ")}\n`,
);

for (const [i, { turn, invoked, ok }] of BASELINE_TURNS.entries()) {
  // The quality gate. Emission is per turn and opt-in, so a turn you would not
  // want the graph to learn from simply never enters it.
  if (!ok) {
    console.log(`  turn ${i + 1}  "${turn}" -> ${invoked}   SKIPPED (unsuccessful)`);
    continue;
  }

  // The query first: invocations attribute to the session's most recent one.
  capture.experimentalRecordBaselineQuery(turn);
  capture.recordEvent({ type: "invoke_start", tool_id: invoked, args_size_bytes: 0 });

  const soFar = await serving.experimentalInitializeIntentGraph(readFileSync(logPath, "utf8"), {
    origins: "baseline",
    provenance: "seeded",
  });
  const r = await readiness(soFar, serving);
  const note =
    r.mature === 1 && r.clusters === 1 && r.observations === SUPPORT_FULL
      ? "   <- support hit 3: full arm weight"
      : r.coverage.hits === r.coverage.probed
        ? "   <- every probe covered"
        : "";
  console.log(
    `  turn ${i + 1}  clusters=${r.clusters} mature=${r.mature} obs=${r.observations} ` +
      `seeded=${r.seeded} ghosts=${r.ghosts.length} ` +
      `coverage=${r.coverage.hits}/${r.coverage.probed}${note}`,
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
  console.log(`    observations  : ${intent.support} (${intent.seeded_support ?? 0} seeded)`);
  console.log(`    invoked       : ${edges}`);
  console.log(`    phrasings     : ${intent.members.length}`);
}

// What catalog drift looks like. A graph outlives the catalog it was built
// against — tools get renamed or removed. Those edges are dropped silently at
// rank time, so a graph can look populated and boost nothing.
const drifted = await readiness(graph, serving, (id) => id !== "gh_run_list" && serving.has(id));
console.log(
  `\n  if gh_run_list left the catalog: ghosts=${drifted.ghosts.length} ` +
    `(${drifted.ghosts.join(", ")}) — those edges would rank nothing`,
);
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
  clusters   distinct intents found so far
  mature     clusters at support >= 3 — below that the boost is ramped down
  obs        confirmed observations; seeded says how many came from capture
  ghosts     edges naming tools the serving catalog no longer has (want 0)
  coverage   held-out queries that matched a cluster — the generalisation check

Only coverage is measured against queries the graph has not seen. Clusters and
support both rise whether or not it generalises, so gate on coverage and treat
the rest as context — as a report for a person to read, not an auto-trigger.`);
