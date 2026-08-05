import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { IntentGraph } from "@ratel-ai/sdk";
import { show } from "./show-graph.js";
import { BASELINE_TURNS, buildCatalog, HELD_OUT, topIds } from "./tools.js";

const QUERY = "why is the build broken";
const logPath = join(mkdtempSync(join(tmpdir(), "ratel-baseline-")), "telemetry.jsonl");

const SUPPORT_FULL = 3;

interface Readiness {
  clusters: number;
  support: number;
  observations: number;
  fromBaseline: number;
  coverage: { hits: number; probed: number };
}

async function readiness(graph: IntentGraph, turn: string): Promise<Readiness> {
  const intents: { support: number; seeded_support?: number; members: string[] }[] = JSON.parse(
    graph.toJson(),
  ).intents;
  const landed = intents.find((it) => it.members.includes(turn));

  const probe = await buildCatalog({ kind: "memory", sessionId: "probe" });
  probe.experimentalEnableAdaptiveRanking(graph);
  for (const query of HELD_OUT) probe.search(query, 5);
  const boosts = probe
    .drainTraceEvents()
    // biome-ignore lint/suspicious/noExplicitAny: drained events are untyped
    .filter((e): e is { type: string; intent: string | null } => (e as any).type === "usage_boost");

  return {
    clusters: graph.clusterCount,
    support: landed?.support ?? 0,
    observations: intents.reduce((n, it) => n + it.support, 0),
    fromBaseline: intents.reduce((n, it) => n + (it.seeded_support ?? 0), 0),
    coverage: { hits: boosts.filter((e) => e.intent !== null).length, probed: boosts.length },
  };
}

const capture = await buildCatalog({ kind: "jsonl", sessionId: "session-1", path: logPath });
const serving = await buildCatalog();

console.log(`query: "${QUERY}"`);
console.log(`  cold (BM25 only) : ${topIds(capture, QUERY).join(" > ")}`);

const probes = HELD_OUT.map((q) => `"${q}"`).join(", ");
console.log(`\nA. collecting — scoring after each turn against held-out: ${probes}\n`);

for (const [i, [turn, invoked]] of BASELINE_TURNS.entries()) {
  capture.experimentalBaselineTurn(turn).invoked(invoked).record();

  const soFar = await serving.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"));
  const r = await readiness(soFar, turn);
  const support = r.support >= SUPPORT_FULL ? `${r.support} (full)` : `${r.support}/${SUPPORT_FULL}`;
  console.log(
    `  turn ${String(i + 1).padStart(2)}  ${invoked.padEnd(13)} clusters=${r.clusters} ` +
      `support=${support.padEnd(9)} ` +
      `obs=${String(r.observations).padEnd(2)} fromBaseline=${r.fromBaseline} ` +
      `coverage=${r.coverage.hits}/${r.coverage.probed}`,
  );
}

console.log(`\n  log -> ${logPath}`);

const graph = await serving.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"));
show(JSON.parse(graph.toJson()));

serving.experimentalEnableAdaptiveRanking(graph, { origins: "agent" });
console.log(`\nC. after seeding : ${topIds(serving, QUERY).join(" > ")}`);
