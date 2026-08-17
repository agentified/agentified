import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type IntentGraph, ToolCatalog } from "@ratel-ai/sdk";
import { show } from "./show-graph.js";
import { BASELINE_TURNS, buildCatalog, HELD_OUT, topIds } from "./tools.js";

const QUERY = "why is the build broken";
const LIVE_QUERY = "is the build broken today";
const logPath = join(mkdtempSync(join(tmpdir(), "ratel-baseline-")), "trace.jsonl");

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

console.log("OFFLINE MODE");

for (const [i, [turn, invoked]] of BASELINE_TURNS.entries()) {
  capture.experimentalBaselineTurn(turn).invoked(invoked).record();

  const soFar = await serving.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"));
  const r = await readiness(soFar, turn);
  const support = r.support >= SUPPORT_FULL ? `${r.support} (full)` : `${r.support}/${SUPPORT_FULL}`;
  console.log(
    `  turn ${String(i + 1).padStart(2)}  ${`"${turn}"`.padEnd(30)} ${invoked.padEnd(13)} ` +
      `clusters=${r.clusters} ` +
      `support=${support.padEnd(9)} ` +
      `obs=${String(r.observations).padEnd(2)} fromBaseline=${r.fromBaseline} ` +
      `coverage=${r.coverage.hits}/${r.coverage.probed}`,
  );
}

console.log(`\n  log -> ${logPath}`);

const graph = await serving.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"));
show(JSON.parse(graph.toJson()));

serving.experimentalEnableAdaptiveRanking(graph, { origins: "agent" });

// Snapshot what seeding alone achieved. ONLINE MODE below keeps learning into
// this same graph, so reading either the graph or `serving` after that point
// answers a different question than phase B did.
const seededOrder = topIds(serving, QUERY).join(" > ");
const seededClusters = graph.clusterCount;
console.log(`\nC. after seeding : ${seededOrder}`);

console.log("\nONLINE MODE (learning from agent searches only)");
for (const origin of ["direct", "agent"] as const) {
  serving.search(LIVE_QUERY, 5, origin);
  await serving.invoke("gh_run_list", {});
  const intents: { support: number; seeded_support?: number }[] = JSON.parse(
    graph.toJson(),
  ).intents;
  const obs = intents.reduce((n, it) => n + it.support, 0);
  const seeded = intents.reduce((n, it) => n + (it.seeded_support ?? 0), 0);
  console.log(
    `  ${origin.padEnd(8)} search  ${`"${LIVE_QUERY}"`.padEnd(30)} gh_run_list   ` +
      `obs=${String(obs).padEnd(3)} fromBaseline=${seeded}`,
  );
}

// ── E. the same capture, from a host that never sees a whole turn ────────────
//
// Everything above assumes one process holds the turn open. A per-request
// server does not: the search and the invocation that follows are different
// requests, on possibly different machines. Same ten turns, same graph, none of
// that assumption.

console.log("\nE. distributed capture (no process sees a whole turn)");

// Stands in for the durable store a real host would use — Postgres, a queue,
// whatever it already runs. Lines arrive from anywhere; order is not assumed.
const collected: string[] = [];

// Stands in for the cross-request stash: Redis, or a table with a TTL. A query
// has to outlive the request that produced it, because the invocation it
// belongs to arrives in a later one.
const pendingQuery = new Map<string, string>();

// The requests as a server actually receives them, with turns overlapping: two
// searches land before either invocation does. The stash is what absorbs that;
// nothing is recorded until a turn is whole.
type Request =
  | { kind: "search"; turnId: string; query: string }
  | { kind: "invoke"; turnId: string; toolId: string };

const requests: Request[] = [];
for (let i = 0; i < BASELINE_TURNS.length; i += 2) {
  const overlapping = BASELINE_TURNS.slice(i, i + 2).map(([query, toolId], k) => ({
    turnId: `turn-${i + k + 1}`,
    query,
    toolId,
  }));
  for (const { turnId, query } of overlapping) requests.push({ kind: "search", turnId, query });
  for (const { turnId, toolId } of overlapping) requests.push({ kind: "invoke", turnId, toolId });
}

for (const request of requests) {
  if (request.kind === "search") {
    // Nothing is recorded yet: a search with no invocation after it is not a turn.
    pendingQuery.set(request.turnId, request.query);
    continue;
  }

  const query = pendingQuery.get(request.turnId);
  if (query === undefined) {
    // The stash expired, or this invocation followed no search of ours. Drop it
    // rather than guess: a wrong pair is evidence the graph cannot tell from a
    // right one.
    continue;
  }
  pendingQuery.delete(request.turnId);

  // A fresh catalog, because a per-request server builds one per request. A
  // recorder needs no registered tools, no embedder and no graph.
  //
  // sessionId is per turn. Recording the turn whole already emits its search and
  // its invoke back to back, so a shared id pairs correctly as long as that
  // adjacency survives — replay tracks a pending query per session, in log
  // order. What a per-turn id buys is safety once lines from many processes are
  // merged into one log and their relative order is no longer guaranteed: only
  // then can one turn's search end up followed by another turn's invoke.
  const recorder = new ToolCatalog({
    trace: {
      kind: "callback",
      sessionId: request.turnId,
      onEvent: (line) => collected.push(line),
    },
  });
  recorder.experimentalRecordBaselineTurn({ query, invoked: [request.toolId] });
}

// Delivery is asynchronous — recording queues the line and returns. A real host
// flushes before its handler resolves; a script waits a tick.
await new Promise((resolve) => setTimeout(resolve, 20));

// Lines carry the same wire form the jsonl sink writes (bar the per-record
// ts and event_id, which replay ignores), so the log a host
// reassembles is the same artifact phase B read off disk.
const distributed = await serving.experimentalBuildIntentGraph(collected.join("\n"));

// Rank against the distributed graph on a catalog of its own, so the answer
// cannot be coloured by what `serving` has since learned.
const check = await buildCatalog();
check.experimentalEnableAdaptiveRanking(distributed);
const distributedOrder = topIds(check, QUERY).join(" > ");

const same = distributed.clusterCount === seededClusters && distributedOrder === seededOrder;

console.log(`  lines collected  ${collected.length} (from ${BASELINE_TURNS.length} turns)`);
console.log(`  sessions         ${new Set(collected.map((l) => JSON.parse(l).session_id)).size}`);
console.log(`  clusters         ${distributed.clusterCount}`);
console.log(`  same as phase B  ${same ? "yes" : "NO"}`);
