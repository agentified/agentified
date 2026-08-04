// Watching a graph mature while capture is still running.
//
// `experimentalInitializeIntentGraph` is a pure function of (log, policy) and
// returns a DETACHED graph — so you can rebuild from the log so far as often as
// you like, score it, and decide when it is worth switching on. Nothing you
// serve is touched until you call `experimentalEnableAdaptiveRanking`.
//
// Run this on a cron against a growing log. Report the numbers; let a person
// read them. A threshold that auto-enables would flip on coverage that looks
// healthy while the clusters behind it are wrong.
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildCatalog, HELD_OUT, readiness } from "./tools.js";

const logPath = join(mkdtempSync(join(tmpdir(), "ratel-monitor-")), "telemetry.jsonl");

// Two intents, so the second appears partway through and coverage climbs.
const TURNS: [string, string][] = [
  ["why is the build broken", "gh_run_list"],
  ["is the build broken again", "gh_run_list"],
  ["the build broken on main", "gh_run_list"],
  ["rotate the signing key", "vault_rotate"],
];

const capture = await buildCatalog({ kind: "jsonl", sessionId: "session-1", path: logPath });
const serving = await buildCatalog();

console.log(`probing with held-out queries: ${HELD_OUT.map((q) => `"${q}"`).join(", ")}\n`);

for (const [i, [turn, invoked]] of TURNS.entries()) {
  capture.experimentalRecordBaselineQuery(turn);
  capture.recordEvent({ type: "invoke_start", tool_id: invoked, args_size_bytes: 0 });

  // Rebuild from the log SO FAR. Detached, so this is safe mid-capture.
  const graph = await serving.experimentalInitializeIntentGraph(readFileSync(logPath, "utf8"), {
    origins: "baseline",
    provenance: "seeded",
  });
  const r = await readiness(graph, serving);

  const note =
    r.mature > 0 && i > 0 && r.coverage.hits === r.coverage.probed
      ? "   <- every probe covered"
      : r.mature === 1 && i === 2
        ? "   <- support hit 3: full arm weight"
        : "";
  console.log(
    `turn ${i + 1}: clusters=${r.clusters} mature=${r.mature} obs=${r.observations} ` +
      `seeded=${r.seeded} ghosts=${r.ghosts.length} ` +
      `coverage=${r.coverage.hits}/${r.coverage.probed}${note}`,
  );
}

// What catalog drift looks like. A graph outlives the catalog it was built
// against — tools get renamed, removed, or (in the TS SDK) registered as
// passthroughs that never enter the retrieval index. Those edges are dropped
// silently at rank time, so a graph can look populated and boost nothing.
const finalGraph = await serving.experimentalInitializeIntentGraph(
  readFileSync(logPath, "utf8"),
  { origins: "baseline", provenance: "seeded" },
);
const shrunk = await buildCatalog();          // pretend gh_run_list was removed
const drifted = await readiness(finalGraph, {
  ...shrunk,
  has: (id: string) => id !== "gh_run_list" && shrunk.has(id),
} as typeof shrunk);
console.log(
  `\nif gh_run_list left the catalog: ghosts=${drifted.ghosts.length} ` +
    `(${drifted.ghosts.join(", ")}) — those edges rank nothing`,
);

console.log(`
Reading the columns:
  clusters   distinct intents found so far
  mature     clusters at support >= 3 — below that the boost is ramped down
  obs        confirmed observations; seeded says how many came from capture
  ghosts     edges naming tools the serving catalog no longer has (want 0)
  coverage   held-out queries that matched a cluster — the generalisation check

Only coverage is measured against queries the graph has not seen. Clusters and
support both rise whether or not it generalises, so gate on coverage and treat
the rest as context.`);
