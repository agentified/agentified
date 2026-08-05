/**
 * Pretty-print an intent graph.
 *
 *     pnpm exec tsx src/show-graph.ts [path]   # defaults to ./intent_graph.json
 *
 * Reads the `protocol/v1` wire form, so it renders a graph written by either the
 * TypeScript or the Python demo. Centroids are summarized rather than printed —
 * 384 floats per cluster is what makes the raw JSON unreadable.
 */
import { readFileSync } from "node:fs";

/** Core's ramp: arm weight is capped once support reaches this. */
const SUPPORT_FULL = 3;

const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const CYAN = "\x1b[36m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const OFF = "\x1b[0m";

interface WireIntent {
  id: string;
  label: string;
  terms?: string[];
  members: string[];
  centroid?: number[];
  support: number;
  seeded_support?: number;
  last_ts?: number;
  tools: Record<string, number>;
  skills: Record<string, number>;
}

interface WireGraph {
  v: number;
  rev?: number;
  built_from_ts?: number;
  model?: string;
  intents: WireIntent[];
}

const ts = (ms?: number) =>
  ms ? `${new Date(ms).toISOString().slice(0, 19).replace("T", " ")}Z` : "—";

const bar = (fraction: number, width = 12) => {
  const filled = Math.round(Math.min(1, Math.max(0, fraction)) * width);
  return "█".repeat(filled) + "·".repeat(width - filled);
};

/** Print a graph already parsed from its wire form. */
export function show(graph: WireGraph, source = ""): void {
  const intents = graph.intents ?? [];
  console.log();
  console.log(`${BOLD}intent graph${OFF}  ${DIM}${source}${OFF}`);
  console.log(
    `  schema v${graph.v}   rev ${graph.rev ?? 0}   ` +
      `clusters ${intents.length}   built ${ts(graph.built_from_ts)}`,
  );
  if (graph.model) {
    // The fingerprint is a long pinned identity (repo, revision, pooling,
    // query prefix). Show the repo; the rest only matters on a mismatch.
    const repo =
      graph.model
        .split("|")
        .find((p) => p.startsWith("repo="))
        ?.slice("repo=".length) ?? graph.model;
    console.log(`  model  ${DIM}${repo.split(":").pop()}${OFF}`);
  }
  if (intents.length === 0) {
    console.log(`\n  ${DIM}(cold — nothing learned yet)${OFF}\n`);
    return;
  }

  for (const intent of intents) {
    const support = intent.support ?? 0;
    const seeded = intent.seeded_support ?? 0;
    const edges = { ...intent.tools, ...intent.skills };
    const members = intent.members ?? [];

    console.log();
    console.log(`${CYAN}┌ ${intent.id}${OFF}  ${BOLD}${intent.label}${OFF}`);
    const ramp = Math.min(1, support / SUPPORT_FULL);
    const note = ramp >= 1 ? "full weight" : `${Math.round(ramp * 100)}% of full weight`;
    const origin = seeded ? `, ${seeded} from a capture` : "";
    console.log(
      `${CYAN}│${OFF} support   ${GREEN}${bar(ramp)}${OFF} ${support}  ${DIM}(${note}${origin})${OFF}`,
    );

    if (intent.terms?.length) {
      console.log(`${CYAN}│${OFF} terms     ${DIM}${intent.terms.join(", ")}${OFF}`);
    }

    const weights = Object.values(edges);
    if (weights.length) {
      const top = Math.max(...weights);
      console.log(`${CYAN}│${OFF} edges`);
      for (const [name, weight] of Object.entries(edges).sort((a, b) => b[1] - a[1])) {
        const kind = name in intent.skills ? "skill" : "tool ";
        console.log(
          `${CYAN}│${OFF}   ${YELLOW}${bar(weight / top, 8)}${OFF} ` +
            `${weight.toFixed(1).padStart(5)}  ${DIM}${kind}${OFF} ${name}`,
        );
      }
    }

    const plural = members.length === 1 ? "query" : "queries";
    console.log(`${CYAN}│${OFF} members   ${DIM}(${members.length} ${plural})${OFF}`);
    for (const member of members) {
      console.log(`${CYAN}│${OFF}   ${member === intent.label ? "*" : " "} ${member}`);
    }

    const tail = intent.centroid
      ? `${intent.centroid.length} dims`
      : `${DIM}none — lexical cluster${OFF}`;
    console.log(`${CYAN}│${OFF} centroid  ${tail}`);
    console.log(`${CYAN}└${OFF} last seen ${ts(intent.last_ts)}`);
  }

  console.log(`\n${DIM}* = cluster label (the most central member)${OFF}`);
}

if (process.argv[1]?.endsWith("show-graph.ts")) {
  const path = process.argv[2] ?? "intent_graph.json";
  try {
    show(JSON.parse(readFileSync(path, "utf8")), path);
  } catch {
    console.error(`no graph at ${path} — run the demo first`);
    process.exit(1);
  }
}
