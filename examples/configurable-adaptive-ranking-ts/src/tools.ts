import { type IntentGraph, ToolCatalog, type TraceSinkConfig } from "@ratel-ai/sdk";

/**
 * A catalog where lexical retrieval is confidently wrong: "why is the build
 * broken" hits `docker_build` on the token *build*, while the tool people
 * actually reach for is `gh_run_list`. That gap is what usage history closes.
 */
export async function buildCatalog(trace?: TraceSinkConfig): Promise<ToolCatalog> {
  const catalog = new ToolCatalog(trace ? { trace } : {});
  await catalog.register([
    {
      id: "docker_build",
      name: "docker_build",
      description: "Build a Docker image from a Dockerfile",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "built",
    },
    {
      id: "gh_run_list",
      name: "gh_run_list",
      description: "List CI workflow runs and whether the build passed",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "listed",
    },
    {
      id: "vault_rotate",
      name: "vault_rotate",
      description: "Rotate a signing key in the vault",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "rotated",
    },
    {
      id: "read_file",
      name: "read_file",
      description: "Read a file from disk",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "read",
    },
  ]);
  return catalog;
}

/**
 * What the customer's agent did on its own, before Ratel ranked anything.
 *
 * `ok` is their success signal — an eval verdict, a thumbs-up, a completed
 * workflow. Only successful turns are seeded, which is the main defence against
 * teaching the graph a mistake.
 */
export const BASELINE_TURNS = [
  { turn: "why is the build broken", invoked: "gh_run_list", ok: true },
  { turn: "is the build broken again", invoked: "gh_run_list", ok: true },
  { turn: "the build broken on main", invoked: "gh_run_list", ok: true },
  { turn: "why is the build broken", invoked: "docker_build", ok: false },
];

export const topIds = (catalog: ToolCatalog, query: string) =>
  catalog.search(query, 3).map((h) => h.toolId);

/** Queries the graph has never been trained on — the coverage probe. */
export const HELD_OUT = ["why is the build broken", "rotate the signing key"];

/** What a maturity check reports about a candidate graph. */
export interface Readiness {
  clusters: number;
  /** Clusters whose arm carries FULL weight. Below this the boost is ramped
   *  down proportionally, so a graph of support-1 clusters barely moves ranking. */
  mature: number;
  observations: number;
  seeded: number;
  /** Edges naming ids the serving catalog no longer defines. These are dropped
   *  silently at rank time, so a non-zero count means the graph looks populated
   *  but will boost less than it appears to — or nothing at all. */
  ghosts: string[];
  /** Held-out queries that matched some cluster, over the number probed. The
   *  only metric measured against queries the graph has NOT seen: cluster count
   *  and support both rise whether or not it generalises. */
  coverage: { hits: number; probed: number };
}

/** Ratel's full-confidence threshold (`SUPPORT_FULL` in core).
 *  Not yet exposed by the SDK — see the README's "rough edges". */
const SUPPORT_FULL = 3;

/**
 * Score a candidate graph without attaching it to anything you serve.
 *
 * `serving` supplies the tool ids to check edges against; the coverage probe
 * runs on a throwaway catalog so the graph under test never touches live
 * ranking.
 */
export async function readiness(
  graph: IntentGraph,
  serving: ToolCatalog,
  heldOut: readonly string[] = HELD_OUT,
): Promise<Readiness> {
  const parsed = JSON.parse(graph.toJson());
  const intents: {
    support: number;
    seeded_support?: number;
    tools: Record<string, number>;
  }[] = parsed.intents;

  const probe = await buildCatalog({ kind: "memory", sessionId: "readiness-probe" });
  probe.experimentalEnableAdaptiveRanking(graph);
  for (const query of heldOut) probe.search(query, 5);
  const boosts = probe
    .drainTraceEvents()
    .filter((e): e is { type: string; intent: string | null } => (e as any).type === "usage_boost");

  return {
    clusters: graph.clusterCount,
    mature: intents.filter((it) => it.support >= SUPPORT_FULL).length,
    observations: intents.reduce((n, it) => n + it.support, 0),
    seeded: intents.reduce((n, it) => n + (it.seeded_support ?? 0), 0),
    ghosts: intents.flatMap((it) => Object.keys(it.tools).filter((id) => !serving.has(id))),
    coverage: { hits: boosts.filter((e) => e.intent !== null).length, probed: boosts.length },
  };
}
