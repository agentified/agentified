import { ToolCatalog, type TraceSinkConfig } from "@ratel-ai/sdk";

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
  { turn: "rotate the signing key", invoked: "vault_rotate", ok: true },
];

export const topIds = (catalog: ToolCatalog, query: string) =>
  catalog.search(query, 3).map((h) => h.toolId);


