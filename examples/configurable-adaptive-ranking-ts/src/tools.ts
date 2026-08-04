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
 * Every invocation becomes evidence — nothing in a trace says whether a turn
 * went well, so none is filtered. That rests on a precondition: seed from an
 * agent that already performs well. See the README's caveats for what happens
 * when it does not.
 */
export const BASELINE_TURNS: [turn: string, invoked: string][] = [
  ["why is the build broken", "gh_run_list"],
  ["is the build broken again", "gh_run_list"],
  ["the build broken on main", "gh_run_list"],
  ["rotate the signing key", "vault_rotate"],
  ["the build is broken", "gh_run_list"],
  ["rotate signing key now", "vault_rotate"],
  ["read a file from disk", "read_file"],
  ["rotate the signing key again", "vault_rotate"],
  ["read the file from disk", "read_file"],
  ["build broken after merge", "gh_run_list"],
];

export const topIds = (catalog: ToolCatalog, query: string) =>
  catalog.search(query, 3).map((h) => h.toolId);


