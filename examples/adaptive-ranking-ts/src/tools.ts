import { ToolCatalog } from "@ratel-ai/sdk";
import type { ExecutableTool } from "@ratel-ai/sdk";

// A catalog where lexical retrieval is confidently wrong: "why is the build
// broken" scores `docker_build` top on the token *build*, but the tool people
// actually reach for is `gh_run_list`. Usage learning is what closes that gap —
// no better description could, because the mismatch is in the user's words, not
// the tool's.
export const TOOLS: ExecutableTool[] = [
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
];

export async function buildCatalog(): Promise<ToolCatalog> {
  const catalog = new ToolCatalog();
  await catalog.register(TOOLS);
  return catalog;
}

// One real session: what the user searched, and the tool they actually invoked
// afterwards. Every build question ends in `gh_run_list`, never `docker_build`
// — that is the signal the graph turns into a ranking boost.
export const SESSION: ReadonlyArray<{ query: string; invoked: string }> = [
  { query: "why is the build broken", invoked: "gh_run_list" },
  { query: "is the build broken again", invoked: "gh_run_list" },
  { query: "did the build pass on main", invoked: "gh_run_list" },
  { query: "rotate the signing key", invoked: "vault_rotate" },
];

// One confirmed observation: search (so the graph sees the query), then invoke
// what you actually wanted (the signal the graph learns from). This is exactly
// what a real agent loop does — the graph just listens in.
export async function learn(catalog: ToolCatalog, query: string, invoked: string): Promise<void> {
  catalog.search(query, 5);
  await catalog.invoke(invoked, {});
}

export const topIds = (catalog: ToolCatalog, query: string, k = 3): string[] =>
  catalog.search(query, k).map((h) => h.toolId);
