"""A demo tool catalog — the Python mirror of `examples/adaptive-ranking-ts/src/tools.ts`.

A catalog where lexical retrieval is confidently wrong: "why is the build
broken" scores ``docker_build`` top on the token *build*, but the tool people
actually reach for is ``gh_run_list``. Usage learning is what closes that gap —
no better description could, because the mismatch is in the user's words, not
the tool's.
"""

from __future__ import annotations

from ratel_ai import ExecutableTool, ToolCatalog

TOOLS: list[ExecutableTool] = [
    ExecutableTool(
        id="docker_build",
        name="docker_build",
        description="Build a Docker image from a Dockerfile",
        execute=lambda _a: "built",
    ),
    ExecutableTool(
        id="gh_run_list",
        name="gh_run_list",
        description="List CI workflow runs and whether the build passed",
        execute=lambda _a: "listed",
    ),
    ExecutableTool(
        id="vault_rotate",
        name="vault_rotate",
        description="Rotate a signing key in the vault",
        execute=lambda _a: "rotated",
    ),
    ExecutableTool(
        id="read_file",
        name="read_file",
        description="Read a file from disk",
        execute=lambda _a: "read",
    ),
]

# One real session: what the user searched, and the tool they actually invoked
# afterwards. Every build question ends in ``gh_run_list``, never
# ``docker_build`` — that is the signal the graph turns into a ranking boost.
SESSION: list[tuple[str, str]] = [
    ("why is the build broken", "gh_run_list"),
    ("is the build broken again", "gh_run_list"),
    ("did the build pass on main", "gh_run_list"),
    ("rotate the signing key", "vault_rotate"),
]


async def build_catalog() -> ToolCatalog:
    catalog = ToolCatalog()
    await catalog.register(TOOLS)
    return catalog


async def learn(catalog: ToolCatalog, query: str, invoked: str) -> None:
    """One confirmed observation: search (so the graph sees the query), then
    invoke what you actually wanted (the signal the graph learns from). This is
    exactly what a real agent loop does — the graph just listens in."""
    catalog.search(query, 5)
    await catalog.invoke(invoked, {})


def top_ids(catalog: ToolCatalog, query: str, k: int = 3) -> list[str]:
    return [h.tool_id for h in catalog.search(query, k)]
