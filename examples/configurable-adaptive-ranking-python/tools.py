from __future__ import annotations
from ratel_ai import ExecutableTool, ToolCatalog, TraceSinkConfig


async def build_catalog(trace: TraceSinkConfig | None = None) -> ToolCatalog:

    catalog = ToolCatalog(trace=trace) if trace else ToolCatalog()
    await catalog.register(
        [
            ExecutableTool(
                id="docker_build",
                name="docker_build",
                description="Build a Docker image from a Dockerfile",
                execute=lambda _args: "built",
            ),
            ExecutableTool(
                id="gh_run_list",
                name="gh_run_list",
                description="List CI workflow runs and whether the build passed",
                execute=lambda _args: "listed",
            ),
            ExecutableTool(
                id="vault_rotate",
                name="vault_rotate",
                description="Rotate a signing key in the vault",
                execute=lambda _args: "rotated",
            ),
            ExecutableTool(
                id="read_file",
                name="read_file",
                description="Read a file from disk",
                execute=lambda _args: "read",
            ),
        ]
    )
    return catalog


BASELINE_TURNS = [
    ("why is the build broken", "gh_run_list"),
    ("is the build broken again", "gh_run_list"),
    ("the build broken on main", "gh_run_list"),
    ("rotate the signing key", "vault_rotate"),
    ("the build is broken", "gh_run_list"),
    ("rotate signing key now", "vault_rotate"),
    ("read a file from disk", "read_file"),
    ("rotate the signing key again", "vault_rotate"),
    ("read the file from disk", "read_file"),
    ("build broken after merge", "gh_run_list"),
]



def top_ids(catalog: ToolCatalog, query: str) -> list[str]:
    return [hit.tool_id for hit in catalog.search(query, 3)]


HELD_OUT = [
    "is the build broken today",
    "rotate the key",
    "read the file",
    "is CI green on my branch",
]