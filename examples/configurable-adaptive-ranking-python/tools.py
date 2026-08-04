"""Catalog and baseline turns for the seeding demo.

The Python mirror of ``examples/configurable-adaptive-ranking-ts/src/tools.ts``.
"""

from __future__ import annotations

from ratel_ai import ExecutableTool, ToolCatalog, TraceSinkConfig


async def build_catalog(trace: TraceSinkConfig | None = None) -> ToolCatalog:
    """A catalog where lexical retrieval is confidently wrong.

    "why is the build broken" hits ``docker_build`` on the token *build*, while
    the tool people actually reach for is ``gh_run_list``. That gap is what
    usage history closes.
    """
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
    {"turn": "why is the build broken", "invoked": "gh_run_list", "ok": True},
    {"turn": "is the build broken again", "invoked": "gh_run_list", "ok": True},
    {"turn": "the build broken on main", "invoked": "gh_run_list", "ok": True},
    {"turn": "why is the build broken", "invoked": "docker_build", "ok": False},
    {"turn": "rotate the signing key", "invoked": "vault_rotate", "ok": True},
]
"""What the customer's agent did on its own, before Ratel ranked anything.

``ok`` is their success signal — an eval verdict, a thumbs-up, a completed
workflow. Only successful turns are seeded, which is the main defence against
teaching the graph a mistake.
"""


def top_ids(catalog: ToolCatalog, query: str) -> list[str]:
    return [hit.tool_id for hit in catalog.search(query, 3)]


