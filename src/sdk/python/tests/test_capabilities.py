"""Tests for the capability tools — mirrors `src/sdk/ts/src/capabilities.test.ts`."""

from ratel_ai import (
    INVOKE_TOOL_ID,
    SEARCH_CAPABILITIES_ID,
    ExecutableTool,
    Skill,
    SkillCatalog,
    ToolCatalog,
    TraceSinkConfig,
    UpstreamServerInfo,
    invoke_tool_tool,
    search_capabilities_tool,
)
from ratel_ai.capabilities import format_upstream_line


def _tool(tool_id: str, description: str, execute=lambda args: {}) -> ExecutableTool:
    return ExecutableTool(
        id=tool_id,
        name=tool_id.split("__")[-1],
        description=description,
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object"},
        execute=execute,
    )


def test_factories_set_ids_and_descriptions() -> None:
    catalog = ToolCatalog()
    search = search_capabilities_tool(catalog)
    invoke = invoke_tool_tool(catalog)
    assert search.id == SEARCH_CAPABILITIES_ID
    assert invoke.id == INVOKE_TOOL_ID
    assert "Discover capabilities" in search.description
    assert search.input_schema["required"] == ["query"]


async def test_search_capabilities_groups_tool_hits_by_upstream() -> None:
    catalog = ToolCatalog()
    await catalog.register(_tool("github__create_issue", "Create a GitHub issue on a repo."))
    await catalog.register(_tool("github__list_issues", "List GitHub issues on a repo."))
    await catalog.register(_tool("local_read", "Read a file from disk."))
    search = search_capabilities_tool(
        catalog,
        upstream_servers=[
            UpstreamServerInfo(name="github", description="GitHub API", instructions="be nice")
        ],
    )
    result = await search.execute({"query": "create a github issue", "topKTools": 5})
    groups = result["tools"]["groups"]
    servers = {g["server"]["name"] for g in groups}
    assert "github" in servers
    gh_group = next(g for g in groups if g["server"]["name"] == "github")
    assert gh_group["server"]["description"] == "GitHub API"
    assert gh_group["server"]["instructions"] == "be nice"
    hit = gh_group["hits"][0]
    assert set(hit) == {"toolId", "score", "description", "inputSchema"}
    # no skill catalog wired → empty skills bucket
    assert result["skills"] == []


async def test_search_capabilities_returns_skills_bucket_when_wired() -> None:
    tools = ToolCatalog()
    await tools.register(_tool("local_read", "Read a file from disk."))
    skills = SkillCatalog()
    await skills.register(
        Skill(
            id="vercel-deploy",
            name="vercel-deploy",
            description="How to deploy to Vercel: env vars, preview vs production, rollbacks.",
            tags=["vercel", "deployment"],
        )
    )
    search = search_capabilities_tool(tools, skills)
    result = await search.execute({"query": "deploy to vercel"})
    assert result["skills"][0]["skillId"] == "vercel-deploy"
    assert "Vercel" in result["skills"][0]["description"]


async def test_search_capabilities_pulls_skill_declared_tools() -> None:
    # A matched skill's declared tools ride into the tools bucket, additively
    # and deduped against query hits; an unknown declared id is skipped.
    tools = ToolCatalog()
    await tools.register(_tool("vercel__push", "Deploy the project to Vercel production."))
    await tools.register(_tool("fs__read_file", "Read a file from local disk."))
    skills = SkillCatalog()
    await skills.register(
        Skill(
            id="vercel-deploy",
            name="vercel-deploy",
            description="How to deploy to Vercel: env vars, preview vs production, rollbacks.",
            # one dep also a query hit, one not, one absent from the catalog
            tools=["vercel__push", "fs__read_file", "ghost__missing"],
        )
    )
    search = search_capabilities_tool(tools, skills)
    result = await search.execute({"query": "deploy to vercel", "topKTools": 5, "topKSkills": 3})
    tool_ids = [h["toolId"] for g in result["tools"]["groups"] for h in g["hits"]]
    assert "fs__read_file" in tool_ids  # rode in on the skill
    assert tool_ids.count("vercel__push") == 1  # query hit + dep, not doubled
    assert "ghost__missing" not in tool_ids  # absent from catalog, skipped


async def test_search_capabilities_never_starves_skills() -> None:
    # Many matching tools must not crowd the skill out of its own bucket.
    tools = ToolCatalog()
    for i in range(8):
        await tools.register(_tool(f"deploy__tool_{i}", "deploy the project to production"))
    skills = SkillCatalog()
    await skills.register(
        Skill(id="vercel-deploy", name="vercel-deploy", description="Deploy to Vercel.")
    )
    search = search_capabilities_tool(tools, skills)
    result = await search.execute(
        {"query": "deploy to production", "topKTools": 5, "topKSkills": 3}
    )
    tool_count = sum(len(g["hits"]) for g in result["tools"]["groups"])
    assert tool_count <= 5
    assert any(s["skillId"] == "vercel-deploy" for s in result["skills"])


async def test_search_capabilities_records_gateway_search_event() -> None:
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="s"))
    await catalog.register(_tool("local_read", "Read a file from disk."))
    catalog.drain_trace_events()
    search = search_capabilities_tool(catalog)
    await search.execute({"query": "read", "topKTools": 3})
    events = [e for e in catalog.drain_trace_events() if e["type"] == "gateway_search"]
    assert events and events[0]["origin"] == "agent" and events[0]["top_k"] == 3


def test_search_description_lists_upstreams() -> None:
    catalog = ToolCatalog()
    search = search_capabilities_tool(
        catalog,
        upstream_servers=[UpstreamServerInfo(name="github", description="GitHub", tool_count=12)],
    )
    assert "upstream MCP servers" in search.description
    assert "- github — GitHub (12 tools)" in search.description


def test_format_upstream_line_flags_auth() -> None:
    line = format_upstream_line(UpstreamServerInfo(name="slack", needs_auth=True))
    assert line == "- slack (auth required)"


async def test_invoke_tool_runs_a_catalog_tool() -> None:
    catalog = ToolCatalog()
    await catalog.register(
        _tool("echo", "Echo the message back.", execute=lambda args: {"echo": args["msg"]})
    )
    invoke = invoke_tool_tool(catalog)
    result = await invoke.execute({"toolId": "echo", "args": {"msg": "hello"}})
    assert result == {"echo": "hello"}


async def test_invoke_tool_unknown_id_returns_error_payload() -> None:
    catalog = ToolCatalog()
    invoke = invoke_tool_tool(catalog)
    result = await invoke.execute({"toolId": "missing", "args": {}})
    assert "unknown toolId" in result["error"]
    assert result["isError"] is True


async def test_invoke_tool_accepts_flattened_args() -> None:
    catalog = ToolCatalog()
    await catalog.register(
        _tool("echo", "Echo back.", execute=lambda args: {"echo": args.get("msg")})
    )
    invoke = invoke_tool_tool(catalog)
    # args not nested under "args" — fall back to top-level minus toolId
    result = await invoke.execute({"toolId": "echo", "msg": "flat"})
    assert result == {"echo": "flat"}


async def test_invoke_tool_rejects_non_object_args() -> None:
    catalog = ToolCatalog()
    await catalog.register(
        _tool("echo", "Echo back.", execute=lambda args: {"echo": args.get("msg")})
    )
    invoke = invoke_tool_tool(catalog)
    # `args` present but a string → reject rather than forwarding stray keys
    result = await invoke.execute({"toolId": "echo", "args": "oops", "msg": "x"})
    assert "must be an object" in result["error"]
    assert result["isError"] is True


async def test_invoke_tool_unauthorized_triggers_callback_and_needs_auth() -> None:
    class UnauthorizedError(Exception):
        pass

    def boom(args):
        raise UnauthorizedError("401")

    seen = []

    async def on_unauthorized(upstream: str) -> None:
        seen.append(upstream)

    catalog = ToolCatalog()
    await catalog.register(_tool("github__create_issue", "Create issue.", execute=boom))
    invoke = invoke_tool_tool(catalog, on_unauthorized=on_unauthorized)
    result = await invoke.execute({"toolId": "github__create_issue", "args": {}})
    assert result["error"] == "needs_auth"
    # needs_auth is a failed call (the tool didn't run) — host promotes on isError.
    assert result["isError"] is True
    assert result["upstream"] == "github"
    assert seen == ["github"]


async def test_invoke_tool_generic_error_is_reported() -> None:
    def boom(args):
        raise RuntimeError("nope")

    catalog = ToolCatalog()
    await catalog.register(_tool("flaky", "Flaky tool.", execute=boom))
    invoke = invoke_tool_tool(catalog)
    result = await invoke.execute({"toolId": "flaky", "args": {}})
    assert "threw: nope" in result["error"]
    assert result["isError"] is True


async def test_invoke_tool_explicit_none_args_is_not_forwarded() -> None:
    catalog = ToolCatalog()
    # Echo returns exactly the args dict it was invoked with.
    await catalog.register(_tool("x__echo", "Echo args.", execute=lambda args: dict(args)))
    invoke = invoke_tool_tool(catalog)
    # explicit None → {} (no leftover "args" key), not {"args": None}
    assert await invoke.execute({"toolId": "x__echo", "args": None}) == {}
    # a genuinely flattened call still passes its keys through (minus toolId/args)
    assert await invoke.execute({"toolId": "x__echo", "foo": 1, "args": None}) == {"foo": 1}


async def test_invoke_tool_missing_tool_id_returns_error_not_keyerror() -> None:
    catalog = ToolCatalog()
    invoke = invoke_tool_tool(catalog)
    # No "toolId" key at all — must be a recoverable structured error, not a KeyError.
    result = await invoke.execute({"args": {}})
    assert "unknown toolId" in result["error"]
    assert result["isError"] is True


async def test_invoke_tool_guidance_is_discovery_tool_neutral() -> None:
    # invoke_tool is shared by the deprecated search_tools and search_capabilities;
    # its guidance must name neither, or it misdirects a compat-only deployment.
    invoke = invoke_tool_tool(ToolCatalog())
    tool_id_desc = invoke.input_schema["properties"]["toolId"]["description"]
    assert "search_capabilities" not in invoke.description
    assert "search_capabilities" not in tool_id_desc
    result = await invoke.execute({"toolId": "nope", "args": {}})
    assert "search_capabilities" not in result["error"]


async def _skill_catalog() -> SkillCatalog:
    c = SkillCatalog()
    await c.register(
        Skill(id="vercel-deploy", name="vercel-deploy", description="Deploy to Vercel.")
    )
    return c


async def test_search_capabilities_clamps_non_positive_top_k() -> None:
    catalog = ToolCatalog()
    await catalog.register(_tool("a__read", "read a file from disk"))
    await catalog.register(_tool("a__send", "send an email message"))
    search = search_capabilities_tool(catalog)

    async def count(extra: dict) -> int:
        result = await search.execute({"query": "read a file or send an email", **extra})
        return sum(len(g["hits"]) for g in result["tools"]["groups"])

    baseline = await count({})  # default top-K
    # 0 / negative / bool / float all fall back to the default — never zero hits,
    # and never an unbounded set. Mirrors the TS clampTopK behaviour exactly.
    assert await count({"topKTools": 0}) == baseline
    assert await count({"topKTools": -3}) == baseline
    assert await count({"topKTools": True}) == baseline
    assert await count({"topKTools": 1.5}) == baseline
    assert await count({"topKTools": 1}) == 1


async def test_search_capabilities_description_mentions_skills_only_when_wired() -> None:
    catalog = ToolCatalog()
    await catalog.register(_tool("a__read", "read a file"))

    tools_only = search_capabilities_tool(catalog)
    assert "get_skill_content" not in tools_only.description
    assert "skill" not in tools_only.description.lower()

    with_skills = search_capabilities_tool(catalog, await _skill_catalog())
    assert "get_skill_content" in with_skills.description

    # an empty skill catalog is treated as no skills
    empty = search_capabilities_tool(catalog, SkillCatalog())
    assert "get_skill_content" not in empty.description
