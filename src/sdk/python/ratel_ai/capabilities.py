"""Capability tools — the Python mirror of `src/sdk/ts/src/capabilities.ts`.

`search_capabilities_tool` and `invoke_tool_tool` give an agent a self-service
surface over a `ToolCatalog` (and an optional `SkillCatalog`): discover tools and
skills by natural-language query, then run a tool by id (or load a skill's body
via `get_skill_content`). Tool descriptions and JSON schemas here are a product
contract shown to the model — kept verbatim with the TS SDK.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Union

from .catalog import ExecutableTool, ToolCatalog
from .fact_catalog import FactCatalog
from .skill_catalog import SkillCatalog
from .telemetry import record_auth_needed

SEARCH_CAPABILITIES_ID = "search_capabilities"
"""Id (and name) of the discovery tool built by `search_capabilities_tool`."""

INVOKE_TOOL_ID = "invoke_tool"
"""Id (and name) of the invocation tool built by `invoke_tool_tool`."""

_DEFAULT_TOP_K_TOOLS = 5
_DEFAULT_TOP_K_SKILLS = 3
_DEFAULT_TOP_K_FACTS = 3
_MAX_TOP_K = 50


def _clamp_top_k(value: Any, fallback: int) -> int:
    """Clamp a model-supplied top-K to a positive int in [1, _MAX_TOP_K].

    Falls back to `fallback` for anything else (None, 0, negative, bool, float).
    Mirrors the TS SDK's `clampTopK` so the two SDKs treat the same input the same
    way (`bool` is excluded even though it subclasses `int`).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return fallback
    return min(value, _MAX_TOP_K)


# The discovery prompt shown to the model. The skills clause is only included
# when a non-empty skill catalog is wired in — otherwise the tool would advertise
# a `skills` bucket and `get_skill_content` that don't exist (always `skills: []`).
_SEARCH_INTRO = (
    "Discover capabilities beyond the ones already in your direct tool list. Call "
    "this BEFORE refusing a request, falling back to a generic capability (web "
    "fetch, shell, built-in search), or improvising a multi-step task: a "
    "purpose-built capability may be in the catalog but not pre-loaded. Pass a "
    "natural-language query describing what you want to do."
)
_RESULT_TOOLS_ONLY = " You get back a `tools` bucket (executable) — run one via invoke_tool."
_RESULT_TOOLS_AND_SKILLS = (
    " You get back two independent buckets: `tools` (run one via invoke_tool) and "
    "`skills` (reusable playbooks — load one's instructions via get_skill_content, "
    "then follow it). Skills have their own result budget, so they are never "
    "crowded out by tools."
)

_MAX_DESCRIPTION_LEN = 160


@dataclass
class UpstreamServerInfo:
    """An upstream MCP server, as advertised in the capability-tool descriptions.

    Pass a list of these to `search_capabilities_tool` (or the deprecated
    `search_tools_tool`) to append a "this catalog aggregates…" listing to the
    tool description shown to the model, and to enrich each search result's
    server group with the upstream's description and instructions.

    Attributes:
        name: the catalog namespace of the server — the `<name>` prefix of its
            `<name>__<tool>` tool ids.
        description: what the server offers; compacted to one line in listings.
        instructions: the server's usage instructions (from its MCP
            `initialize` result), surfaced on matching server groups.
        tool_count: number of tools the server contributed, shown in listings.
        needs_auth: True when the upstream rejected its boot connection with a
            401 / re-auth needed; listings append "(auth required)".
    """

    name: str
    description: str | None = None
    instructions: str | None = None
    tool_count: int | None = None
    needs_auth: bool = False


def format_upstream_line(s: UpstreamServerInfo) -> str:
    """Render one upstream server as a listing bullet for a tool description.

    Format: ``- <name> — <description> (<N> tools) (auth required)``, where the
    description is whitespace-collapsed and truncated to ~160 chars, and each
    trailing part appears only when the corresponding field is set.

    Args:
        s: the upstream server to render.

    Returns:
        A single `- `-prefixed line, ready to join into the
        "this catalog aggregates…" listing.
    """
    line = f"- {s.name}"
    if s.description:
        line += f" — {_compact_description(s.description)}"
    if s.tool_count is not None:
        line += f" ({s.tool_count} tools)"
    if s.needs_auth:
        line += " (auth required)"
    return line


def _compact_description(s: str) -> str:
    collapsed = " ".join(s.split())
    if len(collapsed) <= _MAX_DESCRIPTION_LEN:
        return collapsed
    cut = collapsed[: _MAX_DESCRIPTION_LEN - 1]
    last_space = cut.rfind(" ")
    head = cut[:last_space] if last_space > 80 else cut
    return f"{head.rstrip()}…"


def _build_search_description(has_skills: bool, upstreams: Sequence[UpstreamServerInfo]) -> str:
    base = _SEARCH_INTRO + (_RESULT_TOOLS_AND_SKILLS if has_skills else _RESULT_TOOLS_ONLY)
    if not upstreams:
        return base
    listing = "\n".join(format_upstream_line(u) for u in upstreams)
    return (
        f"{base}\n\n"
        f"This catalog aggregates tools from these upstream MCP servers:\n{listing}"
    )


def search_capabilities_tool(
    catalog: ToolCatalog,
    skill_catalog: SkillCatalog | None = None,
    fact_catalog: FactCatalog | None = None,
    *,
    upstream_servers: Sequence[UpstreamServerInfo] | None = None,
) -> ExecutableTool:
    """Build the `search_capabilities` tool: unified discovery over tools, skills, and facts.

    The returned tool ranks three independent buckets, each with its own top-K
    budget — so a relevant skill or fact is never starved out of the results by a
    large number of matching tools (and the BM25 corpora are never
    score-compared). Tools land grouped by upstream server; a matched skill's
    declared tools are pulled into the tools bucket additively (score 0), so the
    agent gets the playbook and its toolkit in one turn. Facts have no tools, so
    nothing is pulled in on their behalf — each fact carries its `body` inline
    (facts are small constant content, so there is no second round-trip to load
    it). The `skills` bucket (and the mention of `get_skill_content` in the
    description) is only advertised when a non-empty `skill_catalog` is wired in.

    Args:
        catalog: the tool catalog to search.
        skill_catalog: optional skill catalog, ranked against the same query
            in its own bucket.
        fact_catalog: ⚠️ Experimental (see `ratel_ai.experimental`) — optional fact
            catalog, ranked against the same query in its own bucket (the
            retrieval-gated tier; always-on facts inject via the grounding
            freshness gate, not here). Omit it and `facts` is `[]`.
        upstream_servers: upstream MCP servers to advertise in the tool
            description and to enrich result server groups with.

    Returns:
        An `ExecutableTool` to put in the agent's direct tool list.
    """
    upstreams = list(upstream_servers or [])
    upstream_by_name = {u.name: u for u in upstreams}
    has_skills = skill_catalog is not None and skill_catalog.size() > 0

    async def execute(input: dict[str, Any]) -> dict[str, Any]:
        query = input["query"]
        k_tools = _clamp_top_k(input.get("topKTools"), _DEFAULT_TOP_K_TOOLS)
        k_skills = _clamp_top_k(input.get("topKSkills"), _DEFAULT_TOP_K_SKILLS)
        k_facts = _clamp_top_k(input.get("topKFacts"), _DEFAULT_TOP_K_FACTS)
        started_at = time.monotonic()
        tool_hits = await catalog.search_async(query, k_tools, "agent")
        catalog.record_event(
            {
                "type": "gateway_search",
                "query": query,
                "origin": "agent",
                "top_k": k_tools,
                "hits": len(tool_hits),
                "took_ms": int((time.monotonic() - started_at) * 1000),
            }
        )
        order: list[str] = []
        groups: dict[str, dict[str, Any]] = {}
        seen_tools: set[str] = set()

        # Add a tool to its server group, deduped. `score` is the BM25 query score
        # for a real match, or 0 for a skill-declared dependency (it rode in on the
        # skill, it was never matched by the query).
        def add_tool(tool_id: str, score: float) -> None:
            if tool_id in seen_tools:
                return
            tool = catalog.get(tool_id)
            if tool is None:  # a declared id the catalog doesn't have: skip
                return
            seen_tools.add(tool_id)
            sep = tool_id.find("__")
            server_name = tool_id[:sep] if sep > 0 else tool_id
            group = groups.get(server_name)
            if group is None:
                meta = upstream_by_name.get(server_name)
                server: dict[str, Any] = {"name": server_name}
                if meta is not None and meta.description:
                    server["description"] = meta.description
                if meta is not None and meta.instructions:
                    server["instructions"] = meta.instructions
                group = {"server": server, "hits": []}
                groups[server_name] = group
                order.append(server_name)
            group["hits"].append(
                {
                    "toolId": tool_id,
                    "score": score,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
            )

        for h in tool_hits:
            add_tool(h.tool_id, h.score)

        # Skills are ranked in their own bucket against the same query (reserved
        # budget → never starved by tools).
        skills: list[dict[str, Any]] = []
        if skill_catalog is not None:
            for sh in await skill_catalog.search_async(query, k_skills, "agent"):
                sk = skill_catalog.get(sh.skill_id)
                skills.append(
                    {
                        "skillId": sh.skill_id,
                        "score": sh.score,
                        "description": _compact_description(sk.description) if sk else "",
                    }
                )

            # A matched skill's instructions name the tools they call. Pull those
            # into the tools bucket so the agent gets the playbook and its toolkit
            # in one turn — additively (score 0), beyond topKTools, deduped.
            for s in skills:
                sk = skill_catalog.get(s["skillId"])
                for tool_id in sk.tools if sk else []:
                    add_tool(tool_id, 0)

        # Facts rank in their own reserved-budget bucket, like skills. The body
        # rides inline (facts are small constant content — no get_*_content
        # round-trip), and facts have no tools, so nothing is pulled into the
        # tools bucket on their behalf.
        facts: list[dict[str, Any]] = []
        if fact_catalog is not None:
            for fh in await fact_catalog.search_async(query, k_facts, "agent"):
                fact = fact_catalog.get(fh.fact_id)
                facts.append(
                    {
                        "factId": fh.fact_id,
                        "score": fh.score,
                        "description": _compact_description(fact.description) if fact else "",
                        "body": fact.body if fact else "",
                    }
                )

        return {
            "tools": {"groups": [groups[n] for n in order]},
            "skills": skills,
            "facts": facts,
        }

    return ExecutableTool(
        id=SEARCH_CAPABILITIES_ID,
        name=SEARCH_CAPABILITIES_ID,
        description=_build_search_description(has_skills, upstreams),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "describe what you want to do"},
                "topKTools": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "max tools to return (default 5)",
                },
                "topKSkills": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "max skills to return (default 3)",
                },
                "topKFacts": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "max facts to return (default 3)",
                },
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "tools": {
                    "type": "object",
                    "properties": {
                        "groups": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "server": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "description": {"type": "string"},
                                            "instructions": {"type": "string"},
                                        },
                                        "required": ["name"],
                                    },
                                    "hits": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "toolId": {"type": "string"},
                                                "score": {"type": "number"},
                                                "description": {"type": "string"},
                                                "inputSchema": {"type": "object"},
                                            },
                                        },
                                    },
                                },
                                "required": ["server", "hits"],
                            },
                        },
                    },
                    "required": ["groups"],
                },
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skillId": {"type": "string"},
                            "score": {"type": "number"},
                            "description": {"type": "string"},
                        },
                        "required": ["skillId", "score", "description"],
                    },
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "factId": {"type": "string"},
                            "score": {"type": "number"},
                            "description": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["factId", "score", "description", "body"],
                    },
                },
            },
            "required": ["tools", "skills", "facts"],
        },
        execute=execute,
    )


OnUnauthorized = Callable[[str], Union[Awaitable[None], None]]
"""Notified when the underlying tool raises `UnauthorizedError`, with the
upstream server name inferred from the toolId. May be sync or async.
"""


def invoke_tool_tool(
    catalog: ToolCatalog,
    *,
    on_unauthorized: OnUnauthorized | None = None,
) -> ExecutableTool:
    """Build the `invoke_tool` tool: run any catalog tool by id.

    The returned tool resolves `toolId`, forwards the nested `args` object to
    `ToolCatalog.invoke`, and never raises into the host: an unknown id,
    malformed `args`, or a tool exception all come back as a structured
    `{"error": ..., "isError": True}` payload the model can recover from. A
    flattened call (arguments at the top level instead of nested under `args`)
    is tolerated. When the tool raises an `UnauthorizedError`, the payload is
    `{"error": "needs_auth", ...}` with a re-auth hint and, for namespaced
    `<server>__<tool>` ids, the upstream server name.

    Args:
        catalog: the catalog to resolve and run tool ids against.
        on_unauthorized: called with the upstream server name when a tool
            raises `UnauthorizedError`; may be sync or async. Skipped when the
            tool id has no `<server>__` namespace prefix.

    Returns:
        An `ExecutableTool` to put in the agent's direct tool list.
    """

    async def execute(input: dict[str, Any]) -> Any:
        tool_id = input.get("toolId")
        if not isinstance(tool_id, str) or not catalog.has(tool_id):
            # Missing/non-string id: structured error, not a KeyError — a malformed
            # model call must be recoverable, not crash the host (TS parity).
            catalog.record_event(
                {
                    "type": "gateway_error",
                    "tool_id": tool_id if isinstance(tool_id, str) else "",
                    "error": "unknown_tool_id",
                }
            )
            return {
                "error": (
                    f"unknown toolId: {tool_id}. "
                    "Use the catalog's search tool to discover available ids."
                ),
                "isError": True,
            }
        nested = input.get("args")
        if nested is None:
            # No `args` given — tolerate a flattened call. Drop `args` too, so an
            # explicit `args: None` can't forward a stray `args` key to the tool.
            args = {k: v for k, v in input.items() if k not in ("toolId", "args")}
        elif isinstance(nested, dict):
            args = nested
        else:
            # `args` present but not an object — reject rather than forwarding
            # stray top-level keys as arguments.
            return {
                "error": (
                    f"invalid args for {tool_id}: "
                    "`args` must be an object containing the tool's arguments."
                ),
                "isError": True,
            }
        started_at = time.monotonic()
        try:
            result = await catalog.invoke(tool_id, args)
            catalog.record_event(
                {
                    "type": "gateway_invoke",
                    "tool_id": tool_id,
                    "took_ms": int((time.monotonic() - started_at) * 1000),
                }
            )
            return result
        except Exception as err:
            if _is_unauthorized_error(err):
                upstream = _upstream_from_tool_id(tool_id)
                if upstream and on_unauthorized is not None:
                    maybe = on_unauthorized(upstream)
                    if inspect.isawaitable(maybe):
                        await maybe
                record_auth_needed(upstream)
                catalog.record_event(
                    {"type": "gateway_error", "tool_id": tool_id, "error": "needs_auth"}
                )
                payload: dict[str, Any] = {
                    "error": "needs_auth",
                    "isError": True,
                    "hint": "call the auth tool to re-authorize"
                    + (f" {upstream}" if upstream else ""),
                }
                if upstream:
                    payload["upstream"] = upstream
                return payload
            catalog.record_event({"type": "gateway_error", "tool_id": tool_id, "error": str(err)})
            return {"error": f"tool {tool_id} threw: {err}", "isError": True}

    return ExecutableTool(
        id=INVOKE_TOOL_ID,
        name=INVOKE_TOOL_ID,
        description=(
            "Invoke a tool from the catalog by its id. Use this to call tools that "
            "aren't in your direct tool list — first find one via the catalog's "
            "search tool, then run it here. Pass the tool's arguments nested "
            "under the `args` field — do NOT flatten them to the top level."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "toolId": {
                    "type": "string",
                    "description": (
                        "id of the tool to invoke "
                        "(use the catalog's search tool to find available ids)"
                    ),
                },
                "args": {
                    "type": "object",
                    "description": (
                        "arguments object matching the tool's inputSchema, "
                        "nested as a single object"
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["toolId", "args"],
        },
        output_schema={"type": "object"},
        execute=execute,
    )


def _is_unauthorized_error(err: BaseException) -> bool:
    return type(err).__name__ == "UnauthorizedError"


def _upstream_from_tool_id(tool_id: str) -> str | None:
    idx = tool_id.find("__")
    if idx <= 0:
        return None
    return tool_id[:idx]
