"""Upstream MCP ingestion — the Python mirror of `src/sdk/ts/src/mcp.ts`.

`register_mcp_server` lists every paginated upstream MCP `tools/list` page (no live
refresh), registers each tool into a `ToolCatalog` under a namespaced id
(`<server>__<tool>`), and wires each executor to the upstream `call_tool`. It emits
the same `upstream_*` trace **event types** the TS SDK emits (ADR-0007).

The `mcp` package is an optional dependency (`pip install ratel-ai[mcp]`) and is
imported lazily, so the base SDK installs without it.

Divergence from the TS SDK, by design: the Python MCP client is built around
async context managers, so the **caller owns the `ClientSession` lifecycle**
(set up the transport + session with `async with`) and passes the initialized
session in. A consequence of that ownership split: unlike the TS SDK — which reads
`server_instructions` from the live handshake and derives the `transport` label
from the transport class — those two fields are *caller-supplied* here
(`transport_label`, `instructions`). They default to `"unknown"` / `None`; pass the
values from your `await session.initialize()` result to make the emitted
`upstream_register` payload byte-identical to the TS SDK's.

The returned handle's `close()` exists for symmetry and defaults to a no-op; pass
`on_close` if you want the handle to tear something down.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .catalog import ExecutableTool, ToolCatalog
from .telemetry import RuntimeEventProjection, trace_upstream_register

_MCP_LIST_MAX_PAGES = 64

McpToolsListErrorCode = Literal["RepeatedCursor", "PaginationExceeded"]


class McpToolsListError(Exception):
    """Raised when MCP `tools/list` pagination is invalid or exceeds the page cap."""

    def __init__(self, message: str, code: McpToolsListErrorCode) -> None:
        """Set the error message and pagination failure code."""
        super().__init__(message)
        self.code = code
        self.name = "McpToolsListError"


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "register_mcp_server requires the 'mcp' package. "
            "Install it with: pip install 'ratel-ai[mcp]'"
        ) from err


def _next_cursor(result: Any) -> str | None:
    nc = getattr(result, "nextCursor", None)
    if nc is None:
        nc = getattr(result, "next_cursor", None)
    return nc


def _paginated_list_params(cursor: str) -> Any:
    for module_name in ("mcp.types", "mcp_types"):
        try:
            types_mod = __import__(module_name, fromlist=["PaginatedRequestParams"])
            params_cls = types_mod.PaginatedRequestParams
            return params_cls(cursor=cursor)
        except (ImportError, AttributeError):
            continue
    from types import SimpleNamespace

    return SimpleNamespace(cursor=cursor)


async def _fetch_tools_page(session: Any, cursor: str | None) -> Any:
    if cursor is None:
        return await session.list_tools()
    try:
        return await session.list_tools(cursor)
    except TypeError:
        return await session.list_tools(params=_paginated_list_params(cursor))


async def _list_all_mcp_tools(session: Any) -> list[Any]:
    tools: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _ in range(_MCP_LIST_MAX_PAGES):
        result = await _fetch_tools_page(session, cursor)
        tools.extend(result.tools)
        next_cursor = _next_cursor(result)
        # MCP: only absent nextCursor ends pagination ("" is a valid cursor).
        if next_cursor is None:
            return tools
        if next_cursor in seen_cursors:
            raise McpToolsListError(
                "MCP tools/list returned a repeated nextCursor",
                "RepeatedCursor",
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise McpToolsListError(
        f"MCP tools/list exceeded {_MCP_LIST_MAX_PAGES} pages",
        "PaginationExceeded",
    )


def _build_registered_mcp_tools(
    catalog: ToolCatalog,
    session: Any,
    name: str,
    tools: list[Any],
) -> tuple[list[str], list[ExecutableTool]]:
    tool_ids: list[str] = []
    registered: list[ExecutableTool] = []
    for tool in tools:
        tool_id = f"{name}__{tool.name}"
        registered.append(
            ExecutableTool(
                id=tool_id,
                name=tool.name,
                description=getattr(tool, "description", None) or "",
                input_schema=getattr(tool, "inputSchema", None) or {},
                output_schema=getattr(tool, "outputSchema", None) or {"type": "object"},
                execute=_make_executor(catalog, session, name, tool_id, tool.name),
            )
        )
        tool_ids.append(tool_id)
    return tool_ids, registered


@dataclass
class McpServerHandle:
    """What `register_mcp_server` returns: the registration's outcome.

    Attributes:
        tool_ids: namespaced `<server>__<tool>` ids in upstream list order (all
            pages). Duplicate tool names may appear more than once;
            `ToolCatalog.register` keeps the last row per id.
        server_instructions: the `instructions` value passed to
            `register_mcp_server` (the caller reads it from its own
            `initialize` result), or `None`.
        close: async teardown — the `on_close` passed to
            `register_mcp_server`, or a no-op. The session itself stays
            caller-owned either way.
    """

    tool_ids: list[str]
    server_instructions: str | None
    close: Callable[[], Awaitable[None]]


async def _noop_close() -> None:
    return None


async def register_mcp_server(
    catalog: ToolCatalog,
    *,
    name: str,
    session: Any,
    transport_label: str = "unknown",
    instructions: str | None = None,
    on_close: Callable[[], Awaitable[None]] | None = None,
) -> McpServerHandle:
    """Ingest an initialized MCP `ClientSession` into the catalog.

    Args:
        catalog: the catalog to register the upstream tools into.
        name: namespace prefix for tool ids (`<name>__<tool>`).
        session: an initialized `mcp.ClientSession` owned by the caller.
        transport_label: recorded on the `upstream_register` trace event.
        instructions: the upstream's server instructions (from `initialize`), if any.
        on_close: optional async teardown invoked by the handle's `close()`.

    Returns:
        An `McpServerHandle` with the registered tool ids.

    Raises:
        ImportError: if the optional `mcp` package is not installed
            (`pip install 'ratel-ai[mcp]'`).
        McpToolsListError: if `tools/list` repeats a cursor or exceeds the page cap.
    """
    _require_mcp()

    # The whole registration (list + ingest) is one `ratel.upstream.register` span;
    # per-tool invocations later get their own `execute_tool` spans (ADR-0007).
    async def _run(
        report_tool_count: Callable[[int], None],
        projection: RuntimeEventProjection,
    ) -> McpServerHandle:
        tools = await _list_all_mcp_tools(session)
        report_tool_count(len(tools))
        catalog.record_event(
            {
                "type": "upstream_register",
                "server": name,
                "transport": transport_label,
                "tool_count": len(tools),
            },
            projection,
        )

        tool_ids, registered = _build_registered_mcp_tools(catalog, session, name, tools)
        await catalog.register(registered)

        return McpServerHandle(
            tool_ids=tool_ids,
            server_instructions=instructions,
            close=on_close or _noop_close,
        )

    return await trace_upstream_register(name, transport_label, _run)


def _make_executor(
    catalog: ToolCatalog,
    session: Any,
    server: str,
    tool_id: str,
    tool_name: str,
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    async def execute(args: dict[str, Any]) -> Any:
        started_at = time.monotonic()
        try:
            result = await session.call_tool(tool_name, args)
            catalog.record_event(
                {
                    "type": "upstream_invoke",
                    "server": server,
                    "tool_id": tool_id,
                    "took_ms": int((time.monotonic() - started_at) * 1000),
                }
            )
            return result
        except Exception as err:
            catalog.record_event(
                {
                    "type": "upstream_error",
                    "server": server,
                    "tool_id": tool_id,
                    "error": str(err),
                }
            )
            raise

    return execute
