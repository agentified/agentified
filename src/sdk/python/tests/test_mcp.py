"""Tests for MCP ingestion — mirrors `src/sdk/ts/src/mcp.test.ts`.

The upstream session is duck-typed, so the ingestion logic (id namespacing,
trace events, executor wiring) is exercised with a fake session and does not
require the optional `mcp` package. A separate test pins the helpful error when
`mcp` is absent.
"""

import importlib.util
from collections.abc import Callable
from typing import Any

import pytest

from ratel_ai import EmbedderError, ToolCatalog, TraceSinkConfig, register_mcp_server


@pytest.fixture
def skip_mcp_import_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ratel_ai.mcp._require_mcp", lambda: None)


def _cursor_from_list_tools_params(params: Any) -> str | None:
    if params is None:
        return None
    if isinstance(params, dict):
        return params.get("cursor")
    return getattr(params, "cursor", None)


class _FakeTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema
        self.outputSchema = None


class _FakeListResult:
    def __init__(self, tools, *, next_cursor: str | None = None):
        self.tools = tools
        self.next_cursor = next_cursor


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def list_tools(self, *, params=None):
        return _FakeListResult(
            [_FakeTool("create_issue", "Create a GitHub issue.", {"type": "object"})]
        )

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return {"ok": True, "name": name}


class _PaginatedFakeSession:
    def __init__(
        self,
        pages: list[dict[str, Any]]
        | Callable[[str | None], dict[str, Any] | Exception],
    ) -> None:
        self._pages = pages
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _resolve_page(self, cursor: str | None) -> dict[str, Any]:
        pages = self._pages
        if callable(pages):
            result = pages(cursor)
            if isinstance(result, Exception):
                raise result
            return result
        if cursor is None:
            if not pages:
                raise RuntimeError("paginated fake MCP has no first page")
            return pages[0]
        for i in range(len(pages) - 1):
            if pages[i].get("next_cursor") == cursor:
                return pages[i + 1]
        raise RuntimeError(f"unexpected tools/list cursor: {cursor}")

    def _tools_from_page(self, page: dict[str, Any]) -> list[_FakeTool]:
        out: list[_FakeTool] = []
        for spec in page.get("tools", []):
            if isinstance(spec, _FakeTool):
                out.append(spec)
                continue
            name = spec["name"]
            description = spec.get("description") or ""
            out.append(_FakeTool(name, description, {"type": "object"}))
            if "handler" in spec:
                self._handlers[name] = spec["handler"]
        return out

    async def list_tools(self, *, params=None) -> _FakeListResult:
        page = self._resolve_page(_cursor_from_list_tools_params(params))
        tools = self._tools_from_page(page)
        next_cursor = page.get("next_cursor")
        if next_cursor is None and "nextCursor" in page:
            next_cursor = page["nextCursor"]
        return _FakeListResult(tools, next_cursor=next_cursor)

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        self.calls.append((name, args))
        handler = self._handlers.get(name)
        if handler is not None:
            return handler(args)
        return {"ok": True, "name": name}


async def test_register_mcp_server_namespaces_and_wires(skip_mcp_import_check) -> None:
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="s"))
    session = _FakeSession()

    handle = await register_mcp_server(
        catalog,
        name="github",
        session=session,
        transport_label="memory",
        instructions="be nice",
    )

    assert handle.tool_ids == ["github__create_issue"]
    assert handle.server_instructions == "be nice"
    assert catalog.has("github__create_issue")

    register_events = [e for e in catalog.drain_trace_events() if e["type"] == "upstream_register"]
    assert register_events[0]["server"] == "github"
    assert register_events[0]["tool_count"] == 1
    assert register_events[0]["transport"] == "memory"

    result = await catalog.invoke("github__create_issue", {"title": "bug"})
    assert result == {"ok": True, "name": "create_issue"}
    assert session.calls == [("create_issue", {"title": "bug"})]

    invoke_events = [e for e in catalog.drain_trace_events() if e["type"] == "upstream_invoke"]
    assert invoke_events[0]["tool_id"] == "github__create_issue"

    await handle.close()  # default no-op close is awaitable


async def test_register_mcp_server_records_upstream_error(skip_mcp_import_check) -> None:
    class _BoomSession(_FakeSession):
        async def call_tool(self, name, args):
            raise RuntimeError("upstream down")

    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="s"))
    await register_mcp_server(catalog, name="github", session=_BoomSession())
    catalog.drain_trace_events()
    with pytest.raises(RuntimeError, match="upstream down"):
        await catalog.invoke("github__create_issue", {})
    errors = [e for e in catalog.drain_trace_events() if e["type"] == "upstream_error"]
    assert errors[0]["server"] == "github"


async def test_register_mcp_server_embeds_during_registration(skip_mcp_import_check) -> None:
    # Embedding now happens inside `catalog.register` (RAT-379/async-register),
    # which `register_mcp_server` awaits — so a broken model surfaces the error
    # right out of `register_mcp_server` itself, not from a later, separate build.
    catalog = ToolCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})

    with pytest.raises(EmbedderError, match="/missing/ratel-model"):
        await register_mcp_server(catalog, name="github", session=_FakeSession())

    # Metadata registration happens before the embedding pass inside
    # `catalog.register`, so it persists even though the embed itself failed.
    assert catalog.has("github__create_issue")


async def test_register_mcp_server_requires_mcp_when_absent() -> None:
    if importlib.util.find_spec("mcp") is not None:
        pytest.skip("mcp is installed; the absent-dependency path cannot be exercised")
    with pytest.raises(ImportError, match=r"ratel-ai\[mcp\]"):
        await register_mcp_server(ToolCatalog(), name="x", session=object())


async def test_register_mcp_server_lists_every_page_and_aggregates_tool_count(
    skip_mcp_import_check,
) -> None:
    session = _PaginatedFakeSession(
        [
            {
                "tools": [
                    {
                        "name": "page_one",
                        "description": "first page tool",
                        "handler": lambda _args: {"page": 1},
                    }
                ],
                "next_cursor": "page-2",
            },
            {
                "tools": [
                    {
                        "name": "page_two",
                        "description": "second page tool",
                        "handler": lambda _args: {"page": 2},
                    }
                ],
            },
        ]
    )
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="t"))
    handle = await register_mcp_server(catalog, name="demo", session=session)

    assert handle.tool_ids == ["demo__page_one", "demo__page_two"]
    assert catalog.has("demo__page_one")
    assert catalog.has("demo__page_two")

    hits = catalog.search("second page tool", 5)
    assert any(h.tool_id == "demo__page_two" for h in hits)

    register_events = [e for e in catalog.drain_trace_events() if e["type"] == "upstream_register"]
    assert register_events[0]["tool_count"] == 2

    result = await catalog.invoke("demo__page_two", {})
    assert result == {"page": 2}


async def test_register_mcp_server_follows_empty_tools_page(skip_mcp_import_check) -> None:
    session = _PaginatedFakeSession(
        [
            {"tools": [], "next_cursor": "after-empty"},
            {
                "tools": [
                    {
                        "name": "after_empty",
                        "description": "tool listed after an empty page",
                        "handler": lambda _args: {"ok": True},
                    }
                ],
            },
        ]
    )
    catalog = ToolCatalog()
    handle = await register_mcp_server(catalog, name="demo", session=session)

    assert handle.tool_ids == ["demo__after_empty"]
    assert catalog.has("demo__after_empty")


async def test_register_mcp_server_follows_empty_string_next_cursor(skip_mcp_import_check) -> None:
    session = _PaginatedFakeSession(
        [
            {"tools": [], "next_cursor": ""},
            {
                "tools": [
                    {
                        "name": "after_empty_cursor",
                        "description": "tool listed after empty-string cursor",
                        "handler": lambda _args: {"ok": True},
                    }
                ],
            },
        ]
    )
    catalog = ToolCatalog()
    handle = await register_mcp_server(catalog, name="demo", session=session)

    assert handle.tool_ids == ["demo__after_empty_cursor"]
    assert catalog.has("demo__after_empty_cursor")


async def test_register_mcp_server_rejects_repeated_next_cursor(skip_mcp_import_check) -> None:
    session = _PaginatedFakeSession(
        [
            {
                "tools": [
                    {
                        "name": "stuck",
                        "description": "stuck in pagination",
                        "handler": lambda _args: {},
                    }
                ],
                "next_cursor": "loop",
            },
            {
                "tools": [
                    {
                        "name": "again",
                        "description": "another page",
                        "handler": lambda _args: {},
                    }
                ],
                "next_cursor": "loop",
            },
        ]
    )
    catalog = ToolCatalog()

    with pytest.raises(Exception, match=r"repeated nextCursor"):
        await register_mcp_server(catalog, name="demo", session=session)
    assert not catalog.has("demo__stuck")


async def test_register_mcp_server_rejects_when_page_cap_exceeded(skip_mcp_import_check) -> None:
    page_num = 0

    def resolve(cursor: str | None) -> dict[str, Any]:
        nonlocal page_num
        if cursor is not None and not cursor.startswith("cap-page-"):
            raise RuntimeError(f"unexpected cursor: {cursor}")
        page_num += 1
        return {
            "tools": (
                [
                    {
                        "name": "endless",
                        "description": "never finishes",
                        "handler": lambda _args: {},
                    }
                ]
                if page_num == 1
                else []
            ),
            "next_cursor": f"cap-page-{page_num}",
        }

    session = _PaginatedFakeSession(resolve)
    catalog = ToolCatalog()

    with pytest.raises(Exception, match=r"exceeded 64 pages"):
        await register_mcp_server(catalog, name="demo", session=session)
    assert not catalog.has("demo__endless")


async def test_register_mcp_server_does_not_mutate_catalog_when_later_page_fails(
    skip_mcp_import_check,
) -> None:
    def resolve(cursor: str | None) -> dict[str, Any] | Exception:
        if cursor is None:
            return {
                "tools": [
                    {
                        "name": "first_page",
                        "description": "only on page one",
                        "handler": lambda _args: {},
                    }
                ],
                "next_cursor": "page-2",
            }
        return RuntimeError("list page two failed")

    session = _PaginatedFakeSession(resolve)
    catalog = ToolCatalog()

    with pytest.raises(RuntimeError, match="list page two failed"):
        await register_mcp_server(catalog, name="demo", session=session)
    assert not catalog.has("demo__first_page")


async def test_register_mcp_server_retries_after_list_failure_with_new_session(
    skip_mcp_import_check,
) -> None:
    catalog = ToolCatalog()

    failing = _PaginatedFakeSession(
        lambda cursor: (
            {
                "tools": [
                    {
                        "name": "first_page",
                        "description": "only on page one",
                        "handler": lambda _args: {},
                    }
                ],
                "next_cursor": "page-2",
            }
            if cursor is None
            else RuntimeError("list page two failed")
        )
    )
    with pytest.raises(RuntimeError, match="list page two failed"):
        await register_mcp_server(catalog, name="demo", session=failing)

    retry_session = _PaginatedFakeSession(
        lambda cursor: (
            {
                "tools": [
                    {
                        "name": "first_page",
                        "description": "only on page one",
                        "handler": lambda _args: {},
                    }
                ],
            }
            if cursor is None
            else RuntimeError("unexpected cursor on retry")
        )
    )
    handle = await register_mcp_server(catalog, name="demo", session=retry_session)
    assert handle.tool_ids == ["demo__first_page"]
    assert catalog.has("demo__first_page")


async def test_register_mcp_server_empty_unpaginated_list(skip_mcp_import_check) -> None:
    session = _PaginatedFakeSession([{"tools": []}])
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="memory", session_id="t"))
    handle = await register_mcp_server(catalog, name="demo", session=session)

    assert handle.tool_ids == []
    register_events = [e for e in catalog.drain_trace_events() if e["type"] == "upstream_register"]
    assert register_events[0]["tool_count"] == 0


async def test_register_mcp_server_duplicate_tool_name_last_page_wins(
    skip_mcp_import_check,
) -> None:
    session = _PaginatedFakeSession(
        [
            {
                "tools": [
                    {
                        "name": "dup",
                        "description": "first listing",
                        "handler": lambda _args: {"version": "first"},
                    }
                ],
                "next_cursor": "page-2",
            },
            {
                "tools": [
                    {
                        "name": "dup",
                        "description": "second listing wins",
                        "handler": lambda _args: {"version": "second"},
                    }
                ],
            },
        ]
    )
    catalog = ToolCatalog()
    handle = await register_mcp_server(catalog, name="demo", session=session)

    assert handle.tool_ids == ["demo__dup", "demo__dup"]
    result = await catalog.invoke("demo__dup", {})
    assert result == {"version": "second"}
