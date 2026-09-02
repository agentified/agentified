"""Tests for public registry facades backed by the PyO3 extension."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from ratel_ai import SearchHit, Skill, SkillRegistry, Tool, ToolRegistry


async def _register_read_file(reg: ToolRegistry) -> None:
    await reg.register(
        "read_file",
        "read_file",
        "Read a file from local disk and return its textual contents.",
        {"properties": {"path": {"type": "string"}}},
        {"properties": {"contents": {"type": "string"}}},
    )


async def test_register_and_search_returns_hit() -> None:
    reg = ToolRegistry()
    await _register_read_file(reg)
    hits = reg.search("read a text file", 5)
    assert len(hits) >= 1
    assert isinstance(hits[0], SearchHit)
    assert hits[0].tool_id == "read_file"
    assert hits[0].score > 0


async def test_search_hit_carries_a_normalized_score() -> None:
    # `score` is on three incomparable scales, so it was never displayable.
    # Asserting the RULE, not merely that the field exists — a mirror struct
    # that passed `score` twice would satisfy a presence check.
    reg = ToolRegistry()
    await _register_read_file(reg)
    await reg.register(
        "write_file",
        "write_file",
        "Write textual contents to a file on local disk.",
        {"properties": {"path": {"type": "string"}}},
        {"properties": {"written": {"type": "boolean"}}},
    )
    hits = reg.search("read a text file", 5)
    assert len(hits) > 1
    for hit in hits:
        assert 0.0 < hit.relevance <= 1.0
        assert hit.relevance != hit.score
    # Monotone with the raw score, and the weakest hit is not pinned to zero —
    # min-max would put it there whatever it matched.
    assert [h.relevance for h in hits] == sorted(
        (h.relevance for h in hits), reverse=True
    )
    assert hits[-1].relevance > 0
    assert "relevance=" in repr(hits[0])


async def test_skill_hit_carries_a_normalized_score() -> None:
    reg = SkillRegistry()
    await reg.register(
        [
            Skill(
                id="api-docs",
                name="api-docs",
                description="Write REST api reference documentation.",
                tags=["api"],
                tools=[],
                metadata={},
                body="# api-docs",
            ),
            Skill(
                id="slides",
                name="slides",
                description="Build a frontend slides deck from scratch.",
                tags=["frontend"],
                tools=[],
                metadata={},
                body="# slides",
            ),
        ]
    )
    hits = reg.search("api reference", 5)
    assert len(hits) >= 1
    for hit in hits:
        assert 0.0 < hit.relevance <= 1.0
        assert hit.relevance != hit.score
    assert "relevance=" in repr(hits[0])

async def test_registry_register_many_no_longer_exists() -> None:
    # register_many / build_embeddings / rebuild_embeddings were folded into
    # the variadic, self-embedding `register` (RAT-379/async-register).
    for registry in (ToolRegistry(), SkillRegistry()):
        assert not hasattr(registry, "register_many")
        assert not hasattr(registry, "build_embeddings")
        assert not hasattr(registry, "rebuild_embeddings")


async def test_register_item_and_register_iterable() -> None:
    reg = ToolRegistry()
    await reg.register(Tool(id="read", name="read", description="Read a file from disk"))
    await reg.register([Tool(id="send", name="send", description="Send an email message")])

    assert reg.search("send email", 5)[0].tool_id == "send"


async def test_skill_registry_register_item_and_register_iterable() -> None:
    reg = SkillRegistry()
    await reg.register(Skill(id="auth", name="auth", description="Set up login"))
    await reg.register([Skill(id="deploy", name="deploy", description="Deploy an app")])

    assert reg.search("deploy", 5)[0].skill_id == "deploy"


@pytest.mark.parametrize(
    ("registry", "item"),
    [
        (ToolRegistry(), Tool(id="read", name="read", description="Read a file")),
        (SkillRegistry(), Skill(id="auth", name="auth", description="Set up auth")),
    ],
)
def test_dense_submission_failure_releases_registry_busy_state(registry, item) -> None:
    async def exercise() -> None:
        await asyncio.get_running_loop().shutdown_default_executor()
        # `_build` is the internal dense-build primitive an eager `register` now
        # drives; exercised directly since these bm25-default registries have no
        # embedding config to route a real `register(..., method="semantic")`
        # eager build through, but the executor-submission failure it triggers
        # here is model-independent.
        with pytest.raises(RuntimeError, match="Executor shutdown"):
            await registry._build()
        await registry.register(item)

    asyncio.run(exercise())


async def test_registry_async_lifecycle_has_tool_and_skill_parity() -> None:
    tools = ToolRegistry()
    skills = SkillRegistry()

    await tools._build()
    await tools._rebuild()
    await skills._build()
    await skills._rebuild()

    assert await tools.search_async("anything", 5) == []
    assert await skills.search_async("anything", 5) == []


@pytest.mark.parametrize(
    ("registry_type", "item", "hit_attribute"),
    [
        (
            ToolRegistry,
            Tool(id="read", name="read", description="Read a file"),
            "tool_id",
        ),
        (
            SkillRegistry,
            Skill(id="auth", name="auth", description="Set up authentication"),
            "skill_id",
        ),
    ],
)
async def test_async_bm25_does_not_queue_behind_dense_operation(
    controlled_embedding_endpoint: tuple[str, threading.Event, threading.Event],
    registry_type: type[ToolRegistry] | type[SkillRegistry],
    item: Tool | Skill,
    hit_attribute: str,
) -> None:
    endpoint, request_started, send_response = controlled_embedding_endpoint
    registry: Any = registry_type(embedding={"url": endpoint, "model": "test-model"})
    await registry.register(item)
    build = asyncio.create_task(registry._build())
    for _ in range(200):
        if request_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert request_started.is_set()
    try:
        hits = await asyncio.wait_for(
            registry.search_async("read authentication", 5, method="bm25"),
            timeout=0.1,
        )
    finally:
        send_response.set()
        await build

    assert getattr(hits[0], hit_attribute) == item.id


def test_search_empty_registry_returns_empty() -> None:
    reg = ToolRegistry()
    assert reg.search("anything", 5) == []


async def test_search_with_origin_accepts_agent_and_direct() -> None:
    reg = ToolRegistry()
    await _register_read_file(reg)
    assert reg.search_with_origin("read file", 3, "agent")[0].tool_id == "read_file"
    assert reg.search_with_origin("read file", 3, "direct")[0].tool_id == "read_file"


async def test_every_origin_wire_value_reaches_its_own_variant() -> None:
    # The native maps origin strings with a `_ => Direct` fallback so version
    # skew can't fail an infallible search. The cost is that a newly added
    # variant would be silently recorded as "direct" with no compile error.
    # The binding's Cargo.toml sets `test = false` (a Rust test binary there
    # links libpython and fails to find it), so this guard lives here.
    reg = ToolRegistry()
    reg.set_trace_sink("memory", "origins")
    await _register_read_file(reg)

    for value in ("direct", "agent", "baseline"):
        reg.search_with_origin("read file", 3, value)

    recorded = [e["origin"] for e in reg.drain_trace_events() if e["type"] == "search"]
    assert recorded == ["direct", "agent", "baseline"], (
        f"each wire value must survive the round trip, got {recorded}"
    )


async def test_an_unrecognized_origin_degrades_to_direct() -> None:
    reg = ToolRegistry()
    reg.set_trace_sink("memory", "unknown-origin")
    await _register_read_file(reg)

    reg.search_with_origin("read file", 3, "from_the_future")

    recorded = [e["origin"] for e in reg.drain_trace_events() if e["type"] == "search"]
    assert recorded == ["direct"], "unknown origins must not fail the search"


async def test_memory_sink_captures_and_drains_events() -> None:
    reg = ToolRegistry()
    reg.set_trace_sink("memory", "sess-1")
    await _register_read_file(reg)  # emits an index_churn event
    reg.search("read file", 3)  # emits a search event
    events = reg.drain_trace_events()
    types = [e["type"] for e in events]
    assert "index_churn" in types
    assert "search" in types
    # every envelope is stamped
    assert all(e["session_id"] == "sess-1" for e in events)
    # draining is destructive
    assert reg.drain_trace_events() == []


async def test_noop_sink_drains_nothing() -> None:
    reg = ToolRegistry()
    await _register_read_file(reg)
    reg.search("read file", 3)
    assert reg.drain_trace_events() == []


def test_record_event_accepts_valid_event() -> None:
    reg = ToolRegistry()
    reg.set_trace_sink("memory", "sess-2")
    reg.record_event({"type": "invoke_start", "tool_id": "x", "args_size_bytes": 12})
    events = reg.drain_trace_events()
    assert events[0]["type"] == "invoke_start"
    assert events[0]["tool_id"] == "x"
    assert events[0]["args_size_bytes"] == 12


def test_record_event_rejects_invalid_event() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="invalid trace event"):
        reg.record_event({"type": "not_a_real_event"})


def test_memory_sink_requires_session_id() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="session_id"):
        reg.set_trace_sink("memory")
