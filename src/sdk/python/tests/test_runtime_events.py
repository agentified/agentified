"""Public runtime-events and catalog-snapshot contract (ADR-0019)."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from ratel_ai import (
    RUNTIME_EVENT_MAX_HITS,
    RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
    RUNTIME_EVENT_MAX_QUERY_BYTES,
    RUNTIME_EVENT_TYPES,
    ExecutableTool,
    RuntimeCatalog,
    RuntimeEvents,
    Skill,
    SkillCatalog,
    ToolCatalog,
)


@pytest.mark.asyncio
async def test_returns_complete_serializable_catalog_without_executable_content() -> None:
    tools = ToolCatalog()
    skills = SkillCatalog()
    await tools.register(
        [
            ExecutableTool(
                id="z_tool",
                name="Z tool",
                description="Last by id",
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                output_schema={"type": "string"},
                execute=lambda _args: {"secret": "must never escape"},
            ),
            ExecutableTool(
                id="a_tool",
                name="A tool",
                description="First by id",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                execute=lambda _args: {},
            ),
        ]
    )
    await skills.register(
        Skill(
            id="skill-a",
            name="Skill A",
            description="Public skill metadata",
            tags=["public"],
            tools=["a_tool"],
            metadata={"stacks": ["python"]},
            body="May contain private instructions",
        )
    )

    snapshot = RuntimeCatalog(tools, skills, source_id="service-a").snapshot()

    assert snapshot == {
        "source_id": "service-a",
        "tools": [
            {
                "id": "a_tool",
                "name": "A tool",
                "description": "First by id",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            {
                "id": "z_tool",
                "name": "Z tool",
                "description": "Last by id",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "output_schema": {"type": "string"},
            },
        ],
        "skills": [
            {
                "id": "skill-a",
                "name": "Skill A",
                "description": "Public skill metadata",
                "tags": ["public"],
                "tools": ["a_tool"],
                "metadata": {"stacks": ["python"]},
            }
        ],
    }
    encoded = json.dumps(snapshot)
    assert "execute" not in encoded
    assert "private instructions" not in encoded


@pytest.mark.asyncio
async def test_merges_tool_and_skill_events_for_sync_handlers_off_thread() -> None:
    tools = ToolCatalog()
    skills = SkillCatalog()
    events = RuntimeEvents(
        [tools, skills],
        session_id="session-public",
        source_id="source-public",
        queue_capacity=16,
        batch_size=8,
    )
    received: list[dict[str, object]] = []
    handler_threads: list[int] = []
    producer_thread = threading.get_ident()

    def handler(batch: list[dict[str, object]]) -> None:
        received.extend(batch)
        handler_threads.append(threading.get_ident())

    subscription = events.subscribe(handler)
    await tools.register(
        ExecutableTool(
            id="read_file",
            name="read_file",
            description="Read a file",
            execute=lambda _args: {},
        )
    )
    await skills.register(
        Skill(id="api-design", name="API design", description="Design an API", body="Private")
    )
    tools.search("read", 1)
    skills.search("api", 1)

    await subscription.flush()

    assert {event["type"] for event in received} >= {
        "index_churn",
        "skill_churn",
        "search",
        "skill_search",
    }
    assert all(event["session_id"] == "session-public" for event in received)
    assert all(event["source_id"] == "source-public" for event in received)
    assert all(handler_thread != producer_thread for handler_thread in handler_threads)
    subscription.unsubscribe()


def test_matches_frozen_cross_language_event_vocabulary() -> None:
    fixtures = json.loads(
        (Path(__file__).parents[3] / "telemetry" / "conformance" / "fixtures.json").read_text()
    )

    assert fixtures["runtime_events"] == {
        "version": 2,
        "max_payload_bytes": RUNTIME_EVENT_MAX_PAYLOAD_BYTES,
        "max_query_bytes": RUNTIME_EVENT_MAX_QUERY_BYTES,
        "max_hits": RUNTIME_EVENT_MAX_HITS,
        "otel_event_id_attribute": "ratel.event.id",
        "required_envelope_fields": [
            "v",
            "event_id",
            "ts",
            "session_id",
            "source_id",
            "type",
        ],
        "event_types": list(RUNTIME_EVENT_TYPES),
    }


@pytest.mark.asyncio
async def test_bounds_query_hits_and_payload_before_delivery() -> None:
    tools = ToolCatalog()
    skills = SkillCatalog()
    events = RuntimeEvents([tools, skills])
    received: list[dict[str, object]] = []
    subscription = events.subscribe(lambda batch: received.extend(batch))
    await tools.register(
        [
            ExecutableTool(
                id=f"tool-{index:03d}",
                name=f"Tool {index:03d}",
                description="padding " * 1_000,
                execute=lambda _args: {},
            )
            for index in range(120)
        ]
    )

    tools.search("padding " * 1_000, 120)
    await subscription.flush()

    event = next(item for item in received if item["type"] == "search")
    assert len(str(event["query"]).encode()) <= RUNTIME_EVENT_MAX_QUERY_BYTES
    assert len(event["hits"]) == RUNTIME_EVENT_MAX_HITS  # type: ignore[arg-type]
    assert len(json.dumps(event, separators=(",", ":")).encode()) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES
    subscription.unsubscribe()


@pytest.mark.asyncio
async def test_marshals_async_handlers_to_the_subscribing_event_loop() -> None:
    tools = ToolCatalog()
    skills = SkillCatalog()
    events = RuntimeEvents([tools, skills], session_id="session-async")
    received: list[dict[str, object]] = []
    handler_threads: list[int] = []
    event_loop_thread = threading.get_ident()

    async def handler(batch: list[dict[str, object]]) -> None:
        await asyncio.sleep(0)
        received.extend(batch)
        handler_threads.append(threading.get_ident())

    subscription = events.subscribe(handler)
    tools.search("anything", 1)

    await subscription.flush()

    assert [event["type"] for event in received] == ["search"]
    assert handler_threads == [event_loop_thread]
    subscription.unsubscribe()


@pytest.mark.asyncio
async def test_marshals_handlers_that_return_an_awaitable_to_the_event_loop() -> None:
    tools = ToolCatalog()
    events = RuntimeEvents([tools])
    received: list[dict[str, object]] = []
    handler_threads: list[int] = []
    event_loop_thread = threading.get_ident()

    async def async_handler(batch: list[dict[str, object]]) -> None:
        received.extend(batch)
        handler_threads.append(threading.get_ident())

    subscription = events.subscribe(lambda batch: async_handler(batch))
    tools.search("anything", 1)

    await subscription.flush()

    assert [event["type"] for event in received] == ["search"]
    assert handler_threads == [event_loop_thread]
    subscription.unsubscribe()
