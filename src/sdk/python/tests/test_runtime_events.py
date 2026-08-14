"""Public runtime-events and catalog-snapshot contract (ADR-0020)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
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


class TestDefaultSourceId:
    """The default source_id chain (ADR-0020): OTEL_SERVICE_NAME, then service.name in
    OTEL_RESOURCE_ATTRIBUTES, then the service name the telemetry helper recorded from a
    programmatic configure_telemetry/init, then "ratel"."""

    @pytest.fixture(autouse=True)
    def _no_service_name_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)

    def test_falls_back_to_the_recorded_telemetry_service_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ratel_ai_telemetry import otlp

        monkeypatch.setattr(otlp, "recorded_service_name", lambda: "checkout-agent")

        assert RuntimeEvents(()).source_id == "checkout-agent"

    def test_otel_service_name_env_wins_over_the_recorded_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ratel_ai_telemetry import otlp

        monkeypatch.setattr(otlp, "recorded_service_name", lambda: "recorded-loser")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-winner")

        assert RuntimeEvents(()).source_id == "env-winner"

    def test_resource_attributes_env_wins_over_the_recorded_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ratel_ai_telemetry import otlp

        monkeypatch.setattr(otlp, "recorded_service_name", lambda: "recorded-loser")
        monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=attr-winner")

        assert RuntimeEvents(()).source_id == "attr-winner"

    def test_falls_back_to_ratel_when_nothing_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ratel_ai_telemetry import otlp

        monkeypatch.setattr(otlp, "recorded_service_name", lambda: None)

        assert RuntimeEvents(()).source_id == "ratel"

    def test_tolerates_a_telemetry_release_predating_the_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ratel-ai floors ratel-ai-telemetry below the seam's release; an older helper
        # without recorded_service_name must degrade to the bare fallback, not raise.
        from ratel_ai_telemetry import otlp

        monkeypatch.delattr(otlp, "recorded_service_name")

        assert RuntimeEvents(()).source_id == "ratel"


def test_rolls_back_earlier_native_subscriptions_when_a_later_source_rejects() -> None:
    unsubscribed: list[str] = []

    class FirstSubscription:
        dropped_count = 0

        def unsubscribe(self) -> None:
            unsubscribed.append("first")

        def flush(self) -> None:
            pass

    class FirstSource:
        def subscribe_events(
            self,
            handler: object,
            *,
            session_id: str,
            source_id: str,
            queue_capacity: int,
            batch_size: int,
        ) -> FirstSubscription:
            return FirstSubscription()

    class BusySource:
        def subscribe_events(
            self,
            handler: object,
            *,
            session_id: str,
            source_id: str,
            queue_capacity: int,
            batch_size: int,
        ) -> FirstSubscription:
            raise RuntimeError("registry busy")

    events = RuntimeEvents([FirstSource(), BusySource()])  # type: ignore[list-item]

    with pytest.raises(RuntimeError, match="registry busy"):
        events.subscribe(lambda batch: None)
    assert unsubscribed == ["first"]


def test_unsubscribe_still_delivers_envelopes_already_queued_by_native() -> None:
    tools = ToolCatalog()
    events = RuntimeEvents([tools], queue_capacity=16, batch_size=1)
    first_batch_started = threading.Event()
    release_first_batch = threading.Event()
    queued_batch_delivered = threading.Event()

    def handler(batch: list[dict[str, object]]) -> None:
        if not first_batch_started.is_set():
            first_batch_started.set()
            release_first_batch.wait(timeout=1)
        if any(event.get("query") == "already-queued" for event in batch):
            queued_batch_delivered.set()

    subscription = events.subscribe(handler)
    tools.search("first", 1)
    assert first_batch_started.wait(timeout=1)
    tools.search("already-queued", 1)

    subscription.unsubscribe()
    release_first_batch.set()

    assert queued_batch_delivered.wait(timeout=2)


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


def test_counts_and_warns_when_an_async_handlers_event_loop_is_closed() -> None:
    tools = ToolCatalog()
    events = RuntimeEvents([tools], batch_size=1)

    async def handler(_batch: list[dict[str, object]]) -> None:
        pass

    async def subscribe():
        return events.subscribe(handler)

    subscription = asyncio.run(subscribe())

    with pytest.warns(RuntimeWarning, match="event loop is closed") as warning_records:
        tools.search("after-close", 1)
        tools.search("still-closed", 1)
        deadline = time.monotonic() + 1
        while subscription.dropped_count < 2 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert subscription.dropped_count == 2
    assert len(warning_records) == 1
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
