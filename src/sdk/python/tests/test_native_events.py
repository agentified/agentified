"""Native push bridge tests; the public Python API lands in Phase 5."""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ratel_ai._native import IntentGraph, SkillRegistry, ToolRegistry


def test_pushes_worker_thread_search_events_in_batches() -> None:
    registry = ToolRegistry()
    registry.register("read_file", "read_file", "Read a file", {}, {})
    batches: list[list[dict[str, Any]]] = []
    callback_threads: list[int] = []
    subscription = registry.subscribe_trace_events(
        lambda batch: (batches.append(batch), callback_threads.append(threading.get_ident())),
        "session-python",
        "source-python",
        16,
        8,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        producer_thread = executor.submit(threading.get_ident).result()
        worker_search = executor.submit(
            registry._search_with_method,
            "read",
            1,
            "direct",
            "bm25",
        )
        for index in range(8):
            registry.search(f"burst-{index}", 1)
        worker_search.result()
    subscription.flush()

    assert any(len(batch) > 1 for batch in batches)
    event = next(
        event
        for batch in batches
        for event in batch
        if event["type"] == "search" and event["query"] == "read"
    )
    assert event | {
        "v": 2,
        "session_id": "session-python",
        "source_id": "source-python",
        "query": "read",
    } == event
    assert callback_threads
    assert callback_threads[0] != producer_thread


def test_pushes_skill_registry_events_through_the_same_seam() -> None:
    registry = SkillRegistry()
    registry.register("api-design", "api-design", "Design an API", [], [], {}, "Use nouns")
    events: list[dict[str, Any]] = []
    subscription = registry.subscribe_trace_events(
        lambda batch: events.extend(batch),
        "session-skill",
    )

    registry._search_with_method("api", 1, "direct", "bm25")
    subscription.flush()

    assert any(
        event["type"] == "skill_search" and event["session_id"] == "session-skill"
        for event in events
    )


def test_drops_oldest_and_reports_loss_while_python_callback_is_stalled() -> None:
    registry = ToolRegistry()
    callback_started = threading.Event()
    release_callback = threading.Event()
    events: list[dict[str, Any]] = []

    def callback(batch: list[dict[str, Any]]) -> None:
        events.extend(batch)
        if not callback_started.is_set():
            callback_started.set()
            release_callback.wait(timeout=1)

    subscription = registry.subscribe_trace_events(
        callback,
        "session-drop",
        queue_capacity=2,
        batch_size=1,
    )
    registry.search("first", 1)
    assert callback_started.wait(timeout=1)

    for index in range(64):
        registry.search(f"query-{index}", 1)
    release_callback.set()
    subscription.flush()

    assert subscription.dropped_count > 0
    assert any(
        event["type"] == "events_dropped" and event["reason"] == "queue_overflow"
        for event in events
    )


def test_keeps_callbacks_and_usage_learning_across_base_sink_rewrap() -> None:
    registry = ToolRegistry()
    graph = IntentGraph()
    registry.enable_adaptive_ranking(graph)
    events: list[dict[str, Any]] = []
    subscription = registry.subscribe_trace_events(
        lambda batch: events.extend(batch),
        "session-learner",
    )
    registry.set_trace_sink("memory", "session-learner")

    registry.search("read file", 1)
    registry.record_event({"type": "invoke_start", "tool_id": "read_file", "args_size_bytes": 0})
    subscription.flush()

    assert graph.cluster_count == 1
    assert {event["type"] for event in events} >= {"search", "invoke_start"}
