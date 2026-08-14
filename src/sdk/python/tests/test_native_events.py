"""Native push bridge tests; the public Python API lands in Phase 5."""

import gc
import subprocess
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from ratel_ai._native import IntentGraph, SkillRegistry, ToolRegistry


@pytest.mark.parametrize("accessor", ["subscription.unsubscribe()", "subscription.dropped_count"])
def test_subscription_access_does_not_deadlock_a_callback_draining_flush(accessor: str) -> None:
    script = textwrap.dedent(
        f"""
        import threading
        import time

        from ratel_ai._native import ToolRegistry

        registry = ToolRegistry()
        callback_started = threading.Event()

        def callback(_batch):
            if not callback_started.is_set():
                callback_started.set()
                time.sleep(0.5)

        subscription = registry.subscribe_trace_events(
            callback,
            "session-flush-access",
            queue_capacity=16,
            batch_size=1,
        )
        registry.search("queued", 1)
        assert callback_started.wait(timeout=1)
        for index in range(100):
            registry.search(f"queued-{{index}}", 1)

        flushing = threading.Thread(target=subscription.flush)
        flushing.start()
        time.sleep(0.05)
        {accessor}
        flushing.join(timeout=2)
        assert not flushing.is_alive()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=4,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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

    dropped_count = subscription.dropped_count
    assert dropped_count > 0
    assert any(
        event["type"] == "events_dropped" and event["reason"] == "queue_overflow"
        for event in events
    )
    subscription.unsubscribe()
    assert subscription.dropped_count == dropped_count


def test_unsubscribe_releases_the_callback_dispatcher() -> None:
    registry = ToolRegistry()
    released = threading.Event()

    class Callback:
        def __call__(self, _batch: list[dict[str, Any]]) -> None:
            pass

        def __del__(self) -> None:
            released.set()

    callback = Callback()
    subscription = registry.subscribe_trace_events(callback, "session-dispatcher")
    del callback

    subscription.unsubscribe()

    for _ in range(100):
        _ = subscription.dropped_count
        gc.collect()
        if released.wait(timeout=0.01):
            break
    assert released.is_set()


def test_keeps_base_sink_synchronous_and_independently_stamped_after_subscribing() -> None:
    registry = ToolRegistry()
    registry.set_trace_sink("memory", "base-session")
    subscription = registry.subscribe_trace_events(
        lambda batch: None,
        "stream-session",
        queue_capacity=1,
        batch_size=1,
    )

    total = 5_000
    for index in range(total):
        registry.search(f"persist-{index}", 1)

    drained = registry.drain_trace_events()
    persisted = [event for event in drained if event["type"] == "search"]
    assert len(persisted) == total
    assert all(event["session_id"] == "base-session" for event in persisted)
    assert not any(event["type"] == "events_dropped" for event in drained)
    subscription.unsubscribe()


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
