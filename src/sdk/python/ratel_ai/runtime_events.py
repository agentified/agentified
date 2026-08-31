"""Public runtime-event stream and catalog snapshot seams (ADR-0020)."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import secrets
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

from ._native import NativeEventSubscription

if TYPE_CHECKING:
    from .catalog import ToolCatalog
    from .skill_catalog import SkillCatalog

RuntimeEvent = dict[str, Any]
# typing.Optional, not `None | ...`: this alias evaluates at import time, and PEP 604
# unions raise TypeError on CPython 3.9 (`from __future__ import annotations` defers
# only annotations, never assignments).
RuntimeEventHandler = Callable[[list[RuntimeEvent]], Optional[Awaitable[None]]]

RUNTIME_EVENT_TYPES = (
    "catalog_definition",
    "search",
    "skill_search",
    "gateway_search",
    "invoke_start",
    "invoke_end",
    "invoke_error",
    "gateway_invoke",
    "gateway_error",
    "skill_invoke",
    "index_churn",
    "skill_churn",
    "upstream_register",
    "upstream_invoke",
    "upstream_error",
    "auth_refresh",
    "auth_needs",
    "auth_flow_start",
    "auth_flow_end",
    "experiment_selection",
    "experiment_results",
    "experiment_comparison",
    "experiment_skip",
    "experiment_fallback",
    "experiment_drop",
    "experiment_invocation",
    "experiment_outcome",
    "events_dropped",
)
RUNTIME_EVENT_MAX_PAYLOAD_BYTES = 64 * 1_024
RUNTIME_EVENT_MAX_QUERY_BYTES = 4 * 1_024
RUNTIME_EVENT_MAX_HITS = 100

_REQUIRED_ENVELOPE_FIELDS = {"v", "event_id", "ts", "session_id", "source_id", "type"}
_CATALOG_CRITICAL_FIELDS = ("kind", "id", "name", "content_hash")
_CATALOG_SCHEMA_FIELDS = ("input_schema", "output_schema")
_CATALOG_DEFINITION_FIELDS = {
    *_CATALOG_CRITICAL_FIELDS,
    *_CATALOG_SCHEMA_FIELDS,
    "description",
    "tags",
    "searchable_description",
    "searchable_description_overridden",
}


def new_runtime_event_id(now_ms: int | None = None) -> str:
    """Mint a client ULID for one runtime event and OTel projection."""
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    timestamp = now_ms if now_ms is not None else int(time.time() * 1_000)
    encoded_time = ""
    for _ in range(10):
        encoded_time = alphabet[timestamp % 32] + encoded_time
        timestamp //= 32
    entropy = int.from_bytes(secrets.token_bytes(10), "big")
    encoded_entropy = ""
    for shift in range(75, -1, -5):
        encoded_entropy += alphabet[(entropy >> shift) & 31]
    return encoded_time + encoded_entropy


class _EventSource(Protocol):
    def experimental_enable_catalog_definitions(self) -> None: ...

    def subscribe_events(
        self,
        handler: Callable[[list[RuntimeEvent]], object],
        *,
        session_id: str,
        source_id: str,
        queue_capacity: int,
        batch_size: int,
    ) -> NativeEventSubscription: ...


class _SubscriptionState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = True
        self._dropped_count = 0
        self._warned_closed_loop = False

    def unsubscribe(self) -> bool:
        with self._lock:
            if not self._active:
                return False
            self._active = False
            return True

    def record_closed_loop_drop(self, count: int) -> None:
        with self._lock:
            self._dropped_count += count
            warn = not self._warned_closed_loop
            self._warned_closed_loop = True
        if warn:
            warnings.warn(
                "runtime events dropped because the async handler's event loop is closed",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count


class RuntimeEventSubscription:
    """Handle over one merged bounded runtime-event subscription."""

    def __init__(
        self,
        subscriptions: list[NativeEventSubscription],
        pending: set[asyncio.Future[None]],
        state: _SubscriptionState,
    ) -> None:
        """Create a subscription over native handles and pending handler work."""
        self._subscriptions = subscriptions
        self._pending = pending
        self._state = state

    async def flush(self) -> None:
        """Wait until accepted native and async-handler work completes."""
        await asyncio.gather(
            *(asyncio.to_thread(subscription.flush) for subscription in self._subscriptions)
        )
        await asyncio.sleep(0)
        while self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)
            await asyncio.sleep(0)

    def unsubscribe(self) -> None:
        """Stop accepting new events; already-queued native envelopes still drain."""
        if not self._state.unsubscribe():
            return
        for subscription in self._subscriptions:
            subscription.unsubscribe()

    @property
    def dropped_count(self) -> int:
        """Envelopes lost to native overflow or a closed async-handler loop."""
        native_dropped = sum(subscription.dropped_count for subscription in self._subscriptions)
        return native_dropped + self._state.dropped_count


class RuntimeEvents:
    """Merged push stream over tool and skill catalogs."""

    def __init__(
        self,
        sources: Sequence[_EventSource],
        *,
        session_id: str | None = None,
        source_id: str | None = None,
        queue_capacity: int = 1_024,
        batch_size: int = 64,
        experimental_catalog_definitions: bool = False,
    ) -> None:
        """Create a stream sharing one identity across all event sources.

        ``source_id`` defaults to the env-var-configured OTel ``service.name``
        (``OTEL_SERVICE_NAME``, then ``service.name`` in
        ``OTEL_RESOURCE_ATTRIBUTES``), then the service name recorded by this
        SDK's own telemetry configuration (``configure_telemetry(service_name=...)``
        / ``ratel_ai_telemetry.init``), falling back to ``"ratel"``. Env vars keep
        precedence, matching the OTel convention. Any other programmatically
        configured OTel resource is not read; pass ``source_id`` explicitly in
        that case.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.source_id = source_id or _default_source_id()
        self._sources = tuple(sources)
        if experimental_catalog_definitions:
            for source in sources:
                source.experimental_enable_catalog_definitions()
        self._queue_capacity = queue_capacity
        self._batch_size = batch_size

    def subscribe(self, handler: RuntimeEventHandler) -> RuntimeEventSubscription:
        """Subscribe to bounded asynchronous runtime-event batches."""
        state = _SubscriptionState()
        pending: set[asyncio.Future[None]] = set()
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def schedule_awaitable(awaitable: Awaitable[None]) -> None:
            if loop is None:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
                return

            def schedule() -> None:
                future = asyncio.ensure_future(awaitable, loop=loop)
                pending.add(future)
                future.add_done_callback(lambda finished: _finish_async_handler(finished, pending))

            try:
                on_target_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_target_loop = False
            if on_target_loop:
                schedule()
            else:
                if not _schedule_on_loop(loop, schedule, state, 1):
                    close = getattr(awaitable, "close", None)
                    if callable(close):
                        close()

        if inspect.iscoroutinefunction(handler):
            if loop is None:
                raise RuntimeError("async runtime-event handlers require a running event loop")
            async_handler = cast(Callable[[list[RuntimeEvent]], Coroutine[Any, Any, None]], handler)

            def deliver(batch: list[RuntimeEvent]) -> None:
                normalized = [_normalize_runtime_event(event) for event in batch]

                def schedule() -> None:
                    schedule_awaitable(async_handler(normalized))

                _schedule_on_loop(loop, schedule, state, len(normalized))

        else:

            def deliver(batch: list[RuntimeEvent]) -> None:
                try:
                    result = handler([_normalize_runtime_event(event) for event in batch])
                    if inspect.isawaitable(result):
                        schedule_awaitable(result)
                except Exception:
                    pass

        # Subscribe incrementally so a later source's failure explicitly unwinds the
        # native handles already granted (newest first), mirroring the TS SDK rather
        # than leaving the rollback to garbage collection.
        subscriptions: list[NativeEventSubscription] = []
        try:
            for source in self._sources:
                subscriptions.append(
                    source.subscribe_events(
                        deliver,
                        session_id=self.session_id,
                        source_id=self.source_id,
                        queue_capacity=self._queue_capacity,
                        batch_size=self._batch_size,
                    )
                )
        except BaseException:
            for subscription in reversed(subscriptions):
                subscription.unsubscribe()
            raise
        return RuntimeEventSubscription(subscriptions, pending, state)


def _schedule_on_loop(
    loop: asyncio.AbstractEventLoop,
    callback: Callable[[], None],
    state: _SubscriptionState,
    dropped_count: int,
) -> bool:
    if loop.is_closed():
        state.record_closed_loop_drop(dropped_count)
        return False
    try:
        loop.call_soon_threadsafe(callback)
    except RuntimeError:
        if not loop.is_closed():
            raise
        state.record_closed_loop_drop(dropped_count)
        return False
    return True


def _default_source_id() -> str:
    if service_name := os.getenv("OTEL_SERVICE_NAME"):
        return service_name
    attributes = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
    for entry in attributes.split(","):
        key, separator, value = entry.strip().partition("=")
        if separator and key == "service.name" and value:
            return value
    if recorded := _recorded_telemetry_service_name():
        return recorded
    return "ratel"


def _recorded_telemetry_service_name() -> str | None:
    """The service.name a programmatic `configure_telemetry`/`init` installed, or None.

    Read lazily from the OTel-free seam the telemetry helper records into
    (`ratel_ai_telemetry.recorded_service_name`, ADR-0020), so the env vars keep
    precedence and a telemetry release predating the seam degrades to the bare
    fallback instead of raising.
    """
    try:
        from ratel_ai_telemetry.otlp import recorded_service_name
    except ImportError:  # telemetry helper predates the seam
        return None
    return recorded_service_name()


def _finish_async_handler(
    task: asyncio.Future[None],
    pending: set[asyncio.Future[None]],
) -> None:
    pending.discard(task)
    if task.cancelled():
        return
    task.exception()


def _normalize_runtime_event(event: RuntimeEvent) -> RuntimeEvent:
    normalized = cast(RuntimeEvent, _sanitize_value(event))
    if _serialized_size(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES:
        return normalized

    _trim_catalog_schemas(normalized)
    if _serialized_size(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES:
        return normalized

    for key in tuple(normalized):
        if key not in _REQUIRED_ENVELOPE_FIELDS and not _is_product_fact_field(key):
            del normalized[key]
            normalized["payload_truncated"] = True
            if _serialized_size(normalized) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES:
                return normalized

    bounded = {key: normalized[key] for key in normalized if key in _REQUIRED_ENVELOPE_FIELDS}
    bounded["payload_truncated"] = True
    for key, value in _prioritized_product_fact_items(normalized):
        if key in _REQUIRED_ENVELOPE_FIELDS or not _is_product_fact_field(key):
            continue
        bounded[key] = _sanitize_bounded_value(value)
        if _serialized_size(bounded) > RUNTIME_EVENT_MAX_PAYLOAD_BYTES:
            del bounded[key]
    return bounded


def _trim_catalog_schemas(value: RuntimeEvent) -> None:
    if value.get("type") != "catalog_definition":
        return
    fields = sorted(
        (key for key in _CATALOG_SCHEMA_FIELDS if key in value),
        key=lambda key: _serialized_size({key: value[key]}),
        reverse=True,
    )
    for key in fields:
        del value[key]
        value["payload_truncated"] = True
        if _serialized_size(value) <= RUNTIME_EVENT_MAX_PAYLOAD_BYTES:
            break


def _prioritized_product_fact_items(value: RuntimeEvent) -> list[tuple[str, Any]]:
    items = [
        (key, item)
        for key, item in value.items()
        if key not in _REQUIRED_ENVELOPE_FIELDS and _is_product_fact_field(key)
    ]
    if value.get("type") != "catalog_definition":
        return items
    return [
        *((key, value[key]) for key in _CATALOG_CRITICAL_FIELDS if key in value),
        *((key, item) for key, item in items if key not in _CATALOG_CRITICAL_FIELDS),
    ]


def _sanitize_value(value: Any, key: str = "") -> Any:
    if isinstance(value, str):
        limit = RUNTIME_EVENT_MAX_QUERY_BYTES if key == "query" else 4_096
        return _truncate_utf8(value, limit)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value[:RUNTIME_EVENT_MAX_HITS]]
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_value(child_value, str(child_key))
            for child_key, child_value in list(value.items())[:100]
        }
    return value


def _sanitize_bounded_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_utf8(value, 512)
    if isinstance(value, list):
        return [_sanitize_bounded_value(item) for item in value[:RUNTIME_EVENT_MAX_HITS]]
    if isinstance(value, dict):
        return {str(key): _sanitize_bounded_value(child) for key, child in list(value.items())[:32]}
    return value


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode()
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode(errors="ignore")


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":")).encode())


def _is_product_fact_field(key: str) -> bool:
    return key.endswith(("_id", "_ids", "_ms", "_count", "_score", "_scores")) or key in {
        *_CATALOG_DEFINITION_FIELDS,
        "query",
        "target",
        "origin",
        "top_k",
        "hits",
        "outcome",
        "error",
        "error_class",
        "transport",
        "role",
        "cold",
        "agreement",
        "reason",
        "label",
        "score",
        "attributed",
        "rank",
        "turn",
        "action",
    }


class RuntimeCatalog:
    """Complete serializable tool and skill state for snapshot publication."""

    def __init__(
        self,
        tools: ToolCatalog,
        skills: SkillCatalog,
        *,
        source_id: str,
    ) -> None:
        """Create a snapshot view sharing the stream's stable source identity."""
        self._tools = tools
        self._skills = skills
        self._source_id = source_id

    def snapshot(self) -> dict[str, Any]:
        """Return the current full replacement snapshot."""
        return {
            "source_id": self._source_id,
            "tools": self._tools.snapshot(),
            "skills": self._skills.snapshot(),
        }
