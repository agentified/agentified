"""OpenTelemetry emission for the SDK funnel (ADR-0007).

The Python mirror of `src/sdk/ts/src/telemetry.ts`.

The catalog / capability-tool / skill / MCP paths call these helpers to open a span around
each operation, alongside the local `record_event` stream (untouched). Span names
and attribute keys come from the OTel-free `ratel_ai_telemetry` vocabulary.

Emission is **transparent and optional**: the `ratel_ai_telemetry` vocabulary ships
with the base install (it is OTel-free), but the OpenTelemetry API is imported lazily.
Without OpenTelemetry present, every helper is a straight pass-through — zero overhead,
no spans. A host that registers its own OpenTelemetry provider (or installs
`ratel-ai[otlp]`) lights emission up; spans then go to whatever provider is registered,
exactly as a host deployment wires it. Content (`ratel.search.query`, tool args/result,
and the content events) is emitted only when the ecosystem capture gate is on (default
off).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import Any, TypedDict, TypeVar

import rfc8785

from .runtime_events import new_runtime_event_id

try:
    import ratel_ai_telemetry as _telemetry_vocabulary
    from opentelemetry import _logs as _otel_logs
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from ratel_ai_telemetry import (
        EXECUTE_TOOL,
        GEN_AI_OPERATION_NAME,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
        GEN_AI_TOOL_NAME,
        RATEL_AUTH_FLOW,
        RATEL_AUTH_OUTCOME,
        RATEL_ORIGIN,
        RATEL_SEARCH,
        RATEL_SEARCH_HIT_COUNT,
        RATEL_SEARCH_QUERY,
        RATEL_SEARCH_RESULTS,
        RATEL_SEARCH_TARGET,
        RATEL_SEARCH_TOP_K,
        RATEL_SKILL_ID,
        RATEL_SKILL_LOAD,
        RATEL_TOOL_ARGS_SIZE_BYTES,
        RATEL_TOOL_EXECUTION_DETAILS,
        RATEL_UPSTREAM_REGISTER,
        RATEL_UPSTREAM_SERVER,
        RATEL_UPSTREAM_TOOL_COUNT,
        RATEL_UPSTREAM_TRANSPORT,
        AuthOutcome,
        ContentCapture,
        content_capture_mode,
    )

    # Compatibility with telemetry helpers released before the catalog
    # definition vocabulary; the SDK and helper ship on separate trains.
    RATEL_CATALOG_CONTENT_HASH: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_CONTENT_HASH", "ratel.catalog.content_hash"
    )
    RATEL_CATALOG_DEFINITION: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_DEFINITION", "ratel.catalog.definition"
    )
    RATEL_CATALOG_DESCRIPTION: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_DESCRIPTION", "ratel.catalog.description"
    )
    RATEL_CATALOG_ID: str = getattr(_telemetry_vocabulary, "RATEL_CATALOG_ID", "ratel.catalog.id")
    RATEL_CATALOG_INPUT_SCHEMA: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_INPUT_SCHEMA", "ratel.catalog.input_schema"
    )
    RATEL_CATALOG_KIND: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_KIND", "ratel.catalog.kind"
    )
    RATEL_CATALOG_NAME: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_NAME", "ratel.catalog.name"
    )
    RATEL_CATALOG_OUTPUT_SCHEMA: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_OUTPUT_SCHEMA", "ratel.catalog.output_schema"
    )
    RATEL_CATALOG_SCHEMA_OMITTED: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_SCHEMA_OMITTED", "ratel.catalog.schema_omitted"
    )
    RATEL_CATALOG_SEARCHABLE_DESCRIPTION: str = getattr(
        _telemetry_vocabulary,
        "RATEL_CATALOG_SEARCHABLE_DESCRIPTION",
        "ratel.catalog.searchable_description",
    )
    RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN: str = getattr(
        _telemetry_vocabulary,
        "RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN",
        "ratel.catalog.searchable_description_overridden",
    )
    RATEL_CATALOG_TAGS: str = getattr(
        _telemetry_vocabulary, "RATEL_CATALOG_TAGS", "ratel.catalog.tags"
    )
    EXPERIMENTAL_CATALOG_DEFINITIONS_ENV: str = getattr(
        _telemetry_vocabulary,
        "EXPERIMENTAL_CATALOG_DEFINITIONS_ENV",
        "RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS",
    )

    RATEL_EVENT_ID: str
    try:
        from ratel_ai_telemetry import RATEL_EVENT_ID as _IMPORTED_RATEL_EVENT_ID
    except ImportError:
        # Compatibility with telemetry helpers released before ADR-0020. The
        # Python helper exports this constant on the same release train.
        RATEL_EVENT_ID = "ratel.event.id"
    else:
        RATEL_EVENT_ID = _IMPORTED_RATEL_EVENT_ID

    _ENABLED = True
except ModuleNotFoundError:
    _ENABLED = False

_TRACER_NAME = "ratel-ai"
_CATALOG_SCHEMA_MAX_ATTRIBUTE_BYTES = 64 * 1_024
_MAX_SAFE_INTEGER = 2**53 - 1

#: `ratel.search.target` values (mirror `SearchTarget` so catalog call sites stay
#: decoupled from the telemetry vocabulary module).
SEARCH_TARGET_TOOL = "tool"
SEARCH_TARGET_SKILL = "skill"
SEARCH_TARGET_FACT = "fact"

T = TypeVar("T")


class RuntimeEventProjection(TypedDict, total=False):
    """Identity shared by one runtime envelope and its OTel projection."""

    event_id: str
    invocation_id: str
    trace_id: str
    span_id: str


def record_catalog_definitions(
    kind: str,
    definitions: Sequence[Any],
    emitted_hashes: dict[str, str],
) -> None:
    """Project changed catalog definitions into the opt-in Logs channel."""
    if (
        not _ENABLED
        or os.getenv(EXPERIMENTAL_CATALOG_DEFINITIONS_ENV, "").lower() != "true"
        or not _capture_content_on_event()
    ):
        return
    for definition in definitions:
        tags = list(getattr(definition, "tags", []))
        override = getattr(definition, "experimental_searchable_description", None)
        searchable_description = definition.description if override is None else override
        input_schema = getattr(definition, "input_schema", None) if kind == "tool" else None
        output_schema = getattr(definition, "output_schema", None) if kind == "tool" else None
        content = {
            "kind": kind,
            "id": definition.id,
            "name": definition.name,
            "description": definition.description,
            "tags": tags,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "searchable_description": searchable_description,
            "searchable_description_overridden": override is not None,
        }
        if _has_unsafe_integer(content):
            continue
        try:
            content_hash = hashlib.sha256(_canonical_json(content).encode()).hexdigest()
            if kind == "tool":
                canonical_input_schema = _canonical_json(input_schema)
                canonical_output_schema = _canonical_json(output_schema)
                input_schema_omitted = _catalog_schema_exceeds_attribute_limit(
                    canonical_input_schema
                )
                output_schema_omitted = _catalog_schema_exceeds_attribute_limit(
                    canonical_output_schema
                )
        except rfc8785.IntegerDomainError:
            # Catalog mutation already succeeded. Content telemetry is lossy and
            # must never turn a supported schema into a failed registration.
            continue
        if emitted_hashes.get(definition.id) == content_hash:
            continue
        attributes: dict[str, Any] = {
            RATEL_CATALOG_KIND: kind,
            RATEL_CATALOG_ID: definition.id,
            RATEL_CATALOG_NAME: definition.name,
            RATEL_CATALOG_DESCRIPTION: definition.description,
            RATEL_CATALOG_TAGS: tags,
            RATEL_CATALOG_SEARCHABLE_DESCRIPTION: searchable_description,
            RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN: override is not None,
            RATEL_CATALOG_CONTENT_HASH: content_hash,
        }
        if kind == "tool":
            if not input_schema_omitted:
                attributes[RATEL_CATALOG_INPUT_SCHEMA] = canonical_input_schema
            if not output_schema_omitted:
                attributes[RATEL_CATALOG_OUTPUT_SCHEMA] = canonical_output_schema
            if input_schema_omitted or output_schema_omitted:
                attributes[RATEL_CATALOG_SCHEMA_OMITTED] = True
        _logger().emit(event_name=RATEL_CATALOG_DEFINITION, attributes=attributes)
        emitted_hashes[definition.id] = content_hash


def _canonical_json(value: Any) -> str:
    return rfc8785.dumps(value).decode("utf-8")


def _catalog_schema_exceeds_attribute_limit(value: str) -> bool:
    return len(value.encode()) > _CATALOG_SCHEMA_MAX_ATTRIBUTE_BYTES


def _has_unsafe_integer(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return abs(value) > _MAX_SAFE_INTEGER
    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer() and abs(value) > _MAX_SAFE_INTEGER
    if isinstance(value, list):
        return any(_has_unsafe_integer(item) for item in value)
    if isinstance(value, dict):
        return any(_has_unsafe_integer(item) for item in value.values())
    return False


def _tracer() -> Any:
    return _otel_trace.get_tracer(_TRACER_NAME)


def _logger() -> Any:
    return _otel_logs.get_logger(_TRACER_NAME)


def _capture_content_on_span() -> bool:
    mode = content_capture_mode()
    return mode in (ContentCapture.SPAN_ONLY, ContentCapture.SPAN_AND_EVENT)


def _capture_content_on_event() -> bool:
    mode = content_capture_mode()
    return mode in (ContentCapture.EVENT_ONLY, ContentCapture.SPAN_AND_EVENT)


#: Sentinel distinguishing "no result" (error path) from a result that is falsy/None.
_UNSET: Any = object()


def _event_projection(span: Any = None, invocation_id: str | None = None) -> RuntimeEventProjection:
    projection = RuntimeEventProjection(event_id=new_runtime_event_id())
    if invocation_id is not None:
        projection["invocation_id"] = invocation_id
    if span is None:
        return projection
    span.set_attribute(RATEL_EVENT_ID, projection["event_id"])
    context = span.get_span_context()
    if context.is_valid:
        projection["trace_id"] = f"{context.trace_id:032x}"
        projection["span_id"] = f"{context.span_id:016x}"
    return projection


def _add_tool_content_event(
    tool_id: str,
    args: Any,
    event_id: str,
    result: Any = _UNSET,
) -> None:
    """Emit structured tool arguments and result as an OpenTelemetry EventRecord."""
    attributes = {
        GEN_AI_OPERATION_NAME: EXECUTE_TOOL,
        GEN_AI_TOOL_NAME: tool_id,
        GEN_AI_TOOL_CALL_ARGUMENTS: _safe_log_value(args),
        RATEL_EVENT_ID: event_id,
    }
    if result is not _UNSET:
        attributes[GEN_AI_TOOL_CALL_RESULT] = _safe_log_value(result)
    _logger().emit(event_name=RATEL_TOOL_EXECUTION_DETAILS, attributes=attributes)


def _add_search_results_event(query: str, event_id: str) -> None:
    """Emit the Opt-In ``ratel.search.results`` EventRecord carrying the search text.

    Hit ids/scores/BM25 timing are local-stream only; the OTLP glue carries the gated
    query it has (CONVENTIONS.md § ratel.search).
    """
    _logger().emit(
        event_name=RATEL_SEARCH_RESULTS,
        attributes={RATEL_SEARCH_QUERY: query, RATEL_EVENT_ID: event_id},
    )


def _args_size_bytes(args: Any) -> int:
    try:
        return len(json.dumps(args))
    except Exception:
        return 0


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(_normalize_content(value))
    except Exception:
        return ""


def _safe_log_value(value: Any) -> Any:
    try:
        return _normalize_content(value)
    except Exception:
        return None


def _normalize_content(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json", by_alias=True, exclude_none=True)
        except TypeError:
            dumped = model_dump()
        if type(value).__name__ == "CallToolResult" and isinstance(dumped, dict):
            stable = ("content", "structuredContent", "isError")
            return _normalize_content({key: dumped[key] for key in stable if key in dumped})
        return _normalize_content(dumped)
    if isinstance(value, Enum):
        return _normalize_content(value.value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_content(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [_normalize_content(item) for item in value]
        item_types = {type(item) for item in items if item is not None}
        if len(item_types) > 1:
            # OTel Python 1.41 (the last Python 3.9-compatible line) rejects mixed-type
            # AnyValue arrays and silently replaces them with null. Preserve the JSON value
            # losslessly as a typed, indexed map until that SDK limitation can be removed.
            return {
                "ratel.type": "array",
                "ratel.items": {str(index): item for index, item in enumerate(items)},
            }
        return items
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
        return value
    return str(value)


async def trace_execute_tool(
    tool_id: str,
    args: dict[str, Any],
    run: Callable[[RuntimeEventProjection], Awaitable[T]],
) -> T:
    """Wrap a tool invocation in a standard `execute_tool` span.

    The OTel gen_ai tool operation (`gen_ai.operation.name = execute_tool`,
    enriched with `ratel.*`), so a generic backend understands it (ADR-0007).
    No-op pass-through when telemetry is disabled.
    """
    if not _ENABLED:
        return await run(_event_projection(invocation_id=new_runtime_event_id()))
    with _tracer().start_as_current_span(
        f"{EXECUTE_TOOL} {tool_id}", kind=SpanKind.INTERNAL
    ) as span:
        projection = _event_projection(span, new_runtime_event_id())
        span.set_attribute(GEN_AI_OPERATION_NAME, EXECUTE_TOOL)
        span.set_attribute(GEN_AI_TOOL_NAME, tool_id)
        span.set_attribute(RATEL_TOOL_ARGS_SIZE_BYTES, _args_size_bytes(args))
        if _capture_content_on_span():
            span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, _safe_json(args))
        # start_as_current_span records the exception + sets ERROR status on raise.
        try:
            result = await run(projection)
        except BaseException:
            if _capture_content_on_event():
                _add_tool_content_event(tool_id, args, projection["event_id"])
            raise
        if _capture_content_on_span():
            span.set_attribute(GEN_AI_TOOL_CALL_RESULT, _safe_json(result))
        if _capture_content_on_event():
            _add_tool_content_event(tool_id, args, projection["event_id"], result=result)
        span.set_status(Status(StatusCode.OK))
        return result


def trace_search(
    target: str,
    query: str,
    top_k: int,
    origin: str,
    run: Callable[[RuntimeEventProjection], T],
) -> T:
    """Wrap a capability search (tool or skill) in a `ratel.search` span.

    Synchronous: the native BM25 search returns inline; the hit count becomes
    `ratel.search.hit_count`.
    """
    if not _ENABLED:
        return run(_event_projection())
    with _tracer().start_as_current_span(RATEL_SEARCH, kind=SpanKind.INTERNAL) as span:
        projection = _event_projection(span)
        span.set_attribute(RATEL_SEARCH_TARGET, target)
        span.set_attribute(RATEL_SEARCH_TOP_K, top_k)
        span.set_attribute(RATEL_ORIGIN, origin)
        if _capture_content_on_span():
            span.set_attribute(RATEL_SEARCH_QUERY, query)
        hits = run(projection)
        span.set_attribute(RATEL_SEARCH_HIT_COUNT, len(hits))  # type: ignore[arg-type]
        if _capture_content_on_event():
            _add_search_results_event(query, projection["event_id"])
        span.set_status(Status(StatusCode.OK))
        return hits


async def trace_search_async(
    target: str,
    query: str,
    top_k: int,
    origin: str,
    run: Callable[[RuntimeEventProjection], Awaitable[T]],
) -> T:
    """Wrap asynchronous BM25, semantic, or hybrid retrieval in a `ratel.search` span."""
    if not _ENABLED:
        return await run(_event_projection())
    with _tracer().start_as_current_span(RATEL_SEARCH, kind=SpanKind.INTERNAL) as span:
        projection = _event_projection(span)
        span.set_attribute(RATEL_SEARCH_TARGET, target)
        span.set_attribute(RATEL_SEARCH_TOP_K, top_k)
        span.set_attribute(RATEL_ORIGIN, origin)
        if _capture_content_on_span():
            span.set_attribute(RATEL_SEARCH_QUERY, query)
        hits = await run(projection)
        span.set_attribute(RATEL_SEARCH_HIT_COUNT, len(hits))  # type: ignore[arg-type]
        if _capture_content_on_event():
            _add_search_results_event(query, projection["event_id"])
        span.set_status(Status(StatusCode.OK))
        return hits


def trace_skill_load(skill_id: str, run: Callable[[RuntimeEventProjection], T]) -> T:
    """Wrap a skill-content load in a `ratel.skill.load` span."""
    if not _ENABLED:
        return run(_event_projection())
    with _tracer().start_as_current_span(RATEL_SKILL_LOAD, kind=SpanKind.INTERNAL) as span:
        projection = _event_projection(span)
        span.set_attribute(RATEL_SKILL_ID, skill_id)
        body = run(projection)
        span.set_status(Status(StatusCode.OK))
        return body


async def trace_upstream_register(
    server: str,
    transport: str,
    run: Callable[[Callable[[int], None], RuntimeEventProjection], Awaitable[T]],
) -> T:
    """Wrap an upstream-MCP registration in a `ratel.upstream.register` span.

    `run` receives a `report_tool_count` callback to set
    `ratel.upstream.tool_count` once the tool list is known.
    """
    if not _ENABLED:
        return await run(lambda _n: None, _event_projection())
    with _tracer().start_as_current_span(RATEL_UPSTREAM_REGISTER, kind=SpanKind.INTERNAL) as span:
        projection = _event_projection(span)
        span.set_attribute(RATEL_UPSTREAM_SERVER, server)
        span.set_attribute(RATEL_UPSTREAM_TRANSPORT, transport)
        result = await run(
            lambda n: span.set_attribute(RATEL_UPSTREAM_TOOL_COUNT, n),
            projection,
        )
        span.set_status(Status(StatusCode.OK))
        return result


def record_auth_needed(server: str | None = None) -> RuntimeEventProjection:
    """Mark an upstream tool call that failed with a 401 / needs-reauthorization.

    Emits a short `ratel.auth.flow` span carrying `ratel.auth.outcome = needs_auth`.
    """
    if not _ENABLED:
        return _event_projection()
    span = _tracer().start_span(RATEL_AUTH_FLOW, kind=SpanKind.INTERNAL)
    projection = _event_projection(span)
    if server:
        span.set_attribute(RATEL_UPSTREAM_SERVER, server)
    span.set_attribute(RATEL_AUTH_OUTCOME, AuthOutcome.NEEDS_AUTH.value)
    span.end()
    return projection


def _resolve_capture_override(
    capture_content: ContentCapture | str | None,
    include_span_and_events: bool | None,
) -> ContentCapture | str | None:
    """Resolve the capture mode `configure_telemetry` should set.

    `capture_content` wins over `include_span_and_events`; returns None to leave the
    gate env-driven. The bool sugar maps to the wire strings `set_content_capture`
    normalizes (True -> full capture, False -> none).
    """
    if capture_content is not None:
        return capture_content
    if include_span_and_events is not None:
        return "SPAN_AND_EVENT" if include_span_and_events else "NO_CONTENT"
    return None


class _TelemetryHandle:
    """Per-call shutdown behavior over the shutdown handle init() may reuse.

    Attribute access delegates to the underlying handle so callers can still use handle
    methods such as ``force_flush`` without configure_telemetry mutating the shared handle's
    ``shutdown`` method.
    """

    def __init__(self, inner: Any, shutdown: Callable[[], Any]) -> None:
        self._inner = inner
        self._shutdown = shutdown

    def shutdown(self) -> Any:
        """Run this configure call's generation-scoped teardown, then stop the provider."""
        return self._shutdown()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def configure_telemetry(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    logs_endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    service_name: str | None = None,
    capture_content: ContentCapture | str | None = None,
    include_span_and_events: bool | None = None,
    export_all_spans: bool = False,
) -> Any:
    """Register Ratel-owned OTLP trace and Logs exporters for the greenfield case.

    Ships the spans and EventRecords this SDK emits to Ratel Cloud (or any OTLP endpoint) by
    delegating to `ratel_ai_telemetry.init`, which needs the OpenTelemetry SDK —
    install it with ``pip install 'ratel-ai[otlp]'``. A host already running its
    own OpenTelemetry providers should skip this (SDK telemetry flows to them) and add
    both `ratel_span_processor` and `ratel_log_record_processor`.

    ``capture_content`` / ``include_span_and_events`` opt into message/tool content
    capture in code via `set_content_capture`: ``capture_content`` sets an exact mode,
    ``include_span_and_events`` is bool sugar (True -> ``SPAN_AND_EVENT``, False ->
    ``NO_CONTENT``); ``capture_content`` wins when both are given. A provided option
    beats ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` (the env var is the
    fallback when neither is given, as in OTel). The returned handle's ``shutdown()``
    clears the override, restoring env-driven behavior; the clear is
    generation-scoped, so a stale handle shutting down late never clobbers an
    override a newer `configure_telemetry`/`set_content_capture` call installed.

    Args:
        api_key: Ratel Cloud API key override; defaults to ``RATEL_API_KEY``.
        endpoint: OTLP endpoint override; defaults to ``RATEL_OTLP_ENDPOINT``, then
            the superseded ``RATEL_URL`` (which warns).
        logs_endpoint: OTLP logs endpoint override; defaults to the sibling
            ``/v1/logs`` URL derived from `endpoint`.
        headers: Extra headers sent with every export request.
        service_name: ``service.name`` resource attribute; defaults per `init`.
            Once the provider is installed, the effective name is also recorded
            as the default runtime-events ``source_id`` (ADR-0020), after the
            ``OTEL_SERVICE_NAME`` / ``OTEL_RESOURCE_ATTRIBUTES`` env vars —
            so the facts lane and the OTel lane share one identity.
        capture_content: Exact content-capture mode to set (see above).
        include_span_and_events: Boolean sugar for `capture_content` (see above).
        export_all_spans: Export every span, not just the ``gen_ai.*``/``ratel.*``
            signal. Default False: this high-level path defaults to
            ``ratel_signal_filter`` so unrelated HTTP/database/application spans are
            not shipped to Ratel (privacy + cost). Set True to forward all spans.

    Returns:
        A per-call shutdown handle (``handle.shutdown()`` / ``handle.force_flush()``),
        the same shape on every path. Attribute access delegates to the shared handle
        `init` returns; because that handle is shared across callers, shutting it down
        stops export for all of them.

    Raises:
        ModuleNotFoundError: if the OpenTelemetry exporter is not installed
            (``pip install 'ratel-ai[otlp]'``).
        ValueError: if `capture_content` is not a recognized mode — raised before
            any exporter is wired, so a bad option has no side effects.
    """
    try:
        from ratel_ai_telemetry import (
            clear_content_capture,
            init,
            ratel_event_filter,
            ratel_signal_filter,
            set_content_capture,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "configure_telemetry() needs the OpenTelemetry exporter. Install the extra: "
            "pip install 'ratel-ai[otlp]' — or register your own OpenTelemetry provider, "
            "since the SDK emits ratel.*/gen_ai.* telemetry to whichever providers are active."
        ) from exc

    # High-level config defaults to the ratel.*/gen_ai.* signal filter, so unrelated
    # application spans are not shipped (privacy + cost); opt in to all spans explicitly.
    # init() itself keeps its accept-all turnkey default (CONVENTIONS.md § init() surface).
    span_filter = None if export_all_spans else ratel_signal_filter
    log_filter = ratel_event_filter

    capture = _resolve_capture_override(capture_content, include_span_and_events)
    if capture is None:
        # No override: the env var keeps ruling; nothing to set or undo. Still wrap so the
        # return shape (a per-call handle delegating to the shared provider) matches the
        # capture path below, rather than leaking init()'s shared handle directly.
        handle = init(
            api_key=api_key,
            endpoint=endpoint,
            logs_endpoint=logs_endpoint,
            headers=headers,
            service_name=service_name,
            span_filter=span_filter,
            log_filter=log_filter,
        )
        return _TelemetryHandle(handle, handle.shutdown)

    # Apply (and validate — an unrecognized mode raises ValueError) the override *before*
    # wiring the exporter, so a bad option fails loud with no provider side effects; unwind
    # it if init() itself raises.
    generation = set_content_capture(capture)
    try:
        provider = init(
            api_key=api_key,
            endpoint=endpoint,
            logs_endpoint=logs_endpoint,
            headers=headers,
            service_name=service_name,
            span_filter=span_filter,
            log_filter=log_filter,
        )
    except BaseException:
        clear_content_capture(generation)
        raise

    def shutdown_and_clear() -> Any:
        clear_content_capture(generation)
        return provider.shutdown()

    # Keep teardown per configure call. init() is idempotent and may return the same provider
    # to multiple callers; mutating provider.shutdown would make every reference observe the
    # newest generation's wrapper and let a stale handle clear a newer privacy override.
    return _TelemetryHandle(provider, shutdown_and_clear)
