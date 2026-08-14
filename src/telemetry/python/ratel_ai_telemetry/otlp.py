"""The OTLP exporter surface over the standard OpenTelemetry Python SDK.

Two entry points, no custom transport and no schema (ADR-0007, CONVENTIONS.md § init()
surface):

- ``init()`` — the turnkey greenfield path: wires OTLP http/protobuf trace and Logs
  exporters and registers providers Ratel owns.
- Span/log processor and exporter helpers — compose Ratel onto providers a partner already
  owns (Langfuse, the Vercel AI SDK, ...).

The OpenTelemetry SDK is an optional ``[otlp]`` extra: this module imports it lazily, inside
the functions that need it, so the config/gate helpers (``resolve_otlp_config``,
``content_capture_mode``, ``ratel_signal_filter``) and this whole submodule import OTel-free.
Only wiring an exporter needs the extra.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, Protocol, cast

from . import CAPTURE_CONTENT_ENV

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LogRecordProcessor, ReadWriteLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExporter
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from opentelemetry.sdk.trace.export import SpanExporter


class TelemetryHandle(Protocol):
    """What init() returns: a shutdown handle, not a full provider.

    Both the enabled path and the disabled/no-op path satisfy this, so the return type is
    honest either way. Emit telemetry through the global OpenTelemetry APIs, not off this
    handle — init() registers both signal providers globally.
    """

    def shutdown(self) -> Any: ...

    def force_flush(self, timeout_millis: int = 30_000) -> bool: ...


#: Env var whose value is the default OTLP traces endpoint.
OTLP_ENDPOINT_ENV = "RATEL_OTLP_ENDPOINT"

#: Superseded spelling, still read as a fallback. ``RATEL_URL`` also names the SDK's
#: catalog source (ADR-0003), so overloading it as the OTLP endpoint conflated two
#: unrelated destinations. Setting it still works and warns.
LEGACY_ENDPOINT_ENV = "RATEL_URL"

#: Backwards-compatible alias of :data:`OTLP_ENDPOINT_ENV`, the var callers set.
ENDPOINT_ENV = OTLP_ENDPOINT_ENV

#: Env var whose value is the default API key when api_key= is omitted.
API_KEY_ENV = "RATEL_API_KEY"

#: service.name used when the caller does not pass one.
DEFAULT_SERVICE_NAME = "ratel"

#: Raised when a function needs the OpenTelemetry SDK but the [otlp] extra is absent.
_EXTRA_HINT = (
    "ratel telemetry needs the OpenTelemetry SDK. Install the extra: "
    "pip install 'ratel-ai-telemetry[otlp]'."
)

# Stable attribute used to recognize the provider Ratel installed, including after this
# module is reloaded. Its value is the provider itself, so the check is class-identity-free
# (survives importlib.reload, which rebinds this module's classes).
_PROVIDER_HANDLE_ATTR = "_ratel_ai_telemetry_handle"

# Set True on the provider once its handle's shutdown() runs, so a later init() reports the
# terminal state loudly instead of silently handing back a provider whose exporter is dead.
_PROVIDER_SHUTDOWN_ATTR = "_ratel_ai_telemetry_shutdown"

# The LoggerProvider paired with the tracer-provider handle, plus a readiness marker set
# only after both globals are installed. They make idempotence a two-signal invariant.
_PROVIDER_LOGGER_ATTR = "_ratel_ai_telemetry_logger_provider"
_PROVIDER_READY_ATTR = "_ratel_ai_telemetry_ready"

# The effective service.name stamped on the provider at install, so re-entry can restore
# the module-level record after an importlib.reload wiped it (the provider survives in
# OTel's global; this module's state does not).
_PROVIDER_SERVICE_NAME_ATTR = "_ratel_ai_telemetry_service_name"

# Python callers may initialize from concurrent worker threads. Serialize the set-once global
# registrations so no caller can observe and return a tracer provider before Logs is ready.
_INIT_LOCK = RLock()

# The effective service.name of the provider init() installed, for OTel-free readers
# (the SDK's runtime-events default source_id, ADR-0020). None until an enabled init()
# actually installs the provider, so readers never see a name no telemetry carries.
_recorded_service_name: str | None = None


@dataclass(frozen=True)
class OtlpConfig:
    """Resolved exporter configuration; the pure core of init()."""

    url: str
    logs_url: str
    headers: dict[str, str]
    service_name: str


def resolve_otlp_config(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    logs_endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    service_name: str | None = None,
    env: Mapping[str, str] | None = None,
) -> OtlpConfig:
    """Resolve init() options into concrete exporter config.

    Accepts either api_key= (falling back to RATEL_API_KEY; endpoint defaults to
    RATEL_OTLP_ENDPOINT, then the superseded RATEL_URL; Authorization: Bearer) or
    endpoint=/headers= (custom endpoint / collector). The forms compose: explicit
    endpoint/api_key values win over their environment fallbacks. An explicit api_key sets
    the Bearer header; the RATEL_API_KEY fallback applies only when neither api_key nor an
    explicit Authorization header is given, so ambient env never clobbers auth the caller
    set on purpose. env is injectable so the precedence is testable without a network.
    """
    resolved_env = os.environ if env is None else env
    url = endpoint if endpoint is not None else _resolve_endpoint_from_env(resolved_env)
    if not url:
        raise ValueError(
            f"ratel telemetry init: no endpoint. Pass endpoint= or set {OTLP_ENDPOINT_ENV} "
            f"(use api_key= or {API_KEY_ENV} for Bearer auth)."
        )
    resolved_headers: dict[str, str] = dict(headers or {})
    if api_key:
        resolved_headers["Authorization"] = f"Bearer {api_key}"
    elif resolved_env.get(API_KEY_ENV) and not _has_authorization_header(resolved_headers):
        resolved_headers["Authorization"] = f"Bearer {resolved_env[API_KEY_ENV]}"
    return OtlpConfig(
        url=url,
        logs_url=logs_endpoint or _derive_logs_url(url),
        headers=resolved_headers,
        service_name=service_name or DEFAULT_SERVICE_NAME,
    )


def _resolve_endpoint_from_env(env: Mapping[str, str]) -> str | None:
    """The OTLP traces endpoint from the environment: dedicated var, then legacy.

    Reading the legacy var warns rather than failing, so a patch release never breaks an
    install that only sets RATEL_URL. Warning only on the fallback keeps the common path
    silent, including for callers who set both during a migration.
    """
    url = env.get(OTLP_ENDPOINT_ENV)
    if url:
        return url
    legacy = env.get(LEGACY_ENDPOINT_ENV)
    if legacy:
        warnings.warn(
            f"{LEGACY_ENDPOINT_ENV} as the OTLP endpoint is deprecated and will be dropped "
            f"in the next minor; set {OTLP_ENDPOINT_ENV} instead. {LEGACY_ENDPOINT_ENV} "
            "also selects the SDK's catalog source, which is a different destination.",
            DeprecationWarning,
            stacklevel=3,
        )
    return legacy


def _derive_logs_url(traces_url: str) -> str:
    return re.sub(r"/v1/traces(?=/?(?:[?#]|$))", "/v1/logs", traces_url, count=1)


def _has_authorization_header(headers: Mapping[str, str]) -> bool:
    """Whether the caller already supplied an Authorization header (any casing)."""
    return any(key.lower() == "authorization" for key in headers)


#: Predicate deciding whether a finished span is forwarded to Ratel.
SpanFilter = Callable[["ReadableSpan"], bool]
LogFilter = Callable[["ReadWriteLogRecord"], bool]


def ratel_signal_filter(span: ReadableSpan) -> bool:
    """Default span filter: forward only signal-bearing spans — a ratel.* span name, or
    any attribute key under gen_ai.* / ratel.*.

    This is what lets Ratel share a provider with e.g. Langfuse + the Vercel AI SDK and
    ingest only the gen_ai/ratel signal (the AI SDK's gen_ai.* spans + Ratel's own
    ratel.search / execute_tool), dropping the framework's ai.* wrapper noise.
    """
    if span.name.startswith("ratel."):
        return True
    attributes = span.attributes or {}
    return any(key.startswith("gen_ai.") or key.startswith("ratel.") for key in attributes)


def ratel_event_filter(record: ReadWriteLogRecord) -> bool:
    """Default log filter: forward only named gen_ai.* / ratel.* EventRecords."""
    event_name = record.log_record.event_name
    return bool(
        event_name
        and (event_name.startswith("gen_ai.") or event_name.startswith("ratel."))
    )


def _accept_all_spans(_span: ReadableSpan) -> bool:
    """The filter init() uses: it owns the provider, so it exports every span."""
    return True


def _accept_all_logs(_record: ReadWriteLogRecord) -> bool:
    """The filter init() uses: it owns the provider, so it exports every EventRecord."""
    return True


class _NoOpSpanProcessor:
    """OTel-free disabled-mode processor with the standard lifecycle surface."""

    def on_start(self, span: object, parent_context: object | None = None) -> None:
        return None

    def on_end(self, span: object) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class _NoOpLogRecordProcessor:
    """OTel-free disabled-mode log processor with the standard lifecycle surface."""

    def on_emit(self, log_record: object) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class _NoOpHandle:
    """OTel-free disabled-mode shutdown handle: the same surface as init()'s live return."""

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _owned_provider(provider: object) -> object | None:
    """Return the Ratel-owned provider, if this is ours (identity marker, reload-safe)."""
    handle = getattr(provider, _PROVIDER_HANDLE_ATTR, None)
    return handle if handle is provider else None


def _owned_provider_is_ready(provider: object, logger_provider: object) -> bool:
    return bool(
        _owned_provider(provider) is not None
        and getattr(provider, _PROVIDER_READY_ATTR, False)
        and getattr(provider, _PROVIDER_LOGGER_ATTR, None) is logger_provider
    )


def _foreign_provider_error() -> RuntimeError:
    return RuntimeError(
        "ratel telemetry init(): an OpenTelemetry TracerProvider or LoggerProvider is already "
        "registered globally, so init() (the turnkey path that owns the providers) cannot "
        "take over. To send Ratel telemetry alongside existing providers (e.g. Langfuse + "
        "the Vercel AI SDK), add ratel_span_processor(api_key=...) and "
        "ratel_log_record_processor(api_key=...) to the host providers instead of init()."
    )


def _already_shut_down_error() -> RuntimeError:
    return RuntimeError(
        "ratel telemetry init(): telemetry was already shut down in this process. The "
        "OpenTelemetry global tracer provider is set once per process, so re-initialization "
        "after shutdown is not supported — init() once and shut down only at process exit."
    )


def ratel_span_exporter(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> SpanExporter:
    """Build the OTLP http/protobuf span exporter at the resolved Ratel endpoint.

    The standalone exporter for callers wiring their own span-processor; ratel_span_processor
    batches over it. Carries no resource — the caller's provider owns service.name. Needs the
    [otlp] extra; raises a clear error otherwise.
    """
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_EXTRA_HINT) from exc
    cfg = resolve_otlp_config(api_key=api_key, endpoint=endpoint, headers=headers)
    return OTLPSpanExporter(endpoint=cfg.url, headers=dict(cfg.headers))


def ratel_log_exporter(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    logs_endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> LogRecordExporter:
    """Build the OTLP http/protobuf log exporter at the resolved logs endpoint."""
    try:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_EXTRA_HINT) from exc
    cfg = resolve_otlp_config(
        api_key=api_key,
        endpoint=endpoint,
        logs_endpoint=logs_endpoint,
        headers=headers,
    )
    return OTLPLogExporter(endpoint=cfg.logs_url, headers=dict(cfg.headers))


def ratel_span_processor(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    span_filter: SpanFilter | None = None,
    enabled: bool = True,
) -> SpanProcessor:
    """A BatchSpanProcessor over the Ratel OTLP exporter that forwards only the spans passing
    span_filter (default ratel_signal_filter; pass ``lambda _s: True`` to forward everything).

    Add it to your own provider (``provider.add_span_processor(ratel_span_processor(...))``) to
    send Ratel telemetry alongside another provider — no global side effects, no resource.
    Greenfield apps that want Ratel to own the provider should call init() instead. Pass
    enabled=False for an OTel-free no-op processor that needs no endpoint. Enabled processors
    need the [otlp] extra and raise a clear error without it.
    """
    if not enabled:
        return cast("SpanProcessor", _NoOpSpanProcessor())

    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_EXTRA_HINT) from exc
    active_filter = span_filter or ratel_signal_filter
    exporter = ratel_span_exporter(api_key=api_key, endpoint=endpoint, headers=headers)

    class _RatelSpanProcessor(BatchSpanProcessor):
        def on_end(self, span: ReadableSpan) -> None:
            if active_filter(span):
                super().on_end(span)

    return _RatelSpanProcessor(exporter)


def ratel_log_record_processor(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    logs_endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    log_filter: LogFilter | None = None,
    enabled: bool = True,
) -> LogRecordProcessor:
    """A filtered BatchLogRecordProcessor for a host-owned LoggerProvider."""
    if not enabled:
        return cast("LogRecordProcessor", _NoOpLogRecordProcessor())

    try:
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_EXTRA_HINT) from exc
    active_filter = log_filter or ratel_event_filter
    exporter = ratel_log_exporter(
        api_key=api_key,
        endpoint=endpoint,
        logs_endpoint=logs_endpoint,
        headers=headers,
    )

    class _RatelLogRecordProcessor(BatchLogRecordProcessor):
        def on_emit(self, log_record: ReadWriteLogRecord) -> None:
            if active_filter(log_record):
                super().on_emit(log_record)

    return _RatelLogRecordProcessor(exporter)


def init(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    logs_endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    service_name: str | None = None,
    span_filter: SpanFilter | None = None,
    log_filter: LogFilter | None = None,
    enabled: bool = True,
) -> TelemetryHandle:
    """Wire OTLP http/protobuf exporters + batch processors + one service.name resource,
    register global tracer and logger providers, and return a shutdown handle (call
    handle.shutdown() / handle.force_flush()). Emit through the global OTel APIs.

    init() owns the global provider, so it exports every span by default (unlike
    ratel_span_processor, whose default gen_ai.*/ratel.* filter exists for sharing a provider);
    pass span_filter to narrow it. Repeated calls return the original Ratel-owned handle. It
    raises — pointing at ratel_span_processor — if a foreign provider is already registered
    globally. On first setup, pass enabled=False for an OTel-free no-op handle that needs no
    endpoint; once Ratel owns the provider, repeated calls return it regardless of options.
    Enabled initialization needs the [otlp] extra and raises a clear error without it.

    Shutdown is terminal: after handle.shutdown(), a later init() raises rather than hand back
    the dead provider (OTel's global provider is set once per process). Note that the handle is
    shared across repeated calls, so shutting it down stops export for every caller.
    """
    with _INIT_LOCK:
        return _init_locked(
            api_key=api_key,
            endpoint=endpoint,
            logs_endpoint=logs_endpoint,
            headers=headers,
            service_name=service_name,
            span_filter=span_filter,
            log_filter=log_filter,
            enabled=enabled,
        )


def recorded_service_name() -> str | None:
    """The effective ``service.name`` of the provider init() installed, or None.

    The OTel-free identity seam (ADR-0020): the SDK's runtime-events default
    ``source_id`` reads it so the facts lane and the OTel lane share one identity
    when telemetry is configured programmatically. Recorded only when an enabled
    init() actually installs the provider — a disabled or failed init records
    nothing, and re-initialization returns the original provider's name.
    """
    return _recorded_service_name


def _init_locked(
    *,
    api_key: str | None,
    endpoint: str | None,
    logs_endpoint: str | None,
    headers: Mapping[str, str] | None,
    service_name: str | None,
    span_filter: SpanFilter | None,
    log_filter: LogFilter | None,
    enabled: bool,
) -> TelemetryHandle:
    if not enabled:
        # Preserve an already-active, live Ratel provider without importing OTel on the
        # genuinely disabled/base-only path. A prior successful init loaded this module already.
        trace_module = sys.modules.get("opentelemetry.trace")
        logs_module = sys.modules.get("opentelemetry._logs")
        if trace_module is not None and logs_module is not None:
            current_provider = trace_module.get_tracer_provider()
            current_logger_provider = logs_module.get_logger_provider()
            if _owned_provider_is_ready(current_provider, current_logger_provider) and not getattr(
                current_provider, _PROVIDER_SHUTDOWN_ATTR, False
            ):
                return cast("TelemetryHandle", current_provider)
        return _NoOpHandle()

    try:
        from opentelemetry import _logs, trace
        from opentelemetry._logs._internal import ProxyLoggerProvider
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import ProxyTracerProvider
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_EXTRA_HINT) from exc
    # get_tracer_provider() returns the default ProxyTracerProvider until someone installs a
    # real one. Re-entry returns the exact handle Ratel installed; any other real provider
    # belongs to the caller and remains a loud composition error.
    current_provider = trace.get_tracer_provider()
    owned = _owned_provider(current_provider)
    if owned is not None:
        if getattr(owned, _PROVIDER_SHUTDOWN_ATTR, False):
            raise _already_shut_down_error()
        if not _owned_provider_is_ready(current_provider, _logs.get_logger_provider()):
            raise _foreign_provider_error()
        _restore_recorded_service_name(owned)
        return cast("TelemetryHandle", owned)
    if not isinstance(current_provider, ProxyTracerProvider):
        raise _foreign_provider_error()
    if not isinstance(_logs.get_logger_provider(), ProxyLoggerProvider):
        raise _foreign_provider_error()
    cfg = resolve_otlp_config(
        api_key=api_key,
        endpoint=endpoint,
        logs_endpoint=logs_endpoint,
        headers=headers,
        service_name=service_name,
    )
    resource = Resource.create({SERVICE_NAME: cfg.service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        ratel_span_processor(
            api_key=api_key,
            endpoint=endpoint,
            headers=headers,
            span_filter=_accept_all_spans if span_filter is None else span_filter,
        )
    )
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        ratel_log_record_processor(
            api_key=api_key,
            endpoint=endpoint,
            logs_endpoint=logs_endpoint,
            headers=headers,
            log_filter=_accept_all_logs if log_filter is None else log_filter,
        )
    )
    setattr(provider, _PROVIDER_HANDLE_ATTR, provider)
    setattr(provider, _PROVIDER_LOGGER_ATTR, logger_provider)
    _attach_logger_lifecycle(provider, logger_provider)
    trace.set_tracer_provider(provider)
    winner = trace.get_tracer_provider()
    if winner is not provider:
        # Lost a first-init race (OTel's global is set-once). If the winner is Ratel's own
        # provider, honor idempotence and return it; only a truly foreign winner is an error.
        provider.shutdown()
        owned_winner = _owned_provider(winner)
        if owned_winner is not None and _owned_provider_is_ready(
            winner, _logs.get_logger_provider()
        ):
            _restore_recorded_service_name(owned_winner)
            return cast("TelemetryHandle", owned_winner)
        raise _foreign_provider_error()
    _logs.set_logger_provider(logger_provider)
    if _logs.get_logger_provider() is not logger_provider:
        provider.shutdown()
        raise _foreign_provider_error()
    setattr(provider, _PROVIDER_READY_ATTR, True)
    setattr(provider, _PROVIDER_SERVICE_NAME_ATTR, cfg.service_name)
    _restore_recorded_service_name(provider)
    return cast("TelemetryHandle", provider)


def _restore_recorded_service_name(provider: object) -> None:
    """Sync the module-level record from the name stamped on the installed provider."""
    global _recorded_service_name
    stamped = getattr(provider, _PROVIDER_SERVICE_NAME_ATTR, None)
    if isinstance(stamped, str):
        _recorded_service_name = stamped


def _attach_logger_lifecycle(provider: object, logger_provider: object) -> None:
    """Make the tracer-provider handle flush and shut down both signal providers.

    One-time, single-owner wrap set at provider creation (unlike the removed multi-generation
    shutdown mutation). It also records terminal state so a later init() fails loud instead of
    returning providers whose exporters are already stopped.
    """
    shutdown_name = "shutdown"
    force_flush_name = "force_flush"
    original_shutdown = getattr(provider, shutdown_name)
    original_force_flush = getattr(provider, force_flush_name)
    logger_shutdown = getattr(logger_provider, shutdown_name)
    logger_force_flush = getattr(logger_provider, force_flush_name)

    def _shutdown() -> Any:
        failure: BaseException | None = None
        try:
            original_shutdown()
        except BaseException as exc:
            failure = exc
        try:
            logger_shutdown()
        except BaseException as exc:
            if failure is None:
                failure = exc
        finally:
            setattr(provider, _PROVIDER_SHUTDOWN_ATTR, True)
        if failure is not None:
            raise failure
        return None

    def _force_flush(timeout_millis: int = 30_000) -> bool:
        trace_flushed = bool(original_force_flush(timeout_millis))
        logs_flushed = bool(logger_force_flush(timeout_millis))
        return trace_flushed and logs_flushed

    setattr(provider, shutdown_name, _shutdown)
    setattr(provider, force_flush_name, _force_flush)


class ContentCapture(str, Enum):
    """Message/tool content capture modes for
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT (CONVENTIONS.md § Capture
    gating). Default off.
    """

    NO_CONTENT = "NO_CONTENT"
    SPAN_ONLY = "SPAN_ONLY"
    EVENT_ONLY = "EVENT_ONLY"
    SPAN_AND_EVENT = "SPAN_AND_EVENT"


#: Module-level programmatic override; None means unset (env-driven).
_content_capture_override: ContentCapture | None = None

#: Monotonically increasing token identifying the most recent set_content_capture call,
#: so a stale holder (e.g. an old telemetry handle's shutdown) cannot clear an override a
#: newer caller owns via clear_content_capture.
_content_capture_generation = 0


def _parse_content_capture(raw: str) -> ContentCapture | None:
    """The single normalizer for capture-mode strings, shared by the env parser and the
    programmatic setter: trim + case-insensitive, with the legacy boolean forms (true/1
    -> full capture, false/0 -> none). None when unrecognized — the two callers diverge
    there (content_capture_mode defaults, set_content_capture raises).
    """
    normalized = raw.strip().upper()
    if normalized in ("NO_CONTENT", "FALSE", "0"):
        return ContentCapture.NO_CONTENT
    if normalized == "SPAN_ONLY":
        return ContentCapture.SPAN_ONLY
    if normalized == "EVENT_ONLY":
        return ContentCapture.EVENT_ONLY
    if normalized in ("SPAN_AND_EVENT", "TRUE", "1"):
        return ContentCapture.SPAN_AND_EVENT
    return None


def set_content_capture(mode: ContentCapture | str | None) -> int:
    """Programmatically set the content-capture mode. While set, content_capture_mode()
    returns this mode regardless of OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT —
    programmatic config wins over the environment, matching how OpenTelemetry treats env
    vars as the fallback for code-level configuration. Pass None to clear the override
    unconditionally and return to env-driven parsing.

    The mode is validated like the env var (case-insensitive, legacy true/false/1/0
    accepted) and raises a ValueError on anything unrecognized — failing loud at config
    time instead of storing a value that would both disable capture and mask the env var.

    Returns a generation token identifying this call as the current owner of the override;
    pass it to clear_content_capture to clear only if no newer set has happened since (the
    safe form for shutdown/teardown hooks).
    """
    global _content_capture_override, _content_capture_generation
    if mode is None:
        _content_capture_override = None
        _content_capture_generation += 1
        return _content_capture_generation
    parsed = _parse_content_capture(mode)
    if parsed is None:
        valid = ", ".join(member.value for member in ContentCapture)
        raise ValueError(
            f"set_content_capture: unrecognized mode {mode!r}. Valid values: {valid} "
            "(case-insensitive; legacy true/false/1/0 also accepted), or None to clear."
        )
    _content_capture_override = parsed
    _content_capture_generation += 1
    return _content_capture_generation


def clear_content_capture(generation: int) -> None:
    """Clear the programmatic content-capture override, but only when generation — the
    token returned by set_content_capture — still identifies the most recent set. A stale
    token no-ops, so an old handle shutting down late cannot clobber an override a newer
    caller installed (and silently re-enable, or disable, capture via the env fallback).
    For an unconditional clear, use set_content_capture(None).
    """
    global _content_capture_override
    if generation != _content_capture_generation:  # a newer set owns the slot
        return
    _content_capture_override = None


def content_capture_mode(env: Mapping[str, str] | None = None) -> ContentCapture:
    """Parse the ecosystem content-capture gate. A mode set via set_content_capture wins
    outright (env is the fallback, as in OTel); otherwise defaults to NO_CONTENT when
    unset/empty/unrecognized. The legacy boolean form maps true to full capture
    (SPAN_AND_EVENT) and false to none.
    """
    if _content_capture_override is not None:
        return _content_capture_override
    resolved_env = os.environ if env is None else env
    raw = resolved_env.get(CAPTURE_CONTENT_ENV)
    if raw is None or raw.strip() == "":
        return ContentCapture.NO_CONTENT
    parsed = _parse_content_capture(raw)
    return parsed if parsed is not None else ContentCapture.NO_CONTENT
