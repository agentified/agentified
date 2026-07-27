"""Tests for the composable OTLP span-processor (coexistence with another provider)."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from ratel_ai_telemetry.otlp import (
    ENDPOINT_ENV,
    init,
    ratel_event_filter,
    ratel_log_exporter,
    ratel_log_record_processor,
    ratel_signal_filter,
    ratel_span_exporter,
    ratel_span_processor,
)

ENDPOINT = "http://localhost:4318/v1/traces"


def span(name: str, attributes: dict[str, Any] | None = None) -> Any:
    # The filter reads only a span's name + attribute keys, so a minimal shape suffices.
    return SimpleNamespace(name=name, attributes=attributes or {})


def event(event_name: str | None = None) -> Any:
    return SimpleNamespace(
        log_record=SimpleNamespace(event_name=event_name, attributes={})
    )


class TestRatelEventFilter:
    def test_forwards_only_gen_ai_and_ratel_event_records(self) -> None:
        assert ratel_event_filter(event("ratel.search.results"))
        assert ratel_event_filter(event("gen_ai.client.inference.operation.details"))
        assert not ratel_event_filter(event("app.started"))
        assert not ratel_event_filter(event())


class TestRatelSignalFilter:
    def test_forwards_ratel_named_spans(self) -> None:
        assert ratel_signal_filter(span("ratel.search"))
        assert ratel_signal_filter(span("ratel.skill.load"))

    def test_forwards_spans_with_gen_ai_or_ratel_attributes(self) -> None:
        assert ratel_signal_filter(span("execute_tool", {"gen_ai.operation.name": "execute_tool"}))
        assert ratel_signal_filter(span("execute_tool", {"ratel.origin": "agent"}))
        assert ratel_signal_filter(span("chat gpt-4o", {"gen_ai.request.model": "gpt-4o"}))

    def test_drops_spans_without_the_signal(self) -> None:
        assert not ratel_signal_filter(span("ai.generate_text", {"ai.model.id": "gpt-4o"}))
        assert not ratel_signal_filter(span("GET /health"))


class TestRatelSpanExporter:
    def test_builds_an_otlp_exporter(self) -> None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        assert isinstance(ratel_span_exporter(endpoint=ENDPOINT), OTLPSpanExporter)

    def test_raises_without_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENDPOINT_ENV, raising=False)
        with pytest.raises(ValueError, match=ENDPOINT_ENV):
            ratel_span_exporter(api_key="k")


class TestRatelLogExporter:
    def test_builds_an_otlp_logs_exporter(self) -> None:
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        assert isinstance(ratel_log_exporter(endpoint=ENDPOINT), OTLPLogExporter)


class TestRatelSpanProcessor:
    def test_disabled_processor_is_a_noop_without_config_or_otel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENDPOINT_ENV, raising=False)
        real_import = builtins.__import__

        def block_otel(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "opentelemetry" or name.startswith("opentelemetry."):
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_otel)
        proc = ratel_span_processor(enabled=False)

        proc.on_start(span("ratel.search"))
        proc.on_end(span("ratel.search"))
        assert proc.force_flush()
        assert proc.shutdown() is None

    def test_forwards_only_signal_bearing_spans_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        forwarded: list[str] = []
        monkeypatch.setattr(BatchSpanProcessor, "on_end", lambda _self, s: forwarded.append(s.name))
        proc = ratel_span_processor(endpoint=ENDPOINT)
        proc.on_end(span("ratel.search"))
        proc.on_end(span("execute_tool", {"gen_ai.operation.name": "execute_tool"}))
        proc.on_end(span("ai.generate_text", {"ai.model.id": "gpt-4o"}))
        proc.shutdown()
        assert forwarded == ["ratel.search", "execute_tool"]

    def test_respects_a_custom_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        forwarded: list[str] = []
        monkeypatch.setattr(BatchSpanProcessor, "on_end", lambda _self, s: forwarded.append(s.name))
        proc = ratel_span_processor(endpoint=ENDPOINT, span_filter=lambda _s: True)
        proc.on_end(span("ai.generate_text", {"ai.model.id": "gpt-4o"}))
        proc.shutdown()
        assert forwarded == ["ai.generate_text"]


class TestRatelLogRecordProcessor:
    def test_forwards_only_signal_events_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        forwarded: list[str | None] = []
        monkeypatch.setattr(
            BatchLogRecordProcessor,
            "on_emit",
            lambda _self, record: forwarded.append(record.log_record.event_name),
        )
        processor = ratel_log_record_processor(endpoint=ENDPOINT)
        processor.on_emit(event("ratel.search.results"))
        processor.on_emit(event("gen_ai.client.inference.operation.details"))
        processor.on_emit(event("app.started"))
        processor.shutdown()

        assert forwarded == [
            "ratel.search.results",
            "gen_ai.client.inference.operation.details",
        ]


class TestInitGuard:
    def test_raises_pointing_at_the_processor_when_a_provider_is_registered(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        trace.set_tracer_provider(TracerProvider())
        with pytest.raises(RuntimeError, match="ratel_span_processor"):
            init(endpoint=ENDPOINT)

    def test_rejects_a_foreign_logger_provider(self) -> None:
        from opentelemetry import _logs
        from opentelemetry.sdk._logs import LoggerProvider

        _logs.set_logger_provider(LoggerProvider())
        with pytest.raises(RuntimeError, match="ratel_log_record_processor"):
            init(endpoint=ENDPOINT)


def test_new_processor_surface_is_importable_top_level() -> None:
    """The coexistence surface resolves through the lazy top-level accessor (ADR-0007)."""
    import ratel_ai_telemetry
    from ratel_ai_telemetry.otlp import ratel_log_record_processor as otlp_log_processor
    from ratel_ai_telemetry.otlp import ratel_span_processor as otlp_processor

    assert ratel_ai_telemetry.ratel_span_processor is otlp_processor
    assert ratel_ai_telemetry.ratel_log_record_processor is otlp_log_processor
