"""Shared test fixtures for the telemetry helper."""

from __future__ import annotations

import pytest
from opentelemetry import trace as trace_api
from opentelemetry._logs import _internal as logs_api
from opentelemetry.util._once import Once


def _reset_trace_globals() -> None:
    # init()'s loud-fail guard throws when a provider is already registered, and OTel's
    # global tracer provider is process-wide set-once — so reset it around every test to
    # keep cases (and other test modules) from leaking a provider into each other.
    trace_api._TRACER_PROVIDER_SET_ONCE = Once()
    trace_api._TRACER_PROVIDER = None


def _reset_log_globals() -> None:
    logs_api._LOGGER_PROVIDER_SET_ONCE = Once()
    logs_api._LOGGER_PROVIDER = None


@pytest.fixture(autouse=True)
def reset_otel_providers() -> object:
    _reset_trace_globals()
    _reset_log_globals()
    yield
    _reset_trace_globals()
    _reset_log_globals()
