"""Contract-against-the-pin conformance, driven by the shared ../../conformance/fixtures.json.

Each fixture is built into a span from this helper's own ratel.* constants through the real
OTel SDK; the emitted span must match the fixture's expected wire name + attributes exactly.
The same fixtures drive the TS helper, so the two cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ratel_ai_telemetry import (
    EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_ID,
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
    SEMCONV_VERSION,
)

_FIXTURES = json.loads(
    (Path(__file__).resolve().parents[2] / "conformance" / "fixtures.json").read_text()
)

# Logical span id -> the span-name constant under test.
SPAN_NAME = {
    "execute_tool": EXECUTE_TOOL,
    "ratel_search": RATEL_SEARCH,
    "ratel_skill_load": RATEL_SKILL_LOAD,
    "ratel_upstream_register": RATEL_UPSTREAM_REGISTER,
    "ratel_auth_flow": RATEL_AUTH_FLOW,
}

# Logical attribute id -> the attribute-key constant under test.
ATTR_KEY = {
    "gen_ai_operation_name": GEN_AI_OPERATION_NAME,
    "gen_ai_tool_name": GEN_AI_TOOL_NAME,
    "gen_ai_tool_call_id": GEN_AI_TOOL_CALL_ID,
    "gen_ai_tool_call_arguments": GEN_AI_TOOL_CALL_ARGUMENTS,
    "gen_ai_tool_call_result": GEN_AI_TOOL_CALL_RESULT,
    "ratel_origin": RATEL_ORIGIN,
    "ratel_tool_args_size_bytes": RATEL_TOOL_ARGS_SIZE_BYTES,
    "ratel_upstream_server": RATEL_UPSTREAM_SERVER,
    "ratel_search_target": RATEL_SEARCH_TARGET,
    "ratel_search_top_k": RATEL_SEARCH_TOP_K,
    "ratel_search_hit_count": RATEL_SEARCH_HIT_COUNT,
    "ratel_search_query": RATEL_SEARCH_QUERY,
    "ratel_skill_id": RATEL_SKILL_ID,
    "ratel_upstream_transport": RATEL_UPSTREAM_TRANSPORT,
    "ratel_upstream_tool_count": RATEL_UPSTREAM_TOOL_COUNT,
    "ratel_auth_outcome": RATEL_AUTH_OUTCOME,
}

# Logical event id -> the event-name constant under test.
EVENT_NAME = {
    "ratel_search_results": RATEL_SEARCH_RESULTS,
    "ratel_tool_execution_details": RATEL_TOOL_EXECUTION_DETAILS,
}


def test_fixtures_share_the_pinned_semconv_version() -> None:
    assert _FIXTURES["semconv_version"] == SEMCONV_VERSION


@pytest.mark.parametrize("fixture", _FIXTURES["fixtures"], ids=lambda f: f["name"])
def test_fixture_emits_pinned_keys(fixture: dict[str, Any]) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("conformance")
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    _logs.set_logger_provider(logger_provider)
    logger = _logs.get_logger("conformance")

    span = tracer.start_span(SPAN_NAME[fixture["span"]])
    for field, value in fixture["set"].items():
        span.set_attribute(ATTR_KEY[field], value)
    for event in fixture.get("emit_events", []):
        attributes = {
            ATTR_KEY[field]: value for field, value in event["attributes"].items()
        }
        logger.emit(event_name=EVENT_NAME[event["event"]], attributes=attributes)
    span.end()

    emitted = exporter.get_finished_spans()
    assert len(emitted) == 1
    assert emitted[0].name == fixture["expect_name"]
    assert dict(emitted[0].attributes or {}) == fixture["expect_attributes"]
    events = [
        {
            "name": readable.log_record.event_name,
            "attributes": json.loads(
                json.dumps(dict(readable.log_record.attributes or {}))
            ),
        }
        for readable in log_exporter.get_finished_logs()
    ]
    assert events == fixture.get("expect_events", [])
    provider.shutdown()
    logger_provider.shutdown()
