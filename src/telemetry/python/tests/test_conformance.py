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
    RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
    RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
    RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
    RATEL_EXPERIMENT_AGREEMENT_K,
    RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
    RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
    RATEL_EXPERIMENT_AGREEMENT_TOP1,
    RATEL_EXPERIMENT_ARM,
    RATEL_EXPERIMENT_COLD,
    RATEL_EXPERIMENT_COMPARISON,
    RATEL_EXPERIMENT_DROP,
    RATEL_EXPERIMENT_DROP_REASON,
    RATEL_EXPERIMENT_DURATION_MS,
    RATEL_EXPERIMENT_EFFECTIVE_ARM,
    RATEL_EXPERIMENT_FALLBACK,
    RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
    RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
    RATEL_EXPERIMENT_HIT_COUNT,
    RATEL_EXPERIMENT_ID,
    RATEL_EXPERIMENT_INVOCATION,
    RATEL_EXPERIMENT_INVOCATION_AGE_MS,
    RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
    RATEL_EXPERIMENT_INVOCATION_RANK,
    RATEL_EXPERIMENT_OUTCOME,
    RATEL_EXPERIMENT_OUTCOME_LABEL,
    RATEL_EXPERIMENT_OUTCOME_SCORE,
    RATEL_EXPERIMENT_RANKING_ERROR,
    RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
    RATEL_EXPERIMENT_RESULT_ATTRS,
    RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
    RATEL_EXPERIMENT_RESULT_IDS,
    RATEL_EXPERIMENT_RESULT_SCORES,
    RATEL_EXPERIMENT_RESULTS,
    RATEL_EXPERIMENT_ROLE,
    RATEL_EXPERIMENT_SELECTION_ID,
    RATEL_EXPERIMENT_SERVED_ARM,
    RATEL_EXPERIMENT_SERVED_DURATION_MS,
    RATEL_EXPERIMENT_SERVED_HIT_COUNT,
    RATEL_EXPERIMENT_SERVED_OUTCOME,
    RATEL_EXPERIMENT_SHADOW_ARM,
    RATEL_EXPERIMENT_SHADOW_DURATION_MS,
    RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
    RATEL_EXPERIMENT_SHADOW_OUTCOME,
    RATEL_EXPERIMENT_SKIP,
    RATEL_EXPERIMENT_SKIP_ARM,
    RATEL_EXPERIMENT_SKIP_CONCURRENCY,
    RATEL_EXPERIMENT_SKIP_REASON,
    RATEL_EXPERIMENT_TURN,
    RATEL_EXPERIMENT_UNIT,
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
    "ratel_experiment_arm": RATEL_EXPERIMENT_ARM,
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
    "ratel_experiment_agreement_exact_order": RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
    "ratel_experiment_agreement_item_attrs": RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
    "ratel_experiment_agreement_jaccard_at_k": RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
    "ratel_experiment_agreement_k": RATEL_EXPERIMENT_AGREEMENT_K,
    "ratel_experiment_agreement_overlap_count": RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
    "ratel_experiment_agreement_result_attrs": RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
    "ratel_experiment_agreement_top1": RATEL_EXPERIMENT_AGREEMENT_TOP1,
    "ratel_experiment_arm": RATEL_EXPERIMENT_ARM,
    "ratel_experiment_cold": RATEL_EXPERIMENT_COLD,
    "ratel_experiment_drop_reason": RATEL_EXPERIMENT_DROP_REASON,
    "ratel_experiment_duration_ms": RATEL_EXPERIMENT_DURATION_MS,
    "ratel_experiment_effective_arm": RATEL_EXPERIMENT_EFFECTIVE_ARM,
    "ratel_experiment_fallback_effective_arm": RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
    "ratel_experiment_fallback_reused_shadow": RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
    "ratel_experiment_hit_count": RATEL_EXPERIMENT_HIT_COUNT,
    "ratel_experiment_id": RATEL_EXPERIMENT_ID,
    "ratel_experiment_invocation_age_ms": RATEL_EXPERIMENT_INVOCATION_AGE_MS,
    "ratel_experiment_invocation_attributed": RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
    "ratel_experiment_invocation_rank": RATEL_EXPERIMENT_INVOCATION_RANK,
    "ratel_experiment_outcome": RATEL_EXPERIMENT_OUTCOME,
    "ratel_experiment_outcome_label": RATEL_EXPERIMENT_OUTCOME_LABEL,
    "ratel_experiment_outcome_score": RATEL_EXPERIMENT_OUTCOME_SCORE,
    "ratel_experiment_ranking_error": RATEL_EXPERIMENT_RANKING_ERROR,
    "ratel_experiment_result_attributes_error": RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
    "ratel_experiment_result_attrs": RATEL_EXPERIMENT_RESULT_ATTRS,
    "ratel_experiment_result_attrs_encoding_error": RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
    "ratel_experiment_result_ids": RATEL_EXPERIMENT_RESULT_IDS,
    "ratel_experiment_result_scores": RATEL_EXPERIMENT_RESULT_SCORES,
    "ratel_experiment_role": RATEL_EXPERIMENT_ROLE,
    "ratel_experiment_served_arm": RATEL_EXPERIMENT_SERVED_ARM,
    "ratel_experiment_served_duration_ms": RATEL_EXPERIMENT_SERVED_DURATION_MS,
    "ratel_experiment_served_hit_count": RATEL_EXPERIMENT_SERVED_HIT_COUNT,
    "ratel_experiment_served_outcome": RATEL_EXPERIMENT_SERVED_OUTCOME,
    "ratel_experiment_selection_id": RATEL_EXPERIMENT_SELECTION_ID,
    "ratel_experiment_shadow_arm": RATEL_EXPERIMENT_SHADOW_ARM,
    "ratel_experiment_shadow_duration_ms": RATEL_EXPERIMENT_SHADOW_DURATION_MS,
    "ratel_experiment_shadow_hit_count": RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
    "ratel_experiment_shadow_outcome": RATEL_EXPERIMENT_SHADOW_OUTCOME,
    "ratel_experiment_skip_arm": RATEL_EXPERIMENT_SKIP_ARM,
    "ratel_experiment_skip_concurrency": RATEL_EXPERIMENT_SKIP_CONCURRENCY,
    "ratel_experiment_skip_reason": RATEL_EXPERIMENT_SKIP_REASON,
    "ratel_experiment_turn": RATEL_EXPERIMENT_TURN,
    "ratel_experiment_unit": RATEL_EXPERIMENT_UNIT,
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
    "ratel_experiment_comparison": RATEL_EXPERIMENT_COMPARISON,
    "ratel_experiment_drop": RATEL_EXPERIMENT_DROP,
    "ratel_experiment_fallback": RATEL_EXPERIMENT_FALLBACK,
    "ratel_experiment_invocation": RATEL_EXPERIMENT_INVOCATION,
    "ratel_experiment_outcome": RATEL_EXPERIMENT_OUTCOME,
    "ratel_experiment_results": RATEL_EXPERIMENT_RESULTS,
    "ratel_experiment_skip": RATEL_EXPERIMENT_SKIP,
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
