"""Contract-against-the-pin tests for the ratel.* telemetry vocabulary.

Each constant is asserted against the vocabulary pinned in ../CONVENTIONS.md.
"""

from ratel_ai_telemetry import (
    CAPTURE_CONTENT_ENV,
    EXECUTE_TOOL,
    GEN_AI_INFERENCE_DETAILS,
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
    AuthOutcome,
    Origin,
    SearchTarget,
)


def test_pins_the_otel_gen_ai_semconv_version() -> None:
    assert SEMCONV_VERSION == "1.42.0"


def test_gates_content_capture_on_the_ecosystem_env_var() -> None:
    assert CAPTURE_CONTENT_ENV == "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def test_names_the_ratel_spans_per_the_pin() -> None:
    assert RATEL_SEARCH == "ratel.search"
    assert RATEL_SKILL_LOAD == "ratel.skill.load"
    assert RATEL_UPSTREAM_REGISTER == "ratel.upstream.register"
    assert RATEL_AUTH_FLOW == "ratel.auth.flow"


def test_names_the_event_records_per_the_pin() -> None:
    assert RATEL_SEARCH_RESULTS == "ratel.search.results"
    assert RATEL_TOOL_EXECUTION_DETAILS == "ratel.tool.execution.details"
    assert GEN_AI_INFERENCE_DETAILS == "gen_ai.client.inference.operation.details"


def test_tool_invocation_is_gen_ai_execute_tool_not_ratel_invoke() -> None:
    assert EXECUTE_TOOL == "execute_tool"
    assert EXECUTE_TOOL != "ratel.invoke"


def test_ratel_attribute_keys_match_the_pin() -> None:
    assert RATEL_ORIGIN == "ratel.origin"
    assert RATEL_SEARCH_TARGET == "ratel.search.target"
    assert RATEL_SEARCH_TOP_K == "ratel.search.top_k"
    assert RATEL_SEARCH_HIT_COUNT == "ratel.search.hit_count"
    assert RATEL_SEARCH_QUERY == "ratel.search.query"
    assert RATEL_TOOL_ARGS_SIZE_BYTES == "ratel.tool.args_size_bytes"
    assert RATEL_UPSTREAM_SERVER == "ratel.upstream.server"
    assert RATEL_UPSTREAM_TRANSPORT == "ratel.upstream.transport"
    assert RATEL_UPSTREAM_TOOL_COUNT == "ratel.upstream.tool_count"
    assert RATEL_SKILL_ID == "ratel.skill.id"
    assert RATEL_AUTH_OUTCOME == "ratel.auth.outcome"


def test_gen_ai_interop_keys_stay_under_gen_ai_never_renamed_into_ratel() -> None:
    assert GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert GEN_AI_TOOL_NAME == "gen_ai.tool.name"
    assert GEN_AI_TOOL_CALL_ID == "gen_ai.tool.call.id"
    assert GEN_AI_TOOL_CALL_ARGUMENTS == "gen_ai.tool.call.arguments"
    assert GEN_AI_TOOL_CALL_RESULT == "gen_ai.tool.call.result"
    for key in (
        GEN_AI_OPERATION_NAME,
        GEN_AI_TOOL_NAME,
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
    ):
        assert key.startswith("gen_ai.")
        assert not key.startswith("ratel.")


def test_every_ratel_attribute_key_is_namespaced() -> None:
    for key in (
        RATEL_ORIGIN,
        RATEL_SEARCH_TARGET,
        RATEL_SEARCH_TOP_K,
        RATEL_SEARCH_HIT_COUNT,
        RATEL_SEARCH_QUERY,
        RATEL_TOOL_ARGS_SIZE_BYTES,
        RATEL_UPSTREAM_SERVER,
        RATEL_UPSTREAM_TRANSPORT,
        RATEL_UPSTREAM_TOOL_COUNT,
        RATEL_SKILL_ID,
        RATEL_AUTH_OUTCOME,
    ):
        assert key.startswith("ratel.")


def test_attribute_keys_are_unique() -> None:
    keys = [
        RATEL_ORIGIN,
        RATEL_SEARCH_TARGET,
        RATEL_SEARCH_TOP_K,
        RATEL_SEARCH_HIT_COUNT,
        RATEL_SEARCH_QUERY,
        RATEL_TOOL_ARGS_SIZE_BYTES,
        RATEL_UPSTREAM_SERVER,
        RATEL_UPSTREAM_TRANSPORT,
        RATEL_UPSTREAM_TOOL_COUNT,
        RATEL_SKILL_ID,
        RATEL_AUTH_OUTCOME,
        GEN_AI_OPERATION_NAME,
        GEN_AI_TOOL_NAME,
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
    ]
    assert len(set(keys)) == len(keys)


def test_origin_maps_to_wire_strings() -> None:
    assert Origin.DIRECT == "direct"
    assert Origin.AGENT == "agent"


def test_search_target_maps_to_wire_strings() -> None:
    assert SearchTarget.TOOL == "tool"
    assert SearchTarget.SKILL == "skill"


def test_auth_outcome_maps_to_wire_strings() -> None:
    assert AuthOutcome.OK == "ok"
    assert AuthOutcome.REFRESHED == "refreshed"
    assert AuthOutcome.NEEDS_AUTH == "needs_auth"
    assert AuthOutcome.FAILED == "failed"
