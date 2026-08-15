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
    RATEL_CATALOG_CONTENT_HASH,
    RATEL_CATALOG_DEFINITION,
    RATEL_CATALOG_DESCRIPTION,
    RATEL_CATALOG_ID,
    RATEL_CATALOG_INPUT_SCHEMA,
    RATEL_CATALOG_KIND,
    RATEL_CATALOG_NAME,
    RATEL_CATALOG_OUTPUT_SCHEMA,
    RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
    RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
    RATEL_CATALOG_TAGS,
    RATEL_EVENT_ID,
    RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
    RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
    RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
    RATEL_EXPERIMENT_AGREEMENT_K,
    RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
    RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
    RATEL_EXPERIMENT_AGREEMENT_TOP1,
    RATEL_EXPERIMENT_ARM,
    RATEL_EXPERIMENT_ARM_BAGGAGE_KEY,
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
    RATEL_EXPERIMENT_ID_BAGGAGE_KEY,
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
    RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY,
    RATEL_EXPERIMENT_SELECTION_ID,
    RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY,
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
    RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY,
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
    ExperimentArmOutcome,
    ExperimentArmRole,
    ExperimentDropReason,
    ExperimentSkipReason,
    Origin,
    SearchTarget,
)


def test_pins_the_otel_gen_ai_semconv_version() -> None:
    assert SEMCONV_VERSION == "1.42.0"


def test_gates_content_capture_on_the_ecosystem_env_var() -> None:
    assert CAPTURE_CONTENT_ENV == "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def test_names_the_ratel_spans_per_the_pin() -> None:
    assert RATEL_SEARCH == "ratel.search"
    assert RATEL_EXPERIMENT_ARM == "ratel.experiment.arm"
    assert RATEL_SKILL_LOAD == "ratel.skill.load"
    assert RATEL_UPSTREAM_REGISTER == "ratel.upstream.register"
    assert RATEL_AUTH_FLOW == "ratel.auth.flow"


def test_names_the_runtime_event_join_key() -> None:
    assert RATEL_EVENT_ID == "ratel.event.id"


def test_names_the_event_records_per_the_pin() -> None:
    assert RATEL_CATALOG_DEFINITION == "ratel.catalog.definition"
    assert RATEL_EXPERIMENT_RESULTS == "ratel.experiment.results"
    assert RATEL_EXPERIMENT_COMPARISON == "ratel.experiment.comparison"
    assert RATEL_EXPERIMENT_SKIP == "ratel.experiment.skip"
    assert RATEL_EXPERIMENT_FALLBACK == "ratel.experiment.fallback"
    assert RATEL_EXPERIMENT_DROP == "ratel.experiment.drop"
    assert RATEL_EXPERIMENT_INVOCATION == "ratel.experiment.invocation"
    assert RATEL_EXPERIMENT_OUTCOME == "ratel.experiment.outcome"
    assert RATEL_SEARCH_RESULTS == "ratel.search.results"
    assert RATEL_TOOL_EXECUTION_DETAILS == "ratel.tool.execution.details"
    assert GEN_AI_INFERENCE_DETAILS == "gen_ai.client.inference.operation.details"


def test_catalog_definition_attribute_vocabulary() -> None:
    assert [
        RATEL_CATALOG_KIND,
        RATEL_CATALOG_ID,
        RATEL_CATALOG_NAME,
        RATEL_CATALOG_DESCRIPTION,
        RATEL_CATALOG_TAGS,
        RATEL_CATALOG_INPUT_SCHEMA,
        RATEL_CATALOG_OUTPUT_SCHEMA,
        RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
        RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
        RATEL_CATALOG_CONTENT_HASH,
    ] == [
        "ratel.catalog.kind",
        "ratel.catalog.id",
        "ratel.catalog.name",
        "ratel.catalog.description",
        "ratel.catalog.tags",
        "ratel.catalog.input_schema",
        "ratel.catalog.output_schema",
        "ratel.catalog.searchable_description",
        "ratel.catalog.searchable_description_overridden",
        "ratel.catalog.content_hash",
    ]


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
    assert RATEL_EXPERIMENT_ID == "ratel.experiment.id"
    assert RATEL_EXPERIMENT_SELECTION_ID == "ratel.experiment.selection_id"
    assert RATEL_EXPERIMENT_ARM == "ratel.experiment.arm"
    assert RATEL_EXPERIMENT_ROLE == "ratel.experiment.role"
    assert RATEL_EXPERIMENT_UNIT == "ratel.experiment.unit"
    assert RATEL_EXPERIMENT_COLD == "ratel.experiment.cold"
    assert RATEL_EXPERIMENT_OUTCOME == "ratel.experiment.outcome"
    assert RATEL_EXPERIMENT_DURATION_MS == "ratel.experiment.duration_ms"
    assert RATEL_EXPERIMENT_HIT_COUNT == "ratel.experiment.hit_count"
    assert RATEL_EXPERIMENT_RANKING_ERROR == "ratel.experiment.ranking_error"
    assert (
        RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR
        == "ratel.experiment.result_attributes_error"
    )
    assert (
        RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR
        == "ratel.experiment.result_attrs_encoding_error"
    )
    assert RATEL_EXPERIMENT_RESULT_IDS == "ratel.experiment.result_ids"
    assert RATEL_EXPERIMENT_RESULT_SCORES == "ratel.experiment.result_scores"
    assert RATEL_EXPERIMENT_RESULT_ATTRS == "ratel.experiment.result_attrs"
    assert RATEL_EXPERIMENT_SERVED_ARM == "ratel.experiment.served.arm"
    assert RATEL_EXPERIMENT_SERVED_OUTCOME == "ratel.experiment.served.outcome"
    assert RATEL_EXPERIMENT_SERVED_DURATION_MS == "ratel.experiment.served.duration_ms"
    assert RATEL_EXPERIMENT_SERVED_HIT_COUNT == "ratel.experiment.served.hit_count"
    assert RATEL_EXPERIMENT_SHADOW_ARM == "ratel.experiment.shadow.arm"
    assert RATEL_EXPERIMENT_SHADOW_OUTCOME == "ratel.experiment.shadow.outcome"
    assert RATEL_EXPERIMENT_SHADOW_DURATION_MS == "ratel.experiment.shadow.duration_ms"
    assert RATEL_EXPERIMENT_SHADOW_HIT_COUNT == "ratel.experiment.shadow.hit_count"
    assert RATEL_EXPERIMENT_AGREEMENT_TOP1 == "ratel.experiment.agreement.top1"
    assert (
        RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER
        == "ratel.experiment.agreement.exact_order"
    )
    assert (
        RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT
        == "ratel.experiment.agreement.overlap_count"
    )
    assert (
        RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K
        == "ratel.experiment.agreement.jaccard_at_k"
    )
    assert RATEL_EXPERIMENT_AGREEMENT_K == "ratel.experiment.agreement.k"
    assert (
        RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS
        == "ratel.experiment.agreement.item_attrs"
    )
    assert (
        RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS
        == "ratel.experiment.agreement.result_attrs"
    )
    assert RATEL_EXPERIMENT_EFFECTIVE_ARM == "ratel.experiment.effective_arm"
    assert RATEL_EXPERIMENT_SKIP_ARM == "ratel.experiment.skip.arm"
    assert RATEL_EXPERIMENT_SKIP_CONCURRENCY == "ratel.experiment.skip.concurrency"
    assert RATEL_EXPERIMENT_SKIP_REASON == "ratel.experiment.skip.reason"
    assert (
        RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM
        == "ratel.experiment.fallback.effective_arm"
    )
    assert (
        RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW
        == "ratel.experiment.fallback.reused_shadow"
    )
    assert RATEL_EXPERIMENT_DROP_REASON == "ratel.experiment.drop.reason"
    assert (
        RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED
        == "ratel.experiment.invocation.attributed"
    )
    assert RATEL_EXPERIMENT_INVOCATION_RANK == "ratel.experiment.invocation.rank"
    assert RATEL_EXPERIMENT_INVOCATION_AGE_MS == "ratel.experiment.invocation.age_ms"
    assert RATEL_EXPERIMENT_TURN == "ratel.experiment.turn"
    assert RATEL_EXPERIMENT_OUTCOME_LABEL == "ratel.experiment.outcome.label"
    assert RATEL_EXPERIMENT_OUTCOME_SCORE == "ratel.experiment.outcome.score"


def test_experiment_baggage_keys_alias_matching_span_attributes() -> None:
    assert RATEL_EXPERIMENT_ID_BAGGAGE_KEY == RATEL_EXPERIMENT_ID
    assert RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY == RATEL_EXPERIMENT_SELECTION_ID
    assert RATEL_EXPERIMENT_ARM_BAGGAGE_KEY == RATEL_EXPERIMENT_ARM
    assert RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY == RATEL_EXPERIMENT_ROLE
    assert RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY == RATEL_EXPERIMENT_UNIT


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
        RATEL_EXPERIMENT_ID,
        RATEL_EXPERIMENT_SELECTION_ID,
        RATEL_EXPERIMENT_ARM,
        RATEL_EXPERIMENT_ROLE,
        RATEL_EXPERIMENT_UNIT,
        RATEL_EXPERIMENT_COLD,
        RATEL_EXPERIMENT_OUTCOME,
        RATEL_EXPERIMENT_DURATION_MS,
        RATEL_EXPERIMENT_HIT_COUNT,
        RATEL_EXPERIMENT_RANKING_ERROR,
        RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
        RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
        RATEL_EXPERIMENT_RESULT_IDS,
        RATEL_EXPERIMENT_RESULT_SCORES,
        RATEL_EXPERIMENT_RESULT_ATTRS,
        RATEL_EXPERIMENT_SERVED_ARM,
        RATEL_EXPERIMENT_SERVED_OUTCOME,
        RATEL_EXPERIMENT_SERVED_DURATION_MS,
        RATEL_EXPERIMENT_SERVED_HIT_COUNT,
        RATEL_EXPERIMENT_SHADOW_ARM,
        RATEL_EXPERIMENT_SHADOW_OUTCOME,
        RATEL_EXPERIMENT_SHADOW_DURATION_MS,
        RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
        RATEL_EXPERIMENT_AGREEMENT_TOP1,
        RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
        RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
        RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
        RATEL_EXPERIMENT_AGREEMENT_K,
        RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
        RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
        RATEL_EXPERIMENT_EFFECTIVE_ARM,
        RATEL_EXPERIMENT_SKIP_ARM,
        RATEL_EXPERIMENT_SKIP_CONCURRENCY,
        RATEL_EXPERIMENT_SKIP_REASON,
        RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
        RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
        RATEL_EXPERIMENT_DROP_REASON,
        RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
        RATEL_EXPERIMENT_INVOCATION_RANK,
        RATEL_EXPERIMENT_INVOCATION_AGE_MS,
        RATEL_EXPERIMENT_TURN,
        RATEL_EXPERIMENT_OUTCOME_LABEL,
        RATEL_EXPERIMENT_OUTCOME_SCORE,
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
        RATEL_EXPERIMENT_ID,
        RATEL_EXPERIMENT_SELECTION_ID,
        RATEL_EXPERIMENT_ARM,
        RATEL_EXPERIMENT_ROLE,
        RATEL_EXPERIMENT_UNIT,
        RATEL_EXPERIMENT_COLD,
        RATEL_EXPERIMENT_OUTCOME,
        RATEL_EXPERIMENT_DURATION_MS,
        RATEL_EXPERIMENT_HIT_COUNT,
        RATEL_EXPERIMENT_RANKING_ERROR,
        RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
        RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
        RATEL_EXPERIMENT_RESULT_IDS,
        RATEL_EXPERIMENT_RESULT_SCORES,
        RATEL_EXPERIMENT_RESULT_ATTRS,
        RATEL_EXPERIMENT_SERVED_ARM,
        RATEL_EXPERIMENT_SERVED_OUTCOME,
        RATEL_EXPERIMENT_SERVED_DURATION_MS,
        RATEL_EXPERIMENT_SERVED_HIT_COUNT,
        RATEL_EXPERIMENT_SHADOW_ARM,
        RATEL_EXPERIMENT_SHADOW_OUTCOME,
        RATEL_EXPERIMENT_SHADOW_DURATION_MS,
        RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
        RATEL_EXPERIMENT_AGREEMENT_TOP1,
        RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
        RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
        RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
        RATEL_EXPERIMENT_AGREEMENT_K,
        RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
        RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
        RATEL_EXPERIMENT_EFFECTIVE_ARM,
        RATEL_EXPERIMENT_SKIP_ARM,
        RATEL_EXPERIMENT_SKIP_CONCURRENCY,
        RATEL_EXPERIMENT_SKIP_REASON,
        RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
        RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
        RATEL_EXPERIMENT_DROP_REASON,
        RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
        RATEL_EXPERIMENT_INVOCATION_RANK,
        RATEL_EXPERIMENT_INVOCATION_AGE_MS,
        RATEL_EXPERIMENT_TURN,
        RATEL_EXPERIMENT_OUTCOME_LABEL,
        RATEL_EXPERIMENT_OUTCOME_SCORE,
        GEN_AI_OPERATION_NAME,
        GEN_AI_TOOL_NAME,
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
    ]
    assert len(set(keys)) == len(keys)


def test_event_record_names_are_unique() -> None:
    names = [
        RATEL_SEARCH_RESULTS,
        RATEL_TOOL_EXECUTION_DETAILS,
        RATEL_EXPERIMENT_RESULTS,
        RATEL_EXPERIMENT_COMPARISON,
        RATEL_EXPERIMENT_SKIP,
        RATEL_EXPERIMENT_FALLBACK,
        RATEL_EXPERIMENT_DROP,
        RATEL_EXPERIMENT_INVOCATION,
        RATEL_EXPERIMENT_OUTCOME,
        GEN_AI_INFERENCE_DETAILS,
    ]
    assert len(set(names)) == len(names)


def test_origin_maps_to_wire_strings() -> None:
    assert Origin.DIRECT == "direct"
    assert Origin.AGENT == "agent"
    assert Origin.BASELINE == "baseline"


def test_search_target_maps_to_wire_strings() -> None:
    assert SearchTarget.TOOL == "tool"
    assert SearchTarget.SKILL == "skill"
    assert SearchTarget.FACT == "fact"


def test_auth_outcome_maps_to_wire_strings() -> None:
    assert AuthOutcome.OK == "ok"
    assert AuthOutcome.REFRESHED == "refreshed"
    assert AuthOutcome.NEEDS_AUTH == "needs_auth"
    assert AuthOutcome.FAILED == "failed"


def test_experiment_arm_role_maps_to_wire_strings() -> None:
    assert ExperimentArmRole.SERVING == "serving"
    assert ExperimentArmRole.SHADOW == "shadow"


def test_experiment_arm_outcome_maps_to_wire_strings() -> None:
    assert ExperimentArmOutcome.OK == "ok"
    assert ExperimentArmOutcome.EMPTY == "empty"
    assert ExperimentArmOutcome.TIMEOUT == "timeout"
    assert ExperimentArmOutcome.ERROR == "error"


def test_experiment_skip_reason_maps_to_wire_strings() -> None:
    assert ExperimentSkipReason.CAPACITY == "capacity"


def test_experiment_drop_reason_maps_to_wire_strings() -> None:
    assert ExperimentDropReason.ARM_FAILED == "arm-failed"
    assert ExperimentDropReason.FALLBACK_CONSUMED == "fallback-consumed"
    assert ExperimentDropReason.SELECTION_FAILED == "selection-failed"
    assert ExperimentDropReason.SERVED_RANKING_FAILED == "served-ranking-failed"
    assert ExperimentDropReason.COMPARISON_FAILED == "comparison-failed"
