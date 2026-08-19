"""`ratel-ai-telemetry` — the ratel.* telemetry vocabulary.

See the wire contract in ../CONVENTIONS.md. Emitting the vocabulary is done through
the standard OpenTelemetry Python SDK; this package adds no transport and no schema
(ADR-0007). These constants are the ratel.* overlay Ratel owns, plus the small subset
of gen_ai.* keys the overlay emits directly on the execute_tool span (borrowed verbatim
from OpenTelemetry, never renamed).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final

#: The pinned OpenTelemetry semantic-conventions version this vocabulary tracks
#: (the gen_ai group). The pin is the contract; consumers read against this exact
#: version, never "latest" (CONVENTIONS.md § The pin).
SEMCONV_VERSION: Final = "1.42.0"

#: The ecosystem instrumentation env var gating message/tool content capture.
#: Default off; honored by init() rather than a Ratel-invented flag
#: (CONVENTIONS.md § Capture gating). Values: legacy boolean, or the enum
#: NO_CONTENT (default) / SPAN_ONLY / EVENT_ONLY / SPAN_AND_EVENT.
CAPTURE_CONTENT_ENV: Final = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

#: Ratel-specific opt-in for the experimental catalog-definition event.
#: The generic content-capture gate must also allow EventRecord content.
EXPERIMENTAL_CATALOG_DEFINITIONS_ENV: Final = "RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS"

# ---------------------------------------------------------------------------
# Span names (CONVENTIONS.md, Tier 2)
# ---------------------------------------------------------------------------

#: ratel.search — capability search (unifies tool-search and skill-search).
RATEL_SEARCH: Final = "ratel.search"

#: ratel.experiment.arm — one serving or shadow experiment-arm dispatch.
RATEL_EXPERIMENT_ARM: Final = "ratel.experiment.arm"

#: execute_tool — the gen_ai.operation.name value for a tool invocation.
#: Deliberately the standard OTel gen_ai operation, not a bespoke ratel.invoke span,
#: so a generic OTel backend already understands it (locked 2026-07-05). The invoke
#: is enriched with ratel.* attributes.
EXECUTE_TOOL: Final = "execute_tool"

#: ratel.skill.load — skill content load (get_skill_content).
RATEL_SKILL_LOAD: Final = "ratel.skill.load"

#: ratel.upstream.register — upstream-MCP ingest.
RATEL_UPSTREAM_REGISTER: Final = "ratel.upstream.register"

#: ratel.auth.flow — MCP auth flow.
RATEL_AUTH_FLOW: Final = "ratel.auth.flow"

# ---------------------------------------------------------------------------
# EventRecord names (CONVENTIONS.md)
# ---------------------------------------------------------------------------

#: ratel.search.results — Opt-In search-content event; gated like content.
RATEL_SEARCH_RESULTS: Final = "ratel.search.results"

#: ratel.tool.execution.details — Opt-In structured tool arguments/result event.
RATEL_TOOL_EXECUTION_DETAILS: Final = "ratel.tool.execution.details"

#: ratel.catalog.definition — experimental, opt-in complete catalog definition event.
RATEL_CATALOG_DEFINITION: Final = "ratel.catalog.definition"

#: ratel.experiment.results — ranked measurement for one experiment arm.
RATEL_EXPERIMENT_RESULTS: Final = "ratel.experiment.results"

#: ratel.experiment.comparison — one shadow-vs-served comparison.
RATEL_EXPERIMENT_COMPARISON: Final = "ratel.experiment.comparison"

#: ratel.experiment.skip — one shadow dispatch skipped for capacity.
RATEL_EXPERIMENT_SKIP: Final = "ratel.experiment.skip"

#: ratel.experiment.fallback — one successful fallback selection.
RATEL_EXPERIMENT_FALLBACK: Final = "ratel.experiment.fallback"

#: ratel.experiment.drop — one peer comparison that could not be emitted.
RATEL_EXPERIMENT_DROP: Final = "ratel.experiment.drop"

#: ratel.experiment.invocation — attributed or unattributed tool invocation.
RATEL_EXPERIMENT_INVOCATION: Final = "ratel.experiment.invocation"

#: gen_ai.client.inference.operation.details — inference request/response content.
GEN_AI_INFERENCE_DETAILS: Final = "gen_ai.client.inference.operation.details"

# ---------------------------------------------------------------------------
# ratel.* attribute keys (CONVENTIONS.md, Tier 2)
# ---------------------------------------------------------------------------

#: ratel.origin — direct library call vs agent-synthesized (shared attribute).
RATEL_ORIGIN: Final = "ratel.origin"

#: ratel.event.id — runtime-event/OTel deduplication and join key.
RATEL_EVENT_ID: Final = "ratel.event.id"

#: ratel.catalog.kind — "tool", "skill", or "fact".
RATEL_CATALOG_KIND: Final = "ratel.catalog.kind"
#: ratel.catalog.id — stable catalog entry id.
RATEL_CATALOG_ID: Final = "ratel.catalog.id"
#: ratel.catalog.name — model-facing catalog entry name.
RATEL_CATALOG_NAME: Final = "ratel.catalog.name"
#: ratel.catalog.description — model-facing catalog entry description.
RATEL_CATALOG_DESCRIPTION: Final = "ratel.catalog.description"
#: ratel.catalog.tags — search tags; empty for tools.
RATEL_CATALOG_TAGS: Final = "ratel.catalog.tags"
#: ratel.catalog.input_schema — canonical JSON tool input schema.
RATEL_CATALOG_INPUT_SCHEMA: Final = "ratel.catalog.input_schema"
#: ratel.catalog.output_schema — canonical JSON tool output schema.
RATEL_CATALOG_OUTPUT_SCHEMA: Final = "ratel.catalog.output_schema"
#: ratel.catalog.searchable_description — effective searchable description.
RATEL_CATALOG_SEARCHABLE_DESCRIPTION: Final = "ratel.catalog.searchable_description"
#: whether the effective searchable description came from an explicit override.
RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN: Final = (
    "ratel.catalog.searchable_description_overridden"
)
#: ratel.catalog.content_hash — canonical definition SHA-256.
RATEL_CATALOG_CONTENT_HASH: Final = "ratel.catalog.content_hash"

#: ratel.search.target — "tool", "skill", or "fact" (see SearchTarget).
RATEL_SEARCH_TARGET: Final = "ratel.search.target"

#: ratel.search.top_k — requested result count.
RATEL_SEARCH_TOP_K: Final = "ratel.search.top_k"

#: ratel.search.hit_count — results returned.
RATEL_SEARCH_HIT_COUNT: Final = "ratel.search.hit_count"

#: ratel.search.query — the search text (content, gated like message content).
RATEL_SEARCH_QUERY: Final = "ratel.search.query"

#: ratel.tool.args_size_bytes — argument payload size on the execute_tool span.
RATEL_TOOL_ARGS_SIZE_BYTES: Final = "ratel.tool.args_size_bytes"

#: ratel.upstream.server — upstream MCP server backing a tool / auth flow.
RATEL_UPSTREAM_SERVER: Final = "ratel.upstream.server"

#: ratel.upstream.transport — "stdio" / "http" / "sse" / ...
RATEL_UPSTREAM_TRANSPORT: Final = "ratel.upstream.transport"

#: ratel.upstream.tool_count — tools ingested on register.
RATEL_UPSTREAM_TOOL_COUNT: Final = "ratel.upstream.tool_count"

#: ratel.skill.id — skill loaded on the ratel.skill.load span.
RATEL_SKILL_ID: Final = "ratel.skill.id"

#: ratel.auth.outcome — "ok" / "refreshed" / "needs_auth" / "failed" (see AuthOutcome).
RATEL_AUTH_OUTCOME: Final = "ratel.auth.outcome"

#: ratel.experiment.id — configured experiment id.
RATEL_EXPERIMENT_ID: Final = "ratel.experiment.id"

#: ratel.experiment.selection_id — opaque selection correlation id.
RATEL_EXPERIMENT_SELECTION_ID: Final = "ratel.experiment.selection_id"

#: ratel.experiment.role — immutable dispatch role (see ExperimentArmRole).
RATEL_EXPERIMENT_ROLE: Final = "ratel.experiment.role"

#: ratel.experiment.unit — pseudonymous 16-hex unit hash.
RATEL_EXPERIMENT_UNIT: Final = "ratel.experiment.unit"

#: ratel.experiment.cold — whether warmup was unresolved when selection began.
RATEL_EXPERIMENT_COLD: Final = "ratel.experiment.cold"

#: ratel.experiment.outcome — arm completion attr and reported-outcome EventRecord name.
RATEL_EXPERIMENT_OUTCOME: Final = "ratel.experiment.outcome"

#: ratel.experiment.duration_ms — arm callback duration in milliseconds.
RATEL_EXPERIMENT_DURATION_MS: Final = "ratel.experiment.duration_ms"

#: ratel.experiment.hit_count — ranked result count when ranking succeeds.
RATEL_EXPERIMENT_HIT_COUNT: Final = "ratel.experiment.hit_count"

#: ratel.experiment.ranking_error — error type when ranking projection fails.
RATEL_EXPERIMENT_RANKING_ERROR: Final = "ratel.experiment.ranking_error"

#: ratel.experiment.result_attributes_error — result-level projection error type.
RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR: Final = "ratel.experiment.result_attributes_error"

#: ratel.experiment.result_attrs_encoding_error — item-attribute encoding error type.
RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR: Final = "ratel.experiment.result_attrs_encoding_error"

#: ratel.experiment.result_ids — ordered ranked result identifiers.
RATEL_EXPERIMENT_RESULT_IDS: Final = "ratel.experiment.result_ids"

#: ratel.experiment.result_scores — ordered finite scores when every item has one.
RATEL_EXPERIMENT_RESULT_SCORES: Final = "ratel.experiment.result_scores"

#: ratel.experiment.result_attrs — capture-gated item attrs aligned to result ids.
RATEL_EXPERIMENT_RESULT_ATTRS: Final = "ratel.experiment.result_attrs"

#: ratel.experiment.served.arm — effective served arm in a peer comparison.
RATEL_EXPERIMENT_SERVED_ARM: Final = "ratel.experiment.served.arm"

#: ratel.experiment.served.outcome — effective served arm outcome.
RATEL_EXPERIMENT_SERVED_OUTCOME: Final = "ratel.experiment.served.outcome"

#: ratel.experiment.served.duration_ms — effective served arm callback duration.
RATEL_EXPERIMENT_SERVED_DURATION_MS: Final = "ratel.experiment.served.duration_ms"

#: ratel.experiment.served.hit_count — effective served ranking size.
RATEL_EXPERIMENT_SERVED_HIT_COUNT: Final = "ratel.experiment.served.hit_count"

#: ratel.experiment.shadow.arm — compared shadow arm.
RATEL_EXPERIMENT_SHADOW_ARM: Final = "ratel.experiment.shadow.arm"

#: ratel.experiment.shadow.outcome — compared shadow arm outcome.
RATEL_EXPERIMENT_SHADOW_OUTCOME: Final = "ratel.experiment.shadow.outcome"

#: ratel.experiment.shadow.duration_ms — compared shadow arm callback duration.
RATEL_EXPERIMENT_SHADOW_DURATION_MS: Final = "ratel.experiment.shadow.duration_ms"

#: ratel.experiment.shadow.hit_count — compared shadow ranking size.
RATEL_EXPERIMENT_SHADOW_HIT_COUNT: Final = "ratel.experiment.shadow.hit_count"

#: ratel.experiment.agreement.top1 — whether first ranked ids agree.
RATEL_EXPERIMENT_AGREEMENT_TOP1: Final = "ratel.experiment.agreement.top1"

#: ratel.experiment.agreement.exact_order — whether complete ordered ids agree.
RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER: Final = "ratel.experiment.agreement.exact_order"

#: ratel.experiment.agreement.overlap_count — complete-set overlap count.
RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT: Final = "ratel.experiment.agreement.overlap_count"

#: ratel.experiment.agreement.jaccard_at_k — rank-window Jaccard agreement.
RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K: Final = "ratel.experiment.agreement.jaccard_at_k"

#: ratel.experiment.agreement.k — rank window used for Jaccard.
RATEL_EXPERIMENT_AGREEMENT_K: Final = "ratel.experiment.agreement.k"

#: ratel.experiment.agreement.item_attrs — rank-zero shared-key agreement map.
RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS: Final = "ratel.experiment.agreement.item_attrs"

#: ratel.experiment.agreement.result_attrs — result-level union-key agreement map.
RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS: Final = "ratel.experiment.agreement.result_attrs"

#: ratel.experiment.effective_arm — caller-visible arm after fallback.
RATEL_EXPERIMENT_EFFECTIVE_ARM: Final = "ratel.experiment.effective_arm"

#: ratel.experiment.skip.arm — shadow arm skipped for capacity.
RATEL_EXPERIMENT_SKIP_ARM: Final = "ratel.experiment.skip.arm"

#: ratel.experiment.skip.concurrency — configured shadow concurrency.
RATEL_EXPERIMENT_SKIP_CONCURRENCY: Final = "ratel.experiment.skip.concurrency"

#: ratel.experiment.skip.reason — skip reason (see ExperimentSkipReason).
RATEL_EXPERIMENT_SKIP_REASON: Final = "ratel.experiment.skip.reason"

#: ratel.experiment.fallback.effective_arm — fallback arm that served.
RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM: Final = "ratel.experiment.fallback.effective_arm"

#: ratel.experiment.fallback.reused_shadow — whether fallback reused admitted shadow work.
RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW: Final = "ratel.experiment.fallback.reused_shadow"

#: ratel.experiment.drop.reason — comparison drop reason (see ExperimentDropReason).
RATEL_EXPERIMENT_DROP_REASON: Final = "ratel.experiment.drop.reason"

#: ratel.experiment.invocation.attributed — whether an invocation matched a selection.
RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED: Final = "ratel.experiment.invocation.attributed"

#: ratel.experiment.invocation.rank — zero-based first rank, or -1 when absent.
RATEL_EXPERIMENT_INVOCATION_RANK: Final = "ratel.experiment.invocation.rank"

#: ratel.experiment.invocation.age_ms — selection age at invocation report time.
RATEL_EXPERIMENT_INVOCATION_AGE_MS: Final = "ratel.experiment.invocation.age_ms"

#: ratel.experiment.turn — optional caller-supplied turn annotation.
RATEL_EXPERIMENT_TURN: Final = "ratel.experiment.turn"

#: ratel.experiment.outcome.label — free non-empty reported outcome label.
RATEL_EXPERIMENT_OUTCOME_LABEL: Final = "ratel.experiment.outcome.label"

#: ratel.experiment.outcome.score — finite reported outcome score.
RATEL_EXPERIMENT_OUTCOME_SCORE: Final = "ratel.experiment.outcome.score"

# Experiment baggage keys intentionally equal their matching span-attribute keys
# (ADR-0019). Explicit aliases keep baggage construction discoverable.
RATEL_EXPERIMENT_ID_BAGGAGE_KEY: Final = RATEL_EXPERIMENT_ID
RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY: Final = RATEL_EXPERIMENT_SELECTION_ID
RATEL_EXPERIMENT_ARM_BAGGAGE_KEY: Final = RATEL_EXPERIMENT_ARM
RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY: Final = RATEL_EXPERIMENT_ROLE
RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY: Final = RATEL_EXPERIMENT_UNIT

# ---------------------------------------------------------------------------
# gen_ai.* interop keys (CONVENTIONS.md, Tier 1 — borrowed verbatim)
#
# Only the subset the ratel.* overlay directly emits on the execute_tool span.
# These are OpenTelemetry's, not ours: exposed so callers avoid stringly-typing
# them, never renamed into ratel.*.
# ---------------------------------------------------------------------------

#: gen_ai.operation.name — set to EXECUTE_TOOL for a tool invocation.
GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"

#: gen_ai.tool.name — the capability tool id.
GEN_AI_TOOL_NAME: Final = "gen_ai.tool.name"

#: gen_ai.tool.call.id — tool call id, when available.
GEN_AI_TOOL_CALL_ID: Final = "gen_ai.tool.call.id"

#: gen_ai.tool.call.arguments — tool arguments (Opt-In content, gated).
GEN_AI_TOOL_CALL_ARGUMENTS: Final = "gen_ai.tool.call.arguments"

#: gen_ai.tool.call.result — tool result (Opt-In content, gated).
GEN_AI_TOOL_CALL_RESULT: Final = "gen_ai.tool.call.result"

# Tier 1 content, carried on the gen_ai.client.inference.operation.details
# EventRecord (never span attributes; CONVENTIONS.md § Tier 1 content). Each
# holds a structured v1.42.0 message list ({role, parts[], name?}).

#: gen_ai.system_instructions — the system prompt as a bare parts[] (Opt-In content).
GEN_AI_SYSTEM_INSTRUCTIONS: Final = "gen_ai.system_instructions"

#: gen_ai.input.messages — the input message list (Opt-In content).
GEN_AI_INPUT_MESSAGES: Final = "gen_ai.input.messages"

#: gen_ai.output.messages — generated outputs; every message includes finish_reason.
GEN_AI_OUTPUT_MESSAGES: Final = "gen_ai.output.messages"


class Origin(str, Enum):
    """Whether a ratel.* span was a direct library call or synthesized by the agent
    inside its loop. Carried by ratel.origin; mirrors the local trace Origin (ADR-0007).

    A str-Enum: each member equals its exact wire string, so it is usable directly
    as an OTel attribute value.
    """

    DIRECT = "direct"
    AGENT = "agent"
    BASELINE = "baseline"


class SearchTarget(str, Enum):
    """What a ratel.search span was searching. Carried by ratel.search.target; folds
    capability-tool search, skill search, and fact search into one span shape."""

    TOOL = "tool"
    SKILL = "skill"
    FACT = "fact"


class AuthOutcome(str, Enum):
    """Outcome of an MCP auth flow. Carried by ratel.auth.outcome; NEEDS_AUTH is the
    401-driven AuthNeeds case (ADR-0007 auth_needs)."""

    OK = "ok"
    REFRESHED = "refreshed"
    NEEDS_AUTH = "needs_auth"
    FAILED = "failed"


class ExperimentArmRole(str, Enum):
    """Immutable role assigned when an experiment arm is dispatched."""

    SERVING = "serving"
    SHADOW = "shadow"


class ExperimentArmOutcome(str, Enum):
    """Completion outcome of one experiment-arm dispatch."""

    OK = "ok"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"


class ExperimentSkipReason(str, Enum):
    """Reason a requested shadow dispatch did not start."""

    CAPACITY = "capacity"


class ExperimentDropReason(str, Enum):
    """Terminal reason a peer-eligible shadow produced no comparison."""

    ARM_FAILED = "arm-failed"
    FALLBACK_CONSUMED = "fallback-consumed"
    SELECTION_FAILED = "selection-failed"
    SERVED_RANKING_FAILED = "served-ranking-failed"
    COMPARISON_FAILED = "comparison-failed"


# The OTLP exporter surface (init() + config/gate) lives in the opt-in .otlp
# submodule behind the [otlp] extra, so plain constant imports stay OTel-free
# (ADR-0007). A module-level __getattr__ lazily resolves it, so the ergonomic
# `from ratel_ai_telemetry import init` keeps working without eagerly importing it.
_OTLP_EXPORTS: Final = frozenset(
    {
        "init",
        "resolve_otlp_config",
        "content_capture_mode",
        "set_content_capture",
        "clear_content_capture",
        "ContentCapture",
        "OtlpConfig",
        "API_KEY_ENV",
        "ENDPOINT_ENV",
        "OTLP_ENDPOINT_ENV",
        "LEGACY_ENDPOINT_ENV",
        "DEFAULT_SERVICE_NAME",
        "ratel_event_filter",
        "ratel_log_exporter",
        "ratel_log_record_processor",
        "ratel_signal_filter",
        "ratel_span_exporter",
        "ratel_span_processor",
        "recorded_service_name",
        "SpanFilter",
        "LogFilter",
        "TelemetryHandle",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily resolve the OTLP exporter surface from the opt-in .otlp submodule.

    Keeps `from ratel_ai_telemetry import init` working while a plain
    `import ratel_ai_telemetry` pulls no OpenTelemetry SDK. init() itself raises a
    clear error if the [otlp] extra is not installed.
    """
    if name in _OTLP_EXPORTS:
        from . import otlp

        return getattr(otlp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CAPTURE_CONTENT_ENV",
    "EXPERIMENTAL_CATALOG_DEFINITIONS_ENV",
    "EXECUTE_TOOL",
    "GEN_AI_INFERENCE_DETAILS",
    "GEN_AI_INPUT_MESSAGES",
    "GEN_AI_OPERATION_NAME",
    "GEN_AI_OUTPUT_MESSAGES",
    "GEN_AI_SYSTEM_INSTRUCTIONS",
    "GEN_AI_TOOL_CALL_ARGUMENTS",
    "GEN_AI_TOOL_CALL_ID",
    "GEN_AI_TOOL_CALL_RESULT",
    "GEN_AI_TOOL_NAME",
    "RATEL_AUTH_FLOW",
    "RATEL_AUTH_OUTCOME",
    "RATEL_CATALOG_CONTENT_HASH",
    "RATEL_CATALOG_DEFINITION",
    "RATEL_CATALOG_DESCRIPTION",
    "RATEL_CATALOG_ID",
    "RATEL_CATALOG_INPUT_SCHEMA",
    "RATEL_CATALOG_KIND",
    "RATEL_CATALOG_NAME",
    "RATEL_CATALOG_OUTPUT_SCHEMA",
    "RATEL_CATALOG_SEARCHABLE_DESCRIPTION",
    "RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN",
    "RATEL_CATALOG_TAGS",
    "RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER",
    "RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS",
    "RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K",
    "RATEL_EXPERIMENT_AGREEMENT_K",
    "RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT",
    "RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS",
    "RATEL_EXPERIMENT_AGREEMENT_TOP1",
    "RATEL_EXPERIMENT_ARM",
    "RATEL_EXPERIMENT_ARM_BAGGAGE_KEY",
    "RATEL_EXPERIMENT_COLD",
    "RATEL_EXPERIMENT_COMPARISON",
    "RATEL_EXPERIMENT_DROP",
    "RATEL_EXPERIMENT_DROP_REASON",
    "RATEL_EXPERIMENT_DURATION_MS",
    "RATEL_EXPERIMENT_EFFECTIVE_ARM",
    "RATEL_EXPERIMENT_FALLBACK",
    "RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM",
    "RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW",
    "RATEL_EXPERIMENT_HIT_COUNT",
    "RATEL_EXPERIMENT_ID",
    "RATEL_EXPERIMENT_ID_BAGGAGE_KEY",
    "RATEL_EXPERIMENT_INVOCATION",
    "RATEL_EXPERIMENT_INVOCATION_AGE_MS",
    "RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED",
    "RATEL_EXPERIMENT_INVOCATION_RANK",
    "RATEL_EXPERIMENT_OUTCOME",
    "RATEL_EXPERIMENT_OUTCOME_LABEL",
    "RATEL_EXPERIMENT_OUTCOME_SCORE",
    "RATEL_EXPERIMENT_RANKING_ERROR",
    "RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR",
    "RATEL_EXPERIMENT_RESULT_ATTRS",
    "RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR",
    "RATEL_EXPERIMENT_RESULT_IDS",
    "RATEL_EXPERIMENT_RESULT_SCORES",
    "RATEL_EXPERIMENT_RESULTS",
    "RATEL_EXPERIMENT_ROLE",
    "RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY",
    "RATEL_EXPERIMENT_SERVED_ARM",
    "RATEL_EXPERIMENT_SERVED_DURATION_MS",
    "RATEL_EXPERIMENT_SERVED_HIT_COUNT",
    "RATEL_EXPERIMENT_SERVED_OUTCOME",
    "RATEL_EXPERIMENT_SELECTION_ID",
    "RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY",
    "RATEL_EXPERIMENT_SHADOW_ARM",
    "RATEL_EXPERIMENT_SHADOW_DURATION_MS",
    "RATEL_EXPERIMENT_SHADOW_HIT_COUNT",
    "RATEL_EXPERIMENT_SHADOW_OUTCOME",
    "RATEL_EXPERIMENT_SKIP",
    "RATEL_EXPERIMENT_SKIP_ARM",
    "RATEL_EXPERIMENT_SKIP_CONCURRENCY",
    "RATEL_EXPERIMENT_SKIP_REASON",
    "RATEL_EXPERIMENT_TURN",
    "RATEL_EXPERIMENT_UNIT",
    "RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY",
    "RATEL_ORIGIN",
    "RATEL_SEARCH",
    "RATEL_SEARCH_HIT_COUNT",
    "RATEL_SEARCH_QUERY",
    "RATEL_SEARCH_RESULTS",
    "RATEL_SEARCH_TARGET",
    "RATEL_SEARCH_TOP_K",
    "RATEL_SKILL_ID",
    "RATEL_SKILL_LOAD",
    "RATEL_TOOL_ARGS_SIZE_BYTES",
    "RATEL_TOOL_EXECUTION_DETAILS",
    "RATEL_UPSTREAM_REGISTER",
    "RATEL_UPSTREAM_SERVER",
    "RATEL_UPSTREAM_TOOL_COUNT",
    "RATEL_UPSTREAM_TRANSPORT",
    "SEMCONV_VERSION",
    "AuthOutcome",
    "ExperimentArmOutcome",
    "ExperimentArmRole",
    "ExperimentDropReason",
    "ExperimentSkipReason",
    "Origin",
    "SearchTarget",
    # The OTLP exporter surface, lazily resolved from `.otlp` via __getattr__ above
    # (importing any of these — or `import *` — loads the submodule but still no OTel
    # SDK; only wiring an exporter needs the [otlp] extra).
    "init",
    "resolve_otlp_config",
    "content_capture_mode",
    "set_content_capture",
    "clear_content_capture",
    "ContentCapture",
    "OtlpConfig",
    "API_KEY_ENV",
    "ENDPOINT_ENV",
    "OTLP_ENDPOINT_ENV",
    "LEGACY_ENDPOINT_ENV",
    "DEFAULT_SERVICE_NAME",
    "ratel_event_filter",
    "ratel_log_exporter",
    "ratel_log_record_processor",
    "ratel_signal_filter",
    "ratel_span_exporter",
    "ratel_span_processor",
    "recorded_service_name",
    "SpanFilter",
    "LogFilter",
    "TelemetryHandle",
]
