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

# ---------------------------------------------------------------------------
# Span names (CONVENTIONS.md, Tier 2)
# ---------------------------------------------------------------------------

#: ratel.search — capability search (unifies tool-search and skill-search).
RATEL_SEARCH: Final = "ratel.search"

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

#: gen_ai.client.inference.operation.details — inference request/response content.
GEN_AI_INFERENCE_DETAILS: Final = "gen_ai.client.inference.operation.details"

# ---------------------------------------------------------------------------
# ratel.* attribute keys (CONVENTIONS.md, Tier 2)
# ---------------------------------------------------------------------------

#: ratel.origin — direct library call vs agent-synthesized (shared attribute).
RATEL_ORIGIN: Final = "ratel.origin"

#: ratel.search.target — "tool" or "skill" (see SearchTarget).
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
    capability-tool search and skill search into one span shape."""

    TOOL = "tool"
    SKILL = "skill"


class AuthOutcome(str, Enum):
    """Outcome of an MCP auth flow. Carried by ratel.auth.outcome; NEEDS_AUTH is the
    401-driven AuthNeeds case (ADR-0007 auth_needs)."""

    OK = "ok"
    REFRESHED = "refreshed"
    NEEDS_AUTH = "needs_auth"
    FAILED = "failed"


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
    "SpanFilter",
    "LogFilter",
    "TelemetryHandle",
]
