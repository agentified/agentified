//! `ratel-ai-telemetry` — the `ratel.*` telemetry vocabulary.
//!
//! See `README.md` and the wire contract in `../CONVENTIONS.md` for design.

#![warn(missing_docs)]

/// The pinned OpenTelemetry semantic-conventions version this vocabulary tracks
/// (the `gen_ai` group). The pin is the contract; consumers read against this
/// exact version, never "latest". Bumping it is a reviewed change with its own
/// PR and, if the shape changed, a superseding ADR (CONVENTIONS.md § The pin).
pub const SEMCONV_VERSION: &str = "1.42.0";

/// The ecosystem instrumentation env var gating message/tool content capture.
/// Default off; the standard OTel gen_ai gate rather than a Ratel-invented flag
/// (CONVENTIONS.md § Capture gating). This crate is constants-only — the TS/Python
/// `init()` helpers read it. Values: legacy boolean, or the enum `NO_CONTENT`
/// (default) / `SPAN_ONLY` / `EVENT_ONLY` / `SPAN_AND_EVENT`.
pub const CAPTURE_CONTENT_ENV: &str = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

/// Ratel-specific opt-in for the experimental catalog-definition event.
/// The generic content-capture gate must also allow EventRecord content.
pub const EXPERIMENTAL_CATALOG_DEFINITIONS_ENV: &str = "RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS";

// ---------------------------------------------------------------------------
// Span names (CONVENTIONS.md, Tier 2)
// ---------------------------------------------------------------------------

/// `ratel.search` — capability search (unifies tool-search and skill-search).
pub const RATEL_SEARCH: &str = "ratel.search";
/// `ratel.experiment.arm` — one serving or shadow experiment-arm dispatch.
pub const RATEL_EXPERIMENT_ARM: &str = "ratel.experiment.arm";
/// `execute_tool` — the `gen_ai.operation.name` value for a tool invocation.
///
/// Deliberately the standard OTel `gen_ai` operation, not a bespoke
/// `ratel.invoke` span, so a generic OTel backend already understands it
/// (locked 2026-07-05). The invoke is enriched with `ratel.*` attributes.
pub const EXECUTE_TOOL: &str = "execute_tool";
/// `ratel.skill.load` — skill content load (`get_skill_content`).
pub const RATEL_SKILL_LOAD: &str = "ratel.skill.load";
/// `ratel.upstream.register` — upstream-MCP ingest.
pub const RATEL_UPSTREAM_REGISTER: &str = "ratel.upstream.register";
/// `ratel.auth.flow` — MCP auth flow.
pub const RATEL_AUTH_FLOW: &str = "ratel.auth.flow";

// ---------------------------------------------------------------------------
// EventRecord names (CONVENTIONS.md)
// ---------------------------------------------------------------------------

/// `ratel.search.results` — Opt-In search-content event; gated like content.
pub const RATEL_SEARCH_RESULTS: &str = "ratel.search.results";
/// `ratel.tool.execution.details` — Opt-In structured tool arguments/result event.
pub const RATEL_TOOL_EXECUTION_DETAILS: &str = "ratel.tool.execution.details";
/// `ratel.catalog.definition` — experimental, opt-in complete catalog definition event.
pub const RATEL_CATALOG_DEFINITION: &str = "ratel.catalog.definition";
/// `ratel.experiment.results` — ranked measurement for one experiment arm.
pub const RATEL_EXPERIMENT_RESULTS: &str = "ratel.experiment.results";
/// `ratel.experiment.comparison` — one shadow-vs-served comparison.
pub const RATEL_EXPERIMENT_COMPARISON: &str = "ratel.experiment.comparison";
/// `ratel.experiment.skip` — one shadow dispatch skipped for capacity.
pub const RATEL_EXPERIMENT_SKIP: &str = "ratel.experiment.skip";
/// `ratel.experiment.fallback` — one successful fallback selection.
pub const RATEL_EXPERIMENT_FALLBACK: &str = "ratel.experiment.fallback";
/// `ratel.experiment.drop` — one peer comparison that could not be emitted.
pub const RATEL_EXPERIMENT_DROP: &str = "ratel.experiment.drop";
/// `ratel.experiment.invocation` — attributed or unattributed tool invocation.
pub const RATEL_EXPERIMENT_INVOCATION: &str = "ratel.experiment.invocation";
/// `gen_ai.client.inference.operation.details` — inference request/response content.
pub const GEN_AI_INFERENCE_DETAILS: &str = "gen_ai.client.inference.operation.details";

// ---------------------------------------------------------------------------
// `ratel.*` attribute keys (CONVENTIONS.md, Tier 2)
// ---------------------------------------------------------------------------

/// `ratel.event.id` — runtime-event/OTel deduplication and join key.
pub const RATEL_EVENT_ID: &str = "ratel.event.id";
/// `ratel.catalog.kind` — `tool`, `skill`, or `fact`.
pub const RATEL_CATALOG_KIND: &str = "ratel.catalog.kind";
/// `ratel.catalog.id` — stable catalog entry id.
pub const RATEL_CATALOG_ID: &str = "ratel.catalog.id";
/// `ratel.catalog.name` — model-facing catalog entry name.
pub const RATEL_CATALOG_NAME: &str = "ratel.catalog.name";
/// `ratel.catalog.description` — model-facing catalog entry description.
pub const RATEL_CATALOG_DESCRIPTION: &str = "ratel.catalog.description";
/// `ratel.catalog.tags` — search tags; empty for tools.
pub const RATEL_CATALOG_TAGS: &str = "ratel.catalog.tags";
/// `ratel.catalog.input_schema` — canonical JSON tool input schema.
pub const RATEL_CATALOG_INPUT_SCHEMA: &str = "ratel.catalog.input_schema";
/// `ratel.catalog.output_schema` — canonical JSON tool output schema.
pub const RATEL_CATALOG_OUTPUT_SCHEMA: &str = "ratel.catalog.output_schema";
/// `ratel.catalog.schema_omitted` — one or more oversized schema attributes were omitted.
pub const RATEL_CATALOG_SCHEMA_OMITTED: &str = "ratel.catalog.schema_omitted";
/// `ratel.catalog.searchable_description` — effective searchable description.
pub const RATEL_CATALOG_SEARCHABLE_DESCRIPTION: &str = "ratel.catalog.searchable_description";
/// `ratel.catalog.searchable_description_overridden` — whether an override is set.
pub const RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN: &str =
    "ratel.catalog.searchable_description_overridden";
/// `ratel.catalog.use_definition_overrides` — runtime opted into externally-sourced definition overrides.
pub const RATEL_CATALOG_USE_DEFINITION_OVERRIDES: &str = "ratel.catalog.use_definition_overrides";
/// `ratel.catalog.content_hash` — canonical definition SHA-256.
pub const RATEL_CATALOG_CONTENT_HASH: &str = "ratel.catalog.content_hash";
/// `ratel.origin` — direct library call vs agent-synthesized (shared attribute).
pub const RATEL_ORIGIN: &str = "ratel.origin";
/// `ratel.search.target` — `tool`, `skill`, or `fact` (see [`SearchTarget`]).
pub const RATEL_SEARCH_TARGET: &str = "ratel.search.target";
/// `ratel.search.top_k` — requested result count.
pub const RATEL_SEARCH_TOP_K: &str = "ratel.search.top_k";
/// `ratel.search.hit_count` — results returned.
pub const RATEL_SEARCH_HIT_COUNT: &str = "ratel.search.hit_count";
/// `ratel.search.query` — the search text (content, gated like message content).
pub const RATEL_SEARCH_QUERY: &str = "ratel.search.query";
/// `ratel.tool.args_size_bytes` — argument payload size on the `execute_tool` span.
pub const RATEL_TOOL_ARGS_SIZE_BYTES: &str = "ratel.tool.args_size_bytes";
/// `ratel.upstream.server` — upstream MCP server backing a tool / auth flow.
pub const RATEL_UPSTREAM_SERVER: &str = "ratel.upstream.server";
/// `ratel.upstream.transport` — `stdio` / `http` / `sse` / ...
pub const RATEL_UPSTREAM_TRANSPORT: &str = "ratel.upstream.transport";
/// `ratel.upstream.tool_count` — tools ingested on register.
pub const RATEL_UPSTREAM_TOOL_COUNT: &str = "ratel.upstream.tool_count";
/// `ratel.skill.id` — skill loaded on the `ratel.skill.load` span.
pub const RATEL_SKILL_ID: &str = "ratel.skill.id";
/// `ratel.auth.outcome` — `ok` / `refreshed` / `needs_auth` / `failed` (see [`AuthOutcome`]).
pub const RATEL_AUTH_OUTCOME: &str = "ratel.auth.outcome";
/// `ratel.experiment.id` — configured experiment id.
pub const RATEL_EXPERIMENT_ID: &str = "ratel.experiment.id";
/// `ratel.experiment.selection_id` — opaque selection correlation id.
pub const RATEL_EXPERIMENT_SELECTION_ID: &str = "ratel.experiment.selection_id";
/// `ratel.experiment.role` — immutable dispatch role (see [`ExperimentArmRole`]).
pub const RATEL_EXPERIMENT_ROLE: &str = "ratel.experiment.role";
/// `ratel.experiment.unit` — pseudonymous 16-hex unit hash.
pub const RATEL_EXPERIMENT_UNIT: &str = "ratel.experiment.unit";
/// `ratel.experiment.cold` — whether warmup was unresolved when selection began.
pub const RATEL_EXPERIMENT_COLD: &str = "ratel.experiment.cold";
/// `ratel.experiment.outcome` — arm completion attr and reported-outcome EventRecord name.
pub const RATEL_EXPERIMENT_OUTCOME: &str = "ratel.experiment.outcome";
/// `ratel.experiment.duration_ms` — arm callback duration in milliseconds.
pub const RATEL_EXPERIMENT_DURATION_MS: &str = "ratel.experiment.duration_ms";
/// `ratel.experiment.hit_count` — ranked result count when ranking succeeds.
pub const RATEL_EXPERIMENT_HIT_COUNT: &str = "ratel.experiment.hit_count";
/// `ratel.experiment.ranking_error` — error type when ranking projection fails.
pub const RATEL_EXPERIMENT_RANKING_ERROR: &str = "ratel.experiment.ranking_error";
/// `ratel.experiment.result_attributes_error` — result-level projection error type.
pub const RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR: &str =
    "ratel.experiment.result_attributes_error";
/// `ratel.experiment.result_attrs_encoding_error` — item-attribute encoding error type.
pub const RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR: &str =
    "ratel.experiment.result_attrs_encoding_error";
/// `ratel.experiment.result_ids` — ordered ranked result identifiers.
pub const RATEL_EXPERIMENT_RESULT_IDS: &str = "ratel.experiment.result_ids";
/// `ratel.experiment.result_scores` — ordered finite scores when every item has one.
pub const RATEL_EXPERIMENT_RESULT_SCORES: &str = "ratel.experiment.result_scores";
/// `ratel.experiment.result_attrs` — capture-gated item attrs aligned to result ids.
pub const RATEL_EXPERIMENT_RESULT_ATTRS: &str = "ratel.experiment.result_attrs";
/// `ratel.experiment.served.arm` — effective served arm in a peer comparison.
pub const RATEL_EXPERIMENT_SERVED_ARM: &str = "ratel.experiment.served.arm";
/// `ratel.experiment.served.outcome` — effective served arm outcome.
pub const RATEL_EXPERIMENT_SERVED_OUTCOME: &str = "ratel.experiment.served.outcome";
/// `ratel.experiment.served.duration_ms` — effective served arm callback duration.
pub const RATEL_EXPERIMENT_SERVED_DURATION_MS: &str = "ratel.experiment.served.duration_ms";
/// `ratel.experiment.served.hit_count` — effective served ranking size.
pub const RATEL_EXPERIMENT_SERVED_HIT_COUNT: &str = "ratel.experiment.served.hit_count";
/// `ratel.experiment.shadow.arm` — compared shadow arm.
pub const RATEL_EXPERIMENT_SHADOW_ARM: &str = "ratel.experiment.shadow.arm";
/// `ratel.experiment.shadow.outcome` — compared shadow arm outcome.
pub const RATEL_EXPERIMENT_SHADOW_OUTCOME: &str = "ratel.experiment.shadow.outcome";
/// `ratel.experiment.shadow.duration_ms` — compared shadow arm callback duration.
pub const RATEL_EXPERIMENT_SHADOW_DURATION_MS: &str = "ratel.experiment.shadow.duration_ms";
/// `ratel.experiment.shadow.hit_count` — compared shadow ranking size.
pub const RATEL_EXPERIMENT_SHADOW_HIT_COUNT: &str = "ratel.experiment.shadow.hit_count";
/// `ratel.experiment.agreement.top1` — whether first ranked ids agree.
pub const RATEL_EXPERIMENT_AGREEMENT_TOP1: &str = "ratel.experiment.agreement.top1";
/// `ratel.experiment.agreement.exact_order` — whether complete ordered ids agree.
pub const RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER: &str = "ratel.experiment.agreement.exact_order";
/// `ratel.experiment.agreement.overlap_count` — complete-set overlap count.
pub const RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT: &str =
    "ratel.experiment.agreement.overlap_count";
/// `ratel.experiment.agreement.jaccard_at_k` — rank-window Jaccard agreement.
pub const RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K: &str = "ratel.experiment.agreement.jaccard_at_k";
/// `ratel.experiment.agreement.k` — rank window used for Jaccard.
pub const RATEL_EXPERIMENT_AGREEMENT_K: &str = "ratel.experiment.agreement.k";
/// `ratel.experiment.agreement.item_attrs` — rank-zero shared-key agreement map.
pub const RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS: &str = "ratel.experiment.agreement.item_attrs";
/// `ratel.experiment.agreement.result_attrs` — result-level union-key agreement map.
pub const RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS: &str = "ratel.experiment.agreement.result_attrs";
/// `ratel.experiment.effective_arm` — caller-visible arm after fallback.
pub const RATEL_EXPERIMENT_EFFECTIVE_ARM: &str = "ratel.experiment.effective_arm";
/// `ratel.experiment.skip.arm` — shadow arm skipped for capacity.
pub const RATEL_EXPERIMENT_SKIP_ARM: &str = "ratel.experiment.skip.arm";
/// `ratel.experiment.skip.concurrency` — configured shadow concurrency.
pub const RATEL_EXPERIMENT_SKIP_CONCURRENCY: &str = "ratel.experiment.skip.concurrency";
/// `ratel.experiment.skip.reason` — skip reason (see [`ExperimentSkipReason`]).
pub const RATEL_EXPERIMENT_SKIP_REASON: &str = "ratel.experiment.skip.reason";
/// `ratel.experiment.fallback.effective_arm` — fallback arm that served.
pub const RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM: &str = "ratel.experiment.fallback.effective_arm";
/// `ratel.experiment.fallback.reused_shadow` — whether fallback reused admitted shadow work.
pub const RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW: &str = "ratel.experiment.fallback.reused_shadow";
/// `ratel.experiment.drop.reason` — comparison drop reason (see [`ExperimentDropReason`]).
pub const RATEL_EXPERIMENT_DROP_REASON: &str = "ratel.experiment.drop.reason";
/// `ratel.experiment.invocation.attributed` — whether an invocation matched a selection.
pub const RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED: &str = "ratel.experiment.invocation.attributed";
/// `ratel.experiment.invocation.rank` — zero-based first rank, or -1 when absent.
pub const RATEL_EXPERIMENT_INVOCATION_RANK: &str = "ratel.experiment.invocation.rank";
/// `ratel.experiment.invocation.age_ms` — selection age at invocation report time.
pub const RATEL_EXPERIMENT_INVOCATION_AGE_MS: &str = "ratel.experiment.invocation.age_ms";
/// `ratel.experiment.turn` — optional caller-supplied turn annotation.
pub const RATEL_EXPERIMENT_TURN: &str = "ratel.experiment.turn";
/// `ratel.experiment.outcome.label` — free non-empty reported outcome label.
pub const RATEL_EXPERIMENT_OUTCOME_LABEL: &str = "ratel.experiment.outcome.label";
/// `ratel.experiment.outcome.score` — finite reported outcome score.
pub const RATEL_EXPERIMENT_OUTCOME_SCORE: &str = "ratel.experiment.outcome.score";

/// Baggage key matching [`RATEL_EXPERIMENT_ID`].
pub const RATEL_EXPERIMENT_ID_BAGGAGE_KEY: &str = RATEL_EXPERIMENT_ID;
/// Baggage key matching [`RATEL_EXPERIMENT_SELECTION_ID`].
pub const RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY: &str = RATEL_EXPERIMENT_SELECTION_ID;
/// Baggage key matching [`RATEL_EXPERIMENT_ARM`].
pub const RATEL_EXPERIMENT_ARM_BAGGAGE_KEY: &str = RATEL_EXPERIMENT_ARM;
/// Baggage key matching [`RATEL_EXPERIMENT_ROLE`].
pub const RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY: &str = RATEL_EXPERIMENT_ROLE;
/// Baggage key matching [`RATEL_EXPERIMENT_UNIT`].
pub const RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY: &str = RATEL_EXPERIMENT_UNIT;

// ---------------------------------------------------------------------------
// `gen_ai.*` interop keys (CONVENTIONS.md, Tier 1 — borrowed verbatim)
//
// Only the subset the `ratel.*` overlay directly emits on the `execute_tool`
// span. These are OpenTelemetry's, not ours: exposed so callers avoid
// stringly-typing them, never renamed into `ratel.*`.
// ---------------------------------------------------------------------------

/// `gen_ai.operation.name` — set to [`EXECUTE_TOOL`] for a tool invocation.
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
/// `gen_ai.tool.name` — the capability tool id.
pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
/// `gen_ai.tool.call.id` — tool call id, when available.
pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
/// `gen_ai.tool.call.arguments` — tool arguments (Opt-In content, gated).
pub const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
/// `gen_ai.tool.call.result` — tool result (Opt-In content, gated).
pub const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";

// Tier 1 content, carried on the `gen_ai.client.inference.operation.details`
// EventRecord (never span attributes; CONVENTIONS.md § Tier 1 content). Each
// holds a structured v1.42.0 message list (`{ role, parts[], name? }`).

/// `gen_ai.system_instructions` — the system prompt as a bare `parts[]` (Opt-In content).
pub const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
/// `gen_ai.input.messages` — the input message list (Opt-In content).
pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
/// `gen_ai.output.messages` — generated outputs; every message includes `finish_reason`.
pub const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";

/// Where a `ratel.*` span's search came from. Emitted as the `ratel.origin`
/// attribute; mirrors the local trace `Origin` (ADR-0007).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Origin {
    /// A direct library/SDK call — wire value `direct`.
    Direct,
    /// A call the agent synthesized inside its loop (via the capability
    /// tools) — wire value `agent`.
    Agent,
    /// A query recorded while Ratel was observing but not serving retrieval —
    /// wire value `baseline`.
    Baseline,
}

impl Origin {
    /// The wire value carried by `ratel.origin`.
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Direct => "direct",
            Origin::Agent => "agent",
            Origin::Baseline => "baseline",
        }
    }
}

/// What a `ratel.search` span was searching. Emitted as `ratel.search.target`;
/// folds capability-tool search, skill search, and fact search into one span shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchTarget {
    /// The span searched the tool catalog — wire value `tool`.
    Tool,
    /// The span searched the skill catalog — wire value `skill`.
    Skill,
    /// The span searched the fact catalog — wire value `fact`.
    Fact,
}

impl SearchTarget {
    /// The wire value carried by `ratel.search.target`.
    pub fn as_str(self) -> &'static str {
        match self {
            SearchTarget::Tool => "tool",
            SearchTarget::Skill => "skill",
            SearchTarget::Fact => "fact",
        }
    }
}

/// Outcome of an MCP auth flow. Emitted as `ratel.auth.outcome`; `needs_auth`
/// is the 401-driven `AuthNeeds` case (ADR-0007 `auth_needs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthOutcome {
    /// Existing credentials were valid; no interaction needed — wire value `ok`.
    Ok,
    /// Credentials were refreshed without user interaction — wire value
    /// `refreshed`.
    Refreshed,
    /// The upstream challenged for auth (e.g. a 401); user interaction is
    /// required — wire value `needs_auth`, the ADR-0007 `auth_needs` case.
    NeedsAuth,
    /// The flow errored without producing credentials — wire value `failed`.
    Failed,
}

impl AuthOutcome {
    /// The wire value carried by `ratel.auth.outcome`.
    pub fn as_str(self) -> &'static str {
        match self {
            AuthOutcome::Ok => "ok",
            AuthOutcome::Refreshed => "refreshed",
            AuthOutcome::NeedsAuth => "needs_auth",
            AuthOutcome::Failed => "failed",
        }
    }
}

/// Immutable role assigned when an experiment arm is dispatched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentArmRole {
    /// The dispatch may supply the caller-visible result — wire value `serving`.
    Serving,
    /// The dispatch runs for evaluation — wire value `shadow`.
    Shadow,
}

impl ExperimentArmRole {
    /// The wire value carried by `ratel.experiment.role`.
    pub fn as_str(self) -> &'static str {
        match self {
            ExperimentArmRole::Serving => "serving",
            ExperimentArmRole::Shadow => "shadow",
        }
    }
}

/// Completion outcome of one experiment-arm dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentArmOutcome {
    /// Selection returned a non-empty ranking — wire value `ok`.
    Ok,
    /// Selection returned an empty ranking — wire value `empty`.
    Empty,
    /// The arm rejected with a `TimeoutError` — wire value `timeout`.
    Timeout,
    /// Any other arm, transform, or ranking failure — wire value `error`.
    Error,
}

impl ExperimentArmOutcome {
    /// The wire value carried by `ratel.experiment.outcome`.
    pub fn as_str(self) -> &'static str {
        match self {
            ExperimentArmOutcome::Ok => "ok",
            ExperimentArmOutcome::Empty => "empty",
            ExperimentArmOutcome::Timeout => "timeout",
            ExperimentArmOutcome::Error => "error",
        }
    }
}

/// Reason a requested shadow dispatch did not start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentSkipReason {
    /// No shadow slot was available — wire value `capacity`.
    Capacity,
}

impl ExperimentSkipReason {
    /// The wire value carried by `ratel.experiment.skip.reason`.
    pub fn as_str(self) -> &'static str {
        match self {
            ExperimentSkipReason::Capacity => "capacity",
        }
    }
}

/// Terminal reason a peer-eligible shadow produced no comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentDropReason {
    /// Shadow selection, transform, or ranking failed — wire value `arm-failed`.
    ArmFailed,
    /// The shadow result became the effective fallback — wire value `fallback-consumed`.
    FallbackConsumed,
    /// No arm produced an effective result — wire value `selection-failed`.
    SelectionFailed,
    /// The effective served ranking failed — wire value `served-ranking-failed`.
    ServedRankingFailed,
    /// Comparison itself failed — wire value `comparison-failed`.
    ComparisonFailed,
}

impl ExperimentDropReason {
    /// The wire value carried by `ratel.experiment.drop.reason`.
    pub fn as_str(self) -> &'static str {
        match self {
            ExperimentDropReason::ArmFailed => "arm-failed",
            ExperimentDropReason::FallbackConsumed => "fallback-consumed",
            ExperimentDropReason::SelectionFailed => "selection-failed",
            ExperimentDropReason::ServedRankingFailed => "served-ranking-failed",
            ExperimentDropReason::ComparisonFailed => "comparison-failed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_maps_to_wire_strings() {
        assert_eq!(Origin::Direct.as_str(), "direct");
        assert_eq!(Origin::Agent.as_str(), "agent");
        assert_eq!(Origin::Baseline.as_str(), "baseline");
    }

    #[test]
    fn search_target_maps_to_wire_strings() {
        assert_eq!(SearchTarget::Tool.as_str(), "tool");
        assert_eq!(SearchTarget::Skill.as_str(), "skill");
        assert_eq!(SearchTarget::Fact.as_str(), "fact");
    }

    #[test]
    fn auth_outcome_maps_to_wire_strings() {
        assert_eq!(AuthOutcome::Ok.as_str(), "ok");
        assert_eq!(AuthOutcome::Refreshed.as_str(), "refreshed");
        assert_eq!(AuthOutcome::NeedsAuth.as_str(), "needs_auth");
        assert_eq!(AuthOutcome::Failed.as_str(), "failed");
    }

    #[test]
    fn experiment_arm_role_maps_to_wire_strings() {
        assert_eq!(ExperimentArmRole::Serving.as_str(), "serving");
        assert_eq!(ExperimentArmRole::Shadow.as_str(), "shadow");
    }

    #[test]
    fn experiment_arm_outcome_maps_to_wire_strings() {
        assert_eq!(ExperimentArmOutcome::Ok.as_str(), "ok");
        assert_eq!(ExperimentArmOutcome::Empty.as_str(), "empty");
        assert_eq!(ExperimentArmOutcome::Timeout.as_str(), "timeout");
        assert_eq!(ExperimentArmOutcome::Error.as_str(), "error");
    }

    #[test]
    fn experiment_skip_reason_maps_to_wire_strings() {
        assert_eq!(ExperimentSkipReason::Capacity.as_str(), "capacity");
    }

    #[test]
    fn experiment_drop_reason_maps_to_wire_strings() {
        assert_eq!(ExperimentDropReason::ArmFailed.as_str(), "arm-failed");
        assert_eq!(
            ExperimentDropReason::FallbackConsumed.as_str(),
            "fallback-consumed"
        );
        assert_eq!(
            ExperimentDropReason::SelectionFailed.as_str(),
            "selection-failed"
        );
        assert_eq!(
            ExperimentDropReason::ServedRankingFailed.as_str(),
            "served-ranking-failed"
        );
        assert_eq!(
            ExperimentDropReason::ComparisonFailed.as_str(),
            "comparison-failed"
        );
    }

    #[test]
    fn ratel_attribute_keys_match_the_pin() {
        assert_eq!(RATEL_EVENT_ID, "ratel.event.id");
        assert_eq!(RATEL_ORIGIN, "ratel.origin");
        assert_eq!(RATEL_SEARCH_TARGET, "ratel.search.target");
        assert_eq!(RATEL_SEARCH_TOP_K, "ratel.search.top_k");
        assert_eq!(RATEL_SEARCH_HIT_COUNT, "ratel.search.hit_count");
        assert_eq!(RATEL_SEARCH_QUERY, "ratel.search.query");
        assert_eq!(RATEL_TOOL_ARGS_SIZE_BYTES, "ratel.tool.args_size_bytes");
        assert_eq!(RATEL_UPSTREAM_SERVER, "ratel.upstream.server");
        assert_eq!(RATEL_UPSTREAM_TRANSPORT, "ratel.upstream.transport");
        assert_eq!(RATEL_UPSTREAM_TOOL_COUNT, "ratel.upstream.tool_count");
        assert_eq!(RATEL_SKILL_ID, "ratel.skill.id");
        assert_eq!(RATEL_AUTH_OUTCOME, "ratel.auth.outcome");
        assert_eq!(RATEL_EXPERIMENT_ID, "ratel.experiment.id");
        assert_eq!(
            RATEL_EXPERIMENT_SELECTION_ID,
            "ratel.experiment.selection_id"
        );
        assert_eq!(RATEL_EXPERIMENT_ARM, "ratel.experiment.arm");
        assert_eq!(RATEL_EXPERIMENT_ROLE, "ratel.experiment.role");
        assert_eq!(RATEL_EXPERIMENT_UNIT, "ratel.experiment.unit");
        assert_eq!(RATEL_EXPERIMENT_COLD, "ratel.experiment.cold");
        assert_eq!(RATEL_EXPERIMENT_OUTCOME, "ratel.experiment.outcome");
        assert_eq!(RATEL_EXPERIMENT_DURATION_MS, "ratel.experiment.duration_ms");
        assert_eq!(RATEL_EXPERIMENT_HIT_COUNT, "ratel.experiment.hit_count");
        assert_eq!(
            RATEL_EXPERIMENT_RANKING_ERROR,
            "ratel.experiment.ranking_error"
        );
        assert_eq!(
            RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR,
            "ratel.experiment.result_attributes_error"
        );
        assert_eq!(
            RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR,
            "ratel.experiment.result_attrs_encoding_error"
        );
        assert_eq!(RATEL_EXPERIMENT_RESULT_IDS, "ratel.experiment.result_ids");
        assert_eq!(
            RATEL_EXPERIMENT_RESULT_SCORES,
            "ratel.experiment.result_scores"
        );
        assert_eq!(
            RATEL_EXPERIMENT_RESULT_ATTRS,
            "ratel.experiment.result_attrs"
        );
        assert_eq!(RATEL_EXPERIMENT_SERVED_ARM, "ratel.experiment.served.arm");
        assert_eq!(
            RATEL_EXPERIMENT_SERVED_OUTCOME,
            "ratel.experiment.served.outcome"
        );
        assert_eq!(
            RATEL_EXPERIMENT_SERVED_DURATION_MS,
            "ratel.experiment.served.duration_ms"
        );
        assert_eq!(
            RATEL_EXPERIMENT_SERVED_HIT_COUNT,
            "ratel.experiment.served.hit_count"
        );
        assert_eq!(RATEL_EXPERIMENT_SHADOW_ARM, "ratel.experiment.shadow.arm");
        assert_eq!(
            RATEL_EXPERIMENT_SHADOW_OUTCOME,
            "ratel.experiment.shadow.outcome"
        );
        assert_eq!(
            RATEL_EXPERIMENT_SHADOW_DURATION_MS,
            "ratel.experiment.shadow.duration_ms"
        );
        assert_eq!(
            RATEL_EXPERIMENT_SHADOW_HIT_COUNT,
            "ratel.experiment.shadow.hit_count"
        );
        assert_eq!(
            RATEL_EXPERIMENT_AGREEMENT_TOP1,
            "ratel.experiment.agreement.top1"
        );
        assert_eq!(
            RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER,
            "ratel.experiment.agreement.exact_order"
        );
        assert_eq!(
            RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT,
            "ratel.experiment.agreement.overlap_count"
        );
        assert_eq!(
            RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K,
            "ratel.experiment.agreement.jaccard_at_k"
        );
        assert_eq!(RATEL_EXPERIMENT_AGREEMENT_K, "ratel.experiment.agreement.k");
        assert_eq!(
            RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS,
            "ratel.experiment.agreement.item_attrs"
        );
        assert_eq!(
            RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS,
            "ratel.experiment.agreement.result_attrs"
        );
        assert_eq!(
            RATEL_EXPERIMENT_EFFECTIVE_ARM,
            "ratel.experiment.effective_arm"
        );
        assert_eq!(RATEL_EXPERIMENT_SKIP_ARM, "ratel.experiment.skip.arm");
        assert_eq!(
            RATEL_EXPERIMENT_SKIP_CONCURRENCY,
            "ratel.experiment.skip.concurrency"
        );
        assert_eq!(RATEL_EXPERIMENT_SKIP_REASON, "ratel.experiment.skip.reason");
        assert_eq!(
            RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM,
            "ratel.experiment.fallback.effective_arm"
        );
        assert_eq!(
            RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW,
            "ratel.experiment.fallback.reused_shadow"
        );
        assert_eq!(RATEL_EXPERIMENT_DROP_REASON, "ratel.experiment.drop.reason");
        assert_eq!(
            RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED,
            "ratel.experiment.invocation.attributed"
        );
        assert_eq!(
            RATEL_EXPERIMENT_INVOCATION_RANK,
            "ratel.experiment.invocation.rank"
        );
        assert_eq!(
            RATEL_EXPERIMENT_INVOCATION_AGE_MS,
            "ratel.experiment.invocation.age_ms"
        );
        assert_eq!(RATEL_EXPERIMENT_TURN, "ratel.experiment.turn");
        assert_eq!(
            RATEL_EXPERIMENT_OUTCOME_LABEL,
            "ratel.experiment.outcome.label"
        );
        assert_eq!(
            RATEL_EXPERIMENT_OUTCOME_SCORE,
            "ratel.experiment.outcome.score"
        );
    }

    #[test]
    fn experiment_baggage_keys_alias_matching_span_attributes() {
        assert_eq!(RATEL_EXPERIMENT_ID_BAGGAGE_KEY, RATEL_EXPERIMENT_ID);
        assert_eq!(
            RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY,
            RATEL_EXPERIMENT_SELECTION_ID
        );
        assert_eq!(RATEL_EXPERIMENT_ARM_BAGGAGE_KEY, RATEL_EXPERIMENT_ARM);
        assert_eq!(RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY, RATEL_EXPERIMENT_ROLE);
        assert_eq!(RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY, RATEL_EXPERIMENT_UNIT);
    }

    #[test]
    fn semconv_pin_is_explicit() {
        // The pin IS the contract; consumers read against this exact version,
        // not "latest". Bumping it is a reviewed change (CONVENTIONS.md § The pin).
        assert_eq!(SEMCONV_VERSION, "1.42.0");
    }

    #[test]
    fn content_capture_gate_is_the_ecosystem_env_var() {
        assert_eq!(
            CAPTURE_CONTENT_ENV,
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        );
        assert_eq!(
            EXPERIMENTAL_CATALOG_DEFINITIONS_ENV,
            "RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS"
        );
    }

    #[test]
    fn event_record_names_match_the_pin() {
        assert_eq!(RATEL_CATALOG_DEFINITION, "ratel.catalog.definition");
        assert_eq!(RATEL_EXPERIMENT_RESULTS, "ratel.experiment.results");
        assert_eq!(RATEL_EXPERIMENT_COMPARISON, "ratel.experiment.comparison");
        assert_eq!(RATEL_EXPERIMENT_SKIP, "ratel.experiment.skip");
        assert_eq!(RATEL_EXPERIMENT_FALLBACK, "ratel.experiment.fallback");
        assert_eq!(RATEL_EXPERIMENT_DROP, "ratel.experiment.drop");
        assert_eq!(RATEL_EXPERIMENT_INVOCATION, "ratel.experiment.invocation");
        assert_eq!(RATEL_EXPERIMENT_OUTCOME, "ratel.experiment.outcome");
        assert_eq!(RATEL_SEARCH_RESULTS, "ratel.search.results");
        assert_eq!(RATEL_TOOL_EXECUTION_DETAILS, "ratel.tool.execution.details");
        assert_eq!(
            GEN_AI_INFERENCE_DETAILS,
            "gen_ai.client.inference.operation.details"
        );
    }

    #[test]
    fn catalog_definition_attribute_vocabulary_matches_the_pin() {
        assert_eq!(
            [
                RATEL_CATALOG_KIND,
                RATEL_CATALOG_ID,
                RATEL_CATALOG_NAME,
                RATEL_CATALOG_DESCRIPTION,
                RATEL_CATALOG_TAGS,
                RATEL_CATALOG_INPUT_SCHEMA,
                RATEL_CATALOG_OUTPUT_SCHEMA,
                RATEL_CATALOG_SCHEMA_OMITTED,
                RATEL_CATALOG_SEARCHABLE_DESCRIPTION,
                RATEL_CATALOG_SEARCHABLE_DESCRIPTION_OVERRIDDEN,
                RATEL_CATALOG_USE_DEFINITION_OVERRIDES,
                RATEL_CATALOG_CONTENT_HASH,
            ],
            [
                "ratel.catalog.kind",
                "ratel.catalog.id",
                "ratel.catalog.name",
                "ratel.catalog.description",
                "ratel.catalog.tags",
                "ratel.catalog.input_schema",
                "ratel.catalog.output_schema",
                "ratel.catalog.schema_omitted",
                "ratel.catalog.searchable_description",
                "ratel.catalog.searchable_description_overridden",
                "ratel.catalog.use_definition_overrides",
                "ratel.catalog.content_hash",
            ]
        );
    }

    #[test]
    fn gen_ai_content_message_keys_match_the_pin() {
        // Tier 1 content carried on the inference-details event (CONVENTIONS.md
        // § Tier 1 content): borrowed verbatim from gen_ai, never renamed.
        assert_eq!(GEN_AI_SYSTEM_INSTRUCTIONS, "gen_ai.system_instructions");
        assert_eq!(GEN_AI_INPUT_MESSAGES, "gen_ai.input.messages");
        assert_eq!(GEN_AI_OUTPUT_MESSAGES, "gen_ai.output.messages");
        for key in [
            GEN_AI_SYSTEM_INSTRUCTIONS,
            GEN_AI_INPUT_MESSAGES,
            GEN_AI_OUTPUT_MESSAGES,
        ] {
            assert!(key.starts_with("gen_ai."), "{key} is not under gen_ai.*");
            assert!(
                !key.starts_with("ratel."),
                "{key} must not be renamed into ratel.*"
            );
        }
    }

    #[test]
    fn span_names_match_the_pin() {
        assert_eq!(RATEL_SEARCH, "ratel.search");
        assert_eq!(RATEL_EXPERIMENT_ARM, "ratel.experiment.arm");
        assert_eq!(RATEL_SKILL_LOAD, "ratel.skill.load");
        assert_eq!(RATEL_UPSTREAM_REGISTER, "ratel.upstream.register");
        assert_eq!(RATEL_AUTH_FLOW, "ratel.auth.flow");
    }

    #[test]
    fn tool_invocation_is_the_gen_ai_execute_tool_operation_not_ratel_invoke() {
        // Locked 2026-07-05: invoke rides an `execute_tool` gen_ai span for
        // OTel-backend interop, not a bespoke `ratel.invoke` span.
        assert_eq!(EXECUTE_TOOL, "execute_tool");
        assert_ne!(EXECUTE_TOOL, "ratel.invoke");
    }

    #[test]
    fn gen_ai_interop_keys_match_the_pin() {
        // Tier 1: borrowed verbatim from OTel gen_ai; the execute_tool overlay
        // rides these, so they must stay under gen_ai.* (interop), never ratel.*.
        assert_eq!(GEN_AI_OPERATION_NAME, "gen_ai.operation.name");
        assert_eq!(GEN_AI_TOOL_NAME, "gen_ai.tool.name");
        assert_eq!(GEN_AI_TOOL_CALL_ID, "gen_ai.tool.call.id");
        assert_eq!(GEN_AI_TOOL_CALL_ARGUMENTS, "gen_ai.tool.call.arguments");
        assert_eq!(GEN_AI_TOOL_CALL_RESULT, "gen_ai.tool.call.result");
        for key in [
            GEN_AI_OPERATION_NAME,
            GEN_AI_TOOL_NAME,
            GEN_AI_TOOL_CALL_ID,
            GEN_AI_TOOL_CALL_ARGUMENTS,
            GEN_AI_TOOL_CALL_RESULT,
        ] {
            assert!(key.starts_with("gen_ai."), "{key} is not under gen_ai.*");
            assert!(
                !key.starts_with("ratel."),
                "{key} must not be renamed into ratel.*"
            );
        }
    }

    #[test]
    fn every_ratel_attribute_key_is_namespaced() {
        for key in [
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
        ] {
            assert!(key.starts_with("ratel."), "{key} is not under ratel.*");
        }
    }

    #[test]
    fn attribute_keys_are_unique() {
        // A copy-paste dup (two concepts sharing a wire key) passes every
        // per-constant assert above; only a uniqueness check catches it.
        let keys = [
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
            GEN_AI_SYSTEM_INSTRUCTIONS,
            GEN_AI_INPUT_MESSAGES,
            GEN_AI_OUTPUT_MESSAGES,
        ];
        let unique: std::collections::HashSet<&str> = keys.iter().copied().collect();
        assert_eq!(unique.len(), keys.len(), "duplicate attribute key");
    }

    #[test]
    fn span_names_are_unique() {
        // Same copy-paste risk as attribute_keys_are_unique, for the span names.
        let names = [
            RATEL_SEARCH,
            EXECUTE_TOOL,
            RATEL_EXPERIMENT_ARM,
            RATEL_SKILL_LOAD,
            RATEL_UPSTREAM_REGISTER,
            RATEL_AUTH_FLOW,
        ];
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "duplicate span name");
    }

    #[test]
    fn event_names_are_unique() {
        // Same copy-paste risk for the EventRecord names.
        let names = [
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
        ];
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "duplicate event name");
    }
}
