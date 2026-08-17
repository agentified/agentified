/**
 * `@ratel-ai/telemetry` — the `ratel.*` telemetry vocabulary.
 *
 * See the wire contract in `../CONVENTIONS.md`. Emitting the vocabulary is done
 * through the standard OpenTelemetry JS SDK; this package adds no transport and
 * no schema (ADR-0007). These constants are the `ratel.*` overlay Ratel owns,
 * plus the small subset of `gen_ai.*` keys the overlay emits directly on the
 * `execute_tool` span (borrowed verbatim from OpenTelemetry, never renamed).
 *
 * This package is OTel-free (ADR-0007): the only behavior it carries beyond the
 * constants is the content-capture gate, re-exported from `./content-capture`.
 * Wiring a provider — endpoint, auth, exporters, processors, flush — is the
 * host's job.
 */

export {
  ContentCapture,
  clearContentCapture,
  contentCaptureMode,
  setContentCapture,
} from "./content-capture.js";

/**
 * The pinned OpenTelemetry semantic-conventions version this vocabulary tracks
 * (the `gen_ai` group). The pin is the contract; consumers read against this
 * exact version, never "latest" (CONVENTIONS.md § The pin).
 */
export const SEMCONV_VERSION = "1.42.0";

/**
 * The ecosystem instrumentation env var gating message/tool content capture.
 * Default off; the shared ecosystem name rather than a Ratel-invented flag
 * (CONVENTIONS.md § Capture gating). Values: legacy boolean, or the enum
 * `NO_CONTENT` (default) / `SPAN_ONLY` / `EVENT_ONLY` / `SPAN_AND_EVENT`.
 */
export const CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT";

// ---------------------------------------------------------------------------
// Span names (CONVENTIONS.md, Tier 2)
// ---------------------------------------------------------------------------

/** `ratel.search` — capability search (unifies tool-search and skill-search). */
export const RATEL_SEARCH = "ratel.search";

/** `ratel.experiment.arm` — one serving or shadow experiment-arm dispatch. */
export const RATEL_EXPERIMENT_ARM = "ratel.experiment.arm";

/**
 * `execute_tool` — the `gen_ai.operation.name` value for a tool invocation.
 *
 * Deliberately the standard OTel `gen_ai` operation, not a bespoke `ratel.invoke`
 * span, so a generic OTel backend already understands it (locked 2026-07-05). The
 * invoke is enriched with `ratel.*` attributes.
 */
export const EXECUTE_TOOL = "execute_tool";

/** `ratel.skill.load` — skill content load (`get_skill_content`). */
export const RATEL_SKILL_LOAD = "ratel.skill.load";

/** `ratel.upstream.register` — upstream-MCP ingest. */
export const RATEL_UPSTREAM_REGISTER = "ratel.upstream.register";

/** `ratel.auth.flow` — MCP auth flow. */
export const RATEL_AUTH_FLOW = "ratel.auth.flow";

// ---------------------------------------------------------------------------
// EventRecord names (CONVENTIONS.md)
// ---------------------------------------------------------------------------

/**
 * `ratel.search.results` — Opt-In search-content event; gated like content.
 */
export const RATEL_SEARCH_RESULTS = "ratel.search.results";

/** `ratel.tool.execution.details` — Opt-In structured tool arguments/result event. */
export const RATEL_TOOL_EXECUTION_DETAILS = "ratel.tool.execution.details";

/** `ratel.experiment.results` — ranked measurement for one experiment arm. */
export const RATEL_EXPERIMENT_RESULTS = "ratel.experiment.results";

/** `ratel.experiment.comparison` — one shadow-vs-served comparison. */
export const RATEL_EXPERIMENT_COMPARISON = "ratel.experiment.comparison";

/** `ratel.experiment.skip` — one shadow dispatch skipped for capacity. */
export const RATEL_EXPERIMENT_SKIP = "ratel.experiment.skip";

/** `ratel.experiment.fallback` — one successful fallback selection. */
export const RATEL_EXPERIMENT_FALLBACK = "ratel.experiment.fallback";

/** `ratel.experiment.drop` — one peer comparison that could not be emitted. */
export const RATEL_EXPERIMENT_DROP = "ratel.experiment.drop";

/** `ratel.experiment.invocation` — attributed or unattributed tool invocation. */
export const RATEL_EXPERIMENT_INVOCATION = "ratel.experiment.invocation";

/**
 * `gen_ai.client.inference.operation.details` — the event that carries inference
 * request and response content. Borrowed from gen_ai (Tier 1).
 */
export const GEN_AI_INFERENCE_DETAILS = "gen_ai.client.inference.operation.details";

// ---------------------------------------------------------------------------
// `ratel.*` attribute keys (CONVENTIONS.md, Tier 2)
// ---------------------------------------------------------------------------

/** `ratel.origin` — direct library call vs agent-synthesized (shared attribute). */
export const RATEL_ORIGIN = "ratel.origin";

/** `ratel.event.id` — runtime-event/OTel deduplication and join key. */
export const RATEL_EVENT_ID = "ratel.event.id";

/** `ratel.search.target` — `tool`, `skill`, or `fact` (see {@link SearchTarget}). */
export const RATEL_SEARCH_TARGET = "ratel.search.target";

/** `ratel.search.top_k` — requested result count. */
export const RATEL_SEARCH_TOP_K = "ratel.search.top_k";

/** `ratel.search.hit_count` — results returned. */
export const RATEL_SEARCH_HIT_COUNT = "ratel.search.hit_count";

/** `ratel.search.query` — the search text (content, gated like message content). */
export const RATEL_SEARCH_QUERY = "ratel.search.query";

/** `ratel.tool.args_size_bytes` — argument payload size on the `execute_tool` span. */
export const RATEL_TOOL_ARGS_SIZE_BYTES = "ratel.tool.args_size_bytes";

/** `ratel.upstream.server` — upstream MCP server backing a tool / auth flow. */
export const RATEL_UPSTREAM_SERVER = "ratel.upstream.server";

/** `ratel.upstream.transport` — `stdio` / `http` / `sse` / ... */
export const RATEL_UPSTREAM_TRANSPORT = "ratel.upstream.transport";

/** `ratel.upstream.tool_count` — tools ingested on register. */
export const RATEL_UPSTREAM_TOOL_COUNT = "ratel.upstream.tool_count";

/** `ratel.skill.id` — skill loaded on the `ratel.skill.load` span. */
export const RATEL_SKILL_ID = "ratel.skill.id";

/** `ratel.auth.outcome` — `ok` / `refreshed` / `needs_auth` / `failed` (see {@link AuthOutcome}). */
export const RATEL_AUTH_OUTCOME = "ratel.auth.outcome";

/** `ratel.experiment.id` — configured experiment id. */
export const RATEL_EXPERIMENT_ID = "ratel.experiment.id";

/** `ratel.experiment.selection_id` — opaque selection correlation id. */
export const RATEL_EXPERIMENT_SELECTION_ID = "ratel.experiment.selection_id";

/** `ratel.experiment.role` — immutable dispatch role (see {@link ExperimentArmRole}). */
export const RATEL_EXPERIMENT_ROLE = "ratel.experiment.role";

/** `ratel.experiment.unit` — pseudonymous 16-hex unit hash. */
export const RATEL_EXPERIMENT_UNIT = "ratel.experiment.unit";

/** `ratel.experiment.cold` — whether warmup was unresolved when selection began. */
export const RATEL_EXPERIMENT_COLD = "ratel.experiment.cold";

/**
 * `ratel.experiment.outcome` — arm completion attribute and reported-outcome EventRecord name
 * (see {@link ExperimentArmOutcome}).
 */
export const RATEL_EXPERIMENT_OUTCOME = "ratel.experiment.outcome";

/** `ratel.experiment.duration_ms` — arm callback duration in milliseconds. */
export const RATEL_EXPERIMENT_DURATION_MS = "ratel.experiment.duration_ms";

/** `ratel.experiment.hit_count` — ranked result count when ranking succeeds. */
export const RATEL_EXPERIMENT_HIT_COUNT = "ratel.experiment.hit_count";

/** `ratel.experiment.ranking_error` — error type when ranking projection fails. */
export const RATEL_EXPERIMENT_RANKING_ERROR = "ratel.experiment.ranking_error";

/** `ratel.experiment.result_attributes_error` — result-level projection error type. */
export const RATEL_EXPERIMENT_RESULT_ATTRIBUTES_ERROR = "ratel.experiment.result_attributes_error";

/** `ratel.experiment.result_attrs_encoding_error` — gated item-attribute encoding error type. */
export const RATEL_EXPERIMENT_RESULT_ATTRS_ENCODING_ERROR =
  "ratel.experiment.result_attrs_encoding_error";

/** `ratel.experiment.result_ids` — ordered ranked result identifiers. */
export const RATEL_EXPERIMENT_RESULT_IDS = "ratel.experiment.result_ids";

/** `ratel.experiment.result_scores` — ordered finite scores when every item has one. */
export const RATEL_EXPERIMENT_RESULT_SCORES = "ratel.experiment.result_scores";

/** `ratel.experiment.result_attrs` — capture-gated item attributes aligned to result ids. */
export const RATEL_EXPERIMENT_RESULT_ATTRS = "ratel.experiment.result_attrs";

/** `ratel.experiment.served.arm` — effective served arm in a peer comparison. */
export const RATEL_EXPERIMENT_SERVED_ARM = "ratel.experiment.served.arm";

/** `ratel.experiment.served.outcome` — effective served arm outcome. */
export const RATEL_EXPERIMENT_SERVED_OUTCOME = "ratel.experiment.served.outcome";

/** `ratel.experiment.served.duration_ms` — effective served arm callback duration. */
export const RATEL_EXPERIMENT_SERVED_DURATION_MS = "ratel.experiment.served.duration_ms";

/** `ratel.experiment.served.hit_count` — effective served ranking size. */
export const RATEL_EXPERIMENT_SERVED_HIT_COUNT = "ratel.experiment.served.hit_count";

/** `ratel.experiment.shadow.arm` — compared shadow arm. */
export const RATEL_EXPERIMENT_SHADOW_ARM = "ratel.experiment.shadow.arm";

/** `ratel.experiment.shadow.outcome` — compared shadow arm outcome. */
export const RATEL_EXPERIMENT_SHADOW_OUTCOME = "ratel.experiment.shadow.outcome";

/** `ratel.experiment.shadow.duration_ms` — compared shadow arm callback duration. */
export const RATEL_EXPERIMENT_SHADOW_DURATION_MS = "ratel.experiment.shadow.duration_ms";

/** `ratel.experiment.shadow.hit_count` — compared shadow ranking size. */
export const RATEL_EXPERIMENT_SHADOW_HIT_COUNT = "ratel.experiment.shadow.hit_count";

/** `ratel.experiment.agreement.top1` — whether first ranked ids agree. */
export const RATEL_EXPERIMENT_AGREEMENT_TOP1 = "ratel.experiment.agreement.top1";

/** `ratel.experiment.agreement.exact_order` — whether complete ordered ids agree. */
export const RATEL_EXPERIMENT_AGREEMENT_EXACT_ORDER = "ratel.experiment.agreement.exact_order";

/** `ratel.experiment.agreement.overlap_count` — complete-set overlap count. */
export const RATEL_EXPERIMENT_AGREEMENT_OVERLAP_COUNT = "ratel.experiment.agreement.overlap_count";

/** `ratel.experiment.agreement.jaccard_at_k` — rank-window Jaccard agreement. */
export const RATEL_EXPERIMENT_AGREEMENT_JACCARD_AT_K = "ratel.experiment.agreement.jaccard_at_k";

/** `ratel.experiment.agreement.k` — rank window used for Jaccard. */
export const RATEL_EXPERIMENT_AGREEMENT_K = "ratel.experiment.agreement.k";

/** `ratel.experiment.agreement.item_attrs` — rank-zero shared-key agreement map. */
export const RATEL_EXPERIMENT_AGREEMENT_ITEM_ATTRS = "ratel.experiment.agreement.item_attrs";

/** `ratel.experiment.agreement.result_attrs` — result-level union-key agreement map. */
export const RATEL_EXPERIMENT_AGREEMENT_RESULT_ATTRS = "ratel.experiment.agreement.result_attrs";

/** `ratel.experiment.effective_arm` — caller-visible arm after fallback. */
export const RATEL_EXPERIMENT_EFFECTIVE_ARM = "ratel.experiment.effective_arm";

/** `ratel.experiment.skip.arm` — shadow arm skipped for capacity. */
export const RATEL_EXPERIMENT_SKIP_ARM = "ratel.experiment.skip.arm";

/** `ratel.experiment.skip.concurrency` — configured shadow concurrency. */
export const RATEL_EXPERIMENT_SKIP_CONCURRENCY = "ratel.experiment.skip.concurrency";

/** `ratel.experiment.skip.reason` — skip reason (see {@link ExperimentSkipReason}). */
export const RATEL_EXPERIMENT_SKIP_REASON = "ratel.experiment.skip.reason";

/** `ratel.experiment.fallback.effective_arm` — fallback arm that served. */
export const RATEL_EXPERIMENT_FALLBACK_EFFECTIVE_ARM = "ratel.experiment.fallback.effective_arm";

/** `ratel.experiment.fallback.reused_shadow` — whether fallback reused admitted shadow work. */
export const RATEL_EXPERIMENT_FALLBACK_REUSED_SHADOW = "ratel.experiment.fallback.reused_shadow";

/** `ratel.experiment.drop.reason` — comparison drop reason (see {@link ExperimentDropReason}). */
export const RATEL_EXPERIMENT_DROP_REASON = "ratel.experiment.drop.reason";

/** `ratel.experiment.invocation.attributed` — whether an invocation matched a selection. */
export const RATEL_EXPERIMENT_INVOCATION_ATTRIBUTED = "ratel.experiment.invocation.attributed";

/** `ratel.experiment.invocation.rank` — zero-based first rank, or -1 when absent. */
export const RATEL_EXPERIMENT_INVOCATION_RANK = "ratel.experiment.invocation.rank";

/** `ratel.experiment.invocation.age_ms` — selection age at invocation report time. */
export const RATEL_EXPERIMENT_INVOCATION_AGE_MS = "ratel.experiment.invocation.age_ms";

/** `ratel.experiment.turn` — optional caller-supplied turn annotation. */
export const RATEL_EXPERIMENT_TURN = "ratel.experiment.turn";

/** `ratel.experiment.outcome.label` — free non-empty reported outcome label. */
export const RATEL_EXPERIMENT_OUTCOME_LABEL = "ratel.experiment.outcome.label";

/** `ratel.experiment.outcome.score` — finite reported outcome score. */
export const RATEL_EXPERIMENT_OUTCOME_SCORE = "ratel.experiment.outcome.score";

// ---------------------------------------------------------------------------
// Experiment baggage keys (ADR-0019)
//
// The controlled dispatch stamp uses the same wire keys as the matching span
// attributes. Explicit aliases keep baggage construction discoverable without
// allowing the vocabulary to drift.
// ---------------------------------------------------------------------------

/** Baggage key matching {@link RATEL_EXPERIMENT_ID}. */
export const RATEL_EXPERIMENT_ID_BAGGAGE_KEY = RATEL_EXPERIMENT_ID;

/** Baggage key matching {@link RATEL_EXPERIMENT_SELECTION_ID}. */
export const RATEL_EXPERIMENT_SELECTION_ID_BAGGAGE_KEY = RATEL_EXPERIMENT_SELECTION_ID;

/** Baggage key matching {@link RATEL_EXPERIMENT_ARM}. */
export const RATEL_EXPERIMENT_ARM_BAGGAGE_KEY = RATEL_EXPERIMENT_ARM;

/** Baggage key matching {@link RATEL_EXPERIMENT_ROLE}. */
export const RATEL_EXPERIMENT_ROLE_BAGGAGE_KEY = RATEL_EXPERIMENT_ROLE;

/** Baggage key matching {@link RATEL_EXPERIMENT_UNIT}. */
export const RATEL_EXPERIMENT_UNIT_BAGGAGE_KEY = RATEL_EXPERIMENT_UNIT;

// ---------------------------------------------------------------------------
// `gen_ai.*` interop keys (CONVENTIONS.md, Tier 1 — borrowed verbatim)
//
// Only the subset the `ratel.*` overlay directly emits on the `execute_tool`
// span. These are OpenTelemetry's, not ours: exposed so callers avoid
// stringly-typing them, never renamed into `ratel.*`.
// ---------------------------------------------------------------------------

/** `gen_ai.operation.name` — set to {@link EXECUTE_TOOL} for a tool invocation. */
export const GEN_AI_OPERATION_NAME = "gen_ai.operation.name";

/** `gen_ai.tool.name` — the capability tool id. */
export const GEN_AI_TOOL_NAME = "gen_ai.tool.name";

/** `gen_ai.tool.call.id` — tool call id, when available. */
export const GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id";

/** `gen_ai.tool.call.arguments` — tool arguments (Opt-In content, gated). */
export const GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments";

/** `gen_ai.tool.call.result` — tool result (Opt-In content, gated). */
export const GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result";

// Tier 1 content, carried on the `gen_ai.client.inference.operation.details`
// EventRecord (never span attributes; CONVENTIONS.md § Tier 1 content). Each
// holds a structured v1.42.0 message list (`{ role, parts[], name? }`).

/** `gen_ai.system_instructions` — the system prompt as a bare `parts[]` (Opt-In content). */
export const GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions";

/** `gen_ai.input.messages` — the input message list (Opt-In content). */
export const GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages";

/** `gen_ai.output.messages` — generated outputs; every message includes `finish_reason`. */
export const GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages";

// ---------------------------------------------------------------------------
// Enum wire values (CONVENTIONS.md, Tier 2)
//
// Modelled as `as const` value maps: the key is the ergonomic name, the value
// is the exact wire string carried by the corresponding attribute. Each has a
// companion type so callers get the closed union.
// ---------------------------------------------------------------------------

/**
 * Where a `ratel.*` span's search came from. Carried by `ratel.origin`; mirrors
 * the local trace `Origin` (ADR-0007).
 *
 * `Baseline` marks a query recorded while Ratel was observing but not serving
 * retrieval.
 */
export const Origin = {
  Direct: "direct",
  Agent: "agent",
  Baseline: "baseline",
} as const;
export type Origin = (typeof Origin)[keyof typeof Origin];

/**
 * What a `ratel.search` span was searching. Carried by `ratel.search.target`;
 * folds capability-tool search, skill search, and fact search into one span shape.
 */
export const SearchTarget = {
  Tool: "tool",
  Skill: "skill",
  Fact: "fact",
} as const;
export type SearchTarget = (typeof SearchTarget)[keyof typeof SearchTarget];

/**
 * Outcome of an MCP auth flow. Carried by `ratel.auth.outcome`; `NeedsAuth` is
 * the 401-driven `AuthNeeds` case (ADR-0007 `auth_needs`).
 */
export const AuthOutcome = {
  Ok: "ok",
  Refreshed: "refreshed",
  NeedsAuth: "needs_auth",
  Failed: "failed",
} as const;
export type AuthOutcome = (typeof AuthOutcome)[keyof typeof AuthOutcome];

/** Immutable role assigned when an experiment arm is dispatched. */
export const ExperimentArmRole = {
  Serving: "serving",
  Shadow: "shadow",
} as const;
export type ExperimentArmRole = (typeof ExperimentArmRole)[keyof typeof ExperimentArmRole];

/** Completion outcome of one experiment-arm dispatch. */
export const ExperimentArmOutcome = {
  Ok: "ok",
  Empty: "empty",
  Timeout: "timeout",
  Error: "error",
} as const;
export type ExperimentArmOutcome = (typeof ExperimentArmOutcome)[keyof typeof ExperimentArmOutcome];

/** Reason a requested shadow dispatch did not start. */
export const ExperimentSkipReason = {
  Capacity: "capacity",
} as const;
export type ExperimentSkipReason = (typeof ExperimentSkipReason)[keyof typeof ExperimentSkipReason];

/** Terminal reason a peer-eligible shadow produced no comparison. */
export const ExperimentDropReason = {
  ArmFailed: "arm-failed",
  FallbackConsumed: "fallback-consumed",
  SelectionFailed: "selection-failed",
  ServedRankingFailed: "served-ranking-failed",
  ComparisonFailed: "comparison-failed",
} as const;
export type ExperimentDropReason = (typeof ExperimentDropReason)[keyof typeof ExperimentDropReason];
