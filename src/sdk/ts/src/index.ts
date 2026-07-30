/**
 * `@ratel-ai/sdk` — TypeScript SDK for Ratel, the context engineering platform
 * for AI agents. In-process, no infra: a native (Rust) BM25/semantic/hybrid
 * index behind {@link ToolCatalog} and {@link SkillCatalog}, MCP ingestion via
 * {@link registerMcpServer}, and the framework-neutral capability tools
 * ({@link searchCapabilitiesTool}, {@link invokeToolTool},
 * {@link getSkillContentTool}) that let a model discover and run what the
 * catalogs hold. Everything emits OTel `ratel.*`/`gen_ai.*` spans plus a local
 * trace stream (ADR-0007).
 *
 * @packageDocumentation
 */

// The catalog's JSON-Schema spelling, re-exported so framework adapters type
// their CatalogRegistration schemas without a cast.
export type { JSONSchema7 } from "json-schema";
export type {
  AdaptiveRankingStatus,
  ReplaceOutcome,
  SearchHit,
  Skill,
  SkillHit,
  Tool,
} from "../native/index.cjs";
export type {
  CapabilitiesSearchOptions,
  CapabilitySkillHit,
  CapabilityToolGroup,
  CapabilityToolHit,
  InvokeToolError,
  InvokeToolToolOptions,
  SearchCapabilitiesOptions,
  SearchCapabilitiesResult,
  UpstreamServerInfo,
} from "./capabilities.js";
export {
  formatUpstreamLine,
  INVOKE_TOOL_ERROR_CAUSE,
  INVOKE_TOOL_ID,
  invokeToolTool,
  isInvokeToolError,
  runCapabilitiesSearch,
  SEARCH_CAPABILITIES_ID,
  searchCapabilitiesTool,
} from "./capabilities.js";
export type {
  EmbeddingModelConfig,
  EmbeddingSpec,
  ExecutableTool,
  Executor,
  InputValidationResult,
  InputValidator,
  SearchMethod,
  SearchOrigin,
  ToolCatalogOptions,
  TraceSinkConfig,
} from "./catalog.js";
export { ToolCatalog } from "./catalog.js";
// Deprecated pre-0.2.0 surface (see compat.ts) — kept so `@ratel-ai/sdk@0.1.x`
// callers keep compiling and running after upgrading to 0.2.0. Slated for removal (RAT-250).
export type {
  SearchToolHit,
  SearchToolsGroup,
  SearchToolsResult,
  SearchToolsToolOptions,
} from "./compat.js";
export { SEARCH_TOOLS_ID, searchToolsTool } from "./compat.js";
export { DimensionMismatchError, EmbedderError } from "./errors.js";
export type {
  Experiment,
  ExperimentArmOutcome,
  ExperimentArmRole,
  ExperimentConfig,
  ExperimentEvaluationReference,
  ExperimentRankedItem,
  ExperimentReportedOutcome,
  ExperimentSelection,
  ExperimentSelectOptions,
  ExperimentSplit,
} from "./experiment.js";
export { experimentalDefineExperiment } from "./experiment.js";
export type { McpServerHandle, McpToolsListErrorCode, RegisterMcpServerOptions } from "./mcp.js";
export { McpToolsListError, registerMcpServer } from "./mcp.js";
// The framework-adapter SPI and factory (ADR-0013): `ratel(config).adaptTo(adapter)`.
export type {
  AdaptedBase,
  AdaptedRatel,
  AdaptedToolCollection,
  CatalogRegistration,
  ExperimentalPassthroughToolExposure,
  Ratel,
  RatelAdapter,
  RatelConfig,
  RecallRef,
  ToolCollection,
} from "./ratel.js";
export { ratel } from "./ratel.js";
/** Adaptive usage ranking: the shared read model of what users invoke (ADR-0014). */
export { IntentGraph, SkillRegistry, ToolRegistry } from "./registry.js";
export type { PendingReplace, SkillCatalogOptions } from "./skill-catalog.js";
export { SkillCatalog } from "./skill-catalog.js";
export { GET_SKILL_CONTENT_ID, getSkillContentTool } from "./skill-tools.js";
// OpenTelemetry emission of the ratel.*/gen_ai.* funnel. The SDK emits to whatever OTel
// provider the host has registered and never registers one itself — delivery is the host's
// `new NodeSDK({ spanProcessors })`. `ContentCapture`/`setContentCapture`/
// `clearContentCapture` (re-exported from @ratel-ai/telemetry) control the message/tool
// content-capture gate programmatically.
export { ContentCapture, clearContentCapture, setContentCapture } from "./telemetry.js";
