/**
 * ⚠️ **Experimental.** The facts / grounding API — constant grounding content
 * an agent always needs (a shop's address, a brand's voice) plus the
 * transcript-derived re-injection freshness gate (ADR-0014).
 *
 * This surface is behind the `experimental` namespace on purpose: it is new and
 * may change or be removed without a major-version bump. Reach it as
 * `experimental.FactCatalog` (`import { experimental } from "@ratel-ai/sdk"`) so
 * that dependence on an unstable API is explicit at the import site. Constructing
 * a {@link FactCatalog} also logs a one-time warning (silence it with
 * `RATEL_EXPERIMENTAL_SILENCE=1`).
 *
 * @module
 */

export type { Fact, FactHit } from "../native/index.cjs";
export type { CapabilityFactHit } from "./capabilities.js";
export type { FactCatalogOptions } from "./fact-catalog.js";
export { FactCatalog } from "./fact-catalog.js";
export type {
  FactCandidate,
  GroundingItem,
  GroundingResult,
  GroundingSnapshotItem,
  GroundOptions,
  InjectionDecision,
  InjectionDecisionReason,
  InjectionReason,
  PlanInjectionInput,
} from "./grounding.js";
export { FACT_ID_PATTERN, Pin, planInjection } from "./grounding.js";
export { FactRegistry } from "./registry.js";
