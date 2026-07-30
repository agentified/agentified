import type { ExperimentRankingAgreement } from "./experiment-evaluation.js";
import type { ExperimentInvocationAttribution } from "./experiment-invocation.js";
import type {
  ExperimentArmOutcome,
  ExperimentArmRole,
  ExperimentRankedItem,
} from "./experiment-types.js";

/** @internal Valid delayed outcome ready for telemetry emission. */
export interface ExperimentOutcomeEvaluation {
  experimentId: string;
  selectionId: string;
  label?: string;
  score?: number;
}

/** @internal Controlled identity and dispatch facts for one experiment arm. */
export interface ExperimentArmEvaluation<Arm extends string = string> {
  experimentId: string;
  selectionId: string;
  unitHash: string;
  arm: Arm;
  role: ExperimentArmRole;
  cold: boolean;
  attributes?: Record<string, string | number | boolean>;
}

/** @internal Observable completion facts for one experiment arm. */
export interface ExperimentArmCompletionEvaluation {
  outcome: ExperimentArmOutcome;
  durationMs: number;
  hitCount?: number;
  ranking?: readonly ExperimentRankedItem[];
  failure?: { error: unknown };
  rankingFailure?: { error: unknown };
  resultAttributesFailure?: { error: unknown };
}

/** @internal Telemetry handle that keeps an arm's explicit context and span together. */
export interface ExperimentArmEvaluationHandle {
  run<T>(callback: () => T): T;
  complete(evaluation: ExperimentArmCompletionEvaluation): void;
  event(eventName: string, attributes: Record<string, unknown>): void;
}

/** @internal Why an admitted peer shadow produced no comparison. */
export type ExperimentPeerDropReason =
  | "arm-failed"
  | "fallback-consumed"
  | "selection-failed"
  | "served-ranking-failed"
  | "comparison-failed";

/** @internal One completed served-vs-shadow ranking comparison. */
export interface ExperimentComparisonEvaluation<Arm extends string = string> {
  selectionId: string;
  served: {
    arm: Arm;
    outcome: ExperimentArmOutcome;
    durationMs: number;
    hitCount: number;
  };
  shadow: {
    arm: Arm;
    outcome: ExperimentArmOutcome;
    durationMs: number;
    hitCount: number;
  };
  agreement: ExperimentRankingAgreement;
}

/** @internal One admitted shadow that could not produce a comparison. */
export interface ExperimentDropEvaluation<Arm extends string = string> {
  selectionId: string;
  shadowArm: Arm;
  reason: ExperimentPeerDropReason;
}

/** @internal One shadow dispatch skipped because instance capacity was full. */
export interface ExperimentSkipEvaluation<Arm extends string = string> {
  skippedArm: Arm;
  concurrency: number;
}

/** @internal One successful fallback decision. */
export interface ExperimentFallbackEvaluation<Arm extends string = string> {
  effectiveArm: Arm;
  reusedShadow: boolean;
}

/** @internal One invocation attribution ready for telemetry emission. */
export interface ExperimentInvocationEvaluation<Arm extends string = string> {
  experimentId: string;
  unitHash: string;
  toolId: string;
  turn?: number;
  attribution: ExperimentInvocationAttribution<Arm>;
}

/** @internal Evaluation records consumed by the SDK telemetry layer. */
export interface ExperimentEvaluationSink<Arm extends string = string> {
  arm?(evaluation: ExperimentArmEvaluation<Arm>): ExperimentArmEvaluationHandle;
  comparison?(
    evaluation: ExperimentComparisonEvaluation<Arm>,
    arm?: ExperimentArmEvaluationHandle,
  ): void;
  drop?(evaluation: ExperimentDropEvaluation<Arm>, arm?: ExperimentArmEvaluationHandle): void;
  skip?(evaluation: ExperimentSkipEvaluation<Arm>, arm?: ExperimentArmEvaluationHandle): void;
  fallback?(
    evaluation: ExperimentFallbackEvaluation<Arm>,
    arm?: ExperimentArmEvaluationHandle,
  ): void;
  invocation?(evaluation: ExperimentInvocationEvaluation<Arm>): void;
  outcome?(evaluation: ExperimentOutcomeEvaluation): void;
}
