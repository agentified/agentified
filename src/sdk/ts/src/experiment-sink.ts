import type { ExperimentRankingAgreement } from "./experiment-evaluation.js";
import type { ExperimentInvocationAttribution } from "./experiment-invocation.js";
import type {
  ExperimentArmOutcome,
  ExperimentArmRole,
  ExperimentRankedItem,
} from "./experiment-types.js";
import { newRuntimeEventId, type RuntimeEvents } from "./runtime-events.js";

interface ExperimentEventIdentity {
  eventId?: string;
}

/** @internal Valid delayed outcome ready for telemetry emission. */
export interface ExperimentOutcomeEvaluation extends ExperimentEventIdentity {
  experimentId: string;
  selectionId: string;
  label?: string;
  score?: number;
}

/** @internal Controlled identity and dispatch facts for one experiment arm. */
export interface ExperimentArmEvaluation<Arm extends string = string>
  extends ExperimentEventIdentity {
  experimentId: string;
  selectionId: string;
  unitHash: string;
  arm: Arm;
  role: ExperimentArmRole;
  cold: boolean;
  attributes?: Record<string, string | number | boolean>;
}

/** @internal Observable completion facts for one experiment arm. */
export interface ExperimentArmCompletionEvaluation extends ExperimentEventIdentity {
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
  event(eventName: string, attributes: Record<string, unknown>, eventId?: string): void;
}

/** @internal Why an admitted peer shadow produced no comparison. */
export type ExperimentPeerDropReason =
  | "arm-failed"
  | "fallback-consumed"
  | "selection-failed"
  | "served-ranking-failed"
  | "comparison-failed";

/** @internal One completed served-vs-shadow ranking comparison. */
export interface ExperimentComparisonEvaluation<Arm extends string = string>
  extends ExperimentEventIdentity {
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
export interface ExperimentDropEvaluation<Arm extends string = string>
  extends ExperimentEventIdentity {
  selectionId: string;
  shadowArm: Arm;
  reason: ExperimentPeerDropReason;
}

/** @internal One shadow dispatch skipped because instance capacity was full. */
export interface ExperimentSkipEvaluation<Arm extends string = string>
  extends ExperimentEventIdentity {
  skippedArm: Arm;
  concurrency: number;
}

/** @internal One successful fallback decision. */
export interface ExperimentFallbackEvaluation<Arm extends string = string>
  extends ExperimentEventIdentity {
  effectiveArm: Arm;
  reusedShadow: boolean;
}

/** @internal One invocation attribution ready for telemetry emission. */
export interface ExperimentInvocationEvaluation<Arm extends string = string>
  extends ExperimentEventIdentity {
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

/** @internal Compose OTel and runtime-event projections while minting each identity once. */
export function composeExperimentEvaluationSinks<Arm extends string>(
  first: ExperimentEvaluationSink<Arm>,
  second: ExperimentEvaluationSink<Arm>,
): ExperimentEvaluationSink<Arm> {
  const handles = new WeakMap<
    ExperimentArmEvaluationHandle,
    readonly [ExperimentArmEvaluationHandle | undefined, ExperimentArmEvaluationHandle | undefined]
  >();
  const pair = (handle?: ExperimentArmEvaluationHandle) =>
    handle === undefined
      ? ([undefined, undefined] as const)
      : (handles.get(handle) ?? [handle, handle]);
  const stamp = <T extends ExperimentEventIdentity>(evaluation: T): T & { eventId: string } => ({
    ...evaluation,
    eventId: evaluation.eventId ?? newRuntimeEventId(),
  });

  return {
    arm(evaluation) {
      const stamped = stamp(evaluation);
      const firstHandle = first.arm?.(stamped);
      const secondHandle = second.arm?.(stamped);
      const handle: ExperimentArmEvaluationHandle = {
        run(callback) {
          const runSecond = () => secondHandle?.run(callback) ?? callback();
          return firstHandle?.run(runSecond) ?? runSecond();
        },
        complete(completion) {
          const stampedCompletion = stamp(completion);
          firstHandle?.complete(stampedCompletion);
          secondHandle?.complete(stampedCompletion);
        },
        event(eventName, attributes, eventId = newRuntimeEventId()) {
          firstHandle?.event(eventName, attributes, eventId);
          secondHandle?.event(eventName, attributes, eventId);
        },
      };
      handles.set(handle, [firstHandle, secondHandle]);
      return handle;
    },
    comparison(evaluation, handle) {
      const stamped = stamp(evaluation);
      const [firstHandle, secondHandle] = pair(handle);
      first.comparison?.(stamped, firstHandle);
      second.comparison?.(stamped, secondHandle);
    },
    drop(evaluation, handle) {
      const stamped = stamp(evaluation);
      const [firstHandle, secondHandle] = pair(handle);
      first.drop?.(stamped, firstHandle);
      second.drop?.(stamped, secondHandle);
    },
    skip(evaluation, handle) {
      const stamped = stamp(evaluation);
      const [firstHandle, secondHandle] = pair(handle);
      first.skip?.(stamped, firstHandle);
      second.skip?.(stamped, secondHandle);
    },
    fallback(evaluation, handle) {
      const stamped = stamp(evaluation);
      const [firstHandle, secondHandle] = pair(handle);
      first.fallback?.(stamped, firstHandle);
      second.fallback?.(stamped, secondHandle);
    },
    invocation(evaluation) {
      const stamped = stamp(evaluation);
      first.invocation?.(stamped);
      second.invocation?.(stamped);
    },
    outcome(evaluation) {
      const stamped = stamp(evaluation);
      first.outcome?.(stamped);
      second.outcome?.(stamped);
    },
  };
}

/** @internal Project experiment evaluation records into the public runtime vocabulary. */
export function createExperimentRuntimeEventSink<Arm extends string>(
  events: RuntimeEvents,
): ExperimentEvaluationSink<Arm> {
  const arms = new WeakMap<ExperimentArmEvaluationHandle, ExperimentArmEvaluation<Arm>>();
  const emit = (type: string, eventId: string | undefined, fields: Record<string, unknown>) =>
    events.emit({ type, ...(eventId === undefined ? {} : { event_id: eventId }), ...fields });
  const armFields = (evaluation: ExperimentArmEvaluation<Arm>) => ({
    experiment_id: evaluation.experimentId,
    selection_id: evaluation.selectionId,
    unit: evaluation.unitHash,
    arm: evaluation.arm,
    role: evaluation.role,
  });
  const fromHandle = (handle?: ExperimentArmEvaluationHandle) =>
    handle === undefined ? undefined : arms.get(handle);

  return {
    arm(evaluation) {
      emit("experiment_selection", evaluation.eventId, {
        ...armFields(evaluation),
        cold: evaluation.cold,
        ...(evaluation.attributes === undefined ? {} : { attributes: evaluation.attributes }),
      });
      const handle: ExperimentArmEvaluationHandle = {
        run: (callback) => callback(),
        complete(completion) {
          emit("experiment_results", completion.eventId, {
            ...armFields(evaluation),
            outcome: completion.outcome,
            duration_ms: completion.durationMs,
            ...(completion.hitCount === undefined ? {} : { hit_count: completion.hitCount }),
            ...(completion.ranking === undefined
              ? {}
              : {
                  result_ids: completion.ranking.map((item) => item.id),
                  ...(completion.ranking.every((item) => item.score !== undefined)
                    ? { result_scores: completion.ranking.map((item) => item.score) }
                    : {}),
                }),
            ...(completion.failure === undefined
              ? {}
              : { error_class: errorClass(completion.failure.error) }),
          });
        },
        event() {},
      };
      arms.set(handle, evaluation);
      return handle;
    },
    comparison(evaluation, handle) {
      const arm = fromHandle(handle);
      emit("experiment_comparison", evaluation.eventId, {
        ...(arm === undefined ? {} : armFields(arm)),
        selection_id: evaluation.selectionId,
        served_arm: evaluation.served.arm,
        served_outcome: evaluation.served.outcome,
        served_duration_ms: evaluation.served.durationMs,
        served_hit_count: evaluation.served.hitCount,
        shadow_arm: evaluation.shadow.arm,
        shadow_outcome: evaluation.shadow.outcome,
        shadow_duration_ms: evaluation.shadow.durationMs,
        shadow_hit_count: evaluation.shadow.hitCount,
        agreement: evaluation.agreement,
      });
    },
    skip(evaluation, handle) {
      const arm = fromHandle(handle);
      emit("experiment_skip", evaluation.eventId, {
        ...(arm === undefined ? {} : armFields(arm)),
        skipped_arm: evaluation.skippedArm,
        concurrency: evaluation.concurrency,
        reason: "capacity",
      });
    },
    fallback(evaluation, handle) {
      const arm = fromHandle(handle);
      emit("experiment_fallback", evaluation.eventId, {
        ...(arm === undefined ? {} : armFields(arm)),
        effective_arm: evaluation.effectiveArm,
        reused_shadow: evaluation.reusedShadow,
      });
    },
    drop(evaluation, handle) {
      const arm = fromHandle(handle);
      emit("experiment_drop", evaluation.eventId, {
        ...(arm === undefined ? {} : armFields(arm)),
        selection_id: evaluation.selectionId,
        shadow_arm: evaluation.shadowArm,
        reason: evaluation.reason,
      });
    },
    invocation(evaluation) {
      emit("experiment_invocation", evaluation.eventId, {
        experiment_id: evaluation.experimentId,
        unit: evaluation.unitHash,
        tool_id: evaluation.toolId,
        ...(evaluation.turn === undefined ? {} : { turn: evaluation.turn }),
        attributed: evaluation.attribution.attributed,
        ...(evaluation.attribution.attributed
          ? {
              selection_id: evaluation.attribution.selectionId,
              effective_arm: evaluation.attribution.effectiveArm,
              rank: evaluation.attribution.rank,
              age_ms: evaluation.attribution.ageMs,
            }
          : {}),
      });
    },
    outcome(evaluation) {
      emit("experiment_outcome", evaluation.eventId, {
        experiment_id: evaluation.experimentId,
        selection_id: evaluation.selectionId,
        ...(evaluation.label === undefined ? {} : { label: evaluation.label }),
        ...(evaluation.score === undefined ? {} : { score: evaluation.score }),
      });
    },
  };
}

function errorClass(error: unknown): string {
  return error instanceof Error ? error.name : typeof error;
}
