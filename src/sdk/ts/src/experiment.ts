import { randomUUID } from "node:crypto";
import { type ArmDispatcher, type ArmRun, startArm } from "./experiment-arm.js";
import { resolveEvaluationK } from "./experiment-evaluation.js";
import {
  createExperimentInvocationBuffer,
  type ExperimentInvocationBuffer,
  hashUnitId,
} from "./experiment-invocation.js";
import {
  createDeferred,
  type Deferred,
  ensureWarmup,
  notifyEvaluation,
  type WarmupEntry,
} from "./experiment-runtime.js";
import {
  completePeerEvaluation,
  type ServedEvaluationSignal,
  trackShadow,
} from "./experiment-shadow.js";
import type { ExperimentEvaluationSink } from "./experiment-sink.js";
import type { Experiment, ExperimentConfig, ExperimentSelection } from "./experiment-types.js";
import {
  assignArm,
  validateReportedOutcome,
  validateSplitCoverage,
} from "./experiment-validation.js";
import { createExperimentTelemetrySink, validateExperimentAttributes } from "./telemetry.js";

/**
 * Define an opt-in retrieval experiment over host-supplied asynchronous selectors.
 *
 * Configuration errors throw synchronously. Each selection serves one arm, may run
 * other arms as bounded detached shadows, and exposes {@link Experiment.drain} for
 * orderly shutdown.
 */
export function experimentalDefineExperiment<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
): Experiment<Params, Result, Arm> {
  return defineExperimentInternal(config, createExperimentTelemetrySink());
}

/** @internal Builds an experiment with an evaluation sink for package-level composition. */
export function defineExperimentInternal<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  sink: ExperimentEvaluationSink<Arm> = {},
): Experiment<Params, Result, Arm> {
  validateSplitCoverage(config);
  const armNames = Object.keys(config.arms) as Arm[];
  const shadowConcurrency = config.shadowPolicy?.concurrency ?? 1;
  const pendingShadows = new Set<Promise<void>>();
  const warmups = new Map<Arm, WarmupEntry>();
  const comparesPeers = config.evaluation.references.includes("peer-selection");
  const invocationReference = config.evaluation.references.find(
    (reference) => reference !== "peer-selection",
  );
  const invocationBuffer =
    invocationReference === undefined
      ? undefined
      : createExperimentInvocationBuffer<Arm>(invocationReference);
  let shadowsInFlight = 0;

  return {
    select(params, options) {
      const assignedArm = assignArm(config, options);
      if (options.shadow === true && armNames.length < 2) {
        throw new Error(
          "experimentalDefineExperiment.select: shadow requires a second declared arm",
        );
      }
      const telemetryAttributes = validateExperimentAttributes(options.attributes);
      const selectionId = randomUUID();
      const unitHash = hashUnitId(options.unitId);
      const k = resolveEvaluationK(options.k, config.evaluation.k);
      const servedEvaluation = createDeferred<ServedEvaluationSignal<Result, Arm>>();

      const coldAtStart = new Map(
        armNames.map((arm) => [
          arm,
          config.arms[arm].warmup !== undefined && warmups.get(arm)?.resolved !== true,
        ]),
      );
      const dispatch: ArmDispatcher<Result, Arm> = (arm, role, onCallbackSettled) => {
        void ensureWarmup(config, warmups, arm);
        return startArm(
          config,
          arm,
          params,
          role,
          options,
          telemetryAttributes,
          coldAtStart.get(arm) ?? false,
          selectionId,
          unitHash,
          sink,
          onCallbackSettled,
        );
      };
      const assigned = dispatch(assignedArm, "serving");
      const shadows = new Map<Arm, ArmRun<Result, Arm>>();
      if (options.shadow === true) {
        for (const arm of armNames) {
          if (arm === assignedArm) {
            continue;
          }
          if (shadowsInFlight >= shadowConcurrency) {
            notifyEvaluation(
              sink.skip,
              { skippedArm: arm, concurrency: shadowConcurrency },
              assigned.telemetry,
            );
            continue;
          }
          shadowsInFlight += 1;
          const shadow = dispatch(arm, "shadow", () => {
            shadowsInFlight -= 1;
          });
          shadows.set(arm, shadow);
          trackShadow(
            pendingShadows,
            comparesPeers
              ? completePeerEvaluation(shadow, servedEvaluation.promise, sink)
              : shadow.completion,
          );
        }
      }

      return completeSelection(config, {
        assigned,
        assignedArm,
        dispatch,
        k,
        selectionId,
        servedEvaluation,
        shadows,
        invocationBuffer,
        sink,
        unitId: options.unitId,
      });
    },
    reportInvocation(args) {
      if (invocationBuffer === undefined) {
        return;
      }
      try {
        const unitHash = hashUnitId(args.unitId);
        for (const attribution of invocationBuffer.evaluateInvocation(args)) {
          notifyEvaluation(sink.invocation, {
            experimentId: config.id,
            unitHash,
            toolId: args.toolId,
            ...(args.turn === undefined ? {} : { turn: args.turn }),
            attribution,
          });
        }
      } catch {
        // Invocation attribution is observational and cannot fail application work.
      }
    },
    reportOutcome(args) {
      if (config.evaluation.outcome !== true) {
        throw new Error(
          "experimentalDefineExperiment.reportOutcome: outcome evaluation is not enabled",
        );
      }
      validateReportedOutcome(args);
      notifyEvaluation(sink.outcome, {
        experimentId: config.id,
        selectionId: args.selectionId,
        ...(args.label === undefined ? {} : { label: args.label }),
        ...(args.score === undefined ? {} : { score: args.score }),
      });
    },
    warm: () =>
      Promise.all(armNames.map((arm) => ensureWarmup(config, warmups, arm))).then(() => undefined),
    drain: () => Promise.allSettled([...pendingShadows]).then(() => undefined),
  };
}

async function completeSelection<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  args: {
    assignedArm: Arm;
    assigned: ArmRun<Result, Arm>;
    shadows: ReadonlyMap<Arm, ArmRun<Result, Arm>>;
    dispatch: ArmDispatcher<Result, Arm>;
    selectionId: string;
    k: number;
    servedEvaluation: Deferred<ServedEvaluationSignal<Result, Arm>>;
    invocationBuffer?: ExperimentInvocationBuffer<Arm>;
    sink: ExperimentEvaluationSink<Arm>;
    unitId: string;
  },
): Promise<ExperimentSelection<Result, Arm>> {
  let signal: ServedEvaluationSignal<Result, Arm> = {
    status: "selection-failed",
    selectionId: args.selectionId,
  };

  try {
    let effectiveArm = args.assignedArm;
    let consumedShadow: Arm | undefined;
    let completed = await args.assigned.completion;

    if (!completed.ok) {
      if (
        completed.source !== "select" ||
        config.fallbackArm === undefined ||
        config.fallbackArm === args.assignedArm
      ) {
        throw completed.error;
      }
      effectiveArm = config.fallbackArm;
      const reusedShadow = args.shadows.get(effectiveArm);
      const fallback = reusedShadow ?? args.dispatch(effectiveArm, "serving");
      completed = await fallback.completion;
      if (!completed.ok) {
        throw completed.error;
      }
      if (reusedShadow !== undefined) {
        consumedShadow = effectiveArm;
      }
      notifyEvaluation(
        args.sink.fallback,
        { effectiveArm, reusedShadow: reusedShadow !== undefined },
        args.assigned.telemetry,
      );
    }

    signal = {
      status: "served",
      selectionId: args.selectionId,
      effectiveArm,
      completion: completed,
      consumedShadow,
      k: args.k,
    };
    if (args.invocationBuffer !== undefined && completed.ranking !== undefined) {
      try {
        args.invocationBuffer.recordSelection({
          unitId: args.unitId,
          selectionId: args.selectionId,
          effectiveArm,
          ids: completed.ranking.map((item) => item.id),
        });
      } catch {
        // Invocation attribution is observational and cannot fail a selection.
      }
    }
    return {
      result: completed.result,
      selectionId: args.selectionId,
      assignedArm: args.assignedArm,
      effectiveArm,
      durationMs: completed.durationMs,
    };
  } finally {
    args.servedEvaluation.resolve(signal);
  }
}
