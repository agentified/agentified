import type { ExperimentEvaluationAttributes } from "./experiment-evaluation.js";
import type {
  ExperimentArmCompletionEvaluation,
  ExperimentArmEvaluation,
  ExperimentArmEvaluationHandle,
  ExperimentEvaluationSink,
} from "./experiment-sink.js";
import type {
  ExperimentArmOutcome,
  ExperimentArmRole,
  ExperimentConfig,
  ExperimentRankedItem,
  ExperimentSelectOptions,
} from "./experiment-types.js";

type ArmCallbackResult<Result> =
  | { ok: true; result: Result; durationMs: number }
  | { ok: false; error: unknown; durationMs: number };

/** @internal Terminal outcome of one arm callback, transform, and ranking projection. */
export type ArmCompletion<Result> =
  | {
      ok: true;
      result: Result;
      durationMs: number;
      outcome: ExperimentArmOutcome;
      ranking?: ExperimentRankedItem[];
      rankingFailure?: { error: unknown };
      resultAttributes?: ResultAttributesProjection;
    }
  | {
      ok: false;
      error: unknown;
      durationMs: number;
      outcome: "timeout" | "error";
      source: "select" | "transform";
    };

/** @internal Result-level attribute projection captured for served-vs-shadow agreement. */
export type ResultAttributesProjection =
  | { ok: true; attributes: ExperimentEvaluationAttributes }
  | { ok: false; error: unknown };

/** @internal A completed arm that produced a result. */
export type SuccessfulArmCompletion<Result> = Extract<ArmCompletion<Result>, { ok: true }>;

/** @internal A dispatched arm and its pending completion. */
export interface ArmRun<Result, Arm extends string> {
  arm: Arm;
  cold: boolean;
  role: ExperimentArmRole;
  telemetry?: ExperimentArmEvaluationHandle;
  completion: Promise<ArmCompletion<Result>>;
}

/** @internal Starts one arm in a given role, optionally signalling when its callback settles. */
export type ArmDispatcher<Result, Arm extends string> = (
  arm: Arm,
  role: ExperimentArmRole,
  onCallbackSettled?: () => void,
) => ArmRun<Result, Arm>;

/** @internal Dispatch one arm: run its selector under telemetry, transform, and project ranking. */
export function startArm<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  arm: Arm,
  params: Params,
  role: ExperimentArmRole,
  options: ExperimentSelectOptions<Result, Arm>,
  telemetryAttributes: Record<string, string | number | boolean> | undefined,
  cold: boolean,
  selectionId: string,
  unitHash: string,
  sink: ExperimentEvaluationSink<Arm>,
  onCallbackSettled?: () => void,
): ArmRun<Result, Arm> {
  const telemetry = startArmTelemetry(sink, {
    experimentId: config.id,
    selectionId,
    unitHash,
    arm,
    role,
    cold,
    attributes: telemetryAttributes,
  });
  const complete = () =>
    startArmCallback(config, arm, params, role, onCallbackSettled)
      .then((settled): ArmCompletion<Result> => {
        if (!settled.ok) {
          return {
            ...settled,
            outcome: isTimeoutError(settled.error) ? "timeout" : "error",
            source: "select",
          };
        }

        let result: Result;
        try {
          result =
            options.transform === undefined ? settled.result : options.transform(settled.result);
        } catch (error) {
          return {
            ok: false,
            error,
            durationMs: settled.durationMs,
            outcome: "error",
            source: "transform",
          };
        }

        try {
          const ranking = snapshotRanking(config.ranking(result));
          const resultAttributes = projectResultAttributes(config, result);
          return {
            ok: true,
            result,
            durationMs: settled.durationMs,
            outcome: ranking.length === 0 ? "empty" : "ok",
            ranking,
            resultAttributes,
          };
        } catch (error) {
          return {
            ok: true,
            result,
            durationMs: settled.durationMs,
            outcome: "error",
            rankingFailure: { error },
          };
        }
      })
      .then((completed) => {
        completeArmTelemetry(telemetry, {
          outcome: completed.outcome,
          durationMs: completed.durationMs,
          ...(!completed.ok
            ? { failure: { error: completed.error } }
            : completed.rankingFailure === undefined
              ? {}
              : {
                  failure: completed.rankingFailure,
                  rankingFailure: completed.rankingFailure,
                }),
          ...(!completed.ok || completed.ranking === undefined
            ? {}
            : { hitCount: completed.ranking.length, ranking: completed.ranking }),
          ...(completed.ok && completed.resultAttributes?.ok === false
            ? { resultAttributesFailure: { error: completed.resultAttributes.error } }
            : {}),
        });
        return completed;
      });
  const completion = runArmTelemetry(telemetry, complete);
  return { arm, cold, role, telemetry, completion };
}

function startArmTelemetry<Arm extends string>(
  sink: ExperimentEvaluationSink<Arm>,
  evaluation: ExperimentArmEvaluation<Arm>,
): ExperimentArmEvaluationHandle | undefined {
  try {
    return sink.arm?.(evaluation);
  } catch {
    return undefined;
  }
}

function startArmCallback<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  arm: Arm,
  params: Params,
  role: ExperimentArmRole,
  onSettled?: () => void,
): Promise<ArmCallbackResult<Result>> {
  const startedAt = performance.now();
  try {
    return Promise.resolve(config.arms[arm].select(params, { role })).then(
      (result) => {
        onSettled?.();
        return {
          ok: true,
          result,
          durationMs: performance.now() - startedAt,
        };
      },
      (error: unknown) => {
        onSettled?.();
        return {
          ok: false,
          error,
          durationMs: performance.now() - startedAt,
        };
      },
    );
  } catch (error) {
    onSettled?.();
    return Promise.resolve({
      ok: false,
      error,
      durationMs: performance.now() - startedAt,
    });
  }
}

function isTimeoutError(error: unknown): boolean {
  if ((typeof error !== "object" || error === null) && typeof error !== "function") {
    return false;
  }
  try {
    return (error as { name?: unknown }).name === "TimeoutError";
  } catch {
    return false;
  }
}

function snapshotRanking(ranking: readonly ExperimentRankedItem[]): ExperimentRankedItem[] {
  return ranking.map((item) => ({
    id: item.id,
    ...(item.score === undefined ? {} : { score: item.score }),
    ...(item.attrs === undefined ? {} : { attrs: { ...item.attrs } }),
  }));
}

function projectResultAttributes<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  result: Result,
): ResultAttributesProjection | undefined {
  const project = config.evaluation.attributes;
  if (project === undefined || !config.evaluation.references.includes("peer-selection")) {
    return undefined;
  }
  try {
    return { ok: true, attributes: { ...project(result) } };
  } catch (error) {
    return { ok: false, error };
  }
}

function completeArmTelemetry(
  telemetry: ExperimentArmEvaluationHandle | undefined,
  evaluation: ExperimentArmCompletionEvaluation,
): void {
  try {
    telemetry?.complete(evaluation);
  } catch {
    // Telemetry delivery is observational and cannot fail application work.
  }
}

function runArmTelemetry<T>(
  telemetry: ExperimentArmEvaluationHandle | undefined,
  callback: () => T,
): T {
  if (telemetry === undefined) {
    return callback();
  }

  let state: "pending" | "running" | "returned" | "threw" = "pending";
  let result!: T;
  let callbackError: unknown;
  const runOnce = () => {
    if (state === "returned") {
      return result;
    }
    if (state === "threw") {
      throw callbackError;
    }
    state = "running";
    try {
      result = callback();
      state = "returned";
      return result;
    } catch (error) {
      callbackError = error;
      state = "threw";
      throw error;
    }
  };

  try {
    telemetry.run(runOnce);
    if (state === "pending") {
      return runOnce();
    }
  } catch {
    if (state === "pending") {
      return runOnce();
    }
  }
  if (state === "returned") {
    return result;
  }
  throw callbackError;
}
