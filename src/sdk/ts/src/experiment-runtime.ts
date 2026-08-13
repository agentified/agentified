import type { ExperimentArmEvaluationHandle } from "./experiment-sink.js";
import type { ExperimentConfig } from "./experiment-types.js";

/** @internal Warmup lifecycle entry tracked per arm. */
export interface WarmupEntry {
  resolved: boolean;
  promise: Promise<void>;
}

/** @internal Externally-resolvable promise used to join serving and shadow work. */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

/**
 * @internal Deliver one evaluation record to its sink hook, containing failures.
 *
 * Evaluation delivery is observational and must never fail application work.
 */
export function notifyEvaluation<Evaluation>(
  emit: ((evaluation: Evaluation, arm?: ExperimentArmEvaluationHandle) => void) | undefined,
  evaluation: Evaluation,
  telemetry?: ExperimentArmEvaluationHandle,
): void {
  try {
    if (telemetry === undefined) {
      emit?.(evaluation);
    } else {
      emit?.(evaluation, telemetry);
    }
  } catch {
    // Evaluation delivery is observational and cannot fail application work.
  }
}

/** @internal Create an externally-resolvable promise. */
export function createDeferred<T>(): Deferred<T> {
  let resolve = (_value: T): void => {};
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

/**
 * @internal Ensure an arm's warmup runs at most once; rejection keeps the arm cold and retryable.
 */
export function ensureWarmup<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  warmups: Map<Arm, WarmupEntry>,
  arm: Arm,
): Promise<void> {
  const warmup = config.arms[arm].warmup;
  if (warmup === undefined) {
    return Promise.resolve();
  }

  const existing = warmups.get(arm);
  if (existing !== undefined) {
    return existing.promise;
  }

  const entry: WarmupEntry = { resolved: false, promise: Promise.resolve() };
  warmups.set(arm, entry);
  try {
    entry.promise = Promise.resolve(warmup()).then(
      () => {
        entry.resolved = true;
      },
      () => {
        if (warmups.get(arm) === entry) {
          warmups.delete(arm);
        }
      },
    );
  } catch {
    warmups.delete(arm);
  }
  return entry.promise;
}
