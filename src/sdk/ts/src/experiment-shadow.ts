import type { ArmRun, SuccessfulArmCompletion } from "./experiment-arm.js";
import { compareRankings } from "./experiment-evaluation.js";
import { notifyEvaluation } from "./experiment-runtime.js";
import type {
  ExperimentArmEvaluationHandle,
  ExperimentEvaluationSink,
  ExperimentPeerDropReason,
} from "./experiment-sink.js";

/** @internal Serving-selection signal awaited by admitted peer shadows. */
export type ServedEvaluationSignal<Result, Arm extends string> =
  | {
      status: "served";
      selectionId: string;
      effectiveArm: Arm;
      completion: SuccessfulArmCompletion<Result>;
      consumedShadow?: Arm;
      k: number;
    }
  | {
      status: "selection-failed";
      selectionId: string;
    };

/** @internal Compare one admitted shadow against the served selection, or record why it dropped. */
export async function completePeerEvaluation<Result, Arm extends string>(
  shadow: ArmRun<Result, Arm>,
  servedPromise: Promise<ServedEvaluationSignal<Result, Arm>>,
  sink: ExperimentEvaluationSink<Arm>,
): Promise<void> {
  const [shadowCompletion, served] = await Promise.all([shadow.completion, servedPromise]);
  if (!shadowCompletion.ok || shadowCompletion.ranking === undefined) {
    notifyDrop(sink, served.selectionId, shadow.arm, "arm-failed", shadow.telemetry);
    return;
  }
  if (served.status === "selection-failed") {
    notifyDrop(sink, served.selectionId, shadow.arm, "selection-failed", shadow.telemetry);
    return;
  }
  if (served.consumedShadow === shadow.arm) {
    notifyDrop(sink, served.selectionId, shadow.arm, "fallback-consumed", shadow.telemetry);
    return;
  }
  if (served.completion.ranking === undefined) {
    notifyDrop(sink, served.selectionId, shadow.arm, "served-ranking-failed", shadow.telemetry);
    return;
  }

  try {
    const servedResultAttributes = served.completion.resultAttributes;
    const shadowResultAttributes = shadowCompletion.resultAttributes;
    const comparableResultAttributes =
      servedResultAttributes?.ok === true && shadowResultAttributes?.ok === true
        ? {
            servedResultAttrs: servedResultAttributes.attributes,
            shadowResultAttrs: shadowResultAttributes.attributes,
          }
        : {};
    const agreement = compareRankings(served.completion.ranking, shadowCompletion.ranking, {
      k: served.k,
      ...comparableResultAttributes,
    });
    notifyEvaluation(
      sink.comparison,
      {
        selectionId: served.selectionId,
        served: {
          arm: served.effectiveArm,
          outcome: served.completion.outcome,
          durationMs: served.completion.durationMs,
          hitCount: served.completion.ranking.length,
        },
        shadow: {
          arm: shadow.arm,
          outcome: shadowCompletion.outcome,
          durationMs: shadowCompletion.durationMs,
          hitCount: shadowCompletion.ranking.length,
        },
        agreement,
      },
      shadow.telemetry,
    );
  } catch {
    notifyDrop(sink, served.selectionId, shadow.arm, "comparison-failed", shadow.telemetry);
  }
}

function notifyDrop<Arm extends string>(
  sink: ExperimentEvaluationSink<Arm>,
  selectionId: string,
  shadowArm: Arm,
  reason: ExperimentPeerDropReason,
  telemetry?: ExperimentArmEvaluationHandle,
): void {
  notifyEvaluation(sink.drop, { selectionId, shadowArm, reason }, telemetry);
}

/** @internal Track a detached shadow pipeline so {@link Experiment.drain} can await it. */
export function trackShadow(pending: Set<Promise<void>>, completion: Promise<unknown>): void {
  const contained = completion.then(
    () => undefined,
    () => undefined,
  );
  pending.add(contained);
  void contained.then(() => {
    pending.delete(contained);
  });
}
