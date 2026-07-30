import { createHash } from "node:crypto";
import type {
  ExperimentConfig,
  ExperimentEvaluationReference,
  ExperimentReportedOutcome,
  ExperimentSelectOptions,
} from "./experiment-types.js";

/** @internal Reject a misconfigured experiment definition synchronously at define time. */
export function validateSplitCoverage<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
): void {
  const arms = Object.keys(config.arms);
  if (arms.length === 0) {
    throw new Error("experimentalDefineExperiment: arms must not be empty");
  }
  if (config.fallbackArm !== undefined && !arms.includes(config.fallbackArm)) {
    throw new Error(
      `experimentalDefineExperiment: fallbackArm "${config.fallbackArm}" is not a declared arm`,
    );
  }
  const concurrency = config.shadowPolicy?.concurrency;
  if (concurrency !== undefined && (!Number.isInteger(concurrency) || concurrency < 1)) {
    throw new Error(
      "experimentalDefineExperiment: shadowPolicy.concurrency must be a positive integer",
    );
  }
  validateEvaluation(config.evaluation);
  if (config.split === undefined) {
    return;
  }

  const allocations = new Set(config.split.map(({ arm }) => arm));
  if (
    allocations.size !== config.split.length ||
    allocations.size !== arms.length ||
    arms.some((arm) => !allocations.has(arm as Arm))
  ) {
    throw new Error(
      "experimentalDefineExperiment: split must allocate every declared arm exactly once",
    );
  }

  let total = 0;
  for (const allocation of config.split) {
    if (!Number.isSafeInteger(allocation.weight) || allocation.weight < 0) {
      throw new Error(
        "experimentalDefineExperiment: split weight must be a non-negative safe integer",
      );
    }
    total += allocation.weight;
    if (!Number.isSafeInteger(total)) {
      throw new Error("experimentalDefineExperiment: split total must be a positive safe integer");
    }
  }
  if (total === 0) {
    throw new Error("experimentalDefineExperiment: split total must be a positive safe integer");
  }
}

function validateEvaluation<Result>(evaluation: {
  k?: number;
  references: readonly ExperimentEvaluationReference[];
  attributes?: (result: Result) => Record<string, string | number | boolean | null>;
  outcome?: boolean;
}): void {
  if (evaluation.references.length === 0) {
    throw new Error("experimentalDefineExperiment: evaluation.references must not be empty");
  }
  if (evaluation.k !== undefined && (!Number.isInteger(evaluation.k) || evaluation.k < 1)) {
    throw new Error("experimentalDefineExperiment: evaluation.k must be an integer >= 1");
  }

  let peerReferences = 0;
  let invocationReferences = 0;
  for (const reference of evaluation.references) {
    if (reference === "peer-selection") {
      peerReferences += 1;
      continue;
    }

    if (typeof reference !== "object" || reference === null || reference.kind !== "invocation") {
      throw new Error(
        'experimentalDefineExperiment: evaluation reference.kind must be "invocation"',
      );
    }
    if (
      reference.attribution !== undefined &&
      reference.attribution !== "last-selection" &&
      reference.attribution !== "all-in-window"
    ) {
      throw new Error(
        'experimentalDefineExperiment: invocation attribution must be "last-selection" or "all-in-window"',
      );
    }
    invocationReferences += 1;
    const { turns, maxAgeMs } = reference.window;
    if (turns === undefined && maxAgeMs === undefined) {
      throw new Error(
        "experimentalDefineExperiment: invocation window requires at least one bound",
      );
    }
    if (turns !== undefined && (!Number.isSafeInteger(turns) || turns < 1)) {
      throw new Error(
        "experimentalDefineExperiment: invocation window.turns must be a positive safe integer",
      );
    }
    if (maxAgeMs !== undefined && (!Number.isFinite(maxAgeMs) || maxAgeMs <= 0)) {
      throw new Error(
        "experimentalDefineExperiment: invocation window.maxAgeMs must be positive and finite",
      );
    }
  }

  if (peerReferences > 1) {
    throw new Error("experimentalDefineExperiment: duplicate peer-selection reference");
  }
  if (invocationReferences > 1) {
    throw new Error("experimentalDefineExperiment: duplicate invocation reference");
  }
}

/** @internal Resolve the arm for one selection from an explicit override or the deterministic split. */
export function assignArm<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  options: ExperimentSelectOptions<Result, Arm>,
): Arm {
  if (options.arm !== undefined) {
    if (!Object.hasOwn(config.arms, options.arm)) {
      throw new Error(
        `experimentalDefineExperiment.select: arm "${options.arm}" is not a declared arm`,
      );
    }
    return options.arm;
  }
  if (config.split === undefined) {
    throw new Error("experimentalDefineExperiment.select: arm or split is required");
  }

  const total = config.split.reduce((sum, allocation) => sum + allocation.weight, 0);
  const digest = createHash("sha256")
    .update(JSON.stringify([config.id, options.unitId]), "utf8")
    .digest("hex");
  const bucket = Number(BigInt(`0x${digest}`) % BigInt(total));
  let cumulative = 0;
  for (const allocation of config.split) {
    cumulative += allocation.weight;
    if (bucket < cumulative) {
      return allocation.arm;
    }
  }

  throw new Error("experimentalDefineExperiment: split has no positive allocation");
}

/** @internal Reject a malformed delayed outcome report synchronously. */
export function validateReportedOutcome(
  args: { selectionId: string } & ExperimentReportedOutcome,
): void {
  if (typeof args.selectionId !== "string" || args.selectionId.length === 0) {
    throw new Error(
      "experimentalDefineExperiment.reportOutcome: selectionId must be a non-empty string",
    );
  }
  if (args.label === undefined && args.score === undefined) {
    throw new Error("experimentalDefineExperiment.reportOutcome: label or score is required");
  }
  if (args.label !== undefined && (typeof args.label !== "string" || args.label.length === 0)) {
    throw new Error("experimentalDefineExperiment.reportOutcome: label must be a non-empty string");
  }
  if (
    args.score !== undefined &&
    (typeof args.score !== "number" || !Number.isFinite(args.score))
  ) {
    throw new Error("experimentalDefineExperiment.reportOutcome: score must be finite");
  }
}
