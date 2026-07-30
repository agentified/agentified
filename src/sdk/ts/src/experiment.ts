import { createHash, randomUUID } from "node:crypto";
import {
  compareRankings,
  type ExperimentEvaluationAttributes,
  type ExperimentRankingAgreement,
  resolveEvaluationK,
} from "./experiment-evaluation.js";
import {
  createExperimentInvocationBuffer,
  type ExperimentInvocationAttribution,
  type ExperimentInvocationBuffer,
  hashUnitId,
} from "./experiment-invocation.js";

/** Whether an arm dispatch is serving the caller or running as detached shadow work. */
export type ExperimentArmRole = "serving" | "shadow";

/** Normalized outcome of one experiment-arm dispatch. */
export type ExperimentArmOutcome = "ok" | "empty" | "timeout" | "error";

/** One ranked measurement projected from an arm result. */
export interface ExperimentRankedItem {
  /** Stable, non-content identifier used for rank comparison. */
  id: string;
  /** Optional arm-specific score; scores are emitted but never compared across arms. */
  score?: number;
  /** Optional item facts used by item-level evaluation. */
  attrs?: Record<string, string | number | boolean | null>;
}

/** Ordered integer-weight allocation used for deterministic arm assignment. */
export type ExperimentSplit<Arm extends string = string> = readonly Readonly<{
  /** Declared arm receiving this allocation. */
  arm: Arm;
  /** Non-negative safe-integer allocation weight. */
  weight: number;
}>[];

/** Label, score, or both supplied for a delayed experiment outcome. */
export type ExperimentReportedOutcome =
  | {
      /** Free-form non-empty outcome label. */
      label: string;
      /** Optional finite outcome score. */
      score?: number;
    }
  | {
      /** Optional free-form non-empty outcome label. */
      label?: string;
      /** Finite outcome score. */
      score: number;
    };

/** A served-vs-shadow comparison or invocation-attribution evaluation source. */
export type ExperimentEvaluationReference =
  | "peer-selection"
  | {
      /** Selects invocation attribution. */
      kind: "invocation";
      /** Bounds the in-process selection window used for attribution. */
      window: {
        /** Retain the last N completed selections for the unit. */
        turns?: number;
        /** Retain selections no older than this elapsed duration. */
        maxAgeMs?: number;
      };
      /** Whether to attribute only the newest match or every match in the window. */
      attribution?: "last-selection" | "all-in-window";
    };

/** The effective result and correlation data returned by a selection. */
export interface ExperimentSelection<Result, Arm extends string = string> {
  /** Transformed result that reaches the caller. */
  result: Result;
  /** Opaque identifier joining later reports and descendant telemetry. */
  selectionId: string;
  /** Arm chosen explicitly or by the deterministic split. */
  assignedArm: Arm;
  /** Arm whose result reached the caller after any fallback. */
  effectiveArm: Arm;
  /** Effective arm callback duration, excluding transform and evaluation work. */
  durationMs: number;
}

/** Per-call assignment, shadow, transform, and telemetry options. */
export interface ExperimentSelectOptions<Result, Arm extends string = string> {
  /** Stable assignment unit; raw values remain in process. */
  unitId: string;
  /** Explicit arm override. Required when the experiment has no split. */
  arm?: Arm;
  /** Attempt every non-assigned arm as detached shadow work. */
  shadow?: boolean;
  /** Per-request Jaccard rank window override. */
  k?: number;
  /** Synchronous visibility transform applied before evaluation and return. */
  transform?: (result: Result) => Result;
  /** Non-content scalar attributes copied to experiment telemetry. */
  attributes?: Record<string, string | number | boolean>;
}

/** A configured retrieval experiment instance. */
export interface Experiment<Params, Result, Arm extends string = string> {
  /** Select an arm, optionally run shadows, and return the effective transformed result. */
  select(
    params: Params,
    options: ExperimentSelectOptions<Result, Arm>,
  ): Promise<ExperimentSelection<Result, Arm>>;
  /** Report a downstream tool invocation for configured attribution evaluation. */
  reportInvocation(args: { unitId: string; toolId: string; turn?: number }): void;
  /** Append a delayed label or score for a prior selection. */
  reportOutcome(args: { selectionId: string } & ExperimentReportedOutcome): void;
  /** Start unresolved arm warmups concurrently; failures are contained and retryable. */
  warm(): Promise<void>;
  /** Await a snapshot of detached shadow pipelines started before this call. */
  drain(): Promise<void>;
}

/** Definition-time arms, assignment, evaluation, and lifecycle policy. */
export interface ExperimentConfig<Params, Result, Arm extends string = string> {
  /** Stable experiment identifier included in deterministic assignment. */
  id: string;
  /** Named host-supplied selectors. */
  arms: Record<
    Arm,
    {
      /** Run this arm with the exact caller parameters and immutable dispatch role. */
      select: (params: Params, context: { role: ExperimentArmRole }) => Promise<Result>;
      /** Optional readiness warmup; rejection keeps the arm cold and permits retry. */
      warmup?: () => Promise<void>;
    }
  >;
  /** Deterministic allocation used when a call does not explicitly name an arm. */
  split?: ExperimentSplit<NoInfer<Arm>>;
  /** Observational projection from a transformed result to ranked measurement items. */
  ranking: (result: Result) => ExperimentRankedItem[];
  /** Comparison and attribution policy. */
  evaluation: {
    /** Default Jaccard evaluation window. */
    k?: number;
    /** Non-empty evaluation sources enabled for this experiment. */
    references: readonly ExperimentEvaluationReference[];
    /** Optional result-level facts used for served-vs-shadow agreement. */
    attributes?: (result: Result) => Record<string, string | number | boolean | null>;
    /** Whether delayed outcome reports are enabled. */
    outcome?: boolean;
  };
  /** Arm used only when the assigned selector rejects. */
  fallbackArm?: NoInfer<Arm>;
  /** Detached shadow capacity policy. */
  shadowPolicy?: {
    /** Per-instance skip-never-queue concurrency, defaulting to one. */
    concurrency?: number;
  };
}

/** @internal Valid delayed outcome ready for telemetry emission. */
export interface ExperimentOutcomeEvaluation {
  experimentId: string;
  selectionId: string;
  label?: string;
  score?: number;
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
  comparison?(evaluation: ExperimentComparisonEvaluation<Arm>): void;
  drop?(evaluation: ExperimentDropEvaluation<Arm>): void;
  invocation?(evaluation: ExperimentInvocationEvaluation<Arm>): void;
  outcome?(evaluation: ExperimentOutcomeEvaluation): void;
}

type ArmCallbackResult<Result> =
  | { ok: true; result: Result; durationMs: number }
  | { ok: false; error: unknown; durationMs: number };

type ArmCompletion<Result> =
  | {
      ok: true;
      result: Result;
      durationMs: number;
      outcome: ExperimentArmOutcome;
      ranking?: ExperimentRankedItem[];
      resultAttributes?: ResultAttributesProjection;
    }
  | {
      ok: false;
      error: unknown;
      durationMs: number;
      outcome: "timeout" | "error";
      source: "select" | "transform";
    };

type ResultAttributesProjection =
  | { ok: true; attributes: ExperimentEvaluationAttributes }
  | { ok: false; error: unknown };

type SuccessfulArmCompletion<Result> = Extract<ArmCompletion<Result>, { ok: true }>;

type ServedEvaluationSignal<Result, Arm extends string> =
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

interface ArmRun<Result, Arm extends string> {
  arm: Arm;
  cold: boolean;
  role: ExperimentArmRole;
  completion: Promise<ArmCompletion<Result>>;
}

type ArmDispatcher<Result, Arm extends string> = (
  arm: Arm,
  role: ExperimentArmRole,
  onCallbackSettled?: () => void,
) => ArmRun<Result, Arm>;

interface WarmupEntry {
  resolved: boolean;
  promise: Promise<void>;
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

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
  return defineExperimentInternal(config);
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
      const selectionId = randomUUID();
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
          coldAtStart.get(arm) ?? false,
          onCallbackSettled,
        );
      };
      const assigned = dispatch(assignedArm, "serving");
      const shadows = new Map<Arm, ArmRun<Result, Arm>>();
      if (options.shadow === true) {
        for (const arm of armNames) {
          if (arm === assignedArm || shadowsInFlight >= shadowConcurrency) {
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

function ensureWarmup<Params, Result, Arm extends string>(
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

function startArm<Params, Result, Arm extends string>(
  config: ExperimentConfig<Params, Result, Arm>,
  arm: Arm,
  params: Params,
  role: ExperimentArmRole,
  options: ExperimentSelectOptions<Result, Arm>,
  cold: boolean,
  onCallbackSettled?: () => void,
): ArmRun<Result, Arm> {
  const callback = startArmCallback(config, arm, params, role, onCallbackSettled);
  const completion = callback.then((settled): ArmCompletion<Result> => {
    if (!settled.ok) {
      return {
        ...settled,
        outcome: isTimeoutError(settled.error) ? "timeout" : "error",
        source: "select",
      };
    }

    let result: Result;
    try {
      result = options.transform === undefined ? settled.result : options.transform(settled.result);
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
      const ranking = config.ranking(result);
      const resultAttributes = projectResultAttributes(config, result);
      return {
        ok: true,
        result,
        durationMs: settled.durationMs,
        outcome: ranking.length === 0 ? "empty" : "ok",
        ranking,
        resultAttributes,
      };
    } catch {
      return {
        ok: true,
        result,
        durationMs: settled.durationMs,
        outcome: "error",
      };
    }
  });
  return { arm, cold, role, completion };
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
    return { ok: true, attributes: project(result) };
  } catch (error) {
    return { ok: false, error };
  }
}

async function completePeerEvaluation<Result, Arm extends string>(
  shadow: ArmRun<Result, Arm>,
  servedPromise: Promise<ServedEvaluationSignal<Result, Arm>>,
  sink: ExperimentEvaluationSink<Arm>,
): Promise<void> {
  const [shadowCompletion, served] = await Promise.all([shadow.completion, servedPromise]);
  if (!shadowCompletion.ok || shadowCompletion.ranking === undefined) {
    notifyDrop(sink, served.selectionId, shadow.arm, "arm-failed");
    return;
  }
  if (served.status === "selection-failed") {
    notifyDrop(sink, served.selectionId, shadow.arm, "selection-failed");
    return;
  }
  if (served.consumedShadow === shadow.arm) {
    notifyDrop(sink, served.selectionId, shadow.arm, "fallback-consumed");
    return;
  }
  if (served.completion.ranking === undefined) {
    notifyDrop(sink, served.selectionId, shadow.arm, "served-ranking-failed");
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
    notifyEvaluation(sink.comparison, {
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
    });
  } catch {
    notifyDrop(sink, served.selectionId, shadow.arm, "comparison-failed");
  }
}

function notifyDrop<Arm extends string>(
  sink: ExperimentEvaluationSink<Arm>,
  selectionId: string,
  shadowArm: Arm,
  reason: ExperimentPeerDropReason,
): void {
  notifyEvaluation(sink.drop, { selectionId, shadowArm, reason });
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

function trackShadow(pending: Set<Promise<void>>, completion: Promise<unknown>): void {
  const contained = completion.then(
    () => undefined,
    () => undefined,
  );
  pending.add(contained);
  void contained.then(() => {
    pending.delete(contained);
  });
}

function createDeferred<T>(): Deferred<T> {
  let resolve = (_value: T): void => {};
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

function notifyEvaluation<Evaluation>(
  emit: ((evaluation: Evaluation) => void) | undefined,
  evaluation: Evaluation,
): void {
  try {
    emit?.(evaluation);
  } catch {
    // Evaluation delivery is observational and cannot fail application work.
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

function validateReportedOutcome(args: { selectionId: string } & ExperimentReportedOutcome): void {
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

function validateSplitCoverage<Params, Result, Arm extends string>(
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

function assignArm<Params, Result, Arm extends string>(
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
