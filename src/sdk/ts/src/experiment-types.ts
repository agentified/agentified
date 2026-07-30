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
