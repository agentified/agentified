import * as sdk from "./index.js";
import { experimentalDefineExperiment } from "./index.js";

interface Params {
  query: string;
}

interface Result {
  ids: string[];
}

const experiment = experimentalDefineExperiment({
  id: "search-ranking",
  arms: {
    control: {
      select: async (params: Params): Promise<Result> => ({ ids: [params.query] }),
    },
    candidate: {
      select: async (params: Params): Promise<Result> => ({ ids: [params.query] }),
    },
  },
  split: [
    { arm: "control", weight: 90 },
    { arm: "candidate", weight: 10 },
  ],
  ranking: (result) => result.ids.map((id) => ({ id })),
  evaluation: { references: ["peer-selection"] },
});

experiment.select({ query: "build failure" }, { arm: "candidate", unitId: "user-1" });

// @ts-expect-error arm names are inferred from the declared arms
experiment.select({ query: "build failure" }, { arm: "missing", unitId: "user-1" });

type Selection = Awaited<ReturnType<typeof experiment.select>>;
declare const selection: Selection;
const assignedArm: "control" | "candidate" = selection.assignedArm;
const effectiveArm: "control" | "candidate" = selection.effectiveArm;
void [assignedArm, effectiveArm];

// @ts-expect-error ADR-0016 defers the stable factory alias
sdk.defineExperiment;

const widenedSplit = [
  { arm: "control", weight: 90 },
  { arm: "candidate", weight: 10 },
];

experimentalDefineExperiment({
  id: "widened-split",
  arms: {
    control: {
      select: async (params: Params): Promise<Result> => ({ ids: [params.query] }),
    },
    candidate: {
      select: async (params: Params): Promise<Result> => ({ ids: [params.query] }),
    },
  },
  // @ts-expect-error an extracted split must preserve declared arm literals
  split: widenedSplit,
  ranking: (result) => result.ids.map((id) => ({ id })),
  evaluation: { references: ["peer-selection"] },
});

experimentalDefineExperiment({
  id: "fallback-inference",
  arms: {
    control: {
      select: async (params: Params): Promise<Result> => ({ ids: [params.query] }),
    },
  },
  // @ts-expect-error fallback must name a declared arm and cannot widen the arm union
  fallbackArm: "missing",
  ranking: (result) => result.ids.map((id) => ({ id })),
  evaluation: { references: ["peer-selection"] },
});
