export type ExperimentEvaluationScalar = string | number | boolean | null;

export type ExperimentEvaluationAttributes = Readonly<Record<string, ExperimentEvaluationScalar>>;

export interface ExperimentEvaluationRankedItem {
  id: string;
  score?: number;
  attrs?: ExperimentEvaluationAttributes;
}

export interface ExperimentRankingAgreement {
  top1: boolean;
  exactOrder: boolean;
  overlapCount: number;
  jaccardAtK: number;
  k: number;
  itemAttrs?: Record<string, boolean>;
  resultAttrs?: Record<string, boolean>;
}

export interface ExperimentRankingComparisonOptions {
  k: number;
  servedResultAttrs?: ExperimentEvaluationAttributes;
  shadowResultAttrs?: ExperimentEvaluationAttributes;
}

const DEFAULT_EVALUATION_K = 10;

/** Compare two ordered experiment rankings without comparing their scores. */
export function compareRankings(
  served: readonly ExperimentEvaluationRankedItem[],
  shadow: readonly ExperimentEvaluationRankedItem[],
  options: ExperimentRankingComparisonOptions,
): ExperimentRankingAgreement {
  const servedIds = served.map((item) => item.id);
  const shadowIds = shadow.map((item) => item.id);
  const servedWindow = new Set(servedIds.slice(0, options.k));
  const shadowWindow = new Set(shadowIds.slice(0, options.k));
  const agreement: ExperimentRankingAgreement = {
    top1: servedIds[0] === shadowIds[0],
    exactOrder: sameOrder(servedIds, shadowIds),
    overlapCount: intersectionSize(new Set(servedIds), new Set(shadowIds)),
    jaccardAtK: jaccard(servedWindow, shadowWindow),
    k: options.k,
  };

  if (served[0] !== undefined && shadow[0] !== undefined) {
    agreement.itemAttrs = compareSharedAttributes(served[0].attrs, shadow[0].attrs);
  }
  if (options.servedResultAttrs !== undefined && options.shadowResultAttrs !== undefined) {
    agreement.resultAttrs = compareResultAttributes(
      options.servedResultAttrs,
      options.shadowResultAttrs,
    );
  }
  return agreement;
}

/** Resolve the Jaccard rank window for one selection. */
export function resolveEvaluationK(requestK?: number, configuredK?: number): number {
  return isValidK(requestK) ? requestK : (configuredK ?? DEFAULT_EVALUATION_K);
}

function sameOrder(served: readonly string[], shadow: readonly string[]): boolean {
  return served.length === shadow.length && served.every((id, index) => id === shadow[index]);
}

function intersectionSize(served: ReadonlySet<string>, shadow: ReadonlySet<string>): number {
  let count = 0;
  for (const id of served) {
    if (shadow.has(id)) {
      count += 1;
    }
  }
  return count;
}

function jaccard(served: ReadonlySet<string>, shadow: ReadonlySet<string>): number {
  if (served.size === 0 && shadow.size === 0) {
    return 1;
  }
  const intersection = intersectionSize(served, shadow);
  return intersection / (served.size + shadow.size - intersection);
}

function compareSharedAttributes(
  served: ExperimentEvaluationAttributes = {},
  shadow: ExperimentEvaluationAttributes = {},
): Record<string, boolean> {
  const sharedKeys = Object.keys(served).filter((key) => Object.hasOwn(shadow, key));
  return Object.fromEntries(sharedKeys.map((key) => [key, served[key] === shadow[key]]));
}

function compareResultAttributes(
  served: ExperimentEvaluationAttributes,
  shadow: ExperimentEvaluationAttributes,
): Record<string, boolean> {
  const keys = new Set([...Object.keys(served), ...Object.keys(shadow)]);
  return Object.fromEntries(
    [...keys].map((key) => [
      key,
      Object.hasOwn(served, key) && Object.hasOwn(shadow, key) && served[key] === shadow[key],
    ]),
  );
}

function isValidK(value: number | undefined): value is number {
  return value !== undefined && Number.isInteger(value) && value >= 1;
}
