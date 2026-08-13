import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";

const MAX_TRACKED_UNITS = 1_000;
const MAX_SELECTIONS_PER_UNIT = 32;
const EXPIRY_SWEEP_BATCH = 8;

/** @internal */
export interface ExperimentInvocationPolicy {
  window: {
    turns?: number;
    maxAgeMs?: number;
  };
  attribution?: "last-selection" | "all-in-window";
}

/** @internal */
export interface ExperimentInvocationSelection<Arm extends string = string> {
  unitId: string;
  selectionId: string;
  effectiveArm: Arm;
  ids: readonly string[];
}

/** @internal */
export type ExperimentInvocationAttribution<Arm extends string = string> =
  | { attributed: false }
  | {
      attributed: true;
      selectionId: string;
      effectiveArm: Arm;
      rank: number;
      ageMs: number;
    };

/** @internal */
export interface ExperimentInvocationBuffer<Arm extends string = string> {
  recordSelection(selection: ExperimentInvocationSelection<Arm>): void;
  evaluateInvocation(args: {
    unitId: string;
    toolId: string;
  }): readonly ExperimentInvocationAttribution<Arm>[];
}

interface SelectionRecord<Arm extends string> {
  selectionId: string;
  effectiveArm: Arm;
  ids: readonly string[];
  completedAtMs: number;
}

/**
 * Build one bounded in-process attribution buffer for an experiment instance.
 *
 * @internal
 */
export function createExperimentInvocationBuffer<Arm extends string = string>(
  policy: ExperimentInvocationPolicy,
  clock: () => number = monotonicNow,
): ExperimentInvocationBuffer<Arm> {
  const selections = new Map<string, SelectionRecord<Arm>[]>();

  return {
    recordSelection(selection) {
      const completedAtMs = clock();
      sweepExpiredSelections(selections, policy.window.maxAgeMs, completedAtMs);
      const unitHash = hashUnitId(selection.unitId);
      const records = selections.get(unitHash) ?? [];
      selections.delete(unitHash);
      records.push({
        selectionId: selection.selectionId,
        effectiveArm: selection.effectiveArm,
        ids: [...selection.ids],
        completedAtMs,
      });
      if (records.length > MAX_SELECTIONS_PER_UNIT) {
        records.splice(0, records.length - MAX_SELECTIONS_PER_UNIT);
      }
      selections.set(unitHash, records);
      while (selections.size > MAX_TRACKED_UNITS) {
        const oldest = selections.keys().next();
        if (oldest.done) {
          break;
        }
        selections.delete(oldest.value);
      }
    },
    evaluateInvocation({ unitId, toolId }) {
      const nowMs = clock();
      sweepExpiredSelections(selections, policy.window.maxAgeMs, nowMs);
      const records = selections.get(hashUnitId(unitId));
      if (records === undefined) {
        return [{ attributed: false }];
      }

      const positional =
        policy.window.turns === undefined ? records : records.slice(-policy.window.turns);
      const inWindow = positional.filter(
        (record) =>
          policy.window.maxAgeMs === undefined ||
          elapsedMs(nowMs, record.completedAtMs) <= policy.window.maxAgeMs,
      );
      const matched = policy.attribution === "all-in-window" ? inWindow : inWindow.slice(-1);
      if (matched.length === 0) {
        return [{ attributed: false }];
      }

      return matched.map((record) => ({
        attributed: true,
        selectionId: record.selectionId,
        effectiveArm: record.effectiveArm,
        rank: record.ids.indexOf(toolId),
        ageMs: elapsedMs(nowMs, record.completedAtMs),
      }));
    },
  };
}

/** Hash a raw assignment unit to the pseudonymous in-memory join key. @internal */
export function hashUnitId(unitId: string): string {
  return createHash("sha256").update(unitId, "utf8").digest("hex").slice(0, 16);
}

function sweepExpiredSelections<Arm extends string>(
  selections: Map<string, SelectionRecord<Arm>[]>,
  maxAgeMs: number | undefined,
  nowMs: number,
): void {
  if (maxAgeMs === undefined) {
    return;
  }

  let examined = 0;
  for (const [unitHash, records] of selections) {
    if (examined >= EXPIRY_SWEEP_BATCH) {
      return;
    }
    examined += 1;
    const live = records.filter((record) => elapsedMs(nowMs, record.completedAtMs) <= maxAgeMs);
    if (live.length === 0) {
      selections.delete(unitHash);
    } else if (live.length !== records.length) {
      records.splice(0, records.length, ...live);
    }
  }
}

function elapsedMs(nowMs: number, completedAtMs: number): number {
  return Math.max(0, nowMs - completedAtMs);
}

function monotonicNow(): number {
  return performance.now();
}
