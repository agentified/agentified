import { describe, expect, it, vi } from "vitest";
import { type ExperimentEvaluationReference, experimentalDefineExperiment } from "./index.js";

interface SearchParams {
  query: string;
}

interface SearchResult {
  ids: string[];
}

interface Deferred<T> {
  promise: Promise<T>;
  reject(reason: unknown): void;
  resolve(value: T): void;
}

function deferred<T>(): Deferred<T> {
  let resolve = (_value: T): void => {};
  let reject = (_reason: unknown): void => {};
  const promise = new Promise<T>((settle, fail) => {
    resolve = settle;
    reject = fail;
  });
  return { promise, reject, resolve };
}

describe("experimentalDefineExperiment", () => {
  it("rejects an experiment with no declared arms", () => {
    expect(() =>
      experimentalDefineExperiment<SearchParams, SearchResult, never>({
        id: "search",
        arms: {},
        ranking: (result) => result.ids.map((id) => ({ id })),
        evaluation: { references: ["peer-selection"] },
      }),
    ).toThrow(/arms.*not.*empty/i);
  });

  it("serves an explicitly selected arm through the public SDK export", async () => {
    const params = { query: "build failure" };
    const served = { ids: ["inspect-ci"] };
    const select = vi.fn(async () => served);
    const experiment = experimentalDefineExperiment({
      id: "search-ranking",
      arms: {
        control: { select },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(params, {
      arm: "control",
      unitId: "user-1",
    });

    expect(selection.result).toBe(served);
    expect(selection.assignedArm).toBe("control");
    expect(selection.effectiveArm).toBe("control");
    expect(select).toHaveBeenCalledWith(params, { role: "serving" });
  });

  it("assigns a unit with the canonical weighted SHA-256 split", async () => {
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: async () => ({ ids: ["legacy"] }) },
        ratel: { select: async () => ({ ids: ["ratel"] }) },
        hybrid: { select: async () => ({ ids: ["hybrid"] }) },
      },
      split: [
        { arm: "legacy", weight: 2 },
        { arm: "ratel", weight: 1 },
        { arm: "hybrid", weight: 3 },
      ],
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select({ query: "build failure" }, { unitId: "unit-b" });

    // SHA-256(JSON.stringify(["search", "unit-b"])) mod 6 is 2.
    expect(selection.assignedArm).toBe("ratel");
    expect(selection.result.ids).toEqual(["ratel"]);
  });

  it("lets an explicit zero-weight arm override the configured split", async () => {
    const candidate = vi.fn(async () => ({ ids: ["candidate"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        control: { select: async () => ({ ids: ["control"] }) },
        candidate: { select: candidate },
      },
      split: [
        { arm: "control", weight: 1 },
        { arm: "candidate", weight: 0 },
      ],
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "candidate", unitId: "unit-a" },
    );

    expect(selection.assignedArm).toBe("candidate");
    expect(candidate).toHaveBeenCalledOnce();
  });

  it("rejects a split that does not allocate every declared arm", () => {
    expect(() =>
      experimentalDefineExperiment({
        id: "search",
        arms: {
          legacy: { select: async () => ({ ids: ["legacy"] }) },
          ratel: { select: async () => ({ ids: ["ratel"] }) },
        },
        split: [{ arm: "legacy", weight: 1 }],
        ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
        evaluation: { references: ["peer-selection"] },
      }),
    ).toThrow(/split.*every declared arm/i);
  });

  it("rejects non-integer, unsafe, and zero-total split weights", () => {
    for (const weight of [-1, 0, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() =>
        experimentalDefineExperiment({
          id: "search",
          arms: {
            legacy: { select: async () => ({ ids: ["legacy"] }) },
          },
          split: [{ arm: "legacy", weight }],
          ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
          evaluation: { references: ["peer-selection"] },
        }),
      ).toThrow(/split.*weight|split.*total/i);
    }
  });

  it("rejects an undeclared fallback arm at definition time", () => {
    expect(() =>
      experimentalDefineExperiment({
        id: "search",
        arms: {
          legacy: { select: async () => ({ ids: ["legacy"] }) },
        },
        ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
        evaluation: { references: ["peer-selection"] },
        fallbackArm: "missing" as "legacy",
      }),
    ).toThrow(/fallbackArm "missing".*not.*declared/i);
  });

  it("rejects non-positive and fractional shadow concurrency", () => {
    for (const concurrency of [0, -1, 1.5]) {
      expect(() =>
        experimentalDefineExperiment({
          id: "search",
          arms: {
            legacy: { select: async () => ({ ids: ["legacy"] }) },
          },
          ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
          evaluation: { references: ["peer-selection"] },
          shadowPolicy: { concurrency },
        }),
      ).toThrow(/shadowPolicy\.concurrency.*positive integer/i);
    }
  });

  it("rejects invalid evaluation bounds and duplicate references", () => {
    const define = (evaluation: Parameters<typeof experimentalDefineExperiment>[0]["evaluation"]) =>
      experimentalDefineExperiment({
        id: "search",
        arms: {
          legacy: {
            select: async (_params: SearchParams) => ({ ids: ["legacy"] }),
          },
        },
        ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
        evaluation,
      });

    expect(() => define({ references: [] })).toThrow(/references.*not.*empty/i);
    expect(() => define({ k: 0, references: ["peer-selection"] })).toThrow(
      /evaluation\.k.*integer.*1/i,
    );
    expect(() =>
      define({
        references: [{ kind: "invocation", window: {} }],
      }),
    ).toThrow(/invocation window.*bound/i);
    expect(() => define({ references: ["peer-selection", "peer-selection"] })).toThrow(
      /duplicate.*peer-selection/i,
    );
    expect(() =>
      define({
        references: [
          { kind: "other", window: { turns: 1 } } as unknown as ExperimentEvaluationReference,
        ],
      }),
    ).toThrow(/reference\.kind.*invocation/i);
    expect(() =>
      define({
        references: [
          {
            kind: "invocation",
            window: { turns: 1 },
            attribution: "first-selection",
          } as unknown as ExperimentEvaluationReference,
        ],
      }),
    ).toThrow(/attribution.*last-selection.*all-in-window/i);
  });

  it("rejects an unknown explicit arm synchronously before dispatch", () => {
    const select = vi.fn(async () => ({ ids: ["legacy"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    expect(() =>
      experiment.select(
        { query: "build failure" },
        { arm: "missing" as "legacy", unitId: "unit-a" },
      ),
    ).toThrow(/arm "missing".*not.*declared/i);
    expect(select).not.toHaveBeenCalled();
  });

  it("requires either an explicit arm or a configured split before dispatch", () => {
    const select = vi.fn(async () => ({ ids: ["legacy"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    expect(() => experiment.select({ query: "build failure" }, { unitId: "unit-a" })).toThrow(
      /arm or split.*required/i,
    );
    expect(select).not.toHaveBeenCalled();
  });

  it("rejects shadow mode on a single-arm experiment before dispatch", () => {
    const select = vi.fn(async () => ({ ids: ["legacy"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    expect(() =>
      experiment.select(
        { query: "build failure" },
        { arm: "legacy", shadow: true, unitId: "unit-a" },
      ),
    ).toThrow(/shadow requires a second declared arm/i);
    expect(select).not.toHaveBeenCalled();
  });

  it("transforms a served result before ranking and returning it", async () => {
    const raw = { ids: ["visible", "hidden"] };
    const transformed = { ids: ["visible"] };
    const ranking = vi.fn((result: SearchResult) => result.ids.map((id) => ({ id })));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: async () => raw },
      },
      ranking,
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "build failure" },
      {
        arm: "legacy",
        unitId: "unit-a",
        transform: () => transformed,
      },
    );

    expect(selection.result).toBe(transformed);
    expect(ranking).toHaveBeenCalledWith(transformed);
  });

  it("preserves an intentional null transform result", async () => {
    const experiment = experimentalDefineExperiment<SearchParams, string | null, "legacy">({
      id: "search",
      arms: {
        legacy: { select: async () => "raw" },
      },
      ranking: (result) => (result === null ? [] : [{ id: result }]),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "build failure" },
      {
        arm: "legacy",
        unitId: "unit-a",
        transform: () => null,
      },
    );

    expect(selection.result).toBeNull();
  });

  it("measures only the effective arm callback duration", async () => {
    vi.useFakeTimers();
    try {
      const experiment = experimentalDefineExperiment({
        id: "search",
        arms: {
          legacy: {
            select: async () => {
              vi.advanceTimersByTime(40);
              return { ids: ["legacy"] };
            },
          },
        },
        ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
        evaluation: { references: ["peer-selection"] },
      });

      const selection = await experiment.select(
        { query: "build failure" },
        {
          arm: "legacy",
          unitId: "unit-a",
          transform: (result) => {
            vi.advanceTimersByTime(100);
            return result;
          },
        },
      );

      expect(selection.durationMs).toBe(40);
    } finally {
      vi.useRealTimers();
    }
  });

  it("serves a fresh fallback when the assigned arm rejects", async () => {
    const assignedError = new Error("candidate unavailable");
    const fallback = { ids: ["legacy"] };
    const candidate = vi.fn(async () => {
      throw assignedError;
    });
    const legacy = vi.fn(async () => fallback);
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        candidate: { select: candidate },
        legacy: { select: legacy },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "legacy",
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "candidate", unitId: "unit-a" },
    );

    expect(selection.result).toBe(fallback);
    expect(selection.assignedArm).toBe("candidate");
    expect(selection.effectiveArm).toBe("legacy");
    expect(legacy).toHaveBeenCalledWith(expect.anything(), { role: "serving" });
  });

  it("falls back when a rejection has a throwing name accessor", async () => {
    const rejected = Object.defineProperty({}, "name", {
      get() {
        throw new Error("name unavailable");
      },
    });
    const fallback = vi.fn(async () => ({ ids: ["legacy"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        candidate: {
          select: async () => {
            throw rejected;
          },
        },
        legacy: { select: fallback },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "legacy",
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "candidate", unitId: "unit-a" },
    );

    expect(selection.result.ids).toEqual(["legacy"]);
    expect(fallback).toHaveBeenCalledOnce();
  });

  it("does not fall back from an empty successful ranking", async () => {
    const fallback = vi.fn(async () => ({ ids: ["legacy"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        candidate: { select: async () => ({ ids: [] }) },
        legacy: { select: fallback },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "legacy",
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "candidate", unitId: "unit-a" },
    );

    expect(selection.result.ids).toEqual([]);
    expect(selection.effectiveArm).toBe("candidate");
    expect(fallback).not.toHaveBeenCalled();
  });

  it("contains ranking projection failures without falling back", async () => {
    const served = { ids: ["candidate"] };
    const fallback = vi.fn(async () => ({ ids: ["legacy"] }));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        candidate: { select: async () => served },
        legacy: { select: fallback },
      },
      ranking: () => {
        throw new Error("projection failed");
      },
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "legacy",
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "candidate", unitId: "unit-a" },
    );

    expect(selection.result).toBe(served);
    expect(selection.effectiveArm).toBe("candidate");
    expect(fallback).not.toHaveBeenCalled();
  });

  it("reuses a pending fallback shadow without changing its role or transforming twice", async () => {
    const fallback = { ids: ["legacy"] };
    const pendingFallback = deferred<SearchResult>();
    const legacy = vi.fn(() => pendingFallback.promise);
    const transform = vi.fn((result: SearchResult) => result);
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        candidate: {
          select: async () => {
            throw new Error("candidate unavailable");
          },
        },
        legacy: { select: legacy },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "legacy",
    });

    const selecting = experiment.select(
      { query: "build failure" },
      {
        arm: "candidate",
        shadow: true,
        unitId: "unit-a",
        transform,
      },
    );
    let selected = false;
    void selecting.then(() => {
      selected = true;
    });
    await new Promise((resolve) => setImmediate(resolve));
    expect(selected).toBe(false);

    pendingFallback.resolve(fallback);
    const selection = await selecting;
    await experiment.drain();

    expect(selection.result).toBe(fallback);
    expect(legacy).toHaveBeenCalledOnce();
    expect(legacy).toHaveBeenCalledWith(expect.anything(), { role: "shadow" });
    expect(transform).toHaveBeenCalledOnce();
  });

  it("starts a capacity-skipped fallback fresh as serving and warms it then", async () => {
    const occupied = deferred<SearchResult>();
    const fallbackWarmup = vi.fn(async () => {});
    const fallback = vi.fn(
      async (_params: SearchParams, { role }: { role: "serving" | "shadow" }) => ({
        ids: [role],
      }),
    );
    const assigned = vi
      .fn<() => Promise<SearchResult>>()
      .mockResolvedValueOnce({ ids: ["legacy"] })
      .mockRejectedValueOnce(new Error("legacy unavailable"));
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: assigned },
        ratel: { select: () => occupied.promise },
        hybrid: { select: fallback, warmup: fallbackWarmup },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      fallbackArm: "hybrid",
    });
    const options = {
      arm: "legacy" as const,
      shadow: true,
      unitId: "unit-a",
    };

    await experiment.select({ query: "first" }, options);
    expect(fallbackWarmup).not.toHaveBeenCalled();

    const selection = await experiment.select({ query: "second" }, options);
    expect(selection.result.ids).toEqual(["serving"]);
    expect(fallback).toHaveBeenCalledWith(expect.anything(), { role: "serving" });
    expect(fallbackWarmup).toHaveBeenCalledOnce();

    occupied.resolve({ ids: ["ratel"] });
    await experiment.drain();
  });

  it("starts serving before every admitted shadow without awaiting them", async () => {
    const started: string[] = [];
    const ratel = deferred<SearchResult>();
    const hybrid = deferred<SearchResult>();
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: {
          select: async () => {
            started.push("legacy:serving");
            return { ids: ["legacy"] };
          },
        },
        ratel: {
          select: async (_params, { role }) => {
            started.push(`ratel:${role}`);
            return ratel.promise;
          },
        },
        hybrid: {
          select: async (_params, { role }) => {
            started.push(`hybrid:${role}`);
            return hybrid.promise;
          },
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      shadowPolicy: { concurrency: 2 },
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "legacy", shadow: true, unitId: "unit-a" },
    );

    expect(selection.result.ids).toEqual(["legacy"]);
    expect(started).toEqual(["legacy:serving", "ratel:shadow", "hybrid:shadow"]);

    ratel.resolve({ ids: ["ratel"] });
    hybrid.resolve({ ids: ["hybrid"] });
    await experiment.drain();
  });

  it("contains a detached shadow transform failure", async () => {
    const served = { ids: ["legacy"] };
    const shadowed = { ids: ["ratel"] };
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: async () => served },
        ratel: { select: async () => shadowed },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "build failure" },
      {
        arm: "legacy",
        shadow: true,
        unitId: "unit-a",
        transform: (result) => {
          if (result === shadowed) {
            throw new Error("shadow filter failed");
          }
          return result;
        },
      },
    );

    expect(selection.result).toBe(served);
    await expect(experiment.drain()).resolves.toBeUndefined();
  });

  it("never emits an unhandled rejection from detached or warmup failures", async () => {
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    try {
      const experiment = experimentalDefineExperiment({
        id: "search",
        arms: {
          legacy: { select: async () => ({ ids: ["legacy"] }) },
          ratel: {
            select: async () => {
              throw new Error("shadow failed");
            },
            warmup: async () => {
              throw new Error("warmup failed");
            },
          },
        },
        ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
        evaluation: { references: ["peer-selection"] },
      });

      await experiment.select(
        { query: "build failure" },
        { arm: "legacy", shadow: true, unitId: "unit-a" },
      );
      await new Promise((resolve) => setImmediate(resolve));

      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      process.off("unhandledRejection", unhandled);
    }
  });

  it("skips excess shadows across calls and admits new work after callback settlement", async () => {
    const firstShadow = deferred<SearchResult>();
    const shadow = vi
      .fn<() => Promise<SearchResult>>()
      .mockReturnValueOnce(firstShadow.promise)
      .mockResolvedValue({ ids: ["ratel"] });
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: async () => ({ ids: ["legacy"] }) },
        ratel: { select: shadow },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });
    const options = {
      arm: "legacy" as const,
      shadow: true,
      unitId: "unit-a",
    };

    await experiment.select({ query: "first" }, options);
    await experiment.select({ query: "second" }, options);
    expect(shadow).toHaveBeenCalledTimes(1);

    firstShadow.resolve({ ids: ["ratel"] });
    await experiment.drain();
    await experiment.select({ query: "third" }, options);
    expect(shadow).toHaveBeenCalledTimes(2);
  });

  it("drains a snapshot of rejected detached shadow work without rejecting", async () => {
    const shadow = deferred<SearchResult>();
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: async () => ({ ids: ["legacy"] }) },
        ratel: { select: () => shadow.promise },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    await experiment.select(
      { query: "build failure" },
      { arm: "legacy", shadow: true, unitId: "unit-a" },
    );
    let drained = false;
    const draining = experiment.drain().then(() => {
      drained = true;
    });
    await Promise.resolve();
    expect(drained).toBe(false);

    shadow.reject(new Error("shadow failed"));
    await expect(draining).resolves.toBeUndefined();
    expect(drained).toBe(true);
  });

  it("does not extend an existing drain snapshot with later shadows", async () => {
    const firstShadow = deferred<SearchResult>();
    const laterShadow = deferred<SearchResult>();
    const shadow = vi
      .fn<() => Promise<SearchResult>>()
      .mockReturnValueOnce(firstShadow.promise)
      .mockReturnValueOnce(laterShadow.promise);
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: { select: async () => ({ ids: ["legacy"] }) },
        ratel: { select: shadow },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
      shadowPolicy: { concurrency: 2 },
    });
    const options = {
      arm: "legacy" as const,
      shadow: true,
      unitId: "unit-a",
    };

    await experiment.select({ query: "first" }, options);
    const firstDrain = experiment.drain();
    await experiment.select({ query: "later" }, options);

    firstShadow.resolve({ ids: ["ratel"] });
    await expect(firstDrain).resolves.toBeUndefined();

    let laterDrained = false;
    const laterDrain = experiment.drain().then(() => {
      laterDrained = true;
    });
    await Promise.resolve();
    expect(laterDrained).toBe(false);
    laterShadow.resolve({ ids: ["ratel"] });
    await laterDrain;
  });

  it("warms every declared arm concurrently and never rejects", async () => {
    const legacyGate = deferred<void>();
    const legacyWarmup = vi.fn(() => legacyGate.promise);
    const ratelWarmup = vi.fn(async () => {
      throw new Error("embedding unavailable");
    });
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        legacy: {
          select: async () => ({ ids: ["legacy"] }),
          warmup: legacyWarmup,
        },
        ratel: {
          select: async () => ({ ids: ["ratel"] }),
          warmup: ratelWarmup,
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const warming = experiment.warm();
    expect(legacyWarmup).toHaveBeenCalledOnce();
    expect(ratelWarmup).toHaveBeenCalledOnce();

    legacyGate.resolve();
    await expect(warming).resolves.toBeUndefined();
  });

  it("starts assigned-arm warmup lazily without delaying selection", async () => {
    const warming = deferred<void>();
    const warmup = vi.fn(() => warming.promise);
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        ratel: {
          select: async () => ({ ids: ["ratel"] }),
          warmup,
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "build failure" },
      { arm: "ratel", unitId: "unit-a" },
    );

    expect(warmup).toHaveBeenCalledOnce();
    expect(selection.result.ids).toEqual(["ratel"]);
    warming.resolve();
  });

  it("deduplicates concurrent warmups and memoizes success", async () => {
    const warming = deferred<void>();
    const warmup = vi.fn(() => warming.promise);
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        ratel: {
          select: async () => ({ ids: ["ratel"] }),
          warmup,
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const first = experiment.warm();
    const concurrent = experiment.warm();
    expect(warmup).toHaveBeenCalledOnce();
    warming.resolve();
    await Promise.all([first, concurrent]);

    await experiment.warm();
    expect(warmup).toHaveBeenCalledOnce();
  });

  it("retries an arm warmup after a contained failure", async () => {
    const warmup = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new Error("registration failed"))
      .mockResolvedValue(undefined);
    const experiment = experimentalDefineExperiment({
      id: "search",
      arms: {
        ratel: {
          select: async () => ({ ids: ["ratel"] }),
          warmup,
        },
      },
      ranking: (result: SearchResult) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    await expect(experiment.warm()).resolves.toBeUndefined();
    await expect(experiment.warm()).resolves.toBeUndefined();
    await experiment.warm();

    expect(warmup).toHaveBeenCalledTimes(2);
  });
});
