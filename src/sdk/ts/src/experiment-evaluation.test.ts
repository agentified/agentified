import { describe, expect, it } from "vitest";
import { compareRankings, resolveEvaluationK } from "./experiment-evaluation.js";

describe("resolveEvaluationK", () => {
  it("uses a valid request window before the configured window", () => {
    expect(resolveEvaluationK(3, 7)).toBe(3);
  });

  it.each([
    0,
    -1,
    1.5,
    Number.NaN,
    Number.POSITIVE_INFINITY,
  ])("ignores invalid request window %s", (requestK) => {
    expect(resolveEvaluationK(requestK, 7)).toBe(7);
  });

  it("uses the configured window when the request omits one", () => {
    expect(resolveEvaluationK(undefined, 7)).toBe(7);
  });

  it("defaults to ten when no valid request or configured window exists", () => {
    expect(resolveEvaluationK(0)).toBe(10);
  });
});

describe("compareRankings", () => {
  it("compares ids over the specified rank window", () => {
    const agreement = compareRankings(
      [{ id: "a" }, { id: "b" }, { id: "z" }],
      [{ id: "b" }, { id: "c" }, { id: "y" }],
      { k: 2 },
    );

    expect(agreement).toEqual({
      top1: false,
      exactOrder: false,
      overlapCount: 1,
      jaccardAtK: 1 / 3,
      k: 2,
      itemAttrs: {},
    });
  });

  it("treats two empty rankings as exact agreement", () => {
    expect(compareRankings([], [], { k: 3 })).toEqual({
      top1: true,
      exactOrder: true,
      overlapCount: 0,
      jaccardAtK: 1,
      k: 3,
    });
  });

  it("uses the complete de-duplicated sets for overlap", () => {
    const agreement = compareRankings(
      [{ id: "a" }, { id: "a" }, { id: "b" }, { id: "c" }],
      [{ id: "c" }, { id: "c" }, { id: "b" }, { id: "a" }],
      { k: 1 },
    );

    expect(agreement.overlapCount).toBe(3);
  });

  it("slices before de-duplicating the Jaccard window", () => {
    const agreement = compareRankings(
      [{ id: "a" }, { id: "a" }, { id: "b" }],
      [{ id: "a" }, { id: "b" }, { id: "c" }],
      { k: 2 },
    );

    expect(agreement.jaccardAtK).toBe(1 / 2);
  });

  it("compares only shared rank-zero item attributes", () => {
    const agreement = compareRankings(
      [
        { id: "a", attrs: { domain: "docs", shared: null, servedOnly: true } },
        { id: "b", attrs: { ignored: "same" } },
      ],
      [
        { id: "a", attrs: { domain: "code", shared: null, shadowOnly: true } },
        { id: "b", attrs: { ignored: "same" } },
      ],
      { k: 2 },
    );

    expect(agreement.itemAttrs).toEqual({ domain: false, shared: true });
  });

  it("compares result attributes over the union with strict missing and null semantics", () => {
    const agreement = compareRankings([], [], {
      k: 2,
      servedResultAttrs: {
        sameNull: null,
        sameFalse: false,
        different: "docs",
        servedOnly: null,
      },
      shadowResultAttrs: {
        sameNull: null,
        sameFalse: false,
        different: "code",
        shadowOnly: null,
      },
    });

    expect(agreement.resultAttrs).toEqual({
      sameNull: true,
      sameFalse: true,
      different: false,
      servedOnly: false,
      shadowOnly: false,
    });
  });

  it("includes an empty result map when both projections succeed empty", () => {
    const agreement = compareRankings([], [], {
      k: 1,
      servedResultAttrs: {},
      shadowResultAttrs: {},
    });

    expect(agreement.resultAttrs).toEqual({});
  });

  it("omits result attributes when either projection is unavailable", () => {
    const agreement = compareRankings([], [], {
      k: 1,
      servedResultAttrs: {},
    });

    expect(agreement).not.toHaveProperty("resultAttrs");
  });
});
