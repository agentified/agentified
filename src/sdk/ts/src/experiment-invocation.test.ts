import { describe, expect, it } from "vitest";
import { createExperimentInvocationBuffer, hashUnitId } from "./experiment-invocation.js";

describe("createExperimentInvocationBuffer", () => {
  it("attributes an invocation to the newest completed selection by default", () => {
    let nowMs = 100;
    const buffer = createExperimentInvocationBuffer<"legacy" | "candidate">(
      { window: { maxAgeMs: 1_000 } },
      () => nowMs,
    );
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-1",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 150;

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })).toEqual([
      {
        attributed: true,
        selectionId: "selection-1",
        effectiveArm: "legacy",
        rank: 0,
        ageMs: 50,
      },
    ]);
  });

  it("attributes all-in-window to the last N selections in oldest-to-newest order", () => {
    let nowMs = 0;
    const buffer = createExperimentInvocationBuffer<"legacy" | "candidate">(
      { window: { turns: 2 }, attribution: "all-in-window" },
      () => nowMs,
    );
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-1",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 10;
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-2",
      effectiveArm: "candidate",
      ids: ["read-logs", "inspect-ci"],
    });
    nowMs = 20;
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-3",
      effectiveArm: "candidate",
      ids: ["retry-build", "read-logs", "inspect-ci"],
    });

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })).toEqual([
      {
        attributed: true,
        selectionId: "selection-2",
        effectiveArm: "candidate",
        rank: 1,
        ageMs: 10,
      },
      {
        attributed: true,
        selectionId: "selection-3",
        effectiveArm: "candidate",
        rank: 2,
        ageMs: 0,
      },
    ]);
  });

  it("requires selections to satisfy both positional and age bounds", () => {
    let nowMs = 0;
    const buffer = createExperimentInvocationBuffer(
      {
        window: { turns: 2, maxAgeMs: 150 },
        attribution: "all-in-window",
      },
      () => nowMs,
    );
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-1",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 900;
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-2",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 1_000;
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-3",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 1_100;

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })).toEqual([
      {
        attributed: true,
        selectionId: "selection-3",
        effectiveArm: "legacy",
        rank: 0,
        ageMs: 100,
      },
    ]);
  });

  it("retains at most 32 completed selections per unit", () => {
    let nowMs = 0;
    const buffer = createExperimentInvocationBuffer(
      {
        window: { maxAgeMs: 60_000 },
        attribution: "all-in-window",
      },
      () => nowMs,
    );
    for (let index = 1; index <= 33; index += 1) {
      buffer.recordSelection({
        unitId: "unit-a",
        selectionId: `selection-${index}`,
        effectiveArm: "legacy",
        ids: ["inspect-ci"],
      });
      nowMs += 1;
    }

    expect(
      buffer
        .evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })
        .map((match) => (match.attributed ? match.selectionId : "unattributed")),
    ).toEqual(Array.from({ length: 32 }, (_, index) => `selection-${index + 2}`));
  });

  it("retains at most 1000 units and evicts the least recently recorded unit", () => {
    let nowMs = 0;
    const buffer = createExperimentInvocationBuffer({ window: { maxAgeMs: 60_000 } }, () => nowMs);
    for (let index = 0; index < 1_000; index += 1) {
      buffer.recordSelection({
        unitId: `unit-${index}`,
        selectionId: `selection-${index}`,
        effectiveArm: "legacy",
        ids: ["inspect-ci"],
      });
      nowMs += 1;
    }
    buffer.recordSelection({
      unitId: "unit-0",
      selectionId: "selection-0-latest",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    buffer.recordSelection({
      unitId: "unit-1000",
      selectionId: "selection-1000",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });

    expect(
      ["unit-0", "unit-1"].map(
        (unitId) => buffer.evaluateInvocation({ unitId, toolId: "inspect-ci" })[0]?.attributed,
      ),
    ).toEqual([true, false]);
  });

  it("represents a unit with no completed selection as unattributed", () => {
    const buffer = createExperimentInvocationBuffer({
      window: { maxAgeMs: 1_000 },
    });

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })).toEqual([
      { attributed: false },
    ]);
  });

  it("uses rank -1 when the invoked tool is absent from a matched ranking", () => {
    const buffer = createExperimentInvocationBuffer({
      window: { turns: 1 },
    });
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-1",
      effectiveArm: "legacy",
      ids: ["read-logs"],
    });

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })[0]).toMatchObject({
      attributed: true,
      rank: -1,
    });
  });

  it("clamps a negative clock delta to zero", () => {
    let nowMs = 100;
    const buffer = createExperimentInvocationBuffer({ window: { maxAgeMs: 1_000 } }, () => nowMs);
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-1",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 90;

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })[0]).toMatchObject({
      attributed: true,
      ageMs: 0,
    });
  });

  it("expires completed selections lazily after maxAgeMs", () => {
    let nowMs = 0;
    const buffer = createExperimentInvocationBuffer({ window: { maxAgeMs: 1_000 } }, () => nowMs);
    buffer.recordSelection({
      unitId: "unit-a",
      selectionId: "selection-1",
      effectiveArm: "legacy",
      ids: ["inspect-ci"],
    });
    nowMs = 1_001;

    expect(buffer.evaluateInvocation({ unitId: "unit-a", toolId: "inspect-ci" })).toEqual([
      { attributed: false },
    ]);
  });
});

describe("hashUnitId", () => {
  it("returns the first 16 lowercase SHA-256 hex characters", () => {
    expect(hashUnitId("unit-a")).toBe("d7bce2267437426d");
  });
});
