import { context, propagation } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { experimentalDefineExperiment } from "@ratel-ai/sdk";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { experimentalRatelSpanOutputProcessor } from "./observability.js";

describe("experimentalRatelSpanOutputProcessor", () => {
  beforeEach(() => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
  });

  afterEach(() => {
    context.disable();
  });

  it("leaves spans unchanged outside an experiment", async () => {
    const processor = experimentalRatelSpanOutputProcessor();
    const span = { attributes: { model: "gpt-5" } };

    expect(processor.process(span)).toBe(span);
    expect(span.attributes).toEqual({ model: "gpt-5" });
    expect(processor.process()).toBeUndefined();
    await expect(processor.shutdown()).resolves.toBeUndefined();
  });

  it("copies the active experiment join onto a Mastra span", () => {
    const span = {
      attributes: {
        model: "gpt-5",
        "ratel.experiment.id": "counterfeit",
      },
    };
    const experimentContext = propagation.setBaggage(
      context.active(),
      propagation.createBaggage({
        "ratel.experiment.id": { value: "retrieval-v2" },
        "ratel.experiment.selection_id": { value: "selection-42" },
        "ratel.experiment.arm": { value: "hybrid" },
        "ratel.experiment.role": { value: "serving" },
        "ratel.experiment.unit": { value: "0123456789abcdef" },
      }),
    );

    const processed = context.with(experimentContext, () =>
      experimentalRatelSpanOutputProcessor().process(span),
    );

    expect(processed?.attributes).toEqual({
      model: "gpt-5",
      "ratel.experiment.id": "retrieval-v2",
      "ratel.experiment.selection_id": "selection-42",
      "ratel.experiment.arm": "hybrid",
      "ratel.experiment.role": "serving",
      "ratel.experiment.unit": "0123456789abcdef",
    });
  });

  it("keeps the experiment captured on the span's first processor pass", () => {
    const processor = experimentalRatelSpanOutputProcessor();
    const span = { attributes: { model: "gpt-5" } };
    const servingContext = propagation.setBaggage(
      context.active(),
      propagation.createBaggage({
        "ratel.experiment.id": { value: "retrieval-v2" },
        "ratel.experiment.selection_id": { value: "selection-serving" },
        "ratel.experiment.arm": { value: "hybrid" },
        "ratel.experiment.role": { value: "serving" },
        "ratel.experiment.unit": { value: "0123456789abcdef" },
      }),
    );
    const shadowContext = propagation.setBaggage(
      context.active(),
      propagation.createBaggage({
        "ratel.experiment.id": { value: "retrieval-v2" },
        "ratel.experiment.selection_id": { value: "selection-shadow" },
        "ratel.experiment.arm": { value: "semantic" },
        "ratel.experiment.role": { value: "shadow" },
        "ratel.experiment.unit": { value: "fedcba9876543210" },
      }),
    );

    context.with(servingContext, () => processor.process(span));
    context.with(shadowContext, () => processor.process(span));

    expect(span.attributes).toEqual({
      model: "gpt-5",
      "ratel.experiment.id": "retrieval-v2",
      "ratel.experiment.selection_id": "selection-serving",
      "ratel.experiment.arm": "hybrid",
      "ratel.experiment.role": "serving",
      "ratel.experiment.unit": "0123456789abcdef",
    });
  });

  it("copies a real experiment arm's join to a processor span", async () => {
    const processor = experimentalRatelSpanOutputProcessor();
    let mastraSpan: { attributes?: unknown } | undefined;
    const experiment = experimentalDefineExperiment({
      id: "retrieval-v2",
      arms: {
        hybrid: {
          select: async () => {
            mastraSpan = { attributes: { model: "gpt-5" } };
            processor.process(mastraSpan);
            return { ids: ["deploy"] };
          },
        },
      },
      ranking: (result: { ids: string[] }) => result.ids.map((id) => ({ id })),
      evaluation: { references: ["peer-selection"] },
    });

    const selection = await experiment.select(
      { query: "ship app" },
      { arm: "hybrid", unitId: "user-1" },
    );

    expect(mastraSpan?.attributes).toEqual({
      model: "gpt-5",
      "ratel.experiment.id": "retrieval-v2",
      "ratel.experiment.selection_id": selection.selectionId,
      "ratel.experiment.arm": "hybrid",
      "ratel.experiment.role": "serving",
      "ratel.experiment.unit": "c6c289e49e9c05b2",
    });
  });
});
