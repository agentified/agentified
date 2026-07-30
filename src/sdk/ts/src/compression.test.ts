import { describe, expect, it } from "vitest";
import { ExperimentalCompression } from "./compression.js";

/**
 * A model path that cannot exist, so any attempt to load fails loudly. Using it
 * is how a test proves a code path never reached the model — which is the only
 * way to assert gate-before-load without a 700 MB download in CI.
 */
const UNLOADABLE = { local: "/definitely/not/here/ratel-compression" } as const;

const compressor = (defaults?: ConstructorParameters<typeof ExperimentalCompression>[0]) =>
  new ExperimentalCompression({ model: UNLOADABLE, ...defaults });

/** Long enough to clear the word gate, so the model would be reached. */
const LONG = "word ".repeat(100);

describe("ExperimentalCompression", () => {
  it("returns a short prompt verbatim without loading a model", async () => {
    const input = "translate this to french";
    const result = await compressor().compress(input);

    expect(result.text).toBe(input);
    expect(result.stats.gate).toBe("too_short_words");
    expect(result.stats.modelTokensIn).toBe(0);
    expect(result.kept).toEqual([]);
    expect(result.dropped).toEqual([]);
  });

  it("returns the input verbatim at rate 1 without loading a model", async () => {
    const result = await compressor().compress(LONG, { rate: 1 });
    expect(result.text).toBe(LONG);
    expect(result.stats.gate).toBe("rate_one");
  });

  it("narrows stats.gate to a switchable union", async () => {
    const { stats } = await compressor().compress("too short");
    // Type-level: `gate` is the union, not `string`. Compiles only if narrowed.
    const reason: "too_short_words" | "too_short_tokens" | "rate_one" | undefined = stats.gate;
    expect(reason).toBe("too_short_words");
  });

  it("applies constructor defaults, and lets a call override them", async () => {
    const c = new ExperimentalCompression({
      model: UNLOADABLE,
      defaults: { minWords: 10_000 },
    });
    // The default gates a prompt that would otherwise reach the model.
    expect((await c.compress(LONG)).stats.gate).toBe("too_short_words");
    // A per-call override wins, so the load is attempted and fails.
    await expect(c.compress(LONG, { minWords: 1 })).rejects.toThrow(/compression model/);
  });

  it("rejects a lookbehind pattern rather than silently matching differently", async () => {
    // Valid in JS, unsupported by Rust's `regex` crate. It must be a fast,
    // explicit config error — not a pattern that quietly protects nothing, and
    // not one paid for after a ~700 MB load.
    await expect(compressor().compress(LONG, { protect: [/(?<=a)b/] })).rejects.toThrow(
      /invalid protect pattern/,
    );
  });

  it("passes a RegExp across as its source pattern", async () => {
    // Reaching the model load proves the pattern compiled natively; an invalid
    // one would have thrown a config error first.
    await expect(compressor().compress(LONG, { protect: [/\$[\d,]+/] })).rejects.toThrow(
      /compression model/,
    );
  });

  it("reports a missing local model as a load error naming what is absent", async () => {
    await expect(compressor().compress(LONG)).rejects.toThrow(/model\.safetensors/);
  });

  it("reports an uncached HuggingFace model without touching the network", async () => {
    const c = new ExperimentalCompression({
      model: { huggingface: "ratel-test/definitely-not-a-real-repo" },
    });
    await expect(c.compress(LONG)).rejects.toThrow(/huggingface-cli download/);
  });

  it("rejects a bare repo-id string instead of guessing the source", () => {
    // A bare string is a local directory path; a repo id needs the explicit key.
    expect(() => new ExperimentalCompression({ model: "microsoft/llmlingua-2" })).toThrow(
      /"huggingface"/,
    );
  });

  it("accepts a bare path string as a local model", () => {
    expect(() => new ExperimentalCompression({ model: "/models/llmlingua" })).not.toThrow();
  });

  it("throws for an unparseable model config at construction, not on first use", () => {
    expect(
      () =>
        new ExperimentalCompression({
          // Two primary sources is a configuration error.
          model: { huggingface: "a/b", local: "/m" } as never,
        }),
    ).toThrow(/conflicting compression keys/);
  });
});
