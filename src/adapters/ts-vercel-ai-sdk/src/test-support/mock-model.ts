// A local mock provider rather than `ai/test`'s: upstream renames the class per
// major (`MockLanguageModelV2` on ai@5, `MockLanguageModelV3` on ai@6/7), so
// importing it would break the ai@5–7 compat matrix these suites run under.
// One local name resolves on every row. Never shipped — this dir is
// build-excluded.

/** What the AI SDK hands a provider call; only `prompt` is ever asserted on. */
export interface ModelCall {
  prompt: unknown[];
}

/** Token counts the AI SDK requires on every model reply. */
export const usage = { inputTokens: 1, outputTokens: 1, totalTokens: 2 };

/**
 * The `ai` major this run verifies, set per row by the compat matrix. Suites
 * skip on this row identity, never on feature detection: a self-fulfilling
 * `skipIf(!featureFound)` would quietly evaporate the day `ai` moves the seam
 * under test, and report green.
 */
export const aiSdkMajor = Number.parseInt(process.env.AI_SDK_VERSION ?? "7", 10);

/** Replays canned provider results in call order, recording what it was called with. */
export class MockLanguageModelV2 {
  readonly specificationVersion = "v2";
  readonly provider = "mock-provider";
  readonly modelId = "mock-model";
  readonly supportedUrls = {};
  readonly doGenerateCalls: ModelCall[] = [];
  readonly doStreamCalls: ModelCall[] = [];

  constructor(
    private readonly generateResults: unknown[] = [],
    private readonly streamResults: unknown[] = [],
  ) {}

  async doGenerate(options: ModelCall): Promise<unknown> {
    const index = this.doGenerateCalls.push(options) - 1;
    return this.generateResults[index];
  }

  async doStream(options: ModelCall): Promise<unknown> {
    const index = this.doStreamCalls.push(options) - 1;
    return this.streamResults[index];
  }
}
