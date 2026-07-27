// One-time "this API is experimental" nudge for the facts / grounding surface.
// Accesses `process`/`console` through `globalThis` so the module needs no
// `@types/node` and runs in any host (Node, workers, bundlers).

const globalRef = globalThis as {
  console?: { warn?: (message: string) => void };
  process?: { env?: Record<string, string | undefined> };
};

let warned = false;

/**
 * Warn once per process that the facts / grounding API is experimental — unless
 * `RATEL_EXPERIMENTAL_SILENCE` is set. Called from {@link FactCatalog}'s
 * constructor so any entry into the feature trips it exactly once.
 */
export function warnExperimentalFactsOnce(): void {
  if (warned || globalRef.process?.env?.RATEL_EXPERIMENTAL_SILENCE) return;
  warned = true;
  globalRef.console?.warn?.(
    "ratel: the facts / grounding API is experimental and may change without a major-version " +
      'bump — import it from the `experimental` namespace (`import { experimental } from "@ratel-ai/sdk"`). ' +
      "Set RATEL_EXPERIMENTAL_SILENCE=1 to silence this warning.",
  );
}

/**
 * Reset the one-time guard. Test-only — lets a test assert the warning fires
 * without another test having already tripped the process-wide flag.
 */
export function resetExperimentalWarningForTest(): void {
  warned = false;
}
