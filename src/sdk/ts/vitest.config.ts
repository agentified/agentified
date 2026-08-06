import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Nearly every test here is model-free and finishes in milliseconds. The
    // exceptions are the ones that embed — building an intent graph, or a
    // semantic/hybrid search — because the first of them to run downloads
    // `bge-small-en-v1.5` (~130MB) into the shared HuggingFace cache.
    //
    // On a developer machine that cache is warm and the default 5s is plenty.
    // On a cold CI runner the download alone exceeds it, so whichever test
    // happens to be first fails while every later one passes — which reads as a
    // flaky test rather than a missing cache.
    testTimeout: 60_000,
  },
});
