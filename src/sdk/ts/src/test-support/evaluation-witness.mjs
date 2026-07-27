// A module that flags its own evaluation. `isPackageInstalled` must still report it as
// present: the probe resolves, it never imports. If the probe ever starts executing what
// it resolves, this flag is what will catch it.
globalThis.__ratelEvaluationWitness = true;
