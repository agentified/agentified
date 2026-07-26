/**
 * Whether a package is present in the dependency tree, answered without executing it.
 *
 * The SDK uses this to *describe* the caller's environment, never to load from it:
 * `unadaptedError` names the adapter for whichever framework it finds installed, so a
 * framework-shaped tool registered on the native path gets an error that points at the
 * exact package to install rather than generic advice.
 */

import { createRequire } from "node:module";

/**
 * Whether `specifier` resolves from this module (i.e. is installed). Resolution only —
 * the package is never evaluated, so probing is free of side effects and safe for
 * packages that would throw on import.
 *
 * A specifier that resolves but is quirky (an `exports`-map edge case, say) reads as
 * *installed*: this answers "is it in the tree", and only a module-not-found code is
 * evidence that it isn't.
 */
export function isPackageInstalled(specifier: string): boolean {
  try {
    createRequire(import.meta.url).resolve(specifier);
    return true;
  } catch (err) {
    return !isModuleNotFound(err);
  }
}

/** A resolve/import reports a module-not-found code only when the package is absent. */
function isModuleNotFound(err: unknown): boolean {
  const code = (err as { code?: unknown })?.code;
  return code === "ERR_MODULE_NOT_FOUND" || code === "MODULE_NOT_FOUND";
}
