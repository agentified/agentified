/** Must stay identical to native `REGISTRY_BUSY_MESSAGE`. */
const REGISTRY_BUSY_MESSAGE =
  "registry busy; await the active operation before registering more items";

const artifactOperations = new WeakMap<object, number>();

/** Refuse corpus mutation while an artifact source resolve/warm is in flight. */
export function assertNotArtifactBusy(registry: object): void {
  if ((artifactOperations.get(registry) ?? 0) > 0) {
    throw new Error(REGISTRY_BUSY_MESSAGE);
  }
}

/**
 * Warm from an artifact whose bytes come from `resolve`. The corpus is closed
 * to mutation for the whole call — including the asynchronous resolve.
 */
export async function warmFromEmbeddingArtifactSource(
  registry: {
    experimentalWarmEmbeddingsFromArtifact(bytes: Buffer, onMiss: "error" | "embed"): Promise<void>;
  },
  resolve: () => Promise<{ bytes: Buffer; onMiss: "error" | "embed" }>,
): Promise<void> {
  artifactOperations.set(registry, (artifactOperations.get(registry) ?? 0) + 1);
  try {
    const { bytes, onMiss } = await resolve();
    await registry.experimentalWarmEmbeddingsFromArtifact(bytes, onMiss);
  } finally {
    artifactOperations.set(registry, (artifactOperations.get(registry) ?? 1) - 1);
  }
}
