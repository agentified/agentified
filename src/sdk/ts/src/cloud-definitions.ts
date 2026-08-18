interface SearchableDefinition {
  readonly id: string;
  readonly description: string;
  readonly searchableDescription?: string;
}

/** @internal Whether two definitions produce the same retrieval-description component. */
export function hasSameRetrievalDescription(
  left: SearchableDefinition,
  right: SearchableDefinition,
): boolean {
  return (
    (left.searchableDescription ?? left.description) ===
    (right.searchableDescription ?? right.description)
  );
}

/** @internal Apply one Cloud retrieval description and diagnose local shadowing. */
export function withCloudDefinition<T extends SearchableDefinition>(
  kind: "tool" | "skill" | "fact",
  definition: T,
  overrides: ReadonlyMap<string, string>,
  warnedIds: Set<string>,
): T {
  const searchableDescription = overrides.get(definition.id);
  if (searchableDescription === undefined) {
    warnedIds.delete(definition.id);
    return definition;
  }
  if (definition.searchableDescription === undefined) {
    warnedIds.delete(definition.id);
  } else if (!warnedIds.has(definition.id)) {
    warnCloudShadow(kind, definition.id);
    warnedIds.add(definition.id);
  }
  return { ...definition, searchableDescription };
}

function warnCloudShadow(kind: "tool" | "skill" | "fact", entryId: string): void {
  try {
    console.warn(
      `ratel: Cloud retrieval description for ${kind} "${entryId}" shadows the local ` +
        "searchableDescription while useCloudDefinitions is enabled",
    );
  } catch {
    // Diagnostics must never break registration or overlay refresh.
  }
}
