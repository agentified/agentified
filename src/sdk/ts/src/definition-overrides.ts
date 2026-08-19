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

/** @internal Apply one override retrieval description and diagnose local shadowing. */
export function withDefinitionOverride<T extends SearchableDefinition>(
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
    warnOverrideShadow(kind, definition.id);
    warnedIds.add(definition.id);
  }
  return { ...definition, searchableDescription };
}

function warnOverrideShadow(kind: "tool" | "skill" | "fact", entryId: string): void {
  try {
    console.warn(
      `ratel: definition override for ${kind} "${entryId}" shadows the local ` +
        "searchableDescription while useDefinitionOverrides is enabled",
    );
  } catch {
    // Diagnostics must never break registration or overlay refresh.
  }
}
