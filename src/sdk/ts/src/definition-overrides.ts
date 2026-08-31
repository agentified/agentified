import { DefinitionOverlayError } from "./errors.js";
import type {
  ExperimentalDefinitionOverlayResponse,
  ExperimentalDefinitionOverride,
} from "./runtime-events.js";

const DEFINITION_OVERRIDE_KINDS = new Set(["tool", "skill", "fact"]);
const MAX_DEFINITION_OVERRIDE_ENTRY_ID_BYTES = 512;
const MAX_DEFINITION_OVERRIDE_SEARCHABLE_DESCRIPTION_BYTES = 16_384;

/** @internal Controls provisional cross-catalog override application. */
export interface DefinitionOverrideApplyOptions {
  readonly adopt?: boolean;
  readonly emitDefinitions?: boolean;
}

interface SearchableDefinition {
  readonly id: string;
  readonly description: string;
  readonly experimentalSearchableDescription?: string;
}

/** Validate and snapshot one untrusted source response. */
export function validateDefinitionOverlayResponse(
  value: unknown,
): ExperimentalDefinitionOverlayResponse {
  if (!isRecord(value) || (value.status !== 200 && value.status !== 304)) {
    throw new DefinitionOverlayError(
      "definition overlay response status must be 200 or 304",
      "invalid_status",
    );
  }
  if (value.status === 304) return { status: 304 };
  if (!isRecord(value.body)) {
    throw new DefinitionOverlayError(
      "definition overlay response body must be an object",
      "invalid_payload",
    );
  }
  const overrides = validateDefinitionOverrides(value.body.overrides);
  if (typeof value.etag !== "string" || value.etag.trim().length === 0) {
    throw new DefinitionOverlayError(
      "definition overlay response etag must be a non-empty string",
      "invalid_etag",
    );
  }
  return { status: 200, etag: value.etag, body: { overrides } };
}

/** Validate and snapshot a complete untrusted override set before catalog mutation. */
export function validateDefinitionOverrides(value: unknown): ExperimentalDefinitionOverride[] {
  if (!Array.isArray(value)) {
    throw new DefinitionOverlayError(
      "definition overlay overrides must be an array",
      "invalid_payload",
    );
  }
  return value.map((candidate, index) => {
    if (!isRecord(candidate)) {
      throw invalidOverride(index, "must be an object");
    }
    if (!DEFINITION_OVERRIDE_KINDS.has(candidate.kind as string)) {
      throw invalidOverride(index, "kind must be tool, skill, or fact");
    }
    if (typeof candidate.entryId !== "string") {
      throw invalidOverride(index, "entryId must be a string");
    }
    if (utf8Length(candidate.entryId) > MAX_DEFINITION_OVERRIDE_ENTRY_ID_BYTES) {
      throw invalidOverride(
        index,
        `entryId exceeds ${MAX_DEFINITION_OVERRIDE_ENTRY_ID_BYTES} UTF-8 bytes`,
      );
    }
    if (typeof candidate.searchableDescription !== "string") {
      throw invalidOverride(index, "searchableDescription must be a string");
    }
    if (
      utf8Length(candidate.searchableDescription) >
      MAX_DEFINITION_OVERRIDE_SEARCHABLE_DESCRIPTION_BYTES
    ) {
      throw invalidOverride(
        index,
        `searchableDescription exceeds ${MAX_DEFINITION_OVERRIDE_SEARCHABLE_DESCRIPTION_BYTES} UTF-8 bytes`,
      );
    }
    return {
      kind: candidate.kind as ExperimentalDefinitionOverride["kind"],
      entryId: candidate.entryId,
      searchableDescription: candidate.searchableDescription,
    };
  });
}

/** @internal Whether two definitions produce the same retrieval-description component. */
export function hasSameRetrievalDescription(
  left: SearchableDefinition,
  right: SearchableDefinition,
): boolean {
  return (
    (left.experimentalSearchableDescription ?? left.description) ===
    (right.experimentalSearchableDescription ?? right.description)
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
  if (definition.experimentalSearchableDescription === undefined) {
    warnedIds.delete(definition.id);
  } else if (!warnedIds.has(definition.id)) {
    warnOverrideShadow(kind, definition.id);
    warnedIds.add(definition.id);
  }
  return { ...definition, experimentalSearchableDescription: searchableDescription };
}

function warnOverrideShadow(kind: "tool" | "skill" | "fact", entryId: string): void {
  try {
    console.warn(
      `ratel: definition override for ${kind} "${entryId}" shadows the local ` +
        "experimentalSearchableDescription while definition overrides are enabled",
    );
  } catch {
    // Diagnostics must never break registration or overlay refresh.
  }
}

function invalidOverride(index: number, message: string): DefinitionOverlayError {
  return new DefinitionOverlayError(
    `definition overlay override ${index} ${message}`,
    "invalid_payload",
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function utf8Length(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}
