---
name: changelog
description: Update per-package CHANGELOG.md files for a Ratel release. Drafts entries with git-cliff (scoped per package), lets you curate, then writes the CHANGELOGs. Handles both RC entries and GA-graduation collapse (merging X.Y.Z-rc.* sections into a single X.Y.Z section). Invoke before tagging a release.
---

# /changelog

Updates a release unit's `CHANGELOG.md` in preparation for tagging it. Ratel releases
**per unit** (ADR-0008): each unit ships on its own tag (`<prefix>-v*`) at its own version.
For workflow-wired units, `.github/workflows/release.yml` rejects a tag whose unit CHANGELOG
doesn't contain the version being released. `vercel-ai-sdk` is temporarily published
manually via `scripts/publish-rc.sh`; still run this skill before its tag even though the
workflow gate does not cover it yet.

Run it **once per unit** you're releasing.

## Release units it touches

The units and their manifests/CHANGELOGs live in one registry —
`scripts/release-units.mjs` — which every release tool reads. Run `node scripts/release-units.mjs --list`
for the authoritative set; the current eight units are:

| Unit | Registry | CHANGELOG path |
|---|---|---|
| `core` | `ratel-ai-core` (crates.io) | `src/core/CHANGELOG.md` |
| `sdk-ts` | `@ratel-ai/sdk` (npm) | `src/sdk/ts/CHANGELOG.md` |
| `sdk-py` | `ratel-ai` (PyPI) | `src/sdk/python/CHANGELOG.md` |
| `telemetry-core` | `ratel-ai-telemetry` (crates.io) | `src/telemetry/core/CHANGELOG.md` |
| `telemetry-ts` | `@ratel-ai/telemetry` (npm) | `src/telemetry/ts/CHANGELOG.md` |
| `telemetry-py` | `ratel-ai-telemetry` (PyPI) | `src/telemetry/python/CHANGELOG.md` |
| `vercel-ai-sdk` | `@ratel-ai/vercel-ai-sdk` (npm) | `src/adapters/ts-vercel-ai-sdk/CHANGELOG.md` |
| `mastra` | `@ratel-ai/mastra` (npm) | `src/adapters/ts-mastra/CHANGELOG.md` |

`@ratel-ai/mcp-server` lives in [ratel-ai/ratel-mcp](https://github.com/ratel-ai/ratel-mcp) and maintains its own CHANGELOG there.

## Procedure

### 1. Pick the unit and read its version

If the user hasn't named the unit, run `node scripts/releasable.mjs` — it lists which
units have commits since their last release tag. Pick the unit `$UNIT` being released.

Read its canonical version:

```bash
node scripts/release-units.mjs --version "$UNIT"   # -> $TARGET
```

If the user supplies a different version explicitly, prefer that and warn them the working
tree disagrees.

### 2. Determine the diff range

The range is from the unit's **own** last release tag to `HEAD`:

```bash
prefix=$(node scripts/release-units.mjs --tag-prefix "$UNIT")
FROM=$(git describe --tags --match "${prefix}*" --abbrev=0 2>/dev/null || true)
```

If `$FROM` is empty the unit has never shipped; the whole history is in range.

### 3. Generate the draft

```bash
bash .claude/skills/changelog/draft.sh --unit "$UNIT"
```

It emits a single `### <package-name>` block for the unit, containing either
Keep-a-Changelog sections (`### Added`, `### Fixed`, `### Changed`) or the sentinel
`_No user-facing changes._`. (Omit `--unit` to draft every unit at once.) With no
`<from-ref>` argument the script ranges each unit from its own last tag automatically.

If `draft.sh` exits 127, git-cliff is missing. Tell the user how to install it (the
script's stderr already does), and stop.

### 4. Branch on RC vs GA

Inspect `$TARGET`, and edit only **this unit's** CHANGELOG:

- **RC** (`X.Y.Z-rc.N`): prepend a new section above the most recent versioned section:
  ```
  ## [X.Y.Z-rc.N] - YYYY-MM-DD

  <draft content for the unit, or the sentinel>
  ```
  Use today's date in `YYYY-MM-DD` (UTC).

- **GA** (no `-rc` suffix): enter **GA-collapse mode**:
  1. Find every `## [X.Y.Z-rc.*]` section already present that matches the same
     `MAJOR.MINOR.PATCH` as `$TARGET`.
  2. Union their bullet entries (per subsection: `### Added`, `### Changed`, `### Fixed`)
     with the new draft entries from step 3 (commits since the last RC tag).
  3. Deduplicate bullets within each subsection (case-insensitive, whitespace-normalised).
  4. Drop the `_No user-facing changes._` sentinel if any real entries exist; keep it only
     if the unioned set is empty.
  5. Replace all the matched RC sections with a single `## [X.Y.Z] - YYYY-MM-DD` section
     containing the merged content.
  6. Leave non-matching prior versions (e.g. `## [0.1.4]`) untouched.

### 5. Curate with the user

Show the unit's CHANGELOG pending changes in the conversation. Ask the user to confirm or
edit. Common curation moves:

- Rephrase bullets for user-facing clarity (the draft uses commit subjects verbatim).
- Drop bullets that are not user-visible (internal refactors that slipped past
  `cliff.toml`'s skip rules).
- Merge duplicates that survived deduplication.
- Promote / demote between Added / Changed / Fixed if the commit prefix was wrong.

### 6. Write the file

Once approved, write **only this unit's** CHANGELOG using the Edit tool. **Do not commit.**
The release commit is the user's responsibility — they typically include the CHANGELOG
alongside the version bump in a single `release: <unit>-vX.Y.Z` commit.

If releasing more than one unit, repeat from step 1 for each.

### 7. Remind

Tell the user:

- The CHANGELOG is staged in the working tree (unstaged).
- Next step is the release commit + `<unit>-v<version>` tag + push.
- For workflow-wired units, the `release.yml` `tag-version-check` job verifies the unit's
  CHANGELOG contains the tag version; if it doesn't, the release is blocked.
- For `vercel-ai-sdk`, publish manually with `scripts/publish-rc.sh` after the tag push; its
  CHANGELOG is not workflow-gated yet.

## Conventions

- **Date format**: `YYYY-MM-DD` in UTC.
- **Subsection order**: `### Added`, `### Changed`, `### Fixed`, `### Removed`, `### Deprecated`, `### Security`. Omit empty subsections.
- **Sentinel**: `_No user-facing changes._` for a unit with no in-scope commits.
- **Keep `## [Unreleased]` at the top** — it stays empty between releases.

## Why this exists

ADR 0008 (`docs/adr/0008-release-engineering.md`) records the per-unit tag/version split, the
CHANGELOG mechanism, and the rationale. Read it if you're unsure why something is structured
the way it is.
