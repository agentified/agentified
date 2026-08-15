# 5. First-class skills: unified capability search, two surfacing paths

Date: 2026-07-05

## Status

Accepted

Compacted 2026-07 from pre-compaction ADR-0012 (first-class skills, 2026-06-09).

## Context

The gateway's wedge is tool selection; skills (SKILL.md playbooks: YAML frontmatter carrying
the data-model fields below, then a Markdown body as the payload) deserve the same treatment. A skill differs from a tool in two ways that shape
the design: it is **read, not executed** (the body is the payload), and it surfaces in two
different situations: while using a tool, and while working from a terse no-tool intent
prompt. Skills are advisory (a task completes without one, just worse), so skill-surfacing
fails silently and must be made measurable. To actually save tokens the gateway must be the
sole loader of the skills it manages: the host must not auto-scan them.

## Decision

### Source

Skills load from a Ratel-managed folder (default `~/.ratel/skills/`) that the host does not
auto-scan. With a remote source configured, the same catalog hydrates from the pull-sync
contract instead ([ADR-0003](0003-catalog-source-interface.md)); either way the gateway is
the only loader.

### Surface: one search call, two reserved buckets

- **`search_capabilities(query, topKTools?, topKSkills?)`** returns
  `{ tools: { groups }, skills: [...] }`: two independently-ranked buckets, each with its own
  top-K budget. Skills can never be starved by tool matches, and BM25 scores are never
  compared across the two text shapes. A matched skill also contributes its declared `tools`
  to the tools bucket, so the agent gets a playbook and its toolkit in one turn.
- **`invoke_tool(toolId, args)`** runs a tool.
- **`get_skill_content(skillId)`** returns `{ body }`. The `ratel()` factory's `modelTools()` always
  advertises it so the exposed set never depends on registration order
  ([ADR-0013](0013-framework-adapter-spi.md)); loading from an empty catalog returns a structured
  error, not a missing tool. The piecemeal builders keep a size-gated default (pinnable via
  `advertiseSkills`).

The agent must search to find a load-bearing tool, so bundling skills into that same response
means the tool's necessity carries the skill: more reliable than a separate, skippable
skill-search tool.

### Skill data model

`Skill` is `{ id, name, description, searchable_description?, tags, tools, metadata, body }`.
Catalog protocol v1 keeps its frozen shape; v2 adds the optional override:

- **`tags`**: indexed. Author labels and task phrases ("frontend", "login form") folded into
  the BM25 text so a terse intent prompt matches.
- **`tools`**: a typed dependency edge, not indexed; surfaced additively in the tools bucket.
- **`metadata`** (`map<string, string[]>`): free-form, not indexed; context for higher layers
  (e.g. `{"stacks": ["react"]}` for the push-path ranker).

Ranking runs over `name`, the effective searchable description, and `tags` per ADR-0021. `body`,
`tools`, and `metadata` are not indexed. All beyond `{id, name, description}` is optional at the
loader boundary.

### Two surfacing mechanisms

- **Pull**: the `search_capabilities` skills bucket, for tool-adjacent work.
- **Push**: a prompt-submit preload hook for no-tool work; experimental and non-load-bearing.
  Methodology: task-phrase tags bridge lexically-sparse prompts; detected project stacks
  *boost* skills whose `metadata["stacks"]` match (context narrows, intent picks); a
  clear-winner gate fires nothing on vague prompts or ties; BM25 plus tags plus boost plus
  gate ships first, semantic layers only if the retrieval benchmark shows BM25 underperforms.

### Telemetry

Tool and skill activity stay distinguishable on the trace stream: `gateway_search` (tool
hits), `skill_search` (the skills bucket), `skill_invoke` (`get_skill_content`); the
offered-to-loaded funnel is derivable ([ADR-0007](0007-telemetry-two-streams.md)).

## Consequences

- One discovery call, with skills guaranteed a reserved slot: starvation is impossible by
  construction.
- Push-path quality is a measurable target (recall@K plus over-fire rate on negative
  prompts); it does not ship blind.
- Rejected: separate `search_tools` / `search_skills` tools (skills become skippable); one
  merged ranked list (starvation plus cross-type score incomparability); a `relatedSkills`
  side-channel on tool results (a redundant third mechanism); folding project-stack terms
  into the push query (drowns weak prompts; the boost replaces it).
