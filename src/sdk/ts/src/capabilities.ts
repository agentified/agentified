import { isAsyncIterable, isPromiseLike } from "./async.js";
import type {
  ExecutableTool,
  InputValidationResult,
  SearchOrigin,
  ToolCatalog,
} from "./catalog.js";
import { compactDescription } from "./compact.js";
import type { SkillCatalog } from "./skill-catalog.js";
import { recordAuthNeeded, upstreamFromToolId } from "./telemetry.js";

/**
 * Wire id (`"search_capabilities"`) of the discovery capability tool built by
 * {@link searchCapabilitiesTool} — the name the model calls it by.
 */
export const SEARCH_CAPABILITIES_ID = "search_capabilities" as const;
/**
 * Wire id (`"invoke_tool"`) of the execution capability tool built by
 * {@link invokeToolTool} — the name the model calls it by.
 */
export const INVOKE_TOOL_ID = "invoke_tool" as const;

/** Default `tools` bucket size, and the fallback for an invalid host `search` top-K. */
export const DEFAULT_TOP_K_TOOLS = 5;
const DEFAULT_TOP_K_SKILLS = 3;
const MAX_TOP_K = 50;

/**
 * Clamp a model-supplied top-K to a positive integer in [1, MAX_TOP_K], falling
 * back to `fallback` for anything else (undefined, 0, negative, non-integer,
 * NaN). Tools and skills — the capability funnel, the host-driven
 * {@link ToolCollection.search}, and the TS and Python SDKs — run the same input
 * through this, so a stray `topK` can't silently return zero results (or, via a
 * negative wrapping to u32 in the native layer, an unbounded set).
 */
export function clampTopK(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) return fallback;
  return Math.min(value, MAX_TOP_K);
}

// The discovery prompt the model sees. By default the skills clause is only
// included when a non-empty skill catalog is wired in — otherwise the tool would
// advertise a `skills` bucket and `get_skill_content` that don't exist. Hosts
// that always expose `get_skill_content` pin the clause via `advertiseSkills`.
const SEARCH_INTRO =
  "Discover capabilities beyond the ones already in your direct tool list. Call this BEFORE refusing " +
  "a request, falling back to a generic capability (web fetch, shell, built-in search), or improvising " +
  "a multi-step task: a purpose-built capability may be in the catalog but not pre-loaded. Pass a " +
  "natural-language query describing what you want to do.";
const RESULT_TOOLS_ONLY = " You get back a `tools` bucket (executable) — run one via invoke_tool.";
const RESULT_TOOLS_AND_SKILLS =
  " You get back two independent buckets: `tools` (run one via invoke_tool) and `skills` (reusable " +
  "playbooks — load one's instructions via get_skill_content, then follow it). Skills have their own " +
  "result budget, so they are never crowded out by tools.";

/**
 * Descriptive metadata about one upstream MCP server behind the catalog. Fed
 * to {@link searchCapabilitiesTool} via {@link SearchCapabilitiesOptions}: the
 * servers are listed in the tool's description (via
 * {@link formatUpstreamLine}), and `description`/`instructions` enrich the
 * matching {@link CapabilityToolGroup.server} in results.
 */
export interface UpstreamServerInfo {
  /**
   * Server name; must match the `name` the server was registered under
   * (`registerMcpServer`), i.e. the `<name>__` prefix of its tool ids.
   */
  name: string;
  /** One-line summary of what the server offers; compacted in the listing. */
  description?: string;
  /** The server's own usage instructions (e.g. `McpServerHandle.serverInstructions`). */
  instructions?: string;
  /** Number of tools the server contributes; shown in the listing when set. */
  toolCount?: number;
  /** True when the upstream rejected its boot connection with 401 / requires re-authorization. */
  needsAuth?: boolean;
}

/** Options for {@link searchCapabilitiesTool}. */
export interface SearchCapabilitiesOptions {
  /** Upstream MCP servers to advertise in the tool description and result groups. */
  upstreamServers?: readonly UpstreamServerInfo[];
  /**
   * Override the skills clause in the tool description. Default: present only
   * when the skill catalog is non-empty at build time. Hosts that always expose
   * `get_skill_content` (the `ratel()` facade) pass `true` so the description is
   * byte-identical regardless of when the first skill registers.
   */
  advertiseSkills?: boolean;
}

/** One ranked tool in the `tools` bucket of {@link SearchCapabilitiesResult}. */
export interface CapabilityToolHit {
  /** Catalog id to pass to `invoke_tool` (`<server>__<tool>` for MCP-proxied tools). */
  toolId: string;
  /**
   * Retrieval score from the tool catalog's search (scale depends on the
   * catalog's method — BM25 by default), or `0` when the tool was pulled in as
   * a matched skill's declared dependency rather than by the query itself.
   */
  score: number;
  /** The tool's description, as registered. */
  description: string;
  /** The tool's input JSON Schema, so the model can call it without another lookup. */
  inputSchema: Record<string, unknown>;
}

/**
 * Tool hits grouped by upstream server. The group key is the `<server>__` id
 * prefix (a plain, un-prefixed tool id groups under itself); `description` and
 * `instructions` are attached when a matching {@link UpstreamServerInfo} was
 * provided.
 */
export interface CapabilityToolGroup {
  /** The owning server: its name, plus optional description/instructions metadata. */
  server: {
    /** Server name (the `<server>__` prefix of the group's tool ids). */
    name: string;
    /** The server's one-line summary, when known. */
    description?: string;
    /** The server's usage instructions, when known. */
    instructions?: string;
  };
  /** The server's ranked hits, in overall result order. */
  hits: CapabilityToolHit[];
}

/** One ranked skill in the `skills` bucket of {@link SearchCapabilitiesResult}. */
export interface CapabilitySkillHit {
  /** Skill id to pass to `get_skill_content`. */
  skillId: string;
  /** Retrieval score from the skill catalog's search (BM25 by default). */
  score: number;
  /** The skill's description, compacted to ~160 chars for the listing. */
  description: string;
}

/**
 * Result shape of the `search_capabilities` tool: two independently-ranked
 * buckets with separate top-K budgets. Scores are comparable within a bucket,
 * not across the two (tools and skills are indexed as different text shapes).
 */
export interface SearchCapabilitiesResult {
  /** Executable tool hits, grouped by upstream server. */
  tools: {
    /** Groups in ranking order (a server appears where its best hit ranked). */
    groups: CapabilityToolGroup[];
  };
  /** Skill hits — playbooks to load via `get_skill_content`. */
  skills: CapabilitySkillHit[];
}

/**
 * Format one upstream server as the bullet line the capability-tool
 * descriptions embed (`- <name> — <description> (<n> tools) (auth required)`,
 * omitting the parts that are unset). Exported for the deprecated
 * `searchToolsTool` shim, which builds the same listing.
 *
 * @param s - The server to describe.
 * @returns A single `- `-prefixed line with the description compacted.
 */
export function formatUpstreamLine(s: UpstreamServerInfo): string {
  let line = `- ${s.name}`;
  if (s.description) line += ` — ${compactDescription(s.description)}`;
  if (typeof s.toolCount === "number") line += ` (${s.toolCount} tools)`;
  if (s.needsAuth) line += " (auth required)";
  return line;
}

function buildSearchDescription(hasSkills: boolean, opts?: SearchCapabilitiesOptions): string {
  const base = SEARCH_INTRO + (hasSkills ? RESULT_TOOLS_AND_SKILLS : RESULT_TOOLS_ONLY);
  const upstreams = opts?.upstreamServers ?? [];
  if (upstreams.length === 0) return base;
  const list = upstreams.map(formatUpstreamLine).join("\n");
  return `${base}\n\nThis catalog aggregates tools from these upstream MCP servers:\n${list}`;
}

/**
 * Build the `search_capabilities` capability tool: unified discovery over
 * tools AND skills. Its result carries two independently-ranked buckets, each
 * with its own top-K budget — so a relevant skill can never be starved out of
 * the results by a large number of matching tools (and we avoid comparing BM25
 * scores across the two different text shapes).
 *
 * The tool takes `{ query, topKTools?, topKSkills? }` (defaults 5 and 3;
 * values above 50 are capped, anything else non-positive or non-integer falls
 * back to the default) and resolves to a {@link SearchCapabilitiesResult}. A
 * matched skill's declared tool dependencies are pulled into the `tools`
 * bucket additively — score `0`, beyond the `topKTools` budget, deduped
 * against query hits. The skills clause of the tool description appears when
 * `skillCatalog` is non-empty at build time (override with
 * {@link SearchCapabilitiesOptions.advertiseSkills}); the result's `skills`
 * bucket always ranks the live catalog. Each call records a `gateway_search`
 * event on the local trace stream.
 *
 * @param toolCatalog - Catalog the `tools` bucket is ranked from.
 * @param skillCatalog - Optional catalog the `skills` bucket is ranked from.
 * @param opts - Upstream-server metadata for the description and result groups.
 * @returns The tool, ready to expose to the model (and to register alongside
 *   {@link invokeToolTool}, which executes what this discovers).
 *
 * @example
 * ```ts
 * import { searchCapabilitiesTool, type SearchCapabilitiesResult } from "@ratel-ai/sdk";
 *
 * const discovery = searchCapabilitiesTool(toolCatalog, skillCatalog, {
 *   upstreamServers: [{ name: "github", description: "GitHub API", toolCount: 30 }],
 * });
 * const result = (await discovery.execute({
 *   query: "open a pull request",
 * })) as SearchCapabilitiesResult;
 * ```
 */
export function searchCapabilitiesTool(
  toolCatalog: ToolCatalog,
  skillCatalog?: SkillCatalog,
  opts?: SearchCapabilitiesOptions,
): ExecutableTool {
  const upstreams = opts?.upstreamServers ?? [];
  const hasSkills =
    opts?.advertiseSkills ?? (skillCatalog !== undefined && skillCatalog.size() > 0);
  return {
    id: SEARCH_CAPABILITIES_ID,
    name: SEARCH_CAPABILITIES_ID,
    description: buildSearchDescription(hasSkills, opts),
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "describe what you want to do" },
        topKTools: { type: "integer", minimum: 1, description: "max tools to return (default 5)" },
        topKSkills: {
          type: "integer",
          minimum: 1,
          description: "max skills to return (default 3)",
        },
      },
      required: ["query"],
    },
    outputSchema: {
      type: "object",
      properties: {
        tools: {
          type: "object",
          properties: {
            groups: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  server: {
                    type: "object",
                    properties: {
                      name: { type: "string" },
                      description: { type: "string" },
                      instructions: { type: "string" },
                    },
                    required: ["name"],
                  },
                  hits: {
                    type: "array",
                    items: {
                      type: "object",
                      properties: {
                        toolId: { type: "string" },
                        score: { type: "number" },
                        description: { type: "string" },
                        inputSchema: { type: "object" },
                      },
                    },
                  },
                },
                required: ["server", "hits"],
              },
            },
          },
          required: ["groups"],
        },
        skills: {
          type: "array",
          items: {
            type: "object",
            properties: {
              skillId: { type: "string" },
              score: { type: "number" },
              description: { type: "string" },
            },
            required: ["skillId", "score", "description"],
          },
        },
      },
      required: ["tools", "skills"],
    },
    execute: async (input) => {
      const { query, topKTools, topKSkills } = input as {
        query: string;
        topKTools?: number;
        topKSkills?: number;
      };
      return runCapabilitiesSearch(toolCatalog, query, {
        topKTools,
        topKSkills,
        skillCatalog,
        origin: "agent",
        upstreamServers: upstreams,
      });
    },
  };
}

/** Options for {@link runCapabilitiesSearch}. */
export interface CapabilitiesSearchOptions {
  /** Max tools bucket size; capped at 50, default 5 (same as the tool); 0, negative, or non-integer values fall back to the default. */
  topKTools?: number;
  /** Max skills bucket size; capped at 50, default 3; invalid values fall back to the default. */
  topKSkills?: number;
  /** Catalog the `skills` bucket is ranked from (and whose declared tool deps ride in). */
  skillCatalog?: SkillCatalog;
  /**
   * Who initiated the search — stamped on the `gateway_search` trace event and
   * the `ratel.search` span. Default `"agent"` (a model-synthesized call); the
   * host-driven recall path passes `"direct"`.
   */
  origin?: SearchOrigin;
  /** Upstream-server metadata attached to the matching result groups. */
  upstreamServers?: readonly UpstreamServerInfo[];
}

/**
 * Rank a query into the `search_capabilities` result shape — the single source
 * of truth for that shape, shared by {@link searchCapabilitiesTool} (agent
 * origin) and both `ratel()` recall paths, standalone core and adapted views
 * (direct origin), so they can never drift. Ranks the `tools` bucket (grouped by
 * upstream server, a matched skill's declared tool deps pulled in additively at
 * score `0`, deduped) and the independently-budgeted `skills` bucket, caps both
 * top-Ks at 50 (invalid values fall back to their defaults), and records one
 * `gateway_search` event with the given `origin`.
 *
 * @param toolCatalog - Catalog the `tools` bucket is ranked from.
 * @param query - Natural-language description of what the caller wants to do.
 * @param opts - Bucket sizes, skill catalog, origin, and upstream metadata.
 * @returns A promise for the {@link SearchCapabilitiesResult}.
 */
export async function runCapabilitiesSearch(
  toolCatalog: ToolCatalog,
  query: string,
  opts: CapabilitiesSearchOptions = {},
): Promise<SearchCapabilitiesResult> {
  const kTools = clampTopK(opts.topKTools, DEFAULT_TOP_K_TOOLS);
  const kSkills = clampTopK(opts.topKSkills, DEFAULT_TOP_K_SKILLS);
  const origin = opts.origin ?? "agent";
  const upstreamByName = new Map((opts.upstreamServers ?? []).map((u) => [u.name, u]));
  const skillCatalog = opts.skillCatalog;
  const startedAt = Date.now();

  const toolHits = await toolCatalog.searchAsync(query, kTools, origin);
  toolCatalog.recordEvent({
    type: "gateway_search",
    query,
    origin,
    top_k: kTools,
    hits: toolHits.length,
    took_ms: Date.now() - startedAt,
  });

  const order: string[] = [];
  const groups = new Map<string, CapabilityToolGroup>();
  const seenTools = new Set<string>();
  // Add a tool to its server group, deduped. `score` is the BM25 query score
  // for a real match, or 0 for a skill-declared dependency (it rode in on the
  // skill, it was never matched by the query).
  const addTool = (toolId: string, score: number): void => {
    if (seenTools.has(toolId)) return;
    const tool = toolCatalog.get(toolId);
    if (!tool) return; // a declared id the catalog doesn't have: skip
    seenTools.add(toolId);
    const sep = toolId.indexOf("__");
    const serverName = sep > 0 ? toolId.slice(0, sep) : toolId;
    let group = groups.get(serverName);
    if (!group) {
      const meta = upstreamByName.get(serverName);
      group = {
        server: {
          name: serverName,
          ...(meta?.description ? { description: meta.description } : {}),
          ...(meta?.instructions ? { instructions: meta.instructions } : {}),
        },
        hits: [],
      };
      groups.set(serverName, group);
      order.push(serverName);
    }
    group.hits.push({
      toolId,
      score,
      description: tool.description ?? "",
      inputSchema: (tool.inputSchema ?? {}) as Record<string, unknown>,
    });
  };
  for (const h of toolHits) addTool(h.toolId, h.score);

  // Skills are ranked in their own bucket against the same query (reserved
  // budget → never starved by tools). SkillCatalog.search emits its own
  // skill_search trace for the funnel.
  const skills: CapabilitySkillHit[] = skillCatalog
    ? (await skillCatalog.searchAsync(query, kSkills, origin)).map((h) => ({
        skillId: h.skillId,
        score: h.score,
        description: compactDescription(skillCatalog.get(h.skillId)?.description ?? ""),
      }))
    : [];

  // A matched skill's instructions name the tools they call. Pull those into
  // the tools bucket so the agent gets the playbook and its toolkit in one
  // turn — additively (score 0), beyond topKTools, deduped against query hits.
  if (skillCatalog) {
    for (const s of skills) {
      for (const toolId of skillCatalog.get(s.skillId)?.tools ?? []) {
        addTool(toolId, 0);
      }
    }
  }

  return {
    // biome-ignore lint/style/noNonNullAssertion: order entries are guaranteed by construction
    tools: { groups: order.map((n) => groups.get(n)!) },
    skills,
  };
}

/** Options for {@link invokeToolTool}. */
export interface InvokeToolToolOptions {
  /** Notified when the underlying tool throws UnauthorizedError, with the upstream name inferred from the toolId. */
  onUnauthorized?: (upstream: string) => void | Promise<void>;
}

/** Non-enumerable cause carried by a structured error from a target executor. */
export const INVOKE_TOOL_ERROR_CAUSE: unique symbol = Symbol.for(
  "@ratel-ai/sdk.invoke-tool-error-cause",
) as never;

/** A generic `invoke_tool` error whose original target failure is available to framework adapters. */
export interface InvokeToolError {
  /** Human-readable failure passed to the model. */
  error: string;
  /** Stable discriminator used by generic capability hosts. */
  isError: true;
  /** Original thrown value, hidden from enumeration and serialization. */
  readonly [INVOKE_TOOL_ERROR_CAUSE]: unknown;
}

/** Whether `value` is a structured error produced from a target executor failure. */
export function isInvokeToolError(value: unknown): value is InvokeToolError {
  return (
    value !== null && typeof value === "object" && Object.hasOwn(value, INVOKE_TOOL_ERROR_CAUSE)
  );
}

/**
 * Build the `invoke_tool` capability tool: the execution counterpart to
 * {@link searchCapabilitiesTool}. Takes `{ toolId, args }`, delegates input
 * parsing to the selected catalog tool's live validator, then runs the
 * prevalidated input through {@link ToolCatalog.invokeValidatedRaw}. That
 * preserves a target `AsyncIterable` so framework adapters can expose
 * preliminary outputs even when validation itself was asynchronous.
 *
 * Failures come back as structured `{ error, isError: true }` results, not
 * rejections, so the model can read and recover from them: an unknown
 * `toolId` (with a hint to search first), a non-object `args`, or a thrown
 * executor error. Target executor errors also carry their original thrown
 * value under the non-enumerable {@link INVOKE_TOOL_ERROR_CAUSE} marker so an
 * adapter can restore its framework's native error lifecycle. A tool that
 * throws `UnauthorizedError` gets special
 * handling — `opts.onUnauthorized` fires with the upstream name inferred from
 * the `<server>__` id prefix, a `ratel.auth.flow` span records the outcome,
 * and the result is `{ error: "needs_auth", isError: true, upstream?, hint }`.
 * A call with `args` missing (or `null`) is tolerated by treating the
 * remaining top-level keys as the arguments. The capability executor's optional
 * opaque context is forwarded unchanged to the selected catalog executor; the
 * core never reads or records it. Outcomes are recorded as
 * `gateway_invoke` / `gateway_error` events on the local trace stream.
 *
 * @param catalog - Catalog whose tools this executes.
 * @param opts - Optional auth-failure callback.
 * @returns The tool, ready to expose to the model.
 */
export function invokeToolTool(
  catalog: ToolCatalog,
  opts: InvokeToolToolOptions = {},
): ExecutableTool {
  const prevalidatedInputs = new WeakSet<object>();
  return {
    id: INVOKE_TOOL_ID,
    name: INVOKE_TOOL_ID,
    description:
      "Invoke a tool from the catalog by its id. Use this to call tools that aren't in your direct tool list — " +
      "first find one via the catalog's search tool, then run it here. " +
      "Pass the tool's arguments nested under the `args` field — do NOT flatten them to the top level.",
    inputSchema: {
      type: "object",
      properties: {
        toolId: {
          type: "string",
          description:
            "id of the tool to invoke (use the catalog's search tool to find available ids)",
        },
        args: {
          type: "object",
          description:
            "arguments object matching the tool's inputSchema, nested as a single object",
          additionalProperties: true,
        },
      },
      required: ["toolId", "args"],
    },
    outputSchema: { type: "object" },
    validateInput: (input) => validateInvokeInput(catalog, input, prevalidatedInputs),
    execute: (input, context) => {
      const inputObj = input as Record<string, unknown>;
      const toolId = inputObj.toolId as string;
      if (!catalog.has(toolId)) {
        catalog.recordEvent({
          type: "gateway_error",
          tool_id: toolId,
          error: "unknown_tool_id",
        });
        return {
          error: `unknown toolId: ${toolId}. Use the catalog's search tool to discover available ids.`,
          isError: true,
        };
      }
      const prevalidated = prevalidatedInputs.delete(inputObj);
      const nested = inputObj.args;
      let args: unknown;
      if (prevalidated) {
        args = nested;
      } else if (nested === undefined || nested === null) {
        // No `args` given — tolerate a flattened call by treating the remaining
        // top-level keys as the arguments. Drop `args` too, so an explicit
        // `args: null` can't forward a stray `args` key to the tool.
        args = Object.fromEntries(
          Object.entries(inputObj).filter(([k]) => k !== "toolId" && k !== "args"),
        );
      } else if (typeof nested === "object" && !Array.isArray(nested)) {
        args = nested as Record<string, unknown>;
      } else {
        // `args` is present but not an object (string/array/number) — reject
        // rather than silently forwarding stray top-level keys as arguments.
        return {
          error: `invalid args for ${toolId}: \`args\` must be an object containing the tool's arguments.`,
          isError: true,
        };
      }
      const startedAt = Date.now();
      try {
        return observeGatewayResult(
          prevalidated
            ? catalog.invokeValidatedRaw(toolId, args, context)
            : catalog.invokeRaw(toolId, args as Record<string, unknown>, context),
          () => {
            catalog.recordEvent({
              type: "gateway_invoke",
              tool_id: toolId,
              took_ms: Date.now() - startedAt,
            });
          },
          (error) => handleInvokeError(error, toolId, startedAt, catalog, opts),
        );
      } catch (err) {
        return handleInvokeError(err, toolId, startedAt, catalog, opts);
      }
    },
  };
}

function validateInvokeInput(
  catalog: ToolCatalog,
  input: unknown,
  prevalidatedInputs: WeakSet<object>,
): InputValidationResult | PromiseLike<InputValidationResult> {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    return { success: true, value: input };
  }
  const inputObj = input as Record<string, unknown>;
  if (typeof inputObj.toolId !== "string") return { success: true, value: input };

  const nested = inputObj.args;
  let args: Record<string, unknown>;
  if (nested === undefined || nested === null) {
    args = Object.fromEntries(
      Object.entries(inputObj).filter(([key]) => key !== "toolId" && key !== "args"),
    );
  } else if (typeof nested === "object" && !Array.isArray(nested)) {
    args = nested as Record<string, unknown>;
  } else {
    return { success: true, value: input };
  }

  return mapValidationValue(catalog.validateInput(inputObj.toolId, args), (validated) => {
    const transformed = { ...inputObj, args: validated };
    prevalidatedInputs.add(transformed);
    return transformed;
  });
}

/** Map the success value of a (possibly async) validation result; a failure passes through unchanged. */
function mapValidationValue(
  result: InputValidationResult | PromiseLike<InputValidationResult>,
  onSuccess: (value: unknown) => unknown,
): InputValidationResult | PromiseLike<InputValidationResult> {
  const settle = (validated: InputValidationResult): InputValidationResult =>
    validated.success ? { success: true, value: onSuccess(validated.value) } : validated;
  return isPromiseLike(result) ? Promise.resolve(result).then(settle) : settle(result);
}

function observeGatewayResult(
  result: unknown,
  onSuccess: () => void,
  onError: (error: unknown) => unknown,
): unknown {
  if (isAsyncIterable(result)) {
    return observeGatewayIterable(result, onSuccess, onError);
  }
  if (isPromiseLike(result)) {
    return Promise.resolve(result).then(
      (value) => {
        onSuccess();
        return value;
      },
      (error) => onError(error),
    );
  }
  onSuccess();
  return result;
}

async function* observeGatewayIterable(
  iterable: AsyncIterable<unknown>,
  onSuccess: () => void,
  onError: (error: unknown) => unknown,
): AsyncGenerator<unknown> {
  let failed = false;
  try {
    for await (const value of iterable) yield value;
  } catch (error) {
    failed = true;
    yield await onError(error);
  } finally {
    if (!failed) onSuccess();
  }
}

function handleInvokeError(
  error: unknown,
  toolId: string,
  startedAt: number,
  catalog: ToolCatalog,
  opts: InvokeToolToolOptions,
): unknown {
  if (isUnauthorizedError(error)) {
    const upstream = upstreamFromToolId(toolId);
    const finish = () => {
      recordAuthNeeded(upstream);
      catalog.recordEvent({
        type: "gateway_error",
        tool_id: toolId,
        error: "needs_auth",
      });
      const payload: { error: string; isError: true; upstream?: string; hint: string } = {
        error: "needs_auth",
        isError: true,
        hint: `call the auth tool to re-authorize${upstream ? ` ${upstream}` : ""}`,
      };
      if (upstream) payload.upstream = upstream;
      return payload;
    };
    if (upstream && opts.onUnauthorized) {
      return Promise.resolve(opts.onUnauthorized(upstream)).then(finish);
    }
    return finish();
  }

  const message = error instanceof Error ? error.message : String(error);
  catalog.recordEvent({
    type: "gateway_error",
    tool_id: toolId,
    error: message,
    took_ms: Date.now() - startedAt,
  });
  return brandedInvokeError(`tool ${toolId} threw: ${message}`, error);
}

function brandedInvokeError(message: string, cause: unknown): InvokeToolError {
  const payload = { error: message, isError: true as const } as InvokeToolError;
  Object.defineProperty(payload, INVOKE_TOOL_ERROR_CAUSE, {
    value: cause,
    enumerable: false,
    configurable: false,
    writable: false,
  });
  return payload;
}

function isUnauthorizedError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return err.name === "UnauthorizedError";
}
