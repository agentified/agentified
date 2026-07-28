import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { Tool } from "@modelcontextprotocol/sdk/types.js";

import type { ExecutableTool, ToolCatalog } from "./catalog.js";
import { traceUpstreamRegister } from "./telemetry.js";

const MCP_LIST_MAX_PAGES = 64;

export type McpToolsListErrorCode = "RepeatedCursor" | "PaginationExceeded";

export class McpToolsListError extends Error {
  readonly code: McpToolsListErrorCode;

  constructor(message: string, code: McpToolsListErrorCode) {
    super(message);
    this.name = "McpToolsListError";
    this.code = code;
  }
}

async function listAllMcpTools(client: Client): Promise<Tool[]> {
  const tools: Tool[] = [];
  let cursor: string | undefined;
  const seenCursors = new Set<string>();
  for (let page = 0; page < MCP_LIST_MAX_PAGES; page++) {
    const result = await client.listTools(cursor === undefined ? undefined : { cursor });
    tools.push(...result.tools);
    // MCP: only absent nextCursor ends pagination ("" is a valid cursor).
    if (result.nextCursor === undefined) {
      return tools;
    }
    if (seenCursors.has(result.nextCursor)) {
      throw new McpToolsListError(
        "MCP tools/list returned a repeated nextCursor",
        "RepeatedCursor",
      );
    }
    seenCursors.add(result.nextCursor);
    cursor = result.nextCursor;
  }
  throw new McpToolsListError(
    `MCP tools/list exceeded ${MCP_LIST_MAX_PAGES} pages`,
    "PaginationExceeded",
  );
}

function buildRegisteredMcpTools(
  catalog: ToolCatalog,
  client: Client,
  serverName: string,
  tools: Tool[],
): { toolIds: string[]; registered: ExecutableTool[] } {
  const toolIds: string[] = [];
  const registered: ExecutableTool[] = [];
  for (const tool of tools) {
    const id = `${serverName}__${tool.name}`;
    registered.push({
      id,
      name: tool.name,
      description: tool.description ?? "",
      inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema ?? { type: "object" },
      execute: async (args) => {
        const startedAt = Date.now();
        try {
          const result = await client.callTool({
            name: tool.name,
            arguments: args as Record<string, unknown>,
          });
          catalog.recordEvent({
            type: "upstream_invoke",
            server: serverName,
            tool_id: id,
            took_ms: Date.now() - startedAt,
          });
          return result;
        } catch (err) {
          catalog.recordEvent({
            type: "upstream_error",
            server: serverName,
            tool_id: id,
            error: (err as Error).message ?? String(err),
          });
          throw err;
        }
      },
    });
    toolIds.push(id);
  }
  return { toolIds, registered };
}

/** Options for {@link registerMcpServer}. */
export interface RegisterMcpServerOptions {
  /**
   * Namespace for the server's tools inside the catalog: each tool is
   * registered as `<name>__<toolName>`. Also the `server` name that trace
   * events and result groups report for these tools.
   */
  name: string;
  /**
   * An MCP client transport for the server (e.g. `StdioClientTransport`,
   * `StreamableHTTPClientTransport`, or an `InMemoryTransport` pair in tests).
   * {@link registerMcpServer} connects it; it must not be connected already.
   */
  transport: Transport;
}

/** What {@link registerMcpServer} returns: the ingested ids plus lifecycle control. */
export interface McpServerHandle {
  /**
   * Namespaced ids in upstream list order (all pages). Duplicate names may
   * appear more than once; {@link ToolCatalog.register} keeps the last row.
   */
  toolIds: string[];
  /**
   * The usage instructions the server declared during the MCP initialize
   * handshake, or `undefined` if it declared none. Useful as
   * `UpstreamServerInfo.instructions` when building capability tools.
   */
  serverInstructions: string | undefined;
  /**
   * Close the underlying MCP client connection. The proxied tools stay in the
   * catalog but invoking them after close fails.
   */
  close: () => Promise<void>;
}

/**
 * Ingest an MCP server into a {@link ToolCatalog}: connect over the given
 * transport, list every paginated `tools/list` page (no live refresh), and register each as an
 * {@link ToolCatalog.register | executable tool} whose executor proxies
 * `callTool` on the live client. A missing tool description registers as `""`;
 * a missing output schema as `{ type: "object" }`.
 *
 * The whole registration is one `ratel.upstream.register` OTel span and an
 * `upstream_register` local trace event; each later invocation records
 * `upstream_invoke` (or `upstream_error`) alongside the catalog's own events
 * (ADR-0007). Rejects if connecting or listing tools fails.
 *
 * @param catalog - Catalog that receives the proxied tools.
 * @param options - Server name (the id namespace) and transport.
 * @returns A handle with the registered ids, the server's instructions, and
 *   `close()`.
 *
 * @example
 * ```ts
 * import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
 * import { registerMcpServer, ToolCatalog } from "@ratel-ai/sdk";
 *
 * const catalog = new ToolCatalog();
 * const github = await registerMcpServer(catalog, {
 *   name: "github",
 *   transport: new StdioClientTransport({ command: "github-mcp-server" }),
 * });
 * // github.toolIds → ["github__create_issue", "github__get_pull_request", ...]
 * const result = await catalog.invoke("github__create_issue", {
 *   title: "Flaky test on main",
 * });
 * await github.close();
 * ```
 */
export async function registerMcpServer(
  catalog: ToolCatalog,
  options: RegisterMcpServerOptions,
): Promise<McpServerHandle> {
  const { name, transport } = options;
  const transportLabel = transportKind(transport);

  // The whole registration (connect + list + ingest) is one `ratel.upstream.register`
  // span; per-tool invocations later get their own `execute_tool` spans (ADR-0007).
  return traceUpstreamRegister(name, transportLabel, async (reportToolCount) => {
    const client = new Client({ name: "@ratel-ai/sdk", version: "0.0.0" });
    try {
      await client.connect(transport);

      const serverInstructions = client.getInstructions();

      const tools = await listAllMcpTools(client);
      reportToolCount(tools.length);
      catalog.recordEvent({
        type: "upstream_register",
        server: name,
        transport: transportLabel,
        tool_count: tools.length,
      });
      const { toolIds, registered } = buildRegisteredMcpTools(catalog, client, name, tools);
      await catalog.register(registered);

      return {
        toolIds,
        serverInstructions,
        close: async () => {
          await client.close();
        },
      };
    } catch (err) {
      await client.close().catch(() => {});
      throw err;
    }
  });
}

function transportKind(transport: Transport): string {
  const ctor = (transport as { constructor?: { name?: string } }).constructor;
  return ctor?.name ?? "unknown";
}
