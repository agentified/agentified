import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface, type Interface } from "node:readline";

/** Test-only helper (not part of the published package); shared by the tool-
 * and skill-catalog test suites. */
export interface DelayedEmbeddingServer {
  url: string;
  requests: string[][];
  /**
   * Arm a one-shot hold for the next request whose `input` array length is
   * `>= minBatch`. Resolves only after the child ACKs (`{ type: "armed" }`).
   * Single-input queries still complete immediately.
   */
  armBatchHold: (minBatch?: number) => Promise<void>;
  /** Resolves once a held batch request has arrived and is waiting. */
  waitForHeld: () => Promise<void>;
  /** Release the currently held batch request (if any). */
  releaseHeld: () => void;
  close: () => Promise<void>;
}

type ControlMessage = { type: "armed"; minBatch: number } | { type: "held"; n: number };

/** An OpenAI-compatible embedding endpoint that sleeps briefly per request, run
 * in a child process (real wall-clock delay) — shared by tool- and
 * skill-catalog tests that assert dense registration/search doesn't block
 * Node's event loop. Supports an optional batch hold/release barrier for
 * concurrency regressions. */
export async function startDelayedEmbeddingServer(delayMs = 120): Promise<DelayedEmbeddingServer> {
  const source = `
    const http = require("node:http");
    const readline = require("node:readline");
    let holdMinBatch = 0;
    const held = [];
    const rl = readline.createInterface({ input: process.stdin });
    rl.on("line", (line) => {
      const text = String(line).trim();
      if (text.startsWith("HOLD ")) {
        holdMinBatch = Number(text.slice(5)) || 0;
        process.stdout.write(JSON.stringify({ type: "armed", minBatch: holdMinBatch }) + "\\n");
        return;
      }
      if (text === "RELEASE") {
        while (held.length > 0) {
          const resume = held.shift();
          resume();
        }
      }
    });
    function respond(response, payload, inputs) {
      setTimeout(() => {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({
          model: payload.model,
          data: inputs.map((_, index) => ({ index, embedding: [index + 1, 1] })),
        }));
      }, ${delayMs});
    }
    const server = http.createServer((request, response) => {
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk) => { body += chunk; });
      request.on("end", () => {
        const payload = JSON.parse(body);
        const inputs = Array.isArray(payload.input) ? payload.input : [payload.input];
        process.stdout.write(JSON.stringify(inputs) + "\\n");
        if (holdMinBatch > 0 && inputs.length >= holdMinBatch) {
          holdMinBatch = 0;
          process.stdout.write(JSON.stringify({ type: "held", n: inputs.length }) + "\\n");
          held.push(() => respond(response, payload, inputs));
          return;
        }
        respond(response, payload, inputs);
      });
    });
    server.listen(0, "127.0.0.1", () => process.stdout.write(String(server.address().port) + "\\n"));
    process.on("SIGTERM", () => server.close(() => process.exit(0)));
  `;
  const child: ChildProcessWithoutNullStreams = spawn(process.execPath, ["-e", source], {
    stdio: ["pipe", "pipe", "inherit"],
  });
  const lines: Interface = createInterface({ input: child.stdout });
  let startupTimer: ReturnType<typeof setTimeout> | undefined;
  const [line] = (await Promise.race([
    once(lines, "line"),
    once(child, "exit").then(([code]) => {
      throw new Error(`embedding test server exited during startup (${String(code)})`);
    }),
    new Promise<never>((_, reject) => {
      startupTimer = setTimeout(
        () => reject(new Error("embedding test server startup timed out")),
        5_000,
      );
    }),
  ]).finally(() => clearTimeout(startupTimer))) as [string];
  const requests: string[][] = [];
  let heldWaiters: Array<() => void> = [];
  let heldCount = 0;
  let armedWaiters: Array<(msg: ControlMessage & { type: "armed" }) => void> = [];

  lines.on("line", (requestLine) => {
    const parsed = JSON.parse(requestLine) as string[] | ControlMessage;
    if (Array.isArray(parsed)) {
      requests.push(parsed);
      return;
    }
    if (parsed?.type === "armed") {
      const waiters = armedWaiters;
      armedWaiters = [];
      for (const resolve of waiters) resolve(parsed);
      return;
    }
    if (parsed?.type === "held") {
      heldCount += 1;
      const waiters = heldWaiters;
      heldWaiters = [];
      for (const resolve of waiters) resolve();
    }
  });

  return {
    url: `http://127.0.0.1:${Number(line)}/v1/embeddings`,
    requests,
    armBatchHold: (minBatch = 2) => {
      heldCount = 0;
      const armed = new Promise<void>((resolve) => {
        armedWaiters.push(() => resolve());
      });
      child.stdin.write(`HOLD ${minBatch}\n`);
      return armed;
    },
    waitForHeld: () => {
      if (heldCount > 0) return Promise.resolve();
      return new Promise<void>((resolve) => {
        heldWaiters.push(resolve);
      });
    },
    releaseHeld: () => {
      child.stdin.write("RELEASE\n");
    },
    close: async () => {
      lines.close();
      if (child.exitCode !== null) return;
      const exited = once(child, "exit");
      child.kill("SIGKILL");
      await exited;
    },
  };
}
