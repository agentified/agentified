// Compile-only assertions that the adapter's outputs line up with `@mastra/core`'s
// surface, so a drift in Mastra's types fails `tsc -p tsconfig.type-tests.json`
// rather than at a host's call site.
import { Agent, type ToolsInput } from "@mastra/core/agent";
import type { InputProcessor, Processor } from "@mastra/core/processors";
import { ratel } from "@ratel-ai/sdk";
import { mastra } from "./mastra.js";

const view = ratel().adaptTo(mastra());

// `modelTools()` returns tools assignable straight to an Agent's `ToolsInput` — no cast.
const tools: ToolsInput = view.modelTools();

// `recallProcessor()` is a Mastra `Processor` and, more specifically, an
// `InputProcessor` (id + processInput), so it drops into `inputProcessors`.
const processor: Processor = view.recallProcessor();
const inputProcessor: InputProcessor = view.recallProcessor();

// Both slot into a real Agent construction with no cast.
const agent = new Agent({
  id: "type-test",
  name: "type-test",
  instructions: "test",
  model: "openai/gpt-4o-mini",
  tools: view.modelTools(),
  inputProcessors: [view.recallProcessor()],
});

void tools;
void processor;
void inputProcessor;
void agent;
