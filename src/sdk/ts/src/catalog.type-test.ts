import {
  DimensionMismatchError,
  EmbedderError,
  IntentGraph,
  type ObservationPolicyOptions,
  type EmbeddingSpec,
  type Executor,
  type Skill,
  SkillCatalog,
  SkillRegistry,
  ToolCatalog,
  ToolRegistry,
} from "./index.js";

const legacyExecutor: Executor = (_input) => ({});
const contextualExecutor: Executor = (_input, context) => context;
// @ts-expect-error executor context is optional and opaque; implementations must narrow it
const requiredNarrowContext: Executor = (_input, _context: { tenantId: string }) => ({});
void [legacyExecutor, contextualExecutor, requiredNarrowContext];

const valid: EmbeddingSpec[] = [
  { huggingface: "org/model", revision: "main", download: false },
  { local: "/models/local", pooling: "mean" },
  { ollama: "nomic-embed-text" },
  { url: "https://example.test/v1/embeddings", model: "embed", apiKeyEnv: "API_KEY" },
];

// @ts-expect-error embedding sources are mutually exclusive
const mixedSources: EmbeddingSpec = { ollama: "nomic", url: "https://example.test", model: "m" };

// @ts-expect-error apiKeyEnv is valid only for an explicit URL endpoint
const ollamaWithApiKey: EmbeddingSpec = { ollama: "nomic", apiKeyEnv: "API_KEY" };

// @ts-expect-error an object embedding config requires exactly one source
const emptyConfig: EmbeddingSpec = {};

new ToolRegistry("/models/local");
new SkillRegistry({ huggingface: "org/model", revision: "main" });

// @ts-expect-error raw tool registries reject mixed embedding sources too
new ToolRegistry({ ollama: "nomic", url: "https://example.test", model: "m" });

// @ts-expect-error raw skill registries require exactly one object-form source
new SkillRegistry({});

void [valid, mixedSources, ollamaWithApiKey, emptyConfig];

// `register` is async and variadic — a single item or a readonly array — on
// both the raw registries and the high-level catalogs (RAT-379/async-register).
async function registerShapes(): Promise<void> {
  const tools = new ToolRegistry();
  await tools.register({
    id: "read",
    name: "read",
    description: "Read a file",
    inputSchema: {},
    outputSchema: {},
  });
  await tools.register([
    { id: "write", name: "write", description: "Write a file", inputSchema: {}, outputSchema: {} },
  ]);

  const skills = new SkillRegistry();
  await skills.register({ id: "deploy", name: "deploy", description: "Deploy an app" });
  await skills.register([
    { id: "lint", name: "lint", description: "Lint the code" } satisfies Skill,
  ]);

  const catalog = new ToolCatalog();
  await catalog.register({
    id: "read_file",
    name: "read_file",
    description: "Read a file",
    inputSchema: {},
    outputSchema: {},
    execute: async () => ({}),
  });
  await catalog.register([
    {
      id: "write_file",
      name: "write_file",
      description: "Write a file",
      inputSchema: {},
      outputSchema: {},
      execute: async () => ({}),
    },
  ]);

  const skillCatalog = new SkillCatalog();
  await skillCatalog.register({ id: "deploy", name: "deploy", description: "Deploy an app" });
  await skillCatalog.register([{ id: "lint", name: "lint", description: "Lint the code" }]);
}
void registerShapes;

// registerMany / buildEmbeddings / rebuildEmbeddings were folded into the
// variadic, self-embedding `register` (RAT-379/async-register) — gone from
// the public surface of the catalogs and the raw registries alike.
// @ts-expect-error registerMany removed from ToolRegistry
new ToolRegistry().registerMany;
// @ts-expect-error buildEmbeddings removed from ToolRegistry
new ToolRegistry().buildEmbeddings;
// @ts-expect-error rebuildEmbeddings removed from ToolRegistry
new ToolRegistry().rebuildEmbeddings;

// @ts-expect-error registerMany removed from SkillRegistry
new SkillRegistry().registerMany;
// @ts-expect-error buildEmbeddings removed from SkillRegistry
new SkillRegistry().buildEmbeddings;
// @ts-expect-error rebuildEmbeddings removed from SkillRegistry
new SkillRegistry().rebuildEmbeddings;

// @ts-expect-error registerMany removed from ToolCatalog
new ToolCatalog().registerMany;
// @ts-expect-error buildEmbeddings removed from ToolCatalog
new ToolCatalog().buildEmbeddings;
// @ts-expect-error rebuildEmbeddings removed from ToolCatalog
new ToolCatalog().rebuildEmbeddings;

// @ts-expect-error registerMany removed from SkillCatalog
new SkillCatalog().registerMany;
// @ts-expect-error buildEmbeddings removed from SkillCatalog
new SkillCatalog().buildEmbeddings;
// @ts-expect-error rebuildEmbeddings removed from SkillCatalog
new SkillCatalog().rebuildEmbeddings;

// Typed embedding errors: DimensionMismatchError is an EmbedderError, both are Errors,
// and every EmbedderError carries a string `code` (parity with Python's hierarchy).
const embedderError: EmbedderError = new DimensionMismatchError("x");
const asError: Error = embedderError;
const code: string = new EmbedderError("x", "Load").code;
void asError;
void code;

// ---- observation policy: a closed set, so typos are compile errors ---------

const validPolicies: ObservationPolicyOptions[] = [
  {},
  { origins: "any" },
  { origins: "baseline", provenance: "seeded" },
  { origins: "direct", confirmation: "succeeded", provenance: "live" },
  { origins: "agent" },
];
void validPolicies;

// @ts-expect-error "baselien" is not an origin — the whole point of narrowing
const typoOrigin: ObservationPolicyOptions = { origins: "baselien" };
// @ts-expect-error confirmation is attempted | succeeded, not an arbitrary string
const typoConfirmation: ObservationPolicyOptions = { confirmation: "done" };
// @ts-expect-error provenance is live | seeded
const typoProvenance: ObservationPolicyOptions = { provenance: "seedd" };
void [typoOrigin, typoConfirmation, typoProvenance];

// The same options reach both learning paths, so neither accepts a value the
// other rejects.
const catalogForPolicy = new ToolCatalog();
void catalogForPolicy.experimentalBuildIntentGraph("", { origins: "baseline" });
void catalogForPolicy.experimentalEnableAdaptiveRanking(new IntentGraph(), {
  confirmation: "succeeded",
  warnOnModelMismatch: false,
});
// @ts-expect-error a typo is caught on the live path too, not just offline
void catalogForPolicy.experimentalEnableAdaptiveRanking(new IntentGraph(), { origins: "nope" });
