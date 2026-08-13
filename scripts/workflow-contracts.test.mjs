import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const ciWorkflow = readFileSync(new URL("../.github/workflows/ci.yml", import.meta.url), "utf8");
const verifyInstallWorkflow = readFileSync(
  new URL("../.github/workflows/verify-install.yml", import.meta.url),
  "utf8",
);

test("main push path filtering checks out the repository first", () => {
  const changesJob = jobBody(ciWorkflow, "changes");
  const checkout = changesJob.indexOf("uses: actions/checkout@");
  const pathsFilter = changesJob.indexOf("uses: dorny/paths-filter@");

  assert.notEqual(checkout, -1, "changes job must check out the repository");
  assert.ok(checkout < pathsFilter, "checkout must run before paths-filter");
});

for (const job of ["verify-vercel-ai-sdk", "verify-mastra"]) {
  test(`${job} lets npm resolve the adapter's declared SDK peer`, () => {
    const body = jobBody(verifyInstallWorkflow, job);

    assert.doesNotMatch(body, /"@ratel-ai\/sdk@\$\{\{/);
  });
}

function jobBody(workflow, job) {
  const start = workflow.indexOf(`\n  ${job}:`);
  assert.notEqual(start, -1, `missing ${job} job`);

  const rest = workflow.slice(start + 1);
  const nextJob = rest.slice(1).search(/^  [a-z][a-z0-9-]*:\s*$/m);
  return nextJob === -1 ? rest : rest.slice(0, nextJob + 1);
}
