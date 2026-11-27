import assert from "node:assert/strict";
import { test } from "node:test";

import apiCalls from "../pkg/api-calls.json" with { type: "json" };
import { Engine, RUN_AS } from "./engine.ts";

// Each of the bindings' methods whose code the API's module holds runs
// only from an Engine call that waits for the module: one of the same
// name, or one that runs it under its own (`RUN_AS`). Otherwise a call
// before the module is in would reach a placeholder and throw.
test("every call into the API's module waits for it", () => {
  const calls = Object.getOwnPropertyNames(Engine.prototype);
  const runAs = new Set<string>(Object.values(RUN_AS).flat());
  for (const name of apiCalls)
    assert.ok(
      calls.includes(name) || runAs.has(name),
      `the bindings' ${name} runs from no Engine call that waits for the API's module`,
    );
  for (const name of Object.keys(RUN_AS))
    assert.ok(calls.includes(name), `RUN_AS names ${name}, no Engine call`);
});
