import assert from "node:assert/strict";
import { test } from "node:test";

import type { Bindings, ModuleName } from "./builds.ts";
import { ModuleNotLoadedError } from "./trap.ts";

// A split build in miniature: core.wasm's `call` reaches a module's
// function through slot 0 of its table, a placeholder until the module
// (here named `deferred`) writes the function (42) there.
const PRIMARY = [
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 127, 2, 26, 1, 20, 112, 108,
  97, 99, 101, 104, 111, 108, 100, 101, 114, 46, 100, 101, 102, 101, 114, 114,
  101, 100, 1, 48, 0, 0, 3, 2, 1, 0, 4, 5, 1, 112, 1, 1, 1, 7, 16, 2, 4, 99, 97,
  108, 108, 0, 1, 5, 116, 97, 98, 108, 101, 1, 0, 9, 7, 1, 0, 65, 0, 11, 1, 0,
  10, 9, 1, 7, 0, 65, 0, 17, 0, 0, 11,
];
const DEFERRED = [
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 127, 2, 20, 1, 7, 112, 114,
  105, 109, 97, 114, 121, 5, 116, 97, 98, 108, 101, 1, 112, 1, 1, 1, 3, 2, 1, 0,
  9, 7, 1, 0, 65, 0, 11, 1, 0, 10, 6, 1, 4, 0, 65, 42, 11,
];

/** `wasm` with custom sections appended, as xtask's split writes them. */
function module(wasm: number[], sections: Record<string, string>) {
  const bytes = [...wasm];
  for (const [name, payload] of Object.entries(sections)) {
    const body = [
      name.length,
      ...new TextEncoder().encode(name),
      ...new TextEncoder().encode(payload),
    ];
    bytes.push(0, body.length, ...body);
  }
  return new WebAssembly.Module(new Uint8Array(bytes));
}

/** This build in a fresh context (`builds.ts`'s state is its
 * context's), and a module of it whose pair is `pair`. */
async function context(pair: string) {
  const builds = await import(`./builds.ts?${pair}`);
  let instance: WebAssembly.Instance | undefined;
  const glue = {
    __wbg_get_imports: () => ({}),
    __wbg_finalize_init: (i: WebAssembly.Instance) => (instance = i),
  } as unknown as Bindings;
  const primary = module(PRIMARY, { "orbis.pair": "0123456789abcdef" });
  builds.instantiate(glue, { backend: "webgpu", module: primary });
  const call = instance!.exports.call as () => number;
  const name = "deferred" as ModuleName;
  return {
    builds,
    call,
    name,
    split: module(DEFERRED, { "orbis.pair": pair }),
  };
}

test("a call into a module not in yet throws, as a trap", async () => {
  const { call } = await context("0123456789abcdef");
  assert.throws(call, (e) => e instanceof ModuleNotLoadedError);
});

test("a build's module writes its functions into the table", async () => {
  const { builds, call, name, split } = await context("0123456789abcdef");
  builds.takeModule("webgpu", name, split);
  assert.equal(call(), 42);
});

test("another build's module stays out rather than link into it", async () => {
  const { builds, call, name, split } = await context("fedcba9876543210");
  assert.throws(() => builds.takeModule("webgpu", name, split), {
    message: "core-deferred.wasm is another build's",
  });
  assert.throws(call, (e) => e instanceof ModuleNotLoadedError);
});
