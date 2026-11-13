import assert from "node:assert/strict";
import { mock, test } from "node:test";

const { ADAPTER_MS, fallbackBuild, pickBuild } = await import("./builds.ts");

// A "compiled" build is the name of the file fetched for it.
const fetched: string[] = [];
globalThis.fetch = (async (url: URL) => {
  fetched.push(url.pathname.split("/").pop()!);
  return url;
}) as unknown as typeof fetch;
WebAssembly.compileStreaming = (async (source: Promise<URL>) =>
  (await source).pathname.split("/").pop()) as never;

/** A WebGPU that gives an adapter (or not), counting the asks. */
function gpu(adapter: boolean) {
  const asked = { count: 0 };
  const api = {
    requestAdapter: async () => {
      asked.count++;
      return adapter ? {} : null;
    },
  } as unknown as GPU;
  return { api, asked };
}

async function pick(...args: Parameters<typeof pickBuild>) {
  fetched.length = 0;
  const build = await pickBuild(...args);
  return { ...build, fetched: [...fetched] };
}

test("auto draws WebGPU where it gives an adapter, and WebGL 2 saying why where it doesn't", async () => {
  const yes = gpu(true);
  assert.deepEqual(await pick(yes.api), {
    backend: "webgpu",
    module: "core_bg.wasm",
    fetched: ["core_bg.wasm"],
  });
  assert.deepEqual(await pick(gpu(false).api, "auto"), {
    backend: "webgl2",
    module: "core-gl_bg.wasm",
    noWebgpu: "it found no GPU here",
    fetched: ["core_bg.wasm", "core-gl_bg.wasm"],
  });
  assert.deepEqual(await pick(undefined), {
    backend: "webgl2",
    module: "core-gl_bg.wasm",
    noWebgpu: "this browser has none",
    fetched: ["core-gl_bg.wasm"],
  });
});

test("webgl2 asks WebGPU nothing and fetches only core-gl.wasm", async () => {
  const yes = gpu(true);
  assert.deepEqual(await pick(yes.api, "webgl2"), {
    backend: "webgl2",
    module: "core-gl_bg.wasm",
    forced: true,
    fetched: ["core-gl_bg.wasm"],
  });
  assert.equal(yes.asked.count, 0);
});

test("webgpu draws WebGPU, and fails saying why where it gives no adapter", async () => {
  assert.deepEqual(await pick(gpu(true).api, "webgpu"), {
    backend: "webgpu",
    module: "core_bg.wasm",
    forced: true,
    fetched: ["core_bg.wasm"],
  });
  await assert.rejects(pick(gpu(false).api, "webgpu"), {
    message: "Orbis can't draw with WebGPU, as asked: it found no GPU here",
  });
  await assert.rejects(pick(undefined, "webgpu"), {
    message: "Orbis can't draw with WebGPU, as asked: this browser has none",
  });
});

test("auto draws WebGL 2 where WebGPU fails or never answers, saying so", async () => {
  const failing = {
    requestAdapter: async () => {
      throw new Error("the GPU process is gone");
    },
  } as unknown as GPU;
  assert.deepEqual(await pick(failing), {
    backend: "webgl2",
    module: "core-gl_bg.wasm",
    noWebgpu: "it failed: the GPU process is gone",
    unexpected: true,
    fetched: ["core_bg.wasm", "core-gl_bg.wasm"],
  });
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const silent = {
      requestAdapter: () => new Promise(() => {}),
    } as unknown as GPU;
    const picked = pickBuild(silent);
    mock.timers.tick(ADAPTER_MS);
    assert.deepEqual(await picked, {
      backend: "webgl2",
      module: "core-gl_bg.wasm",
      noWebgpu: `it didn't answer for an adapter in ${ADAPTER_MS / 1000} s`,
      unexpected: true,
    });
  } finally {
    mock.timers.reset();
  }
});

test("a map whose WebGPU stopped working falls back to WebGL 2, saying why", async () => {
  assert.deepEqual(await fallbackBuild("the GPU was lost"), {
    backend: "webgl2",
    module: "core-gl_bg.wasm",
    noWebgpu: "the GPU was lost",
    unexpected: true,
  });
});
