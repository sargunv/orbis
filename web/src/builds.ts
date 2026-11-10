// The map's two wasm builds of the same crates: core.wasm draws with
// WebGPU, core-gl.wasm with WebGL 2 where WebGPU gives no adapter or the
// map asks for it. A context picks one and compiles it once; its job and
// placement workers instantiate that compiled module with the same
// build's bindings.

/** The GPU API a map draws with. */
export type Backend = "webgpu" | "webgl2";

/** The GPU API a map asks for: one of them, or `auto` for WebGPU where it
 * gives an adapter and WebGL 2 otherwise. */
export type BackendChoice = Backend | "auto";

/** A compiled build. */
export interface Build {
  backend: Backend;
  module: WebAssembly.Module;
  /** Why not WebGPU, for core-gl.wasm picked by `auto`. */
  noWebgpu?: string;
  /** The map asked for this backend rather than leaving it to `auto`. */
  forced?: true;
}

/** A build's wasm-bindgen bindings: both export the same API. */
export type Bindings = typeof import("../pkg/core.js");

const WASM = {
  webgpu: new URL("../pkg/core_bg.wasm", import.meta.url),
  webgl2: new URL("../pkg/core-gl_bg.wasm", import.meta.url),
};

/** The bindings of `backend`'s build. */
export function bindings(backend: Backend): Promise<Bindings> {
  return backend === "webgpu"
    ? import("../pkg/core.js")
    : (import("../pkg/core-gl.js") as Promise<unknown> as Promise<Bindings>);
}

const compile = (backend: Backend) =>
  WebAssembly.compileStreaming(fetch(WASM[backend]));

/** Why WebGPU gives no adapter here, if it doesn't. */
async function noWebgpu(gpu: GPU | undefined): Promise<string | undefined> {
  if (!gpu) return "this browser has none";
  try {
    if (await gpu.requestAdapter()) return undefined;
    return "it found no GPU here";
  } catch (e) {
    return `it failed: ${e instanceof Error ? e.message : e}`;
  }
}

/** This context's build for `choice`, compiled. `auto`: core.wasm where
 * WebGPU gives an adapter, core-gl.wasm otherwise; where WebGPU is exposed
 * core.wasm downloads while the adapter is asked for, so a WebGPU map
 * waits for nothing new, and without it core-gl.wasm downloads at once.
 * `webgl2` downloads only core-gl.wasm and asks WebGPU nothing; `webgpu`
 * rejects, saying why, where WebGPU gives no adapter. `gpu` is this
 * context's WebGPU, if it has one. */
export function pickBuild(
  gpu = (globalThis.navigator as { gpu?: GPU } | undefined)?.gpu,
  choice: BackendChoice = "auto",
): Promise<Build> {
  if (choice === "webgl2")
    return compile("webgl2").then((module) => ({
      backend: "webgl2",
      module,
      forced: true,
    }));
  const likely = gpu ? compile("webgpu") : undefined;
  likely?.catch(() => {});
  return (async () => {
    const missing = await noWebgpu(gpu);
    if (missing === undefined)
      return {
        backend: "webgpu",
        module: await likely!,
        ...(choice === "webgpu" && { forced: true as const }),
      };
    if (choice === "webgpu")
      throw new Error(`Orbis can't draw with WebGPU, as asked: ${missing}`);
    return {
      backend: "webgl2",
      module: await compile("webgl2"),
      noWebgpu: missing,
    };
  })();
}
