// The map's two wasm builds of the same crates: core.wasm draws with
// WebGPU, core-gl.wasm with WebGL 2 where WebGPU gives no adapter or the
// map asks for it. A context picks one and compiles it once; its job and
// placement workers instantiate that compiled module with the same
// build's bindings.
//
// core.wasm holds what a map's first frames run, and core-deferred.wasm
// the rest (xtask/src/split.rs), which the map fetches once it is idle
// and hands to its workers. Until then core.wasm calls each deferred
// function through a placeholder: the first call fetches and compiles
// core-deferred.wasm on the spot, which writes the real functions into
// core.wasm's table, and goes on.

import { DeferredLoadError } from "./trap.ts";

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

/** Each build's bindings, an import apiece: as a conditional's two arms,
 * the minifier merged their preloads into one, and the page preloaded
 * core-gl.js for WebGPU (and not core.js). */
const BINDINGS: Record<Backend, () => Promise<Bindings>> = {
  webgpu: () => import("../pkg/core.js"),
  webgl2: () =>
    import("../pkg/core-gl.js") as Promise<unknown> as Promise<Bindings>,
};

/** The bindings of `backend`'s build. */
export function bindings(backend: Backend): Promise<Bindings> {
  return BINDINGS[backend]();
}

/** The deferred half of each build that has one. */
const DEFERRED: Partial<Record<Backend, URL>> = {
  webgpu: new URL("../pkg/core-deferred.wasm", import.meta.url),
};
/** The custom section of a split module naming its placeholder table's
 * export; a whole module has none. */
const SECTION = "orbis.deferred";
/** The custom section both halves of a split build carry, naming their
 * pair: a deferred half takes only its own primary. */
const PAIR = "orbis.pair";
/** Where a split module imports its placeholders from. */
const PLACEHOLDERS = "placeholder.deferred";

const compile = (backend: Backend) =>
  WebAssembly.compileStreaming(fetch(WASM[backend]));

/** This context's map instance, while its deferred half isn't in. */
let waiting:
  | { instance: WebAssembly.Instance; module: WebAssembly.Module; url: URL }
  | undefined;
/** The compiled deferred half, once this context has it. */
let deferred: WebAssembly.Module | undefined;
let fetching: Promise<WebAssembly.Module> | undefined;
const started = new WeakSet<Bindings>();

/** Binds `glue` to an instance of `build`'s module: this context's map
 * code (once per context, as wasm-bindgen's `initSync`). */
export function instantiate(
  glue: Bindings,
  build: Pick<Build, "backend" | "module">,
) {
  if (started.has(glue)) return;
  started.add(glue);
  const { module } = build;
  const imports = glue.__wbg_get_imports() as Record<
    string,
    WebAssembly.ModuleImports
  >;
  const [section] = WebAssembly.Module.customSections(module, SECTION);
  const url = DEFERRED[build.backend];
  let table: WebAssembly.Table | undefined;
  if (section && url) {
    // Each placeholder is named by its slot in the table.
    imports[PLACEHOLDERS] = new Proxy(
      {},
      {
        get:
          (_, slot) =>
          (...args: unknown[]) => {
            link();
            return table!.get(Number(slot))(...args);
          },
      },
    );
  }
  const instance = new WebAssembly.Instance(module, imports);
  if (section && url) {
    table = instance.exports[
      new TextDecoder().decode(section)
    ] as WebAssembly.Table;
    waiting = { instance, module, url };
    if (deferred) link();
  }
  glue.__wbg_finalize_init(instance, module);
}

let loaded: Promise<Bindings> | undefined;

/** core.wasm's bindings, instantiated on this thread if nothing here has
 * yet, with its deferred half: what a page whose map runs in a worker
 * loads for regions. */
export function loadCore(): Promise<Bindings> {
  loaded ??= (async () => {
    const glue = await bindings("webgpu");
    if (started.has(glue)) return glue;
    const build = {
      backend: "webgpu" as const,
      module: await compile("webgpu"),
    };
    instantiate(glue, build);
    await fetchDeferred(build);
    return glue;
  })();
  return loaded;
}

/** Starts fetching `build`'s deferred half, if it has one, and resolves
 * to it compiled (once per context); this context's instance takes it
 * as soon as it compiles. */
export function fetchDeferred(
  build: Pick<Build, "backend" | "module">,
): Promise<WebAssembly.Module> | undefined {
  const url = DEFERRED[build.backend];
  if (!url || !WebAssembly.Module.customSections(build.module, SECTION).length)
    return undefined;
  fetching ??= WebAssembly.compileStreaming(
    fetch(url, { priority: "low" } as RequestInit),
  ).then(
    (module) => {
      takeDeferred(module);
      return module;
    },
    (e) => {
      // A placeholder's call can still fetch it.
      fetching = undefined;
      throw e;
    },
  );
  return fetching;
}

/** This context's deferred half, compiled, if it has it. */
export function deferredModule(): WebAssembly.Module | undefined {
  return deferred;
}

/** Hands this context the compiled deferred half, which its instance
 * takes now or when it is made. */
export function takeDeferred(module: WebAssembly.Module): void {
  deferred ??= module;
  if (waiting) link();
}

/** Makes the deferred half with this context's instance's exports, which
 * writes its functions into the instance's table: compiled already, or
 * fetched and compiled now, for a placeholder called before. */
function link(): void {
  if (!waiting) return;
  const { instance, module, url } = waiting;
  try {
    deferred ??= new WebAssembly.Module(fetchNow(url));
    // Another build's deferred half (a stale copy, a cache) would write
    // other functions into the table, or fail to link, its error thrown
    // through the Rust frames of the call that needed it.
    const pair = (m: WebAssembly.Module) =>
      new TextDecoder().decode(WebAssembly.Module.customSections(m, PAIR)[0]);
    if (pair(deferred) !== pair(module))
      throw new Error("it is another build's");
    new WebAssembly.Instance(deferred, { primary: instance.exports });
  } catch (e) {
    throw new DeferredLoadError(url, e);
  }
  waiting = undefined;
}

/** `url`'s bytes, synchronously, as a placeholder needs them. */
function fetchNow(url: URL): ArrayBuffer {
  const request = new XMLHttpRequest();
  request.open("GET", url.href, false);
  // A page's synchronous requests can't ask for bytes, only for text.
  const page = typeof document !== "undefined";
  if (page) request.overrideMimeType("text/plain; charset=x-user-defined");
  else request.responseType = "arraybuffer";
  request.send();
  if (request.status !== 200)
    throw new Error(`${url.pathname}: HTTP ${request.status}`);
  if (!page) return request.response as ArrayBuffer;
  const text = request.responseText;
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i) & 0xff;
  return bytes.buffer;
}

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
