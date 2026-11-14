// The map's two wasm builds of the same crates: core.wasm draws with
// WebGPU, core-gl.wasm with WebGL 2 where WebGPU gives no adapter or the
// map asks for it. A context picks one and compiles it once; its job and
// placement workers instantiate that compiled module with the same
// build's bindings.
//
// Each build holds what plain vector maps run, and leaves each feature a
// style may declare to a module of its own (xtask/src/split.rs):
// `core-<name>.wasm` and `core-gl-<name>.wasm`. A map asks for a module
// once its style declares the feature, and its style waits until every
// thread of the map has the module: this context fetches and compiles it
// once, instantiates it against the build's memory and table, which
// writes its functions into the build's table, and hands it to the
// workers before the map sends them work that needs it. Nothing loads
// synchronously: core.wasm calls a module's functions through
// placeholders until it's in, and a placeholder's call throws, a bug in
// the split's paths or in the map's gate.

import { ModuleNotLoadedError } from "./trap.ts";

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
  /** WebGPU was expected to draw and didn't: it failed, didn't answer, or
   * its map stopped working. */
  unexpected?: true;
  /** The map asked for this backend rather than leaving it to `auto`. */
  forced?: true;
}

/** A build's wasm-bindgen bindings: both export the same API. */
export type Bindings = typeof import("../pkg/core.js");

const WASM = {
  webgpu: new URL("../pkg/core_bg.wasm", import.meta.url),
  webgl2: new URL("../pkg/core-gl_bg.wasm", import.meta.url),
};

/** The feature modules, as orb-runtime's `Modules` names them. */
export type ModuleName =
  | "mlt"
  | "archive"
  | "sqlite"
  | "dem"
  | "globe"
  | "geojson"
  | "api"
  | "bidi";

/** The feature modules in orb-runtime's `Modules::NAMES` order, which
 * the map's module bits follow. */
export const MODULE_NAMES: readonly ModuleName[] = [
  "mlt",
  "archive",
  "sqlite",
  "dem",
  "globe",
  "geojson",
  "api",
  "bidi",
];

/** Each build's feature modules: literal URLs, which the bundler hashes. */
const MODULES: Record<Backend, Record<ModuleName, URL>> = {
  webgpu: {
    mlt: new URL("../pkg/core-mlt.wasm", import.meta.url),
    archive: new URL("../pkg/core-archive.wasm", import.meta.url),
    sqlite: new URL("../pkg/core-sqlite.wasm", import.meta.url),
    dem: new URL("../pkg/core-dem.wasm", import.meta.url),
    globe: new URL("../pkg/core-globe.wasm", import.meta.url),
    geojson: new URL("../pkg/core-geojson.wasm", import.meta.url),
    api: new URL("../pkg/core-api.wasm", import.meta.url),
    bidi: new URL("../pkg/core-bidi.wasm", import.meta.url),
  },
  webgl2: {
    mlt: new URL("../pkg/core-gl-mlt.wasm", import.meta.url),
    archive: new URL("../pkg/core-gl-archive.wasm", import.meta.url),
    sqlite: new URL("../pkg/core-gl-sqlite.wasm", import.meta.url),
    dem: new URL("../pkg/core-gl-dem.wasm", import.meta.url),
    globe: new URL("../pkg/core-gl-globe.wasm", import.meta.url),
    geojson: new URL("../pkg/core-gl-geojson.wasm", import.meta.url),
    api: new URL("../pkg/core-gl-api.wasm", import.meta.url),
    bidi: new URL("../pkg/core-gl-bidi.wasm", import.meta.url),
  },
};

/** Where a build imports a module's placeholders from: this, then the
 * module's name. */
const PLACEHOLDERS = "placeholder.";
/** The custom section every module of a split build carries, naming its
 * build: a module takes only its own build's instance. */
const PAIR = "orbis.pair";

/** The custom section a feature module's data comes in, then its name:
 * read-only data only its code reads, which the build's memory gets as
 * the module comes in (orb-web's `setModuleData`). */
const DATA = "orbis.data.";

/** Hands `glue`'s instance module `name`'s data, if `module` carries it. */
const handData = (glue: Bindings, module: WebAssembly.Module, name: string) => {
  for (const data of WebAssembly.Module.customSections(module, DATA + name))
    glue.setModuleData(name, new Uint8Array(data));
};

/** The build `module` names, if it names one. */
const pair = (module: WebAssembly.Module) => {
  const [section] = WebAssembly.Module.customSections(module, PAIR);
  return section && new TextDecoder().decode(section);
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

const compile = (url: URL, init?: RequestInit) =>
  WebAssembly.compileStreaming(fetch(url, init));

/** What this context has of one build. A context holds both builds once
 * a map falls back from WebGPU to WebGL 2. */
interface Held {
  /** Its instance, once made, and the modules it calls through
   * placeholders: none for a whole build (a profiled or panicking one),
   * which holds every module's code. */
  instance?: {
    exports: WebAssembly.Exports;
    split: Set<string>;
    pair: string | undefined;
    glue: Bindings;
  };
  /** Its modules, compiled, and those linked into its instance. */
  modules: Map<ModuleName, WebAssembly.Module>;
  linked: Set<ModuleName>;
  /** Its modules being fetched. */
  fetching: Map<ModuleName, Promise<WebAssembly.Module>>;
}
const held = (): Held => ({
  modules: new Map(),
  linked: new Set(),
  fetching: new Map(),
});
const builds: Record<Backend, Held> = { webgpu: held(), webgl2: held() };
const started = new WeakSet<Bindings>();

/** Binds `glue` to an instance of `build`'s module: this context's map
 * code (once per context, as wasm-bindgen's `initSync`), with the modules
 * this context already has. */
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
  const split = new Set<string>();
  for (const { module: from } of WebAssembly.Module.imports(module)) {
    if (!from.startsWith(PLACEHOLDERS) || imports[from]) continue;
    const name = from.slice(PLACEHOLDERS.length);
    split.add(name);
    imports[from] = new Proxy(
      {},
      {
        get: (_, slot) => () => {
          throw new ModuleNotLoadedError(name, String(slot));
        },
      },
    );
  }
  const made = new WebAssembly.Instance(module, imports);
  const own = builds[build.backend];
  own.instance = { exports: made.exports, split, pair: pair(module), glue };
  glue.__wbg_finalize_init(made, module);
  // A whole build carries its modules' data itself.
  for (const name of MODULE_NAMES) handData(glue, module, name);
  // What runs outside a map's gate, whatever its style (offline regions'
  // extracts, read in every context), loads its modules through this.
  glue.setModuleLoader((name: string) =>
    isModuleName(name) && split.has(name)
      ? loadModule(build.backend, name)
      : undefined,
  );
  for (const name of own.modules.keys()) {
    // Another build's module stays out: asked for again, it loads again.
    try {
      link(own, name);
    } catch (e) {
      console.warn(e);
    }
  }
}

/** Writes module `name`'s functions into `build`'s instance, if both are
 * here and it isn't yet. Another build's module (a stale copy, a cache)
 * would write other functions into the table: it throws, and stays out. */
function link(build: Held, name: ModuleName): void {
  const { instance, modules, linked } = build;
  const module = modules.get(name);
  if (!instance || !module || linked.has(name)) return;
  if (instance.split.has(name)) {
    if (pair(module) !== instance.pair) {
      modules.delete(name);
      throw new Error(`core-${name}.wasm is another build's`);
    }
    // Its data first: nothing can call its code before it's in.
    handData(instance.glue, module, name);
    new WebAssembly.Instance(module, { primary: instance.exports });
  }
  linked.add(name);
}

/** Hands this context `backend`'s module `name`, compiled elsewhere: the
 * build's instance takes it now, or when it's made. */
export function takeModule(
  backend: Backend,
  name: ModuleName,
  module: WebAssembly.Module,
) {
  const build = builds[backend];
  if (!build.modules.has(name)) build.modules.set(name, module);
  link(build, name);
}

/** Fetches and compiles `backend`'s module `name` (once per context), and
 * links it into this context's instance; resolves to it compiled, for
 * the context's workers. */
export function loadModule(
  backend: Backend,
  name: ModuleName,
): Promise<WebAssembly.Module> {
  const { modules, fetching } = builds[backend];
  const have = modules.get(name);
  if (have) return Promise.resolve(have);
  let load = fetching.get(name);
  if (!load) {
    // The API's module isn't on a first frame's way.
    const init = name === "api" ? ({ priority: "low" } as RequestInit) : {};
    load = compile(MODULES[backend][name], init).then(
      (module) => {
        takeModule(backend, name, module);
        return module;
      },
      (e) => {
        // Asked again, it's fetched again.
        fetching.delete(name);
        throw e;
      },
    );
    fetching.set(name, load);
  }
  return load;
}

/** `backend`'s modules this context has, compiled: what a worker it
 * spawns on that build starts with. */
export function heldModules(
  backend: Backend,
): [ModuleName, WebAssembly.Module][] {
  return [...builds[backend].modules];
}

/** Whether `name` is a feature module's name. */
export function isModuleName(name: string): name is ModuleName {
  return name in MODULES.webgpu;
}

let loaded: Promise<Bindings> | undefined;

/** core.wasm's bindings, instantiated on this thread if nothing here has
 * yet, with its API module: what regions load on a page with no map yet. */
export function loadCore(): Promise<Bindings> {
  loaded ??= (async () => {
    const glue = await bindings("webgpu");
    if (!started.has(glue))
      instantiate(glue, {
        backend: "webgpu",
        module: await compile(WASM.webgpu),
      });
    await loadModule("webgpu", "api");
    return glue;
  })();
  return loaded;
}

/** How long WebGPU is given to answer for an adapter. It answers in tens
 * of milliseconds; a browser whose GPU process is wedged may never answer,
 * and a map waiting on it would never start. */
export const ADAPTER_MS = 5000;

/** Why WebGPU gives no adapter here, if it doesn't, and whether that is
 * unexpected: it failed, or didn't answer in `ADAPTER_MS`. */
async function noWebgpu(
  gpu: GPU | undefined,
): Promise<{ why?: string; unexpected?: true }> {
  if (!gpu) return { why: "this browser has none" };
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<{ why: string; unexpected: true }>((resolve) => {
    timer = setTimeout(
      () =>
        resolve({
          why: `it didn't answer for an adapter in ${ADAPTER_MS / 1000} s`,
          unexpected: true,
        }),
      ADAPTER_MS,
    );
  });
  const answer = Promise.resolve()
    .then(() => gpu.requestAdapter())
    .then(
      (adapter) => (adapter ? {} : { why: "it found no GPU here" }),
      (e) => ({
        why: `it failed: ${e instanceof Error ? e.message : e}`,
        unexpected: true as const,
      }),
    );
  try {
    return await Promise.race([answer, late]);
  } finally {
    clearTimeout(timer);
  }
}

/** core-gl.wasm, compiled. */
const gl = (
  build: Pick<Build, "noWebgpu" | "forced" | "unexpected">,
): Promise<Build> =>
  compile(WASM.webgl2).then((module) => ({
    backend: "webgl2",
    module,
    ...build,
  }));

/** core-gl.wasm, for a map whose WebGPU stopped working (`why`). */
export function fallbackBuild(why: string): Promise<Build> {
  return gl({ noWebgpu: why, unexpected: true });
}

/** This context's build for `choice`, compiled. `auto`: core.wasm where
 * WebGPU gives an adapter, core-gl.wasm otherwise; where WebGPU is exposed
 * core.wasm downloads while the adapter is asked for, so a WebGPU map
 * waits for nothing new, and without it core-gl.wasm downloads at once.
 * `webgl2` downloads only core-gl.wasm and asks WebGPU nothing; `webgpu`
 * rejects, saying why, where WebGPU gives no adapter, as where it doesn't
 * answer in `ADAPTER_MS`. `gpu` is this context's WebGPU, if it has one. */
export function pickBuild(
  gpu = (globalThis.navigator as { gpu?: GPU } | undefined)?.gpu,
  choice: BackendChoice = "auto",
): Promise<Build> {
  if (choice === "webgl2") return gl({ forced: true });
  const likely = gpu ? compile(WASM.webgpu) : undefined;
  likely?.catch(() => {});
  return (async () => {
    const { why, unexpected } = await noWebgpu(gpu);
    if (why === undefined)
      return {
        backend: "webgpu",
        module: await likely!,
        ...(choice === "webgpu" && { forced: true as const }),
      };
    if (choice === "webgpu")
      throw new Error(`Orbis can't draw with WebGPU, as asked: ${why}`);
    return gl({ noWebgpu: why, ...(unexpected && { unexpected }) });
  })();
}
