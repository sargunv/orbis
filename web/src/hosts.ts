// The wasm map, with its job workers and placement worker (workers.ts),
// and its recovery from a lost GPU: in the map worker or on the page.

import type { OrbisMap as Core } from "../pkg/core.js";
import {
  bindings,
  instantiate,
  loadModule,
  type Backend,
  type Build,
  type ModuleName,
} from "./builds.ts";
import { takePreload } from "./preload.ts";
import { onRegionsChanged } from "./regions.ts";
import { answerWorker, connectModule, connectWorker } from "./requests.ts";
import { guard, trapError } from "./trap.ts";
import type { TileLod } from "./types.ts";
import {
  spawnJob,
  spawnPlacement,
  terminateWorkers,
  type Workers,
} from "./workers.ts";

export type { Core };

/** How many times each job worker slot is given a new worker. */
const REPLACEMENTS = 3;

/** The map options the core takes at creation (`MapOptions`). */
export interface CoreOptions {
  fadeDuration?: number;
  crossSourceCollisions?: boolean;
  localIdeographFontFamily?: string | false;
  antialias?: boolean;
  terrainSkirts?: boolean;
  previews?: readonly string[] | "all";
  stableLabels?: boolean;
  tileLod?: TileLod;
  missingTiles?: "parent" | "empty";
}

export interface HostCallbacks {
  /** The map has new data to draw. */
  wake(): void;
  error(e: unknown): void;
  /** The wasm map trapped, a bug: it takes no more calls, every one
   * throwing `e`, and its workers are gone. */
  trapped(e: Error): void;
  /** Sets up the new wasm map. */
  setup(core: Core): void;
}

/** The wasm map and what it runs on. */
export interface Host {
  readonly core: Core;
  /** The GPU API it draws with: its build's. */
  readonly backend: Backend;
  resize(width: number, height: number, ratio: number): void;
  /** Loses the GPU on purpose, to exercise recovery. */
  loseDevice(): void;
  /** Traps the placement worker's wasm, as a bug would, to exercise its
   * replacement. */
  trapPlacementWorker(): void;
  /** Traps job worker `index`'s wasm, to exercise its replacement. */
  trapJobWorker(index: number): void;
  /** How many job workers the map has. */
  readonly jobWorkers: number;
  /** Draws on a new GPU after a loss, on the same canvas. */
  recover(): Promise<void>;
  /** Loads feature modules into the map and its workers: resolves once
   * every one of them has them. */
  loadModules(names: readonly ModuleName[]): Promise<void>;
  /** The feature modules the map and its workers have. */
  modules(): ReadonlySet<ModuleName>;
  destroy(): void;
}

/** The map on this thread, drawing into `canvas`, once `compiling` has
 * compiled its build; `workers`, spawned on the same build, are its. */
export async function startHost(
  compiling: Promise<Build>,
  workers: Workers,
  canvas: HTMLCanvasElement | OffscreenCanvas,
  style: string,
  options: CoreOptions,
  callbacks: HostCallbacks,
): Promise<Host> {
  const receive =
    (apply: (data: unknown) => void) =>
    ({ data }: MessageEvent) => {
      try {
        apply(data);
      } catch (e) {
        callbacks.error(e);
      }
    };
  // Replies still on their way when the map stops find it gone.
  let stopped = false;
  const reply = (apply: (data: unknown) => void) =>
    receive((data) => {
      if (!stopped) apply(data);
    });
  const jobs = workers.jobs;
  const ports = jobs.map((j) => j.port);
  let placement = workers.placement;
  let build: Build;
  let core: Core;
  let stop: (e: unknown) => Error;
  let destroy = () => {};
  let reloadRegions: () => void;
  try {
    build = await compiling;
    const glue = await bindings(build.backend);
    const { createMap: createCoreMap, setPreloads, setRequestHook } = glue;
    reloadRegions = glue.reloadRegions;
    instantiate(glue, build);
    connectModule(setRequestHook);
    setPreloads(takePreload);
    const created: Core = await createCoreMap(
      canvas,
      style,
      options,
      callbacks.wake,
      ports.length,
      (worker: number, message: unknown) =>
        at(ports, worker).postMessage(message),
      (message: unknown, transfer: Transferable[]) =>
        placement.port.postMessage(message, transfer),
    );
    ({ core, stop } = guard(created, (e) => {
      destroy();
      callbacks.trapped(e);
    }));
  } catch (e) {
    terminateWorkers(workers);
    // A trap making the map is a bug, not this thread's lack.
    if (e instanceof WebAssembly.RuntimeError) throw trapError(e);
    // Neither WebGPU nor WebGL 2 draws here, or the one asked for doesn't.
    const picked = await compiling.catch(() => undefined);
    const message = e instanceof Error ? e.message : e;
    if (picked?.forced && picked.backend === "webgl2")
      throw new Error(`Orbis can't draw with WebGL 2, as asked: ${message}`);
    if (picked?.noWebgpu)
      throw new Error(
        `Orbis needs WebGPU or WebGL 2. WebGPU: ${picked.noWebgpu}; WebGL 2: ${message}`,
      );
    throw e;
  }
  // A placement worker whose wasm trapped says so and closes; one that
  // dies fires `error`. Either way another takes its place, and the map
  // sends it what it places from.
  const listen = () => {
    const current = placement;
    const replace = (why: string) => {
      if (placement !== current) return;
      current.worker.terminate();
      placement = spawnPlacement(build);
      listen();
      core.placementWorkerLost(why);
    };
    current.port.onmessage = reply((data) => {
      const lost = (data as { lost?: string }).lost;
      if (lost !== undefined) replace(lost);
      else core.receivePlacement(data);
    });
    current.worker.addEventListener("error", (e) => {
      e.preventDefault();
      if (!stopped) replace(e.message || "the worker died");
    });
  };
  listen();
  // A job worker whose wasm trapped says so and closes; one that dies
  // fires `error`. Either way another takes its place, which the map
  // sends its style before its loads go again; a slot that has lost
  // REPLACEMENTS workers is given up, and its share goes to the others.
  const released = ports.map(connectWorker);
  const replaced = jobs.map(() => 0);
  const listenJob = (i: number) => {
    const { port, worker } = at(jobs, i);
    let gone = false;
    const lose = (why: string) => {
      if (gone || stopped) return;
      gone = true;
      worker.terminate();
      port.close();
      at(released, i)();
      const replace = at(replaced, i) < REPLACEMENTS;
      replaced[i] = at(replaced, i) + 1;
      if (replace) {
        const job = spawnJob(build);
        jobs[i] = job;
        ports[i] = job.port;
        released[i] = connectWorker(job.port);
        listenJob(i);
      }
      core.jobWorkerLost(i, why, replace);
    };
    port.onmessage = reply((data) => {
      const lost = (data as { lost?: string }).lost;
      if (lost !== undefined) lose(lost);
      else if (!answerWorker(port, data)) core.receiveJob(data);
    });
    worker.addEventListener("error", (e) => {
      e.preventDefault();
      lose(e.message || "the worker died");
    });
  };
  jobs.forEach((_, i) => listenJob(i));
  const regions = onRegionsChanged(reloadRegions);
  // In the map worker the module is the map's alone: a trap outside a
  // call (a future's poll, a callback from the browser) stops it too. On
  // the page the module may serve other maps, and a trap there can't be
  // told apart from other code's.
  const inWorker =
    typeof WorkerGlobalScope !== "undefined" &&
    self instanceof WorkerGlobalScope;
  const uncaught = (e: ErrorEvent | PromiseRejectionEvent) => {
    const error = e instanceof ErrorEvent ? e.error : e.reason;
    if (!(error instanceof WebAssembly.RuntimeError)) return;
    e.preventDefault();
    stop(error);
  };
  if (inWorker) {
    self.addEventListener("error", uncaught);
    self.addEventListener("unhandledrejection", uncaught);
  }
  destroy = () => {
    if (stopped) return;
    stopped = true;
    for (const port of [...ports, placement.port]) port.close();
    regions();
    for (const r of released) r();
    for (const w of [...jobs.map((j) => j.worker), placement.worker])
      w.terminate();
    if (inWorker) {
      self.removeEventListener("error", uncaught);
      self.removeEventListener("unhandledrejection", uncaught);
    }
  };
  callbacks.setup(core);
  let size: [number, number, number] | undefined;
  const loaded = new Set<ModuleName>();
  return {
    core,
    backend: build.backend,
    resize: (width, height, ratio) => {
      size = [width, height, ratio];
      core.resize(width, height, ratio);
    },
    loseDevice: () => core.loseDevice(),
    trapPlacementWorker: () => placement.port.postMessage({ trap: true }),
    trapJobWorker: (index) =>
      at(ports, index % ports.length).postMessage({ trap: true }),
    jobWorkers: ports.length,
    async recover() {
      // The map takes calls while its new device is made.
      core.recovered(await core.newDevice(canvas));
      if (size) core.resize(...size);
    },
    async loadModules(names) {
      await Promise.all(
        names.map(async (name) => {
          const module = await loadModule(build.backend, name);
          if (stopped || loaded.has(name)) return;
          // On the ports the map's work goes over, so a worker has the
          // module before any work that needs it; workers spawned later
          // start with it (workers.ts).
          for (const port of [...ports, placement.port])
            port.postMessage({ module: name, compiled: module });
          loaded.add(name);
        }),
      );
    },
    modules: () => loaded,
    destroy: () => {
      if (stopped) return;
      destroy();
      core.free();
    },
  };
}

/** The slot at `i` of an array the caller keeps one entry per worker in. */
function at<T>(slots: T[], i: number): T {
  const slot = slots[i];
  if (slot === undefined) throw new RangeError(`no worker slot ${i}`);
  return slot;
}
