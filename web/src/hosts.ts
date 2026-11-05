// The wasm map on the page's thread, with its job workers and
// placement worker (workers.ts), and its recovery from a lost GPU.

import init, { createMap as createCoreMap, reloadRegions, setRequestHook, type OrbisMap as Core } from "../pkg/core.js";
import { onRegionsChanged } from "./regions.ts";
import { answerWorker, connectModule, connectWorker } from "./requests.ts";
import { spawnPlacement, spawnWorkers } from "./workers.ts";

export type { Core };

/** GL JS's map options the core takes at creation (`MapOptions`). */
export interface CoreOptions {
  fadeDuration?: number;
  crossSourceCollisions?: boolean;
  localIdeographFontFamily?: string | false;
  antialias?: boolean;
  terrainSkirts?: boolean;
}

export interface HostCallbacks {
  /** The map has new data to draw. */
  wake(): void;
  error(e: unknown): void;
  /** Sets up the new wasm map. */
  setup(core: Core): void;
}

/** The wasm map and what it runs on. */
export interface Host {
  readonly core: Core;
  resize(width: number, height: number, ratio: number): void;
  /** Loses the GPU on purpose, to exercise recovery. */
  loseDevice(): void;
  /** Traps the placement worker's wasm, as a bug would, to exercise its
   * replacement. */
  trapPlacementWorker(): void;
  /** Draws on a new GPU after a loss, on the same canvas. */
  recover(): Promise<void>;
  destroy(): void;
}

/** The map on this thread, drawing into `canvas`. */
export async function startMain(module: WebAssembly.Module, canvas: HTMLCanvasElement, style: string,
  options: CoreOptions, jobWorkers: number, callbacks: HostCallbacks): Promise<Host> {
  const receive = (apply: (data: unknown) => void) => ({ data }: MessageEvent) => {
    try {
      apply(data);
    } catch (e) {
      callbacks.error(e);
    }
  };
  // Replies still on their way when the map stops find it gone.
  let stopped = false;
  const reply = (apply: (data: unknown) => void) => receive((data) => { if (!stopped) apply(data); });
  const spawned = spawnWorkers(module, jobWorkers);
  const { ports } = spawned;
  const workers = spawned.workers.slice(0, -1);
  let placement = { port: spawned.placement, worker: spawned.workers[spawned.workers.length - 1] };
  await init({ module_or_path: module });
  connectModule(setRequestHook);
  const core = await createCoreMap(canvas, style, options, callbacks.wake, ports.length,
    (worker: number, message: unknown) => ports[worker].postMessage(message),
    (message: unknown, transfer: Transferable[]) => placement.port.postMessage(message, transfer));
  // A placement worker whose wasm trapped says so and closes; one that
  // dies fires `error`. Either way another takes its place, and the map
  // sends it what it places from.
  const listen = () => {
    const current = placement;
    const replace = (why: string) => {
      if (placement !== current) return;
      current.worker.terminate();
      placement = spawnPlacement(module);
      listen();
      core.placementWorkerLost(why);
    };
    current.port.onmessage = reply((data) => {
      const lost = (data as { lost?: string }).lost;
      if (lost !== undefined) replace(lost);
      else core.receivePlacement(data);
    });
    current.worker.addEventListener("error", (e) => { e.preventDefault(); if (!stopped) replace(e.message || "the worker died"); });
  };
  listen();
  // A job worker whose wasm trapped says so and closes; one that dies
  // fires `error`. Either way its requests fail and are retried on the
  // others.
  const released = ports.map(connectWorker);
  ports.forEach((port, i) => {
    port.onmessage = reply((data) => {
      const lost = (data as { lost?: string }).lost;
      if (lost !== undefined) core.jobWorkerLost(i, lost);
      else if (!answerWorker(port, data)) core.receiveJob(data);
    });
    workers[i].addEventListener("error", (e) => { e.preventDefault(); if (!stopped) core.jobWorkerLost(i, e.message || "the worker died"); });
  });
  const regions = onRegionsChanged(reloadRegions);
  callbacks.setup(core);
  let size: [number, number, number] | undefined;
  return {
    core,
    resize: (width, height, ratio) => {
      size = [width, height, ratio];
      core.resize(width, height, ratio);
    },
    loseDevice: () => core.loseDevice(),
    trapPlacementWorker: () => placement.port.postMessage({ trap: true }),
    async recover() {
      await core.recover(canvas);
      if (size) core.resize(...size);
    },
    destroy: () => {
      stopped = true;
      for (const port of [...ports, placement.port]) port.close();
      regions();
      for (const r of released) r();
      for (const w of [...workers, placement.worker]) w.terminate();
      core.free();
    },
  };
}
