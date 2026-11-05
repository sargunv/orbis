// The wasm map on the page's thread (plan §7.2), with its job workers and
// placement worker (plan §7.1, workers.ts), and its recovery from a lost
// GPU (plan §8.8): the same tier again where it works, else WebGL2.

import init, { createMap as createCoreMap, reloadRegions, setRequestHook, type OrbisMap as Core } from "../pkg/core.js";
import { onRegionsChanged } from "./regions.ts";
import { answerWorker, connectModule, connectWorker } from "./requests.ts";
import { spawnPlacement, spawnWorkers } from "./workers.ts";

export type { Core };

const COMPAT = new URL("../pkg/compat_bg.wasm", import.meta.url);

/** The renderer's module for a tier: core.wasm (WebGPU), or compat.wasm
 * (WebGL2, loaded the first time it's needed). Job and placement workers
 * run core.wasm either way. */
async function rendererModule(tier: "modern" | "compat", module: WebAssembly.Module) {
  if (tier === "modern") {
    await init({ module_or_path: module });
    connectModule(setRequestHook);
    return { create: createCoreMap, reload: reloadRegions };
  }
  const compat = await import("../pkg/compat.js");
  await compat.default({ module_or_path: COMPAT });
  connectModule(compat.setRequestHook);
  return { create: compat.createMap as unknown as typeof createCoreMap, reload: compat.reloadRegions };
}

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
  /** Sets up a new wasm map: the first, and one on another tier. */
  setup(core: Core): void;
}

/** The wasm map and what it runs on; `core` changes when a recovery
 * moves the map to another tier. */
export interface Host {
  readonly core: Core;
  resize(width: number, height: number, ratio: number): void;
  /** Loses the GPU on purpose, to exercise recovery; with `switchTier`,
   * recovery skips the same tier. */
  loseDevice(switchTier: boolean): void;
  /** Traps the placement worker's wasm, as a bug would, to exercise its
   * replacement. */
  trapPlacementWorker(): void;
  /** Draws on a new GPU after a loss: the same tier on the same canvas
   * where it works, else WebGL2 in a fresh canvas `replace` puts in the
   * old one's place. Resolves to the new tier. */
  recover(replace: (canvas: HTMLCanvasElement) => void): Promise<"modern" | "compat">;
  destroy(): void;
}

/** The map on this thread, drawing into `canvas` at `tier`. */
export async function startMain(module: WebAssembly.Module, canvas: HTMLCanvasElement, style: string,
  options: CoreOptions, jobWorkers: number, callbacks: HostCallbacks, tier: "modern" | "compat"): Promise<Host> {
  const receive = (apply: (data: unknown) => void) => ({ data }: MessageEvent) => {
    try {
      apply(data);
    } catch (e) {
      callbacks.error(e);
    }
  };
  // A map with its job and placement workers.
  const start = async (tier: "modern" | "compat", canvas: HTMLCanvasElement, style: string) => {
    // Replies still on their way when the map stops find it gone.
    let stopped = false;
    const reply = (apply: (data: unknown) => void) => receive((data) => { if (!stopped) apply(data); });
    const spawned = spawnWorkers(module, jobWorkers);
    const { ports } = spawned;
    const workers = spawned.workers.slice(0, -1);
    let placement = { port: spawned.placement, worker: spawned.workers[spawned.workers.length - 1] };
    const { create, reload } = await rendererModule(tier, module);
    const core = await create(canvas, style, options, callbacks.wake, ports.length,
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
    const regions = onRegionsChanged(reload);
    callbacks.setup(core);
    const stop = () => {
      stopped = true;
      for (const port of [...ports, placement.port]) port.close();
      regions();
      for (const r of released) r();
      for (const w of [...workers, placement.worker]) w.terminate();
      core.free();
    };
    return { core, stop, trap: () => placement.port.postMessage({ trap: true }) };
  };
  let { core, stop, trap } = await start(tier, canvas, style);
  let size: [number, number, number] | undefined;
  let skipTier = false;
  // A lost WebGL context is seen by the next frame, which this asks for.
  const lost = (e: Event) => { e.preventDefault(); callbacks.wake(); };
  canvas.addEventListener("webglcontextlost", lost);
  return {
    get core() { return core; },
    resize: (width, height, ratio) => {
      size = [width, height, ratio];
      core.resize(width, height, ratio);
    },
    loseDevice: (switchTier) => { skipTier = switchTier; core.loseDevice(); },
    trapPlacementWorker: () => trap(),
    async recover(replace) {
      try {
        if (skipTier) throw new Error("switching tiers");
        const tier = await core.recover(canvas) as "modern" | "compat";
        if (size) core.resize(...size);
        return tier;
      } catch {
        skipTier = false;
        // The canvas keeps its old context; WebGL2 needs another, and the
        // map a new compat.wasm instance with its own workers, at the
        // camera the lost one had.
        const fresh = document.createElement("canvas");
        fresh.style.cssText = canvas.style.cssText;
        fresh.width = canvas.width;
        fresh.height = canvas.height;
        // The map's own state: its style as edited, its camera, and what the
        // app set at run time.
        const json = core.getStyle();
        const state = core.exportState();
        const c = core.camera();
        const camera = { center: [c.lon, c.lat], zoom: c.zoom, bearing: c.bearing, pitch: c.pitch, roll: c.roll, elevation: c.elevation };
        stop();
        ({ core, stop, trap } = await start("compat", fresh, `data:application/json,${encodeURIComponent(json)}`));
        core.jumpTo(camera);
        core.importState(state);
        if (size) core.resize(...size);
        canvas.removeEventListener("webglcontextlost", lost);
        canvas = fresh;
        canvas.addEventListener("webglcontextlost", lost);
        replace(fresh);
        return "compat";
      }
    },
    destroy: () => { canvas.removeEventListener("webglcontextlost", lost); stop(); },
  };
}
