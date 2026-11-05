// The wasm map on the page's thread (plan §7.2). Its tile jobs and labels
// go to job workers and a placement worker (plan §7.1, workers.ts).

import init, { createMap as createCoreMap, type OrbisMap as CoreMap } from "../pkg/core.js";

const COMPAT = new URL("../pkg/compat_bg.wasm", import.meta.url);

/** The renderer role's module for a tier: core.wasm (WebGPU), or
 * compat.wasm (WebGL2, loaded the first time it's needed). Job and
 * placement workers run core.wasm either way. */
async function rendererModule(tier: "modern" | "compat", module: WebAssembly.Module) {
  if (tier === "modern") {
    await init({ module_or_path: module });
    return createCoreMap;
  }
  const compat = await import("../pkg/compat.js");
  await compat.default({ module_or_path: COMPAT });
  return compat.createMap as unknown as typeof createCoreMap;
}
import { renderFrame, type FrameAck, type Renderer } from "./frames.ts";
import type { Hit } from "./types.ts";
import { spawnWorkers } from "./workers.ts";

export interface HostCallbacks {
  /** The map has new data to draw. */
  wake(): void;
  error(e: unknown): void;
}

/** Calls the renderer answers asynchronously, by name. */
export type RequestMethod = "getStyle" | "semanticSnapshot" | "queryRendered" | "querySource"
  | "addImage" | "updateImage" | "removeImage" | "hasImage" | "retryFailedTiles";

export interface Host {
  setCamera(camera: Float64Array): void;
  resize(width: number, height: number, ratio: number): void;
  edit(commands: string): Promise<number>;
  request(method: RequestMethod, ...args: unknown[]): Promise<unknown>;
  /** The topmost feature under a CSS pixel, from the last frame. */
  hitTest(x: number, y: number): Hit | null;
  render(): FrameAck;
  /** The wasm map: for benches and tests. */
  readonly core: CoreMap;
  /** Loses the GPU on purpose, to exercise recovery; with `switchTier`,
   * recovery skips the same tier. */
  loseDevice(switchTier: boolean): void;
  /** Draws on a new GPU after a loss: the same tier on the same canvas
   * where it works, else WebGL2 in a fresh canvas `replace` puts in the
   * old one's place. Resolves to the new tier. */
  recover(replace: (canvas: HTMLCanvasElement) => void, camera: Float64Array): Promise<"modern" | "compat">;
  destroy(): void;
}

/** The map on this thread, drawing into `canvas` at `tier`. */
export async function startMain(module: WebAssembly.Module, canvas: HTMLCanvasElement, style: string,
  jobWorkers: number, callbacks: HostCallbacks, tier: "modern" | "compat"): Promise<Host> {
  const receive = (apply: (data: unknown) => void) => ({ data }: MessageEvent) => {
    try {
      apply(data);
    } catch (e) {
      callbacks.error(e);
    }
  };
  // A map with its job and placement workers.
  const start = async (tier: "modern" | "compat", canvas: HTMLCanvasElement, style: string) => {
    const { workers, ports, placement } = spawnWorkers(module, jobWorkers);
    const create = await rendererModule(tier, module);
    const core = await create(canvas, style, callbacks.wake, ports.length,
      (worker: number, message: unknown) => ports[worker].postMessage(message),
      (message: unknown, transfer: Transferable[]) => placement.postMessage(message, transfer));
    placement.onmessage = receive((data) => core.receivePlacement(data));
    for (const port of ports) port.onmessage = receive((data) => core.receiveJob(data));
    return { core, workers };
  };
  let { core, workers } = await start(tier, canvas, style);
  let size: [number, number, number] | undefined;
  let skipTier = false;
  // `core.recover` holds the map across its await: calls in between wait
  // for it (wasm-bindgen refuses them), as the latest size and camera.
  let recovering = false;
  let heldCamera: Float64Array | undefined;
  // A lost WebGL context is seen by the next frame, which this asks for.
  const lost = (e: Event) => { e.preventDefault(); callbacks.wake(); };
  canvas.addEventListener("webglcontextlost", lost);
  const stop = () => {
    for (const w of workers) w.terminate();
    core.free();
  };
  return {
    get core() { return core; },
    setCamera: (camera) => { if (recovering) heldCamera = camera; else core.setCamera(camera); },
    resize: (width, height, ratio) => {
      size = [width, height, ratio];
      if (!recovering) core.resize(width, height, ratio);
    },
    edit: async (commands) => core.edit(commands),
    request: async (method, ...args) =>
      (core as unknown as Record<string, (...a: unknown[]) => unknown>)[method](...args),
    hitTest: (x, y) => core.hitTest(x, y) as Hit | null,
    render: () => renderFrame(core as unknown as Renderer),
    loseDevice: (switchTier) => { skipTier = switchTier; core.loseDevice(); },
    async recover(replace, camera) {
      try {
        if (skipTier) throw new Error("switching tiers");
        recovering = true;
        try {
          return await core.recover(canvas) as "modern" | "compat";
        } finally {
          recovering = false;
          if (size) core.resize(...size);
          if (heldCamera) core.setCamera(heldCamera);
          heldCamera = undefined;
        }
      } catch {
        skipTier = false;
        // The canvas keeps its old context; WebGL2 needs another, and the
        // map a new compat.wasm instance with its own workers.
        const fresh = document.createElement("canvas");
        fresh.style.cssText = canvas.style.cssText;
        fresh.width = canvas.width;
        fresh.height = canvas.height;
        const json = core.getStyle();
        stop();
        ({ core, workers } = await start("compat", fresh,
          `data:application/json,${encodeURIComponent(json)}`));
        core.setCamera(camera);
        if (size) core.resize(...size);
        canvas.removeEventListener("webglcontextlost", lost);
        canvas = fresh;
        canvas.addEventListener("webglcontextlost", lost);
        replace(fresh);
        return "compat";
      }
    },
    destroy: stop,
  };
}
