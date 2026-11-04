// The two places a map can draw (plan §7.2), behind one interface: on the
// page's thread, or in a render worker. Either way its tile jobs and labels
// go to job workers and a placement worker (plan §7.1).

import init, { createMap as createCoreMap, type OrbisMap as CoreMap } from "../pkg/core.js";
import { renderFrame, type FrameAck, type Renderer } from "./frames.ts";
import type { Hit } from "./types.ts";

export interface HostCallbacks {
  /** A frame the worker drew (worker hosts; the page draws main hosts' itself). */
  frame(ack: FrameAck): void;
  /** The map has new data to draw (main hosts). */
  wake(): void;
  error(e: unknown): void;
}

/** Calls the renderer answers asynchronously, by name. */
export type RequestMethod = "getStyle" | "semanticSnapshot" | "queryRendered" | "querySource";

export interface Host {
  readonly thread: "main" | "worker";
  setCamera(camera: Float64Array): void;
  resize(width: number, height: number, ratio: number): void;
  edit(commands: string): Promise<number>;
  request(method: RequestMethod, ...args: unknown[]): Promise<unknown>;
  /** Main hosts answer hit tests from the last frame; worker hosts leave it to the ui role. */
  hitTest?(x: number, y: number): Hit | null;
  /** Main hosts draw on the page's frames. */
  render?(): FrameAck;
  /** The wasm map, in main-thread mode: for benches and tests. */
  readonly core?: CoreMap;
  destroy(): void;
}

const JOB_WORKERS = Math.min(Math.max((navigator.hardwareConcurrency ?? 4) - 2, 1), 4);

/** Spawns a render worker and asks whether WebGPU works there. */
export async function probeWorker(): Promise<{ worker: Worker; ok: boolean; adapter?: string; reason?: string }> {
  const worker = new Worker(new URL("./render-worker.ts", import.meta.url), { type: "module" });
  const probe = await new Promise<{ ok: boolean; adapter?: string; reason?: string }>((resolve) => {
    worker.onmessage = ({ data }) => resolve(data);
    worker.onerror = (e) => resolve({ ok: false, reason: e.message ?? "worker failed" });
    worker.postMessage({ type: "probe" });
  });
  return { worker, ...probe };
}

/** Spawns the job workers and the placement worker; the map talks to them
 * over `ports` and `placement`. */
function spawnWorkers(module: WebAssembly.Module) {
  const workers: Worker[] = [];
  const ports: MessagePort[] = [];
  for (let i = 0; i < JOB_WORKERS; i++) {
    const channel = new MessageChannel();
    const job = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module" });
    job.postMessage({ module, port: channel.port2 }, [channel.port2]);
    ports.push(channel.port1);
    workers.push(job);
  }
  const placement = new MessageChannel();
  const placer = new Worker(new URL("./placement-worker.ts", import.meta.url), { type: "module" });
  placer.postMessage({ module, port: placement.port2 }, [placement.port2]);
  workers.push(placer);
  return { workers, ports, placement: placement.port1 };
}

/** The map on this thread, drawing into `canvas`. */
export async function startMain(module: WebAssembly.Module, canvas: HTMLCanvasElement, style: string,
  callbacks: HostCallbacks): Promise<Host> {
  const { workers, ports, placement } = spawnWorkers(module);
  await init({ module_or_path: module });
  const core = await createCoreMap(canvas, style, callbacks.wake, ports.length,
    (worker: number, message: unknown) => ports[worker].postMessage(message),
    (message: unknown, transfer: Transferable[]) => placement.postMessage(message, transfer));
  const receive = (apply: (data: unknown) => void) => ({ data }: MessageEvent) => {
    try {
      apply(data);
    } catch (e) {
      callbacks.error(e);
    }
  };
  placement.onmessage = receive((data) => core.receivePlacement(data));
  for (const port of ports) port.onmessage = receive((data) => core.receiveJob(data));
  return {
    thread: "main",
    core,
    setCamera: (camera) => core.setCamera(camera),
    resize: (width, height, ratio) => core.resize(width, height, ratio),
    edit: async (commands) => core.edit(commands),
    request: async (method, ...args) =>
      (core as unknown as Record<string, (...a: unknown[]) => unknown>)[method](...args),
    hitTest: (x, y) => core.hitTest(x, y) as Hit | null,
    render: () => renderFrame(core as unknown as Renderer),
    destroy: () => {
      for (const w of workers) w.terminate();
      core.free();
    },
  };
}

/** Hands `canvas` to the probed render worker and spawns the job workers
 * and the placement worker, wired to it directly. */
export function startWorker(module: WebAssembly.Module, worker: Worker, canvas: HTMLCanvasElement,
  style: string, callbacks: HostCallbacks): Host {
  const spawned = spawnWorkers(module);
  const workers = [worker, ...spawned.workers];
  const { ports, placement } = spawned;
  const offscreen = canvas.transferControlToOffscreen();
  // Calls in flight, by id, until the render worker answers.
  const pending = new Map<number, { resolve(value: unknown): void; reject(e: Error): void }>();
  let next = 0;
  const call = (message: object, transfer: Transferable[] = []) => new Promise<unknown>((resolve, reject) => {
    const id = next++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...message, id }, transfer);
  });
  worker.onmessage = ({ data }) => {
    if (data.type === "frame") callbacks.frame(data.ack);
    else if (data.type === "error") callbacks.error(data.message);
    else if (data.type === "reply" || data.type === "edited") {
      const call = pending.get(data.id);
      if (!call) return;
      pending.delete(data.id);
      if (data.error !== undefined) call.reject(new Error(data.error));
      else call.resolve(data.type === "edited" ? data.generation : data.result);
    }
  };
  worker.onerror = (e) => callbacks.error(e.message);
  worker.postMessage({ type: "init", module, canvas: offscreen, ports, placement, style },
    [offscreen, ...ports, placement]);
  return {
    thread: "worker",
    setCamera: (camera) => worker.postMessage({ type: "camera", camera }),
    resize: (width, height, ratio) => worker.postMessage({ type: "resize", width, height, ratio }),
    edit: (commands) => call({ type: "edit", commands }) as Promise<number>,
    request: (method, ...args) => call({ type: "request", method, args }),
    destroy: () => { for (const w of workers) w.terminate(); },
  };
}
