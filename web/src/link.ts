// How the page's side of a map (map.ts) reaches the map itself
// (engine.ts): by message to the map worker (map-worker.ts), or by direct
// calls where the map runs on the page. Either way calls answer with
// promises, in order, and the map reports after its frames.

import type { Engine, InputKind, MapState, Report } from "./engine.ts";
import type { CoreOptions } from "./hosts.ts";
import { answerWorker, connectWorker } from "./requests.ts";
import type { CameraOptions, Constraints } from "./types.ts";

/** The map's methods the page calls. */
type Calls = Omit<{ [K in keyof Engine]: Engine[K] }, "state" | "input" | "wake">;
export type Method = keyof Calls;
type Result<K extends Method> = Awaited<ReturnType<Calls[K]>>;

export interface Link {
  /** Whether the map runs in the map worker. */
  readonly worker: boolean;
  /** The map's state: now on the page; from a worker, as of its last
   * report or answer. */
  state(): MapState;
  call<K extends Method>(method: K, ...args: Parameters<Calls[K]>): Promise<Result<K>>;
  /** Input at the page's event time, without an answer. */
  input(kind: InputKind, args: (number | string)[], time: number): void;
  /** Sends reports to `onreport`, those that came before it first. */
  listen(onreport: (report: Report) => void): void;
  destroy(): void;
}

/** Reports, held until the page listens. */
function reports() {
  const held: Report[] = [];
  let target: ((report: Report) => void) | undefined;
  return {
    deliver: (report: Report) => { if (target) target(report); else held.push(report); },
    listen: (onreport: (report: Report) => void) => {
      target = onreport;
      for (const report of held.splice(0)) onreport(report);
    },
  };
}

/** A link to an engine on this thread, made with `start(report)`. */
export async function directLink(start: (report: (r: Report) => void) => Promise<Engine>): Promise<Link> {
  const { deliver, listen } = reports();
  const engine = await start(deliver);
  return {
    worker: false,
    state: () => engine.state(),
    call(method, ...args) {
      try {
        return Promise.resolve((engine[method] as (...a: unknown[]) => never)(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    },
    input: (kind, args, time) => engine.input(kind, args, time),
    listen,
    destroy: () => engine.remove(),
  };
}

/** What the map worker starts from. */
export interface WorkerInit {
  canvas: OffscreenCanvas;
  /** The style's URL (a `data:` URL for a document). */
  style: string;
  /** What to fetch ahead: the style as the app gave it. */
  preload: string | object;
  options: CoreOptions;
  camera?: CameraOptions;
  constraints?: Constraints;
  size: [number, number, number];
  /** How many job workers, as the page counts its cores. */
  jobWorkers: number;
  /** The page's `performance.timeOrigin`: event times are on its clock. */
  origin: number;
}

/** The map worker's messages to the page. */
export type FromWorker =
  | { ready: MapState }
  | { unsupported: string }
  | { failed: string }
  | { reply: number; result?: unknown; error?: string; state: MapState }
  | { report: Report };

/** The page's messages to the map worker (beside request transform
 * answers, requests.ts). */
export type ToWorker =
  | { init: WorkerInit }
  | { call: number; method: Method; args: unknown[] }
  | { input: InputKind; args: (number | string)[]; time: number };

/**
 * Starts the map in a worker, drawing into `canvas`'s offscreen self,
 * where it compiles core.wasm and starts its job and placement workers.
 * Resolves to its link, or to why the worker can't draw (no WebGPU
 * there): the page can still run the map itself, on another canvas.
 */
export function workerLink(canvas: HTMLCanvasElement, init: Omit<WorkerInit, "canvas">): Promise<Link | string> {
  const worker = new Worker(new URL("./map-worker.ts", import.meta.url), { type: "module" });
  const offscreen = canvas.transferControlToOffscreen();
  const post = (message: ToWorker, transfer: Transferable[] = []) => worker.postMessage(message, transfer);
  // The page's request transform reaches the worker's fetches through it.
  const release = connectWorker(worker);
  post({ init: { ...init, canvas: offscreen } }, [offscreen]);
  const { deliver, listen } = reports();
  let state: MapState | undefined;
  let next = 0;
  const waiting = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();
  let stopped = false;
  const link: Link = {
    worker: true,
    state: () => state!,
    call(method, ...args) {
      if (stopped) return Promise.reject(new Error("the map was removed"));
      const call = next++;
      // Video frames move to the worker rather than copy.
      post({ call, method, args }, args.filter((a) => a instanceof ImageBitmap) as ImageBitmap[]);
      return new Promise((resolve, reject) => waiting.set(call, { resolve, reject }));
    },
    input: (kind, args, time) => { if (!stopped) post({ input: kind, args, time }); },
    listen,
    destroy() {
      if (stopped) return;
      stopped = true;
      release();
      worker.terminate();
      for (const { reject } of waiting.values()) reject(new Error("the map was removed"));
      waiting.clear();
    },
  };
  return new Promise((resolve, reject) => {
    worker.onerror = (e) => {
      e.preventDefault();
      const message = e.message || "the map worker died";
      if (state) deliver({ state, events: [{ type: "error", data: { message } }] });
      else {
        link.destroy();
        reject(new Error(message));
      }
    };
    worker.onmessage = ({ data }: MessageEvent<FromWorker>) => {
      if (answerWorker(worker, data)) return;
      if ("report" in data) {
        state = data.report.state;
        deliver(data.report);
      } else if ("reply" in data) {
        state = data.state;
        const call = waiting.get(data.reply);
        waiting.delete(data.reply);
        if (data.error !== undefined) call?.reject(new Error(data.error));
        else call?.resolve(data.result as never);
      } else if ("ready" in data) {
        state = data.ready;
        resolve(link);
      } else {
        link.destroy();
        if ("unsupported" in data) resolve(data.unsupported);
        else reject(new Error(data.failed));
      }
    };
  });
}
