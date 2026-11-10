// The map worker: the map (engine.ts) drawing into the page's canvas as
// an OffscreenCanvas, with its job and placement workers, off the page's
// thread. The page (link.ts) sends it the map's calls and input, and it
// answers each call and reports after each frame.
//
// Page → worker: `ToWorker` (link.ts), and the page's answers to request
// transform questions (requests.ts). Worker → page: `FromWorker`.
import { Engine, type Report } from "./engine.ts";
import type { FromWorker, ToWorker, WorkerInit } from "./link.ts";
import { preloadStyle } from "./preload.ts";
import {
  askPage,
  setRequestTransform,
  type RequestTransform,
} from "./requests.ts";
import { spawnWorkers } from "./workers.ts";

const post = (message: FromWorker, transfer: Transferable[] = []) =>
  self.postMessage(message, transfer);
/** The page's request transform, asked over this worker's port. */
const page = askPage(self as unknown as MessagePort);
let engine: Engine | undefined;
const CORE = new URL("../pkg/core_bg.wasm", import.meta.url);

/** An answer's pixels, which move to the page. */
const buffers = (value: unknown): Transferable[] => {
  if (ArrayBuffer.isView(value)) return [value.buffer as ArrayBuffer];
  const pixels = (value as { pixels?: unknown } | null)?.pixels;
  return ArrayBuffer.isView(pixels) ? [pixels.buffer as ArrayBuffer] : [];
};

async function start(init: WorkerInit): Promise<void> {
  // Without WebGPU here the page runs the map itself.
  const gpu = (navigator as WorkerNavigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    post({ unsupported: "this browser has no WebGPU in workers" });
    return;
  }
  // The style's first fetches start, the workers load and the GPU adapter
  // wakes while core.wasm compiles.
  preloadStyle(init.preload, init.size[2], init.camera?.zoom);
  const module = WebAssembly.compileStreaming(fetch(CORE));
  module.catch(() => {});
  const workers = spawnWorkers(module, init.jobWorkers);
  void gpu.requestAdapter().catch(() => {});
  try {
    engine = await Engine.start(
      module,
      workers,
      {
        canvas: init.canvas,
        style: init.style,
        options: init.options,
        camera: init.camera,
        constraints: init.constraints,
        size: init.size,
        // Event times are on the page's clock.
        clock: init.origin - performance.timeOrigin,
        drawOnInput: true,
      },
      (report: Report) =>
        post(
          { report },
          report.points ? [report.points.buffer as ArrayBuffer] : [],
        ),
    );
  } catch (e) {
    // No adapter, or no device, is likely this worker's lack: the page
    // tries on its own thread, and says why if it can't either.
    post({ unsupported: String(e instanceof Error ? e.message : e) });
    return;
  }
  post({ ready: engine.state() });
}

self.onmessage = ({ data }: MessageEvent<ToWorker>) => {
  if (page.answer(data)) return;
  const { requestHook } = data as { requestHook?: boolean };
  if (requestHook !== undefined) {
    setRequestTransform(
      requestHook ? (page.hook as RequestTransform) : undefined,
    );
  } else if ("input" in data) {
    try {
      engine?.input(data.input, data.args, data.time);
    } catch (e) {
      console.error(e);
    }
  } else if ("call" in data) {
    const { call, method, args } = data;
    const map = engine!;
    new Promise((resolve) =>
      resolve((map[method] as (...a: unknown[]) => unknown)(...args)),
    ).then(
      (result) =>
        post({ reply: call, result, state: map.state() }, buffers(result)),
      (e) =>
        post({
          reply: call,
          error: String(e instanceof Error ? e.message : e),
          state: map.state(),
        }),
    );
  } else if ("init" in data) {
    void start(data.init);
  }
};
