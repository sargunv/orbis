// The map worker: the map (engine.ts) drawing into the page's canvas as
// an OffscreenCanvas, with its job and placement workers, off the page's
// thread. The page (link.ts) sends it the map's calls and input, and it
// answers each call and reports after each frame.
//
// Page → worker: `ToWorker` (link.ts), and the page's answers to request
// transform questions (requests.ts). Worker → page: `FromWorker`.
import { pickBuild } from "./builds.ts";
import { Engine, type MapState, type Report } from "./engine.ts";
import type { FromWorker, ToWorker, WorkerInit } from "./link.ts";
import { preloadStyle } from "./preload.ts";
import {
  askPage,
  setRequestTransform,
  type RequestTransform,
} from "./requests.ts";
import { isTrap } from "./trap.ts";
import { spawnWorkers } from "./workers.ts";

const post = (message: FromWorker, transfer: Transferable[] = []) =>
  self.postMessage(message, transfer);
/** The page's request transform, asked over this worker's port. */
const page = askPage(self as unknown as MessagePort);
let engine: Engine | undefined;

/** An answer's pixels, which move to the page. */
const buffers = (value: unknown): Transferable[] => {
  if (ArrayBuffer.isView(value)) return [value.buffer as ArrayBuffer];
  const pixels = (value as { pixels?: unknown } | null)?.pixels;
  return ArrayBuffer.isView(pixels) ? [pixels.buffer as ArrayBuffer] : [];
};

async function start(init: WorkerInit): Promise<void> {
  // Without WebGPU here, a page that has it runs the map itself; one
  // without, or a map that asks for WebGL 2, draws WebGL 2 here.
  const gpu = (navigator as WorkerNavigator & { gpu?: GPU }).gpu;
  if (!gpu && init.pageWebgpu && init.backend !== "webgl2") {
    post({ unsupported: "this browser has no WebGPU in workers" });
    return;
  }
  // The style's first fetches start and the workers load while the build
  // is picked and compiles.
  preloadStyle(init.preload, init.size[2], init.camera?.zoom);
  const build = pickBuild(gpu, init.backend);
  build.catch(() => {});
  const workers = spawnWorkers(build, init.jobWorkers);
  let ready: MapState;
  try {
    engine = await Engine.start(
      build,
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
      },
      (report: Report) =>
        post(
          { report },
          report.points ? [report.points.buffer as ArrayBuffer] : [],
        ),
    );
    ready = engine.state();
  } catch (e) {
    const message = String(e instanceof Error ? e.message : e);
    // No adapter, or no device, is likely this worker's lack: the page
    // tries on its own thread, and says why if it can't either. A trap is
    // a bug that thread would hit too: the map fails to start.
    post(isTrap(e) ? { failed: message } : { unsupported: message });
    return;
  }
  post({ ready, backend: engine.backend });
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
