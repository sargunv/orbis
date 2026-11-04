// The render worker: owns the map and draws into the page's transferred
// canvas, with the camera the page's ui role sends (plan §7.1). Tile jobs
// go to the job workers, and labels to the placement worker, over the
// ports the page set up.
//
// Page → worker: {type: "probe"}; {type: "init", module, canvas, ports,
// placement, style}; {type: "camera", camera}; {type: "resize", width,
// height, ratio}; {type: "edit", id, commands}; {type: "request", id,
// method, args}. Worker → page: {type: "probe", ok, adapter | reason};
// {type: "frame", ack}; {type: "edited", id, generation | error}; {type:
// "reply", id, result | error}; {type: "error", message}.
import init, { createWorkerMap, type OrbisMap } from "../pkg/core.js";
import { frameLoop, renderFrame, type Renderer } from "./frames.ts";

const fail = (e: unknown) => self.postMessage({ type: "error", message: String(e) });
let map: OrbisMap | undefined;
let ready: Promise<void> | undefined;
// The last frame drawn: whether it was idle, and whether a camera drew it
// since this worker's last animation frame.
let last = { idle: true, camera: false };
const draw = (camera: boolean) => {
  const ack = renderFrame(map as unknown as Renderer);
  const transfer = [ack.hitSnapshot?.buffer, ack.elevation?.buffer].filter((b) => b !== undefined);
  self.postMessage({ type: "frame", ack }, transfer);
  last = { idle: ack.idle, camera };
  return !ack.idle;
};
const loop = frameLoop(() => {
  if (!map) return false;
  if (last.camera) {
    last.camera = false;
    return !last.idle;
  }
  return draw(false);
}, fail);
// A camera draws once the messages queued before it are in, so when frames
// fall behind the page's, a backlog of cameras draws only the newest
// rather than every one in turn.
const cameraDraw = new MessageChannel();
let cameraQueued = false;
cameraDraw.port1.onmessage = () => {
  cameraQueued = false;
  try {
    if (map && draw(true)) loop.request();
  } catch (e) {
    fail(e);
  }
};

self.onmessage = async ({ data }: MessageEvent) => {
  switch (data.type) {
    case "probe":
      self.postMessage({ type: "probe", ...await probe() });
      return;
    case "init":
      ready = start(data);
      ready.catch(fail);
      return;
  }
  try {
    await ready;
  } catch {
    return;
  }
  if (!map) return;
  switch (data.type) {
    case "camera":
      map.setCamera(data.camera);
      if (!cameraQueued) {
        cameraQueued = true;
        cameraDraw.port2.postMessage(null);
      }
      break;
    case "resize":
      map.resize(data.width, data.height, data.ratio);
      loop.request();
      break;
    case "edit":
      try {
        const generation = map.edit(data.commands);
        self.postMessage({ type: "edited", id: data.id, generation });
        loop.request();
      } catch (e) {
        self.postMessage({ type: "edited", id: data.id, error: String(e) });
      }
      break;
    case "request":
      try {
        const method = (map as unknown as Record<string, (...args: unknown[]) => unknown>)[data.method];
        const result = await method.apply(map, data.args);
        self.postMessage({ type: "reply", id: data.id, result });
      } catch (e) {
        self.postMessage({ type: "reply", id: data.id, error: String(e) });
      }
      break;
  }
};

/** Whether WebGPU works in this worker (plan M0 exit 7). */
async function probe() {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { ok: false, reason: "no navigator.gpu" };
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no adapter" };
    const { vendor, architecture, description } = adapter.info ?? {};
    return { ok: true, adapter: [vendor, architecture, description].filter(Boolean).join(" ") };
  } catch (e) {
    return { ok: false, reason: String(e) };
  }
}

async function start({ module, canvas, ports, placement, style }: {
  module: WebAssembly.Module; canvas: OffscreenCanvas; ports: MessagePort[];
  placement: MessagePort; style: string;
}) {
  await init({ module_or_path: module });
  map = await createWorkerMap(canvas, style, loop.request, ports.length,
    (worker: number, message: unknown) => ports[worker].postMessage(message),
    (message: unknown, transfer: Transferable[]) => placement.postMessage(message, transfer));
  placement.onmessage = ({ data }) => {
    try {
      map!.receivePlacement(data);
    } catch (e) {
      fail(e);
    }
  };
  for (const port of ports) {
    port.onmessage = ({ data }) => {
      try {
        map!.receiveJob(data);
      } catch (e) {
        fail(e);
      }
    };
  }
  loop.request();
}
