// The render worker: owns the map and draws into the page's transferred
// canvas, with the camera the page's ui role sends (plan §7.1). Tile jobs
// go to job workers, and labels to a placement worker, that it spawns.
//
// Page → worker: {type: "probe"}; {type: "init", module, canvas, style,
// jobWorkers}; {type: "camera", camera}; {type: "resize", width,
// height, ratio}; {type: "acks", every}; {type: "edit", id, commands};
// {type: "request", id, method, args}. Worker → page: {type: "probe", ok, adapter | reason};
// {type: "frame", ack}; {type: "edited", id, generation | error}; {type:
// "reply", id, result | error}; {type: "error", message}.
import init, { createWorkerMap, type OrbisMap } from "../pkg/core.js";
import { ackIdle, frameLoop, renderFrame, type FrameAck, type Renderer } from "./frames.ts";
import { spawnWorkers } from "./workers.ts";

const fail = (e: unknown) => self.postMessage({ type: "error", message: String(e) });
let map: OrbisMap | undefined;
let ready: Promise<void> | undefined;
// The last frame drawn: whether it was idle, and whether a camera drew it
// since this worker's last animation frame.
let last = { idle: true, camera: false };
// Frames go to the page when it needs them: every one while it has
// annotations or `render` listeners; otherwise those with events, picking
// or terrain changes or a change of idleness, and the last before a pause
// (`pending` until then), which shows the camera the map rests at.
let every = false;
let pending: FrameAck | undefined;
let sentIdle = false;
// Sending transfers the ack's arrays: read them first.
const send = (ack: FrameAck) => {
  sentIdle = ackIdle(ack);
  pending = undefined;
  const transfer = [ack.stats.buffer, ack.hits?.buffer, ack.elevation?.buffer].filter((b) => b !== undefined);
  self.postMessage({ type: "frame", ack }, transfer);
};
const draw = (camera: boolean) => {
  const ack = renderFrame(map as unknown as Renderer);
  const idle = ackIdle(ack);
  last = { idle, camera };
  if (every || ack.events || ack.hits || ack.elevation || ack.warnings || idle !== sentIdle) send(ack);
  else pending = ack;
  return !idle;
};
// Whether something asked for a frame, rather than one pending to send.
let wanted = false;
const loop = frameLoop(() => {
  if (!map) return false;
  if (last.camera) {
    last.camera = false;
    return !last.idle || pending !== undefined;
  }
  if (!wanted && last.idle) {
    // A frame without a new camera: the last one drawn goes out.
    if (pending) send(pending);
    return false;
  }
  wanted = false;
  return draw(false) || pending !== undefined;
}, fail);
const request = () => {
  wanted = true;
  loop.request();
};
// A camera draws once the messages queued before it are in, so when frames
// fall behind the page's, a backlog of cameras draws only the newest
// rather than every one in turn.
const cameraDraw = new MessageChannel();
let cameraQueued = false;
cameraDraw.port1.onmessage = () => {
  cameraQueued = false;
  try {
    if (map && (draw(true) || pending)) loop.request();
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
      request();
      break;
    case "acks":
      every = data.every;
      if (every && pending) send(pending);
      break;
    case "edit":
      try {
        const generation = map.edit(data.commands);
        self.postMessage({ type: "edited", id: data.id, generation });
        request();
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

async function start({ module, canvas, style, jobWorkers }: {
  module: WebAssembly.Module; canvas: OffscreenCanvas; style: string; jobWorkers: number;
}) {
  const { ports, placement } = spawnWorkers(module, jobWorkers);
  await init({ module_or_path: module });
  map = await createWorkerMap(canvas, style, request, ports.length,
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
  request();
}
