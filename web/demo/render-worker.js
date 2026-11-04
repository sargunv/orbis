// The render worker: owns the map and draws into the page's transferred
// canvas. Tile jobs go to the job workers, and labels to the placement
// worker, over the ports the page set up.
//
// Page → worker: {type: "probe"}; {type: "init", module, canvas, ports,
// placement, style}; input as {type: "pan" | "zoomAt" | "rotatePitch" | "jumpTo" | "resize", args};
// {type: "edit", id, commands}, a transaction as a JSON array of map calls.
// Worker → page: {type: "probe", ok, adapter | reason}; {type: "frame",
// ack}; {type: "edited", id, generation | error}; {type: "error", message}.
import init, { createWorkerMap } from "./pkg/core.js";
import { frameLoop } from "./frames.js";

const INPUT = new Set(["pan", "zoomAt", "rotatePitch", "jumpTo", "resize", "getStyle", "hitTest", "semanticSnapshot"]);
// Queries: they change nothing, so draw no frame.
const QUERIES = new Set(["getStyle", "hitTest", "semanticSnapshot"]);
const fail = (e) => self.postMessage({ type: "error", message: String(e) });
const frames = frameLoop((ack) => self.postMessage({ type: "frame", ack },
  ack.hitSnapshot ? [ack.hitSnapshot.buffer] : []), fail);
let map;
let ready;

self.onmessage = async ({ data }) => {
  if (data.type === "probe") {
    self.postMessage({ type: "probe", ...await probe() });
  } else if (data.type === "init") {
    ready = start(data);
    ready.catch(fail);
  } else if (data.type === "edit") {
    try {
      await ready;
      const generation = map.edit(data.commands);
      self.postMessage({ type: "edited", id: data.id, generation });
      frames.request();
    } catch (e) {
      self.postMessage({ type: "edited", id: data.id, error: String(e) });
    }
  } else if (INPUT.has(data.type)) {
    try {
      await ready;
    } catch {
      return;
    }
    try {
      const result = map[data.type](...data.args);
      if (data.request !== undefined) self.postMessage({ type: "reply", request: data.request, result });
      if (!QUERIES.has(data.type)) frames.request();
    } catch (e) {
      if (data.request !== undefined) self.postMessage({ type: "reply", request: data.request, error: String(e) });
      else fail(e);
    }
  }
};

// Whether WebGPU works in this worker (plan M0 exit 7).
async function probe() {
  if (!navigator.gpu) return { ok: false, reason: "no navigator.gpu" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no adapter" };
    const { vendor, architecture, description } = adapter.info ?? {};
    return { ok: true, adapter: [vendor, architecture, description].filter(Boolean).join(" ") };
  } catch (e) {
    return { ok: false, reason: String(e) };
  }
}

async function start({ module, canvas, ports, placement, style }) {
  await init({ module_or_path: module });
  map = await createWorkerMap(canvas, style, frames.request, ports.length,
    (worker, message) => ports[worker].postMessage(message),
    (message, transfer) => placement.postMessage(message, transfer));
  placement.onmessage = ({ data }) => {
    try {
      map.receivePlacement(data);
    } catch (e) {
      fail(e);
    }
  };
  for (const port of ports) {
    port.onmessage = ({ data }) => {
      try {
        map.receiveJob(data);
      } catch (e) {
        fail(e);
      }
    };
  }
  frames.start(map);
}
