// The web demo: the same map as `cargo run -p orb-demo`, drawing through
// WebGPU. In worker mode (the default where it works) this page is the ui
// role: it takes input and shows frame acks, while a render worker owns the
// map and job workers lay out tiles (plan §7.1). `?thread=main` runs
// everything on this thread.
import init, { createMap, PublishedHitIndex } from "./pkg/core.js";
import { frameLoop } from "./frames.js";
import { editApi } from "./edit.js";
import { demoStyle } from "./styles.js";
import { loadShaper } from "./shaper.js";

const params = new URLSearchParams(location.search);
const STYLE = demoStyle(params);
const THREAD = params.get("thread"); // "main", "worker", or automatic
const JOB_WORKERS = Math.min(Math.max((navigator.hardwareConcurrency ?? 4) - 2, 1), 4);

const canvas = document.getElementById("map");
const hud = document.getElementById("hud");
let status = "";
export let latestFrame;
let publishedHits;

const kib = (bytes) => `${(bytes / 1024).toFixed(1)} KiB`;
function showFrame(ack) {
  if (publishedHits) {
    if (ack.hitSnapshot) publishedHits.update(ack.hitSnapshot);
    publishedHits.advance(ack.generation);
  }
  latestFrame = ack;
  window.dispatchEvent(new CustomEvent("orbis-frame", {detail:ack}));
  for (const warning of ack.warnings) console.warn(warning);
  const jobs = ack.jobsDone ? ` · ${ack.jobsDone} jobs on ${JOB_WORKERS} workers · ` +
    `${kib(ack.headerBytes)} headers into wasm` : "";
  hud.textContent = `${status}\n` +
    `z${ack.zoom.toFixed(2)} · ${ack.ms.toFixed(1)} ms CPU · ${ack.draws} draws · ` +
    `${ack.dispatches} dispatch · ${ack.candidates} candidates · ${ack.tilesDrawn} tiles · ` +
    `${ack.tilesLoading} loading · bundle ${ack.bundleRecorded ? "recorded" : "replayed"} · ` +
    `wrote ${ack.placementsWritten} placements, ${ack.paintsWritten} paints\n` +
    `uploaded ${kib(ack.uploadedFromJs)} from JS buffers, ${kib(ack.uploadedFromWasm)} from wasm` +
    (ack.texelsFromJs || ack.texelsFromWasm ? ` · texels ${kib(ack.texelsFromJs)} from ` +
      `JS images, ${kib(ack.texelsFromWasm)} from wasm` : "") +
    jobs + (ack.idle ? " · idle" : "");
}

// Asks a fresh render worker whether WebGPU works there.
async function probeWorker() {
  const worker = new Worker(new URL("./render-worker.js", import.meta.url), { type: "module" });
  const probe = await new Promise((resolve) => {
    worker.onmessage = ({ data }) => resolve(data);
    worker.onerror = (e) => resolve({ ok: false, reason: e.message ?? "worker failed" });
    worker.postMessage({ type: "probe" });
  });
  console.info(`orbis: WebGPU in a worker: ${probe.ok ? `yes (${probe.adapter})` : `no (${probe.reason})`}`);
  return { worker, probe };
}

// Runs the map on this thread; returns its input sink and edit function.
async function startMain(module) {
  await init({ module_or_path: module });
  const frames = frameLoop(showFrame, (e) => fail(e));
  const map = await createMap(canvas, STYLE, frames.request, loadShaper);
  frames.start(map);
  const send = (method, ...args) => {
    const result = map[method](...args);
    if (!["hitTest", "getStyle", "semanticSnapshot"].includes(method)) frames.request();
    return result;
  };
  const apply = async (commands) => {
    const generation = map.edit(commands);
    frames.request();
    return generation;
  };
  return { send, apply, map, query: (method,...args) => map[method](...args), hitTest:(x,y) => map.hitTest(x,y) };
}

// Hands the canvas to the render worker and spawns the job workers and the
// placement worker, wired to it directly; returns the input sink.
async function startWorker(module, worker) {
  await init({module_or_path:module});
  publishedHits = new PublishedHitIndex();
  const ports = [];
  for (let i = 0; i < JOB_WORKERS; i++) {
    const channel = new MessageChannel();
    const job = new Worker(new URL("./job-worker.js", import.meta.url), { type: "module" });
    job.postMessage({ module, port: channel.port2 }, [channel.port2]);
    ports.push(channel.port1);
  }
  const placement = new MessageChannel();
  new Worker(new URL("./placement-worker.js", import.meta.url), { type: "module" })
    .postMessage({ module, port: placement.port2 }, [placement.port2]);
  const offscreen = canvas.transferControlToOffscreen();
  // Edits in flight, by id, until the render worker answers.
  const edits = new Map();
  let nextEdit = 0;
  const queries = new Map();
  let nextQuery = 0;
  worker.onmessage = ({ data }) => {
    if (data.type === "reply") {
      const {resolve,reject} = queries.get(data.request);
      queries.delete(data.request);
      if (data.error) reject(new Error(data.error)); else resolve(data.result);
    } else if (data.type === "frame") showFrame(data.ack);
    else if (data.type === "error") fail(data.message);
    else if (data.type === "edited") {
      const { resolve, reject } = edits.get(data.id);
      edits.delete(data.id);
      if (data.error === undefined) resolve(data.generation);
      else reject(new Error(data.error));
    }
  };
  worker.onerror = (e) => fail(e.message);
  worker.postMessage({ type: "init", module, canvas: offscreen, ports, placement: placement.port1, style: STYLE },
    [offscreen, ...ports, placement.port1]);
  const send = (method, ...args) => worker.postMessage({ type: method, args });
  const apply = (commands) => new Promise((resolve, reject) => {
    const id = nextEdit++;
    edits.set(id, { resolve, reject });
    worker.postMessage({ type: "edit", id, commands });
  });
  const query = (method,...args) => new Promise((resolve,reject) => {
    const request = nextQuery++;
    queries.set(request,{resolve,reject});
    worker.postMessage({type:method,args,request});
  });
  return { send, apply, query, hitTest:(x,y) => publishedHits.hitTest(x,y) };
}

function fail(e) {
  hud.textContent = String(e);
  console.error(e);
}

async function main() {
  if (!navigator.gpu) {
    hud.textContent = "WebGPU is not available in this browser.";
    return;
  }
  const module = WebAssembly.compileStreaming(fetch(new URL("./pkg/core_bg.wasm", import.meta.url)));
  const { worker, probe } = await probeWorker();
  const offscreen = typeof OffscreenCanvas !== "undefined" && "transferControlToOffscreen" in canvas;
  const thread = THREAD ?? (offscreen && probe.ok ? "worker" : "main");
  if (thread === "worker" && !(offscreen && probe.ok)) {
    throw new Error(`worker mode needs OffscreenCanvas and WebGPU in workers (${probe.reason ?? "no OffscreenCanvas"})`);
  }
  status = `${thread} thread · WebGPU in workers: ${probe.ok ? probe.adapter || "yes" : `no, ${probe.reason}`}`;
  hud.textContent = `${status}\nloading…`;

  const size = () => [Math.round(canvas.clientWidth * devicePixelRatio),
    Math.round(canvas.clientHeight * devicePixelRatio), devicePixelRatio];
  [canvas.width, canvas.height] = size();
  let started;
  if (thread === "worker") {
    started = await startWorker(await module, worker);
  } else {
    worker.terminate();
    started = await startMain(await module);
  }
  const { send } = started;
  // The edit API, for the console and test scripts: in main-thread mode
  // `orbis.map` is the map itself.
  globalThis.orbis = { ...editApi(started.apply), map: started.map,
    hitTest: started.hitTest, getStyle: () => started.query("getStyle"),
    // What the labels on screen say, for assistive technology (plan §13).
    semanticSnapshot: async () => JSON.parse(await started.query("semanticSnapshot")) };

  const hash = location.hash.slice(1).split("/").map(Number);
  if (hash.length >= 3 && hash.every(Number.isFinite)) {
    const [zoom, lat, lon, bearing = 0, pitch = 0] = hash;
    send("jumpTo", lon, lat, zoom, bearing, pitch);
  }
  new ResizeObserver(() => send("resize", ...size())).observe(canvas);

  // One pointer pans, or with the right button or Ctrl rotates (sideways)
  // and tilts (up and down), at GL JS's rates. Two pointers pinch-zoom and
  // twist-rotate around their midpoint, panning with it, or tilt when they
  // move up or down together.
  const BEARING_PER_PX = 0.8;
  const PITCH_PER_PX = 0.5;
  const pointers = new Map();
  const pair = () => {
    const [a, b] = pointers.values();
    return {
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      distance: Math.hypot(a.x - b.x, a.y - b.y),
      angle: Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI,
    };
  };
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY, rotate: e.button === 2 || e.ctrlKey });
  });
  canvas.addEventListener("pointermove", (e) => {
    const pointer = pointers.get(e.pointerId);
    if (!pointer) return;
    const [dx, dy] = [e.offsetX - pointer.x, e.offsetY - pointer.y];
    if (pointers.size === 1) {
      if (pointer.rotate) {
        send("rotatePitch", dx * BEARING_PER_PX, -dy * PITCH_PER_PX);
      } else {
        send("pan", pointer.x, pointer.y, e.offsetX, e.offsetY);
      }
      pointer.x = e.offsetX;
      pointer.y = e.offsetY;
    } else if (pointers.size === 2) {
      const before = pair();
      pointer.x = e.offsetX;
      pointer.y = e.offsetY;
      const after = pair();
      const twist = ((after.angle - before.angle + 540) % 360) - 180;
      const spread = before.distance > 0 ? after.distance / before.distance : 1;
      const together = Math.abs(spread - 1) < 0.01 && Math.abs(twist) < 0.5;
      if (together && Math.abs(after.y - before.y) > Math.abs(after.x - before.x)) {
        send("rotatePitch", 0, -(after.y - before.y) * PITCH_PER_PX);
        return;
      }
      send("pan", before.x, before.y, after.x, after.y);
      if (before.distance > 0 && after.distance > 0) {
        send("zoomAt", after.x, after.y, Math.log2(spread));
      }
      // Fingers turning clockwise turn the map clockwise: north goes right.
      send("rotatePitch", -twist, 0);
    }
  });
  const release = (e) => pointers.delete(e.pointerId);
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    send("zoomAt", e.offsetX, e.offsetY, -e.deltaY / (e.deltaMode === 1 ? 6 : 300));
  }, { passive: false });
  return started;
}

export const whenReady = main().catch((e) => { fail(e); throw e; });
export async function call(method,...args) {
  const host = await whenReady;
  if (method === "hitTest") return host.hitTest(...args);
  return method === "edit" ? host.apply(...args) : host.query(method,...args);
}
