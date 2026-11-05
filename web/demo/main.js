// The web demo: OpenFreeMap's live Liberty map (the committed Tokyo extract
// with `?style=tokyo-mvt|tokyo-mlt`), drawing through WebGPU with the
// `orbis` package: on this thread by default (decision 0021), or with
// `?thread=worker` as the ui role, owning the camera and input while a
// render worker draws (plan §7.1).
//
// Tap or click a feature to see its properties in a popup; hovering names
// the feature under the pointer; a pin marks Tokyo Station.
import { createMap } from "orbis";
import { LIVE_CAMERA, demoFlight, demoStyle, isLive } from "./styles.js";
import { expose, feel, hashCamera } from "./expose.js";

const params = new URLSearchParams(location.search);
const hud = document.getElementById("hud");
let status = "";
let hovered = "";

const kib = (bytes) => `${(bytes / 1024).toFixed(1)} KiB`;
function showFrame(ack) {
  for (const warning of ack.warnings) console.warn(warning);
  const jobs = ack.jobsDone ? ` · ${ack.jobsDone} jobs · ${kib(ack.headerBytes)} headers into wasm` : "";
  hud.textContent = `${status}${hovered ? ` · ${hovered}` : ""}\n` +
    `z${ack.zoom.toFixed(2)} · ${ack.ms.toFixed(1)} ms CPU · ${ack.draws} draws · ` +
    `${ack.dispatches} dispatch · ${ack.candidates} candidates · ${ack.tilesDrawn} tiles · ` +
    `${ack.tilesLoading} loading · bundle ${ack.bundleRecorded ? "recorded" : "replayed"} · ` +
    `wrote ${ack.placementsWritten} placements, ${ack.paintsWritten} paints\n` +
    `uploaded ${kib(ack.uploadedFromJs)} from JS buffers, ${kib(ack.uploadedFromWasm)} from wasm` +
    (ack.texelsFromJs || ack.texelsFromWasm ? ` · texels ${kib(ack.texelsFromJs)} from ` +
      `JS images, ${kib(ack.texelsFromWasm)} from wasm` : "") +
    jobs + (ack.idle ? " · idle" : "");
}

function fail(e) {
  hud.textContent = String(e);
  console.error(e);
}

function popupElement(feature) {
  const element = document.createElement("div");
  element.className = "popup demo-annotation";
  const title = document.createElement("strong");
  title.textContent = `${feature.layer?.id ?? feature.source}${feature.id !== undefined ? ` #${feature.id}` : ""}`;
  const list = document.createElement("dl");
  for (const [key, value] of Object.entries(feature.properties).slice(0, 8)) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
    list.append(dt, dd);
  }
  element.append(title, list);
  return element;
}

async function main() {
  const map = await createMap({
    container: document.getElementById("map"),
    style: demoStyle(params),
    thread: params.get("thread") ?? undefined,
    renderer: params.get("renderer") ?? undefined,
    camera: hashCamera() ?? (isLive(params) ? LIVE_CAMERA : undefined),
    feel: feel(),
  });
  status = `${map.thread} thread`;
  // The tier the map draws at, and how long a device loss took to recover:
  // to the first frame on the new device, and to the next complete one.
  let lostAt;
  map.on("capability", ({ tier, recoveredMs }) => {
    status = `${map.thread} thread · ${tier === "modern" ? "WebGPU" : "WebGL2"}`;
    console.info(`orbis: ${status}`);
    if (recoveredMs === undefined) return;
    const first = Math.round(recoveredMs);
    const idle = map.on("idle", () => {
      idle();
      status += ` · recovered from device loss: first frame ${first} ms, complete ${Math.round(performance.now() - lostAt)} ms`;
      console.info(`orbis: ${status}`);
      if (map.stats) showFrame(map.stats);
    });
  });
  // `?lose=<ms>` loses the GPU that long after loading, to show recovery
  // (plan §8.8); with `&switch`, WebGPU's loss switches to WebGL2.
  // `orbis.loseDevice(switchTier)` any time.
  if (params.has("lose")) {
    const loaded = map.on("load", () => {
      loaded();
      setTimeout(() => { lostAt = performance.now(); map.loseDevice(params.has("switch")); }, Number(params.get("lose")) || 0);
    });
  }
  hud.textContent = `${status}\nloading…`;
  expose(map);
  map.on("render", showFrame);
  map.on("error", ({ message }) => console.error(message));

  // `?style=alps&flight` flies a loop over the Alps until the user takes
  // the camera.
  const flight = demoFlight(params);
  if (flight) {
    (async () => {
      for (let leg = 0; ; leg = (leg + 1) % flight.length) {
        const arrived = await map.camera.flyTo(flight[leg]);
        if (!arrived) return;
      }
    })();
  }

  if (!["scripts", "alps"].includes(params.get("style"))) {
    const pin = document.createElement("div");
    pin.className = "pin demo-annotation";
    pin.title = "Tokyo Station";
    map.annotations.add({ element: pin, lngLat: [139.7671, 35.6812] });
  }

  let popup;
  map.on("click", async ({ point, lngLat }) => {
    popup?.remove();
    popup = undefined;
    const [feature] = await map.queryRenderedFeatures(point);
    if (!feature || !lngLat) return;
    popup = map.annotations.add({ element: popupElement(feature), lngLat, anchor: "bottom", offset: [0, -8] });
  });
  map.canvas.addEventListener("pointermove", (e) => {
    if (e.buttons) return;
    const hit = map.hitTest([e.offsetX, e.offsetY]);
    hovered = hit ? `${hit.layer}${hit.id !== null ? ` #${hit.id}` : ""}` : "";
    map.canvas.style.cursor = hit ? "pointer" : "";
  });
  return map;
}

export const whenReady = main().catch((e) => { fail(e); throw e; });
