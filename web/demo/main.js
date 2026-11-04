// The web demo: OpenFreeMap's live Liberty map (the committed Tokyo extract
// with `?style=tokyo-mvt|tokyo-mlt`), drawing through WebGPU with the
// `orbis` package. In worker mode (the default where it
// works) this page is the ui role: it owns the camera and input while a
// render worker draws (plan §7.1). `?thread=main` runs everything here.
//
// Tap or click a feature to see its properties in a popup; hovering names
// the feature under the pointer; a pin marks Tokyo Station.
import { createMap } from "orbis";
import { LIVE_CAMERA, demoStyle, isLive } from "./styles.js";
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
    thread: params.get("thread") ?? "auto",
    camera: hashCamera() ?? (isLive(params) ? LIVE_CAMERA : undefined),
    feel: feel(),
  });
  status = `${map.thread} thread`;
  hud.textContent = `${status}\nloading…`;
  expose(map);
  map.on("render", showFrame);
  map.on("error", ({ message }) => console.error(message));

  if (params.get("style") !== "scripts") {
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
