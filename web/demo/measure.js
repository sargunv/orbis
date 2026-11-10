// The measurement page (`web/scripts/measure.mjs` drives it): the
// demo's Tokyo map with a blue circle drawn by the map and a cyan DOM
// annotation at the same point, in Orbis (`?engine=orbis`; on the page,
// not in a worker, with `&worker=0`) or the pinned GL JS (`?engine=gljs`).
// It logs, for each frame that shows new input, where the circle is drawn
// and when the latest input it shows arrived, so the driver can
// read input-to-photon latency and overlay skew off composited frames.
import { demoStyle } from "./styles.js";

const params = new URLSearchParams(location.search);
const engine = params.get("engine") ?? "orbis";
const status = document.getElementById("status");
const container = document.getElementById("map");
// Where the circle is: right of the starting centre, so pinches and twists
// around the centre move it.
// `?at=lon,lat,zoom,bearing,pitch` moves the start, for a dense or tilted view.
const [lon, lat, ZOOM, BEARING, PITCH] = (
  params.get("at") ?? "139.74,35.685,15,0,0"
)
  .split(",")
  .map(Number);
const CENTER = [lon, lat];
const START = {
  center: CENTER,
  zoom: ZOOM,
  bearing: BEARING || 0,
  pitch: PITCH || 0,
};
const OFFSET_PX = 120;

// Main-thread frames, counted for the idle check.
let rafs = 0;
const raf = requestAnimationFrame.bind(window);
window.requestAnimationFrame = (callback) =>
  raf((t) => {
    rafs++;
    callback(t);
  });

const log = [];
let renders = 0;
let lastInput = 0;
for (const type of [
  "pointerdown",
  "pointermove",
  "touchstart",
  "touchmove",
  "mousemove",
]) {
  window.addEventListener(
    type,
    (e) => {
      lastInput = e.timeStamp;
    },
    { capture: true, passive: true },
  );
}

const circleLngLat = async (project, unproject) => {
  const [x, y] = await project(CENTER);
  return unproject([x + OFFSET_PX, y]);
};

async function startOrbis() {
  const { createMap } = await import("orbis");
  const map = await createMap({
    container,
    style: demoStyle(params, "tokyo"),
    camera: START,
    worker: params.get("worker") !== "0",
  });
  await map.once("idle");
  const at = await circleLngLat(
    (l) => map.project(l),
    (p) => map.unproject(p),
  );
  await map.edit((tx) =>
    tx
      .addSource("measure", {
        type: "geojson",
        data: {
          type: "Feature",
          geometry: { type: "Point", coordinates: at },
          properties: {},
        },
      })
      .addLayer({
        id: "measure",
        type: "circle",
        source: "measure",
        paint: { "circle-radius": 12, "circle-color": "#0000ff" },
      }),
  );
  // The annotation, shown or not, gives each frame's circle: frames that
  // show new input log where the circle is drawn for it.
  const element = document.createElement("div");
  element.className = "mark";
  if (params.get("annotation") === "0") element.style.display = "none";
  const mark = map.annotations.add({ element, lngLat: at });
  let shown;
  map.on("render", ({ inputTime }) => {
    renders++;
    const p = mark.point;
    if (p && inputTime !== undefined && inputTime !== shown)
      log.push({ t: inputTime, x: p[0], y: p[1] });
    shown = inputTime;
  });
  await map.once("idle");
  // Resolves once idle, at once if it is.
  const idle = () =>
    map.stats?.idle && !map.camera.moving
      ? Promise.resolve()
      : map.once("idle");
  const home = () => {
    map.camera.jumpTo(START);
    return idle();
  };
  return { idle, home, stats: () => map.stats };
}

async function startGlJs() {
  const [gl, { Protocol }, { default: workerUrl }] = await Promise.all([
    import("maplibre-gl"),
    import("pmtiles"),
    import("maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url"),
  ]);
  const maplibregl = gl.default ?? gl;
  await import("maplibre-gl/dist/maplibre-gl.css");
  maplibregl.setWorkerUrl(workerUrl);
  maplibregl.addProtocol("pmtiles", new Protocol().tile);
  const styleUrl = demoStyle(params, "tokyo");
  const style = await (await fetch(styleUrl)).json();
  // GL JS wants absolute URLs.
  const absolute = (url) => new URL(url, styleUrl).href;
  style.sprite = absolute(style.sprite);
  style.glyphs = absolute(style.glyphs)
    .replace(/%7B/g, "{")
    .replace(/%7D/g, "}");
  for (const source of Object.values(style.sources)) {
    if (source.url?.startsWith("pmtiles://"))
      source.url = `pmtiles://${absolute(source.url.slice(10))}`;
  }
  const map = new maplibregl.Map({
    container,
    style,
    ...START,
    maxPitch: 85,
    attributionControl: false,
  });
  await map.once("idle");
  const at = await circleLngLat(
    (l) => {
      const p = map.project(l);
      return [p.x, p.y];
    },
    (p) => map.unproject(p).toArray(),
  );
  map.addSource("measure", {
    type: "geojson",
    data: {
      type: "Feature",
      geometry: { type: "Point", coordinates: at },
      properties: {},
    },
  });
  map.addLayer({
    id: "measure",
    type: "circle",
    source: "measure",
    paint: { "circle-radius": 12, "circle-color": "#0000ff" },
  });
  const element = document.createElement("div");
  element.className = "mark";
  if (params.get("annotation") !== "0")
    new maplibregl.Marker({ element }).setLngLat(at).addTo(map);
  // GL JS applies input in its render frame: the camera there is the one
  // for the latest input it has seen.
  map.on("move", () => {
    const p = map.project(at);
    log.push({ t: lastInput, x: p.x, y: p.y });
  });
  map.on("render", () => renders++);
  await map.once("idle");
  const idle = () =>
    map.loaded() && !map.isMoving() ? Promise.resolve() : map.once("idle");
  const home = () => {
    map.jumpTo(START);
    return idle();
  };
  return { idle, home, stats: () => null };
}

// No map: what receiving the driver's input costs the page by itself.
async function startNone() {
  container.style.touchAction = "none";
  container.addEventListener("pointermove", (e) =>
    log.push({ t: e.timeStamp, x: e.offsetX, y: e.offsetY }),
  );
  return { idle: async () => {}, home: async () => {}, stats: () => null };
}

window.measure = (async () => {
  const started = performance.now();
  const map =
    engine === "gljs"
      ? await startGlJs()
      : engine === "none"
        ? await startNone()
        : await startOrbis();
  status.textContent = `${engine} · ready in ${(performance.now() - started).toFixed(0)} ms`;
  return {
    engine,
    loadMs: performance.now() - started,
    timeOrigin: performance.timeOrigin,
    /** Clears the input log. */
    reset() {
      log.length = 0;
    },
    log: () => log.slice(),
    /** Main-thread animation frames and map frames over `ms` of quiet. */
    async quiet(ms) {
      const [r0, f0] = [rafs, renders];
      await new Promise((resolve) => setTimeout(resolve, ms));
      return { rafs: rafs - r0, renders: renders - f0 };
    },
    idle: map.idle,
    /** Back to the starting camera, once idle there. */
    home: map.home,
  };
})();
