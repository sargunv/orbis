// The binding scenario (plan §14.1, decision 0018) through the orbis
// package; `web/scripts/scenario.mjs` drives it in each thread mode. The
// page makes the map and records its events and frames; the driver reads
// pixels off screenshots and moves the pointer.
import { createMap } from "orbis";

/** A red background and a blue box around the origin, with a feature id. */
const STYLE = {
  version: 8,
  center: [0, 0],
  zoom: 0,
  sources: {
    box: {
      type: "geojson",
      data: {
        type: "Feature", id: 7, properties: {},
        geometry: { type: "Polygon", coordinates: [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]] },
      },
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#ff0000" } },
    { id: "box", type: "fill", source: "box", paint: { "fill-color": "#0000ff" } },
  ],
};

const params = new URLSearchParams(location.search);
const events = [];
let renders = 0;
let rafs = 0;
const raf = requestAnimationFrame.bind(window);
window.requestAnimationFrame = (callback) => raf((t) => { rafs++; callback(t); });

window.scenario = (async () => {
  const map = await createMap({
    container: document.getElementById("map"), style: STYLE,
    thread: params.get("thread") ?? undefined, renderer: params.get("renderer") ?? undefined,
  });
  const generations = [];
  for (const type of ["styleload", "load", "idle", "movestart", "moveend", "click", "error"]) {
    map.on(type, (e) => events.push({ type, ...(type === "click" ? { point: e.point, lngLat: e.lngLat } : {}) }));
  }
  map.on("render", (stats) => { renders++; generations.push(stats.generation); });
  await map.once("idle");
  return {
    map,
    thread: map.thread,
    /** The events since the last call. */
    events: () => events.splice(0),
    idle: () => (map.stats?.idle && !map.camera.moving ? Promise.resolve() : map.once("idle")),
    /** Resolves once a frame shows `generation`. */
    shows: (generation) => new Promise((resolve) => {
      const check = () => (generations.some((g) => g >= generation) ? resolve() : setTimeout(check, 10));
      check();
    }),
    /** Main-thread animation frames and map frames over `ms`. */
    async quiet(ms) {
      const [r0, f0] = [rafs, renders];
      await new Promise((resolve) => setTimeout(resolve, ms));
      return { rafs: rafs - r0, renders: renders - f0 };
    },
  };
})();
