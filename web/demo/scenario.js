// The binding scenario (plan §14.1, decision 0018) through the orbis
// package; `pnpm -C web e2e` drives it on each renderer. The
// page makes the map and records its events and frames; the driver reads
// pixels off screenshots and moves the pointer.
import { createMap, setRequestTransform } from "orbis";

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
    renderer: params.get("renderer") ?? undefined,
  });
  const generations = [];
  for (const type of ["styleload", "load", "idle", "movestart", "moveend", "click", "error", "styleimagemissing"]) {
    map.on(type, (e) => events.push({ type, ...(type === "click" ? { point: e.point, lngLat: e.lngLat } : {}), ...(e.id ? { id: e.id } : {}) }));
  }
  map.on("render", (stats) => { renders++; generations.push(stats.generation); });
  await map.once("idle");
  return {
    map,
    setRequestTransform,
    /** The events since the last call. */
    events: () => events.splice(0),
    idle: () => (map.stats?.idle && !map.camera.moving ? Promise.resolve() : map.once("idle")),
    /** Resolves once the next frame draws and the map idles again. */
    drawn: () => map.once("render").then(() => (map.stats?.idle ? undefined : map.once("idle"))),
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
