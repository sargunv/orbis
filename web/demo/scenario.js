// The binding scenario through the orbis
// package; `pnpm -C web e2e` drives it (`?worker=0` keeps the map on the
// page, `?backend=webgl2` draws it with WebGL 2). The
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
        type: "Feature",
        id: 7,
        properties: {},
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [-10, -10],
              [10, -10],
              [10, 10],
              [-10, 10],
              [-10, -10],
            ],
          ],
        },
      },
    },
  },
  layers: [
    {
      id: "background",
      type: "background",
      paint: { "background-color": "#ff0000" },
    },
    {
      id: "box",
      type: "fill",
      source: "box",
      paint: { "fill-color": "#0000ff" },
    },
  ],
};

const params = new URLSearchParams(location.search);
const events = [];
let renders = 0;
let rafs = 0;
const raf = requestAnimationFrame.bind(window);
window.requestAnimationFrame = (callback) =>
  raf((t) => {
    rafs++;
    callback(t);
  });

window.scenario = (async () => {
  const map = await createMap({
    container: document.getElementById("map"),
    style: STYLE,
    worker: params.get("worker") !== "0",
    backend: params.get("backend") ?? undefined,
  });
  // Before a frame: the canvas's size, for fits and projections.
  const firstProjection = await map.project([0, 0]);
  const generations = [];
  for (const type of [
    "styleload",
    "load",
    "idle",
    "movestart",
    "moveend",
    "click",
    "contextmenu",
    "error",
    "styleimagemissing",
  ]) {
    map.on(type, (e) =>
      events.push({
        type,
        ...(e.point ? { point: e.point, lngLat: e.lngLat } : {}),
        ...(e.id ? { id: e.id } : {}),
      }),
    );
  }
  map.on("render", (stats) => {
    renders++;
    generations.push(stats.generation);
  });
  await map.once("idle");
  return {
    map,
    setRequestTransform,
    firstProjection,
    /** The events since the last call. */
    events: () => events.splice(0),
    /** Resolves once the map has the calls made so far and is idle. */
    idle: async () => {
      await map.hasImage("");
      if (!map.stats?.idle || map.camera.moving) await map.once("idle");
    },
    /** Resolves once the next frame draws and the map idles again. */
    drawn: () =>
      map
        .once("render")
        .then(() => (map.stats?.idle ? undefined : map.once("idle"))),
    /** Resolves once a frame shows `generation` (or the one it resolves to). */
    shows: async (generation) => {
      const shown = await generation;
      await new Promise((resolve) => {
        const check = () =>
          generations.some((g) => g >= shown)
            ? resolve()
            : setTimeout(check, 10);
        check();
      });
    },
    /** Resolves once the map has every call made so far, its state with them. */
    sync: () => map.hasImage(""),
    /** Main-thread animation frames and map frames over `ms`. */
    async quiet(ms) {
      const [r0, f0] = [rafs, renders];
      await new Promise((resolve) => setTimeout(resolve, ms));
      return { rafs: rafs - r0, renders: renders - f0 };
    },
  };
})();
