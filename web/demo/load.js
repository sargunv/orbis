// The load-timing page (`web/scripts/load.mjs` drives it): one map of
// `?style=<url>` at `?at=lon,lat,zoom,bearing,pitch` in Orbis
// (`?engine=orbis`), the pinned GL JS (`gljs`) or mln-rs's web build
// (`mlnrs`, its module at `?mlnrs=<url>`), timed from this script's start
// to the engine's first `idle`. The title becomes `done {json}` with
// `performance.now()` at the script's start, the map's creation and idle
// (and `performance.timeOrigin`, to place them against the harness's
// request times), or `error <message>`.
const t0 = performance.now();
const params = new URLSearchParams(location.search);
const engine = params.get("engine") ?? "orbis";
const style = params.get("style");
const [lon, lat, zoom, bearing, pitch] = (params.get("at") ?? "0,0,0,0,0").split(",").map(Number);
const container = document.getElementById("map");

const engines = {
  async orbis() {
    const { createMap } = await import("orbis");
    const map = await createMap({ container, style, camera: { center: [lon, lat], zoom, bearing, pitch } });
    return [performance.now(), () => map.once("idle")];
  },
  async gljs() {
    const [gl, { default: workerUrl }] = await Promise.all([
      import("maplibre-gl"), import("maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url"),
      import("maplibre-gl/dist/maplibre-gl.css"),
    ]);
    const maplibregl = gl.default ?? gl;
    maplibregl.setWorkerUrl(workerUrl);
    const map = new maplibregl.Map({ container, style, center: [lon, lat], zoom, bearing, pitch, attributionControl: false });
    map.on("error", (e) => console.error(String(e.error)));
    return [performance.now(), () => new Promise((resolve) => map.once("idle", resolve))];
  },
  async mlnrs() {
    const { Map } = await import(/* @vite-ignore */ params.get("mlnrs"));
    const map = new Map({ container, style, center: [lon, lat], zoom, bearing, pitch, interactive: false });
    map.on("error", (e) => console.error(String(e.error)));
    return [performance.now(), () => map.once("idle")];
  },
};

try {
  const [created, idle] = await engines[engine]();
  await idle();
  document.title = "done " + JSON.stringify({ origin: performance.timeOrigin, script: t0, created, idle: performance.now() });
} catch (e) {
  document.title = "error " + e;
}
