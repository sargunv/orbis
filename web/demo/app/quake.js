// The Earthquake entry: Mapterhorn's Alps as one host-fed `raster-dem`
// source (data/earthquake.json, which `cargo xtask demo-data styles`
// bakes from data/alps.json), whose heights a worker of the demo's own
// makes and feeds the map through its elevation port (quake-worker.js),
// so the page only passes on the tiles the map asks for and clicks. A
// click moves the epicentre, but one on the Matterhorn's marker, which
// flies to it (app/marker.js).

import { onMarker } from "./marker.js";

/** The source earthquake.json feeds terrain and hillshade from. */
export const QUAKE_SOURCE = "quake";

/**
 * Runs the quake on `map` while `active()` says its entry shows: a worker
 * starts with the entry's first tile request, its shock at `epicentre()`,
 * and stops with `stop()`.
 */
export function attachQuake(map, { active, epicentre }) {
  let worker;
  let centre;
  const start = () => {
    centre ??= epicentre();
    worker = new Worker(new URL("./quake-worker.js", import.meta.url), {
      type: "module",
    });
    const port = map.elevationPort();
    worker.postMessage(
      { start: { source: QUAKE_SOURCE, epicentre: centre }, port },
      [port],
    );
    return worker;
  };
  map.on("tilerequest", ({ source, z, x, y }) => {
    if (source !== QUAKE_SOURCE || !active()) return;
    (worker ?? start()).postMessage({ tile: [z, x, y] });
  });
  map.on("click", async ({ point, lngLat }) => {
    if (!active()) return;
    if (!worker || !lngLat || (await onMarker(map, point))) return;
    centre = lngLat;
    worker.postMessage({ epicentre: lngLat });
  });
  return {
    stop() {
      worker?.terminate();
      worker = undefined;
      centre = undefined;
    },
  };
}
