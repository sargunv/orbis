// The Earthquake entry: Mapterhorn's Alps as one host-fed `raster-dem`
// source (`restyle: "quake"`), whose heights a worker of the demo's own
// makes and feeds the map through its elevation port (quake-worker.js),
// so the page only passes on the tiles the map asks for and clicks. A
// click moves the epicentre, but one on the Matterhorn's marker, which
// flies to it (app/marker.js).

import { onMarker } from "./marker.js";

/** The source the restyled style feeds terrain and hillshade from. */
export const QUAKE_SOURCE = "quake";

/** The Matterhorn, which the quake starts under. */
const MATTERHORN = [7.6586, 45.9763];

/**
 * A marker on the Matterhorn, which the quake starts under, so a viewer
 * who wandered off can click back to it.
 */
const marker = {
  source: {
    type: "geojson",
    data: {
      type: "Feature",
      geometry: { type: "Point", coordinates: MATTERHORN },
      properties: {},
    },
  },
  layers: [
    {
      id: "marker",
      type: "circle",
      source: "marker",
      paint: {
        "circle-radius": 7,
        "circle-color": "#d7263d",
        "circle-stroke-color": "#ffffff",
        "circle-stroke-width": 2.5,
      },
    },
    {
      id: "marker-label",
      type: "symbol",
      source: "marker",
      layout: {
        "text-field": "Matterhorn",
        "text-font": ["Noto Sans Bold"],
        "text-size": 13,
        "text-anchor": "bottom",
        "text-offset": [0, -0.9],
      },
      paint: {
        "text-color": "#7a1020",
        "text-halo-color": "#ffffff",
        "text-halo-width": 1.5,
      },
    },
  ],
};

/**
 * The Alps style (data/alps.json) with its two Mapterhorn sources made one
 * that the app feeds (`"tiles": ["orbis-host:"]`): the terrain stands on it
 * and the hillshade shades it, so both follow the moving ground. Its tiles
 * stop at zoom 11, which keeps the heights a step writes to what a view
 * of the valleys needs: a few dozen tiles of 256² heights.
 */
export function quakeStyle(style) {
  const sources = Object.fromEntries(
    Object.entries(style.sources).filter(([, s]) => s.type !== "raster-dem"),
  );
  sources.marker = marker.source;
  sources[QUAKE_SOURCE] = {
    type: "raster-dem",
    tiles: ["orbis-host:"],
    encoding: "terrarium",
    tileSize: 512,
    maxzoom: 11,
    attribution:
      '<a href="https://mapterhorn.com/attribution">© Mapterhorn</a>',
  };
  const dem = new Set(
    Object.entries(style.sources)
      .filter(([, s]) => s.type === "raster-dem")
      .map(([id]) => id),
  );
  return {
    ...style,
    sources,
    layers: [
      ...style.layers.map((layer) =>
        dem.has(layer.source) ? { ...layer, source: QUAKE_SOURCE } : layer,
      ),
      ...marker.layers,
    ],
    terrain: { source: QUAKE_SOURCE, exaggeration: 1.6 },
  };
}

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
