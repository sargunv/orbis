// The Earthquake entry: Mapterhorn's Alps as one host-fed `raster-dem`
// source (`restyle: "quake"`), whose heights a worker of the demo's own
// makes and feeds the map through its elevation port (quake-worker.js),
// so the page only passes on the tiles the map asks for and clicks. A
// click moves the epicentre.

/** The source the restyled style feeds terrain and hillshade from. */
export const QUAKE_SOURCE = "quake";

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
    layers: style.layers.map((layer) =>
      dem.has(layer.source) ? { ...layer, source: QUAKE_SOURCE } : layer,
    ),
    terrain: { source: QUAKE_SOURCE, exaggeration: 1.6 },
  };
}

/**
 * Runs the quake on `map` while `active()` says its entry shows: a worker
 * starts with the entry's first tile request, its shock at `epicentre()`,
 * and stops with `stop()`. `onStats` hears its rates once a second.
 */
export function attachQuake(map, { active, epicentre, onStats }) {
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
    worker.onmessage = ({ data }) => data.stats && onStats?.(data.stats);
    return worker;
  };
  map.on("tilerequest", ({ source, z, x, y }) => {
    if (source !== QUAKE_SOURCE || !active()) return;
    (worker ?? start()).postMessage({ tile: [z, x, y] });
  });
  map.on("click", ({ lngLat }) => {
    if (!worker || !lngLat || !active()) return;
    centre = lngLat;
    worker.postMessage({ epicentre: lngLat });
  });
  return {
    /** Holds the ground still, or lets it move again. */
    pause(paused) {
      worker?.postMessage({ paused });
    },
    stop() {
      worker?.terminate();
      worker = undefined;
      centre = undefined;
    },
  };
}
