import mvt from "./data/tokyo-mvt.json?url";
import mlt from "./data/tokyo-mlt.json?url";

import playground from "./data/playground.json?url";
import scripts from "./data/scripts.json?url";
import arctic from "./data/crs/arctic.json?url";
import greenland from "./data/crs/greenland.json?url";
import nz from "./data/crs/nz.json?url";
import alps from "./data/alps.json?url";
import alpsFlight from "./data/alps-flight.json";

/** OpenFreeMap's live Liberty style over the planet: the demo's default. */
export const LIVE = "https://tiles.openfreemap.org/styles/liberty";
/** Where the live map opens: central Tokyo, around Tokyo Station. */
export const LIVE_CAMERA = { center: [139.767, 35.681], zoom: 13 };

/**
 * The style a demo page shows. `style=tokyo-mvt` or `tokyo-mlt` (or
 * `format=mvt|mlt`) is the committed Tokyo extract, for benchmarks,
 * snapshots and offline work; `style=scripts` the Cairo, Delhi and Bangkok
 * page's; `style=arctic` the Arctic on EPSG:3413, `style=nz` New Zealand on
 * NZTM2000 (EPSG:2193) and `style=greenland` Greenland's ice on Web
 * Mercator (scene documents, plan §11.4); `style=alps` the Valais Alps on
 * terrain (decision 0019); any other
 * `style` a URL. Without one, the playground has its own
 * and every other page `fallback`: "live" (OpenFreeMap) or "tokyo".
 */
export function demoStyle(params, fallback = "live") {
  const style = params.get("style");
  const format = params.get("format");
  if (format && !["mvt", "mlt"].includes(format)) throw new Error("format must be mvt or mlt");
  const tokyo = { "tokyo-mvt": mvt, "tokyo-mlt": mlt }[style] ?? (format ? { mvt, mlt }[format] : undefined);
  const named = { scripts, arctic, greenland, nz, alps }[style];
  const url = named
    ?? tokyo ?? style
    ?? (location.pathname.endsWith("playground.html") ? playground : fallback === "tokyo" ? mvt : LIVE);
  return new URL(url, location.href).href;
}

/** Whether `params` show the live default map. */
export function isLive(params) {
  return !params.get("style") && !params.get("format");
}

/**
 * The camera flight a demo style flies with `?flight`: legs of flyTo
 * options, looping until the user moves the map.
 */
export function demoFlight(params) {
  return params.has("flight") && params.get("style") === "alps" ? alpsFlight.legs : undefined;
}
