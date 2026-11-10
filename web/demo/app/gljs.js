// MapLibre GL JS, the pinned version, as an engine for the comparison
// (compare.js). It loads only when a comparison opens.
import {
  addProtocol,
  getVersion,
  Map as GlMap,
  setWorkerUrl,
} from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import { Protocol } from "pmtiles";

import "maplibre-gl/dist/maplibre-gl.css";

// The bundle moves GL JS away from its worker; the Tokyo extract is PMTiles.
setWorkerUrl(workerUrl);
addProtocol("pmtiles", new Protocol().tile);

export const label = `GL JS ${getVersion()}`;

/** A GL JS map of `style` in `container`, at `camera` within `constraints`. */
export function open(
  container,
  { style, camera: { center, zoom, bearing, pitch, roll }, constraints = {} },
) {
  const map = new GlMap({
    container,
    style,
    center,
    zoom,
    bearing,
    pitch,
    roll,
    // GL JS reads a given but undefined minZoom as 0, not its default -2
    // (Orbis's too), which stopped its globe zooming out at a larger size.
    minZoom: constraints.minZoom ?? -2,
    maxZoom: constraints.maxZoom,
    // Orbis lets pitch reach 85 degrees by default, GL JS 60.
    minPitch: constraints.minPitch,
    maxPitch: constraints.maxPitch ?? 85,
    maxBounds: constraints.bounds ?? undefined,
    // The app shows the style's attribution for both maps.
    attributionControl: false,
  });
  map.on("error", ({ error }) =>
    console.error(`GL JS: ${error?.message ?? error}`),
  );
  return {
    camera: () => ({
      center: map.getCenter().toArray(),
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch(),
      roll: map.getRoll(),
    }),
    jumpTo: ({ center, zoom, bearing, pitch, roll }) =>
      map.jumpTo({ center, zoom, bearing, pitch, roll }),
    onMove: (listener) => {
      const subscription = map.on("move", listener);
      return () => subscription.unsubscribe();
    },
    stop: () => map.stop(),
    setStyle: (next) => map.setStyle(next, { diff: true }),
    style: () => map.getStyle(),
    loaded: () => map.loaded() && !map.isMoving(),
    remove: () => map.remove(),
  };
}
