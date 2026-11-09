// A second Orbis map, as an engine for the comparison (compare.js), for a
// side that picks Orbis while the app's own map shows the other.
import { createMap } from "orbis";

export const label = "Orbis";

/** An Orbis map of `style` in `container`, at `camera` within `constraints`. */
export async function open(container, { style, camera, constraints }) {
  const map = await createMap({ container, style, camera, constraints });
  let idle = false;
  map.on("render", (ack) => { idle = ack.idle; });
  map.on("error", ({ message }) => console.error(`Orbis (compared): ${message}`));
  return {
    camera: () => map.camera.current,
    jumpTo: ({ center, zoom, bearing, pitch, roll }) => map.camera.jumpTo({ center, zoom, bearing, pitch, roll }),
    onMove: (listener) => map.on("move", listener),
    stop: () => map.camera.stop(),
    setStyle: (next) => map.setStyle(next),
    loaded: () => idle,
    remove: () => map.remove(),
  };
}
