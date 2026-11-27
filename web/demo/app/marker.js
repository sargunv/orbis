// Scenes' markers: a landmark a scene's style marks in its layers `marker`
// and `marker-label` (the Earthquake's, data/earthquake.json, too); a click
// on it flies to the scene's `marker.view`, flyTo options, so a viewer who
// wandered off finds the way back.

/** The layers a marker draws in. */
const LAYERS = ["marker", "marker-glow", "marker-label"];

/** Whether the click at `point` landed on the marker. */
export async function onMarker(map, point) {
  const hit = await map.hitTest(point);
  return LAYERS.includes(hit?.layer);
}

/** Flies `map` to the marker's view when it's clicked in `scene()`. */
export function attachMarker(map, scene) {
  map.on("click", async ({ point }) => {
    const view = scene().marker?.view;
    if (view && (await onMarker(map, point))) map.camera.flyTo(view);
  });
}
