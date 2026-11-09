// The light switch in the ⋯ menu, for an entry that offers it (the
// Terrain one): where the style's illumination is anchored. On the map,
// the light stays fixed to the ground as 3D terrain maps light it; on the
// viewport, it turns with the camera as suits a flat hillshade. A switch is
// one edit of every hillshade layer's `hillshade-illumination-anchor` and
// the style's `light` anchor, so a comparison's other maps follow it as
// they follow the editor's edits.
const HILLSHADE = "hillshade-illumination-anchor";
/** The spec's default for both anchors. */
const SPEC_DEFAULT = "viewport";

/** `style`'s anchor: its first hillshade layer's, else its light's, else the spec's default. */
export function anchorOf(style) {
  const hillshade = style.layers.find((l) => l.type === "hillshade");
  return hillshade?.paint?.[HILLSHADE] ?? style.light?.anchor ?? SPEC_DEFAULT;
}

/** Anchors `map`'s style's illumination to `anchor`, "map" or "viewport", in one edit. */
export async function setAnchor(map, anchor) {
  const style = await map.getStyle();
  return map.edit((tx) => {
    for (const layer of style.layers) if (layer.type === "hillshade") tx.setPaintProperty(layer.id, HILLSHADE, anchor);
    tx.setLight({ ...style.light, anchor });
  });
}

/** The menu's control. `onPick(anchor)` hears the user's choices. */
export function lightSwitch({ onPick }) {
  const root = document.getElementById("light-choice");
  const options = [...root.querySelectorAll("[data-anchor]")];
  for (const option of options) option.addEventListener("click", () => onPick(option.dataset.anchor));
  // Arrow keys move the choice, as in a radio group.
  root.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }[e.key];
    if (!step) return;
    e.preventDefault();
    const at = options.findIndex((o) => o.getAttribute("aria-checked") === "true");
    const next = options[(at + step + options.length) % options.length];
    next.focus();
    onPick(next.dataset.anchor);
  });
  return {
    /** Shows `anchor` as chosen, or hides the control for null. */
    show(anchor) {
      root.hidden = anchor === null;
      for (const option of options) {
        const on = option.dataset.anchor === anchor;
        option.setAttribute("aria-checked", String(on));
        option.tabIndex = on ? 0 : -1;
      }
    },
  };
}
