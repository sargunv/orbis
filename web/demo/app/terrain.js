// The terrain switch in an entry's options, for an entry that offers it
// (`terrain` in styles.js): the style's 3D terrain on or off. A switch is
// one edit of the style's `terrain`, so nothing reloads, and a comparison's
// other maps follow it as they follow the editor's edits.

/** Whether `style` draws 3D terrain. */
export const terrainOf = (style) => Boolean(style.terrain);

/** Puts `terrain` (a v8 `terrain`, or null for none) on `map`'s style in one edit. */
export function setTerrain(map, terrain) {
  return map.edit((tx) => tx.setTerrain(terrain));
}

/** The control. `onPick(on)` hears the user's choices. */
export function terrainSwitch({ onPick }) {
  const button = document.getElementById("terrain-toggle");
  button.addEventListener("click", () => onPick(button.getAttribute("aria-checked") !== "true"));
  return {
    /** Shows terrain as on or off, or hides the control for null. */
    show(on) {
      button.hidden = on === null;
      button.setAttribute("aria-checked", String(on === true));
    },
  };
}
