// The map picker: the catalogue by provider, each entry with a chip in its
// style's colours. Tokyo's offline extract carries its format toggle.
import { h, hidden } from "./dom.js";

export function picker(groups, { onPick }) {
  const popover = document.getElementById("scenes");
  const button = document.getElementById("scene-button");
  const list = document.getElementById("scene-groups");
  let current = { scene: null, format: undefined };

  const option = (scene) => {
    const pick = (format) => { popover.hidePopover(); onPick(scene, format); };
    const row = h("button", {
      class: "scene-option", type: "button", "data-scene": scene.key,
      onclick: () => pick(scene.formats ? (current.scene === scene ? current.format : "mvt") : undefined),
    },
    h("span", { class: "chip", "aria-hidden": "true" }),
    h("span", { class: "scene-option-text" },
      h("span", { class: "scene-option-name" }, scene.name),
      scene.detail && h("span", { class: "scene-option-detail" }, scene.detail)),
    scene.dark && h("span", { class: "dark-flag" }, "Dark"));
    const chip = row.querySelector(".chip");
    chip.style.setProperty("--a", scene.swatch[0]);
    chip.style.setProperty("--b", scene.swatch[1]);
    chip.style.setProperty("--c", scene.swatch[2]);
    if (!scene.formats) return h("li", {}, row);
    const formats = h("span", { class: "formats", role: "group", "aria-label": "Tile format" },
      Object.keys(scene.formats).map((format) => h("button", {
        type: "button", "data-format": format, "aria-pressed": "false",
        onclick: () => pick(format),
      }, format.toUpperCase(), hidden(` tiles for ${scene.name}`))));
    return h("li", { class: "scene-row" }, row, formats);
  };

  list.replaceChildren(...groups.map((group) => h("section", { class: "scene-group" },
    h("h3", {}, group.name),
    h("ul", { class: "scene-list" }, group.scenes.map(option)))));

  // On wide screens the sheet opens under the button, wherever the editor
  // has moved it.
  popover.addEventListener("beforetoggle", (e) => {
    if (e.newState !== "open") return;
    const rect = button.getBoundingClientRect();
    popover.style.setProperty("--sheet-top", `${rect.bottom + 8}px`);
    popover.style.setProperty("--sheet-left", `${rect.left}px`);
  });
  popover.addEventListener("toggle", (e) => {
    if (e.newState === "open") list.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
  });

  return {
    show(scene, format) {
      current = { scene, format };
      document.getElementById("scene-name").textContent = scene.name;
      document.getElementById("scene-provider").textContent =
        scene.formats ? `${scene.provider}, ${format.toUpperCase()} tiles` : scene.provider;
      button.setAttribute("aria-label", `Map: ${scene.name}, ${scene.provider}. Choose a map`);
      for (const row of list.querySelectorAll(".scene-option")) {
        row.setAttribute("aria-current", String(row.dataset.scene === scene.key));
      }
      for (const toggle of list.querySelectorAll("[data-format]")) {
        toggle.setAttribute("aria-pressed", String(scene.formats && toggle.dataset.format === format));
      }
    },
  };
}
