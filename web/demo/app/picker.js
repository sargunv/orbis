// The map picker: the catalogue by provider, each entry with a chip in its
// style's colours.
import { h } from "./dom.js";

export function picker(groups, { onPick }) {
  const popover = document.getElementById("scenes");
  const button = document.getElementById("scene-button");
  const list = document.getElementById("scene-groups");

  const option = (scene) => {
    const row = h(
      "button",
      {
        class: "scene-option",
        type: "button",
        "data-scene": scene.key,
        onclick: () => {
          popover.hidePopover();
          onPick(scene);
        },
      },
      h("span", { class: "chip", "aria-hidden": "true" }),
      h(
        "span",
        { class: "scene-option-text" },
        h("span", { class: "scene-option-name" }, scene.name),
        scene.detail &&
          h("span", { class: "scene-option-detail" }, scene.detail),
      ),
      scene.dark && h("span", { class: "dark-flag" }, "Dark"),
    );
    const chip = row.querySelector(".chip");
    chip.style.setProperty("--a", scene.swatch[0]);
    chip.style.setProperty("--b", scene.swatch[1]);
    chip.style.setProperty("--c", scene.swatch[2]);
    return h("li", {}, row);
  };

  list.replaceChildren(
    ...groups.map((group) =>
      h(
        "section",
        { class: "scene-group" },
        h("h3", {}, group.name),
        h("ul", { class: "scene-list" }, group.scenes.map(option)),
      ),
    ),
  );

  // On wide screens the sheet opens under the button.
  popover.addEventListener("beforetoggle", (e) => {
    if (e.newState !== "open") return;
    const rect = button.getBoundingClientRect();
    popover.style.setProperty("--sheet-top", `${rect.bottom + 8}px`);
    popover.style.setProperty("--sheet-left", `${rect.left}px`);
  });
  popover.addEventListener("toggle", (e) => {
    if (e.newState === "open")
      list
        .querySelector('[aria-current="true"]')
        ?.scrollIntoView({ block: "nearest" });
  });

  return {
    /** The entry shown: one of the catalogue's, or one opened by URL (the Tokyo extract in `format`, a custom style). */
    show(scene, format) {
      document.getElementById("scene-name").textContent = scene.name;
      document.getElementById("scene-provider").textContent = scene.formats
        ? `${scene.provider}, ${format.toUpperCase()} tiles`
        : scene.provider;
      button.setAttribute(
        "aria-label",
        `Map: ${scene.name}, ${scene.provider}. Choose a map`,
      );
      for (const row of list.querySelectorAll(".scene-option")) {
        row.setAttribute(
          "aria-current",
          String(row.dataset.scene === scene.key),
        );
      }
    },
  };
}
