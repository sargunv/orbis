// The map picker: the catalogue's groups, each opening its entries as a
// sub-menu with a way back, each entry with a chip in its style's colours.
// It opens on the groups, the current entry's marked.
import { h } from "./dom.js";

export function picker(groups, { onPick }) {
  const popover = document.getElementById("scenes");
  const button = document.getElementById("scene-button");
  const list = document.getElementById("scene-groups");
  const title = document.getElementById("scenes-title");
  const back = document.getElementById("scene-back");

  /** A chip in `swatch`'s colours: land, water and a line. */
  const chip = (swatch) => {
    const element = h("span", { class: "chip", "aria-hidden": "true" });
    element.style.setProperty("--a", swatch[0]);
    element.style.setProperty("--b", swatch[1]);
    element.style.setProperty("--c", swatch[2]);
    return element;
  };

  const option = (scene) =>
    h(
      "li",
      {},
      h(
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
        chip(scene.swatch),
        h(
          "span",
          { class: "scene-option-text" },
          h("span", { class: "scene-option-name" }, scene.name),
          scene.detail &&
            h("span", { class: "scene-option-detail" }, scene.detail),
        ),
        (scene.orbisOnly || scene.dark) &&
          h(
            "span",
            { class: "scene-flags" },
            scene.orbisOnly &&
              h(
                "span",
                {
                  class: "option-tag",
                  title:
                    "Shows an Orbis extension beyond the style spec: GL JS and MapLibre Native can't draw it",
                },
                "Orbis only",
              ),
            scene.dark && h("span", { class: "dark-flag" }, "Dark"),
          ),
      ),
    );

  const groupRow = (group) =>
    h(
      "li",
      {},
      h(
        "button",
        {
          class: "scene-option group-option",
          type: "button",
          "data-group": group.name,
          "aria-label": `${group.name}: ${group.scenes.length} maps`,
          onclick: () => open(group),
        },
        h(
          "span",
          { class: "chip-stack", "aria-hidden": "true" },
          group.scenes.slice(0, 3).map((scene) => chip(scene.swatch)),
        ),
        h(
          "span",
          { class: "scene-option-text" },
          h("span", { class: "scene-option-name" }, group.name),
          h(
            "span",
            { class: "scene-option-detail" },
            group.scenes.map((s) => s.name).join(", "),
          ),
        ),
        h("span", { class: "chevron next", "aria-hidden": "true" }),
      ),
    );

  const groupList = h("ul", { class: "scene-list" }, groups.map(groupRow));
  const panels = new Map(
    groups.map((group) => [
      group,
      h(
        "ul",
        {
          class: "scene-list",
          "data-group-scenes": group.name,
          "aria-label": group.name,
          hidden: true,
        },
        group.scenes.map(option),
      ),
    ]),
  );
  list.replaceChildren(groupList, ...panels.values());

  /** Shows `group`'s entries, or with none the groups. */
  let shown = null;
  function open(group) {
    const from = shown;
    shown = group;
    groupList.hidden = Boolean(group);
    for (const [g, panel] of panels) panel.hidden = g !== group;
    back.hidden = !group;
    title.textContent = group ? group.name : "Choose a map";
    list.scrollTop = 0;
    if (group) {
      const panel = panels.get(group);
      (
        panel.querySelector('[aria-current="true"]') ??
        panel.querySelector("button")
      ).focus({ preventScroll: true });
      panel
        .querySelector('[aria-current="true"]')
        ?.scrollIntoView({ block: "nearest" });
    } else if (from) {
      const row = groupList.querySelector(
        `[data-group="${CSS.escape(from.name)}"]`,
      );
      row.focus({ preventScroll: true });
      row.scrollIntoView({ block: "nearest" });
    }
  }
  back.addEventListener("click", () => open(null));
  // Left, or Backspace, goes back up from a group too.
  list.addEventListener("keydown", (e) => {
    if (shown && (e.key === "ArrowLeft" || e.key === "Backspace")) {
      e.preventDefault();
      open(null);
    }
  });

  // On wide screens the sheet opens under the button.
  popover.addEventListener("beforetoggle", (e) => {
    if (e.newState !== "open") return;
    open(null);
    const rect = button.getBoundingClientRect();
    popover.style.setProperty("--sheet-top", `${rect.bottom + 8}px`);
    popover.style.setProperty("--sheet-left", `${rect.left}px`);
  });
  popover.addEventListener("toggle", (e) => {
    if (e.newState === "open")
      groupList
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
      for (const row of list.querySelectorAll("[data-scene]")) {
        row.setAttribute(
          "aria-current",
          String(row.dataset.scene === scene.key),
        );
      }
      for (const group of groups) {
        groupList
          .querySelector(`[data-group="${CSS.escape(group.name)}"]`)
          .setAttribute(
            "aria-current",
            String(group.scenes.some((s) => s.key === scene.key)),
          );
      }
    },
  };
}
