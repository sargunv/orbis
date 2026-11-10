// Offline areas: download what the map shows, from its zoom to three
// deeper, into the browser's private file system; maps draw downloaded
// areas without the network (orbis's offline regions).
import { deleteRegion, downloadRegion, regions } from "orbis";

import { h } from "./dom.js";

const mib = (bytes) => `${(bytes / 1048576).toFixed(1)} MiB`;

/** `view()` resolves to the map's visible bounds, zoom, style URL and a name, or null. */
export function offline({ view }) {
  const dialog = document.getElementById("offline");
  const list = h("ul", { class: "regions" });
  const status = h("p", { class: "offline-status", role: "status" });
  const button = h(
    "button",
    { type: "button", class: "primary", onclick: () => void download() },
    "Download this view",
  );
  dialog.querySelector(".offline-body").replaceChildren(button, status, list);

  async function refresh() {
    const found = await regions();
    list.replaceChildren(
      ...found
        .sort((a, b) => b.id - a.id)
        .map((region) =>
          h(
            "li",
            {},
            h(
              "span",
              { class: "region-text" },
              h("span", { class: "region-name" }, region.name),
              h(
                "span",
                { class: "region-detail" },
                `z${region.zooms[0]}–${region.zooms[1]}, ${mib(region.bytes)}${region.complete ? "" : ", incomplete"}`,
              ),
            ),
            h(
              "button",
              {
                type: "button",
                class: "tool quiet",
                title: "Delete",
                onclick: async () => {
                  await deleteRegion(region.id);
                  await refresh();
                },
              },
              h("span", { class: "icon icon-close", "aria-hidden": "true" }),
              h("span", { class: "visually-hidden" }, `Delete ${region.name}`),
            ),
          ),
        ),
    );
    if (!found.length)
      list.append(
        h("li", { class: "region-empty" }, "No areas downloaded yet."),
      );
  }

  async function download() {
    const current = await view();
    if (!current) return;
    button.disabled = true;
    status.textContent = "Starting…";
    const started = performance.now();
    try {
      await downloadRegion(current, (event) => {
        if (event.type === "progress") {
          status.textContent = `${event.done} of ${event.total} requests, ${mib(event.bytes)}`;
        } else {
          const seconds = ((performance.now() - started) / 1000).toFixed(1);
          status.textContent =
            event.type === "complete"
              ? `Downloaded ${mib(event.bytes)} in ${seconds} s${event.failed ? `; ${event.failed} requests failed` : ""}.`
              : `The download failed: ${event.message}`;
          button.disabled = false;
          void refresh();
        }
      });
      void refresh();
    } catch (e) {
      status.textContent = `The download failed: ${e instanceof Error ? e.message : e}`;
      button.disabled = false;
    }
  }

  return {
    open() {
      void refresh();
      dialog.showModal();
    },
  };
}
