// The feature popup: what was clicked, its properties, and a way into the
// style editor at the layer that drew it.
import { h, hidden, icon } from "./dom.js";

const MAX_ROWS = 12;

export function featurePopup(feature, { onClose, onEdit }) {
  const layer = feature.layer?.id ?? feature.source;
  const properties = Object.entries(feature.properties);
  const name = feature.properties["name:latin"] ?? feature.properties.name ?? feature.properties["name:en"];
  const where = feature.sourceLayer ?? feature.source;
  const meta = feature.id !== undefined ? `Feature ${feature.id} in ${where}` : `In ${where}`;
  return h("div", { class: "popup demo-annotation", role: "dialog", "aria-label": `Feature in ${layer}` },
    h("div", { class: "popup-head" },
      h("div", {},
        h("div", { class: "popup-title" }, typeof name === "string" ? name : layer),
        typeof name === "string" && h("div", { class: "popup-meta" }, `Layer ${layer}`),
        h("div", { class: "popup-meta" }, meta))),
    h("button", { class: "tool quiet", type: "button", title: "Close", onclick: onClose }, icon("close"), hidden("Close")),
    properties.length > 0 && h("dl", {}, properties.slice(0, MAX_ROWS).flatMap(([key, value]) => [
      h("dt", {}, key),
      h("dd", {}, typeof value === "object" ? JSON.stringify(value) : String(value)),
    ])),
    properties.length > MAX_ROWS && h("div", { class: "popup-meta" }, `${properties.length - MAX_ROWS} more properties`),
    feature.layer && h("div", { class: "popup-actions" },
      h("button", { class: "text-button", type: "button", onclick: () => onEdit(feature.layer.id) }, "Edit layer")));
}
