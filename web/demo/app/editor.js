// The style editor: the active style's layers by source layer, with
// visibility toggles and paint controls generated from the v8 spec, each
// change one transaction through `map.edit`; the whole style as JSON for
// anything else. The footer counts the frames until an edit shows.
//
// In development with ORB_PLAYGROUND_STYLE set, accepted styles and hover
// state also go to that file, which `orb-demo --watch-style` follows.
import spec from "virtual:paint-spec";
import { h, hidden, icon } from "./dom.js";
import { HIGHLIGHT_PREFIX } from "./highlight.js";

const isExpression = (value) => Array.isArray(value) || (value !== null && typeof value === "object");
const humanize = (text) => text.replace(/[_-]+/g, " ").replace(/^./, (c) => c.toUpperCase());
const visible = (layer) => layer.layout?.visibility !== "none";
const ownLayers = (style) => style.layers.filter((l) => !l.id.startsWith(HIGHLIGHT_PREFIX));
const MAIN_COLOR = { fill: "fill-color", line: "line-color", circle: "circle-color", symbol: "text-color",
  background: "background-color", "fill-extrusion": "fill-extrusion-color", heatmap: "heatmap-color" };

// CSS colours to #rrggbb and alpha, through a canvas.
const context = new OffscreenCanvas(1, 1).getContext("2d", { willReadFrequently: true });
function parseColor(text) {
  context.clearRect(0, 0, 1, 1);
  context.fillStyle = "#000";
  context.fillStyle = text;
  context.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = context.getImageData(0, 0, 1, 1).data;
  const hex = `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
  return { hex, alpha: a / 255 };
}
function withHex(hex, alpha) {
  if (alpha >= 0.999) return hex;
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${Number(alpha.toFixed(3))})`;
}

/** A slider's range for a number property. */
function range(name, min, max, value) {
  const lo = min ?? (/translate|offset/.test(name) ? -Math.max(20, Math.abs(value) * 2) : 0);
  let hi = max;
  if (hi === null) {
    if (/opacity|intensity|strength|exaggeration/.test(name)) hi = 1;
    else if (/rotate/.test(name)) hi = 360;
    else if (/blur/.test(name)) hi = 20;
    else hi = Math.max(24, Math.ceil(Math.abs(value) * 2));
  }
  hi = Math.max(hi, value);
  return { lo, hi, step: hi - lo <= 2 ? 0.01 : hi - lo <= 40 ? 0.1 : 1 };
}

export function editor({ onClose, onReplaced }) {
  const root = document.getElementById("editor");
  let map;
  let style;
  let original;
  let frame = 0;
  let pending = [];
  let tab = "layers";
  let hovered;
  const expanded = new Set();

  // Edits: control changes queue per property; one transaction in flight.
  const queued = new Map();
  let inflight = false;

  const applied = h("output", { class: "applied", "aria-live": "polite" }, "Edits apply as you make them");
  const inspect = h("button", { class: "text-button inspect", type: "button", hidden: true,
    onclick: () => hovered && reveal(hovered) });
  const filter = h("input", { class: "field layer-filter", type: "search", placeholder: "Filter layers",
    "aria-label": "Filter layers", oninput: () => renderLayers() });
  const groups = h("div", { class: "layer-groups" });
  const json = h("textarea", { spellcheck: "false", wrap: "off", "aria-label": "Style JSON",
    "aria-describedby": "json-error", oninput: () => { clearTimeout(jsonTimer); jsonTimer = setTimeout(applyJson, 300); } });
  const jsonError = h("div", { class: "json-error", id: "json-error", role: "alert" });
  let jsonTimer;
  const tabs = {
    layers: h("section", { class: "tab-panel", id: "editor-layers", role: "tabpanel", "aria-labelledby": "tab-layers" },
      filter, groups),
    json: h("section", { class: "tab-panel json-panel", id: "editor-json", role: "tabpanel", "aria-labelledby": "tab-json", hidden: true },
      json, jsonError),
  };
  const tabButton = (key, label) => h("button", {
    type: "button", role: "tab", id: `tab-${key}`, "aria-controls": `editor-${key}`, "aria-selected": String(key === tab),
    onclick: () => showTab(key),
  }, label);
  const tabList = h("div", { class: "tabs", role: "tablist" }, tabButton("layers", "Layers"), tabButton("json", "JSON"));
  root.replaceChildren(
    h("button", { class: "editor-grip", type: "button", "aria-label": "Resize the editor",
      onclick: () => root.classList.toggle("tall") }),
    h("header", { class: "editor-head" },
      h("h2", {}, "Style"),
      h("button", { class: "text-button", type: "button", onclick: () => reset() }, "Reset"),
      h("button", { class: "tool quiet", type: "button", title: "Close the editor", onclick: () => onClose() },
        icon("close"), hidden("Close the editor"))),
    tabList, tabs.layers, tabs.json,
    h("footer", { class: "editor-foot" }, applied, inspect));
  root.addEventListener("keydown", (e) => { if (e.key === "Escape") onClose(); });

  function showTab(key) {
    tab = key;
    for (const button of tabList.children) button.setAttribute("aria-selected", String(button.id === `tab-${key}`));
    tabs.layers.hidden = key !== "layers";
    tabs.json.hidden = key !== "json";
    if (key === "json") {
      json.value = JSON.stringify({ ...style, layers: ownLayers(style) }, null, 2);
      json.removeAttribute("aria-invalid");
      jsonError.textContent = "";
    } else renderLayers();
  }

  async function edit(build) {
    const before = frame;
    const started = performance.now();
    const generation = await map.edit(build);
    pending.push({ generation, frame: before, ms: performance.now() - started });
    return generation;
  }

  function set(layer, kind, name, value, control) {
    queued.set(`${layer}\0${kind}\0${name}`, { layer, kind, name, value, control });
    void pump();
  }
  async function pump() {
    if (inflight) return;
    inflight = true;
    try {
      while (queued.size) {
        const batch = [...queued.values()];
        queued.clear();
        try {
          await edit((tx) => {
            for (const c of batch) {
              if (c.kind === "paint") tx.setPaintProperty(c.layer, c.name, c.value);
              else tx.setLayoutProperty(c.layer, c.name, c.value);
            }
          });
          for (const c of batch) {
            const layer = style.layers.find((l) => l.id === c.layer);
            if (!layer) continue;
            layer[c.kind] ??= {};
            if (c.value === undefined) delete layer[c.kind][c.name];
            else layer[c.kind][c.name] = c.value;
            c.control?.ok();
          }
        } catch (e) {
          for (const c of batch) c.control?.fail(String(e instanceof Error ? e.message : e));
        }
      }
    } finally { inflight = false; }
  }

  async function applyJson() {
    let next;
    try { next = JSON.parse(json.value); } catch (e) {
      jsonError.textContent = String(e.message);
      json.setAttribute("aria-invalid", "true");
      return;
    }
    try {
      await edit((tx) => tx.setStyle(next));
      style = structuredClone(next);
      onReplaced();
      jsonError.textContent = "";
      json.removeAttribute("aria-invalid");
    } catch (e) {
      jsonError.textContent = String(e instanceof Error ? e.message : e);
      json.setAttribute("aria-invalid", "true");
    }
  }

  async function reset() {
    const next = JSON.parse(original);
    await edit((tx) => tx.setStyle(next));
    style = next;
    onReplaced();
    if (tab === "json") showTab("json"); else renderLayers();
  }

  // The layer list.
  function groupKey(layer) {
    if (layer["source-layer"]) return layer["source-layer"];
    if (layer.type === "background") return "background";
    return layer.source ?? layer.type;
  }

  function renderLayers() {
    if (!style || tab !== "layers") return;
    const query = filter.value.trim().toLowerCase();
    const layers = ownLayers(style).filter((l) => !query || l.id.toLowerCase().includes(query)
      || (l["source-layer"] ?? "").includes(query));
    const byGroup = new Map();
    for (const layer of layers) {
      const key = groupKey(layer);
      if (!byGroup.has(key)) byGroup.set(key, []);
      byGroup.get(key).push(layer);
    }
    const small = layers.length <= 14;
    const scroll = groups.scrollTop;
    groups.replaceChildren(...[...byGroup].map(([key, members]) => {
      const allVisible = members.every(visible);
      const toggle = h("button", {
        class: "eye", type: "button", "aria-pressed": String(allVisible),
        title: allVisible ? `Hide ${humanize(key)}` : `Show ${humanize(key)}`,
        onclick: (e) => {
          e.preventDefault();
          e.stopPropagation();
          for (const layer of members) set(layer.id, "layout", "visibility", allVisible ? "none" : "visible");
          queueMicrotask(() => void settle().then(renderLayers));
        },
      }, icon(allVisible ? "eye" : "eye-off"), hidden(allVisible ? `Hide ${humanize(key)}` : `Show ${humanize(key)}`));
      const open = small || query || members.some((l) => expanded.has(l.id));
      return h("details", { class: "layer-group", open },
        h("summary", {}, h("span", {}, humanize(key), " ", h("span", { class: "count" }, String(members.length))),
          h("span", { style: { marginLeft: "auto" } }, toggle)),
        members.map(layerRow));
    }));
    groups.scrollTop = scroll;
    markHovered();
  }

  const settle = async () => { while (inflight || queued.size) await new Promise((r) => setTimeout(r, 16)); };

  function layerRow(layer) {
    const open = expanded.has(layer.id);
    const color = layer.paint?.[MAIN_COLOR[layer.type]];
    const dot = h("span", { class: `type-dot ${layer.type}`, "aria-hidden": "true" });
    if (typeof color === "string") dot.style.setProperty("--dot", color);
    const shown = visible(layer);
    const props = open ? propsPanel(layer) : null;
    const row = h("div", { class: `layer${open ? " open" : ""}${shown ? "" : " hidden-layer"}`, "data-layer": layer.id },
      h("div", { class: "layer-row" },
        h("button", {
          class: "eye", type: "button", "aria-pressed": String(shown), title: shown ? "Hide layer" : "Show layer",
          onclick: () => {
            set(layer.id, "layout", "visibility", shown ? "none" : "visible");
            void settle().then(renderLayers);
          },
        }, icon(shown ? "eye" : "eye-off"), hidden(`${shown ? "Hide" : "Show"} ${layer.id}`)),
        h("button", {
          class: "layer-name", type: "button", "aria-expanded": String(open),
          onclick: () => { if (open) expanded.delete(layer.id); else expanded.add(layer.id); renderLayers(); },
        }, dot, h("span", {}, layer.id)),
        h("span", { class: "layer-type" }, layer.type)),
      props);
    return row;
  }

  function propsPanel(layer) {
    const props = spec[layer.type] ?? [];
    const set = props.filter(([name]) => layer.paint?.[name] !== undefined);
    const unset = props.filter(([name]) => layer.paint?.[name] === undefined);
    return h("div", { class: "props" },
      set.length === 0 && h("div", { class: "prop-name" }, "No paint properties set"),
      set.map((p) => propControl(layer, p)),
      unset.length > 0 && h("details", { class: "more-props" },
        h("summary", {}, `${unset.length} more at their defaults`),
        h("div", { class: "props", style: { padding: "0" } }, unset.map((p) => propControl(layer, p)))));
  }

  function propControl(layer, [name, type, fallback, min, max, units, values]) {
    const value = layer.paint?.[name] ?? fallback;
    const isSet = layer.paint?.[name] !== undefined;
    const id = `prop-${layer.id}-${name}`.replace(/[^\w-]/g, "_");
    const error = h("div", { class: "prop-error", role: "alert" });
    const fields = [];
    const control = {
      ok() { error.textContent = ""; for (const f of fields) f.removeAttribute("aria-invalid"); },
      fail(message) { error.textContent = message; for (const f of fields) f.setAttribute("aria-invalid", "true"); },
    };
    const write = (v) => set(layer.id, "paint", name, v, control);
    const label = h("label", { class: "prop-name", for: id }, units ? `${name} (${units})` : name);
    let input;
    if (isExpression(value) || ["array", "numberArray", "colorArray", "resolvedImage", "padding"].includes(type)) {
      input = h("textarea", { class: "field", id, spellcheck: "false", rows: isExpression(value) ? 3 : 1,
        value: value === null ? "" : JSON.stringify(value),
        onchange: (e) => {
          try { write(JSON.parse(e.target.value)); } catch (err) {
            if (type === "resolvedImage" || type === "color") write(e.target.value); else control.fail(String(err.message));
          }
        } });
      fields.push(input);
    } else if (type === "color") {
      const { hex, alpha } = parseColor(value ?? "#000000");
      const text = h("input", { class: "field", id, value: value ?? "", "aria-label": `${name} value`,
        onchange: (e) => { picker.value = parseColor(e.target.value).hex; write(e.target.value); } });
      const picker = h("input", { type: "color", value: hex, "aria-label": `${name} colour`,
        oninput: (e) => { const v = withHex(e.target.value, alpha); text.value = v; write(v); } });
      label.removeAttribute("for");
      label.id = `${id}-label`;
      fields.push(text);
      input = h("div", { class: "prop-control" }, picker, text);
    } else if (type === "number") {
      const v = typeof value === "number" ? value : 0;
      const { lo, hi, step } = range(name, min, max, v);
      const number = h("input", { class: "field num", type: "number", id, value: String(v), step: String(step),
        min: min ?? undefined, max: max ?? undefined,
        oninput: (e) => { if (e.target.value !== "" && Number.isFinite(+e.target.value)) { slider.value = e.target.value; write(+e.target.value); } } });
      const slider = h("input", { type: "range", min: String(lo), max: String(hi), step: String(step), value: String(v),
        "aria-label": `${name} slider`,
        oninput: (e) => { number.value = e.target.value; write(+e.target.value); } });
      fields.push(number);
      input = h("div", { class: "prop-control" }, slider, number);
    } else if (type === "enum") {
      input = h("select", { class: "field", id, onchange: (e) => write(e.target.value) },
        values.map((v) => h("option", { value: v, selected: v === value }, v)));
      fields.push(input);
    } else if (type === "boolean") {
      input = h("input", { type: "checkbox", id, checked: !!value, onchange: (e) => write(e.target.checked) });
    }
    if (!input) return null;
    const wrapped = input.classList?.contains("prop-control") ? input : h("div", { class: "prop-control" }, input);
    return h("div", { class: `prop${isSet ? "" : " unset"}` }, label, wrapped, error);
  }

  function markHovered() {
    for (const row of groups.querySelectorAll(".layer.hovered")) row.classList.remove("hovered");
    if (hovered) groups.querySelector(`.layer[data-layer="${CSS.escape(hovered)}"]`)?.classList.add("hovered");
  }

  function reveal(layerId) {
    if (tab !== "layers") showTab("layers");
    filter.value = "";
    expanded.add(layerId);
    renderLayers();
    const row = groups.querySelector(`.layer[data-layer="${CSS.escape(layerId)}"]`);
    row?.scrollIntoView({ block: "nearest" });
    row?.querySelector(".layer-name")?.focus({ preventScroll: true });
  }

  return {
    get open() { return !root.hidden; },
    get style() { return style; },
    /** A new map: its style as loaded is what Reset returns to. */
    async attach(next) {
      map = next;
      frame = 0;
      pending = [];
      expanded.clear();
      hovered = undefined;
      style = await map.getStyle();
      original = JSON.stringify(style);
      if (tab === "json" && !root.hidden) showTab("json"); else renderLayers();
    },
    /** Re-reads the style after an edit made outside the editor; Reset still returns to the style as loaded. */
    async refresh() {
      if (!map) return;
      style = await map.getStyle();
      if (tab === "json" && !root.hidden) showTab("json"); else renderLayers();
    },
    show(on) {
      root.hidden = !on;
      if (on) showTab(tab);
    },
    frame(ack) {
      frame++;
      pending = pending.filter((edit) => {
        if (ack.generation < edit.generation) return true;
        const frames = frame - edit.frame;
        applied.textContent = `Applied in ${frames} frame${frames === 1 ? "" : "s"}, ${edit.ms.toFixed(1)} ms`;
        applied.dataset.generation = String(edit.generation);
        applied.classList.add("fresh");
        clearTimeout(applied.fade);
        applied.fade = setTimeout(() => applied.classList.remove("fresh"), 900);
        return false;
      });
    },
    /** The feature under the pointer, as a hit. */
    inspect(hit) {
      hovered = hit?.layer;
      inspect.hidden = !hit;
      inspect.textContent = hit ? `${hit.layer}${hit.id !== null ? ` #${hit.id}` : ""}` : "";
      inspect.title = hit ? `Edit ${hit.layer}` : "";
      markHovered();
    },
    reveal,
  };
}
