// The map options panel: switches for what the entry's style draws
// (terrain, the light's anchor and its sun's time, buildings and their shadows, the globe and
// its sky, labels, the colour space), grouped, and only those the style
// can take. Each choice is one edit of the live style through `map.edit`,
// so nothing reloads, and a comparison's other maps follow it as they
// follow every edit (compare.js), ignoring what they can't draw. A choice
// other than the entry's own stays in the URL (`&terrain=1`, `&light=map`,
// `&shadows=0`, …), so a link reproduces the view; the style as the entry
// designs it, before the URL's choices, is what each option's default and
// its "on" read from.
import { h } from "./dom.js";

const COLOR_SPACE = "dev.sargunv.orbis:color-space";
const GROUND_LIGHTING = "dev.sargunv.orbis:ground-lighting";
const DATETIME = "dev.sargunv.orbis:datetime";
const HILLSHADE_ANCHOR = "hillshade-illumination-anchor";
const AMBIENT_OCCLUSION = ["wall-intensity", "ground-intensity"].map(
  (p) => `dev.sargunv.orbis:fill-extrusion-ambient-occlusion-${p}`,
);

const ofType = (style, type) => style.layers.filter((l) => l.type === type);
const visible = (layer) => layer.layout?.visibility !== "none";
/** Whether a number property as written (a number or an expression) is on: not zero, `fallback` for unset. */
const nonzero = (value, fallback) => (value ?? fallback) !== 0;

/** An option that shows or hides every `type` layer, back to each one's visibility as designed. */
const layersOf = (type) => ({
  supported: ({ designed }) => ofType(designed, type).some(visible),
  read: (style) => ofType(style, type).some(visible),
  apply(tx, on, { style, designed }) {
    const designedShown = new Set(
      ofType(designed, type)
        .filter(visible)
        .map((l) => l.id),
    );
    for (const layer of ofType(style, type)) {
      tx.setLayoutProperty(
        layer.id,
        "visibility",
        on && designedShown.has(layer.id) ? "visible" : "none",
      );
    }
  },
});

/** An option that sets one number in the style's `sky`: `off` for off, the designed value (or unset) for on. */
const skyNumber = (
  name,
  fallback,
  off,
  isOn = (v) => (v ?? fallback) !== off,
) => ({
  supported: ({ designed }) =>
    Boolean(designed.sky) && isOn(designed.sky[name]),
  read: (style) => Boolean(style.sky) && isOn(style.sky[name]),
  apply: (tx, on, { style, designed }) =>
    tx.setSky({ ...style.sky, [name]: on ? designed.sky[name] : off }),
});

const projectionType = (style) => {
  const type = style.projection?.type;
  return type === undefined
    ? "mercator"
    : typeof type === "string"
      ? type
      : "globe";
};
/** The entry's projection: the one it applies, else its style's. */
const projectionByDesign = ({ scene, designed }) =>
  scene.projection?.type ?? projectionType(designed);

/** The extrusion layers whose ambient occlusion the style designs. */
const occluded = (style) =>
  ofType(style, "fill-extrusion").filter((l) =>
    AMBIENT_OCCLUSION.some((p) => nonzero(l.paint?.[p], 0)),
  );

/**
 * The options, in the panel's order. Each has a `key` (its URL parameter
 * and `data-option`), a `label`, and either `kind: "switch"` or `choices`;
 * `supported(context)` says whether the entry's style can take it,
 * `read(style)` gives its value in a style, `apply(tx, value, context)`
 * edits the style to it, and `initial(context)`, where given, is the
 * entry's default when not as the style was designed (terrain the switch
 * puts in). An entry's `off` lists switches its style designs that start
 * off (styles.js). `needs` names an option, and its value, without which this
 * one draws nothing, and says so: it's disabled then. `orbisOnly` marks a preview GL JS
 * and MapLibre Native don't draw.
 */
export const OPTIONS = [
  {
    key: "terrain",
    group: "Terrain and light",
    label: "Terrain",
    kind: "switch",
    supported: ({ scene, terrain }) => Boolean(scene.terrain && terrain),
    read: (style) => Boolean(style.terrain),
    apply: (tx, on, { terrain }) => tx.setTerrain(on ? terrain : null),
    // An entry with its style's own terrain draws it by default; one with
    // terrain of its own puts it in only when asked.
    initial: ({ scene }) => scene.terrain === true,
  },
  {
    key: "hillshade",
    group: "Terrain and light",
    label: "Hillshade",
    kind: "switch",
    ...layersOf("hillshade"),
  },
  {
    key: "light",
    group: "Terrain and light",
    label: "Light",
    note: "Fixed to the ground, or turning with the view",
    choices: [
      ["map", "Map"],
      ["viewport", "Viewport"],
    ],
    supported: ({ scene }) => Boolean(scene.light),
    // The first hillshade layer's anchor, else the light's, else the spec's default.
    read: (style) =>
      ofType(style, "hillshade")[0]?.paint?.[HILLSHADE_ANCHOR] ??
      style.light?.anchor ??
      "viewport",
    apply(tx, anchor, { style }) {
      for (const layer of ofType(style, "hillshade"))
        tx.setPaintProperty(layer.id, HILLSHADE_ANCHOR, anchor);
      tx.setLight({ ...style.light, anchor });
    },
  },
  {
    key: "night",
    group: "Terrain and light",
    label: "Night side",
    note: "The ground darkened where the sun is down",
    kind: "switch",
    orbisOnly: true,
    supported: ({ designed }) => nonzero(designed.light?.[GROUND_LIGHTING], 0),
    read: (style) => nonzero(style.light?.[GROUND_LIGHTING], 0),
    apply: (tx, on, { style, designed }) =>
      tx.setLight({
        ...style.light,
        [GROUND_LIGHTING]: on ? designed.light[GROUND_LIGHTING] : 0,
      }),
  },
  {
    key: "now",
    group: "Terrain and light",
    label: "Sun now",
    note: "Where the sun stands at this moment",
    kind: "switch",
    orbisOnly: true,
    supported: ({ designed }) => nonzero(designed.light?.[GROUND_LIGHTING], 0),
    read: (style) => style.light?.[DATETIME] !== undefined,
    // The time is a map-anchored sun's: it puts the light on the map, as
    // the Light option's Map does.
    apply(tx, on, { style }) {
      const { [DATETIME]: _, ...light } = style.light ?? {};
      if (on)
        for (const layer of ofType(style, "hillshade"))
          tx.setPaintProperty(layer.id, HILLSHADE_ANCHOR, "map");
      tx.setLight(
        on ? { ...light, anchor: "map", [DATETIME]: Date.now() } : light,
      );
    },
  },
  {
    key: "buildings",
    group: "Buildings",
    label: "3D buildings",
    kind: "switch",
    ...layersOf("fill-extrusion"),
  },
  {
    key: "shadows",
    group: "Buildings",
    label: "Shadows",
    kind: "switch",
    orbisOnly: true,
    needs: ["buildings", true, "With 3D buildings on"],
    supported: ({ designed }) =>
      nonzero(designed.light?.["dev.sargunv.orbis:shadow-intensity"], 0) &&
      ofType(designed, "fill-extrusion").length > 0,
    read: (style) =>
      nonzero(style.light?.["dev.sargunv.orbis:shadow-intensity"], 0),
    apply: (tx, on, { style, designed }) =>
      tx.setLight({
        ...style.light,
        "dev.sargunv.orbis:shadow-intensity": on
          ? designed.light["dev.sargunv.orbis:shadow-intensity"]
          : 0,
      }),
  },
  {
    key: "ao",
    group: "Buildings",
    label: "Ambient occlusion",
    kind: "switch",
    orbisOnly: true,
    needs: ["buildings", true, "With 3D buildings on"],
    supported: ({ designed }) => occluded(designed).length > 0,
    read: (style) => occluded(style).length > 0,
    apply(tx, on, { style, designed }) {
      const present = new Set(style.layers.map((l) => l.id));
      for (const layer of occluded(designed).filter((l) => present.has(l.id))) {
        for (const name of AMBIENT_OCCLUSION) {
          if (layer.paint[name] !== undefined)
            tx.setPaintProperty(layer.id, name, on ? layer.paint[name] : 0);
        }
      }
    },
  },
  {
    key: "projection",
    group: "Globe and sky",
    label: "Projection",
    choices: [
      ["globe", "Globe"],
      ["mercator", "Mercator"],
    ],
    supported: (context) =>
      ["globe", "mercator"].includes(projectionByDesign(context)),
    read: projectionType,
    apply: (tx, type, { style }) =>
      tx.setProjection({ ...style.projection, type }),
    initial: projectionByDesign,
  },
  // The spec's defaults: atmosphere 0.8, fog over the far half of the view.
  {
    key: "atmosphere",
    group: "Globe and sky",
    label: "Atmosphere",
    kind: "switch",
    needs: ["projection", "globe", "On the globe"],
    ...skyNumber("atmosphere-blend", 0.8, 0),
  },
  // Fog draws only over terrain; a fog-ground blend of 1 puts it past the horizon.
  {
    key: "fog",
    group: "Globe and sky",
    label: "Fog",
    kind: "switch",
    needs: ["terrain", true, "With terrain on"],
    ...skyNumber("fog-ground-blend", 0.5, 1, (v) => (v ?? 0.5) < 1),
  },
  {
    key: "stars",
    group: "Globe and sky",
    label: "Stars",
    kind: "switch",
    orbisOnly: true,
    needs: ["projection", "globe", "On the globe"],
    ...skyNumber("dev.sargunv.orbis:star-opacity", 0, 0),
  },
  {
    key: "labels",
    group: "Map",
    label: "Labels",
    kind: "switch",
    ...layersOf("symbol"),
  },
  {
    key: "colors",
    group: "Map",
    label: "Colours",
    note: "Mixed in linear light, or in sRGB as GL JS mixes them",
    orbisOnly: true,
    choices: [
      ["linear", "Linear"],
      ["srgb", "sRGB"],
    ],
    supported: ({ designed }) =>
      ["linear", "srgb"].includes(designed.metadata?.[COLOR_SPACE]),
    read: (style) =>
      style.metadata?.[COLOR_SPACE] === "linear" ? "linear" : "srgb",
    // The metadata isn't an edit of its own: the style, diffed, with it changed.
    apply: (tx, space, { style }) =>
      tx.setStyle({
        ...style,
        metadata: { ...style.metadata, [COLOR_SPACE]: space },
      }),
  },
];
const encode = (option, value) =>
  option.kind === "switch" ? (value ? "1" : "0") : value;
function decode(option, text) {
  if (text === null) return undefined;
  if (option.kind === "switch")
    return text === "1" ? true : text === "0" ? false : undefined;
  return option.choices.some(([value]) => value === text) ? text : undefined;
}

/** The renderer's choices: `?backend=` (the package's `backend` option), Auto by default. */
const BACKENDS = [
  ["auto", "Auto"],
  ["webgpu", "WebGPU"],
  ["webgl2", "WebGL 2"],
];
const BACKEND_NAMES = { webgpu: "WebGPU", webgl2: "WebGL 2" };
/** The renderer the URL asks for: `auto`, `webgpu` or `webgl2`. */
export function wantedBackend(query = new URLSearchParams(location.search)) {
  const value = query.get("backend");
  return BACKENDS.some(([v]) => v === value) ? value : "auto";
}

/**
 * The panel and its button. `params()` reads the URL's query and
 * `setParam(name, value)` writes one (null removes it); `onBackend(value)`
 * starts the map again on another renderer.
 */
export function mapOptions({ params, setParam, onBackend }) {
  const panel = document.getElementById("options");
  const body = document.getElementById("options-body");
  let map;
  let context;
  let available = [];
  /** Each available option's control: `show(value, disabled)`. */
  const controls = new Map();
  let values = {};
  let ready = false;
  /** Edits in flight: a re-read waits for them, so it doesn't show a choice undone. */
  let editing = 0;
  /**
   * The panel's edits, one at a time: each reads the style as the one
   * before left it, as some write a whole root property (the light) or
   * the whole style (the colour space). `generation` counts attached
   * styles: an edit queued for an older one is dropped.
   */
  let queue = Promise.resolve();
  let generation = 0;
  const enqueue = (work) => {
    editing++;
    queue = queue
      .then(work)
      .catch((e) => console.error(e))
      .finally(() => {
        editing--;
      });
    return queue;
  };

  // On wide screens the panel opens beside its button.
  panel.addEventListener("beforetoggle", (e) => {
    if (e.newState !== "open") return;
    const rect = document
      .getElementById("options-button")
      .getBoundingClientRect();
    panel.style.setProperty("--sheet-top", `${rect.top}px`);
    panel.style.setProperty("--sheet-right", `${innerWidth - rect.left + 8}px`);
  });

  /** The entry's default: off where it says so, else the option's own, else as designed. */
  const initial = (option) => {
    if (context.scene.off?.includes(option.key)) return false;
    return option.initial
      ? option.initial(context)
      : option.read(context.designed);
  };
  const wanted = (option) =>
    decode(option, params().get(option.key)) ?? initial(option);
  const disabled = (option) => {
    if (!ready) return true;
    const [key, value] = option.needs ?? [];
    return Boolean(key && controls.has(key) && values[key] !== value);
  };

  // The renderer, which every map has: its row stays through every render.
  const renderer = rendererGroup(wantedBackend(params()), onBackend);

  function render(loading = false) {
    controls.clear();
    const groups = new Map();
    for (const option of available) {
      if (!groups.has(option.group)) groups.set(option.group, []);
      groups.get(option.group).push(option);
    }
    body.replaceChildren(
      ...[...groups].map(([name, options], i) =>
        h(
          "section",
          {
            class: "option-group",
            "aria-labelledby": `option-group-${i}`,
          },
          h("h3", { id: `option-group-${i}` }, name),
          options.map(control),
        ),
      ),
    );
    if (!available.length) {
      body.replaceChildren(
        h(
          "p",
          { class: "options-empty" },
          loading ? "Loading the map…" : "This map has no options.",
        ),
      );
    }
    body.append(renderer.element);
  }
  render(true);

  const text = (option) =>
    h(
      "span",
      { class: "option-text" },
      h(
        "span",
        { class: "option-label", id: `option-${option.key}` },
        option.label,
        option.orbisOnly &&
          h(
            "span",
            {
              class: "option-tag",
              title:
                "A preview of a style-spec proposal: GL JS and MapLibre Native don't draw it",
            },
            "Orbis only",
          ),
      ),
      option.note && h("span", { class: "option-note" }, option.note),
    );

  function control(option) {
    if (option.kind === "switch") {
      const row = h(
        "button",
        {
          type: "button",
          class: "option-row",
          role: "switch",
          "data-option": option.key,
          "aria-checked": "false",
          "aria-labelledby": `option-${option.key}`,
          onclick: () =>
            pick(option, row.getAttribute("aria-checked") !== "true"),
        },
        text(option),
        h("span", { class: "switch", "aria-hidden": "true" }),
      );
      controls.set(option.key, {
        show(value, off) {
          row.setAttribute("aria-checked", String(value === true));
          row.disabled = off;
          row.title = off && ready ? option.needs[2] : "";
        },
      });
      return row;
    }
    const buttons = option.choices.map(([value, label]) =>
      h(
        "button",
        {
          type: "button",
          role: "radio",
          "data-value": value,
          "aria-checked": "false",
          tabindex: "-1",
          onclick: () => pick(option, value),
        },
        label,
      ),
    );
    const group = h(
      "span",
      {
        class: "segmented",
        role: "radiogroup",
        "aria-labelledby": `option-${option.key}`,
        // The arrow keys move the choice, as in a radio group.
        onkeydown(e) {
          const step = {
            ArrowLeft: -1,
            ArrowUp: -1,
            ArrowRight: 1,
            ArrowDown: 1,
          }[e.key];
          if (!step) return;
          e.preventDefault();
          const at = buttons.findIndex(
            (b) => b.getAttribute("aria-checked") === "true",
          );
          const next = buttons[(at + step + buttons.length) % buttons.length];
          next.focus();
          pick(option, next.dataset.value);
        },
      },
      buttons,
    );
    controls.set(option.key, {
      show(value, off) {
        row.title = off && ready ? (option.needs?.[2] ?? "") : "";
        for (const button of buttons) {
          const on = button.dataset.value === value;
          button.setAttribute("aria-checked", String(on));
          button.tabIndex = on ? 0 : -1;
          button.disabled = off;
        }
      },
    });
    const row = h(
      "div",
      { class: "option-row", "data-option": option.key },
      text(option),
      group,
    );
    return row;
  }

  function showAll() {
    for (const option of available)
      controls.get(option.key).show(values[option.key], disabled(option));
  }

  /** Applies `option`'s `value` to `style`, the map's as it now stands, if it differs; the style after. */
  async function set(option, value, style) {
    if (option.read(style) === value) return style;
    await map.edit((tx) => option.apply(tx, value, { ...context, style }));
    return map.getStyle();
  }

  async function pick(option, value) {
    if (!ready || values[option.key] === value) return;
    values = { ...values, [option.key]: value };
    showAll();
    const picked = generation;
    await enqueue(async () => {
      if (picked === generation) await set(option, value, await map.getStyle());
    });
    await refresh();
  }

  /** Reads every option's value from the map's style into the panel and the URL. */
  async function refresh() {
    if (!ready || editing) return;
    const style = await map.getStyle();
    if (!ready || editing) return;
    values = Object.fromEntries(available.map((o) => [o.key, o.read(style)]));
    showAll();
    for (const option of OPTIONS) {
      const value = values[option.key];
      setParam(
        option.key,
        available.includes(option) && value !== initial(option)
          ? encode(option, value)
          : null,
      );
    }
  }

  return {
    /** Another style is on its way: the panel waits for its options. */
    loading() {
      ready = false;
      generation++;
      available = [];
      render(true);
    },
    /**
     * `next`'s style is in for `scene`: `designed` is it as the entry
     * designs it, before the URL's choices, and `terrain` the terrain its
     * switch puts in. Applies the URL's choices (or the entry's defaults
     * where it has none), then shows them.
     */
    async attach(next, { scene, designed, terrain }) {
      map = next;
      context = { scene, designed, terrain };
      available = OPTIONS.filter((o) => o.supported(context));
      ready = false;
      values = {};
      render();
      showAll();
      const attached = ++generation;
      await enqueue(async () => {
        let style = await map.getStyle();
        for (const option of available)
          style = await set(option, wanted(option), style);
      });
      if (attached !== generation) return;
      ready = true;
      await refresh();
    },
    /** The map draws with `backend` (`webgpu` or `webgl2`), or didn't start (null). */
    drawing(backend) {
      renderer.drawing(backend);
    },
    /** Each available option's value, for the headless checks. */
    get values() {
      return ready ? { ...values } : null;
    },
    /** Whether the panel is open. */
    get open() {
      return panel.matches(":popover-open");
    },
  };
}

/** The renderer's row, its `wanted` choice picked: picking another calls `onPick`. */
function rendererGroup(wanted, onPick) {
  const active = h("span", { class: "option-label" }, "Starting…");
  const buttons = BACKENDS.map(([value, label]) =>
    h(
      "button",
      {
        type: "button",
        role: "radio",
        "data-value": value,
        "aria-checked": String(value === wanted),
        tabindex: value === wanted ? "0" : "-1",
        title:
          value === "webgl2"
            ? "In progress (#140): terrain draws flat, and the map says what else it draws otherwise"
            : undefined,
        onclick: () => value !== wanted && onPick(value),
      },
      label,
    ),
  );
  const row = h(
    "div",
    { class: "option-row wrap", "data-option": "backend" },
    h(
      "span",
      { class: "option-text" },
      active,
      h(
        "span",
        { class: "option-note" },
        "In progress, WebGL 2 draws terrain flat (#140)",
      ),
    ),
    h(
      "span",
      {
        class: "segmented",
        role: "radiogroup",
        "aria-labelledby": "option-group-renderer",
        // The arrow keys move the focus; a choice starts the map again,
        // so it waits for Enter or a click.
        onkeydown(e) {
          const step = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
          if (!step) return;
          e.preventDefault();
          const at = buttons.indexOf(document.activeElement);
          buttons[(at + step + buttons.length) % buttons.length].focus();
        },
      },
      buttons,
    ),
  );
  return {
    element: h(
      "section",
      { class: "option-group", "aria-labelledby": "option-group-renderer" },
      h("h3", { id: "option-group-renderer" }, "Renderer"),
      row,
    ),
    drawing(backend) {
      active.textContent = backend
        ? `Drawing with ${BACKEND_NAMES[backend]}`
        : "The map didn't start";
      row.dataset.active = backend ?? "";
    },
  };
}

/** The URL parameters the panel keeps, which belong to one entry. */
export const OPTION_PARAMS = OPTIONS.map((o) => o.key);
