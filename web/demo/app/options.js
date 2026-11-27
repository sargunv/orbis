// The map options panel: switches for what the entry's style draws
// (terrain and its relief, the light's anchor, buildings, the globe and
// its sky, labels), grouped, and only those the style can take. Each
// choice is one edit of the live style through `map.edit`, so nothing
// reloads, and a comparison's other maps follow it as they follow every
// edit (compare.js), ignoring what they can't draw. A choice
// other than the entry's own stays in the URL (`&terrain=1`, `&light=map`,
// `&blend=1`, …), so a link reproduces the view; the style as the entry
// designs it, before the URL's choices, is what each option's default and
// its "on" read from. An Orbis-only option starts off whatever the entry
// designs, its keys out of the style, so that by default every entry draws
// what GL JS and MapLibre Native could from the same style, Orbis's
// previews a switch away.
import catalogue from "../data/options.json";
import { h } from "./dom.js";

const HILLSHADE_ANCHOR = "hillshade-illumination-anchor";
/** Style-spec#1661's blends for the layer types the spec gives none. */
const LAYER_BLENDS = {
  hillshade: "dev.sargunv.orbis:hillshade-layer-blend",
  "color-relief": "dev.sargunv.orbis:color-relief-layer-blend",
};
/**
 * A blended layer's `metadata` key holding the paint it takes with its
 * blend off: colours made for the layer drawn normally.
 */
const UNBLENDED = "demo:unblended";
const ofType = (style, type) => style.layers.filter((l) => l.type === type);
const visible = (layer) => layer.layout?.visibility !== "none";

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

/**
 * An option that sets the number `name` in the style's `sky`: on at any
 * value but `off` (an expression included), with `default` standing in
 * for an unset one; on puts back the designed value (or unsets it).
 */
const skyNumber = ({ name, default: fallback, off }) => {
  const isOn = (sky) => Boolean(sky) && (sky[name] ?? fallback) !== off;
  return {
    supported: ({ designed }) => isOn(designed.sky),
    read: (style) => isOn(style.sky),
    apply: (tx, on, { style, designed }) =>
      tx.setSky({ ...style.sky, [name]: on ? designed.sky[name] : off }),
  };
};

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

/** The layers the style blends by style-spec#1661's preview. */
const blended = (style) =>
  style.layers.filter((l) => {
    const blend = l.paint?.[LAYER_BLENDS[l.type]];
    return blend !== undefined && blend !== "normal";
  });

/** What each option data/options.json gives no `layers` or `sky` does, by its key. */
const BESPOKE = {
  terrain: {
    supported: ({ scene, terrain }) => Boolean(scene.terrain && terrain),
    read: (style) => Boolean(style.terrain),
    apply: (tx, on, { terrain }) => tx.setTerrain(on ? terrain : null),
  },
  blend: {
    supported: ({ designed }) => blended(designed).length > 0,
    read: (style) => blended(style).length > 0,
    // Each layer's blend with the paint designed for it, or neither and
    // the paint it takes drawn normally.
    apply(tx, on, { style, designed }) {
      const present = new Set(style.layers.map((l) => l.id));
      for (const layer of blended(designed).filter((l) => present.has(l.id))) {
        const unblended = layer.metadata?.[UNBLENDED] ?? {};
        for (const name of Object.keys(unblended))
          tx.setPaintProperty(
            layer.id,
            name,
            on ? (layer.paint[name] ?? null) : unblended[name],
          );
        const blend = LAYER_BLENDS[layer.type];
        tx.setPaintProperty(layer.id, blend, on ? layer.paint[blend] : null);
      }
    },
  },
  light: {
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
  projection: {
    supported: (context) =>
      ["globe", "mercator"].includes(projectionByDesign(context)),
    read: projectionType,
    apply: (tx, type, { style }) =>
      tx.setProjection({ ...style.projection, type }),
    initial: projectionByDesign,
  },
};

/**
 * The options, in the panel's order, from data/options.json, which
 * orb-demo's panel reads too. Each has a `key` (its URL parameter and
 * `data-option`), a `group` and a `label`, an optional `note`, and
 * `choices`, `[value, label]` pairs, or none for a switch. `needs` names
 * an option, its `value` without which this one draws nothing, and `why`:
 * it's disabled then, saying so (fog draws only over terrain, the
 * atmosphere only around the globe). `orbisOnly` marks a preview GL JS and
 * MapLibre Native don't draw: off by default (a choice's `initial` is the
 * spec's), and off takes its keys out of the style. An entry's `off` lists
 * switches its style designs that start off (styles.js).
 *
 * What an option does is `layers` (a layer type it shows and hides), `sky`
 * (a number in the sky it turns off, `skyNumber`: `default` is the spec's,
 * and fog's `off` is 1 because a fog-ground blend of 1 puts the fog past
 * the horizon), or else code here and
 * in orb-demo's options.rs under its key (BESPOKE): `supported(context)`
 * says whether the entry's style can take it, `read(style)` gives its
 * value in a style, `apply(tx, value, context)` edits the style to it, and
 * `initial(context)`, where given, is the entry's default when not as the
 * style was designed.
 */
export const OPTIONS = catalogue.map((option) => {
  const does = option.layers
    ? layersOf(option.layers)
    : option.sky
      ? skyNumber(option.sky)
      : BESPOKE[option.key];
  if (!does)
    throw new Error(`data/options.json: give ${option.key} code in BESPOKE`);
  return { ...option, ...does };
});
const encode = (option, value) => (option.choices ? value : value ? "1" : "0");
function decode(option, text) {
  if (text === null) return undefined;
  if (!option.choices)
    return text === "1" ? true : text === "0" ? false : undefined;
  return option.choices.some(([value]) => value === text) ? text : undefined;
}

/** The entry's default: off where it says so or for an Orbis-only switch, else the option's own, else as designed. */
function initialOf(option, context) {
  if (context.scene.off?.includes(option.key)) return false;
  if (option.initial) return option.initial(context);
  if (option.orbisOnly) return false;
  return option.read(context.designed);
}
/** The value `query` asks of `option`, else its default. */
const wantedOf = (option, context, query) =>
  decode(option, query.get(option.key)) ?? initialOf(option, context);

/** The edits a transaction makes, made on a copy of `style` as JSON: `tx.style` is the result. */
function draft(style) {
  const tx = { style: structuredClone(style) };
  const put = (object, name, value) => {
    if (value === null || value === undefined) delete object[name];
    else object[name] = value;
  };
  const edit = (id, part, name, value) => {
    const layer = tx.style.layers.find((l) => l.id === id);
    layer[part] = { ...layer[part] };
    put(layer[part], name, value);
  };
  Object.assign(tx, {
    setPaintProperty: (id, name, value) => edit(id, "paint", name, value),
    setLayoutProperty: (id, name, value) => edit(id, "layout", name, value),
    setLight: (light) => put(tx.style, "light", light),
    setSky: (sky) => put(tx.style, "sky", sky),
    setTerrain: (terrain) => put(tx.style, "terrain", terrain),
    setProjection: (projection) => put(tx.style, "projection", projection),
  });
  return tx;
}

/**
 * `designed`, the demo's own style for `scene` (styles.js's `own`), with the
 * options `query` asks for and the entry's defaults for the rest: what the
 * map loads, so that it opens as it draws, without an Orbis-only preview
 * switched on and then off. `terrain` is the terrain its switch puts in.
 */
export function opening(designed, { scene, terrain }, query) {
  const context = { scene, designed, terrain };
  const tx = draft(designed);
  for (const option of OPTIONS.filter((o) => o.supported(context))) {
    const value = wantedOf(option, context, query);
    if (option.read(tx.style) !== value)
      option.apply(tx, value, { ...context, style: tx.style });
  }
  return tx.style;
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
   * before left it, as some write a whole root property (the light).
   * `generation` counts attached
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
  const initial = (option) => initialOf(option, context);
  const wanted = (option) => wantedOf(option, context, params());
  const disabled = (option) => {
    if (!ready) return true;
    const { key, value } = option.needs ?? {};
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
    if (!option.choices) {
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
          row.title = off && ready ? option.needs.why : "";
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
        row.title = off && ready ? (option.needs?.why ?? "") : "";
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
        onclick: () => value !== wanted && onPick(value),
      },
      label,
    ),
  );
  const row = h(
    "div",
    { class: "option-row wrap", "data-option": "backend" },
    h("span", { class: "option-text" }, active),
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
