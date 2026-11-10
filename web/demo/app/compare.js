// The comparison: two maps of the same style side by side, split by a
// draggable divider, each side drawn by the engine it picks (Orbis, GL JS or
// MapLibre Native) and taking its own input. The app's Orbis map draws the
// first side that picks Orbis; every other side gets a map of its own over
// it, clipped to that side. The map that last took input leads and the
// others jump to its camera, so they don't echo each other's moves. The
// other maps take Orbis's style as it stands (the entry's projection and
// sky applied, less Orbis's preview properties, for which GL JS's
// validation refuses the whole style) and follow its edits, the map
// options'.
//
// An engine loads only when a side picks it. Each is a module with a
// `label` and `open(container, { style, camera, constraints })`, which
// returns, or resolves to, the map as `{ camera, jumpTo, onMove, stop,
// setStyle, loaded, remove }`, and `style` where it can give its style. In ENGINES, `unsupported(scene)` says why it
// can't show an entry, `once` why it can't draw both sides, and `isolated`
// that it needs a cross-origin isolated page (isolation.js).
import { h } from "./dom.js";
import { isolate, isolated } from "./isolation.js";

const ENGINES = {
  orbis: { name: "Orbis", load: () => import("./orbis.js") },
  gljs: {
    name: "GL JS",
    load: () => import("./gljs.js"),
    unsupported: (scene) => scene.unsupported?.gljs,
  },
  // Of two Native maps on a page, one sometimes stopped short of its last
  // update until the next input moved it.
  native: {
    name: "MapLibre Native",
    load: () => import("./native.js"),
    unsupported: (scene) => scene.unsupported?.native,
    once: "MapLibre Native draws one side at a time",
    isolated: true,
  },
};
/**
 * Properties of Orbis's previews that the other engines don't know, by
 * where they sit: style-spec#1889's stars, #1797's shadows, #1796's ambient
 * occlusion and #1047's rounded fills.
 */
const PREVIEWS = {
  sky: ["backdrop-color", "star-opacity"],
  light: ["shadow-intensity"],
  paint: [
    "wall-intensity",
    "wall-radius",
    "ground-intensity",
    "ground-radius",
  ].map((p) => `fill-extrusion-ambient-occlusion-${p}`),
  layout: ["fill-rounded-corner-distance"],
};
/** `object` less the properties `names` and their transitions; absent stays absent. */
const without = (object, names) =>
  object &&
  Object.fromEntries(
    Object.entries(object).filter(
      ([k]) => !names.some((p) => k === p || k === `${p}-transition`),
    ),
  );
/** What `?compare` alone compares. */
const DEFAULT = ["orbis", "gljs"];
const SIDES = ["left", "right"];

/** The engines `?compare=<left>,<right>` names, `?compare` alone the default, and no `?compare` none. */
export function parsePair(value) {
  if (value === null) return null;
  const named = value.split(",");
  const [left, right] = DEFAULT.map((fallback, i) =>
    ENGINES[named[i]] ? named[i] : fallback,
  );
  if (left !== right || !ENGINES[left].once) return [left, right];
  return [left, left === DEFAULT[1] ? DEFAULT[0] : DEFAULT[1]];
}
/** `?compare`'s value for `pair`: empty for the default. */
export const formatPair = (pair) =>
  pair.every((key, i) => key === DEFAULT[i]) ? "" : pair.join(",");

// Where the divider sits, as a fraction of the stage's width; it stays for
// the next comparison.
let split = 0.5;

/**
 * The switch in the app's menu, and the comparison while it's open.
 * `onChange(pair)` hears each change of engines (null when it closes);
 * `beforeReload` runs before a reload into isolation.
 */
export function comparison({ onChange, beforeReload }) {
  const toggle = document.getElementById("compare-toggle");
  let pair = null;
  let last = DEFAULT;
  let map;
  let scene;
  let shown;
  let isolating;

  // A side's engine that needs isolation reloads the page into it, once.
  function checkIsolation() {
    const needed = pair?.some(
      (key) =>
        ENGINES[key].isolated && !(scene && ENGINES[key].unsupported?.(scene)),
    );
    if (!needed || isolated() || isolating) return;
    // On success the page reloads; on failure the sides that need it say why.
    isolating = isolate(
      Object.keys(ENGINES).filter((k) => ENGINES[k].isolated),
      { beforeReload },
    );
    isolating.catch(update);
  }

  function update() {
    shown?.close();
    shown = undefined;
    toggle.disabled = !map;
    toggle.setAttribute("aria-checked", String(Boolean(pair)));
    if (!pair || !map) return;
    shown = show(map, scene, pair, { isolating, onPick });
  }
  function onPick(side, key) {
    pair = pair.map((k, i) => (i === side ? key : k));
    last = pair;
    onChange(pair);
    checkIsolation();
    update();
  }
  toggle.addEventListener("click", () => {
    pair = pair ? null : last;
    onChange(pair);
    checkIsolation();
    update();
  });

  return {
    /** Compares the engines of `next`, a pair of ENGINES' keys, or closes with null. */
    want(next) {
      pair = next;
      if (pair) last = pair;
      checkIsolation();
      update();
    },
    /** Compares `next`'s map of `entry`, once its style is in; with no map, closes. */
    attach(next, entry) {
      map = next;
      scene = entry;
      update();
    },
    /** Each side's engine, camera, whether it's settled and leads, and its note, for the headless checks. */
    get sides() {
      return shown?.sides();
    },
    /** Each side's style, where its engine gives it, for the headless checks. */
    styles() {
      return shown?.styles();
    },
  };
}

function show(map, scene, pair, { isolating, onPick }) {
  const stage = map.container.parentElement;
  const style = async () => {
    const now = await map.getStyle();
    const sky = without(now.sky, PREVIEWS.sky);
    const light = without(now.light, PREVIEWS.light);
    const layers = now.layers.map((l) => ({
      ...l,
      ...(l.paint && { paint: without(l.paint, PREVIEWS.paint) }),
      ...(l.layout && { layout: without(l.layout, PREVIEWS.layout) }),
    }));
    return { ...now, ...(sky && { sky }), ...(light && { light }), layers };
  };
  // The app's map, as a map the others follow and lead.
  const own = {
    camera: () => map.camera.current,
    jumpTo: (camera) => map.camera.jumpTo(camera),
    stop: () => map.camera.stop(),
    loaded: () => document.getElementById("app").dataset.map === "idle",
    style: () => map.getStyle(),
  };
  const ownSide = pair.indexOf("orbis");
  let closed = false;
  const sides = pair.map((key, index) => {
    const engine = ENGINES[key];
    const side = {
      key,
      index,
      engine,
      map: index === ownSide ? own : undefined,
      note: undefined,
    };
    const select = h(
      "select",
      {
        class: "compare-engine-picker",
        "aria-label": `The ${SIDES[index]} map's engine`,
        onchange: () => onPick(index, select.value),
      },
      Object.entries(ENGINES).map(([k, e]) => {
        const reason =
          e.unsupported?.(scene) ??
          (e.once && pair[1 - index] === k ? e.once : undefined);
        return h(
          "option",
          {
            value: k,
            selected: k === key,
            disabled: Boolean(reason) && k !== key,
            title: reason,
          },
          reason ? `${e.name} (can't)` : e.name,
        );
      }),
    );
    side.select = select;
    side.label = h("label", { class: `compare-label ${SIDES[index]}` }, select);
    if (side.map) return side;
    // Every side but the app map's gets a box of its own inside a layer
    // clipped to its side: GL JS's stylesheet makes its container
    // `position: relative`, which would undo the layer's absolute inset
    // and leave the map no height.
    side.inner = h("div", { class: "compare-engine" });
    side.layer = h("div", { class: `compare-map ${SIDES[index]}` }, side.inner);
    const say = (text) => {
      side.note = text;
      side.inner.replaceChildren(h("p", { class: "compare-note" }, text));
    };
    const reason = engine.unsupported?.(scene);
    if (reason) say(reason);
    else if (engine.isolated && !crossOriginIsolated) {
      if (!isolating) say(`${engine.name} needs a cross-origin isolated page`);
      else {
        say(`Reloading so ${engine.name} can share memory between threads…`);
        isolating.catch((e) =>
          say(
            `${engine.name} can't run here: ${e instanceof Error ? e.message : e}`,
          ),
        );
      }
    } else {
      side.inner.append(
        h("p", { class: "compare-note" }, `Loading ${engine.name}…`),
      );
      side.opening = (async () => {
        const module = await engine.load();
        if (closed) return;
        side.inner.replaceChildren();
        const opened = await module.open(side.inner, {
          style: await style(),
          camera: map.camera.current,
          constraints: scene.constraints,
        });
        if (closed) {
          opened.remove();
          return;
        }
        side.map = opened;
        const option = select.querySelector(`option[value="${key}"]`);
        option.textContent = module.label;
        // It opened at the camera of when it started; it catches up with the leader's.
        opened.jumpTo(leader.camera());
        offs.push(
          opened.onMove(() => {
            if (leader === opened) follow(opened);
          }),
        );
      })().catch((e) => {
        if (closed) return;
        console.error(e);
        say(
          `${engine.name} couldn't start: ${e instanceof Error ? e.message : e}`,
        );
      });
    }
    return side;
  });
  const divider = h("div", {
    class: "compare-divider",
    role: "slider",
    tabindex: "0",
    "aria-label": "Divider between the maps",
    "aria-valuemin": "0",
    "aria-valuemax": "100",
  });
  const added = [
    ...sides.map((s) => s.layer).filter(Boolean),
    divider,
    ...sides.map((s) => s.label),
  ];
  map.container.after(...added);

  // Input on a side makes its map lead and stops the others' motion, such
  // as a drag's momentum; input off the maps (the zoom buttons, say) makes
  // the app's lead. The capture phase sees it before any map does.
  let leader = own;
  const maps = () => [
    own,
    ...sides.map((s) => s.map).filter((m) => m && m !== own),
  ];
  const follow = (from) => {
    const camera = from.camera();
    for (const m of maps()) if (m !== from) m.jumpTo(camera);
  };
  const listening = new AbortController();
  const { signal } = listening;
  for (const type of ["pointerdown", "wheel", "keydown"]) {
    stage.addEventListener(
      type,
      (e) => {
        if (
          e.target === divider ||
          sides.some((s) => s.label.contains(e.target))
        )
          return;
        const layer = sides.find((s) => s.layer?.contains(e.target));
        // A press keeps its pointer on the map it began on, so a release over
        // the chrome still ends a GL JS drag, which listens on its container.
        if (
          layer &&
          type === "pointerdown" &&
          !e.target.hasPointerCapture(e.pointerId)
        ) {
          e.target.setPointerCapture(e.pointerId);
        }
        const next = layer?.map ?? own;
        if (next === leader) return;
        leader = next;
        for (const m of maps()) if (m !== leader) m.stop();
      },
      { capture: true, passive: true, signal },
    );
  }
  const offs = [
    map.on("move", () => {
      if (leader === own) follow(own);
    }),
  ];
  // Style edits, at most once a frame.
  let restyle = 0;
  offs.push(
    map.on("styledata", () => {
      restyle ||= requestAnimationFrame(async () => {
        restyle = 0;
        const next = await style();
        for (const m of maps()) if (m !== own) m.setStyle(next);
      });
    }),
  );

  // The divider: a drag or the arrow keys move it.
  const place = (fraction) => {
    split = Math.min(1, Math.max(0, fraction));
    stage.style.setProperty("--split", `${split * 100}%`);
    divider.setAttribute("aria-valuenow", String(Math.round(split * 100)));
  };
  place(split);
  divider.addEventListener(
    "pointerdown",
    (e) => divider.setPointerCapture(e.pointerId),
    { signal },
  );
  divider.addEventListener(
    "pointermove",
    (e) => {
      if (!divider.hasPointerCapture(e.pointerId)) return;
      const rect = stage.getBoundingClientRect();
      place((e.clientX - rect.left) / rect.width);
    },
    { signal },
  );
  divider.addEventListener(
    "keydown",
    (e) => {
      const step = e.shiftKey ? 0.1 : 0.02;
      const next = {
        ArrowLeft: split - step,
        ArrowRight: split + step,
        Home: 0,
        End: 1,
      }[e.key];
      if (next === undefined) return;
      e.preventDefault();
      place(next);
    },
    { signal },
  );

  return {
    sides: () =>
      sides.map((s) => ({
        engine: s.key,
        note: s.note,
        own: s.map === own,
        leads: Boolean(s.map) && s.map === leader,
        camera: s.map?.camera(),
        loaded: s.map?.loaded() ?? false,
      })),
    styles: () => Promise.all(sides.map((s) => s.map?.style?.() ?? null)),
    close() {
      closed = true;
      listening.abort();
      for (const off of offs) off();
      cancelAnimationFrame(restyle);
      for (const s of sides) if (s.map !== own) s.map?.remove();
      for (const element of added) element.remove();
    },
  };
}
