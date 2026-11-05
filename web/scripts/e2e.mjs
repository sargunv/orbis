// End-to-end checks in headless Chromium:
//
// - `scenario`: the binding scenario (plan §14.1) through the orbis package
//   on web/dist/scenario.html, on WebGPU and WebGL2: an inline style,
//   pixels, a hit test, a rendered-features query, an edit's generation,
//   style images, camera moves with roll and elevation, project and
//   unproject, a drag, a click, idle drawing nothing, and an error.
// - `edits`: every edit class through the map app's edit API
//   (`globalThis.orbis`, web/demo/expose.js) must resolve and settle, and
//   the cost of a feature-state write (plan §10.2's ≤0.05 ms target) over
//   `--writes` writes.
// - `editor`: the map app's style editor, queries and hover state on the
//   editor sample (`?style=playground&edit`), and frames per edit.
// - `offline` (only when asked for: it needs the network): downloads the
//   live map's view at `--at` from the Offline areas dialog, then reloads
//   with every request but the demo server's failing and checks the map
//   draws without errors. `cargo test -p orb-api --test regions --
//   --ignored` does the same natively.
//
//   pnpm e2e [--only=scenario,edits,editor,offline] [--configs=webgpu,webgl2]
//     [--writes=500] [--at=#13/35.6830/139.7520]
import assert from "node:assert/strict";
import { decodePng } from "./png.mjs";
import { launchChromium, serveDemo } from "./serve.mjs";

const arg = (name, fallback) => process.argv.slice(2).find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const only = arg("only", "scenario,edits,editor").split(",");
const configs = arg("configs", "webgpu,webgl2").split(",");
const writes = Number(arg("writes", 500));
const at = arg("at", "#13/35.6830/139.7520");
const SIZE = 200;
const CENTER = [SIZE / 2, SIZE / 2];

// A grid of points with generated ids, drawn through a state lookup.
const features = [];
for (let i = 0; i < 1600; i++) {
  features.push({ type: "Feature", properties: {}, geometry: { type: "Point",
    coordinates: [139.735 + (i % 40) * 0.0004, 35.682 + Math.floor(i / 40) * 0.0003] } });
}
const GRID = [
  ["addSource", "grid", { type: "geojson", generateId: true, data: { type: "FeatureCollection", features } }],
  ["addLayer", { id: "grid", type: "circle", source: "grid", paint: { "circle-radius": 4,
    "circle-color": ["case", ["boolean", ["feature-state", "hover"], false], "#f00", "#00f"] } }],
];
const EDITS = [
  ["paint", [["setPaintProperty", "water", "fill-color", "#f00"]]],
  ["binding flip", [["setPaintProperty", "road_minor", "line-color",
    ["match", ["get", "class"], "minor", "#f00", "#00f"]]]],
  ["filter", [["setFilter", "road_minor", ["==", ["get", "class"], "service"]]]],
  ["layers", [["moveLayer", "water", "road_minor"], ["removeLayer", "park"]]],
  ["source", GRID],
  ["source data", [["setSourceData", "grid", { type: "FeatureCollection", features: features.slice(0, 800) }]]],
  ["feature state", [["setFeatureState", { source: "grid", id: 7 }, { hover: true }]]],
  ["global state", [["setGlobalStateProperty", "unused", 1]]],
];

/** A page that collects its errors and warnings. */
async function open(browser, viewport) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (["error", "warning"].includes(m.type())) errors.push(m.text()); });
  return { page, errors };
}

async function scenario(browser, server, config) {
  const { page, errors } = await open(browser, { width: 400, height: 300 });
  await page.goto(`${server.url}scenario.html?renderer=${config}`);
  const scenario = (fn, arg) => page.evaluate(([fn, arg]) => window.scenario.then((s) => new Function("s", "arg", `return (${fn})(s, arg)`)(s, arg)), [fn.toString(), arg]);
  const pixels = async () => decodePng(await page.locator("#map").screenshot());
  const pixel = (image, [x, y]) => {
    const i = (Math.round(y) * image.width + Math.round(x)) * image.channels;
    return [...image.pixels.subarray(i, i + 3)];
  };

  // The style loads, draws and idles: styleload, load, then idle.
  const first = await scenario((s) => ({ events: s.events().map((e) => e.type) }));
  assert.equal(first.events[0], "styleload", first.events.join());
  assert.ok(first.events.includes("load"), first.events.join());
  assert.equal(first.events.at(-1), "idle", first.events.join());
  let image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [0, 0, 255], "the box");
  assert.deepEqual(pixel(image, [2, 2]), [255, 0, 0], "the background");

  // Picking and queries.
  const picked = await scenario((s, [c, corner]) => ({
    hit: s.map.hitTest(c), none: s.map.hitTest(corner),
  }), [CENTER, [2, 2]]);
  assert.deepEqual([picked.hit?.layer, picked.hit?.id, picked.hit?.source], ["box", "7", "box"]);
  assert.equal(picked.none, null);
  const features = await scenario((s, c) => s.map.queryRenderedFeatures(c), CENTER);
  assert.equal(features.length, 1);
  assert.deepEqual([features[0].id, features[0].layer.id, features[0].geometry.type], [7, "box", "Polygon"]);

  // Projection, synchronous in both modes.
  const projected = await scenario((s, c) => ({ p: s.map.project([0, 0]), g: s.map.unproject(c) }), CENTER);
  assert.ok(Math.hypot(projected.p[0] - CENTER[0], projected.p[1] - CENTER[1]) < 1e-6, String(projected.p));
  assert.ok(Math.hypot(...projected.g) < 1e-6, String(projected.g));

  // An edit resolves to the generation whose frames show it.
  const generation = await scenario(async (s) => {
    const generation = await s.map.edit((tx) => tx.setPaintProperty("box", "fill-color", "#00ff00"));
    await s.shows(generation);
    await s.idle();
    return generation;
  });
  assert.ok(generation > 0);
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [0, 255, 0], "the edited box");

  // A pattern the style lacks is reported; added, it draws, and new
  // pixels replace it in place.
  const solid = (rgb) => ({ width: 8, height: 8, data: Array.from({ length: 64 }, () => [...rgb, 255]).flat() });
  const missing = await scenario(async (s) => {
    s.events();
    await s.shows(await s.map.edit((tx) => tx.setPaintProperty("box", "fill-pattern", "dot")));
    await s.idle();
    return { events: s.events(), has: await s.map.hasImage("dot") };
  });
  assert.ok(missing.events.some((e) => e.type === "styleimagemissing" && e.id === "dot"), JSON.stringify(missing.events));
  assert.equal(missing.has, false);
  const added = await scenario(async (s, image) => {
    const data = (rgb) => ({ ...image, data: new Uint8Array(image.data.map((v, i) => (i % 4 === 3 ? 255 : rgb[i % 4]))) });
    await s.map.addImage("dot", data([255, 255, 0]), { pixelRatio: 1 });
    const drawn = s.drawn();
    const again = (() => { try { s.map.addImage("dot", data([0, 0, 0])); return null; } catch (e) { return String(e); } })();
    await drawn;
    return { has: await s.map.hasImage("dot"), again };
  }, solid([0, 0, 0]));
  assert.ok(added.has && added.again, JSON.stringify(added));
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [255, 255, 0], "the added pattern");
  await scenario(async (s, image) => {
    const drawn = s.drawn();
    await s.map.updateImage("dot", { ...image, data: new Uint8Array(image.data) });
    await drawn;
  }, solid([255, 0, 255]));
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [255, 0, 255], "the updated pattern");
  const removed = await scenario(async (s) => {
    await s.map.removeImage("dot");
    const again = (() => { try { s.map.removeImage("dot"); return null; } catch (e) { return String(e); } })();
    const generation = await s.map.edit((tx) => tx.setPaintProperty("box", "fill-pattern", null));
    await s.shows(generation);
    await s.idle();
    return { has: await s.map.hasImage("dot"), again };
  });
  assert.ok(!removed.has && removed.again, JSON.stringify(removed));

  // Camera moves carry a roll and an elevation; unclamped, the center
  // keeps the elevation it is given.
  const moved = await scenario(async (s) => {
    const c = s.map.camera;
    c.jumpTo({ roll: 30 });
    const rolled = c.current.roll;
    c.centerClampedToGround = false;
    c.setCenterElevation(120);
    const raised = c.current.elevation;
    const arrived = await c.easeTo({ roll: 0, elevation: 0, duration: 50 });
    const { roll, elevation } = c.current;
    c.centerClampedToGround = true;
    await s.idle();
    return { rolled, raised, arrived, roll, elevation };
  });
  assert.deepEqual(moved, { rolled: 30, raised: 120, arrived: true, roll: 0, elevation: 0 });

  // A drag moves the camera, with move events.
  await scenario((s) => s.events());
  const before = await scenario((s) => s.map.camera.current.center);
  await page.mouse.move(40, 100);
  await page.mouse.down();
  for (const x of [50, 70, 100, 140]) await page.mouse.move(x, 100);
  await page.mouse.up();
  const dragged = await scenario(async (s) => { await s.idle(); return { center: s.map.camera.current.center, events: s.events().map((e) => e.type) }; });
  assert.ok(Math.abs(dragged.center[0] - before[0]) > 1, `${before} → ${dragged.center}`);
  assert.ok(dragged.events.includes("movestart") && dragged.events.includes("moveend"), dragged.events.join());

  // A click is reported where it was, and an idle map draws nothing.
  await page.mouse.click(...CENTER);
  // Gesture events are reported with the next animation frame.
  const clicked = await scenario(async (s) => {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await s.idle();
    return s.events().filter((e) => e.type === "click");
  });
  assert.equal(clicked.length, 1);
  assert.deepEqual(clicked[0].point, CENTER);
  const quiet = await scenario((s) => s.quiet(1000));
  assert.deepEqual(quiet, { rafs: 0, renders: 0 }, "idle draws nothing");

  // Errors: an edit of a layer that isn't there fails, and the map keeps its style.
  const rejected = await scenario((s) => { try { s.map.edit((tx) => tx.setPaintProperty("nope", "fill-color", "#000")); return null; } catch (e) { return String(e); } });
  assert.ok(rejected, "a bad edit rejects");
  const hit = await scenario((s) => s.map.hitTest(s.map.project([0, 0]))?.layer);
  assert.equal(hit, "box");
  assert.deepEqual(errors, []);
  console.log(`scenario: ${config} passed`);
  await page.close();
}

async function edits(browser, server) {
  const { page, errors } = await open(browser, { width: 1024, height: 768 });
  await page.goto(`${server.url}?style=tokyo-mvt#15/35.685/139.74`);
  const idle = () => page.waitForFunction(() =>
    document.getElementById("app").dataset.map === "idle", null, { timeout: 60_000 });
  await idle();
  for (const [name, commands] of EDITS) {
    const started = Date.now();
    const generation = await page.evaluate(
      (commands) => globalThis.orbis.edit((tx) => {
        for (const [call, ...args] of commands) tx[call](...args);
      }), commands);
    await page.waitForTimeout(50);
    await idle();
    console.log(`edits: ${name} → generation ${generation}, idle in ${Date.now() - started} ms`);
  }
  const cost = await page.evaluate(async (writes) => {
    const { map } = globalThis.orbis;
    const write = (i) => map.setFeatureState({ source: "grid", id: 123 }, { hover: i % 2 === 0 });
    // Frames' render times over `writes` frames, each asked for by `poke`.
    const frames = async (poke) => {
      let total = 0;
      for (let i = 0; i < writes; i++) {
        const drawn = map.once("render");
        poke(i);
        total += (await drawn).ms;
      }
      return total / writes;
    };
    await frames(write);
    const still = await frames(() => map.camera.jumpTo({}));
    const written = await frames(write);
    let t = performance.now();
    for (let i = 0; i < writes; i++) write(i);
    const edits = performance.now() - t;
    return { still, written, edit: edits / writes };
  }, writes);
  console.log(`edits: feature-state write: edit call ${cost.edit.toFixed(4)} ms; frame with a write ` +
    `${cost.written.toFixed(3)} ms, without ${cost.still.toFixed(3)} ms (${writes} frames each)`);
  assert.deepEqual(errors, []);
  await page.close();
}

async function editor(browser, server) {
  const { page, errors } = await open(browser, { width: 1280, height: 800 });
  await page.goto(`${server.url}?style=playground&edit`);
  await page.waitForFunction(() => window.app?.editor.style && document.querySelector("#app").dataset.map === "idle",
    null,{timeout:60000});
  const map = page.locator("#map");
  const hit = await page.evaluate(async () => {
    const canvas = document.querySelector("#map canvas");
    const result = orbis.hitTest(canvas.clientWidth/2,canvas.clientHeight/2);
    if (result instanceof Promise) throw new Error("hitTest must be synchronous");
    return result;
  });
  assert.equal(hit.layer,"stations");
  assert.equal(hit.id,"31");
  assert.equal(hit.source,"district");
  assert.equal(hit.sourceLayer,null);
  const bounds = await map.boundingBox();
  const center = [bounds.x+bounds.width/2,bounds.y+bounds.height/2];
  const inspected = () => page.locator(".inspect").evaluate(e => e.hidden ? "" : e.textContent);
  await page.mouse.move(...center);
  await page.waitForFunction(() => document.querySelector(".inspect").textContent === "stations #31");
  await page.waitForFunction(() => document.querySelector("#app").dataset.map === "idle");
  const before = await map.screenshot();
  await page.mouse.move(bounds.x+10,bounds.y+bounds.height-10);
  await page.waitForFunction(() => document.querySelector(".inspect").hidden);
  await page.waitForFunction(() => document.querySelector("#app").dataset.map === "idle");
  assert.notDeepEqual(await map.screenshot(),before,"hover changes pixels");
  assert.equal(await inspected(),"");

  // A paint control: the background's colour, one transaction.
  const applied = () => page.locator(".applied").evaluate(e => Number(e.dataset.generation ?? 0));
  let generation = await applied();
  await page.locator(".layer-name", {hasText:"paper"}).click();
  await page.locator(".layer.open .prop-control input.field").first().fill("#ccddee");
  await page.locator(".layer.open .prop-control input.field").first().press("Enter");
  await page.waitForFunction(g => Number(document.querySelector(".applied").dataset.generation ?? 0) > g, generation);
  assert.equal((await page.evaluate(() => app.editor.style.layers[0].paint["background-color"])),"#ccddee");
  // Visibility.
  generation = await applied();
  await page.locator('.layer[data-layer="route"] .eye').click();
  await page.waitForFunction(g => Number(document.querySelector(".applied").dataset.generation ?? 0) > g, generation);
  assert.equal((await page.evaluate(() => app.editor.style.layers.find(l => l.id === "route").layout.visibility)),"none");

  // The JSON tab: invalid JSON keeps the map; a valid edit applies.
  await page.locator("#tab-json").click();
  const json = page.locator("#editor-json textarea");
  const original = await json.inputValue();
  await json.fill("{broken");
  await page.waitForFunction(() => document.querySelector("#json-error").textContent.length > 0);
  const edited = JSON.parse(original);
  edited.layers[0].paint["background-color"] = "#aabbcc";
  generation = await applied();
  await json.fill(JSON.stringify(edited,null,2));
  await page.waitForFunction(g => Number(document.querySelector(".applied").dataset.generation ?? 0) > g, generation);
  assert.equal(await page.locator("#json-error").textContent(),"");
  console.log(`editor: ${await page.locator(".applied").textContent()}; topmost station 31, hover pixel change, controls and validation recovery passed`);

  // Measure frame acknowledgements, rather than elapsed wall time.
  const measurements = await page.evaluate(async () => {
    const apply = async (name,commands) => {
      const frames = [];
      const before = app.map.stats;
      const listener = e => frames.push(e.detail);
      window.addEventListener("orbis-frame",listener);
      const generation = await orbis.edit(tx => { for (const [method,...args] of commands) tx[method](...args); });
      await new Promise(resolve => {
        if (frames.some(f => f.generation >= generation)) return resolve();
        const done = e => { if (e.detail.generation >= generation) { window.removeEventListener("orbis-frame",done); resolve(); } };
        window.addEventListener("orbis-frame",done);
      });
      window.removeEventListener("orbis-frame",listener);
      const visible = frames.find(f => f.generation >= generation);
      return {name,frames:frames.findIndex(f => f.generation >= generation)+1,generation,
        meshUploads:visible.meshesUploaded-before.meshesUploaded,
        rowUploads:visible.rowsUploaded-before.rowsUploaded};
    };
    return [
      await apply("paint",[["setPaintProperty","route","line-color","#345678"]]),
      await apply("binding flip",[["setPaintProperty","route","line-color",["match",["get","kind"],"route","#aabbcc","#ffffff"]]]),
      await apply("layout",[["setLayoutProperty","route","line-cap","round"]]),
    ];
  });
  console.log(`editor: ${JSON.stringify(measurements)}`);
  assert.equal(measurements[0].frames,1);
  assert.equal(measurements[1].meshUploads,0);

  // Replacing the style drops the hovered feature.
  await page.mouse.move(...center);
  await page.waitForFunction(() => document.querySelector(".inspect").textContent === "stations #31");
  generation = await applied();
  await json.fill(JSON.stringify({version:8,sources:{},layers:[edited.layers[0]]}));
  await page.waitForFunction(g => Number(document.querySelector(".applied").dataset.generation ?? 0) > g, generation);
  await page.waitForFunction(() => app.map.stats.draws < 5 && app.map.stats.idle);
  await page.mouse.move(center[0]+1,center[1]);
  await page.waitForFunction(() => document.querySelector(".inspect").hidden);
  assert.deepEqual(errors,[]);
  await page.close();
}

async function offline(browser, server) {
  const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  const errors = [];
  let cut = false;
  page.on("console", (m) => {
    if (m.type() !== "error" || !cut) return;
    // The browser logs each request the cut network refuses.
    if (!m.text().includes("ERR_INTERNET_DISCONNECTED")) errors.push(m.text());
  });
  const idle = () => page.waitForFunction(() => ["idle", "error"].includes(document.getElementById("app")?.dataset.map),
    null, { timeout: 60_000 });
  await page.goto(`${server.url}?style=liberty${at}`);
  await idle();
  await page.click("#menu-button");
  await page.click("#offline-button");
  const started = Date.now();
  await page.click("#offline .primary");
  await page.waitForFunction(() => /Downloaded|failed/.test(document.querySelector(".offline-status")?.textContent ?? ""),
    null, { timeout: 180_000 });
  const status = await page.textContent(".offline-status");
  console.log(`offline: ${status} (${Date.now() - started} ms)`);
  assert.ok(status.startsWith("Downloaded"), status);

  cut = true;
  await page.route((u) => !u.href.startsWith(server.url), (route) => route.abort("internetdisconnected"));
  const reopened = Date.now();
  await page.reload();
  await idle();
  const state = await page.evaluate(() => document.getElementById("app").dataset.map);
  console.log(`offline: reopened ${state} in ${Date.now() - reopened} ms`);
  assert.equal(state, "idle");
  assert.deepEqual(errors, []);
  await page.close();
}

const CHECKS = {
  scenario: async (browser, server) => { for (const config of configs) await scenario(browser, server, config); },
  edits,
  editor,
  offline,
};
const server = await serveDemo();
const browser = await launchChromium();
let failed = 0;
try {
  for (const name of only) {
    const check = CHECKS[name];
    if (!check) throw new Error(`unknown check ${name}`);
    try {
      await check(browser, server);
    } catch (e) {
      failed++;
      console.error(`${name} failed: ${e.message}`);
    }
  }
} finally {
  await browser.close();
  server.close();
}
if (failed) process.exit(1);
