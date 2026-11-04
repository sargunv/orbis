// Drives the demo's edit API (web/demo/edit.js) in headless Chromium, in
// both thread modes: each edit class must resolve and settle. In main-thread
// mode it also times a feature-state write on the scene thread: the edit
// call, plus a frame's extra time over a frame without one, as totals over
// many writes (plan §10.2's ≤0.05 ms target).
//
//   pnpm edits [--writes=500]
import { launchChromium, serveDemo } from "./serve.mjs";

const writes = Number(process.argv.find((a) => a.startsWith("--writes="))?.slice(9) ?? 500);

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
  ["feature state", [["setFeatureState", { source: "grid", id: 7 }, { hover: true }]]],
  ["global state", [["setGlobalStateProperty", "unused", 1]]],
];

const server = await serveDemo();
const browser = await launchChromium();
try {
  for (const thread of ["worker", "main"]) {
    const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });
    let errors = 0;
    page.on("console", (m) => {
      if (["error", "warning"].includes(m.type())) { errors++; console.error(`page: ${m.text()}`); }
    });
    page.on("pageerror", (e) => { errors++; console.error(`page: ${e.message}`); });
    await page.goto(`${server.url}?thread=${thread}#15/35.685/139.74`);
    const idle = () => page.waitForFunction(() =>
      document.getElementById("hud").textContent.endsWith(" · idle"), null, { timeout: 60_000 });
    await idle();
    for (const [name, commands] of EDITS) {
      const started = Date.now();
      const generation = await page.evaluate(
        (commands) => globalThis.orbis.edit((tx) => {
          for (const [call, ...args] of commands) tx[call](...args);
        }), commands);
      await page.waitForTimeout(50);
      await idle();
      console.log(`${thread}: ${name} → generation ${generation}, idle in ${Date.now() - started} ms`);
    }
    if (thread === "main") {
      const cost = await page.evaluate((writes) => {
        const map = globalThis.orbis.map;
        const write = (i) => JSON.stringify([["setFeatureState", { source: "grid", id: 123 }, { hover: i % 2 === 0 }]]);
        for (let i = 0; i < 20; i++) { map.edit(write(i)); map.render().free(); }
        let t = performance.now();
        for (let i = 0; i < writes; i++) map.render().free();
        const frames = performance.now() - t;
        t = performance.now();
        for (let i = 0; i < writes; i++) { map.edit(write(i)); map.render().free(); }
        const written = performance.now() - t;
        t = performance.now();
        for (let i = 0; i < writes; i++) map.edit(write(i));
        const edits = performance.now() - t;
        return { frame: frames / writes, write: (written - frames) / writes, edit: edits / writes };
      }, writes);
      console.log(`main: feature-state write ${cost.write.toFixed(4)} ms ` +
        `(edit call ${cost.edit.toFixed(4)} ms; frame ${cost.frame.toFixed(3)} ms; ${writes} writes)`);
    }
    if (errors) throw new Error(`${thread}: ${errors} page warnings/errors`);
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}
