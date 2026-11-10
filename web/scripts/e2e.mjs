// End-to-end checks in headless Chromium:
//
// - `scenario`: the binding scenario through the orbis package
//   on web/dist/scenario.html: an inline style,
//   pixels, a hit test, a rendered-features query, an edit's generation,
//   style images, camera moves with roll and elevation, project and
//   unproject (for the camera now and another), padding, the free camera, a
//   clustered source's queries, a GeoJSON diff, a canvas source, a request
//   transform, a source the app feeds, a drag, a click, idle drawing nothing, and an error.
// - `edits`: every edit class through the map app's edit API
//   (`globalThis.orbis`, web/demo/expose.js) must resolve and settle, and
//   the cost of a feature-state write (≤0.05 ms target) over
//   `--writes` writes.
// - `playground`: the map app's map of web/demo/data/playground.json
//   through its API: the topmost feature by hit test and query, the
//   style's hover state through feature state, paint and layout edits, a
//   bad edit applying none of its calls, frames per edit, and a style
//   replaced whole.
// - `switching`: entries switch on one map: another style on the same
//   source eases its paint by the style's transition and requests no
//   tiles, and back goes back without a reload; picking any entry keeps
//   the camera, its style's own camera never applying.
// - `light`: the Alps terrain style's light choice in the map options,
//   offline: `&light=viewport` anchors its hillshade and light to the
//   viewport as the style loads, the choice edits them in place and keeps
//   the URL in step, and another entry hides it and drops `&light`.
// - `terrain`: the Alps terrain style's terrain switch in the map options,
//   offline: on by default with no `&terrain`, a click or the keyboard
//   edits the style's terrain away and back without a reload, keeping the
//   URL in step, `&terrain=0` applies as the style loads, and another
//   entry hides it and drops `&terrain`.
// - `failing`: the Alps terrain style and the Globe, offline, their tiles
//   and DEM tiles failing, switched to the Tokyo extract, which draws; then
//   the map's wasm trapped on purpose stops it with one `error` saying so,
//   and calls after fail with that message.
// - `options`: the Globe's map options, offline (the Tokyo extract's style
//   standing in for Protomaps Light): the URL's choices apply as it loads,
//   every option is listed in its group, clicks and keys edit the live
//   style without a reload and keep the URL, options that need another
//   (shadows 3D buildings, fog terrain) wait for it, a reload reproduces
//   the choices, opening the panel moves nothing, and at phone width it's
//   a bottom sheet that fits.
// - `resize`: the Alps terrain style, offline, with circles over it,
//   resized three times: each resize draws the map at the new size, not
//   black, with no WebGPU error in any submit (a bundle bound to a target
//   the resize destroyed failed its submit).
// - `compare`: the comparison on the Tokyo extract: the plain page fetches
//   no other engine and isn't isolated; each pair of engines (Orbis, GL JS
//   and MapLibre Native, Orbis and GL JS twice too) loads only its engines,
//   isolates the page only for Native, fills the stage on both sides, and
//   a drag on either side reaches that side's map, which leads the other;
//   Native draws one side at a time; and GL JS's map follows the map
//   options (labels off, the globe).
// - `gestures`: Compose's gesture set on the scenario page, from input
//   synthesized through the DevTools protocol (mouse and touch): mouse and
//   touch flings, measured against mln-ffi's pan decay (the release
//   velocity from the page's own event times, then the glide's distance,
//   half-life and length), a pinch's zoom glide, Shift-drag box zoom with
//   its box shown, Shift-double-click, double-tap, two-finger tap and
//   double-tap-drag zoom, and a turn near north settling on north.
// - `readback`: decoded images read back exactly where a canvas's
//   readback is perturbed, as fingerprinting browsers (Brave) perturb it: a
//   DEM tile and a sprite drawn on fixture.html come out the same with every
//   context's `getImageData` flipping texels' low bits, and differently once
//   `VideoFrame` is gone too, so the noise reaches what a canvas reads.
// - `boot`: which build a map picks, in the worker and on the page: WebGPU
//   in Chromium, WebGL 2 (core-gl.wasm) in Firefox, whose adapter is
//   blocklisted headless, and in Chromium without a GPU or without WebGPU;
//   `backend: "webgl2"` loads only core-gl.wasm in Chromium with WebGPU,
//   and `backend: "webgpu"` without it fails saying why. The empty style
//   draws transparent on either, with no errors or other warnings. Then the map app's
//   `?backend=webgl2` draws WebGL 2, its renderer row saying so, and Auto
//   starts it again on WebGPU at the same camera.
// - `split`: the Tokyo extract and the Globe, offline, reach their first
//   idle without calling into core-deferred.wasm (a call would fetch it
//   synchronously: web/deferred.txt defers something first frames run,
//   and `pnpm coverage --tour --write` refreshes it), then fetch it.
// - `webgl2`: in a Chromium without WebGPU (#140), the scenario in the
//   worker and on the page; the Tokyo extract at two views, within 1% of
//   pixels of WebGPU's; a style given terrain draws flat and reports it
//   once (`degraded`); and a lost GPU recovers on WebGL 2.
// - `offline` (only when asked for: it needs the network): downloads the
//   live map's view at `--at` from the Offline areas dialog, then reloads
//   with every request but the demo server's failing and checks the map
//   draws without errors. `cargo test -p orb-api --test regions --
//   --ignored` does the same natively.
//
//   pnpm e2e [--only=scenario,webgl2,edits,playground,switching,light,terrain,failing,options,resize,compare,gestures,readback,boot,split,offline]
//     [--writes=500] [--at=#13/35.6830/139.7520] [--pairs=orbis,native;gljs,gljs]
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium, firefox } from "playwright";

import { decodePng, encodePng } from "./png.mjs";
import { launchChromium, serveDemo } from "./serve.mjs";

const arg = (name, fallback) =>
  process.argv
    .slice(2)
    .find((a) => a.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;
const only = arg(
  "only",
  "scenario,webgl2,edits,playground,switching,light,terrain,failing,options,resize,compare,gestures,readback,boot,split",
).split(",");
const writes = Number(arg("writes", 500));
const at = arg("at", "#13/35.6830/139.7520");
const SIZE = 200;
const CENTER = [SIZE / 2, SIZE / 2];

// A grid of points with generated ids, drawn through a state lookup.
const features = [];
for (let i = 0; i < 1600; i++) {
  features.push({
    type: "Feature",
    properties: {},
    geometry: {
      type: "Point",
      coordinates: [
        139.735 + (i % 40) * 0.0004,
        35.682 + Math.floor(i / 40) * 0.0003,
      ],
    },
  });
}
const GRID = [
  [
    "addSource",
    "grid",
    {
      type: "geojson",
      generateId: true,
      data: { type: "FeatureCollection", features },
    },
  ],
  [
    "addLayer",
    {
      id: "grid",
      type: "circle",
      source: "grid",
      paint: {
        "circle-radius": 4,
        "circle-color": [
          "case",
          ["boolean", ["feature-state", "hover"], false],
          "#f00",
          "#00f",
        ],
      },
    },
  ],
];
const EDITS = [
  ["paint", [["setPaintProperty", "water", "fill-color", "#f00"]]],
  [
    "binding flip",
    [
      [
        "setPaintProperty",
        "road_minor",
        "line-color",
        ["match", ["get", "class"], "minor", "#f00", "#00f"],
      ],
    ],
  ],
  [
    "filter",
    [["setFilter", "road_minor", ["==", ["get", "class"], "service"]]],
  ],
  [
    "layers",
    [
      ["moveLayer", "water", "road_minor"],
      ["removeLayer", "park"],
    ],
  ],
  ["source", GRID],
  [
    "source data",
    [
      [
        "setSourceData",
        "grid",
        { type: "FeatureCollection", features: features.slice(0, 800) },
      ],
    ],
  ],
  [
    "feature state",
    [["setFeatureState", { source: "grid", id: 7 }, { hover: true }]],
  ],
  ["global state", [["setGlobalStateProperty", "unused", 1]]],
];

/** A page that collects its errors and warnings. */
async function open(browser, viewport) {
  const page = await browser.newPage({ viewport });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (["error", "warning"].includes(m.type())) errors.push(m.text());
  });
  return { page, errors };
}

/** The scenario with the map in its worker, then on the page's thread. */
async function scenario(browser, server) {
  await scenarioOn(browser, server, "worker");
  await scenarioOn(browser, server, "page");
}

async function scenarioOn(browser, server, thread, backend = "webgpu") {
  const { page, errors } = await open(browser, { width: 400, height: 300 });
  const query = new URLSearchParams({
    ...(thread === "page" && { worker: "0" }),
    ...(backend === "webgl2" && { backend }),
  });
  await page.goto(`${server.url}scenario.html?${query}`);
  assert.deepEqual(
    await page.evaluate(() =>
      window.scenario.then((s) => [s.map.inWorker, s.map.backend]),
    ),
    [thread === "worker", backend],
    thread,
  );
  const scenario = (fn, arg) =>
    page.evaluate(
      ([fn, arg]) =>
        window.scenario.then((s) =>
          new Function("s", "arg", `return (${fn})(s, arg)`)(s, arg),
        ),
      [fn.toString(), arg],
    );
  const pixels = async () => decodePng(await page.locator("#map").screenshot());
  const pixel = (image, [x, y]) => {
    const i = (Math.round(y) * image.width + Math.round(x)) * image.channels;
    return [...image.pixels.subarray(i, i + 3)];
  };

  // The style loads, draws and idles: styleload, load, then idle.
  const first = await scenario((s) => ({
    events: s.events().map((e) => e.type),
  }));
  assert.equal(first.events[0], "styleload", first.events.join());
  assert.ok(first.events.includes("load"), first.events.join());
  assert.equal(first.events.at(-1), "idle", first.events.join());
  let image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [0, 0, 255], "the box");
  assert.deepEqual(pixel(image, [2, 2]), [255, 0, 0], "the background");

  // Picking and queries.
  const picked = await scenario(
    async (s, [c, corner]) => ({
      hit: await s.map.hitTest(c),
      none: await s.map.hitTest(corner),
    }),
    [CENTER, [2, 2]],
  );
  assert.deepEqual(
    [picked.hit?.layer, picked.hit?.id, picked.hit?.source],
    ["box", "7", "box"],
  );
  assert.equal(picked.none, null);
  const features = await scenario(
    (s, c) => s.map.queryRenderedFeatures(c),
    CENTER,
  );
  assert.equal(features.length, 1);
  assert.deepEqual(
    [features[0].id, features[0].layer.id, features[0].geometry.type],
    [7, "box", "Polygon"],
  );

  // Before the first frame, the map knew its size.
  const unframed = await scenario((s) => s.firstProjection);
  assert.deepEqual(unframed, CENTER, "projected before a frame");

  // Projection.
  const projected = await scenario(
    async (s, c) => ({
      p: await s.map.project([0, 0]),
      g: await s.map.unproject(c),
    }),
    CENTER,
  );
  assert.ok(
    Math.hypot(projected.p[0] - CENTER[0], projected.p[1] - CENTER[1]) < 1e-6,
    String(projected.p),
  );
  assert.ok(Math.hypot(...projected.g) < 1e-6, String(projected.g));

  // Padding moves the center; a camera's projection answers before any
  // move; the eye round-trips.
  const camera = await scenario(async (s) => {
    s.map.camera.jumpTo({ padding: { top: 40 } });
    const padded = await s.map.project([0, 0]);
    const closer = await s.map.project([10, 0], {
      zoom: s.map.camera.current.zoom + 1,
    });
    const zoom = s.map.camera.current.zoom;
    s.map.camera.jumpToFree(s.map.camera.free);
    await s.sync();
    const after = s.map.camera.current;
    s.map.camera.jumpTo({ padding: 0 });
    return { padded, closer, zoom, after };
  });
  assert.ok(
    Math.abs(camera.padded[1] - CENTER[1] - 20) < 1e-6,
    String(camera.padded),
  );
  assert.ok(
    Math.abs(
      camera.closer[0] - CENTER[0] - (2 ** camera.zoom * 1024 * 10) / 360,
    ) < 1e-6,
    String(camera.closer),
  );
  assert.ok(
    Math.abs(camera.after.zoom - camera.zoom) < 1e-9 &&
      camera.after.padding.top === 40,
    JSON.stringify(camera.after),
  );

  // An edit resolves to the generation whose frames show it.
  const generation = await scenario(async (s) => {
    const generation = await s.map.edit((tx) =>
      tx.setPaintProperty("box", "fill-color", "#00ff00"),
    );
    await s.shows(generation);
    await s.idle();
    return generation;
  });
  assert.ok(generation > 0);
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [0, 255, 0], "the edited box");

  // A pattern the style lacks is reported; added, it draws, and new
  // pixels replace it in place.
  const solid = (rgb) => ({
    width: 8,
    height: 8,
    data: Array.from({ length: 64 }, () => [...rgb, 255]).flat(),
  });
  const missing = await scenario(async (s) => {
    s.events();
    await s.shows(
      await s.map.edit((tx) =>
        tx.setPaintProperty("box", "fill-pattern", "dot"),
      ),
    );
    await s.idle();
    return { events: s.events(), has: await s.map.hasImage("dot") };
  });
  assert.ok(
    missing.events.some(
      (e) => e.type === "styleimagemissing" && e.id === "dot",
    ),
    JSON.stringify(missing.events),
  );
  assert.equal(missing.has, false);
  const added = await scenario(
    async (s, image) => {
      const data = (rgb) => ({
        ...image,
        data: new Uint8Array(
          image.data.map((v, i) => (i % 4 === 3 ? 255 : rgb[i % 4])),
        ),
      });
      await s.map.addImage("dot", data([255, 255, 0]), { pixelRatio: 1 });
      const drawn = s.drawn();
      const again = await s.map.addImage("dot", data([0, 0, 0])).then(
        () => null,
        (e) => String(e),
      );
      await drawn;
      return { has: await s.map.hasImage("dot"), again };
    },
    solid([0, 0, 0]),
  );
  assert.ok(added.has && added.again, JSON.stringify(added));
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [255, 255, 0], "the added pattern");
  await scenario(
    async (s, image) => {
      const drawn = s.drawn();
      await s.map.updateImage("dot", {
        ...image,
        data: new Uint8Array(image.data),
      });
      await drawn;
    },
    solid([255, 0, 255]),
  );
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [255, 0, 255], "the updated pattern");
  const removed = await scenario(async (s) => {
    await s.map.removeImage("dot");
    const again = await s.map.removeImage("dot").then(
      () => null,
      (e) => String(e),
    );
    const generation = await s.map.edit((tx) =>
      tx.setPaintProperty("box", "fill-pattern", null),
    );
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
    await s.sync();
    const rolled = c.current.roll;
    c.centerClampedToGround = false;
    c.setCenterElevation(120);
    await s.sync();
    const raised = c.current.elevation;
    const arrived = await c.easeTo({ roll: 0, elevation: 0, duration: 50 });
    const { roll, elevation } = c.current;
    c.centerClampedToGround = true;
    await s.idle();
    return { rolled, raised, arrived, roll, elevation };
  });
  assert.deepEqual(moved, {
    rolled: 30,
    raised: 120,
    arrived: true,
    roll: 0,
    elevation: 0,
  });

  // A drag moves the camera, with move events.
  await scenario((s) => s.events());
  const before = await scenario((s) => s.map.camera.current.center);
  await page.mouse.move(40, 100);
  await page.mouse.down();
  for (const x of [50, 70, 100, 140]) await page.mouse.move(x, 100);
  await page.mouse.up();
  const dragged = await scenario(async (s) => {
    await s.idle();
    return {
      center: s.map.camera.current.center,
      events: s.events().map((e) => e.type),
    };
  });
  assert.ok(
    Math.abs(dragged.center[0] - before[0]) > 1,
    `${before} → ${dragged.center}`,
  );
  assert.ok(
    dragged.events.includes("movestart") && dragged.events.includes("moveend"),
    dragged.events.join(),
  );

  // A click is reported where it was, and an idle map draws nothing.
  await page.mouse.click(...CENTER);
  // Gesture events are reported with the frame that follows.
  const clicked = await scenario(async (s) => {
    const events = [];
    for (
      const until = performance.now() + 1000;
      !events.some((e) => e.type === "click") && performance.now() < until;
    ) {
      await new Promise((resolve) => setTimeout(resolve, 10));
      events.push(...s.events());
    }
    await s.idle();
    return [...events, ...s.events()].filter((e) => e.type === "click");
  });
  assert.equal(clicked.length, 1);
  assert.deepEqual(clicked[0].point, CENTER);
  const quiet = await scenario((s) => s.quiet(1000));
  assert.deepEqual(quiet, { rafs: 0, renders: 0 }, "idle draws nothing");

  // A right click is a context menu, not a click.
  await page.mouse.click(...CENTER, { button: "right" });
  const menu = await scenario(async (s) => {
    const events = [];
    for (
      const until = performance.now() + 1000;
      !events.some((e) => e.type === "contextmenu") &&
      performance.now() < until;
    ) {
      await new Promise((resolve) => setTimeout(resolve, 10));
      events.push(...s.events());
    }
    await s.idle();
    return [...events, ...s.events()].filter((e) => e.point);
  });
  assert.deepEqual(
    menu.map((e) => [e.type, e.point]),
    [["contextmenu", CENTER]],
  );

  // With zoom off, a wheel scrolls the page and the map stays.
  const unzoomed = await scenario(async (s) => {
    s.map.setGestures({ zoom: false });
    await s.sync();
    const zoom = s.map.camera.current.zoom;
    const wheel = new WheelEvent("wheel", {
      deltaY: -100,
      cancelable: true,
      bubbles: true,
    });
    s.map.canvas.dispatchEvent(wheel);
    await s.idle();
    const after = s.map.camera.current.zoom;
    s.map.setGestures({});
    await s.sync();
    return { prevented: wheel.defaultPrevented, moved: after !== zoom };
  });
  assert.deepEqual(unzoomed, { prevented: false, moved: false });

  // Errors: an edit of a layer that isn't there fails, and the map keeps its style.
  const rejected = await scenario((s) =>
    s.map
      .edit((tx) => tx.setPaintProperty("nope", "fill-color", "#000"))
      .then(
        () => null,
        (e) => String(e),
      ),
  );
  assert.ok(rejected, "a bad edit rejects");
  const hit = await scenario(
    async (s) => (await s.map.hitTest(await s.map.project([0, 0])))?.layer,
  );
  assert.equal(hit, "box");

  // A clustered source: its points' cluster, whose queries answer.
  const clustered = await scenario(async (s) => {
    const features = Array.from({ length: 12 }, (_, i) => ({
      type: "Feature",
      id: i,
      properties: { n: i },
      geometry: { type: "Point", coordinates: [i * 0.0001, 0] },
    }));
    await s.shows(
      s.map.edit((tx) =>
        tx
          .addSource("points", {
            type: "geojson",
            cluster: true,
            clusterProperties: { sum: ["+", ["get", "n"]] },
            data: { type: "FeatureCollection", features },
          })
          .addLayer({
            id: "points",
            type: "circle",
            source: "points",
            paint: { "circle-radius": 6 },
          }),
      ),
    );
    await s.idle();
    const [cluster] = await s.map.querySourceFeatures("points", {
      filter: ["has", "point_count"],
    });
    const id = cluster.properties.cluster_id;
    return {
      sum: cluster.properties.sum,
      leaves: (await s.map.getClusterLeaves("points", id, 5, 10)).length,
      zoom: await s.map.getClusterExpansionZoom("points", id),
      children: (await s.map.getClusterChildren("points", id)).length,
    };
  });
  assert.equal(clustered.sum, 66, JSON.stringify(clustered));
  assert.equal(clustered.leaves, 2, JSON.stringify(clustered));
  assert.ok(
    clustered.zoom >= 1 && clustered.children > 0,
    JSON.stringify(clustered),
  );

  // A canvas source shows the pixels handed in.
  await scenario(async (s) => {
    await s.shows(
      s.map.edit((tx) =>
        tx
          .addSource("drawing", {
            type: "canvas",
            canvas: "drawing",
            coordinates: [
              [-120, 70],
              [120, 70],
              [120, -70],
              [-120, -70],
            ],
          })
          .addLayer({
            id: "drawing",
            type: "raster",
            source: "drawing",
            paint: { "raster-fade-duration": 0 },
          }),
      ),
    );
    s.map.camera.jumpTo({
      center: [0, 0],
      zoom: 0,
      bearing: 0,
      pitch: 0,
      roll: 0,
    });
    s.map.setSourceFrame("drawing", {
      width: 2,
      height: 2,
      data: new Uint8Array(16).map((_, i) =>
        i % 4 === 1 || i % 4 === 3 ? 255 : 0,
      ),
    });
    await s.idle();
  });
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [0, 255, 0], "the canvas");

  // A diff changes a GeoJSON source's data in the job workers.
  const updated = await scenario(async (s, c) => {
    await s.shows(
      s.map.edit((tx) =>
        tx.updateSourceData("box", {
          update: [
            { id: 7, addOrUpdateProperties: [{ key: "name", value: "seven" }] },
          ],
        }),
      ),
    );
    await s.idle();
    return (await s.map.queryRenderedFeatures(c, { layers: ["box"] }))[0]
      ?.properties;
  }, CENTER);
  assert.deepEqual(updated, { name: "seven" });

  // The style reads back.
  const readback = await scenario(async (s) => ({
    order: await s.map.getLayersOrder(),
    sources: await s.map.getSourceIds(),
    box: await s.map.getLayer("box"),
    none: await s.map.getLayer("nope"),
  }));
  assert.deepEqual(readback.order.slice(0, 2), ["background", "box"]);
  assert.ok(
    readback.sources.includes("box") &&
      readback.box.type === "fill" &&
      readback.none === undefined,
    JSON.stringify(readback),
  );

  // A request transform rewrites what the job workers fetch.
  await scenario(async (s) => {
    const square = {
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [-5, -5],
            [5, -5],
            [5, 5],
            [-5, 5],
            [-5, -5],
          ],
        ],
      },
    };
    s.setRequestTransform((url) =>
      url.startsWith("app://")
        ? {
            url: `data:application/json,${encodeURIComponent(JSON.stringify(square))}`,
            headers: { "X-Orbis": "1" },
          }
        : undefined,
    );
    await s.shows(
      s.map.edit((tx) =>
        tx
          .addSource("remote", { type: "geojson", data: "app://square.json" })
          .addLayer({
            id: "remote",
            type: "fill",
            source: "remote",
            paint: { "fill-color": "#00ffff" },
          }),
      ),
    );
    await s.idle();
    s.setRequestTransform(undefined);
  });
  image = await pixels();
  assert.deepEqual(
    pixel(image, CENTER),
    [0, 255, 255],
    "the transformed source",
  );

  // A source the app feeds: the map asks for its tile, the app answers.
  await scenario(async (s) => {
    s.map.on("tilerequest", ({ source, z, x, y }) => {
      if (source !== "app") return;
      s.map.setHostTile(
        "app",
        { z, x, y },
        {
          shapes: {
            type: "Feature",
            properties: {},
            geometry: {
              type: "Polygon",
              coordinates: [
                [
                  [-20, -20],
                  [20, -20],
                  [20, 20],
                  [-20, 20],
                  [-20, -20],
                ],
              ],
            },
          },
        },
      );
    });
    await s.shows(
      s.map.edit((tx) =>
        tx
          .addSource("app", {
            type: "vector",
            tiles: ["orbis-host:"],
            maxzoom: 0,
          })
          .addLayer({
            id: "app",
            type: "fill",
            source: "app",
            "source-layer": "shapes",
            paint: { "fill-color": "#ff00ff" },
          }),
      ),
    );
    await s.idle();
  });
  image = await pixels();
  assert.deepEqual(pixel(image, CENTER), [255, 0, 255], "the app's tile");
  assert.deepEqual(errors, []);
  console.log(`scenario: passed (${thread})`);
  await page.close();
}

async function edits(browser, server) {
  const { page, errors } = await open(browser, { width: 1024, height: 768 });
  await page.goto(`${server.url}?style=tokyo-mvt#15/35.685/139.74`);
  const idle = () =>
    page.waitForFunction(
      () => document.getElementById("app").dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
  await idle();
  for (const [name, commands] of EDITS) {
    const started = Date.now();
    const generation = await page.evaluate(
      (commands) =>
        globalThis.orbis.edit((tx) => {
          for (const [call, ...args] of commands) tx[call](...args);
        }),
      commands,
    );
    await page.waitForTimeout(50);
    await idle();
    console.log(
      `edits: ${name} → generation ${generation}, idle in ${Date.now() - started} ms`,
    );
  }
  // A placement worker that traps is replaced: labels place again.
  const labels = () =>
    page.evaluate(
      async () =>
        (await globalThis.orbis.map.queryRenderedFeatures()).filter(
          (f) => f.layer?.type === "symbol",
        ).length,
    );
  const before = await labels();
  await page.evaluate(() => globalThis.orbis.trapPlacementWorker());
  await page.waitForTimeout(100);
  await page.evaluate(() => globalThis.orbis.map.camera.jumpTo({ zoom: 15.5 }));
  await page.waitForTimeout(50);
  await idle();
  const after = await labels();
  assert.ok(
    before > 0 && after > 0,
    `labels before the trap ${before}, after ${after}`,
  );
  console.log(
    `edits: placement worker trapped and replaced; ${before} → ${after} labels`,
  );

  // Job workers that trap while a new view loads are replaced, and the
  // new ones have the style with the app's edits: the loads lost go again,
  // and the view's tiles draw without errors, the grid's among them.
  await page.evaluate(() =>
    globalThis.orbis.map.camera.jumpTo({
      center: [139.741, 35.6835],
      zoom: 16.2,
    }),
  );
  await page.waitForFunction(
    () => document.getElementById("app").dataset.map !== "idle",
  );
  await page.waitForTimeout(20);
  await page.evaluate(() => globalThis.orbis.trapJobWorkers());
  await page.waitForTimeout(50);
  await idle();
  const drawn = await page.evaluate(async () => {
    const { map } = globalThis.orbis;
    return {
      all: (await map.queryRenderedFeatures()).length,
      grid: (await map.queryRenderedFeatures(undefined, { layers: ["grid"] }))
        .length,
    };
  });
  assert.ok(
    drawn.all > 0 && drawn.grid > 0,
    `features after the trap ${JSON.stringify(drawn)}`,
  );
  console.log(
    `edits: job workers trapped and replaced; ${drawn.all} features drawn, ${drawn.grid} of the grid`,
  );

  // A device loss keeps what the app set at run time:
  // feature state (the grid's feature 7 above), images, a
  // GeoJSON source's diffed data and a transformed request.
  const recovered = await page.evaluate(async () => {
    const { map, setRequestTransform } = globalThis.orbis;
    map.camera.jumpTo({
      center: [139.74, 35.688],
      zoom: 15,
      bearing: 0,
      pitch: 0,
    });
    map.addImage("e2e-dot", {
      width: 2,
      height: 2,
      data: new Uint8Array(16).fill(255),
    });
    // Squares north of the grid, one per source.
    const box = (x) => [
      [x, 35.691],
      [x + 0.002, 35.691],
      [x + 0.002, 35.693],
      [x, 35.693],
      [x, 35.691],
    ];
    const square = (x, properties = {}) => ({
      type: "Feature",
      id: 1,
      properties,
      geometry: { type: "Polygon", coordinates: [box(x)] },
    });
    const centre = (x) => [x + 0.001, 35.692];
    const [geoX, remoteX] = [139.734, 139.744];
    setRequestTransform((url) =>
      url.startsWith("app://")
        ? {
            url: `data:application/json,${encodeURIComponent(JSON.stringify(square(remoteX)))}`,
          }
        : undefined,
    );
    const fill = (id, color, extra = {}) => ({
      id,
      type: "fill",
      source: id,
      paint: { "fill-color": color },
      ...extra,
    });
    await map.edit((tx) =>
      tx
        .addSource("e2e-geo", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: [square(geoX, { name: "a" })],
          },
        })
        .addLayer(fill("e2e-geo", "#ff0000"))
        .addSource("e2e-remote", { type: "geojson", data: "app://square.json" })
        .addLayer(fill("e2e-remote", "#00ffff")),
    );
    await map.edit((tx) =>
      tx.updateSourceData("e2e-geo", {
        update: [
          { id: 1, addOrUpdateProperties: [{ key: "name", value: "b" }] },
        ],
      }),
    );
    await map.once("idle");
    const shown = async () => {
      const query = async (id, x) =>
        (
          await map.queryRenderedFeatures(await map.project(centre(x)), {
            layers: [id],
          })
        )[0]?.properties ?? null;
      return {
        geo: await query("e2e-geo", geoX),
        remote: await query("e2e-remote", remoteX),
      };
    };
    const before = await shown();
    const restored = map.once("restored");
    map.loseDevice();
    await restored;
    await map.once("idle");
    const at = await map.project([139.735 + 7 * 0.0004, 35.682]);
    const [feature] = await map.queryRenderedFeatures(at, { layers: ["grid"] });
    const after = await shown();
    map.edit((tx) => tx.removeLayer("e2e-geo").removeLayer("e2e-remote"));
    setRequestTransform(undefined);
    return {
      image: await map.hasImage("e2e-dot"),
      state: feature?.state,
      before,
      after,
    };
  });
  const sources = { geo: { name: "b" }, remote: {} };
  assert.deepEqual(recovered.before, sources, JSON.stringify(recovered));
  assert.deepEqual(
    recovered,
    { image: true, state: { hover: true }, before: sources, after: sources },
    JSON.stringify(recovered),
  );
  console.log(
    "edits: lost the GPU, recovered with the app's state and sources",
  );

  const cost = await page.evaluate(async (writes) => {
    const { map } = globalThis.orbis;
    const write = (i) =>
      map.setFeatureState({ source: "grid", id: 123 }, { hover: i % 2 === 0 });
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
  console.log(
    `edits: feature-state write: edit call ${cost.edit.toFixed(4)} ms; frame with a write ` +
      `${cost.written.toFixed(3)} ms, without ${cost.still.toFixed(3)} ms (${writes} frames each)`,
  );
  assert.deepEqual(errors, []);
  await page.close();
}

async function playground(browser, server) {
  const { page, errors } = await open(browser, { width: 1280, height: 800 });
  const style = encodeURIComponent(`${server.url}e2e/data/playground.json`);
  await page.goto(`${server.url}?style=${style}#15/35.6835/139.7655`);
  const idle = () =>
    page.waitForFunction(
      () => document.querySelector("#app").dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
  await idle();
  // The topmost feature at the centre: station 31, by hit test and by query.
  const centre = await page.evaluate(async () => {
    const { map } = app;
    const point = [map.canvas.clientWidth / 2, map.canvas.clientHeight / 2];
    const { layer, id, source, sourceLayer } = await map.hitTest(point);
    const [feature] = await map.queryRenderedFeatures(point);
    return {
      hit: { layer, id, source, sourceLayer },
      feature: feature && {
        layer: feature.layer?.id,
        id: String(feature.id),
        kind: feature.properties.kind,
      },
    };
  });
  assert.deepEqual(centre.hit, {
    layer: "stations",
    id: "31",
    source: "district",
    sourceLayer: null,
  });
  assert.deepEqual(centre.feature, {
    layer: "stations",
    id: "31",
    kind: "station",
  });

  // The style's own hover state, through feature state: the station's
  // pixels turn the style's hover colour, and back when the state goes.
  // A state write keeps the style's generation: the frame that shows it
  // is polled for.
  const station = () =>
    page.evaluate(async () => {
      const image = await app.map.readPixels();
      const i =
        (Math.floor(image.height / 2) * image.width +
          Math.floor(image.width / 2)) *
        4;
      return Array.from(image.data.slice(i, i + 3));
    });
  assert.deepEqual(await station(), [255, 255, 255], "the station draws white");
  await page.evaluate(() =>
    app.map.setFeatureState({ source: "district", id: 31 }, { hover: true }),
  );
  await settles(
    page,
    station,
    [0xe7, 0x5f, 0x35],
    "hover state turns the station orange",
  );
  await page.evaluate(() =>
    app.map.removeFeatureState({ source: "district", id: 31 }, "hover"),
  );
  await settles(
    page,
    station,
    [255, 255, 255],
    "removing the state draws the station as before",
  );

  // Paint and layout edits show in the style; a bad edit rejects and applies none.
  await page.evaluate(() =>
    app.map.edit((tx) => {
      tx.setPaintProperty("paper", "background-color", "#ccddee");
      tx.setLayoutProperty("route", "visibility", "none");
    }),
  );
  const read = () =>
    page.evaluate(async () => {
      const style = await app.map.getStyle();
      const layer = (id) => style.layers.find((l) => l.id === id);
      return {
        paper: layer("paper").paint["background-color"],
        route: layer("route").layout?.visibility,
      };
    });
  assert.deepEqual(await read(), { paper: "#ccddee", route: "none" });
  const rejected = await page.evaluate(() =>
    app.map
      .edit((tx) => {
        tx.setPaintProperty("paper", "background-color", "#aabbcc");
        tx.setPaintProperty("nope", "fill-color", "#000");
      })
      .then(
        () => false,
        () => true,
      ),
  );
  assert.ok(rejected, "an edit naming a missing layer rejects");
  assert.deepEqual(
    await read(),
    { paper: "#ccddee", route: "none" },
    "a rejected edit applies none of its calls",
  );

  // Measure frame acknowledgements, rather than elapsed wall time.
  const measurements = await page.evaluate(async () => {
    const apply = async (name, commands) => {
      const frames = [];
      const before = app.map.stats;
      const listener = (e) => frames.push(e.detail);
      window.addEventListener("orbis-frame", listener);
      const generation = await app.map.edit((tx) => {
        for (const [method, ...args] of commands) tx[method](...args);
      });
      await new Promise((resolve) => {
        if (frames.some((f) => f.generation >= generation)) return resolve();
        const done = (e) => {
          if (e.detail.generation >= generation) {
            window.removeEventListener("orbis-frame", done);
            resolve();
          }
        };
        window.addEventListener("orbis-frame", done);
      });
      window.removeEventListener("orbis-frame", listener);
      const visible = frames.find((f) => f.generation >= generation);
      return {
        name,
        frames: frames.findIndex((f) => f.generation >= generation) + 1,
        generation,
        meshUploads: visible.meshesUploaded - before.meshesUploaded,
        rowUploads: visible.rowsUploaded - before.rowsUploaded,
      };
    };
    return [
      await apply("paint", [
        ["setPaintProperty", "stations", "circle-color", "#345678"],
      ]),
      await apply("binding flip", [
        [
          "setPaintProperty",
          "stations",
          "circle-color",
          ["match", ["get", "kind"], "station", "#aabbcc", "#ffffff"],
        ],
      ]),
      await apply("layout", [
        ["setLayoutProperty", "route", "line-cap", "round"],
      ]),
    ];
  });
  console.log(`playground: ${JSON.stringify(measurements)}`);
  assert.equal(measurements[0].frames, 1);
  assert.equal(measurements[1].meshUploads, 0);

  // A style replaced whole drops its features: nothing to hit at the centre.
  await page.evaluate(() =>
    app.map.edit((tx) =>
      tx.setStyle({
        version: 8,
        sources: {},
        layers: [
          {
            id: "paper",
            type: "background",
            paint: { "background-color": "#aabbcc" },
          },
        ],
      }),
    ),
  );
  await settles(
    page,
    () =>
      page.evaluate(() =>
        app.map.hitTest([
          app.map.canvas.clientWidth / 2,
          app.map.canvas.clientHeight / 2,
        ]),
      ),
    null,
    "nothing to hit once the style is replaced",
  );
  assert.deepEqual(errors, []);
  await page.close();
  console.log(
    "playground: station 31 on top, hover state, edits, a rejected edit and a replaced style passed",
  );
}

/** The Tokyo extract's style with its URLs absolute under the e2e mount, so a copy elsewhere draws alike. */
async function tokyoStyle(server) {
  const text = await readFile(join(DATA, "tokyo-mvt.json"), "utf8");
  return JSON.parse(
    text
      .replaceAll('"./', `"${server.url}e2e/data/`)
      .replaceAll("pmtiles://./", `pmtiles://${server.url}e2e/data/`),
  );
}

async function switching(browser, server) {
  const { page, errors } = await open(browser, { width: 800, height: 600 });
  // Two basemaps on the one source: the second's water is red, and each
  // eases paint over a second and a half.
  const transition = { duration: 1500, delay: 0 };
  const plain = { ...(await tokyoStyle(server)), transition };
  const red = structuredClone(plain);
  red.layers.find((l) => l.id === "water").paint["fill-color"] = "#ff0000";
  await writeFile(join(server.styles, "plain.json"), JSON.stringify(plain));
  await writeFile(join(server.styles, "red.json"), JSON.stringify(red));
  const [first, second] = ["plain", "red"].map(
    (name) => `${server.url}e2e/styles/${name}.json`,
  );
  const idle = () =>
    page.waitForFunction(
      () => document.getElementById("app").dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
  // The extract's range requests, from the page and its workers.
  let archive = 0;
  page.on("request", (r) => {
    if (r.url().includes("tokyo-mvt.pmtiles")) archive++;
  });
  await page.goto(
    `${server.url}?style=${encodeURIComponent(first)}#15/35.685/139.74`,
  );
  await idle();
  assert.ok(archive > 0, "the first load reads the archive");

  // A point the water draws, and its colour through a switch to `style`.
  // Points on the map itself, not under its controls.
  const onMap = (test) =>
    page.evaluate(async (test) => {
      const canvas = app.map.canvas;
      const rect = canvas.getBoundingClientRect();
      for (let y = 20; y < canvas.clientHeight; y += 20) {
        for (let x = 20; x < canvas.clientWidth; x += 20) {
          if (document.elementFromPoint(rect.x + x, rect.y + y) !== canvas)
            continue;
          const hit = await app.map.hitTest([x, y]);
          if (hit && (test === "any" || hit.layer === test)) return [x, y];
        }
      }
      return null;
    }, test);
  const water = await onMap("water");
  assert.ok(water, "the view has water");
  const through = async (style) => {
    archive = 0;
    const seen = await page.evaluate(
      async ([style, [x, y]]) => {
        const { map } = app;
        const colour = async () => {
          const image = await map.readPixels();
          const ratio = image.width / map.canvas.clientWidth;
          const i =
            (Math.round(y * ratio) * image.width + Math.round(x * ratio)) * 4;
          return Array.from(image.data.slice(i, i + 3));
        };
        const before = await colour();
        const loaded = map.once("styleload");
        if (style) app.show(style);
        else history.back();
        const late = new Promise((_, reject) =>
          setTimeout(() => reject(new Error("no styleload")), 20_000),
        );
        await Promise.race([loaded, late]);
        // The transition takes 1.5 s: past it, the colour has settled.
        const colours = [];
        for (const until = performance.now() + 2500; performance.now() < until;)
          colours.push(await colour());
        return { before, colours, after: await colour() };
      },
      [style, water],
    );
    return { ...seen, requests: archive };
  };
  await page.evaluate(() => {
    window.firstMap = app.map;
    window.notReloaded = true;
  });
  const close = (a, b) => a.every((v, i) => Math.abs(v - b[i]) <= 6);
  const eases = ({ before, colours, after }) =>
    !close(before, after) &&
    colours.some((c) => !close(c, before) && !close(c, after));
  const there = await through(second);
  assert.ok(
    close(there.after, [255, 0, 0]),
    `the water turns red: ${JSON.stringify(there.after)}`,
  );
  assert.ok(eases(there), `the water eases: ${JSON.stringify(there)}`);
  assert.equal(there.requests, 0, "the shared source reads no tiles again");
  // Back, by the browser's history, on the same page and map.
  const back = await through(null);
  assert.ok(
    eases(back) && close(back.after, there.before),
    `back eases to the first: ${JSON.stringify(back)}`,
  );
  assert.equal(back.requests, 0);
  assert.ok(
    await page.evaluate(
      () =>
        window.notReloaded &&
        app.map === window.firstMap &&
        document.querySelectorAll("#map canvas").length === 1,
    ),
    "one page, one map",
  );
  assert.ok(page.url().includes(encodeURIComponent(first)), page.url());
  console.log(
    `switching: one map; the water eases over ${there.colours.length} frames read, ` +
      `back over ${back.colours.length}, requesting no tiles`,
  );
  assert.deepEqual(errors, []);
  await page.close();

  // Picking any entry keeps the camera, whatever its style's own camera:
  // every remote style here is the extract's, set at a view far away, and
  // the rest of the network fails. Arctic's plane ends short of Tokyo, so
  // its constraint clamps the camera, last.
  const far = {
    ...(await tokyoStyle(server)),
    center: [7.75, 46.02],
    zoom: 3,
    bearing: -32,
    pitch: 72,
  };
  const picking = await browser.newPage({
    viewport: { width: 800, height: 600 },
  });
  await picking.route(
    (u) => !u.href.startsWith(server.url),
    (route) => {
      if (/\/styles?\/|style\.json/.test(route.request().url()))
        route.fulfill({ json: far });
      else route.abort("internetdisconnected");
    },
  );
  await picking.goto(`${server.url}#15/35.687/139.74/20/30`);
  await picking.waitForFunction(
    () => document.getElementById("app").dataset.map === "idle",
    null,
    { timeout: 60_000 },
  );
  const camera = () =>
    picking.evaluate(() => {
      const { center, zoom, bearing, pitch } = app.map.camera.current;
      return { center, zoom, bearing, pitch };
    });
  const start = await camera();
  const keys = await picking.evaluate(() =>
    [...document.querySelectorAll("[data-scene]")].map((e) => e.dataset.scene),
  );
  assert.ok(keys.length > 10 && keys.includes("arctic"), keys.join());
  const pick = async (key) => {
    await picking.click("#scene-button");
    await picking.click(`[data-scene="${key}"]`);
    await picking.waitForFunction(
      () => document.getElementById("app").dataset.map !== "loading",
      null,
      { timeout: 20_000 },
    );
    // Past the style's load, where a style's camera or a flight would move it.
    await picking.waitForTimeout(500);
    return camera();
  };
  const same = (a, b) =>
    Math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]) < 1e-6 &&
    ["zoom", "bearing", "pitch"].every((k) => Math.abs(a[k] - b[k]) < 1e-6);
  for (const key of [...keys.filter((k) => k !== "arctic"), "liberty"]) {
    const now = await pick(key);
    assert.ok(
      same(now, start),
      `${key} keeps the camera: ${JSON.stringify(now)}, not ${JSON.stringify(start)}`,
    );
  }
  const arctic = await pick("arctic");
  assert.ok(
    arctic.zoom > 10 &&
      Math.abs(arctic.bearing - start.bearing) < 1e-6 &&
      arctic.pitch === start.pitch &&
      Math.hypot(
        arctic.center[0] - start.center[0],
        arctic.center[1] - start.center[1],
      ) < 20,
    `Arctic clamps the camera into its plane: ${JSON.stringify(arctic)}`,
  );
  await picking.close();
  console.log(
    `switching: ${keys.length} entries keep the camera; Arctic clamps it into its plane`,
  );
}

/** Opens the map options panel on `page`, if it's closed. */
async function openOptions(page) {
  if (await page.evaluate(() => app.options.open)) return;
  await page.click("#options-button");
  await page.waitForFunction(() => app.options.open);
}
const option = (page, key) => page.locator(`#options [data-option="${key}"]`);
/** The renderer row's picked choice and the backend it says the map draws with. */
const renderer = (page) =>
  page.evaluate(() => {
    const row = document.querySelector('#options [data-option="backend"]');
    return {
      picked: row.querySelector('[aria-checked="true"]').dataset.value,
      active: row.dataset.active,
    };
  });

/** Polls `read` until it gives `expected`: the map answers edits and queries in its own time. */
async function settles(page, read, expected, message) {
  let seen;
  for (
    const until = Date.now() + 10_000;
    Date.now() < until;
    await page.waitForTimeout(50)
  ) {
    seen = await read();
    if (JSON.stringify(seen) === JSON.stringify(expected)) return;
  }
  assert.deepEqual(seen, expected, message);
}

async function light(browser, server) {
  // Wide enough that the panel opens beside its button.
  const page = await browser.newPage({
    viewport: { width: 1200, height: 800 },
  });
  // The style is the demo's; its tiles would come from the network, which
  // fails here: the switch only edits the style.
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route(
    (u) => !u.href.startsWith(server.url),
    (route) => route.abort("internetdisconnected"),
  );
  await page.goto(`${server.url}?style=alps&light=viewport`);
  await page.waitForFunction(() => app.options.values, null, {
    timeout: 60_000,
  });
  // Every illumination anchor in the map's style.
  const anchors = () =>
    page.evaluate(async () => {
      const of = (style) =>
        style && [
          ...style.layers
            .filter((l) => l.type === "hillshade")
            .map((l) => l.paint["hillshade-illumination-anchor"]),
          style.light.anchor,
        ];
      return {
        map: of(await app.map.getStyle()),
        checked: document.querySelector(
          '#options [data-option="light"] [aria-checked="true"]',
        )?.dataset.value,
        light: new URLSearchParams(location.search).get("light"),
      };
    });
  const all = (anchor) => ({
    map: [anchor, anchor],
    checked: anchor,
    light: anchor === "map" ? null : anchor,
  });
  await settles(
    page,
    anchors,
    all("viewport"),
    "&light=viewport applies as the style loads",
  );
  await openOptions(page);
  await option(page, "light").locator('[data-value="map"]').click();
  await settles(
    page,
    anchors,
    all("map"),
    "the style's own anchor, map, leaves the URL without &light",
  );
  // The arrow keys move the choice, as in a radio group.
  await page.keyboard.press("ArrowRight");
  await settles(
    page,
    anchors,
    all("viewport"),
    "the arrow keys choose the viewport",
  );
  assert.equal(
    await page.evaluate(() => document.activeElement?.dataset.value),
    "viewport",
    "focus follows the choice",
  );
  // And back.
  await page.keyboard.press("ArrowLeft");
  await settles(page, anchors, all("map"), "the arrow keys choose the map");
  await page.evaluate(() => app.show("tokyo"));
  await page.waitForFunction(
    () =>
      app.scene === "tokyo" &&
      !document.querySelector('#options [data-option="light"]'),
  );
  assert.equal(
    new URL(page.url()).searchParams.has("light"),
    false,
    "another entry drops &light",
  );
  assert.deepEqual(errors, []);
  await page.close();
  console.log(
    "light: &light, the switch and another entry agree on the anchor",
  );
}

async function terrain(browser, server) {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 800 },
  });
  // As in `light`: the switch only edits the style.
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route(
    (u) => !u.href.startsWith(server.url),
    (route) => route.abort("internetdisconnected"),
  );
  await page.goto(`${server.url}?style=alps`);
  const toggle = option(page, "terrain");
  await page.waitForFunction(() => app.options.values, null, {
    timeout: 60_000,
  });
  // Whether the map's style has terrain, the switch, and the URL.
  const state = () =>
    page.evaluate(async () => ({
      map: Boolean((await app.map.getStyle()).terrain),
      checked: document
        .querySelector('#options [data-option="terrain"]')
        .getAttribute("aria-checked"),
      terrain: new URLSearchParams(location.search).get("terrain"),
    }));
  const on = { map: true, checked: "true", terrain: null };
  const off = { map: false, checked: "false", terrain: "0" };
  await settles(
    page,
    state,
    on,
    "the Alps draw their terrain by default, with no &terrain",
  );
  // A page that reloads loses this.
  await page.evaluate(() => {
    window.stayed = true;
  });
  await openOptions(page);
  await toggle.click();
  await settles(
    page,
    state,
    off,
    "the switch takes the terrain away, with &terrain=0",
  );
  await toggle.focus();
  await page.keyboard.press("Space");
  await settles(
    page,
    state,
    on,
    "the keyboard puts the style's terrain back, and the URL drops &terrain",
  );
  assert.ok(
    await page.evaluate(() => window.stayed === true),
    "the switch edits the style without a reload",
  );
  // &terrain=0 applies as the style loads.
  await page.goto(`${server.url}?style=alps&terrain=0`);
  await page.waitForFunction(() => app.options.values, null, {
    timeout: 60_000,
  });
  await settles(page, state, off, "&terrain=0 applies as the style loads");
  await openOptions(page);
  await toggle.click();
  await settles(
    page,
    state,
    on,
    "turning it on restores the style's own terrain",
  );
  await page.evaluate(() => app.show("tokyo"));
  await page.waitForFunction(
    () =>
      app.scene === "tokyo" &&
      !document.querySelector('#options [data-option="terrain"]'),
  );
  assert.equal(
    new URL(page.url()).searchParams.has("terrain"),
    false,
    "another entry drops &terrain",
  );
  assert.deepEqual(errors, []);
  await page.close();
  console.log(
    "terrain: the switch, &terrain and another entry agree, with no reload",
  );
}

async function failing(browser, server) {
  // Errors that say the map itself broke, not that a tile failed.
  const broke = (errors) =>
    errors.filter((e) =>
      /unreachable|recursive use|RuntimeError|stopped/.test(e),
    );
  const entries = [
    [
      "alps",
      (page) =>
        page.route(
          (u) => !u.href.startsWith(server.url),
          (route) => route.abort("internetdisconnected"),
        ),
    ],
    ["globe", (page) => offlineGlobe(page, server)],
  ];
  let page, errors;
  for (const [entry, offline] of entries) {
    ({ page, errors } = await open(browser, { width: 800, height: 600 }));
    await offline(page);
    await page.goto(`${server.url}?style=${entry}`);
    // Long enough for tiles and DEM tiles to fail between frames too, whose
    // sources a switch to a style with fewer took away (#160).
    await page.waitForFunction(() => app.map, null, { timeout: 60_000 });
    await page.waitForTimeout(3000);
    await page.evaluate(() => app.show("tokyo"));
    await page.waitForFunction(
      () =>
        app.scene === "tokyo" &&
        document.getElementById("app").dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
    assert.deepEqual(broke(errors), [], `${entry} → tokyo`);
    console.log(`failing: ${entry} with its tiles failing → tokyo draws`);
    if (entry !== "globe") await page.close();
  }

  // A snapshot of a style that failed to load fails at once, saying why,
  // rather than waiting out its minute.
  const refused = await page.evaluate(async () => {
    const started = performance.now();
    app.map.setStyle("/no-such-style.json");
    const message = await app.map.snapshot(64, 64, 1).then(
      () => "resolved",
      (e) => e.message,
    );
    return { message, ms: performance.now() - started };
  });
  assert.notEqual(refused.message, "resolved", JSON.stringify(refused));
  assert.doesNotMatch(refused.message, /settle/, JSON.stringify(refused));
  assert.ok(refused.ms < 10_000, JSON.stringify(refused));
  console.log(
    `failing: a snapshot of a missing style fails in ${refused.ms.toFixed(0)} ms: ${refused.message.split("\n")[0]}`,
  );

  // A trap stops the map cleanly: one `error` says so, and calls after
  // fail with its message rather than "recursive use of an object".
  const seen = await page.evaluate(async () => {
    const { map } = app;
    const events = [];
    map.on("error", ({ message }) => events.push(message));
    const settle = () => new Promise((resolve) => setTimeout(resolve, 300));
    map.trap();
    await settle();
    const call = await map.getStyle().then(
      () => "resolved",
      (e) => e.message,
    );
    map.camera.jumpTo({ zoom: 14 });
    await settle();
    return { events, call };
  });
  assert.equal(seen.events.length, 1, JSON.stringify(seen));
  assert.match(seen.events[0], /^the map stopped: its wasm trapped/);
  assert.equal(seen.call, seen.events[0]);
  assert.deepEqual(
    errors.filter((e) => /recursive use/.test(e)),
    [],
    "no call reaches the trapped module",
  );
  await page.close();
  console.log(
    `failing: a trap stops the map with one error: ${seen.events[0]}`,
  );
}

/** The Globe entry offline: Protomaps Light stood in for by the Tokyo extract's style, with a buildings layer to extrude. */
async function offlineGlobe(page, server) {
  const tokyo = await tokyoStyle(server);
  const style = {
    ...tokyo,
    layers: [
      ...tokyo.layers,
      {
        id: "buildings",
        type: "fill",
        source: "openmaptiles",
        "source-layer": "buildings",
        paint: { "fill-color": "#ccc" },
      },
    ],
  };
  await page.route(
    (u) => !u.href.startsWith(server.url),
    (route) => {
      if (route.request().url().includes("api.protomaps.com/styles/"))
        route.fulfill({ json: style });
      else route.abort("internetdisconnected");
    },
  );
}

// The Globe's options, in the panel's order, and their values as it designs them.
const GLOBE_OPTIONS = {
  terrain: false,
  hillshade: true,
  light: "viewport",
  night: true,
  now: false,
  buildings: true,
  shadows: true,
  ao: true,
  projection: "globe",
  atmosphere: true,
  fog: true,
  stars: true,
  labels: true,
  colors: "linear",
};

async function options(browser, server) {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 800 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await offlineGlobe(page, server);
  await page.goto(
    `${server.url}?style=globe&shadows=0&projection=mercator&colors=srgb#14/35.687/139.74`,
  );
  await page.waitForFunction(
    () => app.scene === "globe" && app.options.values,
    null,
    { timeout: 60_000 },
  );
  // The panel's values, what the map's style draws, the disabled rows and the URL's choices.
  const state = () =>
    page.evaluate(async () => {
      const style = await app.map.getStyle();
      const layer = (id) => style.layers.find((l) => l.id === id);
      const query = new URLSearchParams(location.search);
      return {
        values: app.options.values,
        style: {
          shadows: style.light["dev.sargunv.orbis:shadow-intensity"],
          now: typeof style.light["dev.sargunv.orbis:datetime"],
          anchor: style.light.anchor,
          projection: style.projection.type,
          terrain: Boolean(style.terrain),
          colors: style.metadata["dev.sargunv.orbis:color-space"],
          buildings: layer("buildings_3d").layout.visibility ?? "visible",
          ao: layer("buildings_3d").paint[
            "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity"
          ],
          labels: [
            ...new Set(
              style.layers
                .filter((l) => l.type === "symbol")
                .map((l) => l.layout?.visibility ?? "visible"),
            ),
          ],
        },
        disabled: [...document.querySelectorAll("#options [data-option]")]
          .filter((row) => row.disabled || row.querySelector("button:disabled"))
          .map((row) => row.dataset.option),
        url: Object.fromEntries(
          [...query].filter(([k]) => k !== "style").sort(),
        ),
      };
    });
  const expect = (values, style, disabled, url) => ({
    values: { ...GLOBE_OPTIONS, ...values },
    style: {
      shadows: 0.9,
      now: "undefined",
      anchor: "viewport",
      projection: "globe",
      terrain: false,
      colors: "linear",
      buildings: "visible",
      ao: 0.6,
      labels: ["visible"],
      ...style,
    },
    disabled,
    url: Object.fromEntries(Object.entries(url).sort()),
  });
  // The URL's choices apply as the style loads; fog waits for terrain.
  await settles(
    page,
    state,
    expect(
      { shadows: false, projection: "mercator", colors: "srgb" },
      { shadows: 0, projection: "mercator", colors: "srgb" },
      ["atmosphere", "fog", "stars"],
      { shadows: "0", projection: "mercator", colors: "srgb" },
    ),
    "the URL's choices apply as the Globe loads",
  );
  await page.evaluate(() => {
    window.stayed = true;
  });

  // Opening the panel moves nothing; it lists the Globe's options, grouped, clear of its button.
  const boxes = () =>
    page.evaluate(() =>
      ["scene-button", "options-button", "menu-button", "zoom-in", "map"].map(
        (id) => document.getElementById(id).getBoundingClientRect().toJSON(),
      ),
    );
  const before = await boxes();
  await openOptions(page);
  assert.deepEqual(await boxes(), before, "opening the panel moves nothing");
  const panel = await page.evaluate(() => ({
    keys: [...document.querySelectorAll("#options [data-option]")].map(
      (row) => row.dataset.option,
    ),
    groups: [...document.querySelectorAll("#options .option-group h3")].map(
      (h) => h.textContent,
    ),
    box: document.getElementById("options").getBoundingClientRect().toJSON(),
    button: document
      .getElementById("options-button")
      .getBoundingClientRect()
      .toJSON(),
  }));
  // The renderer's row, last, which every map has: Auto, drawing WebGPU.
  assert.deepEqual(panel.keys, [...Object.keys(GLOBE_OPTIONS), "backend"]);
  assert.deepEqual(panel.groups, [
    "Terrain and light",
    "Buildings",
    "Globe and sky",
    "Map",
    "Renderer",
  ]);
  assert.deepEqual(await renderer(page), {
    picked: "auto",
    active: "webgpu",
  });
  assert.ok(
    panel.box.right <= panel.button.left &&
      panel.box.top >= 0 &&
      panel.box.bottom <= 800,
    `the panel opens beside its button: ${JSON.stringify(panel)}`,
  );

  // Each choice edits the live style and keeps the URL.
  await option(page, "shadows").click();
  await option(page, "projection").locator('[data-value="globe"]').click();
  await settles(
    page,
    state,
    expect({ colors: "srgb" }, { colors: "srgb" }, ["fog"], { colors: "srgb" }),
    "shadows back on and the globe leave the URL",
  );
  // The sun now: a time, which puts the light on the map.
  await option(page, "now").click();
  await settles(
    page,
    state,
    expect(
      { colors: "srgb", now: true, light: "map" },
      { colors: "srgb", now: "number", anchor: "map" },
      ["fog"],
      { colors: "srgb", now: "1", light: "map" },
    ),
    "the sun now fixes the light to the map",
  );
  await option(page, "now").click();
  await option(page, "light").locator('[data-value="viewport"]').click();
  await settles(
    page,
    state,
    expect({ colors: "srgb" }, { colors: "srgb" }, ["fog"], { colors: "srgb" }),
    "the designed sun back",
  );
  await option(page, "buildings").click();
  await settles(
    page,
    state,
    expect(
      { colors: "srgb", buildings: false },
      { colors: "srgb", buildings: "none" },
      ["shadows", "ao", "fog"],
      { colors: "srgb", buildings: "0" },
    ),
    "without 3D buildings, shadows and AO wait",
  );
  await option(page, "buildings").click();
  await option(page, "ao").click();
  await option(page, "terrain").click();
  // The keyboard: a switch takes Space, a choice the arrow keys.
  await option(page, "labels").focus();
  await page.keyboard.press("Space");
  await option(page, "colors").locator('[aria-checked="true"]').focus();
  await page.keyboard.press("ArrowLeft");
  const chosen = expect(
    { ao: false, terrain: true, labels: false },
    { ao: 0, terrain: true, labels: ["none"] },
    [],
    { ao: "0", terrain: "1", labels: "0" },
  );
  await settles(
    page,
    state,
    chosen,
    "AO, terrain, labels and colours by click and key",
  );
  assert.ok(
    await page.evaluate(() => window.stayed),
    "the switches edit the style without a reload",
  );

  // Escape closes the panel, focus back on its button.
  await page.keyboard.press("Escape");
  assert.deepEqual(
    await page.evaluate(() => [app.options.open, document.activeElement?.id]),
    [false, "options-button"],
  );

  // The link reproduces the view.
  await page.reload();
  await page.waitForFunction(
    () => app.scene === "globe" && app.options.values,
    null,
    { timeout: 60_000 },
  );
  await settles(page, state, chosen, "a reload draws the same choices");
  // (`light` and `terrain` check that another entry drops them, and
  // `failing` that a switch away from this offline Globe draws.)
  assert.deepEqual(errors, []);
  await page.close();

  // At phone width the panel is a sheet across the bottom: on screen, under
  // the picker, scrolling to its last option, with no sideways scroll.
  const phone = await browser.newPage({
    viewport: { width: 360, height: 640 },
  });
  await offlineGlobe(phone, server);
  await phone.goto(`${server.url}?style=globe#14/35.687/139.74`);
  await phone.waitForFunction(() => app.options.values, null, {
    timeout: 60_000,
  });
  const scene = await phone.evaluate(() =>
    document.getElementById("scene-button").getBoundingClientRect().toJSON(),
  );
  await openOptions(phone);
  // Once it has risen into place.
  const sheet = await phone.evaluate(async () => {
    const panel = document.getElementById("options");
    await Promise.all(panel.getAnimations().map((a) => a.finished));
    return {
      box: panel.getBoundingClientRect().toJSON(),
      scroll: document.documentElement.scrollWidth,
    };
  });
  assert.ok(
    sheet.box.left === 0 &&
      sheet.box.right === 360 &&
      Math.abs(sheet.box.bottom - 640) < 1 &&
      sheet.box.top > scene.bottom &&
      sheet.scroll <= 360,
    `the sheet: ${JSON.stringify(sheet)}`,
  );
  await option(phone, "labels").click();
  // An option below the fold, scrolled to, takes a tap and keeps the URL.
  await phone.waitForFunction(
    () =>
      app.options.values?.labels === false &&
      new URLSearchParams(location.search).get("labels") === "0",
  );
  await phone.close();
  console.log(
    "options: the Globe's options apply from the URL, edit the style in place by click and key, " +
      "keep the URL, survive a reload, move nothing as the panel opens, and fit a phone",
  );
}

/** Logs every WebGPU submit's validation errors on the page's console, from
 * the page and its workers (a worker's reach the page over a channel): Chrome
 * reports them through DevTools' log, which Playwright doesn't show from
 * workers, and only once the device next does work. Runs as an init script
 * and, through `Worker`, in each worker. */
function reportGpuErrors() {
  const channel = new BroadcastChannel("e2e-gpu-errors");
  const report = globalThis.document
    ? (m) => console.error(m)
    : (m) => channel.postMessage(m);
  if (globalThis.document) channel.onmessage = (e) => report(e.data);
  const request = globalThis.GPUAdapter?.prototype.requestDevice;
  if (request) {
    GPUAdapter.prototype.requestDevice = async function (...args) {
      const device = await request.apply(this, args);
      const { queue } = device;
      const submit = queue.submit;
      queue.submit = (...buffers) => {
        device.pushErrorScope("validation");
        submit.apply(queue, buffers);
        device.popErrorScope().then((e) => e && report(`WebGPU: ${e.message}`));
      };
      return device;
    };
  }
  const Base = globalThis.Worker;
  if (!Base) return;
  const patch = `data:text/javascript,${encodeURIComponent(`(${reportGpuErrors})()`)}`;
  globalThis.Worker = class extends Base {
    constructor(url, options) {
      const source = `import ${JSON.stringify(patch)}; import ${JSON.stringify(String(url))};`;
      super(
        URL.createObjectURL(new Blob([source], { type: "text/javascript" })),
        options,
      );
    }
  };
}

async function resize(browser, server) {
  const { page, errors } = await open(browser, { width: 1000, height: 700 });
  await page.addInitScript(`(${reportGpuErrors})()`);
  await page.route(
    (u) => !u.href.startsWith(server.url),
    (route) => route.abort("internetdisconnected"),
  );
  const idle = () =>
    page.waitForFunction(
      () => document.getElementById("app").dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
  await page.goto(`${server.url}?style=alps`);
  await page.waitForFunction(() => app.map, null, { timeout: 60_000 });
  await idle();
  // Offline, every layer drapes; circles draw over the terrain, reading the
  // terrain's depth for what it hides.
  await page.evaluate(async () => {
    const { map } = globalThis.orbis;
    const [lon, lat] = map.camera.current.center;
    const features = [];
    for (let i = 0; i < 100; i++) {
      const at = [
        lon + ((i % 10) - 4.5) * 0.02,
        lat + (Math.floor(i / 10) - 4.5) * 0.015,
      ];
      features.push({
        type: "Feature",
        properties: {},
        geometry: { type: "Point", coordinates: at },
      });
    }
    await map.edit((tx) =>
      tx
        .addSource("e2e-points", {
          type: "geojson",
          data: { type: "FeatureCollection", features },
        })
        .addLayer({
          id: "e2e-points",
          type: "circle",
          source: "e2e-points",
          paint: { "circle-radius": 6 },
        }),
    );
  });
  await idle();
  for (const viewport of [
    { width: 1200, height: 800 },
    { width: 640, height: 480 },
    { width: 1000, height: 700 },
  ]) {
    await page.setViewportSize(viewport);
    // Frames at the new size, then rest.
    await page.evaluate(
      () =>
        new Promise((r) =>
          requestAnimationFrame(() => requestAnimationFrame(r)),
        ),
    );
    await idle();
    const { width, height, channels, pixels } = decodePng(
      await page.locator("#map").screenshot(),
    );
    let dark = 0;
    const colours = new Set();
    for (let i = 0; i < pixels.length; i += channels) {
      if (pixels[i] + pixels[i + 1] + pixels[i + 2] < 24) dark++;
      colours.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]);
    }
    const share = dark / (width * height);
    assert.ok(
      share < 0.2 && colours.size > 256,
      `at ${viewport.width}×${viewport.height} the map is ${(share * 100).toFixed(0)}% black in ${colours.size} colours`,
    );
    // Offline, the style's live basemap fails to load; nothing else may.
    assert.deepEqual(
      errors.filter((e) => !/network error/.test(e)),
      [],
      `at ${viewport.width}×${viewport.height}`,
    );
  }
  await page.close();
  console.log(
    "resize: the Alps terrain draws after each of three resizes, without a WebGPU error",
  );
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
  const idle = () =>
    page.waitForFunction(
      () =>
        ["idle", "error"].includes(document.getElementById("app")?.dataset.map),
      null,
      { timeout: 60_000 },
    );
  await page.goto(`${server.url}?style=liberty${at}`);
  await idle();
  await page.click("#menu-button");
  await page.click("#offline-button");
  const started = Date.now();
  await page.click("#offline .primary");
  await page.waitForFunction(
    () =>
      /Downloaded|failed/.test(
        document.querySelector(".offline-status")?.textContent ?? "",
      ),
    null,
    { timeout: 180_000 },
  );
  const status = await page.textContent(".offline-status");
  console.log(`offline: ${status} (${Date.now() - started} ms)`);
  assert.ok(status.startsWith("Downloaded"), status);

  cut = true;
  await page.route(
    (u) => !u.href.startsWith(server.url),
    (route) => route.abort("internetdisconnected"),
  );
  const reopened = Date.now();
  await page.reload();
  await idle();
  const state = await page.evaluate(
    () => document.getElementById("app").dataset.map,
  );
  console.log(`offline: reopened ${state} in ${Date.now() - reopened} ms`);
  assert.equal(state, "idle");
  assert.deepEqual(errors, []);
  await page.close();
}

async function split(browser, server) {
  const entries = [
    ["tokyo", () => {}],
    ["globe", (page) => offlineGlobe(page, server)],
  ];
  for (const [entry, offline] of entries) {
    // Offline, the Globe's tiles fail and say so: `failing` checks that.
    const page = await browser.newPage({
      viewport: { width: 800, height: 600 },
    });
    // How each request for core-deferred.wasm went: `fetch` once the map is
    // idle, `xhr` from a placeholder, in any of the map's threads.
    const loads = [];
    page.context().on("request", (r) => {
      if (/\/core-deferred[^/]*\.wasm/.test(r.url()))
        loads.push(r.resourceType());
    });
    await offline(page);
    await page.goto(`${server.url}?style=${entry}`);
    await page.waitForFunction(
      () => document.getElementById("app")?.dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
    assert.ok(
      !loads.includes("xhr"),
      `${entry}: its first frames called into core-deferred.wasm; \`pnpm coverage --tour --write\` refreshes web/deferred.txt`,
    );
    const deadline = Date.now() + 10_000;
    while (!loads.includes("fetch") && Date.now() < deadline)
      await page.waitForTimeout(100);
    assert.deepEqual(loads, ["fetch"], `${entry}: fetched once idle`);
    await page.close();
    console.log(
      `split: ${entry} idle without core-deferred.wasm, then fetched it`,
    );
  }
}

/** The comparison's pairs the check opens, as `?compare=` values: `""` is Orbis and GL JS. */
const PAIRS = arg(
  "pairs",
  ";gljs,orbis;orbis,native;native,gljs;orbis,orbis;gljs,gljs",
).split(";");
const ENGINE_REQUESTS = {
  gljs: /maplibre-gl|\/gljs-/,
  native: /\/native-|\/native\/\d|isolate\.js/,
};

async function compare(browser, server) {
  // One context, as one browser: the isolating service worker, once
  // registered, stays for later pages.
  const context = await browser.newContext({
    viewport: { width: 800, height: 500 },
  });
  const open = async (url) => {
    const page = await context.newPage();
    const pageErrors = [];
    page.on("pageerror", (e) => pageErrors.push(e.message));
    // Console errors explain a failure, but resources missing on purpose (the
    // Tokyo extract's glyph ranges) log some.
    const logged = [];
    page.on("console", (m) => {
      if (m.type() === "error" && !/Failed to load resource/.test(m.text()))
        logged.push(m.text());
    });
    const requests = [];
    page.on("request", (r) => requests.push(r.url()));
    await page.goto(url);
    const loaded = (engine) =>
      requests.filter((url) => ENGINE_REQUESTS[engine].test(url));
    return { page, pageErrors, loaded, logged };
  };
  const idle = (page) =>
    page.waitForFunction(
      () => document.getElementById("app").dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
  // Every side's map at one camera once they settle. A page comparing with
  // Native reloads into isolation first, which ends a wait in the page, so
  // this polls.
  const settled = async ({ page, logged }, name) => {
    const state = () =>
      page
        .evaluate(() => {
          const sides = window.app?.map && app.compare.sides;
          const own = sides && app.map.camera.current;
          const close = (a, b) => Math.abs(a - b) < 1e-7;
          const done = Boolean(
            sides?.every((s) => s.loaded) &&
            document.getElementById("app").dataset.map === "idle" &&
            sides.every(
              ({ camera }) =>
                close(camera.center[0], own.center[0]) &&
                close(camera.center[1], own.center[1]) &&
                close(camera.zoom, own.zoom) &&
                close(camera.bearing, own.bearing),
            ),
          );
          return { done, own, sides };
        })
        .catch((e) => ({ done: false, error: e.message }));
    const start = Date.now();
    for (let now = await state(); !now.done; now = await state()) {
      if (Date.now() - start > 60_000) {
        throw new Error(
          `${name}: the maps didn't settle at one camera: ${JSON.stringify(now)}; logged ${JSON.stringify(logged)}`,
        );
      }
      await page.waitForTimeout(100);
    }
  };

  {
    const { page, pageErrors, loaded } = await open(
      `${server.url}?style=tokyo-mvt`,
    );
    await idle(page);
    assert.deepEqual(
      [...loaded("gljs"), ...loaded("native")],
      [],
      "the plain page fetches no other engine",
    );
    assert.equal(await page.evaluate(() => crossOriginIsolated), false);
    assert.deepEqual(pageErrors, []);
    await page.close();
  }

  {
    // Native draws one side at a time: a second is GL JS instead, and the
    // other side's picker offers no Native.
    const { page } = await open(
      `${server.url}?style=tokyo-mvt&compare=native,native`,
    );
    await page.waitForFunction(
      () => app.compare.sides?.length === 2 && crossOriginIsolated,
      null,
      { timeout: 60_000 },
    );
    assert.deepEqual(
      await page.evaluate(() => app.compare.sides.map((s) => s.engine)),
      ["native", "gljs"],
    );
    assert.equal(
      await page.evaluate(
        () =>
          document.querySelector('.compare-label.right option[value="native"]')
            .disabled,
      ),
      true,
      "the right side offers no second Native",
    );
    await page.close();
  }

  for (const pair of PAIRS) {
    const engines = pair ? pair.split(",") : ["orbis", "gljs"];
    const opened = await open(
      `${server.url}?style=tokyo-mvt&compare${pair ? `=${pair}` : ""}#15/35.687/139.74`,
    );
    const { page, pageErrors, loaded } = opened;
    const name = engines.join(" and ");
    await settled(opened, name);
    // Only a page comparing with MapLibre Native is isolated; it reloads into it.
    assert.equal(
      await page.evaluate(() => crossOriginIsolated),
      engines.includes("native"),
      `${name}: isolation`,
    );
    for (const engine of ["gljs", "native"]) {
      assert.equal(
        loaded(engine).length > 0,
        engines.includes(engine),
        `${name}: ${engine} loads only when picked`,
      );
    }
    // Each side's canvas fills the stage, and the side's half is its own.
    const sides = await page.evaluate(() => app.compare.sides);
    assert.deepEqual(
      sides.map((s) => s.engine),
      engines,
    );
    for (const [i, side] of ["left", "right"].entries()) {
      const box = await page
        .locator(sides[i].own ? "#map canvas" : `.compare-map.${side} canvas`)
        .first()
        .boundingBox();
      assert.ok(
        box && box.width === 800 && box.height === 500,
        `${name}: the ${side} canvas fills the stage: ${JSON.stringify(box)}`,
      );
      assert.ok(
        await page.evaluate(
          ([own, side, x]) =>
            document
              .querySelector(own ? "#map" : `.compare-map.${side}`)
              .contains(document.elementFromPoint(x, 250)),
          [sides[i].own, side, 200 + 400 * i],
        ),
        `${name}: the ${side} half is its map's`,
      );
    }
    // A drag on either side reaches that side's map, which leads; the other follows.
    for (const [i, x] of [600, 200, 600].entries()) {
      const side = i % 2 === 0 ? 1 : 0;
      const before = await page.evaluate(() => app.map.camera.current.center);
      await page.mouse.move(x, 250);
      await page.mouse.down();
      await page.mouse.move(x - 100, 300, { steps: 10 });
      await page.mouse.up();
      await settled(opened, name);
      const after = await page.evaluate(() => ({
        center: app.map.camera.current.center,
        leads: app.compare.sides.map((s) => s.leads),
      }));
      assert.ok(
        Math.hypot(after.center[0] - before[0], after.center[1] - before[1]) >
          1e-4,
        `${name}: a drag on the ${engines[side]} side moves the maps`,
      );
      assert.ok(
        after.leads[side],
        `${name}: the ${engines[side]} side leads after its drag`,
      );
    }
    assert.deepEqual(pageErrors, [], name);
    await page.close();
  }
  {
    // The map options reach the other engine: GL JS's map hides the labels
    // and turns to the globe with Orbis's.
    const opened = await open(
      `${server.url}?style=tokyo-mvt&compare#15/35.687/139.74`,
    );
    const { page, pageErrors } = opened;
    await settled(opened, "options");
    const other = () =>
      page.evaluate(async () => {
        const style = (await app.compare.styles())[1];
        return {
          projection: style.projection?.type ?? "mercator",
          labels: [
            ...new Set(
              style.layers
                .filter((l) => l.type === "symbol")
                .map((l) => l.layout?.visibility ?? "visible"),
            ),
          ],
        };
      });
    assert.deepEqual(await other(), {
      projection: "mercator",
      labels: ["visible"],
    });
    await openOptions(page);
    await option(page, "labels").click();
    await option(page, "projection").locator('[data-value="globe"]').click();
    await settles(
      page,
      other,
      { projection: "globe", labels: ["none"] },
      "GL JS follows the map options",
    );
    assert.deepEqual(pageErrors, []);
    await page.close();
  }
  await context.close();
  console.log("compare: GL JS follows the map options");
  console.log(
    `compare: the plain page fetches no other engine and isn't isolated; ${PAIRS.length} pairs ` +
      "fill the stage, load only their engines, isolate only for MapLibre Native, and follow drags on either side; " +
      "a second Native is GL JS",
  );
}

/** The slope at the newest sample of the least-squares quadratic through
 * the samples of the last 100 ms back to the first gap over 40 ms, per
 * second, and zero for a release more than 40 ms after it: Compose's
 * `VelocityTracker1D`, written apart from the core's for a check. */
function releaseVelocity(samples, release) {
  const [newest] = samples.at(-1);
  if (release - newest > 40) return 0;
  const points = [];
  let previous = newest;
  for (const [t, v] of [...samples].reverse()) {
    if (newest - t > 100 || previous - t > 40) break;
    previous = t;
    points.push([t - newest, v]);
  }
  if (points.length < 3) return 0;
  // Normal equations for y = a + b x + c x², solved by elimination.
  const m = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  for (const [x, y] of points) {
    const p = [1, x, x * x];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) m[i][j] += p[i] * p[j];
      m[i][3] += p[i] * y;
    }
  }
  for (let i = 0; i < 3; i++) {
    for (let k = i + 1; k < 3; k++) {
      const f = m[k][i] / m[i][i];
      for (let j = i; j < 4; j++) m[k][j] -= f * m[i][j];
    }
  }
  const c = m[2][3] / m[2][2];
  const b = (m[1][3] - m[1][2] * c) / m[1][1];
  return b * 1000;
}

/** mln-ffi's (and MapLibre Compose's) pan momentum: friction halving the
 * speed every 140 ms, stopping at 10 px/s, from releases of 200 to
 * 10 000 px/s. */
const PAN = { halfLife: 140, stop: 10, min: 200, max: 10_000 };

async function gestures(browser, server) {
  const { page, errors } = await open(browser, { width: 400, height: 300 });
  // On the page's thread, where the log below reads each frame's time, and a
  // projection answers for the camera at its call.
  await page.goto(`${server.url}scenario.html?worker=0`);
  const scenario = (fn, arg) =>
    page.evaluate(
      ([fn, arg]) =>
        window.scenario.then((s) =>
          new Function("s", "arg", `return (${fn})(s, arg)`)(s, arg),
        ),
      [fn.toString(), arg],
    );
  const cdp = await page.context().newCDPSession(page);
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const SHIFT = 8;
  const mouse = (
    type,
    x,
    y,
    { button = "left", modifiers = 0, clickCount = 1 } = {},
  ) =>
    cdp.send("Input.dispatchMouseEvent", {
      type,
      x,
      y,
      button,
      modifiers,
      clickCount,
      buttons: type === "mouseReleased" ? 0 : { left: 1, right: 2 }[button],
    });
  const touch = (type, points) =>
    cdp.send("Input.dispatchTouchEvent", {
      type,
      touchPoints: points.map(([x, y], id) => ({ x, y, id })),
    });
  /** The camera at rest, at zoom 4 over the box unless `at` says otherwise. */
  const reset = (at = {}) =>
    scenario(async (s, at) => {
      s.map.camera.jumpTo({
        center: [0, 0],
        zoom: 4,
        bearing: 0,
        pitch: 0,
        ...at,
      });
      await s.idle();
    }, at);
  const camera = () =>
    scenario(async (s) => {
      await s.idle();
      return s.map.camera.current;
    });

  // The page logs pointer events (before the map takes them) and, on each
  // camera move, the frame's time and where the origin shows.
  await scenario((s) => {
    const raf = window.requestAnimationFrame;
    let frameTime = 0;
    window.requestAnimationFrame = (callback) =>
      raf((t) => {
        frameTime = t;
        callback(t);
      });
    window.gestureLog = { pointers: [], frames: [], atRelease: null };
    for (const type of ["pointerdown", "pointermove", "pointerup"]) {
      addEventListener(
        type,
        (e) => {
          window.gestureLog.pointers.push({
            type,
            t: e.timeStamp,
            x: e.clientX,
            y: e.clientY,
            kind: e.pointerType,
          });
          if (type !== "pointerup") return;
          const release = (window.gestureLog.atRelease = {
            t: e.timeStamp,
            at: null,
          });
          s.map.project([0, 0]).then((at) => {
            release.at = at;
          });
        },
        true,
      );
    }
    s.map.on("move", () => {
      const frame = { t: frameTime, at: null, zoom: s.map.camera.current.zoom };
      window.gestureLog.frames.push(frame);
      s.map.project([0, 0]).then((at) => {
        frame.at = at;
      });
    });
  });
  const log = (clear) =>
    page.evaluate((clear) => {
      const out = structuredClone(window.gestureLog);
      if (clear)
        Object.assign(window.gestureLog, {
          pointers: [],
          frames: [],
          atRelease: null,
        });
      return out;
    }, clear);

  /** A horizontal fling: `steps` moves of `dx` px, one a frame (the
   * protocol answers each once a frame has taken it), released at once.
   * Returns the measurements against mln-ffi's decay. */
  async function fling(kind, dx, steps = 8) {
    await reset();
    await log(true);
    let x = 40;
    if (kind === "mouse") await mouse("mousePressed", x, 100);
    else await touch("touchStart", [[x, 100]]);
    for (let i = 0; i < steps; i++) {
      x += dx;
      if (kind === "mouse") await mouse("mouseMoved", x, 100);
      else await touch("touchMove", [[x, 100]]);
    }
    if (kind === "mouse") await mouse("mouseReleased", x, 100);
    else await touch("touchEnd", []);
    await camera();
    const { pointers, frames, atRelease } = await log(true);
    const moves = pointers
      .filter((p) => p.type === "pointermove")
      .map((p) => [p.t, p.x]);
    const velocity = releaseVelocity(moves, atRelease.t);
    const k = Math.LN2 / (PAN.halfLife / 1000);
    const speed = Math.min(velocity, PAN.max);
    const expected =
      speed >= PAN.min
        ? {
            distance: (speed - PAN.stop) / k,
            duration: (Math.log(speed / PAN.stop) / k) * 1000,
          }
        : { distance: 0, duration: 0 };
    const glide = frames.filter((f) => f.t > atRelease.t);
    const last = glide.at(-1) ?? { t: atRelease.t, at: atRelease.at };
    const distance = last.at[0] - atRelease.at[0];
    // The remaining distance plus stop / k falls as e^(−kt): its log's
    // slope over the glide's frames gives the measured half-life.
    const fit = glide
      .slice(0, -1)
      .map((f) => [
        (f.t - atRelease.t) / 1000,
        Math.log(last.at[0] - f.at[0] + PAN.stop / k),
      ])
      .filter(([, y]) => Number.isFinite(y));
    const n = fit.length;
    const [sx, sy] = fit.reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
    const sxx = fit.reduce((a, [x]) => a + x * x, 0);
    const sxy = fit.reduce((a, [x, y]) => a + x * y, 0);
    const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    return {
      kind,
      velocity,
      distance,
      expected,
      halfLife: (Math.LN2 / -slope) * 1000,
      duration: last.t - atRelease.t,
      frames: glide.length,
    };
  }

  const results = [];
  // One move a frame: the slow fling's 10 px is about 600 px/s at 60 fps,
  // three times the slowest release that glides, so a loaded machine's
  // slower frames still release a fling.
  for (const [kind, dx] of [
    ["mouse", 10],
    ["mouse", 20],
    ["touch", 10],
    ["touch", 30],
  ]) {
    const r = await fling(kind, dx);
    results.push(r);
    assert.ok(
      r.velocity > PAN.min,
      `${kind} fling of ${dx} px/move: ${r.velocity} px/s`,
    );
    assert.ok(
      Math.abs(r.distance - r.expected.distance) <
        Math.max(1, r.expected.distance * 0.02),
      `${kind}: glided ${r.distance.toFixed(1)} px, mln-ffi's decay gives ${r.expected.distance.toFixed(1)} from ${r.velocity.toFixed(0)} px/s`,
    );
    assert.ok(
      Math.abs(r.halfLife - PAN.halfLife) < 5,
      `${kind}: half-life ${r.halfLife.toFixed(1)} ms`,
    );
    // Frames land on vsync: the last moving one is the first at or after
    // the end (allowing a dropped frame).
    assert.ok(
      r.duration >= r.expected.duration - 1 &&
        r.duration < r.expected.duration + 40,
      `${kind}: glided ${r.duration.toFixed(0)} ms of ${r.expected.duration.toFixed(0)}`,
    );
  }

  // A slow drag and a drag held still before release don't glide.
  for (const held of [0, 120]) {
    await reset();
    await log(true);
    await mouse("mousePressed", 40, 100);
    for (let i = 1; i <= 8; i++)
      await mouse("mouseMoved", 40 + (held ? 10 : 1) * i, 100);
    await wait(held);
    await mouse("mouseReleased", held ? 120 : 48, 100);
    await camera();
    const { frames, atRelease } = await log(true);
    assert.equal(
      frames.filter((f) => f.t > atRelease.t + 1).length,
      0,
      held ? "held still" : "slow",
    );
  }

  // A pinch spreading fast keeps zooming after release, by at most 2.5.
  await reset();
  await log(true);
  await touch("touchStart", [
    [90, 100],
    [110, 100],
  ]);
  for (let i = 1; i <= 8; i++)
    await touch("touchMove", [
      [90 - 8 * i, 100],
      [110 + 8 * i, 100],
    ]);
  const pinched = await scenario((s) => s.map.camera.current.zoom);
  await touch("touchEnd", []);
  const pinchEnd = (await camera()).zoom;
  assert.ok(
    pinchEnd - pinched > 0.1 && pinchEnd - pinched <= 2.5,
    `pinch glide: ${pinched} → ${pinchEnd}`,
  );

  // Shift-drag draws a box and zooms to it: 100 × 80 in 200 × 200 is one level.
  await reset();
  await mouse("mousePressed", 50, 50, { modifiers: SHIFT });
  await mouse("mouseMoved", 100, 90, { modifiers: SHIFT });
  await mouse("mouseMoved", 150, 130, { modifiers: SHIFT });
  const shown = await page.evaluate(() =>
    [...document.querySelectorAll("#map div")]
      .map((d) => d.getBoundingClientRect())
      .find(
        (r) =>
          r.width === 100 && r.height === 80 && r.left === 50 && r.top === 50,
      ),
  );
  assert.ok(shown, "the box shows while it's drawn");
  await mouse("mouseReleased", 150, 130, { modifiers: SHIFT });
  const boxed = await camera();
  assert.ok(Math.abs(boxed.zoom - 5) < 1e-6, `box zoom: ${boxed.zoom}`);

  // Shift-double-click zooms out; double-tap zooms in; a two-finger tap
  // zooms out; double-tap-and-drag zooms.
  const zoomAfter = async (gesture) => {
    await reset();
    await gesture();
    return (await camera()).zoom - 4;
  };
  const clicks = await zoomAfter(async () => {
    for (const clickCount of [1, 2]) {
      await mouse("mousePressed", 100, 100, { modifiers: SHIFT, clickCount });
      await mouse("mouseReleased", 100, 100, { modifiers: SHIFT, clickCount });
      await wait(50);
    }
  });
  assert.ok(Math.abs(clicks + 1) < 1e-6, `shift-double-click: ${clicks}`);
  const taps = await zoomAfter(async () => {
    for (let i = 0; i < 2; i++) {
      await touch("touchStart", [[100, 100]]);
      await touch("touchEnd", []);
      await wait(50);
    }
  });
  assert.ok(Math.abs(taps - 1) < 1e-6, `double-tap: ${taps}`);
  const twoFinger = await zoomAfter(async () => {
    await touch("touchStart", [
      [80, 100],
      [120, 100],
    ]);
    await wait(50);
    await touch("touchEnd", []);
  });
  assert.ok(Math.abs(twoFinger + 1) < 1e-6, `two-finger tap: ${twoFinger}`);
  const quick = await zoomAfter(async () => {
    await touch("touchStart", [[100, 60]]);
    await touch("touchEnd", []);
    await wait(50);
    await touch("touchStart", [[100, 60]]);
    for (let i = 1; i <= 8; i++) await touch("touchMove", [[100, 60 + 8 * i]]);
    await wait(100);
    await touch("touchEnd", []);
  });
  assert.ok(
    Math.abs(quick - 0.5) < 0.05,
    `double-tap-drag 64 px down: ${quick}`,
  );

  // A right-drag turn of a few degrees, held, settles on north.
  await reset();
  await mouse("mousePressed", 100, 100, { button: "right" });
  for (let i = 1; i <= 4; i++)
    await mouse("mouseMoved", 100 + 2 * i, 100, { button: "right" });
  const turned = await scenario((s) => s.map.camera.current.bearing);
  await wait(100);
  await mouse("mouseReleased", 108, 100, { button: "right" });
  const settled = await camera();
  assert.ok(Math.abs(turned) > 3 && Math.abs(turned) < 7, `turned ${turned}`);
  assert.ok(Math.abs(settled.bearing) < 1e-9, `settled at ${settled.bearing}`);

  assert.deepEqual(errors, []);
  await page.close();
  for (const r of results) {
    console.log(
      `gestures: ${r.kind} fling at ${r.velocity.toFixed(0)} px/s glided ${r.distance.toFixed(1)} px ` +
        `(mln-ffi: ${r.expected.distance.toFixed(1)}) over ${r.duration.toFixed(0)} ms ` +
        `(${r.expected.duration.toFixed(0)}) in ${r.frames} frames, half-life ${r.halfLife.toFixed(1)} ms (140)`,
    );
  }
  console.log(
    `gestures: pinch glide +${(pinchEnd - pinched).toFixed(2)} levels; box, shift-double-click, double-tap, ` +
      `two-finger tap, double-tap-drag (${quick.toFixed(3)}) and north snap (from ${turned.toFixed(1)}°) as expected`,
  );
}

/** Patches this context, and every worker it starts, before their own
 * scripts run: `noise` flips the low bit of the red byte of about half the
 * texels a 2D canvas's `getImageData` returns (a DEM's 6.5 km in Mapbox
 * encoding), `!frames` takes `VideoFrame` away. */
function perturbReadback(noise, frames) {
  if (noise) {
    for (const Context of [
      globalThis.CanvasRenderingContext2D,
      globalThis.OffscreenCanvasRenderingContext2D,
    ]) {
      if (!Context) continue;
      const read = Context.prototype.getImageData;
      Context.prototype.getImageData = function (...args) {
        const image = read.apply(this, args);
        for (let i = 0; i < image.data.length; i += 4)
          image.data[i] ^= Math.imul(i, 0x9e3779b1) >>> 31;
        return image;
      };
    }
  }
  if (!frames) delete globalThis.VideoFrame;
  const Base = globalThis.Worker;
  if (!Base) return;
  const patch = `data:text/javascript,${encodeURIComponent(`(${perturbReadback})(${noise}, ${frames})`)}`;
  globalThis.Worker = class extends Base {
    constructor(url, options) {
      // Static imports run in order: the patch, then the worker's script.
      const source = `import ${JSON.stringify(patch)}; import ${JSON.stringify(String(url))};`;
      super(
        URL.createObjectURL(new Blob([source], { type: "text/javascript" })),
        options,
      );
    }
  };
}

async function readback(browser, server) {
  // A Mapbox-encoded DEM of hills 800 m high, every tile the same.
  const size = 256;
  const dem = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const metres = 1000 + 800 * Math.sin(x / 20) * Math.cos(y / 25);
      const v = Math.round((metres + 10000) * 10);
      dem.set([v >> 16, (v >> 8) & 255, v & 255, 255], (y * size + x) * 4);
    }
  }
  await writeFile(join(server.styles, "dem.png"), encodePng(size, size, dem));
  const style = {
    version: 8,
    sprite: `${server.url}e2e/data/sprites/ofm`,
    sources: {
      dem: {
        type: "raster-dem",
        tiles: [`${server.url}e2e/styles/dem.png`],
        tileSize: size,
        encoding: "mapbox",
        maxzoom: 14,
      },
      point: { type: "geojson", data: { type: "Point", coordinates: [0, 0] } },
    },
    terrain: { source: "dem" },
    layers: [
      {
        id: "background",
        type: "background",
        paint: { "background-color": "#ddd" },
      },
      { id: "hillshade", type: "hillshade", source: "dem" },
      {
        id: "icon",
        type: "symbol",
        source: "point",
        layout: { "icon-image": "pedestrian_polygon", "icon-size": 2 },
      },
    ],
    center: [0, 0],
    zoom: 12,
    pitch: 60,
  };
  const request = {
    native: true,
    style: JSON.stringify(style),
    width: 256,
    height: 256,
    ratio: 1,
    max_pitch: 60,
    cross_source_collisions: true,
    timeout_ms: 20_000,
    operations: [],
  };
  const draw = async (noise, frames) => {
    const { page, errors } = await open(browser, { width: 256, height: 256 });
    try {
      if (noise || !frames)
        await page.addInitScript(`(${perturbReadback})(${noise}, ${frames})`);
      await page.goto(`${server.url}fixture.html`);
      await page.waitForFunction(() => window.fixture);
      const { rgba, warnings } = await page.evaluate(
        (r) => window.fixture(r),
        request,
      );
      assert.deepEqual(
        [...warnings, ...errors],
        [],
        `noise ${noise}, frames ${frames}`,
      );
      return Buffer.from(rgba, "base64");
    } finally {
      await page.close();
    }
  };
  const clean = await draw(false, true);
  const noisy = await draw(true, true);
  const canvas = await draw(true, false);
  const differing = (a, b) => a.reduce((n, v, i) => n + (v !== b[i]), 0);
  assert.equal(
    differing(clean, noisy),
    0,
    "a perturbed canvas changed what was drawn",
  );
  // Spikes in the DEM change most of the picture.
  const reached = differing(clean, canvas) / clean.length;
  assert.ok(
    reached > 0.25,
    `the noise never reached a canvas readback (${reached})`,
  );
  console.log(
    `readback: the DEM and sprite exact under a perturbed canvas (${(reached * 100).toFixed(0)}% of bytes differ through it)`,
  );
}

/** The scenario, the Tokyo extract and a style's gaps on WebGL 2 (#140), in
 * a Chromium without WebGPU. */
async function webgl2(webgpuBrowser, server) {
  const browser = await launchChromium({ webgpu: false });
  try {
    await scenarioOn(browser, server, "worker", "webgl2");
    await scenarioOn(browser, server, "page", "webgl2");
    console.log("webgl2: the scenario, in the worker and on the page");

    // The extract's views draw as on WebGPU, give or take antialiasing.
    const size = { width: 600, height: 400 };
    const shot = async (browser, backend, view) => {
      const { page, errors } = await open(browser, size);
      await page.goto(
        `${server.url}?style=tokyo-mvt&backend=${backend}${view}`,
      );
      await page.waitForFunction(
        () => document.getElementById("app")?.dataset.map === "idle",
        null,
        { timeout: 60_000 },
      );
      assert.equal(await page.evaluate(() => app.map.backend), backend);
      const image = decodePng(await page.locator("#map").screenshot());
      await page.close();
      assert.deepEqual(errors, [], `${backend} ${view}`);
      return image;
    };
    for (const view of ["#13/35.6830/139.7520", "#16/35.676/139.745/30/60"]) {
      const [gpu, gl] = [
        await shot(webgpuBrowser, "webgpu", view),
        await shot(browser, "webgl2", view),
      ];
      let differ = 0;
      for (let i = 0; i < gpu.pixels.length; i += gpu.channels) {
        const d = [0, 1, 2].map((c) =>
          Math.abs(gpu.pixels[i + c] - gl.pixels[i + c]),
        );
        if (Math.max(...d) > 24) differ++;
      }
      const share = differ / (gpu.width * gpu.height);
      assert.ok(share < 0.01, `${view}: ${(share * 100).toFixed(2)}% differ`);
      console.log(
        `webgl2: the extract at ${view} as on WebGPU (${(share * 100).toFixed(2)}% of pixels differ)`,
      );
    }

    // A style with terrain draws flat, and says so once; a GPU lost
    // recovers on WebGL 2 and draws again.
    const { page, errors } = await open(browser, { width: 300, height: 200 });
    await page.goto(`${server.url}scenario.html?backend=webgl2`);
    const lost = await page.evaluate(() =>
      window.scenario.then(async ({ map }) => {
        const degraded = [];
        map.on("degraded", (d) => degraded.push(d));
        map.edit((tx) =>
          tx
            .addSource("dem", {
              type: "raster-dem",
              tiles: ["http://127.0.0.1:9/{z}/{x}/{y}.png"],
            })
            .setTerrain({ source: "dem" }),
        );
        await map.once("idle");
        const restored = map.once("restored");
        map.loseDevice();
        await restored;
        await map.once("idle");
        return { degraded, backend: map.backend };
      }),
    );
    assert.deepEqual(lost, {
      degraded: [{ gap: "terrain", layers: [] }],
      backend: "webgl2",
    });
    const image = decodePng(await page.locator("#map").screenshot());
    const at =
      ((image.height >> 1) * image.width + (image.width >> 1)) * image.channels;
    assert.deepEqual([...image.pixels.subarray(at, at + 3)], [0, 0, 255]);
    // The gap's console warning, and the DEM tiles nothing serves.
    const unexpected = errors.filter(
      (e) =>
        !/draws terrain otherwise than on WebGPU|127\.0\.0\.1:9|ERR_CONNECTION_REFUSED|Failed to load resource/.test(
          e,
        ),
    );
    assert.deepEqual(unexpected, []);
    await page.close();
    console.log(
      "webgl2: terrain degrades flat and says so; a lost GPU recovers",
    );
  } finally {
    await browser.close();
  }
  await firefoxAttributeZero(server);
}

/**
 * Firefox on desktop GL fakes vertex attribute 0 for a program without
 * one, sized by the draw's largest index, and refuses draws past 10M
 * indices (#179): every WebGL 2 program has one, so Firefox warns of no
 * emulation. Its warnings are capped at 32 a context, which texture
 * uploads use up, so the cap is raised.
 */
async function firefoxAttributeZero(server) {
  const browser = await firefox.launch({
    firefoxUserPrefs: { "webgl.max-warnings-per-context": 100_000 },
  });
  try {
    const { page, errors } = await open(browser, { width: 600, height: 400 });
    await page.goto(
      `${server.url}?style=tokyo-mvt&backend=webgl2#15/35.68/139.76`,
    );
    await page.waitForFunction(
      () => document.getElementById("app")?.dataset.map === "idle",
      null,
      { timeout: 60_000 },
    );
    assert.equal(await page.evaluate(() => app.map.backend), "webgl2");
    await page.close();
    assert.deepEqual(
      errors.filter((e) => /vertex attrib 0/.test(e)),
      [],
      "Firefox emulates attribute 0",
    );
    console.log("webgl2: Firefox draws without emulating attribute 0");
  } finally {
    await browser.close();
  }
}

const CHECKS = {
  scenario,
  webgl2,
  edits,
  playground,
  switching,
  light,
  terrain,
  failing,
  options,
  resize,
  compare,
  gestures,
  readback,
  boot,
  split,
  offline,
};
/** What an empty style shows, on either backend: the page. */
const BOOTED = [255, 255, 255];

async function boot(webgpuBrowser, server) {
  const noWebgpu = () =>
    chromium.launch({
      args: [
        "--disable-features=WebGPU,Vulkan,WebGPUService",
        "--use-angle=metal",
      ],
    });
  // Each browser, the backend its map draws with, and what the map asks for.
  const browsers = [
    ["Chromium", "webgpu", () => webgpuBrowser],
    // WebGL 2 asked for, though WebGPU has an adapter: only core-gl.wasm loads.
    ["Chromium asked for WebGL 2", "webgl2", () => webgpuBrowser, "webgl2"],
    ["Firefox", "webgl2", () => firefox.launch()],
    [
      "Chromium without a GPU",
      "webgl2",
      () => chromium.launch({ args: ["--disable-gpu"] }),
    ],
    ["Chromium without WebGPU", "webgl2", noWebgpu],
    // WebGPU asked for where it has none: the map doesn't start, saying why.
    ["Chromium without WebGPU asked for it", null, noWebgpu, "webgpu"],
  ];
  for (const [name, backend, launch, asked] of browsers) {
    const browser = await launch();
    try {
      for (const worker of [true, false]) {
        const where = `${name}, ${worker ? "worker" : "page"}`;
        const { page, errors } = await open(browser, {
          width: 300,
          height: 300,
        });
        const builds = [];
        page.on("request", (r) => {
          const build = r.url().match(/\/(core(?:-gl)?)_bg[^/]*\.wasm/)?.[1];
          if (build) builds.push(build);
        });
        const query = new URLSearchParams({
          ...(!worker && { worker: "0" }),
          ...(asked && { backend: asked }),
        });
        await page.goto(`${server.url}boot.html?${query}`);
        const booted = await page.evaluate(() => window.booted);
        if (!backend) {
          assert.match(
            booted.error ?? "",
            /^Orbis can't draw with WebGPU, as asked: /,
            `${where}: ${JSON.stringify(booted)}`,
          );
          assert.ok(
            !builds.includes("core-gl"),
            `${where}: core-gl.wasm loads: ${builds}`,
          );
          await page.close();
          console.log(`boot: ${where}: ${booted.error}`);
          continue;
        }
        assert.equal(booted.error, undefined, where);
        assert.equal(booted.backend, backend, where);
        if (asked === "webgl2")
          assert.deepEqual([...new Set(builds)], ["core-gl"], where);
        // Firefox and Chromium hand a worker the canvas.
        assert.equal(booted.inWorker, worker, where);
        await page.waitForTimeout(100);
        const image = decodePng(await page.locator("#map").screenshot());
        const at = (100 * image.width + 100) * image.channels;
        const pixel = [...image.pixels.subarray(at, at + 3)];
        assert.ok(
          pixel.every((v, i) => Math.abs(v - BOOTED[i]) <= 1),
          `${where}: ${pixel}`,
        );
        // Browsers warn that WebGPU has no adapter (Firefox's is
        // blocklisted), Firefox of a deprecated extension wgpu reads, and
        // Chromium without a GPU that its software compositor reads the
        // canvas back; Firefox's WebGL that partial uploads into the data
        // tables are slower than whole ones.
        const unexpected = errors.filter(
          (e) =>
            !/WebGPU is disabled by blocklist|^No available adapters\.$|^Failed to create WebGPU Context Provider$|WEBGL_debug_renderer_info|GPU stall due to ReadPixels|WebGL warning: texSubImage: |After reporting 32, no further warnings/.test(
              e,
            ),
        );
        assert.deepEqual(unexpected, [], where);
        await page.close();
        console.log(`boot: ${where}: ${backend}`);
      }
    } finally {
      if (browser !== webgpuBrowser) await browser.close();
    }
  }

  // The map app's `?backend=webgl2` draws WebGL 2 though WebGPU has an
  // adapter, and its renderer row says so; Auto starts it again on WebGPU.
  const { page } = await open(webgpuBrowser, { width: 1200, height: 800 });
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  const style = encodeURIComponent(`${server.url}e2e/data/playground.json`);
  const drawing = (backend) =>
    page.waitForFunction(
      (backend) =>
        app.map?.backend === backend &&
        document.querySelector('#options [data-option="backend"]')?.dataset
          .active === backend,
      backend,
      { timeout: 60_000 },
    );
  await page.goto(
    `${server.url}?style=${style}&backend=webgl2#15/35.68/139.76`,
  );
  await drawing("webgl2");
  assert.deepEqual(await renderer(page), {
    picked: "webgl2",
    active: "webgl2",
  });
  // What WebGL 2 draws otherwise shows under the scene button while the
  // style uses it, its layers in the title.
  const badge = () =>
    page.evaluate(() => {
      const b = document.getElementById("degraded");
      return b.hidden ? null : { text: b.textContent, title: b.title };
    });
  assert.equal(await badge(), null);
  const extrude = (on) =>
    page.evaluate(
      (on) =>
        app.map.edit((tx) =>
          on
            ? tx.addLayer({
                id: "blocks-3d",
                type: "fill-extrusion",
                source: "district",
                paint: { "fill-extrusion-height": 20 },
              })
            : tx.removeLayer("blocks-3d"),
        ),
      on,
    );
  await extrude(true);
  await page.waitForFunction(() => !document.getElementById("degraded").hidden);
  assert.deepEqual(await badge(), {
    text: "WebGL 2: coarser 3D depth",
    title: "coarser 3D depth: blocks-3d",
  });
  await extrude(false);
  await page.waitForFunction(() => document.getElementById("degraded").hidden);
  await openOptions(page);
  await Promise.all([
    page.waitForEvent("load"),
    option(page, "backend").locator('[data-value="auto"]').click(),
  ]);
  await drawing("webgpu");
  assert.deepEqual(await renderer(page), { picked: "auto", active: "webgpu" });
  const url = new URL(page.url());
  assert.equal(url.searchParams.get("backend"), null, "Auto leaves the URL");
  assert.equal(url.hash, "#15.00/35.68000/139.76000", "the camera stays");
  assert.deepEqual(pageErrors, []);
  await page.close();
  console.log("boot: the map app's renderer: WebGL 2 asked for, then Auto");
}

const DATA = fileURLToPath(new URL("../demo/data/", import.meta.url));
// Styles a check writes, served beside the demo's data.
const styles = await mkdtemp(join(tmpdir(), "orbis-e2e-"));
const server = {
  ...(await serveDemo({
    mounts: { "e2e/data/": DATA, "e2e/styles/": styles },
  })),
  styles,
};
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
  await rm(styles, { recursive: true, force: true });
}
if (failed) process.exit(1);
