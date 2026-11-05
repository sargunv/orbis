// The binding scenario (plan §14.1, decision 0018) through the orbis
// package in headless Chromium, as Rust, C, Kotlin and Swift run it: an
// inline style, pixels, a hit test, a rendered-features query, an edit's
// generation, project and unproject, a drag, a click event, idle drawing
// nothing, and an error. On web/dist/scenario.html, in each thread mode,
// and on WebGL2.
//
//   pnpm scenario [--configs=main,worker,webgl2]
import assert from "node:assert/strict";
import { decodePng } from "./png.mjs";
import { launchChromium, serveDemo } from "./serve.mjs";

const configs = (process.argv.slice(2).find((a) => a.startsWith("--configs="))?.slice(10) ?? "main,worker,webgl2").split(",");
const SIZE = 200;
const CENTER = [SIZE / 2, SIZE / 2];

const server = await serveDemo();
const browser = await launchChromium();
let failed = 0;
try {
  for (const config of configs) {
    const page = await browser.newPage({ viewport: { width: 400, height: 300 } });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    const query = config === "webgl2" ? "thread=main&renderer=webgl2" : `thread=${config}&renderer=webgpu`;
    try {
      await page.goto(`${server.url}scenario.html?${query}`);
      const scenario = (fn, arg) => page.evaluate(([fn, arg]) => window.scenario.then((s) => new Function("s", "arg", `return (${fn})(s, arg)`)(s, arg)), [fn.toString(), arg]);
      const pixels = async () => decodePng(await page.locator("#map").screenshot());
      const pixel = (image, [x, y]) => {
        const i = (Math.round(y) * image.width + Math.round(x)) * image.channels;
        return [...image.pixels.subarray(i, i + 3)];
      };

      // The style loads, draws and idles: styleload, load, then idle.
      const first = await scenario((s) => ({ thread: s.thread, events: s.events().map((e) => e.type) }));
      assert.equal(first.thread, config === "worker" ? "worker" : "main");
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
      const rejected = await scenario((s) => s.map.edit((tx) => tx.setPaintProperty("nope", "fill-color", "#000")).then(() => null, (e) => String(e)));
      assert.ok(rejected, "a bad edit rejects");
      const hit = await scenario((s) => s.map.hitTest(s.map.project([0, 0]))?.layer);
      assert.equal(hit, "box");
      assert.deepEqual(errors, []);
      console.log(`scenario: ${config} passed`);
    } catch (e) {
      failed++;
      console.error(`scenario: ${config} failed: ${e.message}`);
    } finally {
      await page.close();
    }
  }
} finally {
  await browser.close();
  server.close();
}
if (failed) process.exit(1);
