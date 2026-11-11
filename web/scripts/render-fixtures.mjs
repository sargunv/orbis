// The Chrome transport for orb-fixtures: web/demo/fixture.html, which
// draws each fixture through the orbis package. Rust sends one JSON request
// per line; a render's response names RGBA bytes written by this process
// (a GL JS fixture's premultiplied, a Native one's straight), a query's
// (`query` set) carries its features' JSON. Fixture selection, metadata,
// unpremultiplying and pixel comparison belong to Rust. The first argument
// is the assets directory to serve (GL JS's by default); `--offline` serves
// MapLibre Native's cache, whose `https://` and `mapbox://` URLs are files
// too. `--android` draws them in Chrome on the phone assigned to this
// worktree (`devices-cli assign`), its WebGPU or with `--webgl2` its
// WebGL 2.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

import { launchChromium, openAndroid, serveDemo } from "./serve.mjs";

const assets = process.argv[2]
  ? resolve(process.argv[2]) + sep
  : fileURLToPath(new URL("../../fixtures/gl-js/assets/", import.meta.url));
const offline = process.argv.includes("--offline");
// `--webgl2`: Chromium without WebGPU, its maps on WebGL 2
// (`--platform chrome-webgl2`).
const webgl2 = process.argv.includes("--webgl2");
const android = process.argv.includes("--android");
const reply = (value) => process.stdout.write(JSON.stringify(value) + "\n");
const server = await serveDemo({ assets, offline });
// A phone's Chrome has WebGPU; `?backend=webgl2` puts its maps on WebGL 2.
const { browser, page, close } = android
  ? await openAndroid(server, { physical: true })
  : await launchChromium({ unpaced: true, webgpu: !webgl2 }).then(
      async (browser) => ({
        browser,
        page: await browser.newPage(),
        close: () => browser.close(),
      }),
    );
try {
  page.on("console", (m) => {
    if (["error", "warning"].includes(m.type())) console.error(m.text());
  });
  page.on("pageerror", (e) => console.error(e));
  // `ORB_GLSL` names a JSON file of rewrites applied to every GLSL source
  // the page's maps compile, on the page's thread (the map leaves its
  // worker): `{"rewrites": [{"match": regex, "replace": text, "stage":
  // "vertex"|"fragment", "shader": a substring the source must hold}],
  // "dump": dir}`, `dump` writing each fixture's sources to a directory
  // numbered in request order, for bisecting a driver's miscompile without
  // rebuilding the core.
  const glsl = process.env.ORB_GLSL
    ? JSON.parse(await readFile(process.env.ORB_GLSL, "utf8"))
    : undefined;
  let dumped = 0;
  if (glsl)
    await page.addInitScript((config) => {
      delete HTMLElement.prototype.onpointerrawupdate;
      const sources = (globalThis.__glsl = []);
      const stages = new WeakMap();
      const gl = WebGL2RenderingContext.prototype;
      const create = gl.createShader;
      gl.createShader = function (type) {
        const shader = create.call(this, type);
        if (shader)
          stages.set(
            shader,
            type === this.VERTEX_SHADER ? "vertex" : "fragment",
          );
        return shader;
      };
      const source = gl.shaderSource;
      gl.shaderSource = function (shader, text) {
        const stage = stages.get(shader);
        for (const r of config.rewrites ?? []) {
          if (r.stage && r.stage !== stage) continue;
          if (r.shader && !text.includes(r.shader)) continue;
          text = text.replace(new RegExp(r.match, "g"), r.replace);
        }
        if (config.dump) sources.push({ stage, text, shader });
        return source.call(this, shader, text);
      };
      // With `dump`, ANGLE's translation for the driver beside each
      // source (`WEBGL_debug_shaders`), read once it compiles.
      const compile = gl.compileShader;
      gl.compileShader = function (shader) {
        compile.call(this, shader);
        const entry = sources.findLast((s) => s.shader === shader);
        const debug = config.dump && this.getExtension("WEBGL_debug_shaders");
        if (entry && debug)
          entry.translated = debug.getTranslatedShaderSource(shader);
      };
    }, glsl);
  const open = () =>
    page
      .goto(`${server.url}fixture.html${webgl2 ? "?backend=webgl2" : ""}`)
      .then(() => page.waitForFunction(() => window.fixture));
  await open();
  const maxSize = await page.evaluate(async (webgl2) => {
    if (!webgl2)
      return (await navigator.gpu.requestAdapter()).limits
        .maxTextureDimension2D;
    const gl = new OffscreenCanvas(1, 1).getContext("webgl2");
    return gl.getParameter(gl.MAX_TEXTURE_SIZE);
  }, webgl2);
  reply({
    ready: true,
    browser: android
      ? await page.evaluate(() => navigator.userAgent)
      : browser.version(),
    max_size: maxSize,
  });
  for await (const line of createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  })) {
    const request = JSON.parse(line);
    try {
      // A watchdog for device loss or a readback that never completes.
      const result = await Promise.race([
        page.evaluate((r) => window.fixture(r), request),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("render/readback timeout")),
            request.timeout_ms + 10_000,
          ),
        ),
      ]);
      if (glsl?.dump) {
        const dir = resolve(glsl.dump, String(dumped++).padStart(4, "0"));
        await mkdir(dir, { recursive: true });
        const sources = await page.evaluate(() =>
          globalThis.__glsl.splice(0).map(({ stage, text, translated }) => ({
            stage,
            text,
            translated,
          })),
        );
        for (const [i, { stage, text, translated }] of sources.entries()) {
          await writeFile(resolve(dir, `${i}.${stage}.glsl`), text);
          if (translated)
            await writeFile(
              resolve(dir, `${i}.${stage}.angle.glsl`),
              translated,
            );
        }
      }
      if (request.query) {
        reply({ features: result });
      } else {
        await writeFile(request.output, Buffer.from(result.rgba, "base64"));
        delete result.rgba;
        reply(result);
      }
    } catch (error) {
      // A new page discards a stuck map before the next request.
      await open();
      reply({ error: String(error) });
    }
  }
} catch (error) {
  reply({ error: String(error) });
  process.exitCode = 1;
} finally {
  await close();
  server.close();
}
