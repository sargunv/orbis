// The Chrome transport for orb-fixtures: web/demo/fixture.html, which
// draws each fixture through the orbis package. Rust sends one JSON request
// per line; a render's response names premultiplied RGBA bytes written by
// this process, a query's (`query` set) carries its features' JSON.
// Fixture selection, metadata, unpremultiplying and pixel comparison
// belong to Rust.
import { writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { launchChromium, serveDemo } from "./serve.mjs";

const assets = fileURLToPath(new URL("../../fixtures/gl-js/assets/", import.meta.url));
const reply = (value) => process.stdout.write(JSON.stringify(value) + "\n");
const server = await serveDemo({ assets });
const browser = await launchChromium({ unpaced: true });
try {
  const page = await browser.newPage();
  page.on("console", (m) => { if (["error", "warning"].includes(m.type())) console.error(m.text()); });
  page.on("pageerror", (e) => console.error(e));
  const open = () => page.goto(`${server.url}fixture.html`).then(() => page.waitForFunction(() => window.fixture));
  await open();
  const maxSize = await page.evaluate(async () => (await navigator.gpu.requestAdapter()).limits.maxTextureDimension2D);
  reply({ ready: true, browser: browser.version(), max_size: maxSize });
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    const request = JSON.parse(line);
    try {
      // A watchdog for device loss or a readback that never completes.
      const result = await Promise.race([page.evaluate((r) => window.fixture(r), request),
        new Promise((_, reject) => setTimeout(() => reject(new Error("render/readback timeout")), request.timeout_ms + 10_000))]);
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
  await browser.close();
  server.close();
}
