// The style editor's native viewer bridge, smoke-tested through native
// offscreen rendering.
// Build orb-demo --release; ORB_PLAYGROUND_STYLE is a mktemp file.
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { launchChromium } from "./serve.mjs";
if (!process.env.ORB_PLAYGROUND_STYLE) throw new Error("set ORB_PLAYGROUND_STYLE to a mktemp file");
const root = fileURLToPath(new URL("../../",import.meta.url));
const server = await createServer({configFile:`${root}web/vite.config.js`,server:{host:"127.0.0.1",port:0}});
await server.listen();
const port = server.httpServer.address().port;
const render = async () => {
  const {stderr} = await promisify(execFile)(`${root}target/release/orb-demo`,[
    new URL("../demo/data/playground.json",import.meta.url).href,
    "--watch-style",process.env.ORB_PLAYGROUND_STYLE,"--snapshot","/dev/null",
  ],{cwd:root,timeout:60000});
  assert.match(stderr,/visible in 1 frames/);
  return stderr.split("\n").find(line => line.startsWith("playground:"));
};
const browser = await launchChromium();
try {
  const page = await browser.newPage({viewport:{width:1280,height:800}});
  await page.goto(`http://127.0.0.1:${port}/orbis/?style=playground&edit&thread=main`);
  await page.waitForFunction(() => window.app?.editor.style && document.querySelector("#app").dataset.map === "idle",
    null,{timeout:60000});
  const sent = () => page.locator(".native").evaluate(e => Number(e.dataset.sent ?? 0));
  let last = await sent();
  await page.locator(".layer-name", {hasText:"paper"}).click();
  await page.locator(".layer.open .prop-control input.field").first().fill("#aabbcc");
  await page.locator(".layer.open .prop-control input.field").first().press("Enter");
  await page.waitForFunction(() => document.querySelector(".applied").textContent.startsWith("Applied in"));
  await page.waitForFunction(([l]) => Number(document.querySelector(".native").dataset.sent ?? 0) > l
    && app.editor.style.layers[0].paint["background-color"] === "#aabbcc", [last]);
  await page.waitForFunction(() => document.querySelector(".native").textContent.startsWith("Native viewer linked"));
  // The bridge debounces; wait for the write that carries the edit.
  const read = async () => JSON.parse(await readFile(process.env.ORB_PLAYGROUND_STYLE,"utf8"));
  let commands = await read();
  for (let i = 0; commands[0][1].layers[0].paint["background-color"] !== "#aabbcc" && i < 50; i++) {
    await page.waitForTimeout(100);
    commands = await read();
  }
  assert.equal(commands[0][1].layers[0].paint["background-color"],"#aabbcc");
  console.log(`native paint: ${await render()}`);
  const bounds=await page.locator("#map").boundingBox();
  last = await sent();
  await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
  await page.waitForFunction(() => document.querySelector(".inspect").textContent === "stations #31");
  await page.waitForFunction(([l]) => Number(document.querySelector(".native").dataset.sent ?? 0) > l, [last]);
  commands = await read();
  for (let i = 0; commands.at(-1)[0] !== "setFeatureState" && i < 50; i++) {
    await page.waitForTimeout(100);
    commands = await read();
  }
  assert.equal(commands.at(-1)[0],"setFeatureState");
  assert.equal(commands.at(-1)[1].id,"31");
  console.log(`native hover: ${await render()}`);
} finally { await browser.close(); await server.close(); }
