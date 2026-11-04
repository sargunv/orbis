// Local development bridge smoke test through native offscreen rendering.
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
  await page.goto(`http://127.0.0.1:${port}/orbis/playground.html?thread=main`);
  await page.waitForFunction(() => window.playground);
  const style = JSON.parse(await page.locator("#style").inputValue());
  style.layers[0].paint["background-color"] = "#aabbcc";
  await page.locator("#style").fill(JSON.stringify(style));
  await page.waitForFunction(() => document.querySelector("#edit-status").textContent.startsWith("Applied in"));
  await page.waitForFunction(() => document.querySelector("#native").textContent.startsWith("Style and hover sent"));
  let commands = JSON.parse(await readFile(process.env.ORB_PLAYGROUND_STYLE,"utf8"));
  assert.equal(commands[0][1].layers[0].paint["background-color"],"#aabbcc");
  console.log(`native paint: ${await render()}`);
  const bounds=await page.locator("#map").boundingBox();
  await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
  await page.waitForFunction(() => document.querySelector("#hover").textContent === "stations / 31");
  // The file write completes before the hover label updates.
  commands = JSON.parse(await readFile(process.env.ORB_PLAYGROUND_STYLE,"utf8"));
  assert.equal(commands.at(-1)[0],"setFeatureState");
  assert.equal(commands.at(-1)[1].id,"31");
  console.log(`native hover: ${await render()}`);
} finally { await browser.close(); await server.close(); }
