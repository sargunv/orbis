// Integration checks for the editor, worker query bridge and hover state.
import assert from "node:assert/strict";
import { launchChromium, serveDemo } from "./serve.mjs";
const server = await serveDemo({isolated:true});
const browser = await launchChromium();
try {
  for (const thread of ["main","worker"]) {
    const page = await browser.newPage({viewport:{width:1280,height:800}});
    const errors = [];
    page.on("pageerror",e => errors.push(e.message));
    page.on("console",m => { if (["error","warning"].includes(m.type())) errors.push(m.text()); });
    await page.goto(`${server.url}playground.html?thread=${thread}`);
    await page.waitForFunction(() => window.playground, null,{timeout:60000});
    const hit = await page.evaluate(async () => {
      const canvas = document.querySelector("#map");
      const result = orbis.hitTest(canvas.clientWidth/2,canvas.clientHeight/2);
      if (result instanceof Promise) throw new Error("hitTest must be synchronous");
      return result;
    });
    assert.equal(hit.layer,"stations");
    assert.equal(hit.id,"31");
    assert.equal(hit.source,"district");
    assert.equal(hit.sourceLayer,null);
    const bounds = await page.locator("#map").boundingBox();
    await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
    await page.waitForFunction(() => document.querySelector("#hover").textContent === "stations / 31");
    const before = await page.locator("#map").screenshot();
    await page.mouse.move(10,10);
    await page.waitForFunction(() => document.querySelector("#hover").textContent.startsWith("Hover"));
    await page.waitForTimeout(50);
    assert.notDeepEqual(await page.locator("#map").screenshot(),before,"hover changes pixels");
    const original = await page.locator("#style").inputValue();
    await page.locator("#style").fill("{broken");
    await page.waitForFunction(() => document.querySelector("#error").textContent.length > 0);
    assert.equal(await page.evaluate(() => JSON.parse(orbis.map?.getStyle?.() ?? "{}").version ?? 8),8);
    const edited = JSON.parse(original);
    edited.layers[0].paint["background-color"] = "#ccddee";
    await page.locator("#style").fill(JSON.stringify(edited,null,2));
    await page.waitForFunction(() => document.querySelector("#edit-status").textContent.startsWith("Applied in"));
    assert.equal(await page.locator("#error").textContent(),"");
    console.log(`${thread}: ${await page.locator("#edit-status").textContent()}; topmost station 31, hover pixel change and validation recovery passed`);
    // Measure actual frame acknowledgements, rather than elapsed wall time.
    const measurements = await page.evaluate(async () => {
      const apply = async (name,commands) => {
        const frames = [];
        const before = playground.latestFrame;
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
    console.log(`${thread}: ${JSON.stringify(measurements)}`);
    assert.equal(measurements[0].frames,1);
    assert.equal(measurements[1].meshUploads,0);
    await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
    await page.waitForFunction(() => document.querySelector("#hover").textContent === "stations / 31");
    await page.locator("#style").fill(JSON.stringify({version:8,sources:{},layers:[edited.layers[0]]}));
    await page.waitForFunction(() => playground.latestFrame.draws < 5 && playground.latestFrame.idle);
    await page.mouse.move(bounds.x+bounds.width/2+1,bounds.y+bounds.height/2);
    await page.waitForFunction(() => document.querySelector("#hover").textContent.startsWith("Hover"));
    assert.deepEqual(errors,[]);
    await page.close();
  }
} finally { await browser.close(); server.close(); }
