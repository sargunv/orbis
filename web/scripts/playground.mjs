// Integration checks for the map app's style editor, the worker query
// bridge and hover state, on the editor sample (`?style=playground`).
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
    await page.goto(`${server.url}?style=playground&edit&thread=${thread}`);
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
    console.log(`${thread}: ${await page.locator(".applied").textContent()}; topmost station 31, hover pixel change, controls and validation recovery passed`);

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
    console.log(`${thread}: ${JSON.stringify(measurements)}`);
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
} finally { await browser.close(); server.close(); }
