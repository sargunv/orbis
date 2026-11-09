import assert from "node:assert/strict";
import { test } from "node:test";

const { preloadStyle, takePreload } = await import("./preload.ts");
const { setRequestTransform } = await import("./requests.ts");

const fetched: string[] = [];
globalThis.fetch = (async (url: string) => {
  fetched.push(url);
  const style = { sources: { a: { url: "tiles.json" }, b: { url: "https://b.test/tj" }, c: { type: "geojson" } }, sprite: "sprites/s?k=1" };
  return new Response(url === "https://a.test/style.json" ? JSON.stringify(style) : "{}");
}) as typeof fetch;

test("a style's URL, TileJSON and sprite are fetched ahead, as the map asks for them, once each", async () => {
  preloadStyle("https://a.test/style.json", 2);
  assert.ok(await takePreload("https://a.test/style.json"));
  assert.equal(takePreload("https://a.test/style.json"), undefined, "taken once");
  await new Promise((resolve) => setTimeout(resolve, 10));
  for (const url of ["https://a.test/tiles.json", "https://b.test/tj", "https://a.test/sprites/s@2x.json?k=1", "https://a.test/sprites/s@2x.png?k=1"]) {
    assert.ok(takePreload(url), url);
  }
});

test("the page's request transform applies, and a refused request isn't fetched", async () => {
  fetched.length = 0;
  setRequestTransform((url) => (url.includes("refused") ? null : { url: `${url}?signed` }));
  preloadStyle({ version: 8, sources: { a: { url: "https://c.test/refused" }, b: { url: "https://c.test/ok" } }, sprite: [] }, 1);
  await assert.rejects(takePreload("https://c.test/refused")!);
  assert.ok(await takePreload("https://c.test/ok"));
  assert.deepEqual(fetched, ["https://c.test/ok?signed"]);
  setRequestTransform(undefined);
});

test("the first glyph range of each constant font stack shown at the camera's zoom is fetched ahead", async () => {
  fetched.length = 0;
  const label = (layout: object, extra: object = {}) =>
    ({ type: "symbol", layout: { "text-field": "{name}", ...layout }, ...extra });
  preloadStyle({
    version: 8, glyphs: "https://g.test/{fontstack}/{range}.pbf", sources: {},
    layers: [
      label({ "text-font": ["A", "B"] }),
      label({}),
      label({ "text-font": ["literal", ["C"]] }),
      label({ "text-font": ["step", ["zoom"], ["literal", ["D"]], 8, ["literal", ["E"]]] }),
      label({ "text-font": ["F"] }, { minzoom: 10 }),
      label({ "text-font": ["G"], visibility: "none" }),
      { type: "line", layout: {} },
    ],
  }, 1, 5);
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.deepEqual(fetched.sort(), [
    "https://g.test/A,B/0-255.pbf", "https://g.test/C/0-255.pbf",
    "https://g.test/Open Sans Regular,Arial Unicode MS Regular/0-255.pbf",
  ]);
  for (const url of fetched) assert.ok(takePreload(url), url);
});
