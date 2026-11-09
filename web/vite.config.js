import { execFileSync } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const demo = fileURLToPath(new URL("./demo/", import.meta.url));
const web = fileURLToPath(new URL("./", import.meta.url));
const base = "/orbis/";

// The build's commit, for the app's About sheet and debug overlay, so it's
// clear which version a browser loaded.
function commit() {
  const git = (...args) => execFileSync("git", args, { cwd: demo, encoding: "utf8" }).trim();
  const sha = git("rev-parse", "HEAD");
  return { sha, short: git("rev-parse", "--short", sha), dirty: git("status", "--porcelain") !== "" };
}

// `virtual:paint-spec`: each layer type's paint properties from the v8
// spec (the style editor's controls), as [name, type, default, minimum,
// maximum, units, enum values].
function paintSpec() {
  const id = "virtual:paint-spec";
  return {
    name: "paint-spec",
    resolveId: (source) => (source === id ? `\0${id}` : undefined),
    async load(source) {
      if (source !== `\0${id}`) return;
      const require = createRequire(import.meta.url);
      const spec = require("@maplibre/maplibre-gl-style-spec/dist/latest.json");
      const types = {};
      for (const [key, props] of Object.entries(spec)) {
        if (!key.startsWith("paint_")) continue;
        types[key.slice("paint_".length)] = Object.entries(props).map(([name, p]) =>
          [name, p.type, p.default ?? null, p.minimum ?? null, p.maximum ?? null, p.units ?? null,
            p.type === "enum" ? Object.keys(p.values) : null]);
      }
      return `export default ${JSON.stringify(types)};`;
    },
  };
}

// Vite handles JS, workers and wasm. JSON's pmtiles:// strings aren't JS
// imports: emit the archives first and include their hashed URLs in each
// emitted style, so a change to either also changes the style's hash.
//
// Glyphs and sprites are fetched by template (`{fontstack}/{range}`), so
// they keep their names under data/.
function extractStyles() {
  let data;
  const emitData = async (ctx) => {
    for (const dir of ["glyphs", "sprites"]) {
      const root = resolve(demo, "data", dir);
      for (const entry of await readdir(root, { recursive: true, withFileTypes: true })) {
        if (!entry.isFile()) continue;
        const path = resolve(entry.parentPath, entry.name);
        ctx.emitFile({ type: "asset", fileName: `data/${path.slice(resolve(demo, "data").length + 1)}`, source: await readFile(path) });
      }
    }
  };
  return {
    name: "extract-styles",
    apply: "build",
    enforce: "pre",
    buildStart() { data = undefined; },
    async load(id) {
      if (!/\/data\/tokyo-(mvt|mlt)\.json\?url$/.test(id)) return;
      const path = id.slice(0, -"?url".length);
      this.addWatchFile(path);
      const style = JSON.parse(await readFile(path, "utf8"));
      for (const source of Object.values(style.sources)) {
        if (!source.url?.startsWith("pmtiles://./")) continue;
        const archive = resolve(dirname(path), source.url.slice("pmtiles://".length));
        this.addWatchFile(archive);
        const ref = this.emitFile({ type: "asset", name: basename(archive), source: await readFile(archive) });
        // Fetch runs in both the page and workers under assets/, so use a
        // URL rooted at the deployment base rather than the current script.
        source.url = `pmtiles://${base}${this.getFileName(ref)}`;
      }
      for (const key of ["glyphs", "sprite"]) {
        if (!style[key]?.startsWith("./")) continue;
        style[key] = `${base}data/${style[key].slice("./".length)}`;
        data ??= emitData(this);
      }
      await data;
      const ref = this.emitFile({ type: "asset", name: basename(path), source: JSON.stringify(style) });
      return `export default import.meta.ROLLUP_FILE_URL_${ref};`;
    },
  };
}

export default defineConfig({
  root: demo,
  base,
  publicDir: false,
  // The demo consumes the package by name.
  resolve: { alias: { orbis: fileURLToPath(new URL("./src/index.ts", import.meta.url)) } },
  server: { fs: { allow: [web] } },
  define: { __ORBIS_COMMIT__: JSON.stringify(commit()) },
  plugins: [extractStyles(), paintSpec()],
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    rolldownOptions: {
      input: Object.fromEntries(["index", "bench", "fixture", "load", "measure", "scenario"].map((name) => [name, `${demo}${name}.html`])),
    },
  },
  worker: { format: "es" },
});
