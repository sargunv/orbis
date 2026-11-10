import { execFileSync } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { basename, dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const demo = fileURLToPath(new URL("./demo/", import.meta.url));
const web = fileURLToPath(new URL("./", import.meta.url));
const base = "/orbis/";

// The build's commit, for the app's About sheet and debug overlay, so it's
// clear which version a browser loaded.
function commit() {
  const git = (...args) =>
    execFileSync("git", args, { cwd: demo, encoding: "utf8" }).trim();
  const sha = git("rev-parse", "HEAD");
  return {
    sha,
    short: git("rev-parse", "--short", sha),
    dirty: git("status", "--porcelain") !== "",
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
      for (const entry of await readdir(root, {
        recursive: true,
        withFileTypes: true,
      })) {
        if (!entry.isFile()) continue;
        const path = resolve(entry.parentPath, entry.name);
        ctx.emitFile({
          type: "asset",
          fileName: `data/${path.slice(resolve(demo, "data").length + 1)}`,
          source: await readFile(path),
        });
      }
    }
  };
  return {
    name: "extract-styles",
    apply: "build",
    enforce: "pre",
    buildStart() {
      data = undefined;
    },
    async load(id) {
      if (!/\/data\/tokyo-(mvt|mlt)\.json\?url$/.test(id)) return;
      const path = id.slice(0, -"?url".length);
      this.addWatchFile(path);
      const style = JSON.parse(await readFile(path, "utf8"));
      for (const source of Object.values(style.sources)) {
        if (!source.url?.startsWith("pmtiles://./")) continue;
        const archive = resolve(
          dirname(path),
          source.url.slice("pmtiles://".length),
        );
        this.addWatchFile(archive);
        const ref = this.emitFile({
          type: "asset",
          name: basename(archive),
          source: await readFile(archive),
        });
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
      const ref = this.emitFile({
        type: "asset",
        name: basename(path),
        source: JSON.stringify(style),
      });
      return `export default import.meta.ROLLUP_FILE_URL_${ref};`;
    },
  };
}

// Files served at fixed paths, outside the hashed assets: the service
// worker that isolates a page for an engine that needs it (app/isolation.js)
// at the base, so its scope covers the demo; and MapLibre Native's browser
// module (app/native.js), whose `.mjs` finds its `.wasm` and starts its
// threads' workers by their own names beside it, so they keep them, under
// the version's directory. `virtual:native-module` exports that module's URL.
function fixedFiles() {
  const id = "virtual:native-module";
  const require = createRequire(import.meta.url);
  const native = dirname(
    require.resolve("@maplibre/native-ffi-opengl-wasm32/package.json"),
  );
  const {
    version,
  } = require("@maplibre/native-ffi-opengl-wasm32/package.json");
  const dir = `native/${version}/webgl/`;
  const files = async () => {
    const out = { "isolate.js": resolve(demo, "isolate.js") };
    for (const name of [
      "maplibre_native_ffi.mjs",
      "maplibre_native_ffi.wasm",
      "LICENSE",
    ])
      out[`${dir}${name}`] = resolve(native, name);
    for (const entry of await readdir(resolve(native, "licenses")))
      out[`${dir}licenses/${entry}`] = resolve(native, "licenses", entry);
    return out;
  };
  const types = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".wasm": "application/wasm",
  };
  return {
    name: "fixed-files",
    resolveId: (source) => (source === id ? `\0${id}` : undefined),
    load(source) {
      if (source === `\0${id}`)
        return `export const version = ${JSON.stringify(version)};\nexport default ${JSON.stringify(`${base}${dir}maplibre_native_ffi.mjs`)};`;
    },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const path = new URL(req.url, "http://x").pathname;
        const file =
          path.startsWith(base) && (await files())[path.slice(base.length)];
        if (!file) {
          next();
          return;
        }
        res.setHeader("content-type", types[extname(file)] ?? "text/plain");
        res.end(await readFile(file));
      });
    },
    async generateBundle() {
      if (this.environment?.name && this.environment.name !== "client") return;
      for (const [fileName, path] of Object.entries(await files())) {
        this.emitFile({
          type: "asset",
          fileName,
          source: await readFile(path),
        });
      }
    },
  };
}

export default defineConfig({
  root: demo,
  base,
  publicDir: false,
  // The demo consumes the package by name.
  resolve: {
    alias: { orbis: fileURLToPath(new URL("./src/index.ts", import.meta.url)) },
  },
  server: { fs: { allow: [web] } },
  define: { __ORBIS_COMMIT__: JSON.stringify(commit()) },
  plugins: [extractStyles(), fixedFiles()],
  // The comparison imports GL JS and MapLibre Native on demand; bundling
  // them up front saves the dev server a reload when one first opens.
  optimizeDeps: { include: ["maplibre-gl", "pmtiles", "@maplibre/native-ffi"] },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    rolldownOptions: {
      input: Object.fromEntries(
        ["index", "bench", "fixture", "load", "measure", "scenario"].map(
          (name) => [name, `${demo}${name}.html`],
        ),
      ),
    },
  },
  worker: { format: "es" },
});
