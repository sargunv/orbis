import { execFileSync } from "node:child_process";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const demo = fileURLToPath(new URL("./demo/", import.meta.url));
const base = "/orbis/";

// Replaces %ORBIS_COMMIT% in the pages with the build's commit, linked on
// GitHub, so it's clear which version a browser loaded.
function commit() {
  const git = (...args) => execFileSync("git", args, { cwd: demo, encoding: "utf8" }).trim();
  const sha = git("rev-parse", "HEAD");
  const dirty = git("status", "--porcelain") ? "-dirty" : "";
  const link = `<a href="https://github.com/sargunv/orbis/commit/${sha}">${git("rev-parse", "--short", sha)}${dirty}</a>`;
  return { name: "commit", transformIndexHtml: (html) => html.replaceAll("%ORBIS_COMMIT%", link) };
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
        if (!source.url?.startsWith("pmtiles://./data/")) continue;
        const archive = resolve(dirname(path), "..", source.url.slice("pmtiles://".length));
        this.addWatchFile(archive);
        const ref = this.emitFile({ type: "asset", name: basename(archive), source: await readFile(archive) });
        // Fetch runs in both the page and workers under assets/, so use a
        // URL rooted at the deployment base rather than the current script.
        source.url = `pmtiles://${base}${this.getFileName(ref)}`;
      }
      for (const key of ["glyphs", "sprite"]) {
        if (!style[key]?.startsWith("./data/")) continue;
        style[key] = `${base}${style[key].slice("./".length)}`;
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
  plugins: [extractStyles(), commit(), {
    name: "playground-native",
    configureServer(server) {
      const path = process.env.ORB_PLAYGROUND_STYLE;
      let writes = Promise.resolve();
      server.middlewares.use("/orbis/__playground/style", async (req, res) => {
        if (req.method !== "PUT") { res.statusCode = 405; res.end(); return; }
        if (req.headers.origin !== `http://${req.headers.host}`) { res.statusCode = 403; res.end(); return; }
        try {
          let body = "";
          for await (const chunk of req) {
            body += chunk;
            if (body.length > 4 * 1024 * 1024) throw new Error("style exceeds 4 MiB");
          }
          JSON.parse(body);
          if (path) {
            writes = writes.catch(() => {}).then(() => writeFile(path,body));
            await writes;
          }
          res.setHeader("Content-Type","application/json");
          res.end(JSON.stringify({linked:!!path}));
        } catch (e) { res.statusCode = 400; res.end(String(e)); }
      });
    },
  }],
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    assetsInlineLimit: 0,
    rolldownOptions: {
      input: Object.fromEntries(["index", "probe", "bench", "playground", "cities"].map((name) => [name, `${demo}${name}.html`])),
    },
  },
  worker: { format: "es" },
});
