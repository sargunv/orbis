// What the head-to-head page scripts (load.mjs, memory.mjs, gesture-h2h.mjs)
// load: their cameras, the h2h cache, mln-rs's browser page and MapLibre
// Native's real-world render tests served offline.
import { execFileSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";

/** Cameras as lon,lat,zoom,bearing,pitch. */
export const VIEWS = {
  // Cameras of xtask/h2h/views.txt.
  liberty: {
    ams13: "4.89,52.37,13,0,0",
    paris14p60: "2.35,48.857,14,90,60",
    world2: "10,30,1.5,0,0",
    tokyo16: "139.70,35.69,16,0,0",
  },
  // MapLibre Native's real-world render tests (their `setCenter` and `setZoom`).
  realWorld: {
    chicago: "-87.6942445,41.8703965,13,0,0",
    nepal: "85.48805250000001,28.115547,13,0,0",
    norway: "10.8181,64.863174,12,0,0",
    sanfrancisco: "-122.448635,37.7669995,15,0,0",
    uruguay: "-56.509552,-32.865788,9,0,0",
  },
  // The Alps terrain style (`alps.json`): its own view, the Matterhorn close up, the Bernese Alps from afar.
  terrain: {
    alps11p72: "7.7491,46.0207,11.6,-32,72",
    matterhorn13p60: "7.6586,45.9763,13,60,60",
    oberland9p45: "8.0,46.55,9,0,45",
  },
  // The committed Tokyo extract (`--style=tokyo`).
  tokyo: { tokyo15: "139.74,35.685,15,0,0" },
  // Liberty on the globe.
  globe: {
    world1: "10,30,1.5,0,0",
    europe3p40: "10,48,3,20,40",
    japan5p30: "138,36,5,-20,30",
  },
};

/** The h2h cache, as `cargo xtask h2h` finds it. */
function cacheDir() {
  if (process.env.ORB_H2H_CACHE) return process.env.ORB_H2H_CACHE;
  if (process.platform === "darwin")
    return join(homedir(), "Library/Caches/orbis-h2h");
  return join(
    process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"),
    "orbis-h2h",
  );
}

const root = fileURLToPath(new URL("../../", import.meta.url));
/** The mln-rs commit `xtask/h2h/Cargo.toml` pins, which `cargo xtask h2h` builds natively. */
const MLNRS_REV = readFileSync(
  join(root, "xtask/h2h/Cargo.toml"),
  "utf8",
).match(/^mln-rs = .*rev = "([0-9a-f]{40})"/m)?.[1];
if (!MLNRS_REV) throw new Error("xtask/h2h/Cargo.toml pins no mln-rs rev");

export const MLNRS = { name: `maplibre/mln-rs@${MLNRS_REV.slice(0, 8)}` };

/**
 * `MISE_IGNORED_CONFIG_PATHS` with Cargo's git checkouts added, as
 * `cargo xtask h2h` sets it (`mise_ignored`): mln-rs's checkout has a
 * mise.toml, which mise won't run a tool under (cargo, rustc through
 * sccache) until it is trusted.
 */
function miseIgnored() {
  const checkouts = join(
    process.env.CARGO_HOME ?? join(homedir(), ".cargo"),
    "git/checkouts",
  );
  return [
    ...(process.env.MISE_IGNORED_CONFIG_PATHS ?? "")
      .split(delimiter)
      .filter(Boolean),
    checkouts,
  ].join(delimiter);
}

/**
 * Cargo's checkout of the pinned mln-rs, the one `cargo xtask h2h` builds:
 * fetched if need be, or with `offline`, undefined where Cargo hasn't
 * fetched it.
 */
function mlnrsCheckout({ offline = false } = {}) {
  let metadata;
  try {
    metadata = JSON.parse(
      execFileSync(
        "cargo",
        [
          "metadata",
          "--format-version=1",
          "--locked",
          ...(offline ? ["--offline"] : []),
          "--manifest-path",
          join(root, "xtask/h2h/Cargo.toml"),
        ],
        {
          maxBuffer: 1 << 28,
          stdio: ["ignore", "pipe", offline ? "ignore" : "inherit"],
        },
      ),
    );
  } catch (e) {
    if (offline) return undefined;
    throw e;
  }
  const manifest = metadata.packages.find(
    (p) => p.name === "mln-rs" && p.source?.includes(MLNRS_REV),
  )?.manifest_path;
  if (!manifest) throw new Error(`cargo metadata found no ${MLNRS.name}`);
  return dirname(manifest);
}

/** The real-world render tests' cache in the pinned mln-rs checkout, if Cargo has fetched it. */
export function realWorldCache() {
  const checkout = mlnrsCheckout({ offline: true });
  const db =
    checkout &&
    join(checkout, "tools/render-test/testdata/metrics/cache-style.db");
  return db && existsSync(db) ? db : undefined;
}

/**
 * mln-rs's browser page (its examples/web) at the h2h pin, built in the
 * cache as its `mise run web:build --profile release` builds it: the map's
 * module for WebGPU and again for WebGL 2, and the font-face shaper's, each
 * through wasm-bindgen and the pin's own wasm-opt flags, then the page
 * bundled by Vite. Scripts open its index.html and drive it through
 * `window.mln` (its web/main.ts). Built once per pin, with the wasm-bindgen
 * and wasm-opt mise provides.
 */
export function mlnrs() {
  const dir = join(cacheDir(), `mln-rs-web-${MLNRS_REV.slice(0, 8)}`);
  if (!existsSync(join(dir, "index.html"))) buildMlnrs(dir);
  return dir + "/";
}

function buildMlnrs(dir) {
  const checkout = mlnrsCheckout();
  const bindgen = readFileSync(join(checkout, "Cargo.lock"), "utf8").match(
    /name = "wasm-bindgen"\nversion = "([^"]+)"/,
  )?.[1];
  const have = execFileSync("wasm-bindgen", ["--version"], {
    encoding: "utf8",
  });
  if (!have.includes(` ${bindgen}`))
    throw new Error(
      `${MLNRS.name} needs wasm-bindgen ${bindgen}, not ${have.trim()}: pin that version in mise.toml`,
    );
  const target = join(cacheDir(), "target");
  const partial = `${dir}.partial-${process.pid}`;
  rmSync(partial, { recursive: true, force: true });
  try {
    // The page's sources, with the style catalogue its styles.ts imports
    // beside them, copied so that wasm-bindgen's output, the packages and the
    // build land here and Cargo's checkout stays as it fetched it.
    const page = join(partial, "web");
    cpSync(join(checkout, "examples/web"), page, { recursive: true });
    cpSync(
      join(checkout, "examples/styles.json"),
      join(partial, "styles.json"),
    );
    const pkg = join(page, "web/pkg");
    // Tools' output on stderr: load.mjs's stdout is its report.
    const run = (command, args, { cwd = checkout, env = {} } = {}) =>
      execFileSync(command, args, {
        cwd,
        env: {
          ...process.env,
          MISE_IGNORED_CONFIG_PATHS: miseIgnored(),
          ...env,
        },
        stdio: ["ignore", 2, "inherit"],
      });
    // From the checkout's directory, for its own toolchain and Cargo config;
    // the `s` opt-level is web:build's, for the browser build only.
    const cargo = (...packages) =>
      run(
        "cargo",
        [
          "build",
          "--release",
          "--locked",
          "--target=wasm32-unknown-unknown",
          "--target-dir",
          target,
          ...packages,
        ],
        { env: { CARGO_PROFILE_RELEASE_OPT_LEVEL: "s" } },
      );
    const bind = (module, out) => {
      run("wasm-bindgen", [
        "--target=web",
        "--out-dir",
        out,
        join(target, `wasm32-unknown-unknown/release/${module}.wasm`),
      ]);
      run("bash", [
        "-c",
        'source tools/web-wasm-opt.sh && web_wasm_opt "$1" --strip-debug',
        "web_wasm_opt",
        join(out, `${module}_bg.wasm`),
      ]);
    };
    console.error(`mlnrs: building ${MLNRS.name}'s browser page`);
    cargo("-p=mln-rs-web", "-p=mln-rs-web-shaper");
    bind("mln_web", pkg);
    bind("mln_web_shaper", pkg);
    // The map's module again, drawing with WebGL 2, which the page bundles too.
    cargo("-p=mln-rs-web", "--no-default-features", "--features=webgl");
    bind("mln_web", join(pkg, "webgl"));
    run("pnpm", ["install", "--frozen-lockfile"], { cwd: page });
    run("pnpm", ["build"], { cwd: page });
    rmSync(dir, { recursive: true, force: true });
    renameSync(join(page, "dist"), dir);
  } finally {
    rmSync(partial, { recursive: true, force: true });
  }
}

/**
 * The cache's real-world styles, their tiles, sprites and glyphs as files
 * under `dir`, the styles' `local://` and `mapbox://` URLs pointing at `base`.
 */
export function extractCache(db, dir, base) {
  rmSync(dir, { recursive: true, force: true });
  const cached = new DatabaseSync(db, { readOnly: true });
  const bytes = (data, compressed) =>
    compressed ? inflateSync(data) : Buffer.from(data);
  const write = (path, data) => {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), data);
  };
  for (const { url, data, compressed } of cached
    .prepare("SELECT url, data, compressed FROM resources")
    .all()) {
    const style = url.match(/^http:\/\/styles\/(.*\.json)$/)?.[1];
    const asset = url.match(/^mapbox:\/\/((?:sprites|fonts)\/.*)$/)?.[1];
    if (style) {
      const doc = bytes(data, compressed)
        .toString()
        .replaceAll("local://", base)
        .replaceAll("mapbox://", base);
      write(`styles/${style}`, doc);
    } else if (asset) {
      write(decodeURIComponent(asset), bytes(data, compressed));
    }
  }
  const tiles = cached.prepare(
    "SELECT url_template, z, x, y, data, compressed FROM tiles WHERE url_template LIKE 'http://mvt-fixtures/%'",
  );
  for (const { url_template, z, x, y, data, compressed } of tiles.all()) {
    const path = url_template
      .slice("http://".length)
      .replace("{z}", z)
      .replace("{x}", x)
      .replace("{y}", y);
    write(path, bytes(data, compressed));
  }
  cached.close();
}

export const median = (v) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor((s.length - 1) / 2)] : NaN;
};
