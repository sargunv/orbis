// What the head-to-head page scripts (load.mjs, memory.mjs) load: their
// cameras, the h2h cache, mln-rs's browser build and MapLibre Native's
// real-world render tests served offline.
import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";

/** Cameras as lon,lat,zoom,bearing,pitch. */
export const VIEWS = {
  // mln-rs's liberty-compare cameras.
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
export function cacheDir() {
  if (process.env.ORB_H2H_CACHE) return process.env.ORB_H2H_CACHE;
  if (process.platform === "darwin")
    return join(homedir(), "Library/Caches/orbis-h2h");
  return join(
    process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"),
    "orbis-h2h",
  );
}

/** The real-world render tests' cache in the pinned mln-rs checkout that `cargo xtask h2h` builds, if there is one. */
export function realWorldCache() {
  const checkouts = join(
    process.env.CARGO_HOME ?? join(homedir(), ".cargo"),
    "git/checkouts",
  );
  if (!existsSync(checkouts)) return undefined;
  for (const repo of readdirSync(checkouts).filter((d) =>
    d.startsWith("mln-rs-"),
  )) {
    for (const rev of readdirSync(join(checkouts, repo))) {
      const db = join(
        checkouts,
        repo,
        rev,
        "tools/render-test/testdata/metrics/cache-style.db",
      );
      if (existsSync(db)) return db;
    }
  }
  return undefined;
}

const root = fileURLToPath(new URL("../../", import.meta.url));
/** The mln-rs commit `xtask/h2h/Cargo.toml` pins, which `cargo xtask h2h` builds natively. */
const MLNRS_REV = readFileSync(
  join(root, "xtask/h2h/Cargo.toml"),
  "utf8",
).match(/^mln-rs = .*rev = "([0-9a-f]{40})"/m)?.[1];
if (!MLNRS_REV) throw new Error("xtask/h2h/Cargo.toml pins no mln-rs rev");

export const MLNRS = { name: `louwers/mln-rs@${MLNRS_REV.slice(0, 8)}` };

/**
 * mln-rs's browser build (its examples/web) at the h2h pin, as its
 * `just profile=release web` builds it, in the cache: wasm-bindgen's
 * package, the map's and the tile workers' Web Workers, and as index.js
 * `mlnrs-map.js`, the `Map` demo/load.js drives. Built once per pin, with
 * the wasm-bindgen and wasm-opt mise provides.
 */
export function mlnrs() {
  const dir = join(cacheDir(), `mln-rs-web-${MLNRS_REV.slice(0, 8)}`);
  if (!existsSync(join(dir, "pkg/mln_web_bg.wasm"))) buildMlnrs(dir);
  copyFileSync(
    fileURLToPath(new URL("./mlnrs-map.js", import.meta.url)),
    join(dir, "index.js"),
  );
  return dir + "/";
}

function buildMlnrs(dir) {
  // Cargo's checkout of the pin, fetched if need be.
  const metadata = JSON.parse(
    execFileSync(
      "cargo",
      [
        "metadata",
        "--format-version=1",
        "--locked",
        "--manifest-path",
        join(root, "xtask/h2h/Cargo.toml"),
      ],
      { maxBuffer: 1 << 28, stdio: ["ignore", "pipe", "inherit"] },
    ),
  );
  const checkout = dirname(
    metadata.packages.find(
      (p) => p.name === "mln-rs" && p.source?.includes(MLNRS_REV),
    ).manifest_path,
  );
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
  // Cargo's output on stderr: load.mjs's stdout is its report.
  const run = (command, args, env = {}) =>
    execFileSync(command, args, {
      cwd: checkout,
      env: { ...process.env, ...env },
      stdio: ["ignore", 2, "inherit"],
    });
  console.error(`mlnrs: building ${MLNRS.name}'s browser build`);
  // The checkout's own toolchain and Cargo config, from its directory; the
  // `s` opt-level is `just web`'s, for the browser build only.
  run(
    "cargo",
    [
      "build",
      "--release",
      "--locked",
      "--target=wasm32-unknown-unknown",
      "-p=mln-rs-web",
      "-p=mln-rs-web-shaper",
      "--target-dir",
      target,
    ],
    { CARGO_PROFILE_RELEASE_OPT_LEVEL: "s" },
  );
  const partial = `${dir}.partial-${process.pid}`;
  rmSync(partial, { recursive: true, force: true });
  for (const module of ["mln_web", "mln_web_shaper"]) {
    run("wasm-bindgen", [
      "--target=web",
      "--out-dir",
      join(partial, "pkg"),
      join(target, `wasm32-unknown-unknown/release/${module}.wasm`),
    ]);
    const wasm = join(partial, `pkg/${module}_bg.wasm`);
    // `just web`'s flags.
    run("wasm-opt", [
      "-Oz",
      "--converge",
      "--one-caller-inline-max-function-size=20",
      "--zero-filled-memory",
      "--low-memory-unused",
      "--strip-debug",
      "--strip-producers",
      "--strip-target-features",
      wasm,
      "-o",
      wasm,
    ]);
  }
  for (const file of ["map-worker.js", "worker.js"])
    copyFileSync(join(checkout, "examples/web/web", file), join(partial, file));
  rmSync(dir, { recursive: true, force: true });
  renameSync(partial, dir);
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
