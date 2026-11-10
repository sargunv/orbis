// What the head-to-head page scripts (load.mjs, memory.mjs) load: their
// cameras, the h2h cache, mln-rs's web build and MapLibre Native's
// real-world render tests served offline.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
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

export const MLNRS = {
  name: "@bartlwrs/mln-rs@0.1.0",
  tarball: "https://registry.npmjs.org/@bartlwrs/mln-rs/-/mln-rs-0.1.0.tgz",
  integrity:
    "sha512-6rLjFbfTOpqYJz7HC4r4Ci3IrF34WuRjFZxHRW4+2vyLatA6VHMGcVYw3bV8SffvfH6+zplemkOCudP380ydAA==",
};

/** mln-rs's web build, fetched by version, checked against its npm integrity and unpacked into the cache: its `dist/` directory. */
export async function mlnrs() {
  const dir = join(cacheDir(), "mln-rs-web-0.1.0");
  const dist = join(dir, "package/dist/");
  if (existsSync(join(dist, "index.js"))) return dist;
  const bytes = Buffer.from(await (await fetch(MLNRS.tarball)).arrayBuffer());
  const integrity =
    "sha512-" + createHash("sha512").update(bytes).digest("base64");
  if (integrity !== MLNRS.integrity)
    throw new Error(
      `${MLNRS.tarball}: integrity ${integrity}, not ${MLNRS.integrity}`,
    );
  const partial = `${dir}.partial-${process.pid}`;
  rmSync(partial, { recursive: true, force: true });
  mkdirSync(partial, { recursive: true });
  writeFileSync(join(partial, "package.tgz"), bytes);
  execFileSync("tar", ["-xzf", join(partial, "package.tgz"), "-C", partial]);
  rmSync(dir, { recursive: true, force: true });
  execFileSync("mv", [partial, dir]);
  return dist;
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
