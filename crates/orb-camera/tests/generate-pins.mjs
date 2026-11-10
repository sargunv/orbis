import { createHash } from "node:crypto";
// The 4,000-camera Web Mercator pin: cameras from a
// seeded generator, resolved by the unmodified GL JS transform at the commit
// pinned in xtask/src/fixtures.rs, with screen points it unprojects and, for
// every tenth camera, its tile cover. See README.md for setup.
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
const upstream = resolve(process.argv[2]);
const referenceHashes = {
  "geo/projection/covering_tiles.ts":
    "b606d8e2838a532c95ffff8df36fecd094972e31b24cb2b3b3c43a210dda1f91",
  "geo/projection/mercator_transform.ts":
    "50b88d9fea8938bacd95fa36256644198dab55e47119d69a262bd2f3b3d39c57",
  "geo/projection/mercator_utils.ts":
    "6feeaf70b24be0f88ba2198472eec26bc4a2324f3ebabda7e09c7eee0f3d082f",
  "geo/camera_state.ts":
    "0ad5a7e5e5ad58cc717f25a00c9d1282381beba00abe21d128180433da25b5a8",
};
for (const [path, expected] of Object.entries(referenceHashes)) {
  const actual = createHash("sha256")
    .update(await readFile(`${upstream}/src/${path}`))
    .digest("hex");
  if (actual !== expected)
    throw new Error(`reference source differs from the pinned commit: ${path}`);
}
const from = async (path) =>
  import(pathToFileURL(`${upstream}/src/${path}.ts`));
const { MercatorTransform } = await from("geo/projection/mercator_transform");
const { LngLat } = await from("geo/lng_lat");
const { coveringTiles } = await from("geo/projection/covering_tiles");
const Point = (
  await import(
    pathToFileURL(`${upstream}/node_modules/@mapbox/point-geometry/index.js`)
  )
).default;
const pins = await readFile(
  new URL("../../../xtask/src/fixtures.rs", import.meta.url),
  "utf8",
);
const commit = pins.match(/const COMMIT: &str = "([a-f0-9]{40})";/)[1];

// xorshift32, seeded: the same cameras on every run.
let seed = 0x9e3779b9;
const random = () => {
  seed ^= seed << 13;
  seed >>>= 0;
  seed ^= seed >>> 17;
  seed ^= seed << 5;
  seed >>>= 0;
  return seed / 4294967296;
};
const round = (v, digits) => Number(v.toFixed(digits));
// Rounded so the JSON stays short and GL JS sees exactly what pins.rs reads.
const between = (a, b) => round(a + (b - a) * random(), 8);
const sizes = [
  [1024, 768],
  [512, 512],
  [390, 844],
  [1920, 1080],
  [300, 600],
];
const cameras = [];
for (let i = 0; i < 4000; i++) {
  const [width, height] = sizes[Math.floor(random() * sizes.length)];
  const transform = new MercatorTransform({ maxZoom: 24, maxPitch: 85 });
  transform.resize(width, height);
  transform.setZoom(between(0, 22));
  transform.setCenter(new LngLat(between(-180, 180), between(-84, 84)));
  transform.setBearing(between(-180, 180));
  transform.setPitch(i % 4 === 0 ? 0 : between(0, 85));
  // The screen points are pins.rs's FRACTIONS of the viewport.
  const points = [
    [0.5, 0.5],
    [0.1, 0.9],
    [0.9, 0.95],
    [0.25, 0.6],
    [0.8, 0.7],
  ].map(([u, v]) => {
    const ll = transform.screenPointToLocation(
      new Point(u * width, v * height),
    );
    return [round(ll.lng, 10), round(ll.lat, 10)];
  });
  const camera = {
    lon: transform.center.lng,
    lat: transform.center.lat,
    zoom: transform.zoom,
    pitch: transform.pitch,
    bearing: transform.bearing,
    fov: transform.fov,
    width,
    height,
    points,
  };
  if (i % 10 === 0) {
    camera.tiles = coveringTiles(transform, {
      tileSize: 512,
      maxzoom: 22,
      minzoom: 0,
    })
      .map((t) => [t.wrap, t.canonical.z, t.canonical.x, t.canonical.y])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
  }
  cameras.push(camera);
}
await writeFile(
  new URL("./data/pins.json", import.meta.url),
  `${JSON.stringify({ commit, cameras })}\n`,
);
console.log(
  `${cameras.length} pinned cameras, ${cameras.filter((c) => c.tiles).length} covers`,
);
