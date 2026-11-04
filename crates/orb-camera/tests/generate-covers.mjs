// Run against the unmodified GL JS sources at the commit pinned in xtask/src/fixtures.rs.
// See README.md beside this script for setup and regeneration.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
const upstream = resolve(process.argv[2]);
// Reject changed reference implementations rather than silently blessing new vectors.
const referenceHashes = {
  "geo/projection/covering_tiles.ts": "b606d8e2838a532c95ffff8df36fecd094972e31b24cb2b3b3c43a210dda1f91",
  "geo/projection/mercator_transform.ts": "50b88d9fea8938bacd95fa36256644198dab55e47119d69a262bd2f3b3d39c57",
  "geo/projection/mercator_covering_tiles_details_provider.ts": "061d402a71bfa66c0ae7bb2bac3796d5090d429e96b2b7d4194e4898898cf6a7",
  "geo/projection/mercator_utils.ts": "6feeaf70b24be0f88ba2198472eec26bc4a2324f3ebabda7e09c7eee0f3d082f",
  "geo/camera_state.ts": "0ad5a7e5e5ad58cc717f25a00c9d1282381beba00abe21d128180433da25b5a8",
  "util/primitives/aabb.ts": "323fd0363891c9bfecd7276794a0c91b70fa40b56e174250d5a7d5be63cd09a6",
  "util/primitives/frustum.ts": "39336cbe0009e67aefd756d63c943bdb808feff20abfe0f56163cd43096c08cb"
};
for (const [path, expected] of Object.entries(referenceHashes)) {
  const actual = createHash('sha256').update(await readFile(`${upstream}/src/${path}`)).digest('hex');
  if (actual !== expected) throw new Error(`reference source differs from the pinned commit: ${path}`);
}
const from = async (path) => import(pathToFileURL(`${upstream}/src/${path}.ts`));
const { MercatorTransform } = await from('geo/projection/mercator_transform');
const { LngLat } = await from('geo/lng_lat');
const { coveringTiles } = await from('geo/projection/covering_tiles');
const pins = await readFile(new URL('../../../xtask/src/fixtures.rs', import.meta.url), 'utf8');
const commit = pins.match(/const COMMIT: &str = "([a-f0-9]{40})";/)[1];
const cases = [];
function add(name, options) {
  const { lon = -74.0113, lat = 40.7075, zoom = 14.48, pitch = 0,
    bearing = 0, width = 1024, height = 768, fov = 36.86989764584402,
    maxPitch = 85, tileSize = 512, maxzoom = 22, minzoom = 0, maxContentElevation = 0 } = options;
  const transform = new MercatorTransform({ maxZoom: 24, maxPitch: options.defaultMaxPitch ? undefined : maxPitch });
  transform.resize(width, height);
  transform.setZoom(zoom);
  transform.setCenter(new LngLat(lon, lat));
  transform.setFov(fov);
  transform.setBearing(bearing);
  transform.setPitch(pitch);
  const tiles = coveringTiles(transform, { tileSize, maxzoom, minzoom, maxContentElevation })
    .map(t => [t.wrap, t.canonical.z, t.canonical.x, t.canonical.y])
    .sort((a,b) => a[0]-b[0] || a[1]-b[1] || a[2]-b[2] || a[3]-b[3]);
  cases.push({ name, lon: transform.center.lng, lat: transform.center.lat,
    zoom: transform.zoom, pitch: transform.pitch, bearing: transform.bearing,
    fov: transform.fov, width, height, tileSize, minzoom, maxzoom, maxContentElevation, tiles });
}
for (const zoom of [2.2, 10.3, 14.48, 18]) {
  for (const pitch of [0, 45, 60, 60.1, 80, 85]) {
    for (const bearing of [0, 30]) {
      add(`z${zoom}-p${pitch}-b${bearing}`, { zoom, pitch, bearing });
    }
  }
}
for (const tileSize of [256, 1024]) {
  for (const pitch of [0, 60, 80, 85]) {
    add(`tile${tileSize}-p${pitch}`, { tileSize, pitch, bearing: 135 });
  }
}
add('high-content', { pitch: 80, maxContentElevation: 1200 });
add('high-zoom-close-eye', { zoom: 24, pitch: 85, maxzoom: 24 });
add('wide-world-copies', { lon: 0, lat: 0, zoom: 0, width: 1600, height: 500, pitch: 80, bearing: 90 });
add('source-maxzoom', { pitch: 80, maxzoom: 12 });
add('source-minzoom', { pitch: 80, minzoom: 12 });
add('default-maxPitch-clamps-80-to-60', { pitch: 80, defaultMaxPitch: true });
add('interaction-cap-clamps-90-to-85', { pitch: 90 });
add('wide-fov', { pitch: 80, fov: 50 });
add('antimeridian', { lon: 179.8, lat: -30, zoom: 3.5, pitch: 80, bearing: 270 });
add('world-copies', { lon: 0, lat: 0, zoom: 0, width: 1000, height: 500 });
await writeFile(new URL('./data/covers.json', import.meta.url), `${JSON.stringify({ commit, cases }, null, 2)}\n`);
console.log(`${cases.length} golden covers`);
