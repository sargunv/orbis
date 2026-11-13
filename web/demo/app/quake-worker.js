// The Earthquake entry's ground, off the page's thread: Mapterhorn's
// terrain tiles decoded to heights, a damped radial wave from an
// epicentre added to them, and each step fed to the map's host-fed
// `raster-dem` source through its elevation port, at most one step in
// flight, so the map shows every step it is given at the rate it can.
//
// From the page (quake.js): `{ start, port }` once, then `{ tile }` for
// each tile the map asks for and `{ epicentre }` on a click.

/** Heights a tile side: Mapterhorn's 512-texel tiles averaged 2×2. */
const SIDE = 256;
const MAPTERHORN = "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp";
/** Metres around the Earth at the equator: Web Mercator's world. */
const EARTH = 40075016.686;

/**
 * The wave: a front at `speed` m/s, behind which the ground rings at
 * `hertz`, its swing `amplitude` metres at the epicentre, spreading
 * (halving by `spread` m out, as 1/√) and dying away over `decay`
 * seconds; a new shock every `period` seconds.
 */
const WAVE = {
  speed: 3000,
  hertz: 0.5,
  amplitude: 900,
  spread: 15000,
  decay: 4,
  period: 12,
};
/** The wave's table: one entry per `BIN` metres out to `REACH`. */
const BIN = 40;
const REACH = 400_000;
const BINS = REACH / BIN + 2;

let port;
let source = "quake";
/** Tiles by "z/x/y": their base heights, and their distance from the
 * epicentre in bins, per texel. */
const tiles = new Map();
let epicentre = { x: 0.5, y: 0.5, lat: 0 };
let shockAt = performance.now();
let inFlight = false;
let nextId = 0;

/** World x, y in [0, 1) of a longitude and latitude. */
function world([lon, lat]) {
  const s = Math.sin((lat * Math.PI) / 180);
  return {
    x: lon / 360 + 0.5,
    y: 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI),
    lat,
  };
}

/** Per texel of tile z/x/y, its distance from the epicentre in bins. */
function distances(z, x, y) {
  const n = 2 ** z;
  const scale = (EARTH * Math.cos((epicentre.lat * Math.PI) / 180)) / BIN;
  const out = new Float32Array(SIDE * SIDE);
  for (let j = 0; j < SIDE; j++) {
    const dy = ((y + (j + 0.5) / SIDE) / n - epicentre.y) * scale;
    for (let i = 0; i < SIDE; i++) {
      const dx = ((x + (i + 0.5) / SIDE) / n - epicentre.x) * scale;
      out[j * SIDE + i] = Math.min(Math.hypot(dx, dy), BINS - 2);
    }
  }
  return out;
}

/** A Mapterhorn tile's heights at SIDE² (terrarium), or sea level where
 * it has none. */
async function baseHeights(z, x, y) {
  const heights = new Float32Array(SIDE * SIDE);
  try {
    const url = MAPTERHORN.replace("{z}", z)
      .replace("{x}", x)
      .replace("{y}", y);
    const response = await fetch(url);
    if (!response.ok) return heights;
    const bitmap = await createImageBitmap(await response.blob(), {
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
    const { width, height } = bitmap;
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    const rgba = context.getImageData(0, 0, width, height).data;
    const step = width / SIDE;
    for (let j = 0; j < SIDE; j++) {
      for (let i = 0; i < SIDE; i++) {
        let sum = 0;
        for (let v = 0; v < step; v++) {
          for (let u = 0; u < step; u++) {
            const k = 4 * ((j * step + v) * width + i * step + u);
            sum += rgba[k] * 256 + rgba[k + 1] + rgba[k + 2] / 256 - 32768;
          }
        }
        heights[j * SIDE + i] = sum / (step * step);
      }
    }
  } catch (e) {
    console.warn(`quake: tile ${z}/${x}/${y}: ${e}`);
  }
  return heights;
}

/** The wave's swing at each bin out from the epicentre, `t` seconds after the shock. */
function table(t) {
  const { speed, hertz, amplitude, spread, decay } = WAVE;
  const lut = new Float32Array(BINS);
  for (let k = 0; k < BINS; k++) {
    const d = k * BIN;
    const s = t - d / speed;
    if (s <= 0) continue;
    const swing = amplitude / Math.sqrt(1 + d / spread);
    lut[k] = swing * Math.exp(-s / decay) * Math.sin(2 * Math.PI * hertz * s);
  }
  return lut;
}

/** Sends the next step, unless one is still going in. */
function step(now) {
  if (inFlight || !port) return;
  const ready = [...tiles.values()].filter((t) => t.base);
  if (!ready.length) return;
  const lut = table(((now - shockAt) / 1000) % WAVE.period);
  const n = SIDE * SIDE;
  const heights = new Float32Array(ready.length * n);
  const zxy = new Uint32Array(ready.length * 3);
  ready.forEach((tile, i) => {
    zxy.set([tile.z, tile.x, tile.y], i * 3);
    const { base, bins } = tile;
    const out = heights.subarray(i * n, (i + 1) * n);
    for (let k = 0; k < n; k++) {
      const b = bins[k];
      const lo = b | 0;
      out[k] = base[k] + lut[lo] + (lut[lo + 1] - lut[lo]) * (b - lo);
    }
  });
  inFlight = true;
  port.postMessage({ id: nextId++, source, side: SIDE, zxy, heights }, [
    zxy.buffer,
    heights.buffer,
  ]);
}

function frame(now) {
  step(now);
  schedule();
}

const schedule = () =>
  self.requestAnimationFrame
    ? self.requestAnimationFrame(frame)
    : setTimeout(() => frame(performance.now()), 16);

function onPort({ data }) {
  if ("shown" in data) {
    if (data.source !== source) return;
    inFlight = false;
  } else if ("error" in data) {
    console.error(`quake: ${data.error}`);
    inFlight = false;
  } else {
    // Tiles the map no longer holds: it asks again when it needs them.
    for (const { z, x, y } of data.dropped) tiles.delete(`${z}/${x}/${y}`);
  }
}

self.onmessage = ({ data }) => {
  if (data.start) {
    ({ source } = data.start);
    epicentre = world(data.start.epicentre);
    shockAt = performance.now();
    port = data.port;
    port.onmessage = onPort;
    schedule();
  } else if (data.tile) {
    const [z, x, y] = data.tile;
    const key = `${z}/${x}/${y}`;
    if (tiles.has(key)) return;
    const tile = { z, x, y, base: null, bins: distances(z, x, y) };
    tiles.set(key, tile);
    void baseHeights(z, x, y).then((base) => {
      if (tiles.get(key) === tile) tile.base = base;
    });
  } else if (data.epicentre) {
    epicentre = world(data.epicentre);
    shockAt = performance.now();
    for (const tile of tiles.values())
      tile.bins = distances(tile.z, tile.x, tile.y);
  }
};
