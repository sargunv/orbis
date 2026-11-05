# 0037: The memory budget: growable pools and a cache sized by memory

Scene memory caused two visible bugs (decisions 0027 and 0032): the scene's
pools were fixed (16 M words each for vertices, indices and compacted
indices, about 220 MiB from the first frame on every device) and the tile
cache was a count (128 vector and 32 raster tiles per source), so dense or
large views filled the pools with cached tiles and dropped tiles in view.
Plan §8 calls for a budget manager; this is its first shape (#48).

## Design

- **Pools grow, to the device's limits.** The vertex and index pools start
  at 16 MiB and double into a new buffer when the tiles in view need more:
  the contents copy over on the GPU (a new `Pass::CopyBuffer`; on Compat,
  `copyBufferSubData` between GL buffers and row copies between data
  textures), the arena gains the new space as one free block (which also
  ends fragmentation), and the renderer binds the new buffer (scene, drape
  and cull groups, recorded bundles). The compacted index buffer grows
  without a copy: every cull rewrites it. Ceilings are the device's storage
  binding and buffer sizes (128 MiB per pool on default WebGPU limits, 256
  MiB on WebGL2 here). Tile pages grow by half rather than doubling: their
  layers are large (a 512-px raster tile with mips is about 1.4 MiB).
- **Unused tiles give way first.** Every path that needs room (vector,
  label, raster and DEM tiles, relayouts) evicts tiles out of view before a
  pool or page grows, so allocation follows the peak the views have needed,
  not the cache. Raster tiles take a layer the page has, then those of
  unused tiles, and only then grow it.
- **The cache is bounded by memory.** Each resident tile knows its cost (its
  meshes' GPU bytes, its texture layer and its picking data). After the
  cover each frame, tiles out of view from every source are trimmed oldest
  first until their cost is within a quarter of the budget. The count
  limits are gone.
- **The budget is a target, not a cap.** 512 MB on desktops, 192 MB on phones
  and tablets (plan §8; `Map::set_memory_budget`; orb-api picks the mobile
  budget on Android and iOS builds, orb-web from `navigator.userAgentData`
  or the user agent). It sizes the cache, and eviction before growth keeps
  allocation to what views need; a view that needs more than the budget
  still draws, to the device's limits, rather than dropping tiles in view.
  Capping it hard made the globe at 2560×1440 drop tiles that it drew before.
- **Reported.** `FrameReport::memory` (budget, allocation, each pool's use,
  allocation, ceiling and largest free block, textures, the cache), and the
  web debug overlay (`?debug`) shows it.

## Measurements

Headless Chrome 156 on an Apple M6, main thread, live Liberty unless noted;
before is 97376a0. Frame interval p95 / max in ms.

| | before | after |
|---|---|---|
| scene pools at start | 220 MiB, fixed | 76 MiB, growing |
| lower Manhattan rotate, tilt and zoom path (1728×1117 @2) | 18.3 / 23.1; 2 tiles dropped for coarser ones, twice | 17.7 / 20.5; none dropped; 265 MiB peak |
| Manhattan, six rounds of easing about (1280×800 @2) | no drops; 1,553 MiB uploaded | no drops; 1,535 MiB uploaded; 201–265 MiB |
| globe zoom path z2.7 ↔ z5.5 (1728×1117 @2) | 18.5 / 20.6 | 18.2 / 23.3; 354 MiB |
| globe, two rounds at 2560×1440 @2 | no drops (0032) | no drops; 498 MiB (277 MiB textures; 377 MiB when pages doubled) |
| Alps terrain flight (1728×1117 @2) | 18.6 / 247.4, 2 gaps over 50 ms | 18.5 / 28.7, none; 414 MiB |

The gestures benches (`cargo xtask bench --experiment gestures` and
`pnpm -C web bench --experiment=gestures`, live Seattle, three native runs
each) are within run-to-run noise of each other: render CPU p95 2.3–3.4 ms
before, 2.2–4.6 ms after; the page's frame interval p95 17.1–18.2 ms in both.
On the Tokyo extract, render CPU p50 / p99 is 0.22 / 0.71 ms (rotate) and
0.90 / 2.79 ms (zoom) before, 0.20 / 0.40 and 0.52 / 3.05 after.

## Open

- Pools and pages don't shrink: after a large view, allocation stays at its
  peak (the cache then fills the space). Shrinking would copy into a smaller
  buffer once usage stays low.
- Textures dominate on the globe and over terrain (raster and DEM pages).
- Plan §8's CPU side (`TileFrame` caches in job workers, wasm high-water
  budgets and worker recycling) is not part of this.
