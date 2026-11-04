# 0023: Picking on demand, budgeted tile intake, and jobs off the page

Rotating and tilting the web demo at Seattle on live Liberty, or zooming in
and out over Tokyo, paused for whole seconds. Four causes, each measured in
headless Chrome 156 on an Apple M6 with scripted input (DevTools right-drag
and wheel) and with the gestures bench below.

## Causes

1. **The picking index was rebuilt every frame the camera moved.** Each
   such frame projected every hit-testable feature of every drawn tile into
   CSS pixels, with nested allocations per feature: 20–30 ms of render-thread
   CPU per frame at Seattle z14 (`query::ScreenHit::project` and its
   allocator traffic were 5 of the 7 s `Map::render` spent over an 8 s
   rotation). It came with the picking publication, already in the initial
   commit; snapshots were only *sent* when the camera settled, but the index
   behind them was built every frame.
2. **The render worker drew every camera the page sent** (64a3c54,
   camera-paced worker frames). With frames slower than the page's 60 Hz
   (cause 1), cameras queued, each drawn in turn: after a 6 s right-drag the
   map kept drawing stale cameras for 3.6 s (146 frames). At 34a35ec, just
   before, the same drag settled in 0.66 s.
3. **Arrived tiles applied without a budget, and a tile that did not fit
   wrote itself again for every tile it evicted.** Once the scene's arenas
   fill (Seattle, pitched), an upload ran out part way, evicted one cached
   tile and started over: about 250 rewrites in a 6 s rotation, 40–100 MiB in
   one frame, 35–60 ms frames, and a 117 ms task in the GPU process.
4. **Main-thread mode ran tile jobs on the page** (`PoolJobs` with the
   inline executor), against plan §7.1's job workers in every web mode.
   Since 3b2a559 made it the default, a live tile's layout or glyph
   rasterization took 30–190 ms page tasks while zooming.

## What changed

- A frame records only the tiles it drew and its view; the index projects
  them when something first picks from it (`Map::hit_test`,
  `query_rendered`, `hit_snapshot`), at most once per frame. An edit that
  changes the style before then leaves the older index, which answers
  through the ground as the ui role's snapshot does. `Map::hits_revision`
  replaces `hits_rebuilt`; the render worker publishes a snapshot when the
  camera has rested three frames and again when the map is idle, rather
  than every rested frame while tiles arrive.
- The render worker draws a camera after the messages already queued, so a
  backlog draws only the newest camera; with none, it draws at once.
- Arrived tiles apply within plan §7.4's 1.5 ms upload budget, at least one
  a frame, in order; edits' results apply whole. Before writing a tile, the
  scene checks that all its meshes' arena ranges and chunk slots fit
  (`Scene::fits`, first-fit on copies of the free lists) and evicts unused
  tiles until they do; the retry loop remains for label uploads.
- `thread: "main"` keeps drawing on the page but spawns the same job and
  placement workers as the worker mode (`createMap` takes the same job and
  placement hooks as `createWorkerMap`). Decision 0021's main-mode busy times
  predate this.

## Measurements

`pnpm -C web bench --experiment=gestures` (new): 360 frames of rotate and
tilt at z14 and of zoom z12 → z16 → z12, the camera moved once a display
frame, each mode from a fresh map; 1024×768 at devicePixelRatio 1. Frame
interval p50 / p95 / max and render CPU p50 in ms, gaps over 50 ms, and the
time from the path's end to `idle`. Before is 3300efd.

| live Liberty | before | after |
|---|---|---|
| Tokyo, rotate, main | 53.4 / 80.9 / 136.5; CPU 52.4; 236 gaps; idle 0.44 s | 16.7 / 18.7 / 21.7; CPU 0.3; 0 gaps; idle 0.48 s |
| Tokyo, rotate, worker | 154 of 360 frames; 38.9 / 46.3 / 52.7; CPU 38.8; idle 8.75 s | 16.7 / 18.8 / 21.7; CPU 0.3; idle 0.57 s |
| Tokyo, zoom, main | 18.3 / 36.9 / 168.4; CPU 17.6; 8 gaps | 16.6 / 18.4 / 21.6; CPU 0.4; 0 gaps |
| Tokyo, zoom, worker | 17.0 / 23.5 / 40.9; CPU 15.8 | 16.7 / 18.4 / 64.5; CPU 0.4 |
| Seattle, rotate, main | 17.7 / 37.8 / 74.8; CPU 16.9; 4 gaps | 16.7 / 17.6 / 19.3; CPU 0.3; 0 gaps |
| Seattle, rotate, worker | 317 of 360 frames; CPU 17.4; idle 0.92 s | 16.7 / 17.4 / 18.4; CPU 0.3; idle 0.49 s |

`cargo xtask bench --experiment gestures` (new) runs the same paths
natively at a 60 Hz pace and times `Map::render`, tile arrivals included;
2048×1536 offscreen. Render CPU p50 / p99 / max, frames over 16.7 ms:

| | before (64a3c54) | after |
|---|---|---|
| Tokyo extract, rotate | 2.27 / 6.06 / 7.10 | 0.57 / 1.04 / 4.87 |
| Tokyo extract, zoom | 4.28 / 9.50 / 14.38 | 0.64 / 2.80 / 6.03 |
| live Seattle, rotate | 10.06 / 23.98 / 26.14; 46 frames | 0.45 / 3.61 / 5.83; 0 |
| live Seattle, zoom | 5.45 / 9.99 / 25.68; 1 frame | 0.86 / 3.68 / 5.84; 0 |

## Open

The worker's snapshot at rest still costs one picking build (28 ms at Tokyo
z12), 10 ms to encode, and 26 ms to decode on the page; on the page a first
hover after a move builds the index (#43). The scene's tile cache is bounded
by count (128 tiles), not memory, so a pitched city view keeps the arenas
full and evicts on almost every arrival; plan §8's budget manager is the
answer.
