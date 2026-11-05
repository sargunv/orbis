# 0024: Picking without the camera

Decision 0023 made the picking index build on demand, but what it built
and what the render worker published was still in screen pixels: every
camera change made it stale, and the render worker rebuilt and re-sent it
after each gesture (#43). Live Liberty at Tokyo z14 after a rotation, Apple
M6: the build took 24.4 ms, encoding 6.2 ms and the page's decode 21.4 ms
natively, for a 20.9 MB snapshot; in Chrome 156 that was a 67.3 ms render
worker frame and a 56 ms page task at the end of every gesture. Since
decision 0021 the page owns the camera in both thread modes, so the page
can project picking geometry itself.

## What is published

Picking is kept and published in four camera-independent parts:

- **Tile geometry** (`MeshHits`): each render node's hit-testable features
  in a tile, packed by the tile job in tile-local data units as words (the
  job transport's layout), with per-feature and per-mesh bounds. The render
  thread no longer decodes them into nested vectors; the page stores them
  as they arrive. A tile's `TileMapping` (its edges and span, or a warped
  tile's series) places them in world space exactly as drawing does.
- **The drawn list**: the tiles the last frame drew, in drawing order
  (ancestors before descendants), each world copy with its clip regions'
  corners in world space.
- **Labels**: the labels and icons the last placement commit shows, as
  anchors in world space with their collision boxes in pixels at the
  anchor's scale, the form placement produces.
- **The layer table**: each node's layer with the paint that sets pixel
  tolerances (circle radius and stroke width, line width, translate and
  its anchor, extrusion base and height), sent as the document's JSON and
  parsed on the page with the same property parser (`orb_v8::parse_paint`),
  so zoom-dependent values evaluate at the query's zoom.

`Map::publish_hits` sends what changed since a host's `HitPublication`:
tiles when they load, change or go, the drawn list when the cover changes,
labels with each commit and the layer table with the style. The render
worker calls it every frame; a camera move alone publishes nothing.

## How a query runs

A query (the page's `HitIndex::hit_test`, `Map::hit_test` and
`Map::query_rendered`) takes the camera it is asked with. It unprojects its
point or box, grown by the largest tolerance any drawn feature can have, at
the ground and at the tallest extrusion's top, into world bounds; culls
meshes and then features by their world bounds (shifted by each world copy
and by the layer's translate); and projects only the candidates into
pixels for the same exact tests as before. Labels project from their
anchors. Results are unchanged: the binding scenario, the picking tests and
the query fixtures (58 of 83 natively and in Chrome, 56 of them in GL JS's
order, as at 86049c7). One difference: among tiles of the same layer, the
upper is now the one drawn later (a descendant over its fallback), where it
was the order of a hash map.

The map keeps the layer table of the version its last frame drew, so a
style edit leaves picking as that frame showed it until the next frame, as
before. The page's index answers for the camera it has now: the drawn
tiles' features where that camera puts them, and labels as the last commit
placed them, at their anchors.

## Measurements

`cargo xtask bench --experiment gestures` now times publication too
(`Map::publish_hits` and `HitIndex::apply` after each frame). Apple M6,
1024×768, live Liberty at Tokyo:

| path | while moving (360 frames) | at the path's end | first / next hit test at rest |
|---|---|---|---|
| rotate and tilt at z14 | 90 publications, 6.8 ms publishing (max 0.34) and 2.9 ms applying (max 0.18), 15.2 MiB | 0.19 ms publishing, 0.11 ms applying | 0.83 / 0.29 ms |
| zoom z12 → z16 → z12 | 66 publications, 5.6 (max 0.52) and 2.1 ms (max 0.15), 14.1 MiB | 0.11 and 0.10 ms | 1.47 / 0.21 ms |

Against 24.4 + 6.2 + 21.4 ms at the path's end before. The Tokyo extract:
0.10 and 0.05 ms at the end, first hit test 0.47 ms (was 1.7 + 0.5 +
1.5 ms).

In Chrome 156 (`pnpm -C web bench --experiment=gestures`, live Liberty,
worker mode), the render worker's largest frame after the rotation's end
goes from 67.3 ms to 0.4 ms and the page's longest task from 56 ms to none
over 50 ms; zooming, 32.3 → 0.5 ms. `core.wasm` grows 719 → 728 KiB Brotli
(the page's index and the paint parser's entry point).

The page keeps every resident tile's picking geometry, not only the drawn
tiles' projected features, so its memory follows the render worker's tile
cache (tens of MiB for a city view on live Liberty); the render worker's
copy shrinks, from nested vectors to packed words.
