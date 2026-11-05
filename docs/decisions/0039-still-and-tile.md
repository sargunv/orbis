# 0039: Still and Tile output, and tile-scope placement

Plan §14.2's server modes, built as a mode the frame takes rather than a
second path through the runtime, so the snapshotter (M5) and the `orbis` CLI
sit on the same frames as an interactive map.

## Decision

- **`RenderMode`** on `orb_runtime::Map`: `Continuous` (the default),
  `Still` and `Tile`. Outside `Continuous` nothing fades (labels and raster
  tiles), and placement starts from nothing each time it runs, so its
  variable anchors, orientations and fades never depend on earlier frames.
  Everything else is the ordinary frame.
- **`orb_api::Map::render_still`** draws frames until two in a row are
  idle, waiting on the map's wake (a frame's time at most, for clock-driven
  work) between them, and fails after a timeout. **`render_tile`** draws one
  tile of a tile matrix set on the scene's plane: a camera centred on it,
  looking straight down and unrotated, in a target three tiles wide, and
  returns the middle tile's pixels. The camera store isn't touched; the next
  interactive frame takes its camera back.
- **Tile-scope placement** (`orb_symbol::Scope::Tiles`). The scene hands the
  placement lane, beside the frame, each of the 3×3 grid tiles around the
  drawn one as a view of its own (centred on it, the tile's size). Placement
  then runs in two steps:
  1. each grid tile places, in its own view, only the labels anchored in it,
     against each other, as GL JS would in that view;
  2. a label that overlaps a label of another grid tile ranking above it
     (layer, sort key, the tile in row-major order, then the order its tile
     placed it in) is dropped. Being dropped depends only on step 1's
     results, never on what step 2 drops.

  Neither step reads the drawn tile: grid tiles' views are functions of
  their own index, and two tiles' shapes are compared in one tile's view,
  offset by whole tiles. So two tiles drawn apart decide every label that
  crosses their shared edge alike whenever both neighbourhoods hold the
  tiles of that label's competitors, which holds for labels up to half a
  tile in size.

## Why not just a bigger view

Placing against a padded view (as Native's tile mode did before its
`TilePlacement`) agrees on most edges, because GL JS's greedy order is the
same in both tiles' views for labels near the edge. It breaks when a chain
of collisions runs from outside one neighbourhood to the edge: a row of
labels 41 pixels apart, each overlapping the next, shows different labels at
an edge depending on where the view first sees the row. On that row the
padded view disagreed on 3 edge labels; tile scope on none.

## Measured

Apple M-series, release build:

- Still output repeats byte for byte on the Tokyo extract with Liberty's
  labels, also after rendering elsewhere in between (`orb-api`'s `modes`
  test).
- Tile output repeats byte for byte, also after other tiles; over the 3×3
  tiles around Tokyo's Imperial Palace at z15 (512 px), 26 labels cross the
  12 shared edges and all agree to 0.01 px; on a synthetic crowd of 1,260
  labels at z10, 36 agree and none differ.
- `orbis tiles` writes 64 Tokyo tiles (z13–16, 512 px, PNG) in 1.8 s,
  about 28 ms a tile with its neighbourhood; 4 NZTM2000Quad tiles of the
  `nz` scene from live OpenFreeMap tiles in 1.2 s.
- Mirrored GL JS render fixtures natively: unchanged (1179 pass, 116 fail,
  50 skip).

## Not done

- A grid tile past the matrix (a wrapping plane's antimeridian) is left
  out, so labels crossing a Web Mercator pyramid's ±180° edge may differ.
- Tile mode draws nine tiles' pixels for one; scissoring to the middle
  tile is possible if throughput ever matters.
- The C ABI and the web package don't expose the modes yet: the snapshotter
  is their first consumer.
- The `wasm32-wasip2` kernel component is past the first-release cut line.
