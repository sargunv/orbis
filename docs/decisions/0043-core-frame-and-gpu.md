# 0043: The core chain: camera-independent GPU state and a narrow renderer

Decision 0040 set the first-release target. This record holds the core
chain's part of it, the frame and the GPU scene, and what each step changed.

## Decision

- **A camera-only frame writes the frame block, then one cull dispatch and
  the recorded bundle draw it.** Nothing else in the scene depends on the
  camera:
  - **Tile rows hold the tile's place in the world.** The origin is a whole
    number of quanta (an i32) and an f32 remainder. The quantum is a power
    of two sized to the surface, with room for 64 world spans. The frame
    block holds the camera alike. Shaders subtract the whole quanta as
    integers, then the remainders.
    - An f32 hi/lo pair was tried first. On Metal, which compiles with fast
      math, the compiler reassociated the subtraction and the low parts
      vanished: at z22 a 26 cm camera move drew nothing. Integer whole
      quanta can't be reordered away.
    - Globe anchors split the same way, in ECEF metres.
    - The camera's world units per pixel moved to the frame: a row keeps
      only a factor of it, 1 on the map and a drape slot's scale in a drape.
    - The pattern origin still depends on the integer zoom, so rows change
      when the camera crosses one.
  - **Changed rows only.** The scene compares each placement's row with the
    one it holds, and writes the changed rows in one write in `prepare`.
  - **Cull parameters live in the node rows.** Margin, elevation and
    visibility are patched in place. The candidate lists are laid out again
    only when tiles or placements come or go, or a node changes how it
    draws. Hidden nodes keep their candidates; the cull and the heatmap's
    colour pass skip them by a flag.
    - The margin is still evaluated at the frame zoom on the CPU. With
      paint lowered to per-node curves (the data chain), it becomes the
      curve's maximum or a GPU evaluation.
- **Every frame is right for its camera, tested.**
  `crates/orb-api/tests/path.rs` moves a map to a camera frame by frame,
  lets it settle, and compares it with a fresh map at rest at that camera,
  to one step in a channel. It also checks that camera-only frames write no
  rows, and that a z22 corner draws where the CPU projects it.
  - It found the pyramid drawing a missing tile's nearest *ready* ancestor,
    so a coarse tile cached during a gesture stood in where a fresh map
    loads the nearest ancestor in the tileset. A missing tile's stand-in is
    now that ancestor, whatever else is resident.

## Measured

Apple M6, 2048×1536, `cargo xtask bench --experiment gestures`, render CPU:

| path | p50 | p95 | candidate rebuilds | rows written per frame |
|---|---|---|---|---|
| zoom z12 → z16 → z12 | 1.36 → 1.15 ms | 3.71 → 2.37 ms | 360 → 35 of 360 | 9.9 → 0.5 |
| rotate and tilt at z14 | 0.97 → 0.84 ms | 1.61 → 1.23 ms | 54 → 54 | 20.5 → 1.7 |

The rows still written are tiles arriving or leaving, and the background's
cover. Native fixtures are unchanged: 1179 pass, 116 fail, 50 skip.

## The renderer holds tiles

`Renderer::upload_*` turns a tile into an opaque `GpuTile` that the map
keeps in its pyramid, and `release_tile` frees it. The renderer owns all of
a tile's GPU residency, and asks the map for room by failing with `NoRoom`:
the map evicts an unused tile, or uploads again allowed to grow, and
otherwise lets a coarser tile stand in.

- **Image tiles first.** A raster or DEM tile takes a layer of its source's
  page and a quad in each of the source's image nodes. The renderer makes
  the page with the source's first tile, binds it to the source's nodes,
  adds resident tiles' quads to nodes that edits add, and destroys the page
  once no node draws the source and no tile is left. It indexes image tiles
  by source and tile, so a DEM tile's arrival prepares the resident
  neighbours whose borders it fills, and a raster tile finds the ancestor
  it cross-fades from, without the map's bookkeeping. Render nodes name the
  source they draw (`RenderNode`).

- **Vector and symbol meshes next.** A vector tile is a `GpuTile` the map
  fills with `add_mesh` and `add_symbol_mesh`; the renderer keeps its scene
  tiles per layer extent and its patches. `MeshId` stays a handle the map
  holds for the edits that replace a mesh's rows (`set_mesh_rows`) and for
  symbol state. `remap_nodes` drops the meshes of nodes that leave, in
  every tile. Image keys and feature-state indices are still resolved on
  the CPU and passed in as `MeshRows`.

- **The map borrows its renderer.** `Map::render(gpu, &mut Renderer, …)`
  draws with a renderer its host keeps: `orb-api` keeps it beside its
  targets, the web beside its canvas. A renderer has an id. A map that
  meets a new one (on a new device, say) forgets the old one's tiles and
  lays its style out again, so device recovery is: drop the renderer,
  reset the runtime's shared images once (`Images::reset`), and draw.
  What a style lets go of between frames (its tiles, background, state
  table) waits in a queue for the next frame's renderer.

- **The scene is the renderer's.** Placement goes through the renderer
  (`place(tile, extent, rows)`, `unplace`, `set_tile_order`), which fills
  in a tile's own patches. The background is a tile like any other.
  Low-zoom surface patches are a renderer store keyed by tile matrix set
  and tile (`surface_warp`), and the map's globe cache keeps only CPU
  series and footprints. `Scene` isn't re-exported: nothing outside
  `orb-render` reaches it, its tests included.

- **Replaced tiles keep their labels until a placement covers them**
  (#55, GL JS `holdingForFade`). A tile that leaves the draw list with
  symbol buckets keeps drawing them, as last placed, through placements
  only symbol nodes cull (`PlacementTarget::Labels`). Once a placement
  asked for after it left commits, its labels fade out from where they
  are, and the tile is released when they have. Still output holds
  nothing. Zoom z12 → z16 → z12 on Tokyo: label blinks 42–64 → 0 a run.

- **Memory follows what the map draws.** The vertex, index, compacted
  and property pools start at 1 MB and double as tiles need, contents
  copied. The glyph table is the property pool's first block, so it keeps
  its offset as the pool grows. The chunk table grows alike, and the
  candidate and visible lists grow when a layout outgrows them. The image
  atlas starts with one page and adds pages up to eight; a map hands its
  renderer the atlas's bindings each frame. Tokyo at z12, z15 and z16
  pitched: 115.5 MB → 26–45 MB allocated, guarded by
  `crates/orb-api/tests/memory.rs`.
  - Room for an arriving tile: while the pools hold less than the memory
    budget they grow before cached tiles go, and `trim_cache` keeps the
    cache to a quarter of the budget; past the budget, unused tiles make
    room first (decision 0037's order). With pools that start small,
    evicting first would have left no cache at all.

## Still to land

The Map/Renderer split (`plan_frame`, a narrow `Renderer` that owns all
residency), reflected layouts, one offscreen group, one slot allocator,
lazy pools and atlases, one warp per tile, one native thread pool, and
pipeline failures as map errors. Each step updates this record.
