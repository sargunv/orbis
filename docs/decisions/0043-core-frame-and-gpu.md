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

## One way to draw apart from the map

A layer composited once (`*-layer-opacity`, `-blend`), a 3D layer, a
heatmap's density and a terrain stack each drew into textures of their own
and composited back through their own layouts, pipelines and shaders. They
now share `offscreen::Target`, a list of colour attachments with their
multisampled ones, and `Composites`: one bind layout and one shader
(`composite.wgsl`) in two pipelines. *Over* blends a source at an opacity,
or a density through a ramp. *Replace* writes a W3C blend mode over the
canvas beneath. Opacity is each composite's uniform, written when paint,
visibility or a node's candidates change, so no draw is recorded again for
it. Four shaders went, and the module is net −87 lines.

**How a physically based map would slot in (#91).** Opaque layers would
draw into a material-buffer `Target`: albedo, normal, roughness and
metalness, emissive, and height as attachments of one target. That needs
one extension, a list of colour attachments in a HAL `RenderPass`; the
target already holds one. One lighting composite would then read the
attachments with the frame's lights, the atmosphere's image-based light
and the shadow map. It would draw into the map between the opaque segment
of the main pass and the translucent one, as a blend-mode composite opens
a segment today. Translucent layers, labels and the layers composited
once would follow unchanged.

## Pipelines that fail or hang

A pipeline the driver can't make no longer panics. On Modern it goes the
way of a compile the watchdog found hung: the device is lost, the crash
sentinel stays armed and recovery demotes to Compat. Compat is the last
tier, so there the pipeline stays unready, its nodes draw nothing, and the
failure is reported as a hung compile would be (`RuntimeStats::hung_compile`).

The native watchdog ran only when a frame polled the compiler, and a map
waiting only for pipelines asks for no frames: a hung last compile left it
waiting for good. A watchdog thread beside the compile workers now wakes
the host once a compile runs past the limit, and the frame that finds the
device lost reports it rather than the frame after, which no host draws.
With a 300 ms limit the test map draws 2 frames, idles, and is lost after
314 ms.

## One native thread pool

Pipelines compiled on worker threads of their own, up to four beside the
six of the jobs pool: ten threads on an eight-core phone. They now compile
on the runtime's jobs pool. A request queues its job and spawns one task,
which compiles whatever job is at the front of the queue when it runs, so a
pipeline the view needs still moves ahead. Without an executor a pipeline
compiles on the calling thread, as tools and tests want. The jobs pool
no longer joins its workers when dropped: a compile stuck in a driver would
hold the host's shutdown. Fetches (blocking I/O), placement (a latency lane
of its own), the HTTP cache's flusher and GL's compile thread (a current
context) keep their threads. On the Tokyo snapshot to idle, interleaved, 10
runs each: median 267 ms before, 251 ms after.

The browser's double creation (`create*PipelineAsync`, then wgpu's own
pipeline from Dawn's cache) stays until wgpu has async creation
(gfx-rs/wgpu#10438, on #35): wgpu 30 can't wrap a browser pipeline.

## One DEM prepare path

The DEM prepare ran as two compute dispatches per page on Modern and as two
draws per tile on Compat, with the texel functions shared and two bindings
around them. It is now the draws on both tiers: the job is a uniform, each
step a triangle over the tile's layer. Five shader files, two compute
pipelines and their layouts and bind groups went. Native fixtures for
hillshade, color-relief, terrain and raster match main (27/0/1, 9/0/0,
44/30/14, 56/2/1), and Chrome's hillshade too. The Alps still renders the
same picture: 6 runs each, 0.70–1.15 s before, 0.51–0.94 s after.

## Smaller steps

- **The web's compile wake (#70).** Each `create*PipelineAsync` promise
  wakes the map, and a web map waiting only for compiles asks for no
  frames. WebGL2 has no callback and still polls.
- **One resource table.** The two HALs' copies of the id-keyed table are
  `orb_hal::slots::Slots`. Texture pages hand out layers through the
  scene's `Slots`.
- **Terrain before its DEM.** Terrain paint no longer waits for a DEM
  page: tiles draw flat until their DEM arrives, as GL JS's do.
  `terrain/pitched-world` passes. Globe drapes this exposed are #98.
- **Warps per tile.** The low-zoom surface's patches are keyed by tile
  and made when the tile is first requested (its grid is needed then),
  then uploaded once. That is the "one warp per tile" B10 asked for;
  covered tiles that aren't resident (background, terrain cover) need
  them too, so they stay out of the resident tile.

## Teardown (#79)

A dropped map frees what its renderer made: the HAL destroys every
resource class (samplers, shader modules, bind group layouts and
pipelines joined buffers and textures), and `Renderer::destroy` walks a
small `Release` trait over what each piece owns. Ids a piece only binds
stay with their owner. A create, draw and drop loop holds 5 live
resources flat on Modern and 4 on Compat (the emulator, `cargo xtask
android --run --leakcheck`).

## The frame's plan

`Map::plan_frame` is a function of the view and the resident set: per
source, the cover and `TilePyramid::plan`, with no writes. `apply_source`
commits it, sends the loads and keeps the draws for placement. Placement
stays a step of its own. It uploads globe warps as tiles first need them,
and a pure placement plan would split that out and add code.

Warped raster tiles of different zooms used to leave sub-pixel cracks
where they met (#32). A warp mesh's sides now take `2^d` times the edge
steps when the finest drawn tile across them is `d` zooms finer (up to
4), so both tiles share every vertex there. Each frame, `place_source`
keeps every drawn image tile's mesh current for that frame's surface and
neighbours, which also replaced the globe switch's separate rebuild.
GIBS Web Mercator rasters on an EPSG:3413 plane at 1024×768 had 2, 12
and 5 background pixels at seams in three views; they now have 0.

## One downsampler

Raster pages, DEM slopes and the drape atlas mip through one compute
shader over regions (layer, corner, side). The drape atlas is a
one-layer array. Native fixtures are identical per fixture and pixel.

## Drape size classes, measured (#42)

On the Alps gestures bench (1024×768 at pixel ratio 2, 4,695 frame
plans), 99.86% of terrain tiles need the full 1024² slot by their
longest on-screen edge. The terrain cover keeps tiles near 512 px on
screen, and a foreshortened tile is still full width. Square size
classes would save under 1% of atlas memory, so #42 is closed.

## A drape atlas sized to its slots

The drape atlas used to double and never shrink, and each new atlas
blanked every slot for a full redraw. Now it holds the slots in use with
an eighth to spare. It is made again when the slots outgrow it or fill
under four fifths of it. The new atlas takes the kept slots' texels,
every mip level, by a GPU copy (the HAL's `CopyTexture`), slots past its
end move to free ones, and the runtime follows the moves, so nothing
redraws. Only a change of drape stacks blanks it. On the Alps gestures
bench, the atlas's mean per frame went from 341 MiB to 264–278 MiB
across rotate, pan and the flight. The peak stays at 341 MiB: 29 tiles
at 2 stacks reach the 64 slots an 8192-texel texture holds. The three
paths made 54 resizes, each under 1 ms of CPU.

## Map trims

Terrain keeps itself (`Terrain::upkeep`, `arrive`, `release`), and the
atmosphere is a function of the scene, view and camera. Warnings go out
once per style revision, which replaces four warn-once flags. A dead
cross-fade field, the `bend_of` adapter and the duplicated pixel and
node-state conversions went. `map.rs` is 2,569 lines, from 2,775.

## Still to land

- A Compat cull parity test: `compat::cull` as a pure function of the
  tables, compared with Modern's dispatch read back on the same scene.
