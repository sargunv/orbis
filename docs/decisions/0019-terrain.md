# 0019: Terrain: draped slots, DEM references and a camera on the ground

Status: implemented under Mercator, natively and on the web; occlusion of
symbols and circles by terrain follows (see the end). The behavioural
reference is MapLibre GL JS `2b68611a4e2d4f7f076f6c6ebf758e2d53f91471`:
`terrain.ts`, `terrain_tile_manager.ts`, `render_to_texture.ts`,
`draw_terrain.ts`, `terrain_coverage.ts` and the terrain shaders.

## What the v8 `terrain` does

`terrain: {source, exaggeration}` parses into the Style IR and switches
through `setTerrain` transactions without a reload. The DEM source joins
the style plan even when no layer draws it. Terrain keeps its own tile
pyramid and DEM page for that source: a source shared with hillshade loads
its DEM tiles twice rather than bending one cover to two uses (GL JS
changes the source's tile size for terrain and warns about sharing).

## Terrain tiles and their DEM

Terrain tiles are 512 logical pixels at their zoom and are covered as GL
JS covers terrain: by DEM elevation ranges where known, with detail varying
by distance at any pitch. A tile at zoom `z` reads the DEM tile at
`z − 1 − log2(tileSize / 256)`, clamped to the source's zooms and never
deeper than `z`, or its nearest loaded ancestor (missing tiles request
their ancestors). This is GL JS's DEM choice for 256- and 512-pixel DEMs
alike. Each tile is a shared 128-cell grid with skirts, one vertex per DEM
pixel, raised in the vertex shader by GL JS's bilinear `get_elevation`
over the padded DEM page the prepare pass already fills (decision 0007).
Plan §9.4's CDLOD quadtree is this tile quadtree; no geomorphing was needed
for the fixtures or the demo.
Skirts hang a fifth of a tile's width, GL JS's `auto`.
`Map::set_terrain_skirts(false)` is GL JS's `terrainSkirtLength: "none"`,
for translucent drapes, where skirts would show.

## Draping: slots of one atlas, drawn by drape placements

Layers GL JS renders to texture (background, fill, fill outline, line,
raster, hillshade, color-relief) drape. A run of them between layers that
draw in 3D or on screen is a drape stack. Each terrain tile owns a block of
drape slots, one per stack: 1024² texel squares (GL JS's quality factor 2
over 512 pixels) of one mipmapped RGBA8 atlas. Draped families get
pipelines for that format when the target's differs.

The GPU-driven design is kept by making drape slots ordinary placements:

- A tile placement has a target, the map or drape stack `s`
  (`PlacementTarget`). Draped nodes draw only their stack's placements;
  the cull keeps those by clip rectangles alone (`DRAPED` on the node's
  draw word).
- A drape placement maps a source tile's data into a slot, clipped to the
  terrain tile, in drape space: atlas texels with y up. The drape frame is
  the map's with an orthographic atlas projection.
- Pixel sizes convert with a per-placement `world_per_px`: map
  placements carry the camera's, drape placements their slot's at its
  drape zoom (below). Tile rows grew from 64 to 80 bytes for that, the DEM
  reference and the terrain range, then to 120 for the drape zoom.
- One render pass redraws every slot marked this frame: it clears them,
  then replays one bundle per drape zoom among them.
  One compute pass then downsamples them, a dispatch per mip level over
  the redrawn slots (a render pass per level cost 3 to 5 ms each on
  Apple GPUs whatever its size; the compute pass costs 0.35 ms for all
  levels of 19 slots). Slots are aligned, so levels
  never mix slots, and the terrain shader clamps samples to the slot at
  the level it reads, as a texture of its own would clamp to its edge.

Each stack's terrain draws into a layer of its own sharing the 3D depth
with extrusions, then composites into the map at the stack's place, as
extrusion layers do (decision 0011). GL JS's terrain fog (sky `fog-*`,
`horizon-*`, faded in from 60° to 70° of pitch) is applied there.

### Slots don't depend on the camera

A slot draws its terrain tile's content at the tile's own style zoom, its
*drape zoom*, whatever the camera's. That is where the tile spans its 512
logical pixels, at two texels each (GL JS's quality factor). Every draped
property evaluates there: node uniforms, table and feature-state curves,
the GPU interpreter, pattern images at a settled cross-fade, dashes, and
layer minimum and maximum zooms. Widths are in slot texels at that zoom.
GL JS instead renders to texture at the camera's zoom and waits for the
camera to stop before redrawing (#7549). Orbis first did the same, so
lighting and widths lagged a moving camera and caught up tile by tile.
Then it redrew in the frame the zoom changed, which cost the Alps flight
p50/p95 3.7/9.1 → 11.5/16.5 ms (Apple M6, 2048×1536, to GPU done).

How several zooms draw in one GPU-driven pass, without more nodes or
culls:

- A drape placement carries its drape zoom (`TileRow::drape_zoom`; 0 on
  map placements).
- The renderer keeps a drape pass per drape zoom in use (`DrapeZoom`):
  a frame block at that zoom, the draped nodes' uniforms at that zoom
  (written by `Renderer::set_drape_paint`, which leaves a node out where
  its layer doesn't draw), their bind groups, and a bundle.
- The drape render pass replays the bundles of the zooms being redrawn.
  The cull is shared: each node's visible list holds its drape placements
  of every zoom. A draw drops the vertices of placements whose zoom isn't
  its frame's (`drawn_in_pass` in `draw.wgsl`). That multiplies vertex
  work by the number of zooms only in frames that redraw slots of several
  zooms. Every shader reads its zoom from the frame as before.
- The runtime writes the paint for a drape zoom when the zoom first
  appears or the paint inputs change. It does not do this per frame.

A slot is redrawn when what it draws changes. Its key is the tiles drawn
over it, with their meshes (and, for DEM image sources, which neighbours
fill their borders), the style revision, feature-state writes, a revision
per drape zoom that changes when the uniforms written there do, and the
image atlas's pixel revision. Unchanged slots cost nothing per frame.

Two exceptions follow the camera honestly. A draped layer that turns with
the bearing has its zoom's slots keyed on the bearing: a viewport-anchored
hillshade's light, or a viewport-anchored `*-translate`. Lighting
viewport hillshades in the terrain pass instead needs a second slot or
plane per tile, because the hillshade sits between other draped layers
(#83). Feature-state rows hold values for the camera's integer zoom, so
state paint that varies with zoom keys on that zoom (#85).

Because widths are fixed per tile, a draped line's width on screen
ranges from its width at the tile's zoom up to twice that, as the camera
zooms in until the cover splits the tile. Each tile keeps its own scale,
so widths keep their pixel size into the distance, where GL JS's narrow
with perspective, and step by up to 2× at seams between tile zooms. The
terrain fixtures pass as before (31 natively, 31 in Chrome); views at
fractional zooms differ a little more in width (`skirts-none` 0.25% →
1.1% of pixels). Draping one zoom deeper, where a texel is a logical
pixel, halved widths at the fixtures' integer zooms and failed two more.

Over the Alps demo (`cargo xtask bench alps --experiment gestures
--timing work-done`, Apple M6, 2048×1536, to GPU done, p50/p95 ms; the
demo style, median of three runs on a shared machine, then once with its
hillshade anchored to the map):

| | rotate | Alps flight | flight frames > 16.7 ms |
| --- | --- | --- | --- |
| redrawn in the frame the zoom changes | 8.8 / 14.3 | 8.2 / 12.9 | 74 |
| keyed by content, at drape zooms | 10.3 / 17.5 | 8.7 / 13.8 | 101 |
| … hillshade anchored to the map, before | 5.1 / 9.4 | 8.8 / 19.5 | 233 |
| … hillshade anchored to the map, after | 5.6 / 13.0 | 4.3 / 11.2 | 82 |

The demo's hillshade is viewport-anchored, so turning still redraws every
slot. Those frames now replay the draped nodes once per drape zoom, which
costs 0.5 to 1.5 ms. Without a layer that turns, slots redraw only when
their content changes, which halves the flight's p50. Lighting viewport
hillshades in the terrain pass (#83) gives the demo style that too.
`orb-demo`'s `rotating_over_the_alps_matches_a_fresh_view` keeps the camera
turning until its tiles load, then checks that frame against a fresh map
at rest at that camera. Silhouettes differ by about 1,200 pixels for
another reason (#84).

## Elevated content

Each map placement names the DEM tile its content stands on (page layer,
depth and offset: GL JS's `getTerrainData(tileID)`) and the elevation
range under it, which the GPU cull and CPU cover use. Circles and heatmap
points stand on the ground under their anchor. Extrusions rise by the
ground under their feature's centroid, ground-level parts with GL JS's
10 m basement; the centroid is per feature rather than GL JS's per
polygon, carried in the extrusion row (now ten words; constant-paint
extrusions get rows too).

Symbols stand where GL JS's `getSymbolElevation` puts them: their
`symbol-height-offset` (now in the instance rows, two words per instance),
over the ground under each anchor or line vertex when
`symbol-height-anchor` is `ground`. The GPU reads the placement's DEM
reference; placement reads the same DEM tile on the CPU through
`orb_symbol::Ground`, which the map's terrain snapshot implements, so
collision boxes and drawn glyphs agree.

## The camera on the ground

`CameraState::elevation` is the target's height (GL JS's center elevation);
camera-relative world space keeps the datum at z = 0 and the view orbits
the raised target. Rendering never writes the camera (plan §9.4). Camera
stores get the map's terrain through `orb_camera::Elevation`, a snapshot
the map publishes (`Map::elevation`, `Map::elevation_revision`):

- At rest the target stands on the terrain (`settle_on`); where the DEM
  under it is still loading it keeps its elevation and moves once, when
  the data lands (GL JS #8549's jump on `jumpTo` does not happen).
  Unclamped (`set_center_clamped_to_ground(false)` on the camera store
  and on `Map`, GL JS's `centerClampedToGround`), the target keeps the
  elevation it is given (`set_center_elevation`, or `CameraState::elevation`
  directly), so the camera orbits a point in the air.
- Gestures hold the elevation; when they end the target moves to the
  terrain along the view with the eye in place (`recenter_on`, GL JS's
  `recalculateZoomAndCenter`), so the picture doesn't move.
- An app's ease or flight lerps the elevation toward the terrain under its
  destination, re-aimed whenever that changes as DEM data lands, with the
  elevation now kept (GL JS's `_updateElevation`): data landing mid-flight
  bends the path instead of moving the camera.
- The eye and its near plane keep 50 m above the terrain (`keep_above`,
  `View::clearance`): the ground under the eye and a 5×5 grid across the
  near plane, which reaches about 16 pixels ahead, where a slope would
  show cut open. The camera rises straight up by the shortfall, over the
  same ground, so one step suffices and the lift is continuous as the
  camera moves. Every writer goes through it: gestures, animations, API
  calls and new terrain. The terrain requests the DEM tiles under that
  footprint at the nearest tiles' detail, though it lies behind the view.
- The far plane reaches the lowest elevation of the DEM tile under the
  target, as GL JS's `minElevationForCurrentTile` makes it; without it the
  Dead Sea fixtures lost their far terrain.
- `View::unproject_terrain` walks the view ray over the CPU elevations
  (plan §9.4's CPU ray walk). Gestures drag the plane through the target.

CPU elevations are decoded once in the DEM tile's job (`DemGrid`) and
sampled as GL JS's `TerrainCoverage` does: bilinear in the DEM tile the
drawn terrain tile reads, past tile edges in its neighbours. DEM tiles are
data: on every platform the core decodes them itself
(`orb_host::decode_image`, PNG and WebP), and the GPU and the CPU take the
same texels; on the web the job worker transfers them with their f32
elevations. A browser's decode read back through a canvas can differ by a
bit here and there (Brave's fingerprinting protection perturbs readbacks),
which a Terrarium DEM turns into 256 m spikes. The decoders add 87 KiB
brotli to core.wasm (720 to 807).

## On the globe

Terrain uses the globe surface's own placement (decision 0030), not a
second projection:
- **Placement.** Each terrain tile's row carries its patches in space,
  the same ones the layers' tiles use, with the anchor's offset from the
  camera target. The terrain shader evaluates them per grid vertex
  (`project.wgsl`'s `sphere_at`, over the scene's property arena) and
  raises the point by its elevation along the sphere's up
  (`ground_clip_3d`).
- **The hand-over.** As the view hands over between globe and plane it
  mixes the sphere's position with the plane's, as layers do. Drape slots
  stay flat: the drape frame drops the globe.
- **Poles.** A top or bottom row of a set that stops short of the pole
  (Web Mercator's 85.05°) sends its edge skirt to the pole, closing the
  globe with the edge's colour, as the layers' caps do.
- **Fog** stays off on the globe, as in GL JS.

`projection/globe/terrain` went from 22 failures to 21 (natively):
`fill-planet-solid` passes. Pole caps close and match GL JS
(`fill-planet-pole` is 0.5% off, at the limb). The rest are limb and
cover differences shared with the globe without terrain.

The camera's terrain clearance, its ray walk and the terrain cover's
elevation ranges are still planar. On the globe they read the flat
surface's terrain, which is close at the zooms where terrain matters,
since the globe hands over to the plane by zoom 12.

## The web's ui role and placement worker

Threads without the map's DEM tiles get the elevation pyramid (plan §7.1):
each DEM tile resampled to 128² samples at pixel centres, quantized to 16
bits per tile. It follows the DEM tiles alone: a snapshot is taken when one
arrives or leaves, never for the view, and a point reads the deepest DEM
tile loaded there, as the nearest terrain draws. The map's thread publishes
it with the frame in which it changed. (It once carried the drawn terrain
tiles, so every cover change made a snapshot: 170 of them, 97 MiB encoded,
in the gestures bench's 360-frame rotate. And it published at most every
30 frames while the camera moved; before that, only once the camera
settled, so a flight that never paused flew with the target at sea level
through the Alps.) The ui role's
camera stands on it, and its `project`/`unproject` go through it, so the
synchronous `unproject` is approximate. The placement worker receives it
when it changes, before the frames that use it.

Its documented error, from `orb-demo`'s ignored
`pyramid_error_over_the_alps` test over the Alps demo (1024×768, zoom
11.6, pitch 72, 1,120 sampled pixels): the point the ui role unprojects
draws a median 0.87 px from the pixel, 5.3 px at the 95th percentile and
21 px at most (steep faces near the horizon); elevations differ by a
median 13 m. The pyramid is 230 KB for that view. 64² samples were 2.0,
10.8 and 26 px (58 KB); 256² were 0.33, 2.2 and 5.4 px (918 KB).

## Measurements

Fixtures, native (Apple M6, Metal), after the change: 1032 pass, 55 fail,
109 skip of 1196 mirrored, against 1022, 27 and 147 before on the same
mirror (which now includes the `terrain` and `light` directories); no
previously passing fixture fails, and Chrome matches. Of the terrain
directory 8 pass, 19 fail and 12 skip; 13 of the failures differ by under
1% of pixels. All 12 `light` fixtures pass.
Two pitched variable-anchor text fixtures fail intermittently under full
suite load on main and with this change alike; they pass alone.

Frame cost over the Alps demo view (`orb-bench`'s `alps_terrain_frame_cost`,
1280×800, Apple M6, GPU timestamps, medians in ms):

| | CPU render | GPU total | terrain pass | main pass |
| --- | --- | --- | --- | --- |
| at rest | 0.31 | 0.94 | 0.41 | 0.28 |
| turning 0.2°/frame, tiles loading | 6.7 | 2.4 | 1.0 | 0.72 |
| redrawing all 19 slots | | 14.2 | | |

The full redraw's drape pass is 5.1 ms of that, its mips 0.35 ms.

Memory: a slot is 4 MiB plus a third for its mip levels; the atlas grows by
doubling and holds 64 slots at most on an 8192-texel device. The fixture
views use 4 to 12 terrain tiles.

## Demo

`?style=alps&flight` (web) and `orb-demo … alps.json --flight
alps-flight.json` (native) loop a flight over the Valais Alps: OpenFreeMap's
planet draped over Mapterhorn's terrain tiles (open elevation data, mainly
Copernicus GLO-30 and swisstopo there), with hillshade, peaks and fog.
The legs are geodetic `flyTo` targets, nothing in them Mercator's.

## Follow-ups

- Symbols and circles behind terrain: GL JS fades them by the terrain's
  depth; Orbis draws them still.
- Slots for far, small tiles could be smaller than 1024² texels.
