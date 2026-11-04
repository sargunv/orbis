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

## Draping: slots of one atlas, drawn by drape placements

Layers GL JS renders to texture (background, fill, fill outline, line,
raster, hillshade, color-relief) drape. A run of them between layers that
draw in 3D or on screen is a drape stack. Each terrain tile owns a block of
drape slots, one per stack: 1024² texel squares (GL JS's quality factor 2
over 512 pixels) of one mipmapped atlas in the target's colour format.

The GPU-driven design is kept by making drape slots ordinary placements:

- A tile placement has a target, the map or drape stack `s`
  (`PlacementTarget`). Draped nodes draw only their stack's placements;
  the cull keeps those by clip rectangles alone (`DRAPED` on the node's
  draw word).
- A drape placement maps a source tile's data into a slot, clipped to the
  terrain tile, in drape space: atlas texels with y up. The drape frame is
  the map's with an orthographic atlas projection.
- Pixel sizes now convert with a per-placement `world_per_px` (map
  placements carry the camera's; drape placements scale it with their
  slot), since terrain tiles of several zooms share the atlas. Tile rows
  grow from 64 to 80 bytes: that, the DEM reference and the terrain range.
- One render pass redraws every slot marked this frame: it clears them,
  then replays a bundle of the draped nodes over the drape scene group.
  A pass per mip level downsamples them. Slots are aligned, so levels
  never mix slots, and the terrain shader clamps samples to the slot at
  the level it reads, as a texture of its own would clamp to its edge.

A slot is redrawn when the draped content over its tile changes (drawn
tiles, their meshes, the style revision, feature-state writes) and, when
only the zoom or pattern cross-fade changed, once the camera stops: GL JS's
fingerprint policy (#7549). Unchanged slots cost nothing per frame.

Each stack's terrain draws into a layer of its own sharing the 3D depth
with extrusions, then composites into the map at the stack's place, as
extrusion layers do (decision 0011). GL JS's terrain fog (sky `fog-*`,
`horizon-*`, faded in from 60° to 70° of pitch) is applied there.

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
- Gestures hold the elevation; when they end the target moves to the
  terrain along the view with the eye in place (`recenter_on`, GL JS's
  `recalculateZoomAndCenter`), so the picture doesn't move.
- The eye keeps 50 m above the terrain (`keep_above`).
- The far plane reaches the lowest elevation of the DEM tile under the
  target, as GL JS's `minElevationForCurrentTile` makes it; without it the
  Dead Sea fixtures lost their far terrain.
- `View::unproject_terrain` walks the view ray over the CPU elevations
  (plan §9.4's CPU ray walk). Gestures drag the plane through the target.

CPU elevations are decoded once in the DEM tile's job (`DemGrid`) and
sampled as GL JS's `TerrainCoverage` does: bilinear in the DEM tile the
drawn terrain tile reads, past tile edges in its neighbours. On the web a
job worker reads a DEM tile's texels once for this and transfers them as
f32s beside the tile's `ImageBitmap`.

## The web's ui role and placement worker

Threads without the map's DEM tiles get the elevation pyramid (plan §7.1):
each DEM tile resampled to 128² samples at pixel centres, quantized to 16
bits per tile, with the drawn tiles. The render worker publishes it with a
frame once the camera settles after the terrain changed; the ui role's
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

Fixtures, native (Apple M6, Metal), after the change: 1031 pass, 56 fail,
109 skip of 1196 mirrored, against 1022, 27 and 147 before on the same
mirror (which now includes the `terrain` and `light` directories); no
previously passing fixture fails. Of the terrain directory 8 pass, 19 fail
and 12 skip; 14 of the failures differ by under 1% of pixels, and the
symbol cases wait for symbols on terrain. All 12 `light` fixtures pass.
Two pitched variable-anchor text fixtures fail intermittently under full
suite load on main and with this change alike; they pass alone.

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
