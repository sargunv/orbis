# 0008: Shared image pages and deferred image references

Sprites and patterns use the same Runtime-owned RGBA8 texture array, with a
stable 32-byte image table row containing the page, rectangle, pixel ratio and
SDF flag. The initial allocation is eight 1024 × 1024 pages (32 MiB) and 8192
rows. A shelf packer adds one texel of wrapped edge padding for patterns,
transparent corners, and premultiplies straight host pixels on upload. The
atlas also supports transparent icon padding. Pixel dimensions, stretch,
content and text-fit metadata remain available for M2.

A tile job returns image names and property-word offsets, including the image
at layout zoom minus one, layout zoom, and layout zoom plus one. The scene
resolves these names into stable table ids after uploading either core-owned
or host-owned property rows. Resolving a missing image reserves a transparent
row; a later sprite upload fills that row without rebuilding geometry or
recording commands. Image table, texture array and sampler are part of the
existing pass bind group, shared across every tile and family. No tile job
waits for decoding or atlas allocation.

Every loaded style gets its own namespace, including maps sharing host
services. Style replacement and map destruction retire that namespace. The
next render zeros its rows, recycles their ids, and resets a page when its last
resident image leaves. Pages do not move live rectangles. Exhaustion reports a
warning and leaves the affected image transparent; camera changes allocate no
image resources. Atlas sharing is scoped to maps using the same GPU device.

Images a host adds at run time (`Map::add_image`, `update_image`,
`remove_image` and `has_image`, GL JS's `addImage` family) join the sprites'
images in the style's namespace. Removing or replacing an image empties its
rows in place. The ids stay the same, so tiles holding them read the new
pixels, or a transparent row, without new geometry, and pattern layers aren't
laid out again. Symbol layout does read image sizes and metadata, so adding
or removing an image makes a style revision with no document change, which
lays the symbol layers out again from cached tiles. `image` expressions see
the style's images as available, as GL JS's do. Constant expressions that read
`image` are no longer folded at parse time, so `coalesce` picks the first
image the style has. When a tile's layers ask for an image the style lacks,
the map reports the name once as `MapEvent::StyleImageMissing` (GL JS
`styleimagemissing`), and again only after it has been added and removed.

The alternative of baking sprite coordinates into tile vertices would make
layout depend on the atlas and require geometry rewrites when it changes.
Per-tile textures would require per-tile binding changes. Both conflict with
plan §8.2. Indirection permits future icon and glyph consumers without changing
the tile submission model.

Pattern sampling follows GL JS commit
`2b68611a4e2d4f7f076f6c6ebf758e2d53f91471`: split global pixel offsets,
integer-zoom scaling, floor line width, two-image zoom crossfade (300 ms), SDF
fills and patterned fill outlines. Background layers use ordinary fill nodes
with persistent quads in their actual stack positions. Backgrounds, solid or
patterned, draw on every covered tile as GL JS's do; nothing is drawn beyond
the world's edge near the poles (this replaced a single full-viewport quad
for solid backgrounds, which covered that area where GL JS leaves it
transparent). Patterned backgrounds use tiled sampling. Their camera work is placement-only.

That pinned GL JS also slices vector ancestors into child tiles above source
maxzoom before layout. Orbis now does this directly on TileFrames, retaining
properties and rounding clipped intersections on the normalized 8192 grid.
This resets pattern distance per child as the reference does. Camera coverage
uses the reference's constant zoom through its FOV-dependent pitch threshold
(at most 60 degrees), then its default variable zoom policy. Stretching the
ancestor mesh had visible line-pattern phase differences at pitch.

The runtime publishes tiles in the pyramid's stable draw order. Mesh arrival
and arena slot reuse must not change blending at tile edges when a bounded
scene reloads evicted tiles. The scene keeps meshes in that order for all three
submission paths; camera-only frames with unchanged coverage do no new sorting,
uploads or command recording. Upstream's GPU regression covers reverse arrival,
and its bounded/unbounded demo test compares exact pixels (decision 0006).

The initial mirror expands from 99/101 passing to 131/133 on native and
Chrome, with all 32 newly enabled cases passing. On the 422-fixture
mirror including the circle/heatmap work, both pass 271/275 (147
skipped), up from the recorded 237/241 baseline. All 34 newly enabled cases
pass, and all 271 native passes match Chrome.
The GPU test covers late row population, multiple array pages, namespace
retirement and row reuse without bundle recording. The demo's camera-only
test covers solid and patterned backgrounds. Main and worker browser modes
produce identical pixels for pitched lines and data-driven fill patterns.

Against a build without this work, `core.wasm` grows from
398,484 to 414,340 bytes Brotli (+15,856 bytes, 15.48 KiB), leaving 46,460
bytes below the 450 KiB gate. Raw wasm grows from 1,224,682 to 1,271,395
bytes. This delta includes low-zoom and publication fixes made alongside
it. Both builds use the pinned toolchain and the size gate's
Brotli compressor. `cargo xtask check` passes, including all nine demo camera,
fresh/moved snapshot and bounded-scene eviction tests.
