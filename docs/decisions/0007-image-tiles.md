# 0007: Image tiles: decode, upload and DEM preparation

Status: implemented for raster, hillshade and color-relief layers.

Raster and raster-dem tiles are images, not meshes. The questions were where
they decode, how their pixels reach the GPU without passing through the
render worker's wasm memory (M0 exit 5, plan §10.4), how shaders address
them, and where GL JS's per-tile offscreen hillshade pass goes.

**Decode on the job side, through the host.** Jobs fetch tile bytes as for
vector tiles (URL templates, TileJSON, `pmtiles://` with PNG, JPEG and WebP
archive types), then decode with the host: `image` natively, behind the
`ImageDecode` service. In the browser's worker mode the job worker calls
`createImageBitmap` (straight alpha, no colour conversion) and transfers the
`ImageBitmap` to the render worker, which registers it with the HAL and
copies it into a texture layer with `copyExternalImageToTexture`. Pixels
enter neither worker's wasm memory: on a 512 × 512 view with raster,
hillshade and color-relief layers the render worker copied 2,304 KiB of
texels from bitmaps and 0 from wasm, and read 0 KiB of packed headers.
Main-thread mode keeps the existing `ImageDecode` path, reading pixels back
through an `OffscreenCanvas`; its frames match worker mode pixel for pixel.
`copyExternalImageToTexture` requires `RENDER_ATTACHMENT` usage on the
destination, which the uploaded page arrays carry.

Rejected: sending decoded pixels as host bytes (`ArrayBuffer`) through the
existing buffer path. It would need a readback in the job worker (canvas
`getImageData`, a premultiply round-trip) and a texel copy per tile, where a
bitmap transfer moves a handle.

**Texture-array pages per source.** Each image source gets a page, sized by
its first tile: one texture array for rasters (premultiplied, as GL JS
uploads them), three for DEMs sharing layer numbers (raw elevations, a copy
padded by two texels, hillshade slopes). Pages start at 8 layers and double
up to the device limit, copying their layers on the GPU; a node's bind group
holds its source's arrays and is rebuilt, with the bundle, only when a page
grows. A tile is a one-chunk quad mesh whose property row names its layer, so
image tiles share the cull, placements, fallback clip rectangles and eviction
of vector tiles. Image sources cache 32 ready tiles beyond those in view.

**`Prepare` as compute, on arrival.** GL JS backfills each DEM tile's border
from its neighbours on the CPU, then re-renders the tile's slopes in an
offscreen pass. Orbis queues a prepare job when a DEM tile arrives, and for
each resident same-zoom neighbour: a `pad` dispatch fills the two-texel
border from the eight neighbours' raw layers, or the tile's own edge where a
neighbour isn't resident, exactly as `DEMData` initializes and backfills; a
`slope` dispatch ports `hillshade_prepare`. One dispatch of each runs per page
over every pending tile, in frames with arrivals only; camera-only frames
still run one cull dispatch. Hillshade samples the slopes and color-relief
the padded elevations, as GL JS does.

**Pixel alignment.** GL JS draws image tiles with a projection whose
translation snaps the camera centre to a whole pixel. Without it, 20 raster
and 17 hillshade fixtures that now pass failed on scattered edge pixels; the
frame block carries the equivalent world offset (`align`).

**Mipmaps.** Raster pages carry mip levels on Modern, as GL JS's
`useMipmap` does. After uploads, one compute pass makes each level of the
new layers from the one above with a 2×2 box filter; odd edges clamp. Page
growth copies every level. Raster fragments sample with their uv gradients
(taken in uniform control flow), blending levels with up to 16× anisotropy.
GL JS samples `LINEAR_MIPMAP_NEAREST`, anisotropic above 20° of pitch.
`raster-anisotropic-filtering/pitch-45` passes. `raster-npot-mipmap/pitch-60`
goes from 19% to 1% of pixels off: the blended levels differ slightly from
WebGL's `generateMipmap`. Compat keeps one level: WebGL2 has no storage
textures for the pass.

Raster tiles fade in over `raster-fade-duration`, as GL JS's
`getFadeValues` does. The fade is decided once, when a tile arrives, and
written into its row: the time it arrived (the frame block's clock,
milliseconds since the map began) and the resident ancestor it cross-fades
from. The shader computes the fade from the frame's time, so frames that
only move the camera write nothing.
- **An ancestor covered the tile** (zooming in): it cross-fades from the
  ancestor's texels, in the same page.
- **Nothing covered it**: it fades in by opacity.
- **Resident descendants covered it** (zooming out): it shows at once.
  GL JS keeps the children drawn under a fading parent, and the pyramid
  holds no such retention, so a fade-in there would flash the background.
- **Exempt:** image sources don't fade, as in GL JS. Neither do drape
  slots: the drape frame's time is past every fade, as GL JS doesn't fade
  what it renders to texture.

The map isn't idle until the last fade ends. The render test
`raster_tiles_cross_fade_from_their_ancestor_or_fade_in` checks both
paths halfway and at the end. The fixtures run with the duration at 0, or
draped, and are unchanged.

Deviations: hillshades
draw at most four light sources, with a warning; a 204 response is treated as
a missing tile (parents show through) rather than a blank one.
