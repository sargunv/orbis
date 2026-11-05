# 0034: Image sources: one tile, a warped grid, on every surface

v8 `image` sources put one image on four geographic corners. GL JS draws
one tile holding the corners' quad and warps the image onto it in the
raster shader: projectively where the corners describe a perspective view
of a plane, blending toward a bilinear rubber sheet as they stop doing so
(`calculateImageWarp`), over a 16×16 mesh when the warp isn't affine.

## How Orbis draws them

An image source is a raster endpoint with one tile: the image's URL as its
only template, in a one-tile matrix set whose square holds the corners'
quad on the surface's plane. Fetching, decoding, texture pages, cover,
fades and terrain draping are the raster path's.

The tile's mesh is a grid over the image (16 cells a side, or as fine as
the sphere bends on a globe), each vertex at GL JS's warp of its image
position, computed on the CPU (`orb_crs::ImageQuad`) inside the tile. The
tile itself is affine on the plane, and on a globe it takes the set's
sphere patches like any tile, so the image is right on every surface.
Image vertices gained two words: a weight, GL JS's `perspectiveRatio`, so
image positions interpolate projectively across each triangle, and the
image's own units apart from the tile units clipping uses (they differ only
for an image source, whose grid sits in its tile). Image sources sample
the full-size level, without mip levels, as GL JS binds them.

Patches fitted to the warp itself were tried first; a near-degenerate quad
(GL JS's `near-triangle`) has a denominator near zero that no patch series
fits, and the per-vertex grid is GL JS's own approach.

## Results

The mirror gains GL JS's `image/` and `video/` render tests. On native and
Chrome, `image/` passes 18 of 19 (`near-triangle` differs by 0.13% of
pixels inside cells); `terrain/image*` and `projection/globe/image*` pass
but for `terrain/image-projective` (0.71%, drape resampling). Raster,
hillshade and color-relief fixtures are unchanged. `video` sources still
skip: decoding frames is a host service (#67).
