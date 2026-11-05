# 0036: Recent style-spec proposals: what the native model should take

This note surveys the open maplibre-style-spec design proposals in the
rendering lane's area. For each, it records whether Orbis's native model
should take it and how it would land on what exists. Nothing here is
implemented or scheduled. The owner sets priorities. The rule throughout
is the one decisions 0025 and 0026 follow: a feature lands once, in the
native model, and v8's spelling lowers onto it if the spec adopts one.

## Rendering lane

**#1881, a `["scale"]` expression (metres per pixel at the target).**
- *Take.* The plan already has a `resolution` input for scene-document
  layers (§9.3, and §11.3's dependency bitset), and decision 0022 makes style
  zoom per view.
- `scale` is v8's spelling of the same input: ground resolution at the
  target from the view's tile matrix set. It means the same on any CRS
  and on the globe, where zoom doesn't.
- It is camera data, like zoom, so it rides the existing `CameraCurve`
  binding with no relayout.
- Later, exact per-vertex ground sizing could evaluate it on the GPU per
  vertex, for pitched views.

**#1855, variable-width lines.**
- *Take the idea, not the four properties.* Let `line-width` read
  `["line-progress"]`, evaluated per vertex as `line-gradient` already
  reads it. That covers start and end tapers, arrows and brush strokes
  with one property and any curve.
- Absolute per-vertex arrays (`line-widths`) need a per-vertex data input,
  which is a larger addition. Defer it until a style needs it.

**#1797, extrusion shadows.**
- *Take, as real shadows from the lights model*, not per-layer 2D
  projections with their own azimuth. The plan's `Shadow` node (M8)
  casts from the directional light of decision 0026. That light is the
  sun at a time when the scene has one. Shadows fall on the ground,
  terrain, other buildings and models.
- Per layer: the scene document's `layers.<id>` gains cast and receive
  flags; the lights gain a shadow intensity.
- If v8 adopts `fill-extrusion-shadow-*`: opacity and colour lower to the
  layer's shadow strength and tint. Azimuth and length lower to a shadow
  direction only when no scene light gives one.

**#1796, ambient occlusion for extrusions.**
- *Take, in two steps.* First, geometric contact shading at wall bases
  and between adjacent walls, per vertex at layout. It is cheap and
  works on Compat. Second, screen-space AO over the 3D content's depth
  once models arrive in M8.
- Both live in the lights module as an ambient-occlusion intensity and
  radius, applying to all 3D content, not only extrusions.

**#1795, a configurable extrusion gradient.**
- *Take, small.* A wall's darkening runs along the wall, from base to
  top, with a depth and a reference height. GL JS's height-scaled
  darkening stays as the lowering of `vertical-gradient: true`. It belongs
  with the lighting model's response (decision 0026's `lighting.wgsl`).

**#1374, terrain coloured by slope, aspect and curvature.**
- *Take; it fits well.* DEM preparation already computes slopes into
  texture arrays for hillshade (plan §8.5, `Prepare`). Expose `["slope"]`
  and `["aspect"]` beside `["elevation"]` as inputs to `color-relief-color`.
  Curvature needs a second-derivative term in the same prepare pass.
- Matcap and Phong-style shading belong to the lights model, applied to
  the terrain surface, not to a colour ramp.

**#1490, pixel styling for any raster.**
- *Take; Orbis is well placed.* Generalize color-relief's per-texel
  colouring into a raster colour expression over band inputs
  (`["band", n]`), compiled by the GPU bytecode interpreter (decision
  0017) and run per texel.
- The real work is data: float and multi-band textures (R32Float, COG
  and GeoTIFF bands) through the image pages. Its value for Earth
  observation is large.

**#1554, line stroke properties.**
- *Take.* A stroke drawn in the same pass as the line's fill, from the
  same distance to the centre line, as circles do. Within a layer, a
  later line covers an earlier line's stroke and fill, which is what the
  proposal asks for.

**#1642, colour for SDF fill patterns.**
- *Take, small.* The image atlas already holds SDF icons. A pattern
  sampled from an SDF image mixes `fill-pattern-color` (inside) and
  `fill-color` (outside) at the SDF threshold. The same applies to
  `line-pattern` and `background-pattern`.

**Layer blends, stars and a backdrop** (#1798 and #1889) have landed in
decisions 0033 and 0035.

## For the text and symbol lane

- **#1615, SVG icons, and #1605, colour-font icons.** Take vector icons,
  rasterized into the image atlas at the device's pixel ratio and the
  sizes in use, and redrawn when those change. SVG via a Rust rasterizer.
  COLRv1 colour glyphs through the text stack into the same atlas. One
  path serves both, so neither needs a vector renderer on the GPU.
- **#1898, `text-repeat-distance`.** Take: it separates GL JS's
  hard-coded `symbol-spacing / 2` repeat distance. Small, in layout.
- **#1844, `text-writing-mode` on line placement.** Take: line labels
  pick vertical or horizontal per label from the line's angle, among the
  allowed modes. Small, in line-label placement.

## For R1 (projections)

- **#1623, more projection values, and #1865, adaptive Equal Earth.** The
  any-CRS planner already draws Equal Earth as a plane (the Equal Earth
  demo). An adaptive Equal Earth → Mercator is a blend of two planar
  projections in the vertex shader, as the globe already blends with
  Mercator. v8's values would lower onto the planner's projection
  descriptors.
- **#1866, simpler adaptive projection configuration.** `globe` with an
  optional transition range maps onto `orb_crs::Globe`'s zoom
  interpolation as it is. Only the parsing changes.
