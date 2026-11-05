# 0041: Adaptive projections: low-zoom planes over Web Mercator

Equal Earth is the right whole-world map and the wrong street map: at z13
over Chicago (−87.6°, far from its central meridian) it shears the street
grid, as an equal-area projection must. style-spec#1865 proposes
`{"type": "equal-earth"}` as Equal Earth zoomed out that blends linearly to
Mercator over `[6, 7]`; style-spec#1866 proposes
`projection.transition: [start, end] | false` for any projection. This builds
both, extending decision 0030's surface rather than adding a mode.

## Decision

- **A surface is a plane and a low-zoom surface that hands over to it**:
  `orb_crs::Globe` gains `plane`, the low-zoom plane, or `None` for the
  sphere. The amount of the low-zoom surface in a view is a pure function of
  its style zoom (the plane's, decision 0022): 1 through the start, 0 from
  the end, linear between.
- **v8 `projection.type`** takes `equal-earth`, `natural-earth` and
  `equirectangular` (low-zoom planes over Web Mercator, `[6, 7]` by default)
  beside `globe` (`[11, 12]`) and `vertical-perspective` (the sphere at every
  zoom), and `transition` replaces the zooms (`false`: the low-zoom surface
  at every zoom). GL JS's interpolate form still parses where it is one
  handover from a low-zoom type to `mercator`.
- **The scene document's `projection`** takes `transition` too. Equal Earth,
  Natural Earth and equirectangular adapt by default there; national and
  conformal planes (TM/UTM, LCC, polar stereographic, any `EPSG:` code,
  including Equal Earth's 8857) stay fixed unless a `transition` is given,
  since their data may really be on that plane. An adapting projection makes
  the scene's plane Web Mercator with that plane at low zoom; the scene's
  choice wins over the style's.
- **Drawing**: as on the globe, each tile's row carries patches on the
  low-zoom surface (for a plane, its world positions at height 0 through
  the tile's warp onto it, 27 f32 a patch as the sphere's), and
  `ground_clip` mixes the two clip positions by the amount; one frame word
  marks a plane, which has no horizon, poles or sky of a sphere. Geometry is
  cut along the low-zoom patch grid only in tiles laid out below the end
  zoom (the only ones a low-zoom surface can show), so above it nothing of
  the warp is computed: plain Mercator.
- **The CPU uses the low-zoom plane while it shows at all**: the view holds
  the camera's view of that plane, and projection, unprojection, the cover
  (through each tile's cached footprint on it), labels and gestures
  (keeping the ground grabbed under the pointer, at the zoom) go through it.

## Measured

- Native snapshots of the `equal-earth` demo over Chicago: Equal Earth at z2
  and z5, the half-way blend at z6.5, Web Mercator at z8 and an unsheared
  street grid at z13.59.
- `orb-camera`: on the Equal Earth low plane at z5 and z6.5 (pitched 30°),
  positions round-trip through unprojection to 1e-6 px; drags keep the
  grabbed ground within 0.05 px and the zoom to 1e-9.
- Mirrored `projection/` fixtures natively: unchanged (55 pass, 50 fail,
  30 skip), the same tests as `main`.

## Not done

- Terrain's cover on a low-zoom plane computes footprints without the cache
  the sources use.
- Map-aligned offsets (translate) on a low-zoom plane move along the
  plane's grid, not the local east and north, which Equal Earth shears far
  from its central meridian.
