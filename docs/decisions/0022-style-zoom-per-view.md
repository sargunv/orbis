# 0022: Style zoom per view (updates D15)

D15 made style `zoom` the Web Mercator convention everywhere: on any surface,
a view's zoom was ground resolution at the camera's target against the
length of its parallel, `log2(2πa·cos φ / (512 · m/px))`. On Web Mercator
that is constant while panning, because the map's scale itself grows as
`1/cos φ`. On other planes the map's scale doesn't, so the same view's zoom
changed as it panned: dragging an EPSG:3413 map toward the pole lowered the
zoom by 1.5–2 levels with nothing on screen changing, zoom-dependent layers
came and went, and the camera store's zoom limits, applied to that number,
re-derived the resolution from it and zoomed the map in mid-drag (the bug
the owner hit on `?style=arctic`).

The owner delegated the call ("whatever makes the most sense for a next-gen
map engine; we owe MapLibre no API compatibility, so long as v8 styles work
on top of the scene document"), and the orchestrator decided the direction
recorded here. It changes D15's zoom clause; the rest of D15 (surfaces,
per-source tile matrix sets, the planner, the ground-resolution camera)
stands.

## Decision

- **Camera truth is unchanged**: a target and a ground resolution, with
  bearing, pitch and field of view.
- **Style zoom is a style-evaluation input derived per view**, not a camera
  quantity. `CameraState::zoom(projection)` and `set_zoom(projection, z)`
  take the view's plane; `Map::zoom()` and the camera store's `zoom()` give
  the map's.
- **Web Mercator views and the v8 globe keep GL JS's convention**, bit for
  bit (the globe's `1/cos φ` included), so v8 styles render there as
  authored.
- **Other flat planes** (NZTM2000, EPSG:3413, UTM, LCC, …) take the zoom at
  which Web Mercator's ground resolution at the plane's **reference
  latitude** equals the view's ground resolution there:
  `log2(2πa · cos φ_ref · k_ref / (512 · world/px))`, `k_ref` the plane's
  scale factor at the reference. It is one number for the whole view and
  constant as the view pans, so a style's zoom stops mean roughly the map
  scale they mean in Web Mercator near the plane's area of use.
- **Reference latitude per family**: where the family is true to scale, else
  its natural origin. Polar stereographic: the latitude of true scale
  (EPSG:3413 70°N, EPSG:3031 71°S), or the pole for variant A (UPS), past
  ±85.0511° held at that latitude's scale. Lambert conformal conic and
  Albers: midway between the standard parallels. Lambert azimuthal
  equal-area: the origin. Transverse Mercator and UTM: the natural origin
  (true to scale along the whole central meridian, so it has no latitude of
  its own; for NZTM2000 and UTM that is the equator, so NZTM2000's zoom at
  Wellington is 0.4 above Web Mercator's for the same ground resolution).
  Equirectangular, geographic, Equal Earth and Natural Earth: the equator.
- **Zoom limits** (`minZoom`, `maxZoom`) bound the same zoom, so they hold
  still while panning too.
- **Tiles lay out at the view's zoom where they span 512 pixels**: on a
  plane, its set's tiles at one zoom per level (NZTM2000Quad's at
  `z + 2 + log2(0.9996)`); warped tiles at the zoom of their footprint's
  scale on the surface; `WebMercatorQuad` tiles on Web Mercator at their own
  zoom, as before. Tile level of detail still comes from ground resolution
  and each source's set.
- **No override option now.** The scene document is where a style could
  choose another convention if one is ever needed; `["resolution"]` already
  gives resolution-native styling.

## Panning

A drag is rigid: it keeps the map's scale (world units per pixel), and on a
plane whose scale factor varies, ground resolution at the target then moves
with that factor (EPSG:3413: about 3 % between 70°N and the pole). The
style zoom, measured at the fixed reference, does not move at all.

## Evidence

- `orb-camera`: dragging a camera with a minimum zoom across the pole of an
  EPSG:3413 plane keeps world units per pixel and the style zoom to 1e-9
  while ground resolution follows the plane's scale factor, and zooming out
  stops at the limit; Web Mercator's zoom equals GL JS's formula bit for bit
  at four latitudes; NZTM2000 and EPSG:3413 zooms are measured at their
  references and are the same anywhere on the plane.
- GL JS fixtures natively: 1,032 pass, 55 fail, 109 skip, the same failures
  as before the change but two of #18's pitched variable-anchor flakes,
  which passed this run.
- Demos keep their framing with new zooms in their scene documents
  (`arctic` 2.4 → 2.9, `nz` 5.2 → 5.6).
