# 0049: Contour lines worked out on the GPU

#91's terrain item: contour lines from a `raster-dem` source, without the
worker-side isoline tracing MapLibre's contour plugin does.

## Decision

- **A `contour` layer in the scene dialect.** Like `model`, a scene
  document's style may have it and a v8 document alone may not. Its source
  is a `raster-dem`; its paint is `contour-interval` (metres),
  `contour-major` (every this many lines is a major one), `contour-color`,
  `contour-width` and `contour-major-width` (pixels) and `contour-opacity`.
- **Per fragment, from the DEM the image families already read.** A
  contour node draws the source's DEM tiles as color-relief does, from the
  same padded page (`DEM_PADDED`). A fragment's elevation over the interval
  is a level; its distance in pixels to the nearest line is that level's
  fraction over the level's change across a pixel, which antialiases the
  line at its width. Where lines would close to under four pixels apart
  they fade out by two, so steep ground and far slopes under pitch shade
  rather than fill; flat ground draws none.
- **Continuous across tiles.** Each tile's DEM border is filled from the
  same-zoom neighbours the view reads (#84, c44302b4), and every tile
  samples it as color-relief does, so a line leaving one tile enters its
  neighbour where the neighbour's elevation says, whatever loaded before.
  Neighbours of different zooms hold different data; their lines meet as
  closely as the data does.
- **One interval per frame.** `contour-interval` is evaluated at the
  camera's zoom, never a tile's, so a pitched view's tiles of several
  zooms share it; it steps with zoom but doesn't interpolate, since an
  interval that slid would slide every line. A step is the style's own
  zoom-driven change.
- **Draped like any flat layer.** On terrain, contours draw into the drape
  atlas, which the terrain's lighting shades; flat, they draw in the main
  pass under the lit map's lights like color-relief.
- **No labels yet.** Labels along lines need line geometry the symbol
  path places on the CPU; GPU contours have none, so labelling them is a
  follow-up (#108).

## Measured

Apple M6, 2048×1536, `orb-bench` over Zermatt with the Alps hiking scene
(GPU median, two runs equal): flat, contours cost 0.19 ms a frame at z13.2
pitched 45° (1.13 → 1.33 ms) and 0.18 ms at z14.5 (1.02 → 1.20 ms). On
terrain they draw into drape slots, so a steady view costs nothing more
(2.78 → 2.79 ms, 2.16 → 2.16 ms); a slot pays them when it redraws.
`contours_after_a_gesture_are_a_fresh_map_s` (orb-api's path tests)
checks a gesture through a change of interval, flat and draped, against a
fresh map at rest.
