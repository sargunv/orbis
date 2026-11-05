# 0030: The globe: v8 `projection` on a sphere, through patches in space

Plan §9.1 adds a `Sphere` surface for v8 `globe`, §9.2 the `GpuPerVertex`
plan on surfaces and §9.3 adaptive projections. This records how they are
built, and what is left.

## The surface

`orb_crs::Surface` is a plane (a `Projection`) and an optional `Globe`: a
sphere the plane's tiles wrap around, with the zooms it hands over to the
plane between. v8 `projection` parses to it: `globe` is GL JS's
interpolation from a vertical perspective through zoom 11 to Mercator from
zoom 12, `vertical-perspective` a globe at every zoom, and that
interpolation with other zooms is accepted too; other types warn and stay
Web Mercator. Style zoom stays the plane's (decision 0022), so on Web
Mercator the globe keeps GL JS's `1/cos(lat)` scale: at the target the
globe and the plane show the same ground resolution.

The sphere has the WGS84 semi-major axis, so a Mercator world unit at the
equator is a metre on it. Globe space is ECEF metres less the camera
target's position, kept in f64 on the CPU and as camera-relative f32 on the
GPU. The view matrix is the plane's, turned from the target's east, north
and up into ECEF; the CPU's `View` projects plane positions through it and
unprojects by meeting the sphere, so camera moves, picking, labels and
hosts keep working in plane coordinates.

## Tiles on the sphere

Each tile draws through biquadratic patches in space (`sphere::series`):
1–256 patches over the tile through exact ECEF positions at their 3×3
nodes, within a quarter pixel at 512 pixels per tile, 27 f32 each. The
runtime keeps them per tile matrix set and tile and uploads them when a
placement first needs them. Patches now belong to placement rows rather
than scene tiles, so the background's one mesh places on every covered tile
with each tile's patches. A row carries its patches and the camera-relative
position of their anchor; rows carry them beside any planar patches (see
below).

Shaders place everything through one function, `ground_clip`: a tile point
on the plane and on the sphere, moved by a map offset along east and north
and raised along up there, and the two clip positions mixed by how much of
a globe the view is (GL JS's `u_projection_transition`). Depth on the globe
comes from the horizon plane, as GL JS's `globeComputeClippingZ` does, so
the far side lies past the far plane. Vector geometry is cut along each
tile's patch grid in jobs, so straight edges bend with the sphere; image
tiles and the background draw as grids for the same reason.

Labels place through each tile's patches (`orb_symbol::TileGlobe`), exactly
at their anchors, and are hidden past the horizon. The cover keeps tiles
whose bounding caps reach the eye's side of the horizon and whose bounding
boxes meet the frustum, each at the zoom whose tiles span 512 pixels at its
distance.

A projection edit (`setProjection`) changes the surface at once; vector
tiles lay out again from the jobs' cached data for the new curvature, and
image tiles and the background take their grids on the next frame. Nothing
loads again.

## Measured

On an Apple M-series laptop, native:

- Projection switch at zoom 9 on the Tokyo extract: the first frame draws
  the new surface; 0 tile loads, 7 relayouts, idle after about 300 ms
  (label fades included); pixels equal the style loaded with the new
  projection (`projection_switches_load_nothing`).
- Cover, 15 cameras from zoom 1 to 11 at pitch 0–70°
  (`globe_cover_cost`): 4–45 globe tiles against Mercator's 6–18, in
  0.005–0.065 ms against 0.001–0.002 ms. GL JS's ratio (50.7 Hz against
  157 Hz, plan §18 budget 21) does not carry over: Orbis's Mercator cover
  takes about a microsecond, so the budget is restated as the globe's
  absolute cost, well under a tenth of a millisecond.
- Fixtures: native results equal `main`'s, except the three
  `sky/atmosphere-blend` fixtures, which used to skip on
  `property:projection`; they run since the globe's atmosphere joined
  (decision 0025).

## Gestures

Drags, zooms and rotations around a point turn the camera around the
sphere so the ground grabbed stays under the pointer, keeping the zoom and
bearing (GL JS's `versorSetLocationAtPoint`): an arc rotation from what
shows under the pointer to the ground grabbed, then Newton steps on the
target's longitude and latitude, within 0.05 px near the poles and across
the antimeridian. Inertia glides go through the same pan.

## Pole caps

Web Mercator stops at ±85.05°. As in GL JS, the world's top and bottom row
tiles extend to the poles: fills fan their edges along the world's edge to
a pole vertex in jobs, image tiles' grids fan their edge texels, and the
background's grid fans both edges. A pole vertex has y −32768 or 32767
(orb-layout's `NORTH_POLE_Y`, `SOUTH_POLE_Y`), the sentinels GL JS uses,
but read explicitly: the shader puts it at the pole only where the row's
clip rectangles reach past the world's edge (the placement extends an edge
region's rectangle to the pole), and on the tile's edge, as a degenerate
triangle, everywhere else and on the plane.

## Sources on other planes

On a globe surface every tile draws in its own tile space: on the sphere
through its patches in space (the `Geodetic` plan for geodetic sets), and
on the plane, at and past the handover, through its planar patches. Rows
carry both (112 bytes, 28 words); image tiles of warped sources draw as
grids then, rather than as the plane's `RasterWarp` mesh. A tile without
planar patches (one holding a pole, which Web Mercator lacks) has no place
on the plane: it draws on the globe alone and drops out when the globe
hands over. The demo's Blue Marble globe draws GIBS's EPSG:4326 tiles this
way, to the poles.

## Left for later
- Terrain on the globe, and extrusions' depth (the horizon plane's depth is
  GL JS's, not perspective depth).
- Globe gestures: panning drags plane coordinates, which behaves near the
  target but not as GL JS's rotation near the poles.
- The mirror has no `projection/globe` fixtures yet; GL JS's directory
  needs a prefix in `xtask/src/fixtures.rs`.
