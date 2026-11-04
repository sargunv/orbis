# 0020: Any CRS: planes, tile matrix sets and the planner

M6 makes the surface any plane and every source any tile matrix set (plan
§9.1–9.3). This records how the plan's CRS model was built, the choices the
plan left open, and what remains.

## Projections

`orb-crs` has the planar families of plan §9.3 as f64 forward and inverse
maps, private behind `Projection`: ellipsoidal transverse Mercator (Krüger's
series to sixth order, as PROJ's default `tmerc`), polar stereographic
(variants A and B), Lambert conformal conic, Albers, Lambert azimuthal
equal-area (oblique and polar), Equal Earth (on the authalic sphere),
Natural Earth, equirectangular and geographic. Orthographic and vertical
perspective come with the globe (M7); identity waits for a consumer.

Projections come from descriptors: a `type` naming a kind with PROJ-named
parameters (`{"type": "lcc", "lat_1": 33, "lat_2": 45, "lon_0": -96}`) or an
EPSG code (`{"type": "EPSG:2193"}`); `descriptor()` gives back the same
form. The EPSG table lists 30-odd codes on WGS 84, GRS 1980 datums (WGS 84,
NZGD2000, ETRS89, NAD83, GDA94, RGF93, SWEREF99, EUREF-FIN), which are
within a metre of each other, so no datum shift applies. Codes on other
datums (Amersfoort / RD New, British National Grid, Swiss LV95) need a
transformation and are unknown: errors, never a fallback
([#34](https://github.com/sargunv/orbis/issues/34)).

**Agreement.** All 39 descriptors (every EPSG entry and each parametric
kind) agree with PROJ 9.8.1 on 975 positions across their areas of use:
forward within 0.1 mm, inverse within a nanodegree
(`crates/orb-crs/tests/proj_reference.py` writes the reference).

**Ground metres and zoom.** `scale_factor` is world units per ground metre
along the parallel, measured on the sphere of the semi-major axis: the
sphere style zoom is defined on, so resolution, zoom and scale agree in
every projection (on the ellipsoid it would differ from Web Mercator's own
convention by up to 0.3 %). Style zoom, at first Web Mercator's convention
on every plane, is per view since [0022](0022-style-zoom-per-view.md).
Bearing turns the plane's grid north, the axis a projected map's north is
drawn along.

## Tile matrix sets

`TileMatrixSet` is an OGC 2D-TMS whose matrices form a quadtree: each halves
the previous one's cell size and doubles its columns and rows from the same
corner. The first matrix may be of any size and zoom (`WorldCRS84Quad` is
2×1, the UTM grids start at zoom 1 with 1×2), so tile indices keep the
quadtree's parents and children that fallback, overzoom and caches rely on.
Definitions parse from OGC JSON (2.0, and 1.0's `identifier` and
`topLeftCorner`), with the traps of plan §9.2: axis order from `orderedAxes`
or the CRS (EPSG:2193 and EPSG:3035 list northing first), bottom-left
corners counting rows from the bottom, and refusals for coalesced rows
(`GNOSISGlobalGrid`) and scales that don't halve (`CanadianNAD83_LCC`).
Registry: `WebMercatorQuad`, `WorldCRS84Quad`, `NZTM2000Quad`,
`EuropeanETRS89_LAEAQuad`, `UPSArcticWGS84Quad`, `UPSAntarcticWGS84Quad` and
`UTM01WGS84Quad`–`UTM60WGS84Quad`. Non-quadtree sets are
[#29](https://github.com/sargunv/orbis/issues/29).

Tile URLs take `{tileMatrix}`, `{tileRow}`, `{tileCol}` and `{bbox}` (in
the set's units) beside `{z}{x}{y}`. A tile lays out at its **style zoom**,
the view's zoom where it spans 512 pixels (exactly its own zoom in
`WebMercatorQuad`, so Mercator styles are unchanged; per decision 0022 on
other planes), and keeps the layers that the zoom range `[zoom, zoom + 1)`
overlaps.

## The scene document and the surface

Scene document v0 (plan §11.4) holds `style` (a URL relative to the
document, or an inline v8 object), `projection`, `surface: "plane"` and
`sources.<id>.tileMatrixSet` (a registry id or an inline OGC document).
`Map::load_style_url` and `set_style_json` recognize one; a plain v8 style
is the default scene, Web Mercator. `Map::set_projection` is its API
equivalent for the projection. The surface's extent is its first source
set on its plane, else the projection's own; backgrounds and GeoJSON tile in
a one-tile quadtree over it (GeoJSON is geodetic, densified to 1° steps and
tiled on the plane; positions outside the projection are dropped). Camera
stores take the plane from `styleload`.

## The planner

Each source's plan is decided when it resolves:

| Plan | When | Built |
|---|---|---|
| `Affine` | the source's set is on the surface's plane | one affine row per tile, as before |
| `RasterWarp` | a raster on another plane | an error-bounded mesh per tile |
| `GpuPerVertex` | vector tiles on another plane | biquadratic patches per tile, evaluated per vertex |
| — | DEM on another plane | reported per source ([#30](https://github.com/sargunv/orbis/issues/30)) |

`CpuResample` is not built: GpuPerVertex covers every planar pair without
per-family WGSL, and nothing needs exact output in a foreign CRS until
`Tile` mode, which renders through the same plans.

**Cover.** A warped source's cover traverses its own set with each tile's
footprint on the surface (its bounding box, sampled at 9×9 points; a tile
holding a pole that a wrapping plane draws as an edge spans every copy) and
its local scale for level of detail. The surface sets the world copies.

**RasterWarp.** A grid refined until every cell's edge midpoints and centre
are within a quarter pixel at 512 pixels per tile (4 to 64 cells), the
tile's edges always sampled at 64 steps so neighbours share their vertices
(no T-junction cracks). Seam triangles of a wrapping plane are copied near
their first corner; triangles around a pole, which such a plane draws as an
edge, are dropped. Image vertices carry their position as f32 (a warped
tile's frame can be much larger than the view) and their image units apart;
image fragments clip in image units, clamped at the tile's edge, so the
cull no longer narrows image bounds to clip rectangles.

**GpuPerVertex.** Plan §9.3's non-affine rows carry `C − camera`, a
Jacobian and Hessian terms: a second-order series about the tile. Built
that way, adjacent tiles' series disagree on their shared edge by up to the
tolerance, which showed as cracks. Each tile is instead covered by 1–256
biquadratic patches through the f64 positions of their 3×3 nodes (corners,
edge midpoints, centre): the same precision series in Lagrange form, whose
neighbours, patches or same-zoom tiles, share their edges exactly. The
patches go into the property arena once per tile (18 f32 each), relative
to the tile's anchor; a tile row grows by two words that point at them, and
its origin is the anchor relative to the camera, so camera frames still
write rows only. `project.wgsl`'s `tile_to_world` evaluates them per vertex;
`tile_scale_at` gives lines a local scale.

Straight segments between mapped vertices must follow the curve, so jobs
cut a warped tile's geometry along a grid whose cells are within the
patches' chord tolerance (`TileRequest::subdivision`, 1 to 256 cells per
side): lines and rings where they cross grid lines, polygons clipped to the
tile first, and fill triangles cut into grid cells after earcut (which drops
the collinear vertices that cut the rings). Every cut is computed from an
edge's ends in a fixed order, so both triangles and both tiles sharing an
edge cut it alike. A warped tile's fills clip at its edge inclusively.

What isn't exact: the cull passes warped chunks (all eight storage
bindings of the cull stage are in use, so the patches aren't bound there;
cover already culls warped tiles by footprint); placement collides labels
in each tile's linear approximation at its centre while the GPU draws them
through the patches ([#31](https://github.com/sargunv/orbis/issues/31));
tiles of different zooms meet with sub-pixel gaps where a coarse edge's
curve and its finer neighbours' differ within the tolerance
([#32](https://github.com/sargunv/orbis/issues/32)).

## Measurements

Apple M-series, release build, single thread:

| Pair | Series per tile | Mesh per tile | Footprint per tile |
|---|---|---|---|
| Web Mercator → EPSG:3413, z2–9 | 0.09 ms (1–8 patches/side) | 0.34 ms (922 triangles) | 6 µs |
| Web Mercator → EPSG:2193, z4–11 | 0.02 ms (1–2) | 0.55 ms (338) | 25 µs |
| UPS North → Web Mercator, z2–9 | 1.0 ms (1–4) | 5.8 ms (3,514) | 73 µs |

Polar tiles on Web Mercator are the expensive case: Mercator stretches
toward the pole, so their meshes refine to 64 cells, and each position
inverts the polar projection (0.7 µs). Meshes and series are built on the
scene thread when a tile arrives; moving them into jobs is
[#33](https://github.com/sargunv/orbis/issues/33).

GL JS fixtures natively: 1,010 pass, 27 fail, 107 skip before and after
each commit, the same tests; at the last commit 119 images differ from the
previous build by at most 1/255 in a few pixels (shader rounding from the
projection module). `core.wasm` was 680 KiB brotli after the surface and
RasterWarp, 688 KiB after GpuPerVertex (both lanes' changes included).

Demos (web and native): `?style=arctic` (GIBS rasters in GIBS's EPSG:3413
sets, OpenFreeMap vectors through GpuPerVertex, a GeoJSON graticule: three
sets, three plans, one surface), `?style=nz` (OpenFreeMap on NZTM2000) and
`?style=greenland` (Web Mercator over GIBS ice velocity, warped).
