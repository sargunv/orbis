// Tile data units → camera-relative world (`project_tile`).
//
// An affine tile is `origin + (x, -y) * scale`. A warped tile (the planner's
// `GpuPerVertex` plan) is `origin` plus one of its biquadratic patches, from
// f64 positions on the CPU (orb-crs `Series`). Its words in `props` are its
// window, the rectangle of the tile the patches span (min then max, in tile
// fractions, y down), then the patches row by row over an even grid of the
// window, `warp_cells` holding patches per side as x | y << 16. Each patch
// is its 3×3 nodes' positions (corners, edge midpoints, centre; row by row,
// top to bottom), 18 f32, interpolated by quadratic Lagrange polynomials in
// parameters s (right) and t (down) in [-1, 1]. Neighbours share their
// edges' nodes, so tiles and patches meet exactly. Past a side of the
// window short of the tile's edge, points stay on that side, as the
// target's clamp puts them (Web Mercator's past ±85.05°); past the tile's
// own edges, its edge patches extrapolate.

// Whether the view is on its low-zoom surface, the globe or its plane, by
// any amount; never in a plane variant (`plane`).
fn on_globe() -> bool {
    return !plane && frame.globe_amount > 0.0;
}

const WARP_WINDOW_WORDS: u32 = 4u;

// Where a point of a warp's tile is among its patches.
struct WarpCell {
    // Its parameters in its patch.
    st: vec2<f32>,
    // Its patch's first word in `props`.
    at: u32,
    // Parameters per tile fraction, along x and y.
    per: vec2<f32>,
};

// The patch of the warp at `base` (`cells` packed, `words` per patch)
// holding tile point `uv` (fractions).
fn warp_cell(base: u32, cells: u32, words: u32, uv: vec2<f32>) -> WarpCell {
    let lo = vec2<f32>(bitcast<f32>(prop_at(base)), bitcast<f32>(prop_at(base + 1u)));
    let hi = vec2<f32>(bitcast<f32>(prop_at(base + 2u)), bitcast<f32>(prop_at(base + 3u)));
    let kept = clamp(
        uv,
        select(vec2<f32>(-3.0e38), lo, lo > vec2<f32>(0.0)),
        select(vec2<f32>(3.0e38), hi, hi < vec2<f32>(1.0)),
    );
    let count = vec2<u32>(cells & 0xffffu, cells >> 16u);
    let n = vec2<f32>(count);
    let u = (kept - lo) / (hi - lo) * n;
    let cell = clamp(floor(u), vec2<f32>(0.0), n - 1.0);
    var c: WarpCell;
    c.st = (u - cell) * 2.0 - 1.0;
    c.at = base + WARP_WINDOW_WORDS + (u32(cell.y) * count.x + u32(cell.x)) * words;
    c.per = 2.0 * n / (hi - lo);
    return c;
}

// Whether a tile's planar position holds all of it: an affine tile's, or
// a warp's whose window is the whole tile.
fn warp_whole(tile: Tile) -> bool {
    if !warped(tile) {
        return true;
    }
    let b = tile.warp_base;
    let window = vec4<f32>(bitcast<f32>(prop_at(b)), bitcast<f32>(prop_at(b + 1u)),
        bitcast<f32>(prop_at(b + 2u)), bitcast<f32>(prop_at(b + 3u)));
    return all(window == vec4<f32>(0.0, 0.0, 1.0, 1.0));
}

// The point's planar patch.
fn warp_patch(tile: Tile, p: vec2<f32>) -> WarpCell {
    return warp_cell(tile.warp_base, tile.warp_cells, 18u, p / tile.extent);
}

fn warp_node(at: u32, k: u32) -> vec2<f32> {
    let i = at + 2u * k;
    return vec2<f32>(bitcast<f32>(prop_at(i)), bitcast<f32>(prop_at(i + 1u)));
}

fn lagrange(s: f32) -> vec3<f32> {
    return vec3<f32>(s * (s - 1.0) * 0.5, 1.0 - s * s, s * (s + 1.0) * 0.5);
}

fn lagrange_slope(s: f32) -> vec3<f32> {
    return vec3<f32>(s - 0.5, -2.0 * s, s + 0.5);
}

// Σ a[i] b[j] node(i, j).
fn warp_sum(at: u32, a: vec3<f32>, b: vec3<f32>) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    for (var j = 0u; j < 3u; j++) {
        let row = warp_node(at, 3u * j) * a.x + warp_node(at, 3u * j + 1u) * a.y
            + warp_node(at, 3u * j + 2u) * a.z;
        sum += row * b[j];
    }
    return sum;
}

// Tile point `p` less the tile's origin, in world units.
fn tile_local(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    if !warped(tile) {
        return vec2<f32>(p.x, -p.y) * tile.scale;
    }
    let w = warp_patch(tile, p);
    return warp_sum(w.at, lagrange(w.st.x), lagrange(w.st.y));
}

// A drape placement's tile point `p` in atlas texels, the same wherever
// its slot is, which the map's history decides. Its origin's whole quanta
// hold the slot's corner, a multiple of the quantum, and the remainder its
// place in the slot: the point within the slot, snapped onto the
// rasterizer's 1/256 grid, is the same in every slot, and the corner adds
// to it exactly. On the grid, not between its points: a point midway
// rounds by the rasterizer's rule, which can differ by sign (a slot's edge
// lies just off the page at row 0).
// `local` is the point's `tile_local`.
fn drape_texel(tile: Tile, local: vec2<f32>) -> vec2<f32> {
    let whole = vec2<f32>(tile.origin - frame.camera_whole) * frame.quantum;
    return whole + snap_texel((tile.origin_rest - frame.camera_rest) + local);
}

// Where a drape placement's tile origin's whole quanta put it in the page,
// in texels; the origin elsewhere. Screen positions taken from it are the
// same in every slot, and exact: interpolated across a primitive, they
// round alike wherever the slot is.
fn drape_origin(tile: Tile) -> vec2<f32> {
    if !draped_placement(tile) {
        return vec2<f32>(0.0);
    }
    let whole = vec2<f32>(tile.origin - frame.camera_whole) * frame.quantum;
    return vec2<f32>(whole.x, -whole.y);
}

// `v` on the rasterizer's 1/256 grid.
fn snap_texel(v: vec2<f32>) -> vec2<f32> {
    return round(v * 256.0) / 256.0;
}

// The metric of a tile's map onto the view's surface at data point `p`:
// JᵀJ, J the map's Jacobian in world units per data unit (columns along
// data x and y). An affine tile's is its scale squared times the
// identity; a warp's or a sphere's can stretch one direction more than
// the other (geographic tiles on Web Mercator near a pole), and its
// square root of the determinant is the area scale's.
fn tile_metric_at(tile: Tile, p: vec2<f32>) -> mat2x2<f32> {
    let flat = plane_metric_at(tile, p);
    if !on_globe() || tile.globe_cells == 0u {
        return flat;
    }
    // The sphere's metres per data unit, in plane world units at the
    // target: pixels there and here are the same size.
    let w = globe_patch(tile, p);
    let per = w.per / (tile.extent * frame.globe_metres);
    let ds = globe_sum(w.at, lagrange_slope(w.st.x), lagrange(w.st.y)) * per.x;
    let dt = globe_sum(w.at, lagrange(w.st.x), lagrange_slope(w.st.y)) * per.y;
    let globe = mat2x2<f32>(dot(ds, ds), dot(ds, dt), dot(ds, dt), dot(dt, dt));
    if (tile.globe_flags & 1u) != 0u {
        return globe;
    }
    // Mixed as lengths, not their squares: the metrics' square roots.
    let r = metric_root(flat) * (1.0 - frame.globe_amount) + metric_root(globe) * frame.globe_amount;
    return r * r;
}

fn plane_metric_at(tile: Tile, p: vec2<f32>) -> mat2x2<f32> {
    if !warped(tile) {
        let s = tile.scale * tile.scale;
        return mat2x2<f32>(s, 0.0, 0.0, s);
    }
    let w = warp_patch(tile, p);
    let per = w.per / tile.extent;
    let ds = warp_sum(w.at, lagrange_slope(w.st.x), lagrange(w.st.y)) * per.x;
    let dt = warp_sum(w.at, lagrange(w.st.x), lagrange_slope(w.st.y)) * per.y;
    return mat2x2<f32>(dot(ds, ds), dot(ds, dt), dot(ds, dt), dot(dt, dt));
}

// The symmetric square root of a metric.
fn metric_root(m: mat2x2<f32>) -> mat2x2<f32> {
    let s = sqrt(max(determinant(m), 0.0));
    let t = sqrt(max(m[0][0] + m[1][1] + 2.0 * s, 1.0e-30));
    return mat2x2<f32>(m[0][0] + s, m[0][1], m[1][0], m[1][1] + s) * (1.0 / t);
}

// On a low-zoom surface (frame.globe_amount > 0), every tile's patches are
// in space, laid out as planar ones: nine nodes' camera-relative ECEF metres
// (on a low-zoom plane, its world units at height 0), 27 f32, around its
// row's `globe_origin`.
fn globe_patch(tile: Tile, p: vec2<f32>) -> WarpCell {
    return warp_cell(tile.globe_base, tile.globe_cells, 27u, p / tile.extent);
}

fn globe_node(at: u32, k: u32) -> vec3<f32> {
    let i = at + 3u * k;
    return vec3<f32>(bitcast<f32>(prop_at(i)), bitcast<f32>(prop_at(i + 1u)), bitcast<f32>(prop_at(i + 2u)));
}

fn globe_sum(at: u32, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var j = 0u; j < 3u; j++) {
        let row = globe_node(at, 3u * j) * a.x + globe_node(at, 3u * j + 1u) * a.y
            + globe_node(at, 3u * j + 2u) * a.z;
        sum += row * b[j];
    }
    return sum;
}

// A point of a tile's data on the view's surfaces: camera-relative world on
// the plane and, on a globe, camera-relative ECEF metres on the sphere. A
// tile with no place on the plane draws on the globe alone.
struct Ground {
    flat: vec2<f32>,
    globe: vec3<f32>,
    on_plane: bool,
    // A cap's vertex at a pole (`tile_ground`).
    pole: bool,
};

// Vertices at y -32768 and 32767 are at the North and South Pole (caps,
// orb-layout's `NORTH_POLE_Y`): there on a globe where the row's clip
// rectangles reach on to that pole (to the same extreme, as frame.rs's
// `placements` has them on the world's edge rows), and on the edge
// otherwise.
const SPHERE_RADIUS: f32 = 6378137.0;

fn reaches_pole(tile: Tile, north: bool) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        if lo.x <= hi.x && ((north && lo.y < -32767.5) || (!north && hi.y > 32766.5)) {
            return true;
        }
    }
    return false;
}

// Each projection inlines this, and Adreno's compilers inline every
// call, so it calls each helper from one place: `tile_surface` once, and
// the warp's patches and the Mercator sphere each once, in a loop over the
// points they map.
fn tile_ground(tile: Tile, p: vec2<f32>) -> Ground {
    let north = p.y <= -32767.5;
    let pole = north || p.y >= 32766.5;
    let on_edge = vec2<f32>(clamp(p.x, 0.0, tile.extent), select(tile.extent, 0.0, north));
    var g = tile_surface(tile, select(p, on_edge, pole));
    if pole && on_globe() && frame.globe_plane == 0u && reaches_pole(tile, north) {
        g.globe = vec3<f32>(0.0, 0.0, select(-SPHERE_RADIUS, SPHERE_RADIUS, north)) - frame.globe_target;
        g.pole = true;
    }
    return g;
}

fn tile_surface(tile: Tile, p: vec2<f32>) -> Ground {
    var g: Ground;
    if plane {
        // What the rest comes to for an affine placement on the plane, as
        // a plane variant's all are: without it, the moto g's compiler
        // kept the loop and the drape's branch, a quarter of a zoom-12
        // view's vertex time (#180).
        g.on_plane = true;
        g.flat = tile_affine(tile, p);
        return g;
    }
    g.on_plane = (tile.globe_flags & 1u) == 0u;
    let globe = on_globe() && tile.globe_cells > 0u;
    // A point past the tile (a translated label or fill, a stroke's edge)
    // goes onto the sphere at the tile's edge through its patches, and on
    // from there exactly: the patches don't extend. Only a plane position
    // that holds the whole tile carries it on (not one Web Mercator clamps
    // part of).
    let edge = clamp(p, vec2<f32>(0.0), vec2<f32>(tile.extent));
    let carry = globe && any(edge != p) && g.on_plane && frame.globe_plane == 0u && warp_whole(tile);
    // The point's and, carried on, the edge's places from the tile's
    // origin.
    var local = vec2<f32>(0.0);
    var edge_local = vec2<f32>(0.0);
    for (var k = 0u; k < select(select(0u, 1u, g.on_plane), 2u, carry); k++) {
        let l = tile_local(tile, select(p, edge, k == 1u));
        if k == 0u {
            local = l;
        } else {
            edge_local = l;
        }
    }
    if g.on_plane {
        if draped_placement(tile) {
            g.flat = drape_texel(tile, local);
        } else {
            g.flat = tile_origin(tile) + local;
        }
    }
    if globe {
        g.globe = sphere_at(tile.globe_base, tile.globe_cells, tile_globe_origin(tile), edge / tile.extent);
        if carry {
            g.globe += mercator_step(tile_origin(tile) + edge_local, g.flat);
        }
    }
    return g;
}

// The step on the sphere between two camera-relative Web Mercator world
// positions (metres), `start` to `end`: exact to f32's precision of the
// step itself, for points no tile's patches hold. The sphere's positions
// are millions of metres long, and their difference in f32 would be no
// better than half a metre (#162), a pixel from zoom 18: so the step
// comes from the differences of the two points' longitudes and latitudes
// alone, and `start`'s place, known only to f32's precision, just turns it.
fn mercator_step(start: vec2<f32>, end: vec2<f32>) -> vec3<f32> {
    let up = normalize(frame.globe_target);
    let target_lat = asin(clamp(up.z, -1.0, 1.0));
    let target_lon = atan2(up.y, up.x);
    let target_y = SPHERE_RADIUS * log(tan(0.7853982 + target_lat * 0.5));
    let lon = target_lon + start.x / SPHERE_RADIUS;
    // tan(π/4 + lat/2), Mercator's e^(y/R).
    let e = exp((target_y + start.y) / SPHERE_RADIUS);
    let lat = 2.0 * atan(e) - 1.5707964;
    let d_lon = (end.x - start.x) / SPHERE_RADIUS;
    // The latitudes' difference, 2 (atan(e q) - atan(e)) with q the
    // step's e^(dy/R), taken whole.
    let q1 = exp_minus_one((end.y - start.y) / SPHERE_RADIUS);
    let d_lat = 2.0 * atan(e * q1 / (1.0 + e * e * (1.0 + q1)));
    // cos a − cos b and sin a − sin b as products with the half step's
    // sine, which keep its precision.
    let lat_sine = 2.0 * sin(d_lat * 0.5);
    let lon_sine = 2.0 * sin(d_lon * 0.5);
    let cos_lat = cos(lat + d_lat);
    let d_cos_lat = -sin(lat + d_lat * 0.5) * lat_sine;
    let d_sin_lat = cos(lat + d_lat * 0.5) * lat_sine;
    let d_cos_lon = -sin(lon + d_lon * 0.5) * lon_sine;
    let d_sin_lon = cos(lon + d_lon * 0.5) * lon_sine;
    return vec3<f32>(
        cos_lat * d_cos_lon + d_cos_lat * cos(lon),
        cos_lat * d_sin_lon + d_cos_lat * sin(lon),
        d_sin_lat,
    ) * SPHERE_RADIUS;
}

// e^x − 1 to f32's precision relative to x, for small x too.
fn exp_minus_one(x: f32) -> f32 {
    if abs(x) >= 0.1 {
        return exp(x) - 1.0;
    }
    return x * (1.0 + x * 0.5 * (1.0 + x / 3.0 * (1.0 + x * 0.25 * (1.0 + x * 0.2))));
}

// A tile's point on the sphere, camera-relative, from its patches in space
// (first word `base` in `props`, `cells` packed) around `origin`, at `uv`
// (fractions of the tile, y down). Other tile tables (terrain's) call it
// with their own rows' terms.
fn sphere_at(base: u32, cells: u32, origin: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let c = warp_cell(base, cells, 27u, uv);
    return origin + globe_sum(c.at, lagrange(c.st.x), lagrange(c.st.y));
}

// Tile point `p` translated by `shift` world units along the map (fill
// and circle translate). On a globe, as in GL JS, the translation moves the
// point in the tile's own units before it goes onto the sphere.
fn tile_ground_shifted(tile: Tile, p: vec2<f32>, shift: vec2<f32>) -> Ground {
    let globe = on_globe();
    var g = tile_ground(tile, select(p, p + vec2<f32>(shift.x, -shift.y) / tile.scale, globe));
    if !globe {
        // On a drape's grid too, a texel position plus it is exact.
        g.flat += select(shift, snap_texel(shift), draped_placement(tile));
    }
    return g;
}

// Clip position of a ground point moved by `offset` world units along the
// map (x east, y north) and raised `z` world units: on the plane, or on the
// globe along its east, north and up there, mixed as the globe hands over.
// Depth on the globe comes from the horizon plane, which hides its far side.
fn ground_clip(g: Ground, offset: vec2<f32>, z: f32) -> vec4<f32> {
    return ground_clip_depth(g, offset, z, false);
}

// A ground point raised `z` world units, for the terrain's depth test
// (`terrain_visibility`): `projected`, its `ground_clip`, with the depth
// the terrain draws. On a globe that is the sphere's perspective depth
// (`ground_clip_3d`), not the horizon plane's, which no surface's depth
// matches.
fn terrain_seen(projected: vec4<f32>, g: Ground, z: f32) -> vec4<f32> {
    if !on_globe() {
        return projected;
    }
    return ground_clip_3d(g, vec2<f32>(0.0), z);
}

// Whether the globe stands between the eye and camera-relative `p`. A
// point in front of the sea-level horizon plane never is; only behind it
// does the ray test run.
//
// For `solid` points (the ground's, what stands on it) the globe is the
// sphere through the lowest ground the terrain shows, so that ground
// below sea level near the limb (the Caspian's, the Dead Sea's) stays in
// view while what hangs under it (skirts) hides; its own horizon plane
// lies behind the sea level's. It sinks by a pixel at `p`'s distance: the
// ground's last rows before the horizon lie on it to within their
// patches' tolerance and f32's, and a ray grazing it there would
// otherwise enter it just short of them by that rounding alone, hiding
// them in a line along the horizon. A skirt hangs a fifth of a tile, many
// pixels, below it.
//
// Labels take the sea-level sphere, as their placement does (orb-symbol's
// `sphere_hides`): at low zoom a pixel is tens of kilometres at the limb,
// and the sunk sphere let labels degrees behind it show along it.
fn sphere_hides(p: vec3<f32>, solid: bool) -> bool {
    if dot(frame.globe_horizon.xyz, p) + frame.globe_horizon.w >= 0.0 {
        return false;
    }
    let from_center = frame.globe_eye + frame.globe_target;
    let d = p - frame.globe_eye;
    let a = dot(d, d);
    var radius = SPHERE_RADIUS;
    if solid {
        let lowest = radius + frame.terrain_lowest * frame.extrusion_scale * frame.globe_metres;
        radius = lowest - sqrt(a) / frame.camera_to_center;
    }
    let b = dot(from_center, d);
    // The eye's height over it, taken whole: a difference of squares
    // would cancel.
    let reach = length(from_center);
    let c = (reach - radius) * (reach + radius);
    let disc = b * b - a * c;
    if disc <= 0.0 {
        return false;
    }
    let t = (-b - sqrt(disc)) / a;
    return t > 0.0 && t < 0.999;
}

// `ground_clip` for 3D content (extrusions), with perspective depth on the
// globe, as GL JS's `projectTileFor3D` keeps.
fn ground_clip_3d(g: Ground, offset: vec2<f32>, z: f32) -> vec4<f32> {
    return ground_clip_depth(g, offset, z, true);
}

// `ground_clip_3d` with reversed-Z depth, as 3D passes draw. On the plane
// it comes from the frame's depth row, as clip w less z taken whole: w and
// z are nearly equal there, and subtracting them in f32 would leave a
// relative error of up to 10^-5, enough for coplanar faces to fight.
fn ground_clip_reversed(g: Ground, offset: vec2<f32>, z: f32) -> vec4<f32> {
    return ground_clip_reversed_hiding(g, offset, z, true);
}

// `ground_clip_reversed`, where what the sphere hides goes past the far
// plane only if `hide`: content that hides behind the sphere per
// fragment (`sphere_limb`) keeps its own depth.
fn ground_clip_reversed_hiding(g: Ground, offset: vec2<f32>, z: f32, hide: bool) -> vec4<f32> {
    var clip = ground_clip_hiding(g, offset, z, true, hide);
    clip.z = select(clip.w - clip.z, plane_depth(g.flat + offset, z), !on_globe());
    return clip;
}

// Reversed-Z depth of a camera-relative world position on the plane.
fn plane_depth(p: vec2<f32>, z: f32) -> f32 {
    return dot(frame.view_depth, vec4<f32>(p, z, 1.0));
}

// On terrain, what an extrusion layer darkens on the ground draws first
// into a mask (orb-render's terrain decal): flat, on a plane at the
// ground under the camera target, as the camera sees it. The terrain's
// mesh then reads the mask where each of its points falls on that plane,
// so the darkness lies on the surface drawn. Clip x and y shrink by this
// in the mask, which reaches past the view as far: a point the terrain
// lifts or lowers off the plane falls on it off screen.
const GROUND_MASK_SCALE: f32 = 0.8;

// The ground mask's clip position of camera-relative world point `p`.
fn ground_mask_clip(p: vec2<f32>) -> vec4<f32> {
    var clip = frame.view_proj * vec4<f32>(p, frame.terrain_ground * frame.extrusion_scale, 1.0);
    clip.x *= GROUND_MASK_SCALE;
    clip.y *= GROUND_MASK_SCALE;
    // No depth: within the clip volume wherever it lies at least a
    // hundredth of the way to the target in front of the eye, where the
    // mask's triangles clip rather than at the eye itself.
    clip.z = clip.w - 0.01 * frame.camera_to_center;
    return clip;
}

fn ground_clip_depth(g: Ground, offset: vec2<f32>, z: f32, solid: bool) -> vec4<f32> {
    return ground_clip_hiding(g, offset, z, solid, solid);
}

// Whether 3D content of `tile` may reach behind a whole globe's horizon
// plane (not handing over to the plane), to hide behind the sphere per
// fragment. Its geometry reaches at most a tile's width past its square
// (an overzoom slice's), so none lies `reach`, three widths in metres no
// longer than the equator's, from its anchor. Along the sphere the
// plane's value falls by at most its gradient times the sine of the
// angle from under the eye per metre, and the anchor's chord sags under
// it by less than reach²/R; ground below sea level falls by its depth,
// and raised off the sphere where it faces the eye, a point only rises.
// Elsewhere the sphere hides none of it, and the GPU's hidden-surface
// removal stays on.
fn sphere_limb(tile: Tile) -> bool {
    if !on_globe() || frame.globe_plane != 0u || frame.globe_amount < 1.0 {
        return false;
    }
    let t = frame.globe_target;
    let per_unit = frame.globe_metres * length(t) / max(length(t.xy), 1.0);
    let reach = 3.0 * tile.extent * tile.scale * per_unit;
    let n = frame.globe_horizon;
    let anchor = tile_globe_origin(tile);
    // The sine of the anchor's angle from under the eye.
    let off_axis = length(cross(normalize(n.xyz), anchor - frame.globe_eye)) / SPHERE_RADIUS;
    let depth = max(-frame.terrain_lowest * frame.extrusion_scale * frame.globe_metres, 0.0);
    let fall = reach * (off_axis + 2.0 * reach / SPHERE_RADIUS) + depth;
    return dot(n.xyz, anchor) + n.w < length(n.xyz) * fall;
}

// `ground_clip_depth`; a solid point the sphere hides goes past the far
// plane only if `hide`.
fn ground_clip_hiding(g: Ground, offset: vec2<f32>, z: f32, solid: bool, hide: bool) -> vec4<f32> {
    let flat = frame.view_proj * vec4<f32>(g.flat + offset, z, 1.0);
    if !on_globe() {
        return flat;
    }
    if frame.globe_plane != 0u {
        // A low-zoom plane: moved and raised in its own units, the two
        // planes' clip positions mixed as it hands over.
        let low = frame.globe_view_proj * vec4<f32>(g.globe + vec3<f32>(offset, z) * frame.globe_metres, 1.0);
        if !g.on_plane || frame.globe_amount >= 1.0 {
            return low;
        }
        return mix(flat, low, frame.globe_amount);
    }
    let up = normalize(g.globe + frame.globe_target);
    var p = g.globe + up * (z * frame.globe_metres);
    if any(offset != vec2<f32>(0.0)) {
        // The offset turns the point around the sphere by its east and
        // north parts in radians, as GL JS's `globeRotateVector` does.
        let level = vec3<f32>(-up.y, up.x, 0.0);
        let east = select(vec3<f32>(0.0, 1.0, 0.0), normalize(level), length(level) > 1e-6);
        let north = cross(up, east);
        let t = tan(offset * frame.globe_metres / SPHERE_RADIUS);
        // normalize(up + v) − up, taken whole: as a difference of two
        // unit vectors it would round to a metre at the sphere's scale.
        let v = east * t.x + north * t.y;
        let vv = dot(v, v);
        let s = sqrt(1.0 + vv);
        let turned = v / s - up * (vv / (s * (1.0 + s)));
        p += turned * (SPHERE_RADIUS + z * frame.globe_metres);
    }
    var clip = frame.globe_view_proj * vec4<f32>(p, 1.0);
    if solid {
        // Behind the sphere from the eye: past the far plane, before and
        // after the 3D passes reverse the depth, so a triangle the sphere
        // partly hides (a skirt hanging into it) runs away from the eye
        // there rather than in front of the surface.
        if hide && sphere_hides(p, true) {
            clip.z = 2.0 * clip.w;
        }
        if !g.on_plane {
            return clip;
        }
        return mix(flat, clip, frame.globe_amount);
    }
    // Depth from the horizon plane, as GL JS's globe: past the horizon is
    // beyond the far plane.
    let facing = dot(frame.globe_horizon.xyz, p) + frame.globe_horizon.w;
    clip.z = (1.0 - facing * 0.5) * clip.w;
    if !g.on_plane || frame.globe_amount >= 1.0 {
        return clip;
    }
    // As GL JS hands over: the poles' caps only in the last 2%, and the
    // horizon's depth only past a fifth of the way to the globe.
    if g.pole {
        let hidden = pow(max((1.0 - frame.globe_amount) / 0.02, 0.0), 8.0);
        clip.z = mix(clip.z, 100.0 * clip.w, min(hidden, 1.0));
        return clip;
    }
    var mixed = mix(flat, clip, frame.globe_amount);
    mixed.z = mix(0.0, clip.z / clip.w, clamp((frame.globe_amount - 0.2) / 0.8, 0.0, 1.0)) * mixed.w;
    return mixed;
}
