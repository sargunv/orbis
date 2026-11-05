// Tile data units → camera-relative world (plan §9.3, `project_tile`).
//
// An affine tile is `origin + (x, -y) * scale`. A warped tile (the planner's
// `GpuPerVertex` plan) is `origin` plus one of its `warp_cells`² biquadratic
// patches, from f64 positions on the CPU (orb-crs `Series`): each patch is
// its 3×3 nodes' positions (corners, edge midpoints, centre; row by row, top
// to bottom), 18 f32 in `props`, interpolated by quadratic Lagrange
// polynomials in parameters s (right) and t (down) in [-1, 1]. Neighbours
// share their edges' nodes, so tiles and patches meet exactly.

// The point's patch among `count` per side (its index) and its
// parameters there.
fn warp_cell(tile: Tile, count: u32, p: vec2<f32>) -> vec3<f32> {
    let cells = f32(count);
    let u = p / tile.extent * cells;
    let cell = clamp(floor(u), vec2<f32>(0.0), vec2<f32>(cells - 1.0));
    let st = (u - cell) * 2.0 - 1.0;
    return vec3<f32>(st, f32(u32(cell.y) * count + u32(cell.x)));
}

// The point's planar patch (its first word in `props`) and its parameters.
fn warp_patch(tile: Tile, p: vec2<f32>) -> vec3<f32> {
    let c = warp_cell(tile, tile.warp_cells, p);
    return vec3<f32>(c.xy, f32(tile.warp_base + u32(c.z) * 18u));
}

fn warp_node(at: f32, k: u32) -> vec2<f32> {
    let i = u32(at) + 2u * k;
    return vec2<f32>(bitcast<f32>(prop_at(i)), bitcast<f32>(prop_at(i + 1u)));
}

fn lagrange(s: f32) -> vec3<f32> {
    return vec3<f32>(s * (s - 1.0) * 0.5, 1.0 - s * s, s * (s + 1.0) * 0.5);
}

fn lagrange_slope(s: f32) -> vec3<f32> {
    return vec3<f32>(s - 0.5, -2.0 * s, s + 0.5);
}

// Σ a[i] b[j] node(i, j).
fn warp_sum(at: f32, a: vec3<f32>, b: vec3<f32>) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    for (var j = 0u; j < 3u; j++) {
        let row = warp_node(at, 3u * j) * a.x + warp_node(at, 3u * j + 1u) * a.y
            + warp_node(at, 3u * j + 2u) * a.z;
        sum += row * b[j];
    }
    return sum;
}

fn tile_to_world(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    if tile.warp_cells == 0u {
        return tile_affine(tile, p);
    }
    let w = warp_patch(tile, p);
    return tile_origin(tile) + warp_sum(w.z, lagrange(w.x), lagrange(w.y));
}

// World units per data unit at `p`: the tile's scale, or a warp's local
// scale there (the square root of its area scale).
fn tile_scale_at(tile: Tile, p: vec2<f32>) -> f32 {
    let flat = plane_scale_at(tile, p);
    if frame.globe_amount <= 0.0 || tile.globe_cells == 0u {
        return flat;
    }
    // The sphere's metres per data unit, in plane world units at the
    // target: pixels there and here are the same size.
    let w = globe_patch(tile, p);
    let ds = globe_sum(w.z, lagrange_slope(w.x), lagrange(w.y));
    let dt = globe_sum(w.z, lagrange(w.x), lagrange_slope(w.y));
    let metres = sqrt(length(cross(ds, dt))) * 2.0 * f32(tile.globe_cells) / tile.extent;
    let globe = metres / frame.globe_metres;
    if (tile.globe_flags & 1u) != 0u {
        return globe;
    }
    return mix(flat, globe, frame.globe_amount);
}

fn plane_scale_at(tile: Tile, p: vec2<f32>) -> f32 {
    if tile.warp_cells == 0u {
        return tile.scale;
    }
    let per_unit = 2.0 * f32(tile.warp_cells) / tile.extent;
    let w = warp_patch(tile, p);
    let ds = warp_sum(w.z, lagrange_slope(w.x), lagrange(w.y));
    let dt = warp_sum(w.z, lagrange(w.x), lagrange_slope(w.y));
    // Parameters span 2 per patch side: 2 · cells per tile side.
    return sqrt(abs(ds.x * dt.y - ds.y * dt.x)) * per_unit;
}

// On a low-zoom surface (frame.globe_amount > 0), every tile's patches are
// in space: nine nodes' camera-relative ECEF metres (on a low-zoom plane,
// its world units at height 0), 27 f32, around its row's `globe_origin`.
fn globe_patch(tile: Tile, p: vec2<f32>) -> vec3<f32> {
    let c = warp_cell(tile, tile.globe_cells, p);
    return vec3<f32>(c.xy, f32(tile.globe_base + u32(c.z) * 27u));
}

fn globe_node(at: f32, k: u32) -> vec3<f32> {
    let i = u32(at) + 3u * k;
    return vec3<f32>(bitcast<f32>(prop_at(i)), bitcast<f32>(prop_at(i + 1u)), bitcast<f32>(prop_at(i + 2u)));
}

fn globe_sum(at: f32, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
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
// rectangles reach past the tile's edge toward that pole, and on the edge
// otherwise.
const SPHERE_RADIUS: f32 = 6378137.0;

fn reaches_pole(tile: Tile, north: bool) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        if lo.x <= hi.x && ((north && lo.y < 0.0) || (!north && hi.y > tile.extent)) {
            return true;
        }
    }
    return false;
}

fn tile_ground(tile: Tile, p: vec2<f32>) -> Ground {
    let north = p.y <= -32767.5;
    if north || p.y >= 32766.5 {
        var g = tile_surface(tile, vec2<f32>(clamp(p.x, 0.0, tile.extent), select(tile.extent, 0.0, north)));
        if frame.globe_amount > 0.0 && frame.globe_plane == 0u && reaches_pole(tile, north) {
            g.globe = vec3<f32>(0.0, 0.0, select(-SPHERE_RADIUS, SPHERE_RADIUS, north)) - frame.globe_target;
            g.pole = true;
        }
        return g;
    }
    return tile_surface(tile, p);
}

fn tile_surface(tile: Tile, p: vec2<f32>) -> Ground {
    var g: Ground;
    g.on_plane = (tile.globe_flags & 1u) == 0u;
    if g.on_plane {
        g.flat = tile_to_world(tile, p);
    }
    if frame.globe_amount > 0.0 && tile.globe_cells > 0u {
        // A point past the tile (a translated label or fill, a stroke's
        // edge) goes onto the sphere at the tile's edge through its
        // patches, and on from there exactly: the patches don't extend.
        let edge = clamp(p, vec2<f32>(0.0), vec2<f32>(tile.extent));
        g.globe = sphere_at(tile.globe_base, tile.globe_cells, tile_globe_origin(tile), edge / tile.extent);
        if any(edge != p) && g.on_plane && frame.globe_plane == 0u {
            g.globe += mercator_sphere(g.flat) - mercator_sphere(tile_to_world(tile, edge));
        }
    }
    return g;
}

// A camera-relative Web Mercator world position (metres) on the sphere,
// camera-relative: exact, for points no tile's patches hold. Differences
// of two nearby points keep f32's precision about the target.
fn mercator_sphere(world: vec2<f32>) -> vec3<f32> {
    let up = normalize(frame.globe_target);
    let target_lat = asin(clamp(up.z, -1.0, 1.0));
    let target_lon = atan2(up.y, up.x);
    let target_y = SPHERE_RADIUS * log(tan(0.7853982 + target_lat * 0.5));
    let lon = target_lon + world.x / SPHERE_RADIUS;
    let lat = 2.0 * atan(exp((target_y + world.y) / SPHERE_RADIUS)) - 1.5707964;
    let at = vec3<f32>(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)) * SPHERE_RADIUS;
    return at - up * SPHERE_RADIUS;
}

// A tile's point on the sphere, camera-relative, from its patches in space
// (first word `base` in `props`, `cells` per side) around `origin`, at `uv`
// (fractions of the tile, y down). Other tile tables (terrain's) call it
// with their own rows' terms.
fn sphere_at(base: u32, cells: u32, origin: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    let count = f32(cells);
    let u = uv * count;
    let cell = clamp(floor(u), vec2<f32>(0.0), vec2<f32>(count - 1.0));
    let st = (u - cell) * 2.0 - 1.0;
    let at = f32(base + (u32(cell.y) * cells + u32(cell.x)) * 27u);
    return origin + globe_sum(at, lagrange(st.x), lagrange(st.y));
}

// Tile point `p` translated by `shift` world units along the map (fill
// and circle translate). On a globe, as in GL JS, the translation moves the
// point in the tile's own units before it goes onto the sphere.
fn tile_ground_shifted(tile: Tile, p: vec2<f32>, shift: vec2<f32>) -> Ground {
    if frame.globe_amount <= 0.0 {
        var g = tile_ground(tile, p);
        g.flat += shift;
        return g;
    }
    return tile_ground(tile, p + vec2<f32>(shift.x, -shift.y) / tile.scale);
}

// Clip position of a ground point moved by `offset` world units along the
// map (x east, y north) and raised `z` world units: on the plane, or on the
// globe along its east, north and up there, mixed as the globe hands over.
// Depth on the globe comes from the horizon plane, which hides its far side.
fn ground_clip(g: Ground, offset: vec2<f32>, z: f32) -> vec4<f32> {
    return ground_clip_depth(g, offset, z, false);
}

// Whether the sphere stands between the eye and camera-relative `p`. A
// point in front of the horizon plane never is; only behind it does the
// ray test run (in f32 it cancels badly for grazing rays).
fn sphere_hides(p: vec3<f32>) -> bool {
    if dot(frame.globe_horizon.xyz, p) + frame.globe_horizon.w >= 0.0 {
        return false;
    }
    let from_center = frame.globe_eye + frame.globe_target;
    let d = p - frame.globe_eye;
    let a = dot(d, d);
    let b = dot(from_center, d);
    let c = dot(from_center, from_center) - SPHERE_RADIUS * SPHERE_RADIUS;
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

fn ground_clip_depth(g: Ground, offset: vec2<f32>, z: f32, solid: bool) -> vec4<f32> {
    let flat = frame.view_proj * vec4<f32>(g.flat + offset, z, 1.0);
    if frame.globe_amount <= 0.0 {
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
        let turned = normalize(up + east * t.x + north * t.y);
        p += (turned - up) * (SPHERE_RADIUS + z * frame.globe_metres);
    }
    var clip = frame.globe_view_proj * vec4<f32>(p, 1.0);
    if solid {
        // Behind the sphere from the eye: past the far plane, before and
        // after the 3D passes reverse the depth, so a triangle the sphere
        // partly hides (a skirt hanging into it) runs away from the eye
        // there rather than in front of the surface.
        if sphere_hides(p) {
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
