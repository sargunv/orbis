// Tile data units → camera-relative world (plan §9.3, `project_tile`).
//
// An affine tile is `origin + (x, -y) * scale`. A warped tile (the planner's
// `GpuPerVertex` plan) is `origin` plus one of its `warp_cells`² biquadratic
// patches, from f64 positions on the CPU (orb-crs `Series`): each patch is
// its 3×3 nodes' positions (corners, edge midpoints, centre; row by row, top
// to bottom), 18 f32 in `props`, interpolated by quadratic Lagrange
// polynomials in parameters s (right) and t (down) in [-1, 1]. Neighbours
// share their edges' nodes, so tiles and patches meet exactly.

// The point's patch (its index) and its parameters there.
fn warp_cell(tile: Tile, p: vec2<f32>) -> vec3<f32> {
    let cells = f32(tile.warp_cells);
    let u = p / tile.extent * cells;
    let cell = clamp(floor(u), vec2<f32>(0.0), vec2<f32>(cells - 1.0));
    let st = (u - cell) * 2.0 - 1.0;
    return vec3<f32>(st, f32(u32(cell.y) * tile.warp_cells + u32(cell.x)));
}

// The point's planar patch (its first word in `props`) and its parameters.
fn warp_patch(tile: Tile, p: vec2<f32>) -> vec3<f32> {
    let c = warp_cell(tile, p);
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
    return tile.origin + warp_sum(w.z, lagrange(w.x), lagrange(w.y));
}

// World units per data unit at `p`: the tile's scale, or a warp's local
// scale there (the square root of its area scale).
fn tile_scale_at(tile: Tile, p: vec2<f32>) -> f32 {
    if tile.warp_cells == 0u {
        return tile.scale;
    }
    let per_unit = 2.0 * f32(tile.warp_cells) / tile.extent;
    if frame.globe_amount > 0.0 {
        // The sphere's metres per data unit, in plane world units at the
        // target: pixels there and here are the same size.
        let w = globe_patch(tile, p);
        let ds = globe_sum(w.z, lagrange_slope(w.x), lagrange(w.y));
        let dt = globe_sum(w.z, lagrange(w.x), lagrange_slope(w.y));
        let metres = sqrt(length(cross(ds, dt))) * per_unit;
        return mix(tile.scale, metres / frame.globe_metres, frame.globe_amount);
    }
    let w = warp_patch(tile, p);
    let ds = warp_sum(w.z, lagrange_slope(w.x), lagrange(w.y));
    let dt = warp_sum(w.z, lagrange(w.x), lagrange_slope(w.y));
    // Parameters span 2 per patch side: 2 · cells per tile side.
    return sqrt(abs(ds.x * dt.y - ds.y * dt.x)) * per_unit;
}

// On a globe (frame.globe_amount > 0), every tile's patches are in space:
// nine nodes' camera-relative ECEF metres, 27 f32, around its row's
// `globe_origin`.
fn globe_patch(tile: Tile, p: vec2<f32>) -> vec3<f32> {
    let c = warp_cell(tile, p);
    return vec3<f32>(c.xy, f32(tile.warp_base + u32(c.z) * 27u));
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
// the plane and, on a globe, camera-relative ECEF metres on the sphere.
struct Ground {
    flat: vec2<f32>,
    globe: vec3<f32>,
};

fn tile_ground(tile: Tile, p: vec2<f32>) -> Ground {
    var g: Ground;
    if frame.globe_amount <= 0.0 {
        g.flat = tile_to_world(tile, p);
        return g;
    }
    // A globe's rows place the plane affinely (warped sources only
    // approximately, as they hand over).
    g.flat = tile_affine(tile, p);
    if tile.warp_cells > 0u {
        let w = globe_patch(tile, p);
        let origin = vec3<f32>(tile.globe_origin[0], tile.globe_origin[1], tile.globe_origin[2]);
        g.globe = origin + globe_sum(w.z, lagrange(w.x), lagrange(w.y));
    }
    return g;
}

// Clip position of a ground point moved by `offset` world units along the
// map (x east, y north) and raised `z` world units: on the plane, or on the
// globe along its east, north and up there, mixed as the globe hands over.
fn ground_clip(g: Ground, offset: vec2<f32>, z: f32) -> vec4<f32> {
    let flat = frame.view_proj * vec4<f32>(g.flat + offset, z, 1.0);
    if frame.globe_amount <= 0.0 {
        return flat;
    }
    let up = normalize(g.globe + frame.globe_target);
    let level = vec3<f32>(-up.y, up.x, 0.0);
    let east = select(vec3<f32>(0.0, 1.0, 0.0), normalize(level), length(level) > 1e-6);
    let north = cross(up, east);
    let p = g.globe + (east * offset.x + north * offset.y + up * z) * frame.globe_metres;
    var clip = frame.globe_view_proj * vec4<f32>(p, 1.0);
    // Depth from the horizon plane, as GL JS's globe: past the horizon is
    // beyond the far plane.
    let facing = dot(frame.globe_horizon.xyz, p) + frame.globe_horizon.w;
    clip.z = (1.0 - facing * 0.5) * clip.w;
    return mix(flat, clip, frame.globe_amount);
}
