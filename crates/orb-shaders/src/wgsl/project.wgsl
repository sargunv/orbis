// Tile data units → camera-relative world (plan §9.3, `project_tile`).
//
// An affine tile is `origin + (x, -y) * scale`. A warped tile (the planner's
// `GpuPerVertex` plan) is `origin` plus one of its `warp_cells`² biquadratic
// patches, from f64 positions on the CPU (orb-crs `Series`): each patch is
// its 3×3 nodes' positions (corners, edge midpoints, centre; row by row, top
// to bottom), 18 f32 in `props`, interpolated by quadratic Lagrange
// polynomials in parameters s (right) and t (down) in [-1, 1]. Neighbours
// share their edges' nodes, so tiles and patches meet exactly.

// The point's patch (its first word in `props`) and its parameters there.
fn warp_patch(tile: Tile, p: vec2<f32>) -> vec3<f32> {
    let cells = f32(tile.warp_cells);
    let u = p / tile.extent * cells;
    let cell = clamp(floor(u), vec2<f32>(0.0), vec2<f32>(cells - 1.0));
    let st = (u - cell) * 2.0 - 1.0;
    let index = u32(cell.y) * tile.warp_cells + u32(cell.x);
    return vec3<f32>(st, f32(tile.warp_base + index * 18u));
}

fn warp_node(at: f32, k: u32) -> vec2<f32> {
    let i = u32(at) + 2u * k;
    return vec2<f32>(bitcast<f32>(props[i]), bitcast<f32>(props[i + 1u]));
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
    let w = warp_patch(tile, p);
    let ds = warp_sum(w.z, lagrange_slope(w.x), lagrange(w.y));
    let dt = warp_sum(w.z, lagrange(w.x), lagrange_slope(w.y));
    // Parameters span 2 per patch side: 2 · cells per tile side.
    let per_unit = 2.0 * f32(tile.warp_cells) / tile.extent;
    return sqrt(abs(ds.x * dt.y - ds.y * dt.x)) * per_unit;
}
