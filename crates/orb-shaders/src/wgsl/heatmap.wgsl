// GL JS's Gaussian kernel, at the fixture pin (BSD-3-Clause).
struct HeatmapNode {
    radius: f32,
    weight: f32,
    intensity: f32,
    opacity: f32,
    index: u32,
    table_mask: u32,
    _pad: vec2<u32>,
    curves: array<Curve, 2>,
};
@group(1) @binding(0) var<uniform> node: HeatmapNode;
struct HeatmapOut {
    @builtin(position) position: vec4<f32>,
    @location(0) extrude: vec2<f32>,
    @location(1) weight: f32,
};

@vertex
fn vs_heatmap(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> HeatmapOut {
    var out: HeatmapOut;
    out.position = discarded_vertex();
    var v: Pulled;
    if !pull_point(node.index, instance, &v) { return out; }
    let tile = tile_at(v.tile);
    let p = unpack_i16x2(vertex_at(v.vertex));
    if !point_owned(tile, p) { return out; }
    let row = prop_row(v, vertex_at(v.vertex + 1u), 4u);
    var radius = node.radius;
    var weight = node.weight;
    if (node.table_mask & 1u) != 0u { radius = table_number(row, 0u, node.curves[0], tile.zoom); }
    if (node.table_mask & 2u) != 0u { weight = table_number(row, 1u, node.curves[1], tile.zoom); }
    let amplitude = weight * node.intensity * 0.3989422804014327;
    let zero = 1.0 / 255.0 / 16.0;
    if amplitude <= zero || radius <= 0.0 { return out; }
    let scale = sqrt(-2.0 * log(zero / amplitude)) / 3.0;
    out.extrude = point_corner(index) * scale;
    let offset = out.extrude * vec2<f32>(1.0, -1.0) * radius * tile_world_per_px(tile);
    let z = ground_elevation(tile, p) * frame.extrusion_scale;
    out.position = ground_clip(tile_ground(tile, p), offset, z);
    out.weight = weight;
    return out;
}

@fragment
fn fs_heatmap(in: HeatmapOut) -> @location(0) vec4<f32> {
    let density = in.weight * node.intensity * 0.3989422804014327 * exp(-4.5 * dot(in.extrude, in.extrude));
    return vec4<f32>(density, 1.0, 1.0, 1.0);
}
