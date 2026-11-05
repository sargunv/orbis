// Port of MapLibre GL JS's circle shaders at the fixture pin (BSD-3-Clause).
// Property slots: color, radius, blur, opacity, stroke color, stroke width,
// stroke opacity. Geometry contains only position and feature slot.
struct CircleNode {
    color: vec4<f32>,
    stroke_color: vec4<f32>,
    radius: f32,
    blur: f32,
    opacity: f32,
    stroke_width: f32,
    stroke_opacity: f32,
    index: u32,
    table_mask: u32,
    // Bit 0: viewport translate, 1: map pitch scale, 2: map alignment.
    flags: u32,
    translate: vec2<f32>,
    _pad: vec2<f32>,
    curves: array<Curve, 7>,
    state: StateLookup,
    gpu: GpuPaint,
};
@group(1) @binding(0) var<uniform> node: CircleNode;

struct CircleOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) stroke_color: vec4<f32>,
    @location(2) data: vec3<f32>,
    @location(3) radius_ratio: f32,
    @location(4) stroke_width: f32,
};

const CIRCLE_PROPS: u32 = 14u;

fn circle_number(row: u32, state: u32, slot: u32, constant: f32, z: f32, gpu: vec4<f32>) -> f32 {
    if gpu_slot(node.gpu, slot) {
        return gpu.x;
    }
    if from_state(state, node.state, slot) {
        return state_number(state, node.state, slot, node.curves[slot]);
    }
    if (node.table_mask & (1u << slot)) != 0u {
        return table_number(row, slot, node.curves[slot], z);
    }
    return constant;
}

@vertex
fn vs_circle(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> CircleOut {
    var out: CircleOut;
    out.position = discarded_vertex();
    var v: Pulled;
    if !pull_point(node.index, instance, &v) { return out; }
    let tile = tile_at(v.tile);
    let p = unpack_i16x2(vertex_at(v.vertex));
    if !point_owned(tile, p) { return out; }
    let feature = vertex_at(v.vertex + 1u);
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);
    // Colours in slots 0 and 4.
    let gpu = gpu_values(node.gpu, row + CIRCLE_PROPS, state, 0x11u);
    let radius = circle_number(row, state, 1u, node.radius, tile.zoom, gpu[1]);
    let blur = circle_number(row, state, 2u, node.blur, tile.zoom, gpu[2]);
    let opacity = circle_number(row, state, 3u, node.opacity, tile.zoom, gpu[3]);
    let stroke_width = circle_number(row, state, 5u, node.stroke_width, tile.zoom, gpu[5]);
    let stroke_opacity = circle_number(row, state, 6u, node.stroke_opacity, tile.zoom, gpu[6]);
    let total = radius + stroke_width;
    if total <= 0.0 { return out; }
    var color = node.color;
    var stroke = node.stroke_color;
    if (node.table_mask & 1u) != 0u { color = table_color(row, 0u, node.curves[0], tile.zoom); }
    if (node.table_mask & 16u) != 0u { stroke = table_color(row, 4u, node.curves[4], tile.zoom); }
    if from_state(state, node.state, 0u) { color = state_color(state, node.state, 0u, node.curves[0]); }
    if from_state(state, node.state, 4u) { stroke = state_color(state, node.state, 4u, node.curves[4]); }
    if gpu_slot(node.gpu, 0u) { color = gpu[0]; }
    if gpu_slot(node.gpu, 4u) { stroke = gpu[4]; }
    let corner = point_corner(index);
    let center = tile_ground(tile, p);
    let shift = translate_world(node.translate, node.flags & 1u, tile);
    // On terrain, the ground under the anchor (GL JS `get_elevation`).
    let z = ground_elevation(tile, p) * frame.extrusion_scale;
    let projected = ground_clip(center, shift, z);
    // At the camera target (camera-relative world 0), clip.w is GL JS's
    // camera-to-center distance in logical pixels.
    let camera_distance = frame.view_proj[3].w;
    let scale_map = (node.flags & 2u) != 0u;
    if (node.flags & 4u) != 0u {
        var scale = total * tile.world_per_px;
        if !scale_map { scale *= projected.w / camera_distance; }
        let offset = corner * vec2<f32>(1.0, -1.0) * scale;
        out.position = ground_clip(center, shift + offset, z);
    } else {
        out.position = projected;
        let scale = select(projected.w, camera_distance, scale_map);
        out.position.x += corner.x * total * 2.0 / frame.viewport.x * scale;
        out.position.y -= corner.y * total * 2.0 / frame.viewport.y * scale;
    }
    // Terrain in front of the centre hides it (GL JS `calculate_visibility`).
    let visibility = terrain_visibility(projected);
    out.color = color * opacity * visibility;
    out.stroke_color = stroke * stroke_opacity * visibility;
    out.data = vec3<f32>(corner, max(1.0 / frame.pixel_ratio / total, blur));
    out.radius_ratio = radius / total;
    out.stroke_width = stroke_width;
    return out;
}

@fragment
fn fs_circle(in: CircleOut) -> @location(0) vec4<f32> {
    let distance = length(in.data.xy);
    let aa = in.data.z;
    let coverage = 1.0 - smoothstep(-aa, 0.0, distance - 1.0);
    let stroke = select(smoothstep(-aa, 0.0, distance - in.radius_ratio), 0.0, in.stroke_width < 0.01);
    let color = coverage * mix(in.color, in.stroke_color, stroke);
    if all(color < vec4<f32>(0.5 / 255.0)) { discard; }
    return color;
}
