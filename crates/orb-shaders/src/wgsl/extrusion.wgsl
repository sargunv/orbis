// Flat v8 extrusion shading, ported from MapLibre GL JS (BSD-3-Clause).
struct ExtrusionNode {
    color: vec4<f32>,
    height: f32,
    base: f32,
    index: u32,
    table_mask: u32,
    translate: vec2<f32>,
    translate_anchor: u32,
    vertical_gradient: u32,
    light_position: vec3<f32>,
    light_viewport: u32,
    light_color: vec3<f32>,
    light_intensity: f32,
    opacity: f32,
    image_from: u32, image_to: u32, _pad0: u32,
    curves: array<Curve, 3>,
    pattern: Pattern,
    state: StateLookup,
};
@group(1) @binding(0) var<uniform> node: ExtrusionNode;
struct ExtrusionOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    @location(3) pattern_pos: vec2<f32>,
    @location(4) @interpolate(flat) images: vec2<u32>,
};
@vertex
fn vs_extrusion(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) {
        var out: ExtrusionOut;
        out.position = discarded_vertex();
        return out;
    }
    return extrusion_vertex(v);
}
fn extrusion_vertex(v: Pulled) -> ExtrusionOut {
    var out: ExtrusionOut;
    let tile = tiles[v.tile];
    let pos = unpack_i16x2(vertices[v.vertex]);
    let feature = vertices[v.vertex+1u];
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);
    var color = node.color;
    var height = node.height;
    var base = node.base;
    if (node.table_mask & 1u) != 0u { color = table_color(row, 0u, node.curves[0], tile.zoom); }
    if (node.table_mask & 2u) != 0u { height = table_number(row, 1u, node.curves[1], tile.zoom); }
    if (node.table_mask & 4u) != 0u { base = table_number(row, 2u, node.curves[2], tile.zoom); }
    if from_state(state, node.state, 0u) { color = state_color(state, node.state, 0u, node.curves[0]); }
    if from_state(state, node.state, 1u) { height = state_number(state, node.state, 1u, node.curves[1]); }
    if from_state(state, node.state, 2u) { base = state_number(state, node.state, 2u, node.curves[2]); }
    // On terrain, raise the roof by the ground under the feature's
    // centroid (row word 9) and give ground-level parts a 10 m basement,
    // so buildings don't hang over slopes (GL JS's `a_centroid`).
    let gradient_base = max(0.0, base);
    let gradient_height = max(0.0, height);
    let raise = ground_elevation(tile, unpack_i16x2(props[row + 9u]));
    let basement = select(0.0, 10.0, base <= 0.0 && frame.dem_dim > 0.0);
    height = gradient_height + raise;
    base = gradient_base + raise - basement;
    let packed = vertices[v.vertex+2u];
    let top = f32(packed & 1u);
    let normal = vec3<f32>(unpack_i16x2(packed), unpack_i16x2(vertices[v.vertex+3u]).x) / select(16384.0, 16383.0, node.pattern.enabled != 0u);
    let elevation = mix(base, height, top);
    let shift = translate_world(node.translate, node.translate_anchor, tile);
    out.position = frame.view_proj * vec4<f32>(tile_to_world(tile, pos) + shift, elevation * frame.extrusion_scale, 1.0);
    // Modern uses reversed Z; the shared camera matrix remains usable by 2D.
    out.position.z = out.position.w - out.position.z;
    out.tile_pos = pos;
    out.tile = v.tile;
    var light = node.light_position;
    if node.light_viewport != 0u {
        let s = sin(frame.bearing); let c = cos(frame.bearing);
        light = vec3<f32>(light.x*c - light.y*s, light.x*s + light.y*c, light.z);
    }
    var luminance = dot(color.rgb, vec3<f32>(0.2126,0.7152,0.0722));
    if node.pattern.enabled != 0u { luminance = 0.5; }
    var directional = clamp(dot(normal, light), 0.0, 1.0);
    directional = mix(1.0-node.light_intensity, max(1.0-luminance+node.light_intensity, 1.0), directional);
    if normal.z == 0.0 && node.vertical_gradient != 0u {
        directional *= clamp((top+gradient_base)*sqrt(gradient_height/150.0), mix(0.7,0.98,1.0-node.light_intensity), 1.0);
    }
    if node.pattern.enabled != 0u {
        out.images = pattern_ids(row, 6u, node.image_from, node.image_to, node.table_mask, node.pattern);
        var pattern_pos = pos;
        if normal.z == 0.0 {
            let distance = f32(vertices[v.vertex+3u] >> 16u);
            pattern_pos = vec2<f32>(distance, -elevation * exp2(floor(frame.zoom)) / 65536.0 / tile_pixel_ratio(tile));
        }
        // Keep atlas reads in the fragment stage, within core WebGPU's eight
        // storage slots per stage.
        out.pattern_pos = pattern_pos;
        out.color = vec4<f32>(clamp(directional * node.light_color, (vec3<f32>(1.0)-node.light_color)*0.3, vec3<f32>(1.0)), 1.0);
        return out;
    }
    out.color = vec4<f32>(clamp((color.rgb + vec3<f32>(0.03)) * directional * node.light_color,
        (vec3<f32>(1.0)-node.light_color)*0.3, vec3<f32>(1.0)), 1.0);
    return out;
}
@fragment
fn fs_extrusion(in: ExtrusionOut) -> @location(0) vec4<f32> {
    // Shared tile edges can contain real vertical walls. They have zero
    // ground area, and depth selects their nearest surface without blending
    // twice, so include both bounds for 3D clipping.
    let tile = tiles[in.tile];
    var inside = false;
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        inside = inside || (all(in.tile_pos >= lo) && all(in.tile_pos <= hi));
    }
    if !inside { discard; }
    if node.pattern.enabled != 0u {
        let image_a = images[in.images.x]; let image_b = images[in.images.y];
        let pos_a = pattern_position(tile, in.pattern_pos, image_a, node.pattern.from_scale);
        let pos_b = pattern_position(tile, in.pattern_pos, image_b, node.pattern.to_scale);
        let a = pattern_sample(image_a, fract(pos_a), 0.0);
        let b = pattern_sample(image_b, fract(pos_b), 0.0);
        return mix(a,b,node.pattern.fade) * in.color;
    }
    return in.color;
}
