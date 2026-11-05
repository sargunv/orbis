// Extrusions, lit by the style's lights (lighting.wgsl), ported from
// MapLibre GL JS (BSD-3-Clause).
struct ExtrusionNode {
    color: vec4<f32>,
    height: f32,
    base: f32,
    index: u32,
    table_mask: u32,
    translate: vec2<f32>,
    translate_anchor: u32,
    vertical_gradient: u32,
    opacity: f32,
    image_from: u32, image_to: u32,
    // How much of its own colour it shows (0: lit fully).
    emissive: f32,
    curves: array<Curve, 3>,
    pattern: Pattern,
    state: StateLookup,
    // Where the layer fades out before its tiles' buckets end: x the
    // layer's minzoom less the source's centre zoom, y 1 to fade.
    fade: vec2<f32>,
};
@group(1) @binding(0) var<uniform> node: ExtrusionNode;
struct ExtrusionOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    @location(3) pattern_pos: vec2<f32>,
    @location(4) @interpolate(flat) images: vec2<u32>,
    @location(5) ground: vec2<f32>,
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
    let tile = tile_at(v.tile);
    let pos = unpack_i16x2(vertex_at(v.vertex));
    let feature = vertex_at(v.vertex+1u);
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
    let raise = ground_elevation(tile, unpack_i16x2(prop_at(row + 9u)));
    let basement = select(0.0, 10.0, base <= 0.0 && frame.dem_dim > 0.0);
    height = gradient_height + raise;
    base = gradient_base + raise - basement;
    let packed = vertex_at(v.vertex+2u);
    let top = f32(packed & 1u);
    let normal = vec3<f32>(unpack_i16x2(packed), unpack_i16x2(vertex_at(v.vertex+3u)).x) / select(16384.0, 16383.0, node.pattern.enabled != 0u);
    let elevation = mix(base, height, top);
    let shift = translate_world(node.translate, node.translate_anchor, tile);
    let ground = tile_ground_shifted(tile, pos, shift);
    out.position = ground_clip_3d(ground, vec2<f32>(0.0), elevation * frame.extrusion_scale);
    out.ground = ground.flat;
    // Modern uses reversed Z; the shared camera matrix remains usable by 2D.
    out.position.z = out.position.w - out.position.z;
    out.tile_pos = pos;
    out.tile = v.tile;
    var gradient = 1.0;
    if normal.z == 0.0 && node.vertical_gradient != 0u {
        gradient = light_wall_gradient(top + gradient_base, gradient_height);
    }
    if node.pattern.enabled != 0u {
        out.images = pattern_ids(row, 6u, node.image_from, node.image_to, node.table_mask, node.pattern);
        var pattern_pos = pos;
        if normal.z == 0.0 {
            let distance = f32(vertex_at(v.vertex+3u) >> 16u);
            pattern_pos = vec2<f32>(distance, -elevation * exp2(floor(frame.zoom)) / 65536.0 / tile_pixel_ratio(tile));
        }
        // Keep atlas reads in the fragment stage, within core WebGPU's eight
        // storage slots per stage.
        out.pattern_pos = pattern_pos;
        // The pattern's texels multiply the light, as GL JS shades them at
        // mid luminance.
        let light = clamp(light_factor(normal, 0.5, gradient),
            (vec3<f32>(1.0) - frame.light_color.rgb) * 0.3, vec3<f32>(1.0));
        out.color = vec4<f32>(mix(light, vec3<f32>(1.0), node.emissive), 1.0);
        return out;
    }
    out.color = vec4<f32>(light_surface(color.rgb, normal, gradient, node.emissive), 1.0);
    return out;
}
@fragment
fn fs_extrusion(in: ExtrusionOut) -> @location(0) vec4<f32> {
    // Shared tile edges can contain real vertical walls. They have zero
    // ground area, and depth selects their nearest surface without blending
    // twice, so include both bounds for 3D clipping.
    let tile = tile_at(in.tile);
    var inside = false;
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        inside = inside || (all(in.tile_pos >= lo) && all(in.tile_pos <= hi));
    }
    if !inside { discard; }
    // The cover's tiles here, as zoom levels past the centre's: from below
    // the layer's minzoom they have no buckets, so it fades out over the
    // half level before.
    var fade = 1.0;
    if node.fade.y > 0.0 {
        let d = max(length(vec2<f32>(distance(in.ground, frame.lod_eye), frame.lod_height)), 1e-9);
        let lod = frame.lod_offset - log2(d) + frame.lod_behavior * log2(frame.lod_height / d);
        fade = smoothstep(node.fade.x, node.fade.x + 0.5, lod);
        if fade <= 0.0 { discard; }
    }
    if node.pattern.enabled != 0u {
        let image_a = image_at(in.images.x); let image_b = image_at(in.images.y);
        let pos_a = pattern_position(tile, in.pattern_pos, image_a, node.pattern.from_scale);
        let pos_b = pattern_position(tile, in.pattern_pos, image_b, node.pattern.to_scale);
        let a = pattern_sample(image_a, fract(pos_a), 0.0);
        let b = pattern_sample(image_b, fract(pos_b), 0.0);
        return mix(a,b,node.pattern.fade) * in.color * fade;
    }
    return in.color * fade;
}
