// Extrusions, lit by v8's `light` (lighting.wgsl), ported from
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
    // Color, height, base, and per-feature opacity.
    curves: array<Curve, 4>,
    pattern: Pattern,
    state: StateLookup,
};
@group(1) @binding(0) var<uniform> node: ExtrusionNode;
// `table_mask` bits beyond the row slots: opacity and the vertical gradient
// per feature, from the row's flags word.
const EXTRUSION_OPACITY_ROWS: u32 = 8u;
const EXTRUSION_GRADIENT_ROWS: u32 = 16u;
struct ExtrusionOut {
    @builtin(position) position: vec4<f32>,
    //planes @builtin(clip_distances) clip: array<f32, 4>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    @location(3) pattern_pos: vec2<f32>,
    @location(4) @interpolate(flat) images: vec2<u32>,
};
// Extrusions draw in two pipelines: whole buildings, and buildings cut at
// the source's buffer, clipped per fragment to the placement. Only the
// second discards, which would turn off the GPU's hidden-surface removal
// for every building it drew: a quarter of a dense 3D frame.
@vertex
fn vs_extrusion(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, 1u << EXTRUSION_WHOLE);
}
// Every cut building, clipped per fragment: where clip distances aren't.
@vertex
fn vs_extrusion_cut(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, (1u << EXTRUSION_CUT) | (1u << EXTRUSION_CUT_SPLIT));
}
// Where they are (`CLIP_PLANES`): cut buildings of a placement with one clip
// rectangle, clipped by its four planes, which needs no discard...
@vertex
fn vs_extrusion_cut_planes(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, 1u << EXTRUSION_CUT);
}
// ...and those of a placement with several, clipped per fragment.
@vertex
fn vs_extrusion_cut_split(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, 1u << EXTRUSION_CUT_SPLIT);
}
const EXTRUSION_WHOLE: u32 = 1u;
const EXTRUSION_CUT: u32 = 2u;
const EXTRUSION_CUT_SPLIT: u32 = 3u;
// Vertex `index` if its building is drawn here as one of `parts` (bits).
fn extrusion_part(index: u32, parts: u32) -> ExtrusionOut {
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) || (parts & (1u << extrusion_drawn(v))) == 0u {
        var out: ExtrusionOut;
        out.position = discarded_vertex();
        return out;
    }
    var out = extrusion_vertex(v);
    //planes out.clip = extrusion_planes(tile_at(v.tile), out.tile_pos);
    return out;
}
// The distances inside a placement's one clip rectangle from position `p`,
// for clip planes: both bounds count, as `extrusion_inside`'s do.
fn extrusion_planes(tile: Tile, p: vec2<f32>) -> array<f32, 4> {
    var lo = vec2<f32>(0.0);
    var hi = vec2<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let l = unpack_i16x2(tile.clips[i].x);
        let h = unpack_i16x2(tile.clips[i].y);
        if all(l < h) {
            lo = l;
            hi = h;
        }
    }
    return array<f32, 4>(p.x - lo.x, hi.x - p.x, p.y - lo.y, hi.y - p.y);
}
// Whether a placement clips to more than one rectangle.
fn clips_split(tile: Tile) -> bool {
    var count = 0u;
    for (var i = 0u; i < 4u; i++) {
        count += u32(all(unpack_i16x2(tile.clips[i].x) < unpack_i16x2(tile.clips[i].y)));
    }
    return count > 1u;
}
// How a building draws from this placement: 0 not at all, EXTRUSION_WHOLE,
// or cut: EXTRUSION_CUT, or EXTRUSION_CUT_SPLIT where the placement clips to
// several rectangles. A whole building belongs to the one placement holding
// its centroid (half open, as points are owned), which draws all of it,
// past its clip rectangles: its walls and their pattern run on unbroken. A
// building cut at the source's buffer draws from every placement.
fn extrusion_drawn(v: Pulled) -> u32 {
    let row = prop_row(v, vertex_at(v.vertex + 1u), v.row_words);
    if (prop_at(row + EXTRUSION_ROW_WHOLE) & 1u) == 0u {
        return select(EXTRUSION_CUT, EXTRUSION_CUT_SPLIT, clips_split(tile_at(v.tile)));
    }
    return select(0u, EXTRUSION_WHOLE, inside_clips(tile_at(v.tile), unpack_i16x2(prop_at(row + EXTRUSION_ROW_CENTROID))));
}
// Whether a cut building's fragment at `tile_pos` lies in its placement.
// Shared tile edges can contain real vertical walls: they have zero ground
// area, and depth selects their nearest surface without blending twice, so
// both bounds count.
fn extrusion_inside(tile: Tile, tile_pos: vec2<f32>) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        if all(tile_pos >= lo) && all(tile_pos <= hi) {
            return true;
        }
    }
    return false;
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
    // Per-feature opacity and vertical gradient (`orbis:expressions`),
    // packed beside the whole-building bit.
    let flags = prop_at(row + EXTRUSION_ROW_WHOLE);
    var opacity = 1.0;
    if (node.table_mask & EXTRUSION_OPACITY_ROWS) != 0u {
        let at = vec2<f32>(f32((flags >> 8u) & 255u), f32((flags >> 16u) & 255u)) / 255.0;
        opacity = mix(at.x, at.y, curve_t(node.curves[3], tile.zoom));
    }
    var vertical_gradient = node.vertical_gradient != 0u;
    if (node.table_mask & EXTRUSION_GRADIENT_ROWS) != 0u {
        vertical_gradient = (flags & 2u) != 0u;
    }
    // On terrain, raise the roof by the ground under the feature's
    // centroid and give ground-level parts a 10 m basement,
    // so buildings don't hang over slopes (GL JS's `a_centroid`).
    let gradient_base = max(0.0, base);
    let gradient_height = max(0.0, height);
    let centroid = unpack_i16x2(prop_at(row + EXTRUSION_ROW_CENTROID));
    let raise = ground_elevation(tile, centroid);
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
    // Modern uses reversed Z; the shared camera matrix remains usable by 2D.
    out.position.z = out.position.w - out.position.z;
    out.tile_pos = pos;
    out.tile = v.tile;
    var gradient = 1.0;
    if normal.z == 0.0 && vertical_gradient {
        gradient = light_wall_gradient(top + gradient_base, gradient_height);
    }
    if node.pattern.enabled != 0u {
        out.images = pattern_ids(row, EXTRUSION_ROW_PATTERN, node.image_from, node.image_to, node.table_mask, node.pattern);
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
        out.color = vec4<f32>(clamp(light_factor(normal, 0.5, gradient), light_floor(), vec3<f32>(1.0)), 1.0) * opacity;
        return out;
    }
    out.color = vec4<f32>(light_surface(color.rgb, normal, gradient), 1.0) * opacity;
    return out;
}
fn extrusion_shade(in: ExtrusionOut) -> vec4<f32> {
    if node.pattern.enabled == 0u {
        return in.color;
    }
    let tile = tile_at(in.tile);
    let image_a = image_at(in.images.x); let image_b = image_at(in.images.y);
    let pos_a = pattern_position(tile, in.pattern_pos, image_a, node.pattern.from_scale);
    let pos_b = pattern_position(tile, in.pattern_pos, image_b, node.pattern.to_scale);
    let a = pattern_sample(image_a, fract(pos_a), 0.0);
    let b = pattern_sample(image_b, fract(pos_b), 0.0);
    return mix(a, b, node.pattern.fade) * in.color;
}
