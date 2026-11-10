// Fill family: triangles (`vs_fill`) and antialiasing outlines as line
// lists (`vs_fill_outline`, after GL JS's `fill_outline`). Vertex:
// [x | y << 16, feature slot].
// Property row (`FILL_ROW_*`): color, opacity.

struct FillNode {
    color: vec4<f32>,
    opacity: f32,
    // The node's index in the node tables.
    index: u32,
    // Bit i: property i comes from the property row.
    table_mask: u32,
    // 1 when `fill-translate-anchor` is `viewport`.
    translate_anchor: u32,
    // `fill-translate`, in pixels.
    translate: vec2<f32>,
    image_from: u32,
    image_to: u32,
    curves: array<Curve, 2>,
    translate_curve: Curve,
    pattern: Pattern,
    state: StateLookup,
};

@group(1) @binding(0) var<uniform> node: FillNode;

const FILL_STRIDE: u32 = 2u;

struct FillOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tile_pos: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    // The vertex in framebuffer pixels less `screen_origin`, for outline
    // coverage.
    @location(3) screen: vec2<f32>,
    @location(4) opacity: f32,
    @location(5) pattern_pos: vec2<f32>,
    @location(6) @interpolate(flat) images: vec2<u32>,
    // Where the placement owns the fill: where it lands on the map's plane,
    // where it was on a globe and in a drape slot (`tile_pos` is where it
    // lands there), so a translated fill moves whole.
    @location(7) owner_pos: vec2<f32>,
    @location(8) @interpolate(flat) clip: vec2<u32>,
    // An outline's `drape_origin`: carried flat, so its fragments read no
    // tile for it.
    @location(9) @interpolate(flat) screen_origin: vec2<f32>,
};

@vertex
fn vs_fill(@builtin(vertex_index) index: u32) -> FillOut {
    var v: Pulled;
    if !pull_indexed(node.index, index, FILL_STRIDE, &v) {
        return discarded_fill();
    }
    return fill_vertex(v);
}

@vertex
fn vs_fill_outline(@builtin(vertex_index) index: u32) -> FillOut {
    var v: Pulled;
    if !pull_indexed(node.index, index, FILL_STRIDE, &v) {
        return discarded_fill();
    }
    return fill_outline_vertex(v);
}

fn discarded_fill() -> FillOut {
    var out: FillOut;
    out.position = discarded_vertex();
    return out;
}

fn fill_outline_vertex(v: Pulled) -> FillOut {
    var out = fill_vertex(v);
    let ndc = out.position.xy / out.position.w;
    let size = frame.viewport * frame.pixel_ratio;
    out.screen_origin = drape_origin(tile_at(v.tile));
    out.screen = vec2<f32>(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * size - out.screen_origin;
    return out;
}

fn fill_vertex(v: Pulled) -> FillOut {
    var out: FillOut;
    let tile = tile_at(v.tile);
    let pos = unpack_i16x2(vertex_at(v.vertex));
    let feature = vertex_at(v.vertex + 1u);
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);

    var color = node.color;
    if (node.table_mask & 1u) != 0u {
        color = table_color(row, 0u, node.curves[0], tile.zoom);
    }
    if from_state(state, node.state, 0u) {
        color = state_color(state, node.state, 0u, node.curves[0], color);
    }
    var opacity = node.opacity;
    if (node.table_mask & 2u) != 0u {
        opacity = table_number(row, 1u, node.curves[1], tile.zoom);
    }
    if from_state(state, node.state, 1u) {
        opacity = state_number(state, node.state, 1u, node.curves[1], opacity);
    }

    let px = node_translate(node.translate, node.table_mask, row, v.row_words, node.translate_curve, tile.zoom);
    let shift = translate_world(px, node.translate_anchor, tile);
    let ground = tile_ground_shifted(tile, pos, shift);
    out.position = ground_clip(ground, vec2<f32>(0.0), 0.0);
    out.color = color_in(color) * opacity;
    out.opacity = opacity;
    out.pattern_pos = pos;
    out.images = pattern_ids(row, FILL_ROW_PATTERN, node.image_from, node.image_to, node.table_mask, node.pattern);
    // Translated geometry is clipped where it lands, as GL JS's stencil
    // does on a plane; on a globe, where GL JS clips no tile, where it was,
    // so a fill crossing tiles moves whole, as it does into a drape slot,
    // bounded by the slot where it lands.
    out.tile_pos = pos;
    if !on_globe() {
        out.tile_pos += vec2<f32>(shift.x, -shift.y) / tile.scale;
    }
    out.owner_pos = select(out.tile_pos, pos, draped_placement(tile));
    out.tile = v.tile;
    out.clip = fragment_clip(tile, any(shift != vec2<f32>(0.0)));
    return out;
}

// Whether the placement draws a fill's fragment: it owns where the fill
// was or lands (`owner_pos`), and it lands in the placement's bound. A
// warped tile's fills interpolate their tile position across mapped
// triangles and can stray just past the tile's edge, where no other
// placement draws; there it belongs to the edge.
fn fill_drawn(in: FillOut) -> bool {
    if any(in.clip != CLIP_FETCHED) {
        return clip_drawn(in.clip, in.tile, in.owner_pos, in.tile_pos);
    }
    let tile = tile_at(in.tile);
    var owner = in.owner_pos;
    if warped(tile) {
        owner = clamp(owner, vec2<f32>(0.0), vec2<f32>(tile.extent - 0.5));
    }
    return owned(tile, owner) && landed(tile, in.tile_pos);
}

@fragment
fn fs_fill(in: FillOut) -> @location(0) vec4<f32> {
    if !fill_drawn(in) {
        discard;
    }
    if node.pattern.enabled == 1u {
        return fill_pattern_color(in.images, tile_at(in.tile), in.pattern_pos, node.pattern, in.color, in.opacity);
    }
    return in.color;
}

@fragment
fn fs_fill_outline(in: FillOut) -> @location(0) vec4<f32> {
    if !clip_drawn(in.clip, in.tile, in.owner_pos, in.tile_pos) {
        discard;
    }
    let dist = length(in.screen - (in.position.xy - in.screen_origin));
    var color = in.color;
    if node.pattern.enabled == 1u {
        color = fill_pattern_color(in.images, tile_at(in.tile), in.pattern_pos, node.pattern, in.color, in.opacity);
    }
    return color * (1.0 - smoothstep(0.0, 1.0, dist));
}
