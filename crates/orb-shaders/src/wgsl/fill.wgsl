// Fill family: triangles (`vs_fill`) and antialiasing outlines as line
// lists (`vs_fill_outline`, after GL JS's `fill_outline`). Vertex:
// [x | y << 16, feature slot].
// Property row, two words per property (see `table_number`): color, opacity.

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
    pattern: Pattern,
    state: StateLookup,
    gpu: GpuPaint,
};

@group(1) @binding(0) var<uniform> node: FillNode;

const FILL_STRIDE: u32 = 2u;
const FILL_PROPS: u32 = 7u;

struct FillOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tile_pos: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    // The vertex in framebuffer pixels, for outline coverage.
    @location(3) screen: vec2<f32>,
    @location(4) opacity: f32,
    @location(5) pattern_pos: vec2<f32>,
    @location(6) @interpolate(flat) images: vec2<u32>,
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
    out.screen = vec2<f32>(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * size;
    return out;
}

fn fill_vertex(v: Pulled) -> FillOut {
    var out: FillOut;
    let tile = tile_at(v.tile);
    let pos = unpack_i16x2(vertex_at(v.vertex));
    let feature = vertex_at(v.vertex + 1u);
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);

    let gpu = gpu_values(node.gpu, row + FILL_PROPS, state, 1u);
    var color = node.color;
    if (node.table_mask & 1u) != 0u {
        color = table_color(row, 0u, node.curves[0], tile.zoom);
    }
    if gpu_slot(node.gpu, 0u) {
        color = gpu[0];
    }
    if from_state(state, node.state, 0u) {
        color = state_color(state, node.state, 0u, node.curves[0]);
    }
    var opacity = node.opacity;
    if (node.table_mask & 2u) != 0u {
        opacity = table_number(row, 1u, node.curves[1], tile.zoom);
    }
    if gpu_slot(node.gpu, 1u) {
        opacity = gpu[1].x;
    }
    if from_state(state, node.state, 1u) {
        opacity = state_number(state, node.state, 1u, node.curves[1]);
    }

    let shift = translate_world(node.translate, node.translate_anchor, tile);
    out.position = ground_clip(tile_ground_shifted(tile, pos, shift), vec2<f32>(0.0), 0.0);
    out.color = color * opacity;
    out.opacity = opacity;
    out.pattern_pos = pos;
    out.images = pattern_ids(row, 4u, node.image_from, node.image_to, node.table_mask, node.pattern);
    // Translated geometry is clipped where it lands, as GL JS's stencil
    // does.
    out.tile_pos = pos + vec2<f32>(shift.x, -shift.y) / tile.scale;
    out.tile = v.tile;
    return out;
}

// A fill's position for its clip test. A warped tile's fills interpolate
// their tile position across mapped triangles and can stray just past the
// tile's edge, where no other placement draws; there it belongs to the edge.
fn fill_clip_pos(in: FillOut) -> vec2<f32> {
    let tile = tile_at(in.tile);
    if tile.warp_cells == 0u {
        return in.tile_pos;
    }
    return clamp(in.tile_pos, vec2<f32>(0.0), vec2<f32>(tile.extent - 0.5));
}

@fragment
fn fs_fill(in: FillOut) -> @location(0) vec4<f32> {
    if !inside_clips(tile_at(in.tile), fill_clip_pos(in)) {
        discard;
    }
    if node.pattern.enabled == 1u {
        return fill_pattern_color(in.images, tile_at(in.tile), in.pattern_pos, node.pattern, in.color, in.opacity);
    }
    return in.color;
}

@fragment
fn fs_fill_outline(in: FillOut) -> @location(0) vec4<f32> {
    if !inside_clips(tile_at(in.tile), in.tile_pos) {
        discard;
    }
    let dist = length(in.screen - in.position.xy);
    var color = in.color;
    if node.pattern.enabled == 1u {
        color = fill_pattern_color(in.images, tile_at(in.tile), in.pattern_pos, node.pattern, in.color, in.opacity);
    }
    return color * (1.0 - smoothstep(0.0, 1.0, dist));
}
