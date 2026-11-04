// Port of MapLibre GL JS's symbol shaders at the fixture pin
// (symbol_sdf, symbol_icon, symbol_text_and_icon; BSD-3-Clause), with the
// CPU work of `updateLineLabels` and `updateVariableAnchors` moved into the
// vertex stage: each vertex places its glyph along the projected line, or
// shifts its label to the variable anchor placement chose.
//
// One instance per quad: visible chunks of at most 256 quads (orb-scene's
// symbol meshes), six vertices each. A mesh's tables sit behind a header in
// the vertex arena; its state words, paint rows and the glyph row table in
// the property arena.

// Paint slots: opacity, colour, halo colour, halo width, halo blur.
struct SymbolSize {
    // 0 constant, 1 source, 2 camera, 3 composite.
    kind: u32,
    // A constant size; unused otherwise.
    size: f32,
    stop_count: u32,
    _pad: u32,
    // A camera or composite size's interpolation.
    curve: Curve,
    // A camera size's (zoom, size) stops, two per vec4.
    stops: array<vec4<f32>, 4>,
};

struct SymbolNode {
    fill_color: vec4<f32>,
    halo_color: vec4<f32>,
    opacity: f32,
    halo_width: f32,
    halo_blur: f32,
    index: u32,
    table_mask: u32,
    flags: u32,
    // First word of the glyph row table in the property arena.
    glyph_table: u32,
    // Texels on a side of a glyph page and of an image page.
    glyph_page: f32,
    // `*-translate` in pixels.
    translate: vec2<f32>,
    image_page: f32,
    _pad: u32,
    size: SymbolSize,
    // The text size an icon fitted to variable-anchored text moves with.
    text_size: SymbolSize,
    curves: array<Curve, 5>,
    state: StateLookup,
};
@group(1) @binding(0) var<uniform> node: SymbolNode;

@group(0) @binding(9) var image_pages: texture_2d_array<f32>;
@group(0) @binding(10) var atlas_sampler: sampler;
@group(0) @binding(13) var nearest_sampler: sampler;
@group(0) @binding(12) var glyph_pages: texture_2d_array<f32>;

// Flags of SymbolNode.
const IS_TEXT: u32 = 1u;
const LINEAR_ICONS: u32 = 2u;
const PITCH_WITH_MAP: u32 = 4u;
const ROTATE_WITH_MAP: u32 = 8u;
const ALONG_LINE: u32 = 16u;
const ROTATE_IN_SHADER: u32 = 32u;
const VARIABLE_ANCHOR: u32 = 64u;
const IS_OFFSET: u32 = 128u;
const KEEP_UPRIGHT: u32 = 256u;
const TRANSLATE_VIEWPORT: u32 = 512u;
const ICONS_IN_TEXT: u32 = 1024u;

// Header words (orb-scene `symbol_header`).
const H_QUADS: u32 = 0u;
const H_PLACED: u32 = 1u;
const H_GLYPH_OFFSETS: u32 = 2u;
const H_LINES: u32 = 3u;
const H_GLYPHS: u32 = 4u;
const H_IMAGES: u32 = 5u;
const H_INSTANCE_ROWS: u32 = 6u;
const H_TEXT_SIZES: u32 = 7u;
const H_ORDER: u32 = 8u;
const H_COLORS: u32 = 9u;
const H_STATES: u32 = 10u;
// Blocks of state words, one per world copy (a power of two).
const H_COPIES: u32 = 11u;
// Params: the size's layout size and covering zooms, then the text size's.
const H_LAYOUT_SIZE: u32 = 12u;
const H_ZOOM_LO: u32 = 13u;
const H_ZOOM_HI: u32 = 14u;
const H_TEXT_LAYOUT_SIZE: u32 = 15u;
const H_TEXT_ZOOM_LO: u32 = 16u;
const H_TEXT_ZOOM_HI: u32 = 17u;
// Bit 0: the bucket's images need linear filtering.
const H_FLAGS: u32 = 18u;

// State word bits (orb-symbol `state`).
const S_OPACITY: u32 = 0x7fu;
const S_PLACED: u32 = 0x80u;
const S_HIDDEN: u32 = 0x100u;
const S_SHIFTED: u32 = 0x200u;
const S_QUARTER_TURN: u32 = 0x400u;

// Quad flags.
const Q_SDF: u32 = 1u;
const Q_IMAGE: u32 = 2u;
const Q_COLOR: u32 = 4u;

const QUAD_WORDS: u32 = 12u;
const PLACED_WORDS: u32 = 11u;
const STATE_WORDS: u32 = 3u;
const EXTENT: f32 = 8192.0;
const PI: f32 = 3.14159265358979;
const CHUNK_QUADS: u32 = 256u;

// 0: halo and fill in one pass; 1: halo only; 2: fill only (GL JS draws
// them apart when glyphs overlap).
override PASS: u32 = 0u;

struct Placed {
    anchor: vec2<f32>,
    segment: u32,
    writing_mode: u32,
    quad_start: u32,
    quad_count: u32,
    line_start: u32,
    line_count: u32,
    size: vec2<f32>,
    line_offset: vec2<f32>,
    instance: u32,
};

fn read_placed(base: u32, index: u32) -> Placed {
    let at = base + vertices[base + H_PLACED] + index * PLACED_WORDS;
    var p: Placed;
    p.anchor = unpack_i16x2(vertices[at]);
    let w1 = vertices[at + 1u];
    p.segment = w1 & 0xffffu;
    p.writing_mode = (w1 >> 16u) & 0xffu;
    p.quad_start = vertices[at + 2u];
    p.quad_count = vertices[at + 3u];
    p.line_start = vertices[at + 4u];
    p.line_count = vertices[at + 5u];
    let size = vertices[at + 6u];
    p.size = vec2<f32>(f32(size & 0xffffu), f32(size >> 16u)) / 128.0;
    p.line_offset = vec2<f32>(bitcast<f32>(vertices[at + 7u]), bitcast<f32>(vertices[at + 8u]));
    p.instance = vertices[at + 9u];
    return p;
}

// GL JS `Interpolate.interpolationFactor`, clamped to [0, 1].
fn curve_factor(c: Curve, x: f32, lo: f32, hi: f32) -> f32 {
    let d = hi - lo;
    let p = x - lo;
    var t = 0.0;
    switch c.kind {
        case 1u: {
            let base = c.params.x;
            if d == 0.0 {
                t = 0.0;
            } else if abs(base - 1.0) < 1e-6 {
                t = p / d;
            } else {
                t = (pow(base, p) - 1.0) / (pow(base, d) - 1.0);
            }
        }
        case 2u: {
            if d != 0.0 {
                t = unit_bezier(c.params, clamp(p / d, 0.0, 1.0));
            }
        }
        default: {}
    }
    return clamp(t, 0.0, 1.0);
}

// GL JS `evaluateSizeForZoom` and `evaluateSizeForFeature`.
fn feature_size(s: SymbolSize, packed: vec2<f32>, layout_size: f32, lo: f32, hi: f32) -> f32 {
    switch s.kind {
        case 1u: { return packed.x; }
        case 2u: {
            var lower = i32(s.stop_count) - 1;
            loop {
                if lower <= 0 { break; }
                let stop = s.stops[u32(lower) / 2u];
                let z = select(stop.z, stop.x, (u32(lower) & 1u) == 0u);
                if z <= frame.zoom { break; }
                lower -= 1;
            }
            let upper = min(lower + 1, i32(s.stop_count) - 1);
            let a = s.stops[u32(lower) / 2u];
            let b = s.stops[u32(upper) / 2u];
            let za = select(a.z, a.x, (u32(lower) & 1u) == 0u);
            let sa = select(a.w, a.y, (u32(lower) & 1u) == 0u);
            let zb = select(b.z, b.x, (u32(upper) & 1u) == 0u);
            let sb = select(b.w, b.y, (u32(upper) & 1u) == 0u);
            let t = curve_factor(s.curve, frame.zoom, za, zb);
            return min(mix(sa, sb, t), layout_size);
        }
        case 3u: {
            let t = curve_factor(s.curve, frame.zoom, lo, hi);
            return mix(packed.x, packed.y, t);
        }
        default: { return s.size; }
    }
}

// `state` is the feature's state block (see `state_block`).
fn paint_number(row: u32, state: u32, slot: u32, constant: f32, z: f32) -> f32 {
    if from_state(state, node.state, slot) {
        return state_number(state, node.state, slot, node.curves[slot]);
    }
    if (node.table_mask & (1u << slot)) != 0u {
        return table_number(row, slot, node.curves[slot], z);
    }
    return constant;
}

fn paint_color(row: u32, state: u32, slot: u32, constant: vec4<f32>, z: f32) -> vec4<f32> {
    if from_state(state, node.state, slot) {
        return state_color(state, node.state, slot, node.curves[slot]);
    }
    if (node.table_mask & (1u << slot)) != 0u {
        return table_color(row, slot, node.curves[slot], z);
    }
    return constant;
}

// Tile units (and metres up) → clip.
fn tile_clip(tile: Tile, p: vec2<f32>) -> vec4<f32> {
    return frame.view_proj * vec4<f32>(tile_to_world(tile, p), 0.0, 1.0);
}

fn clip_to_px(c: vec4<f32>) -> vec2<f32> {
    return vec2<f32>((c.x / c.w * 0.5 + 0.5) * frame.viewport.x, (-c.y / c.w * 0.5 + 0.5) * frame.viewport.y);
}

fn px_to_clip(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x / frame.viewport.x * 2.0 - 1.0, 1.0 - p.y / frame.viewport.y * 2.0);
}

fn camera_to_center() -> f32 {
    return frame.view_proj[3].w;
}

// GL JS `getTileSkewVectors` without roll: east, south.
fn skew_vectors() -> mat2x2<f32> {
    let cp = cos(frame.pitch);
    let sb = sin(frame.bearing);
    let cb = cos(frame.bearing);
    var east = vec2<f32>(cb * cp, sb * cp);
    var south = vec2<f32>(-sb, cb);
    east = select(vec2<f32>(0.0), normalize(east), length(east) > 1e-9);
    south = select(vec2<f32>(0.0), normalize(south), length(south) > 1e-9);
    return mat2x2<f32>(east, south);
}

// Tile units per logical pixel (GL JS `pixelsToTileUnits(tile, 1, zoom)`).
fn tile_units_per_px(tile: Tile) -> f32 {
    return EXTENT / (512.0 * exp2(frame.zoom - tile.zoom));
}

// The pitched label plane (GL JS `getPitchedLabelPlaneMatrix`): tile units
// → plane pixels, and back (`getGlCoordMatrix`).
fn to_pitched_plane(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    var m = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    if (node.flags & ROTATE_WITH_MAP) == 0u {
        let s = skew_vectors();
        let det = determinant(s);
        if det != 0.0 {
            m = mat2x2<f32>(vec2<f32>(s[1].y, -s[0].y), vec2<f32>(-s[1].x, s[0].x)) * (1.0 / det);
        } else {
            m = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
        }
    }
    return m * p / tile_units_per_px(tile);
}

fn from_pitched_plane(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    var m = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    if (node.flags & ROTATE_WITH_MAP) == 0u {
        m = skew_vectors();
    }
    return m * p * tile_units_per_px(tile);
}

// A tile point in the label plane (GL JS
// `projectTileCoordinatesToLabelPlane`), and its clip w.
struct PlanePoint {
    p: vec2<f32>,
    w: f32,
};

fn to_label_plane(tile: Tile, p: vec2<f32>) -> PlanePoint {
    if (node.flags & PITCH_WITH_MAP) != 0u {
        return PlanePoint(to_pitched_plane(tile, p), 1.0);
    }
    let c = tile_clip(tile, p);
    return PlanePoint(clip_to_px(c), c.w);
}

// A label-plane point in clip space (GL JS
// `projectFromLabelPlaneToClipSpace`), for orientation tests.
fn label_plane_to_clip(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    if (node.flags & PITCH_WITH_MAP) != 0u {
        let c = tile_clip(tile, from_pitched_plane(tile, p));
        return c.xy / c.w;
    }
    return px_to_clip(p);
}

fn line_vertex(base: u32, placed: Placed, i: u32) -> vec2<f32> {
    return unpack_i16x2(vertices[base + vertices[base + H_LINES] + placed.line_start + i]);
}

// A line vertex in the label plane; one behind the camera becomes a point
// far enough along the line (GL JS `projectLineVertexToLabelPlane`).
fn project_line_vertex(tile: Tile, base: u32, placed: Placed, i: u32, previous_tile: vec2<f32>, previous: vec2<f32>, needed: f32) -> vec2<f32> {
    let v = line_vertex(base, placed, i);
    let q = to_label_plane(tile, v);
    if q.w > 0.0 {
        return q.p;
    }
    let unit_vertex = previous_tile + normalize(previous_tile - v);
    let projected = to_label_plane(tile, unit_vertex).p;
    let segment = previous - projected;
    return previous + segment * (needed / length(segment));
}

struct LineGlyph {
    p: vec2<f32>,
    angle: f32,
    ok: bool,
};

fn perp_normal(v: vec2<f32>, offset: f32, direction: f32) -> vec2<f32> {
    let u = normalize(v);
    return vec2<f32>(-u.y, u.x) * offset * direction;
}

// GL JS `isLineDoublingBack`: the line turns straight back at vertex `i`
// (tile coordinates are integers, so the test is exact).
fn doubles_back(base: u32, placed: Placed, i: i32, direction: i32) -> bool {
    let previous = i - direction;
    let next = i + direction;
    if min(previous, next) < 0 || max(previous, next) >= i32(placed.line_count) {
        return false;
    }
    let p = vec2<i32>(line_vertex(base, placed, u32(previous)));
    let c = vec2<i32>(line_vertex(base, placed, u32(i)));
    let n = vec2<i32>(line_vertex(base, placed, u32(next)));
    let a = c - p;
    let b = n - c;
    return a.x * b.y - a.y * b.x == 0 && a.x * b.x + a.y * b.y < 0;
}

// GL JS `findLineIntersection` of the lines through `a1 a2` and `b1 b2`;
// `a2` where they are parallel.
fn line_intersection(a1: vec2<f32>, a2: vec2<f32>, b1: vec2<f32>, b2: vec2<f32>) -> vec2<f32> {
    let denominator = (b2.y - b1.y) * (a2.x - a1.x) - (b2.x - b1.x) * (a2.y - a1.y);
    if denominator == 0.0 {
        return a2;
    }
    let ua = ((b2.x - b1.x) * (a1.y - b1.y) - (b2.y - b1.y) * (a1.x - b1.x)) / denominator;
    return a1 + ua * (a2 - a1);
}

// GL JS `findOffsetIntersectionPoint`: where the offset segment into vertex
// `i` meets the offset segment out of it (a miter join).
fn offset_intersection(tile: Tile, base: u32, placed: Placed, i: i32, normal: vec2<f32>, current: vec2<f32>, offset_previous: vec2<f32>, offset_y: f32, direction: f32, next_previous_tile: vec2<f32>, previous: vec2<f32>, needed: f32) -> vec2<f32> {
    let offset_current = current + normal;
    let n = i + i32(direction);
    if n < 0 || n >= i32(placed.line_count) || doubles_back(base, placed, i, i32(direction)) {
        return offset_current;
    }
    let next = project_line_vertex(tile, base, placed, u32(n), next_previous_tile, previous, needed);
    if all(next == current) {
        return offset_current;
    }
    let next_normal = perp_normal(next - current, offset_y, direction);
    return line_intersection(offset_previous, offset_current, current + next_normal, next + next_normal);
}

// GL JS `placeGlyphAlongLine`.
fn place_glyph(tile: Tile, base: u32, placed: Placed, anchor: vec2<f32>, offset_x: f32, line_offset: vec2<f32>, flip: bool) -> LineGlyph {
    var out: LineGlyph;
    out.ok = false;
    let combined = select(offset_x + line_offset.x, offset_x - line_offset.x, flip);
    var direction = select(-1.0, 1.0, combined > 0.0);
    var angle = 0.0;
    if flip {
        direction = -direction;
        angle = PI;
    }
    if direction < 0.0 {
        angle += PI;
    }
    var index = i32(placed.segment) + select(1, 0, direction > 0.0);
    var current = anchor;
    var previous = anchor;
    var previous_tile = placed.anchor;
    var distance = 0.0;
    var segment_distance = 0.0;
    let abs_offset = abs(combined);
    var segment = vec2<f32>(0.0);
    var start = anchor;
    // The offset line's previous vertex and its miter at the current one.
    var offset_point = vec2<f32>(0.0);
    var have_offset_point = false;
    for (var guard = 0u; guard < 4096u; guard++) {
        if distance + segment_distance > abs_offset {
            break;
        }
        index += i32(direction);
        if index < 0 || index >= i32(placed.line_count) {
            return out;
        }
        let first = distance + segment_distance == 0.0;
        distance += segment_distance;
        previous = current;
        let needed = abs_offset - distance + 1.0;
        current = project_line_vertex(tile, base, placed, u32(index), previous_tile, previous, needed);
        previous_tile = line_vertex(base, placed, u32(index));
        if line_offset.y == 0.0 {
            start = previous;
            segment = current - previous;
        } else {
            // As GL JS's synthetic vertex: the anchor before the first step.
            let next_previous_tile = select(previous_tile, placed.anchor, first);
            var normal: vec2<f32>;
            if all(current == previous) {
                // The anchor sits on the vertex: look one vertex ahead.
                let n = clamp(index + i32(direction), 0, i32(placed.line_count) - 1);
                let next = project_line_vertex(tile, base, placed, u32(n), next_previous_tile, previous, needed);
                normal = select(perp_normal(next - current, line_offset.y, direction), vec2<f32>(0.0), all(next == current));
            } else {
                normal = perp_normal(current - previous, line_offset.y, direction);
            }
            start = select(previous + normal, offset_point, have_offset_point);
            offset_point = offset_intersection(tile, base, placed, index, normal, current, start, line_offset.y, direction, next_previous_tile, previous, needed);
            have_offset_point = true;
            segment = offset_point - start;
        }
        segment_distance = length(segment);
    }
    let t = select(0.0, (abs_offset - distance) / segment_distance, segment_distance > 0.0);
    out.p = start + segment * t;
    let rotate_to_line = (node.flags & ROTATE_WITH_MAP) != 0u;
    out.angle = select(0.0, angle + atan2(current.y - previous.y, current.x - previous.x), rotate_to_line);
    out.ok = true;
    return out;
}

fn glyph_offset(base: u32, quad: u32) -> f32 {
    return bitcast<f32>(vertices[base + vertices[base + H_GLYPH_OFFSETS] + quad]);
}

// A line label's glyph in the label plane: GL JS `placeGlyphsAlongLine`
// with its keep-upright flip and vertical switch, decided per vertex from
// the first and last glyphs.
fn line_glyph(tile: Tile, base: u32, placed: Placed, quad: u32, font_scale: f32) -> LineGlyph {
    var fail: LineGlyph;
    fail.ok = false;
    let anchor = to_label_plane(tile, placed.anchor).p;
    let line_offset = placed.line_offset * font_scale;
    let own = glyph_offset(base, quad) * font_scale;
    let keep_upright = (node.flags & KEEP_UPRIGHT) != 0u;
    let aspect = frame.viewport.x / frame.viewport.y;
    var flip = false;
    if placed.quad_count > 1u {
        let first_offset = glyph_offset(base, placed.quad_start) * font_scale;
        let last_offset = glyph_offset(base, placed.quad_start + placed.quad_count - 1u) * font_scale;
        let first = place_glyph(tile, base, placed, anchor, first_offset, line_offset, false);
        let last = place_glyph(tile, base, placed, anchor, last_offset, line_offset, false);
        if !first.ok || !last.ok {
            return fail;
        }
        if keep_upright {
            let a = label_plane_to_clip(tile, first.p);
            let b = label_plane_to_clip(tile, last.p);
            // A horizontal label with a vertical alternative steeper than
            // flat gives way to it; the vertical one shows only then.
            let steep = abs(b.y - a.y) > abs(b.x - a.x) * aspect;
            if placed.writing_mode == 1u && steep {
                return fail;
            }
            if placed.writing_mode == 2u && !steep {
                return fail;
            }
            if select(a.x > b.x, a.y < b.y, placed.writing_mode == 2u) {
                flip = true;
                let ff = place_glyph(tile, base, placed, anchor, first_offset, line_offset, true);
                let lf = place_glyph(tile, base, placed, anchor, last_offset, line_offset, true);
                if !ff.ok || !lf.ok {
                    return fail;
                }
            }
        }
    } else if keep_upright {
        let segment_end = line_vertex(base, placed, placed.segment + 1u);
        let b = to_label_plane(tile, segment_end);
        let ca = label_plane_to_clip(tile, anchor);
        let cb = label_plane_to_clip(tile, select(anchor + (anchor - to_label_plane(tile, placed.anchor + normalize(placed.anchor - segment_end)).p), b.p, b.w > 0.0));
        flip = select(ca.x > cb.x, ca.y < cb.y, placed.writing_mode == 2u);
    }
    return place_glyph(tile, base, placed, anchor, own, line_offset, flip);
}

struct SymbolOut {
    @builtin(position) position: vec4<f32>,
    // Glyph or image texel coordinates over page size, and the page layer.
    @location(0) @interpolate(flat) layer: u32,
    @location(1) tex: vec2<f32>,
    // Gamma scale, size and total opacity.
    @location(2) data: vec3<f32>,
    @location(3) fill: vec4<f32>,
    @location(4) halo: vec4<f32>,
    // Halo width, halo blur; whether the quad is an SDF, an image.
    @location(5) halo_params: vec2<f32>,
    @location(6) @interpolate(flat) kind: u32,
};

@vertex
fn vs_symbol(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> SymbolOut {
    var out: SymbolOut;
    out.position = discarded_vertex();
    let entry = visible[node_cull[node.index].visible_base + instance / CHUNK_QUADS];
    let chunk = chunks[entry.chunk];
    let local = instance % CHUNK_QUADS;
    if local >= chunk.index_count { return out; }
    let tile = tiles[entry.tile];
    let base = chunk.vertex_base;
    let quad_index = vertices[base + vertices[base + H_ORDER] + chunk.first_index + local];
    let qa = base + vertices[base + H_QUADS] + quad_index * QUAD_WORDS;
    let placed_index = vertices[qa + 10u];
    let placed = read_placed(base, placed_index);
    if !point_owned(tile, placed.anchor) { return out; }

    // The block of the placement's world copy; the placed table runs up to
    // the glyph offsets.
    let placed_count = (vertices[base + H_GLYPH_OFFSETS] - vertices[base + H_PLACED]) / PLACED_WORDS;
    let copy = u32(tile.wrap & i32(vertices[base + H_COPIES] - 1u));
    let state_at = vertices[base + H_STATES] + (copy * placed_count + placed_index) * STATE_WORDS;
    let state = props[state_at];
    if (state & S_HIDDEN) != 0u { return out; }
    let fade_from = f32(state & S_OPACITY) / 127.0;
    let fade_change = select(-frame.symbol_fade_change, frame.symbol_fade_change, (state & S_PLACED) != 0u);
    let fade = clamp(fade_from + fade_change, 0.0, 1.0);

    let slot = vertices[base + vertices[base + H_INSTANCE_ROWS] + placed.instance];
    let row = chunk.prop_base + slot * 10u;
    var pulled: Pulled;
    pulled.state_base = chunk.state_base;
    let state_at_slot = state_block(pulled, slot, node.state);
    let opacity = paint_number(row, state_at_slot, 0u, node.opacity, tile.zoom);
    let total_opacity = opacity * fade;
    if total_opacity < 0.1 { return out; }

    let is_text = (node.flags & IS_TEXT) != 0u;
    let pitch = (node.flags & PITCH_WITH_MAP) != 0u;
    var size = feature_size(node.size, placed.size, bitcast<f32>(vertices[base + H_LAYOUT_SIZE]),
        bitcast<f32>(vertices[base + H_ZOOM_LO]), bitcast<f32>(vertices[base + H_ZOOM_HI]));

    // `*-translate`, in tile units.
    var t = node.translate;
    if (node.flags & TRANSLATE_VIEWPORT) != 0u {
        let s = sin(frame.bearing);
        let c = cos(frame.bearing);
        t = vec2<f32>(t.x * c - t.y * s, t.x * s + t.y * c);
    }
    let translation = t * tile_units_per_px(tile);
    let anchor = placed.anchor + translation;
    let projected_point = tile_clip(tile, anchor);
    let ctcd = camera_to_center();
    let distance_ratio = select(ctcd / projected_point.w, projected_point.w / ctcd, pitch);
    let perspective_ratio = clamp(0.5 + 0.5 * distance_ratio, 0.0, 4.0);

    // The label plane position and angle: GL JS's dynamic `a_projected_pos`.
    var plane_pos: vec2<f32>;
    var segment_angle = select(0.0, -PI / 2.0, placed.writing_mode == 2u);
    var in_plane = false;
    if (node.flags & ALONG_LINE) != 0u {
        // Only labels whose anchor is near the viewport are placed.
        let clip = projected_point.xy / projected_point.w;
        let buffer = vec2<f32>(256.0) / frame.viewport * 2.0 + 1.0;
        if any(abs(clip) > buffer) { return out; }
        // GL JS `updateLineLabels`: the unclamped ratio, sizes over 24.
        let line_ratio = 0.5 + 0.5 * ctcd / projected_point.w;
        let plane_size = select(size * line_ratio, size / line_ratio, pitch);
        let glyph = line_glyph(tile, base, placed, quad_index, plane_size / 24.0);
        if !glyph.ok { return out; }
        plane_pos = glyph.p;
        segment_angle = -glyph.angle;
        in_plane = true;
    } else if (state & S_SHIFTED) != 0u {
        // GL JS `updateVariableAnchorsForBucket` and `getShiftedAnchor`.
        var shift_size = size;
        if !is_text {
            let text_packed = vertices[base + vertices[base + H_TEXT_SIZES] + placed_index];
            shift_size = feature_size(node.text_size,
                vec2<f32>(f32(text_packed & 0xffffu), f32(text_packed >> 16u)) / 128.0,
                bitcast<f32>(vertices[base + H_TEXT_LAYOUT_SIZE]),
                bitcast<f32>(vertices[base + H_TEXT_ZOOM_LO]), bitcast<f32>(vertices[base + H_TEXT_ZOOM_HI]));
        }
        var render_size = shift_size * (0.5 + 0.5 * ctcd / projected_point.w) / 24.0;
        if pitch {
            render_size *= tile_units_per_px(tile);
        }
        let shift = vec2<f32>(bitcast<f32>(props[state_at + 1u]), bitcast<f32>(props[state_at + 2u])) * render_size;
        if pitch {
            var adjusted = shift;
            if (node.flags & ROTATE_WITH_MAP) == 0u {
                let s = sin(frame.bearing);
                let c = cos(frame.bearing);
                adjusted = vec2<f32>(shift.x * c - shift.y * s, shift.x * s + shift.y * c);
            }
            plane_pos = to_pitched_plane(tile, anchor + adjusted);
        } else {
            let projected = to_label_plane(tile, anchor).p;
            if (node.flags & ROTATE_WITH_MAP) != 0u {
                let east = to_label_plane(tile, anchor + vec2<f32>(1.0, 0.0)).p - projected;
                let a = atan(east.y / east.x) + select(0.0, PI, east.x < 0.0);
                let s = sin(a);
                let c = cos(a);
                plane_pos = projected + vec2<f32>(shift.x * c - shift.y * s, shift.x * s + shift.y * c);
            } else {
                plane_pos = projected + shift;
            }
        }
        let quarter = select(placed.writing_mode == 2u, (state & S_QUARTER_TURN) != 0u, !is_text);
        segment_angle = select(0.0, -PI / 2.0, quarter);
        in_plane = true;
    }

    if (node.flags & IS_OFFSET) == 0u {
        size *= perspective_ratio;
    }
    let font_scale = select(size, size / 24.0, is_text);

    var symbol_rotation = 0.0;
    if (node.flags & ROTATE_IN_SHADER) != 0u {
        let b4 = tile_clip(tile, anchor + vec2<f32>(1.0, 0.0));
        let a = projected_point.xy / projected_point.w;
        let b = b4.xy / b4.w;
        symbol_rotation = atan2((b.y - a.y) / (frame.viewport.x / frame.viewport.y), b.x - a.x);
    }
    let angle_sin = sin(segment_angle + symbol_rotation);
    let angle_cos = cos(segment_angle + symbol_rotation);
    let rotation = mat2x2<f32>(vec2<f32>(angle_cos, -angle_sin), vec2<f32>(angle_sin, angle_cos));

    if !in_plane {
        if pitch {
            plane_pos = to_pitched_plane(tile, anchor);
        } else {
            plane_pos = clip_to_px(projected_point);
        }
    }

    // Corners tl, tr, bl, br; two triangles.
    let corner = array<u32, 6>(0u, 1u, 2u, 1u, 2u, 3u)[index % 6u];
    let offset = unpack_i16x2(vertices[qa + corner]);
    let tex_words = vec2<u32>(vertices[qa + 4u], vertices[qa + 5u]);
    let x0 = f32(tex_words.x & 0xffffu);
    let y0 = f32(tex_words.x >> 16u);
    let x1 = f32(tex_words.y & 0xffffu);
    let y1 = f32(tex_words.y >> 16u);
    let px = vec2<f32>(unpack_i16x2(vertices[qa + 6u]));
    let px_br = vec2<f32>(unpack_i16x2(vertices[qa + 7u]));
    let min_font_scale = unpack_i16x2(vertices[qa + 8u]) / 256.0;
    let right = corner == 1u || corner == 3u;
    let bottom = corner >= 2u;
    let pixel_offset = vec2<f32>(select(px.x, px_br.x, right), select(px.y, px_br.y, bottom)) / 16.0;
    let corner_tex = vec2<f32>(select(x0, x1, right), select(y0, y1, bottom));
    let flags = vertices[qa + 11u] >> 16u;
    let texture = vertices[qa + 9u];

    let glyph_offset_px = rotation * (offset / 32.0 * max(min_font_scale, vec2<f32>(font_scale)) + pixel_offset);
    var final_pos: vec4<f32>;
    if pitch {
        let p = plane_pos + glyph_offset_px;
        final_pos = tile_clip(tile, from_pitched_plane(tile, p));
    } else {
        final_pos = vec4<f32>(px_to_clip(plane_pos + glyph_offset_px), 0.0, 1.0);
    }
    out.position = final_pos;

    // Texels: the padded rectangle's origin plus the quad's texels.
    let image = (flags & Q_IMAGE) != 0u;
    if image {
        let rect = vertices[base + vertices[base + H_IMAGES] + texture];
        if rect == 0xffffffffu {
            return out;
        }
        out.layer = rect >> 24u;
        out.tex = (vec2<f32>(f32(rect & 0xfffu), f32((rect >> 12u) & 0xfffu)) + corner_tex) / node.image_page;
    } else {
        let atlas_row = vertices[base + vertices[base + H_GLYPHS] + texture];
        let g = node.glyph_table + atlas_row * 4u;
        let origin = props[g];
        out.layer = props[g + 2u];
        out.tex = (vec2<f32>(f32(origin & 0xffffu), f32(origin >> 16u)) + corner_tex) / node.glyph_page;
        if props[g + 1u] == 0u {
            // Not rasterized yet: nothing to draw.
            out.position = discarded_vertex();
        }
    }
    out.data = vec3<f32>(final_pos.w, size, total_opacity);
    if (flags & Q_COLOR) != 0u {
        out.fill = unpack4x8unorm(vertices[base + vertices[base + H_COLORS] + quad_index]);
    } else {
        out.fill = paint_color(row, state_at_slot, 1u, node.fill_color, tile.zoom);
    }
    out.halo = paint_color(row, state_at_slot, 2u, node.halo_color, tile.zoom);
    out.halo_params = vec2<f32>(paint_number(row, state_at_slot, 3u, node.halo_width, tile.zoom), paint_number(row, state_at_slot, 4u, node.halo_blur, tile.zoom));
    let linear = (node.flags & LINEAR_ICONS) != 0u || (vertices[base + H_FLAGS] & 1u) != 0u
        || abs(frame.pitch) > 0.0;
    out.kind = select(0u, 1u, (flags & Q_SDF) != 0u) | select(0u, 2u, image)
        | select(0u, 4u, linear);
    return out;
}

const SDF_PX: f32 = 8.0;

@fragment
fn fs_symbol(in: SymbolOut) -> @location(0) vec4<f32> {
    let image = (in.kind & 2u) != 0u;
    let sdf = (in.kind & 1u) != 0u;
    let total_opacity = in.data.z;
    if image && !sdf {
        if (in.kind & 4u) != 0u {
            return textureSampleLevel(image_pages, atlas_sampler, in.tex, in.layer, 0.0) * total_opacity;
        }
        return textureSampleLevel(image_pages, nearest_sampler, in.tex, in.layer, 0.0) * total_opacity;
    }
    var dist: f32;
    if image {
        dist = textureSampleLevel(image_pages, atlas_sampler, in.tex, in.layer, 0.0).a;
    } else {
        dist = textureSampleLevel(glyph_pages, atlas_sampler, in.tex, in.layer, 0.0).r;
    }
    let is_text = (node.flags & IS_TEXT) != 0u;
    let size = in.data.y;
    let gamma_scale = in.data.x;
    let font_scale = select(size, size / 24.0, is_text);
    let edge_gamma = 0.105 / frame.pixel_ratio;
    let u_gamma_scale = select(1.0, cos(frame.pitch) * camera_to_center(), (node.flags & PITCH_WITH_MAP) != 0u);
    let gamma = edge_gamma / (font_scale * u_gamma_scale);
    let inner_edge = (256.0 - 64.0) / 256.0;

    var text = vec4<f32>(0.0);
    var halo = vec4<f32>(0.0);
    let halo_width = in.halo_params.x;
    let halo_blur = in.halo_params.y;
    if PASS != 1u {
        let g = gamma * gamma_scale;
        text = total_opacity * smoothstep(inner_edge - g, inner_edge + g, dist) * in.fill;
    }
    let has_halo = halo_width != 0.0 && any(in.halo != vec4<f32>(0.0));
    if PASS != 2u && has_halo {
        let gamma_halo = (halo_blur * 1.19 / SDF_PX + edge_gamma) / (font_scale * u_gamma_scale);
        let inner_edge_halo = inner_edge + gamma_halo * gamma_scale;
        let g = gamma_halo * gamma_scale;
        var alpha = smoothstep(inner_edge_halo - g, inner_edge_halo + g, dist);
        let halo_edge = (6.0 - halo_width / font_scale) / SDF_PX;
        alpha = min(smoothstep(halo_edge - g, halo_edge + g, dist), 1.0 - alpha);
        halo = total_opacity * alpha * in.halo;
    }
    let color = text + (1.0 - text.a) * halo;
    if color.a <= 0.0 { discard; }
    return color;
}
