// Port of MapLibre GL JS's symbol shaders at the fixture pin
// (symbol_sdf, symbol_icon, symbol_text_and_icon; BSD-3-Clause), with the
// CPU work of `updateLineLabels` and `updateVariableAnchors` moved into the
// vertex stage: each vertex places its glyph along the projected line, or
// shifts its label to the variable anchor placement chose.
//
// One instance per quad: visible chunks of at most 256 quads (orb-render's
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
const ROTATE_IN_SHADER: u32 = 32u;
const VARIABLE_ANCHOR: u32 = 64u;
const IS_OFFSET: u32 = 128u;
const KEEP_UPRIGHT: u32 = 256u;
const TRANSLATE_VIEWPORT: u32 = 512u;
const ICONS_IN_TEXT: u32 = 1024u;
// `symbol-height-anchor: ground`: heights count from the terrain.
const HEIGHT_ANCHOR_GROUND: u32 = 2048u;

// Header words (orb-render `symbol_header`).
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
// them apart when glyphs overlap). A uniform rather than an override: one
// pipeline draws all three, so the largest shader compiles once.
@group(2) @binding(0) var<uniform> symbol_pass: vec4<u32>;

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
    let at = base + vertex_at(base + H_PLACED) + index * PLACED_WORDS;
    var p: Placed;
    p.anchor = unpack_i16x2(vertex_at(at));
    let w1 = vertex_at(at + 1u);
    p.segment = w1 & 0xffffu;
    p.writing_mode = (w1 >> 16u) & 0xffu;
    p.quad_start = vertex_at(at + 2u);
    p.quad_count = vertex_at(at + 3u);
    p.line_start = vertex_at(at + 4u);
    p.line_count = vertex_at(at + 5u);
    let size = vertex_at(at + 6u);
    p.size = vec2<f32>(f32(size & 0xffffu), f32(size >> 16u)) / 128.0;
    p.line_offset = vec2<f32>(bitcast<f32>(vertex_at(at + 7u)), bitcast<f32>(vertex_at(at + 8u)));
    p.instance = vertex_at(at + 9u);
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

// The drawn symbol's `symbol-height-offset`, metres.
var<private> symbol_height: f32;

// Tile units → clip, at the symbol's elevation there: its height offset,
// over the terrain when anchored to the ground (GL JS
// `getSymbolElevation`).
fn tile_clip(tile: Tile, p: vec2<f32>) -> vec4<f32> {
    return ground_clip(tile_ground(tile, p), vec2<f32>(0.0), symbol_metres(tile, p) * frame.extrusion_scale);
}

fn symbol_metres(tile: Tile, p: vec2<f32>) -> f32 {
    var metres = symbol_height;
    if (node.flags & HEIGHT_ANCHOR_GROUND) != 0u {
        metres += ground_elevation(tile, p);
    }
    return metres;
}

// Map-pitched glyphs keep their size on a globe, whose tile units shrink
// toward the poles: GL JS's pitched text correction, the cosine of the
// target's latitude over the anchor's, mixed in as the globe hands over.
fn pitched_correction(anchor: Ground) -> f32 {
    if frame.globe_amount <= 0.0 || frame.globe_plane != 0u {
        return 1.0;
    }
    let up = normalize(anchor.globe + frame.globe_target);
    let centre = normalize(frame.globe_target);
    let ratio = sqrt(max(1.0 - centre.z * centre.z, 1e-6)) / sqrt(max(1.0 - up.z * up.z, 1e-6));
    return mix(1.0, ratio, frame.globe_amount);
}

// On a globe, whether the sphere stands between the eye and the symbol's
// anchor at its height, as GL JS hides symbols behind the planet.
fn globe_hides_anchor(anchor: Ground, metres: f32) -> bool {
    if frame.globe_amount <= 0.0 || frame.globe_plane != 0u {
        return false;
    }
    let up = normalize(anchor.globe + frame.globe_target);
    return sphere_hides(anchor.globe + up * (metres * frame.extrusion_scale * frame.globe_metres));
}

fn clip_to_px(c: vec4<f32>) -> vec2<f32> {
    return vec2<f32>((c.x / c.w * 0.5 + 0.5) * frame.viewport.x, (-c.y / c.w * 0.5 + 0.5) * frame.viewport.y);
}

fn px_to_clip(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x / frame.viewport.x * 2.0 - 1.0, 1.0 - p.y / frame.viewport.y * 2.0);
}

// GL JS `getTileSkewVectors`: east, south.
fn skew_vectors() -> mat2x2<f32> {
    let cp = cos(frame.pitch);
    let sb = sin(frame.bearing);
    let cb = cos(frame.bearing);
    let sr = sin(frame.roll);
    let cr = cos(frame.roll);
    var east = vec2<f32>(cb * cp * cr - sb * sr, sb * cp * cr + cb * sr);
    var south = vec2<f32>(-cb * cp * sr - sb * cr, -sb * cp * sr + cb * cr);
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

// A line label's glyph in the label plane, unless it falls off its line.
struct LineGlyph {
    p: vec2<f32>,
    angle: f32,
    ok: bool,
};

struct SymbolOut {
    @builtin(position) position: vec4<f32>,
    // The page layer, the bitmap's padded rectangle's origin on it, and
    // the quad's texels from there (see `atlas_uv`).
    @location(0) @interpolate(flat) layer: u32,
    @location(1) tex: vec2<f32>,
    @location(7) @interpolate(flat) origin: vec2<f32>,
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
    let entry = visible_at(node_cull_at(node.index).visible_base + instance / CHUNK_QUADS);
    let chunk = chunk_at(entry.chunk);
    let local = instance % CHUNK_QUADS;
    if local >= chunk.index_count { return out; }
    let tile = tile_at(entry.tile);
    let base = chunk.vertex_base;
    let quad_index = vertex_at(base + vertex_at(base + H_ORDER) + chunk.first_index + local);
    let qa = base + vertex_at(base + H_QUADS) + quad_index * QUAD_WORDS;
    let placed_index = vertex_at(qa + 10u);
    let placed = read_placed(base, placed_index);
    if !point_owned(tile, placed.anchor) { return out; }

    // The block of the placement's world copy; the placed table runs up to
    // the glyph offsets.
    let placed_count = (vertex_at(base + H_GLYPH_OFFSETS) - vertex_at(base + H_PLACED)) / PLACED_WORDS;
    let copy = u32(tile.wrap & i32(vertex_at(base + H_COPIES) - 1u));
    let state_at = vertex_at(base + H_STATES) + (copy * placed_count + placed_index) * STATE_WORDS;
    let state = prop_at(state_at);
    if (state & S_HIDDEN) != 0u { return out; }
    let fade_from = f32(state & S_OPACITY) / 127.0;
    let fade_change = select(-frame.symbol_fade_change, frame.symbol_fade_change, (state & S_PLACED) != 0u);
    let fade = clamp(fade_from + fade_change, 0.0, 1.0);

    // Instance rows: the feature's property-row slot, its height offset.
    let instance_row = base + vertex_at(base + H_INSTANCE_ROWS) + placed.instance * 2u;
    let slot = vertex_at(instance_row);
    symbol_height = bitcast<f32>(vertex_at(instance_row + 1u));
    let row = chunk.prop_base + slot * 10u;
    var pulled: Pulled;
    pulled.state_base = chunk.state_base;
    let state_at_slot = state_block(pulled, slot, node.state);
    let opacity = paint_number(row, state_at_slot, 0u, node.opacity, tile.zoom);
    let total_opacity = opacity * fade;
    if total_opacity < 0.1 { return out; }

    let is_text = (node.flags & IS_TEXT) != 0u;
    let pitch = (node.flags & PITCH_WITH_MAP) != 0u;
    var size = feature_size(node.size, placed.size, bitcast<f32>(vertex_at(base + H_LAYOUT_SIZE)),
        bitcast<f32>(vertex_at(base + H_ZOOM_LO)), bitcast<f32>(vertex_at(base + H_ZOOM_HI)));

    // `*-translate`, in tile units.
    var t = node.translate;
    if (node.flags & TRANSLATE_VIEWPORT) != 0u {
        let s = sin(frame.bearing);
        let c = cos(frame.bearing);
        t = vec2<f32>(t.x * c - t.y * s, t.x * s + t.y * c);
    }
    let translation = t * tile_units_per_px(tile);
    let anchor = placed.anchor + translation;
    // The anchor on the ground, once for its projection, the globe's
    // horizon and pitched text's correction (Adreno compilers inline
    // every call).
    let anchor_ground = tile_ground(tile, anchor);
    let anchor_metres = symbol_metres(tile, anchor);
    let projected_point = ground_clip(anchor_ground, vec2<f32>(0.0), anchor_metres * frame.extrusion_scale);
    // Terrain in front of the anchor hides it (GL JS `calculate_visibility`).
    let visible_opacity = total_opacity * terrain_visibility(projected_point);
    if visible_opacity < 0.1 { return out; }
    if globe_hides_anchor(anchor_ground, anchor_metres) { return out; }
    // The anchor's east neighbour, which map-rotated shifted labels and
    // in-shader rotation read: projected at one call site.
    let rotate_shifted = !LINE_LABELS && (state & S_SHIFTED) != 0u && !pitch
        && (node.flags & ROTATE_WITH_MAP) != 0u;
    var east_clip = projected_point;
    if rotate_shifted || (node.flags & ROTATE_IN_SHADER) != 0u {
        east_clip = tile_clip(tile, anchor + vec2<f32>(1.0, 0.0));
    }
    let ctcd = frame.camera_to_center;
    let distance_ratio = select(ctcd / projected_point.w, projected_point.w / ctcd, pitch);
    let perspective_ratio = clamp(0.5 + 0.5 * distance_ratio, 0.0, 4.0);

    // The label plane position and angle: GL JS's dynamic `a_projected_pos`.
    var plane_pos: vec2<f32>;
    var segment_angle = select(0.0, -PI / 2.0, placed.writing_mode == 2u);
    var in_plane = false;
    if LINE_LABELS {
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
            let text_packed = vertex_at(base + vertex_at(base + H_TEXT_SIZES) + placed_index);
            shift_size = feature_size(node.text_size,
                vec2<f32>(f32(text_packed & 0xffffu), f32(text_packed >> 16u)) / 128.0,
                bitcast<f32>(vertex_at(base + H_TEXT_LAYOUT_SIZE)),
                bitcast<f32>(vertex_at(base + H_TEXT_ZOOM_LO)), bitcast<f32>(vertex_at(base + H_TEXT_ZOOM_HI)));
        }
        var render_size = shift_size * (0.5 + 0.5 * ctcd / projected_point.w) / 24.0;
        if pitch {
            render_size *= tile_units_per_px(tile);
        }
        let shift = vec2<f32>(bitcast<f32>(prop_at(state_at + 1u)), bitcast<f32>(prop_at(state_at + 2u))) * render_size;
        if pitch {
            var adjusted = shift;
            if (node.flags & ROTATE_WITH_MAP) == 0u {
                let s = sin(frame.bearing);
                let c = cos(frame.bearing);
                adjusted = vec2<f32>(shift.x * c - shift.y * s, shift.x * s + shift.y * c);
            }
            plane_pos = to_pitched_plane(tile, anchor + adjusted);
        } else {
            // Not pitched: the label plane is the screen.
            let projected = clip_to_px(projected_point);
            if (node.flags & ROTATE_WITH_MAP) != 0u {
                let east = clip_to_px(east_clip) - projected;
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
        let a = projected_point.xy / projected_point.w;
        let b = east_clip.xy / east_clip.w;
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
    let offset = unpack_i16x2(vertex_at(qa + corner));
    let tex_words = vec2<u32>(vertex_at(qa + 4u), vertex_at(qa + 5u));
    let x0 = f32(tex_words.x & 0xffffu);
    let y0 = f32(tex_words.x >> 16u);
    let x1 = f32(tex_words.y & 0xffffu);
    let y1 = f32(tex_words.y >> 16u);
    let px = vec2<f32>(unpack_i16x2(vertex_at(qa + 6u)));
    let px_br = vec2<f32>(unpack_i16x2(vertex_at(qa + 7u)));
    let min_font_scale = unpack_i16x2(vertex_at(qa + 8u)) / 256.0;
    let right = corner == 1u || corner == 3u;
    let bottom = corner >= 2u;
    let pixel_offset = vec2<f32>(select(px.x, px_br.x, right), select(px.y, px_br.y, bottom)) / 16.0;
    let corner_tex = vec2<f32>(select(x0, x1, right), select(y0, y1, bottom));
    let flags = vertex_at(qa + 11u) >> 16u;
    let texture = vertex_at(qa + 9u);

    let glyph_offset_px = rotation * (offset / 32.0 * max(min_font_scale, vec2<f32>(font_scale)) + pixel_offset);
    var final_pos: vec4<f32>;
    if pitch {
        let p = plane_pos + glyph_offset_px * pitched_correction(anchor_ground);
        final_pos = tile_clip(tile, from_pitched_plane(tile, p));
    } else {
        final_pos = vec4<f32>(px_to_clip(plane_pos + glyph_offset_px), 0.0, 1.0);
    }
    out.position = final_pos;
    out.tex = corner_tex;

    // Texels: the padded rectangle's origin plus the quad's texels.
    let image = (flags & Q_IMAGE) != 0u;
    if image {
        let rect = vertex_at(base + vertex_at(base + H_IMAGES) + texture);
        if rect == 0xffffffffu {
            return out;
        }
        out.layer = rect >> 24u;
        out.origin = vec2<f32>(f32(rect & 0xfffu), f32((rect >> 12u) & 0xfffu));
    } else {
        let atlas_row = vertex_at(base + vertex_at(base + H_GLYPHS) + texture);
        // The glyph row table is the property arena's first block.
        let g = atlas_row * 4u;
        let origin = prop_at(g);
        out.layer = prop_at(g + 2u);
        out.origin = vec2<f32>(f32(origin & 0xffffu), f32(origin >> 16u));
        if prop_at(g + 1u) == 0u {
            // Not rasterized yet: nothing to draw.
            out.position = discarded_vertex();
        }
    }
    out.data = vec3<f32>(final_pos.w, size, visible_opacity);
    if (flags & Q_COLOR) != 0u {
        out.fill = unpack4x8unorm(vertex_at(base + vertex_at(base + H_COLORS) + quad_index));
    } else {
        out.fill = paint_color(row, state_at_slot, 1u, node.fill_color, tile.zoom);
    }
    out.halo = paint_color(row, state_at_slot, 2u, node.halo_color, tile.zoom);
    out.halo_params = vec2<f32>(paint_number(row, state_at_slot, 3u, node.halo_width, tile.zoom), paint_number(row, state_at_slot, 4u, node.halo_blur, tile.zoom));
    let linear = (node.flags & LINEAR_ICONS) != 0u || (vertex_at(base + H_FLAGS) & 1u) != 0u
        || abs(frame.pitch) > 0.0;
    out.kind = select(0u, 1u, (flags & Q_SDF) != 0u) | select(0u, 2u, image)
        | select(0u, 4u, linear);
    return out;
}

const SDF_PX: f32 = 8.0;

// Where a fragment samples its page. The quad's texels round to 1/1024
// texel, so adding the rectangle's whole-texel origin is exact in f32 and
// the sampler's filter weights don't depend on where the atlas packed the
// bitmap, which the order glyphs and images arrived in decides.
fn atlas_uv(in: SymbolOut, page: f32) -> vec2<f32> {
    return (in.origin + round(in.tex * 1024.0) / 1024.0) / page;
}

@fragment
fn fs_symbol(in: SymbolOut) -> @location(0) vec4<f32> {
    let image = (in.kind & 2u) != 0u;
    let sdf = (in.kind & 1u) != 0u;
    let total_opacity = in.data.z;
    let uv = atlas_uv(in, select(node.glyph_page, node.image_page, image));
    if image && !sdf {
        if (in.kind & 4u) != 0u {
            return textureSampleLevel(image_pages, atlas_sampler, uv, in.layer, 0.0) * total_opacity;
        }
        return textureSampleLevel(image_pages, nearest_sampler, uv, in.layer, 0.0) * total_opacity;
    }
    var dist: f32;
    if image {
        dist = textureSampleLevel(image_pages, atlas_sampler, uv, in.layer, 0.0).a;
    } else {
        dist = textureSampleLevel(glyph_pages, atlas_sampler, uv, in.layer, 0.0).r;
    }
    let is_text = (node.flags & IS_TEXT) != 0u;
    let size = in.data.y;
    let gamma_scale = in.data.x;
    let font_scale = select(size, size / 24.0, is_text);
    let edge_gamma = 0.105 / frame.pixel_ratio;
    let u_gamma_scale = select(1.0, cos(frame.pitch) * frame.camera_to_center, (node.flags & PITCH_WITH_MAP) != 0u);
    let gamma = edge_gamma / (font_scale * u_gamma_scale);
    let inner_edge = (256.0 - 64.0) / 256.0;

    var text = vec4<f32>(0.0);
    var halo = vec4<f32>(0.0);
    let halo_width = in.halo_params.x;
    let halo_blur = in.halo_params.y;
    if symbol_pass.x != 1u {
        let g = gamma * gamma_scale;
        text = total_opacity * smoothstep(inner_edge - g, inner_edge + g, dist) * in.fill;
    }
    let has_halo = halo_width != 0.0 && any(in.halo != vec4<f32>(0.0));
    if symbol_pass.x != 2u && has_halo {
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
