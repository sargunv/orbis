// Port of MapLibre GL JS's symbol shaders at the fixture pin
// (symbol_sdf, symbol_icon, symbol_text_and_icon; BSD-3-Clause), with the
// CPU work of `updateLineLabels` and `updateVariableAnchors` moved to the
// GPU: each quad's glyph is placed along the projected line, or its label
// shifted to the variable anchor placement chose.
//
// Where the device has compute, a dispatch does that once a frame for
// every quad (`symbol_place.wgsl`), writing where it lands, turned and
// scaled, and its paint into two texels of the placement target, and the
// symbol draw (`symbol_draw.wgsl`) puts each of its four corners there.
// Run in every corner of every pass, as it still is on WebGL 2, the
// placement was most of the symbols' time on the moto g (#192).
//
// One instance per quad: visible chunks of at most 64 quads (orb-render's
// symbol meshes), four vertices each (six indices) in the draw and one
// invocation in the dispatch. A mesh's tables sit behind a header in
// the vertex arena; its state words, paint rows and the glyph row table in
// the property arena.
//
// Qualcomm's Vulkan driver on the Adreno X1 loses the device making these
// stages' pipelines (#141), and they were then the only stages to use some
// instructions. Until a run on that driver shows whether those were the
// cause, the stages keep to instructions other stages use: no `degrees`,
// no `&` on `i32`, no `select` between bools, no `vec4` comparison and no
// matrix in a `var`.

// Paint slots: opacity, colour, halo colour, halo width, halo blur, and
// with `["dev.sargunv.orbis:daylight"]` the colour's and halo colour's
// night stops, which share their day stops' curves.
struct SymbolSize {
    // 0 constant, 1 source, 2 camera, 3 composite.
    kind: u32,
    // A constant size; unused otherwise.
    size: f32,
    _pad: vec2<u32>,
    // A camera or composite size's interpolation.
    curve: Curve,
    // A camera size's two stops around the frame's zoom: zoom, size, zoom,
    // size.
    stops: vec4<f32>,
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
    // `*-translate` in pixels; with per-feature translates, the farthest
    // any feature's reaches.
    translate: vec2<f32>,
    image_page: f32,
    // Words per feature of its property rows.
    row_words: u32,
    size: SymbolSize,
    // The text size an icon fitted to variable-anchored text moves with.
    text_size: SymbolSize,
    curves: array<Curve, 5>,
    state: StateLookup,
    // The colours' night stops, and their curves over the daylight (with
    // `DAYLIGHT_FILL` and `DAYLIGHT_HALO`): the first stop's daylight, the
    // last's, 0 for a step or 1 for an exponential curve, and its base.
    night_fill: vec4<f32>,
    night_halo: vec4<f32>,
    daylight_fill: vec4<f32>,
    daylight_halo: vec4<f32>,
    // With `TRANSLATE_ROW_MASK` in `table_mask`, each feature's
    // `*-translate` ends its row (beyond the spec), blended by this curve.
    translate_curve: Curve,
    // To a node uniform's size (orb-render's `NODE_UNIFORM_SIZE`), which
    // `halves` steps by: an `@size` isn't kept by GLSL's std140, and a
    // struct around it would nest past what ANGLE takes.
    _pad: array<vec4<u32>, 2>,
};
// A symbol layer draws its icons and its text in one list, in the order
// they stack: by sort key, a key's icons then its text (MapLibre Native's),
// or all its icons, then all its text. Its icon node's uniform is followed
// by its text node's, `halves` (`symbol_halves.wgsl`, or every node's in
// `symbol_nodes.wgsl`), and each quad reads the half its mesh is of: that
// of `halves` at `half`, the layer's first plus 1 for text (its mesh's
// `H_TEXT`).
var<private> half: u32;

// Flags of SymbolNode (orb-render `NodePaint::uniform_words`).
const IS_TEXT: u32 = 1u;
const LINEAR_ICONS: u32 = 2u;
const PITCH_WITH_MAP: u32 = 4u;
const ROTATE_WITH_MAP: u32 = 8u;
const ROTATE_IN_SHADER: u32 = 16u;
const IS_OFFSET: u32 = 32u;
const KEEP_UPRIGHT: u32 = 64u;
const TRANSLATE_VIEWPORT: u32 = 128u;
// `symbol-height-anchor: ground`: heights count from the terrain.
const HEIGHT_ANCHOR_GROUND: u32 = 256u;
// The colour and the halo colour follow the daylight.
const DAYLIGHT_FILL: u32 = 512u;
const DAYLIGHT_HALO: u32 = 1024u;
// Its labels follow their lines (only the line variant places them so).
const ALONG_LINE: u32 = 2048u;

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
// The mesh's flags: its images need linear filtering, it is its layer's
// text.
const H_FLAGS: u32 = 18u;
const H_LINEAR_IMAGES: u32 = 1u;
const H_TEXT: u32 = 2u;

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
const CHUNK_QUADS: u32 = 64u;

// The placement target: two layers of PLACED_WIDTH × its rows
// texels, a symbol node's quads from its `quad_base` on, a quad's texel in
// each (orb-render's symbol placements):
//
// 0: where it lands in the label plane (x, y), and its rotation scaled by
//    a pitched label's correction on a globe (cos, sin), as f32 bits;
// 1: its size and its opacity as f32 bits, then its colour and halo
//    colour packed as four unorm8s, before `color_in`. A quad with zero
//    opacity draws nothing.
//
// Colours are kept as 8 bits a channel, as GL JS packs its data-driven
// ones, so that a quad takes two texels.
const PLACED_WIDTH: u32 = 2048u;

fn placed_texel_at(slot: u32) -> vec2<u32> {
    return vec2<u32>(slot % PLACED_WIDTH, slot / PLACED_WIDTH);
}

// A quad placed: its two texels, and whether its chunk has it (the texels
// of one that doesn't are never read).
struct PlacedQuad {
    at: vec4<u32>,
    paint: vec4<u32>,
    written: bool,
};

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

// The instance row of placed symbol `index`: its feature's property-row
// slot, then its height offset.
fn symbol_instance_row(base: u32, index: u32) -> u32 {
    let instance = vertex_at(base + vertex_at(base + H_PLACED) + index * PLACED_WORDS + 9u);
    return base + vertex_at(base + H_INSTANCE_ROWS) + instance * 2u;
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
            let t = curve_factor(s.curve, frame.zoom, s.stops.x, s.stops.z);
            return min(mix(s.stops.y, s.stops.w, t), layout_size);
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
    var value = constant;
    if (row_mask(halves[half].table_mask) & (1u << slot)) != 0u {
        value = table_number(row, slot, halves[half].curves[slot], z);
    }
    if from_state(state, halves[half].state, slot) {
        value = state_number(state, halves[half].state, slot, halves[half].curves[slot], value);
    }
    return value;
}

fn paint_color(row: u32, state: u32, slot: u32, constant: vec4<f32>, z: f32) -> vec4<f32> {
    var value = constant;
    // A night stop (slots 5 and 6) shares its day stop's curve.
    let curve = halves[half].curves[select(slot, slot - 4u, slot > 4u)];
    if (row_mask(halves[half].table_mask) & (1u << slot)) != 0u {
        value = table_color(row, slot, curve, z);
    }
    if from_state(state, halves[half].state, slot) {
        value = state_color(state, halves[half].state, slot, curve, value);
    }
    return value;
}

// `["dev.sargunv.orbis:daylight"]` at a label's anchor: ground lighting's
// day side there (`sky.wgsl`'s `fs_ground`), by the sun's elevation over the
// globe's normal at the anchor, mixed toward the target's up as the globe
// hands over to a flat surface, which takes the target's; then mixed toward
// full daylight as the light shades the ground less.
fn label_daylight(anchor: Ground) -> f32 {
    let amount = frame.ground_up.w;
    if amount <= 0.0 {
        return 1.0;
    }
    var normal = frame.ground_up.xyz;
    if on_globe() && frame.globe_plane == 0u {
        let sphere = normalize(anchor.globe + frame.globe_target);
        normal = normalize(mix(normal, sphere, frame.globe_amount));
    }
    let elevation = asin(clamp(dot(normal, frame.ground_sun.xyz), -1.0, 1.0)) * (180.0 / PI);
    let half = frame.ground_sun.w;
    var day = select(0.0, 1.0, elevation >= 0.0);
    if half > 0.0 {
        day = smoothstep(-half, half, elevation);
    }
    return mix(1.0, day, amount);
}

// How far a colour goes from its night stop toward its day stop at
// `daylight`, over its curve `c` (`SymbolNode`'s `daylight_fill`).
fn daylight_t(c: vec4<f32>, daylight: f32) -> f32 {
    if c.z == 0.0 {
        return select(0.0, 1.0, daylight >= c.y);
    }
    var curve: Curve;
    curve.kind = 1u;
    curve.params = vec4<f32>(c.w, 0.0, 0.0, 0.0);
    return curve_factor(curve, clamp(daylight, c.x, c.y), c.x, c.y);
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
    if (halves[half].flags & HEIGHT_ANCHOR_GROUND) != 0u {
        metres += ground_elevation(tile, p);
    }
    return metres;
}

// Map-pitched glyphs keep their size on a globe, whose tile units shrink
// toward the poles: GL JS's pitched text correction, the cosine of the
// target's latitude over the anchor's, mixed in as the globe hands over.
fn pitched_correction(anchor: Ground) -> f32 {
    if !on_globe() || frame.globe_plane != 0u {
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
    if !on_globe() || frame.globe_plane != 0u {
        return false;
    }
    let up = normalize(anchor.globe + frame.globe_target);
    return sphere_hides(anchor.globe + up * (metres * frame.extrusion_scale * frame.globe_metres), false);
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
// Both keep their matrices in `let`s: no stage holds a matrix in a `var`.
fn to_pitched_plane(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    if (halves[half].flags & ROTATE_WITH_MAP) != 0u {
        let m = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
        return m * p / tile_units_per_px(tile);
    }
    let s = skew_vectors();
    let det = determinant(s);
    if det == 0.0 {
        let m = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
        return m * p / tile_units_per_px(tile);
    }
    let m = mat2x2<f32>(vec2<f32>(s[1].y, -s[0].y), vec2<f32>(-s[1].x, s[0].x)) * (1.0 / det);
    return m * p / tile_units_per_px(tile);
}

fn from_pitched_plane(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    if (halves[half].flags & ROTATE_WITH_MAP) != 0u {
        let m = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
        return m * p * tile_units_per_px(tile);
    }
    return skew_vectors() * p * tile_units_per_px(tile);
}

// A line label's glyph in the label plane, unless it falls off its line.
struct LineGlyph {
    p: vec2<f32>,
    angle: f32,
    ok: bool,
};

