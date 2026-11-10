// Helpers shared by every draw pipeline. Group 0 is the scene, group 1 the
// render node; the scene's tables are read through `fetch`.

@group(0) @binding(0) var<uniform> frame: Frame;
// The terrain's DEM page: padded elevations.
@group(0) @binding(14) var dem_pages: texture_2d_array<f32>;
// The terrain's depth (reversed-Z) from its pre-pass, for what it hides;
// 0, the far plane, without terrain.
@group(0) @binding(15) var terrain_depth: texture_2d<f32>;

// How much of a point at `clip` shows past the terrain in front of it (a
// one-texel `terrain_depth` without terrain), 0 to 1: GL JS's
// `calculate_visibility`.
// Partly hidden points average with a point just above them.
fn terrain_visibility(clip: vec4<f32>) -> f32 {
    if textureDimensions(terrain_depth).x <= 1u || clip.w <= 0.0 {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let v = terrain_depth_visibility(ndc);
    if v > 0.95 {
        return 1.0;
    }
    return (v + terrain_depth_visibility(ndc + vec3<f32>(0.0, 0.01, 0.0))) / 2.0;
}

fn terrain_depth_visibility(ndc: vec3<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(terrain_depth));
    let px = vec2<i32>((ndc.xy * vec2<f32>(0.5, -0.5) + 0.5) * vec2<f32>(size));
    if any(px < vec2<i32>(0)) || any(px >= size) {
        return 1.0;
    }
    let surface = textureLoad(terrain_depth, px, 0).r;
    // GL JS fades over `terrain + 0.0001 − point` in GL's NDC z, about
    // 1 − 2·near/distance; reversed depth here is about near/distance, with
    // the same near plane, so half those bounds.
    let behind = surface - (1.0 - ndc.z);
    return 1.0 - clamp((behind - 0.00005) * 1000.0, 0.0, 1.0);
}

fn dem_pixel(p: vec2<i32>, layer: u32) -> f32 {
    let hi = vec2<i32>(textureDimensions(dem_pages)) - 1;
    let texel = textureLoad(dem_pages, clamp(p, vec2<i32>(0), hi), layer, 0);
    // `dem_elevation`'s decode (`dem.wgsl`), terrarium's voids at sea level.
    let void = frame.dem_unpack.w == 32768.0 && all(texel.rgb == vec3<f32>(0.0));
    return select(dot(vec4<f32>(texel.rgb * 255.0, -1.0), frame.dem_unpack), 0.0, void);
}

// The terrain's elevation in metres, with exaggeration, under data point
// `p` of a placement: its DEM, bilinear between pixels as the terrain's
// mesh samples it. Sea level where it has none.
fn ground_elevation(tile: Tile, p: vec2<f32>) -> f32 {
    let layer = tile.dem & 0xffffu;
    if layer == 0xffffu || frame.terrain_exaggeration == 0.0 {
        return 0.0;
    }
    let depth = f32((tile.dem >> 16u) & 255u);
    let offset = vec2<f32>(f32(tile.dem_offset & 0xffffu), f32(tile.dem_offset >> 16u));
    let uv = (p / tile.extent + offset) / exp2(depth);
    let coord = uv * frame.dem_dim + 1.5;
    let f = fract(coord);
    let c = vec2<i32>(floor(coord));
    let tl = dem_pixel(c, layer);
    let tr = dem_pixel(c + vec2<i32>(1, 0), layer);
    let bl = dem_pixel(c + vec2<i32>(0, 1), layer);
    let br = dem_pixel(c + vec2<i32>(1, 1), layer);
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y) * frame.terrain_exaggeration;
}

// A vertex to shade, however the draw found it.
struct Pulled {
    // Row of the tile placement in `tiles`.
    tile: u32,
    // Word offset of the vertex in `vertices`.
    vertex: u32,
    // First word of the mesh's property rows in `props`.
    prop_base: u32,
    // First word of the mesh's feature-state remap, or NO_STATE.
    state_base: u32,
    // Words per feature of the mesh's property rows.
    row_words: u32,
};

// Whether this pass draws placement `tile`: a drape placement only in the
// drape pass of its zoom, whose frame has that zoom.
fn drawn_in_pass(tile: u32) -> bool {
    let zoom = tile_at(tile).drape_zoom;
    return zoom == 0.0 || zoom == frame.zoom;
}

// Maps a vertex of a chunk instance to its vertex, or `None` (false) for
// padding past the chunk's last index.
fn pull(node_index: u32, instance: u32, index: u32, stride: u32, out: ptr<function, Pulled>) -> bool {
    let entry = visible_at(node_cull_at(node_index).visible_base + instance);
    let chunk = chunk_at(entry.chunk);
    if index >= chunk.index_count || !drawn_in_pass(entry.tile) {
        return false;
    }
    (*out).tile = entry.tile;
    (*out).vertex = chunk.vertex_base + (chunk.first_vertex + index_at(chunk.first_index + index)) * stride;
    (*out).prop_base = chunk.prop_base;
    (*out).state_base = chunk.state_base;
    (*out).row_words = chunk.row_words;
    return true;
}

// Maps an indexed node's compacted index (`run << run_shift | vertex`) to
// its vertex, or `None` (false) for a hidden index.
fn pull_indexed(node_index: u32, index: u32, stride: u32, out: ptr<function, Pulled>) -> bool {
    if index == HIDDEN_INDEX {
        return false;
    }
    let cull = node_cull_at(node_index);
    let run = visible_at(cull.visible_base + (index >> cull.run_shift));
    let chunk = chunk_at(run.chunk);
    if !drawn_in_pass(run.tile) {
        return false;
    }
    (*out).tile = run.tile;
    (*out).vertex = chunk.vertex_base + (index & ((1u << cull.run_shift) - 1u)) * stride;
    (*out).prop_base = chunk.prop_base;
    (*out).state_base = chunk.state_base;
    (*out).row_words = chunk.row_words;
    return true;
}

// Whether a fragment the placement of tile `tile` owns at `owner` and
// lands at `p` is its to draw, by its `fragment_clip`.
fn clip_drawn(clip: vec2<u32>, tile: u32, owner: vec2<f32>, p: vec2<f32>) -> bool {
    if all(clip == CLIP_FETCHED) {
        let t = tile_at(tile);
        return owned(t, owner) && landed(t, p);
    }
    return all(p >= unpack_i16x2(clip.x)) && all(p < unpack_i16x2(clip.y));
}

fn prop_row(v: Pulled, slot: u32, stride: u32) -> u32 {
    return v.prop_base + slot * stride;
}

// A node's feature-state lookups (StateLookup): bit i of
// `mask` reads property slot i from the feature's state row when it has
// one. Rows live in the property arena: the node's block of state row s
// starts at `base + s * stride` with a flags word (STATE_HAS: the feature
// has state; STATE_FADING: what it shows changed at the frame time in the
// next word; zero for neither), then per masked slot STATE_SLOT_WORDS
// words: STATE_ZOOMS values at the integer zooms from `zoom` on, which a
// drape slot reads at its own zoom and the camera between two; as many of
// what the feature showed before the change; and the slot's transition,
// its duration and delay in milliseconds.
struct StateLookup {
    mask: u32,
    base: u32,
    stride: u32,
    zoom: f32,
};

const STATE_HAS: u32 = 1u;
const STATE_FADING: u32 = 2u;
const STATE_ZOOMS: u32 = 8u;
const STATE_SLOT_WORDS: u32 = 2u * STATE_ZOOMS + 2u;

// A feature's state block for a node, or NO_STATE when the feature has
// the default state and nothing fades.
fn state_block(v: Pulled, feature: u32, lookup: StateLookup) -> u32 {
    if lookup.stride == 0u || v.state_base == NO_STATE {
        return NO_STATE;
    }
    let index = prop_at(v.state_base + feature);
    if index == 0u {
        return NO_STATE;
    }
    let block = lookup.base + index * lookup.stride;
    if prop_at(block) == 0u {
        return NO_STATE;
    }
    return block;
}

// Whether property `slot` comes from the state block.
fn from_state(block: u32, lookup: StateLookup, slot: u32) -> bool {
    return block != NO_STATE && (lookup.mask & (1u << slot)) != 0u;
}

// The first word of property `slot`'s values in the block.
fn state_slot(block: u32, lookup: StateLookup, slot: u32) -> u32 {
    return block + 2u + countOneBits(lookup.mask & ((1u << slot) - 1u)) * STATE_SLOT_WORDS;
}

// Where the values from word `at` on are at the frame's zoom: the word of
// the integer zoom at or below it (the first or last past the row's
// zooms), and how far toward the next to go, 0 past the last.
fn state_at(at: u32, lookup: StateLookup, c: Curve) -> vec2<u32> {
    let k = u32(clamp(floor(frame.zoom) - lookup.zoom, 0.0, f32(STATE_ZOOMS - 1u)));
    let t = select(curve_t(c, lookup.zoom + f32(k)), 0.0, k + 1u == STATE_ZOOMS);
    return vec2<u32>(at + k, bitcast<u32>(t));
}

fn state_number_at(at: u32, lookup: StateLookup, c: Curve) -> f32 {
    let w = state_at(at, lookup, c);
    let t = bitcast<f32>(w.y);
    let a = bitcast<f32>(prop_at(w.x));
    if t == 0.0 {
        return a;
    }
    return mix(a, bitcast<f32>(prop_at(w.x + 1u)), t);
}

fn state_color_at(at: u32, lookup: StateLookup, c: Curve) -> vec4<f32> {
    let w = state_at(at, lookup, c);
    let t = bitcast<f32>(w.y);
    let a = unpack4x8unorm(prop_at(w.x));
    if t == 0.0 {
        return a;
    }
    return mix(a, unpack4x8unorm(prop_at(w.x + 1u)), t);
}

// GL JS's `easeCubicInOut`.
fn ease_cubic_in_out(t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    let t2 = t * t;
    let t3 = t2 * t;
    return 4.0 * select(3.0 * (t - t2) + t3 - 0.75, t3, t < 0.5);
}

// How far the change of a block whose slot's values start at `at` has
// eased at the frame's time: 1 once done, or with nothing fading.
fn state_fade(block: u32, at: u32) -> f32 {
    if (prop_at(block) & STATE_FADING) == 0u {
        return 1.0;
    }
    let duration = bitcast<f32>(prop_at(at + 2u * STATE_ZOOMS));
    let delay = bitcast<f32>(prop_at(at + 2u * STATE_ZOOMS + 1u));
    let since = frame.time - bitcast<f32>(prop_at(block + 1u)) - delay;
    return ease_cubic_in_out(since / max(duration, 1e-3));
}

// Property `slot` of a feature with a state block: its state's value, or
// `base`, its value without state, once its state is removed; eased from
// what it showed before while its state changes.
fn state_number(block: u32, lookup: StateLookup, slot: u32, c: Curve, base: f32) -> f32 {
    let at = state_slot(block, lookup, slot);
    var value = base;
    if (prop_at(block) & STATE_HAS) != 0u {
        value = state_number_at(at, lookup, c);
    }
    let t = state_fade(block, at);
    if t >= 1.0 {
        return value;
    }
    return mix(state_number_at(at + STATE_ZOOMS, lookup, c), value, t);
}

fn state_color(block: u32, lookup: StateLookup, slot: u32, c: Curve, base: vec4<f32>) -> vec4<f32> {
    let at = state_slot(block, lookup, slot);
    var value = base;
    if (prop_at(block) & STATE_HAS) != 0u {
        value = state_color_at(at, lookup, c);
    }
    let t = state_fade(block, at);
    if t >= 1.0 {
        return value;
    }
    return mix(state_color_at(at + STATE_ZOOMS, lookup, c), value, t);
}

// How a feature-dependent property interpolates between its values at the
// tile's zoom and the next zoom, after GL JS's composite binders.
struct Curve {
    // Exponential: base in x. Cubic bezier: control points.
    params: vec4<f32>,
    // 0: step (no interpolation), 1: exponential, 2: cubic bezier.
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// The interpolation factor between the values at `tile_zoom` and
// `tile_zoom + 1`, clamped to [0, 1].
fn curve_t(c: Curve, tile_zoom: f32) -> f32 {
    let x = frame.zoom - tile_zoom;
    switch c.kind {
        case 1u: {
            let base = c.params.x;
            if abs(base - 1.0) < 1e-6 {
                return clamp(x, 0.0, 1.0);
            }
            return clamp((pow(base, x) - 1.0) / (base - 1.0), 0.0, 1.0);
        }
        case 2u: {
            return clamp(unit_bezier(c.params, clamp(x, 0.0, 1.0)), 0.0, 1.0);
        }
        default: {
            return 0.0;
        }
    }
}

// A number property from a property row: two f32 words, its values at the
// tile's zoom and the next.
fn table_number(row: u32, slot: u32, c: Curve, tile_zoom: f32) -> f32 {
    let a = bitcast<f32>(prop_at(row + 2u * slot));
    let b = bitcast<f32>(prop_at(row + 2u * slot + 1u));
    return mix(a, b, curve_t(c, tile_zoom));
}

// A premultiplied style or image colour as it mixes in the frame's colour
// space: itself, or in linear light. Families call it once a value is
// chosen and interpolated, so a colour interpolates as the spec says,
// then mixes in linear light.
fn color_in(c: vec4<f32>) -> vec4<f32> {
    if frame.linear == 0u {
        return c;
    }
    return premultiplied_to_linear(c);
}

// A colour property from a property row: two rgba8 words.
fn table_color(row: u32, slot: u32, c: Curve, tile_zoom: f32) -> vec4<f32> {
    let a = unpack4x8unorm(prop_at(row + 2u * slot));
    let b = unpack4x8unorm(prop_at(row + 2u * slot + 1u));
    return mix(a, b, curve_t(c, tile_zoom));
}

// A node's `*-translate` in pixels: the node's, or with
// `TRANSLATE_ROW_MASK` the feature's, from the end of its row.
fn node_translate(px: vec2<f32>, table_mask: u32, row: u32, row_words: u32, c: Curve, tile_zoom: f32) -> vec2<f32> {
    if (table_mask & TRANSLATE_ROW_MASK) == 0u {
        return px;
    }
    let at = row + row_words - TRANSLATE_ROW_WORDS;
    let t = mix(unpack_i16x2(prop_at(at)), unpack_i16x2(prop_at(at + 1u)), curve_t(c, tile_zoom));
    return t / f32(TRANSLATE_ROW_UNITS);
}

// A `*-translate` in pixels as a camera-relative world offset, after GL JS's
// `translatePosition`: map-anchored, or turned with the map's bearing for
// the `viewport` anchor.
fn translate_world(px: vec2<f32>, viewport_anchor: u32, tile: Tile) -> vec2<f32> {
    var t = px;
    if viewport_anchor != 0u {
        let s = sin(frame.bearing);
        let c = cos(frame.bearing);
        t = vec2<f32>(t.x * c - t.y * s, t.x * s + t.y * c);
    }
    return vec2<f32>(t.x, -t.y) * tile_world_per_px(tile);
}
