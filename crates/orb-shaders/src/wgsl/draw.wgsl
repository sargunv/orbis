// Helpers shared by every draw pipeline. Group 0 is the scene, group 1 the
// render node; the scene's tables are read through `fetch`.

@group(0) @binding(0) var<uniform> frame: Frame;
// The terrain's DEM page: padded elevations (decision 0019).
@group(0) @binding(14) var dem_pages: texture_2d_array<f32>;

fn dem_pixel(p: vec2<i32>, layer: u32) -> f32 {
    let hi = vec2<i32>(textureDimensions(dem_pages)) - 1;
    let texel = textureLoad(dem_pages, clamp(p, vec2<i32>(0), hi), layer, 0);
    return dot(vec4<f32>(texel.rgb * 255.0, -1.0), frame.dem_unpack);
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

// Maps a vertex of a chunk instance to its vertex, or `None` (false) for
// padding past the chunk's last index.
fn pull(node_index: u32, instance: u32, index: u32, stride: u32, out: ptr<function, Pulled>) -> bool {
    let entry = visible_at(node_cull_at(node_index).visible_base + instance);
    let chunk = chunk_at(entry.chunk);
    if index >= chunk.index_count {
        return false;
    }
    (*out).tile = entry.tile;
    (*out).vertex = chunk.vertex_base + index_at(chunk.first_index + index) * stride;
    (*out).prop_base = chunk.prop_base;
    (*out).state_base = chunk.state_base;
    (*out).row_words = chunk.row_words;
    return true;
}

// Maps an indexed node's compacted index (`run << run_shift | vertex`) to
// its vertex, or `None` (false) for a hidden index.
fn pull_indexed(node_index: u32, index: u32, stride: u32, out: ptr<function, Pulled>) -> bool {
    if index == HIDDEN_DRAW_INDEX {
        return false;
    }
    let cull = node_cull_at(node_index);
    let run = visible_at(cull.visible_base + (index >> cull.run_shift));
    let chunk = chunk_at(run.chunk);
    (*out).tile = run.tile;
    (*out).vertex = chunk.vertex_base + (index & ((1u << cull.run_shift) - 1u)) * stride;
    (*out).prop_base = chunk.prop_base;
    (*out).state_base = chunk.state_base;
    (*out).row_words = chunk.row_words;
    return true;
}

fn prop_row(v: Pulled, slot: u32, stride: u32) -> u32 {
    return v.prop_base + slot * stride;
}

// A node's feature-state lookups (plan §10.2, §11.3 StateLookup): bit i of
// `mask` reads property slot i from the feature's state row when it has
// one. Rows live in the property arena: the node's block of state row s
// starts at `base + s * stride` with a word that is non-zero once the
// feature has state, then two words (values at `zoom` and the next) per
// masked slot.
struct StateLookup {
    mask: u32,
    base: u32,
    stride: u32,
    zoom: f32,
};

// The values of a feature's state block for a node, or NO_STATE when the
// feature has the default state.
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
    return block + 1u;
}

// Whether property `slot` comes from the state block.
fn from_state(block: u32, lookup: StateLookup, slot: u32) -> bool {
    return block != NO_STATE && (lookup.mask & (1u << slot)) != 0u;
}

// Property `slot`'s position among the block's values.
fn state_slot(lookup: StateLookup, slot: u32) -> u32 {
    return count_bits(lookup.mask & ((1u << slot) - 1u));
}

fn state_number(block: u32, lookup: StateLookup, slot: u32, c: Curve) -> f32 {
    return table_number(block, state_slot(lookup, slot), c, lookup.zoom);
}

fn state_color(block: u32, lookup: StateLookup, slot: u32, c: Curve) -> vec4<f32> {
    return table_color(block, state_slot(lookup, slot), c, lookup.zoom);
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

// A colour property from a property row: two rgba8 words.
fn table_color(row: u32, slot: u32, c: Curve, tile_zoom: f32) -> vec4<f32> {
    let a = unpack4x8unorm(prop_at(row + 2u * slot));
    let b = unpack4x8unorm(prop_at(row + 2u * slot + 1u));
    return mix(a, b, curve_t(c, tile_zoom));
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
    return vec2<f32>(t.x, -t.y) * tile.world_per_px;
}
