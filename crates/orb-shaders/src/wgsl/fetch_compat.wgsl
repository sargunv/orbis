// The `fetch` module, Compat: the scene's tables as data textures (plan
// §8.7). orb-hal-gl mirrors each storage buffer a pipeline binds into an
// R32UI texture DATA_WIDTH words wide, word i at texel
// (i % DATA_WIDTH, i / DATA_WIDTH); rows are reassembled here.

@group(0) @binding(1) var tiles: texture_2d<u32>;
@group(0) @binding(2) var chunks: texture_2d<u32>;
@group(0) @binding(3) var visible: texture_2d<u32>;
@group(0) @binding(4) var vertices: texture_2d<u32>;
@group(0) @binding(5) var indices: texture_2d<u32>;
@group(0) @binding(6) var props: texture_2d<u32>;
@group(0) @binding(7) var node_cull: texture_2d<u32>;
@group(0) @binding(8) var image_rects: texture_2d<u32>;

// The compacted index that draws nothing: WebGL rejects indices from 2^31,
// so Compat's runs and hidden indices fit in 31 bits (orb-scene).
const HIDDEN_DRAW_INDEX: u32 = 0x7ffffffeu;

// orb-hal-gl's DATA_WIDTH_LOG2.
const DATA_WIDTH_LOG2: u32 = 12u;

fn data_texel(i: u32) -> vec2<u32> {
    return vec2<u32>(i & ((1u << DATA_WIDTH_LOG2) - 1u), i >> DATA_WIDTH_LOG2);
}

fn tile_word(i: u32) -> u32 { return textureLoad(tiles, data_texel(i), 0).x; }
fn chunk_word(i: u32) -> u32 { return textureLoad(chunks, data_texel(i), 0).x; }
fn visible_word(i: u32) -> u32 { return textureLoad(visible, data_texel(i), 0).x; }
fn node_cull_word(i: u32) -> u32 { return textureLoad(node_cull, data_texel(i), 0).x; }
fn image_word(i: u32) -> u32 { return textureLoad(image_rects, data_texel(i), 0).x; }

fn vertex_at(i: u32) -> u32 { return textureLoad(vertices, data_texel(i), 0).x; }
fn index_at(i: u32) -> u32 { return textureLoad(indices, data_texel(i), 0).x; }
fn prop_at(i: u32) -> u32 { return textureLoad(props, data_texel(i), 0).x; }

// Rows as orb-scene writes them (`types.wgsl` documents the fields).
fn tile_at(i: u32) -> Tile {
    let b = i * TILE_WORDS;
    var t: Tile;
    t.origin = vec2<i32>(bitcast<i32>(tile_word(b)), bitcast<i32>(tile_word(b + 1u)));
    t.scale = bitcast<f32>(tile_word(b + 2u));
    t.zoom = bitcast<f32>(tile_word(b + 3u));
    t.clips[0] = vec2<u32>(tile_word(b + 4u), tile_word(b + 5u));
    t.clips[1] = vec2<u32>(tile_word(b + 6u), tile_word(b + 7u));
    t.clips[2] = vec2<u32>(tile_word(b + 8u), tile_word(b + 9u));
    t.clips[3] = vec2<u32>(tile_word(b + 10u), tile_word(b + 11u));
    t.pattern_origin = vec2<u32>(tile_word(b + 12u), tile_word(b + 13u));
    t.extent = bitcast<f32>(tile_word(b + 14u));
    t.wrap = bitcast<i32>(tile_word(b + 15u));
    t.px_scale = bitcast<f32>(tile_word(b + 16u));
    t.dem = tile_word(b + 17u);
    t.dem_offset = tile_word(b + 18u);
    t.terrain_range = tile_word(b + 19u);
    t.warp_base = tile_word(b + 20u);
    t.warp_cells = tile_word(b + 21u);
    t.globe_origin[0] = bitcast<i32>(tile_word(b + 22u));
    t.globe_origin[1] = bitcast<i32>(tile_word(b + 23u));
    t.globe_origin[2] = bitcast<i32>(tile_word(b + 24u));
    t.globe_base = tile_word(b + 25u);
    t.globe_cells = tile_word(b + 26u);
    t.globe_flags = tile_word(b + 27u);
    t.origin_rest = vec2<f32>(bitcast<f32>(tile_word(b + 28u)), bitcast<f32>(tile_word(b + 29u)));
    t.globe_origin_rest[0] = bitcast<f32>(tile_word(b + 30u));
    t.globe_origin_rest[1] = bitcast<f32>(tile_word(b + 31u));
    t.globe_origin_rest[2] = bitcast<f32>(tile_word(b + 32u));
    return t;
}

fn chunk_at(i: u32) -> Chunk {
    let b = i * 8u;
    return Chunk(chunk_word(b), chunk_word(b + 1u), chunk_word(b + 2u), chunk_word(b + 3u),
        chunk_word(b + 4u), chunk_word(b + 5u), chunk_word(b + 6u), chunk_word(b + 7u));
}

fn visible_at(i: u32) -> Visible {
    return Visible(visible_word(2u * i), visible_word(2u * i + 1u));
}

fn node_cull_at(i: u32) -> NodeCull {
    let b = i * 8u;
    return NodeCull(node_cull_word(b), bitcast<f32>(node_cull_word(b + 1u)), node_cull_word(b + 2u),
        node_cull_word(b + 3u), bitcast<f32>(node_cull_word(b + 4u)), node_cull_word(b + 5u),
        node_cull_word(b + 6u), node_cull_word(b + 7u));
}

fn image_at(i: u32) -> ImageRect {
    let b = i * 8u;
    return ImageRect(vec2<u32>(image_word(b), image_word(b + 1u)), vec2<u32>(image_word(b + 2u), image_word(b + 3u)),
        image_word(b + 4u), bitcast<f32>(image_word(b + 5u)), image_word(b + 6u), image_word(b + 7u));
}
