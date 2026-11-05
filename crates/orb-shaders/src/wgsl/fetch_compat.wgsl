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

// Tile, chunk, visible and node-cull rows are read by functions orb-shaders'
// build script reflects from `types.wgsl`.

fn image_at(i: u32) -> ImageRect {
    let b = i * 8u;
    return ImageRect(vec2<u32>(image_word(b), image_word(b + 1u)), vec2<u32>(image_word(b + 2u), image_word(b + 3u)),
        image_word(b + 4u), bitcast<f32>(image_word(b + 5u)), image_word(b + 6u), image_word(b + 7u));
}
