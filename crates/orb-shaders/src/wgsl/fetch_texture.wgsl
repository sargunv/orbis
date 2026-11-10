// The `fetch` module over data textures, for devices without storage
// buffers in the vertex stage: each table is an R32Uint texture
// DATA_WIDTH words wide, word i at texel (i % DATA_WIDTH, i / DATA_WIDTH).
// The tables' rows are read by functions the build script reflects from
// `types.wgsl` (`rows_texture.wgsl`).

@group(0) @binding(1) var tiles: texture_2d<u32>;
@group(0) @binding(2) var chunks: texture_2d<u32>;
@group(0) @binding(3) var visible: texture_2d<u32>;
@group(0) @binding(4) var vertices: texture_2d<u32>;
@group(0) @binding(5) var indices: texture_2d<u32>;
@group(0) @binding(6) var props: texture_2d<u32>;
@group(0) @binding(7) var node_cull: texture_2d<u32>;
@group(0) @binding(8) var image_rects: texture_2d<u32>;

// orb-render's DATA_WIDTH_LOG2.
const DATA_WIDTH_LOG2: u32 = 12u;

fn data_word(t: texture_2d<u32>, i: u32) -> u32 {
    return textureLoad(t, vec2<u32>(i & ((1u << DATA_WIDTH_LOG2) - 1u), i >> DATA_WIDTH_LOG2), 0).x;
}

fn tile_word(i: u32) -> u32 { return data_word(tiles, i); }
fn chunk_word(i: u32) -> u32 { return data_word(chunks, i); }
fn visible_word(i: u32) -> u32 { return data_word(visible, i); }
fn node_cull_word(i: u32) -> u32 { return data_word(node_cull, i); }
fn image_word(i: u32) -> u32 { return data_word(image_rects, i); }

fn vertex_at(i: u32) -> u32 { return data_word(vertices, i); }
// Mesh indices are u16s, two to a word, the first in the low half.
fn index_at(i: u32) -> u32 { return (data_word(indices, i >> 1u) >> ((i & 1u) << 4u)) & 0xffffu; }
fn prop_at(i: u32) -> u32 { return data_word(props, i); }
