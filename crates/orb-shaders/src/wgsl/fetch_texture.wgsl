// The `fetch` module over data textures (`data_texture.wgsl`), at the
// storage buffers' bindings.

@group(0) @binding(1) var tiles: texture_2d<u32>;
@group(0) @binding(2) var chunks: texture_2d<u32>;
@group(0) @binding(3) var visible: texture_2d<u32>;
@group(0) @binding(4) var vertices: texture_2d<u32>;
@group(0) @binding(5) var indices: texture_2d<u32>;
@group(0) @binding(6) var props: texture_2d<u32>;
@group(0) @binding(7) var node_cull: texture_2d<u32>;
@group(0) @binding(8) var image_rects: texture_2d<u32>;

fn tile_word(i: u32) -> u32 { return data_word(tiles, i); }
fn chunk_word(i: u32) -> u32 { return data_word(chunks, i); }
fn visible_word(i: u32) -> u32 { return data_word(visible, i); }
fn node_cull_word(i: u32) -> u32 { return data_word(node_cull, i); }
fn image_word(i: u32) -> u32 { return data_word(image_rects, i); }

fn vertex_at(i: u32) -> u32 { return data_word(vertices, i); }
// Mesh indices are u16s, two to a word, the first in the low half.
fn index_at(i: u32) -> u32 { return (data_word(indices, i >> 1u) >> ((i & 1u) << 4u)) & 0xffffu; }
fn prop_at(i: u32) -> u32 { return data_word(props, i); }
