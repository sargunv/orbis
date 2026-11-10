// The `fetch` module over data textures (`data_texture.wgsl`), at the
// storage buffers' bindings: the row tables packed into one texture, each
// from the base `row_bases` gives (orb-render's `Tables`), and the arenas
// and image rects each in a texture of their own.

struct RowBases {
    tiles: u32,
    chunks: u32,
    visible: u32,
    node_cull: u32,
};

@group(0) @binding(1) var rows: texture_2d<u32>;
@group(0) @binding(2) var<uniform> row_bases: RowBases;
@group(0) @binding(4) var vertices: texture_2d<u32>;
@group(0) @binding(5) var indices: texture_2d<u32>;
@group(0) @binding(6) var props: texture_2d<u32>;
@group(0) @binding(8) var image_rects: texture_2d<u32>;

fn tile_word(i: u32) -> u32 { return data_word(rows, row_bases.tiles + i); }
fn chunk_word(i: u32) -> u32 { return data_word(rows, row_bases.chunks + i); }
fn visible_word(i: u32) -> u32 { return data_word(rows, row_bases.visible + i); }
fn node_cull_word(i: u32) -> u32 { return data_word(rows, row_bases.node_cull + i); }
fn image_word(i: u32) -> u32 { return data_word(image_rects, i); }

fn vertex_at(i: u32) -> u32 { return data_word(vertices, i); }
// Mesh indices are u16s, two to a word, the first in the low half.
fn index_at(i: u32) -> u32 { return (data_word(indices, i >> 1u) >> ((i & 1u) << 4u)) & 0xffffu; }
fn prop_at(i: u32) -> u32 { return data_word(props, i); }
