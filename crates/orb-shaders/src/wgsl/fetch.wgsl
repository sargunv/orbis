// The `fetch` module, Modern: draw pipelines read the scene's tables from
// storage buffers (plan §8.4). `fetch_compat.wgsl` reads the same words
// from data textures.

@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var<storage, read> chunks: array<Chunk>;
@group(0) @binding(3) var<storage, read> visible: array<Visible>;
@group(0) @binding(4) var<storage, read> vertices: array<u32>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;
@group(0) @binding(6) var<storage, read> props: array<u32>;
@group(0) @binding(7) var<storage, read> node_cull: array<NodeCull>;
@group(0) @binding(8) var<storage, read> image_rects: array<ImageRect>;

// The compacted index that draws nothing.
const HIDDEN_DRAW_INDEX: u32 = HIDDEN_INDEX;

fn tile_at(i: u32) -> Tile { return tiles[i]; }
fn chunk_at(i: u32) -> Chunk { return chunks[i]; }
fn visible_at(i: u32) -> Visible { return visible[i]; }
fn vertex_at(i: u32) -> u32 { return vertices[i]; }
fn index_at(i: u32) -> u32 { return indices[i]; }
fn prop_at(i: u32) -> u32 { return props[i]; }
fn node_cull_at(i: u32) -> NodeCull { return node_cull[i]; }
fn image_at(i: u32) -> ImageRect { return image_rects[i]; }
