// The terrain's `fetch` (as `fetch.wgsl` is the draw pipelines'): its
// pipelines read their tables from storage buffers only through these, so
// a device without storage buffers in the vertex stage can read them from
// textures by swapping this module alone.

@group(0) @binding(1) var<storage, read> terrain_tiles: array<TerrainTile>;
// Grid vertices: x and y in cells (8 bits each), then a skirt bit.
@group(0) @binding(2) var<storage, read> grid: array<u32>;
@group(0) @binding(8) var<storage, read> props: array<u32>;

fn terrain_tile_at(i: u32) -> TerrainTile { return terrain_tiles[i]; }
fn grid_at(i: u32) -> u32 { return grid[i]; }
fn prop_at(i: u32) -> u32 { return props[i]; }
