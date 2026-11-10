// `fetch_terrain.wgsl` over data textures (`data_texture.wgsl`).

@group(0) @binding(1) var terrain_tiles: texture_2d<u32>;
@group(0) @binding(2) var grid: texture_2d<u32>;
@group(0) @binding(8) var props: texture_2d<u32>;

fn terrain_tile_word(i: u32) -> u32 { return data_word(terrain_tiles, i); }
fn grid_at(i: u32) -> u32 { return data_word(grid, i); }
fn prop_at(i: u32) -> u32 { return data_word(props, i); }
