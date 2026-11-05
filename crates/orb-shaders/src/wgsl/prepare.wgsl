// The DEM prepare pass (`Prepare` node), run when DEM tiles
// arrive rather than every frame. For each tile, `pad` draws its elevations
// into a (dim + 4)² layer with a two-texel border taken from the eight
// neighbouring tiles, or from its own edge where a neighbour isn't resident,
// as GL JS's `DEMData` backfills. `slope` then draws GL JS's
// `hillshade_prepare` slopes into a (dim + 2)² layer. Each is a triangle
// over the layer, one fragment per texel, the job from a uniform.

const NO_TILE: u32 = 0xffffffffu;

struct Job {
    unpack: vec4<f32>,
    // Layers of the tiles at (dx, dy) = (−1, −1), (0, −1), (1, −1), (−1, 0),
    // (1, 0), (−1, 1), (0, 1), (1, 1); `NO_TILE` where not resident.
    neighbours: array<vec4<u32>, 2>,
    layer: u32,
    // The tile's zoom level.
    zoom: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(2) var<uniform> job: Job;

@vertex
fn vs_prepare(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
