// The DEM prepare pass (plan §8.5's `Prepare` node), run when DEM tiles
// arrive rather than every frame. For each tile, `pad` copies its elevations
// into a (dim + 4)² layer with a two-texel border taken from the eight
// neighbouring tiles, or from its own edge where a neighbour isn't resident,
// as GL JS's `DEMData` backfills. `slope` then computes GL JS's
// `hillshade_prepare` slopes over a (dim + 2)² layer. The steps' texels
// are computed by `pad_texel` and `slope_texel`, for a compute dispatch
// (`*_cs.wgsl`, Modern) or a draw into each layer (`prepare_fs.wgsl`).

const NO_TILE: u32 = 0xffffffffu;

struct DemJob {
    unpack: vec4<f32>,
    layer: u32,
    // Layers of the tiles at (dx, dy) = (−1, −1), (0, −1), (1, −1), (−1, 0),
    // (1, 0), (−1, 1), (0, 1), (1, 1); `NO_TILE` where not resident.
    neighbours: array<u32, 8>,
    // The tile's zoom level.
    zoom: f32,
    _pad0: u32,
    _pad1: u32,
};
