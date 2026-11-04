// The DEM prepare pass (plan §8.5's `Prepare` node), run when DEM tiles
// arrive rather than every frame. For each tile, `pad` copies its elevations
// into a (dim + 4)² layer with a two-texel border taken from the eight
// neighbouring tiles, or from its own edge where a neighbour isn't resident,
// as GL JS's `DEMData` backfills. `slope` then computes GL JS's
// `hillshade_prepare` slopes over a (dim + 2)² layer. One invocation per
// texel; `z` picks the job.

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

@group(0) @binding(2) var<storage, read> jobs: array<DemJob>;
