//! The meshes the map makes itself: the background's quads and image
//! tiles' grids, bent for a globe or a warp.

use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_render::{GpuTile, MeshData, Renderer, TileKey};
use orb_source::endpoint::TileEndpoint;

use crate::image;

/// Cells per side of an image source's grid: GL JS's granularity for warped
/// image quads.
pub(crate) const IMAGE_SOURCE_CELLS: u32 = 16;

/// A series' patches as the scene takes them, `node` giving a node's
/// words.
pub(crate) fn series_words<V: Copy, const N: usize>(
    series: &orb_crs::Series<V>,
    node: impl Fn(V) -> [f32; N],
) -> (u32, Vec<f32>) {
    let [lo, hi] = series.window.map(|c| c.as_vec2());
    let nodes = (series.patches.iter()).flat_map(|p| p.points.iter().flat_map(|&v| node(v)));
    orb_render::warp_words(series.cells, [lo.x, lo.y, hi.x, hi.y], nodes)
}

/// A series' planar patches as the scene takes them.
pub(crate) fn series_patches(series: &orb_crs::Series) -> (u32, Vec<f32>) {
    series_words(series, |v| v.as_vec2().to_array())
}

/// An image tile's mesh: on a globe, a grid in the tile's own space that
/// the sphere and the plane's patches bend; on another plane, the warp's
/// mesh on the tile's `grid` (made here if its job didn't), its sides
/// refined by `edges`; else `None` for its quad. `Err` when nothing of it
/// reaches the surface.
pub(crate) fn image_mesh(
    globe_warps: &mut crate::globe::GlobeWarps,
    globe: bool,
    warped: bool,
    endpoint: &TileEndpoint,
    tile: TileIndex,
    (grid, edges): (&mut Option<Arc<orb_crs::WarpGrid>>, [u8; 4]),
) -> Result<Option<Arc<image::Warped>>, ()> {
    let tms = &endpoint.tms;
    let bounds = tms.tile_bounds(tile);
    let mesh = match (&endpoint.quad, warped, globe) {
        // An image source: its quad's warp at the vertices of a grid over
        // the image, in its one tile; on a globe, as fine as the sphere
        // bends.
        (Some(quad), _, false) => image::Warped::quad(bounds, quad, IMAGE_SOURCE_CELLS),
        (Some(quad), _, true) => {
            let [x, y] = globe_warps.grid(tms, tile);
            let cells = x.max(y).max(IMAGE_SOURCE_CELLS);
            image::Warped::quad(bounds, quad, cells)
        }
        (None, true, false) => {
            let warp = globe_warps.surface_warp(tms);
            let grid = match grid {
                Some(grid) => grid,
                None => grid.insert(Arc::new(image::warp_grid(&warp, tile).ok_or(())?)),
            };
            image::Warped::new(&warp, tile, grid, edges).ok_or(())?
        }
        (None, false, false) => return Ok(None),
        // On a globe, a grid fine enough for the sphere to bend.
        (None, _, true) => {
            // A square grid, as fine as the finer side needs.
            let [x, y] = globe_warps.grid(tms, tile);
            let (cells, poles) = (x.max(y), orb_crs::sphere::poles(tms, tile));
            if cells <= 1 && poles == [false; 2] {
                return Ok(None);
            }
            image::Warped::grid(bounds, cells, poles)
        }
    };
    Ok(Some(Arc::new(mesh)))
}

/// Cells per side of the background's grid on a globe: fine enough for a
/// whole-world tile.
pub(crate) const BACKGROUND_GLOBE_CELLS: u32 = 64;

/// One scene tile with a tile-sized quad for every background layer, or on
/// a globe a grid the sphere bends.
pub(crate) fn make_background(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &crate::style::StylePlan,
    globe: bool,
) -> Option<GpuTile> {
    let nodes: Vec<u32> = (plan.nodes.iter().enumerate())
        .filter(|(_, n)| n.source == usize::MAX)
        .map(|(i, _)| i as u32)
        .collect();
    if nodes.is_empty() {
        return None;
    }
    let cells = if globe { BACKGROUND_GLOBE_CELLS } else { 1 };
    let n = cells + 1;
    let step = 8192 / cells;
    let mut vertices: Vec<u32> = (0..n * n)
        .flat_map(|k| [(k % n * step) | (k / n * step) << 16, 0])
        .collect();
    let mut triangles: Vec<[u32; 3]> = (0..cells * cells)
        .flat_map(|c| {
            let k = c / cells * n + c % cells;
            [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
        })
        .collect();
    if globe {
        // Caps: the top and bottom edges fanned out to the poles, which
        // only the world's edge rows reach (`project.wgsl`).
        for (row, pole) in [
            (0, orb_layout::fill::NORTH_POLE_Y),
            (cells, orb_layout::fill::SOUTH_POLE_Y),
        ] {
            let tip = (vertices.len() / 2) as u32;
            vertices.extend([u32::from(pole as u16) << 16, 0]);
            triangles.extend((0..cells).map(|i| [row * n + i, row * n + i + 1, tip]));
        }
    }
    let chunks = triangles
        .chunks(256)
        .enumerate()
        .map(|(c, run)| {
            let rows = (c * 256 / 2) as u32 / cells;
            let last = ((c * 256 + run.len() - 1) / 2) as u32 / cells;
            orb_layout::Chunk {
                sort_key: None,
                first_index: (c * 256 * 3) as u32,
                index_count: (run.len() * 3) as u32,
                // The caps' triangles come last and span every row.
                bounds: if c * 256 >= (cells * cells * 2) as usize {
                    orb_layout::TileBox {
                        min: [0, i16::MIN],
                        max: [8192, i16::MAX],
                    }
                } else {
                    orb_layout::TileBox {
                        min: [0, (rows * step) as i16],
                        max: [8192, ((last + 1) * step).min(8192) as i16],
                    }
                },
            }
        })
        .collect();
    let mesh = orb_layout::Mesh {
        vertices,
        indices: triangles.into_iter().flatten().collect(),
        chunks,
    };
    // One quad (or grid) for every covered tile: no source's tile.
    let key = TileKey {
        source: u32::MAX,
        tile: TileIndex::new(0, 0, 0),
    };
    let tile = renderer
        .add_tile(gpu, key, None)
        .expect("a tile without patches fits");
    let data = MeshData::new(&mesh, &[], 0);
    for node in nodes {
        renderer
            .add_mesh(gpu, tile, (node, 8192), &data, &Default::default())
            .expect("background mesh fits");
    }
    Some(tile)
}
