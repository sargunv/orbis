//! Tiles on the globe (plan §9.2's `GpuPerVertex` on a sphere): each tile's
//! quadratic patches in space, kept by tile matrix set and tile and
//! uploaded when a placement first needs them.

use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec3;
use orb_crs::{Series, TileIndex, TileMatrixSet};
use orb_hal::Gpu;
use orb_render::{Scene, TileWarp};

/// A quarter pixel at 512 pixels per tile side.
const TOLERANCE: f64 = 1.0 / 2048.0;
/// Tiles kept before unused ones are dropped.
const CACHED: usize = 2048;

#[derive(Debug, Default)]
pub(crate) struct GlobeWarps {
    tiles: HashMap<(String, TileIndex), GlobeTile>,
    /// Counts frames, to drop what recent ones didn't use.
    frame: u64,
}

#[derive(Debug)]
struct GlobeTile {
    series: Option<Arc<Series<DVec3>>>,
    warp: Option<TileWarp>,
    used: u64,
}

impl GlobeWarps {
    fn entry(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> &mut GlobeTile {
        let frame = self.frame;
        let entry = self
            .tiles
            .entry((tms.id().to_owned(), tile))
            .or_insert_with(|| GlobeTile {
                series: orb_crs::sphere::series(tms, tile, TOLERANCE).map(Arc::new),
                warp: None,
                used: frame,
            });
        entry.used = frame;
        entry
    }

    /// The tile's patches on the sphere, or `None` where its projection
    /// doesn't reach.
    pub(crate) fn series(
        &mut self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Arc<Series<DVec3>>> {
        self.entry(tms, tile).series.clone()
    }

    /// Cells per side to cut the tile's geometry along on the globe.
    pub(crate) fn grid(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> u32 {
        self.series(tms, tile).map_or(1, |s| s.grid())
    }

    /// The tile's patches in the scene and their anchor's ECEF position,
    /// uploading them the first time.
    pub(crate) fn warp(
        &mut self,
        gpu: &mut dyn Gpu,
        scene: &mut Scene,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<(TileWarp, DVec3)> {
        let entry = self.entry(tms, tile);
        let series = entry.series.clone()?;
        if entry.warp.is_none() {
            let patches: Vec<f32> = series
                .patches
                .iter()
                .flat_map(|p| p.points.iter().flat_map(|v| v.as_vec3().to_array()))
                .collect();
            entry.warp = Some(scene.add_warp(gpu, series.cells, &patches).ok()?);
        }
        Some((entry.warp?, series.anchor))
    }

    /// Ends a frame: past the cache's size, tiles this frame didn't place
    /// are dropped and their patches freed.
    pub(crate) fn end_frame(&mut self, scene: &mut Scene) {
        if self.tiles.len() > CACHED {
            let frame = self.frame;
            self.tiles.retain(|_, t| {
                let keep = t.used == frame;
                if !keep && let Some(warp) = t.warp {
                    scene.remove_warp(warp);
                }
                keep
            });
        }
        self.frame += 1;
    }
}
