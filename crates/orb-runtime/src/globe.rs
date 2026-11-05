//! Tiles on the low-zoom surface (plan §9.2's `GpuPerVertex` on a sphere
//! or on a low-zoom plane): each tile's quadratic patches there, kept by
//! tile matrix set and tile and uploaded when a placement first needs them.

use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec3;
use orb_crs::{Footprint, Projection, Series, TileIndex, TileMatrixSet, Warp};
use orb_hal::Gpu;
use orb_render::{Scene, TileWarp};

/// A sixteenth of a pixel at 512 pixels per tile side on the sphere, and a
/// quarter of one on a plane, as the plane's own warps hold.
const TOLERANCE: f64 = 1.0 / 8192.0;
const PLANE_TOLERANCE: f64 = 1.0 / 2048.0;
/// Tiles kept before unused ones are dropped.
const CACHED: usize = 2048;

#[derive(Debug, Default)]
pub(crate) struct GlobeWarps {
    /// The low-zoom plane, or `None` for the sphere.
    plane: Option<Projection>,
    /// Each set's warp onto the low-zoom plane.
    warps: HashMap<String, Warp>,
    tiles: HashMap<(String, TileIndex), GlobeTile>,
    /// Counts frames, to drop what recent ones didn't use.
    frame: u64,
}

#[derive(Debug)]
struct GlobeTile {
    series: Option<Arc<Series<DVec3>>>,
    /// Where the tile lands on a low-zoom plane, for the cover.
    footprint: Option<Footprint>,
    warp: Option<TileWarp>,
    used: u64,
}

impl GlobeWarps {
    /// Puts the tiles on a low-zoom plane, or on the sphere (`None`),
    /// freeing the patches of another.
    pub(crate) fn set_plane(&mut self, scene: &mut Scene, plane: Option<Projection>) {
        if self.plane != plane {
            self.frame += 1;
            self.retain(scene, |_| false);
            self.warps.clear();
            self.plane = plane;
        }
    }

    fn entry(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> &mut GlobeTile {
        let frame = self.frame;
        let key = (tms.id().to_owned(), tile);
        if !self.tiles.contains_key(&key) {
            let (series, footprint) = match self.plane {
                None => (orb_crs::sphere::series(tms, tile, TOLERANCE), None),
                Some(plane) => {
                    let warp = self
                        .warps
                        .entry(tms.id().to_owned())
                        .or_insert_with(|| Warp::new(tms.clone(), plane));
                    (
                        warp.series(tile, PLANE_TOLERANCE).map(lift),
                        warp.footprint(tile),
                    )
                }
            };
            self.tiles.insert(
                key.clone(),
                GlobeTile {
                    series: series.map(Arc::new),
                    footprint,
                    warp: None,
                    used: frame,
                },
            );
        }
        let entry = self.tiles.get_mut(&key).expect("inserted above");
        entry.used = frame;
        entry
    }

    /// Where the tile lands on the low-zoom plane.
    pub(crate) fn footprint(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> Option<Footprint> {
        self.entry(tms, tile).footprint
    }

    /// The tile's patches on the low-zoom surface, or `None` where its
    /// projection doesn't reach.
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
            self.retain(scene, |t| t.used == frame);
        }
        self.frame += 1;
    }

    fn retain(&mut self, scene: &mut Scene, keep: impl Fn(&GlobeTile) -> bool) {
        self.tiles.retain(|_, t| {
            let kept = keep(t);
            if !kept && let Some(warp) = t.warp {
                scene.remove_warp(warp);
            }
            kept
        });
    }
}

/// A plane's patches as positions in space at height 0.
fn lift(series: Series) -> Series<DVec3> {
    let lift = |v: glam::DVec2| v.extend(0.0);
    Series {
        anchor: lift(series.anchor),
        cells: series.cells,
        patches: series
            .patches
            .into_iter()
            .map(|p| orb_crs::Patch {
                points: p.points.map(lift),
            })
            .collect(),
        max_segment: series.max_segment,
        scale: series.scale,
    }
}
