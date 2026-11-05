//! Tiles on the low-zoom surface (plan §9.2's `GpuPerVertex` on a sphere
//! or on a low-zoom plane): each tile's quadratic patches there, kept by
//! tile matrix set and tile; the renderer uploads them when a placement
//! first needs them.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;

use glam::DVec3;
use orb_crs::{Footprint, Projection, Series, TileIndex, TileMatrixSet, Warp};
use orb_hal::Gpu;
use orb_render::{Renderer, TileWarp, WarpKey};

/// A sixteenth of a pixel at 512 pixels per tile side on the sphere, and a
/// quarter of one on a plane, as the plane's own warps hold.
const TOLERANCE: f64 = 1.0 / 8192.0;
const PLANE_TOLERANCE: f64 = 1.0 / 2048.0;
/// Tiles kept before unused ones are dropped.
const CACHED: usize = 2048;

/// Tiles' places on planes other than their own: on the surface's plane
/// (a source planned as a `Warp`) and on the low-zoom surface.
#[derive(Debug)]
pub(crate) struct GlobeWarps {
    surface: Projection,
    /// The low-zoom plane, or `None` for the sphere.
    plane: Option<Projection>,
    /// Each set's warp onto the surface's plane (`false`) or the low-zoom
    /// one (`true`).
    warps: HashMap<(String, bool), Warp>,
    tiles: HashMap<(String, TileIndex), Tile>,
    /// Counts frames, to drop what recent ones didn't use.
    frame: u64,
}

/// A tile's patches on the low-zoom surface, and its footprint there.
type OnGlobe = (Option<Arc<Series<DVec3>>>, Option<Footprint>);

/// A tile's places, each made when first asked for.
#[derive(Debug, Default)]
struct Tile {
    /// On the low-zoom surface: its patches, and on a low-zoom plane its
    /// footprint, for the cover.
    globe: OnceCell<OnGlobe>,
    /// On the surface's plane: its footprint, and a vector tile's patches
    /// (`None` across a wrapping plane's seam, or bent too much).
    footprint: OnceCell<Option<Footprint>>,
    series: OnceCell<Option<Arc<Series>>>,
    used: u64,
}

impl GlobeWarps {
    pub(crate) fn new(surface: Projection) -> Self {
        Self {
            surface,
            plane: None,
            warps: HashMap::new(),
            tiles: HashMap::new(),
            frame: 0,
        }
    }

    /// Puts the tiles on the surface's plane `surface` and its low-zoom
    /// plane, or on the sphere (`None`), freeing those of others.
    pub(crate) fn set_planes(
        &mut self,
        renderer: &mut Renderer,
        surface: Projection,
        plane: Option<Projection>,
    ) {
        if (self.surface, self.plane) != (surface, plane) {
            renderer.clear_surface_warps();
            let frame = self.frame + 1;
            *self = Self {
                plane,
                frame,
                ..Self::new(surface)
            };
        }
    }

    /// `tms`'s warp onto the surface's plane.
    pub(crate) fn surface_warp(&mut self, tms: &TileMatrixSet) -> &Warp {
        self.at(tms, None, Some(false)).0.expect("asked for")
    }

    /// `tile`'s entry, and `tms`'s warp onto the surface's plane (`false`)
    /// or the low-zoom one (`true`) if asked for.
    fn at(
        &mut self,
        tms: &TileMatrixSet,
        tile: Option<TileIndex>,
        low: Option<bool>,
    ) -> (Option<&Warp>, Option<&Tile>) {
        let to = |low| {
            if low {
                self.plane.expect("a low-zoom plane")
            } else {
                self.surface
            }
        };
        let warp = low.map(|low| {
            let to = to(low);
            &*self
                .warps
                .entry((tms.id().to_owned(), low))
                .or_insert_with(|| Warp::new(tms.clone(), to))
        });
        let frame = self.frame;
        let tile = tile.map(|tile| {
            let entry = self.tiles.entry((tms.id().to_owned(), tile)).or_default();
            entry.used = frame;
            &*entry
        });
        (warp, tile)
    }

    fn globe(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> &OnGlobe {
        let low = self.plane.is_some();
        let (warp, entry) = self.at(tms, Some(tile), low.then_some(true));
        entry.expect("asked for").globe.get_or_init(|| match warp {
            None => (
                orb_crs::sphere::series(tms, tile, TOLERANCE).map(Arc::new),
                None,
            ),
            Some(warp) => (
                warp.series(tile, PLANE_TOLERANCE).map(lift).map(Arc::new),
                warp.footprint(tile),
            ),
        })
    }

    /// Where the tile lands on the low-zoom plane.
    pub(crate) fn footprint(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> Option<Footprint> {
        self.globe(tms, tile).1
    }

    /// The tile's patches on the low-zoom surface, or `None` where its
    /// projection doesn't reach.
    pub(crate) fn series(
        &mut self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Arc<Series<DVec3>>> {
        self.globe(tms, tile).0.clone()
    }

    /// Cells per side to cut the tile's geometry along on the globe.
    pub(crate) fn grid(&mut self, tms: &TileMatrixSet, tile: TileIndex) -> u32 {
        self.series(tms, tile).map_or(1, |s| s.grid())
    }

    /// The tile's patches in the scene and their anchor's ECEF position,
    /// uploaded the first time.
    pub(crate) fn warp(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<(TileWarp, DVec3)> {
        let series = self.series(tms, tile)?;
        let key = WarpKey {
            set: crate::terrain::hash(tms.id()),
            tile,
        };
        let warp = renderer.surface_warp(gpu, key, series.cells, || {
            series
                .patches
                .iter()
                .flat_map(|p| p.points.iter().flat_map(|v| v.as_vec3().to_array()))
                .collect()
        })?;
        Some((warp, series.anchor))
    }

    /// Where a tile of another plane lands on the surface's.
    pub(crate) fn surface_footprint(
        &mut self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Footprint> {
        let (warp, entry) = self.at(tms, Some(tile), Some(false));
        let (warp, entry) = (warp.expect("asked for"), entry.expect("asked for"));
        *entry.footprint.get_or_init(|| warp.footprint(tile))
    }

    /// A vector tile of another plane's patches on the surface's, or
    /// `None` if it can't be drawn through them.
    pub(crate) fn surface_series(
        &mut self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Arc<Series>> {
        let (warp, entry) = self.at(tms, Some(tile), Some(false));
        let (warp, entry) = (warp.expect("asked for"), entry.expect("asked for"));
        entry
            .series
            .get_or_init(|| warp.series(tile, PLANE_TOLERANCE).map(Arc::new))
            .clone()
    }

    /// Ends a frame: past the cache's size, tiles this frame didn't use
    /// are dropped.
    pub(crate) fn end_frame(&mut self) {
        if self.tiles.len() > CACHED {
            let frame = self.frame;
            self.tiles.retain(|_, t| t.used == frame);
        }
        self.frame += 1;
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
