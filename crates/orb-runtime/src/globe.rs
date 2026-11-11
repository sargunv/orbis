//! Tiles on the low-zoom surface (`GpuPerVertex` on a sphere or on a
//! low-zoom plane): each tile's quadratic patches there, kept by
//! tile matrix set and tile; the renderer uploads them when a placement
//! first needs them.

use std::cell::RefCell;
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
/// (a source planned as a `Warp`) and on the low-zoom surface. A memo:
/// each part is made when first asked for, from the tile and the planes
/// alone, so lookups take `&self`.
#[derive(Debug)]
pub(crate) struct GlobeWarps {
    surface: Projection,
    /// The low-zoom plane, or `None` for the sphere.
    plane: Option<Projection>,
    /// By set id, each set's warp onto the surface's plane (`[0]`) or the
    /// low-zoom one (`[1]`). Keyed by set first, so a lookup borrows its
    /// id rather than allocating a key.
    warps: RefCell<HashMap<String, [Option<Arc<Warp>>; 2]>>,
    tiles: RefCell<HashMap<String, HashMap<TileIndex, Tile>>>,
    /// Counts frames, to drop what recent ones didn't use.
    frame: u64,
}

/// A tile's places, each made when first asked for.
#[derive(Debug, Default)]
struct Tile {
    /// On the low-zoom surface: its patches.
    globe: Option<Option<Arc<Series<DVec3>>>>,
    /// On a low-zoom plane: its footprint, for the cover, which visits
    /// many tiles that never draw (so it doesn't fit their patches).
    low_footprint: Option<Option<Footprint>>,
    /// On the surface's plane: its footprint, and a vector tile's patches
    /// (`None` across a wrapping plane's seam, or bent too much).
    footprint: Option<Option<Footprint>>,
    series: Option<Option<Arc<Series>>>,
    used: u64,
}

impl GlobeWarps {
    pub(crate) fn new(surface: Projection) -> Self {
        Self {
            surface,
            plane: None,
            warps: RefCell::default(),
            tiles: RefCell::default(),
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

    /// `tms`'s warp onto the surface's plane, or (`low`) the low-zoom one.
    fn onto(&self, tms: &TileMatrixSet, low: bool) -> Arc<Warp> {
        let to = match low {
            true => self.plane.expect("a low-zoom plane"),
            false => self.surface,
        };
        let mut warps = self.warps.borrow_mut();
        let set = match warps.get_mut(tms.id()) {
            Some(set) => set,
            None => warps.entry(tms.id().to_owned()).or_default(),
        };
        Arc::clone(
            set[usize::from(low)].get_or_insert_with(|| Arc::new(Warp::new(tms.clone(), to))),
        )
    }

    /// `f` of `tile`'s entry, marked used this frame.
    fn tile<R>(&self, tms: &TileMatrixSet, tile: TileIndex, f: impl FnOnce(&mut Tile) -> R) -> R {
        let mut tiles = self.tiles.borrow_mut();
        let set = match tiles.get_mut(tms.id()) {
            Some(set) => set,
            None => tiles.entry(tms.id().to_owned()).or_default(),
        };
        let entry = set.entry(tile).or_default();
        entry.used = self.frame;
        f(entry)
    }

    /// `tms`'s warp onto the surface's plane.
    pub(crate) fn surface_warp(&self, tms: &TileMatrixSet) -> Arc<Warp> {
        self.onto(tms, false)
    }

    /// Where the tile lands on the low-zoom plane; `None` on the sphere.
    pub(crate) fn footprint(&self, tms: &TileMatrixSet, tile: TileIndex) -> Option<Footprint> {
        let warp = self.plane.map(|_| self.onto(tms, true))?;
        self.tile(tms, tile, |t| {
            *t.low_footprint.get_or_insert_with(|| warp.footprint(tile))
        })
    }

    /// The tile's patches on the low-zoom surface, or `None` where its
    /// projection doesn't reach.
    pub(crate) fn series(
        &self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Arc<Series<DVec3>>> {
        let warp = self.plane.map(|_| self.onto(tms, true));
        self.tile(tms, tile, |t| {
            let made = t.globe.get_or_insert_with(|| match &warp {
                None => orb_crs::sphere::series(tms, tile, TOLERANCE).map(Arc::new),
                Some(warp) => warp.series(tile, PLANE_TOLERANCE).map(lift).map(Arc::new),
            });
            made.clone()
        })
    }

    /// Cells per side (across, down) to cut the tile's geometry along on
    /// the globe.
    pub(crate) fn grid(&self, tms: &TileMatrixSet, tile: TileIndex) -> [u32; 2] {
        self.series(tms, tile).map_or([1; 2], |s| s.grid())
    }

    /// The tile's patches in the scene and their anchor's ECEF position,
    /// uploaded the first time.
    pub(crate) fn warp(
        &self,
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
        let warp = renderer.surface_warp(gpu, key, || {
            crate::meshes::series_words(&series, |v| v.as_vec3().to_array())
        })?;
        Some((warp, series.anchor))
    }

    /// Where a tile of another plane lands on the surface's.
    pub(crate) fn surface_footprint(
        &self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Footprint> {
        let warp = self.onto(tms, false);
        self.tile(tms, tile, |t| {
            *t.footprint.get_or_insert_with(|| warp.footprint(tile))
        })
    }

    /// A vector tile of another plane's patches on the surface's, or
    /// `None` if it can't be drawn through them.
    pub(crate) fn surface_series(
        &self,
        tms: &TileMatrixSet,
        tile: TileIndex,
    ) -> Option<Arc<Series>> {
        let warp = self.onto(tms, false);
        let made = |t: &mut Tile| {
            let series = || warp.series(tile, PLANE_TOLERANCE).map(Arc::new);
            t.series.get_or_insert_with(series).clone()
        };
        self.tile(tms, tile, made)
    }

    /// Ends a frame: past the cache's size, tiles this frame didn't use
    /// are dropped.
    pub(crate) fn end_frame(&mut self) {
        let tiles = self.tiles.get_mut();
        if tiles.values().map(HashMap::len).sum::<usize>() > CACHED {
            let frame = self.frame;
            for set in tiles.values_mut() {
                set.retain(|_, t| t.used == frame);
            }
        }
        self.frame += 1;
    }
}

/// A plane's patches as positions in space at height 0.
fn lift(series: Series) -> Series<DVec3> {
    let lift = |v: glam::DVec2| v.extend(0.0);
    Series {
        anchor: lift(series.anchor),
        window: series.window,
        cells: series.cells,
        patches: series
            .patches
            .into_iter()
            .map(|p| orb_crs::Patch {
                points: p.points.map(lift),
            })
            .collect(),
        max_cell: series.max_cell,
        scale: series.scale,
    }
}
