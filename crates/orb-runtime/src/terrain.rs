//! Terrain on the scene side: the DEM tiles the
//! terrain stands on, its cover of terrain tiles, their drape slots, and
//! the CPU elevations the camera, cover and queries read.
//!
//! Terrain tiles are 512 logical pixels at their zoom, covered as GL JS's
//! terrain covers; each samples the DEM tile at its zoom less `delta`, so
//! that it spans 128 DEM pixels and its grid has a vertex at each: the
//! DEM detail of GL JS's terrain, whose tiles are twice the DEM tiles'
//! size, one DEM zoom deeper. Over 512-pixel DEM tiles GL JS's tiles are
//! 1024 pixels with the same 128 cells, so the renderer draws every
//! second vertex there (`orb_render`'s grid step), GL JS's mesh density;
//! slots and covers stay 512 pixels, so one atlas size serves every
//! DEM. A terrain tile's drape slots are drawn at the camera's zoom, to a
//! 32nd of a level, as a flat map would draw their content, at as many
//! texels as the tile shows at, and redrawn only when what they show
//! changes: what is draped under the tile, its size on screen, or for
//! stacks whose content scales with it, the zoom's step.

use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use glam::DVec2;
use orb_camera::View;
use orb_crs::{CoveredTile, Metres, TileIndex, TileMatrixSet, WorldRect};
use orb_hal::Gpu;
use orb_render::{
    ClipRect, DemRef, DrapeLayout, PageId, PlacementTarget, Renderer, TerrainDem, TerrainTile,
    TileRow,
};
use orb_source::{Draw, TilePyramid};

use orb_tile::DemGrid;

use crate::frame::DrapeSphere;

/// Terrain tiles' side at their zoom, in logical pixels.
pub(crate) const TERRAIN_TILE_PX: f64 = 512.0;
/// GL JS's terrain tile manager covers up to zoom 22.
const TERRAIN_MAX_ZOOM: u8 = 22;
/// Ready DEM tiles kept beyond those in use.
const DEM_CACHE: usize = 64;

/// A DEM tile the terrain stands on: its layer in the terrain's page and
/// its elevations on the CPU.
#[derive(Debug)]
pub(crate) struct DemTile {
    pub(crate) layer: u32,
    pub(crate) dem: Arc<Dem>,
}

/// A DEM tile's elevations, and the block the elevation
/// [`pyramid`](TerrainElevation::pyramid) carries for them: resampled the
/// first time a pyramid is encoded, then kept for the tile's life, through
/// every snapshot that holds it.
#[derive(Debug)]
pub(crate) struct Dem {
    pub(crate) grid: Arc<DemGrid>,
    coarse: OnceLock<Box<[u32]>>,
}

impl Dem {
    pub(crate) fn new(grid: Arc<DemGrid>) -> Self {
        Self {
            grid,
            coarse: OnceLock::new(),
        }
    }

    /// The grid at [`PYRAMID_DIM`]² samples, at the coarse pixels' centres
    /// (so the coarse tile samples like the full one, GL JS's pixel
    /// placement), quantized to 16 bits: `lo` and `step` as `f32` bits,
    /// then the samples in pairs. Bilinear within this grid alone: for
    /// grids of 128 pixels or a multiple, as DEM tiles are, no sample
    /// weighs a pixel past the edge, so a neighbour could not change it.
    fn coarse(&self) -> &[u32] {
        self.coarse.get_or_init(|| {
            let (n, grid) = (PYRAMID_DIM, &*self.grid);
            let samples: Vec<f32> = (0..n * n)
                .map(|i| {
                    let uv = (DVec2::new(f64::from(i % n), f64::from(i / n)) + 0.5) / f64::from(n);
                    bilinear(grid.dim(), uv, |i, j| grid.texel(i, j)) as f32
                })
                .collect();
            let (lo, hi) = samples
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                    (l.min(v), h.max(v))
                });
            let step = ((hi - lo) / 65535.0).max(f32::MIN_POSITIVE);
            let q: Vec<u16> = samples
                .iter()
                .map(|v| ((v - lo) / step).round().clamp(0.0, 65535.0) as u16)
                .collect();
            [lo.to_bits(), step.to_bits()]
                .into_iter()
                .chain(q.chunks(2).map(|p| u32::from(p[0]) | u32::from(p[1]) << 16))
                .collect()
        })
    }
}

/// GL JS's `get_elevation` at uv of a `dim`-pixel tile whose pixel
/// `(i, j)` is `texel(i, j)`.
fn bilinear(dim: u32, uv: DVec2, texel: impl Fn(i64, i64) -> f32) -> f64 {
    let coord = uv * f64::from(dim) - 0.5;
    let c = coord.floor();
    let f = coord - c;
    let (i, j) = (c.x as i64, c.y as i64);
    let t = |di, dj| f64::from(texel(i + di, j + dj));
    let top = t(0, 0) + (t(1, 0) - t(0, 0)) * f.x;
    let bottom = t(0, 1) + (t(1, 1) - t(0, 1)) * f.x;
    top + (bottom - top) * f.y
}

/// A terrain tile's drape slots, the side they were drawn at, the content
/// each stack's was drawn with (none while blank) and when.
#[derive(Debug)]
struct Slot {
    index: u32,
    shrink: u8,
    content: Vec<u64>,
    /// The plan that drew it ([`Terrain::plan`]).
    drawn: u64,
}

/// A tile slot to redraw.
#[derive(Debug)]
pub(crate) struct Redraw {
    pub(crate) tile: CoveredTile,
    pub(crate) slot: u32,
    /// Its slots' side, in halvings of their page's ([`slot_shrink`]).
    pub(crate) shrink: u8,
    /// The stacks whose content changed.
    pub(crate) stacks: Vec<u8>,
}

/// Cells a side of the grid a terrain tile's footprint is measured over
/// ([`shown_px`]).
const FOOTPRINT_CELLS: usize = 8;

/// How many times terrain tile `tile`'s drape slots halve their page's
/// side in `view` on a target of `pixel_ratio` physical pixels a logical
/// one: as many as leave √2 texels a logical pixel, or a texel a physical
/// one where those are finer, across the tile at its largest on screen
/// ([`shown_px`]). A flat view's tiles show at 362 to 724 pixels, which
/// takes them all 1024 texels on a target of one physical pixel a logical
/// one (fewer soften a checkerboard's edges). A tile of a pitched or globe
/// view that shows smaller holds and draws as few texels as it shows, a
/// quarter a halving, and one off screen the fewest: a function of the
/// camera and the terrain's `ceiling` (its highest ground there could be)
/// alone, as what slots show must be.
pub(crate) fn slot_shrink(
    (view, pixel_ratio): (&View, f64),
    tms: &TileMatrixSet,
    (tile, ceiling): (CoveredTile, Metres),
) -> u8 {
    let texels = shown_px(view, tms, (tile, ceiling)) * pixel_ratio.max(std::f64::consts::SQRT_2);
    let room = f64::from(orb_render::DRAPE_SLOT_PX) / texels.max(1.0);
    let most = f64::from(orb_render::MAX_SHRINK);
    room.log2().floor().clamp(0.0, most) as u8
}

/// The logical pixels terrain tile `tile` spans in `view`, level with the
/// camera's target, where it shows largest: its longest projected side,
/// as large as any where it reaches behind the eye. On a globe, also where
/// part of it shows larger, over a grid of [`FOOTPRINT_CELLS`] cells a
/// side, the longest projected side of a cell that meets the viewport at
/// some height from sea level to `ceiling`, the highest ground there could
/// be, as many times as the grid has cells, leaving out corners beyond the
/// horizon: a tile the horizon hides most of takes the part that shows. A
/// flat view's tiles span their side.
fn shown_px(view: &View, tms: &TileMatrixSet, (tile, ceiling): (CoveredTile, Metres)) -> f64 {
    const N: usize = FOOTPRINT_CELLS;
    let b = tms.tile_bounds(tile.index);
    let period = tms.projection().wrap_period().unwrap_or(0.0);
    let shift = DVec2::new(f64::from(tile.wrap) * period, 0.0);
    let grid = |height: Option<Metres>| {
        let mut at = [[None; N + 1]; N + 1];
        for (j, row) in at.iter_mut().enumerate() {
            for (i, point) in row.iter_mut().enumerate() {
                let f = DVec2::new(i as f64, j as f64) / N as f64;
                let p = b.min + b.size() * f + shift;
                *point = match height {
                    Some(h) => view.project_3d(p, h),
                    None => view.project(p),
                };
            }
        }
        at
    };
    let level = grid(None);
    let corners = [level[0][0], level[0][N], level[N][N], level[N][0]];
    let mut most = (0..4)
        .filter_map(|k| Some(corners[k]?.distance(corners[(k + 1) % 4]?)))
        .fold(0.0, f64::max);
    // Off the globe, a corner that doesn't project is behind the eye: the
    // tile reaches under the camera, where it shows larger than anywhere;
    // and a tile all in front shows at its longest side, wherever it lies.
    if view.globe().is_none() {
        return match corners.iter().all(Option::is_some) {
            true => most,
            false => f64::INFINITY,
        };
    }
    let band = [grid(Some(Metres(0.0))), grid(Some(ceiling))];
    let ring = |at: &[[Option<DVec2>; N + 1]; N + 1], i: usize, j: usize| {
        [at[j][i], at[j][i + 1], at[j + 1][i + 1], at[j + 1][i]]
    };
    let viewport = view.viewport();
    for j in 0..N {
        for i in 0..N {
            let (mut lo, mut hi) = (DVec2::INFINITY, DVec2::NEG_INFINITY);
            for p in (band.iter()).flat_map(|at| ring(at, i, j)).flatten() {
                (lo, hi) = (lo.min(p), hi.max(p));
            }
            let meets =
                hi.x >= 0.0 && hi.y >= 0.0 && lo.x <= viewport.width && lo.y <= viewport.height;
            if !meets {
                continue;
            }
            let corners = ring(&level, i, j);
            for k in 0..4 {
                if let (Some(p), Some(q)) = (corners[k], corners[(k + 1) % 4]) {
                    most = most.max(p.distance(q) * N as f64);
                }
            }
        }
    }
    most
}

#[derive(Debug)]
pub(crate) struct Terrain {
    /// Unique among the map's terrains, for revisions.
    pub(crate) id: u64,
    /// The DEM source's index in the plan and its key.
    pub(crate) source: usize,
    pub(crate) key: u32,
    /// The DEM source's tiles, held while the terrain falls flat after
    /// the style dropped it.
    pub(crate) endpoint: Arc<orb_source::endpoint::TileEndpoint>,
    /// As drawn: the style's, eased ([`crate::root`]).
    pub(crate) exaggeration: f64,
    pub(crate) pyramid: TilePyramid<DemTile>,
    /// The DEM tiles' page, made for the first.
    pub(crate) page: Option<PageId>,
    zooms: DemZooms,
    /// The DEM tiles' size in logical pixels.
    tile_px: f64,
    /// This frame's terrain tiles.
    pub(crate) tiles: Vec<CoveredTile>,
    /// The drawn slots of the cover's tiles: a tile without one is blank.
    slots: FxHashMap<CoveredTile, Slot>,
    /// The drawn slots of tiles the cover split, kept while a blank tile
    /// they cover shows their part of them ([`plan`](Self::plan)).
    retired: FxHashMap<CoveredTile, Slot>,
    generation: u64,
    /// Drape plans made.
    plans: u64,
    elevation: Arc<TerrainElevation>,
    /// The DEM tiles this camera's cover reads, sorted: the elevations hold
    /// these alone (and, while one loads, its nearest ready ancestor), not
    /// whatever else the pyramid keeps from earlier views.
    wanted: Vec<TileIndex>,
    /// The other DEM tiles whose ranges culled tiles of the cover, sorted:
    /// kept so that the next cover culls those tiles alike, and read for
    /// their ranges alone. Which these are depends on the ranges held
    /// before, so nothing drawn reads them: no elevation, no border.
    culling: Vec<TileIndex>,
    /// Per resident DEM tile padded, its layer and the neighbours' layers
    /// its border was last padded from ([`pads`](Self::pads)).
    padded: FxHashMap<TileIndex, (u32, [Option<u32>; 8])>,
    /// DEM tiles arrived, left or are wanted anew since the elevations were
    /// last taken.
    stale: bool,
    pub(crate) revision: u64,
}

impl Terrain {
    /// Terrain over the `raster-dem` source at `source` (its index in the
    /// plan and key) of `tile_px` tiles from `endpoint`.
    pub(crate) fn new(
        id: u64,
        (source, key): (usize, u32),
        exaggeration: f64,
        endpoint: Arc<orb_source::endpoint::TileEndpoint>,
        tms: &TileMatrixSet,
        tile_px: f64,
    ) -> Self {
        let delta = 1 + (tile_px / 256.0).log2().round().max(0.0) as u8;
        let (min_zoom, max_zoom) = (endpoint.minzoom, endpoint.maxzoom);
        Self {
            id,
            source,
            key,
            endpoint,
            exaggeration,
            pyramid: TilePyramid::new(min_zoom, max_zoom, DEM_CACHE),
            page: None,
            zooms: DemZooms {
                delta,
                min: min_zoom,
                max: max_zoom,
            },
            tile_px,
            tiles: Vec::new(),
            slots: FxHashMap::default(),
            retired: FxHashMap::default(),
            generation: 0,
            plans: 0,
            elevation: Arc::new(TerrainElevation::empty(tms.clone(), exaggeration)),
            wanted: Vec::new(),
            culling: Vec::new(),
            padded: FxHashMap::default(),
            stale: true,
            revision: 0,
        }
    }

    /// Draws it at `exaggeration` from now on: the elevations are taken
    /// again.
    pub(crate) fn set_exaggeration(&mut self, exaggeration: f64) {
        if exaggeration != self.exaggeration {
            self.exaggeration = exaggeration;
            self.stale = true;
        }
    }

    /// The ready DEM tile standing for DEM tile `tile`: itself, or its
    /// nearest ready ancestor while it loads.
    fn ready(&self, tile: TileIndex) -> Option<TileIndex> {
        (self.zooms.ancestors(tile)).find(|&at| self.pyramid.get(at).is_some())
    }

    /// The DEM tiles this camera reads: each wanted tile that is ready, or
    /// the ancestor standing for it. The elevations hold these, and these
    /// alone are padded.
    fn read(&self) -> Vec<TileIndex> {
        let mut read: Vec<TileIndex> = (self.wanted.iter())
            .filter_map(|&t| self.ready(t))
            .collect();
        read.sort();
        read.dedup();
        read
    }

    /// The DEM tile a tile at `tile` reads among those the elevations hold
    /// (its own, or the nearest ancestor of it held), as placement's
    /// [`TerrainElevation`] finds it: never one the pyramid keeps from an
    /// earlier view or for its range alone.
    fn dem_for(&self, tile: TileIndex) -> Option<(TileIndex, &DemTile)> {
        let dem = scaled(tile, self.zooms.of(tile.z)?);
        let at = (self.zooms.ancestors(dem)).find(|at| self.elevation.dems.contains_key(at))?;
        Some((at, self.pyramid.get(at)?))
    }

    /// Covers the view with at most `room` tiles, what the drape pages
    /// may hold ([`Map::cover_room`](crate::Map::cover_room)), asks for the
    /// DEM tiles missing and returns the tiles to request.
    ///
    /// The cover culls tiles by the elevations held, and what is held
    /// follows from the cover, so a DEM tile whose range culls a tile is
    /// kept too ([`culling`](Self::culling)). Otherwise a tile its own DEM
    /// culls lets go of that DEM, comes back under its ancestor's wider
    /// range and asks for it again: a view at rest would flip between two
    /// covers forever.
    fn update(
        &mut self,
        (view, tms, focus): (&View, &TileMatrixSet, orb_crs::Geodetic),
        room: usize,
    ) -> Vec<TileIndex> {
        let consulted = Consulted {
            elevation: &self.elevation,
            ranges: Mutex::new(Vec::new()),
        };
        let mut tiles = view.cover_terrain(tms, TERRAIN_TILE_PX, TERRAIN_MAX_ZOOM, &consulted);
        let ranges = consulted.ranges.into_inner().expect("ranges poisoned");
        // The tiles the cover drew or refined: any other tile it read a
        // range for, that range culled.
        let mut kept: FxHashSet<TileIndex> = tiles.iter().map(|t| t.index).collect();
        for t in &tiles {
            let mut at = t.index.parent();
            while let Some(parent) = at.filter(|&p| kept.insert(p)) {
                at = parent.parent();
            }
        }
        let culled_by: Vec<CoveredTile> = (ranges.into_iter())
            .filter(|(tile, _)| !kept.contains(tile))
            .map(|(_, index)| CoveredTile { index, wrap: 0 })
            .collect();
        let period = tms.projection().wrap_period().unwrap_or(0.0);
        let eye = view.eye().truncate();
        coarsen(&mut tiles, room, |t| {
            let offset = DVec2::new(f64::from(t.wrap) * period, 0.0) - view.camera_world();
            let rel = tms.tile_bounds(t.index).translated(offset);
            rel.clamp(eye).distance(eye)
        });
        self.tiles = tiles;
        let mut wanted: Vec<CoveredTile> = self
            .tiles
            .iter()
            .filter_map(|t| {
                Some(CoveredTile {
                    index: scaled(t.index, self.zooms.of(t.index.z)?),
                    wrap: 0,
                })
            })
            .collect();
        // The ground under the eye and its near plane, which the camera
        // keeps above (`View::clearance`), at the nearest tiles' detail.
        if let Some(z) = self
            .tiles
            .iter()
            .map(|t| t.index.z)
            .max()
            .and_then(|z| self.zooms.of(z))
        {
            wanted.extend(
                tms.tiles_in(view.near_footprint(), z)
                    .map(|index| CoveredTile { index, wrap: 0 }),
            );
        }
        wanted.sort();
        wanted.dedup();
        let indices: Vec<TileIndex> = wanted.iter().map(|t| t.index).collect();
        let mut culling: Vec<TileIndex> = (culled_by.iter())
            .map(|t| t.index)
            .filter(|t| indices.binary_search(t).is_err())
            .collect();
        culling.sort();
        culling.dedup();
        if indices != self.wanted || culling != self.culling {
            self.wanted = indices;
            self.culling = culling;
            self.stale = true;
        }
        wanted.extend(culled_by);
        wanted.sort();
        wanted.dedup();
        crate::frame::focus_first(&mut wanted, tms, focus);
        let update = self.pyramid.update(&wanted);
        update.request
    }

    /// The prepares that pad the DEM tiles this camera reads (those it
    /// wants, and the ancestors drawn while they load) from their ready
    /// neighbours among them, for the tiles whose neighbours changed since:
    /// a border is a function of what this camera reads, not of which tiles
    /// were resident before. A tile whose layer was padded before pads
    /// again only the borders whose neighbours changed.
    fn pads(&mut self, unpack: [f32; 4]) -> Vec<orb_render::DemPrepare> {
        let read = self.read();
        let pyramid = &self.pyramid;
        let layer_of = |t: TileIndex| {
            read.binary_search(&t)
                .ok()
                .and_then(|_| pyramid.get(t))
                .map(|d| d.layer)
        };
        // A layer keeps what it was padded from while its tile stays.
        self.padded
            .retain(|&tile, (layer, _)| pyramid.get(tile).is_some_and(|d| d.layer == *layer));
        let mut jobs = Vec::new();
        for &tile in &read {
            let Some(dem) = pyramid.get(tile) else {
                continue;
            };
            let neighbours = orb_render::dem_neighbours(tile, &layer_of);
            let was = self.padded.insert(tile, (dem.layer, neighbours));
            if was != Some((dem.layer, neighbours)) {
                jobs.push(orb_render::DemPrepare {
                    layer: dem.layer,
                    neighbours,
                    borders: was.map(|(_, was)| orb_render::changed_borders(was, neighbours)),
                    zoom: f32::from(tile.z),
                    unpack,
                });
            }
        }
        jobs
    }

    /// This frame's terrain over `view`, its DEM `unpack`ed, with skirts
    /// `skirt` world units long: covers the view, frees the DEM tiles it
    /// no longer keeps, pads borders and gives the renderer its paint.
    /// Returns the frame's terrain and the DEM tiles to load.
    pub(crate) fn upkeep(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        at: (&View, &TileMatrixSet, orb_crs::Geodetic),
        unpack: [f32; 4],
        (skirt, cover_room): (f64, u32),
    ) -> (orb_render::FrameTerrain, Vec<TileIndex>) {
        // The terrain draws before its DEM tiles arrive, and where its
        // source has none: flat, skirts and all, as GL JS draws it.
        let exaggeration = self.exaggeration as f32;
        let skirt = skirt as f32;
        let paint = orb_render::TerrainPaint {
            unpack,
            exaggeration,
            skirt,
            dem_tile_px: self.tile_px as f32,
        };
        renderer.set_terrain(gpu, Some((self.page, paint)));
        let requests = self.update(at, cover_room as usize);
        let evicted = self.pyramid.evict();
        self.stale |= !evicted.is_empty();
        if let Some(page) = self.page {
            for retired in evicted {
                renderer.free_layer(page, retired.layer);
            }
            for job in self.pads(unpack) {
                renderer.prepare_dem(page, job);
            }
        }
        let dem_dim = self.page.map_or(1, |page| renderer.page_size(page)[0]) as f32;
        // The DEM tiles the terrain tiles read, as `plan` gives them: a
        // tile without one draws at sea level.
        let lowest = (self.tiles.iter())
            .filter_map(|t| self.dem_for(t.index))
            .map(|(_, d)| d.dem.grid.min())
            .fold(0.0, f32::min);
        let frame = orb_render::FrameTerrain {
            unpack,
            exaggeration,
            dem_dim,
            lowest: lowest * exaggeration,
        };
        (frame, requests)
    }

    /// A DEM tile's `result` arrived: into the page and the elevations.
    /// `Err` says why it isn't drawn.
    pub(crate) fn arrive(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        tile: TileIndex,
        result: Result<crate::TileOutcome, String>,
        now: f64,
    ) -> Result<(), String> {
        use crate::TileOutcome::{Built, Cancelled, Missing};
        if !self.pyramid.is_loading(tile) {
            return Ok(());
        }
        let (image, grid) = match result {
            Ok(Missing { .. }) => {
                self.pyramid.insert_missing(tile);
                return Ok(());
            }
            // Given up before it loaded: a later cover that wants it asks
            // again.
            Ok(Cancelled) => {
                self.pyramid.forget_loading(tile);
                return Ok(());
            }
            Ok(Built(crate::TileResult {
                image: Some(image),
                dem: Some(dem),
                ..
            })) => (image, dem),
            Ok(Built(_)) | Err(_) => {
                self.pyramid.insert_failed(tile, now);
                return Err(result.err().unwrap_or("no elevations".into()));
            }
        };
        self.insert(gpu, renderer, tile, image, grid)
    }

    /// A built DEM tile into the page and the elevations.
    fn insert(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        tile: TileIndex,
        image: crate::jobs::TileImage,
        grid: Arc<DemGrid>,
    ) -> Result<(), String> {
        let size = [image.width, image.height];
        let page = *(self.page)
            .get_or_insert_with(|| renderer.create_page(gpu, orb_render::PageKind::Terrain, size));
        let layer = match renderer.page_size(page) {
            other if other != size => Err(format!("{}×{} among {other:?} tiles", size[0], size[1])),
            _ => (self.alloc_layer(gpu, renderer, page))
                .map_err(|_| "out of terrain texture layers".into()),
        };
        let layer = layer.inspect_err(|_| self.pyramid.insert_empty(tile))?;
        renderer.write_layer(gpu, page, layer, image.pixels.upload(false));
        self.stale = true;
        let dem = Arc::new(Dem::new(grid));
        if let Some(old) = self.pyramid.insert(tile, DemTile { layer, dem }) {
            renderer.free_layer(page, old.layer);
        }
        Ok(())
    }

    /// DEM tile `tile`, in layer `layer`, stands on `grid` from now on:
    /// the app's new heights, written over its layer. False if the tile
    /// isn't in that layer.
    pub(crate) fn set_heights(&mut self, tile: TileIndex, layer: u32, grid: &Arc<DemGrid>) -> bool {
        let Some(dem) = self.pyramid.get_mut(tile).filter(|d| d.layer == layer) else {
            return false;
        };
        dem.dem = Arc::new(Dem::new(Arc::clone(grid)));
        self.stale = true;
        true
    }

    /// A layer of the DEM page: a free one, else one DEM tiles the cover no
    /// longer reads leave, least recently used first, else one the page
    /// grows to make, to the device's limits.
    fn alloc_layer(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        page: PageId,
    ) -> Result<u32, orb_render::OutOfSpace> {
        loop {
            match renderer.alloc_layer(gpu, page, false) {
                Err(_) => match self.pyramid.evict_lru() {
                    Some(old) => {
                        renderer.free_layer(page, old.layer);
                        self.stale = true;
                    }
                    None => return renderer.alloc_layer(gpu, page, true),
                },
                done => return done,
            }
        }
    }

    /// Gives back its DEM page.
    pub(crate) fn release(self, gpu: &mut dyn Gpu, renderer: &mut Renderer) {
        renderer.set_terrain(gpu, None);
        if let Some(page) = self.page {
            renderer.remove_page(gpu, page);
        }
    }

    /// The CPU elevations of the DEM tiles this camera reads, taken again
    /// when they change.
    pub(crate) fn elevation(&mut self, tms: &TileMatrixSet) -> &Arc<TerrainElevation> {
        if std::mem::take(&mut self.stale) {
            // Each tile, or while it loads its nearest ready ancestor.
            let held = |tiles: Vec<TileIndex>| -> FxHashMap<TileIndex, Arc<Dem>> {
                (tiles.into_iter())
                    .filter_map(|t| Some((t, Arc::clone(&self.pyramid.get(t)?.dem))))
                    .collect()
            };
            let dems = held(self.read());
            let culling = self.culling.iter().filter_map(|&t| self.ready(t));
            let mut culling = held(culling.collect());
            culling.retain(|t, _| !dems.contains_key(t));
            let same = |a: &FxHashMap<TileIndex, Arc<Dem>>, b: &FxHashMap<TileIndex, Arc<Dem>>| {
                a.len() == b.len()
                    && a.iter()
                        .all(|(t, d)| b.get(t).is_some_and(|h| Arc::ptr_eq(h, d)))
            };
            // The same tiles as before: the same elevations, no new revision.
            if same(&dems, &self.elevation.dems)
                && same(&culling, &self.elevation.culling)
                && self.elevation.exaggeration == self.exaggeration
            {
                return &self.elevation;
            }
            let mut elevation = TerrainElevation::empty(tms.clone(), self.exaggeration);
            elevation.dems = dems;
            elevation.culling = culling;
            elevation.zooms = self.zooms;
            self.elevation = Arc::new(elevation);
            self.revision += 1;
        }
        &self.elevation
    }

    /// The elevations as last taken.
    pub(crate) fn snapshot(&self) -> Arc<TerrainElevation> {
        Arc::clone(&self.elevation)
    }

    /// The DEM a placement of `tile` stands on, and the range of
    /// elevations under it.
    pub(crate) fn ground(&self, tile: TileIndex) -> (DemRef, [f32; 2]) {
        let Some((dem_tile, DemTile { layer, dem })) = self.dem_for(tile) else {
            return (DemRef::NONE, [0.0; 2]);
        };
        let depth = tile.z - dem_tile.z;
        let offset = |v: u32| (v - (v >> depth << depth)) as u16;
        let e = self.exaggeration as f32;
        (
            DemRef {
                layer: *layer as u16,
                depth,
                offset: [offset(tile.col), offset(tile.row)],
            },
            [dem.grid.min() * e, dem.grid.max() * e],
        )
    }

    /// Retires the slots of tiles this frame doesn't cover: those drawn
    /// stand in for the blank tiles they cover until these are drawn.
    pub(crate) fn release_slots(&mut self) {
        let keep: FxHashSet<CoveredTile> = self.tiles.iter().copied().collect();
        let gone: Vec<CoveredTile> = (self.slots.keys())
            .filter(|t| !keep.contains(t))
            .copied()
            .collect();
        for tile in gone {
            let slot = self.slots.remove(&tile).expect("listed above");
            self.retired.insert(tile, slot);
        }
        // A tile the cover takes back keeps the slot it retired.
        for tile in &self.tiles {
            if let Some(slot) = self.retired.remove(tile) {
                self.slots.insert(*tile, slot);
            }
        }
    }

    /// Whether every tile of the cover is drawn in the slot of its place,
    /// as [`plan`](Self::plan) leaves them once nothing is deferred.
    #[cfg(test)]
    fn slots_follow_cover(&self) -> bool {
        self.retired.is_empty()
            && (0u32..)
                .zip(&self.tiles)
                .all(|(place, tile)| self.slots.get(tile).is_some_and(|s| s.index == place))
    }

    /// The nearest retired ancestor of `tile`, whose slot it shows while
    /// blank.
    fn stand_in(&self, tile: CoveredTile) -> Option<CoveredTile> {
        let mut at = tile.index.parent();
        while let Some(index) = at {
            let ancestor = CoveredTile {
                index,
                wrap: tile.wrap,
            };
            if self.retired.contains_key(&ancestor) {
                return Some(ancestor);
            }
            at = index.parent();
        }
        None
    }

    /// This frame's terrain tiles for the renderer, with slots assigned
    /// from `layout`, the tile slots' stacks it redraws, and whether it
    /// left any for the frames after. A stack is stale when its draped
    /// content, as `content` hashes it per stack, changed, and every stack
    /// of a tile whose slots' side in `view` did ([`slot_shrink`]).
    ///
    /// What a frame redraws is bounded by `budget`'s texels and slots
    /// (unbounded for none), so that it follows the screen, not how fast
    /// the zoom crosses the drape's steps or how many tiles a cover
    /// splits. Blank tiles go first: a tile the cover split shows its part
    /// of its ancestor's slot, retired, until drawn; one with no such
    /// ancestor (new ground, or a merge) is drawn whatever the budget.
    /// Then stale ones, the longest stale first, each drawn as it was
    /// until a frame redraws it. Of those alike, the ones that show
    /// largest go first, then those nearest the view's centre. Each tile
    /// takes the slot of its place in the cover, whose place in the drape
    /// grid it draws in, as a fresh map's does. A move is a redraw, within
    /// the budget, into a place left free. Once none is blank, stale or
    /// out of place, each slot holds what a fresh map's does. Slots' sides
    /// follow their tiles' footprints in `view` on a target of its pixel
    /// ratio ([`slot_shrink`]). On a globe, `globe` places a tile on the
    /// sphere.
    pub(crate) fn plan(
        &mut self,
        tms: &TileMatrixSet,
        (view, pixel_ratio): (&View, f32),
        (layout, budget): (DrapeLayout, Option<(u64, usize)>),
        content: &dyn Fn(CoveredTile) -> Vec<u64>,
        globe: &mut dyn FnMut(TileIndex) -> Option<orb_render::TerrainGlobe>,
    ) -> (Vec<TerrainTile>, Vec<Redraw>, bool) {
        self.plans += 1;
        if layout.generation != self.generation {
            self.generation = layout.generation;
            self.slots.clear();
            self.retired.clear();
        }
        debug_assert!(self.tiles.len() <= layout.tile_slots as usize);
        let places: FxHashMap<CoveredTile, u32> = self.tiles.iter().copied().zip(0..).collect();
        let period = tms.projection().wrap_period().unwrap_or(0.0);
        // Out of place with no free slot to move through: every tile takes
        // its place in one frame, drawn anew where it moves. So too
        // wherever the budget is unbounded (a still map). A continuous
        // map's pages keep a slot past its cover for this
        // (`Map::cover_room`), so that only pages of a single slot do.
        let held = self.slots.len() + self.retired.len();
        let misplaced = (self.slots.iter()).any(|(t, s)| places[t] != s.index);
        let texels = budget.map(|b| b.0);
        if texels.is_none() || misplaced && held >= layout.tile_slots as usize {
            self.retired.clear();
            self.slots.retain(|t, s| places[t] == s.index);
        }
        let ceiling = Metres(orb_camera::HIGHEST_GROUND.0 * self.exaggeration);
        let mut wants = Vec::new();
        for (place, &tile) in (0u32..).zip(&self.tiles) {
            let content = content(tile);
            let shrink = slot_shrink((view, f64::from(pixel_ratio)), tms, (tile, ceiling));
            let need = match self.slots.get(&tile) {
                None => Need::Blank(self.stand_in(tile)),
                Some(slot) if slot.index != place => Need::Move,
                Some(slot) => {
                    let stacks: Vec<u8> = (0..content.len())
                        .filter(|&s| slot.shrink != shrink || slot.content.get(s) != content.get(s))
                        .map(|s| s as u8)
                        .collect();
                    if stacks.is_empty() {
                        continue;
                    }
                    Need::Stale(stacks)
                }
            };
            wants.push(Want {
                place,
                shrink,
                content,
                need,
            });
        }
        // A retired slot stays only while a blank tile shows it.
        let mut shown: FxHashMap<CoveredTile, u32> = FxHashMap::default();
        for want in &wants {
            if let Need::Blank(Some(stand_in)) = want.need {
                *shown.entry(stand_in).or_default() += 1;
            }
        }
        self.retired.retain(|t, _| shown.contains_key(t));
        let order = |want: &Want| {
            let tile = self.tiles[want.place as usize];
            let rank = match want.need {
                Need::Blank(None) => 0,
                Need::Blank(Some(_)) => 1,
                Need::Stale(_) | Need::Move => 2,
            };
            let drawn = self.slots.get(&tile).map_or(0, |s| s.drawn);
            // Nearest the camera's target first: the view's centre.
            let target = view.camera_world();
            let shift = DVec2::new(f64::from(tile.wrap) * period, 0.0);
            let bounds = tms.tile_bounds(tile.index).translated(shift);
            let distance = bounds.clamp(target).distance(target).to_bits();
            (rank, drawn, want.shrink, distance, want.place)
        };
        wants.sort_by_cached_key(order);
        let mut used: FxHashSet<u32> = (self.slots.values().chain(self.retired.values()))
            .map(|s| s.index)
            .collect();
        let all: Vec<u8> = (0..layout.stack_count() as u8).collect();
        let mut spent = 0;
        let mut redraw = Vec::new();
        let mut deferred = false;
        // A tile out of place whose place another holds, and whether any
        // moved: where none could, the first takes a free slot, so that
        // the place it leaves frees another's (they took each other's).
        let (mut blocked, mut moved) = (None, false);
        // The slots the frame redraws.
        let mut slots = 0;
        for want in wants {
            let tile = self.tiles[want.place as usize];
            let forced = matches!(want.need, Need::Blank(None));
            if !forced && texels.is_some_and(|t| spent >= t) {
                deferred = true;
                continue;
            }
            // Each slot a frame redraws replays the drape's draws, so a
            // continuous map's frame redraws few: what else is due waits
            // for a frame to come.
            if !forced && slots >= budget.map_or(usize::MAX, |b| b.1) {
                deferred = true;
                continue;
            }
            let (index, stacks) = match want.need {
                Need::Stale(stacks) => (self.slots[&tile].index, stacks),
                // Into its place once that's free.
                Need::Move if used.contains(&want.place) => {
                    deferred = true;
                    blocked.get_or_insert((tile, want.shrink, want.content));
                    continue;
                }
                Need::Move => {
                    moved = true;
                    used.remove(&self.slots[&tile].index);
                    (want.place, all.clone())
                }
                Need::Blank(stand_in) => {
                    let free = (0..layout.tile_slots).find(|i| !used.contains(i));
                    let index = match free {
                        _ if !used.contains(&want.place) => want.place,
                        Some(index) => index,
                        // Each retired slot left is shown by one blank
                        // tile alone (they number no more than those, and
                        // every slot is held): this one's, which it no
                        // longer shows once drawn.
                        None => {
                            let stand_in = stand_in.expect("a free slot for a tile without one");
                            shown.remove(&stand_in);
                            let retired = self.retired.remove(&stand_in).expect("kept above");
                            used.remove(&retired.index);
                            retired.index
                        }
                    };
                    // A retired slot no blank tile shows any more is free.
                    if let Some(n) = stand_in.and_then(|s| shown.get_mut(&s)) {
                        *n -= 1;
                        if *n == 0 {
                            let stand_in = stand_in.expect("counted");
                            shown.remove(&stand_in);
                            let retired = self.retired.remove(&stand_in).expect("kept above");
                            used.remove(&retired.index);
                        }
                    }
                    (index, all.clone())
                }
            };
            used.insert(index);
            slots += 1;
            spent += (stacks.iter())
                .map(|&s| u64::from(layout.slot_px(u32::from(s)) >> want.shrink).pow(2))
                .sum::<u64>();
            redraw.push(Redraw {
                tile,
                slot: index,
                shrink: want.shrink,
                stacks,
            });
            self.slots.insert(
                tile,
                Slot {
                    index,
                    shrink: want.shrink,
                    content: want.content,
                    drawn: self.plans,
                },
            );
        }
        let free = (0..layout.tile_slots).find(|i| !used.contains(i));
        let room = texels.is_none_or(|t| spent < t);
        if let (Some((tile, shrink, content)), Some(index), false, true) =
            (blocked, free, moved, room)
        {
            redraw.push(Redraw {
                tile,
                slot: index,
                shrink,
                stacks: all,
            });
            let drawn = self.plans;
            self.slots.insert(
                tile,
                Slot {
                    index,
                    shrink,
                    content,
                    drawn,
                },
            );
        }
        let mut out = Vec::with_capacity(self.tiles.len());
        for tile in &self.tiles {
            // A blank tile shows its part of its ancestor's slot.
            let (slot, in_slot) = match self.slots.get(tile) {
                Some(slot) => (slot, ([0.0; 2], 1.0)),
                None => {
                    let stand_in = self.stand_in(*tile).expect("blank tiles stand in");
                    (&self.retired[&stand_in], within(tile.index, stand_in.index))
                }
            };
            let bounds = tms.tile_bounds(tile.index);
            let origin = DVec2::new(bounds.min.x + f64::from(tile.wrap) * period, bounds.max.y)
                - view.camera_world();
            let dem = self.dem_for(tile.index).map(|(dem_tile, dem)| {
                let (offset, scale) = within(tile.index, dem_tile);
                TerrainDem {
                    layer: dem.layer,
                    offset,
                    scale,
                }
            });
            out.push(TerrainTile {
                origin: [origin.x as f32, origin.y as f32],
                scale: (tms.tile_span(tile.index.z) / f64::from(orb_render::TERRAIN_EXTENT)) as f32,
                dem,
                slot: slot.index,
                within: in_slot,
                globe: globe(tile.index),
                latitudes: orb_render::tile_latitudes(tile.index),
            });
        }
        (out, redraw, deferred)
    }
}

/// What a cover tile's slots need this frame ([`Terrain::plan`]).
#[derive(Debug)]
enum Need {
    /// Drawing: it has no slot, but shows that of its retired ancestor, if
    /// any.
    Blank(Option<CoveredTile>),
    /// These stacks drawn again.
    Stale(Vec<u8>),
    /// Drawing into the slot of its place.
    Move,
}

#[derive(Debug)]
struct Want {
    place: u32,
    shrink: u8,
    content: Vec<u64>,
    need: Need,
}

/// Where `tile` lies in its ancestor (or itself) `at`: tile uv → `at`'s uv
/// as `uv · scale + offset`.
fn within(tile: TileIndex, at: TileIndex) -> ([f32; 2], f32) {
    let depth = tile.z - at.z;
    let scale = 0.5f32.powi(i32::from(depth));
    let d = |v: u32| (v - (v >> depth << depth)) as f32 * scale;
    ([d(tile.col), d(tile.row)], scale)
}

/// Coarsens a cover until it has at most `room` tiles (or no tiles
/// share an ancestor): the tiles under the farthest ancestor of two or
/// more, by `distance` from the eye, become that ancestor, as distance
/// already coarsens the cover.
fn coarsen(tiles: &mut Vec<CoveredTile>, room: usize, distance: impl Fn(CoveredTile) -> f64) {
    while tiles.len() > room {
        let mut under: FxHashMap<CoveredTile, usize> = FxHashMap::default();
        for t in tiles.iter() {
            let mut at = t.index;
            while let Some(parent) = at.parent() {
                let wrap = t.wrap;
                *under
                    .entry(CoveredTile {
                        index: parent,
                        wrap,
                    })
                    .or_default() += 1;
                at = parent;
            }
        }
        let farthest = (under.into_iter())
            .filter(|&(_, n)| n > 1)
            .map(|(t, _)| (distance(t), t))
            .max_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
        let Some((_, merged)) = farthest else {
            break;
        };
        tiles.retain(|t| {
            t.wrap != merged.wrap
                || t.index.z < merged.index.z
                || scaled(t.index, merged.index.z) != merged.index
        });
        tiles.push(merged);
    }
}

/// `tile`'s ancestor (or itself) at zoom `z ≤ tile.z`.
pub(crate) fn scaled(tile: TileIndex, z: u8) -> TileIndex {
    let d = tile.z - z;
    TileIndex::new(z, tile.col >> d, tile.row >> d)
}

/// A drawn tile's place on the surface: its bounds in its world copy (a
/// warped tile's frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Drawn {
    pub(crate) bounds: WorldRect,
}

impl Drawn {
    pub(crate) fn of(
        tms: &TileMatrixSet,
        draw: &Draw,
        frame: Option<WorldRect>,
        period: f64,
    ) -> Self {
        let bounds = frame.unwrap_or_else(|| tms.tile_bounds(draw.tile));
        Self {
            bounds: bounds.translated(DVec2::new(f64::from(draw.wrap) * period, 0.0)),
        }
    }

    /// The part of it that region `region` of its draw (a tile at its
    /// zoom or deeper) covers.
    pub(crate) fn region(&self, tms: &TileMatrixSet, draw: &Draw, region: TileIndex) -> Self {
        let (own, part) = (tms.tile_bounds(draw.tile), tms.tile_bounds(region));
        let to = |p: DVec2| self.bounds.min + (p - own.min) / own.size() * self.bounds.size();
        Self {
            bounds: WorldRect {
                min: to(part.min),
                max: to(part.max),
            },
        }
    }

    /// Whether it overlaps a terrain tile of the surface's matrix set, or
    /// comes within `reach_px` of it, in logical pixels at style zoom
    /// `zoom`, which the tile's slots are drawn at: what a translation
    /// that far moves into them.
    #[cfg(test)]
    pub(crate) fn overlaps(
        &self,
        surface: &TileMatrixSet,
        terrain: CoveredTile,
        (zoom, reach_px): (f64, f64),
    ) -> bool {
        self.within(&Reach::of(surface, terrain, (zoom, None, reach_px)))
    }

    /// Whether it overlaps a terrain tile's [`Reach`].
    pub(crate) fn within(&self, reach: &Reach) -> bool {
        let (a, b, eps) = (self.bounds, reach.bounds, reach.eps);
        a.min.x < b.max.x - eps
            && b.min.x < a.max.x - eps
            && a.min.y < b.max.y - eps
            && b.min.y < a.max.y - eps
    }
}

/// A terrain tile's bounds grown by `reach_px` logical pixels at style
/// zoom `zoom` on `sphere`, wherever in the tile they are
/// ([`Drawn::overlaps`]): worked out once for every drawn
/// tile tested against it, as finding the tile's style zoom inverts the
/// projection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reach {
    bounds: WorldRect,
    eps: f64,
}

impl Reach {
    pub(crate) fn of(
        surface: &TileMatrixSet,
        terrain: CoveredTile,
        (zoom, sphere, reach_px): (f64, Option<DrapeSphere>, f64),
    ) -> Self {
        let b = terrain_bounds(surface, terrain);
        let most = sphere.map_or(1.0, |s| s.most(b.min.y, b.max.y));
        let reach =
            DVec2::splat(reach_px * most * b.size().x / slot_span_px(surface, terrain, zoom));
        Self {
            bounds: WorldRect {
                min: b.min - reach,
                max: b.max + reach,
            },
            eps: 1e-9 * surface.tile_span(0),
        }
    }
}

/// Logical pixels a terrain tile's slots span at style zoom `zoom`: the
/// 512 of the tile at its own zoom, scaled.
fn slot_span_px(surface: &TileMatrixSet, terrain: CoveredTile, zoom: f64) -> f64 {
    TERRAIN_TILE_PX * (zoom - surface.style_zoom(terrain.index)).exp2()
}

/// A terrain tile's bounds in its world copy.
fn terrain_bounds(surface: &TileMatrixSet, terrain: CoveredTile) -> WorldRect {
    let period = surface.projection().wrap_period().unwrap_or(0.0);
    surface
        .tile_bounds(terrain.index)
        .translated(DVec2::new(f64::from(terrain.wrap) * period, 0.0))
}

/// Drape placements of a drawn tile (`extent` data units a side, mapped
/// by its map placements `map_rows`) for a terrain tile and its tile
/// slot, its slots `shrink` halvings of their page's side, drawn at style
/// zoom `zoom` ([`drape_zoom`](crate::frame::drape_zoom)) to show on
/// `sphere`: one row set per
/// drape stack of `stacks` (those its source's layers draw into), each
/// mapping the tile's data into the stack's slot, clipped to the terrain
/// tile and the `reach_px` (logical pixels) a translation moves content
/// into it from, bounded by the terrain tile, with pixel sizes in slot
/// texels at that zoom (on the sphere, where the camera's target has
/// them, and as each point's latitude has them there).
#[expect(clippy::too_many_arguments, reason = "one placement's terms")]
pub(crate) fn drape_rows(
    surface: &TileMatrixSet,
    drawn: Drawn,
    extent: u32,
    (terrain, slot, shrink): (CoveredTile, u32, u8),
    (layout, stacks): (&DrapeLayout, &[u8]),
    (zoom, sphere, reach_px): (f64, Option<DrapeSphere>, f64),
    map_rows: impl Iterator<Item = TileRow>,
    out: &mut Vec<TileRow>,
) {
    let origin = DVec2::new(drawn.bounds.min.x, drawn.bounds.max.y);
    let scale = drawn.bounds.size().x / f64::from(extent);
    let t = terrain_bounds(surface, terrain);
    let t_origin = DVec2::new(t.min.x, t.max.y);
    let span = t.size().x;
    // The terrain tile in the drawn tile's data units.
    let lo = (t_origin - origin) / scale * DVec2::new(1.0, -1.0);
    let hi = lo + DVec2::splat(span / scale);
    let clamp = |v: f64| v.round().clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
    let bound = ClipRect::new([clamp(lo.x), clamp(lo.y)], [clamp(hi.x), clamp(hi.y)]);
    let slot_span_px = slot_span_px(surface, terrain, zoom);
    let most = sphere.map_or(1.0, |s| s.most(t.min.y, t.max.y));
    let reach = DVec2::splat(reach_px * most * span / slot_span_px / scale);
    let sphere = sphere.map(|s| s.row(origin.y, scale));
    let (lo, hi) = (lo - reach, hi + reach);
    let (lo, hi) = ([clamp(lo.x), clamp(lo.y)], [clamp(hi.x), clamp(hi.y)]);
    for row in map_rows {
        let mut clips = [ClipRect::EMPTY; 4];
        let mut any = false;
        for (clip, out) in row.clips.iter().zip(&mut clips) {
            let min = [clip.min[0].max(lo[0]), clip.min[1].max(lo[1])];
            let max = [clip.max[0].min(hi[0]), clip.max[1].min(hi[1])];
            if min[0] < max[0] && min[1] < max[1] {
                *out = ClipRect::new(min, max);
                any = true;
            }
        }
        if !any {
            continue;
        }
        for &stack in stacks {
            let [cx, cy] = layout.corner(slot, u32::from(stack));
            let corner = DVec2::new(f64::from(cx), -f64::from(cy));
            // Texels per world unit, and per logical pixel at the zoom.
            let slot_px = f64::from(layout.slot_px(u32::from(stack)) >> shrink);
            let k = slot_px / span;
            let o = corner + (origin - t_origin) * k;
            out.push(TileRow {
                origin: o.to_array(),
                scale: (scale * k) as f32,
                clips,
                bound,
                px_scale: (slot_px / slot_span_px) as f32,
                sphere,
                target: PlacementTarget::Drape(stack),
                // Slots are flat: the terrain carries them onto a globe.
                globe: None,
                ..row
            });
        }
    }
}

/// A content hash.
pub(crate) fn hash(value: impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// The zooms a terrain's DEM tiles come in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DemZooms {
    /// DEM zooms below a terrain tile's.
    delta: u8,
    min: u8,
    max: u8,
}

impl DemZooms {
    /// The DEM zoom a tile at zoom `z` reads: GL JS's `_getSourceZoom`,
    /// never deeper than the tile.
    fn of(self, z: u8) -> Option<u8> {
        let zd = z.saturating_sub(self.delta).min(self.max).max(self.min);
        (zd <= z).then_some(zd)
    }

    /// DEM tile `tile` and its ancestors down to the least zoom, nearest
    /// first: where a tile that hasn't loaded finds one standing for it.
    fn ancestors(self, tile: TileIndex) -> impl Iterator<Item = TileIndex> {
        std::iter::successors(Some(tile), move |at| {
            (at.z > self.min).then(|| at.parent()).flatten()
        })
    }
}

/// The terrain's elevations on the CPU: a snapshot of the ready DEM tiles
/// the camera and queries hold while the map moves on. It
/// follows DEM arrival and eviction, never the view.
#[derive(Debug)]
pub struct TerrainElevation {
    tms: TileMatrixSet,
    exaggeration: f64,
    zooms: DemZooms,
    dems: FxHashMap<TileIndex, Arc<Dem>>,
    /// DEM tiles the cover reads ranges from and nothing else
    /// (the terrain's `culling`).
    culling: FxHashMap<TileIndex, Arc<Dem>>,
    /// The pyramid, encoded once for every reader of this snapshot.
    encoded: OnceLock<Arc<[u8]>>,
}

impl TerrainElevation {
    fn empty(tms: TileMatrixSet, exaggeration: f64) -> Self {
        Self {
            tms,
            exaggeration,
            zooms: DemZooms {
                delta: 1,
                min: 0,
                max: 0,
            },
            dems: FxHashMap::default(),
            culling: FxHashMap::default(),
            encoded: OnceLock::new(),
        }
    }

    /// Bytes of DEM data held.
    pub fn bytes(&self) -> usize {
        (self.dems.values().chain(self.culling.values()))
            .map(|d| d.grid.bytes())
            .sum()
    }

    /// The tile at zoom `z` containing a world position in the primary
    /// world copy, with the position's uv in it.
    fn locate(&self, world: DVec2, z: u8) -> Option<(TileIndex, DVec2)> {
        // From the corner of the quadtree's grid, which a partial matrix's
        // extent may not start at.
        let corner = self.tms.tile_bounds(TileIndex::new(z, 0, 0));
        let span = self.tms.tile_span(z);
        let x = (world.x - corner.min.x) / span;
        let y = (corner.max.y - world.y) / span;
        if !(x >= 0.0 && y >= 0.0 && x < f64::from(u32::MAX) && y < f64::from(u32::MAX)) {
            return None;
        }
        let tile = TileIndex::new(z, x as u32, y as u32);
        self.tms
            .contains(tile)
            .then(|| (tile, DVec2::new(x.fract(), y.fract())))
    }

    fn wrapped(&self, world: DVec2) -> DVec2 {
        let bounds = self.tms.projection().world_bounds();
        match self.tms.projection().wrap_period() {
            Some(period) => DVec2::new(
                (world.x - bounds.min.x).rem_euclid(period) + bounds.min.x,
                world.y,
            ),
            None => world,
        }
    }

    /// The DEM tile `tile` reads, or its nearest loaded ancestor.
    fn dem_for(&self, tile: TileIndex) -> Option<(TileIndex, &DemGrid)> {
        self.held_for(tile, false)
    }

    /// [`dem_for`](Self::dem_for) among the tiles held, with `culling`
    /// those held for their ranges too.
    fn held_for(&self, tile: TileIndex, culling: bool) -> Option<(TileIndex, &DemGrid)> {
        let dem = scaled(tile, self.zooms.of(tile.z)?);
        self.zooms.ancestors(dem).find_map(|at| {
            let dem =
                (self.dems.get(&at)).or_else(|| culling.then(|| self.culling.get(&at)).flatten());
            Some((at, &*dem?.grid))
        })
    }

    /// A DEM pixel, from the neighbouring tile past an edge as the GPU's
    /// padded pages backfill it.
    fn texel(&self, tile: TileIndex, grid: &DemGrid, i: i64, j: i64) -> f32 {
        let dim = i64::from(grid.dim());
        let (dx, dy) = (i.div_euclid(dim), j.div_euclid(dim));
        if (dx, dy) == (0, 0) || dx.abs() > 1 || dy.abs() > 1 {
            return grid.texel(i, j);
        }
        let (cols, rows) = self.tms.matrix_size(tile.z);
        let row = i64::from(tile.row) + dy;
        if !(0..i64::from(rows)).contains(&row) {
            return grid.texel(i, j);
        }
        let col = (i64::from(tile.col) + dx).rem_euclid(i64::from(cols));
        match self
            .dems
            .get(&TileIndex::new(tile.z, col as u32, row as u32))
        {
            Some(n) if n.grid.dim() == grid.dim() => n.grid.texel(i - dx * dim, j - dy * dim),
            _ => grid.texel(i, j),
        }
    }

    /// GL JS's `get_elevation` at a DEM tile's uv, in metres.
    fn sample(&self, tile: TileIndex, grid: &DemGrid, uv: DVec2) -> f64 {
        bilinear(grid.dim(), uv, |i, j| self.texel(tile, grid, i, j))
    }

    /// Elevation of terrain tile `tile` at its uv, from the DEM tile it
    /// draws with: GL JS's `TerrainCoverage` sample, bilinear between DEM
    /// pixels as the mesh's vertices are.
    fn on_tile(&self, tile: TileIndex, uv: DVec2) -> Option<f64> {
        let (dem_tile, grid) = self.dem_for(tile)?;
        let depth = tile.z - dem_tile.z;
        let scale = 0.5f64.powi(i32::from(depth));
        let offset = |v: u32| f64::from(v - (v >> depth << depth)) * scale;
        let uv = DVec2::new(offset(tile.col), offset(tile.row)) + uv * scale;
        Some(self.sample(dem_tile, grid, uv) * self.exaggeration)
    }
}

/// Elevation samples per side of each tile of the elevation
/// [`pyramid`](TerrainElevation::pyramid).
const PYRAMID_DIM: u32 = 128;

impl TerrainElevation {
    /// The elevation pyramid: these elevations with each DEM
    /// tile resampled to 128² samples quantized to 16 bits, as bytes, for
    /// a camera or placement on another thread (the web's ui role and
    /// placement worker). [`from_pyramid`](Self::from_pyramid) reads them
    /// back. Encoded once a snapshot, from blocks resampled once a DEM
    /// tile: a new snapshot with the same tiles costs a copy.
    pub fn pyramid(&self) -> Arc<[u8]> {
        Arc::clone(self.encoded.get_or_init(|| self.encode()))
    }

    fn encode(&self) -> Arc<[u8]> {
        let block = 2 + (PYRAMID_DIM * PYRAMID_DIM).div_ceil(2) as usize;
        let mut words: Vec<u32> = Vec::with_capacity(16 + (3 + block) * self.dems.len());
        words.extend([
            1,
            (self.exaggeration as f32).to_bits(),
            u32::from(self.zooms.delta),
            u32::from(self.zooms.min),
            u32::from(self.zooms.max),
            self.dems.len() as u32,
        ]);
        let id = self.tms.id().as_bytes();
        words.push(id.len() as u32);
        for chunk in id.chunks(4) {
            let mut b = [0u8; 4];
            b[..chunk.len()].copy_from_slice(chunk);
            words.push(u32::from_le_bytes(b));
        }
        for (tile, dem) in &self.dems {
            words.extend([u32::from(tile.z), tile.col, tile.row]);
            words.extend_from_slice(dem.coarse());
        }
        Arc::from(bytemuck::cast_slice::<u32, u8>(&words))
    }

    /// Reads an [`pyramid`](Self::pyramid).
    pub fn from_pyramid(bytes: &[u8]) -> Result<Self, String> {
        let bad = || "malformed elevation pyramid".to_string();
        if !bytes.len().is_multiple_of(4) {
            return Err(bad());
        }
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        let mut at = 0usize;
        let mut next = || -> Result<u32, String> {
            let w = *words.get(at).ok_or_else(bad)?;
            at += 1;
            Ok(w)
        };
        if next()? != 1 {
            return Err(bad());
        }
        let exaggeration = f64::from(f32::from_bits(next()?));
        let (delta, min_zoom, max_zoom) = (next()? as u8, next()? as u8, next()? as u8);
        let tiles = next()?;
        let id_len = next()? as usize;
        let mut id = Vec::new();
        for _ in 0..id_len.div_ceil(4) {
            id.extend(next()?.to_le_bytes());
        }
        id.truncate(id_len);
        let id = String::from_utf8(id).map_err(|_| bad())?;
        let tms = TileMatrixSet::by_id(&id).map_err(|e| format!("elevation pyramid: {e}"))?;
        let mut elevation = Self::empty(tms, exaggeration);
        elevation.zooms = DemZooms {
            delta,
            min: min_zoom,
            max: max_zoom,
        };
        let n = PYRAMID_DIM;
        for _ in 0..tiles {
            let (z, col, row) = (next()? as u8, next()?, next()?);
            let (lo, step) = (f32::from_bits(next()?), f32::from_bits(next()?));
            let mut samples = Vec::with_capacity((n * n) as usize);
            for _ in 0..(n * n).div_ceil(2) {
                let w = next()?;
                samples.extend([w & 0xffff, w >> 16].map(|q| lo + q as f32 * step));
            }
            samples.truncate((n * n) as usize);
            let grid = DemGrid::from_elevations(n, samples)?;
            elevation.dems.insert(
                TileIndex::new(z, col, row),
                Arc::new(Dem::new(Arc::new(grid))),
            );
        }
        Ok(elevation)
    }
}

/// A pyramid's words ([`TerrainElevation::pyramid`]): its header, through
/// its tile matrix set's id, and each DEM tile's block (its index, then
/// its samples) by tile.
struct PyramidWords<'a> {
    header: &'a [u32],
    blocks: Vec<(TileIndex, &'a [u32])>,
}

/// Words in a DEM tile's block: its index, its range and its samples.
const PYRAMID_BLOCK: usize = 3 + 2 + (PYRAMID_DIM * PYRAMID_DIM).div_ceil(2) as usize;

impl<'a> PyramidWords<'a> {
    /// Reads a pyramid of version `version`, whose block count is its
    /// sixth word, its blocks after its header; what follows them stays.
    fn read(words: &'a [u32], version: u32) -> Result<(Self, &'a [u32]), String> {
        let bad = || "malformed elevation pyramid".to_string();
        if words.first() != Some(&version) || words.len() < 7 {
            return Err(bad());
        }
        let (tiles, id) = (words[5] as usize, (words[6] as usize).div_ceil(4));
        let (header, mut rest) = words.split_at_checked(7 + id).ok_or_else(bad)?;
        let mut blocks = Vec::with_capacity(tiles);
        for _ in 0..tiles {
            let (block, after) = rest.split_at_checked(PYRAMID_BLOCK).ok_or_else(bad)?;
            let tile = TileIndex::new(block[0] as u8, block[1], block[2]);
            blocks.push((tile, block));
            rest = after;
        }
        Ok((Self { header, blocks }, rest))
    }
}

/// What a reader holding pyramid `old` ([`TerrainElevation::pyramid`])
/// needs to hold `new`: `new`'s header, the blocks of DEM tiles `old`
/// lacks or has other samples for, and the tiles only `old` has; a
/// placement worker takes the terrain's changes rather than all of it
/// again. [`apply_pyramid_delta`] reads it.
pub fn pyramid_delta(old: Option<&[u8]>, new: &[u8]) -> Result<Vec<u8>, String> {
    let words = |bytes: &[u8]| -> Vec<u32> { bytemuck::pod_collect_to_vec(bytes) };
    let (old_words, new_words) = (old.map(words), words(new));
    let (new, _) = PyramidWords::read(&new_words, 1)?;
    let old = (old_words.as_deref())
        .map(|w| PyramidWords::read(w, 1).map(|(old, _)| old))
        .transpose()?;
    let held: FxHashMap<TileIndex, &[u32]> = old
        .as_ref()
        .map(|old| old.blocks.iter().copied().collect())
        .unwrap_or_default();
    let added: Vec<&[u32]> = (new.blocks.iter())
        .filter(|(tile, block)| held.get(tile) != Some(block))
        .map(|(_, block)| *block)
        .collect();
    let kept: FxHashSet<TileIndex> = new.blocks.iter().map(|(tile, _)| *tile).collect();
    let removed: Vec<TileIndex> = held.keys().filter(|t| !kept.contains(t)).copied().collect();
    let mut out: Vec<u32> = Vec::with_capacity(new.header.len() + added.len() * PYRAMID_BLOCK);
    out.push(2);
    out.extend_from_slice(&new.header[1..5]);
    out.push(added.len() as u32);
    out.extend_from_slice(&new.header[6..]);
    for block in added {
        out.extend_from_slice(block);
    }
    out.push(removed.len() as u32);
    for tile in removed {
        out.extend([u32::from(tile.z), tile.col, tile.row]);
    }
    Ok(bytemuck::cast_slice(&out).to_vec())
}

/// Pyramid `old` (none for an empty one) with `delta` ([`pyramid_delta`])
/// applied: the pyramid the delta was made from.
pub fn apply_pyramid_delta(old: Option<&[u8]>, delta: &[u8]) -> Result<Vec<u8>, String> {
    let bad = || "malformed elevation pyramid delta".to_string();
    if !delta.len().is_multiple_of(4) {
        return Err(bad());
    }
    let delta_words: Vec<u32> = bytemuck::pod_collect_to_vec(delta);
    let (delta, rest) = PyramidWords::read(&delta_words, 2)?;
    let (&removed, rest) = rest.split_first().ok_or_else(bad)?;
    if rest.len() != removed as usize * 3 {
        return Err(bad());
    }
    let removed: FxHashSet<TileIndex> = (rest.as_chunks::<3>().0.iter())
        .map(|&[z, col, row]| TileIndex::new(z as u8, col, row))
        .collect();
    let old_words: Vec<u32> = old.map(bytemuck::pod_collect_to_vec).unwrap_or_default();
    let old = match old {
        Some(_) => Some(PyramidWords::read(&old_words, 1)?.0),
        None => None,
    };
    let mut blocks: FxHashMap<TileIndex, &[u32]> = (old.iter())
        .flat_map(|old| old.blocks.iter().copied())
        .filter(|(tile, _)| !removed.contains(tile))
        .collect();
    blocks.extend(delta.blocks.iter().copied());
    let mut out: Vec<u32> = vec![1];
    out.extend_from_slice(&delta.header[1..5]);
    out.push(blocks.len() as u32);
    out.extend_from_slice(&delta.header[6..]);
    for block in blocks.values() {
        out.extend_from_slice(block);
    }
    Ok(bytemuck::cast_slice(&out).to_vec())
}

impl orb_symbol::Ground for TerrainElevation {
    /// GL JS's `Terrain.getElevation(tileID, x, y)`: the DEM tile the drawn
    /// tile reads, as its placements' DEM references name it on the GPU.
    fn elevation(&self, tile: orb_symbol::TileId, x: f64, y: f64) -> f64 {
        let index = TileIndex::new(tile.z, tile.x, tile.y);
        let uv = DVec2::new(x, y) / orb_symbol::EXTENT;
        self.on_tile(index, uv).unwrap_or(0.0)
    }

    fn pyramid(&self) -> Arc<[u8]> {
        TerrainElevation::pyramid(self)
    }
}

impl orb_camera::Elevation for TerrainElevation {
    fn at(&self, world: DVec2) -> Option<Metres> {
        let world = self.wrapped(world);
        // The deepest DEM tile loaded, which the nearest terrain draws with.
        (self.zooms.min..=self.zooms.max).rev().find_map(|z| {
            let (tile, uv) = self.locate(world, z)?;
            let dem = self.dems.get(&tile)?;
            Some(Metres(self.sample(tile, &dem.grid, uv) * self.exaggeration))
        })
    }

    fn range(&self, rect: WorldRect, z: u8) -> Option<(Metres, Metres)> {
        self.range_of(rect, z).map(|(_, _, range)| range)
    }

    fn ceiling(&self) -> Metres {
        Metres(orb_camera::HIGHEST_GROUND.0 * self.exaggeration)
    }
}

impl TerrainElevation {
    /// The tile at zoom `z` under `rect`, the DEM tile it reads and that
    /// DEM's range of elevations.
    fn range_of(&self, rect: WorldRect, z: u8) -> Option<(TileIndex, TileIndex, (Metres, Metres))> {
        let (tile, _) = self.locate(self.wrapped(rect.center()), z)?;
        let (dem, grid) = self.held_for(tile, true)?;
        let e = self.exaggeration;
        Some((
            tile,
            dem,
            (
                Metres(f64::from(grid.min()) * e),
                Metres(f64::from(grid.max()) * e),
            ),
        ))
    }
}

/// Elevations that note, per tile asked about, the DEM tile whose range
/// answered.
#[derive(Debug)]
struct Consulted<'a> {
    elevation: &'a TerrainElevation,
    ranges: Mutex<Vec<(TileIndex, TileIndex)>>,
}

impl orb_camera::Elevation for Consulted<'_> {
    fn at(&self, world: DVec2) -> Option<Metres> {
        self.elevation.at(world)
    }

    fn range(&self, rect: WorldRect, z: u8) -> Option<(Metres, Metres)> {
        let (tile, dem, range) = self.elevation.range_of(rect, z)?;
        self.ranges
            .lock()
            .expect("ranges poisoned")
            .push((tile, dem));
        Some(range)
    }

    fn ceiling(&self) -> Metres {
        self.elevation.ceiling()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_camera::Elevation;

    fn tms() -> TileMatrixSet {
        TileMatrixSet::web_mercator().clone()
    }

    /// A 4×4 DEM tile rising eastward by 10 m a pixel from 100 m.
    fn ramp() -> Arc<Dem> {
        let terrarium = [256.0, 1.0, 1.0 / 256.0, 32768.0];
        let texels: Vec<u8> = (0..16)
            .flat_map(|i| {
                let metres = 100 + 10 * (i % 4);
                let v = 32768 + metres;
                [(v >> 8) as u8, (v & 255) as u8, 0, 255]
            })
            .collect();
        Arc::new(Dem::new(Arc::new(
            DemGrid::from_rgba(4, &texels, terrarium).unwrap(),
        )))
    }

    /// A `dim`-pixel DEM tile of rolling hills, 0–1000 m, its phase
    /// shifted by `seed`.
    fn hills(dim: u32, seed: f64) -> Arc<Dem> {
        let hills: Vec<f32> = (0..dim * dim)
            .map(|i| {
                let (x, y) = (f64::from(i % dim) + seed, f64::from(i / dim) - seed);
                (500.0 + 300.0 * (x / 20.0).sin() + 200.0 * (y / 31.0).cos()) as f32
            })
            .collect();
        Arc::new(Dem::new(Arc::new(
            DemGrid::from_elevations(dim, hills).unwrap(),
        )))
    }

    /// The pyramid as encoded before blocks were kept: every DEM tile
    /// resampled with its neighbours, on every call.
    fn reference(e: &TerrainElevation) -> Vec<u8> {
        let mut words: Vec<u32> = vec![
            1,
            (e.exaggeration as f32).to_bits(),
            u32::from(e.zooms.delta),
            u32::from(e.zooms.min),
            u32::from(e.zooms.max),
            e.dems.len() as u32,
        ];
        let id = e.tms.id().as_bytes();
        words.push(id.len() as u32);
        for chunk in id.chunks(4) {
            let mut b = [0u8; 4];
            b[..chunk.len()].copy_from_slice(chunk);
            words.push(u32::from_le_bytes(b));
        }
        let n = PYRAMID_DIM;
        for (tile, dem) in &e.dems {
            let samples: Vec<f32> = (0..n * n)
                .map(|i| {
                    let uv = (DVec2::new(f64::from(i % n), f64::from(i / n)) + 0.5) / f64::from(n);
                    e.sample(*tile, &dem.grid, uv) as f32
                })
                .collect();
            let (lo, hi) = samples
                .iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                    (l.min(v), h.max(v))
                });
            let step = ((hi - lo) / 65535.0).max(f32::MIN_POSITIVE);
            words.extend([
                u32::from(tile.z),
                tile.col,
                tile.row,
                lo.to_bits(),
                step.to_bits(),
            ]);
            let q: Vec<u16> = samples
                .iter()
                .map(|v| ((v - lo) / step).round().clamp(0.0, 65535.0) as u16)
                .collect();
            words.extend(q.chunks(2).map(|p| u32::from(p[0]) | u32::from(p[1]) << 16));
        }
        bytemuck::cast_slice(&words).to_vec()
    }

    /// Each DEM tile's block, by where it lives: the same address is the
    /// same resample.
    fn blocks(e: &TerrainElevation) -> FxHashMap<TileIndex, usize> {
        e.dems
            .iter()
            .filter_map(|(&t, d)| Some((t, d.coarse.get()?.as_ptr() as usize)))
            .collect()
    }

    /// A terrain over 256-pixel DEM tiles to zoom 12 with `dems` arrived.
    fn terrain_with(dems: &[(TileIndex, Arc<Dem>)]) -> Terrain {
        let style = orb_style::v8::parse(
            r#"{"version": 8, "sources": {"dem": {"type": "raster-dem",
                "tiles": ["t/{z}/{x}/{y}.png"], "tileSize": 256, "maxzoom": 12}}, "layers": []}"#,
        )
        .unwrap();
        let endpoint =
            orb_source::endpoint::TileEndpoint::resolve(&style.sources[0].kind, None, tms())
                .unwrap();
        let mut terrain = Terrain::new(1, (0, 0), 1.5, Arc::new(endpoint), &tms(), 256.0);
        arrive(&mut terrain, dems);
        terrain
    }

    /// DEM tiles wanted, as a cover would, and arrived.
    fn arrive(terrain: &mut Terrain, dems: &[(TileIndex, Arc<Dem>)]) {
        let mut wanted: Vec<CoveredTile> = terrain
            .pyramid
            .ready()
            .map(|(index, _)| CoveredTile { index, wrap: 0 })
            .collect();
        wanted.extend(
            dems.iter()
                .map(|&(index, _)| CoveredTile { index, wrap: 0 }),
        );
        wanted.sort();
        terrain.wanted = wanted.iter().map(|t| t.index).collect();
        terrain.pyramid.update(&wanted);
        for (layer, (tile, dem)) in dems.iter().enumerate() {
            let tile_dem = DemTile {
                layer: layer as u32,
                dem: Arc::clone(dem),
            };
            assert!(terrain.pyramid.insert(*tile, tile_dem).is_none());
        }
        terrain.stale = true;
    }

    #[test]
    fn covers_coarsen_their_farthest_tiles_to_fit() {
        let at = |z, col, row| CoveredTile {
            index: TileIndex::new(z, col, row),
            wrap: 0,
        };
        // Four tiles near the eye at (0, 0), four farther, and one alone.
        let near = [at(4, 0, 0), at(4, 1, 0), at(4, 0, 1), at(4, 1, 1)];
        let far = [at(4, 2, 2), at(4, 3, 2), at(4, 2, 3), at(4, 3, 3)];
        let mut tiles: Vec<_> = near.into_iter().chain(far).chain([at(3, 3, 0)]).collect();
        let distance = |t: CoveredTile| {
            let s = f64::from(1u32 << (8 - t.index.z));
            f64::from(t.index.col).hypot(f64::from(t.index.row)) * s
        };
        coarsen(&mut tiles, 6, distance);
        let want: Vec<_> = near.into_iter().chain([at(3, 3, 0), at(3, 1, 1)]).collect();
        assert_eq!(tiles, want, "the far four become their parent");
        // Within the room: as covered.
        let before = tiles.clone();
        coarsen(&mut tiles, 6, distance);
        assert_eq!(tiles, before);
        // Down to what no two tiles share: the root.
        coarsen(&mut tiles, 1, distance);
        assert_eq!(tiles, [at(0, 0, 0)]);
    }

    #[test]
    fn pyramid_blocks_are_resampled_once_a_dem_tile() {
        let tms = tms();
        let quad = |z, c, r| TileIndex::new(z, c, r);
        let mut terrain = terrain_with(&[
            (quad(10, 530, 360), hills(256, 0.0)),
            (quad(10, 531, 360), hills(256, 7.0)),
        ]);
        let first = Arc::clone(terrain.elevation(&tms));
        let bytes = first.pyramid();
        let built = blocks(&first);
        assert_eq!(built.len(), 2);
        // Asked again, and by placement through `Ground`: the same bytes.
        assert!(Arc::ptr_eq(&bytes, &first.pyramid()));
        let ground: Arc<dyn orb_symbol::Ground> = Arc::clone(&first) as _;
        assert!(Arc::ptr_eq(&bytes, &ground.pyramid()));
        // The cover changed: the same snapshot, which follows DEM tiles
        // alone.
        terrain.tiles = vec![CoveredTile {
            index: quad(11, 1060, 720),
            wrap: 0,
        }];
        assert!(Arc::ptr_eq(&first, terrain.elevation(&tms)));
        assert_eq!(*bytes, *reference(&first));
        // One DEM tile arrives: one block resampled.
        arrive(&mut terrain, &[(quad(10, 530, 361), hills(256, 3.0))]);
        let added = Arc::clone(terrain.elevation(&tms));
        added.pyramid();
        let after = blocks(&added);
        assert_eq!(after.len(), 3);
        assert!(built.iter().all(|(t, b)| after[t] == *b));
        // One reloaded with new data: its block alone resampled.
        let reloaded = quad(10, 531, 360);
        for (tile, dem) in terrain.pyramid.ready_mut() {
            if tile == reloaded {
                dem.dem = hills(256, 11.0);
            }
        }
        terrain.stale = true;
        let replaced = Arc::clone(terrain.elevation(&tms));
        replaced.pyramid();
        let last = blocks(&replaced);
        assert_ne!(last[&reloaded], after[&reloaded]);
        assert!(
            after
                .iter()
                .filter(|(t, _)| **t != reloaded)
                .all(|(t, b)| last[t] == *b)
        );
        assert_eq!(*replaced.pyramid(), *reference(&replaced));
    }

    #[test]
    fn dems_kept_for_their_ranges_pad_no_borders_and_hold_no_elevations() {
        let tms = tms();
        let quad = |z, c, r| TileIndex::new(z, c, r);
        let (drawn, culling) = (quad(10, 530, 360), quad(10, 531, 360));
        let mut terrain = terrain_with(&[(drawn, hills(256, 0.0)), (culling, hills(256, 7.0))]);
        terrain.wanted = vec![drawn];
        terrain.culling = vec![culling];
        // The drawn tile's east border is its own edge, not its resident
        // neighbour's.
        let jobs = terrain.pads([0.0; 4]);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].neighbours, [None; 8]);
        // The cover reads the neighbour's range; nothing reads its
        // elevations, nor sends them to placement.
        let elevation = Arc::clone(terrain.elevation(&tms));
        let under = |t: TileIndex| tms.tile_bounds(t).center();
        assert!(elevation.at(under(drawn)).is_some());
        assert!(elevation.at(under(culling)).is_none());
        let child = tms.tile_bounds(quad(11, 1062, 720));
        assert!(elevation.range(child, 11).is_some());
        let placed = TerrainElevation::from_pyramid(&elevation.pyramid()).unwrap();
        assert_eq!(placed.dems.keys().collect::<Vec<_>>(), [&drawn]);
    }

    #[test]
    fn drawn_tiles_stand_on_the_dem_tiles_placement_reads() {
        let tms = tms();
        let quad = |z, c, r| TileIndex::new(z, c, r);
        let (coarse, cached) = (quad(9, 265, 180), quad(10, 530, 360));
        let mut terrain = terrain_with(&[(coarse, hills(256, 0.0)), (cached, hills(256, 7.0))]);
        // A cover coarsened to the drape's room reads the zoom 9 tile; the
        // zoom 10 one under it stays cached from a closer view.
        terrain.wanted = vec![coarse];
        let jobs = terrain.pads([0.0; 4]);
        assert_eq!(jobs.len(), 1, "the cached tile isn't padded");
        let elevation = Arc::clone(terrain.elevation(&tms));
        let tile = quad(11, 1060, 720);
        assert_eq!(elevation.dem_for(tile).map(|(t, _)| t), Some(coarse));
        let (dem, _) = terrain.ground(tile);
        assert_eq!((dem.layer, dem.depth), (0, 2), "the coarse tile's layer");
    }

    #[test]
    fn kept_blocks_encode_as_resampling_did() {
        let tms = tms();
        // DEM tiles of 128 pixels and multiples, as DEM sources serve
        // them, with neighbours on every side of one; at two
        // exaggerations.
        for dim in [128, 256, 512] {
            for exaggeration in [1.0, 2.5] {
                let mut e = TerrainElevation::empty(tms.clone(), exaggeration);
                e.zooms.max = 12;
                for (c, r) in [(530, 360), (529, 360), (531, 360), (530, 359), (530, 361)] {
                    e.dems
                        .insert(TileIndex::new(10, c, r), hills(dim, f64::from(c * 3 + r)));
                }
                assert_eq!(*e.pyramid(), *reference(&e), "{dim} px, ×{exaggeration}");
            }
        }
        // Smaller tiles stand alone: their edges clamp rather than read a
        // neighbour.
        let mut e = TerrainElevation::empty(tms, 1.0);
        e.zooms.max = 12;
        e.dems.insert(TileIndex::new(10, 530, 360), hills(64, 0.0));
        assert_eq!(*e.pyramid(), *reference(&e));
    }

    #[test]
    fn elevations_follow_the_dem_pixels() {
        let tms = tms();
        let mut elevation = TerrainElevation::empty(tms.clone(), 2.0);
        elevation.zooms.delta = 1;
        elevation.zooms.max = 12;
        let dem = TileIndex::new(4, 8, 8);
        elevation.dems.insert(dem, ramp());
        let bounds = tms.tile_bounds(dem);
        // Pixel 1's centre, 3/8 across: 110 m, doubled.
        let at = |u: f64, v: f64| {
            DVec2::new(
                bounds.min.x + u * bounds.size().x,
                bounds.max.y - v * bounds.size().y,
            )
        };
        assert!((elevation.at(at(0.375, 0.5)).unwrap().0 - 220.0).abs() < 1e-9);
        assert_eq!(
            elevation.range(tms.tile_bounds(TileIndex::new(5, 16, 16)), 5),
            Some((Metres(200.0), Metres(260.0)))
        );
        // Nothing loaded elsewhere.
        assert!(elevation.at(at(-1.0, 0.5)).is_none());
    }

    /// A placement worker that holds a pyramid takes the next as the DEM
    /// tiles that came and went, and holds what it would have whole.
    #[test]
    fn pyramid_deltas_carry_only_what_changed() {
        let tms = tms();
        let quad = |z, c, r| TileIndex::new(z, c, r);
        let mut terrain = terrain_with(&[
            (quad(10, 530, 360), hills(256, 0.0)),
            (quad(10, 531, 360), hills(256, 7.0)),
        ]);
        let first = terrain.elevation(&tms).pyramid();
        let whole = pyramid_delta(None, &first).unwrap();
        assert_eq!(apply_pyramid_delta(None, &whole).unwrap(), *first);
        arrive(&mut terrain, &[(quad(10, 530, 361), hills(256, 3.0))]);
        let second = terrain.elevation(&tms).pyramid();
        let delta = pyramid_delta(Some(&first), &second).unwrap();
        assert!(delta.len() < second.len() / 2, "one block of three");
        let held = apply_pyramid_delta(Some(&first), &delta).unwrap();
        let tiles = |bytes: &[u8]| {
            let elevation = TerrainElevation::from_pyramid(bytes).unwrap();
            let mut tiles: Vec<_> = elevation.dems.keys().copied().collect();
            tiles.sort();
            tiles
        };
        assert_eq!(tiles(&held), tiles(&second));
        assert_eq!(held.len(), second.len());
        assert!(apply_pyramid_delta(Some(&first), &delta[..delta.len() - 4]).is_err());
    }

    #[test]
    fn the_pyramid_keeps_elevations_within_metres() {
        let tms = tms();
        let mut full = TerrainElevation::empty(tms.clone(), 1.5);
        full.zooms.max = 12;
        // A 256-pixel tile of rolling hills, 0–1000 m.
        let dim = 256u32;
        let hills: Vec<f32> = (0..dim * dim)
            .map(|i| {
                let (x, y) = (f64::from(i % dim), f64::from(i / dim));
                (500.0 + 300.0 * (x / 20.0).sin() + 200.0 * (y / 31.0).cos()) as f32
            })
            .collect();
        let tile = TileIndex::new(10, 530, 360);
        full.dems.insert(
            tile,
            Arc::new(Dem::new(Arc::new(
                DemGrid::from_elevations(dim, hills).unwrap(),
            ))),
        );
        let coarse = TerrainElevation::from_pyramid(&full.pyramid()).unwrap();
        let bounds = tms.tile_bounds(tile);
        let mut worst = 0.0f64;
        for i in 0..50 {
            for j in 0..50 {
                let p = DVec2::new(
                    bounds.min.x + (f64::from(i) + 0.5) / 50.0 * bounds.size().x,
                    bounds.max.y - (f64::from(j) + 0.5) / 50.0 * bounds.size().y,
                );
                let (a, b) = (full.at(p).unwrap().0, coarse.at(p).unwrap().0);
                worst = worst.max((a - b).abs());
            }
        }
        // Four pixels a sample over hills 20 pixels across.
        assert!(worst < 10.0, "{worst}");
        assert!(worst > 0.0);
    }

    /// A terrain tile's slots take the texels it shows at: full in a view
    /// straight down, fewer for a pitched view's far tiles, more of them
    /// on a target of more physical pixels a logical one.
    #[test]
    fn slots_shrink_with_their_tiles_on_screen() {
        let tms = tms();
        let mercator = *tms.projection();
        let viewport = orb_camera::Viewport {
            width: 1024.0,
            height: 768.0,
        };
        let shrinks_at = |zoom: f64, pitch: f64, ratio: f64| {
            let target = orb_crs::Geodetic::new(7.75, 46.02);
            let mut camera = orb_camera::CameraState::new(target, zoom, &mercator);
            camera.pitch = pitch;
            let view = View::new(&camera, viewport, &orb_crs::Surface::plane(mercator));
            (view.cover(&tms, TERRAIN_TILE_PX, 22).into_iter())
                .map(|t| (t, slot_shrink((&view, ratio), &tms, (t, Metres(0.0)))))
                .collect::<Vec<_>>()
        };
        let shrinks = |zoom, pitch| shrinks_at(zoom, pitch, 1.0);
        // Straight down, every tile shows at 512 pixels or more.
        for zoom in [11.0, 11.5, 11.99] {
            assert!(shrinks(zoom, 0.0).iter().all(|&(_, s)| s == 0), "z{zoom}");
        }
        // A pitched view's far tiles show smaller than its near ones, and
        // their slots with them.
        let tilted = shrinks(11.5, 70.0);
        assert!(tilted.iter().any(|&(_, s)| s >= 1), "{tilted:?}");
        // The two rows under the target, at the bottom of the view.
        let near: Vec<_> = (tilted.iter())
            .filter(|(t, _)| t.index.z == 11 && t.index.row >= 728)
            .collect();
        assert!(
            near.len() >= 4 && near.iter().all(|&&(_, s)| s == 0),
            "{tilted:?}"
        );
        // Twice the physical pixels: no slot smaller, some larger.
        let sharp = shrinks_at(11.5, 70.0, 2.0);
        assert!(
            sharp
                .iter()
                .zip(&tilted)
                .all(|(a, b)| a.0 == b.0 && a.1 <= b.1)
        );
        assert!(sharp.iter().zip(&tilted).any(|(a, b)| a.1 < b.1));
    }

    /// A tile the cover splits shows its part of its parent's slot until
    /// a frame's budget draws it, one a frame; then each moves to the slot
    /// of its place within the budget too, and then none is left.
    #[test]
    fn split_tiles_show_their_parents_slot_until_the_budget_draws_them() {
        let tms = tms();
        let mercator = *tms.projection();
        let camera =
            orb_camera::CameraState::new(orb_crs::Geodetic::new(7.75, 46.02), 11.5, &mercator);
        let viewport = orb_camera::Viewport {
            width: 1024.0,
            height: 768.0,
        };
        let view = View::new(&camera, viewport, &orb_crs::Surface::plane(mercator));
        let layout = DrapeLayout {
            stacks: [1, 0],
            side: 8,
            tile_slots: 6,
            generation: 1,
        };
        let mut terrain = terrain_with(&[]);
        let covered = |index| CoveredTile { index, wrap: 0 };
        let parent = TileIndex::new(11, 1067, 727);
        let children: Vec<CoveredTile> = (0..4)
            .map(|i| covered(TileIndex::new(12, 2 * 1067 + i % 2, 2 * 727 + i / 2)))
            .collect();
        // A tile a frame: what a frame redraws may pass its budget by one.
        let budget = Some((1, usize::MAX));
        let plan = |terrain: &mut Terrain| {
            terrain.release_slots();
            terrain.plan(
                &tms,
                (&view, 1.0),
                (layout, budget),
                &|_| vec![1],
                &mut |_| None,
            )
        };
        terrain.tiles = vec![covered(parent)];
        let (_, redraw, deferred) = plan(&mut terrain);
        assert_eq!((redraw.len(), deferred), (1, false));
        let parent_slot = redraw[0].slot;
        terrain.tiles = children.clone();
        for frame in 0..4 {
            let (tiles, redraw, deferred) = plan(&mut terrain);
            assert_eq!((redraw.len(), deferred), (1, frame < 3), "frame {frame}");
            // The others show their quarter of the parent's slot.
            let standing = (tiles.iter())
                .filter(|t| t.within.1 == 0.5)
                .collect::<Vec<_>>();
            assert_eq!(standing.len(), 3 - frame, "frame {frame}");
            assert!(standing.iter().all(|t| t.slot == parent_slot));
        }
        assert!(terrain.retired.is_empty());
        // The tiles out of place move one a frame.
        let mut frames = 0;
        while !terrain.slots_follow_cover() {
            let (_, redraw, _) = plan(&mut terrain);
            assert!(redraw.len() <= 1);
            frames += 1;
            assert!(frames < 10, "the tiles never took their places");
        }
        let (_, redraw, deferred) = plan(&mut terrain);
        assert_eq!((redraw.len(), deferred), (0, false));
    }

    /// Heights the app writes over resident DEM tiles cull the cover as a
    /// terrain that loaded them does: the ranges, and so the tiles kept,
    /// follow the new heights, not the ones the tiles arrived with.
    #[test]
    fn heights_written_over_cull_as_loaded_ones_do() {
        let tms = tms();
        let mercator = *tms.projection();
        let mut camera =
            orb_camera::CameraState::new(orb_crs::Geodetic::new(7.75, 46.02), 11.0, &mercator);
        camera.pitch = 72.0;
        let viewport = orb_camera::Viewport {
            width: 1024.0,
            height: 768.0,
        };
        let view = View::new(&camera, viewport, &orb_crs::Surface::plane(mercator));
        let grid = |metres: f32| Arc::new(DemGrid::from_elevations(16, vec![metres; 256]).unwrap());
        let (low, high) = (grid(50.0), grid(5200.0));
        // Covers, and loads what each cover asks for, until it asks no more.
        let converge = |terrain: &mut Terrain, heights: &Arc<DemGrid>| {
            for _ in 0..32 {
                terrain.elevation(&tms);
                let asked = terrain.update((&view, &tms, camera.target), 1024);
                if asked.is_empty() {
                    break;
                }
                for tile in asked {
                    let layer = terrain.pyramid.loading() as u32 + 1000 * u32::from(tile.z);
                    let dem = Arc::new(Dem::new(Arc::clone(heights)));
                    terrain.pyramid.insert(tile, DemTile { layer, dem });
                    terrain.stale = true;
                }
            }
            terrain.elevation(&tms);
            terrain.update((&view, &tms, camera.target), 1024);
            terrain.tiles.clone()
        };
        let mut fresh = terrain_with(&[]);
        let tall = converge(&mut fresh, &high);
        let mut moved = terrain_with(&[]);
        let flat = converge(&mut moved, &low);
        assert_ne!(
            flat, tall,
            "the heights don't change the cover: the test shows nothing"
        );
        let resident: Vec<(TileIndex, u32)> = (moved.pyramid.ready())
            .map(|(tile, dem)| (tile, dem.layer))
            .collect();
        for (tile, layer) in resident {
            assert!(moved.set_heights(tile, layer, &high));
        }
        assert_eq!(converge(&mut moved, &high), tall);
        let range = |t: &mut Terrain| {
            let rect = tms.tile_bounds(TileIndex::new(9, 267, 180));
            t.elevation(&tms).range(rect, 9)
        };
        assert_eq!(range(&mut moved), range(&mut fresh));
    }

    /// A drawn tile's regions reach a terrain tile's slots only where they
    /// overlap it: a coarse tile over two terrain tiles draws its west
    /// half's region into the west one alone.
    #[test]
    fn a_drawn_region_reaches_the_slots_it_overlaps() {
        let tms = tms();
        let draw = Draw {
            tile: TileIndex::new(2, 1, 1),
            wrap: 0,
            regions: vec![TileIndex::new(3, 2, 2), TileIndex::new(3, 3, 3)],
        };
        let drawn = Drawn::of(&tms, &draw, None, 0.0);
        let at = |z, col, row| CoveredTile {
            index: TileIndex::new(z, col, row),
            wrap: 0,
        };
        let north_west = drawn.region(&tms, &draw, draw.regions[0]);
        assert_eq!(north_west.bounds, tms.tile_bounds(draw.regions[0]));
        let zoom = tms.style_zoom(TileIndex::new(3, 0, 0));
        assert!(north_west.overlaps(&tms, at(3, 2, 2), (zoom, 0.0)));
        assert!(!north_west.overlaps(&tms, at(3, 3, 3), (zoom, 0.0)));
        // Within reach of a translation, it comes into the neighbour's.
        assert!(north_west.overlaps(&tms, at(3, 3, 2), (zoom, 1.0)));
        // A translation's reach is in pixels at the zoom the slots are
        // drawn at, the camera's: past the neighbour, 512 pixels away, 600
        // reach at the tiles' zoom and 300 a zoom out; 600 a zoom in don't.
        let beyond = at(3, 4, 2);
        assert!(north_west.overlaps(&tms, beyond, (zoom, 600.0)));
        assert!(north_west.overlaps(&tms, beyond, (zoom - 1.0, 300.0)));
        assert!(!north_west.overlaps(&tms, beyond, (zoom + 1.0, 600.0)));
    }
}
