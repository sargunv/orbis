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
//! DEM. A terrain tile's drape slots
//! are drawn at a zoom of its own ([`drape_zoom`]), whatever the camera's,
//! and redrawn only when the draped content under it changes.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use glam::DVec2;
use orb_camera::View;
use orb_crs::{CoveredTile, TileIndex, TileMatrixSet, WorldRect};
use orb_hal::Gpu;
use orb_render::{
    ClipRect, DRAPE_SLOT_PX, DemRef, DrapeLayout, PageId, PlacementTarget, Renderer, TerrainDem,
    TerrainTile, TileRow,
};
use orb_source::{Draw, TilePyramid};
use orb_tile::DemGrid;

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

/// A terrain tile's drape slots and the content each stack's was drawn
/// with.
#[derive(Debug)]
struct Slot {
    index: u32,
    content: Vec<u64>,
}

/// A tile slot to redraw: its terrain tile, tile slot, and the stacks
/// whose content changed.
pub(crate) type Redraw = (CoveredTile, u32, Vec<u8>);

/// The style zoom a terrain tile's drape slots are drawn at, whatever the
/// camera's: the tile's own, where it spans its 512 logical pixels, at
/// [`DRAPE_SLOT_PX`] texels (GL JS's quality factor 2).
pub(crate) fn drape_zoom(tms: &TileMatrixSet, tile: TileIndex) -> u8 {
    tms.style_zoom(tile).round().clamp(0.0, 255.0) as u8
}

#[derive(Debug)]
pub(crate) struct Terrain {
    /// Unique among the map's terrains, for revisions.
    pub(crate) id: u64,
    /// The DEM source's index in the plan and its key.
    pub(crate) source: usize,
    pub(crate) key: u32,
    pub(crate) exaggeration: f64,
    pub(crate) pyramid: TilePyramid<DemTile>,
    /// The DEM tiles' page, made for the first.
    pub(crate) page: Option<PageId>,
    pub(crate) loading: u32,
    /// DEM zooms below a terrain tile's.
    delta: u8,
    /// The DEM tiles' size in logical pixels.
    tile_px: f64,
    min_zoom: u8,
    max_zoom: u8,
    /// This frame's terrain tiles.
    pub(crate) tiles: Vec<CoveredTile>,
    slots: HashMap<CoveredTile, Slot>,
    generation: u64,
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
    /// Per wanted DEM tile, its layer and the neighbours' layers its border
    /// was last padded from ([`pads`](Self::pads)).
    padded: HashMap<TileIndex, (u32, [Option<u32>; 8])>,
    /// DEM tiles arrived, left or are wanted anew since the elevations were
    /// last taken.
    stale: bool,
    pub(crate) revision: u64,
}

impl Terrain {
    /// Terrain over a `raster-dem` source of `tile_px` tiles between
    /// `min_zoom` and `max_zoom`.
    pub(crate) fn new(
        id: u64,
        source: usize,
        key: u32,
        exaggeration: f64,
        tms: &TileMatrixSet,
        (tile_px, min_zoom, max_zoom): (f64, u8, u8),
    ) -> Self {
        let delta = 1 + (tile_px / 256.0).log2().round().max(0.0) as u8;
        Self {
            id,
            source,
            key,
            exaggeration,
            pyramid: TilePyramid::new(min_zoom, max_zoom, DEM_CACHE),
            page: None,
            loading: 0,
            delta,
            tile_px,
            min_zoom,
            max_zoom,
            tiles: Vec::new(),
            slots: HashMap::new(),
            generation: 0,
            elevation: Arc::new(TerrainElevation::empty(tms.clone(), exaggeration)),
            wanted: Vec::new(),
            culling: Vec::new(),
            padded: HashMap::new(),
            stale: true,
            revision: 0,
        }
    }

    /// The DEM zoom a tile at zoom `z` reads: GL JS's `_getSourceZoom`,
    /// never deeper than the tile.
    fn dem_zoom(&self, z: u8) -> Option<u8> {
        let zd = z
            .saturating_sub(self.delta)
            .min(self.max_zoom)
            .max(self.min_zoom);
        (zd <= z).then_some(zd)
    }

    /// The ready DEM tile standing for DEM tile `tile`: itself, or its
    /// nearest ready ancestor while it loads.
    fn ready(&self, tile: TileIndex) -> Option<TileIndex> {
        let mut at = tile;
        loop {
            if self.pyramid.get(at).is_some() {
                return Some(at);
            }
            if at.z <= self.min_zoom {
                return None;
            }
            at = at.parent()?;
        }
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
        let mut at = scaled(tile, self.dem_zoom(tile.z)?);
        loop {
            if self.elevation.dems.contains_key(&at) {
                return Some((at, self.pyramid.get(at)?));
            }
            if at.z <= self.min_zoom {
                return None;
            }
            at = at.parent()?;
        }
    }

    /// Covers the view with at most `budget` tiles, the drape atlas's
    /// room, asks for the DEM tiles missing and returns the tiles to
    /// request.
    ///
    /// The cover culls tiles by the elevations held, and what is held
    /// follows from the cover, so a DEM tile whose range culls a tile is
    /// kept too ([`culling`](Self::culling)). Otherwise a tile its own DEM
    /// culls lets go of that DEM, comes back under its ancestor's wider
    /// range and asks for it again: a view at rest would flip between two
    /// covers forever.
    fn update(&mut self, view: &View, tms: &TileMatrixSet, budget: usize) -> Vec<TileIndex> {
        let consulted = Consulted {
            elevation: &self.elevation,
            ranges: Mutex::new(Vec::new()),
        };
        let mut tiles = view.cover_volume(
            tms,
            TERRAIN_TILE_PX,
            TERRAIN_MAX_ZOOM,
            0.0,
            Some(&consulted),
            0.0,
        );
        let ranges = consulted.ranges.into_inner().expect("ranges poisoned");
        // The tiles the cover drew or refined: any other tile it read a
        // range for, that range culled.
        let mut kept: HashSet<TileIndex> = tiles.iter().map(|t| t.index).collect();
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
        coarsen(&mut tiles, budget, |t| {
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
                    index: scaled(t.index, self.dem_zoom(t.index.z)?),
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
            .and_then(|z| self.dem_zoom(z))
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
        let update = self.pyramid.update(&wanted);
        update.request
    }

    /// The prepares that pad the DEM tiles this camera reads (those it
    /// wants, and the ancestors drawn while they load) from their ready
    /// neighbours among them, for the tiles whose neighbours changed since:
    /// a border is a function of what this camera reads, not of which tiles
    /// were resident before.
    fn pads(&mut self, unpack: [f32; 4]) -> Vec<orb_render::DemPrepare> {
        let read = self.read();
        let pyramid = &self.pyramid;
        let layer_of = |t: TileIndex| {
            read.binary_search(&t)
                .ok()
                .and_then(|_| pyramid.get(t))
                .map(|d| d.layer)
        };
        let mut jobs = Vec::new();
        let mut padded = HashMap::new();
        for &tile in &read {
            let Some(dem) = pyramid.get(tile) else {
                continue;
            };
            let now = (dem.layer, orb_render::dem_neighbours(tile, &layer_of));
            if self.padded.get(&tile) != Some(&now) {
                jobs.push(orb_render::DemPrepare {
                    layer: now.0,
                    neighbours: now.1,
                    zoom: f32::from(tile.z),
                    unpack,
                });
            }
            padded.insert(tile, now);
        }
        self.padded = padded;
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
        (view, tms): (&View, &TileMatrixSet),
        unpack: [f32; 4],
        skirt: f64,
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
        let requests = self.update(view, tms, renderer.drape_room(gpu) as usize);
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
        let frame = orb_render::FrameTerrain {
            unpack,
            exaggeration,
            dem_dim,
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
        result: Result<crate::TileResult, String>,
        now: f64,
    ) -> Result<(), String> {
        self.loading = self.loading.saturating_sub(1);
        if !self.pyramid.is_loading(tile) {
            return Ok(());
        }
        let (image, grid) = match result {
            Ok(crate::TileResult { missing: true, .. }) => {
                self.pyramid.insert_missing(tile);
                return Ok(());
            }
            Ok(crate::TileResult {
                image: Some(image),
                dem: Some(dem),
                ..
            }) => (image, dem),
            Ok(_) | Err(_) => {
                self.pyramid.insert_failed(tile, now);
                return Err(result.err().unwrap_or("no elevations".into()));
            }
        };
        let size = [image.width, image.height];
        let page = *self
            .page
            .get_or_insert_with(|| renderer.create_page(gpu, orb_render::PageKind::Dem, size));
        let layer = match renderer.page_size(page) {
            other if other != size => Err(format!("{}×{} among {other:?} tiles", size[0], size[1])),
            _ => renderer
                .alloc_layer(gpu, page)
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
            let held = |tiles: Vec<TileIndex>| -> HashMap<TileIndex, Arc<Dem>> {
                (tiles.into_iter())
                    .filter_map(|t| Some((t, Arc::clone(&self.pyramid.get(t)?.dem))))
                    .collect()
            };
            let dems = held(self.read());
            let culling = self.culling.iter().filter_map(|&t| self.ready(t));
            let mut culling = held(culling.collect());
            culling.retain(|t, _| !dems.contains_key(t));
            let same = |a: &HashMap<TileIndex, Arc<Dem>>, b: &HashMap<TileIndex, Arc<Dem>>| {
                a.len() == b.len()
                    && a.iter()
                        .all(|(t, d)| b.get(t).is_some_and(|h| Arc::ptr_eq(h, d)))
            };
            // The same tiles as before: the same elevations, no new revision.
            if same(&dems, &self.elevation.dems) && same(&culling, &self.elevation.culling) {
                return &self.elevation;
            }
            let mut elevation = TerrainElevation::empty(tms.clone(), self.exaggeration);
            elevation.dems = dems;
            elevation.culling = culling;
            elevation.delta = self.delta;
            elevation.min_zoom = self.min_zoom;
            elevation.max_zoom = self.max_zoom;
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

    /// Frees the slots of tiles this frame doesn't cover; returns the
    /// tile slots kept, which the atlas carries over.
    pub(crate) fn kept_slots(&mut self) -> Vec<u32> {
        let keep: HashSet<CoveredTile> = self.tiles.iter().copied().collect();
        self.slots.retain(|tile, _| keep.contains(tile));
        self.slots.values().map(|s| s.index).collect()
    }

    /// This frame's terrain tiles for the renderer, with slots assigned
    /// from `layout` after the atlas's `moves`, and which tile slots'
    /// stacks to redraw: those whose draped content, as `content` hashes
    /// it per stack, changed. On a globe, `globe` places a tile on the
    /// sphere.
    pub(crate) fn plan(
        &mut self,
        tms: &TileMatrixSet,
        view: &View,
        (layout, moves): (DrapeLayout, &[(u32, u32)]),
        content: &dyn Fn(CoveredTile) -> Vec<u64>,
        globe: &mut dyn FnMut(TileIndex) -> Option<orb_render::TerrainGlobe>,
    ) -> (Vec<TerrainTile>, Vec<Redraw>) {
        if layout.generation != self.generation {
            self.generation = layout.generation;
            self.slots.clear();
        }
        for slot in self.slots.values_mut() {
            if let Some(&(_, to)) = moves.iter().find(|m| m.0 == slot.index) {
                slot.index = to;
            }
        }
        let used: HashSet<u32> = self.slots.values().map(|s| s.index).collect();
        let mut free = (0..layout.tile_slots).filter(|t| !used.contains(t));
        let period = tms.projection().wrap_period().unwrap_or(0.0);
        let mut out = Vec::with_capacity(self.tiles.len());
        let mut redraw = Vec::new();
        for &tile in &self.tiles {
            let content = content(tile);
            let slot = match self.slots.get_mut(&tile) {
                Some(slot) => {
                    let changed: Vec<u8> = (0..content.len())
                        .filter(|&s| slot.content.get(s) != content.get(s))
                        .map(|s| s as u8)
                        .collect();
                    if !changed.is_empty() {
                        slot.content = content;
                        redraw.push((tile, slot.index, changed));
                    }
                    slot.index
                }
                None => {
                    let index = free.next().expect("the atlas holds every tile");
                    let every = (0..content.len() as u8).collect();
                    self.slots.insert(tile, Slot { index, content });
                    redraw.push((tile, index, every));
                    index
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
                slot,
                globe: globe(tile.index),
                latitudes: orb_render::tile_latitudes(tile.index),
            });
        }
        (out, redraw)
    }
}

/// Where `tile` lies in its ancestor (or itself) `at`: tile uv → `at`'s uv
/// as `uv · scale + offset`.
fn within(tile: TileIndex, at: TileIndex) -> ([f32; 2], f32) {
    let depth = tile.z - at.z;
    let scale = 0.5f32.powi(i32::from(depth));
    let d = |v: u32| (v - (v >> depth << depth)) as f32 * scale;
    ([d(tile.col), d(tile.row)], scale)
}

/// Coarsens a cover until it has at most `budget` tiles (or no tiles
/// share an ancestor): the tiles under the farthest ancestor of two or
/// more, by `distance` from the eye, become that ancestor, as distance
/// already coarsens the cover.
fn coarsen(tiles: &mut Vec<CoveredTile>, budget: usize, distance: impl Fn(CoveredTile) -> f64) {
    while tiles.len() > budget {
        let mut under: HashMap<CoveredTile, usize> = HashMap::new();
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

    /// Whether it overlaps a terrain tile of the surface's matrix set, or
    /// comes within `reach_px` of it, in the tile's logical pixels: what a
    /// translation that far moves into the tile's slots.
    pub(crate) fn overlaps(
        &self,
        surface: &TileMatrixSet,
        terrain: CoveredTile,
        reach_px: f64,
    ) -> bool {
        let b = terrain_bounds(surface, terrain);
        let reach = DVec2::splat(reach_px * b.size().x / TERRAIN_TILE_PX);
        let b = WorldRect {
            min: b.min - reach,
            max: b.max + reach,
        };
        let a = self.bounds;
        let eps = 1e-9 * surface.tile_span(0);
        a.min.x < b.max.x - eps
            && b.min.x < a.max.x - eps
            && a.min.y < b.max.y - eps
            && b.min.y < a.max.y - eps
    }
}

/// A terrain tile's bounds in its world copy.
fn terrain_bounds(surface: &TileMatrixSet, terrain: CoveredTile) -> WorldRect {
    let period = surface.projection().wrap_period().unwrap_or(0.0);
    surface
        .tile_bounds(terrain.index)
        .translated(DVec2::new(f64::from(terrain.wrap) * period, 0.0))
}

/// Drape placements of a drawn tile (`extent` data units a side, mapped
/// by its map placements `map_rows`, placed at the terrain tile's
/// [`drape_zoom`]) for a terrain tile and its tile slot: one row set per
/// drape stack of `stacks` (those its source's layers draw into), each
/// mapping the tile's data into the stack's slot,
/// clipped to the terrain tile and the `reach_px` (logical pixels) a
/// translation moves content into it from, bounded by the terrain tile,
/// with pixel sizes in slot texels at the drape zoom.
#[expect(clippy::too_many_arguments, reason = "one placement's terms")]
pub(crate) fn drape_rows(
    surface: &TileMatrixSet,
    drawn: Drawn,
    extent: u32,
    (terrain, slot): (CoveredTile, u32),
    (layout, stacks): (&DrapeLayout, &[u8]),
    reach_px: f64,
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
    let reach = DVec2::splat(reach_px * span / TERRAIN_TILE_PX / scale);
    let (lo, hi) = (lo - reach, hi + reach);
    let (lo, hi) = ([clamp(lo.x), clamp(lo.y)], [clamp(hi.x), clamp(hi.y)]);
    let k = f64::from(DRAPE_SLOT_PX) / span;
    let zoom = drape_zoom(surface, terrain.index);
    // Texels per logical pixel at the drape zoom: the slot spans the 512
    // logical pixels of the terrain tile at its own zoom.
    let px_scale = f64::from(DRAPE_SLOT_PX)
        / (TERRAIN_TILE_PX * (f64::from(zoom) - surface.style_zoom(terrain.index)).exp2());
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
            let o = corner + (origin - t_origin) * k;
            out.push(TileRow {
                origin: o.to_array(),
                scale: (scale * k) as f32,
                clips,
                bound,
                px_scale: px_scale as f32,
                drape_zoom: f32::from(zoom),
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

/// The terrain's elevations on the CPU: a snapshot of the ready DEM tiles
/// the camera and queries hold while the map moves on. It
/// follows DEM arrival and eviction, never the view.
#[derive(Debug)]
pub struct TerrainElevation {
    tms: TileMatrixSet,
    exaggeration: f64,
    delta: u8,
    min_zoom: u8,
    max_zoom: u8,
    dems: HashMap<TileIndex, Arc<Dem>>,
    /// DEM tiles the cover reads ranges from and nothing else
    /// (the terrain's `culling`).
    culling: HashMap<TileIndex, Arc<Dem>>,
    /// The pyramid, encoded once for every reader of this snapshot.
    encoded: OnceLock<Arc<[u8]>>,
}

impl TerrainElevation {
    fn empty(tms: TileMatrixSet, exaggeration: f64) -> Self {
        Self {
            tms,
            exaggeration,
            delta: 1,
            min_zoom: 0,
            max_zoom: 0,
            dems: HashMap::new(),
            culling: HashMap::new(),
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

    fn dem_zoom(&self, z: u8) -> Option<u8> {
        let zd = z
            .saturating_sub(self.delta)
            .min(self.max_zoom)
            .max(self.min_zoom);
        (zd <= z).then_some(zd)
    }

    /// The DEM tile `tile` reads, or its nearest loaded ancestor.
    fn dem_for(&self, tile: TileIndex) -> Option<(TileIndex, &DemGrid)> {
        self.held_for(tile, false)
    }

    /// [`dem_for`](Self::dem_for) among the tiles held, with `culling`
    /// those held for their ranges too.
    fn held_for(&self, tile: TileIndex, culling: bool) -> Option<(TileIndex, &DemGrid)> {
        let mut at = scaled(tile, self.dem_zoom(tile.z)?);
        loop {
            let dem =
                (self.dems.get(&at)).or_else(|| culling.then(|| self.culling.get(&at)).flatten());
            if let Some(dem) = dem {
                return Some((at, &dem.grid));
            }
            if at.z <= self.min_zoom {
                return None;
            }
            at = at.parent()?;
        }
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
            u32::from(self.delta),
            u32::from(self.min_zoom),
            u32::from(self.max_zoom),
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
        let tms = TileMatrixSet::by_id(&id).map_err(|e| format!("elevation pyramid: {e:?}"))?;
        let mut elevation = Self::empty(tms, exaggeration);
        elevation.delta = delta;
        elevation.min_zoom = min_zoom;
        elevation.max_zoom = max_zoom;
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
    fn at(&self, world: DVec2) -> Option<f64> {
        let world = self.wrapped(world);
        // The deepest DEM tile loaded, which the nearest terrain draws with.
        (self.min_zoom..=self.max_zoom).rev().find_map(|z| {
            let (tile, uv) = self.locate(world, z)?;
            let dem = self.dems.get(&tile)?;
            Some(self.sample(tile, &dem.grid, uv) * self.exaggeration)
        })
    }

    fn range(&self, rect: WorldRect, z: u8) -> Option<(f64, f64)> {
        self.range_of(rect, z).map(|(_, _, range)| range)
    }
}

impl TerrainElevation {
    /// The tile at zoom `z` under `rect`, the DEM tile it reads and that
    /// DEM's range of elevations.
    fn range_of(&self, rect: WorldRect, z: u8) -> Option<(TileIndex, TileIndex, (f64, f64))> {
        let (tile, _) = self.locate(self.wrapped(rect.center()), z)?;
        let (dem, grid) = self.held_for(tile, true)?;
        let e = self.exaggeration;
        Some((
            tile,
            dem,
            (f64::from(grid.min()) * e, f64::from(grid.max()) * e),
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
    fn at(&self, world: DVec2) -> Option<f64> {
        self.elevation.at(world)
    }

    fn range(&self, rect: WorldRect, z: u8) -> Option<(f64, f64)> {
        let (tile, dem, range) = self.elevation.range_of(rect, z)?;
        self.ranges
            .lock()
            .expect("ranges poisoned")
            .push((tile, dem));
        Some(range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_camera::Elevation;

    fn tms() -> TileMatrixSet {
        TileMatrixSet::by_id("WebMercatorQuad").unwrap()
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
            u32::from(e.delta),
            u32::from(e.min_zoom),
            u32::from(e.max_zoom),
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
    fn blocks(e: &TerrainElevation) -> HashMap<TileIndex, usize> {
        e.dems
            .iter()
            .filter_map(|(&t, d)| Some((t, d.coarse.get()?.as_ptr() as usize)))
            .collect()
    }

    /// A terrain over 256-pixel DEM tiles to zoom 12 with `dems` arrived.
    fn terrain_with(dems: &[(TileIndex, Arc<Dem>)]) -> Terrain {
        let mut terrain = Terrain::new(1, 0, 0, 1.5, &tms(), (256.0, 0, 12));
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
        // Under budget: as covered.
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
                e.max_zoom = 12;
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
        e.max_zoom = 12;
        e.dems.insert(TileIndex::new(10, 530, 360), hills(64, 0.0));
        assert_eq!(*e.pyramid(), *reference(&e));
    }

    #[test]
    fn elevations_follow_the_dem_pixels() {
        let tms = tms();
        let mut elevation = TerrainElevation::empty(tms.clone(), 2.0);
        elevation.delta = 1;
        elevation.max_zoom = 12;
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
        assert!((elevation.at(at(0.375, 0.5)).unwrap() - 220.0).abs() < 1e-9);
        assert_eq!(
            elevation.range(tms.tile_bounds(TileIndex::new(5, 16, 16)), 5),
            Some((200.0, 260.0))
        );
        // Nothing loaded elsewhere.
        assert!(elevation.at(at(-1.0, 0.5)).is_none());
    }

    #[test]
    fn the_pyramid_keeps_elevations_within_metres() {
        let tms = tms();
        let mut full = TerrainElevation::empty(tms.clone(), 1.5);
        full.max_zoom = 12;
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
                let (a, b) = (full.at(p).unwrap(), coarse.at(p).unwrap());
                worst = worst.max((a - b).abs());
            }
        }
        // Four pixels a sample over hills 20 pixels across.
        assert!(worst < 10.0, "{worst}");
        assert!(worst > 0.0);
    }

    #[test]
    fn drape_rows_map_tiles_into_their_slots() {
        let tms = tms();
        let layout = DrapeLayout {
            stacks: 2,
            columns: 8,
            tile_slots: 4,
            generation: 1,
        };
        let terrain = CoveredTile {
            index: TileIndex::new(3, 2, 2),
            wrap: 0,
        };
        // A vector tile one zoom deeper, in the terrain tile's south-east.
        let draw = Draw {
            tile: TileIndex::new(4, 5, 5),
            wrap: 0,
            regions: vec![TileIndex::new(4, 5, 5)],
        };
        let drawn = Drawn::of(&tms, &draw, None, 0.0);
        assert!(drawn.overlaps(&tms, terrain, 0.0));
        let row = TileRow {
            origin: [0.0; 2],
            scale: 1.0,
            zoom: 4.0,
            bound: ClipRect::EVERYWHERE,
            clips: [
                ClipRect::new([0, 0], [4096, 4096]),
                ClipRect::EMPTY,
                ClipRect::EMPTY,
                ClipRect::EMPTY,
            ],
            pattern_origin: [0; 2],
            extent: 4096.0,
            wrap: 0,
            px_scale: 1.0,
            drape_zoom: 0.0,
            dem: DemRef::NONE,
            terrain_range: [0.0; 2],
            target: PlacementTarget::Map,
            warp: orb_render::TileWarp::NONE,
            globe: None,
        };
        let mut rows = Vec::new();
        drape_rows(
            &tms,
            drawn,
            4096,
            (terrain, 1),
            (&layout, &[0, 1]),
            0.0,
            [row].into_iter(),
            &mut rows,
        );
        assert_eq!(rows.len(), 2);
        let slot = DRAPE_SLOT_PX as f32;
        // Slot 1, stack 0 is atlas slot 2: column 2. The tile's top-left
        // is the slot's centre; its 4096 units span half a slot.
        let px = f64::from(DRAPE_SLOT_PX);
        let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-9;
        assert!(near(rows[0].origin, [2.5 * px, -0.5 * px]));
        assert_eq!(rows[0].scale, slot / 2.0 / 4096.0);
        // Drawn at the terrain tile's zoom, two texels a logical pixel.
        let zoom = tms.style_zoom(terrain.index);
        assert_eq!(f64::from(rows[0].drape_zoom), zoom);
        assert_eq!(rows[0].px_scale, 2.0);
        assert_eq!(rows[1].target, PlacementTarget::Drape(1));
        assert_eq!(rows[1].origin[0], 3.5 * px);
        // Clipped to the terrain tile: the whole vector tile is inside.
        assert_eq!(rows[0].clips[0], ClipRect::new([0, 0], [4096, 4096]));
        // A neighbour outside contributes nothing.
        let away = Draw {
            tile: TileIndex::new(4, 7, 5),
            wrap: 0,
            regions: vec![TileIndex::new(4, 7, 5)],
        };
        assert!(!Drawn::of(&tms, &away, None, 0.0).overlaps(&tms, terrain, 0.0));
    }
}
