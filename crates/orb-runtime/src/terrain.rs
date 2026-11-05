//! Terrain on the scene side (plan §9.4, decision 0019): the DEM tiles the
//! terrain stands on, its cover of terrain tiles, their drape slots, and
//! the CPU elevations the camera, cover and queries read.
//!
//! Terrain tiles are 512 logical pixels at their zoom, covered as GL JS's
//! terrain covers; each samples the DEM tile at its zoom less `delta`, so
//! that its 128-cell grid has one vertex per DEM pixel: GL JS's density
//! for 256- and 512-pixel DEM tiles alike. A terrain tile redraws its
//! drape slots when the draped content under it changes, and when only
//! the zoom changed once the camera has stopped, as GL JS's render to
//! texture does.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use glam::DVec2;
use orb_camera::View;
use orb_crs::{CoveredTile, TileIndex, TileMatrixSet, WorldRect};
use orb_render::{
    ClipRect, DRAPE_SLOT_PX, DemRef, DrapeLayout, PageId, PlacementTarget, TerrainDem, TerrainTile,
    TileRow,
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

/// A terrain tile's drape slots and what they were drawn with.
#[derive(Debug)]
struct Slot {
    index: u32,
    content: u64,
    /// The zoom and pattern cross-fade they were drawn at.
    look: [u64; 3],
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
    min_zoom: u8,
    max_zoom: u8,
    /// This frame's terrain tiles.
    pub(crate) tiles: Vec<CoveredTile>,
    slots: HashMap<CoveredTile, Slot>,
    free: Vec<u32>,
    next_slot: u32,
    generation: u64,
    elevation: Arc<TerrainElevation>,
    /// DEM tiles arrived or left since the elevations were last taken.
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
            min_zoom,
            max_zoom,
            tiles: Vec::new(),
            slots: HashMap::new(),
            free: Vec::new(),
            next_slot: 0,
            generation: 0,
            elevation: Arc::new(TerrainElevation::empty(tms.clone(), exaggeration)),
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

    /// The ready DEM tile a tile at `tile` reads, or its nearest ready
    /// ancestor while it loads.
    fn dem_for(&self, tile: TileIndex) -> Option<(TileIndex, &DemTile)> {
        let mut at = scaled(tile, self.dem_zoom(tile.z)?);
        loop {
            if let Some(dem) = self.pyramid.get(at) {
                return Some((at, dem));
            }
            if at.z <= self.min_zoom {
                return None;
            }
            at = at.parent()?;
        }
    }

    /// Covers the view, asks for the DEM tiles missing and returns the
    /// tiles to request.
    pub(crate) fn update(&mut self, view: &View, tms: &TileMatrixSet) -> Vec<TileIndex> {
        let tiles = view.cover_volume(
            tms,
            TERRAIN_TILE_PX,
            TERRAIN_MAX_ZOOM,
            0.0,
            Some(&*self.elevation),
        );
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
        let update = self.pyramid.update(&wanted);
        update.request
    }

    /// A DEM tile arrived: its page layer and elevations.
    pub(crate) fn insert(&mut self, tile: TileIndex, dem: DemTile) -> Option<DemTile> {
        self.stale = true;
        self.pyramid.insert(tile, dem)
    }

    pub(crate) fn evict(&mut self) -> Vec<DemTile> {
        let evicted = self.pyramid.evict();
        if !evicted.is_empty() {
            self.stale = true;
        }
        evicted
    }

    /// The CPU elevations, taken again when DEM tiles arrived or left.
    pub(crate) fn elevation(&mut self, tms: &TileMatrixSet) -> &Arc<TerrainElevation> {
        if std::mem::take(&mut self.stale) {
            let dems = self
                .pyramid
                .ready()
                .map(|(tile, dem)| (tile, Arc::clone(&dem.dem)))
                .collect();
            let mut elevation = TerrainElevation::empty(tms.clone(), self.exaggeration);
            elevation.dems = dems;
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

    /// This frame's terrain tiles for the renderer, with slots assigned
    /// from `layout`, and which tile slots to redraw. `content` hashes the
    /// draped content under a terrain tile; `look` is the zoom, bearing and
    /// cross-fade it draws at. A tile whose either changed redraws in this
    /// frame. On a globe, `globe` places a tile on the sphere.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan(
        &mut self,
        tms: &TileMatrixSet,
        view: &View,
        layout: DrapeLayout,
        content: &dyn Fn(CoveredTile) -> u64,
        look: [u64; 3],
        globe: &mut dyn FnMut(TileIndex) -> Option<orb_render::TerrainGlobe>,
    ) -> (Vec<TerrainTile>, Vec<(CoveredTile, u32)>) {
        if layout.generation != self.generation {
            self.generation = layout.generation;
            self.slots.clear();
            self.free.clear();
            self.next_slot = 0;
        }
        let keep: HashSet<CoveredTile> = self.tiles.iter().copied().collect();
        let free = &mut self.free;
        self.slots.retain(|tile, slot| {
            let kept = keep.contains(tile);
            if !kept {
                free.push(slot.index);
            }
            kept
        });
        let period = tms.projection().wrap_period().unwrap_or(0.0);
        let mut out = Vec::with_capacity(self.tiles.len());
        let mut redraw = Vec::new();
        for &tile in &self.tiles {
            let content = content(tile);
            let slot = match self.slots.get_mut(&tile) {
                Some(slot) => {
                    if slot.content != content || slot.look != look {
                        slot.content = content;
                        slot.look = look;
                        redraw.push((tile, slot.index));
                    }
                    slot.index
                }
                None => {
                    let index = self.free.pop().unwrap_or_else(|| {
                        self.next_slot += 1;
                        self.next_slot - 1
                    });
                    self.slots.insert(
                        tile,
                        Slot {
                            index,
                            content,
                            look,
                        },
                    );
                    redraw.push((tile, index));
                    index
                }
            };
            let bounds = tms.tile_bounds(tile.index);
            let origin = DVec2::new(bounds.min.x + f64::from(tile.wrap) * period, bounds.max.y)
                - view.camera_world();
            let dem = self.dem_for(tile.index).map(|(dem_tile, _)| {
                let depth = tile.index.z - dem_tile.z;
                let scale = 0.5f32.powi(i32::from(depth));
                let d = |v: u32| (v - (v >> depth << depth)) as f32 * scale;
                TerrainDem {
                    layer: self.pyramid.get(dem_tile).expect("ready").layer,
                    offset: [d(tile.index.col), d(tile.index.row)],
                    scale,
                }
            });
            out.push(TerrainTile {
                origin: [origin.x as f32, origin.y as f32],
                scale: (tms.tile_span(tile.index.z) / f64::from(orb_render::TERRAIN_EXTENT)) as f32,
                dem,
                slot,
                globe: globe(tile.index),
            });
        }
        (out, redraw)
    }

    /// Tile slots in use: the atlas must hold this many.
    pub(crate) fn slots_needed(&self) -> u32 {
        self.next_slot.max(self.tiles.len() as u32)
    }
}

/// `tile`'s ancestor (or itself) at zoom `z ≤ tile.z`.
fn scaled(tile: TileIndex, z: u8) -> TileIndex {
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

    /// Whether it overlaps a terrain tile of the surface's matrix set.
    pub(crate) fn overlaps(&self, surface: &TileMatrixSet, terrain: CoveredTile) -> bool {
        let (a, b) = (self.bounds, terrain_bounds(surface, terrain));
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
/// by its map placements `map_rows`) for a terrain tile and its tile slot:
/// one row set per drape stack, each mapping the tile's data into the
/// stack's slot and clipped to the terrain tile.
pub(crate) fn drape_rows(
    surface: &TileMatrixSet,
    drawn: Drawn,
    extent: u32,
    (terrain, slot): (CoveredTile, u32),
    layout: &DrapeLayout,
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
    let (lo, hi) = ([clamp(lo.x), clamp(lo.y)], [clamp(hi.x), clamp(hi.y)]);
    let k = f64::from(DRAPE_SLOT_PX) / span;
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
        for stack in 0..layout.stacks {
            let [cx, cy] = layout.corner(slot, stack);
            let corner = DVec2::new(f64::from(cx), -f64::from(cy));
            let o = corner + (origin - t_origin) * k;
            out.push(TileRow {
                origin: [o.x as f32, o.y as f32],
                scale: (scale * k) as f32,
                clips,
                world_per_px: (f64::from(row.world_per_px) * k) as f32,
                target: PlacementTarget::Drape(stack as u8),
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
/// the camera and queries hold while the map moves on (plan §9.4). It
/// follows DEM arrival and eviction, never the view.
#[derive(Debug)]
pub struct TerrainElevation {
    tms: TileMatrixSet,
    exaggeration: f64,
    delta: u8,
    min_zoom: u8,
    max_zoom: u8,
    dems: HashMap<TileIndex, Arc<Dem>>,
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
            encoded: OnceLock::new(),
        }
    }

    /// Bytes of DEM data held.
    pub fn bytes(&self) -> usize {
        self.dems.values().map(|d| d.grid.bytes()).sum()
    }

    /// The tile at zoom `z` containing a world position in the primary
    /// world copy, with the position's uv in it.
    fn locate(&self, world: DVec2, z: u8) -> Option<(TileIndex, DVec2)> {
        let bounds = self.tms.projection().world_bounds();
        let span = self.tms.tile_span(z);
        let x = (world.x - bounds.min.x) / span;
        let y = (bounds.max.y - world.y) / span;
        let (cols, rows) = self.tms.matrix_size(z);
        if !(0.0..f64::from(rows)).contains(&y) || !(0.0..f64::from(cols)).contains(&x) {
            return None;
        }
        let tile = TileIndex::new(z, x as u32, y as u32);
        Some((tile, DVec2::new(x.fract(), y.fract())))
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
        let mut at = scaled(tile, self.dem_zoom(tile.z)?);
        loop {
            if let Some(dem) = self.dems.get(&at) {
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
    /// The elevation pyramid (plan §7.1): these elevations with each DEM
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
        let (tile, _) = self.locate(self.wrapped(rect.center()), z)?;
        let (_, grid) = self.dem_for(tile)?;
        let e = self.exaggeration;
        Some((f64::from(grid.min()) * e, f64::from(grid.max()) * e))
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

    /// DEM tiles requested and arrived.
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
        terrain.pyramid.update(&wanted);
        for (layer, (tile, dem)) in dems.iter().enumerate() {
            let tile_dem = DemTile {
                layer: layer as u32,
                dem: Arc::clone(dem),
            };
            assert!(terrain.insert(*tile, tile_dem).is_none());
        }
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
        assert!(drawn.overlaps(&tms, terrain));
        let row = TileRow {
            origin: [0.0; 2],
            scale: 1.0,
            zoom: 4.0,
            clips: [
                ClipRect::new([0, 0], [4096, 4096]),
                ClipRect::EMPTY,
                ClipRect::EMPTY,
                ClipRect::EMPTY,
            ],
            pattern_origin: [0; 2],
            extent: 4096.0,
            wrap: 0,
            world_per_px: 1.0,
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
            &layout,
            [row].into_iter(),
            &mut rows,
        );
        assert_eq!(rows.len(), 2);
        let slot = DRAPE_SLOT_PX as f32;
        // Slot 1, stack 0 is atlas slot 2: column 2. The tile's top-left
        // is the slot's centre; its 4096 units span half a slot.
        assert_eq!(rows[0].origin, [2.5 * slot, -0.5 * slot]);
        assert_eq!(rows[0].scale, slot / 2.0 / 4096.0);
        assert_eq!(
            rows[0].world_per_px,
            (f64::from(DRAPE_SLOT_PX) / tms.tile_span(3)) as f32
        );
        assert_eq!(rows[1].target, PlacementTarget::Drape(1));
        assert_eq!(rows[1].origin[0], 3.5 * slot);
        // Clipped to the terrain tile: the whole vector tile is inside.
        assert_eq!(rows[0].clips[0], ClipRect::new([0, 0], [4096, 4096]));
        // A neighbour outside contributes nothing.
        let away = Draw {
            tile: TileIndex::new(4, 7, 5),
            wrap: 0,
            regions: vec![TileIndex::new(4, 7, 5)],
        };
        assert!(!Drawn::of(&tms, &away, None, 0.0).overlaps(&tms, terrain));
    }
}
