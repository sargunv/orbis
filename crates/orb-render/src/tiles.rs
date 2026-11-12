//! Tiles the renderer holds: what an upload made of a tile,
//! behind a handle the map keeps in its pyramid.
//!
//! A vector tile holds one scene tile per layer extent, its meshes in
//! them by render node, symbol meshes among them, and the planar patches
//! it draws through. The map refers to a mesh by its [`MeshId`] for the
//! edits that replace its rows and for symbol state.
//!
//! An image tile takes a layer of its source's page, which the source's
//! first tile makes and the source's image nodes bind, and a quad (or a
//! warped mesh) in each of those nodes. Image tiles are indexed by source
//! and tile: a DEM tile's arrival prepares it and the resident neighbours
//! whose borders it fills, and a raster tile cross-fades from its nearest
//! resident ancestor.

use rustc_hash::{FxHashMap, FxHashSet};
use std::ops::Range;
use std::sync::Arc;

use crate::scene::{
    MeshData, MeshId, MeshSize, OutOfSpace, Pixels, SymbolMeshData, Table, TileId, TileRow,
    TileWarp, Words,
};
use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_layout::{Chunk, TileBox};

use crate::Renderer;
use crate::image::{DemPrepare, IMAGE_EXTENT, IMAGE_ROW_WORDS, ImageFade, PageId, PageKind};

/// What uploads found no room for while the pools couldn't grow
/// ([`Renderer::take_refusals`]): per pool (vertices, indices,
/// properties) the words, and the chunk rows, that would have held them,
/// 0 where they fit; and the tile pages they wanted a layer of. A map
/// that doesn't move memory while the camera moves keeps it with what
/// waits, and makes the room at rest ([`Renderer::grow_into`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Room {
    pub(crate) words: [u32; 3],
    pub(crate) chunks: u32,
    pub(crate) pages: Vec<PageId>,
}

impl Room {
    /// Nothing was refused.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Room for both.
    pub fn merge(&mut self, other: &Room) {
        for (a, b) in self.words.iter_mut().zip(other.words) {
            *a = (*a).max(b);
        }
        self.chunks = self.chunks.max(other.chunks);
        for &page in &other.pages {
            if !self.pages.contains(&page) {
                self.pages.push(page);
            }
        }
    }
}

/// A tile the renderer holds, from an upload; [`Renderer::release_tile`]
/// frees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuTile(u32);

/// Which tile of which source; sources by their style key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey {
    pub source: u32,
    pub tile: TileIndex,
}

/// An image tile's mesh where it isn't its own square: on another plane or
/// bent by a globe. Vertices are five words each (position in tile units as
/// two f32, clip position as two u16, weight as f32, image position as two
/// u16), in [`IMAGE_EXTENT`] units.
#[derive(Debug, PartialEq)]
pub struct ImageMesh {
    pub vertices: Vec<u32>,
    pub indices: Vec<u32>,
    pub chunks: Vec<Chunk>,
}

/// An image tile to upload: raster colours or DEM elevations.
#[derive(Debug)]
pub struct ImageBuild<'a> {
    pub key: TileKey,
    pub kind: PageKind,
    pub size: [u32; 2],
    pub pixels: Pixels<'a>,
    /// Its mesh, or `None` for its square.
    pub mesh: Option<Arc<ImageMesh>>,
    /// Planar patches it draws through on a globe, as
    /// [`warp_words`](crate::warp_words) has them.
    pub patches: Option<(u32, &'a [f32])>,
    /// When it arrived, in the frame's milliseconds, if it fades in: it
    /// then cross-fades from its nearest resident ancestor, unless resident
    /// descendants covered it.
    pub fade_from: Option<f32>,
    /// A DEM tile's unpack vector.
    pub unpack: Option<[f32; 4]>,
    /// For a raster tile of a partial tile matrix that reaches past its
    /// set's extent, the texels inside it, `[x0, y0, x1, y1)`: those past
    /// it hold fill, and take the nearest inside before filtering reads
    /// them.
    pub area: Option<[u32; 4]>,
}

/// What a mesh's property rows refer to besides their words.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshRows<'a> {
    /// Image ids to patch into row words: `(word, image)`.
    pub images: &'a [(u32, u32)],
    /// Per feature slot, its feature-state index; empty for the default
    /// state.
    pub states: &'a [u32],
}

/// A tile's patches on the low-zoom surface (a sphere, or a low-zoom
/// plane), by its tile matrix set (`set`, any number that names it) and
/// index: they're shared by every placement of that tile, of any source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WarpKey {
    pub set: u64,
    pub tile: TileIndex,
}

/// Low-zoom surface patches the renderer holds, by key, and the frame each
/// was last placed in.
#[derive(Debug, Default)]
pub(crate) struct SurfaceWarps {
    warps: FxHashMap<WarpKey, (TileWarp, u64)>,
    frame: u64,
}

/// Surface patches kept before those the last frame didn't place go.
const SURFACE_WARPS: usize = 2048;

/// A render node's share of a tile: its geometry in one layer extent and
/// its property rows, which per-feature paint (a circle's radius) reads.
/// Fills, lines and extrusions are meshes; point families (circles,
/// heatmaps) are their points.
#[derive(Clone, Debug)]
pub struct MeshBuild<'a> {
    pub node: u32,
    pub extent: u32,
    /// Its vertices, indices and property rows: ranges of the tile's
    /// [`words`](TileBuild::words), empty for those it shares.
    pub spans: [Range<u32>; 3],
    /// Words per feature of the rows.
    pub row_words: u32,
    pub chunks: &'a [Chunk],
    pub rows: MeshRows<'a>,
    /// The earlier mesh of the build whose vertices, and whose indices,
    /// it draws instead of its own.
    pub shares: [Option<usize>; 2],
}

/// A tile to upload, all of it or none: its words, its meshes by node, and
/// the planar patches it draws through on another plane (as
/// [`warp_words`](crate::warp_words) has them). Symbol streams join with
/// [`add_symbol_mesh`](Renderer::add_symbol_mesh).
#[derive(Clone, Debug)]
pub struct TileBuild<'a> {
    pub key: TileKey,
    /// Every mesh's words, each pool's together (all the vertices, all
    /// the indices, all the rows): they go up as one range of each pool,
    /// with one write per pool.
    pub words: Words<'a>,
    pub meshes: &'a [MeshBuild<'a>],
    pub patches: Option<(u32, &'a [f32])>,
}

/// Per pool (vertices, indices, rows), the words of a tile from its
/// meshes' first span to their last: what it takes of the pool, as one
/// range.
pub fn tile_extents<'a>(spans: impl IntoIterator<Item = &'a [Range<u32>; 3]>) -> [Range<u32>; 3] {
    let mut extents: [Option<Range<u32>>; 3] = Default::default();
    for spans in spans {
        for (extent, span) in extents.iter_mut().zip(spans) {
            if !span.is_empty() {
                let (start, end) = extent.as_ref().map_or((span.start, span.end), |e| {
                    (e.start.min(span.start), e.end.max(span.end))
                });
                *extent = Some(start..end);
            }
        }
    }
    extents.map(Option::unwrap_or_default)
}

/// What a tile whose meshes take `extents` ([`tile_extents`]) and hold
/// `chunks` chunks takes of the scene.
pub fn tile_size(extents: &[Range<u32>; 3], chunks: u32) -> MeshSize {
    let len = |p: usize| extents[p].end - extents[p].start;
    MeshSize {
        vertices: len(0),
        indices: len(1),
        props: len(2),
        chunks,
    }
}

impl TileBuild<'_> {
    fn extents(&self) -> [Range<u32>; 3] {
        tile_extents(self.meshes.iter().map(|m| &m.spans))
    }

    /// What it takes of the scene.
    pub fn size(&self) -> MeshSize {
        let chunks = self.meshes.iter().map(|m| m.chunks.len() as u32).sum();
        tile_size(&self.extents(), chunks)
    }

    /// Whether its meshes' spans lie in its words apart from each other.
    fn spans_apart(&self) -> bool {
        (0..3).all(|pool| {
            let mut spans: Vec<&Range<u32>> = (self.meshes.iter())
                .map(|m| &m.spans[pool])
                .filter(|s| !s.is_empty())
                .collect();
            spans.sort_unstable_by_key(|s| s.start);
            spans.windows(2).all(|w| w[0].end <= w[1].start)
                && spans.last().is_none_or(|s| s.end <= self.words.len())
        })
    }
}

/// A tile's words being written into the pools it holds room in, one
/// range each ([`Renderer::reserve`]): nothing draws them until
/// [`Renderer::finish_tile`], so a tile written over several frames shows
/// whole or not at all.
#[derive(Debug)]
pub struct Reserved {
    /// Per pool, the tile's words it holds (a range of the build's) and
    /// where in the pool they start.
    pools: [(Range<u32>, u32); 3],
    /// Bytes written, over the pools in turn.
    written: u64,
}

impl Reserved {
    /// The bytes written.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// The bytes it has left to write.
    pub fn left(&self) -> u64 {
        let words: u64 = (self.pools.iter())
            .map(|(words, _)| u64::from(words.end - words.start))
            .sum();
        words * 4 - self.written
    }
}

/// Why an upload failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UploadError {
    /// No room: releasing tiles, or uploading again allowed to grow, may
    /// make some.
    NoRoom,
    /// The tile itself can't draw.
    Invalid(String),
}

#[derive(Debug)]
struct Entry {
    key: TileKey,
    /// One scene tile per layer extent.
    parts: Vec<(u32, TileId)>,
    /// Its meshes by render node index.
    meshes: Vec<(u32, MeshId)>,
    image: Option<Image>,
    warp: TileWarp,
    /// Its place in the tile order ([`Renderer::set_tile_order`]), which
    /// parts it gains later take too.
    order: Option<u32>,
}

impl Entry {
    /// The scene's place for part `part` of a tile at `order` in the tile
    /// order: the tile's parts follow each other, in the order they came.
    fn part_place(order: u32, part: usize) -> u64 {
        u64::from(order) << 32 | part as u64
    }
}

#[derive(Debug)]
struct Image {
    layer: u32,
    mesh: Option<Arc<ImageMesh>>,
    fade: ImageFade,
    /// A DEM tile's unpack vector, and the neighbours' layers its border
    /// was last padded from ([`Renderer::pad_dems`]).
    unpack: Option<[f32; 4]>,
    padded: Option<[Option<u32>; 8]>,
}

/// A source's image page and its resident tiles.
#[derive(Debug)]
struct Source {
    page: PageId,
    tiles: FxHashMap<TileIndex, GpuTile>,
}

/// The renderer's tiles.
#[derive(Debug, Default)]
pub(crate) struct Tiles {
    entries: Vec<Option<Entry>>,
    free: Vec<u32>,
    sources: FxHashMap<u32, Source>,
    /// Pages no source uses any more, for the next frame to destroy.
    retired: Vec<PageId>,
    /// Vector tiles drawing through planar patches, which only the paint
    /// and symbol pipelines' full variants draw.
    warped: u32,
}

impl Tiles {
    /// How many vector tiles draw through planar patches.
    pub(crate) fn warped(&self) -> u32 {
        self.warped
    }

    /// Whether an entry is a vector tile drawing through planar patches.
    fn is_warped(entry: &Entry) -> bool {
        entry.image.is_none() && entry.warp.cells() > 0
    }

    fn entry(&self, tile: GpuTile) -> &Entry {
        self.entries[tile.0 as usize]
            .as_ref()
            .expect("a released tile")
    }

    fn insert(&mut self, entry: Entry) -> GpuTile {
        self.warped += u32::from(Self::is_warped(&entry));
        match self.free.pop() {
            Some(i) => {
                self.entries[i as usize] = Some(entry);
                GpuTile(i)
            }
            None => {
                self.entries.push(Some(entry));
                GpuTile(self.entries.len() as u32 - 1)
            }
        }
    }

    /// The page of a source with image tiles.
    pub(crate) fn page(&self, source: u32) -> Option<PageId> {
        self.sources.get(&source).map(|s| s.page)
    }

    fn layer_of(&self, source: u32, tile: TileIndex) -> Option<u32> {
        let found = *self.sources.get(&source)?.tiles.get(&tile)?;
        self.entry(found).image.as_ref().map(|i| i.layer)
    }

    /// Renumbers mesh nodes after a node remap: node `i` became `map[i]`.
    pub(crate) fn remap(&mut self, map: &[Option<u32>]) {
        for entry in self.entries.iter_mut().flatten() {
            entry
                .meshes
                .retain_mut(|(node, _)| match map[*node as usize] {
                    Some(new) => {
                        *node = new;
                        true
                    }
                    None => false,
                });
        }
    }

    /// Image tiles' meshes in nodes leaving: they're removed from the scene
    /// before the nodes are.
    pub(crate) fn meshes_of(&self, node: u32) -> impl Iterator<Item = MeshId> + '_ {
        self.entries
            .iter()
            .flatten()
            .flat_map(move |e| e.meshes.iter().filter(move |m| m.0 == node).map(|m| m.1))
    }

    /// The image tiles of a source.
    pub(crate) fn of_source(&self, source: u32) -> Vec<GpuTile> {
        self.sources
            .get(&source)
            .map(|s| s.tiles.values().copied().collect())
            .unwrap_or_default()
    }

    /// Sources with a page.
    pub(crate) fn sources(&self) -> impl Iterator<Item = (u32, PageId)> + '_ {
        self.sources.iter().map(|(&k, s)| (k, s.page))
    }

    pub(crate) fn forget_source(&mut self, source: u32) -> Option<PageId> {
        let s = self.sources.remove(&source)?;
        debug_assert!(s.tiles.is_empty());
        Some(s.page)
    }

    pub(crate) fn source_empty(&self, source: u32) -> bool {
        self.sources.get(&source).is_none_or(|s| s.tiles.is_empty())
    }
}

impl Renderer {
    /// Uploads an image tile: its pixels into a layer of its source's page,
    /// and a quad (or its mesh) into each of the source's image nodes.
    /// Without `grow`, only room already there is used; with it, the page
    /// and the scene's pools grow within their limits, if memory is
    /// [free](crate::Moves::Free). No source's first page is made while
    /// memory is [held](crate::Moves::Held) ([`has_page`](Self::has_page)).
    pub fn upload_image(
        &mut self,
        gpu: &mut dyn Gpu,
        build: &ImageBuild<'_>,
        grow: bool,
    ) -> Result<GpuTile, UploadError> {
        let ImageBuild {
            key, kind, size, ..
        } = *build;
        if kind == PageKind::Dem && size[0] != size[1] {
            return Err(UploadError::Invalid("DEM tiles must be square".into()));
        }
        let page = match self.tiles.page(key.source) {
            Some(page) => page,
            None if self.moves() == crate::Moves::Held => return Err(UploadError::NoRoom),
            None => {
                let layers = self.page_layers(kind, size);
                let page = self.images.create_page(gpu, kind, size, layers);
                for node in &mut self.nodes {
                    if node.source == Some(key.source)
                        && let Some(image) = &mut node.image
                    {
                        crate::image::Images::bind_page(image, page);
                    }
                }
                self.tiles.sources.insert(
                    key.source,
                    Source {
                        page,
                        tiles: FxHashMap::default(),
                    },
                );
                page
            }
        };
        let expected = self.images.page_size(page);
        if expected != size {
            return Err(UploadError::Invalid(format!(
                "a {}×{} image in a source of {}×{} tiles",
                size[0], size[1], expected[0], expected[1]
            )));
        }
        let nodes: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| self.nodes[i as usize].source == Some(key.source))
            .collect();
        let sizes = vec![image_mesh_size(build.mesh.as_deref()); nodes.len()];
        let fits = self.scene.fits(&sizes) || (grow && self.grow_for(gpu, &sizes));
        if !fits {
            self.scene.refuse(&sizes);
            return Err(UploadError::NoRoom);
        }
        let layer = self
            .alloc_layer(gpu, page, grow)
            .map_err(|OutOfSpace| UploadError::NoRoom)?;
        self.images.write_layer(gpu, page, layer, build.pixels);
        if let (PageKind::Raster, Some(area)) = (kind, build.area) {
            self.images.extend_edges(page, layer, area);
        }

        let fade = match build.fade_from {
            Some(time) => self.fade_in(key, time),
            None => ImageFade::NONE,
        };
        let part = self.scene.insert_tile();
        let mut entry = Entry {
            key,
            parts: vec![(IMAGE_EXTENT, part)],
            meshes: Vec::new(),
            image: Some(Image {
                layer,
                mesh: build.mesh.clone(),
                fade,
                unpack: build.unpack,
                padded: None,
            }),
            warp: TileWarp::NONE,
            order: None,
        };
        let added = nodes
            .iter()
            .try_for_each(|&node| self.add_image_node(gpu, &mut entry, node));
        let warped = match (added, build.patches) {
            (Ok(()), Some((cells, patches))) => self
                .scene
                .add_warp(gpu, cells, patches)
                .map(|warp| entry.warp = warp),
            (added, _) => added,
        };
        if warped.is_err() {
            self.scene.remove_tile(part);
            self.images.free_layer(page, layer);
            return Err(UploadError::NoRoom);
        }

        let handle = self.tiles.insert(entry);
        let replaced = self
            .tiles
            .sources
            .get_mut(&key.source)
            .expect("made above")
            .tiles
            .insert(key.tile, handle);
        debug_assert!(replaced.is_none(), "one resident tile per index");
        Ok(handle)
    }

    /// Writes new texels of the same size over a resident image tile's
    /// layer (a video's next frame); false if the tile isn't resident or
    /// the size differs.
    pub fn rewrite_image(
        &mut self,
        gpu: &mut dyn Gpu,
        key: TileKey,
        size: [u32; 2],
        pixels: Pixels<'_>,
    ) -> bool {
        let (Some(page), Some(layer)) = (
            self.tiles.page(key.source),
            self.tiles.layer_of(key.source, key.tile),
        ) else {
            return false;
        };
        if self.images.page_size(page) != size {
            return false;
        }
        self.images.write_layer(gpu, page, layer, pixels);
        true
    }

    /// Frees a tile: its meshes, warp and texture layer. A source's page
    /// goes with its last tile once no node draws the source.
    pub fn release_tile(&mut self, tile: GpuTile) {
        let entry = self.tiles.entries[tile.0 as usize]
            .take()
            .expect("a tile is released once");
        self.tiles.free.push(tile.0);
        self.tiles.warped -= u32::from(Tiles::is_warped(&entry));
        for (_, part) in entry.parts {
            self.scene.remove_tile(part);
        }
        self.scene.remove_warp(entry.warp);
        if let Some(image) = entry.image {
            let source = self
                .tiles
                .sources
                .get_mut(&entry.key.source)
                .expect("image tiles have a source");
            source.tiles.remove(&entry.key.tile);
            self.images.free_layer(source.page, image.layer);
            self.drop_unused_page(entry.key.source);
        }
    }

    /// Gives an image tile another mesh (a globe came or went): its quads
    /// in every node, and its patches. The pools grow, within their
    /// limits, for a finer mesh than the old. All or nothing: on `Err`
    /// the tile holds no quads and draws nothing until a mesh fits.
    pub fn set_image_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: GpuTile,
        mesh: Option<Arc<ImageMesh>>,
        patches: Option<(u32, &[f32])>,
    ) -> Result<(), OutOfSpace> {
        let mut entry = self.tiles.entries[tile.0 as usize]
            .take()
            .expect("a resident tile");
        for (_, id) in entry.meshes.drain(..) {
            self.scene.remove_mesh(id);
        }
        self.scene.remove_warp(std::mem::take(&mut entry.warp));
        let nodes: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| self.nodes[i as usize].source == Some(entry.key.source))
            .collect();
        let mut sizes = vec![image_mesh_size(mesh.as_deref()); nodes.len()];
        sizes.extend(patches.map(|(_, words)| MeshSize {
            props: words.len() as u32,
            ..MeshSize::default()
        }));
        let image = entry.image.as_mut().expect("image tiles hold an image");
        image.mesh = mesh;
        let mut result = match self.fits(&sizes) || self.grow_for(gpu, &sizes) {
            true => nodes
                .iter()
                .try_for_each(|&node| self.add_image_node(gpu, &mut entry, node)),
            false => Err(OutOfSpace),
        };
        if let (Ok(()), Some((cells, patches))) = (&result, patches) {
            result = self
                .scene
                .add_warp(gpu, cells, patches)
                .map(|warp| entry.warp = warp);
        }
        if result.is_err() {
            // Quads without their patches would draw off the surface.
            for (_, id) in entry.meshes.drain(..) {
                self.scene.remove_mesh(id);
            }
        }
        self.tiles.entries[tile.0 as usize] = Some(entry);
        result
    }

    /// Uploads a tile: all its meshes, in order, or none. Without `grow`,
    /// only room already there is used; with it, the pools grow within
    /// their limits, if memory is [free](crate::Moves::Free). Returns the
    /// tile and its meshes' ids.
    pub fn upload_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        build: &TileBuild<'_>,
        grow: bool,
    ) -> Result<(GpuTile, Vec<MeshId>), UploadError> {
        let size = build.size();
        if grow && !self.fits(&[size]) {
            self.grow_for(gpu, &[size]);
        }
        let mut reserved = self.reserve(build)?;
        self.write_reserved(gpu, build.words, &mut reserved, u64::MAX);
        self.finish_tile(gpu, build, reserved)
    }

    /// Takes room for a tile's words, one range of each pool, without
    /// growing them; [`write_reserved`](Self::write_reserved) writes them
    /// and [`finish_tile`](Self::finish_tile) makes the tile of them, or
    /// [`unreserve`](Self::unreserve) gives the room back.
    pub fn reserve(&mut self, build: &TileBuild<'_>) -> Result<Reserved, UploadError> {
        if !build.spans_apart() {
            return Err(UploadError::Invalid("a tile's streams overlap".into()));
        }
        let size = build.size();
        if !self.fits(&[size]) {
            self.scene.refuse(&[size]);
            return Err(UploadError::NoRoom);
        }
        let extents = build.extents();
        let lens = extents.clone().map(|e| e.end - e.start);
        let block = self.scene.alloc_block(lens).ok_or(UploadError::NoRoom)?;
        Ok(Reserved {
            pools: [0, 1, 2].map(|p| (extents[p].clone(), block[p].start)),
            written: 0,
        })
    }

    /// Writes up to `bytes` more of a reserved tile's `words` (its
    /// build's), in whole words: at most one write per pool. Returns the
    /// bytes written.
    pub fn write_reserved(
        &mut self,
        gpu: &mut dyn Gpu,
        words: Words<'_>,
        reserved: &mut Reserved,
        bytes: u64,
    ) -> u64 {
        let mut budget = bytes / 4;
        let mut skip = reserved.written / 4;
        let mut written = 0;
        for (pool, (range, at)) in reserved.pools.iter().enumerate() {
            let len = u64::from(range.end - range.start);
            if skip >= len {
                skip -= len;
                continue;
            }
            let take = (len - skip).min(budget);
            if take == 0 {
                break;
            }
            let from = range.start + skip as u32;
            let piece = words.slice(from..from + take as u32);
            self.scene.write_pool(gpu, pool, at + skip as u32, piece);
            (skip, budget, written) = (0, budget - take, written + take * 4);
        }
        reserved.written += written;
        written
    }

    /// Gives back a reserved tile's room.
    pub fn unreserve(&mut self, reserved: Reserved) {
        for (pool, (range, at)) in reserved.pools.into_iter().enumerate() {
            self.scene
                .free_words(pool, at..at + (range.end - range.start));
        }
    }

    /// Makes a reserved tile, its words all written, of its build: its
    /// meshes take their words where they were written, and from then on
    /// draw. Returns the tile and its meshes' ids. On `Err` the room is
    /// given back.
    pub fn finish_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        build: &TileBuild<'_>,
        reserved: Reserved,
    ) -> Result<(GpuTile, Vec<MeshId>), UploadError> {
        debug_assert_eq!(reserved.left(), 0, "a tile's words are written first");
        let Ok(tile) = self.add_tile(gpu, build.key, build.patches) else {
            self.unreserve(reserved);
            return Err(UploadError::NoRoom);
        };
        // Empty ranges start at 0, as an arena's empty allocations do.
        let at = |pool: usize, span: &Range<u32>| {
            if span.is_empty() {
                return 0..0;
            }
            let (words, at) = &reserved.pools[pool];
            let start = at + (span.start - words.start);
            start..start + (span.end - span.start)
        };
        // The words between the meshes' (those of meshes left out) go.
        for pool in 0..3 {
            let mut spans: Vec<Range<u32>> = (build.meshes.iter())
                .map(|m| m.spans[pool].clone())
                .filter(|s| !s.is_empty())
                .collect();
            spans.sort_unstable_by_key(|s| s.start);
            let mut from = reserved.pools[pool].0.start;
            for span in spans
                .into_iter()
                .chain(std::iter::once(reserved.pools[pool].0.end..u32::MAX))
            {
                if span.start > from {
                    self.scene.free_words(pool, at(pool, &(from..span.start)));
                }
                from = span.end;
            }
        }
        let mut ids: Vec<MeshId> = Vec::with_capacity(build.meshes.len());
        for (i, mesh) in build.meshes.iter().enumerate() {
            let shared = mesh
                .shares
                .map(|of| of.map(|of| self.scene.mesh_ranges(ids[of])));
            let ranges = [0, 1, 2].map(|p| match (&shared, p) {
                ([Some(of), _], 0) => of[0].clone(),
                ([_, Some(of)], 1) => of[1].clone(),
                _ => at(p, &mesh.spans[p]),
            });
            let part = self.part(tile, mesh.extent);
            let inserted = self.scene.insert_mesh_at(
                gpu,
                part,
                mesh.node,
                ranges,
                (mesh.chunks, mesh.row_words),
                mesh.shares.map(|s| s.is_some()),
            );
            let failed = match inserted {
                Ok(id) => match self.attach(gpu, id, &mesh.rows) {
                    Ok(()) => {
                        self.entry_mut(tile).meshes.push((mesh.node, id));
                        ids.push(id);
                        continue;
                    }
                    Err(OutOfSpace) => {
                        self.scene.remove_mesh(id);
                        i + 1
                    }
                },
                Err(OutOfSpace) => i,
            };
            // The meshes not inserted give back their own words.
            for mesh in &build.meshes[failed..] {
                for (pool, span) in mesh.spans.iter().enumerate() {
                    if !span.is_empty() {
                        self.scene.free_words(pool, at(pool, span));
                    }
                }
            }
            self.release_tile(tile);
            return Err(UploadError::NoRoom);
        }
        Ok((tile, ids))
    }
    /// A vector tile with no meshes yet, drawing through planar patches
    /// (cells per side and their words) if it has any.
    pub fn add_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        key: TileKey,
        patches: Option<(u32, &[f32])>,
    ) -> Result<GpuTile, OutOfSpace> {
        let warp = match patches {
            Some((cells, patches)) => self.scene.add_warp(gpu, cells, patches)?,
            None => TileWarp::NONE,
        };
        Ok(self.tiles.insert(Entry {
            key,
            parts: Vec::new(),
            meshes: Vec::new(),
            image: None,
            warp,
            order: None,
        }))
    }

    /// Adds a mesh for render `node` to a tile, in its part of layer
    /// extent `extent`, with what its rows refer to.
    pub fn add_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: GpuTile,
        (node, extent): (u32, u32),
        data: &MeshData<'_>,
        rows: &MeshRows<'_>,
    ) -> Result<MeshId, OutOfSpace> {
        self.add_mesh_sharing(gpu, tile, (node, extent), data, rows, [None; 2])
    }

    /// [`add_mesh`](Self::add_mesh) drawing the vertices, the indices or
    /// both of meshes of the same tile (`shares`).
    fn add_mesh_sharing(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: GpuTile,
        (node, extent): (u32, u32),
        data: &MeshData<'_>,
        rows: &MeshRows<'_>,
        shares: [Option<MeshId>; 2],
    ) -> Result<MeshId, OutOfSpace> {
        let part = self.part(tile, extent);
        let id = self
            .scene
            .insert_mesh_sharing(gpu, part, node, data, shares)?;
        if let Err(e) = self.attach(gpu, id, rows) {
            self.scene.remove_mesh(id);
            return Err(e);
        }
        self.entry_mut(tile).meshes.push((node, id));
        Ok(id)
    }

    /// Adds a symbol stream for render `node` to a tile, with its rows'
    /// feature-state indices.
    pub fn add_symbol_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: GpuTile,
        node: u32,
        data: &SymbolMeshData<'_>,
        states: &[u32],
    ) -> Result<MeshId, OutOfSpace> {
        let part = self.part(tile, IMAGE_EXTENT);
        let id = self.scene.insert_symbol_mesh(gpu, part, node, data)?;
        if let Err(e) = self.scene.set_mesh_state(gpu, id, states) {
            self.scene.remove_mesh(id);
            return Err(e);
        }
        self.entry_mut(tile).meshes.push((node, id));
        Ok(id)
    }

    /// Removes one of a tile's meshes.
    pub fn remove_mesh(&mut self, tile: GpuTile, mesh: MeshId) {
        self.scene.remove_mesh(mesh);
        self.entry_mut(tile).meshes.retain(|&(_, id)| id != mesh);
    }

    /// Replaces a mesh's property rows, keeping its geometry: `row_words`
    /// per feature, and what they refer to.
    pub fn set_mesh_rows(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        (props, row_words): (&Words<'_>, u32),
        rows: &MeshRows<'_>,
    ) -> Result<(), OutOfSpace> {
        self.scene.set_mesh_props(gpu, mesh, props, row_words)?;
        self.attach(gpu, mesh, rows)
    }

    /// Whether meshes of these sizes would fit without growing the pools.
    pub fn fits(&self, meshes: &[MeshSize]) -> bool {
        self.scene.fits(meshes)
    }

    /// Records that meshes of these sizes found no room, as an upload
    /// refused would ([`take_refusals`](Self::take_refusals)).
    pub fn refuse(&mut self, meshes: &[MeshSize]) {
        self.scene.refuse(meshes);
    }

    /// Grows the pools, within their limits, until meshes of these sizes
    /// fit; whether they do. Only [free](crate::Moves::Free) memory grows
    /// for an upload.
    pub fn grow_for(&mut self, gpu: &mut dyn Gpu, meshes: &[MeshSize]) -> bool {
        if !self.uploads_grow() {
            return self.fits(meshes);
        }
        let fits = self.scene.grow_for(gpu, meshes);
        self.glyphs
            .set_table(self.scene.tables().writer(Table::Props));
        fits
    }

    /// Halves a pool, table or image tile page a view that emptied left
    /// mostly free: one move. Whether one shrank: never while memory is
    /// [held](crate::Moves::Held).
    pub fn shrink(&mut self, gpu: &mut dyn Gpu) -> bool {
        if self.moves() == crate::Moves::Held {
            return false;
        }
        self.refused_pages.clear();
        if self.scene.shrink(gpu) {
            self.glyphs
                .set_table(self.scene.tables().writer(Table::Props));
            return true;
        }
        self.images.shrink(gpu)
    }

    /// Whether [`shrink`](Self::shrink) would shrink anything.
    pub fn would_shrink(&self) -> bool {
        self.scene.would_shrink() || self.images.would_shrink()
    }

    /// Forgets the pools' refusals, as [`shrink`](Self::shrink) does, on
    /// a frame that doesn't shrink.
    pub fn forget_refusals(&mut self) {
        self.scene.forget_refusals();
        self.refused_pages.clear();
    }

    /// Whether a pool, the chunk table or an image tile page is more than
    /// three quarters full, below its ceiling, and growing it fits in
    /// `room` bytes.
    pub fn wants_headroom(&self, room: u64) -> bool {
        self.scene.wants_headroom(room)
            || self.images.wants_headroom(room)
            || self.glyphs.wants_headroom(room)
    }

    /// Grows one of what is more than three quarters full, within its
    /// ceiling and `room` bytes: a map at rest makes room ahead of the next
    /// view, so that a gesture from rest needn't grow them. New memory
    /// costs its clearing, and a device may hold up other work while it
    /// clears (#176: on RADV 43 µs a MiB, holding up submissions; a new
    /// 256 MiB block of wgpu's allocator 10-12 ms), so a frame grows one,
    /// and a gesture that starts waits for one at most. Whether one grew:
    /// never while memory is [held](crate::Moves::Held).
    pub fn grow_headroom(&mut self, gpu: &mut dyn Gpu, room: u64) -> bool {
        if self.moves() == crate::Moves::Held {
            return false;
        }
        if self.scene.grow_headroom(gpu, room) {
            self.glyphs
                .set_table(self.scene.tables().writer(Table::Props));
            return true;
        }
        self.images.grow_headroom(gpu, room) || self.glyphs.grow_headroom(gpu, room)
    }

    /// Grows the pools once, within their limits; whether they grew. Only
    /// [free](crate::Moves::Free) memory grows for an upload.
    pub fn grow(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.uploads_grow() && self.grow_for_style(gpu)
    }

    /// Grows the pools once for an edit's relayout, within their limits,
    /// unless memory is [held](crate::Moves::Held): an edit at rest grows
    /// them as a style's load sizes them, with no upkeep to wait for.
    /// Whether they grew.
    pub fn grow_for_edit(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.moves() != crate::Moves::Held && self.grow_for_style(gpu)
    }

    /// Grows the pools once, within their limits, whatever the
    /// [gate](crate::Moves) says: for what a style's load or edit makes
    /// itself (its background meshes), which has no upkeep to wait for and
    /// no frame to keep smooth. Whether they grew.
    pub fn grow_for_style(&mut self, gpu: &mut dyn Gpu) -> bool {
        let grew = self.scene.grow(gpu);
        self.glyphs
            .set_table(self.scene.tables().writer(Table::Props));
        grew
    }

    /// How many world copies a symbol mesh keeps state words for.
    pub fn symbol_copies(&self, mesh: MeshId) -> u32 {
        self.scene.symbol_copies(mesh)
    }

    /// Keeps state words for `copies` world copies (a power of two), all
    /// hidden until written.
    pub fn set_symbol_copies(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        copies: u32,
    ) -> Result<(), OutOfSpace> {
        self.scene.set_symbol_copies(gpu, mesh, copies)
    }

    /// Writes a symbol mesh's state words for world copy `wrap`.
    pub fn write_symbol_states(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        wrap: i32,
        words: &[u32],
    ) {
        self.scene.write_symbol_states(gpu, mesh, wrap, words);
    }

    /// Sets the order a symbol mesh's quads draw in: every quad once.
    pub fn write_symbol_order(&mut self, gpu: &mut dyn Gpu, mesh: MeshId, order: &[u32]) {
        self.scene.write_symbol_order(gpu, mesh, order);
    }

    fn attach(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        rows: &MeshRows<'_>,
    ) -> Result<(), OutOfSpace> {
        for &(word, image) in rows.images {
            self.scene.set_mesh_image(gpu, mesh, word, image);
        }
        self.scene.set_mesh_state(gpu, mesh, rows.states)
    }

    /// Pads source `source`'s drawn DEM tiles (`drawn`) from their drawn
    /// neighbours: a tile whose drawn neighbours changed since it was last
    /// prepared is prepared again. A border is then a function of what the
    /// view draws, not of which tiles were resident before.
    pub fn pad_dems(&mut self, source: u32, drawn: &FxHashSet<TileIndex>) {
        let Some(page) = self.tiles.sources.get(&source).map(|s| s.page) else {
            return;
        };
        let tiles = &self.tiles;
        let layer_of = |t: TileIndex| {
            drawn
                .contains(&t)
                .then(|| tiles.layer_of(source, t))
                .flatten()
        };
        let mut jobs = Vec::new();
        for &tile in drawn {
            let Some(&found) = tiles.sources[&source].tiles.get(&tile) else {
                continue;
            };
            let Some(image) = &tiles.entry(found).image else {
                continue;
            };
            let Some(unpack) = image.unpack else {
                continue;
            };
            let neighbours = dem_neighbours(tile, &layer_of);
            if image.padded != Some(neighbours) {
                let job = DemPrepare {
                    layer: image.layer,
                    neighbours,
                    borders: image.padded.map(|was| changed_borders(was, neighbours)),
                    zoom: f32::from(tile.z),
                    unpack,
                };
                jobs.push((found, job));
            }
        }
        for (found, job) in jobs {
            if let Some(image) = &mut self.entry_mut(found).image {
                image.padded = Some(job.neighbours);
            }
            self.images.prepare_dem(page, job);
        }
    }

    fn entry_mut(&mut self, tile: GpuTile) -> &mut Entry {
        self.tiles.entries[tile.0 as usize]
            .as_mut()
            .expect("a released tile")
    }

    /// A tile's scene tile of layer extent `extent`, made if need be in the
    /// tile's place in the tile order.
    fn part(&mut self, tile: GpuTile, extent: u32) -> TileId {
        if let Some(&(_, id)) = self.entry_mut(tile).parts.iter().find(|p| p.0 == extent) {
            return id;
        }
        let id = self.scene.insert_tile();
        let entry = self.entry_mut(tile);
        entry.parts.push((extent, id));
        if let Some(order) = entry.order {
            let at = Entry::part_place(order, entry.parts.len() - 1);
            self.scene.order_tile(id, at);
        }
        id
    }

    /// A tile's patches on the low-zoom surface, uploaded the first time a
    /// placement needs them: `patches` makes them, as
    /// [`warp_words`](crate::warp_words) has them. `None` when the property
    /// memory has no room.
    pub fn surface_warp(
        &mut self,
        gpu: &mut dyn Gpu,
        key: WarpKey,
        patches: impl FnOnce() -> (u32, Vec<f32>),
    ) -> Option<TileWarp> {
        let frame = self.surface_warps.frame;
        if let Some((warp, used)) = self.surface_warps.warps.get_mut(&key) {
            *used = frame;
            return Some(*warp);
        }
        let (cells, words) = patches();
        let warp = self.scene.add_warp(gpu, cells, &words).ok()?;
        self.surface_warps.warps.insert(key, (warp, frame));
        Some(warp)
    }

    /// Forgets every low-zoom surface patch: the surface changed.
    pub fn clear_surface_warps(&mut self) {
        for (_, (warp, _)) in self.surface_warps.warps.drain() {
            self.scene.remove_warp(warp);
        }
    }

    /// Ends a frame's surface patches: past the cache's size, those it
    /// didn't place go.
    pub(crate) fn end_surface_warps(&mut self) {
        let warps = &mut self.surface_warps;
        if warps.warps.len() > SURFACE_WARPS {
            let frame = warps.frame;
            let scene = &mut self.scene;
            warps.warps.retain(|_, &mut (warp, used)| {
                let kept = used == frame;
                if !kept {
                    scene.remove_warp(warp);
                }
                kept
            });
        }
        warps.frame += 1;
    }

    /// The layer extents a tile holds meshes at: each places apart.
    pub fn tile_extents(&self, tile: GpuTile) -> Vec<u32> {
        self.tiles.entry(tile).parts.iter().map(|p| p.0).collect()
    }

    /// Places a tile's meshes of layer extent `extent`, those it has and
    /// those added later: one row per placement, none to hide them. Rows
    /// draw through the tile's own planar patches.
    pub fn place(
        &mut self,
        tile: GpuTile,
        extent: u32,
        rows: &[TileRow],
    ) -> Result<(), OutOfSpace> {
        let part = self.part(tile, extent);
        let warp = self.tiles.entry(tile).warp;
        let rows: Vec<TileRow> = rows.iter().map(|&row| TileRow { warp, ..row }).collect();
        self.scene.place_tile(part, &rows)
    }

    /// Keeps only a tile's labels drawing, where it was last placed: a
    /// tile that left the draw list, until a placement covers what
    /// replaced it. Each frame it holds them, `globe` and `ground` are
    /// that frame's for the tile (its place on the low-zoom surface, the
    /// DEM tile it stands on): a held label stays where it is in the world
    /// as the camera moves and the globe hands over, and doesn't read
    /// patches or a DEM layer that went since.
    pub fn hold_symbols(
        &mut self,
        tile: GpuTile,
        globe: Option<crate::RowGlobe>,
        ground: (crate::DemRef, [f32; 2]),
    ) {
        for &(_, part) in &self.tiles.entry(tile).parts {
            self.scene.hold_symbols(part, globe, ground);
        }
    }

    /// Hides a tile: no placements.
    pub fn unplace(&mut self, tile: GpuTile) {
        for &(_, part) in &self.tiles.entry(tile).parts {
            self.scene
                .place_tile(part, &[])
                .expect("removing placements always fits");
        }
    }

    /// Orders source `source`'s tiles within each render node, in drawing
    /// order; its tiles not listed follow, and other sources' keep their
    /// order. The order holds until the source's next: meshes a tile gains
    /// later (a relayout's) draw in its place too.
    pub fn set_tile_order(&mut self, source: u32, tiles: &[GpuTile]) {
        let mut parts = Vec::new();
        let ordered = (self.tiles.entries.iter_mut().flatten())
            .filter(|e| e.key.source == source && e.order.is_some());
        for entry in ordered {
            entry.order = None;
            parts.extend(entry.parts.iter().map(|p| (p.1, None)));
        }
        for (order, &tile) in (0u32..).zip(tiles) {
            let entry = self.entry_mut(tile);
            debug_assert_eq!(entry.key.source, source, "another source's tile");
            entry.order = Some(order);
            let places = (entry.parts.iter().enumerate())
                .map(|(i, p)| (p.1, Some(Entry::part_place(order, i))));
            parts.extend(places);
        }
        self.scene.set_tile_order(parts);
    }

    /// Words of the property buffer for a table outside any tile, such as
    /// the feature-state table; not cleared. The pool grows, within its
    /// limit, to make room.
    pub fn alloc_table(
        &mut self,
        gpu: &mut dyn Gpu,
        words: u32,
    ) -> Result<std::ops::Range<u32>, OutOfSpace> {
        self.scene.alloc_table(words).or_else(|_| {
            let table = MeshSize {
                props: words,
                ..MeshSize::default()
            };
            self.grow_for(gpu, &[table]);
            self.scene.alloc_table(words)
        })
    }

    pub fn free_table(&mut self, table: std::ops::Range<u32>) {
        self.scene.free_table(table);
    }

    /// Writes words of a table from [`alloc_table`](Self::alloc_table),
    /// from word `at` of the property buffer.
    pub fn write_table(&mut self, gpu: &mut dyn Gpu, at: u32, words: &[u32]) {
        self.scene.write_table(gpu, at, words);
    }

    /// GPU bytes a tile holds: its meshes and texture layer.
    pub fn tile_bytes(&self, tile: GpuTile) -> u64 {
        let entry = self.tiles.entry(tile);
        let meshes: u64 = entry
            .meshes
            .iter()
            .map(|&(_, id)| self.scene.mesh_bytes(id))
            .sum();
        let layer = entry.image.as_ref().map_or(0, |_| {
            let page = self.tiles.page(entry.key.source).expect("image source");
            self.images.layer_bytes(page)
        });
        meshes + layer
    }

    /// Adds an image tile's quad (or mesh) to render node `node`.
    fn add_image_node(
        &mut self,
        gpu: &mut dyn Gpu,
        entry: &mut Entry,
        node: u32,
    ) -> Result<(), OutOfSpace> {
        let image = entry.image.as_ref().expect("image tiles hold an image");
        let tile = entry.key.tile;
        let [top, bottom] = tile_latitudes(tile);
        let mut row = [0; IMAGE_ROW_WORDS];
        row[..3].copy_from_slice(&[image.layer, top.to_bits(), bottom.to_bits()]);
        row[3..].copy_from_slice(&image.fade.words());
        let mesh = match &image.mesh {
            Some(mesh) => MeshData {
                vertices: Words::Core(&mesh.vertices),
                indices: Words::Core(&mesh.indices),
                props: Words::Core(&row),
                row_words: IMAGE_ROW_WORDS as u32,
                chunks: &mesh.chunks,
            },
            None => MeshData {
                vertices: Words::Core(&QUAD),
                indices: Words::Core(&QUAD_INDICES),
                props: Words::Core(&row),
                row_words: IMAGE_ROW_WORDS as u32,
                chunks: &[QUAD_CHUNK],
            },
        };
        let id = self.scene.insert_mesh(gpu, entry.parts[0].1, node, &mesh)?;
        entry.meshes.push((node, id));
        Ok(())
    }

    /// New image nodes draw the tiles their source has resident.
    pub(crate) fn add_image_nodes(&mut self, gpu: &mut dyn Gpu, nodes: &[u32]) -> bool {
        let mut complete = true;
        for &node in nodes {
            let Some(source) = self.nodes[node as usize].source else {
                continue;
            };
            if let (Some(page), Some(image)) = (
                self.tiles.page(source),
                self.nodes[node as usize].image.as_mut(),
            ) {
                crate::image::Images::bind_page(image, page);
            }
            // A new node of a source draws its resident tiles: room for
            // their quads first.
            let tiles = self.tiles.of_source(source);
            let sizes: Vec<MeshSize> = (tiles.iter())
                .map(|tile| {
                    let image = self.tiles.entry(*tile).image.as_ref();
                    image_mesh_size(image.and_then(|i| i.mesh.as_deref()))
                })
                .collect();
            if !self.fits(&sizes) {
                self.grow_for(gpu, &sizes);
            }
            for tile in tiles {
                let mut entry = self.tiles.entries[tile.0 as usize]
                    .take()
                    .expect("indexed tiles are resident");
                complete &= self.add_image_node(gpu, &mut entry, node).is_ok();
                self.tiles.entries[tile.0 as usize] = Some(entry);
            }
        }
        complete
    }

    /// Retires a source's page once it holds no tiles and no node draws
    /// the source; the next frame destroys it.
    pub(crate) fn drop_unused_page(&mut self, source: u32) {
        let drawn = self.nodes.iter().any(|n| n.source == Some(source));
        if !drawn
            && self.tiles.source_empty(source)
            && let Some(page) = self.tiles.forget_source(source)
        {
            self.tiles.retired.push(page);
        }
    }

    /// Destroys retired pages.
    pub(crate) fn destroy_retired_pages(&mut self, gpu: &mut dyn Gpu) {
        for page in std::mem::take(&mut self.tiles.retired) {
            self.images.remove_page(gpu, page);
        }
    }

    /// How a tile arriving at `time` fades in, by what of its source is
    /// resident (GL JS's `getFadeValues`, its tile retention reduced to
    /// what the pyramid holds): where descendants covered it (zooming out)
    /// it shows at once; otherwise it cross-fades from the nearest resident
    /// ancestor, or without one fades in.
    fn fade_in(&self, key: TileKey, time: f32) -> ImageFade {
        let layer = |t: TileIndex| self.tiles.layer_of(key.source, t);
        let children = key.tile.children();
        if children
            .iter()
            .any(|&c| layer(c).is_some() || c.children().iter().any(|&g| layer(g).is_some()))
        {
            return ImageFade::NONE;
        }
        let mut ancestor = key.tile.parent();
        let mut depth = 1;
        while let Some(at) = ancestor.filter(|_| depth <= FADE_DEPTH) {
            if let Some(layer) = layer(at) {
                let offset = |v: u32| (v - (v >> depth << depth)) as u16;
                return ImageFade {
                    arrived: time,
                    ancestor: Some((layer, depth, [offset(key.tile.col), offset(key.tile.row)])),
                };
            }
            ancestor = at.parent();
            depth += 1;
        }
        ImageFade {
            arrived: time,
            ancestor: None,
        }
    }
}

/// How many zooms up a fading tile looks for the ancestor it cross-fades
/// from: the row packs the tile's offsets in it in 12 bits.
const FADE_DEPTH: u8 = 12;

fn image_mesh_size(mesh: Option<&ImageMesh>) -> MeshSize {
    match mesh {
        Some(mesh) => MeshSize {
            vertices: mesh.vertices.len() as u32,
            indices: mesh.indices.len() as u32,
            props: IMAGE_ROW_WORDS as u32,
            chunks: mesh.chunks.len() as u32,
        },
        None => MeshSize {
            vertices: QUAD.len() as u32,
            indices: QUAD_INDICES.len() as u32,
            props: IMAGE_ROW_WORDS as u32,
            chunks: 1,
        },
    }
}

/// The quad an image tile on its own plane draws: its corners in tile
/// units, as f32 positions and as image units (the same for a quad).
const QUAD: [u32; 20] = {
    let e = IMAGE_EXTENT;
    let (zero, full, one) = (0f32.to_bits(), (e as f32).to_bits(), 1f32.to_bits());
    [
        zero,
        zero,
        0,
        one,
        0, //
        full,
        zero,
        e,
        one,
        e, //
        zero,
        full,
        e << 16,
        one,
        e << 16, //
        full,
        full,
        e | e << 16,
        one,
        e | e << 16,
    ]
};
/// Indices 0, 1, 2, 2, 1, 3, two to a word (`orb_layout::pack_indices`).
const QUAD_INDICES: [u32; 3] = [1 << 16, 2 | 2 << 16, 1 | 3 << 16];
const QUAD_CHUNK: Chunk = Chunk {
    sort_key: None,
    owner: orb_layout::Owner::Every,
    first_index: 0,
    index_count: 6,
    first_vertex: 0,
    bounds: TileBox {
        min: [0, 0],
        max: [IMAGE_EXTENT as i16, IMAGE_EXTENT as i16],
    },
};

/// The layers of `tile`'s same-zoom neighbours that `layer_of` finds, in
/// [`DemPrepare::neighbours`] order: what its padded border is filled from.
pub fn dem_neighbours(
    tile: TileIndex,
    layer_of: &dyn Fn(TileIndex) -> Option<u32>,
) -> [Option<u32>; 8] {
    OFFSETS.map(|offset| offset_tile(tile, offset).and_then(layer_of))
}

/// The borders ([`DemPrepare::borders`]) a tile prepared with neighbours
/// `was` prepares again for neighbours `now`: those whose neighbour changed.
pub fn changed_borders(was: [Option<u32>; 8], now: [Option<u32>; 8]) -> u8 {
    (0..8)
        .filter(|&k| was[k] != now[k])
        .fold(0, |mask, k| mask | 1 << k)
}

/// Neighbour offsets in [`DemPrepare::neighbours`] order.
const OFFSETS: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// The tile `offset` away at the same zoom, wrapping east and west as GL
/// JS's `_getNeighboringTiles` does; `None` past the poles.
fn offset_tile(tile: TileIndex, (dx, dy): (i32, i32)) -> Option<TileIndex> {
    let n = 1i64 << tile.z;
    let row = i64::from(tile.row) + i64::from(dy);
    if !(0..n).contains(&row) {
        return None;
    }
    let col = (i64::from(tile.col) + i64::from(dx)).rem_euclid(n);
    Some(TileIndex::new(tile.z, col as u32, row as u32))
}

/// The latitudes of a Web Mercator tile's top and bottom edges, in
/// degrees, which hillshade slopes shrink by.
pub fn tile_latitudes(tile: TileIndex) -> [f32; 2] {
    let n = (1u64 << tile.z) as f64;
    [tile.row, tile.row + 1].map(|row| {
        ((1.0 - 2.0 * f64::from(row) / n) * std::f64::consts::PI)
            .sinh()
            .atan()
            .to_degrees() as f32
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_wrap_east_and_west() {
        let tile = TileIndex::new(2, 0, 0);
        assert_eq!(offset_tile(tile, (-1, 0)), Some(TileIndex::new(2, 3, 0)));
        assert_eq!(offset_tile(tile, (0, -1)), None);
        assert_eq!(offset_tile(tile, (1, 1)), Some(TileIndex::new(2, 1, 1)));
    }

    /// Meshes a tile gains after the tile order was set (a relayout's, in
    /// a part of a new extent too) draw in the tile's place in it, not
    /// after every tile: no one need set the order again for them.
    #[test]
    fn meshes_added_later_keep_their_tile_s_place() {
        let Ok(mut gpu) = pollster::block_on(orb_hal_wgpu::WgpuGpu::offscreen()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let images = crate::ImageAtlas::new(&mut gpu, 4, 2, 8);
        let format = crate::TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let capacity = crate::SceneCapacity::default();
        let mut renderer = Renderer::new(&mut gpu, format, capacity, images.bindings());
        renderer.set_nodes(&mut gpu, &[crate::Family::Fill]);
        let mut square = orb_tile::LayerBuilder::new("water", 4096);
        square.begin_feature(None, orb_tile::GeometryType::Polygon);
        square.begin_part();
        square.begin_ring();
        for v in [[0, 0], [4096, 0], [4096, 4096], [0, 4096]] {
            square.push_vertex(v);
        }
        let mesh = orb_layout::fill::layout(&square.finish(), &[0]);
        let tile = |renderer: &mut Renderer, gpu: &mut orb_hal_wgpu::WgpuGpu, row| {
            let key = TileKey {
                source: 0,
                tile: TileIndex::new(2, 0, row),
            };
            renderer.add_tile(gpu, key, None).unwrap()
        };
        let (a, b, c) = (
            tile(&mut renderer, &mut gpu, 0),
            tile(&mut renderer, &mut gpu, 1),
            tile(&mut renderer, &mut gpu, 2),
        );
        let mut add = |renderer: &mut Renderer, tile, extent| {
            let data = MeshData::new(&mesh, &[], 0);
            renderer
                .add_mesh(&mut gpu, tile, (0, extent), &data, &Default::default())
                .unwrap();
        };
        for tile in [c, b, a] {
            add(&mut renderer, tile, 4096);
        }
        renderer.set_tile_order(0, &[a, b]);
        // Another mesh of `a`, and one in a part of another extent.
        add(&mut renderer, a, 4096);
        add(&mut renderer, a, 8192);
        add(&mut renderer, c, 4096);
        let parts = |tile| renderer.tiles.entry(tile).parts.clone();
        let [(_, a0), (_, a1)] = parts(a)[..] else {
            panic!("a has two parts")
        };
        let ([(_, b0)], [(_, c0)]) = (&parts(b)[..], &parts(c)[..]) else {
            panic!("b and c have one part")
        };
        assert_eq!(
            renderer.scene.node_tiles(0),
            [a0, a0, a1, *b0, *c0, *c0],
            "a's meshes, then b's, then c's, which the order leaves out"
        );
    }

    /// One source's tile order leaves another's: a relayout of the other's
    /// tiles still draws in their places in its order.
    #[test]
    fn a_source_s_tile_order_keeps_another_s() {
        let Ok(mut gpu) = pollster::block_on(orb_hal_wgpu::WgpuGpu::offscreen()) else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let images = crate::ImageAtlas::new(&mut gpu, 4, 2, 8);
        let format = crate::TargetFormat {
            color: orb_hal::TextureFormat::Rgba8Unorm,
            sample_count: 1,
        };
        let capacity = crate::SceneCapacity::default();
        let mut renderer = Renderer::new(&mut gpu, format, capacity, images.bindings());
        // A node a source, as a layer draws one source.
        renderer.set_nodes(&mut gpu, &[crate::Family::Fill, crate::Family::Fill]);
        let mut square = orb_tile::LayerBuilder::new("water", 4096);
        square.begin_feature(None, orb_tile::GeometryType::Polygon);
        square.begin_part();
        square.begin_ring();
        for v in [[0, 0], [4096, 0], [4096, 4096], [0, 4096]] {
            square.push_vertex(v);
        }
        let mesh = orb_layout::fill::layout(&square.finish(), &[0]);
        let mut tile = |renderer: &mut Renderer, source, row| {
            let key = TileKey {
                source,
                tile: TileIndex::new(2, 0, row),
            };
            let tile = renderer.add_tile(&mut gpu, key, None).unwrap();
            let data = MeshData::new(&mesh, &[], 0);
            renderer
                .add_mesh(&mut gpu, tile, (source, 4096), &data, &Default::default())
                .unwrap();
            tile
        };
        let (a, b) = (tile(&mut renderer, 0, 0), tile(&mut renderer, 0, 1));
        let (c, d) = (tile(&mut renderer, 1, 0), tile(&mut renderer, 1, 1));
        renderer.set_tile_order(1, &[c, d]);
        renderer.set_tile_order(0, &[b, a]);
        // A relayout of c, its old mesh gone and its new one in.
        let old = renderer.tiles.entry(c).meshes.clone();
        for (_, id) in old {
            renderer.remove_mesh(c, id);
        }
        let data = MeshData::new(&mesh, &[], 0);
        renderer
            .add_mesh(&mut gpu, c, (1, 4096), &data, &Default::default())
            .unwrap();
        let part = |tile| renderer.tiles.entry(tile).parts[0].1;
        assert_eq!(renderer.scene.node_tiles(0), [part(b), part(a)]);
        assert_eq!(
            renderer.scene.node_tiles(1),
            [part(c), part(d)],
            "the relayout of c draws in its place, before d"
        );
    }
}
