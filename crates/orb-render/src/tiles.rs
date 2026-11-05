//! Tiles the renderer holds (plan §5.4): what an upload made of a tile,
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

use std::collections::HashMap;
use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_layout::{Chunk, TileBox};
use orb_scene::{
    MeshData, MeshId, MeshSize, OutOfSpace, Pixels, SymbolMeshData, TileId, TileWarp, Words,
};

use crate::Renderer;
use crate::image::{DemPrepare, IMAGE_EXTENT, IMAGE_ROW_WORDS, ImageFade, PageId, PageKind};

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
    /// Planar patches it draws through on a globe: cells per side and their
    /// words (`orb_scene::WARP_PATCH_WORDS` each).
    pub patches: Option<(u32, &'a [f32])>,
    /// When it arrived, in the frame's milliseconds, if it fades in: it
    /// then cross-fades from its nearest resident ancestor, unless resident
    /// descendants covered it.
    pub fade_from: Option<f32>,
    /// A DEM tile's unpack vector.
    pub unpack: Option<[f32; 4]>,
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
}

#[derive(Debug)]
struct Image {
    layer: u32,
    mesh: Option<Arc<ImageMesh>>,
    fade: ImageFade,
}

/// A source's image page and its resident tiles.
#[derive(Debug)]
struct Source {
    page: PageId,
    tiles: HashMap<TileIndex, GpuTile>,
}

/// The renderer's tiles.
#[derive(Debug, Default)]
pub(crate) struct Tiles {
    entries: Vec<Option<Entry>>,
    free: Vec<u32>,
    sources: HashMap<u32, Source>,
    /// Pages no source uses any more, for the next frame to destroy.
    retired: Vec<PageId>,
}

impl Tiles {
    fn entry(&self, tile: GpuTile) -> &Entry {
        self.entries[tile.0 as usize]
            .as_ref()
            .expect("a released tile")
    }

    fn insert(&mut self, entry: Entry) -> GpuTile {
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
    /// and the scene's pools grow within their limits.
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
            None => {
                let page = self.images.create_page(gpu, kind, size);
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
                        tiles: HashMap::new(),
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
        let fits =
            self.scene.fits(sizes.iter().copied()) || (grow && self.scene.grow_for(gpu, &sizes));
        if !fits {
            return Err(UploadError::NoRoom);
        }
        let layer = if grow {
            self.images.alloc_layer(gpu, page).ok()
        } else {
            self.images.alloc_existing_layer(page)
        }
        .ok_or(UploadError::NoRoom)?;
        self.images.write_layer(gpu, page, layer, build.pixels);

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
            }),
            warp: TileWarp::NONE,
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

        if let Some(unpack) = build.unpack {
            let layer_of = |t: TileIndex| self.tiles.layer_of(key.source, t);
            for job in dem_prepares(key.tile, layer, unpack, &layer_of) {
                self.images.prepare_dem(page, job);
            }
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
    /// in every node, and its patches.
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
        let image = entry.image.as_mut().expect("image tiles hold an image");
        image.mesh = mesh;
        let nodes: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| self.nodes[i as usize].source == Some(entry.key.source))
            .collect();
        let mut result = nodes
            .iter()
            .try_for_each(|&node| self.add_image_node(gpu, &mut entry, node));
        if let (Ok(()), Some((cells, patches))) = (&result, patches) {
            result = self
                .scene
                .add_warp(gpu, cells, patches)
                .map(|warp| entry.warp = warp);
        }
        self.tiles.entries[tile.0 as usize] = Some(entry);
        result
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
        let part = self.part(tile, extent);
        let id = self.scene.insert_mesh(gpu, part, node, data)?;
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
        self.scene.fits(meshes.iter().copied())
    }

    /// Grows the pools, within their limits, until meshes of these sizes
    /// fit; whether they do.
    pub fn grow_for(&mut self, gpu: &mut dyn Gpu, meshes: &[MeshSize]) -> bool {
        self.scene.grow_for(gpu, meshes)
    }

    /// Grows the pools once, within their limits; whether they grew.
    pub fn grow(&mut self, gpu: &mut dyn Gpu) -> bool {
        self.scene.grow(gpu)
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

    fn entry_mut(&mut self, tile: GpuTile) -> &mut Entry {
        self.tiles.entries[tile.0 as usize]
            .as_mut()
            .expect("a released tile")
    }

    /// A tile's scene tile of layer extent `extent`, made if need be.
    fn part(&mut self, tile: GpuTile, extent: u32) -> TileId {
        if let Some(&(_, id)) = self.entry_mut(tile).parts.iter().find(|p| p.0 == extent) {
            return id;
        }
        let id = self.scene.insert_tile();
        self.entry_mut(tile).parts.push((extent, id));
        id
    }

    /// The scene tiles a tile places, one per layer extent.
    pub fn tile_parts(&self, tile: GpuTile) -> &[(u32, TileId)] {
        &self.tiles.entry(tile).parts
    }

    /// The planar patches a tile draws through, or [`TileWarp::NONE`].
    pub fn tile_warp(&self, tile: GpuTile) -> TileWarp {
        self.tiles.entry(tile).warp
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
        let [top, bottom] = [tile.row, tile.row + 1].map(|row| latitude(tile.z, row));
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
            for tile in self.tiles.of_source(source) {
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
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 1, 3];
const QUAD_CHUNK: Chunk = Chunk {
    sort_key: None,
    first_index: 0,
    index_count: 6,
    bounds: TileBox {
        min: [0, 0],
        max: [IMAGE_EXTENT as i16, IMAGE_EXTENT as i16],
    },
};

/// The prepare jobs a DEM tile's arrival at `layer` starts: its own, and
/// its resident neighbours', whose borders it fills. `layer_of` finds a
/// resident tile's layer.
pub fn dem_prepares(
    tile: TileIndex,
    layer: u32,
    unpack: [f32; 4],
    layer_of: &dyn Fn(TileIndex) -> Option<u32>,
) -> Vec<DemPrepare> {
    let arrived = (tile, layer);
    let mut jobs = vec![DemPrepare {
        layer,
        neighbours: neighbours(layer_of, tile, arrived),
        zoom: f32::from(tile.z),
        unpack,
    }];
    for offset in OFFSETS {
        let Some(neighbour) = offset_tile(tile, offset) else {
            continue;
        };
        if let Some(layer) = layer_of(neighbour) {
            jobs.push(DemPrepare {
                layer,
                neighbours: neighbours(layer_of, neighbour, arrived),
                zoom: f32::from(neighbour.z),
                unpack,
            });
        }
    }
    jobs
}

/// Which of a DEM tile's neighbours are resident (`resident`), a bit
/// each: its prepared borders come from them.
pub fn resident_neighbours(tile: TileIndex, resident: &dyn Fn(TileIndex) -> bool) -> u8 {
    OFFSETS
        .iter()
        .enumerate()
        .filter(|&(_, &offset)| offset_tile(tile, offset).is_some_and(resident))
        .fold(0, |bits, (i, _)| bits | 1 << i)
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

/// The layers of `tile`'s resident neighbours, counting `arrived`, a tile
/// not yet resident.
fn neighbours(
    layer_of: &dyn Fn(TileIndex) -> Option<u32>,
    tile: TileIndex,
    arrived: (TileIndex, u32),
) -> [Option<u32>; 8] {
    OFFSETS.map(|offset| {
        let neighbour = offset_tile(tile, offset)?;
        if neighbour == arrived.0 {
            return Some(arrived.1);
        }
        layer_of(neighbour)
    })
}

/// The latitude of a row edge of the Web Mercator tile grid, in degrees.
fn latitude(z: u8, row: u32) -> f32 {
    let n = (1u64 << z) as f64;
    ((1.0 - 2.0 * f64::from(row) / n) * std::f64::consts::PI)
        .sinh()
        .atan()
        .to_degrees() as f32
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
}
