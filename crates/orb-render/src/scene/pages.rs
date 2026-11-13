//! Texture-array pages for image tiles (raster images, DEM elevations) and
//! glyphs, one tile per array layer: up to [`PAGE_CHUNKS`] chunks, each
//! one or more texture arrays that share layer numbers (a DEM tile has its
//! raw elevations, a padded copy and its slopes in the same layer of three
//! arrays).
//!
//! A tile's layer word names its chunk and its layer there
//! ([`CHUNK_SHIFT`]), and shaders bind every chunk, choosing by the word.
//! The first chunk holds the layers the page started with, and chunk `k`
//! as many as the chunks before it together, so that the four hold eight
//! times the first. A page grows by its next chunk, blank, and shrinks by
//! letting go of a chunk once no tile holds a layer of it
//! ([`TilePage::shrink`]): no layer ever moves. New tiles take the lowest
//! chunk's free layers first, so the chunks a view grew the page by fill
//! last and empty as their tiles leave.
//!
//! Past its chunks, a page's last chunk grows into a larger one, copying
//! its layers: a page of one chunk (the terrain's, which every draw that
//! stands on the terrain binds, within WebGL 2's 16 textures a pipeline,
//! and the glyphs') grows only so, and keeps its size.

use crate::scene::arena::Slots;
use orb_hal::{
    Gpu, HostImageId, Pass, TextureCopy, TextureDesc, TextureFormat, TextureId, TextureRegion,
    TextureUsages, TextureViewDimension,
};

use crate::scene::{OutOfSpace, Resized};

/// One of a page's texture arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PageTexture {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// Beyond the copies a page makes when it grows.
    pub usage: TextureUsages,
    /// Mip levels; a page's growth copies them all.
    pub mip_levels: u32,
}

/// Pixels to write into a page layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pixels<'a> {
    /// Tightly packed rows in core memory, ready to store.
    Core(&'a [u8]),
    /// A host-held image, written without entering core memory;
    /// `premultiply` multiplies its colour by alpha on the way.
    Host {
        image: HostImageId,
        premultiply: bool,
    },
    /// Tightly packed rows at byte `offset` of host-held bytes (a DEM
    /// tile's texels in the buffer a job worker transferred), written
    /// without entering core memory.
    HostBytes {
        bytes: orb_hal::HostBytesId,
        offset: u64,
    },
}

/// Bytes of a page texture of `layers` layers, its mip levels included.
pub(crate) fn texture_bytes(desc: &PageTexture, layers: u32) -> u64 {
    let mut bytes = 0;
    for level in 0..desc.mip_levels.max(1) {
        let (w, h) = ((desc.width >> level).max(1), (desc.height >> level).max(1));
        bytes += u64::from(w) * u64::from(h) * u64::from(desc.format.texel_bytes());
    }
    bytes * u64::from(layers)
}

pub(crate) fn desc(desc: &PageTexture, layers: u32) -> TextureDesc<'static> {
    TextureDesc {
        label: desc.label,
        width: desc.width,
        height: desc.height,
        array_layers: layers,
        dimension: TextureViewDimension::D2Array,
        format: desc.format,
        usage: desc.usage | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        sample_count: 1,
        mip_levels: desc.mip_levels,
    }
}

/// The most chunks an image tile page has: an image node binds each, which
/// with the scene's textures and a hillshade's table fills the 16 that
/// wgpu's GL back end binds a pipeline.
pub(crate) const PAGE_CHUNKS: usize = 4;
/// A layer word's chunk lies in its bits from here up, and its layer in
/// the chunk below: a texture holds at most 2048 layers on any device.
const CHUNK_SHIFT: u32 = 11;
const LAYER_MASK: u32 = (1 << CHUNK_SHIFT) - 1;
/// Layers a chunk after the first holds at least, and the least a chunk
/// grows by when it grows in place.
const GROWTH: u32 = 8;

/// The chunk and the layer in it that a layer word names.
fn split(word: u32) -> (usize, u32) {
    ((word >> CHUNK_SHIFT) as usize, word & LAYER_MASK)
}

/// The layer word of layer `layer` of chunk `chunk`.
fn word(chunk: usize, layer: u32) -> u32 {
    (chunk as u32) << CHUNK_SHIFT | layer
}

/// One chunk of a [`TilePage`]: its texture arrays, by the page's
/// textures' order, and its layers.
#[derive(Debug)]
struct Chunk {
    textures: Vec<TextureId>,
    capacity: u32,
    layers: Slots,
}

/// A tile page (see the module's doc).
#[derive(Debug)]
pub(crate) struct TilePage {
    textures: Vec<PageTexture>,
    /// The layers its first chunk holds.
    first: u32,
    max_layers: u32,
    /// The chunks it may have.
    most: usize,
    chunks: Vec<Option<Chunk>>,
    /// Chunks from here up are ones the device refused, which the page
    /// asks for again once a chunk goes.
    refused: usize,
    revision: u32,
    /// What making chunks and moving layers did since the last
    /// [`take_resized`](Self::take_resized).
    resized: Resized,
}

impl TilePage {
    /// A page of texture arrays whose layers are allocated together, of up
    /// to `most` chunks ([`PAGE_CHUNKS`] at most), its first of `layers`
    /// layers, or fewer where the device holds fewer in a texture
    /// ([`Gpu::texture_fits`]).
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        textures: &[PageTexture],
        (layers, most): (u32, usize),
    ) -> Self {
        let max_layers = (gpu.caps().limits.max_texture_array_layers).min(LAYER_MASK + 1);
        let most = most.clamp(1, PAGE_CHUNKS);
        let mut page = Self {
            textures: textures.to_vec(),
            first: 1,
            max_layers,
            most,
            chunks: Vec::new(),
            refused: most,
            revision: 0,
            resized: Resized::default(),
        };
        page.first = page.fitting(gpu, layers);
        // Its first chunk, now: a page is made to take a tile, and has a
        // chunk, as the device makes any texture, though it refuses one.
        let capacity = page.first;
        if !page.make(gpu, 0, capacity) {
            let textures = (page.textures.iter())
                .map(|t| gpu.create_texture(&desc(t, capacity)))
                .collect();
            page.chunks = vec![Some(Chunk {
                textures,
                capacity,
                layers: Slots::new(capacity),
            })];
            page.refused = most;
        }
        page.resized = Resized::default();
        page
    }

    /// `layers`, or fewer where the device doesn't make the textures with
    /// as many: the most of its halvings that it does, at least one.
    fn fitting(&self, gpu: &dyn Gpu, layers: u32) -> u32 {
        let mut layers = layers.clamp(1, self.max_layers);
        while layers > 1 && !(self.textures.iter()).all(|t| gpu.texture_fits(&desc(t, layers))) {
            layers /= 2;
        }
        layers
    }

    /// The layers chunk `k` holds when made: the first chunk's, then as
    /// many as bring the chunks up to it to `2^k` times those (as many as
    /// the chunks before it together), within the device's limits.
    fn chunk_layers(&self, gpu: &dyn Gpu, k: usize) -> u32 {
        if k == 0 {
            return self.first;
        }
        let below: u32 = (self.chunks.iter().take(k).flatten())
            .map(|c| c.capacity)
            .sum();
        let total = self.first.saturating_mul(1 << k);
        self.fitting(gpu, total.saturating_sub(below).max(GROWTH))
    }

    /// The textures of a chunk of `capacity` layers, or `None` where the
    /// device refused one (none then made).
    fn textures_of(&self, gpu: &mut dyn Gpu, capacity: u32) -> Option<Vec<TextureId>> {
        let mut made = Vec::with_capacity(self.textures.len());
        for t in &self.textures {
            match gpu.try_create_texture(&desc(t, capacity)) {
                Some(texture) => made.push(texture),
                None => {
                    for texture in made {
                        gpu.destroy_texture(texture);
                    }
                    return None;
                }
            }
        }
        Some(made)
    }

    /// Makes chunk `k` of `capacity` layers, blank: whether the device
    /// made it.
    fn make(&mut self, gpu: &mut dyn Gpu, k: usize, capacity: u32) -> bool {
        let before = gpu.counters();
        let Some(textures) = self.textures_of(gpu, capacity) else {
            self.refused = self.refused.min(k);
            return false;
        };
        if self.chunks.len() <= k {
            self.chunks.resize_with(k + 1, || None);
        }
        self.chunks[k] = Some(Chunk {
            textures,
            capacity,
            layers: Slots::new(capacity),
        });
        self.revision += 1;
        self.resized.count(gpu, &before);
        true
    }

    /// Moves chunk `k` into textures of `capacity` layers, more than it
    /// has, the layers it holds copied over. The copy is submitted at
    /// once, so writes queued before it land in the old textures first.
    /// Fails, keeping the chunk, where the device refuses one.
    fn grow(&mut self, gpu: &mut dyn Gpu, k: usize, capacity: u32) -> Result<(), OutOfSpace> {
        let before = gpu.counters();
        let made = self.textures_of(gpu, capacity).ok_or(OutOfSpace)?;
        let chunk = self.chunks[k].as_mut().expect("a chunk to resize");
        let kept = chunk.layers.held_below();
        let copies: Vec<Pass> = (self.textures.iter().zip(&chunk.textures).zip(&made))
            .map(|((t, &source), &destination)| Pass::CopyTexture {
                source,
                destination,
                copies: (kept > 0)
                    .then_some(TextureCopy {
                        from: [0; 3],
                        to: [0; 3],
                        size: [t.width, t.height, kept],
                    })
                    .into_iter()
                    .collect(),
            })
            .collect();
        gpu.submit(&copies);
        for texture in std::mem::replace(&mut chunk.textures, made) {
            gpu.destroy_texture(texture);
        }
        chunk.layers.grow(capacity);
        chunk.capacity = capacity;
        self.revision += 1;
        self.resized.count(gpu, &before);
        Ok(())
    }

    /// GPU bytes of the page's chunks.
    pub(crate) fn bytes(&self) -> u64 {
        let layer = self.layer_bytes();
        (self.chunks.iter().flatten())
            .map(|c| u64::from(c.capacity) * layer)
            .sum()
    }

    /// Bytes one layer takes across the page's textures.
    pub(crate) fn layer_bytes(&self) -> u64 {
        self.textures.iter().map(|t| texture_bytes(t, 1)).sum()
    }

    /// Each chunk's `index`th texture array, as given to
    /// [`new`](Self::new), or `placeholder` where the page has no such
    /// chunk: what a binding of the page binds.
    pub(crate) fn textures(
        &self,
        index: usize,
        placeholder: TextureId,
    ) -> [TextureId; PAGE_CHUNKS] {
        std::array::from_fn(|k| match self.chunks.get(k) {
            Some(Some(chunk)) => chunk.textures[index],
            _ => placeholder,
        })
    }

    /// The first chunk's `index`th texture array: a page of one chunk's.
    pub(crate) fn texture(&self, index: usize) -> TextureId {
        self.chunks[0].as_ref().expect("a first chunk").textures[index]
    }

    /// The chunks the page has: each one's index, `index`th texture array
    /// and layers.
    pub(crate) fn chunks(&self, index: usize) -> impl Iterator<Item = (usize, TextureId, u32)> {
        (self.chunks.iter().enumerate())
            .filter_map(move |(k, c)| c.as_ref().map(|c| (k, c.textures[index], c.capacity)))
    }

    /// The `index`th texture array of the chunk layer word `word` lies in,
    /// and its layer there.
    pub(crate) fn texture_of(&self, index: usize, word: u32) -> (TextureId, u32) {
        let (k, layer) = split(word);
        let chunk = self.chunks[k].as_ref().expect("a held layer's chunk");
        (chunk.textures[index], layer)
    }

    /// The chunk layer word `word` lies in.
    pub(crate) fn chunk_of(word: u32) -> usize {
        split(word).0
    }

    /// The size of the `index`th array's layers.
    pub(crate) fn size(&self, index: usize) -> [u32; 2] {
        let t = &self.textures[index];
        [t.width, t.height]
    }

    /// The `index`th array's mip levels.
    pub(crate) fn mip_levels(&self, index: usize) -> u32 {
        self.textures[index].mip_levels
    }

    /// Changes whenever a chunk is made, moves or goes.
    pub(crate) fn revision(&self) -> u32 {
        self.revision
    }

    /// What making chunks and moving layers did since the last call.
    pub(crate) fn take_resized(&mut self) -> Resized {
        std::mem::take(&mut self.resized)
    }

    /// A free layer's word: the lowest of the lowest chunk with one, or
    /// with `grow` the first of a chunk made for it, or past the page's
    /// chunks one its last grows by (by half, at least [`GROWTH`]). Fails
    /// at the device's limits, or where it refuses a texture.
    pub(crate) fn alloc(&mut self, gpu: &mut dyn Gpu, grow: bool) -> Result<u32, OutOfSpace> {
        for (k, chunk) in self.chunks.iter_mut().enumerate() {
            if let Some(layer) = chunk.as_mut().and_then(|c| c.layers.alloc()) {
                return Ok(word(k, layer));
            }
        }
        if !grow {
            return Err(OutOfSpace);
        }
        let missing = (0..self.refused).find(|&k| self.chunks.get(k).is_none_or(Option::is_none));
        let k = match missing {
            Some(k) => {
                let capacity = self.chunk_layers(gpu, k);
                if !self.make(gpu, k, capacity) {
                    return Err(OutOfSpace);
                }
                k
            }
            None => {
                let k = self.chunks.len().checked_sub(1).ok_or(OutOfSpace)?;
                let old = self.chunks[k].as_ref().expect("the last chunk").capacity;
                let capacity = self.fitting(gpu, old + (old / 2).max(GROWTH));
                if capacity <= old {
                    return Err(OutOfSpace);
                }
                self.grow(gpu, k, capacity)?;
                k
            }
        };
        let chunk = self.chunks[k].as_mut().expect("made");
        let layer = chunk.layers.alloc().expect("a new chunk has room");
        Ok(word(k, layer))
    }

    /// Layers its chunks hold.
    pub(crate) fn capacity(&self) -> u32 {
        self.chunks.iter().flatten().map(|c| c.capacity).sum()
    }

    /// Layers handed out and not freed.
    pub(crate) fn used(&self) -> u32 {
        self.chunks.iter().flatten().map(|c| c.layers.used()).sum()
    }

    pub(crate) fn free(&mut self, word: u32) {
        let (k, layer) = split(word);
        let chunk = self.chunks[k].as_mut().expect("a held layer's chunk");
        chunk.layers.free(layer);
    }

    /// Writes a layer of the `index`th array, whole.
    pub(crate) fn write(&self, gpu: &mut dyn Gpu, index: usize, word: u32, pixels: Pixels<'_>) {
        let (texture, layer) = self.texture_of(index, word);
        let t = &self.textures[index];
        let region = TextureRegion {
            texture,
            x: 0,
            y: 0,
            layer,
            width: t.width,
            height: t.height,
            level: 0,
        };
        match pixels {
            Pixels::Core(bytes) => gpu.write_texture(&region, bytes),
            Pixels::Host { image, premultiply } => {
                gpu.write_texture_from_host(&region, image, premultiply)
            }
            Pixels::HostBytes { bytes, offset } => {
                gpu.write_texture_from_host_bytes(&region, bytes, offset)
            }
        }
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        for chunk in self.chunks.into_iter().flatten() {
            for texture in chunk.textures {
                gpu.destroy_texture(texture);
            }
        }
    }

    /// Lets go of the chunks no layer is held in, copying nothing: the
    /// layers held stay where they are, so a chunk goes once its last tile
    /// does. [`alloc`](Self::alloc) hands out the lowest chunk's layers
    /// first, so the higher chunks a view grew the page by take tiles only
    /// while the lower ones are full, and empty as their tiles leave. A
    /// page of one chunk keeps it.
    pub(crate) fn shrink(&mut self, gpu: &mut dyn Gpu) {
        if self.most == 1 {
            return;
        }
        for slot in &mut self.chunks {
            if let Some(chunk) = slot.take_if(|c| c.layers.used() == 0) {
                for texture in chunk.textures {
                    gpu.destroy_texture(texture);
                }
                self.revision += 1;
                // A chunk refused before may fit now that this one went.
                self.refused = self.most;
            }
        }
        while self.chunks.last().is_some_and(Option::is_none) {
            self.chunks.pop();
        }
    }
}
