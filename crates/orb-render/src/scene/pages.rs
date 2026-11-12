//! Texture-array pages for image tiles: raster images and DEM
//! elevations, one tile per array layer, so one binding serves every tile
//! of a source and shaders find a tile's layer through its property row.
//!
//! A page is one or more texture arrays that share layer numbers: a DEM
//! tile has its raw elevations, a padded copy and its slopes in the same
//! layer of three arrays. Pages start at the layers a map's view needs
//! and grow when full, copying their layers into the larger arrays;
//! [`TilePage::revision`] changes so bind groups over the old arrays are
//! rebuilt. Growing and shrinking are their own calls, which a map makes
//! at rest; the growth tests move every kind of page.

use crate::scene::arena::Slots;
use orb_hal::{
    Gpu, HostImageId, Pass, TextureDesc, TextureFormat, TextureId, TextureRegion, TextureUsages,
    TextureViewDimension,
};

use crate::scene::{OutOfSpace, Resized};

/// Layers a page grows by at least.
const GROWTH: u32 = 8;

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

#[derive(Debug)]
pub(crate) struct TilePage {
    textures: Vec<(PageTexture, TextureId)>,
    capacity: u32,
    /// The layers it started with, which it doesn't shrink below.
    first: u32,
    max_layers: u32,
    layers: Slots,
    revision: u32,
    /// What moving the arrays did since the last
    /// [`take_resized`](Self::take_resized).
    resized: Resized,
}

impl TilePage {
    /// GPU bytes of the page's textures at their current layer counts.
    pub(crate) fn bytes(&self) -> u64 {
        self.textures
            .iter()
            .map(|(desc, _)| texture_bytes(desc, self.capacity))
            .sum()
    }

    /// Bytes one layer takes across the page's textures.
    pub(crate) fn layer_bytes(&self) -> u64 {
        self.textures
            .iter()
            .map(|(desc, _)| texture_bytes(desc, 1))
            .sum()
    }

    /// A page of texture arrays whose layers are allocated together, of
    /// `layers` layers to start with, or fewer where the device holds
    /// fewer in a texture ([`Gpu::texture_fits`]).
    pub(crate) fn new(gpu: &mut dyn Gpu, textures: &[PageTexture], layers: u32) -> Self {
        let max_layers = gpu.caps().limits.max_texture_array_layers;
        let mut capacity = layers.clamp(1, max_layers);
        while capacity > 1 && !(textures.iter()).all(|t| gpu.texture_fits(&desc(t, capacity))) {
            capacity /= 2;
        }
        Self {
            textures: textures
                .iter()
                .map(|&t| (t, gpu.create_texture(&desc(&t, capacity))))
                .collect(),
            capacity,
            first: capacity,
            max_layers,
            layers: Slots::new(capacity),
            revision: 0,
            resized: Resized::default(),
        }
    }

    /// The `index`th texture array, as given to [`new`](Self::new).
    pub(crate) fn texture(&self, index: usize) -> TextureId {
        self.textures[index].1
    }

    /// The size of the `index`th array's layers.
    pub(crate) fn size(&self, index: usize) -> [u32; 2] {
        let t = &self.textures[index].0;
        [t.width, t.height]
    }

    /// The `index`th array's mip levels.
    pub(crate) fn mip_levels(&self, index: usize) -> u32 {
        self.textures[index].0.mip_levels
    }

    /// Changes whenever the textures are replaced.
    pub(crate) fn revision(&self) -> u32 {
        self.revision
    }

    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// What moving the arrays did since the last call.
    pub(crate) fn take_resized(&mut self) -> Resized {
        std::mem::take(&mut self.resized)
    }

    /// A free layer, growing the page if it is full.
    pub(crate) fn alloc(&mut self, gpu: &mut dyn Gpu) -> Result<u32, OutOfSpace> {
        if let Some(layer) = self.alloc_existing() {
            return Ok(layer);
        }
        self.grow(gpu)?;
        Ok(self.alloc_existing().expect("grown"))
    }

    /// Layers handed out and not freed.
    pub(crate) fn used(&self) -> u32 {
        self.layers.used()
    }

    /// A layer the arrays already have, without growing them.
    pub(crate) fn alloc_existing(&mut self) -> Option<u32> {
        self.layers.alloc()
    }

    pub(crate) fn free(&mut self, layer: u32) {
        self.layers.free(layer);
    }

    /// Writes a layer of the `index`th array, whole.
    pub(crate) fn write(&self, gpu: &mut dyn Gpu, index: usize, layer: u32, pixels: Pixels<'_>) {
        let (desc, texture) = &self.textures[index];
        let region = TextureRegion {
            texture: *texture,
            x: 0,
            y: 0,
            layer,
            width: desc.width,
            height: desc.height,
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
        for (_, texture) in self.textures {
            gpu.destroy_texture(texture);
        }
    }

    /// Grows the arrays by half (at least [`GROWTH`] layers): pages of
    /// large tiles overshoot less than by doubling.
    /// Fails at the device's most layers, or past the bytes it allocates
    /// a texture ([`Gpu::texture_fits`], [`Gpu::try_create_texture`]),
    /// which then are the page's most.
    pub(crate) fn grow(&mut self, gpu: &mut dyn Gpu) -> Result<(), OutOfSpace> {
        let capacity = self.grown_capacity();
        if capacity <= self.capacity {
            return Err(OutOfSpace);
        }
        let fits = (self.textures.iter()).all(|(t, _)| gpu.texture_fits(&desc(t, capacity)));
        if !fits || self.resize(gpu, capacity).is_err() {
            self.max_layers = self.capacity;
            return Err(OutOfSpace);
        }
        self.layers.grow(capacity);
        Ok(())
    }

    /// The bytes growing the arrays would add, when they hold more than
    /// three quarters of their layers below the device's most: room a map
    /// at rest makes by growing them ([`grow`](Self::grow)).
    pub(crate) fn headroom(&self) -> Option<u64> {
        let capacity = self.grown_capacity();
        (capacity > self.capacity && self.layers.used() * 4 > self.capacity * 3).then(|| {
            (self.textures.iter())
                .map(|(desc, _)| texture_bytes(desc, capacity) - texture_bytes(desc, self.capacity))
                .sum()
        })
    }

    /// Whether [`grow`](Self::grow) would add layers.
    pub(crate) fn can_grow(&self) -> bool {
        self.grown_capacity() > self.capacity
    }

    fn grown_capacity(&self) -> u32 {
        (self.capacity + (self.capacity / 2).max(GROWTH)).min(self.max_layers)
    }

    /// Halves the arrays while every held layer lies in their lowest
    /// quarter, down to their first size: a view of fewer tiles gives back
    /// what a view of many grew them to (layers are handed out lowest
    /// first, so they stay low). Returns whether they shrank.
    pub(crate) fn shrink(&mut self, gpu: &mut dyn Gpu) -> bool {
        let capacity = self.shrunk_capacity();
        if capacity == self.capacity {
            return false;
        }
        if self.resize(gpu, capacity).is_err() {
            return false;
        }
        self.layers.shrink(capacity);
        true
    }

    /// Whether [`shrink`](Self::shrink) would shrink the arrays.
    pub(crate) fn would_shrink(&self) -> bool {
        self.shrunk_capacity() != self.capacity
    }

    fn shrunk_capacity(&self) -> u32 {
        super::shrunk(self.capacity, self.layers.held_below(), self.first)
    }

    /// Moves the arrays into ones of `capacity` layers, the layers held
    /// copied over. The copy is submitted at once, so writes queued before
    /// it land in the old arrays first and later ones after it. Fails,
    /// keeping the arrays, where the device refuses one.
    fn resize(&mut self, gpu: &mut dyn Gpu, capacity: u32) -> Result<(), OutOfSpace> {
        let before = gpu.counters();
        let mut made = Vec::with_capacity(self.textures.len());
        for (t, _) in &self.textures {
            match gpu.try_create_texture(&desc(t, capacity)) {
                Some(texture) => made.push(texture),
                None => {
                    for texture in made {
                        gpu.destroy_texture(texture);
                    }
                    return Err(OutOfSpace);
                }
            }
        }
        let kept = self.layers.held_below().min(capacity);
        let mut copies = Vec::with_capacity(self.textures.len());
        for ((desc, texture), moved) in self.textures.iter_mut().zip(made) {
            copies.push(Pass::CopyTexture {
                source: *texture,
                destination: moved,
                copies: (kept > 0)
                    .then_some(orb_hal::TextureCopy {
                        from: [0; 3],
                        to: [0; 3],
                        size: [desc.width, desc.height, kept],
                    })
                    .into_iter()
                    .collect(),
            });
            *texture = moved;
        }
        gpu.submit(&copies);
        for copy in copies {
            if let Pass::CopyTexture { source, .. } = copy {
                gpu.destroy_texture(source);
            }
        }
        self.capacity = capacity;
        self.revision += 1;
        self.resized.count(gpu, &before);
        Ok(())
    }
}

/// Bytes of a page texture of `layers` layers, its mip levels included.
fn texture_bytes(desc: &PageTexture, layers: u32) -> u64 {
    let mut bytes = 0;
    for level in 0..desc.mip_levels.max(1) {
        let (w, h) = ((desc.width >> level).max(1), (desc.height >> level).max(1));
        bytes += u64::from(w) * u64::from(h) * u64::from(desc.format.texel_bytes());
    }
    bytes * u64::from(layers)
}

fn desc(desc: &PageTexture, layers: u32) -> TextureDesc<'static> {
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
