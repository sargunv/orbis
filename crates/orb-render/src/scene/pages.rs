//! Texture-array pages for image tiles: raster images and DEM
//! elevations, one tile per array layer, so one binding serves every tile
//! of a source and shaders find a tile's layer through its property row.
//!
//! A page is one or more texture arrays that share layer numbers: a DEM
//! tile has its raw elevations, a padded copy and its slopes in the same
//! layer of three arrays. Pages start small and double when full, copying
//! their layers into the larger arrays; [`TilePage::revision`] changes so
//! bind groups over the old arrays are rebuilt.

use crate::scene::arena::Slots;
use orb_hal::{
    Gpu, HostImageId, Pass, TextureDesc, TextureFormat, TextureId, TextureRegion, TextureUsages,
    TextureViewDimension,
};

use crate::scene::OutOfSpace;

/// Layers a page holds before it first grows.
const INITIAL_LAYERS: u32 = 8;

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
}

#[derive(Debug)]
pub(crate) struct TilePage {
    textures: Vec<(PageTexture, TextureId)>,
    capacity: u32,
    max_layers: u32,
    layers: Slots,
    revision: u32,
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

    /// A page of texture arrays whose layers are allocated together.
    pub(crate) fn new(gpu: &mut dyn Gpu, textures: &[PageTexture]) -> Self {
        let max_layers = gpu.caps().limits.max_texture_array_layers;
        let capacity = INITIAL_LAYERS.min(max_layers);
        Self {
            textures: textures
                .iter()
                .map(|&t| (t, create(gpu, &t, capacity)))
                .collect(),
            capacity,
            max_layers,
            layers: Slots::new(capacity),
            revision: 0,
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
        }
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        for (_, texture) in self.textures {
            gpu.destroy_texture(texture);
        }
    }

    /// Grows the arrays by half (at least 8 layers): pages of large tiles
    /// overshoot less than by doubling. The copy is submitted at once, so
    /// writes queued before it land in the old arrays first and later ones
    /// after it.
    pub(crate) fn grow(&mut self, gpu: &mut dyn Gpu) -> Result<(), OutOfSpace> {
        if self.capacity >= self.max_layers {
            return Err(OutOfSpace);
        }
        let capacity =
            (self.capacity + (self.capacity / 2).max(INITIAL_LAYERS)).min(self.max_layers);
        let mut copies = Vec::with_capacity(self.textures.len());
        for (desc, texture) in &mut self.textures {
            let grown = create(gpu, desc, capacity);
            copies.push(Pass::CopyTexture {
                source: *texture,
                destination: grown,
                copies: vec![orb_hal::TextureCopy {
                    from: [0; 3],
                    to: [0; 3],
                    size: [desc.width, desc.height, self.capacity],
                }],
            });
            *texture = grown;
        }
        gpu.submit(&copies);
        for copy in copies {
            if let Pass::CopyTexture { source, .. } = copy {
                gpu.destroy_texture(source);
            }
        }
        self.capacity = capacity;
        self.layers.grow(capacity);
        self.revision += 1;
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

fn create(gpu: &mut dyn Gpu, desc: &PageTexture, layers: u32) -> TextureId {
    gpu.create_texture(&TextureDesc {
        label: desc.label,
        width: desc.width,
        height: desc.height,
        array_layers: layers,
        dimension: TextureViewDimension::D2Array,
        format: desc.format,
        usage: desc.usage | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
        sample_count: 1,
        mip_levels: desc.mip_levels,
        view_formats: &[],
    })
}
