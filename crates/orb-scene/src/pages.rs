//! Texture-array pages for image tiles (plan §8.2): raster images and DEM
//! elevations, one tile per array layer, so one binding serves every tile
//! of a source and shaders find a tile's layer through its property row.
//!
//! A page is one or more texture arrays that share layer numbers: a DEM
//! tile has its raw elevations, a padded copy and its slopes in the same
//! layer of three arrays. Pages start small and double when full, copying
//! their layers into the larger arrays; [`TilePage::revision`] changes so
//! bind groups over the old arrays are rebuilt.

use orb_hal::{
    Gpu, HostImageId, Pass, TextureDesc, TextureFormat, TextureId, TextureRegion, TextureUsages,
    TextureViewDimension,
};

use crate::OutOfSpace;

/// Layers a page holds before it first grows.
const INITIAL_LAYERS: u32 = 8;

/// One of a page's texture arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageTexture {
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
    /// A host-held image, written without entering core memory (plan
    /// §10.4); `premultiply` multiplies its colour by alpha on the way.
    Host {
        image: HostImageId,
        premultiply: bool,
    },
}

#[derive(Debug)]
pub struct TilePage {
    textures: Vec<(PageTexture, TextureId)>,
    capacity: u32,
    max_layers: u32,
    /// Layers below `next` have been handed out; freed ones wait in `free`.
    next: u32,
    free: Vec<u32>,
    revision: u32,
}

impl TilePage {
    /// A page of texture arrays whose layers are allocated together.
    pub fn new(gpu: &mut dyn Gpu, textures: &[PageTexture]) -> Self {
        let max_layers = gpu.caps().limits.max_texture_array_layers;
        let capacity = INITIAL_LAYERS.min(max_layers);
        Self {
            textures: textures
                .iter()
                .map(|&t| (t, create(gpu, &t, capacity)))
                .collect(),
            capacity,
            max_layers,
            next: 0,
            free: Vec::new(),
            revision: 0,
        }
    }

    /// The `index`th texture array, as given to [`new`](Self::new).
    pub fn texture(&self, index: usize) -> TextureId {
        self.textures[index].1
    }

    /// The size of the `index`th array's layers.
    pub fn size(&self, index: usize) -> [u32; 2] {
        let t = &self.textures[index].0;
        [t.width, t.height]
    }

    /// Changes whenever the textures are replaced.
    /// The `index`th array's mip levels.
    pub fn mip_levels(&self, index: usize) -> u32 {
        self.textures[index].0.mip_levels
    }

    pub fn revision(&self) -> u32 {
        self.revision
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Layers in use.
    pub fn len(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A free layer, growing the page if it is full.
    pub fn alloc(&mut self, gpu: &mut dyn Gpu) -> Result<u32, OutOfSpace> {
        if let Some(layer) = self.free.pop() {
            return Ok(layer);
        }
        if self.next == self.capacity {
            self.grow(gpu)?;
        }
        self.next += 1;
        Ok(self.next - 1)
    }

    pub fn free(&mut self, layer: u32) {
        debug_assert!(layer < self.next && !self.free.contains(&layer));
        self.free.push(layer);
    }

    /// Writes a layer of the `index`th array, whole.
    pub fn write(&self, gpu: &mut dyn Gpu, index: usize, layer: u32, pixels: Pixels<'_>) {
        let (desc, texture) = &self.textures[index];
        let region = TextureRegion {
            texture: *texture,
            x: 0,
            y: 0,
            layer,
            width: desc.width,
            height: desc.height,
        };
        match pixels {
            Pixels::Core(bytes) => gpu.write_texture(&region, bytes),
            Pixels::Host { image, premultiply } => {
                gpu.write_texture_from_host(&region, image, premultiply)
            }
        }
    }

    pub fn destroy(self, gpu: &mut dyn Gpu) {
        for (_, texture) in self.textures {
            gpu.destroy_texture(texture);
        }
    }

    /// Doubles the arrays. The copy is submitted at once, so writes queued
    /// before it land in the old arrays first and later ones after it.
    fn grow(&mut self, gpu: &mut dyn Gpu) -> Result<(), OutOfSpace> {
        if self.capacity >= self.max_layers {
            return Err(OutOfSpace);
        }
        let capacity = (self.capacity * 2).min(self.max_layers);
        let mut copies = Vec::with_capacity(self.textures.len());
        for (desc, texture) in &mut self.textures {
            let grown = create(gpu, desc, capacity);
            copies.push(Pass::CopyTextureLayers {
                source: *texture,
                destination: grown,
                layers: self.capacity,
            });
            *texture = grown;
        }
        gpu.submit(&copies);
        for copy in copies {
            if let Pass::CopyTextureLayers { source, .. } = copy {
                gpu.destroy_texture(source);
            }
        }
        self.capacity = capacity;
        self.revision += 1;
        Ok(())
    }
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
    })
}
