//! An atlas's texels in core memory, uploaded one rectangle per layer a
//! frame.
//!
//! Two copies into one layer of a texture in one submission can race on
//! Apple GPUs and lose texels (orb-hal-wgpu's `texture_writes`), and a
//! frame of tile arrivals adds dozens of bitmaps to an atlas's layer.
//! Atlases write their bitmaps here instead, and [`AtlasTexels::flush`]
//! uploads the rectangle around each layer's changes in one write: the
//! GPU never takes two writes into a layer in a frame, and a burst costs
//! one upload a layer, not one a bitmap.

use orb_hal::{Gpu, TextureId, TextureRegion};

#[derive(Debug, Default)]
struct Layer {
    /// Its texels, rows of the atlas's width; empty until first written.
    texels: Vec<u8>,
    /// The rectangle written since the last flush, `[x0, y0, x1, y1)`.
    dirty: Option<[u32; 4]>,
}

/// The texels of an atlas's layers, and what changed since they were last
/// uploaded.
#[derive(Debug)]
pub(crate) struct AtlasTexels {
    width: u32,
    height: u32,
    /// Bytes per texel.
    texel: usize,
    layers: Vec<Layer>,
}

impl AtlasTexels {
    /// The texels of `width` × `height` layers of `texel` bytes each.
    pub(crate) fn new(width: u32, height: u32, texel: usize) -> Self {
        Self {
            width,
            height,
            texel,
            layers: Vec::new(),
        }
    }

    /// Writes `data`, rows of `size[0]` texels, at `at` in `layer`; it
    /// reaches the GPU at the next [`flush`](Self::flush).
    pub(crate) fn write(&mut self, layer: u32, at: [u32; 2], size: [u32; 2], data: &[u8]) {
        let (width, texel) = (self.width as usize, self.texel);
        assert!(at[0] + size[0] <= self.width && at[1] + size[1] <= self.height);
        assert_eq!(data.len(), size[0] as usize * size[1] as usize * texel);
        if self.layers.len() <= layer as usize {
            self.layers.resize_with(layer as usize + 1, Layer::default);
        }
        let l = &mut self.layers[layer as usize];
        if l.texels.is_empty() {
            l.texels = vec![0; width * self.height as usize * texel];
        }
        let row = size[0] as usize * texel;
        for (y, line) in data.chunks_exact(row.max(1)).enumerate() {
            let start = ((at[1] as usize + y) * width + at[0] as usize) * texel;
            l.texels[start..start + row].copy_from_slice(line);
        }
        let rect = [at[0], at[1], at[0] + size[0], at[1] + size[1]];
        l.dirty = Some(match l.dirty {
            Some(d) => [
                d[0].min(rect[0]),
                d[1].min(rect[1]),
                d[2].max(rect[2]),
                d[3].max(rect[3]),
            ],
            None => rect,
        });
    }

    /// Gives `layer`'s core memory back: nothing in it is drawn any more.
    /// What the GPU holds there stays until a write covers it.
    pub(crate) fn forget(&mut self, layer: u32) {
        if let Some(l) = self.layers.get_mut(layer as usize) {
            *l = Layer::default();
        }
    }

    /// Uploads each layer's changes since the last flush into `texture`
    /// (level 0), one write a layer.
    pub(crate) fn flush(&mut self, gpu: &mut dyn Gpu, texture: TextureId) {
        let (width, texel) = (self.width as usize, self.texel);
        let mut rows = Vec::new();
        for (layer, l) in self.layers.iter_mut().enumerate() {
            let Some([x0, y0, x1, y1]) = l.dirty.take() else {
                continue;
            };
            let row = (x1 - x0) as usize * texel;
            rows.clear();
            for y in y0..y1 {
                let start = (y as usize * width + x0 as usize) * texel;
                rows.extend_from_slice(&l.texels[start..start + row]);
            }
            gpu.write_texture(
                &TextureRegion {
                    texture,
                    x: x0,
                    y: y0,
                    layer: layer as u32,
                    width: x1 - x0,
                    height: y1 - y0,
                    level: 0,
                },
                &rows,
            );
        }
    }
}
