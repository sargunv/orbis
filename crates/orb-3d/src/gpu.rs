//! A [`Model`] packed for the GPU (`model.wgsl`): vertex and material
//! words, indices grouped by how they draw, and its images as one texture
//! array. [`GpuModel::to_words`] carries it across web workers in one
//! transferable buffer.

use std::ops::Range;

use crate::{Alpha, Model, TextureRef, Wrap};

/// Words per vertex: position (3 × f32), octahedral normal (2 × snorm16),
/// texture coordinates (2 × f32), linear RGBA8 colour, material.
pub const VERTEX_WORDS: usize = 8;
/// Words per material; see [`GpuModel::materials`].
pub const MATERIAL_WORDS: usize = 16;
/// A material texture slot without a texture.
pub const NO_LAYER: u32 = 0xff;
/// The texture array's largest layer side.
pub const MAX_TEXTURE: u32 = 1024;
/// The texture array's budget, bytes with mips: many large images share
/// smaller layers.
pub const TEXTURE_BUDGET: u64 = 64 << 20;

/// Material flags (word 13, from bit 8).
pub const UNLIT: u32 = 1 << 8;
pub const MASK: u32 = 1 << 9;
pub const BLEND: u32 = 1 << 10;
pub const DOUBLE_SIDED: u32 = 1 << 11;

/// How a range of triangles draws. Opaque classes never discard a
/// fragment, which keeps a tile-based GPU's hidden-surface removal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawClass {
    /// Opaque, back faces culled by the rasterizer.
    Opaque,
    /// Opaque, both faces.
    DoubleSided,
    /// Alpha-masked: texels under the cutoff, and a single-sided
    /// material's back faces, discarded.
    Mask,
    /// Blended over what's behind, without writing depth; a single-sided
    /// material's back faces discarded.
    Blend,
}

impl DrawClass {
    pub const ALL: [Self; 4] = [Self::Opaque, Self::DoubleSided, Self::Mask, Self::Blend];
}

/// A model ready to upload.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuModel {
    /// [`VERTEX_WORDS`] per vertex.
    pub vertices: Vec<u32>,
    /// [`MATERIAL_WORDS`] per material: base colour (linear RGBA), emissive
    /// (linear RGB), alpha cutoff, metallic, roughness, normal scale,
    /// occlusion strength, texture layers (base colour, metallic-roughness,
    /// normal, occlusion in bytes), the emissive layer in the low byte
    /// with flags above, then wrap modes (two bits per axis per slot, base
    /// colour first: 0 repeat, 1 clamp, 2 mirror).
    pub materials: Vec<u32>,
    /// Model-local vertex indices, the [`DrawClass`] ranges in order.
    pub indices: Vec<u32>,
    /// Index ranges per [`DrawClass::ALL`].
    pub draws: [Range<u32>; 4],
    /// Texture layers' side (0 without textures), and each layer's RGBA8
    /// texels with their mips.
    pub texture_size: u32,
    pub layers: Vec<Vec<u8>>,
    /// Metres east, north and up, min and max.
    pub bounds: [[f32; 3]; 2],
}

/// Packs a model. `images` are its decoded images by index (width,
/// height, RGBA8); those that failed to decode leave their slots empty.
pub fn pack(model: &Model, images: &[Option<(u32, u32, Vec<u8>)>]) -> GpuModel {
    // Decoded images become layers in order.
    let mut layer_of = vec![NO_LAYER; images.len()];
    let mut decoded: Vec<(u32, u32, &[u8])> = Vec::new();
    for (i, image) in images.iter().enumerate() {
        if let Some((w, h, rgba)) = image
            && *w > 0
            && *h > 0
            && rgba.len() == (*w * *h * 4) as usize
            && decoded.len() < NO_LAYER as usize
        {
            layer_of[i] = decoded.len() as u32;
            decoded.push((*w, *h, rgba));
        }
    }
    let mut max = MAX_TEXTURE;
    while max > 64 && decoded.len() as u64 * u64::from(max * max) * 4 * 4 / 3 > TEXTURE_BUDGET {
        max /= 2;
    }
    let (texture_size, layers) = if decoded.is_empty() {
        (0, Vec::new())
    } else {
        crate::texture_layers(&decoded, max)
    };

    let slot = |r: Option<TextureRef>| -> (u32, u32) {
        let Some(texture) = r.and_then(|r| model.textures.get(r.texture)) else {
            return (NO_LAYER, 0);
        };
        let layer = texture
            .image
            .and_then(|i| layer_of.get(i).copied())
            .unwrap_or(NO_LAYER);
        let wrap = |w: Wrap| match w {
            Wrap::Repeat => 0,
            Wrap::Clamp => 1,
            Wrap::Mirror => 2,
        };
        (layer, wrap(texture.wrap[0]) | wrap(texture.wrap[1]) << 2)
    };
    let mut materials = Vec::with_capacity(model.materials.len().max(1) * MATERIAL_WORDS);
    let defaults = [crate::Material::default()];
    let list = if model.materials.is_empty() {
        &defaults[..]
    } else {
        &model.materials[..]
    };
    for m in list {
        let f = f32::to_bits;
        let slots = [
            slot(m.base_texture),
            slot(m.metallic_roughness_texture),
            slot(m.normal_texture),
            slot(m.occlusion_texture),
            slot(m.emissive_texture),
        ];
        let (cutoff, alpha) = match m.alpha {
            Alpha::Opaque => (0.0, 0),
            Alpha::Mask(c) => (c, MASK),
            Alpha::Blend => (0.0, BLEND),
        };
        let flags =
            alpha | if m.unlit { UNLIT } else { 0 } | if m.double_sided { DOUBLE_SIDED } else { 0 };
        materials.extend(m.base_color.map(f));
        materials.extend(m.emissive.map(f));
        materials.extend([
            f(cutoff),
            f(m.metallic),
            f(m.roughness),
            f(m.normal_scale),
            f(m.occlusion_strength),
            slots[0].0 | slots[1].0 << 8 | slots[2].0 << 16 | slots[3].0 << 24,
            slots[4].0 | flags,
            slots
                .iter()
                .enumerate()
                .fold(0, |w, (i, s)| w | s.1 << (4 * i)),
            0,
        ]);
    }

    let mut vertices = Vec::with_capacity(model.vertices.len() * VERTEX_WORDS);
    for v in &model.vertices {
        let [x, y, z] = v.position.map(f32::to_bits);
        let [u, w] = v.uv.map(f32::to_bits);
        vertices.extend([
            x,
            y,
            z,
            octahedral(v.normal),
            u,
            w,
            u32::from_le_bytes(v.color),
            v.material,
        ]);
    }

    let class = |material: u32| {
        let m = list.get(material as usize).unwrap_or(&defaults[0]);
        match (m.alpha, m.double_sided) {
            (Alpha::Blend, _) => 3,
            (Alpha::Mask(_), _) => 2,
            (_, true) => 1,
            _ => 0,
        }
    };
    let mut buckets: [Vec<u32>; 4] = Default::default();
    for t in model.indices.as_chunks::<3>().0 {
        let material = model.vertices[t[0] as usize].material;
        buckets[class(material)].extend_from_slice(t);
    }
    let mut indices = Vec::with_capacity(model.indices.len());
    let draws = buckets.map(|b| {
        let start = indices.len() as u32;
        indices.extend(b);
        start..indices.len() as u32
    });
    GpuModel {
        vertices,
        materials,
        indices,
        draws,
        texture_size,
        layers,
        bounds: model.bounds,
    }
}

/// A unit vector as two snorm16s on the octahedron (zero for none).
fn octahedral(n: [f32; 3]) -> u32 {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    if l1 == 0.0 {
        return 0;
    }
    let (mut x, mut y) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * ox.signum();
        y = (1.0 - ox.abs()) * oy.signum();
    }
    let s = |v: f32| ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16) as u16;
    u32::from(s(x)) | u32::from(s(y)) << 16
}

impl GpuModel {
    /// Vertices in the model.
    pub fn vertex_count(&self) -> u32 {
        (self.vertices.len() / VERTEX_WORDS) as u32
    }

    /// The bytes it takes on the GPU: vertex, material and index words and
    /// its texture layers.
    pub fn gpu_bytes(&self) -> u64 {
        let words = self.vertices.len() + self.materials.len() + self.indices.len();
        words as u64 * 4 + self.layers.iter().map(|l| l.len() as u64).sum::<u64>()
    }

    /// One word stream: a header, then every part.
    pub fn to_words(&self) -> Vec<u32> {
        let mut out = vec![
            self.vertices.len() as u32,
            self.materials.len() as u32,
            self.indices.len() as u32,
            self.texture_size,
            self.layers.len() as u32,
        ];
        out.extend(self.draws.iter().map(|r| r.end));
        out.extend(self.bounds.iter().flatten().map(|v| v.to_bits()));
        out.extend(&self.vertices);
        out.extend(&self.materials);
        out.extend(&self.indices);
        for layer in &self.layers {
            out.push(layer.len() as u32 / 4);
            out.extend(
                layer
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|t| u32::from_le_bytes(*t)),
            );
        }
        out
    }

    /// [`to_words`](Self::to_words)'s inverse.
    pub fn from_words(words: &[u32]) -> Result<Self, String> {
        let bad = || "malformed packed model".to_string();
        let mut at = 0;
        let mut take = |n: usize| -> Result<&[u32], String> {
            let part = words.get(at..at + n).ok_or_else(bad)?;
            at += n;
            Ok(part)
        };
        let header = take(15)?.to_vec();
        let [v, m, i, size, layers] = [0, 1, 2, 3, 4].map(|k| header[k] as usize);
        let ends = [header[5], header[6], header[7], header[8]];
        let b: Vec<f32> = header[9..15].iter().map(|w| f32::from_bits(*w)).collect();
        let vertices = take(v)?.to_vec();
        let materials = take(m)?.to_vec();
        let indices = take(i)?.to_vec();
        let mut out_layers = Vec::with_capacity(layers);
        for _ in 0..layers {
            let texels = take(1)?[0] as usize;
            out_layers.push(take(texels)?.iter().flat_map(|w| w.to_le_bytes()).collect());
        }
        if ends[3] as usize != indices.len() || !ends.is_sorted() {
            return Err(bad());
        }
        Ok(Self {
            vertices,
            materials,
            indices,
            draws: [
                0..ends[0],
                ends[0]..ends[1],
                ends[1]..ends[2],
                ends[2]..ends[3],
            ],
            texture_size: size as u32,
            layers: out_layers,
            bounds: [[b[0], b[1], b[2]], [b[3], b[4], b[5]]],
        })
    }
}
