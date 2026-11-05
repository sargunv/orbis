//! 3D content (plan §9.4, M8): glTF models as one stream of triangles in
//! metres on a local east, north, up frame, their materials and images,
//! ready for the renderer. Below the runtime: no I/O, threads or GPU; the
//! runtime fetches documents, buffers and images, and the host decodes the
//! images ([`gltf`]).

#![forbid(unsafe_code)]

pub mod gltf;
pub mod gpu;
pub mod load;

/// A model: triangles in metres, east, north and up from its origin.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    /// Three per triangle, counter-clockwise from outside.
    pub indices: Vec<u32>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    /// Encoded, for the host to decode; textures name them by index.
    pub images: Vec<ImageSource>,
    /// The vertices' extent, min and max per axis.
    pub bounds: [[f32; 3]; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    /// Unit, or zero where the model gives none.
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// Linear RGBA, multiplying the base colour (glTF `COLOR_0`).
    pub color: [u8; 4],
    /// Index into [`Model::materials`].
    pub material: u32,
}

/// A metallic-roughness material (glTF 3.9).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Linear RGBA.
    pub base_color: [f32; 4],
    pub base_texture: Option<TextureRef>,
    pub metallic: f32,
    pub roughness: f32,
    /// Roughness in green, metalness in blue.
    pub metallic_roughness_texture: Option<TextureRef>,
    pub normal_texture: Option<TextureRef>,
    pub normal_scale: f32,
    pub occlusion_texture: Option<TextureRef>,
    pub occlusion_strength: f32,
    /// Linear RGB, times `KHR_materials_emissive_strength`.
    pub emissive: [f32; 3],
    pub emissive_texture: Option<TextureRef>,
    pub alpha: Alpha,
    pub double_sided: bool,
    /// `KHR_materials_unlit`: the base colour as it is.
    pub unlit: bool,
}

impl Default for Material {
    /// glTF's default material: white, fully metallic and rough.
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            base_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha: Alpha::Opaque,
            double_sided: false,
            unlit: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Alpha {
    Opaque,
    /// Discarded below the cutoff.
    Mask(f32),
    Blend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureRef {
    /// Index into [`Model::textures`].
    pub texture: usize,
    /// Which texture coordinate set; only set 0 is read.
    pub tex_coord: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Texture {
    /// Index into [`Model::images`].
    pub image: Option<usize>,
    /// Along u and v.
    pub wrap: [Wrap; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    Repeat,
    Clamp,
    Mirror,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageSource {
    /// Encoded bytes (PNG, JPEG, ...).
    Bytes(Vec<u8>),
    /// A URI relative to the document.
    Uri(String),
}

/// Decoded images as one square layer each of a texture array with its mip
/// levels: `size` texels a side at level 0, the power of two at or below
/// the largest image's side, at most `max`. Each layer's levels follow
/// one another, RGBA8.
pub fn texture_layers(images: &[(u32, u32, &[u8])], max: u32) -> (u32, Vec<Vec<u8>>) {
    let largest = images.iter().map(|&(w, h, _)| w.max(h)).max().unwrap_or(1);
    let size = prev_power_of_two(largest.min(max).max(1));
    let layers = images
        .iter()
        .map(|&(w, h, rgba)| {
            let mut level = resize(rgba, [w, h], size);
            let mut out = level.clone();
            let mut side = size;
            while side > 1 {
                level = halve(&level, side);
                side /= 2;
                out.extend_from_slice(&level);
            }
            out
        })
        .collect();
    (size, layers)
}

fn prev_power_of_two(v: u32) -> u32 {
    1 << (31 - v.leading_zeros())
}

/// Bilinear resampling of RGBA8 to a `size`² square.
fn resize(rgba: &[u8], [w, h]: [u32; 2], size: u32) -> Vec<u8> {
    if [w, h] == [size, size] {
        return rgba.to_vec();
    }
    let texel = |x: u32, y: u32, c: usize| f32::from(rgba[((y * w + x) * 4) as usize + c]);
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        let fy = ((y as f32 + 0.5) * h as f32 / size as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as u32, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..size {
            let fx = ((x as f32 + 0.5) * w as f32 / size as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as u32, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..4 {
                let top = texel(x0, y0, c) * (1.0 - tx) + texel(x1, y0, c) * tx;
                let bottom = texel(x0, y1, c) * (1.0 - tx) + texel(x1, y1, c) * tx;
                out.push((top * (1.0 - ty) + bottom * ty).round() as u8);
            }
        }
    }
    out
}

/// The next mip level of a `side`² RGBA8 level: 2×2 box averages.
fn halve(level: &[u8], side: u32) -> Vec<u8> {
    let half = side / 2;
    let mut out = Vec::with_capacity((half * half * 4) as usize);
    for y in 0..half {
        for x in 0..half {
            for c in 0..4 {
                let at = |dx: u32, dy: u32| {
                    u32::from(level[(((2 * y + dy) * side + 2 * x + dx) * 4) as usize + c])
                };
                out.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
