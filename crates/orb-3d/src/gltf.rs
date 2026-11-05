//! A glTF 2.0 reader: a `.gltf` (JSON) or `.glb` document and its buffers
//! to one [`Model`]. Every triangle primitive of the default scene's nodes
//! is flattened into one vertex and index stream, transformed by its node's
//! world matrix and turned from glTF's axes (+Y up, +Z forward) onto east,
//! north and up: `(x, −z, y)`. Images stay encoded for the host to decode.
//!
//! Supported: accessors of every component type, normalized and sparse;
//! triangles, strips and fans; node matrices and TRS; metallic-roughness
//! materials with base colour, metallic-roughness, normal, occlusion and
//! emissive textures, alpha modes and double-sidedness; vertex colours;
//! `KHR_materials_emissive_strength`, `KHR_materials_unlit` and
//! `KHR_mesh_quantization` (integer positions). Points and lines are
//! skipped, as are morph targets and skins (the bind pose draws).

use std::collections::HashMap;

use base64::Engine;
use glam::{Mat3, Mat4, Quat, Vec3};
use orb_expr::json::{self, Json};

use crate::{Alpha, ImageSource, Material, Model, TextureRef, Vertex, Wrap};

const GLB_MAGIC: u32 = 0x4654_6c67;
const CHUNK_JSON: u32 = 0x4e4f_534a;
const CHUNK_BIN: u32 = 0x004e_4942;

/// A parsed glTF document, before its external buffers are fetched.
#[derive(Debug)]
pub struct Gltf {
    json: Json,
    /// A `.glb`'s binary chunk: buffer 0 when it has no `uri`.
    bin: Option<Vec<u8>>,
}

impl Gltf {
    /// Reads a `.glb` container or a `.gltf` JSON document.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let word = |at: usize| -> Result<u32, String> {
            bytes
                .get(at..at + 4)
                .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
                .ok_or_else(|| "truncated glb".to_string())
        };
        let (text, bin) = if bytes.len() >= 12 && word(0)? == GLB_MAGIC {
            if word(4)? != 2 {
                return Err(format!("glb version {} (2 is supported)", word(4)?));
            }
            let length = (word(8)? as usize).min(bytes.len());
            let (mut at, mut text, mut bin) = (12, None, None);
            while at + 8 <= length {
                let (size, kind) = (word(at)? as usize, word(at + 4)?);
                let chunk = bytes
                    .get(at + 8..at + 8 + size)
                    .ok_or("truncated glb chunk")?;
                match kind {
                    CHUNK_JSON => text = Some(chunk),
                    CHUNK_BIN => bin = Some(chunk.to_vec()),
                    _ => {}
                }
                at += 8 + size.next_multiple_of(4);
            }
            (text.ok_or("glb without a JSON chunk")?, bin)
        } else {
            (bytes, None)
        };
        let text = std::str::from_utf8(text).map_err(|e| format!("glTF JSON: {e}"))?;
        let json = json::parse(text).map_err(|e| format!("glTF JSON: {}", e.message))?;
        let version = json
            .get("asset")
            .and_then(|a| a.get("version"))
            .and_then(Json::as_str)
            .ok_or("glTF without asset.version")?;
        if !version.starts_with("2.") {
            return Err(format!("glTF {version} (2.x is supported)"));
        }
        for required in array(&json, "extensionsRequired") {
            let name = required.as_str().unwrap_or_default();
            if !SUPPORTED.contains(&name) {
                return Err(format!("glTF requires the unsupported extension {name}"));
            }
        }
        Ok(Self { json, bin })
    }

    /// The buffers to fetch, by index, with their URIs relative to the
    /// document: those not in the container nor in a `data:` URI.
    pub fn external_buffers(&self) -> Vec<(usize, String)> {
        array(&self.json, "buffers")
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let uri = b.get("uri")?.as_str()?;
                (!uri.starts_with("data:")).then(|| (i, uri.to_string()))
            })
            .collect()
    }

    /// The model, given the external buffers by index.
    pub fn build(&self, external: &HashMap<usize, Vec<u8>>) -> Result<Model, String> {
        let buffers = self.buffers(external)?;
        let mut builder = Builder {
            doc: &self.json,
            buffers: &buffers,
            model: Model::default(),
            materials: HashMap::new(),
        };
        builder.images()?;
        builder.textures();
        let scene = self
            .json
            .get("scene")
            .and_then(Json::as_f64)
            .map(|s| s as usize)
            .unwrap_or(0);
        let roots: Vec<usize> = match array(&self.json, "scenes").get(scene) {
            Some(scene) => indices(scene.get("nodes")),
            // No scenes: every node no other node parents.
            None => {
                let nodes = array(&self.json, "nodes");
                let children: Vec<usize> = nodes
                    .iter()
                    .flat_map(|n| indices(n.get("children")))
                    .collect();
                (0..nodes.len()).filter(|i| !children.contains(i)).collect()
            }
        };
        for root in roots {
            builder.node(root, Mat4::IDENTITY, 0)?;
        }
        let mut model = builder.model;
        if model.indices.is_empty() {
            return Err("glTF has no triangles".into());
        }
        model.bounds = bounds(&model.vertices);
        Ok(model)
    }

    fn buffers(&self, external: &HashMap<usize, Vec<u8>>) -> Result<Vec<Vec<u8>>, String> {
        array(&self.json, "buffers")
            .iter()
            .enumerate()
            .map(|(i, b)| match b.get("uri").and_then(Json::as_str) {
                Some(uri) if uri.starts_with("data:") => data_uri(uri),
                Some(uri) => external
                    .get(&i)
                    .cloned()
                    .ok_or_else(|| format!("glTF buffer {uri} not loaded")),
                None if i == 0 => self.bin.clone().ok_or("glTF buffer 0 without data".into()),
                None => Err(format!("glTF buffer {i} without a uri")),
            })
            .collect()
    }
}

/// Extensions a model may require.
const SUPPORTED: &[&str] = &[
    "KHR_mesh_quantization",
    "KHR_materials_unlit",
    "KHR_materials_emissive_strength",
];

/// Nested nodes deeper than this are ignored (a cycle guard).
const MAX_DEPTH: usize = 64;

struct Builder<'a> {
    doc: &'a Json,
    buffers: &'a [Vec<u8>],
    model: Model,
    /// glTF material index (or `None`, the default) to the model's.
    materials: HashMap<Option<usize>, u32>,
}

impl Builder<'_> {
    fn images(&mut self) -> Result<(), String> {
        for image in array(self.doc, "images") {
            let source = match (
                image.get("uri").and_then(Json::as_str),
                image.get("bufferView"),
            ) {
                (Some(uri), _) if uri.starts_with("data:") => ImageSource::Bytes(data_uri(uri)?),
                (Some(uri), _) => ImageSource::Uri(uri.to_string()),
                (None, Some(view)) => ImageSource::Bytes(
                    self.view(view.as_f64().unwrap_or(-1.0) as usize)?
                        .0
                        .to_vec(),
                ),
                (None, None) => return Err("glTF image without data".into()),
            };
            self.model.images.push(source);
        }
        Ok(())
    }

    fn textures(&mut self) {
        let samplers = array(self.doc, "samplers");
        for texture in array(self.doc, "textures") {
            let sampler = texture
                .get("sampler")
                .and_then(Json::as_f64)
                .and_then(|s| samplers.get(s as usize));
            let wrap = |key| match sampler.and_then(|s| s.get(key)).and_then(Json::as_f64) {
                Some(33071.0) => Wrap::Clamp,
                Some(33648.0) => Wrap::Mirror,
                _ => Wrap::Repeat,
            };
            self.model.textures.push(crate::Texture {
                image: texture
                    .get("source")
                    .and_then(Json::as_f64)
                    .map(|s| s as usize),
                wrap: [wrap("wrapS"), wrap("wrapT")],
            });
        }
    }

    fn node(&mut self, index: usize, parent: Mat4, depth: usize) -> Result<(), String> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        let Some(node) = array(self.doc, "nodes").get(index) else {
            return Err(format!("glTF node {index} doesn't exist"));
        };
        let local = match node.get("matrix").and_then(Json::as_array) {
            Some(m) if m.len() == 16 => {
                let m: Vec<f32> = m.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();
                Mat4::from_cols_slice(&m)
            }
            _ => {
                let floats = |key, default: &[f32]| -> Vec<f32> {
                    node.get(key)
                        .and_then(Json::as_array)
                        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
                        .unwrap_or_else(|| default.to_vec())
                };
                let t = floats("translation", &[0.0; 3]);
                let r = floats("rotation", &[0.0, 0.0, 0.0, 1.0]);
                let s = floats("scale", &[1.0; 3]);
                Mat4::from_scale_rotation_translation(
                    Vec3::new(s[0], s[1], s[2]),
                    Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize(),
                    Vec3::new(t[0], t[1], t[2]),
                )
            }
        };
        let world = parent * local;
        if let Some(mesh) = node.get("mesh").and_then(Json::as_f64) {
            self.mesh(mesh as usize, world)?;
        }
        for child in indices(node.get("children")) {
            self.node(child, world, depth + 1)?;
        }
        Ok(())
    }

    fn mesh(&mut self, index: usize, world: Mat4) -> Result<(), String> {
        let Some(mesh) = array(self.doc, "meshes").get(index) else {
            return Err(format!("glTF mesh {index} doesn't exist"));
        };
        for primitive in array(mesh, "primitives") {
            self.primitive(primitive, world)?;
        }
        Ok(())
    }

    fn primitive(&mut self, primitive: &Json, world: Mat4) -> Result<(), String> {
        let mode = primitive.get("mode").and_then(Json::as_f64).unwrap_or(4.0) as u32;
        if !(4..=6).contains(&mode) {
            return Ok(());
        }
        let attributes = primitive
            .get("attributes")
            .ok_or("primitive without attributes")?;
        let attribute = |name| {
            attributes
                .get(name)
                .and_then(Json::as_f64)
                .map(|a| a as usize)
        };
        let Some(position) = attribute("POSITION") else {
            return Ok(());
        };
        let positions = self.accessor(position)?;
        let count = positions.count;
        let normals = attribute("NORMAL").map(|a| self.accessor(a)).transpose()?;
        let uvs = attribute("TEXCOORD_0")
            .map(|a| self.accessor(a))
            .transpose()?;
        let colors = attribute("COLOR_0").map(|a| self.accessor(a)).transpose()?;
        let mut corners: Vec<u32> = match primitive.get("indices").and_then(Json::as_f64) {
            Some(i) => self
                .accessor(i as usize)?
                .values
                .iter()
                .map(|&v| v as u32)
                .collect(),
            None => (0..count as u32).collect(),
        };
        corners = match mode {
            5 => (2..corners.len())
                .flat_map(|i| {
                    let (a, b, c) = (corners[i - 2], corners[i - 1], corners[i]);
                    if i % 2 == 0 { [a, b, c] } else { [b, a, c] }
                })
                .collect(),
            6 => (2..corners.len())
                .flat_map(|i| [corners[0], corners[i - 1], corners[i]])
                .collect(),
            _ => {
                corners.truncate(corners.len() / 3 * 3);
                corners
            }
        };
        if corners.iter().any(|&c| c as usize >= count) {
            return Err("glTF index out of range".into());
        }
        let material = self.material(
            primitive
                .get("material")
                .and_then(Json::as_f64)
                .map(|m| m as usize),
        )?;
        let normal_matrix = Mat3::from_mat4(world).inverse().transpose();
        // glTF's axes onto east, north, up.
        let enu = |v: Vec3| [v.x, -v.z, v.y];
        let vertex = |i: usize| {
            let p = world.transform_point3(positions.vec3(i));
            let normal = normals
                .as_ref()
                .map(|n| enu((normal_matrix * n.vec3(i)).normalize_or_zero()))
                .unwrap_or([0.0; 3]);
            let color = colors.as_ref().map_or([255; 4], |c| {
                let v = c.get(i);
                let a = if c.components == 4 { v[3] } else { 1.0 };
                [v[0], v[1], v[2], a].map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
            });
            Vertex {
                position: enu(p),
                normal,
                uv: uvs.as_ref().map_or([0.0; 2], |u| {
                    let v = u.get(i);
                    [v[0], v[1]]
                }),
                color,
                material,
            }
        };
        let base = self.model.vertices.len() as u32;
        // A mirroring transform turns the winding.
        let flip = world.determinant() < 0.0;
        if normals.is_some() {
            self.model.vertices.extend((0..count).map(vertex));
            for t in corners.as_chunks::<3>().0 {
                let [a, b, c] = [t[0], t[1], t[2]].map(|v| base + v);
                self.model
                    .indices
                    .extend(if flip { [a, c, b] } else { [a, b, c] });
            }
        } else {
            // No normals: flat ones, per triangle (glTF 3.7.2.1).
            for t in corners.as_chunks::<3>().0 {
                let mut v: Vec<Vertex> = t.iter().map(|&i| vertex(i as usize)).collect();
                if flip {
                    v.swap(1, 2);
                }
                let p: Vec<Vec3> = v.iter().map(|v| Vec3::from(v.position)).collect();
                let n = (p[1] - p[0])
                    .cross(p[2] - p[0])
                    .normalize_or_zero()
                    .to_array();
                let at = self.model.vertices.len() as u32;
                for mut v in v {
                    v.normal = n;
                    self.model.vertices.push(v);
                }
                self.model.indices.extend([at, at + 1, at + 2]);
            }
        }
        Ok(())
    }

    fn material(&mut self, index: Option<usize>) -> Result<u32, String> {
        if let Some(&m) = self.materials.get(&index) {
            return Ok(m);
        }
        let material = match index {
            None => Material::default(),
            Some(i) => {
                let json = array(self.doc, "materials")
                    .get(i)
                    .ok_or_else(|| format!("glTF material {i} doesn't exist"))?;
                material(json)
            }
        };
        let at = self.model.materials.len() as u32;
        self.model.materials.push(material);
        self.materials.insert(index, at);
        Ok(at)
    }

    /// A buffer view's bytes and stride (0: tightly packed).
    fn view(&self, index: usize) -> Result<(&[u8], usize), String> {
        let view = array(self.doc, "bufferViews")
            .get(index)
            .ok_or_else(|| format!("glTF bufferView {index} doesn't exist"))?;
        let number = |key| view.get(key).and_then(Json::as_f64).unwrap_or(0.0) as usize;
        let buffer = self
            .buffers
            .get(number("buffer"))
            .ok_or("glTF bufferView's buffer doesn't exist")?;
        let (offset, length) = (number("byteOffset"), number("byteLength"));
        let bytes = buffer
            .get(offset..offset + length)
            .ok_or("glTF bufferView out of its buffer")?;
        Ok((bytes, number("byteStride")))
    }

    fn accessor(&self, index: usize) -> Result<Accessor, String> {
        let accessor = array(self.doc, "accessors")
            .get(index)
            .ok_or_else(|| format!("glTF accessor {index} doesn't exist"))?;
        let number = |key| accessor.get(key).and_then(Json::as_f64);
        let count = number("count").ok_or("accessor without count")? as usize;
        let kind = number("componentType").ok_or("accessor without componentType")? as u32;
        let normalized = accessor
            .get("normalized")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let components = match accessor.get("type").and_then(Json::as_str) {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") | Some("MAT2") => 4,
            Some("MAT3") => 9,
            Some("MAT4") => 16,
            other => return Err(format!("accessor type {other:?}")),
        };
        let size = component_size(kind)?;
        let mut values = vec![0.0f32; count * components];
        if let Some(view) = number("bufferView") {
            let (bytes, stride) = self.view(view as usize)?;
            let offset = number("byteOffset").unwrap_or(0.0) as usize;
            let stride = if stride == 0 {
                size * components
            } else {
                stride
            };
            for i in 0..count {
                for c in 0..components {
                    let at = offset + i * stride + c * size;
                    let raw = bytes.get(at..at + size).ok_or("accessor out of its view")?;
                    values[i * components + c] = read(raw, kind, normalized);
                }
            }
        }
        if let Some(sparse) = accessor.get("sparse") {
            let n = sparse.get("count").and_then(Json::as_f64).unwrap_or(0.0) as usize;
            let part = |key| -> Result<(&[u8], usize, u32), String> {
                let p = sparse.get(key).ok_or("sparse accessor without its parts")?;
                let view = p
                    .get("bufferView")
                    .and_then(Json::as_f64)
                    .ok_or("sparse part without view")?;
                let offset = p.get("byteOffset").and_then(Json::as_f64).unwrap_or(0.0) as usize;
                let kind = p
                    .get("componentType")
                    .and_then(Json::as_f64)
                    .unwrap_or(f64::from(kind)) as u32;
                let bytes = self.view(view as usize)?.0;
                Ok((
                    bytes.get(offset..).ok_or("sparse part out of its view")?,
                    0,
                    kind,
                ))
            };
            let (index_bytes, _, index_kind) = part("indices")?;
            let (value_bytes, _, _) = part("values")?;
            let index_size = component_size(index_kind)?;
            for s in 0..n {
                let raw = index_bytes
                    .get(s * index_size..(s + 1) * index_size)
                    .ok_or("sparse indices out of their view")?;
                let target = read(raw, index_kind, false) as usize;
                for c in 0..components {
                    let at = (s * components + c) * size;
                    let raw = value_bytes
                        .get(at..at + size)
                        .ok_or("sparse values out of their view")?;
                    if let Some(v) = values.get_mut(target * components + c) {
                        *v = read(raw, kind, normalized);
                    }
                }
            }
        }
        Ok(Accessor {
            count,
            components,
            values,
        })
    }
}

fn material(json: &Json) -> Material {
    let pbr = json.get("pbrMetallicRoughness");
    let numbers = |j: Option<&Json>, default: &[f32]| -> Vec<f32> {
        j.and_then(Json::as_array)
            .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
            .filter(|v: &Vec<f32>| v.len() == default.len())
            .unwrap_or_else(|| default.to_vec())
    };
    let number =
        |j: Option<&Json>, default: f32| j.and_then(Json::as_f64).map_or(default, |v| v as f32);
    let texture = |j: Option<&Json>| {
        let j = j?;
        Some(TextureRef {
            texture: j.get("index")?.as_f64()? as usize,
            tex_coord: number(j.get("texCoord"), 0.0) as u32,
        })
    };
    let extensions = json.get("extensions");
    let base = numbers(pbr.and_then(|p| p.get("baseColorFactor")), &[1.0; 4]);
    let emissive = numbers(json.get("emissiveFactor"), &[0.0; 3]);
    let strength = number(
        extensions
            .and_then(|e| e.get("KHR_materials_emissive_strength"))
            .and_then(|e| e.get("emissiveStrength")),
        1.0,
    );
    Material {
        base_color: [base[0], base[1], base[2], base[3]],
        base_texture: texture(pbr.and_then(|p| p.get("baseColorTexture"))),
        metallic: number(pbr.and_then(|p| p.get("metallicFactor")), 1.0),
        roughness: number(pbr.and_then(|p| p.get("roughnessFactor")), 1.0),
        metallic_roughness_texture: texture(pbr.and_then(|p| p.get("metallicRoughnessTexture"))),
        normal_texture: texture(json.get("normalTexture")),
        normal_scale: number(json.get("normalTexture").and_then(|t| t.get("scale")), 1.0),
        occlusion_texture: texture(json.get("occlusionTexture")),
        occlusion_strength: number(
            json.get("occlusionTexture").and_then(|t| t.get("strength")),
            1.0,
        ),
        emissive: [emissive[0], emissive[1], emissive[2]].map(|v| v * strength),
        emissive_texture: texture(json.get("emissiveTexture")),
        alpha: match json.get("alphaMode").and_then(Json::as_str) {
            Some("MASK") => Alpha::Mask(number(json.get("alphaCutoff"), 0.5)),
            Some("BLEND") => Alpha::Blend,
            _ => Alpha::Opaque,
        },
        double_sided: json
            .get("doubleSided")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        unlit: extensions
            .and_then(|e| e.get("KHR_materials_unlit"))
            .is_some(),
    }
}

struct Accessor {
    count: usize,
    components: usize,
    values: Vec<f32>,
}

impl Accessor {
    fn get(&self, i: usize) -> [f32; 4] {
        let mut out = [0.0; 4];
        let n = self.components.min(4);
        out[..n].copy_from_slice(&self.values[i * self.components..i * self.components + n]);
        out
    }

    fn vec3(&self, i: usize) -> Vec3 {
        let v = self.get(i);
        Vec3::new(v[0], v[1], v[2])
    }
}

fn component_size(kind: u32) -> Result<usize, String> {
    match kind {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        other => Err(format!("accessor componentType {other}")),
    }
}

/// One component as f32, normalized as glTF 3.11 maps integers.
fn read(raw: &[u8], kind: u32, normalized: bool) -> f32 {
    match kind {
        5120 => {
            let v = f32::from(raw[0] as i8);
            if normalized { (v / 127.0).max(-1.0) } else { v }
        }
        5121 => {
            let v = f32::from(raw[0]);
            if normalized { v / 255.0 } else { v }
        }
        5122 => {
            let v = f32::from(i16::from_le_bytes([raw[0], raw[1]]));
            if normalized {
                (v / 32767.0).max(-1.0)
            } else {
                v
            }
        }
        5123 => {
            let v = f32::from(u16::from_le_bytes([raw[0], raw[1]]));
            if normalized { v / 65535.0 } else { v }
        }
        5125 => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f32,
        _ => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
    }
}

fn data_uri(uri: &str) -> Result<Vec<u8>, String> {
    let (_, data) = uri.split_once(',').ok_or("malformed data URI")?;
    if uri[..uri.len() - data.len()].contains(";base64") {
        base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|e| format!("data URI: {e}"))
    } else {
        Ok(data.as_bytes().to_vec())
    }
}

fn array<'a>(json: &'a Json, key: &str) -> &'a [Json] {
    json.get(key).and_then(Json::as_array).unwrap_or(&[])
}

fn indices(json: Option<&Json>) -> Vec<usize> {
    json.and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Json::as_f64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default()
}

fn bounds(vertices: &[Vertex]) -> [[f32; 3]; 2] {
    let mut b = [[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]];
    for v in vertices {
        for (axis, &p) in v.position.iter().enumerate() {
            b[0][axis] = b[0][axis].min(p);
            b[1][axis] = b[1][axis].max(p);
        }
    }
    b
}
