//! The demo's glTF landmarks (`cargo xtask demo-data models`): a lattice
//! tower, a torii gate and a pagoda, built here from boxes, frusta and
//! prisms in metres and written as binary glTF into
//! `web/demo/data/models`. Flat-shaded, metallic-roughness materials, one
//! primitive per material.

use std::path::Path;

/// East, north and up, metres.
type V = [f32; 3];

struct Material {
    name: &'static str,
    /// Linear RGBA.
    color: [f32; 4],
    metallic: f32,
    roughness: f32,
    /// Linear RGB.
    emissive: [f32; 3],
}

/// One material's flat-shaded polygons.
#[derive(Clone, Default)]
struct Part {
    positions: Vec<V>,
    normals: Vec<V>,
    indices: Vec<u32>,
}

/// Its parts by material.
struct Mesh {
    parts: Vec<Part>,
}

impl Mesh {
    fn new(materials: usize) -> Self {
        Self {
            parts: vec![Part::default(); materials],
        }
    }

    /// A flat polygon, its corners counter-clockwise seen from outside.
    fn face(&mut self, material: usize, corners: &[V]) {
        let [a, b, c] = [corners[0], corners[1], corners[2]];
        let u = sub(b, a);
        let v = sub(c, a);
        let n = normalize([
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]);
        let part = &mut self.parts[material];
        let base = part.positions.len() as u32;
        for &p in corners {
            part.positions.push(p);
            part.normals.push(n);
        }
        for i in 1..corners.len() as u32 - 1 {
            part.indices.extend([base, base + i, base + i + 1]);
        }
    }

    fn vertices(&self) -> usize {
        self.parts.iter().map(|p| p.positions.len()).sum()
    }

    /// A prism or frustum about the vertical axis through `(x, y)`: a
    /// regular `sides`-gon of circumradius `r0` at height `z0`, `r1` at
    /// `z1`, turned by `turn` radians, with its caps.
    #[allow(clippy::too_many_arguments)]
    fn frustum(
        &mut self,
        m: usize,
        [x, y]: [f32; 2],
        (z0, r0): (f32, f32),
        (z1, r1): (f32, f32),
        sides: usize,
        turn: f32,
    ) {
        let ring = |z: f32, r: f32| -> Vec<V> {
            (0..sides)
                .map(|i| {
                    let a = turn + i as f32 * std::f32::consts::TAU / sides as f32;
                    [x + r * a.cos(), y + r * a.sin(), z]
                })
                .collect()
        };
        let (lo, hi) = (ring(z0, r0), ring(z1, r1));
        for i in 0..sides {
            let j = (i + 1) % sides;
            self.face(m, &[lo[i], lo[j], hi[j], hi[i]]);
        }
        if r1 > 0.0 {
            self.face(m, &hi);
        }
        if r0 > 0.0 {
            let mut bottom = lo.clone();
            bottom.reverse();
            self.face(m, &bottom);
        }
    }

    /// A box between two corners.
    fn cuboid(&mut self, m: usize, lo: V, hi: V) {
        let p = |x: usize, y: usize, z: usize| {
            [[lo[0], hi[0]][x], [lo[1], hi[1]][y], [lo[2], hi[2]][z]]
        };
        self.face(m, &[p(0, 0, 0), p(0, 1, 0), p(1, 1, 0), p(1, 0, 0)]);
        self.face(m, &[p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)]);
        self.face(m, &[p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1)]);
        self.face(m, &[p(1, 1, 0), p(0, 1, 0), p(0, 1, 1), p(1, 1, 1)]);
        self.face(m, &[p(0, 1, 0), p(0, 0, 0), p(0, 0, 1), p(0, 1, 1)]);
        self.face(m, &[p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1)]);
    }
}

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize(v: V) -> V {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l == 0.0 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

/// An sRGB hex colour, linear.
fn srgb(hex: u32) -> [f32; 4] {
    let c = |shift: u32| {
        let v = ((hex >> shift) & 0xff) as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    [c(16), c(8), c(0), 1.0]
}

fn material(name: &'static str, hex: u32, metallic: f32, roughness: f32) -> Material {
    Material {
        name,
        color: srgb(hex),
        metallic,
        roughness,
        emissive: [0.0; 3],
    }
}

/// A 333 m lattice tower's silhouette: a tapering body in bands of orange
/// and white, two observation decks, a spire with a warning light.
fn tower() -> (Vec<Material>, Mesh) {
    let mut light = material("light", 0xff3020, 0.0, 0.5);
    light.emissive = [1.0, 0.1, 0.05];
    let materials = vec![
        material("orange", 0xe8542a, 0.3, 0.45),
        material("white", 0xf2f0ea, 0.3, 0.45),
        material("deck", 0x9fb8c8, 0.6, 0.25),
        light,
    ];
    let mut mesh = Mesh::new(materials.len());
    // Half the side at a height, along the body's taper.
    let half = |z: f32| -> f32 {
        if z <= 150.0 {
            40.0 - 26.0 * (z / 150.0).powf(0.6)
        } else {
            14.0 - 10.0 * ((z - 150.0) / 100.0)
        }
    };
    let r = |z: f32| half(z) * std::f32::consts::SQRT_2;
    let turn = std::f32::consts::FRAC_PI_4;
    let bands = [
        0.0, 30.0, 60.0, 90.0, 120.0, 145.0, 160.0, 185.0, 210.0, 235.0, 250.0,
    ];
    for (i, w) in bands.windows(2).enumerate() {
        mesh.frustum(i % 2, [0.0, 0.0], (w[0], r(w[0])), (w[1], r(w[1])), 4, turn);
    }
    // Observation decks.
    mesh.frustum(
        2,
        [0.0, 0.0],
        (145.0, r(145.0) + 6.0),
        (160.0, r(160.0) + 6.0),
        8,
        0.0,
    );
    mesh.frustum(2, [0.0, 0.0], (245.0, 7.0), (252.0, 7.0), 8, 0.0);
    // The spire, banded, and its light.
    let spire = [250.0, 275.0, 300.0, 325.0];
    for (i, w) in spire.windows(2).enumerate() {
        let taper = |z: f32| 4.0 - 2.5 * (z - 250.0) / 75.0;
        mesh.frustum(
            i % 2,
            [0.0, 0.0],
            (w[0], taper(w[0])),
            (w[1], taper(w[1])),
            4,
            turn,
        );
    }
    mesh.frustum(3, [0.0, 0.0], (325.0, 1.2), (333.0, 0.2), 6, 0.0);
    (materials, mesh)
}

/// A 24 m torii: two posts, the top lintel with its black cap, the tie
/// beam below.
fn torii() -> (Vec<Material>, Mesh) {
    let materials = vec![
        material("vermilion", 0xd8401f, 0.0, 0.6),
        material("black", 0x1e1c1a, 0.0, 0.5),
        material("stone", 0x8c8a84, 0.0, 0.9),
    ];
    let mut mesh = Mesh::new(materials.len());
    for x in [-7.0, 7.0] {
        mesh.frustum(2, [x, 0.0], (0.0, 2.0), (1.2, 1.8), 8, 0.0);
        mesh.frustum(0, [x, 0.0], (1.2, 1.3), (20.0, 1.1), 12, 0.0);
    }
    // The tie beam through the posts.
    mesh.cuboid(0, [-9.5, -0.6, 15.0], [9.5, 0.6, 16.6]);
    // The lintel, its ends swept up, under its cap.
    mesh.cuboid(0, [-11.0, -0.9, 20.0], [11.0, 0.9, 21.6]);
    mesh.cuboid(1, [-12.5, -1.1, 21.6], [12.5, 1.1, 22.6]);
    for side in [-1.0, 1.0] {
        let x0 = side * 12.5;
        let x1 = side * 13.8;
        let (a, b) = if side < 0.0 { (x1, x0) } else { (x0, x1) };
        mesh.cuboid(1, [a, -1.1, 22.0], [b, 1.1, 23.4]);
    }
    (materials, mesh)
}

/// A five-storey pagoda: a plinth, five storeys under broad roofs, and the
/// finial.
fn pagoda() -> (Vec<Material>, Mesh) {
    let materials = vec![
        material("plinth", 0xb9b3a6, 0.0, 0.95),
        material("wood", 0xb5432a, 0.0, 0.7),
        material("roof", 0x3e4a48, 0.2, 0.55),
        material("bronze", 0xc8a14a, 0.9, 0.35),
    ];
    let mut mesh = Mesh::new(materials.len());
    let turn = std::f32::consts::FRAC_PI_4;
    let s2 = std::f32::consts::SQRT_2;
    mesh.frustum(0, [0.0, 0.0], (0.0, 11.0 * s2), (2.0, 10.5 * s2), 4, turn);
    let mut z = 2.0;
    for storey in 0..5 {
        let body = 7.0 - storey as f32 * 0.9;
        let roof = body + 4.5;
        let height = if storey == 0 { 7.0 } else { 5.5 };
        mesh.frustum(
            1,
            [0.0, 0.0],
            (z, body * s2),
            (z + height, body * s2),
            4,
            turn,
        );
        z += height;
        // The roof flares out, then rises to the next storey.
        mesh.frustum(
            2,
            [0.0, 0.0],
            (z - 0.3, body * s2),
            (z + 0.4, roof * s2),
            4,
            turn,
        );
        mesh.frustum(
            2,
            [0.0, 0.0],
            (z + 0.4, roof * s2),
            (z + 1.6, (body - 0.6) * s2),
            4,
            turn,
        );
        z += 1.6;
    }
    // The finial and its rings.
    mesh.frustum(3, [0.0, 0.0], (z, 0.6), (z + 11.0, 0.3), 8, 0.0);
    for i in 0..7 {
        let at = z + 2.0 + i as f32 * 1.2;
        mesh.frustum(3, [0.0, 0.0], (at, 1.3), (at + 0.3, 1.3), 12, 0.0);
    }
    (materials, mesh)
}

/// Writes `mesh` as binary glTF, a primitive per material: glTF's axes
/// are east, up and south.
fn glb(materials: &[Material], mesh: &Mesh) -> Vec<u8> {
    let to_gltf = |v: V| [v[0], v[2], -v[1]];
    let mut vertices: Vec<u8> = Vec::new();
    let mut indices: Vec<u8> = Vec::new();
    let mut accessors = Vec::new();
    let mut primitives = Vec::new();
    for (m, part) in mesh.parts.iter().enumerate() {
        if part.indices.is_empty() {
            continue;
        }
        let n = part.positions.len();
        let positions: Vec<V> = part.positions.iter().map(|&p| to_gltf(p)).collect();
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &positions {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let at = vertices.len();
        for v in positions.iter().chain(
            part.normals
                .iter()
                .map(|&n| to_gltf(n))
                .collect::<Vec<_>>()
                .iter(),
        ) {
            for c in v {
                vertices.extend(c.to_le_bytes());
            }
        }
        accessors.push(format!(
            r#"{{"bufferView":0,"byteOffset":{at},"componentType":5126,"count":{n},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
        ));
        accessors.push(format!(
            r#"{{"bufferView":0,"byteOffset":{},"componentType":5126,"count":{n},"type":"VEC3"}}"#,
            at + n * 12
        ));
        accessors.push(format!(
            r#"{{"bufferView":1,"byteOffset":{},"componentType":5125,"count":{},"type":"SCALAR"}}"#,
            indices.len(),
            part.indices.len()
        ));
        for i in &part.indices {
            indices.extend(i.to_le_bytes());
        }
        let first = accessors.len() - 3;
        primitives.push(format!(
            r#"{{"attributes":{{"POSITION":{first},"NORMAL":{}}},"indices":{},"material":{m}}}"#,
            first + 1,
            first + 2
        ));
    }
    let materials: Vec<String> = materials
        .iter()
        .map(|m| {
            let [r, g, b, a] = m.color;
            let [er, eg, eb] = m.emissive;
            format!(
                r#"{{"name":"{}","pbrMetallicRoughness":{{"baseColorFactor":[{r},{g},{b},{a}],"metallicFactor":{},"roughnessFactor":{}}},"emissiveFactor":[{er},{eg},{eb}]}}"#,
                m.name, m.metallic, m.roughness
            )
        })
        .collect();
    let mut bin = vertices;
    let index_start = bin.len();
    bin.extend(&indices);
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"orbis xtask demo-data"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{}]}}],"materials":[{}],"accessors":[{}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{index_start}}},{{"buffer":0,"byteOffset":{index_start},"byteLength":{}}}],"buffers":[{{"byteLength":{}}}]}}"#,
        primitives.join(","),
        materials.join(","),
        accessors.join(","),
        indices.len(),
        bin.len()
    );
    let mut json = json.into_bytes();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((json.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(&json);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(&bin);
    out
}

/// Writes the landmarks into `web/demo/data/models`.
pub(crate) fn generate() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the workspace root")
        .join("web/demo/data/models");
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    for (name, (materials, mesh)) in [("tower", tower()), ("torii", torii()), ("pagoda", pagoda())]
    {
        let bytes = glb(&materials, &mesh);
        let path = root.join(format!("{name}.glb"));
        std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        println!(
            "{}: {} bytes, {} vertices",
            path.display(),
            bytes.len(),
            mesh.vertices()
        );
    }
    Ok(())
}
