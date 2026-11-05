use std::collections::HashMap;

use base64::Engine;

use crate::gltf::Gltf;
use crate::*;

/// Little-endian f32s as bytes.
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One triangle in glTF's axes (y up), its buffer in a data URI, under a
/// node moved 10 m along +x.
fn triangle(extra_accessor: &str, normals: bool) -> String {
    let mut bytes = floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    bytes.extend(floats(&[0.0, 0.0, 1.0].repeat(3)));
    let uri = format!(
        "data:application/octet-stream;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    );
    let normal = if normals { r#", "NORMAL": 1"# } else { "" };
    format!(
        r#"{{"asset": {{"version": "2.0"}}, "scene": 0, "scenes": [{{"nodes": [0]}}],
        "nodes": [{{"mesh": 0, "translation": [10, 0, 0]}}],
        "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0{normal}}}, "material": 0}}]}}],
        "materials": [{{"pbrMetallicRoughness": {{"baseColorFactor": [1, 0, 0, 1], "metallicFactor": 0}},
            "extensions": {{"KHR_materials_emissive_strength": {{"emissiveStrength": 2}}}},
            "emissiveFactor": [0.5, 0, 0], "alphaMode": "MASK"}}],
        "buffers": [{{"byteLength": {len}, "uri": "{uri}"}}],
        "bufferViews": [{{"buffer": 0, "byteLength": 36}}, {{"buffer": 0, "byteOffset": 36, "byteLength": 36}}],
        "accessors": [{{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"{extra_accessor}}},
            {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}}]}}"#,
        len = bytes.len(),
    )
}

#[test]
fn a_triangle_lands_east_north_up_with_its_node_and_material() {
    let model = Gltf::parse(triangle("", true).as_bytes())
        .unwrap()
        .build(&HashMap::new())
        .unwrap();
    // glTF (x, y, z) is east x, north −z, up y; the node moves it 10 m east.
    let positions: Vec<[f32; 3]> = model.vertices.iter().map(|v| v.position).collect();
    assert_eq!(
        positions,
        [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 0.0, 2.0]]
    );
    // glTF +z (toward the viewer) is south.
    assert_eq!(model.vertices[0].normal, [0.0, -1.0, 0.0]);
    assert_eq!(model.indices, [0, 1, 2]);
    assert_eq!(model.bounds, [[10.0, 0.0, 0.0], [11.0, 0.0, 2.0]]);
    let m = model.materials[0];
    assert_eq!(m.base_color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!((m.metallic, m.roughness), (0.0, 1.0));
    assert_eq!(m.emissive, [1.0, 0.0, 0.0]);
    assert_eq!(m.alpha, Alpha::Mask(0.5));
}

#[test]
fn missing_normals_are_flat_and_glb_reads_like_json() {
    let json = triangle("", false);
    let mut glb = Vec::new();
    let padded = json.len().next_multiple_of(4);
    glb.extend(0x4654_6c67u32.to_le_bytes());
    glb.extend(2u32.to_le_bytes());
    glb.extend(((12 + 8 + padded) as u32).to_le_bytes());
    glb.extend((padded as u32).to_le_bytes());
    glb.extend(0x4e4f_534au32.to_le_bytes());
    glb.extend(json.as_bytes());
    glb.resize(12 + 8 + padded, b' ');
    let model = Gltf::parse(&glb).unwrap().build(&HashMap::new()).unwrap();
    // Counter-clockwise from glTF's +z: facing south.
    assert!(model.vertices.iter().all(|v| v.normal == [0.0, -1.0, 0.0]));
}

#[test]
fn sparse_accessors_override_their_base() {
    // Replace vertex 2's position with (0, 4, 0) through a sparse part
    // stored after the normals: index u16 2 at byte 72, value at 76.
    let extra = r#", "sparse": {"count": 1, "indices": {"bufferView": 2, "componentType": 5123},
        "values": {"bufferView": 3}}"#;
    let json = triangle(extra, true);
    let mut bytes = floats(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    bytes.extend(floats(&[0.0, 0.0, 1.0].repeat(3)));
    bytes.extend(2u16.to_le_bytes());
    bytes.extend([0, 0]);
    bytes.extend(floats(&[0.0, 4.0, 0.0]));
    let uri = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let json = json
        .replace(
            r#"{"buffer": 0, "byteOffset": 36, "byteLength": 36}]"#,
            r#"{"buffer": 0, "byteOffset": 36, "byteLength": 36},
            {"buffer": 0, "byteOffset": 72, "byteLength": 2},
            {"buffer": 0, "byteOffset": 76, "byteLength": 12}]"#,
        )
        .replace(
            "\"byteLength\": 72,",
            &format!("\"byteLength\": {},", bytes.len()),
        );
    let start = json.find("base64,").unwrap() + 7;
    let end = start + json[start..].find('"').unwrap();
    let json = format!("{}{uri}{}", &json[..start], &json[end..]);
    let model = Gltf::parse(json.as_bytes())
        .unwrap()
        .build(&HashMap::new())
        .unwrap();
    assert_eq!(model.vertices[2].position, [10.0, 0.0, 4.0]);
}

#[test]
fn external_buffers_are_asked_for_and_unknown_required_extensions_refused() {
    let json = r#"{"asset": {"version": "2.0"}, "buffers": [{"byteLength": 4, "uri": "a.bin"}]}"#;
    assert_eq!(
        Gltf::parse(json.as_bytes()).unwrap().external_buffers(),
        [(0, "a.bin".to_string())]
    );
    let required =
        r#"{"asset": {"version": "2.0"}, "extensionsRequired": ["KHR_draco_mesh_compression"]}"#;
    assert!(Gltf::parse(required.as_bytes()).is_err());
}

#[test]
fn images_become_square_power_of_two_layers_with_mips() {
    let a = [255u8, 0, 0, 255].repeat(6 * 3);
    let b = [0u8, 0, 255, 255].repeat(2 * 2);
    let (size, layers) = texture_layers(&[(6, 3, &a), (2, 2, &b)], 1024);
    assert_eq!(size, 4);
    // 4² + 2² + 1² texels per layer.
    assert!(layers.iter().all(|l| l.len() == (16 + 4 + 1) * 4));
    assert_eq!(&layers[0][..4], &[255, 0, 0, 255]);
    assert_eq!(&layers[1][layers[1].len() - 4..], &[0, 0, 255, 255]);
}

#[test]
fn packs_vertices_materials_and_draw_classes_and_round_trips() {
    let mut model = Gltf::parse(triangle("", true).as_bytes())
        .unwrap()
        .build(&HashMap::new())
        .unwrap();
    // A second, blended material on a copy of the triangle.
    let mut blended = model.materials[0];
    blended.alpha = Alpha::Blend;
    model.materials.push(blended);
    let copies: Vec<Vertex> = model
        .vertices
        .iter()
        .map(|v| Vertex { material: 1, ..*v })
        .collect();
    model.vertices.extend(copies);
    model.indices.extend([3, 4, 5]);
    let packed = gpu::pack(&model, &[]);
    assert_eq!(packed.vertex_count(), 6);
    assert_eq!(packed.draws, [0..3, 3..3, 3..6]);
    assert_eq!(packed.materials.len(), 2 * gpu::MATERIAL_WORDS);
    // A south-facing normal: octahedral (0, −1).
    assert_eq!(packed.vertices[3], 0x8001_0000);
    assert_eq!(packed.materials[13] & gpu::MASK, gpu::MASK);
    assert_eq!(packed.materials[16 + 13] & gpu::BLEND, gpu::BLEND);
    assert_eq!(packed.materials[12], 0xffff_ffff, "no textures");
    let back = gpu::GpuModel::from_words(&packed.to_words()).unwrap();
    assert_eq!(back, packed);
}

#[test]
fn urls_resolve_against_the_document() {
    use crate::load::join;
    let base = "https://host.example/models/tower/scene.gltf?v=2";
    assert_eq!(
        join(base, "scene.bin"),
        "https://host.example/models/tower/scene.bin"
    );
    assert_eq!(
        join(base, "../tex/a.png"),
        "https://host.example/models/tex/a.png"
    );
    assert_eq!(join(base, "/b.png"), "https://host.example/b.png");
    assert_eq!(
        join(base, "https://cdn.example/c.png"),
        "https://cdn.example/c.png"
    );
    assert_eq!(join("file:///data/m.glb", "m.bin"), "file:///data/m.bin");
}

#[test]
fn a_load_fetches_buffers_then_images_and_packs() {
    // The triangle with its buffer in a file of its own and a base colour
    // texture from an image file.
    let json = triangle("", true);
    let start = json.find("data:").unwrap();
    let end = start + json[start..].find('"').unwrap();
    let bin = base64::engine::general_purpose::STANDARD
        .decode(&json[json.find("base64,").unwrap() + 7..end])
        .unwrap();
    let json = format!("{}tri.bin{}", &json[..start], &json[end..]).replace(
        r#""metallicFactor": 0}"#,
        r#""metallicFactor": 0, "baseColorTexture": {"index": 0}}"#,
    );
    let json = json.replacen(
        r#""buffers""#,
        r#""images": [{"uri": "tex.png"}], "textures": [{"source": 0}], "buffers""#,
        1,
    );
    let mut load = crate::load::Load::new("https://h.example/m/a.gltf", json.as_bytes()).unwrap();
    assert_eq!(load.wanted(), ["https://h.example/m/tri.bin"]);
    assert_eq!(load.images().unwrap(), None);
    load.fetched("https://h.example/m/tri.bin", bin);
    // Built: the image is wanted next.
    assert_eq!(load.images().unwrap(), None);
    assert_eq!(load.wanted(), ["https://h.example/m/tex.png"]);
    load.fetched("https://h.example/m/tex.png", b"png".to_vec());
    assert_eq!(load.images().unwrap(), Some(vec![Some(b"png".to_vec())]));
    let packed = load.finish(&[Some((1, 1, vec![255, 0, 0, 255]))]).unwrap();
    assert_eq!(
        (
            packed.vertex_count(),
            packed.texture_size,
            packed.layers.len()
        ),
        (3, 1, 1)
    );
    // The base colour samples layer 0.
    assert_eq!(packed.materials[12] & 0xff, 0);
}

/// Every Khronos sample in the fixtures (`cargo xtask fixtures`) loads,
/// decodes and packs; skipped without them.
#[test]
fn khronos_samples_load_and_pack() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gltf");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("no glTF fixtures: run `cargo xtask fixtures`");
        return;
    };
    let mut names: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    names.sort();
    for path in names
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "glb"))
    {
        let bytes = std::fs::read(path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        let mut load = crate::load::Load::new(&format!("file:///{name}"), &bytes).unwrap();
        assert!(load.wanted().is_empty(), "{name}: a .glb holds its data");
        let images = load.images().unwrap().expect("nothing to fetch");
        let decoded: Vec<_> = images
            .iter()
            .map(|bytes| {
                let image = orb_host::decode_image(bytes.as_ref()?).ok()?;
                Some((image.width, image.height, image.pixels))
            })
            .collect();
        assert!(
            decoded.iter().all(Option::is_some),
            "{name}: every image decodes"
        );
        let packed = load.finish(&decoded).unwrap();
        let triangles: u32 = packed.draws.iter().map(|r| r.len() as u32 / 3).sum();
        assert!(triangles > 0, "{name}");
        eprintln!(
            "{name}: {} vertices, {triangles} triangles, {} layers of {}², {:.1} MiB",
            packed.vertex_count(),
            packed.layers.len(),
            packed.texture_size,
            packed.gpu_bytes() as f64 / f64::from(1 << 20)
        );
    }
}
