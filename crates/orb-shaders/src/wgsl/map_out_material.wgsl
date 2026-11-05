// A map layer's fragment as material, for the lit map (#91, decision
// 0048): its colour is the albedo, and its node's material (the scene
// document's `roughness`, `metalness` and `emissive-strength`) goes into
// the material buffer's second attachment, premultiplied by the
// fragment's coverage, so layers blend as their colours do and the
// lighting pass (`map_light.wgsl`) divides it out. Map layers face up.
struct NodeMaterial {
    roughness: f32,
    metalness: f32,
    // The share of its albedo that shows whatever the lights.
    self_lit: f32,
    _pad: f32,
};
@group(2) @binding(0) var<storage, read> node_materials: array<NodeMaterial>;

struct MapOut {
    @location(0) albedo: vec4<f32>,
    // Roughness, metalness, the self-lit share, 1.
    @location(1) material: vec4<f32>,
};

fn map_out(color: vec4<f32>, node: u32) -> MapOut {
    let m = node_materials[node];
    return MapOut(color, vec4<f32>(m.roughness, m.metalness, m.self_lit, 1.0) * color.a);
}

// What arrives already shaded (3D layers, the terrain) or isn't a surface
// (a heatmap): it shows as it is.
fn map_out_self_lit(color: vec4<f32>) -> MapOut {
    return MapOut(color, vec4<f32>(1.0, 0.0, 1.0, 1.0) * color.a);
}
