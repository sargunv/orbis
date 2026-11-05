// A map layer's fragment lit by a scene's lights (#91, decision 0048): its
// colour is the albedo of its node's material (the scene document's
// `roughness`, `metalness` and `emissive-strength`), shaded in place with
// `pbr_light`. Map layers face up; blending lit colours equals lighting
// blended albedo for their diffuse light, which shares one normal and
// point per pixel. Shadows fall on them as on the unlit ground: the
// ground pass takes the directional light's share away where they fall
// (`shadow_ground.wgsl`), which for matte ground is exact.
struct NodeMaterial {
    roughness: f32,
    metalness: f32,
    // The share of its albedo that shows whatever the lights.
    self_lit: f32,
    _pad: f32,
};
@group(2) @binding(0) var<storage, read> node_materials: array<NodeMaterial>;

struct MapOut {
    @location(0) color: vec4<f32>,
};

fn map_out(color: vec4<f32>, node: u32, position: vec4<f32>) -> MapOut {
    let a = color.a;
    if a <= 0.0 {
        return MapOut(vec4<f32>(0.0));
    }
    let n = node_materials[node];
    var m: Material;
    m.albedo = srgb_to_linear(color.rgb / a);
    m.alpha = a;
    m.roughness = clamp(n.roughness, 0.04, 1.0);
    m.metallic = n.metalness;
    m.self_lit = n.self_lit;
    m.occlusion = 1.0;
    m.emitted = vec3<f32>(0.0);
    m.normal = vec3<f32>(0.0, 0.0, 1.0);
    return MapOut(vec4<f32>(linear_to_srgb(pbr_light(m, scene_lights())) * a, a));
}
