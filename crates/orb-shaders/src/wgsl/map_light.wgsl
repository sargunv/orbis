// The lit map (#91, decision 0048): one full-screen draw in the main pass
// shades the material buffer the map's layers drew (`map_out_material`)
// with `pbr_light` under the style's lights, the directional light kept
// from the ground where shadows fall, over the sky. Labels draw after it,
// unlit.
@group(1) @binding(0) var material_albedo: texture_2d<f32>;
@group(1) @binding(1) var material_params: texture_2d<f32>;

struct MapLightOut { @builtin(position) position: vec4<f32>, };

@vertex
fn vs_map_light(@builtin(vertex_index) index: u32) -> MapLightOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    var out: MapLightOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    return out;
}

// How much of the directional light the ground under pixel `xy` keeps: its
// point on the ground plane, or the terrain's from its depth.
fn ground_shadow(xy: vec2<f32>) -> f32 {
    if shadow.cascades.count == 0u {
        return 1.0;
    }
    let ndc = xy / (frame.viewport * frame.pixel_ratio) * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    var world: vec3<f32>;
    if frame.terrain_exaggeration > 0.0 {
        let depth = textureLoad(terrain_depth, vec2<i32>(xy), 0).r;
        if depth <= 0.0 {
            return 1.0;
        }
        let p = shadow.inverse * vec4<f32>(ndc, 1.0 - depth, 1.0);
        world = p.xyz / p.w;
    } else {
        let near = shadow.inverse * vec4<f32>(ndc, 0.0, 1.0);
        let far = shadow.inverse * vec4<f32>(ndc, 1.0, 1.0);
        let a = near.xyz / near.w;
        let b = far.xyz / far.w;
        if b.z >= a.z || a.z < 0.0 {
            return 1.0;
        }
        world = mix(a, b, a.z / (a.z - b.z));
    }
    let depth = (frame.view_proj * vec4<f32>(world, 1.0)).w;
    return shadow_at(world, vec3<f32>(0.0, 0.0, 1.0), depth);
}

@fragment
fn fs_map_light(in: MapLightOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    let albedo = textureLoad(material_albedo, p, 0);
    let a = albedo.a;
    if a <= 0.0 {
        return vec4<f32>(0.0);
    }
    let params = textureLoad(material_params, p, 0) / a;
    var m: Material;
    m.albedo = srgb_to_linear(albedo.rgb / a);
    m.alpha = a;
    m.roughness = clamp(params.x, 0.04, 1.0);
    m.metallic = clamp(params.y, 0.0, 1.0);
    m.self_lit = clamp(params.z, 0.0, 1.0);
    m.occlusion = 1.0;
    m.emitted = vec3<f32>(0.0);
    m.normal = vec3<f32>(0.0, 0.0, 1.0);
    var lights = scene_lights();
    lights.shadow = ground_shadow(in.position.xy);
    return vec4<f32>(linear_to_srgb(pbr_light(m, lights)) * a, a);
}
