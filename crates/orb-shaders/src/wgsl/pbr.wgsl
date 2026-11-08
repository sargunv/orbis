// Metallic-roughness shading under the style's lights, as functions of
// their inputs alone: no bindings, so any pipeline composes it. Directions are in east, north and up; colours are
// linear.

const PBR_PI: f32 = 3.14159265;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

// The lights at a surface: linear colours times their levels, the unit
// vector toward the directional light and toward the eye, and how much
// of the directional light reaches it (1 lit, 0 in shadow).
struct PbrLights {
    ambient: vec3<f32>,
    directional: vec3<f32>,
    toward_light: vec3<f32>,
    toward_eye: vec3<f32>,
    shadow: f32,
};

// GGX distribution, Smith-Schlick visibility and Schlick's Fresnel: the
// specular reflectance toward the eye.
fn pbr_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let h = normalize(v + l);
    let a = roughness * roughness;
    let a2 = a * a;
    let nh = max(dot(n, h), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let nl = max(dot(n, l), 1e-4);
    let d = a2 / (PBR_PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let k = a / 2.0;
    let vis = 0.25 / ((nv * (1.0 - k) + k) * (nl * (1.0 - k) + k));
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    return d * vis * f;
}

// A surface's material: all its shading needs besides the lights, which
// the lit map's layers fill (`map_out_lit.wgsl`, #91).
// Colours are linear; the normal is a unit vector in
// east, north and up.
struct Material {
    albedo: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    // Ambient light reaching it, 0 to 1.
    occlusion: f32,
    // How much of its albedo shows whatever the lights.
    self_lit: f32,
    // Light it gives off, added after shading.
    emitted: vec3<f32>,
    normal: vec3<f32>,
};

// A material's linear colour under `lights`: Lambert diffuse for its
// non-metal part, GGX specular toward the eye, and the sky (the ambient
// light) mirrored by Fresnel's share, which rougher surfaces blur away. The
// levels add: a matte face toward the light gets the ambient and the
// directional light's levels of its colour, a face away the ambient's.
// Then its self-lit share of the albedo, and the light it gives off.
fn pbr_light(m: Material, lights: PbrLights) -> vec3<f32> {
    let n = m.normal;
    let l = lights.toward_light;
    let nl = max(dot(n, l), 0.0);
    let direct = lights.directional * nl * lights.shadow;
    let f0 = mix(vec3<f32>(0.04), m.albedo, m.metallic);
    let diffuse = m.albedo * (1.0 - m.metallic);
    let ambient = lights.ambient * m.occlusion;
    // Schlick's Fresnel with roughness: rough surfaces mirror little of
    // the sky even at grazing angles.
    let nv = max(dot(n, lights.toward_eye), 0.0);
    let sky = f0 + (max(vec3<f32>(1.0 - m.roughness), f0) - f0) * pow(1.0 - nv, 5.0);
    let lit = diffuse * (ambient + direct)
        + pbr_specular(n, lights.toward_eye, l, f0, m.roughness) * direct
        + sky * ambient * (1.0 - m.roughness * 0.7);
    return mix(lit, m.albedo, m.self_lit) + m.emitted;
}
