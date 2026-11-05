// glTF's metallic-roughness shading under the style's lights (plan §9.4,
// decision 0026), as functions of their inputs alone: no bindings, so any
// pipeline composes it. Directions are in east, north and up; colours are
// linear.

const PBR_PI: f32 = 3.14159265;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

// `orb_3d::gpu`'s octahedral normal: two snorm16s, x low.
fn octahedral_normal(w: u32) -> vec3<f32> {
    let e = vec2<f32>(f32(bitcast<i32>(w << 16u) >> 16u), f32(bitcast<i32>(w) >> 16u)) / 32767.0;
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    if n.z < 0.0 {
        n = vec3<f32>((1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0)), n.z);
    }
    let l = length(n);
    return select(vec3<f32>(0.0, 0.0, 1.0), n / l, l > 0.0);
}

// Turns `v` about east, then north, then up (clockwise seen from above, as
// a heading), by `degrees`: a model layer's `model-rotation`.
fn turn_enu(v: vec3<f32>, degrees: vec3<f32>) -> vec3<f32> {
    let r = radians(degrees);
    let x = vec3<f32>(v.x, v.y * cos(r.x) - v.z * sin(r.x), v.y * sin(r.x) + v.z * cos(r.x));
    let y = vec3<f32>(x.x * cos(r.y) + x.z * sin(r.y), x.y, -x.x * sin(r.y) + x.z * cos(r.y));
    return vec3<f32>(y.x * cos(r.z) + y.y * sin(r.z), -y.x * sin(r.z) + y.y * cos(r.z), y.z);
}

// Texture coordinates wrapped as a glTF sampler says, per axis two bits
// of `wrap` (0 repeat, 1 clamp to edge, 2 mirrored repeat), for a sampler
// that repeats.
fn wrap_uv(uv: vec2<f32>, wrap: u32) -> vec2<f32> {
    var t = uv;
    for (var axis = 0u; axis < 2u; axis++) {
        let mode = (wrap >> (2u * axis)) & 3u;
        if mode == 1u {
            t[axis] = clamp(t[axis], 0.0, 1.0);
        } else if mode == 2u {
            t[axis] = 1.0 - abs(fract(t[axis] * 0.5) * 2.0 - 1.0);
        }
    }
    return t;
}

// A normal map's normal (`sampled`, the texel in [0, 1]) on surface
// normal `n`, with the tangent frame from screen derivatives of the
// position and texture coordinates, so meshes need no tangents.
fn perturb_normal(n: vec3<f32>, dp: mat2x3<f32>, duv: mat2x2<f32>, sampled: vec3<f32>, scale: f32) -> vec3<f32> {
    let t = dp[0] * duv[1].y - dp[1] * duv[0].y;
    let b = dp[1] * duv[0].x - dp[0] * duv[1].x;
    let size = max(length(t), length(b));
    if size == 0.0 {
        return n;
    }
    let m = sampled * 2.0 - 1.0;
    return normalize(t / size * m.x * scale + b / size * m.y * scale + n * m.z);
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

// A surface's material: all its shading needs besides the lights, the one
// model the lit map's layers (`map_out_lit.wgsl`) and models fill (#91,
// decision 0048). Colours are linear; the normal is a unit vector in
// east, north and up.
struct Material {
    albedo: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    // Ambient light reaching it, 0 to 1.
    occlusion: f32,
    // How much of its albedo shows whatever the lights (glTF's unlit is 1;
    // a model layer's `model-emissive-strength`).
    self_lit: f32,
    // Light it gives off, added after shading.
    emitted: vec3<f32>,
    normal: vec3<f32>,
};

// A material's linear colour under `lights`: Lambert diffuse for its
// non-metal part, GGX specular toward the eye, and the ambient light
// reaching metals as a dim reflection rougher surfaces blur away. The
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
    let lit = diffuse * (ambient + direct)
        + pbr_specular(n, lights.toward_eye, l, f0, m.roughness) * direct
        + f0 * ambient * (1.0 - m.roughness * 0.7) * m.metallic;
    return mix(lit, m.albedo, m.self_lit) + m.emitted;
}
