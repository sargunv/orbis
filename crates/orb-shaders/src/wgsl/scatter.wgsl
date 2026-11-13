// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`),
// drawn after its layers with the sky's bindings: GL JS's port of
// glsl-atmosphere (Rayleigh and Mie, five steps along the view ray), in
// metres on an Earth-sized globe, at every pixel.
//
// GL JS marches three more steps toward the sun from each of the five; how
// much light reaches a point that way depends only on its height and the
// sun's angle from its zenith, so `Atmosphere::sunward` (orb-render's
// `atmosphere.rs`) works that march out once, at every texel of a table
// over the two, with the constants below.
@group(1) @binding(0) var sunward: texture_2d<f32>;
@group(1) @binding(1) var sunward_sampler: sampler;

const SCATTER_PI: f32 = 3.141592653589793;
const SCATTER_PLANET: f32 = 6371e3;
const SCATTER_SHELL: f32 = 6471e3;
const SCATTER_RAYLEIGH: vec3<f32> = vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6);
const SCATTER_MIE: f32 = 21e-6;
const SCATTER_RAYLEIGH_HEIGHT: f32 = 8e3;
const SCATTER_MIE_HEIGHT: f32 = 1.2e3;
// Texels of the sunward table: across, the sun's angle; down, the height.
const SUNWARD_SIZE: vec2<f32> = vec2<f32>(256.0, 32.0);

// A ray's intersections with a sphere about the origin: (near, far), near
// > far without one.
fn scatter_sphere(origin: vec3<f32>, ray: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = 2.0 * dot(ray, origin);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - 4.0 * c;
    if d < 0.0 {
        return vec2<f32>(1e5, -1e5);
    }
    return vec2<f32>(-b - sqrt(d), -b + sqrt(d)) / 2.0;
}

// The share of each channel's light that reaches a point `r` from the
// globe's centre from a sun at cosine `mu` from its zenith. The table's
// rows go by the square root of the height, its columns by the cosine's
// distance from the horizon's, square-rooted either side of it, so both
// crowd where the light changes fastest: near the ground, and where the sun
// sets.
fn scatter_sunward(r: f32, mu: f32) -> vec3<f32> {
    let height = max(r - SCATTER_PLANET, 0.0);
    let row = sqrt(min(height / (SCATTER_SHELL - SCATTER_PLANET), 1.0));
    let horizon = -sqrt(height * (2.0 * SCATTER_PLANET + height)) / r;
    let past = (mu - horizon) / select(1.0 + horizon, 1.0 - horizon, mu >= horizon);
    let column = 0.5 + 0.5 * sign(past) * sqrt(abs(past));
    // The table's first and last texels' centres at 0 and 1.
    let uv = (vec2<f32>(column, row) * (SUNWARD_SIZE - 1.0) + 0.5) / SUNWARD_SIZE;
    return textureSampleLevel(sunward, sunward_sampler, uv, 0.0).rgb;
}

// The march along a view ray so far: where the next step lies, the air's
// optical depth behind it (Rayleigh, Mie), and the light scattered.
struct Scatter {
    time: f32,
    depth: vec2<f32>,
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
};

// One `step` along the view ray from `origin`.
fn scatter_step(s: Scatter, origin: vec3<f32>, ray: vec3<f32>, step: f32) -> Scatter {
    let position = origin + ray * s.time;
    let r = length(position);
    let height = r - SCATTER_PLANET;
    let rayleigh = exp(-height / SCATTER_RAYLEIGH_HEIGHT) * step;
    let mie = exp(-height / SCATTER_MIE_HEIGHT) * step;
    let depth = s.depth + vec2<f32>(rayleigh, mie);
    let sun = normalize(atmosphere.globe_sun.xyz);
    let attenuation = exp(-(SCATTER_MIE * depth.y + SCATTER_RAYLEIGH * depth.x))
        * scatter_sunward(r, dot(position, sun) / r);
    return Scatter(s.time + step, depth, s.rayleigh + rayleigh * attenuation, s.mie + mie * attenuation);
}

// Light scattered toward the eye along a view ray through the globe's
// atmosphere, and how much of what's behind shows through (alpha).
fn scatter(ray: vec3<f32>) -> vec4<f32> {
    let g = 0.758;
    let sun_intensity = 22.0;
    let origin = atmosphere.globe_eye.xyz / atmosphere.globe_eye.w * SCATTER_PLANET;
    var p = scatter_sphere(origin, ray, SCATTER_SHELL);
    if p.x > p.y {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    p.x = max(p.x, 0.0);
    let ground = scatter_sphere(origin, ray, SCATTER_PLANET);
    if ground.x <= ground.y && ground.x > 0.0 {
        p.y = min(p.y, ground.x);
    }
    let step = (p.y - p.x) / 5.0;
    // Five steps written out, not looped: naga bounds every loop it writes
    // for Metal with a counter, which cost a tenth of the pass.
    var s = Scatter(p.x + step * 0.5, vec2<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    s = scatter_step(s, origin, ray, step);
    s = scatter_step(s, origin, ray, step);
    s = scatter_step(s, origin, ray, step);
    s = scatter_step(s, origin, ray, step);
    s = scatter_step(s, origin, ray, step);
    let mu = dot(ray, normalize(atmosphere.globe_sun.xyz));
    let gg = g * g;
    let phase_rayleigh = 3.0 / (16.0 * SCATTER_PI) * (1.0 + mu * mu);
    let phase_mie = 3.0 / (8.0 * SCATTER_PI) * ((1.0 - gg) * (mu * mu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));
    let opacity = exp(-(length(SCATTER_RAYLEIGH) * length(s.rayleigh) + SCATTER_MIE * length(s.mie)));
    let color = sun_intensity * (phase_rayleigh * SCATTER_RAYLEIGH * s.rayleigh + phase_mie * SCATTER_MIE * s.mie);
    return vec4<f32>(color, opacity);
}

// GL JS's exposure, premultiplied, by the blend. What's beneath is dimmed
// by the transmittance and the scattered light added, as GL JS adds it:
// gamma-encoded, as if over black, with the transmittance's 1/2.2 power
// (GL JS's gamma-corrected alpha), which lifts a bright map's sunlit face
// toward white.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let scattered = scatter(atmosphere_ray(atmosphere, px, frame.viewport));
    let light = 1.0 - exp(-scattered.rgb);
    let through = pow(scattered.a, 1.0 / ATMOSPHERE_GAMMA);
    let encoded = pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA));
    return vec4<f32>(encoded, 1.0 - through) * atmosphere.scatter_blend;
}
