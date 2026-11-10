// The atmosphere: one sky and fog model for
// every surface, from the style's `sky`. The sky shader draws it above the
// horizon; the terrain shader fogs its surface with it. Shaders bind the
// uniform themselves as `atmosphere`.
//
// The surface gives the horizon. Flat surfaces give a line: a point on it
// and its normal toward the sky, which a rolled camera turns. A globe gives
// its sphere: the sky is measured from its limb and fades to space outside
// a 100 km shell (GL JS's `sky.fragment.glsl` `u_sky_blend` terms), and a
// scattering pass over the map tints the limb (`atmosphere.fragment.glsl`).

struct Atmosphere {
    // Premultiplied style colours, in linear light when `linear` is 1.
    sky_color: vec4<f32>,
    horizon_color: vec4<f32>,
    fog_color: vec4<f32>,
    // style-spec#1889's backdrop, premultiplied, behind the stars and sky.
    backdrop_color: vec4<f32>,
    // Logical pixels, y down.
    horizon: vec2<f32>,
    horizon_normal: vec2<f32>,
    // `sky-horizon-blend` as a distance in logical pixels.
    sky_horizon_blend: f32,
    horizon_fog_blend: f32,
    fog_ground_blend: f32,
    // GL JS's fade from 60° to 70° of pitch; 0 turns fog off.
    fog_opacity: f32,
    // The fog's depth range along the view, in pixels (`View::fog_range_px`).
    fog_near: f32,
    fog_far: f32,
    // style-spec#1889's stars, 0 to 1.
    star_opacity: f32,
    // 1 when colours mix in linear light (`dev.sargunv.orbis:color-space`):
    // the colours above are decoded, fog mixes them as they are, and the
    // scattering adds to what's beneath exactly.
    linear: f32,
    // Clip → camera-relative space on Earth-fixed axes, for view rays.
    ray_inverse: mat4x4<f32>,
    // The globe, mixed in by `globe_amount` (0 on a flat surface): the eye
    // from the globe's centre with the radius in w; toward the sun, unit.
    globe_eye: vec4<f32>,
    globe_sun: vec4<f32>,
    globe_amount: f32,
    // GL JS's camera-to-centre distance, logical pixels.
    globe_eye_distance_px: f32,
    // `atmosphere-blend` with the eye's altitude: the sky's, and the
    // scattering pass's (also by `globe_amount`).
    sky_atmosphere_blend: f32,
    scatter_blend: f32,
};

// The atmosphere's shell above the globe, as a share of its radius.
const ATMOSPHERE_HEIGHT: f32 = 100000.0 / 6371000.0;
const ATMOSPHERE_PI: f32 = 3.141592653589793;

// The view ray through a logical pixel, camera-relative globe axes.
fn atmosphere_ray(a: Atmosphere, px: vec2<f32>, viewport: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(px.x / viewport.x * 2.0 - 1.0, 1.0 - px.y / viewport.y * 2.0);
    let near = a.ray_inverse * vec4<f32>(ndc, 0.0, 1.0);
    let far = a.ray_inverse * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(far.xyz / far.w - near.xyz / near.w);
}

const ATMOSPHERE_GAMMA: f32 = 2.2;

// How far a logical pixel is above the horizon (positive above it), with
// `ray` the view ray through it, and the sky's opacity there. On a globe
// the distance is measured from the limb and the sky fades to space.
fn atmosphere_horizon(a: Atmosphere, px: vec2<f32>, ray: vec3<f32>) -> vec2<f32> {
    var distance = dot(px - a.horizon, a.horizon_normal);
    var opacity = 1.0;
    if a.globe_amount > 0.0 {
        // In units of the radius, for precision.
        let center = -a.globe_eye.xyz / a.globe_eye.w;
        let globe_distance = length(center);
        let angle = acos(clamp(dot(ray, center) / globe_distance, -1.0, 1.0));
        let horizon = asin(min(1.0 / globe_distance, 1.0));
        distance = mix(distance, (angle - horizon) * a.globe_eye_distance_px + 1.0, a.globe_amount);
        let closest = max(dot(ray, center), 0.0);
        let lowest = sqrt(max(globe_distance * globe_distance - closest * closest, 0.0)) - 1.0;
        let inside = 1.0 - smoothstep(ATMOSPHERE_HEIGHT, 10.0 * ATMOSPHERE_HEIGHT, globe_distance - 1.0);
        let through = 1.0 - smoothstep(0.0, ATMOSPHERE_HEIGHT, lowest);
        opacity = mix(1.0, inside * through * (1.0 - a.sky_atmosphere_blend), a.globe_amount);
    }
    return vec2<f32>(distance, opacity);
}

// The sky at a pixel `horizon` gives (`atmosphere_horizon`): GL JS's
// gradient from the horizon colour to the sky colour above the horizon, or
// transparent below it.
fn atmosphere_sky(a: Atmosphere, horizon: vec2<f32>) -> vec4<f32> {
    let distance = horizon.x;
    let opacity = horizon.y;
    if distance <= 0.0 {
        return vec4<f32>(0.0);
    }
    var color = a.sky_color;
    if distance < a.sky_horizon_blend {
        color = mix(a.sky_color, a.horizon_color, pow(1.0 - distance / a.sky_horizon_blend, 2.0));
    }
    return color * opacity;
}

// style-spec#1889's stars along a view ray (unit, Earth-fixed axes, so
// they turn with the globe), `pixel` radians to a logical pixel; the
// premultiplied colour, before `star_opacity`. Two layers of hashed points,
// a sparse bright one and a dense faint one, each one point at most per
// cell of a cube about the eye, with cells of near-equal angle. A star is
// a Gaussian sized in logical pixels at least 0.7 device pixels wide, so
// it looks the same at any resolution and doesn't shimmer as it moves.
fn atmosphere_stars(ray: vec3<f32>, pixel: f32, pixel_ratio: f32) -> vec4<f32> {
    // The face the ray leaves the cube through, and where on it, in
    // equal-angle coordinates in [-1, 1].
    let m = abs(ray);
    var face: u32;
    var axes: mat3x3<f32>;
    if m.x >= m.y && m.x >= m.z {
        face = select(1u, 0u, ray.x > 0.0);
        axes = mat3x3<f32>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(sign(ray.x), 0.0, 0.0));
    } else if m.y >= m.z {
        face = select(3u, 2u, ray.y > 0.0);
        axes = mat3x3<f32>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, sign(ray.y), 0.0));
    } else {
        face = select(5u, 4u, ray.z > 0.0);
        axes = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, sign(ray.z)));
    }
    let local = transpose(axes) * ray;
    let uv = atan(local.xy / local.z) * (4.0 / ATMOSPHERE_PI);
    var sum = vec4<f32>(0.0);
    // Cells per face edge, the share of cells with a star, and the faintest
    // and brightest stars' peak and width (logical pixels).
    let cells = array<f32, 2>(48.0, 128.0);
    let share = array<f32, 2>(0.22, 0.42);
    let peak = array<vec2<f32>, 2>(vec2<f32>(0.35, 1.0), vec2<f32>(0.12, 0.45));
    let width = array<vec2<f32>, 2>(vec2<f32>(0.8, 1.35), vec2<f32>(0.6, 0.8));
    for (var layer = 0u; layer < 2u; layer++) {
        let n = cells[layer];
        let at = (uv * 0.5 + 0.5) * n;
        let cell = floor(at);
        // This cell and the three nearest the ray, within the face.
        let toward = select(vec2<f32>(-1.0), vec2<f32>(1.0), at - cell > vec2<f32>(0.5));
        for (var k = 0u; k < 4u; k++) {
            let c = cell + toward * vec2<f32>(f32(k & 1u), f32(k >> 1u));
            if any(c < vec2<f32>(0.0)) || any(c >= vec2<f32>(n)) {
                continue;
            }
            let h = atmosphere_hash(vec3<u32>(vec2<u32>(c), face * 2u + layer));
            let r = vec3<f32>(h >> vec3<u32>(8u)) / 16777216.0;
            if r.z >= share[layer] {
                continue;
            }
            // Brightness skewed faint, as stars' magnitudes are.
            let bright = pow(r.z / share[layer], 3.0);
            let star_uv = (c + 0.15 + 0.7 * r.xy) / n * 2.0 - 1.0;
            let star = axes * normalize(vec3<f32>(tan(star_uv * (ATMOSPHERE_PI / 4.0)), 1.0));
            let w = max(mix(width[layer].x, width[layer].y, bright), 0.7 / pixel_ratio);
            // The same light spread wider when a star is widened.
            let p = mix(peak[layer].x, peak[layer].y, bright) * pow(mix(width[layer].x, width[layer].y, bright) / w, 2.0);
            let d = length(ray - star) / pixel / w;
            let tint = f32(h.x & 255u) / 255.0;
            let color = mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0, 0.9, 0.76), tint);
            sum += vec4<f32>(color, 1.0) * p * exp(-0.5 * d * d);
        }
    }
    let alpha = min(sum.a, 1.0);
    return vec4<f32>(sum.rgb * (alpha / max(sum.a, 1e-6)), alpha);
}

// A 3D hash of a star cell: Jarzynski and Olano's pcg3d.
fn atmosphere_hash(seed: vec3<u32>) -> vec3<u32> {
    var v = seed * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

// A ray's intersections with a sphere about the origin: (near, far), near
// > far without one.
fn atmosphere_sphere(origin: vec3<f32>, ray: vec3<f32>, radius: f32) -> vec2<f32> {
    let b = 2.0 * dot(ray, origin);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - 4.0 * c;
    if d < 0.0 {
        return vec2<f32>(1e5, -1e5);
    }
    return vec2<f32>(-b - sqrt(d), -b + sqrt(d)) / 2.0;
}

// Light scattered toward the eye along a view ray through the globe's
// atmosphere, and how much of what's behind shows through (alpha): GL
// JS's port of glsl-atmosphere (Rayleigh and Mie, 5 and 3 steps), in
// metres on an Earth-sized globe.
fn atmosphere_scatter(a: Atmosphere, ray: vec3<f32>) -> vec4<f32> {
    let planet = 6371e3;
    let shell = 6471e3;
    let rayleigh = vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6);
    let mie = 21e-6;
    let rayleigh_height = 8e3;
    let mie_height = 1.2e3;
    let g = 0.758;
    let sun_intensity = 22.0;
    let origin = a.globe_eye.xyz / a.globe_eye.w * planet;
    let sun = normalize(a.globe_sun.xyz);
    var p = atmosphere_sphere(origin, ray, shell);
    if p.x > p.y {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    p.x = max(p.x, 0.0);
    let ground = atmosphere_sphere(origin, ray, planet);
    if ground.x <= ground.y && ground.x > 0.0 {
        p.y = min(p.y, ground.x);
    }
    let step = (p.y - p.x) / 5.0;
    var time = p.x + step * 0.5;
    var total_rayleigh = vec3<f32>(0.0);
    var total_mie = vec3<f32>(0.0);
    var depth_rayleigh = 0.0;
    var depth_mie = 0.0;
    let mu = dot(ray, sun);
    let gg = g * g;
    let phase_rayleigh = 3.0 / (16.0 * ATMOSPHERE_PI) * (1.0 + mu * mu);
    let phase_mie = 3.0 / (8.0 * ATMOSPHERE_PI) * ((1.0 - gg) * (mu * mu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));
    for (var i = 0; i < 5; i++) {
        let position = origin + ray * time;
        let height = length(position) - planet;
        let step_rayleigh = exp(-height / rayleigh_height) * step;
        let step_mie = exp(-height / mie_height) * step;
        depth_rayleigh += step_rayleigh;
        depth_mie += step_mie;
        let light_step = atmosphere_sphere(position, sun, shell).y / 3.0;
        var light_time = light_step * 0.5;
        var light_rayleigh = 0.0;
        var light_mie = 0.0;
        for (var j = 0; j < 3; j++) {
            let height = length(position + sun * light_time) - planet;
            light_rayleigh += exp(-height / rayleigh_height) * light_step;
            light_mie += exp(-height / mie_height) * light_step;
            light_time += light_step;
        }
        let attenuation = exp(-(mie * (depth_mie + light_mie) + rayleigh * (depth_rayleigh + light_rayleigh)));
        total_rayleigh += step_rayleigh * attenuation;
        total_mie += step_mie * attenuation;
        time += step;
    }
    let opacity = exp(-(length(rayleigh) * length(total_rayleigh) + mie * length(total_mie)));
    let color = sun_intensity * (phase_rayleigh * rayleigh * total_rayleigh + phase_mie * mie * total_mie);
    return vec4<f32>(color, opacity);
}

// The fog matrix's [0, 1] depth of a point `w` pixels along the view: a GL
// perspective from `fog_near` to `fog_far`.
fn atmosphere_fog_depth(a: Atmosphere, w: f32) -> f32 {
    return a.fog_far / (a.fog_far - a.fog_near) * (1.0 - a.fog_near / max(w, 1e-6));
}

// GL JS's terrain fog over a premultiplied surface colour at a fog depth,
// blended in linear space toward the horizon colour far off: through GL
// JS's 2.2 gamma for encoded colours, as they are in linear light.
// `ground` fogs the whole surface, drawn on or not, as GL JS's terrain
// does; otherwise only what is drawn, by its coverage: a terrain layer
// over the ground then fogs its own content and leaves what lies beneath
// as it was, so the fog lands once however many layers drape the surface.
fn atmosphere_fog(a: Atmosphere, surface: vec4<f32>, depth: f32, ground: bool) -> vec4<f32> {
    if a.fog_opacity <= 0.0 || depth <= a.fog_ground_blend || (!ground && surface.a <= 0.0) {
        return surface;
    }
    let blend = smoothstep(0.0, 1.0, max((depth - a.horizon_fog_blend) / (1.0 - a.horizon_fog_blend), 0.0));
    let factor = pow(max(depth - a.fog_ground_blend, 0.0) / (1.0 - a.fog_ground_blend), 2.0) * a.fog_opacity;
    if a.linear != 0.0 {
        let coverage = select(surface.a, 1.0, ground);
        return mix(surface, mix(a.fog_color, a.horizon_color, blend) * coverage, factor);
    }
    let gamma = vec4<f32>(ATMOSPHERE_GAMMA);
    let fog = mix(pow(a.fog_color, gamma), pow(a.horizon_color, gamma), blend);
    if ground {
        return pow(mix(pow(surface, gamma), fog, factor), 1.0 / gamma);
    }
    // Content alone: its straight colour fogged, covering as it did.
    let straight = vec4<f32>(surface.rgb / surface.a, 1.0);
    return pow(mix(pow(straight, gamma), fog, factor), 1.0 / gamma) * surface.a;
}
