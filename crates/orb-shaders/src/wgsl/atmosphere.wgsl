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
    // Premultiplied style colours.
    sky_color: vec4<f32>,
    horizon_color: vec4<f32>,
    fog_color: vec4<f32>,
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
    _pad0: f32,
    _pad1: f32,
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
    // Premultiplied; behind the stars and the sky.
    backdrop: vec4<f32>,
    // The stars' brightness, and their turn about the pole (radians).
    stars: f32,
    star_rotation: f32,
    _pad2: f32,
    _pad3: f32,
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

// The sky at a logical pixel: GL JS's gradient from the horizon colour to
// the sky colour above the horizon, or transparent below it. On a globe
// the gradient starts at the limb and the sky fades to space.
fn atmosphere_sky(a: Atmosphere, px: vec2<f32>, viewport: vec2<f32>) -> vec4<f32> {
    var distance = dot(px - a.horizon, a.horizon_normal);
    var opacity = 1.0;
    if a.globe_amount > 0.0 {
        // In units of the radius, for precision.
        let ray = atmosphere_ray(a, px, viewport);
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
    if distance <= 0.0 {
        return vec4<f32>(0.0);
    }
    var color = a.sky_color;
    if distance < a.sky_horizon_blend {
        color = mix(a.sky_color, a.horizon_color, pow(1.0 - distance / a.sky_horizon_blend, 2.0));
    }
    return color * opacity;
}

// Four uniform hashes of a cell (PCG4D, Jarzynski and Olano 2020).
fn atmosphere_hash(cell: vec4<u32>) -> vec4<f32> {
    var v = cell * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> vec4<u32>(16u);
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return vec4<f32>(v) / 4294967296.0;
}

// A direction's cube face and its equal-angle position on it, in [-1, 1]:
// cells of the face grid then span nearly equal solid angles.
fn atmosphere_face(d: vec3<f32>) -> vec3<f32> {
    let m = abs(d);
    var face = 0.0;
    var uv = vec2<f32>(0.0);
    if m.x >= m.y && m.x >= m.z {
        face = select(1.0, 0.0, d.x > 0.0);
        uv = d.yz / m.x;
    } else if m.y >= m.z {
        face = select(3.0, 2.0, d.y > 0.0);
        uv = d.xz / m.y;
    } else {
        face = select(5.0, 4.0, d.z > 0.0);
        uv = d.xy / m.z;
    }
    return vec3<f32>(atan(uv) * (4.0 / ATMOSPHERE_PI), face);
}

// The unit direction at an equal-angle position on a cube face.
fn atmosphere_unface(w: vec2<f32>, face: f32) -> vec3<f32> {
    let uv = tan(w * (ATMOSPHERE_PI / 4.0));
    let s = select(-1.0, 1.0, (u32(face) & 1u) == 0u);
    var d = vec3<f32>(s, uv);
    if face >= 4.0 {
        d = vec3<f32>(uv, s);
    } else if face >= 2.0 {
        d = vec3<f32>(uv.x, s, uv.y);
    }
    return normalize(d);
}

// The starfield along a celestial direction, premultiplied: a fixed
// decorative pattern of point stars, a constant number per steradian (two
// layers of hashed cells on an equal-angle cube, at most one star a cell,
// kept off the cell's edges), each a point spread `pixel` radians wide so
// it stays a sharp dot at any zoom and resolution.
fn atmosphere_starfield(d: vec3<f32>, pixel: f32) -> vec3<f32> {
    let at = atmosphere_face(d);
    // Many faint stars, and fewer, brighter ones.
    return atmosphere_stars(d, at, pixel, 0u, vec3<f32>(64.0, 0.35, 0.55))
        + atmosphere_stars(d, at, pixel, 1u, vec3<f32>(20.0, 0.3, 1.0));
}

// One layer of the starfield: `layer` is its cells per face edge, the share
// of them holding a star, and its brightest star.
fn atmosphere_stars(d: vec3<f32>, at: vec3<f32>, pixel: f32, index: u32, layer: vec3<f32>) -> vec3<f32> {
    let cell = floor((at.xy * 0.5 + 0.5) * layer.x);
    let h = atmosphere_hash(vec4<u32>(vec2<u32>(cell), u32(at.z), index));
    if h.x >= layer.y {
        return vec3<f32>(0.0);
    }
    let star = atmosphere_unface((cell + 0.2 + 0.6 * h.yz) / layer.x * 2.0 - 1.0, at.z);
    // Few bright stars, many faint.
    let brightness = layer.z * (0.15 + 0.85 * pow(h.w, 4.0));
    let sigma = pixel * (0.55 + 0.5 * brightness);
    let distance = length(d - star) / sigma;
    // Blue-white to orange, as stars' temperatures run.
    let tint = mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0, 0.86, 0.68), fract(h.w * 37.0));
    return tint * brightness * exp(-0.5 * distance * distance);
}

// What lies behind the sky at a logical pixel, premultiplied: the
// backdrop, and the stars over it where the view ray leaves the surface
// (past a globe's limb, above a flat surface's horizon). They turn with
// the Earth, by the sidereal angle when the scene has a time. `step` is a
// device pixel in logical pixels.
fn atmosphere_space(a: Atmosphere, px: vec2<f32>, viewport: vec2<f32>, step: f32) -> vec4<f32> {
    if a.stars <= 0.0 {
        return a.backdrop;
    }
    let ray = atmosphere_ray(a, px, viewport);
    var seen = dot(px - a.horizon, a.horizon_normal) > 0.0;
    if a.globe_amount > 0.0 {
        let center = -a.globe_eye.xyz / a.globe_eye.w;
        let closest = dot(ray, center);
        seen = closest <= 0.0 || dot(center, center) - closest * closest > 1.0;
    }
    if !seen {
        return a.backdrop;
    }
    let pixel = length(atmosphere_ray(a, px + vec2<f32>(step, 0.0), viewport) - ray);
    let c = cos(a.star_rotation);
    let s = sin(a.star_rotation);
    let celestial = vec3<f32>(c * ray.x - s * ray.y, s * ray.x + c * ray.y, ray.z);
    let light = atmosphere_starfield(celestial, pixel) * a.stars;
    let alpha = min(max(max(light.r, light.g), light.b), 1.0);
    return vec4<f32>(min(light, vec3<f32>(1.0)), alpha) + a.backdrop * (1.0 - alpha);
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

// GL JS's terrain fog over a surface colour at a fog depth, blended in
// linear space toward the horizon colour far off.
fn atmosphere_fog(a: Atmosphere, surface: vec4<f32>, depth: f32) -> vec4<f32> {
    if a.fog_opacity <= 0.0 || depth <= a.fog_ground_blend {
        return surface;
    }
    let gamma = vec4<f32>(ATMOSPHERE_GAMMA);
    let blend = smoothstep(0.0, 1.0, max((depth - a.horizon_fog_blend) / (1.0 - a.horizon_fog_blend), 0.0));
    let fog = mix(pow(a.fog_color, gamma), pow(a.horizon_color, gamma), blend);
    let factor = max(depth - a.fog_ground_blend, 0.0) / (1.0 - a.fog_ground_blend);
    return pow(mix(pow(surface, gamma), fog, pow(factor, 2.0) * a.fog_opacity), 1.0 / gamma);
}
