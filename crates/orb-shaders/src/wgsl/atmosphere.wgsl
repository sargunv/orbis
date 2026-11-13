// The atmosphere: one sky and fog model for
// every surface, from the style's `sky`. The sky shader draws it above the
// horizon; the terrain shader fogs its surface with it. Shaders bind the
// uniform themselves as `atmosphere`.
//
// The surface gives the horizon. Flat surfaces give a line: a point on it
// and its normal toward the sky, which a rolled camera turns. A globe gives
// its sphere: the sky is measured from its limb and fades to space outside
// a 100 km shell (GL JS's `sky.fragment.glsl` `u_sky_blend` terms), and a
// scattering pass over the map tints the limb (`scatter.wgsl`).

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
// the distance is measured from a pixel below the limb and the sky fades
// to space.
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
//
// On either surface the sky reaches a pixel below the horizon, under the
// ground, in the horizon colour: the sky draws per pixel and the ground
// per sample, so the row whose centre lies just below the horizon would
// otherwise keep the samples above the ground's edge clear. The globe's
// distance already counts from a pixel below its limb, the plane's from
// the horizon itself, so the cut moves by the globe's share.
fn atmosphere_sky(a: Atmosphere, horizon: vec2<f32>) -> vec4<f32> {
    let distance = max(horizon.x, 0.0);
    let opacity = horizon.y;
    if horizon.x <= a.globe_amount - 1.0 {
        return vec4<f32>(0.0);
    }
    var color = a.sky_color;
    if distance < a.sky_horizon_blend {
        color = mix(a.sky_color, a.horizon_color, pow(1.0 - distance / a.sky_horizon_blend, 2.0));
    }
    return color * opacity;
}

// The fog matrix's [0, 1] depth of a point `w` pixels along the view: a GL
// perspective from `fog_near` to `fog_far`.
fn atmosphere_fog_depth(a: Atmosphere, w: f32) -> f32 {
    return a.fog_far / (a.fog_far - a.fog_near) * (1.0 - a.fog_near / max(w, 1e-6));
}

// How much of the fog's colour a surface at a fog depth takes.
fn atmosphere_fog_factor(a: Atmosphere, depth: f32) -> f32 {
    if a.fog_opacity <= 0.0 || depth <= a.fog_ground_blend {
        return 0.0;
    }
    return pow(max(depth - a.fog_ground_blend, 0.0) / (1.0 - a.fog_ground_blend), 2.0) * a.fog_opacity;
}

// GL JS's terrain fog over a premultiplied surface colour at a fog depth,
// blended in linear space toward the horizon colour far off: through GL
// JS's 2.2 gamma.
// `ground` fogs the whole surface, drawn on or not, as GL JS's terrain
// does; otherwise only what is drawn, by its coverage: a terrain layer
// over the ground then fogs its own content and leaves what lies beneath
// as it was, so the fog lands once however many layers drape the surface.
fn atmosphere_fog(a: Atmosphere, surface: vec4<f32>, depth: f32, ground: bool) -> vec4<f32> {
    if a.fog_opacity <= 0.0 || depth <= a.fog_ground_blend || (!ground && surface.a <= 0.0) {
        return surface;
    }
    let blend = smoothstep(0.0, 1.0, max((depth - a.horizon_fog_blend) / (1.0 - a.horizon_fog_blend), 0.0));
    let factor = atmosphere_fog_factor(a, depth);
    let gamma = vec4<f32>(ATMOSPHERE_GAMMA);
    let fog = mix(pow(a.fog_color, gamma), pow(a.horizon_color, gamma), blend);
    if ground {
        return pow(mix(pow(surface, gamma), fog, factor), 1.0 / gamma);
    }
    // Content alone: its straight colour fogged, covering as it did.
    let straight = vec4<f32>(surface.rgb / surface.a, 1.0);
    return pow(mix(pow(straight, gamma), fog, factor), 1.0 / gamma) * surface.a;
}
