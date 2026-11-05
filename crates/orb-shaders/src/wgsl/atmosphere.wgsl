// The atmosphere (plan §8.6, decision 0025): one sky and fog model for
// every surface, from the style's `sky`. The sky shader draws it above the
// horizon; the terrain shader fogs its surface with it. Shaders bind the
// uniform themselves as `atmosphere`.
//
// The surface gives the horizon. Flat surfaces give a line: a point on it
// and its normal toward the sky, which a rolled camera turns. GL JS's globe
// sky (`sky.fragment.glsl`'s `u_sky_blend` terms) joins here with the globe
// surface, as a sphere in view space.

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
};

const ATMOSPHERE_GAMMA: f32 = 2.2;

// The sky at a logical pixel: GL JS's gradient from the horizon colour to
// the sky colour, or transparent below the horizon.
fn atmosphere_sky(a: Atmosphere, px: vec2<f32>) -> vec4<f32> {
    let distance = dot(px - a.horizon, a.horizon_normal);
    if distance <= 0.0 {
        return vec4<f32>(0.0);
    }
    if distance < a.sky_horizon_blend {
        return mix(a.sky_color, a.horizon_color, pow(1.0 - distance / a.sky_horizon_blend, 2.0));
    }
    return a.sky_color;
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
