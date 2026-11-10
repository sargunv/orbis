// The atmosphere's sky, drawn before the map layers: pinned GL JS's
// `sky.fragment.glsl` over a flat surface's horizon, over style-spec#1889's
// backdrop and the stars on it, which show above the horizon.
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> atmosphere: Atmosphere;

@vertex
fn vs_sky(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_sky(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(atmosphere, px, frame.viewport);
    // Radians to a logical pixel, from the ray's change across the quad.
    let pixel = 0.5 * (length(dpdx(ray)) + length(dpdy(ray))) * frame.pixel_ratio;
    let horizon = atmosphere_horizon(atmosphere, px, ray);
    let sky = atmosphere_sky(atmosphere, horizon);
    var under = atmosphere.backdrop_color;
    if atmosphere.star_opacity > 0.0 && horizon.x > 0.0 {
        var stars = atmosphere_stars(ray, max(pixel, 1e-7), frame.pixel_ratio);
        if atmosphere.linear != 0.0 {
            // Light, as over black: each premultiplied component decoded,
            // so a star looks as it does in sRGB.
            stars = vec4<f32>(srgb_to_linear(stars.rgb), stars.a);
        }
        stars *= atmosphere.star_opacity;
        under = stars + under * (1.0 - stars.a);
    }
    let color = sky + under * (1.0 - sky.a);
    if color.a <= 0.0 && all(color.rgb <= vec3<f32>(0.0)) {
        discard;
    }
    return color;
}

// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`):
// GL JS's model and exposure, premultiplied, by the blend. What's beneath
// is dimmed by the transmittance and the scattered light added. In sRGB,
// as GL JS adds it: gamma-encoded, as if over black, with the
// transmittance's 1/2.2 power (GL JS's gamma-corrected alpha), which lifts
// a bright map's sunlit face toward white. In linear light, blending adds
// it to what's beneath exactly: a surface keeps its colours, tinted, and
// the long paths toward the limb glow.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(atmosphere, px, frame.viewport);
    let scattered = atmosphere_scatter(atmosphere, ray);
    let light = 1.0 - exp(-scattered.rgb);
    if atmosphere.linear != 0.0 {
        return vec4<f32>(light, 1.0 - scattered.a) * atmosphere.scatter_blend;
    }
    let through = pow(scattered.a, 1.0 / ATMOSPHERE_GAMMA);
    let encoded = pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA));
    return vec4<f32>(encoded, 1.0 - through) * atmosphere.scatter_blend;
}
