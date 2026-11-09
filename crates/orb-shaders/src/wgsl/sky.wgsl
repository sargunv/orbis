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
        let stars = atmosphere_stars(ray, max(pixel, 1e-7), frame.pixel_ratio) * atmosphere.star_opacity;
        under = stars + under * (1.0 - stars.a);
    }
    let color = sky + under * (1.0 - sky.a);
    if color.a <= 0.0 && all(color.rgb <= vec3<f32>(0.0)) {
        discard;
    }
    return color;
}

// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`):
// GL JS's model and exposure. What's beneath is dimmed by the
// transmittance, which on gamma-encoded colours is exactly its 1/2.2 power
// (GL JS's gamma-corrected alpha), and the scattered light is added: by
// default as GL JS adds it, gamma-encoded as if over black, and with
// `orbis:atmosphere` in linear light. Premultiplied, by the blend.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(atmosphere, px, frame.viewport);
    let scattered = atmosphere_scatter(atmosphere, ray);
    let light = 1.0 - exp(-scattered.rgb);
    let through = pow(scattered.a, 1.0 / ATMOSPHERE_GAMMA);
    let encoded = pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA));
    if atmosphere.scatter_linear <= 0.0 {
        return vec4<f32>(encoded, 1.0 - through) * atmosphere.scatter_blend;
    }
    // GL JS's addition is exact over black, but lifts a bright map's
    // sunlit face toward white (clipping its water to cyan), though the thin
    // column of air over it scatters little light next to what the surface
    // reflects. Blending can't read what's beneath, so the light is added
    // exactly for a reference colour: off the globe, the sky over the
    // backdrop, which the shader knows; on it, a mid-grey surface. A
    // surface's own colour then shows through tinted, the long paths toward
    // the limb glow, and a dark surface hazes a little less than it would.
    var under = vec3<f32>(0.5);
    let ground = atmosphere_sphere(atmosphere.globe_eye.xyz / atmosphere.globe_eye.w, ray, 1.0);
    if ground.x > ground.y || ground.y < 0.0 {
        let sky = atmosphere_sky(atmosphere, atmosphere_horizon(atmosphere, px, ray));
        under = sky.rgb + atmosphere.backdrop_color.rgb * (1.0 - sky.a);
    }
    let added = max(atmosphere_lit(under, scattered.a, light) - under * through, vec3<f32>(0.0));
    return vec4<f32>(added, 1.0 - through) * atmosphere.scatter_blend;
}

// An encoded colour dimmed by a transmittance with linear light added.
fn atmosphere_lit(under: vec3<f32>, through: f32, light: vec3<f32>) -> vec3<f32> {
    let gamma = vec3<f32>(ATMOSPHERE_GAMMA);
    return pow(pow(under, gamma) * through + light, 1.0 / gamma);
}
