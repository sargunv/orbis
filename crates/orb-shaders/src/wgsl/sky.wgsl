// The atmosphere's sky, drawn before the map layers: pinned GL JS's
// `sky.fragment.glsl` over a flat surface's horizon.
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
    let color = atmosphere_sky(atmosphere, atmosphere_horizon(atmosphere, px, ray));
    if color.a <= 0.0 && all(color.rgb <= vec3<f32>(0.0)) {
        discard;
    }
    return color;
}

// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`):
// GL JS's model and exposure, premultiplied, by the blend. What's beneath
// is dimmed by the transmittance and the scattered light added, as GL JS
// adds it: gamma-encoded, as if over black, with the transmittance's 1/2.2
// power (GL JS's gamma-corrected alpha), which lifts a bright map's sunlit
// face toward white.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(atmosphere, px, frame.viewport);
    let scattered = atmosphere_scatter(atmosphere, ray);
    let light = 1.0 - exp(-scattered.rgb);
    let through = pow(scattered.a, 1.0 / ATMOSPHERE_GAMMA);
    let encoded = pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA));
    return vec4<f32>(encoded, 1.0 - through) * atmosphere.scatter_blend;
}
