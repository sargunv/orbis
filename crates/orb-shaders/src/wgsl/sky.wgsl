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
    let sky = atmosphere_sky(atmosphere, px, frame.viewport);
    if sky.a <= 0.0 && all(sky.rgb <= vec3<f32>(0.0)) {
        discard;
    }
    return sky;
}

// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`):
// exposure and gamma as GL JS applies them, premultiplied by the blend.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let ray = atmosphere_ray(atmosphere, position.xy / frame.pixel_ratio, frame.viewport);
    let scattered = atmosphere_scatter(atmosphere, ray);
    let color = pow(vec4<f32>(1.0 - exp(-scattered.rgb), scattered.a), vec4<f32>(1.0 / 2.2));
    return vec4<f32>(color.rgb, 1.0 - color.a) * atmosphere.scatter_blend;
}
