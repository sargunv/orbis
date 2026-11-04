// Flat-surface sky: pinned GL JS sky.fragment.glsl, with top-left pixel Y.
struct Sky {
    color: vec4<f32>,
    horizon_color: vec4<f32>,
    blend: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};
@group(0) @binding(0) var<uniform> sky_frame: Frame;
@group(0) @binding(1) var<uniform> sky: Sky;

@vertex
fn vs_sky(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_sky(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let distance = sky_frame.sky_horizon - position.y / sky_frame.pixel_ratio;
    if distance <= 0.0 { discard; }
    let blend_distance = sky.blend * sky_frame.viewport.y / 2.0;
    if distance < blend_distance {
        return mix(sky.color, sky.horizon_color, pow(1.0 - distance / blend_distance, 2.0));
    }
    return sky.color;
}
