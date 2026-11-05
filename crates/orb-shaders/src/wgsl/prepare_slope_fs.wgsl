@fragment
fn fs_slope(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return slope_texel(uniform_job(), vec2<u32>(position.xy));
}
