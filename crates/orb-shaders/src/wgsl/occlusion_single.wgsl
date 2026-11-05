@group(0) @binding(0) var depth: texture_depth_2d;
@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(depth, vec2<i32>(position.xy), 0), 0.0, 0.0, 0.0);
}
