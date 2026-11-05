@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
@fragment
fn fs_occlusion(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    var nearest = 0.0;
    for (var i = 0; i < i32(textureNumSamples(depth)); i++) {
        nearest = max(nearest, textureLoad(depth, vec2<i32>(position.xy), i));
    }
    return vec4<f32>(nearest, 0.0, 0.0, 0.0);
}
