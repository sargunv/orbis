@group(1) @binding(1) var layer_color: texture_2d<f32>;
struct CompositeOut { @builtin(position) position: vec4<f32>, };
@vertex
fn vs_extrusion_composite(@builtin(vertex_index) index: u32) -> CompositeOut {
    let positions = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
    var out: CompositeOut;
    out.position = vec4<f32>(positions[index],0.0,1.0);
    return out;
}
@fragment
fn fs_extrusion_composite(in: CompositeOut) -> @location(0) vec4<f32> {
    return textureLoad(layer_color, vec2<i32>(in.position.xy), 0) * node.opacity;
}
