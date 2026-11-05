// `fill-layer-opacity` and `line-layer-opacity`: a layer drawn into a
// texture of its own composites into the map once at its opacity, so its
// overlaps don't accumulate (GL JS `draw_layer_opacity.ts`). Texels are
// premultiplied.

@group(0) @binding(0) var<uniform> layer_opacity: vec4<f32>;
@group(0) @binding(1) var layer_color: texture_2d<f32>;

@vertex
fn vs_layer_composite(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_layer_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(layer_color, vec2<i32>(position.xy), 0) * layer_opacity.x;
}
