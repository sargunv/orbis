// The 3D depth (terrain and 3D layers, reversed-Z) as a colour, for the
// symbols and circles it hides: the nearest of its samples.
@vertex
fn vs_occlusion(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(positions[index], 0.0, 1.0);
}
