// The table-class stand-ins for `expr.wgsl`: a node without bytecode
// properties never calls them, and its pipeline compiles no interpreter.

fn expr_zoom(zoom: f32, bits: vec2<u32>) -> vec4<u32> {
    return vec4<u32>();
}

fn expr_value(p: u32, inputs: u32, state: u32, zoom: vec4<u32>, color: bool) -> vec4<f32> {
    return vec4<f32>();
}
