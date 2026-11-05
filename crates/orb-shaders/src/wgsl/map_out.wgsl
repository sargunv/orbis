// A map layer's fragment as the map's colour (`map_out_lit.wgsl` lights
// it for a scene's lights).
struct MapOut {
    @location(0) color: vec4<f32>,
};

fn map_out(color: vec4<f32>, node: u32, position: vec4<f32>) -> MapOut {
    return MapOut(color);
}
