// A map layer's fragment as the map's colour: the layers drawn straight
// into the map (`map_out_material.wgsl` is the lit map's).
struct MapOut {
    @location(0) color: vec4<f32>,
};

fn map_out(color: vec4<f32>, node: u32) -> MapOut {
    return MapOut(color);
}

fn map_out_self_lit(color: vec4<f32>) -> MapOut {
    return MapOut(color);
}
