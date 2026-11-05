// A map layer's fragment as the map's colour (`map_out_lit.wgsl` lights
// it for a scene's lights).
struct MapOut {
    @location(0) color: vec4<f32>,
};

fn map_out(color: vec4<f32>, node: u32, position: vec4<f32>) -> MapOut {
    return MapOut(color);
}

// A surface that isn't the flat map (an extrusion's): its colour as the
// unlit map shades it, its albedo (premultiplied), its normal (east,
// north, up), the share of the directional light its shadows leave, the
// ambient light reaching it, its node, and on a wall where along it and how high (metres), that
// position's change across a pixel (taken where control flow is uniform)
// and a value per building, for windows.
struct SurfaceIn {
    color: vec4<f32>,
    albedo: vec4<f32>,
    normal: vec3<f32>,
    shadow: f32,
    occlusion: f32,
    node: u32,
    wall: vec2<f32>,
    wall_width: vec2<f32>,
    seed: u32,
};

fn surface_out(s: SurfaceIn) -> vec4<f32> {
    return s.color;
}
