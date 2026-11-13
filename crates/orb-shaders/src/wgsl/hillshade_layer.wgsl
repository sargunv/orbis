// Hillshade layers (`hillshade.wgsl`). The node's texture array holds each
// tile's prepared slopes, and its table their colours.

@group(1) @binding(0) var<uniform> node: HillshadeNode;
@group(1) @binding(4) var table: texture_2d<f32>;

@vertex
fn vs_hillshade(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> ImageOut {
    return image_vertex(node.index, instance, index);
}

@fragment
fn fs_hillshade(in: ImageOut) -> @location(0) vec4<f32> {
    if image_clipped(in) {
        discard;
    }
    let v_pos = image_pos(in) / IMAGE_EXTENT;
    let size = vec2<f32>(image_size());
    let pixel = sample_image(in, (v_pos * (size - 2.0) + 1.0) / size, node.nearest);
    let latitude = mix(in.latitudes.x, in.latitudes.y, v_pos.y);
    let slope = hillshade_slope(pixel, mercator_stretch(latitude));
    return textureSampleLevel(table, linear_sampler, hillshade_table_uv(slope), 0.0);
}

// A hillshade the terrain shades: its slopes, unlit, into its drape stack,
// where the terrain pass lights them from the frame's bearing
// (`terrain_shade.wgsl`). Premultiplied, so mip levels and slot edges
// that average with blank texels keep the slopes in proportion.
@fragment
fn fs_hillshade_slopes(in: ImageOut) -> @location(0) vec4<f32> {
    if image_clipped(in) {
        discard;
    }
    let v_pos = image_pos(in) / IMAGE_EXTENT;
    let size = vec2<f32>(image_size());
    let pixel = sample_image(in, (v_pos * (size - 2.0) + 1.0) / size, node.nearest);
    return vec4<f32>(pixel.rg, 0.0, 1.0);
}
