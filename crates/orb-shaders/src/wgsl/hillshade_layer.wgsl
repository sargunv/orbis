// Hillshade layers (`hillshade.wgsl`'s methods). The node's texture array
// holds each tile's prepared slopes.

@group(1) @binding(0) var<uniform> node: HillshadeNode;

@vertex
fn vs_hillshade(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> ImageOut {
    return image_vertex(node.index, instance, index);
}

@fragment
fn fs_hillshade(in: ImageOut) -> MapOut {
    if image_clipped(in) {
        discard;
    }
    let v_pos = image_pos(in) / IMAGE_EXTENT;
    let size = vec2<f32>(textureDimensions(images));
    let pixel = sample_image(in, (v_pos * (size - 2.0) + 1.0) / size, node.nearest);
    let latitude = mix(in.latitudes.x, in.latitudes.y, v_pos.y);
    return map_out(hillshade_color(pixel, latitude), node.index, in.up);
}
