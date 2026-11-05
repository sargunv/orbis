// Contour lines of a raster-dem source (the scene document's `contour`
// layers), worked out per fragment: its elevation over the node's interval
// is a level, and its distance in pixels to the nearest whole level is that
// level's fraction over the level's change across a pixel. The node's
// texture array holds each tile's DEM with a two-texel border filled from
// the neighbours the view reads (`prepare.wgsl`), sampled as color-relief
// samples it, so lines meet across tile edges wherever those agree.

struct ContourNode {
    // The source's DEM unpack vector.
    unpack: vec4<f32>,
    // Premultiplied, at the layer's opacity.
    color: vec4<f32>,
    // Metres between lines, at the camera's zoom.
    interval: f32,
    // Every this many lines is a major one.
    major: f32,
    // Logical pixels.
    width: f32,
    major_width: f32,
    index: u32,
};

@group(1) @binding(0) var<uniform> node: ContourNode;

@vertex
fn vs_contour(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> ImageOut {
    return image_vertex(node.index, instance, index);
}

// How much of a pixel `distance` pixels from a line `width` pixels wide
// the line covers.
fn contour_coverage(distance: f32, width: f32) -> f32 {
    return clamp(width * 0.5 + 0.5 - distance, 0.0, 1.0);
}

// The distance in pixels from `level` to the nearest whole level, its
// change across a pixel `slope`; none where the ground is flat.
fn contour_distance(level: f32, slope: f32) -> f32 {
    return select(1e9, abs(fract(level + 0.5) - 0.5) / slope, slope > 1e-7);
}

@fragment
fn fs_contour(in: ImageOut) -> MapOut {
    let dimension = f32(textureDimensions(images).x);
    let v_pos = (image_pos(in) / IMAGE_EXTENT) * ((dimension - 4.0) / dimension) + 2.0 / dimension;
    let level = dem_elevation(sample_image(in, v_pos, 0u), node.unpack) / max(node.interval, 1e-3);
    // Levels per pixel, where control flow is uniform.
    let slope = length(vec2<f32>(dpdx(level), dpdy(level)));
    if image_clipped(in) {
        discard;
    }
    let major = max(node.major, 1.0);
    // Lines closer than a few pixels fill the ground rather than draw it:
    // minor lines fade out as theirs close from four logical pixels to
    // two, major ones as theirs do.
    let px = frame.pixel_ratio;
    let minor = contour_coverage(contour_distance(level, slope), node.width * px)
        * (1.0 - smoothstep(0.25 / px, 0.5 / px, slope));
    let majors = contour_coverage(contour_distance(level / major, slope / major), node.major_width * px)
        * (1.0 - smoothstep(0.25 / px, 0.5 / px, slope / major));
    return map_out(node.color * max(minor, majors), node.index, in.up);
}
