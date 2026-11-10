// Raster layers, after GL JS's `raster` shaders. Textures hold
// premultiplied colour. Arriving tiles cross-fade from the ancestor that
// covered them, or fade in (`image_fade`), over `raster-fade-duration`.

struct RasterNode {
    // GL JS's `u_spin_weights` in xyz.
    spin_weights: vec4<f32>,
    opacity: f32,
    brightness_low: f32,
    brightness_high: f32,
    saturation_factor: f32,
    contrast_factor: f32,
    index: u32,
    // 1 to filter by nearest texel.
    nearest: u32,
    // 1 to read the full-size level alone, without mip levels.
    level0: u32,
    // `raster-fade-duration`, milliseconds; 0 for none.
    fade_duration: f32,
};

@group(1) @binding(0) var<uniform> node: RasterNode;

@vertex
fn vs_raster(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> ImageOut {
    return image_vertex(node.index, instance, index);
}

@fragment
fn fs_raster(in: ImageOut) -> @location(0) vec4<f32> {
    // The texture's footprint, in uniform control flow, for its mip levels.
    let uv = image_pos(in) / IMAGE_EXTENT;
    let ddx = dpdx(uv);
    let ddy = dpdy(uv);
    if image_clipped(in) {
        discard;
    }
    var color: vec4<f32>;
    if node.level0 != 0u {
        color = sample_image(in, uv, node.nearest);
    } else {
        color = sample_image_grad(in, uv, node.nearest, ddx, ddy);
    }
    let fade = image_fade(in, node.fade_duration);
    if fade.y > 0.0 {
        // The ancestor's texels at this point, in the same page.
        let a = image_ancestor_uv(in, uv);
        var ancestor = in;
        ancestor.layer = in.fade.y;
        let older = sample_image_grad(ancestor, a.xy, node.nearest, ddx * a.z, ddy * a.z);
        color = mix(color, older, fade.y);
    }
    color *= fade.x;
    if color.a > 0.0 {
        color = vec4<f32>(color.rgb / color.a, color.a);
    }
    color.a *= node.opacity;
    var rgb = color.rgb;

    // spin
    let w = node.spin_weights.xyz;
    rgb = vec3<f32>(dot(rgb, w.xyz), dot(rgb, w.zxy), dot(rgb, w.yzx));

    // saturation
    let average = (color.r + color.g + color.b) / 3.0;
    rgb += (average - rgb) * node.saturation_factor;

    // contrast
    rgb = (rgb - 0.5) * node.contrast_factor + 0.5;

    // brightness
    let high = vec3<f32>(node.brightness_low);
    let low = vec3<f32>(node.brightness_high);
    return color_in(vec4<f32>(mix(high, low, rgb) * color.a, color.a));
}
