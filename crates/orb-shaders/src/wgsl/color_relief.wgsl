// Color-relief layers, after GL JS's `color_relief` shaders. The node's
// texture array holds each tile's DEM with a two-texel border filled from
// its neighbours (`prepare.wgsl`). The ramp texture's row 0 holds the stop
// colours (premultiplied), row 1 the stop elevations packed as DEM texels.

struct ColorReliefNode {
    // The source's DEM unpack vector.
    unpack: vec4<f32>,
    opacity: f32,
    ramp_size: u32,
    index: u32,
    nearest: u32,
};

@group(1) @binding(0) var<uniform> node: ColorReliefNode;
@group(1) @binding(4) var ramp: texture_2d<f32>;

@vertex
fn vs_color_relief(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> ImageOut {
    return image_vertex(node.index, instance, index);
}

fn elevation_stop(stop: i32) -> f32 {
    return dem_elevation(textureLoad(ramp, vec2<i32>(stop, 1), 0), node.unpack);
}

@fragment
fn fs_color_relief(in: ImageOut) -> @location(0) vec4<f32> {
    if image_clipped(in) {
        discard;
    }
    let dimension = f32(textureDimensions(images).x);
    let epsilon = 2.0 / dimension;
    let scale = (dimension - 4.0) / dimension;
    let v_pos = (image_pos(in) / IMAGE_EXTENT) * scale + epsilon;
    let el = dem_elevation(sample_image(in, v_pos, node.nearest), node.unpack);

    // Binary search.
    var r = i32(node.ramp_size) - 1;
    var l = 0;
    var el_l = elevation_stop(l);
    var el_r = elevation_stop(r);
    while r - l > 1 {
        let m = (r + l) / 2;
        let el_m = elevation_stop(m);
        if el < el_m {
            r = m;
            el_r = el_m;
        } else {
            l = m;
            el_l = el_m;
        }
    }

    let x = (f32(l) + (el - el_l) / (el_r - el_l) + 0.5) / f32(node.ramp_size);
    return node.opacity * textureSampleLevel(ramp, linear_sampler, vec2<f32>(x, 0.25), 0.0);
}
