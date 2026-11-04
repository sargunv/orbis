// Hillshade layers, after GL JS's `hillshade` fragment shader. The node's
// texture array holds each tile's slopes from the DEM prepare pass
// (`prepare.wgsl`): (dim + 2)² texels, a one-texel border around the tile.

const PI: f32 = 3.141592653589793;
// The most light sources a node holds.
const MAX_LIGHTS: u32 = 4u;

const STANDARD: u32 = 0u;
const COMBINED: u32 = 1u;
const IGOR: u32 = 2u;
const MULTIDIRECTIONAL: u32 = 3u;
const BASIC: u32 = 4u;

struct HillshadeNode {
    accent: vec4<f32>,
    shadows: array<vec4<f32>, 4>,
    highlights: array<vec4<f32>, 4>,
    // Radians, per light.
    altitudes: vec4<f32>,
    // Radians, per light, with the map's bearing for viewport anchoring.
    azimuths: vec4<f32>,
    exaggeration: f32,
    method: u32,
    lights: u32,
    index: u32,
    nearest: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var<uniform> node: HillshadeNode;

@vertex
fn vs_hillshade(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> ImageOut {
    return image_vertex(node.index, instance, index);
}

fn get_aspect(deriv: vec2<f32>) -> f32 {
    if deriv.x != 0.0 {
        return atan2(deriv.y, -deriv.x);
    }
    return PI / 2.0 * select(-1.0, 1.0, deriv.y > 0.0);
}

// GLSL's `mod`: x − y · floor(x / y).
fn glsl_mod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn igor_hillshade(d: vec2<f32>) -> vec4<f32> {
    let deriv = d * node.exaggeration * 2.0;
    let aspect = get_aspect(deriv);
    let azimuth = node.azimuths[0] + PI;
    let slope_strength = atan(length(deriv)) * 2.0 / PI;
    let aspect_strength = 1.0 - abs(glsl_mod((aspect + azimuth) / PI + 0.5, 2.0) - 1.0);
    let shadow_strength = slope_strength * aspect_strength;
    let highlight_strength = slope_strength * (1.0 - aspect_strength);
    return node.shadows[0] * shadow_strength + node.highlights[0] * highlight_strength;
}

fn standard_hillshade(deriv: vec2<f32>) -> vec4<f32> {
    let azimuth = node.azimuths[0] + PI;
    let slope = atan(0.625 * length(deriv));
    let aspect = get_aspect(deriv);
    let intensity = node.exaggeration;
    let base = 1.875 - intensity * 1.75;
    let max_value = 0.5 * PI;
    var scaled_slope = slope;
    if intensity != 0.5 {
        scaled_slope = ((pow(base, slope) - 1.0) / (pow(base, max_value) - 1.0)) * max_value;
    }
    let accent = cos(scaled_slope);
    let accent_color = (1.0 - accent) * node.accent * clamp(intensity * 2.0, 0.0, 1.0);
    let shade = abs(glsl_mod((aspect + azimuth) / PI + 0.5, 2.0) - 1.0);
    let shade_color = mix(node.shadows[0], node.highlights[0], shade) * sin(scaled_slope)
        * clamp(intensity * 2.0, 0.0, 1.0);
    return accent_color * (1.0 - shade_color.a) + shade_color;
}

fn basic_hillshade(d: vec2<f32>) -> vec4<f32> {
    let deriv = d * node.exaggeration * 2.0;
    let azimuth = node.azimuths[0] + PI;
    let cos_az = cos(azimuth);
    let sin_az = sin(azimuth);
    let cos_alt = cos(node.altitudes[0]);
    let sin_alt = sin(node.altitudes[0]);
    let cang = (sin_alt - (deriv.y * cos_az * cos_alt - deriv.x * sin_az * cos_alt))
        / sqrt(1.0 + dot(deriv, deriv));
    let shade = clamp(cang, 0.0, 1.0);
    if shade > 0.5 {
        return node.highlights[0] * (2.0 * shade - 1.0);
    }
    return node.shadows[0] * (1.0 - 2.0 * shade);
}

fn multidirectional_hillshade(d: vec2<f32>) -> vec4<f32> {
    let deriv = d * node.exaggeration * 2.0;
    let count = f32(node.lights);
    var color = vec4<f32>(0.0);
    for (var i = 0u; i < node.lights; i++) {
        let cos_alt = cos(node.altitudes[i]);
        let sin_alt = sin(node.altitudes[i]);
        let cos_az = -cos(node.azimuths[i]);
        let sin_az = -sin(node.azimuths[i]);
        let cang = (sin_alt - (deriv.y * cos_az * cos_alt - deriv.x * sin_az * cos_alt))
            / sqrt(1.0 + dot(deriv, deriv));
        let shade = clamp(cang, 0.0, 1.0);
        if shade > 0.5 {
            color += node.highlights[i] * (2.0 * shade - 1.0) / count;
        } else {
            color += node.shadows[i] * (1.0 - 2.0 * shade) / count;
        }
    }
    return color;
}

fn combined_hillshade(d: vec2<f32>) -> vec4<f32> {
    let deriv = d * node.exaggeration * 2.0;
    let azimuth = node.azimuths[0] + PI;
    let cos_az = cos(azimuth);
    let sin_az = sin(azimuth);
    let cos_alt = cos(node.altitudes[0]);
    let sin_alt = sin(node.altitudes[0]);
    var cang = acos((sin_alt - (deriv.y * cos_az * cos_alt - deriv.x * sin_az * cos_alt))
        / sqrt(1.0 + dot(deriv, deriv)));
    cang = clamp(cang, 0.0, PI / 2.0);
    let shade = cang * atan(length(deriv)) * 4.0 / PI / PI;
    let highlight = (PI / 2.0 - cang) * atan(length(deriv)) * 4.0 / PI / PI;
    return node.shadows[0] * shade + node.highlights[0] * highlight;
}

@fragment
fn fs_hillshade(in: ImageOut) -> @location(0) vec4<f32> {
    if image_clipped(in) {
        discard;
    }
    let v_pos = in.tile_pos / IMAGE_EXTENT;
    let size = vec2<f32>(textureDimensions(images));
    let pixel = sample_image(in, (v_pos * (size - 2.0) + 1.0) / size, node.nearest);

    // Slopes shrink by the cosine of the latitude, for Mercator's
    // stretching (GL JS #4807).
    let lat_top = in.latitudes.x;
    let lat_bottom = in.latitudes.y;
    let scale_factor = cos(radians((lat_top - lat_bottom) * (1.0 - v_pos.y) + lat_bottom));
    let deriv = (pixel.rg - 128.0 / 255.0) * 8.0 / scale_factor;

    switch node.method {
        case BASIC: {
            return basic_hillshade(deriv);
        }
        case COMBINED: {
            return combined_hillshade(deriv);
        }
        case IGOR: {
            return igor_hillshade(deriv);
        }
        case MULTIDIRECTIONAL: {
            return multidirectional_hillshade(deriv);
        }
        default: {
            return standard_hillshade(deriv);
        }
    }
}
