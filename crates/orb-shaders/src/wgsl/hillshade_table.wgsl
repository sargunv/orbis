// A hillshade node's table (`hillshade.wgsl`): its colour for every slope,
// by GL JS's `hillshade` fragment shader's methods over the node's paint,
// drawn whenever the paint changes. One triangle over the table, one
// fragment per texel.

const PI: f32 = 3.141592653589793;

@group(0) @binding(0) var<uniform> node: HillshadeNode;

@vertex
fn vs_hillshade_table(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_hillshade_table(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let deriv = hillshade_table_slope(position.xy / HILLSHADE_TABLE_SIDE);
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
        // GL JS's `pow(base, slope)` as `exp2` of a `log2`: `base` is
        // positive for intensities in 0..1, and `pow` pays for the bases
        // that aren't.
        let log_base = log2(base);
        scaled_slope = ((exp2(slope * log_base) - 1.0) / (exp2(max_value * log_base) - 1.0)) * max_value;
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
