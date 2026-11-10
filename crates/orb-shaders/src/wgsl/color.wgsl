// The sRGB transfer function, for the style's linear-light colour space
// (`dev.sargunv.orbis:color-space`): style colours and images are
// sRGB-encoded; in linear light they are decoded before they mix, and the
// target's sRGB format encodes what blending made.

// Encoded components to linear light.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let c0 = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(pow((c0 + 0.055) / 1.055, vec3<f32>(2.4)), c0 / 12.92, c0 <= vec3<f32>(0.04045));
}

// Linear light to encoded components.
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let c0 = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(c0, vec3<f32>(1.0 / 2.4)) - 0.055, c0 * 12.92, c0 <= vec3<f32>(0.0031308));
}

// A premultiplied encoded colour in linear light, premultiplied: its
// unpremultiplied colour decoded, so translucent colours keep their hue.
fn premultiplied_to_linear(c: vec4<f32>) -> vec4<f32> {
    if c.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(srgb_to_linear(c.rgb / c.a) * c.a, c.a);
}

// A premultiplied linear-light colour encoded, premultiplied.
fn premultiplied_to_srgb(c: vec4<f32>) -> vec4<f32> {
    if c.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(linear_to_srgb(c.rgb / c.a) * c.a, c.a);
}
