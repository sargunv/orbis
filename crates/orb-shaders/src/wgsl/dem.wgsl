// GL JS's `getElevation`: metres from a DEM texel's 0–255 channels and the
// source's unpack vector.
fn dem_elevation(texel: vec4<f32>, unpack: vec4<f32>) -> f32 {
    return dot(vec4<f32>(texel.rgb * 255.0, -1.0), unpack);
}
