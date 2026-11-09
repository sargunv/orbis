// GL JS's `getElevation`: metres from a DEM texel's 0–255 channels and the
// source's unpack vector, except that terrarium's no-data texel, black
// (−32768 m), reads as sea level, as the CPU's `DemGrid` reads it.
fn dem_elevation(texel: vec4<f32>, unpack: vec4<f32>) -> f32 {
    let void = unpack.w == 32768.0 && all(texel.rgb == vec3<f32>(0.0));
    return select(dot(vec4<f32>(texel.rgb * 255.0, -1.0), unpack), 0.0, void);
}
