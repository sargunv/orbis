// The `builtins` module for GLSL ES 3.00 (WebGL 2), which has no
// `bitCount`: the built-ins written out.

fn count_one_bits(x: u32) -> u32 {
    var v = x - ((x >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0f0f0f0fu) * 0x01010101u) >> 24u;
}

// GLSL ES 3.00 has no `packUnorm4x8` either: WGSL's `pack4x8unorm`.
fn pack_4x8_unorm(c: vec4<f32>) -> u32 {
    let b = vec4<u32>(round(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
    return b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
}
