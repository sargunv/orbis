// The `builtins` module: the WGSL built-ins some targets lack, which
// pipelines call only through these, so a build for such a target swaps
// this module alone. GLSL ES 3.00 (WebGL 2) has no `bitCount`
// or `packUnorm4x8`.

fn count_one_bits(x: u32) -> u32 { return countOneBits(x); }
fn pack_4x8_unorm(c: vec4<f32>) -> u32 { return pack4x8unorm(c); }
