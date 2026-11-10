// The `builtins` module: the WGSL built-ins some targets lack, which
// pipelines call only through these, so a build for such a target swaps
// this module alone. GLSL ES 3.00 (WebGL 2) has no `bitCount`.

fn count_one_bits(x: u32) -> u32 { return countOneBits(x); }
