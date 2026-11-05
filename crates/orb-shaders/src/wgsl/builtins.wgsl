// Builtins GLSL ES 3.00 lacks, called through these names so that
// `builtins_compat.wgsl` can stand in for them.

fn count_bits(x: u32) -> u32 { return countOneBits(x); }
