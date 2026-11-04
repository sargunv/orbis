// Builtins GLSL ES 3.00 lacks, called through these names so that
// `builtins_compat.wgsl` can stand in for them.

fn count_bits(x: u32) -> u32 { return countOneBits(x); }
fn lowest_bit(x: u32) -> u32 { return firstTrailingBit(x); }
// `m · 2^e` for e in [-126, 127].
fn scale_exp2(m: f32, e: i32) -> f32 { return ldexp(m, e); }
