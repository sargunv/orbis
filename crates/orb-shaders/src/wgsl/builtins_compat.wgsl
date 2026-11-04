// `builtins.wgsl` for Compat, without the builtins GLSL ES 3.00 lacks.

fn count_bits(x: u32) -> u32 {
    var v = x - ((x >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0f0f0f0fu) * 0x01010101u) >> 24u;
}

fn lowest_bit(x: u32) -> u32 {
    if x == 0u {
        return 0xffffffffu;
    }
    return count_bits((x & (0u - x)) - 1u);
}

// `m · 2^e` for e in [-126, 127]: 2^e is exactly the normal float with
// biased exponent e + 127.
fn scale_exp2(m: f32, e: i32) -> f32 {
    return m * bitcast<f32>(u32(e + 127) << 23u);
}
