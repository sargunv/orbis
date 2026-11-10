// The `builtins` module for GLSL ES 3.00 (WebGL 2), which has no
// `bitCount`: the built-ins written out.

fn count_one_bits(x: u32) -> u32 {
    var v = x - ((x >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    return (((v + (v >> 4u)) & 0x0f0f0f0fu) * 0x01010101u) >> 24u;
}
