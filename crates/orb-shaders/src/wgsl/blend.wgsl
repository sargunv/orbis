// W3C Compositing 1's blend modes (`*-layer-blend`), as `Blend` in
// orb-render numbers them: 0 normal, 1 multiply, 2 screen, 3 overlay,
// 4 color, 5 luminosity, 6 plus-lighter.

fn blend_lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.3, 0.59, 0.11));
}

fn blend_clip(c: vec3<f32>) -> vec3<f32> {
    let l = blend_lum(c);
    let n = min(min(c.r, c.g), c.b);
    let x = max(max(c.r, c.g), c.b);
    var out = c;
    if n < 0.0 {
        out = l + (out - l) * l / max(l - n, 1e-6);
    }
    if x > 1.0 {
        out = l + (out - l) * (1.0 - l) / max(x - l, 1e-6);
    }
    return out;
}

fn blend_set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return blend_clip(c + (l - blend_lum(c)));
}

// W3C Compositing 1's B(Cb, Cs) of the separable and non-separable modes,
// on unpremultiplied colours.
fn blend_mix(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1u: {
            return cb * cs;
        }
        case 2u: {
            return cb + cs - cb * cs;
        }
        case 3u: {
            // Overlay is hard light with the layers swapped.
            let screen = cs + (2.0 * cb - 1.0) - cs * (2.0 * cb - 1.0);
            return select(screen, cs * 2.0 * cb, cb <= vec3<f32>(0.5));
        }
        case 4u: {
            return blend_set_lum(cs, blend_lum(cb));
        }
        case 5u: {
            return blend_set_lum(cb, blend_lum(cs));
        }
        default: {
            return cs;
        }
    }
}

// Premultiplied `s` over the backdrop `b` in `mode`.
fn blend_over(mode: u32, s: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    if mode == 6u {
        // plus-lighter: the Porter-Duff plus operator, clamped.
        return min(s + b, vec4<f32>(1.0));
    }
    let cs = s.rgb / max(s.a, 1e-6);
    let cb = b.rgb / max(b.a, 1e-6);
    let mixed = blend_mix(mode, clamp(cb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(cs, vec3<f32>(0.0), vec3<f32>(1.0)));
    let rgb = s.rgb * (1.0 - b.a) + b.rgb * (1.0 - s.a) + s.a * b.a * mixed;
    return vec4<f32>(rgb, s.a + b.a * (1.0 - s.a));
}
