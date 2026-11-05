// Composites (decision 0043): what drew apart from the map, into a target
// of its own, drawn into the map by one full-screen triangle. Texels are
// premultiplied.
//
// - over: the source at an opacity, by the pipeline's premultiplied-alpha
//   blend (a 3D layer, a terrain stack, a layer composited once);
// - ramp: the source's red channel looked up in a colour ramp, sampled
//   linearly (a heatmap's density, at a quarter of the resolution);
// - replace: W3C Compositing 1's blend modes against the map beneath,
//   read from the canvas it was drawn into, written over it in place
//   (`*-layer-blend`).

// x: the opacity; y: the mode (`Mode` in orb-render: 0 over, 1–6 the blend
// modes, 7 ramp).
@group(0) @binding(0) var<uniform> composite: vec4<f32>;
@group(0) @binding(1) var source: texture_2d<f32>;
// The ramp, or the map beneath.
@group(0) @binding(2) var aux: texture_2d<f32>;
@group(0) @binding(3) var linear: sampler;

struct CompositeOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_composite(@builtin(vertex_index) index: u32) -> CompositeOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    var out: CompositeOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_over(in: CompositeOut) -> MapOut {
    if u32(composite.y) == 7u {
        let t = textureSample(source, linear, in.uv).r;
        return map_out_self_lit(textureSample(aux, linear, vec2<f32>(t, 0.5)) * composite.x);
    }
    return map_out_self_lit(textureLoad(source, vec2<i32>(in.position.xy), 0) * composite.x);
}

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

@fragment
fn fs_replace(in: CompositeOut) -> MapOut {
    let texel = vec2<i32>(in.position.xy);
    let s = textureLoad(source, texel, 0) * composite.x;
    let b = textureLoad(aux, texel, 0);
    let mode = u32(composite.y);
    if mode == 6u {
        // plus-lighter: the Porter-Duff plus operator, clamped.
        return map_out_self_lit(min(s + b, vec4<f32>(1.0)));
    }
    let cs = s.rgb / max(s.a, 1e-6);
    let cb = b.rgb / max(b.a, 1e-6);
    let mixed = blend_mix(mode, clamp(cb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(cs, vec3<f32>(0.0), vec3<f32>(1.0)));
    let rgb = s.rgb * (1.0 - b.a) + b.rgb * (1.0 - s.a) + s.a * b.a * mixed;
    return map_out_self_lit(vec4<f32>(rgb, s.a + b.a * (1.0 - s.a)));
}
