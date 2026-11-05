// Layer composites (`*-layer-opacity`, `*-layer-blend`): a layer drawn into
// a texture of its own composites into the map once, so its overlaps don't
// accumulate (GL JS `draw_layer_opacity.ts`), and in any blend mode other
// than normal against the map beneath it, read from the canvas the map so far
// was drawn into. Texels are premultiplied.

// x: the layer opacity; y: the blend mode (`Blend` in orb-render).
@group(0) @binding(0) var<uniform> layer_composite: vec4<f32>;
@group(0) @binding(1) var layer_color: texture_2d<f32>;
@group(0) @binding(2) var layer_backdrop: texture_2d<f32>;

@vertex
fn vs_layer_composite(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// Normal blending, by the pipeline's premultiplied-alpha blend.
@fragment
fn fs_layer_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(layer_color, vec2<i32>(position.xy), 0) * layer_composite.x;
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

// Any other blend mode, written over the canvas in place (no fixed blend).
@fragment
fn fs_layer_blend(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(position.xy);
    let s = textureLoad(layer_color, texel, 0) * layer_composite.x;
    let b = textureLoad(layer_backdrop, texel, 0);
    let mode = u32(layer_composite.y);
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
