// Composites: what drew apart from the map, into a target
// of its own, drawn into the map by one full-screen triangle. Texels are
// premultiplied.
//
// - over: the source at an opacity, by the pipeline's premultiplied-alpha
//   blend (a 3D layer, a terrain stack, a layer composited once);
// - ramp: the source's red channel looked up in a colour ramp, sampled
//   linearly (a heatmap's density, at a quarter of the resolution);
// - replace: W3C Compositing 1's blend modes against the map beneath,
//   read from the canvas it was drawn into, written over it in place
//   (`*-layer-blend`, `blend.wgsl`).

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
fn fs_over(in: CompositeOut) -> @location(0) vec4<f32> {
    if u32(composite.y) == 7u {
        let t = textureSample(source, linear, in.uv).r;
        return textureSample(aux, linear, vec2<f32>(t, 0.5)) * composite.x;
    }
    return textureLoad(source, vec2<i32>(in.position.xy), 0) * composite.x;
}

@fragment
fn fs_replace(in: CompositeOut) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(in.position.xy);
    let s = textureLoad(source, texel, 0) * composite.x;
    return blend_over(u32(composite.y), s, textureLoad(aux, texel, 0));
}

// The linear-light map, drawn into an sRGB target of the renderer's own,
// into the host's target: encoded, premultiplied.
@fragment
fn fs_encode(in: CompositeOut) -> @location(0) vec4<f32> {
    return premultiplied_to_srgb(textureLoad(source, vec2<i32>(in.position.xy), 0));
}
