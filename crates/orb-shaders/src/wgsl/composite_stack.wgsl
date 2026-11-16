// Stacked composites: up to eight images drawn apart, each a render
// target's texel a logical pixel sampled linearly as `composite.wgsl`'s
// sampled and solid modes sample them, put over one another here and drawn
// into the map by one full-screen triangle with premultiplied blending.
// Image-process layers and fills drawn through targets in a row, which
// would each cost the map's pass a full-screen draw, cost it one.

// A layer: its texel times `scale.x`, plus `tint` times the texel's
// coverage: a process's output at an opacity (no tint), or a fill's colour
// through a target (scale 0). Premultiplied.
struct Layer {
    scale: vec4<f32>,
    tint: vec4<f32>,
};
// count.x: how many layers, 1–8, bottom first.
struct Stack {
    count: vec4<f32>,
    layers: array<Layer, 8>,
};
@group(0) @binding(0) var<uniform> stack: Stack;
// How many layers the pipeline samples: a stack of fewer leaves the rest
// blank. Each sample costs every pixel of the map, so smaller stacks draw
// with pipelines that sample fewer.
override stack_layers: u32 = 8u;
@group(0) @binding(1) var linear: sampler;
@group(0) @binding(2) var layer0: texture_2d<f32>;
@group(0) @binding(3) var layer1: texture_2d<f32>;
@group(0) @binding(4) var layer2: texture_2d<f32>;
@group(0) @binding(5) var layer3: texture_2d<f32>;
@group(0) @binding(6) var layer4: texture_2d<f32>;
@group(0) @binding(7) var layer5: texture_2d<f32>;
@group(0) @binding(8) var layer6: texture_2d<f32>;
@group(0) @binding(9) var layer7: texture_2d<f32>;

struct StackOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_stack(@builtin(vertex_index) index: u32) -> StackOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    var out: StackOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return out;
}

// Layer `i` of texel `t`, over `beneath` as premultiplied blending puts it,
// where the stack has it.
fn stack_over(i: u32, t: vec4<f32>, beneath: vec4<f32>) -> vec4<f32> {
    let layer = stack.layers[i];
    let s = t * layer.scale.x + layer.tint * t.a;
    return select(beneath, s + beneath * (1.0 - s.a), i < u32(stack.count.x));
}

// The pipeline's layers' textures are sampled, past the stack's a blank
// texel, ahead of any arithmetic: in branches on the stack's count, each
// sample's latency would add up.
@fragment
fn fs_stack(in: StackOut) -> @location(0) vec4<f32> {
    let t0 = textureSample(layer0, linear, in.uv);
    let t1 = textureSample(layer1, linear, in.uv);
    var c = stack_over(0u, t0, vec4<f32>(0.0));
    c = stack_over(1u, t1, c);
    if stack_layers > 2u {
        let t2 = textureSample(layer2, linear, in.uv);
        let t3 = textureSample(layer3, linear, in.uv);
        c = stack_over(2u, t2, c);
        c = stack_over(3u, t3, c);
    }
    if stack_layers > 4u {
        let t4 = textureSample(layer4, linear, in.uv);
        let t5 = textureSample(layer5, linear, in.uv);
        let t6 = textureSample(layer6, linear, in.uv);
        let t7 = textureSample(layer7, linear, in.uv);
        c = stack_over(4u, t4, c);
        c = stack_over(5u, t5, c);
        c = stack_over(6u, t6, c);
        c = stack_over(7u, t7, c);
    }
    return c;
}
