// Cascaded shadows from the directional light, as functions of the
// cascades and their depth maps: no
// bindings, so casters and receivers compose it.
//
// Each cascade is a depth layer the casters drew from the light: camera-
// relative world → the light's clip space, x and y in [−1, 1], depth in
// [0, 1] growing away from the light. A receiver compares its own depth
// there, after moving along its normal and toward the light by about a
// texel, which keeps surfaces from shadowing themselves.

struct ShadowCascades {
    view_proj: array<mat4x4<f32>, 2>,
    // Each cascade's far end in view depth (clip w); a receiver uses the
    // first that reaches it.
    splits: vec4<f32>,
    // World units per texel of each cascade, for the receivers' offset.
    texel: vec4<f32>,
    // Unit vector toward the light, camera-relative world axes (x east,
    // y north, z up).
    light: vec3<f32>,
    // Cascades drawn this frame: 0 without shadows.
    count: u32,
    // How dark a shadow is: 1 removes all the directional light.
    strength: f32,
    // The directional light's share of level ground's lighting.
    ground: f32,
    _pad: vec2<f32>,
};

// The cascade a receiver at view depth `depth` reads, or `count` past the
// last.
fn shadow_cascade(c: ShadowCascades, depth: f32) -> u32 {
    for (var i = 0u; i < c.count; i++) {
        if depth <= c.splits[i] {
            return i;
        }
    }
    return c.count;
}

// How lit a receiver is in cascade `i`, 0 to 1, outside the map lit: with
// `taps` 4, four bilinear comparisons half a texel either way of its place
// in the map, a tent over the 3×3 texels around it; with 1, one bilinear
// comparison there, over 2×2.
fn shadow_lit(map: texture_depth_2d_array, cmp: sampler_comparison, c: ShadowCascades, i: u32, world: vec3<f32>, normal: vec3<f32>, taps: u32) -> f32 {
    let p = c.view_proj[i] * vec4<f32>(world + (normal * 1.5 + c.light) * c.texel[i], 1.0);
    let ndc = p.xyz / p.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0 {
        return 1.0;
    }
    if taps == 1u {
        return textureSampleCompareLevel(map, cmp, uv, i32(i), ndc.z);
    }
    let half = 0.5 / vec2<f32>(textureDimensions(map));
    var lit = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let at = uv + vec2<f32>(f32(k & 1u) * 2.0 - 1.0, f32(k >> 1u) * 2.0 - 1.0) * half;
        lit += textureSampleCompareLevel(map, cmp, at, i32(i), ndc.z);
    }
    return lit / 4.0;
}

// How much of the directional light reaches a receiver at camera-relative
// `world`, with unit `normal` and view depth `depth`: 1 lit, `1 − strength`
// in full shadow. Near a cascade's far end it blends into the next, and
// past the last it fades out, so no edge shows as the camera moves.
fn shadow_visibility(map: texture_depth_2d_array, cmp: sampler_comparison, c: ShadowCascades, world: vec3<f32>, normal: vec3<f32>, depth: f32, taps: u32) -> f32 {
    let i = shadow_cascade(c, depth);
    if i >= c.count {
        return 1.0;
    }
    var lit = shadow_lit(map, cmp, c, i, world, normal, taps);
    var start = 0.0;
    if i > 0u {
        start = c.splits[i - 1u];
    }
    let edge = clamp((depth - mix(start, c.splits[i], 0.9)) / (0.1 * (c.splits[i] - start)), 0.0, 1.0);
    if edge > 0.0 {
        let next = select(1.0, shadow_lit(map, cmp, c, i + 1u, world, normal, taps), i + 1u < c.count);
        lit = mix(lit, next, edge);
    }
    return 1.0 - c.strength * (1.0 - lit);
}
