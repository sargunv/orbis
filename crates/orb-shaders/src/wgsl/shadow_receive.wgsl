// The scene's shadows: the cascades the shadow passes drew
// this frame (none: count 0) and their depth maps, in the scene group.
struct Shadow {
    cascades: ShadowCascades,
    // Camera clip → camera-relative world, for the ground pass.
    inverse: mat4x4<f32>,
};
@group(0) @binding(16) var<uniform> shadow: Shadow;
@group(0) @binding(17) var shadow_map: texture_depth_2d_array;
@group(0) @binding(18) var shadow_compare: sampler_comparison;

// How much of the directional light reaches a surface at camera-relative
// `world` facing `normal`, at view depth `depth`: 1 lit.
fn shadow_at(world: vec3<f32>, normal: vec3<f32>, depth: f32) -> f32 {
    if shadow.cascades.count == 0u {
        return 1.0;
    }
    return shadow_visibility(shadow_map, shadow_compare, shadow.cascades, world, normal, depth, 4u);
}

// `shadow_at` for the ground under the view, with one bilinear comparison:
// the full-screen ground pass costs more for each tap than the walls do,
// and the ground, seen at a slant, needs the 3×3 tent less.
fn shadow_at_ground(world: vec3<f32>, depth: f32) -> f32 {
    if shadow.cascades.count == 0u {
        return 1.0;
    }
    return shadow_visibility(shadow_map, shadow_compare, shadow.cascades, world, vec3<f32>(0.0, 0.0, 1.0), depth, 1u);
}
