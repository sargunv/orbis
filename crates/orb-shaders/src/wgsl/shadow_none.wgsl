// Compat draws no shadows (plan §8.7): everything is lit.
fn shadow_at(world: vec3<f32>, normal: vec3<f32>, depth: f32) -> f32 {
    return 1.0;
}
