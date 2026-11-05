// Shadows on the ground (decision 0046): one full-screen draw in the main
// pass at the first 3D layer's place, multiplying what lies below by how
// much of the directional light each pixel's ground point keeps. The point
// is the terrain's, from its depth, else the view ray's on the ground
// plane.
struct GroundOut { @builtin(position) position: vec4<f32>, };
@vertex
fn vs_shadow_ground(@builtin(vertex_index) index: u32) -> GroundOut {
    let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: GroundOut;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    return out;
}
@fragment
fn fs_shadow_ground(in: GroundOut) -> @location(0) vec4<f32> {
    let ndc = in.position.xy / (frame.viewport * frame.pixel_ratio) * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    var world: vec3<f32>;
    if frame.terrain_exaggeration > 0.0 {
        let depth = textureLoad(terrain_depth, vec2<i32>(in.position.xy), 0).r;
        if depth <= 0.0 {
            return vec4<f32>(1.0);
        }
        let p = shadow.inverse * vec4<f32>(ndc, 1.0 - depth, 1.0);
        world = p.xyz / p.w;
    } else {
        let near = shadow.inverse * vec4<f32>(ndc, 0.0, 1.0);
        let far = shadow.inverse * vec4<f32>(ndc, 1.0, 1.0);
        let a = near.xyz / near.w;
        let b = far.xyz / far.w;
        if b.z >= a.z || a.z < 0.0 {
            return vec4<f32>(1.0);
        }
        world = mix(a, b, a.z / (a.z - b.z));
    }
    let depth = (frame.view_proj * vec4<f32>(world, 1.0)).w;
    let lit = shadow_at_ground(world, depth);
    return vec4<f32>(vec3<f32>(1.0 - shadow.cascades.ground * (1.0 - lit)), 1.0);
}
