// The style's lights on a surface (plan §9.4, decision 0026): one model
// for v8's `light` and the scene document's lights, which the frame block
// carries. Its response is GL JS's flat extrusion shading, which v8 defines:
// a face away from the light gets the ambient level of the ambient colour,
// a face toward it `max(a + d − luminance, 1)` of the directional colour
// (lighter surfaces lit less, so lit faces don't wash out), in between by
// n·l. v8's intensity i lowers to a = 1 − i, d = 2i.

// The light falling on a surface facing `normal`, as a multiplier, with a
// wall's vertical gradient (1 for none). Normals and the light's position
// are in GL JS's extrusion axes: tile axes (y south) with their horizontal
// parts negated, as its wall normals point and its `sphericalToCartesian`
// places v8's light.
fn light_factor(normal: vec3<f32>, luminance: f32, gradient: f32) -> vec3<f32> {
    let ndotl = clamp(dot(normal, frame.light_position.xyz), 0.0, 1.0);
    let a = frame.light_ambient.w;
    let lit = max(a + frame.light_color.w - luminance, 1.0);
    return mix(a * frame.light_ambient.rgb, lit * frame.light_color.rgb, ndotl) * gradient;
}

// A wall's vertical gradient: darker toward its foot, over its first
// 150 m, never below what the ambient level allows.
fn light_wall_gradient(above_ground: f32, height: f32) -> f32 {
    return clamp(above_ground * sqrt(height / 150.0), mix(0.7, 0.98, frame.light_ambient.w), 1.0);
}

// A lit colour: GL JS's lift of 0.03, its floor for tinted lights, and an
// emissive strength that keeps the surface's own colour.
fn light_surface(color: vec3<f32>, normal: vec3<f32>, gradient: f32, emissive: f32) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let lit = clamp((color + vec3<f32>(0.03)) * light_factor(normal, luminance, gradient),
        (vec3<f32>(1.0) - frame.light_color.rgb) * 0.3, vec3<f32>(1.0));
    return mix(lit, color, emissive);
}
