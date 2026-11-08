// v8's `light` on extrusions, GL JS's flat shading, which v8 defines: a
// face away from the light gets 1 − i of the light's colour, a face toward
// it `max(1 + i − luminance, 1)` (lighter surfaces lit less, so lit faces
// don't wash out), in between by n·l, for intensity i.

// The light falling on a surface facing `normal`, as a multiplier, with a
// wall's vertical gradient (1 for none). Normals and the light's position
// are in GL JS's extrusion axes: tile axes (y south) with their horizontal
// parts negated, as its wall normals point and its `sphericalToCartesian`
// places v8's light.
fn light_factor(normal: vec3<f32>, luminance: f32, gradient: f32) -> vec3<f32> {
    let ndotl = clamp(dot(normal, frame.light_position), 0.0, 1.0);
    let i = frame.light_color.w;
    return mix(1.0 - i, max(1.0 + i - luminance, 1.0), ndotl) * frame.light_color.rgb * gradient;
}

// A wall's vertical gradient: darker toward its foot, over its first
// 150 m, never below what the intensity allows.
fn light_wall_gradient(above_ground: f32, height: f32) -> f32 {
    return clamp(above_ground * sqrt(height / 150.0), mix(0.7, 0.98, 1.0 - frame.light_color.w), 1.0);
}

// The least a face is lit: GL JS's floor for tinted lights.
fn light_floor() -> vec3<f32> {
    return (vec3<f32>(1.0) - frame.light_color.rgb) * 0.3;
}

// A lit colour, with GL JS's lift of 0.03.
fn light_surface(color: vec3<f32>, normal: vec3<f32>, gradient: f32) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return clamp((color + vec3<f32>(0.03)) * light_factor(normal, luminance, gradient), light_floor(), vec3<f32>(1.0));
}
