// The style's lights as `pbr.wgsl` takes them: linear colours times their
// levels, east, north and up. lighting.wgsl keeps GL JS's extrusion axes:
// x and y negated, y south. The eye looks along the view's axis, from the
// bearing and pitch.
fn scene_lights() -> PbrLights {
    var lights: PbrLights;
    lights.ambient = srgb_to_linear(frame.light_ambient.rgb) * frame.light_ambient.w;
    lights.directional = srgb_to_linear(frame.light_color.rgb) * frame.light_color.w;
    lights.toward_light = normalize(vec3<f32>(-frame.light_position.x, frame.light_position.y, frame.light_position.z));
    lights.toward_eye = normalize(vec3<f32>(-sin(frame.bearing) * sin(frame.pitch), -cos(frame.bearing) * sin(frame.pitch), cos(frame.pitch)));
    lights.shadow = 1.0;
    return lights;
}
