// The style's lights as `pbr.wgsl` takes them: linear colours times their
// levels, east, north and up. lighting.wgsl keeps GL JS's extrusion axes:
// x and y negated, y south. The eye looks along the view's axis, from the
// bearing and pitch.
fn scene_lights() -> PbrLights {
    var lights: PbrLights;
    lights.ambient = srgb_to_linear(frame.light_ambient.rgb) * frame.light_ambient.w;
    lights.directional = srgb_to_linear(frame.light_color.rgb) * frame.light_color.w;
    let toward = vec3<f32>(-frame.light_position.x, frame.light_position.y, frame.light_position.z);
    // Without a directional light, none: it's down, and its level zero.
    lights.toward_light = select(vec3<f32>(0.0, 0.0, -1.0), normalize(toward), length(toward) > 0.0);
    lights.toward_eye = normalize(vec3<f32>(-sin(frame.bearing) * sin(frame.pitch), -cos(frame.bearing) * sin(frame.pitch), cos(frame.pitch)));
    lights.shadow = 1.0;
    return lights;
}

// The lights where the Earth's up is `up` (`ground_up`): the directional
// light, given at the camera's target, fades out over its last two degrees
// above that point's own horizon and is gone below it, so a sun lights a
// globe's day side alone.
fn scene_lights_at(up: vec3<f32>) -> PbrLights {
    var lights = scene_lights();
    lights.directional *= light_above(dot(up, lights.toward_light));
    return lights;
}

// A light at sine of elevation `s` over a horizon: its last two degrees
// fade it out.
fn light_above(s: f32) -> f32 {
    return clamp(degrees(asin(clamp(s, -1.0, 1.0))) * 0.5, 0.0, 1.0);
}

// How dark it is where the Earth's up is `up`: 0 by day, 1 at night, as
// the directional light goes from 4° above that point's horizon to 4°
// below it, when lights come on (night lights, lit windows). Without a
// directional light it is night.
fn darkness(lights: PbrLights, up: vec3<f32>) -> f32 {
    return 1.0 - smoothstep(-0.07, 0.07, dot(up, lights.toward_light));
}
