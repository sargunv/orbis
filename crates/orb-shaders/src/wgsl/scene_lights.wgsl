// The style's lights as `pbr.wgsl` takes them, as the frame block carries
// them (orb-scene's `write_frame` works them out once a frame, so no
// fragment repeats it): linear colours times their levels, east, north and
// up at the target. The eye looks along the view's axis.
fn scene_lights() -> PbrLights {
    var lights: PbrLights;
    lights.ambient = frame.lit_ambient.rgb;
    lights.directional = frame.lit_directional.rgb;
    lights.toward_light = frame.lit_toward_light.xyz;
    lights.toward_eye = frame.lit_toward_eye.xyz;
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
// fade it out (sin 2°; the sine is the angle there).
fn light_above(s: f32) -> f32 {
    return clamp(s / 0.0349, 0.0, 1.0);
}

// How dark it is where the Earth's up is `up`: 0 by day, 1 at night, as
// the directional light goes from 4° above that point's horizon to 4°
// below it, when lights come on (night lights, lit windows). Without a
// directional light it is night.
fn darkness(lights: PbrLights, up: vec3<f32>) -> f32 {
    return 1.0 - smoothstep(-0.07, 0.07, dot(up, lights.toward_light));
}
