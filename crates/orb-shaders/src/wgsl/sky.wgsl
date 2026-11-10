// The atmosphere's sky, drawn before the map layers: pinned GL JS's
// `sky.fragment.glsl` over a flat surface's horizon, over style-spec#1889's
// backdrop and the stars on it, which show above the horizon.
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> atmosphere: Atmosphere;

@vertex
fn vs_sky(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_sky(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(atmosphere, px, frame.viewport);
    // Radians to a logical pixel, from the ray's change across the quad.
    let pixel = 0.5 * (length(dpdx(ray)) + length(dpdy(ray))) * frame.pixel_ratio;
    let horizon = atmosphere_horizon(atmosphere, px, ray);
    let sky = atmosphere_sky(atmosphere, horizon);
    var under = atmosphere.backdrop_color;
    if atmosphere.star_opacity > 0.0 && horizon.x > 0.0 {
        var stars = atmosphere_stars(ray, max(pixel, 1e-7), frame.pixel_ratio);
        if atmosphere.linear != 0.0 {
            // Light, as over black: each premultiplied component decoded,
            // so a star looks as it does in sRGB.
            stars = vec4<f32>(srgb_to_linear(stars.rgb), stars.a);
        }
        stars *= atmosphere.star_opacity;
        under = stars + under * (1.0 - stars.a);
    }
    let color = sky + under * (1.0 - sky.a);
    if color.a <= 0.0 && all(color.rgb <= vec3<f32>(0.0)) {
        discard;
    }
    return color;
}

// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`):
// GL JS's model and exposure, premultiplied, by the blend. What's beneath
// is dimmed by the transmittance and the scattered light added. In sRGB,
// as GL JS adds it: gamma-encoded, as if over black, with the
// transmittance's 1/2.2 power (GL JS's gamma-corrected alpha), which lifts
// a bright map's sunlit face toward white. In linear light, blending adds
// it to what's beneath exactly: a surface keeps its colours, tinted, and
// the long paths toward the limb glow.
@fragment
fn fs_scatter(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(atmosphere, px, frame.viewport);
    let scattered = atmosphere_scatter(atmosphere, ray);
    let light = 1.0 - exp(-scattered.rgb);
    if atmosphere.linear != 0.0 {
        return vec4<f32>(light, 1.0 - scattered.a) * atmosphere.scatter_blend;
    }
    let through = pow(scattered.a, 1.0 / ATMOSPHERE_GAMMA);
    let encoded = pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA));
    return vec4<f32>(encoded, 1.0 - through) * atmosphere.scatter_blend;
}

// Ground lighting over the map beneath its labels, multiplying what's
// drawn (`dev.sargunv.orbis:ground-lighting`): the sun's light where it is
// up, the ambient light where it is down, smoothly over the twilight, by
// the sun's elevation over the ground's normal: the globe's where the ray
// meets it, mixed toward the target's up as the globe hands over to a
// flat surface, which takes the target's. Above the horizon the sky is
// lit as the target's ground is, so a night view's sky is dark too, while
// the camera is inside the atmosphere: seen from orbit it is space,
// unlit. The light is in linear light; in sRGB it multiplies encoded
// colours by its encoding, which is the same under GL JS's 2.2 gamma.
@fragment
fn fs_ground(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let a = atmosphere;
    let px = position.xy / frame.pixel_ratio;
    let ray = atmosphere_ray(a, px, frame.viewport);
    var amount = a.ground_day.w;
    var normal = a.ground_up.xyz;
    // The sky starts at the globe's limb, not at GL JS's sky's, a pixel or
    // two inside it (`atmosphere_horizon`'s offset, and its pixels measured
    // as at the view's centre): on a whole globe, half a pixel past where
    // the ray clears it and any terrain that could stand on it. This draws
    // per pixel and the ground per sample, so a pixel the limb crosses is
    // lit if any of its samples could be ground. So the globe's outermost
    // pixels, and mountains on its limb, are lit as the rest of the ground
    // (where the space beside them darkens with them, it was dark already).
    var sky = atmosphere_horizon(a, px, ray).x > a.globe_amount;
    if a.globe_amount > 0.0 {
        let origin = a.globe_eye.xyz / a.globe_eye.w;
        // The height the ray passes closest to the globe's centre at, if
        // ahead, in its radii.
        let toward = -dot(origin, ray);
        let clear = sqrt(max(dot(origin, origin) - toward * toward, 0.0)) - 1.0;
        let pixel = fwidth(clear);
        if a.globe_amount >= 1.0 {
            sky = toward <= 0.0 || clear > a.ground_up.w + 0.5 * pixel;
        }
        if !sky {
            // Where the ray meets the globe, or, over its limb, passes
            // closest to it.
            let hit = atmosphere_sphere(origin, ray, 1.0);
            let along = select(max(-dot(origin, ray), 0.0), max(hit.x, 0.0), hit.x <= hit.y);
            let sphere = normalize(origin + ray * along);
            normal = normalize(mix(normal, sphere, a.globe_amount));
        }
    }
    if sky {
        // As the sky itself fades out with altitude (`atmosphere_horizon`).
        if a.globe_amount > 0.0 {
            let altitude = length(a.globe_eye.xyz / a.globe_eye.w) - 1.0;
            let inside = 1.0 - smoothstep(ATMOSPHERE_HEIGHT, 10.0 * ATMOSPHERE_HEIGHT, altitude);
            amount *= mix(1.0, inside, a.globe_amount);
        }
        if amount <= 0.0 {
            discard;
        }
    }
    // The sine of the sun's elevation, against that of half the twilight.
    let elevation = dot(normal, a.ground_sun.xyz);
    let half = a.ground_night.w;
    var day = select(0.0, 1.0, elevation >= 0.0);
    if half > 0.0 {
        day = smoothstep(-half, half, elevation);
    }
    let light = mix(vec3<f32>(1.0), mix(a.ground_night.rgb, a.ground_day.rgb, day), amount);
    if a.linear != 0.0 {
        return vec4<f32>(light, 1.0);
    }
    return vec4<f32>(pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA)), 1.0);
}
