// v8's `light` on extrusions, GL JS's flat shading, which v8 defines: a
// face away from the light gets 1 − i of the light's colour, a face toward
// it `max(1 + i − luminance, 1)` (lighter surfaces lit less, so lit faces
// don't wash out), in between by n·l, for intensity i. That is an ambient
// light of 1 − i of the colour and a sun's adding the rest by n·l.

// The light falling on a surface facing `normal`, as a multiplier, with a
// wall's vertical gradient (1 for none). Normals and the light's position
// are in GL JS's extrusion axes: tile axes (y south) with their horizontal
// parts negated, as its wall normals point and its `sphericalToCartesian`
// places v8's light.
fn light_factor(normal: vec3<f32>, luminance: f32, gradient: f32) -> vec3<f32> {
    return light_factor_facing(light_facing(normal), luminance, gradient);
}

// On a globe, `normal` at ground point `g` turned onto the axes at the
// camera target, where the light's position (`frame.light_position`)
// stands: one light over the whole sphere, as a viewport's light lights
// a 3D scene, so a wall facing it at the limb is lit as one facing it at
// the target, and a roof far from the target turns with the ground. On
// the plane, and at the target, the normal is as it was. (GL JS turns
// both onto the sphere, but by axes whose horizontal parts the extrusion
// axes negate, so away from the target it lights the faces turned from
// the light.)
fn light_normal(normal: vec3<f32>, g: Ground) -> vec3<f32> {
    if !on_globe() || frame.globe_plane != 0u {
        return normal;
    }
    let at = globe_axes(g.globe + frame.globe_target);
    // The extrusion axes' x points west and y north.
    let n = at[0] * -normal.x + at[1] * normal.y + at[2] * normal.z;
    let light_at = globe_axes(frame.globe_target);
    let turned = vec3<f32>(-dot(n, light_at[0]), dot(n, light_at[1]), dot(n, light_at[2]));
    return normalize(mix(normal, turned, frame.globe_amount));
}

// East, north and up at an ECEF position, unit; east along y at a pole.
fn globe_axes(p: vec3<f32>) -> mat3x3<f32> {
    let up = normalize(p);
    let level = vec3<f32>(-up.y, up.x, 0.0);
    let east = select(vec3<f32>(0.0, 1.0, 0.0), normalize(level), length(level) > 1e-6);
    return mat3x3<f32>(east, cross(up, east), up);
}

// How far a surface facing `normal` turns to the light, 0 to 1.
fn light_facing(normal: vec3<f32>) -> f32 {
    return clamp(dot(normal, frame.light_position), 0.0, 1.0);
}

fn light_factor_facing(facing: f32, luminance: f32, gradient: f32) -> vec3<f32> {
    return (light_ambient() + facing * light_sun(luminance)) * gradient;
}

// The ambient light on every face.
fn light_ambient() -> vec3<f32> {
    return (1.0 - frame.light_color.w) * frame.light_color.rgb;
}

// The sun's light on a face turned to it, beyond the ambient.
fn light_sun(luminance: f32) -> vec3<f32> {
    let i = frame.light_color.w;
    return (max(1.0 + i - luminance, 1.0) - (1.0 - i)) * frame.light_color.rgb;
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
    return light_lit(color, light_facing(normal), gradient);
}

// `light_surface` for a surface turned `facing` to the light.
fn light_lit(color: vec3<f32>, facing: f32, gradient: f32) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return clamp((color + vec3<f32>(0.03)) * light_factor_facing(facing, luminance, gradient), light_floor(), vec3<f32>(1.0));
}
