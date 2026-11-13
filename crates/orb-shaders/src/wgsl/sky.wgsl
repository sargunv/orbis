// The atmosphere's sky, drawn before the map layers: pinned GL JS's
// `sky.fragment.glsl` over a flat surface's horizon.
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
    let color = atmosphere_sky(atmosphere, atmosphere_horizon(atmosphere, px, ray));
    if color.a <= 0.0 && all(color.rgb <= vec3<f32>(0.0)) {
        discard;
    }
    return color;
}

// The globe's scattering over the map (GL JS `atmosphere.fragment.glsl`):
// GL JS's model and exposure, premultiplied, by the blend. What's beneath
// is dimmed by the transmittance and the scattered light added, as GL JS
// adds it: gamma-encoded, as if over black, with the transmittance's 1/2.2
// power (GL JS's gamma-corrected alpha), which lifts a bright map's sunlit
// face toward white.
//
// The light changes slowly with the view ray, so it is marched at the
// corners of a grid of SCATTER_CELL logical pixels and interpolated
// between them rather than at every pixel: its cost follows the logical
// viewport, not the device's pixels (14.7 M rays a frame over a 2560×1440
// window at a pixel ratio of 2). The limb is where it changes fast: a ray
// past it crosses the whole shell, one short of it stops at the ground.
// Pixels within SCATTER_LIMB cells of it march their own ray, so no cell
// they lie in mixes the two sides.

// Logical pixels a side of the scattering grid's cells.
const SCATTER_CELL: f32 = 8.0;
// Cells either side of the limb whose pixels march their own rays: more
// than a cell's diagonal, so that every triangle the rest interpolate lies
// on one side of it.
const SCATTER_LIMB: f32 = 2.0;

struct ScatterOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// The light along `ray` as the map composites it, by the blend.
fn scatter_color(ray: vec3<f32>) -> vec4<f32> {
    let scattered = atmosphere_scatter(atmosphere, ray);
    let light = 1.0 - exp(-scattered.rgb);
    let through = pow(scattered.a, 1.0 / ATMOSPHERE_GAMMA);
    let encoded = pow(light, vec3<f32>(1.0 / ATMOSPHERE_GAMMA));
    return vec4<f32>(encoded, 1.0 - through) * atmosphere.scatter_blend;
}

// Whether the view ray through logical pixel `px` reaches the atmosphere's
// shell, a hundredth of its height wider: those that miss it see none.
fn scatter_reaches(px: vec2<f32>) -> bool {
    let origin = atmosphere.globe_eye.xyz / atmosphere.globe_eye.w * 6371e3;
    let shell = atmosphere_sphere(origin, atmosphere_ray(atmosphere, px, frame.viewport), 6472e3);
    return shell.x <= shell.y && shell.y > 0.0;
}

// Six vertices a cell, its two triangles, the cells row by row over the
// logical viewport (`Atmosphere::commands_after` draws as many). A cell
// whose corners all miss the shell draws nothing: the light there is none
// and the map shows through whole, so a globe small in the view costs the
// pixels around it rather than the view's.
@vertex
fn vs_scatter(@builtin(vertex_index) vertex: u32) -> ScatterOut {
    let columns = u32(ceil(frame.viewport.x / SCATTER_CELL));
    let cell = vertex / 6u;
    let corner = vertex % 6u;
    let step = vec2<u32>(
        u32(corner == 1u || corner == 4u || corner == 5u),
        u32(corner == 2u || corner == 3u || corner == 5u),
    );
    let origin = vec2<u32>(cell % columns, cell / columns);
    let low = min(vec2<f32>(origin) * SCATTER_CELL, frame.viewport);
    let high = min(low + SCATTER_CELL, frame.viewport);
    var out: ScatterOut;
    if !(scatter_reaches(low) || scatter_reaches(high) || scatter_reaches(vec2<f32>(low.x, high.y))
        || scatter_reaches(vec2<f32>(high.x, low.y))) {
        // Every vertex of the cell at one point.
        out.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return out;
    }
    let px = min(vec2<f32>(origin + step) * SCATTER_CELL, frame.viewport);
    out.position = vec4<f32>(px.x / frame.viewport.x * 2.0 - 1.0, 1.0 - px.y / frame.viewport.y * 2.0, 0.0, 1.0);
    out.color = scatter_color(atmosphere_ray(atmosphere, px, frame.viewport));
    return out;
}

@fragment
fn fs_scatter(in: ScatterOut) -> @location(0) vec4<f32> {
    let ray = atmosphere_ray(atmosphere, in.position.xy / frame.pixel_ratio, frame.viewport);
    // How far toward the globe's centre the ray points past the limb, as
    // cosines: positive where it meets the ground, and near the limb a
    // pixel's change of it the same as its neighbours'.
    let distance = length(atmosphere.globe_eye.xyz) / atmosphere.globe_eye.w;
    let limb = sqrt(max(1.0 - 1.0 / (distance * distance), 0.0));
    let past = dot(ray, -normalize(atmosphere.globe_eye.xyz)) - limb;
    let cell = fwidth(past) * SCATTER_CELL * frame.pixel_ratio;
    if abs(past) < SCATTER_LIMB * cell {
        return scatter_color(ray);
    }
    return in.color;
}
