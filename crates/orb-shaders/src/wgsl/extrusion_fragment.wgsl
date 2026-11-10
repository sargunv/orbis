// Extrusions' fragment stages, apart from `extrusion.wgsl` for the
// vertex-only clip-plane shaders, which leave them out.
@fragment
fn fs_extrusion(in: ExtrusionOut) -> @location(0) vec4<f32> {
    return extrusion_shade(in);
}
@fragment
fn fs_extrusion_cut(in: ExtrusionOut) -> @location(0) vec4<f32> {
    if !extrusion_inside(tile_at(in.tile), in.tile_pos) { discard; }
    return extrusion_shade(in);
}

// The sphere's depth, drawn into the depth the 3D passes share before the
// first of them on a whole globe (orb-render's `Extrusions::prepare`), as
// GL JS fills its depth with the globe's tiles: what stands behind the
// sphere hides per fragment (`CLIP_STANDING`), while a wall rising past
// the limb from a foot behind it shows above it. It lies a little over a
// thousandth of its distance deeper than the sea-level sphere, past the
// steps that order buildings' depths (`extrusion_depth_scale`, at most
// 0.05%), so a wall's foot on the ground stays in front of it: about a
// pixel at the limb.
struct SphereDepthOut {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sphere_depth(@builtin(vertex_index) index: u32) -> SphereDepthOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    var out: SphereDepthOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

struct SphereDepth {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

const SPHERE_DEPTH_SINK: f32 = 0.9990234375;

@fragment
fn fs_sphere_depth(in: SphereDepthOut) -> SphereDepth {
    var out: SphereDepth;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0;
    let m = frame.globe_view_proj;
    let r0 = vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
    let r1 = vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
    let r3 = vec4<f32>(m[0].w, m[1].w, m[2].w, m[3].w);
    // The pixel's ray is where the planes of clip x = ndc.x w and
    // y = ndc.y w meet, from the eye on both, away from it as w grows.
    var d = normalize(cross((r0 - in.ndc.x * r3).xyz, (r1 - in.ndc.y * r3).xyz));
    d = select(d, -d, dot(r3.xyz, d) < 0.0);
    let from_center = frame.globe_eye + frame.globe_target;
    let b = dot(from_center, d);
    let reach = length(from_center);
    let c = (reach - SPHERE_RADIUS) * (reach + SPHERE_RADIUS);
    let disc = b * b - c;
    if disc <= 0.0 || b >= 0.0 {
        return out;
    }
    // The nearer root, as c over the farther's sum: no cancellation.
    let t = c / (-b + sqrt(disc));
    let clip = m * vec4<f32>(frame.globe_eye + d * t, 1.0);
    out.depth = clamp((clip.w - clip.z) / clip.w * SPHERE_DEPTH_SINK, 0.0, 1.0);
    return out;
}
