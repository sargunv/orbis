// glTF models at point features (plan §9.4): each instance is a point of a
// resident tile, as circles pull theirs (point.wgsl); the draw's vertices
// are its model's, in metres east, north and up of the point, scaled,
// turned and moved by the layer's paint, and shaded with glTF's
// metallic-roughness model under the style's lights (pbr.wgsl, decision
// 0026). Model words are `orb_3d::gpu`'s.

struct ModelNode {
    // Premultiplied, in the style's (sRGB) space.
    color: vec4<f32>,
    color_mix: f32,
    emissive: f32,
    index: u32,
    table_mask: u32,
    scale: vec3<f32>,
    // Applied to the layer as a whole, by the 3D composite.
    opacity: f32,
    rotation: vec3<f32>,
    _pad0: f32,
    translation: vec3<f32>,
    _pad1: f32,
    // Paint slots: colour, colour mix, emissive strength, then scale,
    // rotation and translation by component.
    curves: array<Curve, 12>,
};
@group(1) @binding(0) var<uniform> node: ModelNode;

// One model drawn for one node: where its vertices and materials start in
// `model_words`.
struct ModelDraw {
    vertex_base: u32,
    material_base: u32,
    _pad: vec2<u32>,
};
@group(2) @binding(0) var<storage, read> model_words: array<u32>;
@group(2) @binding(1) var model_textures: texture_2d_array<f32>;
@group(2) @binding(2) var model_sampler: sampler;
@group(2) @binding(3) var<uniform> model_draw: ModelDraw;

const MODEL_VERTEX_WORDS: u32 = 8u;
const MODEL_MATERIAL_WORDS: u32 = 16u;
const MODEL_NO_LAYER: u32 = 255u;
const MODEL_UNLIT: u32 = 256u;
const MODEL_MASK: u32 = 512u;
const MODEL_BLEND: u32 = 1024u;

struct ModelOut {
    @builtin(position) position: vec4<f32>,
    // Metres east, north and up of the instance, after its paint: for the
    // normal map's derivatives.
    @location(0) local: vec3<f32>,
    // East, north and up.
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // The vertex colour, linear.
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) material: u32,
    // The paint colour (linear, straight alpha), its mix and the emissive
    // strength.
    @location(5) @interpolate(flat) paint: vec4<f32>,
    @location(6) @interpolate(flat) mix_emissive: vec2<f32>,
};

fn model_word_f(i: u32) -> f32 { return bitcast<f32>(model_words[i]); }

fn model_number(row: u32, slot: u32, constant: f32, tile_zoom: f32) -> f32 {
    if (node.table_mask & (1u << slot)) != 0u {
        return table_number(row, slot, node.curves[slot], tile_zoom);
    }
    return constant;
}

fn model_vec3(row: u32, first: u32, constant: vec3<f32>, tile_zoom: f32) -> vec3<f32> {
    return vec3<f32>(
        model_number(row, first, constant.x, tile_zoom),
        model_number(row, first + 1u, constant.y, tile_zoom),
        model_number(row, first + 2u, constant.z, tile_zoom));
}

// Clip position of metres `o` east, north and up of ground point `g`: on
// the plane, or on the globe in its frame there, mixed as the globe hands
// over (`ground_clip_3d`'s placement in a local frame); reversed depth, as
// extrusions draw.
fn model_clip(g: Ground, o: vec3<f32>) -> vec4<f32> {
    let s = frame.extrusion_scale;
    var clip = frame.view_proj * vec4<f32>(g.flat + o.xy * s, o.z * s, 1.0);
    if frame.globe_amount > 0.0 {
        var p: vec3<f32>;
        if frame.globe_plane != 0u {
            p = g.globe + o * frame.globe_metres * s;
        } else {
            let up = normalize(g.globe + frame.globe_target);
            let level = vec3<f32>(-up.y, up.x, 0.0);
            let east = select(vec3<f32>(0.0, 1.0, 0.0), normalize(level), length(level) > 1e-6);
            let north = cross(up, east);
            p = g.globe + east * o.x + north * o.y + up * o.z;
        }
        var globe = frame.globe_view_proj * vec4<f32>(p, 1.0);
        if frame.globe_plane == 0u && sphere_hides(p) {
            globe.z = -globe.w;
        }
        clip = select(mix(clip, globe, frame.globe_amount), globe, !g.on_plane);
    }
    clip.z = clip.w - clip.z;
    return clip;
}

@vertex
fn vs_model(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ModelOut {
    var out: ModelOut;
    out.position = discarded_vertex();
    var v: Pulled;
    if !pull_point(node.index, instance, &v) { return out; }
    let tile = tile_at(v.tile);
    let anchor = unpack_i16x2(vertex_at(v.vertex));
    if !point_owned(tile, anchor) { return out; }
    let row = prop_row(v, vertex_at(v.vertex + 1u), v.row_words);
    var color = node.color;
    if (node.table_mask & 1u) != 0u { color = table_color(row, 0u, node.curves[0], tile.zoom); }
    let color_mix = model_number(row, 1u, node.color_mix, tile.zoom);
    let emissive = model_number(row, 2u, node.emissive, tile.zoom);
    let scale = model_vec3(row, 3u, node.scale, tile.zoom);
    let rotation = model_vec3(row, 6u, node.rotation, tile.zoom);
    let translation = model_vec3(row, 9u, node.translation, tile.zoom);

    let w = model_draw.vertex_base + vertex * MODEL_VERTEX_WORDS;
    let position = vec3<f32>(model_word_f(w), model_word_f(w + 1u), model_word_f(w + 2u));
    let local = turn_enu(position * scale, rotation) + translation;
    // On terrain, the model stands on the ground under its point.
    let ground = ground_elevation(tile, anchor);
    out.position = model_clip(tile_ground(tile, anchor), local + vec3<f32>(0.0, 0.0, ground));
    out.local = local;
    // Normals scale inversely, then turn with the model.
    out.normal = turn_enu(octahedral_normal(model_words[w + 3u]) / scale, rotation);
    out.uv = vec2<f32>(model_word_f(w + 4u), model_word_f(w + 5u));
    out.color = unpack4x8unorm(model_words[w + 6u]);
    out.material = model_words[w + 7u];
    let straight = select(color.rgb / color.a, vec3<f32>(0.0), color.a == 0.0);
    out.paint = vec4<f32>(srgb_to_linear(straight), color.a);
    out.mix_emissive = vec2<f32>(color_mix, emissive);
    return out;
}

// A material's texture `slot` (0–4: base colour, metallic-roughness,
// normal, occlusion, emissive) at `uv`, wrapped as its sampler says, or
// `fallback` without one.
fn model_texel(m: u32, layer: u32, slot: u32, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if layer == MODEL_NO_LAYER {
        return fallback;
    }
    let wrap = (model_words[m + 14u] >> (4u * slot)) & 15u;
    return textureSample(model_textures, model_sampler, wrap_uv(uv, wrap), layer);
}

// The style's lights in east, north and up, and the direction toward the
// eye. lighting.wgsl keeps GL JS's extrusion axes: x and y negated, y
// south.
fn model_lights() -> PbrLights {
    var lights: PbrLights;
    lights.ambient = srgb_to_linear(frame.light_ambient.rgb) * frame.light_ambient.w;
    lights.directional = srgb_to_linear(frame.light_color.rgb) * frame.light_color.w;
    lights.toward_light = normalize(vec3<f32>(-frame.light_position.x, frame.light_position.y, frame.light_position.z));
    lights.toward_eye = normalize(vec3<f32>(-sin(frame.bearing) * sin(frame.pitch), -cos(frame.bearing) * sin(frame.pitch), cos(frame.pitch)));
    lights.shadow = 1.0;
    return lights;
}

@fragment
fn fs_model(in: ModelOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let m = model_draw.material_base + in.material * MODEL_MATERIAL_WORDS;
    let layers = model_words[m + 12u];
    let flags = model_words[m + 13u];
    let factor = vec4<f32>(model_word_f(m), model_word_f(m + 1u), model_word_f(m + 2u), model_word_f(m + 3u));
    let texel = model_texel(m, layers & 255u, 0u, in.uv, vec4<f32>(1.0));
    var base = factor * vec4<f32>(srgb_to_linear(texel.rgb), texel.a) * in.color;
    if (flags & MODEL_MASK) != 0u && base.a < model_word_f(m + 7u) {
        discard;
    }
    let alpha = select(1.0, base.a, (flags & MODEL_BLEND) != 0u);
    let albedo = mix(base.rgb, in.paint.rgb, in.mix_emissive.x);
    if (flags & MODEL_UNLIT) != 0u {
        return vec4<f32>(linear_to_srgb(albedo) * alpha, alpha);
    }
    let mr = model_texel(m, (layers >> 8u) & 255u, 1u, in.uv, vec4<f32>(1.0));
    let metallic = clamp(model_word_f(m + 8u) * mr.b, 0.0, 1.0);
    let roughness = clamp(model_word_f(m + 9u) * mr.g, 0.04, 1.0);
    var n = normalize(in.normal);
    if !front {
        n = -n;
    }
    let normal_layer = (layers >> 16u) & 255u;
    if normal_layer != MODEL_NO_LAYER {
        let sampled = model_texel(m, normal_layer, 2u, in.uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).xyz;
        n = perturb_normal(n, mat2x3<f32>(dpdx(in.local), dpdy(in.local)), mat2x2<f32>(dpdx(in.uv), dpdy(in.uv)), sampled, model_word_f(m + 10u));
    }
    let occlusion = mix(1.0, model_texel(m, (layers >> 24u) & 255u, 3u, in.uv, vec4<f32>(1.0)).r, model_word_f(m + 11u));
    let lit = pbr_shade(albedo, metallic, roughness, occlusion, n, model_lights());
    let emitted = vec3<f32>(model_word_f(m + 4u), model_word_f(m + 5u), model_word_f(m + 6u))
        * srgb_to_linear(model_texel(m, flags & 255u, 4u, in.uv, vec4<f32>(1.0)).rgb);
    let color = mix(lit, albedo, in.mix_emissive.y) + emitted;
    return vec4<f32>(linear_to_srgb(color) * alpha, alpha);
}
