// glTF models at point features (plan §9.4): each instance is a visible
// point of the node (`pull_instance`); the draw's vertices are its
// model's, in metres east, north and up of the point, scaled, turned and
// moved by the layer's paint. A fragment fills a `Material` from glTF's
// metallic-roughness material and the paint, and lights it under the
// style's lights (pbr.wgsl, decision 0026), with the shadows it receives
// (decision 0046) in the directional term. Model words are
// `orb_3d::gpu`'s. Opaque materials shade in a pipeline that never
// discards (`fs_model`), masked and blended ones in `fs_model_masked`.

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
    state: StateLookup,
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
const MODEL_DOUBLE_SIDED: u32 = 2048u;

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
    // Camera-relative world and view depth, for shadows.
    @location(7) world: vec3<f32>,
    @location(8) depth: f32,
};

fn model_word_f(i: u32) -> f32 { return bitcast<f32>(model_words[i]); }

fn model_number(row: u32, state: u32, slot: u32, constant: f32, tile_zoom: f32) -> f32 {
    if from_state(state, node.state, slot) {
        return state_number(state, node.state, slot, node.curves[slot]);
    }
    if (node.table_mask & (1u << slot)) != 0u {
        return table_number(row, slot, node.curves[slot], tile_zoom);
    }
    return constant;
}

fn model_vec3(row: u32, state: u32, first: u32, constant: vec3<f32>, tile_zoom: f32) -> vec3<f32> {
    return vec3<f32>(
        model_number(row, state, first, constant.x, tile_zoom),
        model_number(row, state, first + 1u, constant.y, tile_zoom),
        model_number(row, state, first + 2u, constant.z, tile_zoom));
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
            globe.z = 2.0 * globe.w;
        }
        clip = select(mix(clip, globe, frame.globe_amount), globe, !g.on_plane);
    }
    clip.z = clip.w - clip.z;
    return clip;
}

// A model vertex at its instance: the instance's tile, ground point,
// property row and feature-state block, and the vertex's word, its place in metres east, north
// and up of the point after the paint, and that on the ground under it.
struct ModelPlaced {
    tile: Tile,
    ground: Ground,
    row: u32,
    state: u32,
    w: u32,
    local: vec3<f32>,
    rotation: vec3<f32>,
    scale: vec3<f32>,
    standing: vec3<f32>,
};

fn model_placed(vertex: u32, instance: u32, out: ptr<function, ModelPlaced>) -> bool {
    var v: Pulled;
    if !pull_instance(node.index, instance, &v) { return false; }
    let tile = tile_at(v.tile);
    let anchor = unpack_i16x2(vertex_at(v.vertex));
    if !point_owned(tile, anchor) { return false; }
    let feature = vertex_at(v.vertex + 1u);
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);
    let scale = model_vec3(row, state, 3u, node.scale, tile.zoom);
    let rotation = model_vec3(row, state, 6u, node.rotation, tile.zoom);
    let translation = model_vec3(row, state, 9u, node.translation, tile.zoom);
    let w = model_draw.vertex_base + vertex * MODEL_VERTEX_WORDS;
    let position = vec3<f32>(model_word_f(w), model_word_f(w + 1u), model_word_f(w + 2u));
    let local = turn_enu(position * scale, rotation) + translation;
    // On terrain, the model stands on the ground under its point.
    let ground = ground_elevation(tile, anchor);
    *out = ModelPlaced(tile, tile_ground(tile, anchor), row, state, w, local, rotation, scale, local + vec3<f32>(0.0, 0.0, ground));
    return true;
}

// Camera-relative world (x east, y north, z up) of a placed vertex, on
// the plane, where shadows fall (decision 0046).
fn model_world(p: ModelPlaced) -> vec3<f32> {
    let s = frame.extrusion_scale;
    return vec3<f32>(p.ground.flat + p.standing.xy * s, p.standing.z * s);
}

@vertex
fn vs_model(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ModelOut {
    var out: ModelOut;
    out.position = discarded_vertex();
    var p: ModelPlaced;
    if !model_placed(vertex, instance, &p) { return out; }
    var color = node.color;
    if (node.table_mask & 1u) != 0u { color = table_color(p.row, 0u, node.curves[0], p.tile.zoom); }
    if from_state(p.state, node.state, 0u) { color = state_color(p.state, node.state, 0u, node.curves[0]); }
    let color_mix = model_number(p.row, p.state, 1u, node.color_mix, p.tile.zoom);
    let emissive = model_number(p.row, p.state, 2u, node.emissive, p.tile.zoom);
    out.position = model_clip(p.ground, p.standing);
    out.local = p.local;
    out.world = model_world(p);
    out.depth = out.position.w;
    // Normals scale inversely, then turn with the model.
    out.normal = turn_enu(octahedral_normal(model_words[p.w + 3u]) / p.scale, p.rotation);
    out.uv = vec2<f32>(model_word_f(p.w + 4u), model_word_f(p.w + 5u));
    out.color = unpack4x8unorm(model_words[p.w + 6u]);
    out.material = model_words[p.w + 7u];
    let straight = select(color.rgb / color.a, vec3<f32>(0.0), color.a == 0.0);
    out.paint = vec4<f32>(srgb_to_linear(straight), color.a);
    out.mix_emissive = vec2<f32>(color_mix, emissive);
    return out;
}

// The texture coordinates and their screen derivatives, taken where
// control flow is uniform: samples then need none of it.
struct ModelUv {
    uv: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
};

// A material's texture `slot` (0–4: base colour, metallic-roughness,
// normal, occlusion, emissive) at `uv`, wrapped as its sampler says, or
// `fallback` without one.
fn model_texel(m: u32, layer: u32, slot: u32, uv: ModelUv, fallback: vec4<f32>) -> vec4<f32> {
    if layer == MODEL_NO_LAYER {
        return fallback;
    }
    let wrap = (model_words[m + 14u] >> (4u * slot)) & 15u;
    return textureSampleGrad(model_textures, model_sampler, wrap_uv(uv.uv, wrap), layer, uv.dx, uv.dy);
}

// Material `m`'s base colour (linear, straight alpha) at `uv`, times the
// vertex colour.
fn model_base(m: u32, uv: ModelUv, color: vec4<f32>) -> vec4<f32> {
    let factor = vec4<f32>(model_word_f(m), model_word_f(m + 1u), model_word_f(m + 2u), model_word_f(m + 3u));
    let texel = model_texel(m, model_words[m + 12u] & 255u, 0u, uv, vec4<f32>(1.0));
    return factor * vec4<f32>(srgb_to_linear(texel.rgb), texel.a) * color;
}

// The style's lights, with the directional light's shadow at the
// surface's geometric normal `n`.
fn model_lights(world: vec3<f32>, n: vec3<f32>, depth: f32) -> PbrLights {
    var lights = scene_lights();
    lights.shadow = shadow_at(world, n, depth);
    return lights;
}

// Opaque materials (orb_3d's `DrawClass::Opaque` and `DoubleSided`):
// nothing discards, so a tile-based GPU keeps hidden-surface removal; the
// rasterizer culls a single-sided material's back faces.
@fragment
fn fs_model(in: ModelOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let uv = ModelUv(in.uv, dpdx(in.uv), dpdy(in.uv));
    let dp = mat2x3<f32>(dpdx(in.local), dpdy(in.local));
    let m = model_draw.material_base + in.material * MODEL_MATERIAL_WORDS;
    return model_shade(in, front, uv, dp, m, model_base(m, uv, in.color));
}

// Masked and blended materials, both faces drawn: a single-sided
// material's back faces and texels under a mask's cutoff are discarded.
@fragment
fn fs_model_masked(in: ModelOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let uv = ModelUv(in.uv, dpdx(in.uv), dpdy(in.uv));
    let dp = mat2x3<f32>(dpdx(in.local), dpdy(in.local));
    let m = model_draw.material_base + in.material * MODEL_MATERIAL_WORDS;
    let flags = model_words[m + 13u];
    if !front && (flags & MODEL_DOUBLE_SIDED) == 0u {
        discard;
    }
    let base = model_base(m, uv, in.color);
    if (flags & MODEL_MASK) != 0u && base.a < model_word_f(m + 7u) {
        discard;
    }
    return model_shade(in, front, uv, dp, m, base);
}

// Material `m` at a fragment, its base colour `base`, lit: `uv` and `dp`
// the texture coordinates' and the position's screen derivatives, taken
// where control flow is uniform.
fn model_shade(in: ModelOut, front: bool, uv: ModelUv, dp: mat2x3<f32>, m: u32, base: vec4<f32>) -> vec4<f32> {
    let layers = model_words[m + 12u];
    let flags = model_words[m + 13u];
    var material: Material;
    material.albedo = mix(base.rgb, in.paint.rgb, in.mix_emissive.x);
    material.alpha = select(1.0, base.a, (flags & MODEL_BLEND) != 0u);
    let mr = model_texel(m, (layers >> 8u) & 255u, 1u, uv, vec4<f32>(1.0));
    material.metallic = clamp(model_word_f(m + 8u) * mr.b, 0.0, 1.0);
    material.roughness = clamp(model_word_f(m + 9u) * mr.g, 0.04, 1.0);
    var n = normalize(in.normal);
    if !front {
        n = -n;
    }
    let geometric = n;
    let normal_layer = (layers >> 16u) & 255u;
    if normal_layer != MODEL_NO_LAYER {
        let sampled = model_texel(m, normal_layer, 2u, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).xyz;
        n = perturb_normal(n, dp, mat2x2<f32>(uv.dx, uv.dy), sampled, model_word_f(m + 10u));
    }
    material.normal = n;
    material.occlusion = mix(1.0, model_texel(m, (layers >> 24u) & 255u, 3u, uv, vec4<f32>(1.0)).r, model_word_f(m + 11u));
    material.self_lit = select(in.mix_emissive.y, 1.0, (flags & MODEL_UNLIT) != 0u);
    material.emitted = vec3<f32>(model_word_f(m + 4u), model_word_f(m + 5u), model_word_f(m + 6u))
        * srgb_to_linear(model_texel(m, flags & 255u, 4u, uv, vec4<f32>(1.0)).rgb);
    let color = pbr_light(material, model_lights(in.world, geometric, in.depth));
    return vec4<f32>(linear_to_srgb(color) * material.alpha, material.alpha);
}
