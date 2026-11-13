// Image families: raster, hillshade and color-relief. A tile is one quad over
// its whole extent, or the mesh that warps it onto another plane. Vertex:
// five words: the position in the placement's units as two f32 (a warped
// tile's frame can be much larger than the view, so i16 would crack), the
// tile units placement clips in as [x | y << 16], an f32 weight that makes
// an image source's warp projective across a triangle (1 elsewhere: image
// units interpolate as units · weight over weight), then the image's own
// tile units, which it samples at (the clip units but for an image
// source, whose grid sits in its one tile). Property row
// (`IMAGE_ROW_WORDS`): the tile's layer in the node's texture array, the
// latitudes of its top and bottom edges, then its fade (orb-render's
// `ImageFade`): when it arrived (f32 ms, the frame's clock), its ancestor's
// layer (or none), and that ancestor's depth and the tile's offsets in it
// (8, 12 and 12 bits).
//
// Group 1 is the node: its uniform at binding 0, the linear sampler at 2,
// and its page's chunks at 1 and 5 to 7 (orb-render's `TilePage`): a
// layer word's bits from 11 up name its chunk, those below its layer
// there. Each texture binding takes the one sampler, as a texture unit
// does on WebGL 2, where wgpu gives a texture the last sampler bound with
// it; the nearest texel is loaded rather than sampled.

// GL JS's `EXTENT`: image quads span this many units per tile side.
const IMAGE_EXTENT: f32 = 8192.0;

@group(1) @binding(2) var linear_sampler: sampler;
@group(1) @binding(1) var images_0: texture_2d_array<f32>;
@group(1) @binding(5) var images_1: texture_2d_array<f32>;
@group(1) @binding(6) var images_2: texture_2d_array<f32>;
@group(1) @binding(7) var images_3: texture_2d_array<f32>;

// Texels a side of the page's layers.
fn image_size() -> vec2<u32> {
    return textureDimensions(images_0);
}

struct ImageOut {
    @builtin(position) position: vec4<f32>,
    // The image's tile units times the vertex's weight; `image_pos` divides.
    @location(0) tile_pos: vec2<f32>,
    @location(4) weight: f32,
    // The tile units placement clips in.
    @location(5) clip_pos: vec2<f32>,
    @location(1) @interpolate(flat, either) tile: u32,
    // The tile's texture layer.
    @location(2) @interpolate(flat, either) layer: u32,
    // Latitudes of the tile's top and bottom edges, in degrees.
    @location(3) @interpolate(flat, either) latitudes: vec2<f32>,
    // Its fade's three row words.
    @location(6) @interpolate(flat, either) fade: vec3<u32>,
    // The placement's clip as fragments test it (`fragment_clip`).
    @location(7) @interpolate(flat, either) clip: vec4<f32>,
};

const NO_ANCESTOR: u32 = 0xffffffffu;

fn image_vertex(node_index: u32, instance: u32, index: u32) -> ImageOut {
    var out: ImageOut;
    var v: Pulled;
    if !pull(node_index, instance, index, 5u, &v) {
        out.position = discarded_vertex();
        return out;
    }
    let tile = tile_at(v.tile);
    var pos = bitcast<vec2<f32>>(vec2<u32>(vertex_at(v.vertex), vertex_at(v.vertex + 1u)));
    let weight = bitcast<f32>(vertex_at(v.vertex + 3u));
    var image = unpack_i16x2(vertex_at(v.vertex + 4u));
    var clip = unpack_i16x2(vertex_at(v.vertex + 2u));
    // On an affine tile, where a vertex's place, clip position and image
    // position are one point, the mesh keeps to its placement's clips'
    // box: a coarse tile standing in for a little of itself (in a drape
    // slot, or for the regions its descendants left while a zoom crosses
    // levels) would otherwise cover far more only to discard it, once for
    // each of its placements. On the globe too, whose grid's lines are
    // the tile's axes, so a clamped grid is the grid within the box: a
    // tip of a pole's cap (placed at the pole, clipped and sampled on the
    // tile's edge) stays at the pole where the box reaches it, and else
    // falls in the box with the edge.
    if tile.warp_cells == 0u && weight == 1.0 && all(clip == image) {
        let box = clips_box(tile);
        let hi = max(box.xy, box.zw);
        let north = pos.y <= -32767.5;
        let south = pos.y >= 32766.5;
        let cap = (north && box.y <= -32767.5) || (south && hi.y >= 32766.5);
        if all(pos == clip) || ((north || south) && !cap) {
            pos = clamp(clip, box.xy, hi);
            clip = pos;
            image = pos;
        }
    }
    let ground = tile_ground(tile, pos);
    out.position = ground_clip(ground, frame.align, 0.0);
    out.tile_pos = image * weight;
    out.weight = weight;
    out.clip_pos = clip;
    out.tile = v.tile;
    out.layer = prop_at(v.prop_base);
    out.latitudes = bitcast<vec2<f32>>(vec2<u32>(prop_at(v.prop_base + 1u), prop_at(v.prop_base + 2u)));
    out.fade = vec3<u32>(prop_at(v.prop_base + 3u), prop_at(v.prop_base + 4u), prop_at(v.prop_base + 5u));
    out.clip = fragment_clip(tile, false);
    return out;
}

// GL JS's fade values over `duration` ms at the frame's time: (opacity,
// mix toward the ancestor). With an ancestor the tile cross-fades from it;
// without, it fades in.
fn image_fade(in: ImageOut, duration: f32) -> vec2<f32> {
    if duration <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let since = clamp((frame.time - bitcast<f32>(in.fade.x)) / duration, 0.0, 1.0);
    if in.fade.y != NO_ANCESTOR {
        return vec2<f32>(1.0, 1.0 - since);
    }
    return vec2<f32>(since, 0.0);
}

// Where `uv` (0–1 over the tile) falls in its ancestor, and the scale.
fn image_ancestor_uv(in: ImageOut, uv: vec2<f32>) -> vec3<f32> {
    let depth = in.fade.z & 255u;
    let offset = vec2<f32>(f32((in.fade.z >> 8u) & 4095u), f32(in.fade.z >> 20u));
    let scale = 1.0 / f32(1u << depth);
    return vec3<f32>((offset + uv) * scale, scale);
}

// The fragment's position in the image's tile units.
fn image_pos(in: ImageOut) -> vec2<f32> {
    return in.tile_pos / in.weight;
}

// Whether the fragment belongs to this placement of the tile. A warped
// mesh's interpolated image position can stray just past the tile's edge,
// where no other placement draws; it belongs to the edge. A placement
// drawing its whole square holds every one (`clip_fragments`).
fn image_clipped(in: ImageOut) -> bool {
    if !clip_fragments {
        return false;
    }
    let p = clamp(in.clip_pos, vec2<f32>(0.0), vec2<f32>(IMAGE_EXTENT - 0.5));
    return !clip_drawn(in.clip, in.tile, p, p);
}

// The tile's texel at `uv` (0–1 over its layer) for a footprint of `ddx`
// and `ddy` on screen: mipmapped and anisotropic where linear, as GL JS
// samples raster tiles; nearest from the full-size level.
fn sample_image_grad(in: ImageOut, uv: vec2<f32>, nearest: u32, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    if nearest != 0u {
        return nearest_texel(in.layer, uv);
    }
    let l = i32(in.layer & 2047u);
    switch in.layer >> 11u {
        case 0u: { return textureSampleGrad(images_0, linear_sampler, uv, l, ddx, ddy); }
        case 1u: { return textureSampleGrad(images_1, linear_sampler, uv, l, ddx, ddy); }
        case 2u: { return textureSampleGrad(images_2, linear_sampler, uv, l, ddx, ddy); }
        default: { return textureSampleGrad(images_3, linear_sampler, uv, l, ddx, ddy); }
    }
}

// The tile's texel at `uv` (0–1 over its layer), filtered by nearest texel
// or linearly.
fn sample_image(in: ImageOut, uv: vec2<f32>, nearest: u32) -> vec4<f32> {
    if nearest != 0u {
        return nearest_texel(in.layer, uv);
    }
    let l = i32(in.layer & 2047u);
    switch in.layer >> 11u {
        case 0u: { return textureSampleLevel(images_0, linear_sampler, uv, l, 0.0); }
        case 1u: { return textureSampleLevel(images_1, linear_sampler, uv, l, 0.0); }
        case 2u: { return textureSampleLevel(images_2, linear_sampler, uv, l, 0.0); }
        default: { return textureSampleLevel(images_3, linear_sampler, uv, l, 0.0); }
    }
}

// The full-size level's texel of layer word `layer` nearest `uv`, as a
// nearest sampler clamped to the edge reads it.
fn nearest_texel(layer: u32, uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<i32>(image_size());
    let p = clamp(vec2<i32>(floor(uv * vec2<f32>(size))), vec2<i32>(0), size - 1);
    let l = i32(layer & 2047u);
    switch layer >> 11u {
        case 0u: { return textureLoad(images_0, p, l, 0); }
        case 1u: { return textureLoad(images_1, p, l, 0); }
        case 2u: { return textureLoad(images_2, p, l, 0); }
        default: { return textureLoad(images_3, p, l, 0); }
    }
}
