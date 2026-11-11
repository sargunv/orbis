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
// Group 1 is the node: its uniform at binding 0, its texture array at 1,
// linear and nearest samplers at 2 and 3, and the array again at 5 for
// the nearest sampler. Each texture binding takes one sampler, as a
// texture unit does on WebGL 2, where wgpu gives a texture the last
// sampler bound with it.

// GL JS's `EXTENT`: image quads span this many units per tile side.
const IMAGE_EXTENT: f32 = 8192.0;

@group(1) @binding(1) var images: texture_2d_array<f32>;
@group(1) @binding(2) var linear_sampler: sampler;
@group(1) @binding(3) var nearest_sampler: sampler;
@group(1) @binding(5) var images_nearest: texture_2d_array<f32>;

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
    // box: a coarse tile standing in for a little of itself, as in a drape
    // slot, would otherwise cover far more (all of the drape atlas) only
    // to discard it.
    if tile.warp_cells == 0u && tile.globe_cells == 0u && weight == 1.0 && all(pos == clip) && all(clip == image) {
        let box = clips_box(tile);
        pos = clamp(pos, box.xy, max(box.xy, box.zw));
        clip = pos;
        image = pos;
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
// where no other placement draws; it belongs to the edge.
fn image_clipped(in: ImageOut) -> bool {
    let p = clamp(in.clip_pos, vec2<f32>(0.0), vec2<f32>(IMAGE_EXTENT - 0.5));
    return !clip_drawn(in.clip, in.tile, p, p);
}

// The tile's texel at `uv` (0–1 over its layer) for a footprint of `ddx`
// and `ddy` on screen: mipmapped and anisotropic where linear, as GL JS
// samples raster tiles; nearest from the full-size level.
fn sample_image_grad(in: ImageOut, uv: vec2<f32>, nearest: u32, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    if nearest != 0u {
        return textureSampleLevel(images_nearest, nearest_sampler, uv, in.layer, 0.0);
    }
    return textureSampleGrad(images, linear_sampler, uv, in.layer, ddx, ddy);
}

// The tile's texel at `uv` (0–1 over its layer), filtered by nearest texel
// or linearly.
fn sample_image(in: ImageOut, uv: vec2<f32>, nearest: u32) -> vec4<f32> {
    if nearest != 0u {
        return textureSampleLevel(images_nearest, nearest_sampler, uv, in.layer, 0.0);
    }
    return textureSampleLevel(images, linear_sampler, uv, in.layer, 0.0);
}
