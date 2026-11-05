// Image families: raster, hillshade and color-relief. A tile is one quad over
// its whole extent, or the mesh that warps it onto another plane. Vertex:
// five words: the position in the placement's units as two f32 (a warped
// tile's frame can be much larger than the view, so i16 would crack), the
// tile units placement clips in as [x | y << 16], an f32 weight that makes
// an image source's warp projective across a triangle (1 elsewhere: image
// units interpolate as units · weight over weight), then the image's own
// tile units, which it samples at (the clip units but for an image
// source, whose grid sits in its one tile). Property row
// (`IMAGE_ROW_WORDS`): the tile's layer in the node's texture array, then
// the latitudes of its top and bottom edges.
//
// Group 1 is the node: its uniform at binding 0, its texture array at 1,
// linear and nearest samplers at 2 and 3.

// GL JS's `EXTENT`: image quads span this many units per tile side.
const IMAGE_EXTENT: f32 = 8192.0;

@group(1) @binding(1) var images: texture_2d_array<f32>;
@group(1) @binding(2) var linear_sampler: sampler;
@group(1) @binding(3) var nearest_sampler: sampler;

struct ImageOut {
    @builtin(position) position: vec4<f32>,
    // The image's tile units times the vertex's weight; `image_pos` divides.
    @location(0) tile_pos: vec2<f32>,
    @location(4) weight: f32,
    // The tile units placement clips in.
    @location(5) clip_pos: vec2<f32>,
    @location(1) @interpolate(flat) tile: u32,
    // The tile's texture layer.
    @location(2) @interpolate(flat) layer: u32,
    // Latitudes of the tile's top and bottom edges, in degrees.
    @location(3) @interpolate(flat) latitudes: vec2<f32>,
};

fn image_vertex(node_index: u32, instance: u32, index: u32) -> ImageOut {
    var out: ImageOut;
    var v: Pulled;
    if !pull(node_index, instance, index, 5u, &v) {
        out.position = discarded_vertex();
        return out;
    }
    let tile = tile_at(v.tile);
    let pos = bitcast<vec2<f32>>(vec2<u32>(vertex_at(v.vertex), vertex_at(v.vertex + 1u)));
    out.position = ground_clip(tile_ground(tile, pos), frame.align, 0.0);
    let weight = bitcast<f32>(vertex_at(v.vertex + 3u));
    out.tile_pos = unpack_i16x2(vertex_at(v.vertex + 4u)) * weight;
    out.weight = weight;
    out.clip_pos = unpack_i16x2(vertex_at(v.vertex + 2u));
    out.tile = v.tile;
    out.layer = prop_at(v.prop_base);
    out.latitudes = bitcast<vec2<f32>>(vec2<u32>(prop_at(v.prop_base + 1u), prop_at(v.prop_base + 2u)));
    return out;
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
    return !inside_clips(tile_at(in.tile), p);
}

// The tile's texel at `uv` (0–1 over its layer) for a footprint of `ddx`
// and `ddy` on screen: mipmapped and anisotropic where linear, as GL JS
// samples raster tiles (#3); nearest from the full-size level.
fn sample_image_grad(in: ImageOut, uv: vec2<f32>, nearest: u32, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    if nearest != 0u {
        return textureSampleLevel(images, nearest_sampler, uv, in.layer, 0.0);
    }
    return textureSampleGrad(images, linear_sampler, uv, in.layer, ddx, ddy);
}

// The tile's texel at `uv` (0–1 over its layer), filtered by nearest texel
// or linearly.
fn sample_image(in: ImageOut, uv: vec2<f32>, nearest: u32) -> vec4<f32> {
    if nearest != 0u {
        return textureSampleLevel(images, nearest_sampler, uv, in.layer, 0.0);
    }
    return textureSampleLevel(images, linear_sampler, uv, in.layer, 0.0);
}
