// ImageAtlas table: rect in physical texels, array page, source pixel ratio,
// SDF flag. Missing rows are zero, and sample transparent without division.
struct ImageRect {
    origin: vec2<u32>,
    size: vec2<u32>,
    layer: u32,
    pixel_ratio: f32,
    sdf: u32,
    _pad: u32,
};
@group(0) @binding(8) var<storage, read> images: array<ImageRect>;
@group(0) @binding(9) var image_texture: texture_2d_array<f32>;
@group(0) @binding(10) var image_sampler: sampler;
@group(0) @binding(11) var dash_texture: texture_2d<f32>;

struct Pattern {
    from_scale: f32,
    to_scale: f32,
    fade: f32,
    enabled: u32,
    floor_width: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

fn pattern_ids(row: u32, base: u32, image_from: u32, image_to: u32, mask: u32, pattern: Pattern) -> vec2<u32> {
    if (mask & 64u) != 0u { return vec2<u32>(props[row + base + select(2u, 0u, pattern.from_scale > 1.0)], props[row + base + 1u]); }
    return vec2<u32>(image_from, image_to);
}

fn tile_pixel_ratio(tile: Tile) -> f32 {
    return tile.scale / tile.world_per_px * exp2(floor(frame.zoom) - frame.zoom);
}

fn modulo(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a - b * floor(a / b); }

// GL JS get_pattern_pos: split the global origin before modular arithmetic
// so adjacent tiles agree even at zoom 22 and with uneven pattern sizes.
fn pattern_position(tile: Tile, pos: vec2<f32>, image: ImageRect, scale: f32) -> vec2<f32> {
    let size = max(vec2<f32>(image.size) / max(image.pixel_ratio, 1.0e-6) * scale, vec2<f32>(1.0e-6));
    let upper = vec2<f32>(vec2<i32>(bitcast<vec2<i32>>(tile.pattern_origin) >> vec2<u32>(16u)));
    let lower = vec2<f32>(tile.pattern_origin & vec2<u32>(65535u));
    let offset = modulo(modulo(modulo(upper, size) * 256.0, size) * 256.0 + lower, size);
    return (tile_pixel_ratio(tile) * pos + offset) / size;
}

fn pattern_sample(image: ImageRect, uv: vec2<f32>, padding: f32) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(image_texture));
    let tl = vec2<f32>(image.origin) - padding;
    let br = vec2<f32>(image.origin + image.size) + padding;
    var texel = mix(tl, br, uv);
    // Clamp to the padded subimage's texel centres, just as a standalone
    // GL JS tile atlas clamps its outer edge. Neighbouring images must not
    // contribute when line AA reaches the padding boundary.
    if padding > 0.0 { texel = clamp(texel, tl + 0.5, br - 0.5); }
    let sampled = textureSampleLevel(image_texture, image_sampler, texel / size, i32(image.layer), 0.0);
    return select(vec4<f32>(0.0), sampled, all(image.size > vec2<u32>(0u)));
}

fn fill_pattern_color(ids: vec2<u32>, tile: Tile, pos: vec2<f32>, pattern: Pattern, color: vec4<f32>, opacity: f32) -> vec4<f32> {
    let a = images[ids.x];
    let b = images[ids.y];
    let ca = pattern_sample(a, fract(pattern_position(tile, pos, a, pattern.from_scale)), 0.0);
    let cb = pattern_sample(b, fract(pattern_position(tile, pos, b, pattern.to_scale)), 0.0);
    let ga = max(fwidth(ca.a) * 0.5, 1.0 / 255.0 / 16.0);
    let gb = max(fwidth(cb.a) * 0.5, 1.0 / 255.0 / 16.0);
    let sa = smoothstep(0.75 - ga, 0.75 + ga, ca.a);
    let sb = smoothstep(0.75 - gb, 0.75 + gb, cb.a);
    let color_a = select(ca * opacity, color * sa, a.sdf != 0u);
    let color_b = select(cb * opacity, color * sb, b.sdf != 0u);
    return mix(color_a, color_b, pattern.fade);
}
