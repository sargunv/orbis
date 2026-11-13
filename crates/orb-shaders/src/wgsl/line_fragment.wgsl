// Lines' fragment stages, apart from `line.wgsl` for the vertex-only
// clip-plane shaders, which leave them out.

// A line's coverage of a fragment, antialiased across its edges.
fn line_alpha(normal: vec2<f32>, width2: vec2<f32>, blur: f32, gamma_scale: f32) -> f32 {
    let dist = length(normal) * width2.x;
    let blur2 = (blur + 1.0 / frame.pixel_ratio) * gamma_scale;
    return clamp(min(dist - (width2.y - blur2), width2.x - dist) / blur2, 0.0, 1.0);
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    if clip_fragments && !clip_drawn(in.clip, in.tile, in.owner_pos, in.tile_pos) {
        discard;
    }
    return in.color * line_alpha(in.normal, in.width2, in.blur, in.gamma_scale);
}

@fragment
fn fs_line_styled(in: StyledLineOut) -> @location(0) vec4<f32> {
    if clip_fragments && !clip_drawn(in.clip, in.tile, in.owner_pos, in.tile_pos) {
        discard;
    }
    var alpha = line_alpha(in.normal, in.width2, in.blur, in.gamma_scale);
    if with_patterns && node.pattern.enabled != 0u {
        let a = image_at(in.images.x);
        let b = image_at(in.images.y);
        let size_a = max(vec2<f32>(a.size) / max(a.pixel_ratio, 1.0e-6), vec2<f32>(1.0e-6));
        let size_b = max(vec2<f32>(b.size) / max(b.pixel_ratio, 1.0e-6), vec2<f32>(1.0e-6));
        let ratio = tile_pixel_ratio(tile_at(in.tile));
        let x_a = fract(in.linesofar / (size_a.x * node.pattern.from_scale / ratio) * size_a.y / max(in.floorwidth, 1.0e-6));
        let x_b = fract(in.linesofar / (size_b.x * node.pattern.to_scale / ratio) * size_b.y / max(in.floorwidth, 1.0e-6));
        let y = in.normal.y * 0.5 + 0.5;
        let ca = pattern_sample(a, vec2<f32>(x_a, y), 1.0);
        let cb = pattern_sample(b, vec2<f32>(x_b, y), 1.0);
        return mix(ca, cb, node.pattern.fade) * alpha * in.opacity;
    }
    if node.dashed != 0u {
        let a = image_at(in.images.x);
        let b = image_at(in.images.y);
        let scale = tile_pixel_ratio(tile_at(in.tile)) / max(in.floorwidth, 1.0e-6);
        let ca = dash_sample(a, in.linesofar * scale / node.dash_from_scale, in.normal.y);
        let cb = dash_sample(b, in.linesofar * scale, in.normal.y);
        let gamma = 1.0 / frame.pixel_ratio / max(min(a.pixel_ratio, b.pixel_ratio), 1.0e-6) / max(in.floorwidth, 1.0e-6);
        alpha *= smoothstep(0.5 - gamma, 0.5 + gamma, mix(ca, cb, node.dash_fade));
    }
    if node.gradient != 0u {
        return gradient_sample(image_at(in.ramp), in.progress) * alpha * in.opacity;
    }
    return in.color * alpha;
}

// GL JS LineAtlas: center row and 14-texel span for round dashes, one row
// for butt dashes. The scalar distance is stored in R8 rather than alpha.
fn dash_sample(image: ImageRect, distance: f32, normal: f32) -> f32 {
    let dimensions = vec2<f32>(textureDimensions(dash_texture));
    let uv = vec2<f32>(fract(distance / max(image.pixel_ratio, 1.0e-6)), (f32(image.origin.y) + 0.5 - normal * f32(image.size.y) / 2.0) / dimensions.y);
    // Repeat horizontally across texel centers, without wrapping atlas rows.
    let x = uv.x * dimensions.x - 0.5;
    let a = vec2<f32>((floor(x) + dimensions.x) % dimensions.x + 0.5, uv.y * dimensions.y) / dimensions;
    let b = vec2<f32>((floor(x) + 1.0) % dimensions.x + 0.5, uv.y * dimensions.y) / dimensions;
    return mix(textureSampleLevel(dash_texture, image_sampler, a, 0.0).r, textureSampleLevel(dash_texture, image_sampler, b, 0.0).r, fract(x));
}

fn gradient_texel(image: ImageRect, index: u32) -> vec4<f32> {
    let p = image.origin + vec2<u32>(index % 256u, index / 256u);
    return textureLoad(image_texture, vec2<i32>(p), i32(image.layer), 0);
}

fn gradient_sample(image: ImageRect, progress: f32) -> vec4<f32> {
    let size = image.size.x * image.size.y;
    if size == 0u { return vec4<f32>(0.0); }
    if image.sdf != 0u { return gradient_texel(image, min(u32(clamp(progress, 0.0, 1.0) * f32(size)), size - 1u)); }
    let x = clamp(progress * f32(size) - 0.5, 0.0, f32(size - 1u));
    let i = u32(x);
    return mix(gradient_texel(image, i), gradient_texel(image, min(i + 1u, size - 1u)), fract(x));
}
