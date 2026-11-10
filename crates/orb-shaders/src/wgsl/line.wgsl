// Line family, after GL JS 2b68611a4e2d4f7f076f6c6ebf758e2d53f91471's
// src/shaders/glsl/line.{vertex,fragment}.glsl (BSD-3-Clause).
// Vertex, 4 words (16 bytes): [x i16 | y i16 << 16,
// extrude.x i8 | extrude.y i8 << 8 | offset.x i8 << 16 | offset.y i8 << 24,
// flags (bit 0 round, bit 1 up; bits 2–31 half line distance), feature slot],
// and from a source with `lineMetrics` (`node.metrics`) a fifth, f32 local
// progress. With metrics the packed distance is global progress * 32767 / 2;
// the fifth word retains full precision within the clip for gradient ramps.
// Both vectors are scaled by 63 with JavaScript rounding. Offset normals are
// bounded join miters, easing toward segment normals near hairpins.
// Property row (`LINE_ROW_*`): color, opacity, width, gap width, offset,
// blur.

struct LineNode {
    color: vec4<f32>,
    opacity: f32,
    width: f32,
    gap_width: f32,
    offset: f32,
    blur: f32,
    // The node's index in the node tables.
    index: u32,
    // Bit i: property i comes from the property row.
    table_mask: u32,
    // 1 when `line-translate-anchor` is `viewport`.
    translate_anchor: u32,
    // `line-translate`, in pixels.
    translate: vec2<f32>,
    image_from: u32,
    image_to: u32,
    curves: array<Curve, 6>,
    translate_curve: Curve,
    pattern: Pattern,
    dash_from: u32,
    dash_to: u32,
    dash_from_scale: f32,
    dash_fade: f32,
    dash_floor_width: f32,
    dashed: u32,
    gradient: u32,
    // 1 when vertices carry a fifth word, their progress.
    metrics: u32,
    state: StateLookup,
};

@group(1) @binding(0) var<uniform> node: LineNode;

const LINE_STRIDE: u32 = 4u;
const EXTRUDE_SCALE: f32 = 1.0 / 63.0;

// A dashed, patterned or gradient line's varyings (`vs_line_styled`).
struct StyledLineOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec2<f32>,
    // (outset, inset) in pixels.
    @location(2) width2: vec2<f32>,
    @location(3) gamma_scale: f32,
    @location(4) blur: f32,
    @location(5) tile_pos: vec2<f32>,
    @location(6) @interpolate(flat) tile: u32,
    // Where the placement owns the line: where it lands, but where it was
    // in a drape slot, so a translated line moves whole into it, bounded
    // by the slot where it lands (`tile_pos`).
    @location(7) owner_pos: vec2<f32>,
    @location(8) @interpolate(flat) clip: vec2<u32>,
    @location(9) linesofar: f32,
    @location(10) floorwidth: f32,
    @location(11) @interpolate(flat) images: vec2<u32>,
    @location(12) opacity: f32,
    @location(13) progress: f32,
    @location(14) @interpolate(flat) ramp: u32,
};

// A plain line's (`vs_line`): the styled line's first nine. Every vertex
// output is stored and every fragment interpolates it: on a pitched city
// view the styled ones cost plain lines a tenth of their main pass.
struct LineOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) width2: vec2<f32>,
    @location(3) gamma_scale: f32,
    @location(4) blur: f32,
    @location(5) tile_pos: vec2<f32>,
    @location(6) @interpolate(flat) tile: u32,
    @location(7) owner_pos: vec2<f32>,
    @location(8) @interpolate(flat) clip: vec2<u32>,
};

fn line_prop(row: u32, state: u32, bit: u32, constant: f32, tile_zoom: f32) -> f32 {
    var value = constant;
    if (node.table_mask & (1u << bit)) != 0u {
        value = table_number(row, bit, node.curves[bit], tile_zoom);
    }
    if from_state(state, node.state, bit) {
        value = state_number(state, node.state, bit, node.curves[bit], value);
    }
    return value;
}

fn signed_byte(w: u32, shift: u32) -> f32 {
    return f32(bitcast<i32>(w << (24u - shift)) >> 24u);
}

@vertex
fn vs_line(@builtin(vertex_index) index: u32) -> LineOut {
    let s = pulled_line(index);
    return LineOut(s.position, s.color, s.normal, s.width2, s.gamma_scale, s.blur, s.tile_pos, s.tile, s.owner_pos, s.clip);
}

@vertex
fn vs_line_styled(@builtin(vertex_index) index: u32) -> StyledLineOut {
    return pulled_line(index);
}

fn pulled_line(index: u32) -> StyledLineOut {
    var v: Pulled;
    if !pull_indexed(node.index, index, LINE_STRIDE + node.metrics, &v) {
        var out: StyledLineOut;
        out.position = discarded_vertex();
        return out;
    }
    return line_vertex(v);
}

// Data units per pixel of extrusion at tile point `p`, for a vertex
// whose offset normal is `across` (data units, y down) and `miter` long
// (1 on a straight run or a cap). GL JS lineUniformValues uses
// transform.zoom for u_ratio (tileZoom only scales patterned and dashed
// lines' atlas); in our camera-relative world its reciprocal is
// world_per_px over the tile's scale. A warp or a sphere can stretch one
// direction more than the other, where one scale can't be right both
// ways. With J its Jacobian and K = JᵀJ its metric, a line's normal n
// goes to J⁻ᵀn on screen and its direction d to Jd, and a point a pixels
// off the line on screen is a·|J⁻ᵀn| off it along n in data units.
//
// A join's two segment normals sit at ± its half-angle about the
// bisector b, cos = 1 / miter. The extrusion keeps its distance from
// both lines, in pixels, so a miter meets where the lines' edges meet on
// screen and a bevel's corners stay on them. On a straight run that is
// K⁻¹n / |J⁻ᵀn| across, with d / |Jd| along for its caps. The miter's
// bytes round to a half-angle of up to 9° where there is none, so joins
// whose half-angle is under about 17° (sine 0.3) ease to the straight
// run's frame.
fn line_px_to_data(tile: Tile, p: vec2<f32>, across: vec2<f32>, miter: f32) -> mat2x2<f32> {
    let world_per_px = tile_world_per_px(tile);
    if tile.warp_cells == 0u && (frame.globe_amount <= 0.0 || tile.globe_cells == 0u) {
        let s = world_per_px / tile.scale;
        return mat2x2<f32>(s, 0.0, 0.0, s);
    }
    let k = tile_metric_at(tile, p);
    let inv = mat2x2<f32>(k[1][1], -k[0][1], -k[1][0], k[0][0]) * (1.0 / max(determinant(k), 1.0e-30));
    let b = across * inverseSqrt(max(dot(across, across), 1.0e-12));
    let e = vec2<f32>(-b.y, b.x);
    let raw = sqrt(max(1.0 - 1.0 / (miter * miter), 0.0));
    let blend = smoothstep(0.15, 0.3, raw);
    let s = raw * blend;
    let c = sqrt(1.0 - s * s);
    // With normals n = c b ± s e, solve n · x = |J⁻ᵀn| (n · v) for both,
    // σ = |J⁻ᵀn| = sqrt(nᵀK⁻¹n); their difference, 4cs q / (σ₁ + σ₂),
    // doesn't cancel as the half-angle closes.
    let bb = dot(b, inv * b);
    let ee = dot(e, inv * e);
    let q = dot(e, inv * b);
    let sum = max(sqrt(max(c * c * bb + s * s * ee + 2.0 * c * s * q, 0.0))
        + sqrt(max(c * c * bb + s * s * ee - 2.0 * c * s * q, 0.0)), 1.0e-30);
    // Where the screen's angle is much sharper than the data's, the join
    // the layout chose as a miter would reach far past the bytes' limit
    // (`orb_layout::line`, about twice the half width): it stops there.
    let exact = b * (0.5 * sum) + e * (2.0 * c * c * q / sum);
    let x = exact * min(1.0, 2.0 / max(miter * sqrt(max(dot(exact, k * exact), 0.0)), 1.0e-30));
    let kept = b * (2.0 * s * s * q / sum) + e * (0.5 * sum);
    let along = e * inverseSqrt(max(dot(e, k * e), 1.0e-30));
    let y = mix(along, kept, blend);
    return mat2x2<f32>(x * b.x + y * e.x, x * b.y + y * e.y) * world_per_px;
}

fn line_vertex(v: Pulled) -> StyledLineOut {
    var out: StyledLineOut;
    let tile = tile_at(v.tile);
    let pos = unpack_i16x2(vertex_at(v.vertex));
    let data = vertex_at(v.vertex + 1u);
    let feature = vertex_at(v.vertex + 3u);
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);

    var color = node.color;
    if (node.table_mask & 1u) != 0u {
        color = table_color(row, 0u, node.curves[0], tile.zoom);
    }
    if from_state(state, node.state, 0u) {
        color = state_color(state, node.state, 0u, node.curves[0], color);
    }
    let opacity = line_prop(row, state, 1u, node.opacity, tile.zoom);
    if opacity < 0.01 {
        out.position = discarded_vertex();
        return out;
    }
    let width = line_prop(row, state, 2u, node.width, tile.zoom);
    let gap_width = line_prop(row, state, 3u, node.gap_width, tile.zoom) / 2.0;
    let offset = -line_prop(row, state, 4u, node.offset, tile.zoom);
    let blur = line_prop(row, state, 5u, node.blur, tile.zoom);

    let extrude = vec2<f32>(signed_byte(data, 0u), signed_byte(data, 8u));
    let offset_normal = vec2<f32>(signed_byte(data, 16u), signed_byte(data, 24u));
    let flags = vertex_at(v.vertex + 2u);
    let normal = vec2<f32>(f32(flags & 1u), f32((flags >> 1u) & 1u) * 2.0 - 1.0);

    // Room for the edges' fade (`line_alpha`, a device pixel wide): half
    // a device pixel, so that the line's coverage sums to its width. In a
    // drape slot the fade is a texel wide, and a texel as many pixels as
    // the zoom makes it: half a texel.
    let antialiasing = select(1.0 / frame.pixel_ratio, 1.0 / tile.px_scale, draped_placement(tile)) / 2.0;
    let half_width = width / 2.0;
    var inset = gap_width;
    var outset = gap_width + half_width;
    if gap_width > 0.0 {
        inset += antialiasing;
        outset += half_width;
    }
    if half_width != 0.0 {
        outset += antialiasing;
    }

    let has_offset_normal = any(offset_normal != vec2<f32>(0.0));
    let px_to_data = line_px_to_data(
        tile,
        pos,
        select(extrude, offset_normal, has_offset_normal),
        select(1.0, length(offset_normal) * EXTRUDE_SCALE, has_offset_normal),
    );
    let dist = outset * extrude * EXTRUDE_SCALE;
    let offset2 = offset * offset_normal * EXTRUDE_SCALE * normal.y;

    let px = node_translate(node.translate, node.table_mask, row, v.row_words, node.translate_curve, tile.zoom);
    let shift = translate_world(px, node.translate_anchor, tile);
    let center = pos + px_to_data * offset2 + vec2<f32>(shift.x, -shift.y) / tile.scale;
    let projected_no_extrude = ground_clip(tile_ground(tile, center), vec2<f32>(0.0), 0.0);
    let extruded = center + px_to_data * dist;
    let ground = tile_ground(tile, extruded);
    out.position = ground_clip(ground, vec2<f32>(0.0), 0.0);

    // How much perspective squishes or stretches the extrusion, after both
    // tile positions have been projected (as in GL JS).
    let units_to_px = vec2<f32>(frame.viewport.x / 2.0, -frame.viewport.y / 2.0);
    let projected_extrude = out.position.xy - projected_no_extrude.xy;
    let with_perspective = length(projected_extrude / out.position.w * units_to_px);
    out.gamma_scale = length(dist) / max(with_perspective, 1e-6);

    out.color = color_in(color) * opacity;
    out.opacity = opacity;
    out.linesofar = f32(vertex_at(v.vertex + 2u) >> 2u) * 2.0 * tile.extent / 8192.0;
    out.floorwidth = node.pattern.floor_width;
    if (node.table_mask & 4u) != 0u {
        out.floorwidth = bitcast<f32>(prop_at(row + 4u + select(0u, 1u, floor(frame.zoom) > tile.zoom)));
    }
    out.images = pattern_ids(row, LINE_ROW_PATTERN, node.image_from, node.image_to, node.table_mask, node.pattern);
    if node.metrics != 0u { out.progress = bitcast<f32>(vertex_at(v.vertex + 4u)); }
    if node.gradient != 0u { out.ramp = prop_at(row + LINE_ROW_GRADIENT); }
    if node.dashed != 0u {
        out.images = vec2<u32>(node.dash_from, node.dash_to);
        if (node.table_mask & 128u) != 0u {
            out.images = vec2<u32>(prop_at(row + LINE_ROW_DASH + select(2u, 0u, node.dash_from_scale > 1.0)), prop_at(row + LINE_ROW_DASH + 1u));
        }
        out.floorwidth = node.dash_floor_width;
        if (node.table_mask & 4u) != 0u {
            out.floorwidth = bitcast<f32>(prop_at(row + LINE_ROW_FLOOR_WIDTH));
        }
    }
    out.normal = normal;
    out.width2 = vec2<f32>(outset, inset);
    out.blur = blur;
    out.tile_pos = extruded;
    out.owner_pos = select(extruded, extruded - vec2<f32>(shift.x, -shift.y) / tile.scale, draped_placement(tile));
    out.tile = v.tile;
    out.clip = fragment_clip(tile, any(shift != vec2<f32>(0.0)));
    return out;
}

// A line's coverage of a fragment, antialiased across its edges.
fn line_alpha(normal: vec2<f32>, width2: vec2<f32>, blur: f32, gamma_scale: f32) -> f32 {
    let dist = length(normal) * width2.x;
    let blur2 = (blur + 1.0 / frame.pixel_ratio) * gamma_scale;
    return clamp(min(dist - (width2.y - blur2), width2.x - dist) / blur2, 0.0, 1.0);
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    if !clip_drawn(in.clip, in.tile, in.owner_pos, in.tile_pos) {
        discard;
    }
    return in.color * line_alpha(in.normal, in.width2, in.blur, in.gamma_scale);
}

@fragment
fn fs_line_styled(in: StyledLineOut) -> @location(0) vec4<f32> {
    if !clip_drawn(in.clip, in.tile, in.owner_pos, in.tile_pos) {
        discard;
    }
    var alpha = line_alpha(in.normal, in.width2, in.blur, in.gamma_scale);
    if node.pattern.enabled != 0u {
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
        return color_in(mix(ca, cb, node.pattern.fade)) * alpha * in.opacity;
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
        return color_in(gradient_sample(image_at(in.ramp), in.progress)) * alpha * in.opacity;
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
