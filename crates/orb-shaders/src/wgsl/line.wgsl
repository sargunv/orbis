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
// Its position is invariant: compiled beside fetches from storage buffers
// or from data textures (WebGL 2's driver), the same words make the same
// position, and both drivers draw the same pixels.
struct StyledLineOut {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) @interpolate(flat, either) color: vec4<f32>,
    @location(1) normal: vec2<f32>,
    // (outset, inset) in pixels.
    @location(2) @interpolate(flat, either) width2: vec2<f32>,
    @location(3) gamma_scale: f32,
    @location(4) @interpolate(flat, either) blur: f32,
    @location(5) tile_pos: vec2<f32>,
    @location(6) @interpolate(flat, either) tile: u32,
    // Where the placement owns the line: where it lands, but where it was
    // in a drape slot, so a translated line moves whole into it, bounded
    // by the slot where it lands (`tile_pos`).
    @location(7) owner_pos: vec2<f32>,
    @location(8) @interpolate(flat, either) clip: vec4<f32>,
    //planes @builtin(clip_distances) planes: array<f32, 4>,
    @location(9) linesofar: f32,
    @location(10) @interpolate(flat, either) floorwidth: f32,
    @location(11) @interpolate(flat, either) images: vec2<u32>,
    @location(12) @interpolate(flat, either) opacity: f32,
    @location(13) progress: f32,
    @location(14) @interpolate(flat, either) ramp: u32,
};

// A plain line's (`vs_line`): the styled line's first nine. Every vertex
// output is stored and every fragment interpolates it: on a pitched city
// view the styled ones cost plain lines a tenth of their main pass.
// What a feature holds constant (colour, widths, blur) rides flat, which
// fragments read without interpolating: 4% of the moto g's main pass
// across views (#180).
struct LineOut {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) @interpolate(flat, either) color: vec4<f32>,
    @location(1) normal: vec2<f32>,
    @location(2) @interpolate(flat, either) width2: vec2<f32>,
    @location(3) gamma_scale: f32,
    @location(4) @interpolate(flat, either) blur: f32,
    @location(5) tile_pos: vec2<f32>,
    @location(6) @interpolate(flat, either) tile: u32,
    @location(7) owner_pos: vec2<f32>,
    @location(8) @interpolate(flat, either) clip: vec4<f32>,
    //planes @builtin(clip_distances) planes: array<f32, 4>,
};

fn line_prop(row: u32, state: u32, bit: u32, constant: f32, tile_zoom: f32) -> f32 {
    var value = constant;
    if (row_mask(node.table_mask) & (1u << bit)) != 0u {
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

// Where clip distances are (`CLIP_PLANES`, orb-render's line planes
// pipelines), placements with one clip rectangle clip by its four
// planes, and their fragments test nothing (`clip_fragments`). Element by
// element, as WebKit's Metal translation takes them (`extrusion.wgsl`).
@vertex
fn vs_line(@builtin(vertex_index) index: u32) -> LineOut {
    let s = pulled_line(index);
    var out: LineOut;
    out.position = s.position;
    out.color = s.color;
    out.normal = s.normal;
    out.width2 = s.width2;
    out.gamma_scale = s.gamma_scale;
    out.blur = s.blur;
    out.tile_pos = s.tile_pos;
    out.tile = s.tile;
    out.owner_pos = s.owner_pos;
    out.clip = s.clip;
    //planes let planes = clip_planes(s.clip, s.tile_pos);
    //planes out.planes[0] = planes.x;
    //planes out.planes[1] = planes.y;
    //planes out.planes[2] = planes.z;
    //planes out.planes[3] = planes.w;
    return out;
}

@vertex
fn vs_line_styled(@builtin(vertex_index) index: u32) -> StyledLineOut {
    var out = pulled_line(index);
    //planes let planes = clip_planes(out.clip, out.tile_pos);
    //planes out.planes[0] = planes.x;
    //planes out.planes[1] = planes.y;
    //planes out.planes[2] = planes.z;
    //planes out.planes[3] = planes.w;
    return out;
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
    if !warped(tile) && (!on_globe() || tile.globe_cells == 0u) {
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
    if (row_mask(node.table_mask) & 1u) != 0u {
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

    let px = node_translate(node.translate, row_mask(node.table_mask), row, v.row_words, node.translate_curve, tile.zoom);
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

    out.color = color * opacity;
    out.opacity = opacity;
    out.linesofar = f32(vertex_at(v.vertex + 2u) >> 2u) * 2.0 * tile.extent / 8192.0;
    out.floorwidth = node.pattern.floor_width;
    // A per-feature width's rows, as the camera scales them.
    if (row_mask(node.table_mask) & 4u) != 0u {
        out.floorwidth = camera_scaled(bitcast<f32>(prop_at(row + 4u + select(0u, 1u, floor(frame.zoom) > tile.zoom))), node.curves[2]);
    }
    out.images = pattern_ids(row, LINE_ROW_PATTERN, node.image_from, node.image_to, row_mask(node.table_mask), node.pattern);
    if node.metrics != 0u { out.progress = bitcast<f32>(vertex_at(v.vertex + 4u)); }
    if feature_rows && node.gradient != 0u { out.ramp = prop_at(row + LINE_ROW_GRADIENT); }
    if node.dashed != 0u {
        out.images = vec2<u32>(node.dash_from, node.dash_to);
        if (row_mask(node.table_mask) & 128u) != 0u {
            out.images = vec2<u32>(prop_at(row + LINE_ROW_DASH + select(2u, 0u, node.dash_from_scale > 1.0)), prop_at(row + LINE_ROW_DASH + 1u));
        }
        out.floorwidth = node.dash_floor_width;
        if (row_mask(node.table_mask) & 4u) != 0u {
            out.floorwidth = camera_scaled(bitcast<f32>(prop_at(row + LINE_ROW_FLOOR_WIDTH)), node.curves[2]);
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
