// Extrusions, lit by v8's `light` (lighting.wgsl), ported from
// MapLibre GL JS (BSD-3-Clause).
struct ExtrusionNode {
    color: vec4<f32>,
    height: f32,
    base: f32,
    index: u32,
    table_mask: u32,
    translate: vec2<f32>,
    translate_anchor: u32,
    vertical_gradient: u32,
    opacity: f32,
    image_from: u32, image_to: u32,
    // style-spec#1796's ambient occlusion: the walls' intensity and radius
    // (metres up from the ground), then the ground's (metres out from the
    // walls, extrusion_ground.wgsl). An intensity of 0 is none.
    occlusion: vec4<f32>,
    // Color, height, base, and per-feature opacity.
    curves: array<Curve, 4>,
    translate_curve: Curve,
    pattern: Pattern,
    state: StateLookup,
    elevation: ElevationColor,
};
// `fill-extrusion-color` over `["elevation"]`, the height above the ground,
// drawn per fragment by pipelines specialised for its shape: the elevation
// through up to four operators (codes in `elevation_input`, constants in
// `input`) finds its place among the `elevation_stops` stops' `labels`
// (the largest float past them), and the colours there mix by `curve`, or
// step. Stop i's colour is property slot 0 for the first, 2 + i after:
// from the feature's state, its row where bit i of `rows` is set (the
// row's words after the family's for i > 0), else `colors`.
struct ElevationColor {
    rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    curve: Curve,
    input: vec4<f32>,
    labels: array<vec4<f32>, 4>,
    colors: array<vec4<u32>, 4>,
};
// A pipeline holds only the code its curve runs: on Apple GPUs, one
// shader for every shape (any operator, curve or source of stops) drew a
// 2048 x 1536 frame of Liberty's buildings 0.5 ms slower than plain ones,
// where pipelines specialised for a two-stop ramp take 0.04 ms more. The
// number of stops, 0 for none.
override elevation_stops: u32 = 0u;
// The input operators' codes plus one, five bits each, innermost first.
override elevation_input: u32 = 0u;
// How the colours mix between stops: 0, they step; 1, linearly; 2,
// exponentially; 3, by a cubic bezier.
override elevation_curve: u32 = 0u;
// Where stops may come from beside the node: bit 0, features' rows; bit 1,
// their state.
override elevation_dynamic: u32 = 0u;
@group(1) @binding(0) var<uniform> node: ExtrusionNode;
// `table_mask` bits beyond the row slots: opacity and the vertical gradient
// per feature, from the row's flags word.
const EXTRUSION_OPACITY_ROWS: u32 = 8u;
const EXTRUSION_GRADIENT_ROWS: u32 = 16u;
// A colour over the elevation is lit per fragment: it keeps how far the
// face turns to the light, the vertical gradient, the opacity and how far
// rows blend toward the next zoom in `color`, the height above the ground
// in `pattern_pos`, and the feature's row and state block in `images`.
struct ExtrusionOut {
    @builtin(position) position: vec4<f32>,
    //planes @builtin(clip_distances) clip: array<f32, 4>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(2) @interpolate(flat) tile: u32,
    @location(3) pattern_pos: vec2<f32>,
    @location(4) @interpolate(flat) images: vec2<u32>,
    // A wall's height above the ground, metres, for its ambient occlusion;
    // EXTRUSION_ROOF_ABOVE on roofs, which it leaves alone. On terrain, its
    // elevation, less the DEM's at the fragment (`dem` and `dem_layer`,
    // `ground_dem_at`'s; NO_GROUND_DEM off terrain).
    @location(5) above: f32,
    @location(6) dem: vec2<f32>,
    @location(7) @interpolate(flat) dem_layer: u32,
    // A wall's share of its light that is ambient, which its ambient
    // occlusion dims (`light_ambient_share`).
    @location(8) @interpolate(flat) ambient: f32,
};
// Extrusions draw in two pipelines: whole buildings, and buildings cut at
// the source's buffer, clipped per fragment to the placement. Only the
// second discards, which would turn off the GPU's hidden-surface removal
// for every building it drew: a quarter of a dense 3D frame.
@vertex
fn vs_extrusion(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, 1u << EXTRUSION_WHOLE);
}
// Every cut building, clipped per fragment: where clip distances aren't.
@vertex
fn vs_extrusion_cut(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, (1u << EXTRUSION_CUT) | (1u << EXTRUSION_CUT_SPLIT));
}
// Where they are (`CLIP_PLANES`): cut buildings of a placement with one clip
// rectangle, clipped by its four planes, which needs no discard...
@vertex
fn vs_extrusion_cut_planes(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, 1u << EXTRUSION_CUT);
}
// ...and those of a placement with several, clipped per fragment.
@vertex
fn vs_extrusion_cut_split(@builtin(vertex_index) index: u32) -> ExtrusionOut {
    return extrusion_part(index, 1u << EXTRUSION_CUT_SPLIT);
}
const EXTRUSION_WHOLE: u32 = 1u;
const EXTRUSION_CUT: u32 = 2u;
const EXTRUSION_CUT_SPLIT: u32 = 3u;
// Vertex `index` if its building is drawn here as one of `parts` (bits).
fn extrusion_part(index: u32, parts: u32) -> ExtrusionOut {
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) || (parts & (1u << extrusion_drawn(v))) == 0u {
        var out: ExtrusionOut;
        out.position = discarded_vertex();
        return out;
    }
    var out = extrusion_vertex(v);
    //planes out.clip = extrusion_planes(tile_at(v.tile), out.tile_pos);
    return out;
}
// The distances inside a placement's one clip rectangle from position `p`,
// for clip planes: both bounds count, as `extrusion_inside`'s do.
fn extrusion_planes(tile: Tile, p: vec2<f32>) -> array<f32, 4> {
    var lo = vec2<f32>(0.0);
    var hi = vec2<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let l = unpack_i16x2(tile.clips[i].x);
        let h = unpack_i16x2(tile.clips[i].y);
        if all(l < h) {
            lo = l;
            hi = h;
        }
    }
    return array<f32, 4>(p.x - lo.x, hi.x - p.x, p.y - lo.y, hi.y - p.y);
}
// Whether a placement clips to more than one rectangle.
fn clips_split(tile: Tile) -> bool {
    var count = 0u;
    for (var i = 0u; i < 4u; i++) {
        count += u32(all(unpack_i16x2(tile.clips[i].x) < unpack_i16x2(tile.clips[i].y)));
    }
    return count > 1u;
}
// How a building draws from this placement: 0 not at all, EXTRUSION_WHOLE,
// or cut: EXTRUSION_CUT, or EXTRUSION_CUT_SPLIT where the placement clips to
// several rectangles. A whole building belongs to the placement holding its
// centroid (half open, as points are owned), which draws all of it, past
// its clip rectangles: its walls and their pattern run on unbroken. Every
// other placement leaves it to the tile around it holding its centroid
// where that tile is placed alone over its whole square (`neighbours`),
// which then owns it and keeps it whole too; where that tile is drawn at
// another zoom or not at all, a placement draws its own part, as it draws
// a building cut at the source's buffer.
fn extrusion_drawn(v: Pulled) -> u32 {
    let row = prop_row(v, vertex_at(v.vertex + 1u), v.row_words);
    let tile = tile_at(v.tile);
    let cut = select(EXTRUSION_CUT, EXTRUSION_CUT_SPLIT, clips_split(tile));
    if (prop_at(row + EXTRUSION_ROW_WHOLE) & 1u) == 0u {
        return cut;
    }
    let centroid = unpack_i16x2(prop_at(row + EXTRUSION_ROW_CENTROID));
    if inside_clips(tile, centroid) {
        return EXTRUSION_WHOLE;
    }
    let cell = floor(centroid / tile.extent) + 1.0;
    if all(cell >= vec2<f32>(0.0)) && all(cell <= vec2<f32>(2.0))
        && ((tile.neighbours >> u32(3.0 * cell.y + cell.x)) & 1u) != 0u {
        return 0u;
    }
    return cut;
}
// Whether a cut building's fragment at `tile_pos` lies in its placement.
// Shared tile edges can contain real vertical walls: they have zero ground
// area, and depth selects their nearest surface without blending twice, so
// both bounds count.
fn extrusion_inside(tile: Tile, tile_pos: vec2<f32>) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        if all(tile_pos >= lo) && all(tile_pos <= hi) {
            return true;
        }
    }
    return false;
}
// Overlapping parts of one building in OSM (a podium and the tower on it,
// both from the ground) have coplanar walls, which would z-fight: which
// face shows would flicker with the camera. Each feature's depth instead
// sits whole steps farther than taller ones', so the same face always
// wins, the tallest part's, whose wall runs on unbroken: 32 levels of
// height on a log scale, and within one, 4 by the feature's row. A step is
// a relative depth of 2^-18: at 2^-19 two triangulations of one wall still
// crossed in Seattle's towers. The farthest level moves a surface back by
// 0.05% of its distance, about a pixel's footprint, and never past the
// near plane.
const EXTRUSION_DEPTH_STEP: f32 = 3.814697265625e-6;
fn extrusion_depth_scale(height: f32, feature: u32) -> f32 {
    let tall = min(u32(log2(max(height, 0.0) + 1.0) * 3.1), 31u);
    return 1.0 - f32((31u - tall) * 4u + feature % 4u) * EXTRUSION_DEPTH_STEP;
}
// A roof's height above the ground for ambient occlusion: beyond any
// radius, so it stays unshaded.
const EXTRUSION_ROOF_ABOVE: f32 = 1e30;
// A feature's base, height and opacity, from the node, its row (its
// opacity, beyond the spec, packed beside the whole-building bit) or its
// state.
fn extrusion_feature(tile: Tile, row: u32, state: u32) -> vec3<f32> {
    var height = node.height;
    var base = node.base;
    if (node.table_mask & 2u) != 0u { height = table_number(row, 1u, node.curves[1], tile.zoom); }
    if (node.table_mask & 4u) != 0u { base = table_number(row, 2u, node.curves[2], tile.zoom); }
    if from_state(state, node.state, 1u) { height = state_number(state, node.state, 1u, node.curves[1], height); }
    if from_state(state, node.state, 2u) { base = state_number(state, node.state, 2u, node.curves[2], base); }
    var opacity = 1.0;
    if (node.table_mask & EXTRUSION_OPACITY_ROWS) != 0u {
        let flags = prop_at(row + EXTRUSION_ROW_WHOLE);
        let at = vec2<f32>(f32((flags >> 8u) & 255u), f32((flags >> 16u) & 255u)) / 255.0;
        opacity = mix(at.x, at.y, curve_t(node.curves[3], tile.zoom));
    }
    return vec3<f32>(base, height, opacity);
}
fn extrusion_vertex(v: Pulled) -> ExtrusionOut {
    var out: ExtrusionOut;
    let tile = tile_at(v.tile);
    let pos = unpack_i16x2(vertex_at(v.vertex));
    let feature = vertex_at(v.vertex+1u);
    let row = prop_row(v, feature, v.row_words);
    let state = state_block(v, feature, node.state);
    var color = node.color;
    if (node.table_mask & 1u) != 0u { color = table_color(row, 0u, node.curves[0], tile.zoom); }
    if from_state(state, node.state, 0u) { color = state_color(state, node.state, 0u, node.curves[0], color); }
    let part = extrusion_feature(tile, row, state);
    var base = part.x;
    var height = part.y;
    let opacity = part.z;
    // The target keeps the nearest surface per pixel: a building at zero
    // would hide what stands behind it.
    if opacity <= 0.0 {
        out.position = discarded_vertex();
        return out;
    }
    let flags = prop_at(row + EXTRUSION_ROW_WHOLE);
    var vertical_gradient = node.vertical_gradient != 0u;
    if (node.table_mask & EXTRUSION_GRADIENT_ROWS) != 0u {
        vertical_gradient = (flags & 2u) != 0u;
    }
    // On terrain, raise the roof by the ground under the feature's
    // centroid and give ground-level parts a 10 m basement,
    // so buildings don't hang over slopes (GL JS's `a_centroid`).
    let gradient_base = max(0.0, base);
    let gradient_height = max(0.0, height);
    let centroid = unpack_i16x2(prop_at(row + EXTRUSION_ROW_CENTROID));
    let raise = ground_elevation(tile, centroid);
    let basement = select(0.0, 10.0, base <= 0.0 && frame.dem_dim > 0.0);
    height = gradient_height + raise;
    base = gradient_base + raise - basement;
    let packed = vertex_at(v.vertex+2u);
    let top = f32(packed & 1u);
    let normal = vec3<f32>(unpack_i16x2(packed), unpack_i16x2(vertex_at(v.vertex+3u)).x) / select(16384.0, 16383.0, node.pattern.enabled != 0u);
    let elevation = mix(base, height, top);
    let px = node_translate(node.translate, node.table_mask, row, v.row_words, node.translate_curve, tile.zoom);
    let shift = translate_world(px, node.translate_anchor, tile);
    let ground = tile_ground_shifted(tile, pos, shift);
    out.position = ground_clip_reversed(ground, vec2<f32>(0.0), elevation * frame.extrusion_scale);
    out.position.z *= extrusion_depth_scale(gradient_height, feature);
    out.tile_pos = pos;
    out.tile = v.tile;
    out.above = select(EXTRUSION_ROOF_ABOVE, mix(gradient_base - basement, gradient_height, top), normal.z == 0.0);
    // On terrain a wall's foot is where it meets the ground, which the DEM
    // shapes between the wall's ends (a street's crown, a pit dug beside
    // it): a wall measures its height above the ground per fragment.
    out.dem_layer = NO_GROUND_DEM;
    if normal.z == 0.0 && node.occlusion.x > 0.0 && frame.dem_dim > 0.0 {
        let at = ground_dem_at(tile, pos);
        if at.layer != NO_GROUND_DEM {
            out.above = elevation;
            out.dem = at.coord;
            out.dem_layer = at.layer;
        }
    }
    var gradient = 1.0;
    if normal.z == 0.0 && vertical_gradient {
        gradient = light_wall_gradient(top + gradient_base, gradient_height);
    }
    // Patterns and colours over the elevation are lit as mid luminance.
    var luminance = 0.5;
    if node.pattern.enabled == 0u && elevation_stops == 0u {
        luminance = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    out.ambient = 1.0;
    if normal.z == 0.0 && node.occlusion.x > 0.0 {
        out.ambient = light_ambient_share(light_facing(normal), luminance);
    }
    if node.pattern.enabled != 0u {
        out.images = pattern_ids(row, EXTRUSION_ROW_PATTERN, node.image_from, node.image_to, node.table_mask, node.pattern);
        var pattern_pos = pos;
        if normal.z == 0.0 {
            let distance = f32(vertex_at(v.vertex+3u) >> 16u);
            pattern_pos = vec2<f32>(distance, -elevation * exp2(floor(frame.zoom)) / 65536.0 / tile_pixel_ratio(tile));
        }
        // Keep atlas reads in the fragment stage, within core WebGPU's eight
        // storage slots per stage.
        out.pattern_pos = pattern_pos;
        // The pattern's texels multiply the light, as GL JS shades them at
        // mid luminance.
        out.color = vec4<f32>(clamp(light_factor(normal, 0.5, gradient), light_floor(), vec3<f32>(1.0)), 1.0) * opacity;
        return out;
    }
    if elevation_stops > 0u {
        out.color = vec4<f32>(light_facing(normal), gradient, opacity, curve_t(node.curves[0], tile.zoom));
        out.pattern_pos = vec2<f32>(mix(gradient_base - basement, gradient_height, top), 0.0);
        out.images = vec2<u32>(row, state);
        return out;
    }
    out.color = color_in(vec4<f32>(light_surface(color.rgb, normal, gradient), 1.0) * opacity);
    return out;
}
// The elevation `v` through input operator `i`. The curve's uniforms are
// read at constant indices only, which the GPU keeps in registers: an index
// that varies reads memory for every fragment.
fn elevation_op(i: u32, v: f32) -> f32 {
    let c = node.elevation.input[i];
    switch (elevation_input >> (5u * i)) & 31u {
        case 0u: { return v; }
        case 1u: { return v + c; }
        case 2u: { return v * c; }
        case 3u: { return v - c; }
        case 4u: { return c - v; }
        case 5u: { return v / c; }
        case 6u: { return v % c; }
        case 7u: { return min(v, c); }
        case 8u: { return max(v, c); }
        case 9u: { return abs(v); }
        case 10u: { return floor(v); }
        case 11u: { return ceil(v); }
        // Halves away from zero, as GL JS rounds.
        default: { return sign(v) * floor(abs(v) + 0.5); }
    }
}
// Stop `i`'s colour for the fragment `in`, whose feature's state or row
// it may come from, else `color`, the node's, packed.
fn elevation_stop(i: u32, color: u32, in: ExtrusionOut) -> vec4<f32> {
    let slot = select(2u + i, 0u, i == 0u);
    var value = unpack4x8unorm(color);
    if (elevation_dynamic & 1u) != 0u && (node.elevation.rows & (1u << i)) != 0u {
        let word = in.images.x + select(EXTRUSION_ROW_WORDS + 2u * (i - 1u), 0u, i == 0u);
        value = mix(unpack4x8unorm(prop_at(word)), unpack4x8unorm(prop_at(word + 1u)), in.color.w);
    }
    if (elevation_dynamic & 2u) != 0u && from_state(in.images.y, node.state, slot) {
        value = state_color(in.images.y, node.state, slot, node.curves[0], value);
    }
    return value;
}
// GL JS's `interpolationFactor` of `x` between stops `lower` and `upper`,
// computed so that a large exponent stays in range.
fn elevation_factor(x: f32, lower: f32, upper: f32) -> f32 {
    let d = upper - lower;
    let p = x - lower;
    let c = node.elevation.curve;
    switch elevation_curve {
        case 2u: {
            let base = c.params.x;
            if base > 1.0 {
                let r = pow(base, -d);
                return (pow(base, p - d) - r) / (1.0 - r);
            }
            return (pow(base, p) - 1.0) / (pow(base, d) - 1.0);
        }
        case 3u: { return unit_bezier(c.params, p / d); }
        default: { return p / d; }
    }
}
// What of the stops lies at or below an input: their count, the greatest
// label and its packed colour; and the least label above, if any.
struct ElevationSpan {
    below: u32,
    lower: f32,
    lower_color: u32,
    above: bool,
    upper: f32,
    upper_color: u32,
};
// `span` with four more stops, which follow its own, compared to `x`.
fn elevation_span(span: ElevationSpan, labels: vec4<f32>, colors: vec4<u32>, x: f32) -> ElevationSpan {
    let n = u32(dot(select(vec4<f32>(0.0), vec4<f32>(1.0), labels <= vec4<f32>(x)), vec4<f32>(1.0)));
    var out = span;
    out.below += n;
    // Labels ascend: the greatest at or below is the nth, the least above
    // the next.
    if n > 0u {
        out.lower = select(select(select(labels.x, labels.y, n > 1u), labels.z, n > 2u), labels.w, n > 3u);
        out.lower_color = select(select(select(colors.x, colors.y, n > 1u), colors.z, n > 2u), colors.w, n > 3u);
    }
    if n < 4u && !span.above {
        out.above = true;
        out.upper = select(select(select(labels.w, labels.z, n < 3u), labels.y, n < 2u), labels.x, n < 1u);
        out.upper_color = select(select(select(colors.w, colors.z, n < 3u), colors.y, n < 2u), colors.x, n < 1u);
    }
    return out;
}
// A colour over the elevation at the fragment, lit.
fn elevation_shade(in: ExtrusionOut) -> vec4<f32> {
    let x = elevation_op(3u, elevation_op(2u, elevation_op(1u, elevation_op(0u, in.pattern_pos.x))));
    let labels = node.elevation.labels;
    let colors = node.elevation.colors;
    var span = ElevationSpan(0u, labels[0].x, colors[0].x, false, 0.0, 0u);
    span = elevation_span(span, labels[0], colors[0], x);
    if elevation_stops > 4u {
        span = elevation_span(span, labels[1], colors[1], x);
    }
    if elevation_stops > 8u {
        span = elevation_span(span, labels[2], colors[2], x);
    }
    if elevation_stops > 12u {
        span = elevation_span(span, labels[3], colors[3], x);
    }
    let lower = max(span.below, 1u) - 1u;
    var color = elevation_stop(lower, span.lower_color, in);
    if elevation_curve != 0u && span.below > 0u && span.below < elevation_stops {
        let t = elevation_factor(x, span.lower, span.upper);
        color = mix(color, elevation_stop(lower + 1u, span.upper_color, in), t);
    }
    return color_in(vec4<f32>(light_lit(color.rgb, in.color.x, in.color.y), 1.0) * in.color.z);
}
// The fragment's colour, its wall darkened toward the ground by
// style-spec#1796's ambient occlusion as MapLibre Native's walls PR (#4909)
// shades it: by the intensity at the foot, fading out over the radius, of
// the ambient share of its light. On terrain, from the ground under the
// fragment, wherever along the wall the terrain meets it.
fn extrusion_shade(in: ExtrusionOut) -> vec4<f32> {
    let color = extrusion_color(in);
    let occlusion = node.occlusion;
    if occlusion.x <= 0.0 || occlusion.y <= 0.0 {
        return color;
    }
    var above = in.above;
    if in.dem_layer != NO_GROUND_DEM {
        above -= dem_bilinear(DemAt(in.dem, in.dem_layer));
    }
    let lit = 1.0 - occlusion.x * (1.0 - smoothstep(0.0, occlusion.y, above)) * in.ambient;
    return vec4<f32>(color.rgb * lit, color.a);
}
fn extrusion_color(in: ExtrusionOut) -> vec4<f32> {
    if node.pattern.enabled == 0u {
        if elevation_stops > 0u {
            return elevation_shade(in);
        }
        return in.color;
    }
    let tile = tile_at(in.tile);
    let image_a = image_at(in.images.x); let image_b = image_at(in.images.y);
    let pos_a = pattern_position(tile, in.pattern_pos, image_a, node.pattern.from_scale);
    let pos_b = pattern_position(tile, in.pattern_pos, image_b, node.pattern.to_scale);
    let a = pattern_sample(image_a, fract(pos_a), 0.0);
    let b = pattern_sample(image_b, fract(pos_b), 0.0);
    return color_in(mix(a, b, node.pattern.fade) * in.color);
}
