// Line labels' variant of the symbol shader: each glyph placed along its
// projected line in the vertex stage. Point labels compose
// `symbol_point.wgsl` instead, so their pipeline doesn't compile this:
// Adreno compilers inline every call.

const LINE_LABELS: bool = true;

// A tile point in the label plane (GL JS
// `projectTileCoordinatesToLabelPlane`), and its clip w.
struct PlanePoint {
    p: vec2<f32>,
    w: f32,
};

fn to_label_plane(tile: Tile, p: vec2<f32>) -> PlanePoint {
    if (node.flags & PITCH_WITH_MAP) != 0u {
        return PlanePoint(to_pitched_plane(tile, p), 1.0);
    }
    let c = tile_clip(tile, p);
    return PlanePoint(clip_to_px(c), c.w);
}

// A label-plane point in clip space (GL JS
// `projectFromLabelPlaneToClipSpace`), for orientation tests.
fn label_plane_to_clip(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    if (node.flags & PITCH_WITH_MAP) != 0u {
        let c = tile_clip(tile, from_pitched_plane(tile, p));
        return c.xy / c.w;
    }
    return px_to_clip(p);
}

fn line_vertex(base: u32, placed: Placed, i: u32) -> vec2<f32> {
    return unpack_i16x2(vertex_at(base + vertex_at(base + H_LINES) + placed.line_start + i));
}

// A line vertex in the label plane; one behind the camera becomes a point
// far enough along the line (GL JS `projectLineVertexToLabelPlane`).
// One projection call site, in a loop (Adreno compilers inline every
// call): the vertex, then, behind the camera, a point a unit back along the
// line.
fn project_line_vertex(tile: Tile, base: u32, placed: Placed, i: u32, previous_tile: vec2<f32>, previous: vec2<f32>, needed: f32) -> vec2<f32> {
    let v = line_vertex(base, placed, i);
    var point = v;
    var behind = false;
    var q: PlanePoint;
    loop {
        q = to_label_plane(tile, point);
        if q.w > 0.0 || behind {
            break;
        }
        behind = true;
        point = previous_tile + normalize(previous_tile - v);
    }
    if !behind {
        return q.p;
    }
    let segment = previous - q.p;
    return previous + segment * (needed / length(segment));
}

fn perp_normal(v: vec2<f32>, offset: f32, direction: f32) -> vec2<f32> {
    let u = normalize(v);
    return vec2<f32>(-u.y, u.x) * offset * direction;
}

// GL JS `isLineDoublingBack`: the line turns straight back at vertex `i`
// (tile coordinates are integers, so the test is exact).
fn doubles_back(base: u32, placed: Placed, i: i32, direction: i32) -> bool {
    let previous = i - direction;
    let next = i + direction;
    if min(previous, next) < 0 || max(previous, next) >= i32(placed.line_count) {
        return false;
    }
    let p = vec2<i32>(line_vertex(base, placed, u32(previous)));
    let c = vec2<i32>(line_vertex(base, placed, u32(i)));
    let n = vec2<i32>(line_vertex(base, placed, u32(next)));
    let a = c - p;
    let b = n - c;
    return a.x * b.y - a.y * b.x == 0 && a.x * b.x + a.y * b.y < 0;
}

// GL JS `findLineIntersection` of the lines through `a1 a2` and `b1 b2`;
// `a2` where they are parallel.
fn line_intersection(a1: vec2<f32>, a2: vec2<f32>, b1: vec2<f32>, b2: vec2<f32>) -> vec2<f32> {
    let denominator = (b2.y - b1.y) * (a2.x - a1.x) - (b2.x - b1.x) * (a2.y - a1.y);
    if denominator == 0.0 {
        return a2;
    }
    let ua = ((b2.x - b1.x) * (a1.y - b1.y) - (b2.y - b1.y) * (a1.x - b1.x)) / denominator;
    return a1 + ua * (a2 - a1);
}

// GL JS `findOffsetIntersectionPoint`: where the offset segment into vertex
// `i` meets the offset segment out of it (a miter join); `next` is vertex
// `i + direction` in the label plane, when the line has it.
fn offset_intersection(base: u32, placed: Placed, i: i32, normal: vec2<f32>, current: vec2<f32>, offset_previous: vec2<f32>, offset_y: f32, direction: f32, next: vec2<f32>) -> vec2<f32> {
    let offset_current = current + normal;
    let n = i + i32(direction);
    if n < 0 || n >= i32(placed.line_count) || doubles_back(base, placed, i, i32(direction)) {
        return offset_current;
    }
    if all(next == current) {
        return offset_current;
    }
    let next_normal = perp_normal(next - current, offset_y, direction);
    return line_intersection(offset_previous, offset_current, current + next_normal, next + next_normal);
}

// GL JS `placeGlyphAlongLine`.
fn place_glyph(tile: Tile, base: u32, placed: Placed, anchor: vec2<f32>, offset_x: f32, line_offset: vec2<f32>, flip: bool) -> LineGlyph {
    var out: LineGlyph;
    out.ok = false;
    let combined = select(offset_x + line_offset.x, offset_x - line_offset.x, flip);
    var direction = select(-1.0, 1.0, combined > 0.0);
    var angle = 0.0;
    if flip {
        direction = -direction;
        angle = PI;
    }
    if direction < 0.0 {
        angle += PI;
    }
    var index = i32(placed.segment) + select(1, 0, direction > 0.0);
    var current = anchor;
    var previous = anchor;
    var previous_tile = placed.anchor;
    var distance = 0.0;
    var segment_distance = 0.0;
    let abs_offset = abs(combined);
    var segment = vec2<f32>(0.0);
    var start = anchor;
    // The offset line's previous vertex and its miter at the current one.
    var offset_point = vec2<f32>(0.0);
    var have_offset_point = false;
    for (var guard = 0u; guard < 4096u; guard++) {
        if distance + segment_distance > abs_offset {
            break;
        }
        index += i32(direction);
        if index < 0 || index >= i32(placed.line_count) {
            return out;
        }
        let first = distance + segment_distance == 0.0;
        distance += segment_distance;
        previous = current;
        let needed = abs_offset - distance + 1.0;
        // This vertex and, for an offset line, the one after it, through
        // one call site (Adreno compilers inline every call).
        let n = clamp(index + i32(direction), 0, i32(placed.line_count) - 1);
        var next = vec2<f32>(0.0);
        let vertices = select(1u, 2u, line_offset.y != 0.0);
        for (var k = 0u; k < vertices; k++) {
            // The next vertex's previous is this one, or GL JS's synthetic
            // vertex before the first step: the anchor.
            let from_tile = select(previous_tile, select(line_vertex(base, placed, u32(index)), placed.anchor, first), k == 1u);
            let projected = project_line_vertex(tile, base, placed, u32(select(index, n, k == 1u)), from_tile, previous, needed);
            if k == 0u {
                current = projected;
            } else {
                next = projected;
            }
        }
        previous_tile = line_vertex(base, placed, u32(index));
        if line_offset.y == 0.0 {
            start = previous;
            segment = current - previous;
        } else {
            var normal: vec2<f32>;
            if all(current == previous) {
                // The anchor sits on the vertex: look one vertex ahead.
                normal = select(perp_normal(next - current, line_offset.y, direction), vec2<f32>(0.0), all(next == current));
            } else {
                normal = perp_normal(current - previous, line_offset.y, direction);
            }
            start = select(previous + normal, offset_point, have_offset_point);
            offset_point = offset_intersection(base, placed, index, normal, current, start, line_offset.y, direction, next);
            have_offset_point = true;
            segment = offset_point - start;
        }
        segment_distance = length(segment);
    }
    let t = select(0.0, (abs_offset - distance) / segment_distance, segment_distance > 0.0);
    out.p = start + segment * t;
    let rotate_to_line = (node.flags & ROTATE_WITH_MAP) != 0u;
    out.angle = select(0.0, angle + atan2(current.y - previous.y, current.x - previous.x), rotate_to_line);
    out.ok = true;
    return out;
}

// Label-plane points `a` and `b` in clip space (`xy`, `zw`), through one
// call site (Adreno compilers inline every call).
fn clip_pair(tile: Tile, a: vec2<f32>, b: vec2<f32>) -> vec4<f32> {
    var clip = vec4<f32>(0.0);
    for (var k = 0u; k < 2u; k++) {
        let c = label_plane_to_clip(tile, select(a, b, k == 1u));
        if k == 0u {
            clip = vec4<f32>(c, clip.zw);
        } else {
            clip = vec4<f32>(clip.xy, c);
        }
    }
    return clip;
}

fn glyph_offset(base: u32, quad: u32) -> f32 {
    return bitcast<f32>(vertex_at(base + vertex_at(base + H_GLYPH_OFFSETS) + quad));
}

// A line label's glyph in the label plane: GL JS `placeGlyphsAlongLine`
// with its keep-upright flip and vertical switch, decided per vertex from
// the first and last glyphs. `place_glyph` has one call site, in a loop over
// the placements asked for: first and last glyph, again flipped, then this
// glyph. Adreno compilers inline every call, and five copies of its loop
// made the symbol pipeline take 28 s to compile on an Adreno 619.
// `anchor_clip` is the untranslated anchor in clip space.
fn line_glyph(tile: Tile, base: u32, placed: Placed, quad: u32, font_scale: f32, anchor_clip: vec4<f32>) -> LineGlyph {
    var fail: LineGlyph;
    fail.ok = false;
    // The anchor in the label plane: on the screen, from `vs_symbol`'s
    // projection of it (`anchor_clip`).
    var anchor = clip_to_px(anchor_clip);
    if (node.flags & PITCH_WITH_MAP) != 0u {
        anchor = to_pitched_plane(tile, placed.anchor);
    }
    let line_offset = placed.line_offset * font_scale;
    let own = glyph_offset(base, quad) * font_scale;
    let keep_upright = (node.flags & KEEP_UPRIGHT) != 0u;
    let aspect = frame.viewport.x / frame.viewport.y;
    let several = placed.quad_count > 1u;
    let first_offset = glyph_offset(base, placed.quad_start) * font_scale;
    let last_offset = glyph_offset(base, placed.quad_start + max(placed.quad_count, 1u) - 1u) * font_scale;
    var flip = false;
    if !several && keep_upright {
        // The segment's end, or behind the camera, the anchor's mirror of
        // a point a unit back: one projection call site.
        let segment_end = line_vertex(base, placed, placed.segment + 1u);
        var point = segment_end;
        var behind = false;
        var q: PlanePoint;
        loop {
            q = to_label_plane(tile, point);
            if q.w > 0.0 || behind {
                break;
            }
            behind = true;
            point = placed.anchor + normalize(placed.anchor - segment_end);
        }
        let end = select(q.p, anchor + (anchor - q.p), behind);
        let c = clip_pair(tile, anchor, end);
        // As for several glyphs below: the segment's slope picks the
        // horizontal glyph or its vertical alternative.
        let steep = abs(c.w - c.y) > abs(c.z - c.x) * aspect;
        if (placed.writing_mode == 1u && steep) || (placed.writing_mode == 2u && !steep) {
            return fail;
        }
        flip = select(c.x > c.z, c.y < c.w, placed.writing_mode == 2u);
    }
    // 0, 1: the first and last glyph; 2, 3: both flipped; 4: this glyph.
    var step = select(4u, 0u, several);
    var first = vec2<f32>(0.0);
    var placed_glyph = fail;
    loop {
        let offset = select(select(last_offset, first_offset, step % 2u == 0u), own, step == 4u);
        let flipped = select(step >= 2u, flip, step == 4u);
        let glyph = place_glyph(tile, base, placed, anchor, offset, line_offset, flipped);
        if step == 4u {
            placed_glyph = glyph;
            break;
        }
        if !glyph.ok {
            return fail;
        }
        if step == 0u {
            first = glyph.p;
            step = 1u;
            continue;
        }
        if step == 1u {
            step = 4u;
            if keep_upright {
                let c = clip_pair(tile, first, glyph.p);
                let a = c.xy;
                let b = c.zw;
                // A horizontal label with a vertical alternative steeper
                // than flat gives way to it; the vertical one shows only
                // then.
                let steep = abs(b.y - a.y) > abs(b.x - a.x) * aspect;
                if placed.writing_mode == 1u && steep {
                    return fail;
                }
                if placed.writing_mode == 2u && !steep {
                    return fail;
                }
                if select(a.x > b.x, a.y < b.y, placed.writing_mode == 2u) {
                    flip = true;
                    step = 2u;
                }
            }
            continue;
        }
        step += 1u;
    }
    return placed_glyph;
}
