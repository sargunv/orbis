// The symbol placements (orb-render's): each quad of a symbol node's
// visible chunks placed for the four corners the symbol draw shades. A
// compute dispatch places every symbol node's quads once a frame, into
// their texels of the placement target (`symbol_place_compute.wgsl`);
// without compute (WebGL 2), each corner places its quad itself
// (`symbol_place_inline.wgsl`).

// The quad's half places its glyphs along its line.
fn along_line() -> bool {
    return LINE_LABELS && (halves[half].flags & ALONG_LINE) != 0u;
}

// Quad `instance` of the draw of symbol `node`, whose layer's icons'
// uniform is `halves[first_half]`: 64 for each of its visible chunks, as
// the symbol draw's instances are. Zero opacity (nothing shows) until the
// quad is placed.
fn place_quad(node: NodeCull, first_half: u32, instance: u32) -> PlacedQuad {
    var out: PlacedQuad;
    let entry = visible_at(node.visible_base + instance / CHUNK_QUADS);
    let chunk = chunk_at(entry.chunk);
    let local = instance % CHUNK_QUADS;
    if local >= chunk.index_count { return out; }
    out.written = true;
    let base = chunk.vertex_base;
    half = first_half + select(0u, 1u, (vertex_at(base + H_FLAGS) & H_TEXT) != 0u);
    let tile = tile_at(entry.tile);
    let quad_index = vertex_at(base + vertex_at(base + H_ORDER) + chunk.first_index + local);
    let qa = base + vertex_at(base + H_QUADS) + quad_index * QUAD_WORDS;
    let placed_index = vertex_at(qa + 10u);
    // The placement's state first: most quads of a dense view are hidden,
    // and leave after the fewest reads. The block of the placement's world
    // copy; the placed table runs up to the glyph offsets.
    let placed_count = (vertex_at(base + H_GLYPH_OFFSETS) - vertex_at(base + H_PLACED)) / PLACED_WORDS;
    let copy = bitcast<u32>(tile.wrap) & (vertex_at(base + H_COPIES) - 1u);
    let state_at = vertex_at(base + H_STATES) + (copy * placed_count + placed_index) * STATE_WORDS;
    let state = prop_at(state_at);
    if (state & S_HIDDEN) != 0u { return out; }
    let placed = read_placed(base, placed_index);
    if !point_owned(tile, placed.anchor) { return out; }
    let fade_from = f32(state & S_OPACITY) / 127.0;
    let fade_change = select(-frame.symbol_fade_change, frame.symbol_fade_change, (state & S_PLACED) != 0u);
    let fade = clamp(fade_from + fade_change, 0.0, 1.0);

    // Instance rows: the feature's property-row slot, its height offset.
    let instance_row = base + vertex_at(base + H_INSTANCE_ROWS) + placed.instance * 2u;
    let slot = vertex_at(instance_row);
    symbol_height = bitcast<f32>(vertex_at(instance_row + 1u));
    let row = chunk.prop_base + slot * halves[half].row_words;
    var pulled: Pulled;
    pulled.state_base = chunk.state_base;
    let state_at_slot = state_block(pulled, slot, halves[half].state);
    let opacity = paint_number(row, state_at_slot, 0u, halves[half].opacity, tile.zoom);
    let total_opacity = opacity * fade;
    if total_opacity < 0.1 { return out; }

    let is_text = (halves[half].flags & IS_TEXT) != 0u;
    let pitch = (halves[half].flags & PITCH_WITH_MAP) != 0u;
    var size = feature_size(halves[half].size, placed.size, bitcast<f32>(vertex_at(base + H_LAYOUT_SIZE)),
        bitcast<f32>(vertex_at(base + H_ZOOM_LO)), bitcast<f32>(vertex_at(base + H_ZOOM_HI)));

    // `*-translate`, in tile units: the node's, or the feature's.
    var t = node_translate(halves[half].translate, halves[half].table_mask, row, halves[half].row_words, halves[half].translate_curve, tile.zoom);
    if (halves[half].flags & TRANSLATE_VIEWPORT) != 0u {
        let s = sin(frame.bearing);
        let c = cos(frame.bearing);
        t = vec2<f32>(t.x * c - t.y * s, t.x * s + t.y * c);
    }
    let translation = t * tile_units_per_px(tile);
    let anchor = placed.anchor + translation;
    let line_label = along_line();
    // The anchor's east neighbour, which map-rotated shifted labels and
    // in-shader rotation read. Variable anchors are point labels': WebGL
    // 2's line variant leaves their code out (Adreno's compilers take
    // seconds over each part).
    let rotate_shifted = POINT_LABELS && !line_label && (state & S_SHIFTED) != 0u && !pitch
        && (halves[half].flags & ROTATE_WITH_MAP) != 0u;
    let east = rotate_shifted || (halves[half].flags & ROTATE_IN_SHADER) != 0u;
    // A line label's untranslated anchor, where its glyphs follow the line
    // from, on a screen label plane.
    let line_anchor = line_label && !pitch && any(translation != vec2<f32>(0.0));
    // The anchor on the ground and in clip space, then the east neighbour
    // and the line's anchor, through one projection: Adreno's compilers
    // inline every call, and each projection's globe, warp and terrain
    // terms took seconds to compile on an Adreno 619.
    var anchor_ground: Ground;
    var anchor_metres = 0.0;
    var projected_point = vec4<f32>(0.0);
    var east_clip = vec4<f32>(0.0);
    var line_anchor_clip = vec4<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        if (k == 1u && !east) || (k == 2u && !line_anchor) {
            continue;
        }
        let p = select(select(placed.anchor, anchor + vec2<f32>(1.0, 0.0), k == 1u), anchor, k == 0u);
        let ground = tile_ground(tile, p);
        let metres = symbol_metres(tile, p);
        let clip = ground_clip(ground, vec2<f32>(0.0), metres * frame.extrusion_scale);
        if k == 0u {
            anchor_ground = ground;
            anchor_metres = metres;
            projected_point = clip;
        } else if k == 1u {
            east_clip = clip;
        } else {
            line_anchor_clip = clip;
        }
    }
    if !east {
        east_clip = projected_point;
    }
    if !line_anchor {
        line_anchor_clip = projected_point;
    }
    // Terrain in front of the anchor hides it (GL JS `calculate_visibility`).
    let visible_opacity = total_opacity * terrain_visibility(
        terrain_seen(projected_point, anchor_ground, anchor_metres * frame.extrusion_scale));
    if visible_opacity < 0.1 { return out; }
    if globe_hides_anchor(anchor_ground, anchor_metres) { return out; }
    let ctcd = frame.camera_to_center;
    let distance_ratio = select(ctcd / projected_point.w, projected_point.w / ctcd, pitch);
    let perspective_ratio = clamp(0.5 + 0.5 * distance_ratio, 0.0, 4.0);

    // The label plane position and angle: GL JS's dynamic `a_projected_pos`.
    var plane_pos: vec2<f32>;
    var segment_angle = select(0.0, -PI / 2.0, placed.writing_mode == 2u);
    var in_plane = false;
    if line_label {
        // Only labels whose anchor is near the viewport are placed.
        let clip = projected_point.xy / projected_point.w;
        let buffer = vec2<f32>(256.0) / frame.viewport * 2.0 + 1.0;
        if any(abs(clip) > buffer) { return out; }
        // GL JS `updateLineLabels`: the unclamped ratio, sizes over 24.
        let line_ratio = 0.5 + 0.5 * ctcd / projected_point.w;
        let plane_size = select(size * line_ratio, size / line_ratio, pitch);
        let glyph = line_glyph(tile, base, placed, quad_index, plane_size / 24.0, line_anchor_clip);
        if !glyph.ok { return out; }
        plane_pos = glyph.p;
        segment_angle = -glyph.angle;
        in_plane = true;
    } else if POINT_LABELS && (state & S_SHIFTED) != 0u {
        // GL JS `updateVariableAnchorsForBucket` and `getShiftedAnchor`.
        var shift_size = size;
        if !is_text {
            let text_packed = vertex_at(base + vertex_at(base + H_TEXT_SIZES) + placed_index);
            shift_size = feature_size(halves[half].text_size,
                vec2<f32>(f32(text_packed & 0xffffu), f32(text_packed >> 16u)) / 128.0,
                bitcast<f32>(vertex_at(base + H_TEXT_LAYOUT_SIZE)),
                bitcast<f32>(vertex_at(base + H_TEXT_ZOOM_LO)), bitcast<f32>(vertex_at(base + H_TEXT_ZOOM_HI)));
        }
        var render_size = shift_size * (0.5 + 0.5 * ctcd / projected_point.w) / 24.0;
        if pitch {
            render_size *= tile_units_per_px(tile);
        }
        let shift = vec2<f32>(bitcast<f32>(prop_at(state_at + 1u)), bitcast<f32>(prop_at(state_at + 2u))) * render_size;
        if pitch {
            var adjusted = shift;
            if (halves[half].flags & ROTATE_WITH_MAP) == 0u {
                let s = sin(frame.bearing);
                let c = cos(frame.bearing);
                adjusted = vec2<f32>(shift.x * c - shift.y * s, shift.x * s + shift.y * c);
            }
            plane_pos = to_pitched_plane(tile, anchor + adjusted);
        } else {
            // Not pitched: the label plane is the screen.
            let projected = clip_to_px(projected_point);
            if (halves[half].flags & ROTATE_WITH_MAP) != 0u {
                let east = clip_to_px(east_clip) - projected;
                let a = atan(east.y / east.x) + select(0.0, PI, east.x < 0.0);
                let s = sin(a);
                let c = cos(a);
                plane_pos = projected + vec2<f32>(shift.x * c - shift.y * s, shift.x * s + shift.y * c);
            } else {
                plane_pos = projected + shift;
            }
        }
        let quarter = (is_text && placed.writing_mode == 2u) || (!is_text && (state & S_QUARTER_TURN) != 0u);
        segment_angle = select(0.0, -PI / 2.0, quarter);
        in_plane = true;
    }

    if (halves[half].flags & IS_OFFSET) == 0u {
        size *= perspective_ratio;
    }

    var symbol_rotation = 0.0;
    if (halves[half].flags & ROTATE_IN_SHADER) != 0u {
        let a = projected_point.xy / projected_point.w;
        let b = east_clip.xy / east_clip.w;
        symbol_rotation = atan2((b.y - a.y) / (frame.viewport.x / frame.viewport.y), b.x - a.x);
    }
    // Map-pitched glyphs keep their size on a globe: the corners' offsets
    // turn and scale together.
    let correction = select(1.0, pitched_correction(anchor_ground), pitch);
    let angle_sin = sin(segment_angle + symbol_rotation) * correction;
    let angle_cos = cos(segment_angle + symbol_rotation) * correction;

    if !in_plane {
        if pitch {
            plane_pos = to_pitched_plane(tile, anchor);
        } else {
            plane_pos = clip_to_px(projected_point);
        }
    }

    let fill = paint_color(row, state_at_slot, 1u, halves[half].fill_color, tile.zoom);
    let halo = paint_color(row, state_at_slot, 2u, halves[half].halo_color, tile.zoom);
    var fill_word = pack_4x8_unorm(fill);
    if ((vertex_at(qa + 11u) >> 16u) & Q_COLOR) != 0u {
        fill_word = vertex_at(base + vertex_at(base + H_COLORS) + quad_index);
    }
    out.at = vec4<u32>(bitcast<vec2<u32>>(plane_pos), bitcast<u32>(angle_cos), bitcast<u32>(angle_sin));
    out.paint = vec4<u32>(bitcast<u32>(size), bitcast<u32>(visible_opacity), fill_word, pack_4x8_unorm(halo));
    return out;
}

