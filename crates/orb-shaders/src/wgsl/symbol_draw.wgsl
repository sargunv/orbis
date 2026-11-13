// The symbol draw: each quad's four corners around where its placement
// put it (`placement`: `symbol_placed.wgsl` or `symbol_place_inline.wgsl`),
// and their fragments.

@group(0) @binding(9) var image_pages: texture_2d_array<f32>;
@group(0) @binding(10) var atlas_sampler: sampler;
@group(0) @binding(13) var nearest_sampler: sampler;
// The image pages again, for the nearest sampler: each texture binding
// takes one sampler, as a texture unit does on WebGL 2.
@group(0) @binding(16) var image_pages_nearest: texture_2d_array<f32>;
@group(0) @binding(12) var glyph_pages: texture_2d_array<f32>;

// What the pass draws of the icons (x) and of the text (y): halo and fill
// together, the halo, the fill (GL JS draws them apart when glyphs
// overlap), or nothing (orb-render's `SymbolPass`). A uniform rather than
// an override: one pipeline draws every pass, so the largest shader
// compiles once.
@group(2) @binding(0) var<uniform> symbol_pass: vec4<u32>;
const PASS_WHOLE: u32 = 0u;
const PASS_HALO: u32 = 1u;
const PASS_FILL: u32 = 2u;
const PASS_SKIP: u32 = 3u;

struct SymbolOut {
    @builtin(position) position: vec4<f32>,
    // The page layer, the bitmap's padded rectangle's origin on it, and
    // the quad's texels from there (see `atlas_uv`), taken where the quad
    // covers the pixel: multisampled (on terrain with antialiasing on),
    // a pixel the quad's edge crosses would otherwise take them at its
    // centre, off the quad, and read the bitmap packed beside it.
    @location(0) @interpolate(flat, either) layer: u32,
    @location(1) @interpolate(perspective, centroid) tex: vec2<f32>,
    @location(7) @interpolate(flat, either) origin: vec2<f32>,
    // Gamma scale, size and total opacity.
    @location(2) data: vec3<f32>,
    @location(3) @interpolate(flat, either) fill: vec4<f32>,
    @location(4) @interpolate(flat, either) halo: vec4<f32>,
    // Halo width, halo blur; whether the quad is an SDF (1), an image (2),
    // sampled linearly (4), text (`K_TEXT`) and pitched with the map
    // (`K_PITCH`).
    @location(5) @interpolate(flat, either) halo_params: vec2<f32>,
    @location(6) @interpolate(flat, either) kind: u32,
};
const K_TEXT: u32 = 8u;
const K_PITCH: u32 = 16u;
// Paint slots 3 and 4, the halo's width and blur.
const HALO_SLOTS: u32 = 0x18u;

@vertex
fn vs_symbol(@builtin(vertex_index) index: u32, @builtin(instance_index) instance: u32) -> SymbolOut {
    var out: SymbolOut;
    out.position = discarded_vertex();
    // The draw is the icon node's.
    let node = node_cull_at(halves[0].index);
    let entry = visible_at(node.visible_base + instance / CHUNK_QUADS);
    let chunk = chunk_at(entry.chunk);
    let local = instance % CHUNK_QUADS;
    if local >= chunk.index_count { return out; }
    let base = chunk.vertex_base;
    let mesh_flags = vertex_at(base + H_FLAGS);
    half = select(0u, 1u, (mesh_flags & H_TEXT) != 0u);
    let mode = select(symbol_pass.x, symbol_pass.y, half == 1u);
    if mode == PASS_SKIP { return out; }
    // Where the quad is placed: nothing of it shows at zero opacity,
    // which most quads of a dense view have.
    let placed = placement(node, instance);
    let paint = placed.paint;
    let visible_opacity = bitcast<f32>(paint.y);
    if !(visible_opacity > 0.0) { return out; }
    let at = placed.at;
    let quad_index = vertex_at(base + vertex_at(base + H_ORDER) + chunk.first_index + local);
    let qa = base + vertex_at(base + H_QUADS) + quad_index * QUAD_WORDS;
    let flags = vertex_at(qa + 11u) >> 16u;
    let image = (flags & Q_IMAGE) != 0u;
    // The quad's bitmap in the atlas, read with the quad: an image's rect,
    // or a glyph's row in the glyph row table (the property arena's first
    // block; row 0, unused, for an image). Read after the full variants'
    // projections, these returned another glyph's row on an Adreno 619
    // (native Vulkan, naga's SPIR-V; Tint's of the same source didn't),
    // and labels sampled empty texels (#173, #178).
    let atlas = vertex_at(base + vertex_at(base + select(H_GLYPHS, H_IMAGES, image)) + vertex_at(qa + 9u));
    let glyph_row = select(atlas, 0u, image) * 4u;
    let glyph_origin = prop_at(glyph_row);
    let glyph_ready = prop_at(glyph_row + 1u);
    let glyph_layer = prop_at(glyph_row + 2u);

    let is_text = (halves[half].flags & IS_TEXT) != 0u;
    let pitch = (halves[half].flags & PITCH_WITH_MAP) != 0u;
    let size = bitcast<f32>(paint.x);
    let font_scale = select(size, size / 24.0, is_text);
    let angle_cos = bitcast<f32>(at.z);
    let angle_sin = bitcast<f32>(at.w);
    let rotation = mat2x2<f32>(vec2<f32>(angle_cos, -angle_sin), vec2<f32>(angle_sin, angle_cos));
    let plane_pos = bitcast<vec2<f32>>(at.xy);

    // The feature's row, which a map-pitched quad's height offset and a
    // halo from the feature's row or state need: read only for those.
    let halo_rows = (row_mask(halves[half].table_mask) & HALO_SLOTS) != 0u
        || (feature_rows && feature_state && (halves[half].state.mask & HALO_SLOTS) != 0u);
    var halo_params = vec2<f32>(halves[half].halo_width, halves[half].halo_blur);
    var tile: Tile;
    if pitch || halo_rows {
        tile = tile_at(entry.tile);
        let instance_row = symbol_instance_row(base, vertex_at(qa + 10u));
        let feature = vertex_at(instance_row);
        symbol_height = bitcast<f32>(vertex_at(instance_row + 1u));
        if halo_rows {
            let row = chunk.prop_base + feature * halves[half].row_words;
            var pulled: Pulled;
            pulled.state_base = chunk.state_base;
            let state = state_block(pulled, feature, halves[half].state);
            halo_params = vec2<f32>(paint_number(row, state, 3u, halo_params.x, tile.zoom),
                paint_number(row, state, 4u, halo_params.y, tile.zoom));
        }
    }

    // Corners tl, tr, bl, br, indexed into two triangles (orb-render's
    // `QUAD_INDICES`).
    let corner = index & 3u;
    let offset = unpack_i16x2(vertex_at(qa + corner));
    let tex_words = vec2<u32>(vertex_at(qa + 4u), vertex_at(qa + 5u));
    let x0 = f32(tex_words.x & 0xffffu);
    let y0 = f32(tex_words.x >> 16u);
    let x1 = f32(tex_words.y & 0xffffu);
    let y1 = f32(tex_words.y >> 16u);
    let px = vec2<f32>(unpack_i16x2(vertex_at(qa + 6u)));
    let px_br = vec2<f32>(unpack_i16x2(vertex_at(qa + 7u)));
    let min_font_scale = unpack_i16x2(vertex_at(qa + 8u)) / 256.0;
    let right = corner == 1u || corner == 3u;
    let bottom = corner >= 2u;
    let pixel_offset = vec2<f32>(select(px.x, px_br.x, right), select(px.y, px_br.y, bottom)) / 16.0;
    let corner_tex = vec2<f32>(select(x0, x1, right), select(y0, y1, bottom));

    let glyph_offset_px = rotation * (offset / 32.0 * max(min_font_scale, vec2<f32>(font_scale)) + pixel_offset);
    var final_pos: vec4<f32>;
    if pitch {
        // The quad lies level at the elevation under its placement point
        // (a glyph's on its line, a point label's anchor), as GL JS's
        // pitched symbols stand at their anchor's: corners raised each by
        // the ground under it fold over a slope that faces away from the
        // eye, which draws the glyph mirrored.
        let at = from_pitched_plane(tile, plane_pos);
        let metres = symbol_metres(tile, at) * frame.extrusion_scale;
        final_pos = ground_clip(tile_ground(tile, from_pitched_plane(tile, plane_pos + glyph_offset_px)), vec2<f32>(0.0), metres);
    } else {
        final_pos = vec4<f32>(px_to_clip(plane_pos + glyph_offset_px), 0.0, 1.0);
    }
    out.position = final_pos;
    out.tex = corner_tex;

    // Texels: the padded rectangle's origin plus the quad's texels.
    if image {
        if atlas == 0xffffffffu {
            return out;
        }
        out.layer = atlas >> 24u;
        out.origin = vec2<f32>(f32(atlas & 0xfffu), f32((atlas >> 12u) & 0xfffu));
    } else {
        out.layer = glyph_layer;
        out.origin = vec2<f32>(f32(glyph_origin & 0xffffu), f32(glyph_origin >> 16u));
        if glyph_ready == 0u {
            // Not rasterized yet: nothing to draw.
            out.position = discarded_vertex();
        }
    }
    out.data = vec3<f32>(final_pos.w, size, visible_opacity);
    out.fill = unpack4x8unorm(paint.z);
    out.halo = unpack4x8unorm(paint.w);
    out.halo_params = halo_params;
    let linear = (halves[half].flags & LINEAR_ICONS) != 0u || (mesh_flags & H_LINEAR_IMAGES) != 0u
        || abs(frame.pitch) > 0.0;
    out.kind = select(0u, 1u, (flags & Q_SDF) != 0u) | select(0u, 2u, image)
        | select(0u, 4u, linear) | select(0u, K_TEXT, is_text) | select(0u, K_PITCH, pitch);
    return out;
}

const SDF_PX: f32 = 8.0;

// Where a fragment samples its page. The quad's texels round to 1/1024
// texel, so adding the rectangle's whole-texel origin is exact in f32 and
// the sampler's filter weights don't depend on where the atlas packed the
// bitmap, which the order glyphs and images arrived in decides.
fn atlas_uv(in: SymbolOut, page: f32) -> vec2<f32> {
    return (in.origin + round(in.tex * 1024.0) / 1024.0) / page;
}

@fragment
fn fs_symbol(in: SymbolOut) -> @location(0) vec4<f32> {
    let image = (in.kind & 2u) != 0u;
    let sdf = (in.kind & 1u) != 0u;
    let total_opacity = in.data.z;
    let is_text = (in.kind & K_TEXT) != 0u;
    // The pages' sizes, as the quad's half has them.
    let h = select(0u, 1u, is_text);
    let uv = atlas_uv(in, select(halves[h].glyph_page, halves[h].image_page, image));
    if image && !sdf {
        if (in.kind & 4u) != 0u {
            return textureSampleLevel(image_pages, atlas_sampler, uv, in.layer, 0.0) * total_opacity;
        }
        return textureSampleLevel(image_pages_nearest, nearest_sampler, uv, in.layer, 0.0) * total_opacity;
    }
    var dist: f32;
    if image {
        dist = textureSampleLevel(image_pages, atlas_sampler, uv, in.layer, 0.0).a;
    } else {
        dist = textureSampleLevel(glyph_pages, atlas_sampler, uv, in.layer, 0.0).r;
    }
    let mode = select(symbol_pass.x, symbol_pass.y, is_text);
    let size = in.data.y;
    let gamma_scale = in.data.x;
    let font_scale = select(size, size / 24.0, is_text);
    let edge_gamma = 0.105 / frame.pixel_ratio;
    let u_gamma_scale = select(1.0, cos(frame.pitch) * frame.camera_to_center, (in.kind & K_PITCH) != 0u);
    let gamma = edge_gamma / (font_scale * u_gamma_scale);
    let inner_edge = (256.0 - 64.0) / 256.0;

    var text = vec4<f32>(0.0);
    var halo = vec4<f32>(0.0);
    let halo_width = in.halo_params.x;
    let halo_blur = in.halo_params.y;
    if mode != PASS_HALO {
        let g = gamma * gamma_scale;
        text = total_opacity * smoothstep(inner_edge - g, inner_edge + g, dist) * in.fill;
    }
    let has_halo = halo_width != 0.0 && dot(in.halo, in.halo) != 0.0;
    if mode != PASS_FILL && has_halo {
        let gamma_halo = (halo_blur * 1.19 / SDF_PX + edge_gamma) / (font_scale * u_gamma_scale);
        let inner_edge_halo = inner_edge + gamma_halo * gamma_scale;
        let g = gamma_halo * gamma_scale;
        var alpha = smoothstep(inner_edge_halo - g, inner_edge_halo + g, dist);
        let halo_edge = (6.0 - halo_width / font_scale) / SDF_PX;
        alpha = min(smoothstep(halo_edge - g, halo_edge + g, dist), 1.0 - alpha);
        halo = total_opacity * alpha * in.halo;
    }
    let color = text + (1.0 - text.a) * halo;
    if color.a <= 0.0 { discard; }
    return color;
}
