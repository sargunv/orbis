// What an extrusion layer darkens on the ground beneath its buildings:
// style-spec#1797's shadows and style-spec#1796's ground occlusion. They
// draw into the layer's target before its buildings, as black whose alpha
// is how dark, the darkest kept where shapes overlap (a max blend), so
// overlapping shadows darken once. The buildings then cover what they
// stand over, and the target composites over the map, darkening what lies
// beneath the layer by what is left.
//
// Both reuse the buildings' own vertices, drawn again by these vertex
// stages. A shadow is every roof and the walls turned from the light,
// projected along it onto the ground the building stands on: each ray of
// light through a building enters by its roof or a wall facing the light,
// and leaves by a wall turned from it or into the ground beneath it, which
// the building covers. Occlusion turns each wall into a strip on the
// ground running out from its foot, past both ends by the radius so that
// convex corners round off.
//
// Every vertex of a triangle draws or none does: one moved out of the view
// alone would stretch its triangle across the map.

struct GroundOut {
    @builtin(position) position: vec4<f32>,
    // How far out from the wall, how far along it from its start and its
    // length, all in radii (zeros for a shadow), and the darkness at its
    // foot.
    @location(0) ground: vec4<f32>,
};

// A vertex of a building this placement draws, with what the ground
// passes read of its feature.
struct GroundVertex {
    tile: Tile,
    pos: vec2<f32>,
    feature: u32,
    // Metres above the ground: its part's base and height, clamped to it.
    base: f32,
    height: f32,
    opacity: f32,
    // The ground it stands on, moved by the layer's translate.
    ground: Ground,
};

// Pulled vertex `v`'s building, which this placement draws.
fn ground_vertex(v: Pulled) -> GroundVertex {
    var g: GroundVertex;
    g.tile = tile_at(v.tile);
    g.pos = unpack_i16x2(vertex_at(v.vertex));
    g.feature = vertex_at(v.vertex + 1u);
    let row = prop_row(v, g.feature, v.row_words);
    let part = extrusion_feature(g.tile, row, state_block(v, g.feature, node.state));
    g.base = max(part.x, 0.0);
    g.height = max(part.y, 0.0);
    g.opacity = part.z;
    let px = node_translate(node.translate, node.table_mask, row, v.row_words, node.translate_curve, g.tile.zoom);
    g.ground = tile_ground_shifted(g.tile, g.pos, translate_world(px, node.translate_anchor, g.tile));
    return g;
}

// World units per data unit at `p`, the square root of the tile's area
// scale there: a ground pass's offsets are short enough to take it as one.
fn ground_scale(tile: Tile, p: vec2<f32>) -> f32 {
    return sqrt(sqrt(determinant(tile_metric_at(tile, p))));
}

// Whether a roof vertex: its normal.z word's low half is set.
fn ground_roof(v: Pulled) -> bool {
    return (vertex_at(v.vertex + 3u) & 0xffffu) != 0u;
}

// A wall vertex's outward normal, east and north. Tile axes run y south,
// and the layout's wall normals point in from the wall there, as GL JS's
// do.
fn ground_outward(packed: u32) -> vec2<f32> {
    let n = normalize(unpack_i16x2(packed));
    return vec2<f32>(-n.x, n.y);
}

// How much nearer the terrain's surface ground passes test their depth, as
// a fraction of the distance: they lie flat between DEM samples where the
// terrain's mesh bends, and would sink into it. A pull in depth alone, so
// they stay where they fall on screen.
const GROUND_TERRAIN_PULL: f32 = 0.01;

// The clip position of ground vertex `g` moved `offset` world units along
// the map (east, north), on the ground there: on terrain, the DEM's height
// where it lands.
fn ground_landing(g: GroundVertex, offset: vec2<f32>) -> vec4<f32> {
    if frame.dem_dim <= 0.0 {
        return ground_clip_reversed(g.ground, offset, 0.0);
    }
    let landing = g.pos + vec2<f32>(offset.x, -offset.y) / ground_scale(g.tile, g.pos);
    let z = ground_elevation(g.tile, clamp(landing, vec2<f32>(0.0), vec2<f32>(g.tile.extent)));
    var clip = ground_clip_reversed(g.ground, offset, z * frame.extrusion_scale);
    clip.z *= 1.0 + GROUND_TERRAIN_PULL;
    return clip;
}

// A shadow: each vertex where the light carries it down to the ground, at
// frame.shadow_offset metres east and north per metre it stands up. A wall
// facing the light stays at its foot: its sweep would lie beneath its
// building and the shadows of the rest. Its two ends may judge apart on a
// rounded corner, which leaves part of the sweep, still in the shadow.
@vertex
fn vs_extrusion_shadow(@builtin(vertex_index) index: u32) -> GroundOut {
    var out: GroundOut;
    var v: Pulled;
    if frame.shadow_intensity <= 0.0 || !pull_indexed(node.index, index, 4u, &v)
        || extrusion_drawn(v) == 0u {
        out.position = discarded_vertex();
        return out;
    }
    let g = ground_vertex(v);
    if g.opacity <= 0.0 {
        out.position = discarded_vertex();
        return out;
    }
    let packed = vertex_at(v.vertex + 2u);
    var up = g.height;
    if !ground_roof(v) {
        let lit = dot(ground_outward(packed), frame.shadow_offset) < 0.0;
        up = select(g.height, g.base, (packed & 1u) == 0u || lit);
    }
    out.position = ground_landing(g, frame.shadow_offset * (up * frame.extrusion_scale));
    out.ground = vec4<f32>(0.0, 0.0, 0.0, frame.shadow_intensity * g.opacity);
    return out;
}

// Occlusion: each wall's foot stays, its top folds down to the ground the
// radius out from it, and both ends reach the radius past its corners.
// A part standing off the ground darkens it less, and not at all from the
// radius up.
@vertex
fn vs_extrusion_occlusion(@builtin(vertex_index) index: u32) -> GroundOut {
    var out: GroundOut;
    out.position = discarded_vertex();
    let radius = node.occlusion.w;
    var v: Pulled;
    if radius <= 0.0 || !pull_indexed(node.index, index, 4u, &v) || ground_roof(v)
        || extrusion_drawn(v) == 0u {
        return out;
    }
    let g = ground_vertex(v);
    let intensity = node.occlusion.z * g.opacity * (1.0 - min(g.base / radius, 1.0));
    if intensity <= 0.0 {
        return out;
    }
    // The wall's other end: the vertex at the same height two along, on
    // its far side as this one's end bit says (orb-layout's extrusion
    // layout). Where a chunk's vertices were copied apart and lost it, the
    // vertex stays at its foot, undarkened.
    let packed = vertex_at(v.vertex + 2u);
    let end = (packed >> 16u) & 1u;
    let mate = select(v.vertex - 8u, v.vertex + 8u, end == 1u);
    let along = g.pos - unpack_i16x2(vertex_at(mate));
    let span = length(along);
    if vertex_at(mate + 1u) != g.feature || ((vertex_at(mate + 2u) ^ packed) & 0x10001u) != 0x10000u
        || span <= 0.0 {
        out.position = ground_landing(g, vec2<f32>(0.0));
        out.ground = vec4<f32>(0.0);
        return out;
    }
    let top = f32(packed & 1u);
    let to = vec2<f32>(along.x, -along.y) / span + ground_outward(packed) * top;
    let reach = radius * frame.extrusion_scale;
    out.position = ground_landing(g, to * reach);
    let wall = span * ground_scale(g.tile, g.pos) / reach;
    out.ground = vec4<f32>(top, select(-1.0, wall + 1.0, end == 1u), wall, intensity);
    return out;
}

// The darkness at a fragment: a shadow's throughout; occlusion's from the
// wall's foot out to the radius, rounding past its ends.
@fragment
fn fs_extrusion_ground(in: GroundOut) -> @location(0) vec4<f32> {
    let beyond = max(max(-in.ground.y, in.ground.y - in.ground.z), 0.0);
    let d = length(vec2<f32>(in.ground.x, beyond));
    return vec4<f32>(0.0, 0.0, 0.0, in.ground.w * (1.0 - smoothstep(0.0, 1.0, d)));
}
