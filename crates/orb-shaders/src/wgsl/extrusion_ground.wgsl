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
// the building covers (a raised part's, by its underside: see
// vs_extrusion_shadow). Occlusion turns each wall into a strip on the
// ground running out from its foot, past both ends by the radius so that
// convex corners round off.
//
// On terrain they draw into the ground mask instead (`ground_mask`), which
// the terrain's own mesh then carries onto the surface drawn (orb-render's
// terrain decal): drawn here, a shape lying flat between the DEM's heights
// at its corners would float over the ground or sink into it where the
// terrain bends between them, and the buildings' walls would cover what
// floats.
//
// Every vertex of a triangle draws or none does: one moved out of the view
// alone would stretch its triangle across the map.

// The pipelines that draw into the ground mask (`ground_mask_clip`), one
// sample of the darkness alone.
override ground_mask: bool = false;

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
    // On terrain, the ground's elevation under its feature's centroid,
    // which its building stands on (`extrusion_vertex`), metres; 0 off it.
    raise: f32,
    // Whether its feature's centroid lies in its tile's square, where the
    // placement's DEM reaches: past it, `raise` holds the DEM's edge.
    centroid_in_tile: bool,
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
    let centroid = unpack_i16x2(prop_at(row + EXTRUSION_ROW_CENTROID));
    g.centroid_in_tile = all(centroid >= vec2<f32>(0.0)) && all(centroid <= vec2<f32>(g.tile.extent));
    if frame.dem_dim > 0.0 {
        g.raise = ground_elevation(g.tile, centroid);
    }
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

// The DEM's elevation where ground vertex `g` lands moved `offset` world
// units along the map (east, north), metres: off the placement's DEM
// tile, at its edge, as far as its border reaches.
fn ground_elevation_at(g: GroundVertex, offset: vec2<f32>) -> f32 {
    let landing = g.pos + vec2<f32>(offset.x, -offset.y) / ground_scale(g.tile, g.pos);
    var at = ground_dem_at(g.tile, landing);
    at.coord = clamp(at.coord, vec2<f32>(0.5), vec2<f32>(frame.dem_dim + 2.5));
    return dem_bilinear(at);
}

// The clip position of ground vertex `g` moved `offset` world units along
// the map (east, north), on the ground there: on terrain, the DEM's height
// where it lands, or flat in the ground mask.
fn ground_landing(g: GroundVertex, offset: vec2<f32>) -> vec4<f32> {
    if ground_mask {
        return ground_mask_clip(g.ground.flat + offset);
    }
    if frame.dem_dim <= 0.0 {
        return ground_clip_reversed(g.ground, offset, 0.0);
    }
    let z = ground_elevation_at(g, offset);
    var clip = ground_clip_reversed(g.ground, offset, z * frame.extrusion_scale);
    clip.z *= 1.0 + GROUND_TERRAIN_PULL;
    return clip;
}

// How many times as far as on flat ground a shadow falls at most on ground
// falling away from the light: where the ground falls as steeply as the
// light, the light would carry it on until the ground rises again, and
// each vertex would settle on a different slope, spiking the shadow
// downhill. Ground falling half as steeply as the light already doubles
// it; past that, the terrain shades itself.
const SHADOW_DROP_LIMIT: f32 = 2.0;

// Where the light carries a point of ground vertex `g`'s building `up`
// metres above its base down to the ground, world units east and north of
// it. On terrain the point stands on the ground under its feature's
// centroid, and falls as far as it stands over the ground where it lands:
// two steps toward it from where it would land on the ground under it,
// which slopes gentler than the light settle on, and at most
// SHADOW_DROP_LIMIT times its height. A point at the foot of a part on
// the ground stays where it meets the ground.
fn shadow_reach(g: GroundVertex, up: f32) -> vec2<f32> {
    let along = frame.shadow_offset * frame.extrusion_scale;
    if frame.dem_dim <= 0.0 || up <= 0.0 {
        return along * up;
    }
    let top = g.raise + up;
    let limit = SHADOW_DROP_LIMIT * up;
    var drop = clamp(top - ground_elevation(g.tile, g.pos), 0.0, limit);
    for (var i = 0; i < 2; i++) {
        drop = clamp(top - ground_elevation_at(g, along * drop), 0.0, limit);
    }
    return along * drop;
}

// A shadow: each vertex where the light carries it down to the ground, at
// frame.shadow_offset metres east and north per metre it stands up. A wall
// facing the light, on a part standing on the ground, stays at its foot:
// its sweep would lie beneath its building and the shadows of the rest. A
// raised part's light leaves by its underside, which has no faces to draw,
// so its walls facing the light sweep too. A wall's two ends may judge
// apart on a rounded corner, which leaves part of the sweep, still in the
// shadow.
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
    // On terrain, a building cut at the buffer casts its shadow from the
    // placement holding its centroid: another's DEM doesn't reach it, and
    // would stand the building at the wrong height, its shadow at the
    // wrong length, which nothing covers where it falls on the terrain.
    let stray = frame.dem_dim > 0.0 && !g.centroid_in_tile;
    if g.opacity <= 0.0 || stray {
        out.position = discarded_vertex();
        return out;
    }
    let packed = vertex_at(v.vertex + 2u);
    var up = g.height;
    if !ground_roof(v) {
        let lit = dot(ground_outward(packed), frame.shadow_offset) < 0.0;
        up = select(g.height, g.base, (packed & 1u) == 0u || (lit && g.base <= 0.0));
    }
    out.position = ground_landing(g, shadow_reach(g, up));
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
fn ground_darkness(in: GroundOut) -> f32 {
    let beyond = max(max(-in.ground.y, in.ground.y - in.ground.z), 0.0);
    let d = length(vec2<f32>(in.ground.x, beyond));
    return in.ground.w * (1.0 - smoothstep(0.0, 1.0, d));
}

// As black whose alpha is the darkness, into the layer's target.
@fragment
fn fs_extrusion_ground(in: GroundOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, ground_darkness(in));
}

// As the darkness alone, into the ground mask.
@fragment
fn fs_extrusion_ground_mask(in: GroundOut) -> @location(0) vec4<f32> {
    return vec4<f32>(ground_darkness(in));
}
