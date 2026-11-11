// Cull and compact in one dispatch, a workgroup per render node.
//
// Pulled and point nodes have one workgroup, which walks the node's
// `(chunk, tile placement)` candidates in order, 256 at a time, and
// compacts the survivors into the node's visible list with a workgroup
// prefix sum, so the list keeps candidate order (draw order)
// deterministically. They draw it with `drawIndirect`.
//
// Indexed nodes give every candidate a fixed slot in the compacted index
// buffer, so their batches of 256 are independent. A slot stays put while its
// candidate lives, through layouts of the candidates after tiles come or
// go; the rows a candidate left are holes (`NO_CHUNK`) the cull skips. A
// layout that moves a run's slots copies them (`MOVED`). A candidate
// whose visibility changed since the last cull rewrites its slot, the
// batch's lanes copying together: the chunk's indices tagged with its run
// when it enters the view, hidden indices when it leaves.
// The node draws its slots up to the last visible one with
// `drawIndexedIndirect`, so frames where nothing enters or leaves write no
// indices. The node's workgroup walks its batches in turn: a workgroup for
// each batch the node with the most candidates has, in every node, most
// with nothing to do, cost the moto g's cull as much again once one line
// node had thousands of chunks (#180).

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> tiles: array<Tile>;
@group(0) @binding(2) var<storage, read> chunks: array<Chunk>;
struct Candidate {
    // The chunk, or NO_CHUNK in a hole.
    chunk: u32,
    tile: u32,
    // Indexed nodes: the first index of the candidate's slot, its run, and
    // whether the slot holds the chunk (1), hidden indices (0), one of the
    // two as the slot it was copied from did (MOVED), or nothing yet (any
    // other value).
    slot: u32,
    run: u32,
    drawn: u32,
};

@group(0) @binding(3) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(4) var<storage, read> node_cull: array<NodeCull>;
@group(0) @binding(5) var<storage, read_write> visible_out: array<Visible>;
// Per node, ARGS_WORDS words: the draw's indirect arguments (pulled and
// point nodes: vertex count, instance count, first vertex, first instance;
// indexed nodes: index count, instance count, first index, base vertex,
// first instance), then the cull's counters: at word 5 the chunks kept,
// at 6 the indices it wrote, at 7 the vertex inputs it draws (vertex
// pulling's padded vertices, or the visible chunks' indices).
@group(0) @binding(6) var<storage, read_write> args: array<atomic<u32>>;
// Mesh indices: u16s, two to a word, the first in the low half.
@group(0) @binding(7) var<storage, read> indices: array<u32>;
@group(0) @binding(8) var<storage, read_write> compacted: array<u32>;

const ARGS_WORDS: u32 = 8u;
// An indexed node's candidate row that holds none (orb-render's
// `NO_CHUNK`): its slot is hidden and taken by no candidate.
const NO_CHUNK: u32 = 0xffffffffu;
// A candidate's `drawn` once a layout moved its slot (orb-render's
// `MOVED`): the slot's first index says which it holds.
const MOVED: u32 = 2u;
// Keeps every candidate, culling nothing: what a driver without the cull
// draws, for tests that hold the two to the same buffers and pixels. A
// pipeline constant, false in shipping builds, where it folds away.
override keep_all: bool = false;
// Indices per chunk at most (orb-layout's CHUNK_INDICES).
const CHUNK_INDICES: u32 = 768u;
// Instances per point chunk (orb-layout's MAX_CHUNK_INSTANCES).
const CHUNK_POINTS: u32 = 64u;
// The view-volume planes a clip-space corner lies outside, a bit each:
// right, left, top, bottom (widened by `margin`, in NDC per unit w), far
// and near.
fn outside_planes(p: vec4<f32>, margin: vec2<f32>) -> u32 {
    let extra = margin * max(p.w, 0.0);
    return select(0u, 1u, p.x > p.w + extra.x)
        | select(0u, 2u, p.x < -p.w - extra.x)
        | select(0u, 4u, p.y > p.w + extra.y)
        | select(0u, 8u, p.y < -p.w - extra.y)
        | select(0u, 16u, p.z > p.w)
        | select(0u, 32u, p.z < 0.0);
}

// Whether any of the candidate's chunk can draw: its bounds, inside the
// tile's clip rectangles (unless its meshes draw past them), standing on the ground and raised by
// `elevation`, not wholly outside one plane of the view volume.
//
// No function-scope arrays: this is inlined once per cull mode, and two
// copies of a version that indexed local arrays in loops (its eight
// corners, the tile's copied clip rectangles) miscompiled on an Adreno 830
// (Chrome 156, Vulkan): the dispatch wrote nothing or lost the device.
// A test in orb-shaders keeps the cull shader free of them.
fn candidate_visible(candidate: Candidate, margin_px: f32, draw: u32, elevation: f32, draped: bool, unclipped_meshes: bool) -> bool {
    let point_instances = draw == DRAW_POINTS;
    let chunk = chunks[candidate.chunk];
    let tile = tiles[candidate.tile];

    let margin = margin_px * tile_world_per_px(tile) / tile.scale;
    let before_clip = select(margin, 0.0, point_instances);
    var lo = unpack_i16x2(chunk.bounds_min) - vec2<f32>(before_clip);
    var hi = unpack_i16x2(chunk.bounds_max) + vec2<f32>(before_clip);
    // Only the part inside the tile's clip rectangles can draw. Images clip
    // per fragment in their own units.
    var clip_lo = vec2<f32>(1e9);
    var clip_hi = vec2<f32>(-1e9);
    let unclipped = draw == DRAW_IMAGES || unclipped_meshes;
    if unclipped {
        clip_lo = lo;
        clip_hi = hi;
    }
    for (var i = 0u; i < 4u && !unclipped; i++) {
        let clip = tiles[candidate.tile].clips[i];
        let c_lo = max(lo, unpack_i16x2(clip.x));
        let c_hi = min(hi, unpack_i16x2(clip.y));
        if all(c_lo <= c_hi) {
            clip_lo = min(clip_lo, c_lo);
            clip_hi = max(clip_hi, c_hi);
        }
    }
    if any(clip_lo > clip_hi) {
        return false;
    }
    // A drape placement exists only for a slot to draw: its clip
    // rectangles are the slot's part of the tile. A warped tile's chunks
    // pass too: its patches aren't bound here (the stage's storage buffers
    // are all in use), and cover already kept only tiles whose footprints
    // meet the view.
    if draped || tile.warp_cells != 0u || tile.globe_cells != 0u {
        return true;
    }
    let after_clip = select(0.0, margin, point_instances);
    lo = clip_lo - vec2<f32>(after_clip);
    hi = clip_hi + vec2<f32>(after_clip);
    // On terrain, content stands between the lowest and highest ground
    // under the placement; extrusions rise above it.
    let ground = unpack_i16x2(tile.terrain_range) * TERRAIN_RANGE_UNIT * frame.extrusion_scale;
    let c0 = frame.view_proj * vec4<f32>(tile_affine(tile, lo), ground.x, 1.0);
    let c1 = frame.view_proj * vec4<f32>(tile_affine(tile, vec2<f32>(hi.x, lo.y)), ground.x, 1.0);
    let c2 = frame.view_proj * vec4<f32>(tile_affine(tile, vec2<f32>(lo.x, hi.y)), ground.x, 1.0);
    let c3 = frame.view_proj * vec4<f32>(tile_affine(tile, hi), ground.x, 1.0);
    let rise = ground.y - ground.x + elevation * frame.extrusion_scale;
    let raised = frame.view_proj * vec4<f32>(0.0, 0.0, rise, 0.0);
    let m = 2.0 * margin_px / frame.viewport;
    // Extrusions' shadows (style-spec#1797) reach along the ground away
    // from the light as far as the tallest casts them (none without).
    let shadow = frame.view_proj * vec4<f32>(frame.shadow_offset * (elevation * frame.extrusion_scale), 0.0, 0.0);
    let outside = outside_planes(c0, m) & outside_planes(c1, m) & outside_planes(c2, m)
        & outside_planes(c3, m) & outside_planes(c0 + raised, m) & outside_planes(c1 + raised, m)
        & outside_planes(c2 + raised, m) & outside_planes(c3 + raised, m)
        & outside_planes(c0 + shadow, m) & outside_planes(c1 + shadow, m)
        & outside_planes(c2 + shadow, m) & outside_planes(c3 + shadow, m);
    return outside == 0u;
}

const WORKGROUP: u32 = 256u;

var<workgroup> scan: array<u32, WORKGROUP>;
var<workgroup> node_count: u32;
var<workgroup> node_draw: u32;
// Indexed nodes: the batch's slots to rewrite (source index, or
// HIDDEN_INDEX to hide; count; destination; run tag plus the chunk's first
// vertex), the end of the node's last visible slot, and its counters.
var<workgroup> rewrite_src: array<u32, WORKGROUP>;
var<workgroup> rewrite_count: array<u32, WORKGROUP>;
var<workgroup> rewrite_dst: array<u32, WORKGROUP>;
var<workgroup> rewrite_tag: array<u32, WORKGROUP>;
var<workgroup> rewrites: atomic<u32>;
var<workgroup> extent: atomic<u32>;
var<workgroup> kept: atomic<u32>;
var<workgroup> written_indices: atomic<u32>;
var<workgroup> drawn_indices: atomic<u32>;

@compute @workgroup_size(256)
fn cull(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let node = node_cull[group.x];
    if lane == 0u {
        // No further than the list goes: a count gone stale can't loop the
        // workgroup past it.
        let listed = arrayLength(&candidates);
        node_count = min(node.candidate_count, listed - min(node.visible_base, listed));
        node_draw = node.draw;
    }
    let word = workgroupUniformLoad(&node_draw);
    let draw = word & 255u;
    // A hidden node draws nothing, and an indexed node's slots stay as
    // they are for when it shows.
    if (word & HIDDEN) != 0u {
        if lane == 0u {
            let at = group.x * ARGS_WORDS;
            for (var w = 0u; w < ARGS_WORDS; w++) {
                atomicStore(&args[at + w], 0u);
            }
        }
        return;
    }
    if draw == DRAW_INDEXED {
        cull_indexed(group.x, lane, node);
    } else {
        cull_listed(group.x, lane, node, draw);
    }
}

fn cull_listed(index: u32, lane: u32, node: NodeCull, draw: u32) {
    let points = draw == DRAW_POINTS;
    let count = workgroupUniformLoad(&node_count);
    var written = 0u;
    for (var start = 0u; start < count; start += WORKGROUP) {
        let i = start + lane;
        var keep = 0u;
        var candidate: Candidate;
        if i < count {
            candidate = candidates[node.visible_base + i];
            keep = select(0u, 1u, keep_all || candidate_visible(candidate, node.margin_px, draw, node.elevation, (node.draw & DRAPED) != 0u, (node.draw & UNCLIPPED) != 0u));
        }
        // Inclusive Hillis–Steele scan of `keep` across the workgroup.
        scan[lane] = keep;
        workgroupBarrier();
        for (var offset = 1u; offset < WORKGROUP; offset *= 2u) {
            var add = 0u;
            if lane >= offset {
                add = scan[lane - offset];
            }
            workgroupBarrier();
            scan[lane] += add;
            workgroupBarrier();
        }
        if keep == 1u {
            visible_out[node.visible_base + written + scan[lane] - 1u] = Visible(candidate.chunk, candidate.tile);
        }
        written += workgroupUniformLoad(&scan[WORKGROUP - 1u]);
        // Everyone has read the total before the next batch overwrites it.
        workgroupBarrier();
    }

    if lane == 0u {
        let at = index * ARGS_WORDS;
        let cells = point_cells((node.draw & SURFACE) != 0u);
        let vertices = select(0u, select(CHUNK_VERTICES, 6u * cells * cells, points), written > 0u);
        let instances = written * select(1u, CHUNK_POINTS, points);
        atomicStore(&args[at], vertices);
        atomicStore(&args[at + 1u], instances);
        atomicStore(&args[at + 2u], 0u);
        atomicStore(&args[at + 3u], 0u);
        atomicStore(&args[at + 4u], 0u);
        atomicStore(&args[at + 5u], written);
        atomicStore(&args[at + 6u], 0u);
        atomicStore(&args[at + 7u], vertices * instances);
    }
}

fn cull_indexed(index: u32, lane: u32, node: NodeCull) {
    let count = workgroupUniformLoad(&node_count);
    for (var start = 0u; start < count; start += WORKGROUP) {
        let i = start + lane;
        if i < count && candidates[node.visible_base + i].chunk != NO_CHUNK {
            let candidate = candidates[node.visible_base + i];
            let keep = select(0u, 1u, keep_all || candidate_visible(candidate, node.margin_px, DRAW_INDEXED, node.elevation, (node.draw & DRAPED) != 0u, (node.draw & UNCLIPPED) != 0u));
            let chunk = chunks[candidate.chunk];
            if keep == 1u {
                atomicMax(&extent, candidate.slot + chunk.index_count - node.index_base);
                atomicAdd(&kept, 1u);
                atomicAdd(&drawn_indices, chunk.index_count);
            }
            var drawn = candidate.drawn;
            if drawn == MOVED {
                // Hidden either way: this index, or the CPU driver's restart.
                let hidden = chunk.index_count == 0u || compacted[candidate.slot] >= HIDDEN_INDEX;
                drawn = select(1u, 0u, hidden);
                candidates[node.visible_base + i].drawn = drawn;
            }
            if drawn != keep {
                let at = atomicAdd(&rewrites, 1u);
                rewrite_src[at] = select(HIDDEN_INDEX, chunk.first_index, keep == 1u);
                // A chunk's indices are at most CHUNK_INDICES: a row gone
                // stale can't loop the copy below for long.
                rewrite_count[at] = min(chunk.index_count, CHUNK_INDICES);
                rewrite_dst[at] = candidate.slot;
                // A compacted index counts from the mesh's first vertex, and
                // the mesh's vertices fit below the run's bits.
                rewrite_tag[at] = (candidate.run << node.run_shift) + chunk.first_vertex;
                candidates[node.visible_base + i].drawn = keep;
                atomicAdd(&written_indices, chunk.index_count);
            }
        }
        workgroupBarrier();
        // Every lane copies a slot's indices in turn.
        let n = atomicLoad(&rewrites);
        for (var r = 0u; r < n; r++) {
            let src = rewrite_src[r];
            let dst = rewrite_dst[r];
            let tag = rewrite_tag[r];
            for (var k = lane; k < rewrite_count[r]; k += WORKGROUP) {
                // Arena indices never reach HIDDEN_INDEX.
                if src == HIDDEN_INDEX {
                    compacted[dst + k] = HIDDEN_INDEX;
                } else {
                    let i = src + k;
                    compacted[dst + k] = tag + ((indices[i >> 1u] >> ((i & 1u) << 4u)) & 0xffffu);
                }
            }
        }
        // Every lane has read this batch's rewrites before the next's.
        workgroupBarrier();
        if lane == 0u {
            atomicStore(&rewrites, 0u);
        }
        workgroupBarrier();
    }
    if lane == 0u {
        let at = index * ARGS_WORDS;
        // No further than the node's slots: a slot or chunk row gone stale
        // can't make the draw read past them (indirect calls aren't
        // validated natively, and billions of indices hang the GPU).
        let visible = min(atomicLoad(&extent), node.index_count);
        // A node with nothing in view keeps all-zero arguments: an indexed
        // indirect draw of no indices but one instance loses the device on
        // an Adreno 619 (Chrome 156, Vulkan).
        let drawn = visible > 0u;
        atomicStore(&args[at], visible);
        atomicStore(&args[at + 1u], select(0u, 1u, drawn));
        atomicStore(&args[at + 2u], select(0u, node.index_base, drawn));
        atomicStore(&args[at + 3u], 0u);
        atomicStore(&args[at + 4u], 0u);
        atomicStore(&args[at + 5u], atomicLoad(&kept));
        atomicStore(&args[at + 6u], atomicLoad(&written_indices));
        atomicStore(&args[at + 7u], atomicLoad(&drawn_indices));
    }
}
