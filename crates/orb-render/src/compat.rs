//! The Compat cull (plan §8.3, §8.7): `cull.wgsl` on the CPU, over the
//! scene's CPU copies of its tables, writing what the dispatch writes on
//! Modern: pulled and point nodes' visible lists, indexed nodes' slots of
//! the compacted index buffer, and every node's draw arguments and
//! counters. The draws that read them are Modern's, emulated by
//! `orb-hal-gl`, so Compat records the same bundle.

use orb_hal::{BufferId, Gpu};
use orb_scene::{ARGS_WORDS, CullTables, FrameBlock, Scene};
use orb_shaders::layout::{candidate as cd, chunk as ck, node_cull as nc, tile as t, visible};

// `types.wgsl`'s draw words.
const DRAW_IMAGES: u32 = 0;
const DRAW_POINTS: u32 = 1;
const DRAW_INDEXED: u32 = 2;
const DRAW_INSTANCES: u32 = 3;
const DRAPED: u32 = 256;
const HIDDEN: u32 = 512;
/// `fetch_compat.wgsl`'s `HIDDEN_DRAW_INDEX`: WebGL rejects indices from
/// 2^31.
const HIDDEN_INDEX: u32 = 0x7fff_fffe;
const CHUNK_VERTICES: u32 = 768;
const TERRAIN_RANGE_UNIT: f32 = 4.0;

/// What a frame's cull wrote, for counters.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Culled {
    pub(crate) visible: u32,
    pub(crate) indices_written: u32,
}

fn unpack(w: u32) -> [f32; 2] {
    [
        f32::from(w as u16 as i16),
        f32::from((w >> 16) as u16 as i16),
    ]
}

/// Culls every node and uploads the results.
pub(crate) fn cull(gpu: &mut dyn Gpu, scene: &mut Scene, frame: &FrameBlock) -> Culled {
    let b = *scene.buffers();
    let tables = scene.cull_tables().expect("Compat scenes keep CPU tables");
    let nodes = tables.node_cull.len() / nc::WORDS;
    let mut args = vec![0u32; nodes * ARGS_WORDS];
    let mut culled = Culled::default();
    let mut visible = Vec::new();
    let mut slots = Vec::new();
    for node in 0..nodes {
        let n = &tables.node_cull[node * nc::WORDS..][..nc::WORDS];
        let node_args = &mut args[node * ARGS_WORDS..][..ARGS_WORDS];
        let draw = n[nc::DRAW] & 255;
        // Compat draws no models (plan §8.7).
        if n[nc::DRAW] & HIDDEN != 0 || draw == DRAW_INSTANCES {
            continue;
        }
        let cull = NodeCull {
            base: n[nc::VISIBLE_BASE],
            margin_px: f32::from_bits(n[nc::MARGIN_PX]),
            count: n[nc::CANDIDATE_COUNT],
            draw,
            draped: n[nc::DRAW] & DRAPED != 0,
            elevation: f32::from_bits(n[nc::ELEVATION]),
            index_base: n[nc::INDEX_BASE],
            run_shift: n[nc::RUN_SHIFT],
        };
        if draw == DRAW_INDEXED {
            cull_indexed(tables, frame, &cull, node_args, &mut slots);
            culled.indices_written += node_args[6];
        } else {
            visible.clear();
            cull_listed(tables, frame, &cull, node_args, &mut visible);
            culled.visible += node_args[5];
            if !visible.is_empty() {
                gpu.write_buffer(
                    b.visible,
                    u64::from(cull.base) * visible::WORDS as u64 * 4,
                    bytemuck::cast_slice(&visible),
                );
            }
        }
    }
    write_slots(gpu, b.compacted, &mut slots);
    if !args.is_empty() {
        gpu.write_buffer(b.args, 0, bytemuck::cast_slice(&args));
    }
    culled
}

/// A node's `NodeCull` row.
struct NodeCull {
    base: u32,
    margin_px: f32,
    count: u32,
    draw: u32,
    draped: bool,
    elevation: f32,
    index_base: u32,
    run_shift: u32,
}

fn cull_listed(
    tables: &CullTables,
    frame: &FrameBlock,
    node: &NodeCull,
    args: &mut [u32],
    visible: &mut Vec<u32>,
) {
    let points = node.draw == DRAW_POINTS;
    for i in 0..node.count {
        let c = &tables.candidates[(node.base + i) as usize * cd::WORDS..];
        let (chunk, tile) = (c[cd::CHUNK], c[cd::TILE]);
        if candidate_visible(tables, frame, chunk, tile, node) {
            visible.extend([chunk, tile]);
        }
    }
    let written = (visible.len() / visible::WORDS) as u32;
    let vertices = match (written, points) {
        (0, _) => 0,
        (_, true) => 6,
        (_, false) => CHUNK_VERTICES,
    };
    let instances = written * if points { 256 } else { 1 };
    // One draw's words: the rest of a node's are an instanced node's
    // second draw, and Compat draws no instanced nodes.
    args[..8].copy_from_slice(&[
        vertices,
        instances,
        0,
        0,
        0,
        written,
        0,
        vertices * instances,
    ]);
}

/// Index slots to rewrite: first slot, and the words to write there.
type Slots = Vec<(u32, Vec<u32>)>;

fn cull_indexed(
    tables: &mut CullTables,
    frame: &FrameBlock,
    node: &NodeCull,
    args: &mut [u32],
    slots: &mut Slots,
) {
    let (mut extent, mut kept, mut written, mut drawn) = (0u32, 0u32, 0u32, 0u32);
    for i in 0..node.count {
        let at = (node.base + i) as usize * cd::WORDS;
        let c = &tables.candidates[at..at + cd::WORDS];
        let (chunk, tile, slot, run, was) = (
            c[cd::CHUNK],
            c[cd::TILE],
            c[cd::SLOT],
            c[cd::RUN],
            c[cd::DRAWN],
        );
        let keep = candidate_visible(tables, frame, chunk, tile, node);
        let row = &tables.chunks[chunk as usize * ck::WORDS..];
        let (first, count) = (row[ck::FIRST_INDEX], row[ck::INDEX_COUNT]);
        if keep {
            extent = extent.max(slot + count - node.index_base);
            kept += 1;
            drawn += count;
        }
        if was != u32::from(keep) {
            let words = if keep {
                let tag = run << node.run_shift;
                tables.indices[first as usize..(first + count) as usize]
                    .iter()
                    .map(|&index| tag | index)
                    .collect()
            } else {
                vec![HIDDEN_INDEX; count as usize]
            };
            slots.push((slot, words));
            tables.candidates[at + cd::DRAWN] = u32::from(keep);
            written += count;
        }
    }
    // As on Modern, a node with nothing in view keeps all-zero arguments.
    if extent > 0 {
        args[..3].copy_from_slice(&[extent, 1, node.index_base]);
    }
    args[5] = kept;
    args[6] = written;
    args[7] = drawn;
}

/// Writes rewritten slots, merging adjacent ones into one write.
fn write_slots(gpu: &mut dyn Gpu, compacted: BufferId, slots: &mut Slots) {
    slots.sort_unstable_by_key(|(slot, _)| *slot);
    let mut run: Option<(u32, Vec<u32>)> = None;
    for (slot, words) in slots.drain(..) {
        match &mut run {
            Some((start, buffer)) if *start + buffer.len() as u32 == slot => {
                buffer.extend(words);
            }
            _ => {
                if let Some((start, buffer)) = run.take() {
                    gpu.write_buffer(
                        compacted,
                        u64::from(start) * 4,
                        bytemuck::cast_slice(&buffer),
                    );
                }
                run = Some((slot, words));
            }
        }
    }
    if let Some((start, buffer)) = run {
        gpu.write_buffer(
            compacted,
            u64::from(start) * 4,
            bytemuck::cast_slice(&buffer),
        );
    }
}

/// `cull.wgsl`'s `outside_planes`.
fn outside_planes(p: [f32; 4], margin: [f32; 2]) -> u32 {
    let extra = [margin[0] * p[3].max(0.0), margin[1] * p[3].max(0.0)];
    u32::from(p[0] > p[3] + extra[0])
        | u32::from(p[0] < -p[3] - extra[0]) << 1
        | u32::from(p[1] > p[3] + extra[1]) << 2
        | u32::from(p[1] < -p[3] - extra[1]) << 3
        | u32::from(p[2] > p[3]) << 4
        | u32::from(p[2] < 0.0) << 5
}

fn transform(m: &[[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (col, &x) in m.iter().zip(&v) {
        for (o, &c) in out.iter_mut().zip(col) {
            *o += c * x;
        }
    }
    out
}

/// `cull.wgsl`'s `candidate_visible`.
fn candidate_visible(
    tables: &CullTables,
    frame: &FrameBlock,
    chunk: u32,
    tile: u32,
    node: &NodeCull,
) -> bool {
    let c = &tables.chunks[chunk as usize * ck::WORDS..][..ck::WORDS];
    let row = &tables.tiles[tile as usize * t::WORDS..][..t::WORDS];
    let f = |i: usize| f32::from_bits(row[i]);
    // The row's origin less the camera, in f64 where the GPU subtracts
    // whole quanta and remainders apart.
    let quantum = orb_scene::quantum(frame.world_span);
    let origin = [0, 1].map(|i| {
        let world =
            f64::from(row[t::ORIGIN + i] as i32) * quantum + f64::from(f(t::ORIGIN_REST + i));
        (world - frame.camera[i]) as f32
    });
    let (scale, world_per_px) = (f(t::SCALE), frame.world_per_px * f(t::PX_SCALE));
    let points = node.draw == DRAW_POINTS;
    let margin = node.margin_px * world_per_px / scale;
    let before = if points { 0.0 } else { margin };
    let (bmin, bmax) = (unpack(c[ck::BOUNDS_MIN]), unpack(c[ck::BOUNDS_MAX]));
    let mut lo = [bmin[0] - before, bmin[1] - before];
    let mut hi = [bmax[0] + before, bmax[1] + before];
    let (mut clip_lo, mut clip_hi) = if node.draw == DRAW_IMAGES {
        (lo, hi)
    } else {
        ([1e9; 2], [-1e9; 2])
    };
    if node.draw != DRAW_IMAGES {
        for i in 0..4 {
            let clip = t::CLIPS + 2 * i;
            let (r_lo, r_hi) = (unpack(row[clip]), unpack(row[clip + 1]));
            let c_lo = [lo[0].max(r_lo[0]), lo[1].max(r_lo[1])];
            let c_hi = [hi[0].min(r_hi[0]), hi[1].min(r_hi[1])];
            if c_lo[0] <= c_hi[0] && c_lo[1] <= c_hi[1] {
                clip_lo = [clip_lo[0].min(c_lo[0]), clip_lo[1].min(c_lo[1])];
                clip_hi = [clip_hi[0].max(c_hi[0]), clip_hi[1].max(c_hi[1])];
            }
        }
    }
    if clip_lo[0] > clip_hi[0] || clip_lo[1] > clip_hi[1] {
        return false;
    }
    // As on Modern, warped tiles' chunks pass: cover kept only tiles whose
    // footprints meet the view.
    if node.draped || row[t::WARP_CELLS] != 0 || row[t::GLOBE_CELLS] != 0 {
        return true;
    }
    let after = if points { margin } else { 0.0 };
    lo = [clip_lo[0] - after, clip_lo[1] - after];
    hi = [clip_hi[0] + after, clip_hi[1] + after];
    let range = unpack(row[t::TERRAIN_RANGE]);
    let ground = [
        range[0] * TERRAIN_RANGE_UNIT * frame.extrusion_scale,
        range[1] * TERRAIN_RANGE_UNIT * frame.extrusion_scale,
    ];
    let world = |p: [f32; 2]| [origin[0] + p[0] * scale, origin[1] - p[1] * scale];
    let corner = |p: [f32; 2]| {
        let w = world(p);
        transform(&frame.view_proj, [w[0], w[1], ground[0], 1.0])
    };
    let corners = [
        corner(lo),
        corner([hi[0], lo[1]]),
        corner([lo[0], hi[1]]),
        corner(hi),
    ];
    let raised = transform(
        &frame.view_proj,
        [
            0.0,
            0.0,
            ground[1] - ground[0] + node.elevation * frame.extrusion_scale,
            0.0,
        ],
    );
    let m = [
        2.0 * node.margin_px / frame.viewport[0],
        2.0 * node.margin_px / frame.viewport[1],
    ];
    let mut outside = u32::MAX;
    for p in corners {
        let up = [
            p[0] + raised[0],
            p[1] + raised[1],
            p[2] + raised[2],
            p[3] + raised[3],
        ];
        outside &= outside_planes(p, m) & outside_planes(up, m);
    }
    outside == 0
}
