// The symbol placements, one dispatch for every symbol layer, point and
// line labels alike: a workgroup for each 64 texels of the placement
// target, the quads of one of a symbol node's visible chunks
// (`place_quad`). The workgroups of chunks a node's cull didn't keep this
// frame leave at once.

// The cull's draw arguments and counters, ARGS_WORDS a node (`cull.wgsl`):
// at 5 the chunks it kept.
@group(0) @binding(18) var<storage, read> cull_args: array<u32>;
@group(0) @binding(19) var placed_out: texture_storage_2d_array<rgba32uint, write>;

const ARGS_WORDS: u32 = 8u;

var<workgroup> place_node: u32;

@compute @workgroup_size(64)
fn cs_place(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let first = group.x * CHUNK_QUADS;
    if lane == 0u {
        // The node whose texels these are: the last whose `quad_base` is
        // at most the first's. Every node has one, the texels the symbol
        // nodes before it take, so the bases rise with the node.
        var lo = 0u;
        var hi = max(frame.placed_nodes, 1u);
        for (var guard = 0u; guard < 32u && hi - lo > 1u; guard++) {
            let mid = (lo + hi) / 2u;
            if node_cull_at(mid).quad_base <= first {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        place_node = lo;
    }
    let n = workgroupUniformLoad(&place_node);
    let node = node_cull_at(n);
    let instance = first - node.quad_base + lane;
    if instance / CHUNK_QUADS >= cull_args[n * ARGS_WORDS + 5u] { return; }
    let placed = place_quad(node, n, instance);
    if placed.written {
        let at = placed_texel_at(first + lane);
        textureStore(placed_out, at, 0u, placed.at);
        textureStore(placed_out, at, 1u, placed.paint);
    }
}
