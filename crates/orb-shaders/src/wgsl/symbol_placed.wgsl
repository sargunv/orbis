// The symbol draw's quads as the placement dispatch placed them
// (`symbol_place_compute.wgsl`): their texels of the placement target,
// bound beside the pass rather than in the scene, as the dispatch binds
// the scene and writes the target.
@group(2) @binding(1) var symbol_placed: texture_2d_array<u32>;

// Quad `instance` of symbol `node`'s draw. Nothing of a quad with zero
// opacity shows, which most of a dense view's have: its other texel is
// left unread.
fn placement(node: NodeCull, instance: u32) -> PlacedQuad {
    var out: PlacedQuad;
    let at = placed_texel_at(node.quad_base + instance);
    out.paint = textureLoad(symbol_placed, at, 1u, 0);
    if bitcast<f32>(out.paint.y) > 0.0 {
        out.at = textureLoad(symbol_placed, at, 0u, 0);
        out.written = true;
    }
    return out;
}
