// The symbol draw's quads on a device without compute (WebGL 2): each
// corner places its quad itself (`place_quad`), the draw's node uniforms
// its layer's halves.
fn placement(node: NodeCull, instance: u32) -> PlacedQuad {
    return place_quad(node, 0u, instance);
}
