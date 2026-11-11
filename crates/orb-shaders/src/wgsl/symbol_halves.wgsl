// A symbol layer's node uniforms, its icons' and its text's, which its
// icon node binds (`symbol.wgsl`'s `halves`): the draw's and the drawn
// placement pass's.
@group(1) @binding(0) var<uniform> halves: array<SymbolNode, 2>;
