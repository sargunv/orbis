// Every render node's uniform, by node (`symbol.wgsl`'s `halves`): the
// placement dispatch's, which places every symbol layer's quads at once.
@group(0) @binding(17) var<storage, read> halves: array<SymbolNode>;
