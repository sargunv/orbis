// A terrain layer that shades a hillshade between its stacks: lit from the
// frame's bearing on every frame, so no drape slot turns with the map. The
// hillshade draws its slopes, unlit, into a stack of their own on the
// slope page
// (`hillshade_layer.wgsl`'s `fs_hillshade_slopes`), from whatever tiles its
// source draws there, as any draped layer does. Its node group is group 2:
// its paint (`hillshade.wgsl`'s `node`).

@group(2) @binding(0) var<uniform> node: HillshadeNode;

// The hillshade at the fragment, from its slopes' slot: premultiplied by
// how much of the footprint the source covers.
fn shade_color(in: TerrainOut) -> vec4<f32> {
    let slopes = slope_texel(in, drape_sample(in, terrain.slope_px), layer.slopes);
    if slopes.a < 0.5 / 255.0 {
        return vec4<f32>(0.0);
    }
    let pixel = vec4<f32>(slopes.rg / slopes.a, 1.0, 1.0);
    return hillshade_color(pixel, in.stretch) * slopes.a;
}

@fragment
fn fs_terrain_shaded(in: TerrainOut) -> @location(0) vec4<f32> {
    let at = drape_sample(in, terrain.slot_px);
    let below = drape_color(in, at, layer.below);
    let above = drape_color(in, at, layer.above);
    return fogged(in, over(above, over(shade_color(in), below)));
}
