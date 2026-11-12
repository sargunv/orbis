// A terrain layer that shades a hillshade between its stacks: lit from the
// frame's bearing on every frame, so no drape slot turns with the map. The
// hillshade draws its slopes, unlit, into a stack of their own on the
// slope page
// (`hillshade_layer.wgsl`'s `fs_hillshade_slopes`), from whatever tiles its
// source draws there, as any draped layer does. Its node group is group 2:
// its paint (`hillshade.wgsl`'s `node`). A hillshade that blends other than
// normally blends into the colours below it, which are what it lies on.

@group(2) @binding(0) var<uniform> node: HillshadeNode;
// The terrain layer before this one, as it drew, where the hillshade
// blends into it (`layer.backdrop`).
@group(3) @binding(0) var backdrop: texture_2d<f32>;

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
    let above = drape_color(in, at, layer.above);
    if layer.backdrop != 0u {
        // The layer before, fogged already, under the hillshade blended
        // into it, faded as the fog thickens, which takes its place.
        let beneath = textureLoad(backdrop, vec2<i32>(in.position.xy), 0);
        let fade = 1.0 - atmosphere_fog_factor(atmosphere, in.fog_depth);
        let shaded = blend_over(layer.shade_blend, shade_color(in) * fade, beneath);
        return over(fogged(in, above), shaded);
    }
    let below = drape_color(in, at, layer.below);
    let shaded = blend_over(layer.shade_blend, shade_color(in), below);
    return fogged(in, over(above, shaded));
}
