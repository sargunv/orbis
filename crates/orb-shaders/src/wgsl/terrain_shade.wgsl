// A terrain layer that shades hillshades between its stacks: lit from the
// frame's bearing on every frame, so no drape slot turns with the map. A
// hillshade draws its slopes, unlit, into a stack of its own on the slope
// page (`hillshade_layer.wgsl`'s `fs_hillshade_slopes`), from whatever
// tiles its source draws there, as any draped layer does, and its table
// (`hillshade.wgsl`) colours them: group 2 binds the layer's hillshades'
// tables, in order. Each hillshade lies over what is beneath it in the
// layer, blending into it in its blend mode, under the stack above it.

@group(2) @binding(0) var table_sampler: sampler;
@group(2) @binding(1) var first_table: texture_2d<f32>;
@group(2) @binding(2) var second_table: texture_2d<f32>;
// The terrain layer before this one, as it drew, where the layer's first
// hillshade blends into it (`layer.backdrop`).
@group(3) @binding(0) var backdrop: texture_2d<f32>;

// The layer's hillshade `k` at the fragment, from its slopes' slot at
// `at`: premultiplied by how much of the footprint the source covers.
fn shade_color(in: TerrainOut, at: DrapeSample, k: u32) -> vec4<f32> {
    let slopes = slope_texel(in, at, layer.shades[k].slopes);
    if slopes.a < 0.5 / 255.0 {
        return vec4<f32>(0.0);
    }
    let slope = hillshade_slope(vec4<f32>(slopes.rg / slopes.a, 1.0, 1.0), in.stretch);
    let uv = hillshade_table_uv(slope);
    if k == 0u {
        return textureSampleLevel(first_table, table_sampler, uv, 0.0) * slopes.a;
    }
    return textureSampleLevel(second_table, table_sampler, uv, 0.0) * slopes.a;
}

@fragment
fn fs_terrain_shaded(in: TerrainOut) -> @location(0) vec4<f32> {
    let at = drape_sample(in, terrain.slot_px);
    let slopes_at = drape_sample(in, terrain.slope_px);
    if layer.backdrop != 0u {
        // The layer before, fogged already, under the hillshades blended
        // into it, faded as the fog thickens, which takes their place, and
        // the stacks above them fogged.
        var color = textureLoad(backdrop, vec2<i32>(in.position.xy), 0);
        let fade = 1.0 - atmosphere_fog_factor(atmosphere, in.fog_depth);
        for (var k = 0u; k < 2u && layer.shades[k].slopes != NO_STACK; k++) {
            let shade = layer.shades[k];
            color = blend_over(shade.blend, shade_color(in, slopes_at, k) * fade, color);
            color = over(fogged(in, drape_color(in, at, shade.above)), color);
        }
        return color;
    }
    var color = drape_color(in, at, layer.below);
    for (var k = 0u; k < 2u && layer.shades[k].slopes != NO_STACK; k++) {
        let shade = layer.shades[k];
        color = blend_over(shade.blend, shade_color(in, slopes_at, k), color);
        color = over(drape_color(in, at, shade.above), color);
    }
    return fogged(in, color);
}
