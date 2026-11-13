// A terrain layer that shades hillshades between its stacks: lit from the
// frame's bearing on every frame, so no drape slot turns with the map. A
// hillshade draws its slopes, unlit, into a stack of its own on the slope
// page (`hillshade_layer.wgsl`'s `fs_hillshade_slopes`), from whatever
// tiles its source draws there, as any draped layer does, and its table
// (`hillshade.wgsl`) colours them: group 2 binds the layer's hillshades'
// tables, in order. Each hillshade lies over what is beneath it in the
// layer, blending into it in its blend mode, under the stack above it.
//
// A layer that reads the map beneath it (`fs_terrain_beneath`) draws in
// its place rather than over it: one whose first hillshade blends into
// the map, or one that blends with the map in a mode of its own.

@group(2) @binding(0) var table_sampler: sampler;
@group(2) @binding(1) var first_table: texture_2d<f32>;
@group(2) @binding(2) var second_table: texture_2d<f32>;
// The map beneath the layer, as the pass before drew it.
@group(3) @binding(0) var beneath: texture_2d<f32>;

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

// The layer's colour: its stack below, each hillshade over it in its
// blend mode, under the stack above that.
fn shaded_color(in: TerrainOut) -> vec4<f32> {
    let at = drape_sample(in, terrain.slot_px);
    let slopes_at = drape_sample(in, terrain.slope_px);
    var color = drape_color(in, at, layer.below);
    for (var k = 0u; k < 2u && layer.shades[k].slopes != NO_STACK; k++) {
        let shade = layer.shades[k];
        color = blend_over(shade.blend, shade_color(in, slopes_at, k), color);
        color = over(drape_color(in, at, shade.above), color);
    }
    // Without a hillshade, the stack above where the zoom splits the layer.
    return over(drape_color(in, at, layer.above), color);
}

@fragment
fn fs_terrain_shaded(in: TerrainOut) -> @location(0) vec4<f32> {
    return fogged(in, shaded_color(in)) * layer.opacity;
}

@fragment
fn fs_terrain_beneath(in: TerrainOut) -> @location(0) vec4<f32> {
    let map = textureLoad(beneath, vec2<i32>(in.position.xy), 0);
    if layer.backdrop == 0u {
        // In the layer's blend mode, fading into the fog: what it blends
        // with is fogged already.
        return blend_over(layer.blend, fogged(in, shaded_color(in)) * layer.opacity, map);
    }
    // The map, fogged already, under the hillshades blended into it,
    // faded as the fog thickens, which takes their place, and the stacks
    // above them fogged.
    let at = drape_sample(in, terrain.slot_px);
    let slopes_at = drape_sample(in, terrain.slope_px);
    let fade = 1.0 - atmosphere_fog_factor(atmosphere, in.fog_depth);
    var color = map;
    for (var k = 0u; k < 2u && layer.shades[k].slopes != NO_STACK; k++) {
        let shade = layer.shades[k];
        color = blend_over(shade.blend, shade_color(in, slopes_at, k) * fade, color);
        color = over(fogged(in, drape_color(in, at, shade.above)), color);
    }
    return color;
}
