// A terrain layer that shades a hillshade between its stacks (decision
// 0019): lit from the frame's bearing on every frame, so no drape slot
// turns with the map. The hillshade's node group is group 2: its paint
// (`hillshade.wgsl`'s `node`) and its source's prepared slopes. Each
// quarter of a terrain tile reads one slope tile (raster covers refine a
// zoom deeper than terrain tiles): `shade_tiles[(shade * tiles + tile) * 4
// + quarter]`, quarters west to east, then north to south.

struct ShadeTile {
    // Quarter uv → slope tile uv: `uv * scale + offset`.
    offset: vec2<f32>,
    // Latitudes of the slope tile's top and bottom edges, in degrees.
    latitudes: vec2<f32>,
    scale: f32,
    // The slope tile's layer, or NO_SLOPES.
    layer: u32,
};

const NO_SLOPES: u32 = 0xffffffffu;

@group(0) @binding(9) var<storage, read> shade_tiles: array<ShadeTile>;
@group(2) @binding(0) var<uniform> node: HillshadeNode;
@group(2) @binding(1) var slopes: texture_2d_array<f32>;
@group(2) @binding(2) var slope_linear: sampler;
@group(2) @binding(3) var slope_nearest: sampler;

// The hillshade at the fragment, from mipmapped slopes in the distance.
fn shade_color(in: TerrainOut) -> vec4<f32> {
    let ddx = dpdx(in.uv);
    let ddy = dpdy(in.uv);
    let q = min(vec2<u32>(in.uv * 2.0), vec2<u32>(1u));
    let shade = shade_tiles[(layer.shade * terrain.tiles + in.tile.x) * 4u + q.x + 2u * q.y];
    if shade.layer == NO_SLOPES {
        return vec4<f32>(0.0);
    }
    let uv = (in.uv * 2.0 - vec2<f32>(q)) * shade.scale + shade.offset;
    let size = vec2<f32>(textureDimensions(slopes));
    let at = (uv * (size - 2.0) + 1.0) / size;
    // Slope uv per terrain tile uv.
    let k = 2.0 * shade.scale * (size - 2.0) / size;
    var pixel: vec4<f32>;
    if node.nearest != 0u {
        pixel = textureSampleLevel(slopes, slope_nearest, at, shade.layer, 0.0);
    } else {
        pixel = textureSampleGrad(slopes, slope_linear, at, shade.layer, ddx * k, ddy * k);
    }
    return hillshade_color(pixel, mix(shade.latitudes.x, shade.latitudes.y, uv.y));
}

@fragment
fn fs_terrain_shaded(in: TerrainOut) -> @location(0) vec4<f32> {
    let below = drape_color(in, layer.below);
    let above = drape_color(in, layer.above);
    return fogged(in, over(above, over(shade_color(in), below)));
}
