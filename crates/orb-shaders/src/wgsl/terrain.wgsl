// The terrain surface: one instance per terrain tile of a
// shared grid, raised by its DEM tile, coloured from the tile's drape slots
// for the terrain layer being drawn, fogged by the atmosphere like
// GL JS's `terrain.fragment.glsl`. On a globe it lies on the sphere through the
// tile's patches in space (`project.wgsl`'s `sphere_at`), raised along the
// sphere's up, and its top and bottom rows' skirts close the poles.
//
// Drape slots are squares of the drape atlas's layers, `slot_px` texels
// per side, `columns` per row; a layer holds `per_layer` colour stacks,
// and a tile's slot for colour stack `s` is slot
// `s % per_layer * stride + slot` of layer `s / per_layer`, each stack's
// slots side by side. Slope stacks are on a page of their own, at
// `slope_px` texels a side, likewise. A
// terrain layer draws a stack's slots, two stacks' where the zoom splits
// it, or two around a hillshade it shades between them from its slopes
// (`terrain_shade.wgsl`).
// A tile's slots are the top-left square of their place, its `shrink`
// halving their page's side, which a tile that shows small draws few texels
// of. Mip levels never mix slots, and samples keep to the slot at the level
// they read, as a texture of its own would clamp to its edge.

struct TerrainTile {
    // The tile's north-west corner in camera-relative world space.
    origin: vec2<f32>,
    // World units per terrain unit; a tile spans TERRAIN_EXTENT of them.
    scale: f32,
    // Page layer of its DEM tile, or NO_DEM.
    dem_layer: u32,
    // Tile uv → DEM tile uv: `uv * dem_scale + dem_offset`.
    dem_offset: vec2<f32>,
    dem_scale: f32,
    // Its block of drape slots, and how many times they halve their page's
    // side.
    slot: u32,
    shrink: u32,
    // On a globe: its patches in space (`project.wgsl`'s layout, from word
    // `globe_base` of the props) around `globe_origin`, camera-relative
    // metres, and the poles its skirts close (bit 0 north, bit 1 south).
    globe_origin: array<f32, 3>,
    globe_base: u32,
    globe_cells: u32,
    poles: u32,
    // Latitudes of its top and bottom edges, in degrees, for a hillshade
    // it shades.
    latitudes: vec2<f32>,
};

struct Terrain {
    // GL JS's unpack vector: metres = `r·x + g·y + b·z − w`.
    unpack: vec4<f32>,
    exaggeration: f32,
    // DEM pixels per side, without the 2-texel border.
    dem_dim: f32,
    // World units the skirts hang below the edges.
    skirt: f32,
    // Colour stacks a layer, slots between a tile's colour stacks in a
    // layer, and the colour page's slots per row.
    per_layer: u32,
    stride: u32,
    columns: u32,
    slot_px: f32,
    // Terrain tiles drawn.
    tiles: u32,
    // The same for the slope page's, and its slots' side.
    slope_per_layer: u32,
    slope_stride: u32,
    slope_columns: u32,
    slope_px: f32,
};

// A terrain layer: the colour stacks drawn below and above its shaded
// hillshade, the slope stack (by its rank on the slope page) holding
// that hillshade's slopes, or NO_STACK, and how it fogs: FOG_CONTENT, or
// FOG_GROUND for the map's first terrain layer, the ground, which fogs
// where it draws nothing too, or FOG_FADES for a layer composited in a
// blend mode, over ground fogged already; the blend mode its shaded
// hillshade blends into the stack below in (`blend.wgsl`), and 1 where it
// blends into the terrain layer before it instead (`terrain_shade.wgsl`).
struct Layer {
    below: u32,
    above: u32,
    slopes: u32,
    fog: u32,
    shade_blend: u32,
    backdrop: u32,
    pad1: u32,
    pad2: u32,
};

const FOG_GROUND: u32 = 1u;
const FOG_FADES: u32 = 2u;

const TERRAIN_EXTENT: f32 = 8192.0;
const NO_DEM: u32 = 0xffffffffu;
const NO_STACK: u32 = 0xffffffffu;
// Grid cells per tile side.
const GRID: u32 = 128u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(3) var dem: texture_2d_array<f32>;
@group(0) @binding(4) var drape: texture_2d_array<f32>;
@group(0) @binding(5) var drape_sampler: sampler;
@group(0) @binding(6) var<uniform> terrain: Terrain;
@group(0) @binding(7) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(9) var slope_atlas: texture_2d_array<f32>;
@group(1) @binding(0) var<uniform> layer: Layer;

fn dem_texel(p: vec2<i32>, layer: u32) -> f32 {
    let hi = vec2<i32>(textureDimensions(dem)) - 1;
    let texel = textureLoad(dem, clamp(p, vec2<i32>(0), hi), layer, 0);
    return dem_elevation(texel, terrain.unpack);
}

// GL JS's `get_elevation`: bilinear over the padded DEM, pixel i at
// texel i + 2, in metres with exaggeration.
fn terrain_elevation(tile: TerrainTile, uv: vec2<f32>) -> f32 {
    if tile.dem_layer == NO_DEM {
        return 0.0;
    }
    let coord = (uv * tile.dem_scale + tile.dem_offset) * terrain.dem_dim + 1.5;
    let f = fract(coord);
    let c = vec2<i32>(floor(coord));
    let tl = dem_texel(c, tile.dem_layer);
    let tr = dem_texel(c + vec2<i32>(1, 0), tile.dem_layer);
    let bl = dem_texel(c + vec2<i32>(0, 1), tile.dem_layer);
    let br = dem_texel(c + vec2<i32>(1, 1), tile.dem_layer);
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y) * terrain.exaggeration;
}

struct TerrainOut {
    @builtin(position) @invariant position: vec4<f32>,
    // Tile uv, y down.
    @location(0) uv: vec2<f32>,
    // The tile's index, its block of drape slots and their shrink.
    @location(1) @interpolate(flat, either) tile: vec3<u32>,
    @location(2) fog_depth: f32,
    // Mercator's slope stretch for a hillshade the layer shades
    // (`terrain_shade.wgsl`): per vertex rather than per fragment, which
    // a grid cell's smooth change of latitude doesn't show.
    @location(3) stretch: f32,
};

@vertex
fn vs_terrain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> TerrainOut {
    let tile = terrain_tile_at(instance);
    let g = grid_at(vertex);
    let cell = vec2<f32>(f32(g & 255u), f32((g >> 8u) & 255u));
    let skirt = (g >> 16u) & 1u;
    let uv = cell / f32(GRID);
    let elevation = terrain_elevation(tile, uv);
    var z = elevation * frame.extrusion_scale - f32(skirt) * terrain.skirt;
    let world = tile.origin + vec2<f32>(uv.x, -uv.y) * TERRAIN_EXTENT * tile.scale;
    var clip = frame.view_proj * vec4<f32>(world, z, 1.0);
    let globe = frame.globe_amount > 0.0 && tile.globe_cells > 0u;
    if globe {
        var g: Ground;
        g.flat = world;
        g.on_plane = true;
        let origin = vec3<f32>(tile.globe_origin[0], tile.globe_origin[1], tile.globe_origin[2]);
        g.globe = sphere_at(tile.globe_base, tile.globe_cells, origin, uv);
        // A top or bottom row's skirt reaches the pole its set stops short
        // of, closing the globe with the edge's colour, as the layers' caps
        // do.
        let north = cell.y == 0.0 && (tile.poles & 1u) != 0u;
        let south = cell.y == f32(GRID) && (tile.poles & 2u) != 0u;
        if skirt != 0u && (north || south) && frame.globe_plane == 0u {
            g.globe = vec3<f32>(0.0, 0.0, select(-SPHERE_RADIUS, SPHERE_RADIUS, north)) - frame.globe_target;
            z = 0.0;
        }
        clip = ground_clip_3d(g, vec2<f32>(0.0), z);
    }
    var out: TerrainOut;
    // Reversed-Z, as extrusions draw (`ground_clip_reversed`).
    let depth = select(plane_depth(world, z), clip.w - clip.z, globe);
    out.position = vec4<f32>(clip.xy, depth, clip.w);
    out.uv = uv;
    out.tile = vec3<u32>(instance, tile.slot, tile.shrink);
    out.stretch = 1.0 / cos(radians(mix(tile.latitudes.x, tile.latitudes.y, uv.y)));
    // The view depth is the clip w, mixed as the globe hands over: fog
    // fades in as it does (`fog_opacity`).
    out.fog_depth = atmosphere_fog_depth(atmosphere, clip.w);
    return out;
}

// The terrain's depth, for the symbols and circles it hides: the
// reversed-Z depth the stacks draw, as a colour.
@fragment
fn fs_terrain_depth(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(position.z, 0.0, 0.0, 0.0);
}

// Where a fragment samples its tile's drape slots on a page of `slot_px`
// texels a slot: the texel in a slot and the footprint's gradients, the
// same in every slot of the page and side (the colour page's, or the slope
// page's, shrunk as the tile's). Trilinear and
// anisotropic, as GL JS samples its render-to-texture tiles, and kept in
// the slot as a texture of its own would clamp to its edge.
struct DrapeSample {
    texel: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
};

fn drape_sample(in: TerrainOut, page_px: f32) -> DrapeSample {
    // Exact: a power of two over one.
    let slot_px = page_px / f32(1u << in.tile.z);
    let texel = in.uv * slot_px;
    let dx = dpdx(texel);
    let dy = dpdy(texel);
    // A texel of the level the sampler reads: the footprint's longer side
    // over up to 16 probes along it, no finer than its shorter side.
    let level_texel = max(max(length(dx), length(dy)) / 16.0, min(length(dx), length(dy)));
    // Each probe reads half a texel of the coarser level around it: the
    // sample keeps that far from the slot's edge. The probes span the
    // footprint: near the edge it shrinks until they stay in the slot,
    // rather than the sample moving further off the edge texels.
    let half = 0.5 * exp2(ceil(clamp(log2(level_texel), 0.0, log2(slot_px) - 1.0)));
    let p = clamp(texel, vec2<f32>(half), vec2<f32>(slot_px - half));
    let room = min(p, vec2<f32>(slot_px) - p) - half;
    let need = 0.5 * max(abs(dx), abs(dy));
    let fit = min(1.0, min(room.x / max(need.x, 1e-6), room.y / max(need.y, 1e-6)));
    // On a 1/512-texel grid, finer than samplers resolve: added to any
    // slot's corner in a page of up to 32768 texels a side, exact in f32,
    // so where a tile's slot lies never moves its samples.
    return DrapeSample(round(p * 512.0) / 512.0, dx * fit, dy * fit);
}

// Colour stack `stack`'s slot colour at `at`, none for NO_STACK.
fn drape_color(in: TerrainOut, at: DrapeSample, stack: u32) -> vec4<f32> {
    if stack == NO_STACK {
        return vec4<f32>(0.0);
    }
    let slot = stack % terrain.per_layer * terrain.stride + in.tile.y;
    let size = vec2<f32>(textureDimensions(drape));
    let corner = vec2<f32>(f32(slot % terrain.columns), f32(slot / terrain.columns)) * terrain.slot_px;
    let uv = (corner + at.texel) / size;
    return textureSampleGrad(drape, drape_sampler, uv, stack / terrain.per_layer, at.dx / size, at.dy / size);
}

// Slope stack `rank`'s slot at `at` (a sample at `slope_px`).
fn slope_texel(in: TerrainOut, at: DrapeSample, rank: u32) -> vec4<f32> {
    let slot = rank % terrain.slope_per_layer * terrain.slope_stride + in.tile.y;
    let size = vec2<f32>(textureDimensions(slope_atlas));
    let corner = vec2<f32>(f32(slot % terrain.slope_columns), f32(slot / terrain.slope_columns)) * terrain.slope_px;
    let uv = (corner + at.texel) / size;
    return textureSampleGrad(slope_atlas, drape_sampler, uv, rank / terrain.slope_per_layer, at.dx / size, at.dy / size);
}

// `top` over `bottom`, premultiplied.
fn over(top: vec4<f32>, bottom: vec4<f32>) -> vec4<f32> {
    return top + (1.0 - top.a) * bottom;
}

@fragment
fn fs_terrain(in: TerrainOut) -> @location(0) vec4<f32> {
    let at = drape_sample(in, terrain.slot_px);
    return fogged(in, over(drape_color(in, at, layer.above), drape_color(in, at, layer.below)));
}

// The layer's colour as the atmosphere fogs it: the ground whole, a layer
// over it only where it draws (`atmosphere_fog`), and a blended one fading
// out as the fog thickens, leaving the fogged ground it blends with.
fn fogged(in: TerrainOut, color: vec4<f32>) -> vec4<f32> {
    if layer.fog == FOG_FADES {
        return color * (1.0 - atmosphere_fog_factor(atmosphere, in.fog_depth));
    }
    return atmosphere_fog(atmosphere, color, in.fog_depth, layer.fog == FOG_GROUND);
}
