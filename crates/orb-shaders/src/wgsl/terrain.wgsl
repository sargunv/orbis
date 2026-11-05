// The terrain surface (plan §9.4): one instance per terrain tile of a
// shared grid, raised by its DEM tile, coloured from the tile's drape slot
// for the stack being drawn, fogged by the atmosphere like GL JS's
// `terrain.fragment.glsl`. On a globe it lies on the sphere through the
// tile's patches in space (`project.wgsl`'s `sphere_at`), raised along the
// sphere's up, and its top and bottom rows' skirts close the poles.
//
// Drape slots are squares of the drape atlas, `slot_px` texels per side,
// `columns` per row; a tile's slot for stack `s` is `slot * stacks + s`.
// Mip levels never mix slots, and samples clamp to the slot at the level
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
    // Its block of drape slots.
    slot: u32,
    // On a globe: its patches in space (`globe_cells` per side from word
    // `globe_base` of the props) around `globe_origin`, camera-relative
    // metres, and the poles its skirts close (bit 0 north, bit 1 south).
    globe_origin: array<f32, 3>,
    globe_base: u32,
    globe_cells: u32,
    poles: u32,
};

struct Terrain {
    // GL JS's unpack vector: metres = `r·x + g·y + b·z − w`.
    unpack: vec4<f32>,
    exaggeration: f32,
    // DEM pixels per side, without the 2-texel border.
    dem_dim: f32,
    // World units the skirts hang below the edges.
    skirt: f32,
    stacks: u32,
    columns: u32,
    slot_px: f32,
    // Mip levels of the atlas.
    levels: f32,
};

struct Stack {
    index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

const TERRAIN_EXTENT: f32 = 8192.0;
const NO_DEM: u32 = 0xffffffffu;
// Grid cells per tile side.
const GRID: u32 = 128u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> terrain_tiles: array<TerrainTile>;
// Grid vertices: x and y in cells (8 bits each), then a skirt bit.
@group(0) @binding(2) var<storage, read> grid: array<u32>;
@group(0) @binding(3) var dem: texture_2d_array<f32>;
@group(0) @binding(4) var drape: texture_2d<f32>;
@group(0) @binding(5) var drape_sampler: sampler;
@group(0) @binding(6) var<uniform> terrain: Terrain;
@group(0) @binding(7) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(8) var<storage, read> props: array<u32>;
@group(1) @binding(0) var<uniform> stack: Stack;

fn prop_at(i: u32) -> u32 {
    return props[i];
}

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
    @location(1) @interpolate(flat) slot: u32,
    @location(2) fog_depth: f32,
    // The surface's normal for the lights, in their axes (lighting.wgsl).
    @location(3) normal: vec3<f32>,
};

@vertex
fn vs_terrain(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> TerrainOut {
    let tile = terrain_tiles[instance];
    let g = grid[vertex];
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
        if skirt != 0u && (north || south) {
            g.globe = vec3<f32>(0.0, 0.0, select(-SPHERE_RADIUS, SPHERE_RADIUS, north)) - frame.globe_target;
            z = 0.0;
        }
        clip = ground_clip_3d(g, vec2<f32>(0.0), z);
    }
    var out: TerrainOut;
    // Reversed-Z, as extrusions draw.
    out.position = vec4<f32>(clip.xy, clip.w - clip.z, clip.w);
    out.uv = uv;
    out.slot = tile.slot * terrain.stacks + stack.index;
    // The view depth is the clip w. Fog stays off on a globe, as in GL JS.
    out.fog_depth = select(atmosphere_fog_depth(atmosphere, clip.w), 0.0, globe);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    if frame.light_position.w > 0.0 && tile.dem_layer != NO_DEM {
        // Central differences over a grid cell, in world units; uv y runs
        // south, as tile axes do, whose horizontal parts the lights negate.
        let d = 1.0 / f32(GRID);
        let east = terrain_elevation(tile, uv + vec2<f32>(d, 0.0)) - terrain_elevation(tile, uv - vec2<f32>(d, 0.0));
        let south = terrain_elevation(tile, uv + vec2<f32>(0.0, d)) - terrain_elevation(tile, uv - vec2<f32>(0.0, d));
        let rise = frame.extrusion_scale / (2.0 * TERRAIN_EXTENT * tile.scale * d);
        out.normal = normalize(vec3<f32>(east * rise, south * rise, 1.0));
    }
    return out;
}

// The terrain's depth, for the symbols and circles it hides (#40): the
// reversed-Z depth the stacks draw, as a colour.
@fragment
fn fs_terrain_depth(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(position.z, 0.0, 0.0, 0.0);
}

// The slot's colour at uv, mipmapped and clamped to the slot.
fn drape_color(slot: u32, uv: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(drape));
    let corner = vec2<f32>(f32(slot % terrain.columns), f32(slot / terrain.columns)) * terrain.slot_px;
    let texel = uv * terrain.slot_px;
    let footprint = max(length(dpdx(texel)), length(dpdy(texel)));
    let lod = clamp(log2(max(footprint, 1e-6)), 0.0, terrain.levels - 1.0);
    // Half a texel of the coarser level read: samples stay in the slot.
    let inset = 0.5 * exp2(ceil(lod));
    let p = clamp(texel, vec2<f32>(inset), vec2<f32>(terrain.slot_px - inset));
    return textureSampleLevel(drape, drape_sampler, (corner + p) / size, lod);
}

@fragment
fn fs_terrain(in: TerrainOut) -> @location(0) vec4<f32> {
    var surface = drape_color(in.slot, in.uv);
    if frame.light_position.w > 0.0 {
        // The scene's lights shade the terrain too (decision 0026).
        surface = vec4<f32>(light_surface(surface.rgb, normalize(in.normal), 1.0, 0.0), surface.a);
    }
    return atmosphere_fog(atmosphere, surface, in.fog_depth);
}
