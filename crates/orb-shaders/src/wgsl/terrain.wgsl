// The terrain surface: one instance per terrain tile of a
// shared grid, raised by its DEM tile, coloured from the tile's drape slots
// for the terrain layer being drawn, fogged by the atmosphere like
// GL JS's `terrain.fragment.glsl`. On a globe it lies on the sphere through the
// tile's patches in space (`project.wgsl`'s `sphere_at`), raised along the
// sphere's up, and its top and bottom rows' skirts close the poles.
//
// A tile's drape slots are layers of a texture of their side that the
// draw of its tiles binds, as many texels as the tile shows at: a layer a
// colour stack, those of its texture's `slot`th slot (colour stack `s` in
// layer `slot * colour_stacks + s`). Slope stacks are in a texture of
// their own, at half the texels, likewise. A
// terrain layer draws a stack's slots, or
// a stack's below hillshades it shades from their slopes, each under its
// stack above it (`terrain_shade.wgsl`), over the map at its opacity. It
// draws in the map's pass after the terrain's depth alone
// (`fs_terrain_hidden`), so only the nearest surface shades.

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
    // Its slots' place in their textures.
    slot: u32,
    // Tile uv → the uv of the slots it shows: `uv * slot_scale +
    // slot_offset`, its own, or while blank its part of an ancestor's.
    slot_offset: vec2<f32>,
    slot_scale: f32,
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
    // Terrain tiles drawn.
    tiles: u32,
    // Colour stacks, and slope stacks: each slot's layers in a texture.
    colour_stacks: u32,
    slope_stacks: u32,
    // Grid points a side of the mesh's cells (orb-render's `grid_step`).
    grid_step: u32,
};

// A hillshade a terrain layer shades: the slope stack (by its rank on
// the slope page) holding its slopes, or NO_STACK past the layer's last,
// the colour stack drawn above it, and the blend mode it blends into the
// colours beneath it in (`blend.wgsl`).
struct Shade {
    slopes: u32,
    above: u32,
    blend: u32,
    pad: u32,
};

// A terrain layer: the colour stack drawn below its hillshades; how it
// fogs: FOG_CONTENT, or FOG_GROUND for the map's
// first terrain layer, the ground, which fogs where it draws nothing too,
// or FOG_FADES for a layer in a blend mode, over ground fogged already; 1
// where its first hillshade blends into the map beneath rather than a
// stack below (`terrain_shade.wgsl`); its opacity, and the blend mode it
// blends into the map beneath in (`blend.wgsl`; 0, normal, over it); and
// its hillshades (orb-render's `MAX_SHADES`).
struct Layer {
    below: u32,
    fog: u32,
    backdrop: u32,
    opacity: f32,
    blend: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    shades: array<Shade, 2>,
};

const FOG_GROUND: u32 = 1u;
const FOG_FADES: u32 = 2u;

const TERRAIN_EXTENT: f32 = 8192.0;
const NO_DEM: u32 = 0xffffffffu;
const NO_STACK: u32 = 0xffffffffu;
// Grid cells per tile side.
const GRID: u32 = 128u;
// Whether the vertices work out the drape's footprint (`cell_footprint`):
// not for the terrain's depth alone, whose fragments sample nothing.
override terrain_footprint: bool = true;

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
    // Uv in the slots it shows, y down.
    @location(0) uv: vec2<f32>,
    // Its slots' place in their textures.
    @location(1) @interpolate(flat, either) slot: u32,
    @location(2) fog_depth: f32,
    // Mercator's slope stretch for a hillshade the layer shades
    // (`terrain_shade.wgsl`): per vertex rather than per fragment, which
    // a grid cell's smooth change of latitude doesn't show.
    @location(3) stretch: f32,
    // A framebuffer pixel's step right and down in `uv` over the mesh's
    // cell the triangle is in (`cell_footprint`), packed (`pack_halves`):
    // its provoking vertex's, which is the cell's first grid point
    // (orb-render's `grid`).
    @location(4) @interpolate(flat, either) footprint: vec2<u32>,
};

// Two floats in a word, each its top 16 bits: its sign, exponent and 7
// bits of mantissa (bfloat16), which a footprint's level takes to within
// a hundredth.
fn pack_halves(v: vec2<f32>) -> u32 {
    return (bitcast<u32>(v.x) & 0xffff0000u) | (bitcast<u32>(v.y) >> 16u);
}

fn unpack_halves(word: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(word & 0xffff0000u), bitcast<f32>(word << 16u));
}

// What a whole unit of tile uv along u and along v moves the surface by
// in clip space at the point `sphere` (on a globe, from the globe's
// target), over the cell of the mesh there, whose height climbs `slope`
// world units a unit of uv: on the plane, on the low-zoom surface, mixed
// as the globe hands over. From the surface's tangents rather than its
// points beside it placed again, which would be a second copy of the
// globe's placement, one Adreno's compilers inline and the moto g pays a
// tenth of its terrain's frame for though no frame runs it.
fn surface_steps(tile: TerrainTile, sphere: vec3<f32>, slope: vec2<f32>, globe: bool) -> array<vec4<f32>, 2> {
    let span = TERRAIN_EXTENT * tile.scale;
    let flat_u = vec3<f32>(span, 0.0, slope.x);
    let flat_v = vec3<f32>(0.0, -span, slope.y);
    var du = frame.view_proj * vec4<f32>(flat_u, 0.0);
    var dv = frame.view_proj * vec4<f32>(flat_v, 0.0);
    if globe {
        let metres = frame.globe_metres;
        var u = flat_u * metres;
        var v = flat_v * metres;
        if frame.globe_plane == 0u {
            // Along the sphere, a plane world unit is `globe_metres` at the
            // target's latitude and shrinks with the latitude's cosine.
            let up = normalize(sphere + frame.globe_target);
            let level = vec3<f32>(-up.y, up.x, 0.0);
            let east = select(vec3<f32>(0.0, 1.0, 0.0), normalize(level), length(level) > 1e-6);
            let north = cross(up, east);
            let along = metres * length(up.xy) / max(length(normalize(frame.globe_target).xy), 1e-6);
            u = east * (span * along) + up * (slope.x * metres);
            v = -north * (span * along) + up * (slope.y * metres);
        }
        du = mix(du, frame.globe_view_proj * vec4<f32>(u, 0.0), frame.globe_amount);
        dv = mix(dv, frame.globe_view_proj * vec4<f32>(v, 0.0), frame.globe_amount);
    }
    return array<vec4<f32>, 2>(du, dv);
}

// What a framebuffer pixel's step right and down move the surface by in
// tile uv over a cell of the mesh `h` uv a side whose first grid point is
// at `first` in clip space and whose sides move it by `steps`
// (`surface_steps`): the screen's steps along them at its centre,
// inverted. A cell's fragments all sample with it, as a triangle's do
// with the derivatives GL JS's GPU takes: not ours, coarse or fine, which
// on Apple's GPUs came out otherwise at a few pixels from one frame to
// the next, so that the same frame drawn again sampled them a unit apart,
// and through a hillshade's slopes several. Near the camera, where w goes
// to 0, the footprint shrinks to nothing, and the finest level shows.
fn cell_footprint(first: vec4<f32>, steps: array<vec4<f32>, 2>, h: f32) -> vec4<f32> {
    let du = steps[0];
    let dv = steps[1];
    let centre = first + 0.5 * h * (du + dv);
    let w = max(centre.w, 1e-6);
    // Framebuffer pixels per unit of clip x and y over w, rows down.
    let px = vec2<f32>(0.5, -0.5) * frame.viewport * frame.pixel_ratio;
    let a = (du.xy * w - centre.xy * du.w) / (w * w) * px;
    let b = (dv.xy * w - centre.xy * dv.w) / (w * w) * px;
    var det = a.x * b.y - a.y * b.x;
    det = select(det, 1e-12, abs(det) < 1e-12);
    let step = vec4<f32>(b.y, -a.y, -b.x, a.x) / det;
    return clamp(step, vec4<f32>(-1e4), vec4<f32>(1e4));
}

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
    var sphere = vec3<f32>(0.0);
    if globe {
        var g: Ground;
        g.flat = world;
        g.on_plane = true;
        let origin = vec3<f32>(tile.globe_origin[0], tile.globe_origin[1], tile.globe_origin[2]);
        g.globe = sphere_at(tile.globe_base, tile.globe_cells, origin, uv);
        sphere = g.globe;
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
    out.uv = uv * tile.slot_scale + tile.slot_offset;
    if terrain_footprint {
        // The cell's slope through its first, right and down corners. A
        // skirt's vertex provokes no triangle (orb-render's `grid`), so its
        // lowered point serves.
        let h = f32(terrain.grid_step) / f32(GRID);
        let beside = vec2<f32>(
            terrain_elevation(tile, uv + vec2<f32>(h, 0.0)),
            terrain_elevation(tile, uv + vec2<f32>(0.0, h)),
        );
        let slope = (beside - elevation) / h * frame.extrusion_scale;
        let steps = surface_steps(tile, sphere, slope, globe);
        let f = cell_footprint(clip, steps, h) * tile.slot_scale;
        out.footprint = vec2<u32>(pack_halves(f.xy), pack_halves(f.zw));
    }
    out.slot = tile.slot;
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

// The terrain's depth alone, in a pass that draws more: nothing over the
// colour beneath it, which the pipeline blends over.
@fragment
fn fs_terrain_hidden() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

// Where a fragment samples its tile's drape slots, each a texture of its
// own `slot_px` texels a side: uv and the footprint's gradients.
// Trilinear and anisotropic, as GL JS samples its render-to-texture
// tiles, and kept off the slot's edge by half a texel of the level the
// sampler reads, the footprint shrunk where it would reach past it, on a
// 1/512-texel grid. With the sampler's own clamp to the edge, what a
// moved map showed at a seam differed from a fresh map's by a speck. The
// footprint is the cell's (`cell_footprint`), not the derivatives'.
struct DrapeSample {
    uv: vec2<f32>,
    dx: vec2<f32>,
    dy: vec2<f32>,
};

fn drape_sample(in: TerrainOut, slot_px: f32) -> DrapeSample {
    let texel = in.uv * slot_px;
    let dx = unpack_halves(in.footprint.x) * slot_px;
    let dy = unpack_halves(in.footprint.y) * slot_px;
    // A texel of the level the sampler reads: the footprint's longer side
    // over up to 16 probes along it, no finer than its shorter side.
    let level_texel = max(max(length(dx), length(dy)) / 16.0, min(length(dx), length(dy)));
    let half = 0.5 * exp2(ceil(clamp(log2(level_texel), 0.0, log2(slot_px) - 1.0)));
    let p = clamp(texel, vec2<f32>(half), vec2<f32>(slot_px - half));
    let room = min(p, vec2<f32>(slot_px) - p) - half;
    let need = 0.5 * max(abs(dx), abs(dy));
    let fit = min(1.0, min(room.x / max(need.x, 1e-6), room.y / max(need.y, 1e-6)));
    return DrapeSample(round(p * 512.0) / 512.0 / slot_px, dx * fit / slot_px, dy * fit / slot_px);
}

// Colour stack `stack`'s slot colour at `at`, none for NO_STACK.
fn drape_color(in: TerrainOut, at: DrapeSample, stack: u32) -> vec4<f32> {
    if stack == NO_STACK {
        return vec4<f32>(0.0);
    }
    let layer = in.slot * terrain.colour_stacks + stack;
    return textureSampleGrad(drape, drape_sampler, at.uv, layer, at.dx, at.dy);
}

// Slope stack `rank`'s slot at `at`.
fn slope_texel(in: TerrainOut, at: DrapeSample, rank: u32) -> vec4<f32> {
    let layer = in.slot * terrain.slope_stacks + rank;
    return textureSampleGrad(slope_atlas, drape_sampler, at.uv, layer, at.dx, at.dy);
}

// `top` over `bottom`, premultiplied.
fn over(top: vec4<f32>, bottom: vec4<f32>) -> vec4<f32> {
    return top + (1.0 - top.a) * bottom;
}

@fragment
fn fs_terrain(in: TerrainOut) -> @location(0) vec4<f32> {
    let at = drape_sample(in, f32(textureDimensions(drape).x));
    let color = drape_color(in, at, layer.below);
    return fogged(in, color) * layer.opacity;
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
