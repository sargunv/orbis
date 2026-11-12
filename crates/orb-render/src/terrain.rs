//! Terrain (render graph `Terrain`/`Drape`): the map's 2D
//! layers draped over a DEM-raised surface, after GL JS's render to
//! texture.
//!
//! Each terrain tile has a drape slot per drape stack (a run of draped
//! layers between layers that draw in 3D or on screen, the part of one
//! above its first layer whose drawing changes with the zoom, or a
//! hillshade the terrain shades, whose stack holds its unlit slopes).
//! Slots are squares of mipmapped atlas pages, sized to the slots in use:
//! colour stacks in one, slope stacks at half the texels a side in
//! another, each stack's slots side by side in a layer of its page, as
//! many stacks a layer as fit, so that a page holds as many terrain tiles
//! however many stacks it has. A new page takes the kept slots' texels,
//! so none redraws for it. A frame redraws only the slots the runtime
//! marks: their drape placements, the draped nodes' only candidates, map
//! them into the atlas, so one pass a layer redraws them all, replaying
//! each stack's bundle of its draped nodes over its redrawn slots, then a
//! compute pass a page downsamples their mip levels. Each terrain
//! layer (a stack, two where the zoom splits it, or two around a hillshade
//! lit between them from its slopes' stack and the frame's bearing) draws
//! into a texture of its own with the depth extrusions share, composited
//! in the map at its place.

use crate::TargetFormat;
use crate::pipeline::{DEPTH, Draw};
use crate::release::Release;
use crate::scene::{Backing, DataTable};
use TextureViewDimension::D2Array;
use orb_hal::*;
use orb_shaders::{DRAPE_SHADER, Shader, TERRAIN_SHADE_SHADER, TERRAIN_SHADER};

use crate::offscreen::{Composite, Composites, Mode, Offscreen};

/// Drape texels per terrain tile side: twice the 512 logical pixels a
/// terrain tile covers at its zoom, GL JS's quality factor.
pub const DRAPE_SLOT_PX: u32 = 1024;

/// Slope slots' texels per terrain tile side: half the drape's. A shaded
/// hillshade's slopes come from DEM tiles of at most 512 pixels a side at
/// the terrain tile's zoom, which a 512-texel slot holds sample for
/// sample, and a quarter of the texels lets a viewport-lit hillshade cost
/// a terrain tile a quarter slot rather than a whole one (#142).
pub(crate) const SLOPE_SLOT_PX: u32 = DRAPE_SLOT_PX / 2;

/// The drape's pages by slot side: colour stacks, then slope stacks.
const PAGE_PX: [u32; 2] = [DRAPE_SLOT_PX, SLOPE_SLOT_PX];

/// Slots of `slot_px` texels per side of a texture `max` texels a side, a
/// power of two (see [`atlas_grid`]).
fn atlas_side(max: u32, slot_px: u32) -> u32 {
    1 << (max / slot_px).max(1).ilog2()
}

/// A page's grid: stacks a layer, and the layers' columns and rows.
type Grid = (u32, u32, u32);

/// A page's grid for `tiles` tile slots of `stacks` stacks: stacks a
/// layer and the layers' (columns, rows); or `None` to keep the grid
/// `held`, kept while its stacks have room for the tiles and they fill
/// more than a quarter of it. A layer holds as many of the stacks as fit
/// with their tiles, so that a view of few tiles keeps them all in one,
/// redrawn in one pass, and one of many has a layer a stack. A grid is a
/// power of two slots a side, so that drape space maps to clip space and
/// atlas uv by powers of two, exactly: a slot's texels then don't depend
/// on where in the atlas it lies, which depends on the tiles that came
/// and went before. A new grid holds the slots in the fewest, or on
/// shrinking twice that, so that a view must grow past it again to make
/// another; squarest, columns first. `tiles` is at most `side²`.
fn atlas_grid(
    tiles: u32,
    held: Option<(u32, u32, u32)>,
    stacks: u32,
    side: u32,
) -> Option<(u32, u32, u32)> {
    let tiles = tiles.max(1);
    let room = held.map(|(per_layer, columns, rows)| columns * rows / per_layer);
    if room.is_some_and(|room| tiles <= room && tiles * 4 > room) {
        return None;
    }
    let shrink = room.is_some_and(|room| tiles <= room);
    let per_layer = stacks.min(side * side / tiles).max(1);
    let count = ((per_layer * tiles).next_power_of_two() << u32::from(shrink)).min(side * side);
    let k = count.ilog2();
    Some((per_layer, 1 << k.div_ceil(2), 1 << (k / 2)))
}

/// The grid of a layer a stack for `tiles` tile slots, rounded up to a
/// power of two: the fewest texels a page of them can take.
fn tight_grid(tiles: u32) -> Grid {
    let k = tiles.max(1).next_power_of_two().ilog2();
    (1, 1 << k.div_ceil(2), 1 << (k / 2))
}

/// Whether the device allocates page `page` in `grid` for `stacks` stacks
/// ([`Gpu::texture_fits`]).
fn fits(gpu: &dyn Gpu, page: usize, (per_layer, columns, rows): Grid, stacks: u32) -> bool {
    let px = PAGE_PX[page];
    let size = [columns * px, rows * px];
    gpu.texture_fits(&atlas_desc(size, stacks.div_ceil(per_layer), px))
}

/// The terrain tiles the drape pages hold, with `stacks` colour and slope
/// stacks: the slots of a layer of WebGPU's default texture side (64
/// colour slots), however many stacks there are, and on every device that
/// has it, so that a view's terrain doesn't depend on either. A layer of
/// the device's largest side holds more of the stacks. Where a page of a
/// layer a stack of them takes more bytes than the device allocates a
/// texture (GL's [`Limits::max_texture_bytes`]), it holds half as many,
/// and so on until such a page fits.
pub(crate) fn room(gpu: &dyn Gpu, stacks: [u32; 2]) -> u32 {
    let alloc = allocated(stacks);
    let limits = gpu.caps().limits;
    (0..2)
        .filter(|&p| alloc[p] > 0)
        .map(|p| {
            if alloc[p] > limits.max_texture_array_layers {
                return 0;
            }
            let side = atlas_side(limits.max_texture_dimension_2d, PAGE_PX[p]);
            let mut tiles = side * side;
            while tiles > 0 && !fits(gpu, p, tight_grid(tiles), alloc[p]) {
                tiles /= 2;
            }
            tiles
        })
        .min()
        .expect("a colour page")
}

/// Page `page`'s grid for `tiles` tile slots of `stacks` stacks
/// ([`atlas_grid`]), or `None` to keep `held`: at the device's largest
/// side, or a smaller one where that page takes more bytes than the device
/// allocates, or else a layer a stack ([`tight_grid`]), which a cover
/// within the [`room`] fits.
fn page_grid(
    gpu: &dyn Gpu,
    page: usize,
    tiles: u32,
    held: Option<Grid>,
    stacks: u32,
) -> Option<Grid> {
    let limits = gpu.caps().limits;
    let base = atlas_side(limits.max_texture_dimension_2d, PAGE_PX[page]);
    let mut side = atlas_side(limits.largest_texture_dimension_2d, PAGE_PX[page]);
    loop {
        let grid = atlas_grid(tiles, held, stacks, side)?;
        if fits(gpu, page, grid, stacks) {
            return Some(grid);
        }
        if side <= base {
            return Some(tight_grid(tiles));
        }
        side /= 2;
    }
}

/// Why the drape pages can't hold a view's terrain tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrapeFull {
    /// The style's drape stacks leave no room for them ([`room`]).
    Room,
    /// The device refused a page of this many bytes; pages take fewer from
    /// then on ([`Gpu::try_create_texture`]).
    Refused(u64),
}

/// The stacks each page holds: the colour page at least one, so that it
/// always exists for the terrain to bind.
fn allocated(stacks: [u32; 2]) -> [u32; 2] {
    [stacks[0].max(1), stacks[1]]
}
/// The drape atlas's format: draped layers draw into it with pipelines of
/// their own when the target's differs; mip levels are written as storage.
pub const DRAPE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
/// Grid vertices per terrain tile side less one (`terrain.wgsl`'s `GRID`):
/// one per DEM pixel at the DEM zoom the runtime picks. The mesh drawn
/// uses every [`grid_step`]th of them.
pub const TERRAIN_GRID: u32 = 128;
/// The grid steps the mesh has index ranges for: cells of 1, 2 and 4
/// vertices a side.
const GRID_STEPS: [u32; 3] = [1, 2, 4];

/// The grid step a terrain over DEM tiles of `tile_px` logical pixels
/// draws with: GL JS's mesh density, whose terrain tiles are twice the DEM
/// tiles' size with 128 cells. A vertex every 4 logical pixels over
/// 256-pixel DEM tiles, every 8 over 512-pixel ones, which costs a
/// quarter of the vertices with nothing lost the drape and its shading
/// don't carry (a quarter less GPU time over the Alps).
fn grid_step(tile_px: f32) -> u32 {
    let step = (tile_px / 256.0).round().max(1.0) as u32;
    *GRID_STEPS
        .iter()
        .rev()
        .find(|&&s| s <= step)
        .expect("a step of 1")
}
/// Terrain units per tile side (`terrain.wgsl`'s `TERRAIN_EXTENT`).
pub const TERRAIN_EXTENT: f32 = 8192.0;
const NO_DEM: u32 = u32::MAX;
const NONE: u32 = u32::MAX;
use orb_shaders::layout::{terrain as params, terrain_tile};
/// Uniform slices are 256-byte aligned on every backend.
const UNIFORM_STRIDE: u64 = 256;

/// The terrain's values for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainPaint {
    /// GL JS's unpack vector of the DEM source.
    pub unpack: [f32; 4],
    pub exaggeration: f32,
    /// World units the tile skirts hang below their edges.
    pub skirt: f32,
    /// The DEM source's tile size in logical pixels, which sets the
    /// mesh's density ([`grid_step`]).
    pub dem_tile_px: f32,
}

/// A terrain tile to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainTile {
    /// Its north-west corner in camera-relative world space.
    pub origin: [f32; 2],
    /// World units per terrain unit: it spans [`TERRAIN_EXTENT`].
    pub scale: f32,
    /// Its DEM: page layer, and tile uv → DEM uv as `uv · scale + offset`.
    pub dem: Option<TerrainDem>,
    /// Its block of drape slots (see [`DrapeLayout`]).
    pub slot: u32,
    /// On a globe, where it lies on the sphere.
    pub globe: Option<TerrainGlobe>,
    /// Latitudes of its top and bottom edges, in degrees, which a shaded
    /// hillshade's slopes shrink by.
    pub latitudes: [f32; 2],
}

/// What one terrain draw composites in the map: a stack's slots, the
/// stacks below and above where the zoom splits it, or those below and
/// above a hillshade it shades between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TerrainLayer {
    pub(crate) below: Option<u8>,
    /// The hillshade node, and the stack its unlit slopes draw into.
    pub(crate) shade: Option<(u32, u8)>,
    pub(crate) above: Option<u8>,
    /// A layer's own, composited in a blend mode other than normal.
    pub(crate) blended: bool,
    /// How its hillshade blends into the colours below it: normal, or a
    /// blend mode its layer composites in, which the layer then takes.
    pub(crate) shade_blend: crate::Blend,
    /// The colours its hillshade blends into are the terrain layer's
    /// before it, as it drew them, rather than a stack of its own: the
    /// hillshade lies on that layer, which shades another.
    pub(crate) backdrop: bool,
}

/// How a terrain layer fogs (`Layer.fog` in `terrain.wgsl`): its content,
/// the whole ground, or by fading its coverage.
const FOG_CONTENT: u32 = 0;
const FOG_GROUND: u32 = 1;
const FOG_FADES: u32 = 2;

/// A terrain tile on a globe: its patches in
/// space around `origin`, the anchor less the camera target in metres, and
/// which poles, north then south, its edge skirts close (a top or bottom
/// row of a set that stops short of the pole).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainGlobe {
    pub warp: crate::scene::TileWarp,
    pub origin: [f32; 3],
    pub poles: [bool; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainDem {
    pub layer: u32,
    pub offset: [f32; 2],
    pub scale: f32,
}

/// Where drape slots are. Stacks `0..stacks[0]` are colour stacks, on
/// page 0 at [`DRAPE_SLOT_PX`] texels a side; the rest hold slopes, on
/// page 1 at [`SLOPE_SLOT_PX`]. Page `p`'s layers hold `per_layer[p]`
/// stacks each: tile slot `t`'s stack of rank `r` there is layer
/// `r / per_layer[p]`, slot `r % per_layer[p] · strides[p] + t` of it, at
/// column `slot % columns[p]`, row `slot / columns[p]`. Each stack's
/// slots lie side by side in one layer, so that redrawing one stack of
/// many tiles draws into a few rows. Each layer's drape space is its
/// texels with y up: texel `(x, y)` is drape world `(x, −y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrapeLayout {
    /// Colour stacks, then slope stacks.
    pub stacks: [u32; 2],
    /// Stacks a layer of each page holds.
    pub per_layer: [u32; 2],
    /// Slots per row of each page's layers.
    pub columns: [u32; 2],
    /// Slots of each page's layers between a tile slot's stacks: the tile
    /// slots one of its stacks has room for.
    pub strides: [u32; 2],
    /// Tile slots the pages hold.
    pub tile_slots: u32,
    /// Changes when every slot is blank (new pages without the old ones'
    /// texels, or other stacks): every slot must be redrawn.
    pub generation: u64,
}

impl DrapeLayout {
    /// The stacks of both pages.
    pub fn stack_count(&self) -> u32 {
        self.stacks[0] + self.stacks[1]
    }

    /// The page of stack `stack`, and its rank among the page's stacks.
    pub fn page(&self, stack: u32) -> (usize, u32) {
        match stack.checked_sub(self.stacks[0]) {
            Some(rank) => (1, rank),
            None => (0, stack),
        }
    }

    /// Texels a side of stack `stack`'s slots.
    pub fn slot_px(&self, stack: u32) -> u32 {
        PAGE_PX[self.page(stack).0]
    }

    /// The layer of its page stack `stack` lies in.
    pub fn layer(&self, stack: u32) -> u32 {
        let (page, rank) = self.page(stack);
        rank / self.per_layer[page]
    }

    /// Tile slot `tile_slot`'s slot of `stack`: its page and slot in the
    /// stack's layer.
    fn slot(&self, tile_slot: u32, stack: u32) -> (usize, u32) {
        let (page, rank) = self.page(stack);
        let per_layer = self.per_layer[page];
        (page, rank % per_layer * self.strides[page] + tile_slot)
    }

    /// A slot's top-left texel in its layer.
    pub fn corner(&self, tile_slot: u32, stack: u32) -> [u32; 2] {
        let (page, slot) = self.slot(tile_slot, stack);
        let (columns, px) = (self.columns[page], PAGE_PX[page]);
        [slot % columns * px, slot / columns * px]
    }
}

/// Sorted atlas slots as rectangles of an atlas `columns` slots wide:
/// runs side by side in a row, each joined to the run above it when that
/// spans the same columns, `[column, row, slots, rows]`.
fn slot_runs(slots: &[u32], columns: u32) -> Vec<[u32; 4]> {
    let mut rects: Vec<[u32; 4]> = Vec::new();
    let close = |rects: &mut Vec<[u32; 4]>, [x, y, width]: [u32; 3]| match rects.last_mut() {
        Some(r) if (r[0], r[2]) == (x, width) && r[1] + r[3] == y => r[3] += 1,
        _ => rects.push([x, y, width, 1]),
    };
    let mut run: Option<[u32; 3]> = None;
    for &slot in slots {
        let (x, y) = (slot % columns, slot / columns);
        match &mut run {
            Some(r) if r[1] == y && r[0] + r[2] == x => r[2] += 1,
            _ => {
                if let Some(done) = run.replace([x, y, 1]) {
                    close(&mut rects, done);
                }
            }
        }
    }
    if let Some(done) = run {
        close(&mut rects, done);
    }
    rects
}

/// A drape page's texture: `layers` layers of `size` texels, mipmapped
/// down from slots of `slot_px`.
fn atlas_desc(size: [u32; 2], layers: u32, slot_px: u32) -> TextureDesc<'static> {
    TextureDesc {
        label: "drape atlas",
        width: size[0],
        height: size[1],
        array_layers: layers,
        dimension: TextureViewDimension::D2Array,
        format: DRAPE_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST,
        sample_count: 1,
        mip_levels: slot_px.ilog2() + 1,
    }
}

#[derive(Debug)]
struct Atlas {
    /// Layers of [`DrapeLayout::per_layer`] stacks, mipmapped.
    texture: TextureId,
    /// A layer's texels.
    size: [u32; 2],
    layers: u32,
    /// Texels per slot side.
    slot_px: u32,
    /// Per mip level, a view, which the mip levels are written through.
    levels: Vec<TextureId>,
    /// What draws into each layer's level 0.
    draw: Vec<TextureId>,
    /// A layer's size, for the clear.
    size_uniform: BufferId,
    clear: BindGroupId,
    /// The downsampler's chain of its levels, over the redrawn slots as
    /// its regions, which the clear reads too.
    mips: crate::downsample::Chain,
}

impl Atlas {
    /// A page of `layers` layers of `columns × rows` slots of `slot_px`
    /// texels, or the bytes of one the device refused.
    fn new(
        gpu: &mut dyn Gpu,
        (columns, rows, layers): (u32, u32, u32),
        slot_px: u32,
        clear_layout: BindGroupLayoutId,
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<Self, u64> {
        let size = [columns, rows].map(|n| n * slot_px);
        let levels = slot_px.ilog2() + 1;
        let desc = atlas_desc(size, layers, slot_px);
        let texture = gpu
            .try_create_texture(&desc)
            .ok_or_else(|| level_bytes(&desc, layers) * 4 / 3)?;
        let views: Vec<TextureId> = (0..levels)
            .map(|l| gpu.create_texture_view(texture, l..l + 1))
            .collect();
        let draw = (0..layers)
            .map(|layer| gpu.create_layer_view(texture, layer, 0))
            .collect();
        downsampler.make(gpu);
        let slots = ("drape slots", columns * rows * layers);
        let mips = downsampler.chain(gpu, (texture, &views), slots);
        let size_uniform = gpu.create_buffer(&BufferDesc {
            label: "drape atlas size",
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let words = [size[0] as f32, size[1] as f32, 0.0, 0.0];
        gpu.write_buffer(size_uniform, 0, bytemuck::cast_slice(&words));
        let clear = gpu.create_bind_group(&BindGroupDesc {
            label: "drape clear",
            layout: clear_layout,
            entries: &[
                BindGroupEntry::buffer(0, size_uniform),
                mips.regions.entry(1),
            ],
        });
        Ok(Self {
            texture,
            size,
            layers,
            slot_px,
            levels: views,
            draw,
            size_uniform,
            clear,
            mips,
        })
    }

    /// A layer's slots: columns and rows.
    fn grid(&self) -> (u32, u32) {
        (self.size[0] / self.slot_px, self.size[1] / self.slot_px)
    }

    /// Its bytes, mips included.
    fn bytes(&self) -> u64 {
        let [w, h] = self.size.map(u64::from);
        w * h * u64::from(self.layers) * 4 * 4 / 3
    }
}

impl crate::release::Release for Atlas {
    fn release(self, gpu: &mut dyn Gpu) {
        (
            self.clear,
            self.mips,
            self.levels,
            self.draw,
            self.texture,
            self.size_uniform,
        )
            .release(gpu);
    }
}

#[derive(Debug)]
pub(crate) struct Terrain {
    target: TargetFormat,
    layout: BindGroupLayoutId,
    stack_layout: BindGroupLayoutId,
    /// A shaded hillshade's node group: its image node's.
    shade_layout: BindGroupLayoutId,
    /// What a shaded hillshade blends into, where its layer has a
    /// backdrop: the terrain layer's before it.
    backdrop_layout: BindGroupLayoutId,
    /// Each terrain layer's backdrop group and the texture it binds, and
    /// the blank one for the layers without.
    backdrops: Vec<Option<(TextureId, BindGroupId)>>,
    blank: Option<(TextureId, BindGroupId)>,
    clear_layout: BindGroupLayoutId,
    /// Draping and shading, each at one sample and at four.
    pipelines: [[Option<RenderPipelineId>; 2]; 2],
    /// Clearing the pages' slots.
    clear: RenderPipelineId,
    grid: DataTable,
    grid_indices: BufferId,
    /// The index range of each grid step's mesh, and the step drawn.
    grid_ranges: [std::ops::Range<u32>; GRID_STEPS.len()],
    grid_step: u32,
    tiles: DataTable,
    tile_capacity: u32,
    tile_count: u32,
    params: BufferId,
    /// The terrain layers, and each's group over its slice of
    /// `stack_uniforms`, made for as many as the style has.
    layers: Vec<TerrainLayer>,
    stack_groups: Vec<BindGroupId>,
    stack_uniforms: Option<BufferId>,
    /// Trilinear and anisotropic, as GL JS samples its render-to-texture
    /// tiles: the raster pages' sampler, which the renderer owns.
    linear: SamplerId,
    /// The colour page, and the slope page while a hillshade is shaded.
    atlas: [Option<Atlas>; 2],
    layout_state: DrapeLayout,
    /// The DEM page texture and revision the group binds.
    dem: Option<(TextureId, u32)>,
    /// The group, and the frame and property buffers it binds.
    group: Option<(BindGroupId, (BufferId, BindGroupEntry))>,
    /// Each terrain layer's texture and its composite into the map.
    targets: Vec<Offscreen>,
    /// The terrain's depth as a colour, and the depth it's drawn with: what
    /// hides symbols and circles.
    occlusion: Option<[TextureId; 2]>,
    occlusion_pipeline: Option<RenderPipelineId>,
    target_size: [u32; 2],
    /// The size of `occlusion`, in logical pixels.
    occlusion_size: [u32; 2],
    target_samples: u32,
    /// Slots to redraw this frame: (page, stack, slot in its layer). In a
    /// page's order they're each stack's in turn, so each layer's.
    pending: Vec<(usize, u32, u32)>,
    /// What the last frame's drawn mip levels rendered into, released
    /// by the next.
    mip_targets: crate::downsample::Transient,
    pub(crate) paint: Option<TerrainPaint>,
    /// The atmosphere's uniform, for fog.
    atmosphere: BufferId,
}

/// What a frame's terrain passes drew: render passes, and drape slots
/// redrawn.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TerrainDrawn {
    pub(crate) draws: u32,
    pub(crate) slots: u32,
}

impl Terrain {
    /// Terrain drawn into `target`, fogged by `atmosphere`, shading
    /// hillshades through image node groups of `shade_layout`.
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        atmosphere: BufferId,
        (shade_layout, linear): (BindGroupLayoutId, SamplerId),
    ) -> Self {
        use BindGroupLayoutEntry as E;
        let vs = ShaderStages::VERTEX;
        let fs = ShaderStages::FRAGMENT;
        let vf = vs | fs;
        let backing = Backing::of(gpu.caps());
        let table = |binding, stages| DataTable::layout_entry(backing, binding, stages);
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain",
            entries: &[
                E::uniform(0, vf),
                // Terrain tiles: a shaded hillshade reads their latitudes.
                table(1, vf),
                table(2, vs),
                E::texture(3, vs, false, D2Array),
                E::texture(4, fs, true, D2Array),
                E::sampler(5, fs, true),
                E::uniform(6, vf),
                E::uniform(7, vf),
                // The scene's property arena: globe patches.
                table(8, vs),
                // The slope page (the colour page while there is none).
                E::texture(9, fs, true, D2Array),
            ],
        });
        let stack_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain layer",
            entries: &[E::uniform(0, fs)],
        });
        let backdrop_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain backdrop",
            entries: &[E::texture(0, fs, false, TextureViewDimension::D2)],
        });
        // The atlas's mip regions, as the downsampler keeps them.
        let regions = if cfg!(cpu_driven) && !gpu.caps().compute {
            table(1, vs)
        } else {
            E::storage(1, vs)
        };
        let clear_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "drape clear",
            entries: &[E::uniform(0, vs), regions],
        });
        let clear = Draw {
            topology: PrimitiveTopology::TriangleStrip,
            ..Draw::new(
                &DRAPE_SHADER,
                &[clear_layout],
                TargetFormat {
                    color: DRAPE_FORMAT,
                    sample_count: 1,
                },
                None,
            )
        }
        .make(gpu);

        let (grid, indices, grid_ranges) = grid();
        let grid_buffer = DataTable::new(gpu, "terrain grid", grid.len() as u64);
        grid_buffer.write(gpu, 0, &grid);
        let grid_indices = gpu.create_buffer(&BufferDesc {
            label: "terrain grid indices",
            size: indices.len() as u64 * 4,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
        });
        gpu.write_buffer(grid_indices, 0, bytemuck::cast_slice(&indices));
        let params = gpu.create_buffer(&BufferDesc {
            label: "terrain",
            size: params::WORDS as u64 * 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        Self {
            target,
            layout,
            stack_layout,
            shade_layout,
            backdrop_layout,
            backdrops: Vec::new(),
            blank: None,
            clear_layout,
            pipelines: [[None; 2]; 2],
            clear,
            grid: grid_buffer,
            grid_indices,
            grid_ranges,
            grid_step: 1,
            tiles: DataTable::new(gpu, "terrain tiles", 64 * terrain_tile::WORDS as u64),
            tile_capacity: 64,
            tile_count: 0,
            params,
            layers: Vec::new(),
            stack_groups: Vec::new(),
            stack_uniforms: None,
            linear,
            atlas: [None, None],
            layout_state: DrapeLayout {
                stacks: [1, 0],
                per_layer: [1; 2],
                columns: [1; 2],
                strides: [0; 2],
                tile_slots: 0,
                generation: 0,
            },
            dem: None,
            group: None,
            targets: Vec::new(),
            occlusion: None,
            occlusion_pipeline: None,
            target_size: [0; 2],
            occlusion_size: [0; 2],
            target_samples: 0,
            pending: Vec::new(),
            mip_targets: Default::default(),
            paint: None,
            atmosphere,
        }
    }

    /// The drape pages' bytes, mips included.
    pub(crate) fn atlas_bytes(&self) -> u64 {
        self.atlas.iter().flatten().map(Atlas::bytes).sum()
    }

    pub(crate) fn layout(&self) -> DrapeLayout {
        self.layout_state
    }

    /// This frame's terrain layers, whose slope stacks follow `colour`
    /// colour stacks: each gets a group binding what it draws, and a
    /// texture once sized.
    pub(crate) fn set_layers(&mut self, gpu: &mut dyn Gpu, layers: &[TerrainLayer], colour: u32) {
        if self.layers == layers {
            return;
        }
        let count = (layers.len() as u32).next_power_of_two().max(4);
        if self.stack_groups.len() < count as usize {
            std::mem::take(&mut self.stack_groups).release(gpu);
            self.stack_uniforms.take().release(gpu);
            let buffer = gpu.create_buffer(&BufferDesc {
                label: "terrain layers",
                size: UNIFORM_STRIDE * u64::from(count),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            });
            self.stack_groups = (0..count)
                .map(|layer| {
                    gpu.create_bind_group(&BindGroupDesc {
                        label: "terrain layer",
                        layout: self.stack_layout,
                        entries: &[BindGroupEntry::slice(
                            0,
                            buffer,
                            UNIFORM_STRIDE * u64::from(layer),
                            32,
                        )],
                    })
                })
                .collect();
            self.stack_uniforms = Some(buffer);
        }
        let buffer = self.stack_uniforms.expect("made above");
        for (i, layer) in layers.iter().enumerate() {
            let stack = |s: Option<u8>| s.map_or(NONE, u32::from);
            // A slope stack by its rank on the slope page.
            let slopes = (layer.shade).map_or(NONE, |(_, s)| u32::from(s) - colour);
            // The first layer is the ground: it fogs where it draws
            // nothing too, and the layers over it only their own content,
            // but for a blended one, which fades into the fog: what it
            // blends with is fogged already.
            let fog = match () {
                () if layer.blended => FOG_FADES,
                () if i == 0 => FOG_GROUND,
                () => FOG_CONTENT,
            };
            let blend = layer.shade_blend.mode() as u32;
            let backdrop = u32::from(layer.backdrop);
            let words = [
                stack(layer.below),
                stack(layer.above),
                slopes,
                fog,
                blend,
                backdrop,
                0,
                0,
            ];
            gpu.write_buffer(
                buffer,
                UNIFORM_STRIDE * i as u64,
                bytemuck::cast_slice(&words),
            );
        }
        self.layers = layers.to_vec();
    }

    /// Whether [`reserve`](Self::reserve) of `tile_slots` at rest would
    /// make new pages.
    pub(crate) fn would_reserve(&self, gpu: &dyn Gpu, tile_slots: u32, stacks: [u32; 2]) -> bool {
        tile_slots <= room(gpu, stacks) && self.grids(gpu, tile_slots, stacks, true).1
    }

    /// Each page's grid for `tile_slots` slots ([`atlas_grid`]), and
    /// whether any differs from the pages held.
    fn grids(
        &self,
        gpu: &dyn Gpu,
        tile_slots: u32,
        stacks: [u32; 2],
        may_shrink: bool,
    ) -> ([Option<Grid>; 2], bool) {
        let state = self.layout_state;
        let alloc = allocated(stacks);
        let same = self.atlas[0].is_some() && state.stacks == stacks;
        let mut grids = [None; 2];
        let mut changed = !same;
        for page in 0..2 {
            if alloc[page] == 0 {
                continue;
            }
            let held = (self.atlas[page].as_ref())
                .filter(|_| same)
                .map(|a| (state.per_layer[page], a.grid()))
                .map(|(per_layer, (columns, rows))| (per_layer, columns, rows));
            let holds = held.is_some_and(|(k, c, r)| tile_slots <= c * r / k);
            let grid = (may_shrink || !holds)
                .then(|| page_grid(gpu, page, tile_slots, held, alloc[page]))
                .flatten();
            changed |= grid.is_some();
            grids[page] = grid.or(held);
        }
        (grids, changed)
    }

    /// Sizes the pages for `tile_slots` blocks of `stacks` colour and
    /// slope slots ([`atlas_grid`]); each new page copies the kept slots.
    /// The tile slots in `kept` carry their texels into new pages, those
    /// past the new end moved to free slots; returns the moves (from, to),
    /// or `None` when the pages stay. A change of stacks blanks every slot
    /// (a new [`DrapeLayout::generation`]). Pages that hold the slots
    /// stay unless `may_shrink`, which a map at rest passes. Fails when
    /// the slots exceed the device's texture size ([`room`]), or when the
    /// device refuses a page, which leaves none: the next reserve makes
    /// them anew, smaller.
    pub(crate) fn reserve(
        &mut self,
        gpu: &mut dyn Gpu,
        (tile_slots, kept): (u32, &[u32]),
        (stacks, may_shrink): ([u32; 2], bool),
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<Option<Vec<(u32, u32)>>, DrapeFull> {
        let state = self.layout_state;
        if tile_slots > room(gpu, stacks) {
            return Err(DrapeFull::Room);
        }
        let alloc = allocated(stacks);
        let same = self.atlas[0].is_some() && state.stacks == stacks;
        let (grids, changed) = self.grids(gpu, tile_slots, stacks, may_shrink);
        if !changed {
            return Ok(None);
        }
        let layout = DrapeLayout {
            stacks,
            per_layer: grids.map(|g| g.map_or(1, |(per_layer, ..)| per_layer)),
            columns: grids.map(|g| g.map_or(1, |(_, columns, _)| columns)),
            strides: grids.map(|g| g.map_or(0, |(k, c, r)| c * r / k)),
            tile_slots: (grids.iter().flatten())
                .map(|(k, c, r)| c * r / k)
                .min()
                .expect("a colour page"),
            generation: state.generation + u64::from(!same),
        };
        // Kept slots past the new end move to the lowest free ones.
        let mut free = (0..layout.tile_slots).filter(|t| !kept.contains(t));
        let moves: Vec<(u32, u32)> = (kept.iter().filter(|&&t| t >= layout.tile_slots))
            .map(|&t| (t, free.next().expect("room for every kept slot")))
            .collect();
        let to = |t: u32| moves.iter().find(|m| m.0 == t).map_or(t, |m| m.1);
        for page in 0..2 {
            let old = self.atlas[page].take();
            let Some(grid) = grids[page] else {
                old.release(gpu);
                continue;
            };
            let old = match old {
                // Its grid and slots stay.
                Some(old)
                    if same
                        && (state.per_layer[page], old.grid()) == (grid.0, (grid.1, grid.2))
                        && moves.is_empty() =>
                {
                    self.atlas[page] = Some(old);
                    continue;
                }
                Some(old) if same => Some(old),
                other => {
                    other.release(gpu);
                    None
                }
            };
            let (per_layer, columns, rows) = grid;
            let made = Atlas::new(
                gpu,
                (columns, rows, alloc[page].div_ceil(per_layer)),
                PAGE_PX[page],
                self.clear_layout,
                downsampler,
            );
            let atlas = match made {
                Ok(atlas) => atlas,
                Err(bytes) => {
                    old.release(gpu);
                    self.release_atlas(gpu);
                    self.pending.clear();
                    self.forget_group(gpu);
                    return Err(DrapeFull::Refused(bytes));
                }
            };
            if let Some(old) = old {
                let px = PAGE_PX[page];
                let first = if page == 0 { 0 } else { stacks[0] };
                let copies = (kept.iter())
                    .flat_map(|&t| (first..first + stacks[page]).map(move |s| (t, s)))
                    .map(|(t, s)| {
                        let ([x0, y0], [x1, y1]) = (state.corner(t, s), layout.corner(to(t), s));
                        orb_hal::TextureCopy {
                            from: [x0, y0, state.layer(s)],
                            to: [x1, y1, layout.layer(s)],
                            size: [px, px, 1],
                        }
                    })
                    .collect();
                gpu.submit(&[Pass::CopyTexture {
                    source: old.texture,
                    destination: atlas.texture,
                    copies,
                }]);
                old.release(gpu);
            }
            self.atlas[page] = Some(atlas);
        }
        self.layout_state = layout;
        // Terrain tiles find their slots by these from this frame on.
        self.write_layout(gpu);
        self.pending.clear();
        self.forget_group(gpu);
        Ok(Some(moves))
    }

    /// Writes the layout's words of the terrain's values.
    fn write_layout(&self, gpu: &mut dyn Gpu) {
        let state = self.layout_state;
        let words = [
            (params::PER_LAYER, state.per_layer[0]),
            (params::STRIDE, state.strides[0]),
            (params::COLUMNS, state.columns[0]),
            (params::SLOPE_PER_LAYER, state.per_layer[1]),
            (params::SLOPE_STRIDE, state.strides[1]),
            (params::SLOPE_COLUMNS, state.columns[1]),
        ];
        for (word, value) in words {
            gpu.write_buffer(self.params, word as u64 * 4, bytemuck::bytes_of(&value));
        }
    }

    fn release_atlas(&mut self, gpu: &mut dyn Gpu) {
        for page in &mut self.atlas {
            page.take().release(gpu);
        }
    }

    fn forget_group(&mut self, gpu: &mut dyn Gpu) {
        self.group.take().map(|(group, _)| group).release(gpu);
    }

    /// Redraws tile slots' stacks this frame: per tile slot, the stacks
    /// to redraw.
    pub(crate) fn redraw(&mut self, tile_slots: &[(u32, &[u8])]) {
        let layout = self.layout_state;
        for &(tile, redrawn) in tile_slots {
            let stacks =
                (redrawn.iter().map(|&s| u32::from(s))).filter(|&s| s < layout.stack_count());
            self.pending.extend(stacks.map(|stack| {
                let (page, slot) = layout.slot(tile, stack);
                (page, stack, slot)
            }));
        }
    }

    /// The stacks this frame redraws slots of, in order.
    pub(crate) fn pending_stacks(&self) -> Vec<u32> {
        let mut stacks: Vec<u32> = self.pending.iter().map(|p| p.1).collect();
        stacks.sort_unstable();
        stacks.dedup();
        stacks
    }

    pub(crate) fn set_tiles(&mut self, gpu: &mut dyn Gpu, tiles: &[TerrainTile]) {
        if tiles.len() as u32 > self.tile_capacity {
            self.tiles.destroy(gpu);
            self.tile_capacity = (tiles.len() as u32).next_power_of_two();
            self.tiles = DataTable::new(
                gpu,
                "terrain tiles",
                u64::from(self.tile_capacity) * terrain_tile::WORDS as u64,
            );
            self.forget_group(gpu);
        }
        let words: Vec<u32> = tiles
            .iter()
            .flat_map(|t| {
                use terrain_tile as n;
                let mut b = orb_shaders::Block::new(n::WORDS);
                let (layer, offset, scale) = match t.dem {
                    Some(d) => (d.layer, d.offset, d.scale),
                    None => (NO_DEM, [0.0; 2], 1.0),
                };
                b.f(n::ORIGIN, &t.origin)
                    .f(n::SCALE, &[t.scale])
                    .u(n::DEM_LAYER, &[layer]);
                b.f(n::DEM_OFFSET, &offset)
                    .f(n::DEM_SCALE, &[scale])
                    .u(n::SLOT, &[t.slot]);
                if let Some(g) = t.globe {
                    let poles = u32::from(g.poles[0]) | u32::from(g.poles[1]) << 1;
                    b.f(n::GLOBE_ORIGIN, &g.origin)
                        .u(n::GLOBE_BASE, &[g.warp.base()]);
                    b.u(n::GLOBE_CELLS, &[g.warp.cells()]).u(n::POLES, &[poles]);
                }
                b.f(n::LATITUDES, &t.latitudes);
                b.0
            })
            .collect();
        if !words.is_empty() {
            self.tiles.write(gpu, 0, &words);
        }
        self.tile_count = tiles.len() as u32;
        let at = params::TILES as u64 * 4;
        gpu.write_buffer(self.params, at, bytemuck::bytes_of(&self.tile_count));
    }

    /// Writes the frame's terrain values for a DEM page of `dim`-texel
    /// tiles, bound as `dem` (its padded texture and revision).
    pub(crate) fn set_paint(
        &mut self,
        gpu: &mut dyn Gpu,
        paint: TerrainPaint,
        dem: (TextureId, u32),
        dim: u32,
    ) {
        self.set_dem(gpu, dem);
        self.grid_step = grid_step(paint.dem_tile_px);
        let state = self.layout_state;
        let mut b = orb_shaders::Block::new(params::WORDS);
        b.f(params::UNPACK, &paint.unpack)
            .f(params::EXAGGERATION, &[paint.exaggeration]);
        b.f(params::DEM_DIM, &[dim as f32])
            .f(params::SKIRT, &[paint.skirt]);
        b.u(params::PER_LAYER, &[state.per_layer[0]])
            .u(params::STRIDE, &[state.strides[0]])
            .u(params::COLUMNS, &[state.columns[0]]);
        b.u(params::SLOPE_PER_LAYER, &[state.per_layer[1]])
            .u(params::SLOPE_STRIDE, &[state.strides[1]])
            .u(params::SLOPE_COLUMNS, &[state.columns[1]]);
        b.f(params::SLOT_PX, &[DRAPE_SLOT_PX as f32])
            .f(params::SLOPE_PX, &[SLOPE_SLOT_PX as f32])
            .u(params::TILES, &[self.tile_count]);
        gpu.write_buffer(self.params, 0, bytemuck::cast_slice(&b.0));
        self.paint = Some(paint);
    }

    /// Binds `dem` (the DEM page's padded texture and revision), forgetting
    /// the group over the last one if it moved.
    pub(crate) fn set_dem(&mut self, gpu: &mut dyn Gpu, dem: (TextureId, u32)) {
        if self.dem != Some(dem) {
            self.forget_group(gpu);
            self.dem = Some(dem);
        }
    }

    /// Sizes the terrain layers' textures to the target, `size` device
    /// pixels, and its depth for what the terrain hides to `logical`
    /// pixels, as GL JS's: what it hides is tested a logical pixel at a
    /// time, and a quarter of the texels at a pixel ratio of 2 take a
    /// fraction of the pass. Returns whether they changed.
    pub(crate) fn resize(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        (size, logical): ([u32; 2], [u32; 2]),
        samples: u32,
    ) -> bool {
        let layers = self.layers.len();
        if self.target_size == size
            && self.occlusion_size == logical
            && self.target_samples == samples
            && self.targets.len() == layers
        {
            return false;
        }
        for target in self.targets.drain(..) {
            target.release(gpu);
        }
        self.release_occlusion(gpu);
        let texture = |gpu: &mut dyn Gpu, label, format, usage| {
            gpu.create_texture(&TextureDesc {
                label,
                width: logical[0],
                height: logical[1],
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format,
                usage,
                sample_count: 1,
                mip_levels: 1,
            })
        };
        self.occlusion = Some([
            texture(
                gpu,
                "terrain depth",
                TextureFormat::R32Float,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            ),
            texture(
                gpu,
                "terrain depth pass",
                TextureFormat::Depth32Float,
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
            ),
        ]);
        self.target_size = size;
        self.occlusion_size = logical;
        self.target_samples = samples;
        for _ in 0..layers {
            let mut slot = None;
            let target = ("terrain layer", &[self.target.color][..]);
            Offscreen::fit(
                &mut slot,
                gpu,
                composites,
                target,
                (size, samples),
                (Mode::Over, None),
                1.0,
            );
            self.targets.extend(slot);
        }
        true
    }

    /// The colour texture terrain layer `layer` draws into, once sized.
    pub(crate) fn layer_texture(&self, layer: u8) -> Option<TextureId> {
        Some(self.targets.get(usize::from(layer))?.target.texture(0))
    }

    /// Terrain layer `layer`'s composite into the map, once sized.
    pub(crate) fn layer_composite(&self, layer: u8) -> Option<&Composite> {
        Some(&self.targets.get(usize::from(layer))?.composite)
    }

    /// Whether every terrain pipeline at the current sample count
    /// compiled, making the one for it if needed. Until then terrain draws
    /// nothing.
    pub(crate) fn ready(&mut self, gpu: &mut dyn Gpu) -> bool {
        let samples = self.target_samples.max(1);
        let pipeline = self.pipeline(gpu, samples, false);
        let shaded = (self.layers.iter().any(|l| l.shade.is_some()))
            .then(|| self.pipeline(gpu, samples, true));
        let occlusion = self.occlusion.map(|_| self.occlusion_pipeline(gpu));
        [pipeline.into(), self.clear.into()]
            .into_iter()
            .chain(shaded.map(PipelineId::from))
            .chain(occlusion.map(PipelineId::from))
            .all(|p| gpu.pipeline_ready(p))
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        let mut out: Vec<PipelineId> = vec![self.clear.into()];
        out.extend(
            self.pipelines
                .iter()
                .flatten()
                .flatten()
                .map(|&p| PipelineId::from(p)),
        );
        out
    }

    /// Makes every drape slot blank, as a new atlas does: the map draws
    /// them all again (draped layers whose pipelines arrived since).
    pub(crate) fn redraw_all(&mut self) {
        self.layout_state.generation += 1;
        self.pending.clear();
    }

    /// The pipeline terrain layers draw with at `samples`: shading a
    /// hillshade or not.
    fn pipeline(&mut self, gpu: &mut dyn Gpu, samples: u32, shaded: bool) -> RenderPipelineId {
        let slot = &mut self.pipelines[usize::from(shaded)][usize::from(samples > 1)];
        if let Some(pipeline) = *slot {
            return pipeline;
        }
        let (shader, layouts): (&Shader, &[_]) = if shaded {
            (
                &TERRAIN_SHADE_SHADER,
                &[
                    self.layout,
                    self.stack_layout,
                    self.shade_layout,
                    self.backdrop_layout,
                ],
            )
        } else {
            (&TERRAIN_SHADER, &[self.layout, self.stack_layout])
        };
        let target = TargetFormat {
            sample_count: samples,
            ..self.target
        };
        let pipeline = Draw {
            depth: DEPTH,
            ..Draw::new(shader, layouts, target, None)
        }
        .make(gpu);
        self.pipelines[usize::from(shaded)][usize::from(samples > 1)] = Some(pipeline);
        pipeline
    }

    /// This frame's passes before the map: redrawn drape slots (`drape`
    /// draws the draped nodes into level 0) and their mip levels into
    /// `passes`, and into `layers` each terrain layer's pass into its
    /// texture, in order, the first clearing `depth`, the rest loading it.
    /// The caller places the layer passes among the 3D layers' passes. A
    /// layer shades its hillshade where `shades` has the node's group.
    /// Returns its draws and the slots it redrew.
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (frame, props): (BufferId, BindGroupEntry),
        (drape, shades): (&[(u32, RenderBundleId)], &[Option<BindGroupId>]),
        (depth, depth_store): (TextureId, StoreOp),
        downsampler: &crate::downsample::Downsampler,
        (passes, layers): (&mut Vec<Pass>, &mut Vec<RenderPass>),
    ) -> TerrainDrawn {
        let mut drawn = TerrainDrawn::default();
        if !self.ready(gpu) || !downsampler.ready(gpu) {
            // The map placed this frame's drape for these slots only: it
            // draws them all again once terrain can.
            if !self.pending.is_empty() {
                self.redraw_all();
            }
            return drawn;
        }
        // The last frame, which drew into them, was submitted.
        std::mem::take(&mut self.mip_targets).release(gpu);
        let (Some(colour), Some(dem)) = (&self.atlas[0], self.dem) else {
            return drawn;
        };
        if !self.pending.is_empty() {
            let start = passes.len();
            self.pending.sort_unstable();
            self.pending.dedup();
            drawn.slots = self.pending.len() as u32;
            let layout = self.layout_state;
            for (page, atlas) in self.atlas.iter().enumerate() {
                let Some(atlas) = atlas else {
                    continue;
                };
                // (stack, slot), each stack's in turn, so each layer's.
                let pending: Vec<(u32, u32)> = (self.pending.iter())
                    .filter(|p| p.0 == page)
                    .map(|&(_, stack, slot)| (stack, slot))
                    .collect();
                if pending.is_empty() {
                    continue;
                }
                let (columns, _) = atlas.grid();
                let px = atlas.slot_px;
                let regions: Vec<[u32; 4]> = (pending.iter())
                    .map(|&(stack, s)| {
                        [layout.layer(stack), s % columns * px, s / columns * px, px]
                    })
                    .collect();
                atlas.mips.regions.write(gpu, &regions);
                // A pass a layer: its slots' regions cleared, then each
                // stack's bundle replayed once per rectangle of its redrawn
                // slots, scissored to it: what a coarse tile standing in
                // draws spans far more than the slots it draws into (all of
                // the layer), and would otherwise be rasterized there only
                // to be discarded.
                let mut first = 0;
                for run in pending.chunk_by(|a, b| layout.layer(a.0) == layout.layer(b.0)) {
                    let end = first + run.len() as u32;
                    let mut commands = vec![
                        RenderCommand::SetPipeline(self.clear),
                        RenderCommand::SetBindGroup {
                            index: 0,
                            group: atlas.clear,
                        },
                        RenderCommand::Draw {
                            vertices: 0..4,
                            instances: first..end,
                        },
                    ];
                    first = end;
                    for stack_run in run.chunk_by(|a, b| a.0 == b.0) {
                        let stack = stack_run[0].0;
                        let Some(&(_, bundle)) = drape.iter().find(|d| d.0 == stack) else {
                            continue;
                        };
                        let slots: Vec<u32> = stack_run.iter().map(|p| p.1).collect();
                        for [x, y, width, height] in slot_runs(&slots, columns) {
                            commands.push(RenderCommand::SetScissor([
                                x * px,
                                y * px,
                                width * px,
                                height * px,
                            ]));
                            commands.push(RenderCommand::ExecuteBundles(vec![bundle]));
                        }
                    }
                    passes.push(Pass::Render(RenderPass {
                        label: "drape",
                        colors: vec![ColorAttachment {
                            target: atlas.draw[layout.layer(run[0].0) as usize],
                            resolve: None,
                            store: StoreOp::Store,
                            load: LoadOp::Load,
                        }],
                        depth: None,
                        commands,
                    }));
                    drawn.draws += 1;
                }
                // Each level from the one above, over the page's slots.
                let chain = ("drape mips", &atlas.mips);
                let targets = &mut self.mip_targets;
                let (mips, _) =
                    (downsampler.mip_passes(gpu, chain, &regions, targets)).expect("ready above");
                passes.extend(mips);
            }
            self.pending.clear();
            Pass::group(passes, start, "drape");
        }
        let atlas_texture = colour.texture;
        let slope_texture = self.atlas[1].as_ref().map_or(atlas_texture, |a| a.texture);
        if self.group.is_some_and(|(_, bound)| bound != (frame, props)) {
            // The scene's buffers grew.
            self.forget_group(gpu);
        }
        let group = match self.group {
            Some((group, _)) => group,
            None => {
                let group = gpu.create_bind_group(&BindGroupDesc {
                    label: "terrain",
                    layout: self.layout,
                    entries: &[
                        BindGroupEntry::buffer(0, frame),
                        self.tiles.entry(1),
                        self.grid.entry(2),
                        BindGroupEntry::texture(3, dem.0, D2Array),
                        BindGroupEntry::texture(4, atlas_texture, D2Array),
                        BindGroupEntry::sampler(5, self.linear),
                        BindGroupEntry::buffer(6, self.params),
                        BindGroupEntry::buffer(7, self.atmosphere),
                        props,
                        BindGroupEntry::texture(9, slope_texture, D2Array),
                    ],
                });
                self.group = Some((group, (frame, props)));
                group
            }
        };
        let occlusion_pipeline = self.occlusion.map(|_| self.occlusion_pipeline(gpu));
        let samples = self.target_samples.max(1);
        let pipelines = [false, true].map(|shaded| {
            let wanted = shaded && shades.iter().any(Option::is_some);
            (!shaded || wanted).then(|| self.pipeline(gpu, samples, shaded))
        });
        let step = GRID_STEPS.iter().position(|&s| s == self.grid_step);
        let indices = self.grid_ranges[step.expect("a grid step")].clone();
        // A shaded layer's backdrop: the layer's before it, or blank.
        let backdrops: Vec<Option<BindGroupId>> = (0..self.targets.len())
            .map(|layer| {
                shades.get(layer).copied().flatten()?;
                Some(self.backdrop_group(gpu, layer))
            })
            .collect();
        let draw = |pipeline, layer_group, shade: Option<(BindGroupId, BindGroupId)>| {
            let mut commands = vec![
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup { index: 0, group },
                RenderCommand::SetBindGroup {
                    index: 1,
                    group: layer_group,
                },
            ];
            if let Some((shade, backdrop)) = shade {
                commands.push(RenderCommand::SetBindGroup {
                    index: 2,
                    group: shade,
                });
                commands.push(RenderCommand::SetBindGroup {
                    index: 3,
                    group: backdrop,
                });
            }
            commands.extend([
                RenderCommand::SetIndexBuffer {
                    buffer: self.grid_indices,
                    offset: 0,
                },
                RenderCommand::DrawIndexed {
                    indices: indices.clone(),
                    base_vertex: 0,
                    instances: 0..self.tile_count,
                },
            ]);
            commands
        };
        // The terrain's depth first, for what it hides.
        if let (Some([color, occlusion_depth]), Some(&layer_group), Some(occlusion_pipeline)) = (
            self.occlusion,
            self.stack_groups.first(),
            occlusion_pipeline,
        ) {
            passes.push(Pass::Render(RenderPass {
                label: "terrain depth",
                colors: vec![ColorAttachment {
                    target: color,
                    resolve: None,
                    store: StoreOp::Store,
                    load: LoadOp::Clear([0.0; 4]),
                }],
                depth: Some(DepthAttachment {
                    target: occlusion_depth,
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Discard,
                }),
                commands: draw(occlusion_pipeline, layer_group, None),
            }));
            drawn.draws += 1;
        }
        for (layer, target) in self.targets.iter().enumerate() {
            let shade = (shades.get(layer).copied().flatten()).zip(backdrops[layer]);
            let pipeline = pipelines[usize::from(shade.is_some())].expect("made above");
            layers.push(RenderPass {
                label: "terrain",
                colors: vec![target.target.attachment(0, LoadOp::Clear([0.0; 4]))],
                depth: Some(DepthAttachment {
                    target: depth,
                    load: if layer == 0 {
                        LoadOp::Clear(0.0)
                    } else {
                        LoadOp::Load
                    },
                    store: depth_store,
                }),
                commands: draw(pipeline, self.stack_groups[layer], shade),
            });
            drawn.draws += 1;
        }
        drawn
    }

    /// Terrain layer `layer`'s backdrop group: the texture of the layer
    /// before it, where its hillshade blends into that, else a blank one.
    fn backdrop_group(&mut self, gpu: &mut dyn Gpu, layer: usize) -> BindGroupId {
        let previous = layer
            .checked_sub(1)
            .filter(|_| self.layers.get(layer).is_some_and(|l| l.backdrop))
            .map(|l| self.targets[l].target.texture(0));
        let make = |gpu: &mut dyn Gpu, layout, texture| {
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "terrain backdrop",
                layout,
                entries: &[BindGroupEntry::texture(
                    0,
                    texture,
                    TextureViewDimension::D2,
                )],
            });
            (texture, group)
        };
        let Some(texture) = previous else {
            if let Some((_, group)) = self.blank {
                return group;
            }
            let texture = gpu.create_texture(&TextureDesc {
                label: "terrain backdrop",
                width: 1,
                height: 1,
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format: self.target.color,
                usage: TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_levels: 1,
            });
            let blank = make(gpu, self.backdrop_layout, texture);
            self.blank = Some(blank);
            return blank.1;
        };
        if self.backdrops.len() <= layer {
            self.backdrops.resize(layer + 1, None);
        }
        match self.backdrops[layer] {
            Some((bound, group)) if bound == texture => group,
            stale => {
                stale.map(|(_, group)| group).release(gpu);
                let made = make(gpu, self.backdrop_layout, texture);
                self.backdrops[layer] = Some(made);
                made.1
            }
        }
    }

    /// The drape frame: the map's, at its zoom, drawn into page `page`.
    pub(crate) fn drape_frame(
        &self,
        frame: &crate::scene::FrameBlock,
        page: usize,
    ) -> crate::scene::FrameBlock {
        let [w, h] = self.atlas[page]
            .as_ref()
            .map_or([1.0; 2], |a| a.size.map(|s| s as f32));
        // Drape world (x, −y) for texel (x, y) → clip.
        let view_proj = [
            [2.0 / f64::from(w), 0.0, 0.0, 0.0],
            [0.0, 2.0 / f64::from(h), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        crate::scene::FrameBlock {
            view_proj,
            // Drape placements' origins are in drape space already.
            camera: [0.0; 2],
            viewport: [w, h],
            // Drape placements' `px_scale` is their texels per logical
            // pixel at the map's zoom.
            world_per_px: 1.0,
            pixel_ratio: 1.0,
            align: [0.0; 2],
            pitch: 0.0,
            // Slots are flat: the terrain carries them onto a globe.
            globe: None,
            // Slots are cached: tiles draw into them as they settle, as GL
            // JS doesn't fade what it renders to texture. Feature-state
            // fades keep the frame's `state_time`: the runtime redraws the
            // slots each frame while one runs.
            time: f32::MAX,
            ..*frame
        }
    }

    pub(crate) fn disable(&mut self, gpu: &mut dyn Gpu) {
        self.paint = None;
        self.release_atlas(gpu);
        self.forget_group(gpu);
        for target in self.targets.drain(..) {
            target.release(gpu);
        }
        self.release_occlusion(gpu);
        self.target_size = [0; 2];
        self.dem = None;
        self.pending.clear();
        self.layout_state.tile_slots = 0;
    }

    /// The DEM page texture elevated content reads, while terrain is on.
    pub(crate) fn dem_texture(&self) -> Option<TextureId> {
        self.paint.and(self.dem).map(|(texture, _)| texture)
    }

    pub(crate) fn active(&self) -> bool {
        self.paint.is_some() && self.atlas[0].is_some()
    }

    /// The terrain's depth symbols and circles read, while terrain is on.
    pub(crate) fn occlusion_texture(&self) -> Option<TextureId> {
        self.occlusion
            .filter(|_| self.active())
            .map(|[color, _]| color)
    }

    fn release_occlusion(&mut self, gpu: &mut dyn Gpu) {
        self.occlusion.take().release(gpu);
    }

    /// The terrain drawn as its depth, at one sample.
    fn occlusion_pipeline(&mut self, gpu: &mut dyn Gpu) -> RenderPipelineId {
        if let Some(pipeline) = self.occlusion_pipeline {
            return pipeline;
        }
        let target = TargetFormat {
            color: TextureFormat::R32Float,
            sample_count: 1,
        };
        let pipeline = Draw {
            label: "terrain depth",
            fragment: (&TERRAIN_SHADER, "fs_terrain_depth"),
            depth: DEPTH,
            ..Draw::new(
                &TERRAIN_SHADER,
                &[self.layout, self.stack_layout],
                target,
                None,
            )
        }
        .make(gpu);
        self.occlusion_pipeline = Some(pipeline);
        pipeline
    }
}

/// The shared tile mesh: `(GRID + 1)²` vertices, then a skirt copy of
/// each edge vertex; and per [`GRID_STEPS`] step, the index range of two
/// triangles per cell of that many vertices a side and per edge segment.
fn grid() -> (Vec<u32>, Vec<u32>, [std::ops::Range<u32>; GRID_STEPS.len()]) {
    let n = TERRAIN_GRID;
    let side = n + 1;
    let pack = |x: u32, y: u32, skirt: u32| x | y << 8 | skirt << 16;
    let mut vertices: Vec<u32> = (0..side)
        .flat_map(|y| (0..side).map(move |x| pack(x, y, 0)))
        .collect();
    let at = |x: u32, y: u32| y * side + x;
    // Edges in order along each side: north, south, west, east.
    let edge = |edge_index: u32, i: u32| match edge_index {
        0 => (i, 0),
        1 => (i, n),
        2 => (0, i),
        _ => (n, i),
    };
    let skirt_base = vertices.len() as u32;
    for edge_index in 0..4 {
        for i in 0..side {
            let (x, y) = edge(edge_index, i);
            vertices.push(pack(x, y, 1));
        }
    }
    let mut indices = Vec::new();
    let ranges = GRID_STEPS.map(|s| {
        let start = indices.len() as u32;
        for y in (0..n).step_by(s as usize) {
            for x in (0..n).step_by(s as usize) {
                indices.extend([
                    at(x, y),
                    at(x, y + s),
                    at(x + s, y + s),
                    at(x, y),
                    at(x + s, y + s),
                    at(x + s, y),
                ]);
            }
        }
        for edge_index in 0..4 {
            let base = skirt_base + edge_index * side;
            for i in (0..n).step_by(s as usize) {
                let (x0, y0) = edge(edge_index, i);
                let (x1, y1) = edge(edge_index, i + s);
                let (a, b) = (at(x0, y0), at(x1, y1));
                let (sa, sb) = (base + i, base + i + s);
                indices.extend([a, sa, sb, a, sb, b]);
            }
        }
        start..indices.len() as u32
    });
    (vertices, indices, ranges)
}

impl crate::release::Release for Terrain {
    fn release(mut self, gpu: &mut dyn Gpu) {
        self.release_atlas(gpu);
        self.release_occlusion(gpu);
        self.forget_group(gpu);
        let layouts = (self.layout, self.stack_layout, self.clear_layout);
        let backdrops: Vec<BindGroupId> = (self.backdrops.iter().flatten()).map(|b| b.1).collect();
        (self.backdrop_layout, backdrops, self.blank).release(gpu);
        let pipelines = (self.pipelines, self.clear, self.occlusion_pipeline);
        self.grid.destroy(gpu);
        self.tiles.destroy(gpu);
        let buffers = (self.grid_indices, self.params);
        (layouts, pipelines, buffers, self.stack_uniforms).release(gpu);
        (self.stack_groups, self.targets, self.mip_targets).release(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redrawn_slots_replay_in_rectangles() {
        // Four columns: 2, 3 run together; 4 starts the next row.
        assert_eq!(
            slot_runs(&[0, 2, 3, 4, 5, 7], 4),
            [[0, 0, 1, 1], [2, 0, 2, 1], [0, 1, 2, 1], [3, 1, 1, 1]]
        );
        // Whole rows join: one stack of every tile slot from slot 6 on.
        assert_eq!(
            slot_runs(&(6..20).collect::<Vec<_>>(), 4),
            [[2, 1, 2, 1], [0, 2, 4, 3]]
        );
    }

    #[test]
    fn the_grid_has_cells_and_skirts() {
        let (vertices, indices, ranges) = grid();
        let side = TERRAIN_GRID + 1;
        assert_eq!(vertices.len() as u32, side * side + 4 * side);
        assert!(indices.iter().all(|&i| (i as usize) < vertices.len()));
        let skirts = vertices.iter().filter(|v| *v >> 16 == 1).count() as u32;
        assert_eq!(skirts, 4 * side);
        for (step, range) in GRID_STEPS.into_iter().zip(ranges) {
            let n = TERRAIN_GRID / step;
            assert_eq!(range.len() as u32, n * n * 6 + 4 * n * 6, "step {step}");
            // Every vertex it reads lies on its step's grid.
            for &i in &indices[range.start as usize..range.end as usize] {
                let (x, y) = (vertices[i as usize] & 255, vertices[i as usize] >> 8 & 255);
                assert_eq!((x % step, y % step), (0, 0), "step {step}");
            }
        }
    }

    #[test]
    fn coarser_dem_tiles_draw_with_gl_js_density() {
        let steps = [64.0, 256.0, 512.0, 1024.0, 4096.0].map(grid_step);
        assert_eq!(steps, [1, 1, 2, 4, 4]);
    }

    #[test]
    fn atlas_grids_are_powers_of_two_with_room_to_change() {
        assert_eq!(atlas_grid(0, None, 1, 16), Some((1, 1, 1)));
        assert_eq!(atlas_grid(24, None, 1, 16), Some((1, 8, 4)));
        assert_eq!(atlas_grid(33, None, 1, 16), Some((1, 8, 8)));
        // Kept while it holds them and they fill over a quarter of it.
        assert_eq!(atlas_grid(32, Some((1, 8, 4)), 1, 16), None);
        assert_eq!(atlas_grid(9, Some((1, 8, 4)), 1, 16), None);
        assert_eq!(atlas_grid(33, Some((1, 8, 4)), 1, 16), Some((1, 8, 8)));
        // Shrunk to twice the fewest, half full.
        assert_eq!(atlas_grid(8, Some((1, 8, 4)), 1, 16), Some((1, 4, 4)));
        // Never past the device's largest.
        assert_eq!(atlas_grid(64, None, 1, 8), Some((1, 8, 8)));
    }

    #[test]
    fn stacks_share_a_layer_while_they_fit_it() {
        // Two stacks of 20 tiles: one layer of 64 slots, 32 a stack.
        assert_eq!(atlas_grid(20, None, 2, 8), Some((2, 8, 8)));
        // Of 33: a layer each, a stack holding a whole layer's 64.
        assert_eq!(atlas_grid(33, None, 2, 8), Some((1, 8, 8)));
        assert_eq!(atlas_grid(64, None, 2, 8), Some((1, 8, 8)));
        // Kept on falling back to 20, which fill over a quarter of 64.
        assert_eq!(atlas_grid(20, Some((1, 8, 8)), 2, 8), None);
        // Three stacks of 20: two a layer.
        assert_eq!(atlas_grid(20, None, 3, 8), Some((3, 8, 8)));
        assert_eq!(atlas_grid(22, None, 3, 8), Some((2, 8, 8)));
    }

    #[test]
    fn slots_fill_rows_of_the_atlas() {
        let layout = DrapeLayout {
            stacks: [3, 1],
            per_layer: [2, 1],
            columns: [8, 4],
            strides: [16, 16],
            tile_slots: 16,
            generation: 1,
        };
        // Each stack's slots side by side: stack 1 after stack 0's 16.
        assert_eq!(layout.corner(1, 0), [DRAPE_SLOT_PX, 0]);
        assert_eq!(layout.corner(0, 1), [0, 2 * DRAPE_SLOT_PX]);
        assert_eq!(layout.corner(9, 0), [DRAPE_SLOT_PX, DRAPE_SLOT_PX]);
        assert_eq!((layout.layer(0), layout.layer(1)), (0, 0));
        // Stack 2 starts the next layer.
        assert_eq!(layout.layer(2), 1);
        assert_eq!(layout.corner(1, 2), [DRAPE_SLOT_PX, 0]);
        // Slopes on their own page, at half the texels a side.
        assert_eq!(layout.page(3), (1, 0));
        assert_eq!(layout.corner(1, 3), [SLOPE_SLOT_PX, 0]);
        assert_eq!(layout.corner(5, 3), [SLOPE_SLOT_PX, SLOPE_SLOT_PX]);
    }
}
