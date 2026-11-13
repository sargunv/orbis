//! Terrain (render graph `Terrain`/`Drape`): the map's 2D
//! layers draped over a DEM-raised surface, after GL JS's render to
//! texture.
//!
//! Each terrain tile has a drape slot per drape stack (a run of draped
//! layers between layers that draw in 3D or on screen, the part of one
//! above its first layer whose drawing changes with the zoom, or a
//! hillshade the terrain shades, whose stack holds its unlit slopes).
//! Slots are squares of mipmapped pages (a tile that shows small draws the
//! top-left square of its slots that it shows at,
//! [`TerrainTile::shrink`]): colour stacks in one, slope stacks at half the
//! texels a side in another, a layer of its page a stack. A page is
//! sheets of 2 × 2 slots, each a texture of its own, which lie side by
//! side in one grid of drape space: pages grow by a sheet, blank, and
//! shrink by sheets no slot holds, and no slot ever moves to another
//! texture. A frame redraws only the slots the runtime marks: their drape
//! placements, the draped nodes' only candidates, map them into the grid,
//! so one pass a sheet's layer redraws its slots, replaying each stack's
//! bundle of its draped nodes over them with the viewport on the sheet,
//! then a compute pass a sheet downsamples their mip levels. The terrain
//! draws each sheet's tiles with it bound. Each terrain layer (a stack,
//! two where the zoom splits it, or up to [`MAX_SHADES`] hillshades lit
//! between stacks from their slopes' stacks and the frame's bearing)
//! draws into a texture of its own with the depth extrusions share,
//! composited in the map at its place.

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

/// Slots a side of a drape sheet: each a square of 2 × 2 slots, a texture
/// of its own.
const SHEET_SIDE: u32 = 2;
/// Tile slots a sheet holds: slots `4s` to `4s + 3` are sheet `s`'s.
pub const SHEET_SLOTS: u32 = SHEET_SIDE * SHEET_SIDE;

/// Slots a side of the grid the sheets lie in on this device, whose
/// sides stop at WebGPU's default, 8192: the drape space a page's slots
/// draw in, each sheet a square of it.
fn grid_side(gpu: &dyn Gpu) -> u32 {
    let side = gpu.caps().limits.max_texture_dimension_2d.min(8192) / DRAPE_SLOT_PX;
    (side / SHEET_SIDE * SHEET_SIDE).max(SHEET_SIDE)
}

/// Where tile slot `slot` lies in a grid `side` slots a side: its column
/// and row. Sheets fill the grid's rows of sheets in turn, and each its
/// four places by rows.
fn slot_place(slot: u32, side: u32) -> [u32; 2] {
    let sheets = side / SHEET_SIDE;
    let origin = sheet_origin(slot / SHEET_SLOTS, sheets);
    let [x, y] = sheet_place(slot);
    [origin[0] + x, origin[1] + y]
}

/// Sheet `sheet`'s top-left slot in a grid of `sheets` sheets a side.
fn sheet_origin(sheet: u32, sheets: u32) -> [u32; 2] {
    [sheet % sheets, sheet / sheets].map(|n| n * SHEET_SIDE)
}

/// Where tile slot `slot` lies in its sheet: its column and row.
fn sheet_place(slot: u32) -> [u32; 2] {
    let at = slot % SHEET_SLOTS;
    [at % SHEET_SIDE, at / SHEET_SIDE]
}

/// The terrain tiles the drape can hold with `stacks` colour and slope
/// stacks: the slots of a grid of WebGPU's default texture side (64
/// colour slots) on every device that has it, so that a view's terrain
/// doesn't depend on which, or none where a sheet of the stacks takes
/// more layers or bytes than the device allocates a texture
/// ([`Gpu::texture_fits`], GL's [`Limits::max_texture_bytes`]).
pub(crate) fn room(gpu: &dyn Gpu, stacks: [u32; 2]) -> u32 {
    let alloc = allocated(stacks);
    let fits = (0..2).filter(|&p| alloc[p] > 0).all(|page| {
        alloc[page] <= gpu.caps().limits.max_texture_array_layers
            && gpu.texture_fits(&sheet_desc(alloc[page], PAGE_PX[page]))
    });
    if fits { grid_side(gpu).pow(2) } else { 0 }
}

/// Why the drape pages can't hold a view's terrain tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrapeFull {
    /// The style's drape stacks leave no room for them ([`room`]).
    Room,
    /// The device refused a sheet of this many bytes
    /// ([`Gpu::try_create_texture`]).
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
    /// How many times its slots halve their page's side: each takes the
    /// top-left square of that side of its place in the grid, which keeps
    /// to it at every mip level, so that a tile that shows small draws
    /// few texels.
    pub shrink: u8,
    /// Where it lies in the slots it shows, as `uv · scale + offset`: its
    /// own, or while blank its part of an ancestor's.
    pub within: ([f32; 2], f32),
    /// On a globe, where it lies on the sphere.
    pub globe: Option<TerrainGlobe>,
    /// Latitudes of its top and bottom edges, in degrees, which a shaded
    /// hillshade's slopes shrink by.
    pub latitudes: [f32; 2],
}

/// The hillshades a terrain layer shades at most: each shades a frame's
/// every pixel the layer covers, in the one pass, where a layer of its own
/// each cost another pass over the screen and its composite (1 ms of
/// 5 a 5120×2880 frame of the Globe over Nepal, M6).
pub(crate) const MAX_SHADES: usize = 2;

/// What one terrain draw composites in the map: a stack's slots, the
/// stacks below and above where the zoom splits it, or a stack below the
/// hillshades it shades and those above each of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TerrainLayer {
    pub(crate) below: Option<u8>,
    /// The hillshades it shades, in order from the first: each over what
    /// is beneath it, its colour stack above it on top.
    pub(crate) shades: [Option<Shade>; MAX_SHADES],
    /// Without a hillshade, the stack above `below` where the zoom splits
    /// the layer.
    pub(crate) above: Option<u8>,
    /// A layer's own, composited in a blend mode other than normal.
    pub(crate) blended: bool,
    /// What its first hillshade blends into is the terrain layer's before
    /// it, as it drew it, rather than a stack of its own: the hillshade
    /// lies on that layer, whose hillshades fill it.
    pub(crate) backdrop: bool,
}

impl TerrainLayer {
    /// Whether it shades as many hillshades as a layer takes.
    pub(crate) fn full(&self) -> bool {
        self.shades[MAX_SHADES - 1].is_some()
    }

    /// Whether it shades a hillshade.
    pub(crate) fn shaded(&self) -> bool {
        self.shades[0].is_some()
    }
}

/// A terrain layer's hillshades' tables, in order: none where the layer
/// shades none there, or it hides.
pub(crate) type ShadeTables = [Option<TextureId>; MAX_SHADES];

/// A hillshade a terrain layer lights from its slopes and the frame's
/// bearing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shade {
    pub(crate) node: u32,
    /// The stack its unlit slopes draw into.
    pub(crate) slopes: u8,
    /// The colour stack of the layers above it, up to the next hillshade.
    pub(crate) above: Option<u8>,
    /// How it blends into the colours beneath it: normal, or a blend mode
    /// its layer composites in, which the layer then takes.
    pub(crate) blend: crate::Blend,
}

/// How a terrain layer fogs (`Layer.fog` in `terrain.wgsl`): its content,
/// the whole ground, or by fading its coverage.
const FOG_CONTENT: u32 = 0;
const FOG_GROUND: u32 = 1;
const FOG_FADES: u32 = 2;
/// Words of a terrain layer's uniform (`terrain.wgsl`'s `Layer`).
const LAYER_WORDS: usize = 4 + 4 * MAX_SHADES;

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
/// page 1 at [`SLOPE_SLOT_PX`]. A page has a layer a stack, the stack of
/// rank `r` among the page's in layer `r`, in each of its sheets. Tile
/// slot `t` lies in sheet `t / 4` ([`SHEET_SLOTS`]), and the sheets lie
/// in a grid `side` slots a side, the same for both pages: each a square
/// of 2 × 2 slots, filling the grid's rows of sheets in turn. Where a
/// slot lies is a function of its index alone, whatever the pages hold.
/// Each layer's drape space is the grid's texels with y up: texel
/// `(x, y)` is drape world `(x, −y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrapeLayout {
    /// Colour stacks, then slope stacks.
    pub stacks: [u32; 2],
    /// Slots a side of the grid the sheets lie in.
    pub side: u32,
    /// Tile slots the pages hold: their sheets' slots.
    pub tile_slots: u32,
    /// Changes when every slot is blank (other stacks): every slot must
    /// be redrawn.
    pub generation: u64,
}

impl DrapeLayout {
    /// The stacks of both pages.
    pub fn stack_count(&self) -> u32 {
        self.stacks[0] + self.stacks[1]
    }

    /// The page of stack `stack`, and its rank among the page's stacks:
    /// the layer of the page it lies in.
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
        self.page(stack).1
    }

    /// Tile slot `tile_slot`'s slot of `stack`: its top-left texel in the
    /// grid's drape space.
    pub fn corner(&self, tile_slot: u32, stack: u32) -> [u32; 2] {
        let px = self.slot_px(stack);
        slot_place(tile_slot, self.side).map(|n| n * px)
    }
}

/// The scissor around a stack's redrawn `slots` (slot, side) of a sheet
/// of `px`-texel slots: their places' bounding rectangle in the sheet, or
/// a slot alone's side.
fn scissor(slots: &[(u32, u32)], px: u32) -> [u32; 4] {
    if let [(slot, side)] = *slots {
        let [x, y] = sheet_place(slot);
        return [x * px, y * px, side, side];
    }
    let (mut lo, mut hi) = ([u32::MAX; 2], [0; 2]);
    for &(slot, _) in slots {
        let at = sheet_place(slot);
        lo = [0, 1].map(|i| lo[i].min(at[i]));
        hi = [0, 1].map(|i| hi[i].max(at[i] + 1));
    }
    [
        lo[0] * px,
        lo[1] * px,
        (hi[0] - lo[0]) * px,
        (hi[1] - lo[1]) * px,
    ]
}

/// A drape sheet's texture: `layers` layers of 2 × 2 slots of `slot_px`
/// texels, mipmapped down from those.
fn sheet_desc(layers: u32, slot_px: u32) -> TextureDesc<'static> {
    TextureDesc {
        label: "drape sheet",
        width: SHEET_SIDE * slot_px,
        height: SHEET_SIDE * slot_px,
        array_layers: layers,
        dimension: TextureViewDimension::D2Array,
        format: DRAPE_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
        sample_count: 1,
        mip_levels: slot_px.ilog2() + 1,
    }
}

/// A sheet of a drape page: 2 × 2 slots a stack.
#[derive(Debug)]
struct Sheet {
    /// A layer a stack of the page, mipmapped.
    texture: TextureId,
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

impl Sheet {
    /// A sheet of `layers` layers of `slot_px`-texel slots, or the bytes
    /// of one the device refused.
    fn new(
        gpu: &mut dyn Gpu,
        (layers, slot_px): (u32, u32),
        clear_layout: BindGroupLayoutId,
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<Self, u64> {
        let desc = sheet_desc(layers, slot_px);
        let side = desc.width;
        let levels = desc.mip_levels;
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
        let slots = ("drape slots", SHEET_SLOTS * layers);
        let mips = downsampler.chain(gpu, (texture, &views), slots);
        let size_uniform = gpu.create_buffer(&BufferDesc {
            label: "drape sheet size",
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let words = [side as f32, side as f32, 0.0, 0.0];
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
            layers,
            slot_px,
            levels: views,
            draw,
            size_uniform,
            clear,
            mips,
        })
    }

    /// Its bytes, mips included.
    fn bytes(&self) -> u64 {
        let side = u64::from(SHEET_SIDE * self.slot_px);
        side * side * u64::from(self.layers) * 4 * 4 / 3
    }
}

impl crate::release::Release for Sheet {
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
    /// A shaded layer's hillshades' tables, which colour their slopes.
    shade_layout: BindGroupLayoutId,
    /// Each terrain layer's shade group and the tables it binds.
    shade_groups: Vec<Option<(ShadeTables, BindGroupId)>>,
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
    /// The colour page's sheets, and the slope page's while a hillshade is
    /// shaded, as many.
    sheets: [Vec<Sheet>; 2],
    layout_state: DrapeLayout,
    /// The DEM page texture and revision the groups bind.
    dem: Option<(TextureId, u32)>,
    /// A group a sheet, binding its colour and slope sheets, and the frame
    /// and property buffers they bind.
    groups: Vec<BindGroupId>,
    bound: Option<(BufferId, BindGroupEntry)>,
    /// The terrain tiles of each sheet's slots: the table holds them
    /// sheet by sheet, and each sheet's draw takes its range.
    sheet_tiles: Vec<std::ops::Range<u32>>,
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
    /// Slots to redraw this frame: (page, sheet, stack, tile slot, side),
    /// each page's sheets' in turn, and in a sheet each stack's, so each
    /// layer's.
    pending: Vec<(usize, u32, u32, u32, u32)>,
    /// The style zoom `pending` slots draw at: the drape pass's frame's.
    zoom: f64,
    /// What the last frame's drawn mip levels rendered into, released
    /// by the next.
    mip_targets: crate::downsample::Transient,
    pub(crate) paint: Option<TerrainPaint>,
    /// The atmosphere's uniform, for fog.
    atmosphere: BufferId,
    /// What remaking the drape pages did since the last
    /// [`take_resized`](Self::take_resized).
    resized: crate::scene::Resized,
}

/// What a frame's terrain passes drew: render passes, and drape slots
/// redrawn.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TerrainDrawn {
    pub(crate) draws: u32,
    pub(crate) slots: u32,
}

impl Terrain {
    /// Terrain drawn into `target`, fogged by `atmosphere`, its drape
    /// sampled with `linear`.
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        atmosphere: BufferId,
        linear: SamplerId,
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
        let shade_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain shades",
            entries: &[
                E::sampler(0, fs, true),
                E::texture(1, fs, true, TextureViewDimension::D2),
                E::texture(2, fs, true, TextureViewDimension::D2),
            ],
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
            shade_groups: Vec::new(),
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
            sheets: [Vec::new(), Vec::new()],
            layout_state: DrapeLayout {
                stacks: [1, 0],
                side: grid_side(gpu),
                tile_slots: 0,
                generation: 0,
            },
            dem: None,
            groups: Vec::new(),
            bound: None,
            sheet_tiles: Vec::new(),
            targets: Vec::new(),
            occlusion: None,
            occlusion_pipeline: None,
            target_size: [0; 2],
            occlusion_size: [0; 2],
            target_samples: 0,
            pending: Vec::new(),
            zoom: 0.0,
            mip_targets: Default::default(),
            paint: None,
            atmosphere,
            resized: Default::default(),
        }
    }

    /// What remaking the drape pages did since the last call.
    pub(crate) fn take_resized(&mut self) -> crate::scene::Resized {
        std::mem::take(&mut self.resized)
    }

    /// The drape pages' bytes, mips included.
    pub(crate) fn atlas_bytes(&self) -> u64 {
        self.sheets.iter().flatten().map(Sheet::bytes).sum()
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
                            LAYER_WORDS as u64 * 4,
                        )],
                    })
                })
                .collect();
            self.stack_uniforms = Some(buffer);
        }
        let buffer = self.stack_uniforms.expect("made above");
        for (i, layer) in layers.iter().enumerate() {
            let stack = |s: Option<u8>| s.map_or(NONE, u32::from);
            // The first layer is the ground: it fogs where it draws
            // nothing too, and the layers over it only their own content,
            // but for a blended one, which fades into the fog: what it
            // blends with is fogged already.
            let fog = match () {
                () if layer.blended => FOG_FADES,
                () if i == 0 => FOG_GROUND,
                () => FOG_CONTENT,
            };
            let mut words = vec![
                stack(layer.below),
                stack(layer.above),
                fog,
                u32::from(layer.backdrop),
            ];
            for shade in layer.shades {
                // A slope stack by its rank on the slope page.
                words.extend(match shade {
                    Some(s) => [
                        u32::from(s.slopes) - colour,
                        stack(s.above),
                        s.blend.mode() as u32,
                        0,
                    ],
                    None => [NONE, NONE, 0, 0],
                });
            }
            debug_assert_eq!(words.len(), LAYER_WORDS);
            gpu.write_buffer(
                buffer,
                UNIFORM_STRIDE * i as u64,
                bytemuck::cast_slice(&words),
            );
        }
        self.layers = layers.to_vec();
    }

    /// Makes the pages hold `tile_slots` tile slots of `stacks` colour and
    /// slope stacks, in the fewest sheets that do: sheets are added blank
    /// and the last ones let go, the rest staying as they are, slots and
    /// all. Pages of other stacks are new, blank, of a new
    /// [`DrapeLayout::generation`]: whether it made them.
    /// Fails past the device's [`room`], or when the device refuses a
    /// sheet, which leaves none.
    pub(crate) fn reserve(
        &mut self,
        gpu: &mut dyn Gpu,
        (tile_slots, stacks): (u32, [u32; 2]),
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<bool, DrapeFull> {
        if tile_slots > room(gpu, stacks) {
            return Err(DrapeFull::Room);
        }
        let state = self.layout_state;
        let made = state.stacks != stacks || self.sheets[0].is_empty();
        let sheets = tile_slots.max(1).div_ceil(SHEET_SLOTS);
        let before = gpu.counters();
        let mut changed = made;
        if made {
            self.release_sheets(gpu);
            self.pending.clear();
        }
        let alloc = allocated(stacks);
        let clear_layout = self.clear_layout;
        for (page, held) in self.sheets.iter_mut().enumerate() {
            let want = if alloc[page] > 0 { sheets as usize } else { 0 };
            changed |= held.len() != want;
            while held.len() > want {
                held.pop().release(gpu);
            }
            while held.len() < want {
                let size = (alloc[page], PAGE_PX[page]);
                match Sheet::new(gpu, size, clear_layout, downsampler) {
                    Ok(sheet) => held.push(sheet),
                    Err(bytes) => {
                        self.release_sheets(gpu);
                        return Err(DrapeFull::Refused(bytes));
                    }
                }
            }
        }
        let tile_slots = sheets * SHEET_SLOTS;
        // Slots of the sheets let go draw nothing more.
        self.pending.retain(|p| p.3 < tile_slots);
        if changed {
            self.forget_group(gpu);
            self.resized.count(gpu, &before);
        }
        self.layout_state = DrapeLayout {
            stacks,
            side: grid_side(gpu),
            tile_slots,
            generation: state.generation + u64::from(made),
        };
        Ok(made)
    }

    /// Releases every sheet.
    fn release_sheets(&mut self, gpu: &mut dyn Gpu) {
        std::mem::take(&mut self.sheets).release(gpu);
        self.forget_group(gpu);
    }

    fn forget_group(&mut self, gpu: &mut dyn Gpu) {
        std::mem::take(&mut self.groups).release(gpu);
        self.bound = None;
    }

    /// Redraws tile slots' stacks this frame at style zoom `zoom`: per
    /// tile slot, its [`shrink`](TerrainTile::shrink) and the stacks to
    /// redraw.
    pub(crate) fn redraw(&mut self, zoom: f64, tile_slots: &[(u32, u8, &[u8])]) {
        self.zoom = zoom;
        self.pending.clear();
        let layout = self.layout_state;
        for &(slot, shrink, redrawn) in tile_slots {
            let stacks =
                (redrawn.iter().map(|&s| u32::from(s))).filter(|&s| s < layout.stack_count());
            self.pending.extend(stacks.map(|stack| {
                let page = layout.page(stack).0;
                let side = layout.slot_px(stack) >> shrink;
                (page, slot / SHEET_SLOTS, stack, slot, side)
            }));
        }
    }

    /// The stacks this frame redraws slots of, in order.
    pub(crate) fn pending_stacks(&self) -> Vec<u32> {
        let mut stacks: Vec<u32> = self.pending.iter().map(|p| p.2).collect();
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
        // Sheet by sheet, each sheet's tiles in the order given: a sheet's
        // draw binds it.
        let mut order: Vec<&TerrainTile> = tiles.iter().collect();
        order.sort_by_key(|t| t.slot / SHEET_SLOTS);
        self.sheet_tiles.clear();
        for (at, tile) in (0u32..).zip(&order) {
            let sheet = (tile.slot / SHEET_SLOTS) as usize;
            if self.sheet_tiles.len() <= sheet {
                self.sheet_tiles.resize(sheet + 1, at..at);
            }
            self.sheet_tiles[sheet].end = at + 1;
        }
        let words: Vec<u32> = order
            .into_iter()
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
                    .u(n::SLOT, &[t.slot])
                    .u(n::SHRINK, &[u32::from(t.shrink)]);
                b.f(n::SLOT_OFFSET, &t.within.0)
                    .f(n::SLOT_SCALE, &[t.within.1]);
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
        let mut b = orb_shaders::Block::new(params::WORDS);
        b.f(params::UNPACK, &paint.unpack)
            .f(params::EXAGGERATION, &[paint.exaggeration]);
        b.f(params::DEM_DIM, &[dim as f32])
            .f(params::SKIRT, &[paint.skirt]);
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

    /// Asks for the pipelines the terrain draws with at `samples`, a
    /// shaded hillshade's too where one is (`shaded`): the pipeline set's
    /// ([`Renderer::set_pipelines`](crate::Renderer::set_pipelines)).
    pub(crate) fn request_pipelines(&mut self, gpu: &mut dyn Gpu, samples: u32, shaded: bool) {
        self.pipeline(gpu, samples, false);
        if shaded {
            self.pipeline(gpu, samples, true);
        }
        self.occlusion_pipeline(gpu);
    }

    /// The pipeline terrain layers draw with at `samples`, shading a
    /// hillshade or not, once asked for.
    fn made(&self, samples: u32, shaded: bool) -> Option<RenderPipelineId> {
        self.pipelines[usize::from(shaded)][usize::from(samples > 1)]
    }

    /// Whether every terrain pipeline at the current sample count
    /// compiled. Until then terrain draws nothing.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        let samples = self.target_samples.max(1);
        let shaded = self.layers.iter().any(TerrainLayer::shaded);
        let wanted = [
            Some(Some(self.clear)),
            Some(self.made(samples, false)),
            shaded.then(|| self.made(samples, true)),
            self.occlusion.map(|_| self.occlusion_pipeline),
        ];
        (wanted.into_iter().flatten()).all(|p| p.is_some_and(|p| gpu.pipeline_ready(p.into())))
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

    /// Makes the pipeline terrain layers draw with at `samples`, shading
    /// a hillshade or not, unless it is made.
    fn pipeline(&mut self, gpu: &mut dyn Gpu, samples: u32, shaded: bool) {
        if self.made(samples, shaded).is_some() {
            return;
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
    }

    /// This frame's passes before the map: redrawn drape slots (`drape`
    /// draws the draped nodes into level 0) and their mip levels into
    /// `passes`, and into `layers` each terrain layer's pass into its
    /// texture, in order, the first clearing `depth`, the rest loading it.
    /// The caller places the layer passes among the 3D layers' passes. A
    /// layer shades its hillshades where `shades` has their tables, and
    /// none where it has none (they hide).
    /// Returns its draws and the slots it redrew.
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (frame, props): (BufferId, BindGroupEntry),
        (drape, shades): (&[(u32, RenderBundleId)], &[ShadeTables]),
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
        let Some(dem) = self.dem.filter(|_| !self.sheets[0].is_empty()) else {
            return drawn;
        };
        if !self.pending.is_empty() {
            let start = passes.len();
            let layout = self.layout_state;
            let mut pending = std::mem::take(&mut self.pending);
            pending.sort_unstable();
            pending.dedup();
            drawn.slots = pending.len() as u32;
            let sheets = layout.side / SHEET_SIDE;
            let grid = layout.side as f32;
            for run in pending.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
                let (page, at) = (run[0].0, run[0].1);
                let Some(sheet) = self.sheets[page].get(at as usize) else {
                    continue;
                };
                let px = sheet.slot_px;
                // Whole places, cleared and downsampled, a smaller slot's
                // too: what filters read past its edge is then blank, not
                // what the place held before.
                let regions: Vec<[u32; 4]> = (run.iter())
                    .map(|&(_, _, stack, slot, _)| {
                        let [x, y] = sheet_place(slot);
                        [layout.layer(stack), x * px, y * px, px]
                    })
                    .collect();
                sheet.mips.regions.write(gpu, &regions);
                // The grid's drape space, which the drape frame maps onto
                // a viewport of its size, moved so that the sheet's square
                // of it lands on the sheet.
                let origin = sheet_origin(at, sheets).map(|n| -((n * px) as f32));
                let viewport = [origin[0], origin[1], grid * px as f32, grid * px as f32];
                // A pass a layer: its slots' regions cleared, then each
                // stack's bundle replayed once, scissored to the rectangle
                // around its redrawn slots (a slot alone to its side). The
                // bundle draws every redrawn slot's content, each bounded
                // by its slot, and what other sheets' slots draw falls
                // outside the viewport's square. The scissor also keeps
                // what a coarse tile standing in draws (spanning far more
                // than its slot) from being rasterized over the whole
                // layer only to be discarded. A replay costs the CPU every
                // draw of the stack, so the map redraws few sheets a frame
                // (the runtime's `Terrain::plan`).
                let mut first = 0;
                for layer_run in run.chunk_by(|a, b| layout.layer(a.2) == layout.layer(b.2)) {
                    let end = first + layer_run.len() as u32;
                    let mut commands = vec![
                        RenderCommand::SetPipeline(self.clear),
                        RenderCommand::SetBindGroup {
                            index: 0,
                            group: sheet.clear,
                        },
                        RenderCommand::Draw {
                            vertices: 0..4,
                            instances: first..end,
                        },
                        RenderCommand::SetViewport(viewport),
                    ];
                    first = end;
                    for stack_run in layer_run.chunk_by(|a, b| a.2 == b.2) {
                        let stack = stack_run[0].2;
                        let Some(&(_, bundle)) = drape.iter().find(|d| d.0 == stack) else {
                            continue;
                        };
                        let slots: Vec<(u32, u32)> = stack_run.iter().map(|p| (p.3, p.4)).collect();
                        commands.push(RenderCommand::SetScissor(scissor(&slots, px)));
                        commands.push(RenderCommand::ExecuteBundles(vec![bundle]));
                    }
                    passes.push(Pass::Render(RenderPass {
                        label: "drape",
                        colors: vec![ColorAttachment {
                            target: sheet.draw[layout.layer(layer_run[0].2) as usize],
                            resolve: None,
                            store: StoreOp::Store,
                            load: LoadOp::Load,
                        }],
                        depth: None,
                        commands,
                    }));
                    drawn.draws += 1;
                }
                // Each level from the one above, over the sheet's slots.
                let chain = ("drape mips", &sheet.mips);
                let targets = &mut self.mip_targets;
                let (mips, _) =
                    (downsampler.mip_passes(gpu, chain, &regions, targets)).expect("ready above");
                passes.extend(mips);
            }
            Pass::group(passes, start, "drape");
        }
        if self.bound.is_some_and(|bound| bound != (frame, props)) {
            // The scene's buffers grew.
            self.forget_group(gpu);
        }
        if self.groups.is_empty() {
            self.groups = (self.sheets[0].iter().enumerate())
                .map(|(at, colour)| {
                    let slopes = self.sheets[1].get(at).unwrap_or(colour);
                    gpu.create_bind_group(&BindGroupDesc {
                        label: "terrain",
                        layout: self.layout,
                        entries: &[
                            BindGroupEntry::buffer(0, frame),
                            self.tiles.entry(1),
                            self.grid.entry(2),
                            BindGroupEntry::texture(3, dem.0, D2Array),
                            BindGroupEntry::texture(4, colour.texture, D2Array),
                            BindGroupEntry::sampler(5, self.linear),
                            BindGroupEntry::buffer(6, self.params),
                            BindGroupEntry::buffer(7, self.atmosphere),
                            props,
                            BindGroupEntry::texture(9, slopes.texture, D2Array),
                        ],
                    })
                })
                .collect();
            self.bound = Some((frame, props));
        }
        let occlusion_pipeline = self.occlusion.and(self.occlusion_pipeline);
        let samples = self.target_samples.max(1);
        // A layer that shades hillshades draws its stacks above them
        // whether they show or not: one hidden shades nothing.
        let shaded: Vec<bool> = (0..self.targets.len())
            .map(|layer| self.layers.get(layer).is_some_and(TerrainLayer::shaded))
            .collect();
        let pipelines = [false, true].map(|with| {
            let wanted = with && shaded.contains(&true);
            (!with || wanted)
                .then(|| self.made(samples, with))
                .flatten()
        });
        let step = GRID_STEPS.iter().position(|&s| s == self.grid_step);
        let indices = self.grid_ranges[step.expect("a grid step")].clone();
        // A shaded layer's tables, and its backdrop: the layer's before
        // it, or blank.
        let backdrops: Vec<Option<(BindGroupId, BindGroupId)>> = (0..self.targets.len())
            .map(|layer| {
                if !shaded[layer] {
                    return None;
                }
                let tables = shades.get(layer).copied().unwrap_or_default();
                Some((
                    self.shade_group(gpu, layer, tables),
                    self.backdrop_group(gpu, layer),
                ))
            })
            .collect();
        // Each sheet's tiles with it bound.
        let sheets: Vec<(BindGroupId, std::ops::Range<u32>)> = (self.groups.iter())
            .zip(&self.sheet_tiles)
            .filter(|(_, tiles)| !tiles.is_empty())
            .map(|(&group, tiles)| (group, tiles.clone()))
            .collect();
        let draw = |pipeline, layer_group, shade: Option<(BindGroupId, BindGroupId)>| {
            let mut commands = vec![
                RenderCommand::SetPipeline(pipeline),
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
            commands.push(RenderCommand::SetIndexBuffer {
                buffer: self.grid_indices,
                offset: 0,
            });
            for (group, tiles) in &sheets {
                commands.extend([
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: *group,
                    },
                    RenderCommand::DrawIndexed {
                        indices: indices.clone(),
                        base_vertex: 0,
                        instances: tiles.clone(),
                    },
                ]);
            }
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
            let shade = backdrops[layer];
            let pipeline = pipelines[usize::from(shade.is_some())].expect("ready: in the set");
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

    /// Terrain layer `layer`'s shade group, of the hillshade tables
    /// `tables` (a blank one where a hillshade hides, which shades
    /// nothing).
    fn shade_group(&mut self, gpu: &mut dyn Gpu, layer: usize, tables: ShadeTables) -> BindGroupId {
        if self.shade_groups.len() <= layer {
            self.shade_groups.resize(layer + 1, None);
        }
        if let Some((bound, group)) = self.shade_groups[layer] {
            if bound == tables {
                return group;
            }
            gpu.destroy_bind_group(group);
        }
        let blank = self.blank(gpu).0;
        let [first, second] = tables.map(|t| t.unwrap_or(blank));
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "terrain shades",
            layout: self.shade_layout,
            entries: &[
                BindGroupEntry::sampler(0, self.linear),
                BindGroupEntry::texture(1, first, TextureViewDimension::D2),
                BindGroupEntry::texture(2, second, TextureViewDimension::D2),
            ],
        });
        self.shade_groups[layer] = Some((tables, group));
        group
    }

    /// A blank texture, and the backdrop group binding it.
    fn blank(&mut self, gpu: &mut dyn Gpu) -> (TextureId, BindGroupId) {
        if let Some(blank) = self.blank {
            return blank;
        }
        let texture = gpu.create_texture(&TextureDesc {
            label: "terrain blank",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: self.target.color,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
        });
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "terrain backdrop",
            layout: self.backdrop_layout,
            entries: &[BindGroupEntry::texture(
                0,
                texture,
                TextureViewDimension::D2,
            )],
        });
        self.blank = Some((texture, group));
        (texture, group)
    }

    /// Terrain layer `layer`'s backdrop group: the texture of the layer
    /// before it, where its first hillshade blends into that, else a blank
    /// one.
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
            return self.blank(gpu).1;
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

    /// The drape frame: the map's, at the zoom its slots draw at
    /// ([`redraw`](Self::redraw)), drawn into page `page`.
    pub(crate) fn drape_frame(
        &self,
        frame: &crate::scene::FrameBlock,
        page: usize,
    ) -> crate::scene::FrameBlock {
        // The grid the sheets lie in, which each sheet's passes move their
        // viewport over.
        let [w, h] = [(self.layout_state.side * PAGE_PX[page]) as f32; 2];
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
            // pixel at the slots' zoom.
            world_per_px: 1.0,
            // What shaders blend by the zoom (per-feature values, feature
            // state, patterns' scale) blends at the slots' zoom, as their
            // paint does, not at the zoom of the frame that draws them.
            zoom: self.zoom,
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
        self.release_sheets(gpu);
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
        self.paint.is_some() && !self.sheets[0].is_empty()
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

    /// Makes the pipeline of the terrain drawn as its depth, at one
    /// sample, unless it is made.
    fn occlusion_pipeline(&mut self, gpu: &mut dyn Gpu) {
        if self.occlusion_pipeline.is_some() {
            return;
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
        self.release_sheets(gpu);
        self.release_occlusion(gpu);
        self.forget_group(gpu);
        let layouts = (self.layout, self.stack_layout, self.clear_layout);
        let backdrops: Vec<BindGroupId> = (self.backdrops.iter().flatten()).map(|b| b.1).collect();
        (self.backdrop_layout, backdrops, self.blank).release(gpu);
        let shades: Vec<BindGroupId> = (self.shade_groups.iter().flatten()).map(|s| s.1).collect();
        (self.shade_layout, shades).release(gpu);
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
    fn a_stack_replays_once_over_its_redrawn_slots() {
        // Slots 5 and 6 lie at columns 1 and 0 of rows 0 and 1 of sheet 1.
        assert_eq!(scissor(&[(5, 1024), (6, 512)], 1024), [0, 0, 2048, 2048]);
        // A slot alone takes its side.
        assert_eq!(scissor(&[(7, 256)], 1024), [1024, 1024, 256, 256]);
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
}
