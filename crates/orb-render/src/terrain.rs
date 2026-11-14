//! Terrain (render graph `Terrain`/`Drape`): the map's 2D
//! layers draped over a DEM-raised surface, after GL JS's render to
//! texture.
//!
//! Each terrain tile has a drape slot per drape stack (a run of draped
//! layers between layers that draw in 3D or on screen, or a hillshade the
//! terrain shades, whose stack holds its unlit slopes), of as many texels
//! as the tile shows at (the runtime's `slot_shrink`), up to
//! [`DRAPE_SLOT_PX`] a side. A tile's slots are a texture of their side,
//! mipmapped, a layer a stack: colour stacks in one, slope stacks at half
//! the texels a side in another (two tiles' a texture where GL would make
//! one's layers with a layer more, [`slots_per_texture`]). The drape holds
//! the texels its tiles show, no more: a slot of another side takes
//! another texture, and a texture no slot holds goes once the map
//! settles. A frame redraws only the slots the runtime marks: their drape
//! placements, the draped nodes' only candidates, map each into its place
//! in a grid of drape space, a function of its tile slot alone, and a pass
//! a slot's stack draws the stack's draped nodes' runs of that slot alone
//! (the renderer's `Recorded`) with the viewport moving its place onto its
//! texture's layer, which draws it alike in any texture; then a compute
//! pass a texture downsamples its mip levels.
//!
//! The terrain draws in the map's own pass, its tiles in the order the
//! runtime gives them, each with its slots' textures bound, over a depth
//! attachment that pass never stores: first its depth
//! alone, then each terrain layer (a stack, or up to [`MAX_SHADES`]
//! hillshades lit between stacks from their
//! slopes' stacks and the frame's bearing) at the place of its first
//! node, over what the map drew beneath it, every fragment but the
//! nearest failing the depth test, and of those meeting at it (a skirt
//! and a neighbour's edge) all but the first failing the stencil's, so
//! no sample blends twice. With antialiasing on, the map's passes take
//! four samples a pixel, and so do the terrain's silhouettes, as GL JS's
//! antialiased canvas. A layer that blends with the map
//! beneath it in a mode the blend unit has no factors for reads it from
//! the canvas the pass before drew.

use crate::TargetFormat;
use crate::pipeline::{Attachments, Draw, depth_test};
use crate::release::Release;
use crate::scene::{Backing, DataTable};
use TextureViewDimension::D2Array;
use orb_hal::*;
use orb_shaders::{Shader, TERRAIN_SHADE_SHADER, TERRAIN_SHADER};
use rustc_hash::FxHashMap;

/// Drape texels per terrain tile side at most: twice the 512 logical
/// pixels a terrain tile covers at its zoom, GL JS's quality factor.
pub const DRAPE_SLOT_PX: u32 = 1024;

/// Slope slots' texels per terrain tile side at most: half the drape's. A
/// shaded hillshade's slopes come from DEM tiles of at most 512 pixels a
/// side at the terrain tile's zoom, which a 512-texel slot holds sample
/// for sample, and a quarter of the texels lets a viewport-lit hillshade
/// cost a terrain tile a quarter slot rather than a whole one (#142).
pub(crate) const SLOPE_SLOT_PX: u32 = DRAPE_SLOT_PX / 2;

/// The drape's pages by slot side: colour stacks, then slope stacks.
const PAGE_PX: [u32; 2] = [DRAPE_SLOT_PX, SLOPE_SLOT_PX];

/// The most a slot's side halves: to 64 texels a colour slot.
pub const MAX_SHRINK: u8 = 4;

/// Places a side of the drape grid on this device, whose sides stop at
/// WebGPU's default, 8192: the drape space slots draw in, a place each,
/// [`DRAPE_SLOT_PX`] texels a side.
fn grid_side(gpu: &dyn Gpu) -> u32 {
    (gpu.caps().limits.max_texture_dimension_2d.min(8192) / DRAPE_SLOT_PX).max(1)
}

/// Where tile slot `slot` lies in a grid `side` places a side: its column
/// and row.
fn slot_place(slot: u32, side: u32) -> [u32; 2] {
    [slot % side, slot / side]
}

/// The terrain tiles the drape can hold with `stacks` colour and slope
/// stacks: the places of a grid of WebGPU's default texture side (64) on
/// every device that has it, so that a view's terrain doesn't depend on
/// which, or none where a tile's whole slots take more layers or bytes
/// than the device allocates a texture ([`Gpu::texture_fits`], GL's
/// [`Limits::max_texture_bytes`]).
pub(crate) fn room(gpu: &dyn Gpu, stacks: [u32; 2]) -> u32 {
    let alloc = allocated(stacks);
    let shared = slots_per_texture(gpu, stacks);
    let fits = (0..2).filter(|&p| alloc[p] > 0).all(|page| {
        let desc = slots_desc(alloc[page] * shared, PAGE_PX[page], gpu.caps().compute);
        desc.array_layers <= gpu.caps().limits.max_texture_array_layers && gpu.texture_fits(&desc)
    });
    if fits { grid_side(gpu).pow(2) } else { 0 }
}

/// Why the drape pages can't hold a view's terrain tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrapeFull {
    /// The style's drape stacks leave no room for them ([`room`]).
    Room,
    /// The device refused a texture of this many bytes
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
/// The vertex stage's constants where the terrain draws its depth alone:
/// its fragments sample no drape, so its vertices work out no footprint
/// (`terrain.wgsl`'s `terrain_footprint`).
const DEPTH_ALONE: &[(&str, f64)] = &[("terrain_footprint", 0.0)];

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

/// A tile slot whose drape a frame redraws: its index, how many times its
/// side halves its page's, and the stacks redrawn.
pub type SlotRedraw<'a> = (u32, u8, &'a [u8]);

/// A terrain tile to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainTile {
    /// Its north-west corner in camera-relative world space.
    pub origin: [f32; 2],
    /// World units per terrain unit: it spans [`TERRAIN_EXTENT`].
    pub scale: f32,
    /// Its DEM: page layer, and tile uv → DEM uv as `uv · scale + offset`.
    pub dem: Option<TerrainDem>,
    /// Its tile slot: its place in the drape grid (see [`DrapeLayout`]),
    /// whose texels it shows.
    pub slot: u32,
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
/// every pixel the layer covers, in the one draw of the terrain, where a
/// layer of its own each cost another draw of it, and a third reads the
/// map beneath it from the canvas the pass before drew.
pub(crate) const MAX_SHADES: usize = 2;

/// What one draw of the terrain lays over the map: a stack's slots, or a
/// stack below the hillshades it shades and those above each of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TerrainLayer {
    pub(crate) below: Option<u8>,
    /// The hillshades it shades, in order from the first: each over what
    /// is beneath it, its colour stack above it on top.
    pub(crate) shades: [Option<Shade>; MAX_SHADES],
    /// A layer's own, which blends with the map beneath it in this mode
    /// (normal: over it).
    pub(crate) blend: crate::Blend,
    /// What its first hillshade blends into is the map beneath it, rather
    /// than a stack of its own: the hillshade lies on the terrain layer
    /// before, whose hillshades fill it.
    pub(crate) backdrop: bool,
}

impl TerrainLayer {
    /// Whether it reads the map beneath it: it blends with it in a mode
    /// other than normal, or its first hillshade blends into it.
    pub(crate) fn reads_beneath(&self) -> bool {
        self.backdrop || self.blend != crate::Blend::Normal
    }

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
const LAYER_WORDS: usize = 8 + 4 * MAX_SHADES;
/// The word of a terrain layer's uniform that holds its opacity.
const LAYER_OPACITY: u64 = 3;

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

/// Where drape slots are drawn. Stacks `0..stacks[0]` are colour stacks,
/// on page 0 at up to [`DRAPE_SLOT_PX`] texels a side; the rest hold
/// slopes, on page 1 at up to [`SLOPE_SLOT_PX`]. A tile slot is a place
/// in a grid `side` places a side, a function of its index alone, whose
/// top-left square of its side each of its slots draws in: each layer's
/// drape space is the grid's texels with y up, texel `(x, y)` drape world
/// `(x, −y)`, and a slot draws alike in whichever texture its texels lie,
/// which a pass's viewport maps its place onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrapeLayout {
    /// Colour stacks, then slope stacks.
    pub stacks: [u32; 2],
    /// Places a side of the grid.
    pub side: u32,
    /// Tile slots the grid holds: its places.
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

    /// The page of stack `stack`, and its rank among the page's stacks.
    pub fn page(&self, stack: u32) -> (usize, u32) {
        match stack.checked_sub(self.stacks[0]) {
            Some(rank) => (1, rank),
            None => (0, stack),
        }
    }

    /// Texels a side of stack `stack`'s slots at most.
    pub fn slot_px(&self, stack: u32) -> u32 {
        PAGE_PX[self.page(stack).0]
    }

    /// Stack `stack`'s rank among its page's stacks.
    pub fn rank(&self, stack: u32) -> u32 {
        self.page(stack).1
    }

    /// Tile slot `tile_slot`'s slot of `stack`: its top-left texel in the
    /// grid's drape space.
    pub fn corner(&self, tile_slot: u32, stack: u32) -> [u32; 2] {
        let px = self.slot_px(stack);
        slot_place(tile_slot, self.side).map(|n| n * px)
    }
}

/// A texture of `layers` slot layers `side` texels a side, mipmapped down
/// to a texel, its mip levels written as storage where the device
/// `computes`, or drawn apart and copied in.
fn slots_desc(layers: u32, side: u32, computes: bool) -> TextureDesc<'static> {
    let levels = match computes {
        true => TextureUsages::STORAGE_BINDING,
        false => TextureUsages::COPY_DST,
    };
    TextureDesc {
        label: "drape slots",
        width: side,
        height: side,
        array_layers: layers,
        dimension: TextureViewDimension::D2Array,
        format: DRAPE_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | levels,
        sample_count: 1,
        mip_levels: side.ilog2() + 1,
    }
}

/// The tile slots a texture of each page holds: one, or the fewest more
/// where the device would make one's layers with a layer more than they
/// take (wgpu's GL back end, for one layer or a square of 6n:
/// [`Gpu::allocated_layers`]).
fn slots_per_texture(gpu: &dyn Gpu, stacks: [u32; 2]) -> u32 {
    let exact = |shared: &u32| {
        (0..2).filter(|&p| stacks[p] > 0).all(|page| {
            let desc = slots_desc(stacks[page] * shared, PAGE_PX[page], gpu.caps().compute);
            gpu.allocated_layers(&desc) == desc.array_layers
        })
    };
    (1..=3).find(exact).unwrap_or(1)
}

/// A page's texture of tile slots of one side: a layer a slot's stack,
/// mipmapped, whose texels draw and sample alike whichever texture and
/// layer they lie in.
#[derive(Debug)]
struct SlotTexture {
    texture: TextureId,
    layers: u32,
    side: u32,
    /// What draws into each layer's level 0.
    draw: Vec<TextureId>,
    /// Per mip level, a view, which the mip levels are written through.
    levels: Vec<TextureId>,
    /// The downsampler's chain of its levels, over the redrawn layers as
    /// its regions.
    mips: crate::downsample::Chain,
}

impl SlotTexture {
    /// A texture of `layers` layers `side` texels a side, or the bytes of
    /// one the device refused.
    fn new(
        gpu: &mut dyn Gpu,
        (layers, side): (u32, u32),
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<Self, u64> {
        let desc = slots_desc(layers, side, gpu.caps().compute);
        let texture = gpu
            .try_create_texture(&desc)
            .ok_or_else(|| level_bytes(&desc, layers) * 4 / 3)?;
        let levels: Vec<TextureId> = (0..desc.mip_levels)
            .map(|l| gpu.create_texture_view(texture, l..l + 1))
            .collect();
        let draw = (0..layers)
            .map(|layer| gpu.create_layer_view(texture, layer, 0))
            .collect();
        downsampler.make(gpu);
        let mips = downsampler.chain(gpu, (texture, [side; 2], &levels), ("drape slots", layers));
        Ok(Self {
            texture,
            layers,
            side,
            draw,
            levels,
            mips,
        })
    }

    /// Its bytes, mips included.
    fn bytes(&self) -> u64 {
        u64::from(self.side).pow(2) * u64::from(self.layers) * 4 * 4 / 3
    }
}

impl crate::release::Release for SlotTexture {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.mips, self.levels, self.draw, self.texture).release(gpu);
    }
}

/// Textures of tile slots of one side: each page's, and the group the
/// terrain's draws of their tiles bind, with the tile slot each of its
/// slots holds.
#[derive(Debug)]
struct Texels {
    /// The slots' side, in halvings of their page's.
    level: u8,
    /// The run of tile slots its slots hold, [`slots_per_texture`] of
    /// them: tile slot `t` is slot `t % n` of a texture of run `t / n`, a
    /// function of the tile slot alone, so that a map at rest holds the
    /// textures a fresh one there does.
    run: u32,
    pages: [Option<SlotTexture>; 2],
    group: Option<BindGroupId>,
    holds: Vec<Option<u32>>,
}

impl Texels {
    /// Its bytes, mips included.
    fn bytes(&self) -> u64 {
        self.pages.iter().flatten().map(SlotTexture::bytes).sum()
    }
}

#[derive(Debug)]
pub(crate) struct Terrain {
    /// The map's passes, which the terrain layers draw in.
    target: Attachments,
    layout: BindGroupLayoutId,
    stack_layout: BindGroupLayoutId,
    /// A shaded layer's hillshades' tables, which colour their slopes.
    shade_layout: BindGroupLayoutId,
    /// Each terrain layer's shade group and the tables it binds.
    shade_groups: Vec<Option<(ShadeTables, BindGroupId)>>,
    /// The map beneath a layer that reads it: the canvas the pass before
    /// drew.
    backdrop_layout: BindGroupLayoutId,
    /// Each terrain layer's backdrop group and the texture it binds.
    backdrops: Vec<Option<(TextureId, BindGroupId)>>,
    /// A blank texture, and a backdrop group binding it.
    blank: Option<(TextureId, BindGroupId)>,
    /// Terrain layers over the map, lighting hillshades between their
    /// stacks or not, and those that read the map beneath ([`Kind`]).
    pipelines: [Option<RenderPipelineId>; 3],
    /// The terrain's depth alone, before anything tests it: in the map's
    /// passes ([`MAP`]), and in the 3D layers' ([`THREE_D`]) where theirs
    /// differ (multisampled), each with the attachments it was made for;
    /// and the 3D layers' attachments, once asked for.
    depth_pipelines: [Option<(Attachments, RenderPipelineId)>; 2],
    three_d: Option<Attachments>,
    grid: DataTable,
    grid_indices: BufferId,
    /// The index range of each grid step's mesh, and the step drawn.
    grid_ranges: [std::ops::Range<u32>; GRID_STEPS.len()],
    grid_step: u32,
    tiles: DataTable,
    tile_capacity: u32,
    tile_count: u32,
    params: BufferId,
    /// The terrain layers and their opacities, and each's group over its
    /// slice of `stack_uniforms`, made for as many as the style has.
    layers: Vec<TerrainLayer>,
    opacities: Vec<f32>,
    stack_groups: Vec<BindGroupId>,
    stack_uniforms: Option<BufferId>,
    /// Trilinear and anisotropic, as GL JS samples its render-to-texture
    /// tiles: the raster pages' sampler, which the renderer owns.
    linear: SamplerId,
    /// The slots' textures, by index: the colour page's, and the slope
    /// page's while a hillshade is shaded, a texture of each a tile slot
    /// (or two, [`slots_per_texture`]) of its side; those no slot holds
    /// stay for the next slots of their side until the map settles
    /// ([`trim`](Self::trim)).
    texels: Vec<Option<Texels>>,
    /// Each tile slot's texture, and its slot of it.
    held: std::collections::BTreeMap<u32, (u32, u32)>,
    /// Tile slots a texture holds.
    shared: u32,
    /// Whether the pages were made for the layout's stacks.
    reserved: bool,
    layout_state: DrapeLayout,
    /// The DEM page texture and revision the groups bind.
    dem: Option<(TextureId, u32)>,
    /// The frame and property buffers the textures' groups bind.
    bound: Option<(BufferId, BindGroupEntry)>,
    /// The texture of each of this frame's tiles, in the table's order.
    tile_textures: Vec<u32>,
    /// What this frame's draws of the terrain draw: each tile's group, and
    /// the grid's indices; none while the terrain can't draw.
    drawn: Option<Drawn>,
    /// The terrain's depth as a colour, and the depth it's drawn with: what
    /// hides symbols and circles.
    occlusion: Option<[TextureId; 2]>,
    occlusion_pipeline: Option<RenderPipelineId>,
    /// The size of `occlusion`, in logical pixels.
    occlusion_size: [u32; 2],
    /// Slots to redraw this frame.
    pending: Vec<Redraw>,
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

/// A slot to redraw: a stack of a tile slot, and its texture and layer.
/// They sort by page, then texture: each texture's stacks, whose mip
/// levels are made at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Redraw {
    page: usize,
    texture: u32,
    layer: u32,
    stack: u32,
    tile_slot: u32,
}

/// The terrain's depth in the map's passes.
pub(crate) const MAP: usize = 0;
/// The terrain's depth in the 3D layers' passes.
pub(crate) const THREE_D: usize = 1;

/// The pipelines of terrain layers: over the map, lighting hillshades or
/// not, or reading the map beneath them ([`TerrainLayer::reads_beneath`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Plain,
    Shaded,
    Beneath,
}

impl TerrainLayer {
    fn kind(&self) -> Kind {
        if self.reads_beneath() {
            Kind::Beneath
        } else if self.shaded() {
            Kind::Shaded
        } else {
            Kind::Plain
        }
    }
}

/// What a frame's draws of the terrain draw: each tile's group, by its
/// instance, and the grid's indices at the frame's step.
#[derive(Debug)]
struct Drawn {
    tiles: Vec<Option<BindGroupId>>,
    indices: std::ops::Range<u32>,
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
        target: Attachments,
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
            pipelines: [None; 3],
            depth_pipelines: [None; 2],
            three_d: None,
            grid: grid_buffer,
            grid_indices,
            grid_ranges,
            grid_step: 1,
            tiles: DataTable::new(gpu, "terrain tiles", 64 * terrain_tile::WORDS as u64),
            tile_capacity: 64,
            tile_count: 0,
            params,
            layers: Vec::new(),
            opacities: Vec::new(),
            stack_groups: Vec::new(),
            stack_uniforms: None,
            linear,
            texels: Vec::new(),
            held: Default::default(),
            shared: 1,
            reserved: false,
            layout_state: DrapeLayout {
                stacks: [1, 0],
                side: grid_side(gpu),
                tile_slots: 0,
                generation: 0,
            },
            dem: None,
            bound: None,
            tile_textures: Vec::new(),
            drawn: None,
            occlusion: None,
            occlusion_pipeline: None,
            occlusion_size: [0; 2],
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
        self.texels.iter().flatten().map(Texels::bytes).sum()
    }

    pub(crate) fn layout(&self) -> DrapeLayout {
        self.layout_state
    }

    /// This frame's terrain layers, whose slope stacks follow `colour`
    /// colour stacks, at their `opacities`: each gets a group binding what
    /// it draws.
    pub(crate) fn set_layers(
        &mut self,
        gpu: &mut dyn Gpu,
        (layers, colour): (&[TerrainLayer], u32),
        opacities: &[f32],
    ) {
        if self.layers == layers {
            self.set_opacities(gpu, opacities);
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
                () if layer.blend != crate::Blend::Normal => FOG_FADES,
                () if i == 0 => FOG_GROUND,
                () => FOG_CONTENT,
            };
            let opacity = opacities.get(i).copied().unwrap_or(1.0);
            let mut words = vec![
                stack(layer.below),
                fog,
                u32::from(layer.backdrop),
                opacity.to_bits(),
                layer.blend.mode() as u32,
                0,
                0,
                0,
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
        self.opacities = opacities.to_vec();
    }

    /// Writes the terrain layers' opacities where they changed.
    fn set_opacities(&mut self, gpu: &mut dyn Gpu, opacities: &[f32]) {
        let Some(buffer) = self.stack_uniforms else {
            return;
        };
        for (i, &opacity) in opacities.iter().enumerate().take(self.layers.len()) {
            if self.opacities.get(i) != Some(&opacity) {
                let at = UNIFORM_STRIDE * i as u64 + LAYER_OPACITY * 4;
                gpu.write_buffer(buffer, at, bytemuck::bytes_of(&opacity));
            }
        }
        self.opacities = opacities.to_vec();
    }

    /// Makes the pages for `stacks` colour and slope stacks: unless they
    /// were made for them, new and blank, of a new
    /// [`DrapeLayout::generation`], their textures made as slots need them
    /// ([`redraw`](Self::redraw)): whether it made them. Fails where the
    /// stacks leave the device no [`room`].
    pub(crate) fn reserve(
        &mut self,
        gpu: &mut dyn Gpu,
        stacks: [u32; 2],
    ) -> Result<bool, DrapeFull> {
        let room = room(gpu, stacks);
        if room == 0 {
            return Err(DrapeFull::Room);
        }
        let state = self.layout_state;
        let made = state.stacks != stacks || !self.reserved;
        if made {
            let before = gpu.counters();
            self.release_texels(gpu);
            self.resized.count(gpu, &before);
            self.pending.clear();
            self.shared = slots_per_texture(gpu, allocated(stacks));
            let counts = allocated(stacks);
            let at = params::COLOUR_STACKS as u64 * 4;
            gpu.write_buffer(self.params, at, bytemuck::cast_slice(&counts));
        }
        self.reserved = true;
        self.layout_state = DrapeLayout {
            stacks,
            side: grid_side(gpu),
            tile_slots: room,
            generation: state.generation + u64::from(made),
        };
        Ok(made)
    }

    /// Releases every slot's texture: every slot is blank.
    fn release_texels(&mut self, gpu: &mut dyn Gpu) {
        self.forget_group(gpu);
        for texels in std::mem::take(&mut self.texels).into_iter().flatten() {
            texels.pages.release(gpu);
        }
        self.held.clear();
    }

    /// Lets go of the textures no slot holds, in a frame that settles the
    /// map: a view whose slots change sides keeps them for the next slots
    /// of their sides until then, rather than making and freeing textures
    /// frame after frame. A map at rest then holds its slots' texels and
    /// no more, as a fresh one there does.
    pub(crate) fn trim(&mut self, gpu: &mut dyn Gpu) {
        let before = gpu.counters();
        let mut changed = false;
        for at in &mut self.texels {
            if at
                .as_ref()
                .is_some_and(|t| t.holds.iter().all(Option::is_none))
            {
                let texels = at.take().expect("checked");
                (texels.pages, texels.group).release(gpu);
                changed = true;
            }
        }
        while self.texels.last().is_some_and(Option::is_none) {
            self.texels.pop();
        }
        if changed {
            self.resized.count(gpu, &before);
        }
    }

    /// Forgets the textures' groups, for the next frame to make again.
    fn forget_group(&mut self, gpu: &mut dyn Gpu) {
        for texels in self.texels.iter_mut().flatten() {
            texels.group.take().release(gpu);
        }
        self.bound = None;
    }

    /// Redraws tile slots' stacks this frame at style zoom `zoom`: per
    /// tile slot, how many times its side halves its page's and the
    /// stacks to redraw. The slots none of the frame's `tiles` shows give
    /// their texels back first, for these to take. A slot of another side
    /// than its texels' takes a texture's slot of its side
    /// ([`take`](Self::take)), all its stacks redrawn, and leaves the old
    /// one. Fails where the device refuses a texture, which leaves the
    /// slots blank.
    pub(crate) fn redraw(
        &mut self,
        gpu: &mut dyn Gpu,
        zoom: f64,
        (tiles, tile_slots): (&[TerrainTile], &[SlotRedraw<'_>]),
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<(), DrapeFull> {
        self.zoom = zoom;
        self.pending.clear();
        let shown: std::collections::BTreeSet<u32> = tiles.iter().map(|t| t.slot).collect();
        let gone: Vec<u32> = (self.held.keys())
            .filter(|s| !shown.contains(s))
            .copied()
            .collect();
        for slot in gone {
            let (texture, at) = self.held.remove(&slot).expect("held");
            self.texture_mut(texture).holds[at as usize] = None;
        }
        let layout = self.layout_state;
        let counts = allocated(layout.stacks);
        for &(slot, shrink, redrawn) in tile_slots {
            let level = shrink.min(MAX_SHRINK);
            let (texture, at) = match self.held.get(&slot) {
                Some(&(texture, at)) if self.level(texture) == level => (texture, at),
                held => {
                    if let Some(&(texture, at)) = held {
                        self.texture_mut(texture).holds[at as usize] = None;
                    }
                    let (texture, at) = self.take(gpu, (slot, level), downsampler)?;
                    self.texture_mut(texture).holds[at as usize] = Some(slot);
                    self.held.insert(slot, (texture, at));
                    (texture, at)
                }
            };
            let stacks =
                (redrawn.iter().map(|&s| u32::from(s))).filter(|&s| s < layout.stack_count());
            self.pending.extend(stacks.map(|stack| {
                let (page, rank) = layout.page(stack);
                Redraw {
                    page,
                    texture,
                    layer: at * counts[page] + rank,
                    stack,
                    tile_slot: slot,
                }
            }));
        }
        Ok(())
    }

    /// The slots' side of texture `texture`, in halvings of their page's.
    fn level(&self, texture: u32) -> u8 {
        self.texels[texture as usize]
            .as_ref()
            .expect("a texture")
            .level
    }

    fn texture_mut(&mut self, texture: u32) -> &mut Texels {
        self.texels[texture as usize].as_mut().expect("a texture")
    }

    /// Tile slot `slot`'s texture of `level`, and its slot of it: that of
    /// its run ([`Texels::run`]), else one of `level` that holds no slot,
    /// which a view that moves keeps until it settles, else a new one.
    fn take(
        &mut self,
        gpu: &mut dyn Gpu,
        (slot, level): (u32, u8),
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<(u32, u32), DrapeFull> {
        let (run, at) = (slot / self.shared, slot % self.shared);
        let find = |fits: &dyn Fn(&Texels) -> bool| {
            (self.texels.iter())
                .position(|t| t.as_ref().is_some_and(|t| t.level == level && fits(t)))
        };
        let found =
            find(&|t| t.run == run).or_else(|| find(&|t| t.holds.iter().all(Option::is_none)));
        if let Some(texture) = found {
            let texels = self.texels[texture].as_mut().expect("found");
            debug_assert!(
                texels.holds[at as usize].is_none(),
                "a slot holds one tile slot"
            );
            texels.run = run;
            return Ok((texture as u32, at));
        }
        let before = gpu.counters();
        let counts = allocated(self.layout_state.stacks);
        let mut pages = [None, None];
        for page in 0..2 {
            if counts[page] == 0 {
                continue;
            }
            let size = (counts[page] * self.shared, PAGE_PX[page] >> level);
            match SlotTexture::new(gpu, size, downsampler) {
                Ok(texture) => pages[page] = Some(texture),
                Err(bytes) => {
                    pages.release(gpu);
                    self.release_texels(gpu);
                    return Err(DrapeFull::Refused(bytes));
                }
            }
        }
        let texels = Texels {
            level,
            run,
            pages,
            group: None,
            holds: vec![None; self.shared as usize],
        };
        let index = match self.texels.iter().position(Option::is_none) {
            Some(index) => index,
            None => {
                self.texels.push(None);
                self.texels.len() - 1
            }
        };
        self.texels[index] = Some(texels);
        self.resized.count(gpu, &before);
        Ok((index as u32, at))
    }

    /// The tile slots whose texels the textures hold.
    pub(crate) fn held_slots(&self) -> u32 {
        self.held.len() as u32
    }

    /// The slots this frame redraws, `(stack, tile slot)`, in order.
    pub(crate) fn pending_slots(&self) -> Vec<(u32, u32)> {
        let mut slots: Vec<(u32, u32)> = (self.pending.iter())
            .map(|p| (p.stack, p.tile_slot))
            .collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// This frame's terrain tiles, each drawn from the texels its slot
    /// took in the frame's [`redraw`](Self::redraw) or before. They draw
    /// in the order given, which decides what shows where two meet.
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
        // A tile shows a slot it drew, or its ancestor did.
        let order: Vec<(&TerrainTile, (u32, u32))> = (tiles.iter())
            .filter_map(|t| Some((t, *self.held.get(&t.slot)?)))
            .collect();
        debug_assert_eq!(order.len(), tiles.len(), "every tile shows a slot drawn");
        self.tile_textures = order.iter().map(|&(_, (texture, _))| texture).collect();
        let words: Vec<u32> = (order.iter())
            .flat_map(|&(t, (_, at))| {
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
                    .u(n::SLOT, &[at]);
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
        self.tile_count = order.len() as u32;
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
        let counts = allocated(self.layout_state.stacks);
        b.u(params::TILES, &[self.tile_count])
            .u(params::COLOUR_STACKS, &[counts[0]])
            .u(params::SLOPE_STACKS, &[counts[1]]);
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

    /// Sizes the terrain's depth for what it hides to `logical` pixels, as
    /// GL JS's: what it hides is tested a logical pixel at a time, and a
    /// quarter of the texels at a pixel ratio of 2 take a fraction of the
    /// pass. Returns whether it changed.
    pub(crate) fn resize(&mut self, gpu: &mut dyn Gpu, logical: [u32; 2]) -> bool {
        if self.occlusion.is_some() && self.occlusion_size == logical {
            return false;
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
        self.occlusion_size = logical;
        true
    }

    /// Draws into the map's passes as `target` from now: the pipelines
    /// made for them before are let go, for the next request to make.
    pub(crate) fn retarget(&mut self, gpu: &mut dyn Gpu, target: Attachments) {
        if self.target == target {
            return;
        }
        self.target = target;
        std::mem::take(&mut self.pipelines).release(gpu);
        (self.depth_pipelines.iter_mut())
            .map(|d| d.take().map(|(_, p)| p))
            .collect::<Vec<_>>()
            .release(gpu);
    }

    /// Asks for the pipelines the terrain draws with: its layers' in the
    /// map's passes, a shaded hillshade's where one is (`shaded`), one
    /// reading the map beneath where a layer does (`beneath`), and its
    /// depth alone there and in the 3D layers' passes, of `three_d`, where
    /// the style has them: the pipeline set's
    /// ([`Renderer::set_pipelines`](crate::Renderer::set_pipelines)).
    pub(crate) fn request_pipelines(
        &mut self,
        gpu: &mut dyn Gpu,
        three_d: Option<TargetFormat>,
        (shaded, beneath): (bool, bool),
    ) {
        self.pipeline(gpu, Kind::Plain);
        if shaded {
            self.pipeline(gpu, Kind::Shaded);
        }
        if beneath {
            self.pipeline(gpu, Kind::Beneath);
        }
        self.depth_pipeline(gpu, MAP, self.target);
        if let Some(three_d) = three_d {
            let three_d = Attachments {
                depth: self.target.depth,
                ..three_d.into()
            };
            self.depth_pipeline(gpu, THREE_D, three_d);
        }
        self.occlusion_pipeline(gpu);
    }

    /// Whether every terrain pipeline this frame's layers draw with
    /// compiled, and its depth's in the map's passes. Until then terrain
    /// draws nothing.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        let kinds = (self.layers.iter()).map(|l| self.pipelines[l.kind() as usize]);
        let wanted = self.occlusion.and(self.occlusion_pipeline);
        let mut all = kinds.chain(wanted.into_iter().map(Some));
        all.all(|p| p.is_some_and(|p| gpu.pipeline_ready(p.into()))) && self.depth_ready(gpu, MAP)
    }

    /// Whether the pipeline of the terrain's depth alone in the map's
    /// passes ([`MAP`]) or the 3D layers' ([`THREE_D`]) compiled.
    pub(crate) fn depth_ready(&self, gpu: &mut dyn Gpu, at: usize) -> bool {
        self.depth_pipeline_at(at)
            .is_some_and(|p| gpu.pipeline_ready(p.into()))
    }

    /// The pipeline of the terrain's depth alone in the map's passes
    /// ([`MAP`]) or the 3D layers' ([`THREE_D`]), once made: the map's
    /// serves both where their attachments are alike.
    fn depth_pipeline_at(&self, at: usize) -> Option<RenderPipelineId> {
        let wanted = if at == MAP {
            Some(self.target)
        } else {
            self.three_d
        };
        (self.depth_pipelines.iter().flatten())
            .find(|(made, _)| Some(*made) == wanted)
            .map(|&(_, p)| p)
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        let depth = self.depth_pipelines.iter().flatten().map(|&(_, p)| p);
        let made = (self.pipelines.iter().flatten().copied()).chain(depth);
        made.chain(self.occlusion_pipeline)
            .map(PipelineId::from)
            .collect()
    }

    /// Makes every drape slot blank from the next frame's plan, as a new
    /// atlas does: the map draws them all again (draped layers whose
    /// pipelines arrived since). The slots this frame redraws still draw:
    /// it comes between the plan and the passes when pipelines arrive, and
    /// a slot the plan moved to a texture of another side holds nothing
    /// there until drawn, which showed as a hole in the ground for a frame.
    pub(crate) fn redraw_all(&mut self) {
        self.layout_state.generation += 1;
    }

    /// Makes the pipeline terrain layers of `kind` draw with, unless it is
    /// made: over the map, by premultiplied blending, or in place of it
    /// where they read it.
    fn pipeline(&mut self, gpu: &mut dyn Gpu, kind: Kind) {
        if self.pipelines[kind as usize].is_some() {
            return;
        }
        let shaded = [self.layout, self.stack_layout, self.shade_layout];
        let beneath = [
            self.layout,
            self.stack_layout,
            self.shade_layout,
            self.backdrop_layout,
        ];
        let (shader, entry, layouts): (&Shader, _, &[_]) = match kind {
            Kind::Plain => (&TERRAIN_SHADER, "fs_terrain", &shaded[..2]),
            Kind::Shaded => (&TERRAIN_SHADE_SHADER, "fs_terrain_shaded", &shaded),
            Kind::Beneath => (&TERRAIN_SHADE_SHADER, "fs_terrain_beneath", &beneath),
        };
        let blend = (kind != Kind::Beneath).then_some(BlendState::PremultipliedAlpha);
        // Over the depth alone, a sample of the nearest surface passes,
        // and of the pieces of mesh meeting there (a skirt and a
        // neighbour's edge) the first drawn alone: none blends twice.
        let format = self.target.depth.expect("the map's passes have a depth");
        let pipeline = Draw {
            fragment: (shader, entry),
            depth: Some(DepthState {
                format,
                write: false,
                test: true,
                once: true,
            }),
            ..Draw::new(shader, layouts, self.target, blend)
        }
        .make(gpu);
        self.pipelines[kind as usize] = Some(pipeline);
    }

    /// Makes the pipeline of the terrain's depth alone into `target`'s
    /// passes, the map's ([`MAP`]) or the 3D layers' ([`THREE_D`]), unless
    /// it is made for them, or for the map's passes where the 3D layers'
    /// are alike: its fragments add nothing to the colour.
    fn depth_pipeline(&mut self, gpu: &mut dyn Gpu, at: usize, target: Attachments) {
        if at == THREE_D {
            self.three_d = Some(target);
        }
        if self.depth_pipelines[at].is_some_and(|(made, _)| made == target) {
            return;
        }
        self.depth_pipelines[at].take().map(|(_, p)| p).release(gpu);
        if at == THREE_D && target == self.target {
            return;
        }
        let pipeline = Draw {
            label: "terrain depth alone",
            fragment: (&TERRAIN_SHADER, "fs_terrain_hidden"),
            constants: [DEPTH_ALONE, &[]],
            depth: depth_test(target.depth.expect("the passes have a depth")),
            ..Draw::new(
                &TERRAIN_SHADER,
                &[self.layout],
                target,
                Some(BlendState::PremultipliedAlpha),
            )
        }
        .make(gpu);
        self.depth_pipelines[at] = Some((target, pipeline));
    }

    /// This frame's passes before the map: redrawn drape slots (`drape`
    /// draws the draped nodes into level 0) and their mip levels, and the
    /// terrain's depth for what it hides, into `passes`; and what the
    /// frame's draws of the terrain draw, where every pipeline they take
    /// compiled ([`layer_commands`](Self::layer_commands)).
    /// Returns its draws and the slots it redrew.
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (frame, props): (BufferId, BindGroupEntry),
        drape: &FxHashMap<(u32, u32), Vec<RenderCommand>>,
        downsampler: &crate::downsample::Downsampler,
        passes: &mut Vec<Pass>,
    ) -> TerrainDrawn {
        self.drawn = None;
        let mut drawn = TerrainDrawn::default();
        if !self.ready(gpu) || !downsampler.ready(gpu) {
            // The map placed this frame's drape for these slots only: it
            // draws them all again once terrain can.
            if !self.pending.is_empty() {
                self.redraw_all();
                self.pending.clear();
            }
            return drawn;
        }
        // The last frame, which drew into them, was submitted.
        std::mem::take(&mut self.mip_targets).release(gpu);
        let Some(dem) = self.dem.filter(|_| self.reserved) else {
            return drawn;
        };
        if !self.pending.is_empty() {
            let start = passes.len();
            let layout = self.layout_state;
            let mut pending = std::mem::take(&mut self.pending);
            pending.sort_unstable();
            pending.dedup();
            drawn.slots = pending.len() as u32;
            let grid = layout.side as f32;
            for run in pending.chunk_by(|a, b| (a.page, a.texture) == (b.page, b.texture)) {
                let (page, texture) = (run[0].page, run[0].texture);
                let Some(slots) = (self.texels.get(texture as usize).and_then(Option::as_ref))
                    .and_then(|t| t.pages[page].as_ref())
                else {
                    continue;
                };
                let px = PAGE_PX[page] as f32;
                // A pass a slot's stack: its layer cleared, then the
                // stack's draws of the slot's content into it, the grid's
                // drape space, which the drape frame maps onto a viewport
                // of its size, moved so that the slot's place lands on the
                // layer.
                for p in run {
                    let Some(commands) = drape.get(&(p.stack, p.tile_slot)) else {
                        continue;
                    };
                    let corner = layout.corner(p.tile_slot, p.stack).map(|n| -(n as f32));
                    let viewport = [corner[0], corner[1], grid * px, grid * px];
                    passes.push(Pass::Render(RenderPass {
                        label: "drape",
                        colors: vec![ColorAttachment {
                            target: slots.draw[p.layer as usize],
                            resolve: None,
                            store: StoreOp::Store,
                            load: LoadOp::Clear([0.0; 4]),
                        }],
                        depth: None,
                        commands: std::iter::once(RenderCommand::SetViewport(viewport))
                            .chain(commands.iter().cloned())
                            .collect(),
                    }));
                    drawn.draws += 1;
                }
                // Each level from the one above, over the redrawn layers.
                let regions: Vec<[u32; 4]> =
                    (run.iter()).map(|p| [p.layer, 0, 0, slots.side]).collect();
                slots.mips.regions.write(gpu, &regions);
                let chain = ("drape mips", &slots.mips);
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
        for texels in self.texels.iter_mut().flatten() {
            if texels.group.is_some() {
                continue;
            }
            let Some(colour) = &texels.pages[0] else {
                continue;
            };
            let slopes = texels.pages[1].as_ref().unwrap_or(colour);
            texels.group = Some(gpu.create_bind_group(&BindGroupDesc {
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
            }));
        }
        self.bound = Some((frame, props));
        let step = GRID_STEPS.iter().position(|&s| s == self.grid_step);
        let indices = self.grid_ranges[step.expect("a grid step")].clone();
        let tiles = (self.tile_textures.iter())
            .map(|&texture| self.texels.get(texture as usize)?.as_ref()?.group)
            .collect();
        self.drawn = Some(Drawn { tiles, indices });
        // The terrain's depth, for what it hides.
        if let (Some([color, depth]), Some(pipeline)) = (self.occlusion, self.occlusion_pipeline) {
            let mut commands = vec![RenderCommand::SetPipeline(pipeline)];
            self.draw(&mut commands);
            passes.push(Pass::Render(RenderPass {
                label: "terrain depth",
                colors: vec![ColorAttachment {
                    target: color,
                    resolve: None,
                    store: StoreOp::Store,
                    load: LoadOp::Clear([0.0; 4]),
                }],
                depth: Some(DepthAttachment {
                    target: depth,
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Discard,
                }),
                commands,
            }));
            drawn.draws += 1;
        }
        drawn
    }

    /// The frame's tiles drawn, over the pipeline set before: nothing while
    /// the terrain can't draw. Each tile is a draw of its own, with its
    /// slots' texture bound, never an instance of a draw of tiles that
    /// share one: tiles drawn together showed otherwise where they meet
    /// than drawn apart, by a unit of colour here and there on Apple's GPUs,
    /// which made what shows depend on which texture holds which slot.
    fn draw(&self, commands: &mut Vec<RenderCommand>) {
        let Some(drawn) = &self.drawn else {
            return;
        };
        commands.push(RenderCommand::SetIndexBuffer {
            buffer: self.grid_indices,
            offset: 0,
        });
        for (tile, group) in (0u32..).zip(&drawn.tiles) {
            let Some(group) = *group else {
                continue;
            };
            commands.extend([
                RenderCommand::SetBindGroup { index: 0, group },
                RenderCommand::DrawIndexed {
                    indices: drawn.indices.clone(),
                    base_vertex: 0,
                    instances: tile..tile + 1,
                },
            ]);
        }
    }

    /// The terrain's depth alone, drawn first in a pass that tests it,
    /// the map's ([`MAP`]) or a 3D layer's ([`THREE_D`]): the nearest
    /// surface, which every layer's draw then shades once. Nothing while
    /// the terrain can't draw this frame.
    pub(crate) fn depth_commands(&self, at: usize) -> Vec<RenderCommand> {
        let Some(pipeline) = self.depth_pipeline_at(at).filter(|_| self.drawn.is_some()) else {
            return Vec::new();
        };
        let mut commands = vec![RenderCommand::SetPipeline(pipeline)];
        self.draw(&mut commands);
        commands
    }

    /// Terrain layer `layer`'s draw in the map's pass, after its depth
    /// ([`depth_commands`](Self::depth_commands)): its hillshades shaded
    /// through `shades` (none where it has none there: they hide), over
    /// the map, or where it reads the map beneath, from `beneath`, the
    /// canvas the pass before drew. Nothing while the terrain can't draw
    /// this frame.
    pub(crate) fn layer_commands(
        &mut self,
        gpu: &mut dyn Gpu,
        layer: u8,
        shades: ShadeTables,
        beneath: Option<TextureId>,
    ) -> Vec<RenderCommand> {
        let at = usize::from(layer);
        let (Some(kind), Some(_)) = (self.layers.get(at).map(TerrainLayer::kind), &self.drawn)
        else {
            return Vec::new();
        };
        let Some(pipeline) = self.pipelines[kind as usize] else {
            return Vec::new();
        };
        // A reference of the layer's own: what layers before it marked
        // doesn't keep it from drawing.
        let mut commands = vec![
            RenderCommand::SetStencilReference(u32::from(layer) % 255 + 1),
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup {
                index: 1,
                group: self.stack_groups[at],
            },
        ];
        if kind != Kind::Plain {
            let group = self.shade_group(gpu, at, shades);
            commands.push(RenderCommand::SetBindGroup { index: 2, group });
        }
        if kind == Kind::Beneath {
            let group = self.backdrop_group(gpu, at, beneath);
            commands.push(RenderCommand::SetBindGroup { index: 3, group });
        }
        self.draw(&mut commands);
        commands
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

    /// Terrain layer `layer`'s backdrop group: the map beneath it,
    /// `beneath`, or a blank texture without one.
    fn backdrop_group(
        &mut self,
        gpu: &mut dyn Gpu,
        layer: usize,
        beneath: Option<TextureId>,
    ) -> BindGroupId {
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
        let Some(texture) = beneath else {
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
        // The grid of the slots' places, which each slot's passes move
        // their viewport over.
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
        self.release_texels(gpu);
        self.release_occlusion(gpu);
        self.drawn = None;
        self.dem = None;
        self.pending.clear();
        self.reserved = false;
        self.layout_state.tile_slots = 0;
    }

    /// The DEM page texture elevated content reads, while terrain is on.
    pub(crate) fn dem_texture(&self) -> Option<TextureId> {
        self.paint.and(self.dem).map(|(texture, _)| texture)
    }

    pub(crate) fn active(&self) -> bool {
        self.paint.is_some() && self.reserved
    }

    /// Whether this frame's draws of the terrain draw it
    /// ([`passes`](Self::passes)).
    pub(crate) fn drawing(&self) -> bool {
        self.drawn.is_some()
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
            constants: [DEPTH_ALONE, &[]],
            depth: depth_test(TextureFormat::Depth32Float),
            ..Draw::new(&TERRAIN_SHADER, &[self.layout], target, None)
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
        self.release_texels(gpu);
        self.release_occlusion(gpu);
        let layouts = (self.layout, self.stack_layout);
        let backdrops: Vec<BindGroupId> = (self.backdrops.iter().flatten()).map(|b| b.1).collect();
        (self.backdrop_layout, backdrops, self.blank).release(gpu);
        let shades: Vec<BindGroupId> = (self.shade_groups.iter().flatten()).map(|s| s.1).collect();
        (self.shade_layout, shades).release(gpu);
        let depth = self.depth_pipelines.map(|d| d.map(|(_, p)| p));
        let pipelines = (self.pipelines, depth);
        let pipelines = (pipelines, self.occlusion_pipeline);
        self.grid.destroy(gpu);
        self.tiles.destroy(gpu);
        let buffers = (self.grid_indices, self.params);
        (layouts, pipelines, buffers, self.stack_uniforms).release(gpu);
        (self.stack_groups, self.mip_targets).release(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
