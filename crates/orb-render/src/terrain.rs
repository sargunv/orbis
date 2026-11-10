//! Terrain (render graph `Terrain`/`Drape`): the map's 2D
//! layers draped over a DEM-raised surface, after GL JS's render to
//! texture.
//!
//! Each terrain tile has a block of drape slots, one per drape stack (a run
//! of draped layers between layers that draw in 3D or on screen, or a
//! hillshade the terrain shades, whose stack holds its unlit slopes).
//! Slots are squares of mipmapped atlas pages, sized to the slots in use:
//! colour stacks in one, slope stacks at half the texels a side in
//! another. A new page takes the kept slots' texels, so none redraws for
//! it. A frame
//! redraws only the slots the runtime marks: their drape placements, the
//! draped nodes' only candidates, map them into the atlas, so one pass
//! redraws them all, replaying a bundle of the draped nodes per drape zoom
//! among them, then a compute pass downsamples their mip levels. Each
//! terrain layer (a stack, or two around a hillshade lit between them from
//! its slopes' stack and the frame's bearing) draws into a texture of its own
//! with the depth extrusions share, composited in the map at its place.

use crate::TargetFormat;
use crate::pipeline::{DEPTH, Draw};
use crate::release::Release;
use TextureViewDimension::D2Array;
use orb_hal::*;
use orb_shaders::{DRAPE_SHADER, Shader, TERRAIN_SHADE_SHADER, TERRAIN_SHADER};

use crate::offscreen::{Composite, Composites, Mode, Offscreen};

/// Drape texels per terrain tile side: twice the 512 logical pixels a
/// terrain tile covers at its zoom, GL JS's quality factor.
pub const DRAPE_SLOT_PX: u32 = 1024;

/// Slope slots' texels per terrain tile side: half the drape's. A shaded
/// hillshade's slopes come from DEM tiles of at most 512 pixels a side at
/// the drape zoom, which a 512-texel slot holds sample for sample, and a
/// quarter of the texels lets a viewport-lit hillshade cost a terrain
/// tile a quarter slot rather than a whole one (#142).
pub(crate) const SLOPE_SLOT_PX: u32 = DRAPE_SLOT_PX / 2;

/// The drape's pages by slot side: colour stacks, then slope stacks.
const PAGE_PX: [u32; 2] = [DRAPE_SLOT_PX, SLOPE_SLOT_PX];

/// Slots per side of the largest page of `slot_px` slots the device makes,
/// a power of two (see [`atlas_grid`]).
fn atlas_side(gpu: &dyn Gpu, slot_px: u32) -> u32 {
    let side = gpu.caps().limits.max_texture_dimension_2d / slot_px;
    1 << side.max(1).ilog2()
}

/// The atlas grid (columns, rows) for `slots` drape slots, or `None` to
/// keep the atlas of `held` slots: kept while it holds them and they fill
/// more than a quarter of it. A grid is a power of two slots a side, so
/// that drape space maps to clip space and atlas uv by powers of two,
/// exactly: a slot's texels then don't depend on where in the atlas it
/// lies, which depends on the tiles that came and went before. A new grid
/// holds the slots in the fewest, or on shrinking twice that, so that a
/// view must grow past it again to make another; squarest, columns first.
fn atlas_grid(slots: u32, held: Option<u32>, side: u32) -> Option<(u32, u32)> {
    let slots = slots.max(1);
    if held.is_some_and(|held| slots <= held && slots * 4 > held) {
        return None;
    }
    let shrink = held.is_some_and(|held| slots <= held);
    let count = (slots.next_power_of_two() << u32::from(shrink)).min(side * side);
    let k = count.ilog2();
    Some((1 << k.div_ceil(2), 1 << (k / 2)))
}

/// The terrain tiles the largest drape pages hold, with `stacks` colour
/// and slope stacks.
pub(crate) fn room(gpu: &dyn Gpu, stacks: [u32; 2]) -> u32 {
    let alloc = allocated(stacks);
    (0..2)
        .filter(|&p| alloc[p] > 0)
        .map(|p| atlas_side(gpu, PAGE_PX[p]).pow(2) / alloc[p])
        .min()
        .expect("a colour page")
}

/// The slots per tile each page holds: the colour page at least one, so
/// that it always exists for the terrain to bind.
fn allocated(stacks: [u32; 2]) -> [u32; 2] {
    [stacks[0].max(1), stacks[1]]
}
/// The drape atlas's format: draped layers draw into it with pipelines of
/// their own when the target's differs; mip levels are written as storage.
/// In linear light the colour page is drawn and sampled through its sRGB
/// twin, and its mip levels decode and encode as they average.
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

/// What one terrain draw composites in the map: a stack's slots, or the
/// stacks below and above a hillshade it shades between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TerrainLayer {
    pub(crate) below: Option<u8>,
    /// The hillshade node, and the stack its unlit slopes draw into.
    pub(crate) shade: Option<(u32, u8)>,
    pub(crate) above: Option<u8>,
}

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
/// page 1 at [`SLOPE_SLOT_PX`]. Tile slot `t`'s stack of rank `r` on page
/// `p` is that page's slot `t · stacks[p] + r`, at column
/// `slot % columns[p]`, row `slot / columns[p]`. Each page's drape space
/// is its texels with y up: texel `(x, y)` is drape world `(x, −y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrapeLayout {
    /// Colour stacks, then slope stacks.
    pub stacks: [u32; 2],
    /// Slots per row of each page.
    pub columns: [u32; 2],
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

    /// Tile slot `tile_slot`'s slot of `stack`: its page and slot there.
    fn slot(&self, tile_slot: u32, stack: u32) -> (usize, u32) {
        let (page, rank) = self.page(stack);
        (page, tile_slot * self.stacks[page] + rank)
    }

    /// A slot's top-left texel in its page.
    pub fn corner(&self, tile_slot: u32, stack: u32) -> [u32; 2] {
        let (page, slot) = self.slot(tile_slot, stack);
        let (columns, px) = (self.columns[page], PAGE_PX[page]);
        [slot % columns * px, slot / columns * px]
    }
}

/// Sorted atlas slots as runs side by side in a row of an atlas
/// `columns` slots wide: `[column, row, slots]`.
fn slot_runs(slots: &[u32], columns: u32) -> Vec<[u32; 3]> {
    let mut runs: Vec<[u32; 3]> = Vec::new();
    for &slot in slots {
        let (x, y) = (slot % columns, slot / columns);
        match runs.last_mut() {
            Some(run) if run[1] == y && run[0] + run[2] == x => run[2] += 1,
            _ => runs.push([x, y, 1]),
        }
    }
    runs
}

/// A drape bundle: its zoom, the page it draws into, the bundle.
pub(crate) type DrapeBundle = (u8, usize, RenderBundleId);

#[derive(Debug)]
struct Atlas {
    /// One layer, mipmapped.
    texture: TextureId,
    size: [u32; 2],
    /// Texels per slot side.
    slot_px: u32,
    /// Per mip level, a view, which the mip levels are written through.
    levels: Vec<TextureId>,
    /// What draws into level 0 and what samples every level: the sRGB
    /// twin's views of a linear-light colour page, else level 0's view and
    /// the texture.
    draw: TextureId,
    sampled: TextureId,
    /// The redrawn slots as the downsampler's regions, which the clear
    /// reads too.
    regions: BufferId,
    /// The atlas's size, for the clear.
    size_uniform: BufferId,
    clear: BindGroupId,
    /// The downsampler's group per level after the first.
    mips: Vec<BindGroupId>,
}

impl Atlas {
    /// A page of `columns × rows` slots of `slot_px` texels.
    fn new(
        gpu: &mut dyn Gpu,
        (columns, rows): (u32, u32),
        slot_px: u32,
        clear_layout: BindGroupLayoutId,
        downsampler: &mut crate::downsample::Downsampler,
        linear: bool,
    ) -> Self {
        let size = [columns, rows].map(|n| n * slot_px);
        let levels = slot_px.ilog2() + 1;
        let texture = gpu.create_texture(&TextureDesc {
            label: "drape atlas",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2Array,
            format: DRAPE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: levels,
            view_formats: if linear {
                &[TextureFormat::Rgba8UnormSrgb]
            } else {
                &[]
            },
        });
        let views: Vec<TextureId> = (0..levels)
            .map(|l| gpu.create_texture_view(texture, l..l + 1))
            .collect();
        let (draw, sampled) = if linear {
            let srgb = DRAPE_FORMAT.srgb();
            (
                gpu.create_texture_view_as(texture, 0..1, srgb),
                gpu.create_texture_view_as(texture, 0..levels, srgb),
            )
        } else {
            (views[0], texture)
        };
        let regions = gpu.create_buffer(&BufferDesc {
            label: "drape slots",
            size: u64::from(columns * rows) * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
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
                BindGroupEntry::buffer(1, regions),
            ],
        });
        downsampler.make(gpu);
        let mips = downsampler.groups(gpu, &views, regions, linear);
        Self {
            texture,
            size,
            slot_px,
            levels: views,
            draw,
            sampled,
            regions,
            size_uniform,
            clear,
            mips,
        }
    }

    /// Its slots: columns and rows.
    fn grid(&self) -> (u32, u32) {
        (self.size[0] / self.slot_px, self.size[1] / self.slot_px)
    }

    /// Its bytes, mips included.
    fn bytes(&self) -> u64 {
        let [w, h] = self.size.map(u64::from);
        w * h * 4 * 4 / 3
    }
}

impl crate::release::Release for Atlas {
    fn release(self, gpu: &mut dyn Gpu) {
        let buffers = (self.regions, self.size_uniform);
        // A linear page's own views; else they're level 0's and the texture.
        let views = (self.draw != self.levels[0]).then_some([self.draw, self.sampled]);
        (
            self.clear,
            self.mips,
            self.levels,
            views,
            self.texture,
            buffers,
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
    clear_layout: BindGroupLayoutId,
    /// Draping and shading, each at one sample and at four.
    pipelines: [[Option<RenderPipelineId>; 2]; 2],
    /// Clearing the colour page's slots and the slope page's, which differ
    /// in linear light.
    clear: [RenderPipelineId; 2],
    grid: BufferId,
    grid_indices: BufferId,
    /// The index range of each grid step's mesh, and the step drawn.
    grid_ranges: [std::ops::Range<u32>; GRID_STEPS.len()],
    grid_step: u32,
    tiles: BufferId,
    tile_capacity: u32,
    tile_count: u32,
    params: BufferId,
    /// The terrain layers, and each's group over its slice of
    /// `stack_uniforms`, made for as many as the style has.
    layers: Vec<TerrainLayer>,
    stack_groups: Vec<BindGroupId>,
    stack_uniforms: Option<BufferId>,
    linear: SamplerId,
    /// The colour page, and the slope page while a hillshade is shaded.
    atlas: [Option<Atlas>; 2],
    layout_state: DrapeLayout,
    /// The DEM page texture and revision the group binds.
    dem: Option<(TextureId, u32)>,
    /// The group, and the frame and property buffers it binds.
    group: Option<(BindGroupId, [BufferId; 2])>,
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
    /// Slots to redraw this frame (page, slot) with the drape zoom each
    /// draws at, and those zooms with the pages they draw into.
    pending: Vec<(usize, u32, u8)>,
    pending_draws: Vec<(u8, usize)>,
    pub(crate) paint: Option<TerrainPaint>,
    /// The atmosphere's uniform, for fog.
    atmosphere: BufferId,
}

impl Terrain {
    /// Terrain drawn into `target`, fogged by `atmosphere`, shading
    /// hillshades through image node groups of `shade_layout`.
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        atmosphere: BufferId,
        shade_layout: BindGroupLayoutId,
    ) -> Self {
        use BindGroupLayoutEntry as E;
        let vs = ShaderStages::VERTEX;
        let fs = ShaderStages::FRAGMENT;
        let vf = vs | fs;
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain",
            entries: &[
                E::uniform(0, vf),
                // Terrain tiles: a shaded hillshade reads their latitudes.
                E::storage(1, vf),
                E::storage(2, vs),
                E::texture(3, vs, false, D2Array),
                E::texture(4, fs, true, D2Array),
                E::sampler(5, fs, true),
                E::uniform(6, vf),
                E::uniform(7, vf),
                // The scene's property arena: globe patches.
                E::storage(8, vs),
                // The slope page (the colour page while there is none).
                E::texture(9, fs, true, D2Array),
            ],
        });
        let stack_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain layer",
            entries: &[E::uniform(0, fs)],
        });
        let clear_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "drape clear",
            entries: &[E::uniform(0, vs), E::storage(1, vs)],
        });
        let mut clear_pipeline = |color| {
            let target = TargetFormat {
                color,
                sample_count: 1,
            };
            Draw {
                topology: PrimitiveTopology::TriangleStrip,
                ..Draw::new(&DRAPE_SHADER, &[clear_layout], target, None)
            }
            .make(gpu)
        };
        let slope_clear = clear_pipeline(DRAPE_FORMAT);
        let clear = if target.color.is_srgb() {
            [clear_pipeline(DRAPE_FORMAT.srgb()), slope_clear]
        } else {
            [slope_clear; 2]
        };

        let (grid, indices, grid_ranges) = grid();
        let grid_buffer = gpu.create_buffer(&BufferDesc {
            label: "terrain grid",
            size: grid.len() as u64 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        gpu.write_buffer(grid_buffer, 0, bytemuck::cast_slice(&grid));
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
        // Trilinear and anisotropic, as GL JS samples its render-to-texture
        // tiles.
        let linear = gpu.create_sampler(&SamplerDesc {
            label: "drape",
            filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            anisotropy: 16,
        });
        Self {
            target,
            layout,
            stack_layout,
            shade_layout,
            clear_layout,
            pipelines: [[None; 2]; 2],
            clear,
            grid: grid_buffer,
            grid_indices,
            grid_ranges,
            grid_step: 1,
            tiles: gpu.create_buffer(&BufferDesc {
                label: "terrain tiles",
                size: 64 * terrain_tile::WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
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
                columns: [1; 2],
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
            pending_draws: Vec::new(),
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
                            16,
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
            // nothing too, and the layers over it only their own content.
            let words = [
                stack(layer.below),
                stack(layer.above),
                slopes,
                u32::from(i == 0),
            ];
            gpu.write_buffer(
                buffer,
                UNIFORM_STRIDE * i as u64,
                bytemuck::cast_slice(&words),
            );
        }
        self.layers = layers.to_vec();
    }

    /// Sizes the pages for `tile_slots` blocks of `stacks` colour and
    /// slope slots ([`atlas_grid`]); each new page copies the kept slots.
    /// The tile slots in `kept` carry their texels into new pages, those
    /// past the new end moved to free slots; returns the moves (from, to),
    /// or `None` when the pages stay. A change of stacks blanks every slot
    /// (a new [`DrapeLayout::generation`]). Fails when the slots exceed the
    /// device's texture size ([`room`]).
    pub(crate) fn reserve(
        &mut self,
        gpu: &mut dyn Gpu,
        (tile_slots, kept): (u32, &[u32]),
        stacks: [u32; 2],
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<Option<Vec<(u32, u32)>>, crate::OutOfSpace> {
        let state = self.layout_state;
        if tile_slots > room(gpu, stacks) {
            return Err(crate::OutOfSpace);
        }
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
                .map(Atlas::grid);
            let side = atlas_side(gpu, PAGE_PX[page]);
            let grid = atlas_grid(tile_slots * alloc[page], held.map(|(c, r)| c * r), side);
            changed |= grid.is_some();
            grids[page] = grid.or(held);
        }
        if !changed {
            return Ok(None);
        }
        let layout = DrapeLayout {
            stacks,
            columns: grids.map(|g| g.map_or(1, |(columns, _)| columns)),
            tile_slots: (0..2)
                .filter_map(|p| grids[p].map(|(c, r)| c * r / alloc[p]))
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
                Some(old) if same && old.grid() == grid && moves.is_empty() => {
                    self.atlas[page] = Some(old);
                    continue;
                }
                Some(old) if same => Some(old),
                other => {
                    other.release(gpu);
                    None
                }
            };
            let linear = page == 0 && self.target.color.is_srgb();
            let atlas = Atlas::new(
                gpu,
                grid,
                PAGE_PX[page],
                self.clear_layout,
                downsampler,
                linear,
            );
            if let Some(old) = old {
                let px = PAGE_PX[page];
                let first = if page == 0 { 0 } else { stacks[0] };
                let copies = (kept.iter())
                    .flat_map(|&t| (first..first + stacks[page]).map(move |s| (t, s)))
                    .map(|(t, s)| {
                        let ([x0, y0], [x1, y1]) = (state.corner(t, s), layout.corner(to(t), s));
                        orb_hal::TextureCopy {
                            from: [x0, y0, 0],
                            to: [x1, y1, 0],
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
        self.pending_draws.clear();
        self.forget_group(gpu);
        Ok(Some(moves))
    }

    /// Writes the layout's words of the terrain's values.
    fn write_layout(&self, gpu: &mut dyn Gpu) {
        let state = self.layout_state;
        let words = [
            (params::STACKS, state.stacks[0]),
            (params::COLUMNS, state.columns[0]),
            (params::SLOPE_STACKS, state.stacks[1]),
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

    /// Redraws tile slots' stacks this frame: per tile slot, its drape
    /// zoom and the stacks to redraw.
    pub(crate) fn redraw(&mut self, tile_slots: &[(u32, u8, &[u8])]) {
        let layout = self.layout_state;
        for &(tile, zoom, redrawn) in tile_slots {
            let stacks =
                (redrawn.iter().map(|&s| u32::from(s))).filter(|&s| s < layout.stack_count());
            for stack in stacks {
                let (page, slot) = layout.slot(tile, stack);
                self.pending.push((page, slot, zoom));
                self.pending_draws.push((zoom, page));
            }
        }
        self.pending_draws.sort_unstable();
        self.pending_draws.dedup();
    }

    /// The drape zooms of the slots this frame redraws, with the pages
    /// they draw into.
    pub(crate) fn pending_draws(&self) -> &[(u8, usize)] {
        &self.pending_draws
    }

    pub(crate) fn set_tiles(&mut self, gpu: &mut dyn Gpu, tiles: &[TerrainTile]) {
        if tiles.len() as u32 > self.tile_capacity {
            gpu.destroy_buffer(self.tiles);
            self.tile_capacity = (tiles.len() as u32).next_power_of_two();
            self.tiles = gpu.create_buffer(&BufferDesc {
                label: "terrain tiles",
                size: u64::from(self.tile_capacity) * terrain_tile::WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
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
            gpu.write_buffer(self.tiles, 0, bytemuck::cast_slice(&words));
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
        if self.dem != Some(dem) {
            self.forget_group(gpu);
            self.dem = Some(dem);
        }
        self.grid_step = grid_step(paint.dem_tile_px);
        let state = self.layout_state;
        let mut b = orb_shaders::Block::new(params::WORDS);
        b.f(params::UNPACK, &paint.unpack)
            .f(params::EXAGGERATION, &[paint.exaggeration]);
        b.f(params::DEM_DIM, &[dim as f32])
            .f(params::SKIRT, &[paint.skirt]);
        b.u(params::STACKS, &[state.stacks[0]])
            .u(params::COLUMNS, &[state.columns[0]]);
        b.u(params::SLOPE_STACKS, &[state.stacks[1]])
            .u(params::SLOPE_COLUMNS, &[state.columns[1]]);
        b.f(params::SLOT_PX, &[DRAPE_SLOT_PX as f32])
            .f(params::SLOPE_PX, &[SLOPE_SLOT_PX as f32])
            .u(params::TILES, &[self.tile_count]);
        gpu.write_buffer(self.params, 0, bytemuck::cast_slice(&b.0));
        self.paint = Some(paint);
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
                view_formats: &[],
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
        [pipeline.into(), self.clear[0].into(), self.clear[1].into()]
            .into_iter()
            .chain(shaded.map(PipelineId::from))
            .chain(occlusion.map(PipelineId::from))
            .all(|p| gpu.pipeline_ready(p))
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        let mut out: Vec<PipelineId> = vec![self.clear[0].into()];
        if self.clear[1] != self.clear[0] {
            out.push(self.clear[1].into());
        }
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
        self.pending_draws.clear();
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
                &[self.layout, self.stack_layout, self.shade_layout],
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
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (frame, props): (BufferId, BufferId),
        (drape, shades): (&[DrapeBundle], &[Option<BindGroupId>]),
        (depth, depth_store): (TextureId, StoreOp),
        downsampler: &crate::downsample::Downsampler,
        (passes, layers): (&mut Vec<Pass>, &mut Vec<RenderPass>),
    ) -> u32 {
        let mut draws = 0;
        if !self.ready(gpu) || !downsampler.ready(gpu) {
            // The map placed this frame's drape for these slots only: it
            // draws them all again once terrain can.
            if !self.pending.is_empty() {
                self.redraw_all();
            }
            return 0;
        }
        let (Some(colour), Some(dem)) = (&self.atlas[0], self.dem) else {
            return 0;
        };
        if !self.pending.is_empty() {
            let start = passes.len();
            self.pending.sort_unstable();
            self.pending.dedup();
            for (page, atlas) in self.atlas.iter().enumerate() {
                let Some(atlas) = atlas else {
                    continue;
                };
                let slots: Vec<(u32, u8)> = (self.pending.iter())
                    .filter(|p| p.0 == page)
                    .map(|&(_, slot, zoom)| (slot, zoom))
                    .collect();
                if slots.is_empty() {
                    continue;
                }
                let (columns, _) = atlas.grid();
                let px = atlas.slot_px;
                let regions: Vec<[u32; 4]> = (slots.iter())
                    .map(|&(s, _)| [0, s % columns * px, s / columns * px, px])
                    .collect();
                gpu.write_buffer(atlas.regions, 0, bytemuck::cast_slice(&regions));
                let count = slots.len() as u32;
                let mut commands = vec![
                    RenderCommand::SetPipeline(self.clear[page]),
                    RenderCommand::SetBindGroup {
                        index: 0,
                        group: atlas.clear,
                    },
                    RenderCommand::Draw {
                        vertices: 0..4,
                        instances: 0..count,
                    },
                ];
                // Each zoom's bundle replays once per run of its redrawn
                // slots side by side, scissored to it: what a coarse tile
                // standing in draws spans far more than the slots it draws
                // into (all of the page), and would otherwise be
                // rasterized there only to be discarded.
                for &(zoom, _, bundle) in drape.iter().filter(|d| d.1 == page) {
                    let at: Vec<u32> = (slots.iter())
                        .filter(|p| p.1 == zoom)
                        .map(|p| p.0)
                        .collect();
                    for [x, y, width] in slot_runs(&at, columns) {
                        commands.push(RenderCommand::SetScissor([x * px, y * px, width * px, px]));
                        commands.push(RenderCommand::ExecuteBundles(vec![bundle]));
                    }
                }
                passes.push(Pass::Render(RenderPass {
                    label: "drape",
                    colors: vec![ColorAttachment {
                        target: atlas.draw,
                        resolve: None,
                        store: StoreOp::Store,
                        load: LoadOp::Load,
                    }],
                    depth: None,
                    commands,
                }));
                // Each level from the one above, one dispatch over the slots.
                let mips = downsampler.dispatches(gpu, &atlas.mips, px, count);
                passes.push(Pass::Compute(ComputePass {
                    label: "drape mips",
                    commands: mips.expect("ready above"),
                }));
                draws += 1;
            }
            self.pending.clear();
            self.pending_draws.clear();
            Pass::group(passes, start, "drape");
        }
        let atlas_texture = colour.sampled;
        let slope_texture = self.atlas[1].as_ref().map_or(atlas_texture, |a| a.sampled);
        if self.group.is_some_and(|(_, bound)| bound != [frame, props]) {
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
                        BindGroupEntry::buffer(1, self.tiles),
                        BindGroupEntry::buffer(2, self.grid),
                        BindGroupEntry::texture(3, dem.0, D2Array),
                        BindGroupEntry::texture(4, atlas_texture, D2Array),
                        BindGroupEntry::sampler(5, self.linear),
                        BindGroupEntry::buffer(6, self.params),
                        BindGroupEntry::buffer(7, self.atmosphere),
                        BindGroupEntry::buffer(8, props),
                        BindGroupEntry::texture(9, slope_texture, D2Array),
                    ],
                });
                self.group = Some((group, [frame, props]));
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
        let draw = |pipeline, layer_group, shade: Option<BindGroupId>| {
            let mut commands = vec![
                RenderCommand::SetPipeline(pipeline),
                RenderCommand::SetBindGroup { index: 0, group },
                RenderCommand::SetBindGroup {
                    index: 1,
                    group: layer_group,
                },
            ];
            commands.extend(shade.map(|group| RenderCommand::SetBindGroup { index: 2, group }));
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
            draws += 1;
        }
        for (layer, target) in self.targets.iter().enumerate() {
            let shade = shades.get(layer).copied().flatten();
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
            draws += 1;
        }
        draws
    }

    /// The drape frame at style zoom `zoom`: the map's, drawn into page
    /// `page`.
    pub(crate) fn drape_frame(
        &self,
        frame: &crate::scene::FrameBlock,
        (zoom, page): (u8, usize),
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
            // pixel at the drape zoom.
            world_per_px: 1.0,
            pixel_ratio: 1.0,
            zoom: f64::from(zoom),
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
        self.pending_draws.clear();
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
        let [colour_clear, slope_clear] = self.clear;
        let clear = (
            colour_clear,
            (slope_clear != colour_clear).then_some(slope_clear),
        );
        let pipelines = (self.pipelines, clear, self.occlusion_pipeline);
        let buffers = (self.grid, self.grid_indices, self.tiles, self.params);
        (layouts, pipelines, buffers, self.stack_uniforms).release(gpu);
        (self.stack_groups, self.linear, self.targets).release(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redrawn_slots_replay_in_runs_along_rows() {
        // Four columns: 2, 3 run together; 4 starts the next row.
        assert_eq!(
            slot_runs(&[0, 2, 3, 4, 5, 7], 4),
            [[0, 0, 1], [2, 0, 2], [0, 1, 2], [3, 1, 1]]
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
        assert_eq!(atlas_grid(0, None, 16), Some((1, 1)));
        assert_eq!(atlas_grid(24, None, 16), Some((8, 4)));
        assert_eq!(atlas_grid(33, None, 16), Some((8, 8)));
        // Kept while it holds them and they fill over a quarter of it.
        assert_eq!(atlas_grid(32, Some(32), 16), None);
        assert_eq!(atlas_grid(9, Some(32), 16), None);
        assert_eq!(atlas_grid(33, Some(32), 16), Some((8, 8)));
        // Shrunk to twice the fewest, half full.
        assert_eq!(atlas_grid(8, Some(32), 16), Some((4, 4)));
        // Never past the device's largest.
        assert_eq!(atlas_grid(200, None, 8), Some((8, 8)));
    }

    #[test]
    fn slots_fill_rows_of_the_atlas() {
        let layout = DrapeLayout {
            stacks: [2, 1],
            columns: [8, 4],
            tile_slots: 16,
            generation: 1,
        };
        assert_eq!(layout.corner(0, 1), [DRAPE_SLOT_PX, 0]);
        assert_eq!(layout.corner(4, 0), [0, DRAPE_SLOT_PX]);
        // Slopes on their own page, at half the texels a side.
        assert_eq!(layout.page(2), (1, 0));
        assert_eq!(layout.corner(1, 2), [SLOPE_SLOT_PX, 0]);
        assert_eq!(layout.corner(5, 2), [SLOPE_SLOT_PX, SLOPE_SLOT_PX]);
    }
}
