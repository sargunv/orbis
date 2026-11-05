//! Terrain (plan §9.4, render graph `Terrain`/`Drape`): the map's 2D
//! layers draped over a DEM-raised surface, after GL JS's render to
//! texture (decision 0019).
//!
//! Each terrain tile has a block of drape slots, one per drape stack (a run
//! of draped layers between layers that draw in 3D or on screen, or a
//! hillshade the terrain shades). Slots are squares of one mipmapped
//! atlas, sized to the slots in use: a new one takes the kept slots'
//! texels, so none redraws for it. A frame redraws only the slots the
//! runtime marks: their drape placements, the draped nodes' only
//! candidates, map them into the atlas, so one pass redraws them all,
//! replaying a bundle of the draped nodes per drape zoom among them, then
//! a compute pass downsamples their mip levels. Each terrain layer (a stack, or two around a hillshade shaded
//! between them from the frame's bearing) draws into a texture of its own
//! with the depth extrusions share, composited in the map at its place.

use crate::TargetFormat;
use crate::release::Release;
use orb_hal::*;
use orb_shaders::{DRAPE_SHADER, Shader, TERRAIN_SHADE_SHADER, TERRAIN_SHADER};

use crate::offscreen::{Composite, Composites, Mode, Offscreen};

/// Drape texels per terrain tile side: twice the 512 logical pixels a
/// terrain tile covers at its zoom, GL JS's quality factor.
pub const DRAPE_SLOT_PX: u32 = 1024;

/// Slots per side of the largest drape atlas the device makes.
fn atlas_side(gpu: &dyn Gpu) -> u32 {
    gpu.caps().limits.max_texture_dimension_2d / DRAPE_SLOT_PX
}

/// The terrain tiles the largest drape atlas holds, `stacks` slots each.
pub(crate) fn room(gpu: &dyn Gpu, stacks: u32) -> u32 {
    atlas_side(gpu).pow(2) / stacks.max(1)
}
/// The drape atlas's format: draped layers draw into it with pipelines of
/// their own when the target's differs; mip levels are written as storage.
pub const DRAPE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
/// Grid cells per terrain tile side (`terrain.wgsl`'s `GRID`): one per
/// DEM pixel at the DEM zoom the runtime picks.
pub const TERRAIN_GRID: u32 = 128;
/// Terrain units per tile side (`terrain.wgsl`'s `TERRAIN_EXTENT`).
pub const TERRAIN_EXTENT: f32 = 8192.0;
const NO_DEM: u32 = u32::MAX;
const NONE: u32 = u32::MAX;
use orb_shaders::layout::{shade_tile, terrain as params, terrain_tile};
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
}

/// What one terrain draw composites in the map: a stack's slots, or the
/// stacks below and above a hillshade it shades between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TerrainLayer {
    pub(crate) below: Option<u8>,
    /// The hillshade node, which draws no slots.
    pub(crate) shade: Option<u32>,
    pub(crate) above: Option<u8>,
}

/// Where a shaded hillshade reads a quarter of a terrain tile's slopes: an
/// image tile of its source, and the quarter's uv → that tile's uv as
/// `uv · scale + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainSlopes {
    pub tile: crate::GpuTile,
    pub offset: [f32; 2],
    pub scale: f32,
}

/// A terrain tile on a globe (decision 0030's surface): its patches in
/// space around `origin`, the anchor less the camera target in metres, and
/// which poles, north then south, its edge skirts close (a top or bottom
/// row of a set that stops short of the pole).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainGlobe {
    pub warp: orb_scene::TileWarp,
    pub origin: [f32; 3],
    pub poles: [bool; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainDem {
    pub layer: u32,
    pub offset: [f32; 2],
    pub scale: f32,
}

/// Where drape slots are: tile slot `t`'s stack `s` is atlas slot
/// `t · stacks + s`, at column `slot % columns`, row `slot / columns`,
/// [`DRAPE_SLOT_PX`] texels a side. Drape space is atlas texels with y
/// up: texel `(x, y)` is drape world `(x, −y)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrapeLayout {
    pub stacks: u32,
    pub columns: u32,
    /// Tile slots the atlas holds.
    pub tile_slots: u32,
    /// Changes when every slot is blank (a new atlas without the old one's
    /// texels, or other stacks): every slot must be redrawn.
    pub generation: u64,
}

impl DrapeLayout {
    /// An atlas slot's top-left texel.
    pub fn corner(&self, tile_slot: u32, stack: u32) -> [u32; 2] {
        let slot = tile_slot * self.stacks + stack;
        [
            slot % self.columns * DRAPE_SLOT_PX,
            slot / self.columns * DRAPE_SLOT_PX,
        ]
    }
}

#[derive(Debug)]
struct Atlas {
    /// One layer, mipmapped.
    texture: TextureId,
    size: [u32; 2],
    /// Per mip level, a view: level 0's is drawn into.
    levels: Vec<TextureId>,
    /// The redrawn slots as the downsampler's regions, which the clear
    /// reads too.
    regions: BufferId,
    /// The atlas's size, for the clear.
    size_uniform: BufferId,
    clear: BindGroupId,
    /// The downsampler's group per level after the first.
    mips: Vec<BindGroupId>,
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
    clear: RenderPipelineId,
    grid: BufferId,
    grid_indices: BufferId,
    index_count: u32,
    tiles: BufferId,
    tile_capacity: u32,
    tile_count: u32,
    params: BufferId,
    /// Shaded hillshades' slope tiles, each's for every terrain tile.
    shade_tiles: BufferId,
    shade_capacity: u64,
    /// The terrain layers, and each's group over its slice of
    /// `stack_uniforms`, made for as many as the style has.
    layers: Vec<TerrainLayer>,
    stack_groups: Vec<BindGroupId>,
    stack_uniforms: Option<BufferId>,
    linear: SamplerId,
    atlas: Option<Atlas>,
    layout_state: DrapeLayout,
    /// The DEM page texture and revision the group binds.
    dem: Option<(TextureId, u32)>,
    /// The group, and the frame and property buffers it binds.
    group: Option<(BindGroupId, [BufferId; 2])>,
    /// Each terrain layer's texture and its composite into the map.
    targets: Vec<Offscreen>,
    /// The terrain's depth as a colour, and the depth it's drawn with: what
    /// hides symbols and circles (#40).
    occlusion: Option<[TextureId; 2]>,
    occlusion_pipeline: Option<RenderPipelineId>,
    target_size: [u32; 2],
    target_samples: u32,
    /// Atlas slots to redraw this frame, and the drape zooms they draw at.
    pending: Vec<u32>,
    pending_zooms: Vec<u8>,
    pub(crate) paint: Option<TerrainPaint>,
    /// The atmosphere's uniform, for fog.
    atmosphere: BufferId,
}

fn uniform(binding: u32, visibility: ShaderStages) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Uniform,
    }
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
        let vs = ShaderStages::VERTEX;
        let fs = ShaderStages::FRAGMENT;
        let vf = vs | fs;
        let storage = |binding| BindGroupLayoutEntry {
            binding,
            visibility: vs,
            ty: BindingType::Storage { read_only: true },
        };
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain",
            entries: &[
                uniform(0, vf),
                storage(1),
                storage(2),
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vs,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: false },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: fs,
                    ty: BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: true },
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: fs,
                    ty: BindingType::Sampler { filtering: true },
                },
                uniform(6, vf),
                uniform(7, vf),
                // The scene's property arena: globe patches.
                storage(8),
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: fs,
                    ty: BindingType::Storage { read_only: true },
                },
            ],
        });
        let stack_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain layer",
            entries: &[uniform(0, fs)],
        });
        let clear_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "drape clear",
            entries: &[uniform(0, vs), storage(1)],
        });
        let drape_module = gpu.create_shader_module(&ShaderModuleDesc {
            label: DRAPE_SHADER.label,
            source: ShaderSource::Wgsl(DRAPE_SHADER.source().into()),
        });
        let clear = gpu.create_render_pipeline(&RenderPipelineDesc {
            label: DRAPE_SHADER.label,
            bind_group_layouts: &[clear_layout],
            vertex: ShaderStage {
                module: drape_module,
                entry_point: DRAPE_SHADER.vertex,
                constants: &[],
            },
            fragment: ShaderStage {
                module: drape_module,
                entry_point: DRAPE_SHADER.fragment,
                constants: &[],
            },
            targets: &[ColorTargetState {
                format: DRAPE_FORMAT,
                blend: None,
            }],
            depth: None,
            topology: PrimitiveTopology::TriangleStrip,
            cull_mode: None,
            sample_count: 1,
        });
        gpu.destroy_shader_module(drape_module);

        let (grid, indices) = grid();
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
            address: AddressMode::ClampToEdge,
            mipmap_filter: FilterMode::Linear,
            anisotropy: 16,
            compare: None,
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
            index_count: indices.len() as u32,
            tiles: gpu.create_buffer(&BufferDesc {
                label: "terrain tiles",
                size: 64 * terrain_tile::WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
            tile_capacity: 64,
            tile_count: 0,
            params,
            shade_tiles: gpu.create_buffer(&BufferDesc {
                label: "terrain shade tiles",
                size: 64 * shade_tile::WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
            shade_capacity: 64,
            layers: Vec::new(),
            stack_groups: Vec::new(),
            stack_uniforms: None,
            linear,
            atlas: None,
            layout_state: DrapeLayout {
                stacks: 1,
                columns: 1,
                tile_slots: 0,
                generation: 0,
            },
            dem: None,
            group: None,
            targets: Vec::new(),
            occlusion: None,
            occlusion_pipeline: None,
            target_size: [0; 2],
            target_samples: 0,
            pending: Vec::new(),
            pending_zooms: Vec::new(),
            paint: None,
            atmosphere,
        }
    }

    /// The drape atlas's bytes, mips included.
    pub(crate) fn atlas_bytes(&self) -> u64 {
        let [w, h] = self
            .atlas
            .as_ref()
            .map_or([0; 2], |a| a.size.map(u64::from));
        w * h * 4 * 4 / 3
    }

    pub(crate) fn layout(&self) -> DrapeLayout {
        self.layout_state
    }

    /// This frame's terrain layers: each gets a group binding what it
    /// draws, and a texture once sized.
    pub(crate) fn set_layers(&mut self, gpu: &mut dyn Gpu, layers: &[TerrainLayer]) {
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
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::Buffer {
                                buffer,
                                offset: UNIFORM_STRIDE * u64::from(layer),
                                size: Some(16),
                            },
                        }],
                    })
                })
                .collect();
            self.stack_uniforms = Some(buffer);
        }
        let buffer = self.stack_uniforms.expect("made above");
        let mut shades = 0;
        for (i, layer) in layers.iter().enumerate() {
            let stack = |s: Option<u8>| s.map_or(NONE, u32::from);
            let shade = layer.shade.map_or(NONE, |_| {
                shades += 1;
                shades - 1
            });
            let words = [stack(layer.below), stack(layer.above), shade, 0];
            gpu.write_buffer(
                buffer,
                UNIFORM_STRIDE * i as u64,
                bytemuck::cast_slice(&words),
            );
        }
        self.layers = layers.to_vec();
    }

    /// The shaded hillshades' slope tiles: a `ShadeTile` per quarter of
    /// a terrain tile, each hillshade's after the one before.
    pub(crate) fn set_shade_tiles(&mut self, gpu: &mut dyn Gpu, words: &[u32]) {
        let entries = (words.len() / shade_tile::WORDS) as u64;
        if entries > self.shade_capacity {
            gpu.destroy_buffer(self.shade_tiles);
            self.shade_capacity = entries.next_power_of_two();
            self.shade_tiles = gpu.create_buffer(&BufferDesc {
                label: "terrain shade tiles",
                size: self.shade_capacity * shade_tile::WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            self.forget_group(gpu);
        }
        if !words.is_empty() {
            gpu.write_buffer(self.shade_tiles, 0, bytemuck::cast_slice(words));
        }
    }

    /// Sizes the atlas for `tile_slots` blocks of `stacks` slots, with a
    /// quarter to spare: made again when they outgrow it, or fill less
    /// than half of it and the new one is at least a quarter smaller (each
    /// new atlas copies the kept slots). The tile slots in `kept` carry their texels
    /// into a new atlas, those past its end moved to free slots; returns
    /// the moves (from, to), or `None` when the atlas stays. A change of
    /// stacks blanks every slot (a new [`DrapeLayout::generation`]).
    /// Fails when the slots exceed the device's texture size
    /// ([`room`]).
    pub(crate) fn reserve(
        &mut self,
        gpu: &mut dyn Gpu,
        (tile_slots, kept): (u32, &[u32]),
        stacks: u32,
        downsampler: &mut crate::downsample::Downsampler,
    ) -> Result<Option<Vec<(u32, u32)>>, crate::OutOfSpace> {
        let stacks = stacks.max(1);
        let state = self.layout_state;
        let side = atlas_side(gpu);
        if tile_slots > room(gpu, stacks) {
            return Err(crate::OutOfSpace);
        }
        let want = (tile_slots + tile_slots.div_ceil(4)).clamp(1, room(gpu, stacks));
        let same = self.atlas.is_some() && state.stacks == stacks;
        if same && tile_slots <= state.tile_slots && tile_slots * 2 >= state.tile_slots {
            return Ok(None);
        }
        // The grid of the fewest slots that holds them, squarest first.
        let slots = want * stacks;
        let (columns, rows) = (1..=side)
            .map(|c| (c, slots.div_ceil(c)))
            .filter(|&(_, r)| r <= side)
            .min_by_key(|&(c, r)| (c * r, c.abs_diff(r)))
            .expect("one column fits");
        let held = state.tile_slots * stacks;
        if same && tile_slots <= state.tile_slots && columns * rows * 4 > held * 3 {
            return Ok(None);
        }
        let layout = DrapeLayout {
            stacks,
            columns,
            tile_slots: columns * rows / stacks,
            generation: state.generation + u64::from(!same),
        };
        // Kept slots past the new end move to the lowest free ones.
        let mut free = (0..layout.tile_slots).filter(|t| !kept.contains(t));
        let moves: Vec<(u32, u32)> = (kept.iter().filter(|&&t| t >= layout.tile_slots))
            .map(|&t| (t, free.next().expect("room for every kept slot")))
            .collect();
        let size = [columns, rows].map(|n| n * DRAPE_SLOT_PX);
        let levels = DRAPE_SLOT_PX.ilog2() + 1;
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
        });
        let views: Vec<TextureId> = (0..levels)
            .map(|l| gpu.create_texture_view(texture, l..l + 1))
            .collect();
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
            layout: self.clear_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::buffer(size_uniform),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::buffer(regions),
                },
            ],
        });
        downsampler.make(gpu);
        let mips = downsampler.groups(gpu, &views, regions);
        let atlas = Atlas {
            texture,
            size,
            levels: views,
            regions,
            size_uniform,
            clear,
            mips,
        };
        if let Some(old) = self.atlas.as_ref().filter(|_| same) {
            let to = |t: u32| moves.iter().find(|m| m.0 == t).map_or(t, |m| m.1);
            let copies = (kept.iter().flat_map(|&t| (0..stacks).map(move |s| (t, s))))
                .map(|(t, s)| {
                    let ([x0, y0], [x1, y1]) = (state.corner(t, s), layout.corner(to(t), s));
                    orb_hal::TextureCopy {
                        from: [x0, y0, 0],
                        to: [x1, y1, 0],
                        size: [DRAPE_SLOT_PX, DRAPE_SLOT_PX, 1],
                    }
                })
                .collect();
            gpu.submit(&[Pass::CopyTexture {
                source: old.texture,
                destination: atlas.texture,
                copies,
            }]);
        }
        self.release_atlas(gpu);
        self.atlas = Some(atlas);
        self.layout_state = layout;
        // Terrain tiles find their slots by these from this frame on.
        for (word, value) in [(params::STACKS, stacks), (params::COLUMNS, columns)] {
            gpu.write_buffer(self.params, word as u64 * 4, bytemuck::bytes_of(&value));
        }
        self.pending.clear();
        self.pending_zooms.clear();
        self.forget_group(gpu);
        Ok(Some(moves))
    }

    fn release_atlas(&mut self, gpu: &mut dyn Gpu) {
        if let Some(a) = self.atlas.take() {
            let buffers = (a.regions, a.size_uniform);
            (a.clear, a.mips, a.levels, a.texture, buffers).release(gpu);
        }
    }

    fn forget_group(&mut self, gpu: &mut dyn Gpu) {
        self.group.take().map(|(group, _)| group).release(gpu);
    }

    /// Redraws tile slots' every stack this frame, each at its drape zoom.
    pub(crate) fn redraw(&mut self, tile_slots: &[(u32, u8)]) {
        let stacks = self.layout_state.stacks;
        for &(tile, zoom) in tile_slots {
            self.pending.extend((0..stacks).map(|s| tile * stacks + s));
            self.pending_zooms.push(zoom);
        }
        self.pending_zooms.sort_unstable();
        self.pending_zooms.dedup();
    }

    /// The drape zooms of the slots this frame redraws.
    pub(crate) fn pending_zooms(&self) -> &[u8] {
        &self.pending_zooms
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
        let state = self.layout_state;
        let mut b = orb_shaders::Block::new(params::WORDS);
        b.f(params::UNPACK, &paint.unpack)
            .f(params::EXAGGERATION, &[paint.exaggeration]);
        b.f(params::DEM_DIM, &[dim as f32])
            .f(params::SKIRT, &[paint.skirt]);
        b.u(params::STACKS, &[state.stacks])
            .u(params::COLUMNS, &[state.columns]);
        b.f(params::SLOT_PX, &[DRAPE_SLOT_PX as f32])
            .u(params::TILES, &[self.tile_count]);
        gpu.write_buffer(self.params, 0, bytemuck::cast_slice(&b.0));
        self.paint = Some(paint);
    }

    /// Sizes the terrain layers' textures to the target.
    /// Returns whether they changed.
    pub(crate) fn resize(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        size: [u32; 2],
        samples: u32,
    ) -> bool {
        let layers = self.layers.len();
        if self.target_size == size
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
                width: size[0],
                height: size[1],
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
        self.pending_zooms.clear();
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
        let module = gpu.create_shader_module(&ShaderModuleDesc {
            label: shader.label,
            source: ShaderSource::Wgsl(shader.source().into()),
        });
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            label: shader.label,
            bind_group_layouts: layouts,
            vertex: ShaderStage {
                module,
                entry_point: shader.vertex,
                constants: &[],
            },
            fragment: ShaderStage {
                module,
                entry_point: shader.fragment,
                constants: &[],
            },
            targets: &[ColorTargetState {
                format: self.target.color,
                blend: None,
            }],
            depth: Some(DepthState {
                format: TextureFormat::Depth32Float,
                write: true,
                compare: CompareFunction::GreaterEqual,
            }),
            topology: PrimitiveTopology::TriangleList,
            cull_mode: None,
            sample_count: samples,
        });
        gpu.destroy_shader_module(module);
        self.pipelines[usize::from(shaded)][usize::from(samples > 1)] = Some(pipeline);
        pipeline
    }

    /// This frame's passes before the map: redrawn drape slots (`drape`
    /// draws the draped nodes into level 0), their mip levels, and each
    /// terrain layer into its texture, the first clearing `depth`. A layer
    /// shades its hillshade where `shades` has the node's group.
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        (frame, props): (BufferId, BufferId),
        (drape, shades): (&[RenderBundleId], &[Option<BindGroupId>]),
        (depth, depth_store): (TextureId, StoreOp),
        downsampler: &crate::downsample::Downsampler,
        passes: &mut Vec<Pass>,
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
        let (Some(atlas), Some(dem)) = (&self.atlas, self.dem) else {
            return 0;
        };
        if !self.pending.is_empty() {
            self.pending.sort_unstable();
            self.pending.dedup();
            let columns = self.layout_state.columns;
            let regions: Vec<[u32; 4]> = (self.pending.iter())
                .map(|&s| {
                    [
                        0,
                        s % columns * DRAPE_SLOT_PX,
                        s / columns * DRAPE_SLOT_PX,
                        DRAPE_SLOT_PX,
                    ]
                })
                .collect();
            gpu.write_buffer(atlas.regions, 0, bytemuck::cast_slice(&regions));
            let count = self.pending.len() as u32;
            let quads = |pipeline, group| {
                vec![
                    RenderCommand::SetPipeline(pipeline),
                    RenderCommand::SetBindGroup { index: 0, group },
                    RenderCommand::Draw {
                        vertices: 0..4,
                        instances: 0..count,
                    },
                ]
            };
            let mut commands = quads(self.clear, atlas.clear);
            commands.push(RenderCommand::ExecuteBundles(drape.to_vec()));
            passes.push(Pass::Render(RenderPass {
                label: "drape".into(),
                colors: vec![ColorAttachment {
                    target: atlas.levels[0],
                    resolve: None,
                    store: StoreOp::Store,
                    load: LoadOp::Load,
                }],
                depth: None,
                commands,
            }));
            // Each level from the one above, one dispatch over the slots.
            let mips = downsampler.dispatches(gpu, &atlas.mips, DRAPE_SLOT_PX, count);
            passes.push(Pass::Compute(ComputePass {
                label: "drape mips".into(),
                commands: mips.expect("ready above"),
            }));
            draws += 1;
            self.pending.clear();
            self.pending_zooms.clear();
        }
        let atlas_texture = atlas.texture;
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
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::buffer(frame),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::buffer(self.tiles),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::buffer(self.grid),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Texture {
                                texture: dem.0,
                                dimension: TextureViewDimension::D2Array,
                            },
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::Texture {
                                texture: atlas_texture,
                                dimension: TextureViewDimension::D2Array,
                            },
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::Sampler(self.linear),
                        },
                        BindGroupEntry {
                            binding: 6,
                            resource: BindingResource::buffer(self.params),
                        },
                        BindGroupEntry {
                            binding: 7,
                            resource: BindingResource::buffer(self.atmosphere),
                        },
                        BindGroupEntry {
                            binding: 8,
                            resource: BindingResource::buffer(props),
                        },
                        BindGroupEntry {
                            binding: 9,
                            resource: BindingResource::buffer(self.shade_tiles),
                        },
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
                    format: IndexFormat::Uint32,
                    offset: 0,
                },
                RenderCommand::DrawIndexed {
                    indices: 0..self.index_count,
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
                label: "terrain depth".into(),
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
            passes.push(Pass::Render(RenderPass {
                label: "terrain".into(),
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
            }));
            draws += 1;
        }
        draws
    }

    /// The drape frame at style zoom `zoom`: the map's, drawn into the
    /// atlas.
    pub(crate) fn drape_frame(
        &self,
        frame: &orb_scene::FrameBlock,
        zoom: u8,
    ) -> orb_scene::FrameBlock {
        let [w, h] = self
            .atlas
            .as_ref()
            .map_or([1.0; 2], |a| a.size.map(|s| s as f32));
        // Drape world (x, −y) for texel (x, y) → clip.
        let view_proj = [
            [2.0 / w, 0.0, 0.0, 0.0],
            [0.0, 2.0 / h, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 1.0],
        ];
        orb_scene::FrameBlock {
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
            // JS doesn't fade what it renders to texture.
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
        self.pending_zooms.clear();
        self.layout_state.tile_slots = 0;
    }

    /// The DEM page texture elevated content reads, while terrain is on.
    pub(crate) fn dem_texture(&self) -> Option<TextureId> {
        self.paint.and(self.dem).map(|(texture, _)| texture)
    }

    pub(crate) fn active(&self) -> bool {
        self.paint.is_some() && self.atlas.is_some()
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
        let module = gpu.create_shader_module(&ShaderModuleDesc {
            label: TERRAIN_SHADER.label,
            source: ShaderSource::Wgsl(TERRAIN_SHADER.source().into()),
        });
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            label: "terrain depth",
            bind_group_layouts: &[self.layout, self.stack_layout],
            vertex: ShaderStage {
                module,
                entry_point: TERRAIN_SHADER.vertex,
                constants: &[],
            },
            fragment: ShaderStage {
                module,
                entry_point: "fs_terrain_depth",
                constants: &[],
            },
            targets: &[ColorTargetState {
                format: TextureFormat::R32Float,
                blend: None,
            }],
            depth: Some(DepthState {
                format: TextureFormat::Depth32Float,
                write: true,
                compare: CompareFunction::GreaterEqual,
            }),
            topology: PrimitiveTopology::TriangleList,
            cull_mode: None,
            sample_count: 1,
        });
        gpu.destroy_shader_module(module);
        self.occlusion_pipeline = Some(pipeline);
        pipeline
    }
}

/// The shared tile mesh: `(GRID + 1)²` vertices, then a skirt copy of
/// each edge vertex; two triangles per cell and per edge segment.
fn grid() -> (Vec<u32>, Vec<u32>) {
    let n = TERRAIN_GRID;
    let side = n + 1;
    let pack = |x: u32, y: u32, skirt: u32| x | y << 8 | skirt << 16;
    let mut vertices: Vec<u32> = (0..side)
        .flat_map(|y| (0..side).map(move |x| pack(x, y, 0)))
        .collect();
    let mut indices = Vec::with_capacity((n * n * 6 + 4 * n * 6) as usize);
    let at = |x: u32, y: u32| y * side + x;
    for y in 0..n {
        for x in 0..n {
            indices.extend([
                at(x, y),
                at(x, y + 1),
                at(x + 1, y + 1),
                at(x, y),
                at(x + 1, y + 1),
                at(x + 1, y),
            ]);
        }
    }
    // Edges in order along each side: north, south, west, east.
    for edge_index in 0..4 {
        let edge = |i: u32| match edge_index {
            0 => (i, 0),
            1 => (i, n),
            2 => (0, i),
            _ => (n, i),
        };
        let base = vertices.len() as u32;
        for i in 0..side {
            let (x, y) = edge(i);
            vertices.push(pack(x, y, 1));
        }
        for i in 0..n {
            let (x0, y0) = edge(i);
            let (x1, y1) = edge(i + 1);
            let (a, b) = (at(x0, y0), at(x1, y1));
            let (sa, sb) = (base + i, base + i + 1);
            indices.extend([a, sa, sb, a, sb, b]);
        }
    }
    (vertices, indices)
}

impl crate::release::Release for Terrain {
    fn release(mut self, gpu: &mut dyn Gpu) {
        self.release_atlas(gpu);
        self.release_occlusion(gpu);
        self.forget_group(gpu);
        let layouts = (self.layout, self.stack_layout, self.clear_layout);
        let pipelines = (self.pipelines, self.clear, self.occlusion_pipeline);
        let buffers = (
            self.grid,
            self.grid_indices,
            self.tiles,
            self.params,
            self.shade_tiles,
        );
        (layouts, pipelines, buffers, self.stack_uniforms).release(gpu);
        (self.stack_groups, self.linear, self.targets).release(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_has_cells_and_skirts() {
        let (vertices, indices) = grid();
        let side = TERRAIN_GRID + 1;
        assert_eq!(vertices.len() as u32, side * side + 4 * side);
        assert_eq!(
            indices.len() as u32,
            TERRAIN_GRID * TERRAIN_GRID * 6 + 4 * TERRAIN_GRID * 6
        );
        assert!(indices.iter().all(|&i| (i as usize) < vertices.len()));
        let skirts = vertices.iter().filter(|v| *v >> 16 == 1).count() as u32;
        assert_eq!(skirts, 4 * side);
    }

    #[test]
    fn slots_fill_rows_of_the_atlas() {
        let layout = DrapeLayout {
            stacks: 2,
            columns: 8,
            tile_slots: 16,
            generation: 1,
        };
        assert_eq!(layout.corner(0, 1), [DRAPE_SLOT_PX, 0]);
        assert_eq!(layout.corner(4, 0), [0, DRAPE_SLOT_PX]);
    }
}
