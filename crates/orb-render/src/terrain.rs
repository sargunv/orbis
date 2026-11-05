//! Terrain (plan §9.4, render graph `Terrain`/`Drape`): the map's 2D
//! layers draped over a DEM-raised surface, after GL JS's render to
//! texture (decision 0019).
//!
//! Each terrain tile has a block of drape slots, one per drape stack (a run
//! of draped layers between layers that draw in 3D or on screen). Slots are
//! squares of one mipmapped atlas. A frame redraws only the slots the
//! runtime marks: their drape placements, the draped nodes' only
//! candidates, map them into the atlas, so one pass over a bundle of the
//! draped nodes redraws them all, then a compute pass downsamples their
//! mip levels. Each stack's terrain draws into a layer of its own with the depth
//! extrusions share, composited in the map at the stack's place.

use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_shaders::{DRAPE_SHADER, TERRAIN_SHADER, TERRAIN_STACK_SHADER};

/// Drape texels per terrain tile side: twice the 512 logical pixels a
/// terrain tile covers at its zoom, GL JS's quality factor.
pub const DRAPE_SLOT_PX: u32 = 1024;
/// The drape atlas's format: draped layers draw into it with pipelines of
/// their own when the target's differs; mip levels are written as storage.
pub const DRAPE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
/// Grid cells per terrain tile side (`terrain.wgsl`'s `GRID`): one per
/// DEM pixel at the DEM zoom the runtime picks.
pub const TERRAIN_GRID: u32 = 128;
/// Terrain units per tile side (`terrain.wgsl`'s `TERRAIN_EXTENT`).
pub const TERRAIN_EXTENT: f32 = 8192.0;
const NO_DEM: u32 = u32::MAX;
const TILE_WORDS: usize = 8;
const PARAM_WORDS: usize = 12;
/// Uniform slices are 256-byte aligned on every backend.
const UNIFORM_STRIDE: u64 = 256;
const MAX_STACKS: u32 = 16;

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
    /// Changes whenever the atlas is made again: every slot must be
    /// redrawn.
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
    texture: TextureId,
    size: [u32; 2],
    /// Per mip level, a view to draw into.
    levels: Vec<TextureId>,
    /// Level 0's clear, then per mip level after it its downsample from
    /// the level above.
    groups: Vec<BindGroupId>,
    slots: BufferId,
    level_uniforms: BufferId,
}

#[derive(Debug, Default)]
struct StackTarget {
    color: Option<TextureId>,
    multisample: Option<TextureId>,
    group: Option<BindGroupId>,
}

#[derive(Debug)]
pub(crate) struct Terrain {
    target: TargetFormat,
    layout: BindGroupLayoutId,
    stack_layout: BindGroupLayoutId,
    clear_layout: BindGroupLayoutId,
    mip_layout: BindGroupLayoutId,
    composite_layout: BindGroupLayoutId,
    /// At one sample and at four.
    pipelines: [Option<RenderPipelineId>; 2],
    clear: RenderPipelineId,
    downsample: ComputePipelineId,
    pub(crate) composite: RenderPipelineId,
    grid: BufferId,
    grid_indices: BufferId,
    index_count: u32,
    tiles: BufferId,
    tile_capacity: u32,
    tile_count: u32,
    params: BufferId,
    stack_groups: Vec<BindGroupId>,
    linear: SamplerId,
    atlas: Option<Atlas>,
    layout_state: DrapeLayout,
    /// The DEM page texture and revision the group binds.
    dem: Option<(TextureId, u32)>,
    group: Option<BindGroupId>,
    targets: Vec<StackTarget>,
    target_size: [u32; 2],
    target_samples: u32,
    /// Atlas slots to redraw this frame.
    pending: Vec<u32>,
    pub(crate) paint: Option<TerrainPaint>,
    /// The frame block draped nodes draw with.
    pub(crate) drape_frame: BufferId,
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

fn texture(binding: u32, visibility: ShaderStages, filterable: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Texture {
            sample: TextureSampleType::Float { filterable },
            dimension: TextureViewDimension::D2,
        },
    }
}

impl Terrain {
    pub(crate) fn new(gpu: &mut dyn Gpu, target: TargetFormat, atmosphere: BufferId) -> Self {
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
                texture(4, fs, true),
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: fs,
                    ty: BindingType::Sampler { filtering: true },
                },
                uniform(6, vf),
                uniform(7, vf),
            ],
        });
        let stack_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain stack",
            entries: &[uniform(0, vs)],
        });
        let clear_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "drape clear",
            entries: &[uniform(0, vs), storage(1)],
        });
        let cs = ShaderStages::COMPUTE;
        let mip_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "drape mip",
            entries: &[
                uniform(0, cs),
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: cs,
                    ty: BindingType::Storage { read_only: true },
                },
                texture(2, cs, false),
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: cs,
                    ty: BindingType::StorageTexture {
                        format: DRAPE_FORMAT,
                        dimension: TextureViewDimension::D2,
                    },
                },
            ],
        });
        let composite_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "terrain composite",
            entries: &[texture(0, fs, false)],
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
        let downsample = gpu.create_compute_pipeline(&ComputePipelineDesc {
            label: DRAPE_SHADER.label,
            bind_group_layouts: &[mip_layout],
            compute: ShaderStage {
                module: drape_module,
                entry_point: DRAPE_SHADER.compute,
                constants: &[],
            },
        });
        let composite = draw_pipeline(
            gpu,
            &TERRAIN_STACK_SHADER,
            false,
            PrimitiveTopology::TriangleList,
            target,
            &[composite_layout],
        );

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
            size: PARAM_WORDS as u64 * 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let stack_uniforms = gpu.create_buffer(&BufferDesc {
            label: "terrain stacks",
            size: UNIFORM_STRIDE * u64::from(MAX_STACKS),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let mut words = vec![0u32; (UNIFORM_STRIDE / 4 * u64::from(MAX_STACKS)) as usize];
        for stack in 0..MAX_STACKS {
            words[(UNIFORM_STRIDE / 4 * u64::from(stack)) as usize] = stack;
        }
        gpu.write_buffer(stack_uniforms, 0, bytemuck::cast_slice(&words));
        let stack_groups = (0..MAX_STACKS)
            .map(|stack| {
                gpu.create_bind_group(&BindGroupDesc {
                    label: "terrain stack",
                    layout: stack_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: stack_uniforms,
                            offset: UNIFORM_STRIDE * u64::from(stack),
                            size: Some(16),
                        },
                    }],
                })
            })
            .collect();
        let linear = gpu.create_sampler(&SamplerDesc {
            label: "drape",
            filter: FilterMode::Linear,
            address: AddressMode::ClampToEdge,
            mipmap_filter: FilterMode::Linear,
        });
        let drape_frame = gpu.create_buffer(&BufferDesc {
            label: "drape frame",
            size: orb_scene::FRAME_BYTES,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        Self {
            target,
            layout,
            stack_layout,
            clear_layout,
            mip_layout,
            composite_layout,
            pipelines: [None; 2],
            clear,
            downsample,
            composite,
            grid: grid_buffer,
            grid_indices,
            index_count: indices.len() as u32,
            tiles: gpu.create_buffer(&BufferDesc {
                label: "terrain tiles",
                size: 64 * TILE_WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
            tile_capacity: 64,
            tile_count: 0,
            params,
            stack_groups,
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
            target_size: [0; 2],
            target_samples: 0,
            pending: Vec::new(),
            paint: None,
            drape_frame,
            atmosphere,
        }
    }

    pub(crate) fn layout(&self) -> DrapeLayout {
        self.layout_state
    }

    /// Makes room for `tile_slots` blocks of `stacks` slots. A new atlas
    /// starts a new [`DrapeLayout::generation`]. Fails when the atlas
    /// would exceed the device's texture size.
    pub(crate) fn reserve(
        &mut self,
        gpu: &mut dyn Gpu,
        tile_slots: u32,
        stacks: u32,
    ) -> Result<(), crate::OutOfSpace> {
        let stacks = stacks.clamp(1, MAX_STACKS);
        let state = self.layout_state;
        if self.atlas.is_some() && state.stacks == stacks && state.tile_slots >= tile_slots {
            return Ok(());
        }
        let max = gpu.caps().limits.max_texture_dimension_2d;
        let columns = (max / DRAPE_SLOT_PX).clamp(1, 8);
        let max_rows = max / DRAPE_SLOT_PX;
        // Grow by doubling, so a panning view rarely reallocates.
        let wanted = (tile_slots.max(state.tile_slots).max(4) * stacks).next_power_of_two();
        let rows = wanted.div_ceil(columns);
        if rows > max_rows {
            return Err(crate::OutOfSpace);
        }
        self.release_atlas(gpu);
        let size = [columns * DRAPE_SLOT_PX, rows * DRAPE_SLOT_PX];
        let levels = DRAPE_SLOT_PX.ilog2() + 1;
        let texture = gpu.create_texture(&TextureDesc {
            label: "drape atlas",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: DRAPE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            sample_count: 1,
            mip_levels: levels,
        });
        let views: Vec<TextureId> = (0..levels)
            .map(|l| gpu.create_texture_view(texture, l..l + 1))
            .collect();
        let slots = gpu.create_buffer(&BufferDesc {
            label: "drape slots",
            size: u64::from(columns * rows) * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        let level_uniforms = gpu.create_buffer(&BufferDesc {
            label: "drape levels",
            size: UNIFORM_STRIDE * u64::from(levels),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let mut groups = Vec::new();
        for level in 0..levels {
            let words: [u32; 4] = [
                ((size[0] >> level) as f32).to_bits(),
                ((size[1] >> level) as f32).to_bits(),
                ((DRAPE_SLOT_PX >> level) as f32).to_bits(),
                columns,
            ];
            gpu.write_buffer(
                level_uniforms,
                UNIFORM_STRIDE * u64::from(level),
                bytemuck::cast_slice(&words),
            );
            let mut entries = vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: level_uniforms,
                        offset: UNIFORM_STRIDE * u64::from(level),
                        size: Some(16),
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::buffer(slots),
                },
            ];
            let layout = if level == 0 {
                self.clear_layout
            } else {
                let view = |i: u32| BindingResource::Texture {
                    texture: views[i as usize],
                    dimension: TextureViewDimension::D2,
                };
                entries.extend([
                    BindGroupEntry {
                        binding: 2,
                        resource: view(level - 1),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: view(level),
                    },
                ]);
                self.mip_layout
            };
            groups.push(gpu.create_bind_group(&BindGroupDesc {
                label: "drape level",
                layout,
                entries: &entries,
            }));
        }
        self.atlas = Some(Atlas {
            texture,
            size,
            levels: views,
            groups,
            slots,
            level_uniforms,
        });
        self.layout_state = DrapeLayout {
            stacks,
            columns,
            tile_slots: columns * rows / stacks,
            generation: state.generation + 1,
        };
        self.pending.clear();
        self.forget_group(gpu);
        Ok(())
    }

    fn release_atlas(&mut self, gpu: &mut dyn Gpu) {
        if let Some(atlas) = self.atlas.take() {
            for group in atlas.groups {
                gpu.destroy_bind_group(group);
            }
            for view in atlas.levels {
                gpu.destroy_texture(view);
            }
            gpu.destroy_texture(atlas.texture);
            gpu.destroy_buffer(atlas.slots);
            gpu.destroy_buffer(atlas.level_uniforms);
        }
    }

    fn forget_group(&mut self, gpu: &mut dyn Gpu) {
        if let Some(group) = self.group.take() {
            gpu.destroy_bind_group(group);
        }
    }

    /// Redraws tile slots' every stack this frame.
    pub(crate) fn redraw(&mut self, tile_slots: &[u32]) {
        let stacks = self.layout_state.stacks;
        for &tile in tile_slots {
            self.pending.extend((0..stacks).map(|s| tile * stacks + s));
        }
    }

    pub(crate) fn set_tiles(&mut self, gpu: &mut dyn Gpu, tiles: &[TerrainTile]) {
        if tiles.len() as u32 > self.tile_capacity {
            gpu.destroy_buffer(self.tiles);
            self.tile_capacity = (tiles.len() as u32).next_power_of_two();
            self.tiles = gpu.create_buffer(&BufferDesc {
                label: "terrain tiles",
                size: u64::from(self.tile_capacity) * TILE_WORDS as u64 * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            self.forget_group(gpu);
        }
        let words: Vec<u32> = tiles
            .iter()
            .flat_map(|t| {
                let (layer, offset, scale) = match t.dem {
                    Some(d) => (d.layer, d.offset, d.scale),
                    None => (NO_DEM, [0.0; 2], 1.0),
                };
                [
                    t.origin[0].to_bits(),
                    t.origin[1].to_bits(),
                    t.scale.to_bits(),
                    layer,
                    offset[0].to_bits(),
                    offset[1].to_bits(),
                    scale.to_bits(),
                    t.slot,
                ]
            })
            .collect();
        if !words.is_empty() {
            gpu.write_buffer(self.tiles, 0, bytemuck::cast_slice(&words));
        }
        self.tile_count = tiles.len() as u32;
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
        let mut words = [0u32; PARAM_WORDS];
        let f = |v: f32| v.to_bits();
        words[..4].copy_from_slice(&paint.unpack.map(f));
        words[4] = f(paint.exaggeration);
        words[5] = f(dim as f32);
        words[6] = f(paint.skirt);
        words[7] = state.stacks;
        words[8] = state.columns;
        words[9] = f(DRAPE_SLOT_PX as f32);
        words[10] = f((DRAPE_SLOT_PX.ilog2() + 1) as f32);
        gpu.write_buffer(self.params, 0, bytemuck::cast_slice(&words));
        self.paint = Some(paint);
    }

    /// Sizes the stacks' layers to the target.
    /// Returns whether they changed.
    pub(crate) fn resize(&mut self, gpu: &mut dyn Gpu, size: [u32; 2], samples: u32) -> bool {
        let stacks = self.layout_state.stacks as usize;
        if self.target_size == size
            && self.target_samples == samples
            && self.targets.len() == stacks
        {
            return false;
        }
        for target in self.targets.drain(..) {
            release_target(gpu, target);
        }
        self.target_size = size;
        self.target_samples = samples;
        for _ in 0..stacks {
            let color = gpu.create_texture(&TextureDesc {
                label: "terrain layer",
                width: size[0],
                height: size[1],
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format: self.target.color,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_levels: 1,
            });
            let multisample = (samples > 1).then(|| {
                gpu.create_texture(&TextureDesc {
                    label: "terrain layer multisample",
                    width: size[0],
                    height: size[1],
                    array_layers: 1,
                    dimension: TextureViewDimension::D2,
                    format: self.target.color,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                    sample_count: samples,
                    mip_levels: 1,
                })
            });
            let group = gpu.create_bind_group(&BindGroupDesc {
                label: "terrain composite",
                layout: self.composite_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Texture {
                        texture: color,
                        dimension: TextureViewDimension::D2,
                    },
                }],
            });
            self.targets.push(StackTarget {
                color: Some(color),
                multisample,
                group: Some(group),
            });
        }
        true
    }

    /// Stack `stack`'s composite into the map, for the main bundle.
    pub(crate) fn composite_commands(&self, stack: u32) -> Vec<RenderCommand> {
        let Some(group) = self.targets.get(stack as usize).and_then(|t| t.group) else {
            return Vec::new();
        };
        vec![
            RenderCommand::SetPipeline(self.composite),
            RenderCommand::SetBindGroup { index: 0, group },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ]
    }

    fn pipeline(&mut self, gpu: &mut dyn Gpu, samples: u32) -> RenderPipelineId {
        let slot = usize::from(samples > 1);
        if let Some(pipeline) = self.pipelines[slot] {
            return pipeline;
        }
        let module = gpu.create_shader_module(&ShaderModuleDesc {
            label: TERRAIN_SHADER.label,
            source: ShaderSource::Wgsl(TERRAIN_SHADER.source().into()),
        });
        let pipeline = gpu.create_render_pipeline(&RenderPipelineDesc {
            label: TERRAIN_SHADER.label,
            bind_group_layouts: &[self.layout, self.stack_layout],
            vertex: ShaderStage {
                module,
                entry_point: TERRAIN_SHADER.vertex,
                constants: &[],
            },
            fragment: ShaderStage {
                module,
                entry_point: TERRAIN_SHADER.fragment,
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
        self.pipelines[slot] = Some(pipeline);
        pipeline
    }

    /// This frame's passes before the map: redrawn drape slots (`drape`
    /// draws the draped nodes into level 0), their mip levels, and each
    /// stack's terrain into its layer, the first clearing `depth`.
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: BufferId,
        drape: Option<RenderBundleId>,
        (depth, depth_store): (TextureId, StoreOp),
        passes: &mut Vec<Pass>,
    ) -> u32 {
        let mut draws = 0;
        let (Some(atlas), Some(dem)) = (&self.atlas, self.dem) else {
            return 0;
        };
        if !self.pending.is_empty() {
            self.pending.sort_unstable();
            self.pending.dedup();
            gpu.write_buffer(atlas.slots, 0, bytemuck::cast_slice(&self.pending));
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
            let mut commands = quads(self.clear, atlas.groups[0]);
            commands.extend(drape.map(|b| RenderCommand::ExecuteBundles(vec![b])));
            passes.push(Pass::Render(RenderPass {
                label: "drape".into(),
                color: ColorAttachment {
                    target: atlas.levels[0],
                    resolve: None,
                    store: StoreOp::Store,
                    load: LoadOp::Load,
                },
                depth: None,
                timestamps: None,
                commands,
            }));
            // Each level from the one above, one dispatch over the slots.
            let mut mips = Vec::new();
            for level in 1..atlas.levels.len() {
                let side = (DRAPE_SLOT_PX >> level).div_ceil(8);
                mips.extend([
                    ComputeCommand::SetPipeline(self.downsample),
                    ComputeCommand::SetBindGroup {
                        index: 0,
                        group: atlas.groups[level],
                    },
                    ComputeCommand::Dispatch {
                        x: side,
                        y: side,
                        z: count,
                    },
                ]);
            }
            passes.push(Pass::Compute(ComputePass {
                label: "drape mips".into(),
                timestamps: None,
                commands: mips,
            }));
            draws += 1;
            self.pending.clear();
        }
        let atlas_texture = atlas.texture;
        let group = match self.group {
            Some(group) => group,
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
                                dimension: TextureViewDimension::D2,
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
                    ],
                });
                self.group = Some(group);
                group
            }
        };
        let pipeline = self.pipeline(gpu, self.target_samples.max(1));
        for (stack, target) in self.targets.iter().enumerate() {
            let color = target.color.expect("sized");
            passes.push(Pass::Render(RenderPass {
                label: "terrain".into(),
                color: ColorAttachment {
                    target: target.multisample.unwrap_or(color),
                    resolve: target.multisample.map(|_| color),
                    store: if target.multisample.is_some() {
                        StoreOp::Discard
                    } else {
                        StoreOp::Store
                    },
                    load: LoadOp::Clear([0.0; 4]),
                },
                depth: Some(DepthAttachment {
                    target: depth,
                    load: if stack == 0 {
                        LoadOp::Clear(0.0)
                    } else {
                        LoadOp::Load
                    },
                    store: depth_store,
                }),
                timestamps: None,
                commands: vec![
                    RenderCommand::SetPipeline(pipeline),
                    RenderCommand::SetBindGroup { index: 0, group },
                    RenderCommand::SetBindGroup {
                        index: 1,
                        group: self.stack_groups[stack],
                    },
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
                ],
            }));
            draws += 1;
        }
        draws
    }

    /// The drape frame: the map's, drawn into the atlas.
    pub(crate) fn drape_frame(&self, frame: &orb_scene::FrameBlock) -> orb_scene::FrameBlock {
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
            viewport: [w, h],
            pixel_ratio: 1.0,
            align: [0.0; 2],
            pitch: 0.0,
            ..*frame
        }
    }

    pub(crate) fn disable(&mut self, gpu: &mut dyn Gpu) {
        self.paint = None;
        self.release_atlas(gpu);
        self.forget_group(gpu);
        for target in self.targets.drain(..) {
            release_target(gpu, target);
        }
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
        self.paint.is_some() && self.atlas.is_some()
    }
}

fn release_target(gpu: &mut dyn Gpu, target: StackTarget) {
    for texture in [target.color, target.multisample].into_iter().flatten() {
        gpu.destroy_texture(texture);
    }
    if let Some(group) = target.group {
        gpu.destroy_bind_group(group);
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
