//! Mip levels for the drape atlas and raster pages (`downsample.wgsl`),
//! over regions of an array texture's layers, each `(layer, x, y, side)`
//! at level 0 in a regions table. And before them, raster tiles' texels
//! past their area taken from the nearest inside (`extend`).
//!
//! On a device with compute, one dispatch per level covers every region
//! (`downsample_compute.wgsl`); without (WebGL 2), each level of each
//! layer is a render pass drawing its regions' quads
//! (`downsample_draw.wgsl`). Both write each texel with the same body.

use orb_hal::*;
use orb_shaders::{DOWNSAMPLE_SHADER, EXTEND_DRAW_SHADER, EXTEND_SHADER, MIP_DRAW_SHADER};

use crate::pipeline::{Draw, compute};
use crate::release::Release;
use crate::scene::DataTable;

/// The most mip levels a texture has here.
const LEVELS: u64 = 16;
/// Uniform slices are 256-byte aligned on every backend.
const STRIDE: u64 = 256;

#[derive(Debug, Clone, Copy)]
enum Pipelines {
    Compute {
        mip: ComputePipelineId,
        extend: ComputePipelineId,
    },
    Drawn {
        mip: RenderPipelineId,
        extend: RenderPipelineId,
    },
}

impl Pipelines {
    fn all(self) -> [PipelineId; 2] {
        match self {
            Self::Compute { mip, extend } => [mip.into(), extend.into()],
            Self::Drawn { mip, extend } => [mip.into(), extend.into()],
        }
    }
}

#[derive(Debug)]
pub(crate) struct Downsampler {
    layout: BindGroupLayoutId,
    /// Whether the levels are drawn: the device has no compute.
    drawn: bool,
    /// Made for the first texture that needs it, with `extend`'s.
    pipelines: Option<Pipelines>,
    /// Per level, at `STRIDE` apart: its number; then again with a flag
    /// that the texels are linear light, sRGB-encoded.
    levels: BufferId,
}

/// The regions a downsampler reads: a storage buffer for its compute
/// entry points, a data texture for its drawn ones.
#[derive(Debug)]
pub(crate) enum Regions {
    Buffer(BufferId),
    Table(DataTable),
}

impl Regions {
    /// Queues `regions` into the table from its start.
    pub(crate) fn write(&self, gpu: &mut dyn Gpu, regions: &[[u32; 4]]) {
        match self {
            Self::Buffer(buffer) => gpu.write_buffer(*buffer, 0, bytemuck::cast_slice(regions)),
            Self::Table(table) => table.write(gpu, 0, bytemuck::cast_slice(regions)),
        }
    }

    /// Its binding in a group.
    pub(crate) fn entry(&self, binding: u32) -> BindGroupEntry {
        match self {
            Self::Buffer(buffer) => BindGroupEntry::buffer(binding, *buffer),
            Self::Table(table) => table.entry(binding),
        }
    }
}

impl Release for Regions {
    fn release(self, gpu: &mut dyn Gpu) {
        match self {
            Self::Buffer(buffer) => gpu.destroy_buffer(buffer),
            Self::Table(table) => table.destroy(gpu),
        }
    }
}

/// A texture's levels as the downsampler writes them: per level after the
/// first, its group (reading the level above), and the regions they
/// cover.
#[derive(Debug)]
pub(crate) struct Chain {
    texture: TextureId,
    groups: Vec<BindGroupId>,
    pub(crate) regions: Regions,
}

impl Release for Chain {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.groups, self.regions).release(gpu);
    }
}

/// What a frame's levels used, to release once it was submitted: the
/// render targets drawn levels went through, and extended tiles' scratch.
#[derive(Debug, Default)]
pub(crate) struct Transient {
    textures: Vec<TextureId>,
    regions: Vec<Regions>,
    groups: Vec<BindGroupId>,
}

impl Release for Transient {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.groups, self.regions, self.textures).release(gpu);
    }
}

impl Downsampler {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        // Only WebGL 2's build (`cpu-driven`) meets a device without compute.
        let drawn = cfg!(cpu_driven) && !gpu.caps().compute;
        let entries = if drawn {
            let (vs, vf) = (
                ShaderStages::VERTEX,
                ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            );
            let backing = crate::scene::Backing::of(gpu.caps());
            vec![
                BindGroupLayoutEntry::texture(0, vf, false, TextureViewDimension::D2Array),
                DataTable::layout_entry(backing, 2, vs),
                BindGroupLayoutEntry::uniform(3, vf),
            ]
        } else {
            let cs = ShaderStages::COMPUTE;
            vec![
                BindGroupLayoutEntry::texture(0, cs, false, TextureViewDimension::D2Array),
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: cs,
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2Array,
                    },
                },
                BindGroupLayoutEntry::storage(2, cs),
                BindGroupLayoutEntry::uniform(3, cs),
            ]
        };
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "downsample",
            entries: &entries,
        });
        let levels = gpu.create_buffer(&BufferDesc {
            label: "downsample levels",
            size: 2 * LEVELS * STRIDE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        for linear in 0..2 {
            for level in 0..LEVELS {
                gpu.write_buffer(
                    levels,
                    (linear * LEVELS + level) * STRIDE,
                    bytemuck::cast_slice(&[level as u32, linear as u32]),
                );
            }
        }
        Self {
            layout,
            drawn,
            pipelines: None,
            levels,
        }
    }

    /// Makes the pipelines, for the first texture with mip levels.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu) {
        if self.pipelines.is_some() {
            return;
        }
        let layouts = [self.layout];
        self.pipelines = Some(if self.drawn {
            let mut draw = |shader| {
                Draw {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Draw::new(
                        shader,
                        &layouts,
                        crate::TargetFormat {
                            color: TextureFormat::Rgba8Unorm,
                            sample_count: 1,
                        },
                        None,
                    )
                }
                .make(gpu)
            };
            Pipelines::Drawn {
                mip: draw(&MIP_DRAW_SHADER),
                extend: draw(&EXTEND_DRAW_SHADER),
            }
        } else {
            Pipelines::Compute {
                mip: compute(gpu, &DOWNSAMPLE_SHADER, &layouts, &[]),
                extend: compute(gpu, &EXTEND_SHADER, &layouts, &[]),
            }
        });
    }

    /// Whether its pipelines compiled, or none is needed yet.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        self.pipelines().all(|p| gpu.pipeline_ready(p))
    }

    pub(crate) fn pipelines(&self) -> impl Iterator<Item = PipelineId> {
        self.pipelines.into_iter().flat_map(Pipelines::all)
    }

    /// A table of `count` regions.
    pub(crate) fn regions(&self, gpu: &mut dyn Gpu, label: &str, count: u32) -> Regions {
        if self.drawn {
            Regions::Table(DataTable::new(gpu, label, u64::from(count) * 4))
        } else {
            Regions::Buffer(gpu.create_buffer(&BufferDesc {
                label,
                size: u64::from(count.max(1)) * 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }))
        }
    }

    /// The levels of `texture`, whose views per level are `views`, over
    /// up to `count` regions at once; `linear` when the texels encode
    /// linear light, which averages decoded.
    pub(crate) fn chain(
        &self,
        gpu: &mut dyn Gpu,
        (texture, views): (TextureId, &[TextureId]),
        (label, count): (&str, u32),
        linear: bool,
    ) -> Chain {
        let regions = self.regions(gpu, label, count);
        let first = if linear { LEVELS } else { 0 };
        let groups = (1..views.len())
            .map(|level| {
                let at = (first + level as u64) * STRIDE;
                self.group(
                    gpu,
                    "downsample",
                    views[level - 1],
                    views[level],
                    &regions,
                    at,
                )
            })
            .collect();
        Chain {
            texture,
            groups,
            regions,
        }
    }

    /// A group reading `source`, writing `destination` (compute only;
    /// drawn levels render into it) over `regions`, at level slice `at`.
    fn group(
        &self,
        gpu: &mut dyn Gpu,
        label: &str,
        source: TextureId,
        destination: TextureId,
        regions: &Regions,
        at: u64,
    ) -> BindGroupId {
        let view = |binding, texture| {
            BindGroupEntry::texture(binding, texture, TextureViewDimension::D2Array)
        };
        let mut entries = vec![view(0, source)];
        if !self.drawn {
            entries.push(view(1, destination));
        }
        entries.extend([
            regions.entry(2),
            BindGroupEntry::slice(3, self.levels, at, 16),
        ]);
        gpu.create_bind_group(&BindGroupDesc {
            label,
            layout: self.layout,
            entries: &entries,
        })
    }

    /// The passes making the levels after the first of `regions`, as
    /// written to `chain`'s table, once the pipelines compiled, and their
    /// dispatches or draws; drawn levels' render targets go into
    /// `transient`.
    pub(crate) fn mip_passes(
        &self,
        gpu: &mut dyn Gpu,
        (label, chain): (&'static str, &Chain),
        regions: &[[u32; 4]],
        transient: &mut Transient,
    ) -> Option<(Vec<Pass>, u32)> {
        let pipelines = self
            .pipelines
            .filter(|p| p.all().iter().all(|&p| gpu.pipeline_ready(p)))?;
        if regions.is_empty() {
            return Some((Vec::new(), 0));
        }
        let count = regions.len() as u32;
        match pipelines {
            Pipelines::Compute { mip, .. } => {
                let side = regions.iter().map(|r| r[3]).max().unwrap_or(0);
                let mut commands = Vec::new();
                for (level, &group) in (1..).zip(&chain.groups) {
                    let n = (side >> level).max(1).div_ceil(8);
                    commands.extend([
                        ComputeCommand::SetPipeline(mip),
                        ComputeCommand::SetBindGroup { index: 0, group },
                        ComputeCommand::Dispatch {
                            x: n,
                            y: n,
                            z: count,
                        },
                    ]);
                }
                let passes = vec![Pass::Compute(ComputePass { label, commands })];
                Some((passes, chain.groups.len() as u32))
            }
            Pipelines::Drawn { mip, .. } => {
                let mut layers: Vec<u32> = regions.iter().map(|r| r[0]).collect();
                layers.sort_unstable();
                layers.dedup();
                let mut passes = Vec::new();
                for (level, &group) in (1..).zip(&chain.groups) {
                    for &layer in &layers {
                        let target = gpu.create_layer_view(chain.texture, layer, level, None);
                        transient.textures.push(target);
                        passes.push(drawn_pass(label, mip, group, target, regions, layer));
                    }
                }
                let draws = passes.len() as u32;
                Some((passes, draws))
            }
        }
    }

    /// The passes extending raster tiles' `edges` (each its layer and the
    /// texels inside its area, `[x0, y0, x1, y1)`) in `texture`, of
    /// `side`-texel layers, whose first level `level0` views, once the
    /// pipelines compiled: drawn into scratch layers, then copied back.
    /// What they use goes into `transient`.
    pub(crate) fn extend_passes(
        &self,
        gpu: &mut dyn Gpu,
        (texture, level0, side): (TextureId, TextureId, u32),
        edges: &[(u32, [u32; 4])],
        transient: &mut Transient,
    ) -> Option<Vec<Pass>> {
        let pipelines = self
            .pipelines
            .filter(|p| p.all().iter().all(|&p| gpu.pipeline_ready(p)))?;
        let count = edges.len() as u32;
        let target = if self.drawn {
            TextureUsages::RENDER_ATTACHMENT
        } else {
            TextureUsages::STORAGE_BINDING
        };
        let scratch = gpu.create_texture(&TextureDesc {
            label: "extended tiles",
            width: side,
            height: side,
            array_layers: count,
            dimension: TextureViewDimension::D2Array,
            format: TextureFormat::Rgba8Unorm,
            usage: target | TextureUsages::COPY_SRC,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        let regions = self.regions(gpu, "extended tile areas", count);
        let words: Vec<[u32; 4]> = (edges.iter())
            .map(|&(layer, [x0, y0, x1, y1])| [layer, x0 | y0 << 16, x1 | y1 << 16, 0])
            .collect();
        regions.write(gpu, &words);
        let group = self.group(gpu, "extend", level0, scratch, &regions, 0);
        transient.textures.push(scratch);
        transient.regions.push(regions);
        transient.groups.push(group);
        let mut passes = match pipelines {
            Pipelines::Compute { extend, .. } => {
                let n = side.div_ceil(8);
                vec![Pass::Compute(ComputePass {
                    label: "image edges",
                    commands: vec![
                        ComputeCommand::SetPipeline(extend),
                        ComputeCommand::SetBindGroup { index: 0, group },
                        ComputeCommand::Dispatch {
                            x: n,
                            y: n,
                            z: count,
                        },
                    ],
                })]
            }
            Pipelines::Drawn { extend, .. } => (0..count)
                .map(|i| {
                    let target = gpu.create_layer_view(scratch, i, 0, None);
                    transient.textures.push(target);
                    let one = RenderCommand::Draw {
                        vertices: 0..4,
                        instances: i..i + 1,
                    };
                    render_pass("image edges", extend, group, target, vec![one])
                })
                .collect(),
        };
        passes.push(Pass::CopyTexture {
            source: scratch,
            destination: texture,
            copies: (edges.iter().zip(0..))
                .map(|(&(layer, _), i)| TextureCopy {
                    from: [0, 0, i],
                    to: [0, 0, layer],
                    size: [side, side, 1],
                })
                .collect(),
        });
        Some(passes)
    }
}

/// A drawn level of `layer`: a quad per region of it, into `target`.
fn drawn_pass(
    label: &'static str,
    pipeline: RenderPipelineId,
    group: BindGroupId,
    target: TextureId,
    regions: &[[u32; 4]],
    layer: u32,
) -> Pass {
    let mut draws = Vec::new();
    let mut run: Option<std::ops::Range<u32>> = None;
    for (i, region) in (0..).zip(regions) {
        if region[0] != layer {
            continue;
        }
        match &mut run {
            Some(r) if r.end == i => r.end = i + 1,
            _ => {
                draws.extend(run.take());
                run = Some(i..i + 1);
            }
        }
    }
    draws.extend(run);
    let commands = (draws.into_iter())
        .map(|instances| RenderCommand::Draw {
            vertices: 0..4,
            instances,
        })
        .collect();
    render_pass(label, pipeline, group, target, commands)
}

fn render_pass(
    label: &'static str,
    pipeline: RenderPipelineId,
    group: BindGroupId,
    target: TextureId,
    draws: Vec<RenderCommand>,
) -> Pass {
    let mut commands = vec![
        RenderCommand::SetPipeline(pipeline),
        RenderCommand::SetBindGroup { index: 0, group },
    ];
    commands.extend(draws);
    Pass::Render(RenderPass {
        label,
        colors: vec![ColorAttachment {
            target,
            resolve: None,
            store: StoreOp::Store,
            load: LoadOp::Load,
        }],
        depth: None,
        commands,
    })
}

impl Release for Downsampler {
    fn release(self, gpu: &mut dyn Gpu) {
        let pipelines = self.pipelines.map(|p| match p {
            Pipelines::Compute { mip, extend } => (Some([mip, extend]), None),
            Pipelines::Drawn { mip, extend } => (None, Some([mip, extend])),
        });
        (self.layout, pipelines, self.levels).release(gpu);
    }
}
