//! Mip levels for the drape atlas and raster pages (`downsample.wgsl`):
//! one compute dispatch per level over regions of an array texture's
//! layers, each `(layer, x, y, side)` at level 0 in a regions buffer.

use orb_hal::*;
use orb_shaders::DOWNSAMPLE_SHADER;

use crate::release::Release;

/// The most mip levels a texture has here.
const LEVELS: u64 = 16;
/// Uniform slices are 256-byte aligned on every backend.
const STRIDE: u64 = 256;

#[derive(Debug)]
pub(crate) struct Downsampler {
    layout: BindGroupLayoutId,
    /// Made for the first texture that needs it.
    pipeline: Option<ComputePipelineId>,
    /// Per level, at `STRIDE` apart: its number.
    levels: BufferId,
}

impl Downsampler {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let cs = ShaderStages::COMPUTE;
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty,
        };
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "downsample",
            entries: &[
                entry(
                    0,
                    BindingType::Texture {
                        sample: TextureSampleType::Float { filterable: false },
                        dimension: TextureViewDimension::D2Array,
                    },
                ),
                entry(
                    1,
                    BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2Array,
                    },
                ),
                entry(2, BindingType::Storage { read_only: true }),
                entry(3, BindingType::Uniform),
            ],
        });
        let levels = gpu.create_buffer(&BufferDesc {
            label: "downsample levels",
            size: LEVELS * STRIDE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        for level in 0..LEVELS {
            gpu.write_buffer(levels, level * STRIDE, bytemuck::bytes_of(&(level as u32)));
        }
        Self {
            layout,
            pipeline: None,
            levels,
        }
    }

    /// Makes the pipeline, for the first texture with mip levels.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu) {
        if self.pipeline.is_some() {
            return;
        }
        let module = crate::shader_module(gpu, &DOWNSAMPLE_SHADER);
        self.pipeline = Some(gpu.create_compute_pipeline(&ComputePipelineDesc {
            label: DOWNSAMPLE_SHADER.label,
            bind_group_layouts: &[self.layout],
            compute: ShaderStage {
                module,
                entry_point: DOWNSAMPLE_SHADER.compute,
                constants: &[],
            },
        }));
        gpu.destroy_shader_module(module);
    }

    /// Whether its pipeline compiled, or none is needed yet.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        self.pipeline.is_none_or(|p| gpu.pipeline_ready(p.into()))
    }

    pub(crate) fn pipeline(&self) -> Option<PipelineId> {
        self.pipeline.map(PipelineId::from)
    }

    /// Per level after the first, its group: reading `views[level - 1]`,
    /// writing `views[level]`, over `regions`.
    pub(crate) fn groups(
        &self,
        gpu: &mut dyn Gpu,
        views: &[TextureId],
        regions: BufferId,
    ) -> Vec<BindGroupId> {
        let view = |binding, texture| BindGroupEntry {
            binding,
            resource: BindingResource::Texture {
                texture,
                dimension: TextureViewDimension::D2Array,
            },
        };
        (1..views.len())
            .map(|level| {
                gpu.create_bind_group(&BindGroupDesc {
                    label: "downsample",
                    layout: self.layout,
                    entries: &[
                        view(0, views[level - 1]),
                        view(1, views[level]),
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::buffer(regions),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Buffer {
                                buffer: self.levels,
                                offset: level as u64 * STRIDE,
                                size: Some(16),
                            },
                        },
                    ],
                })
            })
            .collect()
    }

    /// The dispatches making the levels of `count` regions whose side at
    /// level 0 is at most `side`, once the pipeline compiled.
    pub(crate) fn dispatches(
        &self,
        gpu: &mut dyn Gpu,
        groups: &[BindGroupId],
        side: u32,
        count: u32,
    ) -> Option<Vec<ComputeCommand>> {
        let pipeline = self.pipeline.filter(|&p| gpu.pipeline_ready(p.into()))?;
        let mut commands = Vec::new();
        for (level, &group) in (1..).zip(groups) {
            let n = (side >> level).max(1).div_ceil(8);
            commands.extend([
                ComputeCommand::SetPipeline(pipeline),
                ComputeCommand::SetBindGroup { index: 0, group },
                ComputeCommand::Dispatch {
                    x: n,
                    y: n,
                    z: count,
                },
            ]);
        }
        Some(commands)
    }
}

impl Release for Downsampler {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.layout, self.pipeline, self.levels).release(gpu);
    }
}
