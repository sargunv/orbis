//! Mip levels for the drape atlas and raster pages (`downsample.wgsl`):
//! one compute dispatch per level over regions of an array texture's
//! layers, each `(layer, x, y, side)` at level 0 in a regions buffer. And
//! before them, raster tiles' texels past their area taken from the
//! nearest inside (`extend`).

use orb_hal::*;
use orb_shaders::{DOWNSAMPLE_SHADER, EXTEND_SHADER};

use crate::pipeline::compute;
use crate::release::Release;

/// The most mip levels a texture has here.
const LEVELS: u64 = 16;
/// Uniform slices are 256-byte aligned on every backend.
const STRIDE: u64 = 256;

#[derive(Debug)]
pub(crate) struct Downsampler {
    layout: BindGroupLayoutId,
    /// Made for the first texture that needs it, with `extend`'s.
    pipeline: Option<ComputePipelineId>,
    extend: Option<ComputePipelineId>,
    /// Per level, at `STRIDE` apart: its number; then again with a flag
    /// that the texels are linear light, sRGB-encoded.
    levels: BufferId,
}

impl Downsampler {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let cs = ShaderStages::COMPUTE;
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "downsample",
            entries: &[
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
            ],
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
            pipeline: None,
            extend: None,
            levels,
        }
    }

    /// Makes the pipelines, for the first texture with mip levels.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu) {
        if self.pipeline.is_some() {
            return;
        }
        let layouts = [self.layout];
        self.pipeline = Some(compute(gpu, &DOWNSAMPLE_SHADER, &layouts, &[]));
        self.extend = Some(compute(gpu, &EXTEND_SHADER, &layouts, &[]));
    }

    /// Whether its pipelines compiled, or none is needed yet.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        self.pipelines().all(|p| gpu.pipeline_ready(p))
    }

    pub(crate) fn pipelines(&self) -> impl Iterator<Item = PipelineId> {
        [self.pipeline, self.extend]
            .into_iter()
            .flatten()
            .map(PipelineId::from)
    }

    /// Per level after the first, its group: reading `views[level - 1]`,
    /// writing `views[level]`, over `regions`; `linear` when the texels
    /// encode linear light, which averages decoded.
    pub(crate) fn groups(
        &self,
        gpu: &mut dyn Gpu,
        views: &[TextureId],
        regions: BufferId,
        linear: bool,
    ) -> Vec<BindGroupId> {
        let first = if linear { LEVELS } else { 0 };
        let view = |binding, texture| {
            BindGroupEntry::texture(binding, texture, TextureViewDimension::D2Array)
        };
        (1..views.len())
            .map(|level| {
                gpu.create_bind_group(&BindGroupDesc {
                    label: "downsample",
                    layout: self.layout,
                    entries: &[
                        view(0, views[level - 1]),
                        view(1, views[level]),
                        BindGroupEntry::buffer(2, regions),
                        BindGroupEntry::slice(3, self.levels, (first + level as u64) * STRIDE, 16),
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

impl Downsampler {
    /// The group for [`extend`](Self::extend): reading `level0`, a view of
    /// a texture's first level, writing `scratch`'s layers, over
    /// `regions`.
    pub(crate) fn extend_group(
        &self,
        gpu: &mut dyn Gpu,
        level0: TextureId,
        scratch: TextureId,
        regions: BufferId,
    ) -> BindGroupId {
        let view = |binding, texture| {
            BindGroupEntry::texture(binding, texture, TextureViewDimension::D2Array)
        };
        gpu.create_bind_group(&BindGroupDesc {
            label: "extend",
            layout: self.layout,
            entries: &[
                view(0, level0),
                view(1, scratch),
                BindGroupEntry::buffer(2, regions),
                BindGroupEntry::slice(3, self.levels, 0, 16),
            ],
        })
    }

    /// The dispatch extending `count` regions of `side` texels into a
    /// scratch layer each, once the pipeline compiled.
    pub(crate) fn extend(
        &self,
        gpu: &mut dyn Gpu,
        group: BindGroupId,
        side: u32,
        count: u32,
    ) -> Option<Vec<ComputeCommand>> {
        let pipeline = self.extend.filter(|&p| gpu.pipeline_ready(p.into()))?;
        let n = side.div_ceil(8);
        Some(vec![
            ComputeCommand::SetPipeline(pipeline),
            ComputeCommand::SetBindGroup { index: 0, group },
            ComputeCommand::Dispatch {
                x: n,
                y: n,
                z: count,
            },
        ])
    }
}

impl Release for Downsampler {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.layout, self.pipeline, self.extend, self.levels).release(gpu);
    }
}
