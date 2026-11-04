//! A fullscreen sky drawn before map layers. Camera values come from the
//! existing frame buffer; style uniforms are written only when they change.
use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_shaders::SKY_SHADER;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyPaint {
    /// Premultiplied style colours.
    pub color: [f32; 4],
    pub horizon_color: [f32; 4],
    /// Gradient extent as a fraction of half the viewport height.
    pub blend: f32,
}

#[derive(Debug)]
pub(crate) struct Sky {
    pipeline: RenderPipelineId,
    group: BindGroupId,
    uniform: BufferId,
    pub(crate) paint: Option<SkyPaint>,
    pub(crate) enabled: bool,
}

impl Sky {
    pub(crate) fn new(gpu: &mut dyn Gpu, target: TargetFormat, frame: BufferId) -> Self {
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "sky",
            entries: &[0, 1].map(|binding| BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Uniform,
            }),
        });
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "sky style",
            size: 48,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "sky",
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::buffer(frame),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::buffer(uniform),
                },
            ],
        });
        let pipeline = draw_pipeline(
            gpu,
            &SKY_SHADER,
            false,
            PrimitiveTopology::TriangleList,
            target,
            &[layout],
        );
        Self {
            pipeline,
            group,
            uniform,
            paint: None,
            enabled: false,
        }
    }

    pub(crate) fn set(&mut self, gpu: &mut dyn Gpu, paint: Option<SkyPaint>) -> bool {
        self.enabled = paint.is_some();
        let Some(paint) = paint else { return false };
        if self.paint == Some(paint) {
            return false;
        }
        let mut words = [0.0f32; 12];
        words[..4].copy_from_slice(&paint.color);
        words[4..8].copy_from_slice(&paint.horizon_color);
        words[8] = paint.blend;
        gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&words));
        self.paint = Some(paint);
        true
    }

    pub(crate) fn commands(&self) -> Vec<RenderCommand> {
        if !self.enabled {
            return Vec::new();
        }
        vec![
            RenderCommand::SetPipeline(self.pipeline),
            RenderCommand::SetBindGroup {
                index: 0,
                group: self.group,
            },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ]
    }
}
