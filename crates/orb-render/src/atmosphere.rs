//! The atmosphere (plan §8.6's `Atmosphere`, decision 0025): one sky and
//! fog model from the style's `sky`. The sky draws before the map layers;
//! terrain fogs its surface through the same uniform. The surface gives
//! the horizon: a line on flat surfaces, a sphere on the globe once it
//! draws one. Style values are written only when they or the view change.

use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_shaders::SKY_SHADER;

/// The style's `sky`, evaluated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmospherePaint {
    /// Premultiplied style colours.
    pub sky_color: [f32; 4],
    pub horizon_color: [f32; 4],
    pub fog_color: [f32; 4],
    /// Gradient extent as a fraction of half the viewport height.
    pub sky_horizon_blend: f32,
    pub horizon_fog_blend: f32,
    pub fog_ground_blend: f32,
}

/// What a frame's view gives the atmosphere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtmosphereView {
    /// A point on a flat surface's horizon and the unit normal toward the
    /// sky, in logical pixels with y down.
    pub horizon: [f32; 2],
    pub horizon_normal: [f32; 2],
    /// Logical viewport height.
    pub height: f32,
    /// The fog's depth range along the view, in pixels.
    pub fog_range: [f32; 2],
    /// GL JS's fog fade with pitch, 0 to 1.
    pub fog_opacity: f32,
}

const WORDS: usize = 24;

#[derive(Debug)]
pub(crate) struct Atmosphere {
    pub(crate) uniform: BufferId,
    /// The sky's pipeline and bind group, made for the first sky.
    sky: Option<(RenderPipelineId, BindGroupId)>,
    written: Option<(Option<AtmospherePaint>, AtmosphereView)>,
    pub(crate) enabled: bool,
}

impl Atmosphere {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "atmosphere",
            size: WORDS as u64 * 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        Self {
            uniform,
            sky: None,
            written: None,
            enabled: false,
        }
    }

    /// Writes the frame's atmosphere: without a paint there is no sky and
    /// no fog.
    pub(crate) fn set(
        &mut self,
        gpu: &mut dyn Gpu,
        target: TargetFormat,
        frame: BufferId,
        paint: Option<AtmospherePaint>,
        view: AtmosphereView,
    ) {
        self.enabled = paint.is_some();
        if paint.is_some() && self.sky.is_none() {
            self.sky = Some(sky_pipeline(gpu, target, frame, self.uniform));
        }
        if self.written == Some((paint, view)) {
            return;
        }
        let mut words = [0.0f32; WORDS];
        if let Some(p) = paint {
            words[0..4].copy_from_slice(&p.sky_color);
            words[4..8].copy_from_slice(&p.horizon_color);
            words[8..12].copy_from_slice(&p.fog_color);
            words[16] = p.sky_horizon_blend * view.height / 2.0;
            words[17] = p.horizon_fog_blend;
            words[18] = p.fog_ground_blend;
            words[19] = view.fog_opacity;
        }
        words[12..14].copy_from_slice(&view.horizon);
        words[14..16].copy_from_slice(&view.horizon_normal);
        words[20] = view.fog_range[0];
        words[21] = view.fog_range[1];
        gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&words));
        self.written = Some((paint, view));
    }

    pub(crate) fn commands(&self) -> Vec<RenderCommand> {
        let Some((pipeline, group)) = self.sky.filter(|_| self.enabled) else {
            return Vec::new();
        };
        vec![
            RenderCommand::SetPipeline(pipeline),
            RenderCommand::SetBindGroup { index: 0, group },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ]
    }
}

fn sky_pipeline(
    gpu: &mut dyn Gpu,
    target: TargetFormat,
    frame: BufferId,
    uniform: BufferId,
) -> (RenderPipelineId, BindGroupId) {
    let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
        label: "sky",
        entries: &[0, 1].map(|binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Uniform,
        }),
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
    (pipeline, group)
}
