//! The atmosphere (plan §8.6's `Atmosphere`, decision 0025): one sky and
//! fog model from the style's `sky`. The sky draws before the map layers;
//! terrain fogs its surface through the same uniform, and on a globe a
//! scattering pass tints the limb after them. The surface gives the
//! horizon: a line on flat surfaces, a sphere on the globe. Style values
//! are written only when they or the view change.

use crate::{TargetFormat, draw_pipeline};
use orb_hal::*;
use orb_shaders::{SCATTER_SHADER, SKY_SHADER, Shader};

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
    /// How much the globe's atmosphere shows: 0 none, 1 GL JS's full halo.
    pub atmosphere_blend: f32,
}

/// A globe view's sphere, for its sky and scattering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobeSky {
    /// Clip → camera-relative globe space (metres), column-major.
    pub inverse_view_proj: [[f32; 4]; 4],
    /// The eye from the globe's centre, metres.
    pub eye: [f32; 3],
    pub radius: f32,
    /// How much of a globe the view is, in (0, 1].
    pub amount: f32,
    /// GL JS's camera-to-centre distance, logical pixels.
    pub eye_distance_px: f32,
    /// Toward the sun, camera-relative globe axes.
    pub sun: [f32; 3],
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
    /// The globe, when the view is one.
    pub globe: Option<GlobeSky>,
}

const WORDS: usize = 52;

#[derive(Debug)]
pub(crate) struct Atmosphere {
    pub(crate) uniform: BufferId,
    /// The sky's layout, group and pipeline, made for the first sky.
    sky: Option<(BindGroupLayoutId, BindGroupId, RenderPipelineId)>,
    /// The globe's scattering pipeline, made for the first globe sky.
    scatter: Option<RenderPipelineId>,
    scatter_blend: f32,
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
            scatter: None,
            scatter_blend: 0.0,
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
            let (layout, group) = sky_group(gpu, frame, self.uniform);
            self.sky = Some((layout, group, pipeline(gpu, &SKY_SHADER, target, layout)));
        }
        // Compat draws no scattering (plan §8.7).
        let modern = gpu.caps().tier != Tier::Compat;
        if let Some((layout, ..)) = self.sky
            && modern
            && view.globe.is_some()
            && self.scatter.is_none()
        {
            self.scatter = Some(pipeline(gpu, &SCATTER_SHADER, target, layout));
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
        self.scatter_blend = 0.0;
        if let Some(g) = view.globe {
            for (i, v) in g.inverse_view_proj.iter().flatten().enumerate() {
                words[24 + i] = *v;
            }
            words[40..43].copy_from_slice(&g.eye);
            words[43] = g.radius;
            words[44..47].copy_from_slice(&g.sun);
            words[48] = g.amount;
            words[49] = g.eye_distance_px;
            // GL JS's `getAtmosphereAltitudeBlend`: the atmosphere shows
            // from orbit, fading in over nine shell heights above it.
            let shell = g.radius * 100_000.0 / 6_371_000.0;
            let altitude = g.eye.iter().map(|v| v * v).sum::<f32>().sqrt() - g.radius;
            let x = ((altitude - shell) / (9.0 * shell)).clamp(0.0, 1.0);
            let blend = paint.map_or(0.0, |p| p.atmosphere_blend) * x * x * (3.0 - 2.0 * x);
            words[50] = blend;
            words[51] = blend * g.amount;
            self.scatter_blend = blend * g.amount;
        }
        gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&words));
        self.written = Some((paint, view));
    }

    pub(crate) fn commands(&self) -> Vec<RenderCommand> {
        let Some((_, group, pipeline)) = self.sky.filter(|_| self.enabled) else {
            return Vec::new();
        };
        Self::fullscreen(pipeline, group)
    }

    /// The globe's scattering over the map, after its layers.
    pub(crate) fn commands_after(&self) -> Vec<RenderCommand> {
        match (self.sky, self.scatter) {
            (Some((_, group, _)), Some(pipeline)) if self.enabled && self.scatter_blend > 0.0 => {
                Self::fullscreen(pipeline, group)
            }
            _ => Vec::new(),
        }
    }

    fn fullscreen(pipeline: RenderPipelineId, group: BindGroupId) -> Vec<RenderCommand> {
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

/// The sky's bind group (the frame block and the atmosphere's uniform)
/// and its layout, which the scattering shares.
fn sky_group(
    gpu: &mut dyn Gpu,
    frame: BufferId,
    uniform: BufferId,
) -> (BindGroupLayoutId, BindGroupId) {
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
    (layout, group)
}

fn pipeline(
    gpu: &mut dyn Gpu,
    shader: &Shader,
    target: TargetFormat,
    layout: BindGroupLayoutId,
) -> RenderPipelineId {
    draw_pipeline(
        gpu,
        shader,
        false,
        PrimitiveTopology::TriangleList,
        target,
        &[layout],
    )
}
