//! The atmosphere (`Atmosphere`): one sky and
//! fog model from the style's `sky`. The sky draws before the map layers;
//! terrain fogs its surface through the same uniform, and on a globe a
//! scattering pass tints the limb after them. The surface gives the
//! horizon: a line on flat surfaces, a sphere on the globe. Style values
//! are written only when they or the view change.

use crate::TargetFormat;
use crate::pipeline::Draw;
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
    /// style-spec#1889: the premultiplied colour behind the stars and the
    /// sky, and the stars' opacity, 0 to 1.
    pub backdrop_color: [f32; 4],
    pub star_opacity: f32,
}

/// A globe view's sphere, for its sky and scattering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobeSky {
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
    /// GL JS's fog fade with pitch, and its fade in as a globe hands
    /// over, 0 to 1.
    pub fog_opacity: f32,
    /// Clip → camera-relative space with Earth-fixed (ECEF) axes,
    /// column-major, for the view ray through a pixel: on a globe its
    /// inverse view-projection, on a flat surface turned from east, north
    /// and up at the target.
    pub ray_inverse: [[f32; 4]; 4],
    /// The globe, when the view is one.
    pub globe: Option<GlobeSky>,
}

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
            size: orb_shaders::layout::atmosphere::WORDS as u64 * 4,
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

    /// The sky's pipelines, once made.
    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.sky
            .map(|(.., p)| p.into())
            .into_iter()
            .chain(self.scatter.map(PipelineId::from))
            .collect()
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
        if let Some((layout, ..)) = self.sky
            && view.globe.is_some()
            && self.scatter.is_none()
        {
            self.scatter = Some(pipeline(gpu, &SCATTER_SHADER, target, layout));
        }
        if self.written == Some((paint, view)) {
            return;
        }
        use orb_shaders::layout::atmosphere as n;
        let mut b = orb_shaders::Block::new(n::WORDS);
        if let Some(p) = paint {
            // In linear light the colours mix decoded.
            let linear = target.color.is_srgb();
            let color = |c| crate::color_in(c, linear);
            b.f(n::SKY_COLOR, &color(p.sky_color))
                .f(n::HORIZON_COLOR, &color(p.horizon_color));
            b.f(n::FOG_COLOR, &color(p.fog_color))
                .f(n::BACKDROP_COLOR, &color(p.backdrop_color))
                .f(n::STAR_OPACITY, &[p.star_opacity])
                .f(n::LINEAR, &[f32::from(u8::from(linear))]);
            b.f(
                n::SKY_HORIZON_BLEND,
                &[p.sky_horizon_blend * view.height / 2.0],
            );
            b.f(n::HORIZON_FOG_BLEND, &[p.horizon_fog_blend]);
            b.f(n::FOG_GROUND_BLEND, &[p.fog_ground_blend])
                .f(n::FOG_OPACITY, &[view.fog_opacity]);
        }
        b.f(n::RAY_INVERSE, view.ray_inverse.as_flattened());
        b.f(n::HORIZON, &view.horizon)
            .f(n::HORIZON_NORMAL, &view.horizon_normal);
        b.f(n::FOG_NEAR, &[view.fog_range[0]])
            .f(n::FOG_FAR, &[view.fog_range[1]]);
        self.scatter_blend = 0.0;
        if let Some(g) = view.globe {
            let [x, y, z] = g.eye;
            b.f(n::GLOBE_EYE, &[x, y, z, g.radius])
                .f(n::GLOBE_SUN, &g.sun);
            b.f(n::GLOBE_AMOUNT, &[g.amount])
                .f(n::GLOBE_EYE_DISTANCE_PX, &[g.eye_distance_px]);
            // GL JS's `getAtmosphereAltitudeBlend`: the atmosphere shows
            // from orbit, fading in over nine shell heights above it.
            let shell = g.radius * 100_000.0 / 6_371_000.0;
            let altitude = g.eye.iter().map(|v| v * v).sum::<f32>().sqrt() - g.radius;
            let x = ((altitude - shell) / (9.0 * shell)).clamp(0.0, 1.0);
            let blend = paint.map_or(0.0, |p| p.atmosphere_blend) * x * x * (3.0 - 2.0 * x);
            b.f(n::SKY_ATMOSPHERE_BLEND, &[blend])
                .f(n::SCATTER_BLEND, &[blend * g.amount]);
            self.scatter_blend = blend * g.amount;
        }
        gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&b.0));
        self.written = Some((paint, view));
    }

    /// The sky's draw, once its pipeline compiled.
    pub(crate) fn commands(&self, gpu: &mut dyn Gpu) -> Vec<RenderCommand> {
        let Some((_, group, pipeline)) = self
            .sky
            .filter(|&(_, _, p)| self.enabled && gpu.pipeline_ready(p.into()))
        else {
            return Vec::new();
        };
        Self::fullscreen(pipeline, group)
    }

    /// The globe's scattering over the map, after its layers, once its
    /// pipeline compiled.
    pub(crate) fn commands_after(&self, gpu: &mut dyn Gpu) -> Vec<RenderCommand> {
        match (self.sky, self.scatter) {
            (Some((_, group, _)), Some(pipeline))
                if self.enabled
                    && self.scatter_blend > 0.0
                    && gpu.pipeline_ready(pipeline.into()) =>
            {
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
        entries: &[0, 1]
            .map(|binding| BindGroupLayoutEntry::uniform(binding, ShaderStages::FRAGMENT)),
    });
    let group = gpu.create_bind_group(&BindGroupDesc {
        label: "sky",
        layout,
        entries: &[
            BindGroupEntry::buffer(0, frame),
            BindGroupEntry::buffer(1, uniform),
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
    let blend = Some(BlendState::PremultipliedAlpha);
    Draw::new(shader, &[layout], target, blend).make(gpu)
}

impl crate::release::Release for Atmosphere {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.uniform, self.sky, self.scatter).release(gpu);
    }
}
