//! The atmosphere (`Atmosphere`): one sky and
//! fog model from the style's `sky`. The sky draws before the map layers;
//! terrain fogs its surface through the same uniform, and on a globe a
//! scattering pass tints the limb after them. The surface gives the
//! horizon: a line on flat surfaces, a sphere on the globe. Style values
//! are written only when they or the view change.

use crate::TargetFormat;
use crate::pipeline::Draw;
use orb_hal::*;
use orb_shaders::{GROUND_LIGHT_SHADER, SCATTER_SHADER, SKY_SHADER, Shader};

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
    /// `light`'s shading of the ground, when it shades it.
    pub ground: Option<GroundLight>,
}

/// Ground lighting (`dev.sargunv.orbis:ground-lighting`) for a frame: what
/// multiplies the ground beneath the labels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundLight {
    /// How much it shades, in (0, 1].
    pub amount: f32,
    /// The sun's light and the ambient light, premultiplied style colours.
    pub day: [f32; 4],
    pub night: [f32; 4],
    /// The sun elevations, degrees, it passes from night to day over.
    pub twilight: f32,
    /// Toward the sun and the camera's target's up, unit, camera-relative
    /// globe axes.
    pub sun: [f32; 3],
    pub up: [f32; 3],
    /// The highest the ground can stand above a globe (terrain), as a share
    /// of its radius.
    pub relief: f32,
}

#[derive(Debug)]
pub(crate) struct Atmosphere {
    pub(crate) uniform: BufferId,
    /// The layout and group the sky, the scattering and the ground's light
    /// share, made for the first of them.
    group: Option<(BindGroupLayoutId, BindGroupId)>,
    /// The sky's pipeline, made for the first sky.
    sky: Option<RenderPipelineId>,
    /// The globe's scattering pipeline, made for the first globe sky.
    scatter: Option<RenderPipelineId>,
    scatter_blend: f32,
    /// The ground's light, made for the first that shades it.
    ground: Option<RenderPipelineId>,
    written: Option<(Option<AtmospherePaint>, AtmosphereView)>,
    pub(crate) enabled: bool,
    /// The light shades the ground this frame.
    lit: bool,
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
            group: None,
            sky: None,
            scatter: None,
            scatter_blend: 0.0,
            ground: None,
            written: None,
            enabled: false,
            lit: false,
        }
    }

    /// The pipelines the frame draws with, once made.
    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        let sky = [self.sky, self.scatter]
            .into_iter()
            .filter(|_| self.enabled);
        let ground = self.ground.filter(|_| self.lit);
        sky.chain([ground])
            .flatten()
            .map(PipelineId::from)
            .collect()
    }

    /// Whether the frame lights the ground: its light shades it and the
    /// pipeline compiled.
    pub(crate) fn lights_ground(&self, gpu: &mut dyn Gpu) -> bool {
        self.lit && self.ground.is_some_and(|p| gpu.pipeline_ready(p.into()))
    }

    /// The ground's light over the map drawn so far, when it
    /// [`lights_ground`](Self::lights_ground): what a bundle records
    /// beneath the labels.
    pub(crate) fn commands_ground(&self, gpu: &mut dyn Gpu) -> Vec<RenderCommand> {
        match (self.group, self.ground) {
            (Some((_, group)), Some(pipeline)) if self.lights_ground(gpu) => {
                Self::fullscreen(pipeline, group)
            }
            _ => Vec::new(),
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
        self.lit = view.ground.is_some();
        if (self.enabled || self.lit) && self.group.is_none() {
            self.group = Some(sky_group(gpu, frame, self.uniform));
        }
        if let Some((layout, _)) = self.group {
            let blend = Some(BlendState::PremultipliedAlpha);
            if self.enabled && self.sky.is_none() {
                self.sky = Some(pipeline(gpu, &SKY_SHADER, target, layout, blend));
            }
            if self.enabled && view.globe.is_some() && self.scatter.is_none() {
                self.scatter = Some(pipeline(gpu, &SCATTER_SHADER, target, layout, blend));
            }
            if self.lit && self.ground.is_none() {
                let multiply = Some(BlendState::Multiply);
                self.ground = Some(pipeline(
                    gpu,
                    &GROUND_LIGHT_SHADER,
                    target,
                    layout,
                    multiply,
                ));
            }
        }
        if self.written == Some((paint, view)) {
            return;
        }
        use orb_shaders::layout::atmosphere as n;
        let mut b = orb_shaders::Block::new(n::WORDS);
        // In linear light the colours mix decoded.
        let linear = target.color.is_srgb();
        b.f(n::LINEAR, &[f32::from(u8::from(linear))]);
        if let Some(p) = paint {
            let color = |c| crate::color_in(c, linear);
            b.f(n::SKY_COLOR, &color(p.sky_color))
                .f(n::HORIZON_COLOR, &color(p.horizon_color));
            b.f(n::FOG_COLOR, &color(p.fog_color))
                .f(n::BACKDROP_COLOR, &color(p.backdrop_color))
                .f(n::STAR_OPACITY, &[p.star_opacity]);
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
        if let Some(g) = view.ground {
            // In linear light whatever the space: it multiplies light.
            let [dr, dg, db, _] = crate::color_in(g.day, true);
            let [nr, ng, nb, _] = crate::color_in(g.night, true);
            // The sine of the sun elevation half the twilight away.
            let half = (g.twilight / 2.0).to_radians().sin();
            b.f(n::GROUND_DAY, &[dr, dg, db, g.amount])
                .f(n::GROUND_NIGHT, &[nr, ng, nb, half])
                .f(n::GROUND_SUN, &g.sun)
                .f(n::GROUND_UP, &[g.up[0], g.up[1], g.up[2], g.relief]);
        }
        gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&b.0));
        self.written = Some((paint, view));
    }

    /// The sky's draw, once its pipeline compiled.
    pub(crate) fn commands(&self, gpu: &mut dyn Gpu) -> Vec<RenderCommand> {
        match (self.group, self.sky) {
            (Some((_, group)), Some(pipeline))
                if self.enabled && gpu.pipeline_ready(pipeline.into()) =>
            {
                Self::fullscreen(pipeline, group)
            }
            _ => Vec::new(),
        }
    }

    /// The globe's scattering over the map, after its layers, once its
    /// pipeline compiled.
    pub(crate) fn commands_after(&self, gpu: &mut dyn Gpu) -> Vec<RenderCommand> {
        match (self.group, self.scatter) {
            (Some((_, group)), Some(pipeline))
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
    blend: Option<BlendState>,
) -> RenderPipelineId {
    Draw::new(shader, &[layout], target, blend).make(gpu)
}

impl crate::release::Release for Atmosphere {
    fn release(self, gpu: &mut dyn Gpu) {
        (
            self.uniform,
            self.group,
            self.sky,
            self.scatter,
            self.ground,
        )
            .release(gpu);
    }
}
