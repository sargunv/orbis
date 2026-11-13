//! The atmosphere (`Atmosphere`): one sky and
//! fog model from the style's `sky`. The sky draws before the map layers;
//! terrain fogs its surface through the same uniform, and on a globe a
//! scattering pass tints the limb after them. The surface gives the
//! horizon: a line on flat surfaces, a sphere on the globe. Style values
//! are written only when they or the view change.

use crate::pipeline::{Attachments, Draw};
use orb_hal::*;
use orb_shaders::{SCATTER_SHADER, SKY_SHADER, Shader};

/// Texels of the scattering's sunward table across (the sun's angle) and
/// down (the height): `scatter.wgsl`'s `SUNWARD_SIZE`.
const SUNWARD_SIZE: [u32; 2] = [256, 32];

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
    /// The layout and group the sky and the scattering share, made for the
    /// first of them.
    group: Option<(BindGroupLayoutId, BindGroupId)>,
    /// The sky's pipeline, made for the first sky.
    sky: Option<RenderPipelineId>,
    /// The globe's scattering pipeline and its sunward table, made for the
    /// first globe sky.
    scatter: Option<RenderPipelineId>,
    sunward: Option<Sunward>,
    /// The sky's and the scattering's pipelines for the map's other kind
    /// of pass ([`swap_target`](Self::swap_target)).
    kept: [Option<RenderPipelineId>; 2],
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
            group: None,
            sky: None,
            scatter: None,
            sunward: None,
            kept: [None; 2],
            scatter_blend: 0.0,
            written: None,
            enabled: false,
        }
    }

    /// The pipelines the frame draws with, once made.
    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        [self.sky, self.scatter]
            .into_iter()
            .filter(|_| self.enabled)
            .flatten()
            .map(PipelineId::from)
            .collect()
    }

    /// Draws into the map's passes with or without their depth from now
    /// on: its pipelines for the other kind, kept since it last drew into
    /// those, take the place of these, which are kept in turn.
    pub(crate) fn swap_target(&mut self) {
        let current = [self.sky, self.scatter];
        [self.sky, self.scatter] = std::mem::replace(&mut self.kept, current);
    }

    /// Asks for the sky's pipeline unless it is made, and for the globe's
    /// scattering where the map's views may be on a low-zoom surface
    /// (`full_projection`), in the background unless the view is a globe
    /// (`globe`): the pipeline set's for a style with a sky.
    pub(crate) fn request(
        &mut self,
        gpu: &mut dyn Gpu,
        (target, full_projection): (Attachments, bool),
        frame: BufferId,
        globe: bool,
    ) {
        let uniform = self.uniform;
        let (layout, _) = *self
            .group
            .get_or_insert_with(|| sky_group(gpu, frame, uniform));
        let blend = Some(BlendState::PremultipliedAlpha);
        if self.sky.is_none() {
            self.sky = Some(pipeline(gpu, &SKY_SHADER, target, &[layout], blend));
        }
        if self.scatter.is_none() && (globe || full_projection) {
            let sunward = self.sunward.get_or_insert_with(|| Sunward::new(gpu));
            gpu.set_background_compiles(!globe);
            let layouts = [layout, sunward.layout];
            let scatter = pipeline(gpu, &SCATTER_SHADER, target, &layouts, blend);
            gpu.set_background_compiles(false);
            self.scatter = Some(scatter);
        }
    }

    /// Writes the frame's atmosphere: without a paint there is no sky and
    /// no fog. A sky the pipeline set left out asks for its pipelines.
    pub(crate) fn set(
        &mut self,
        gpu: &mut dyn Gpu,
        target: (Attachments, bool),
        frame: BufferId,
        paint: Option<AtmospherePaint>,
        view: AtmosphereView,
    ) {
        self.enabled = paint.is_some();
        if self.enabled {
            self.request(gpu, target, frame, view.globe.is_some());
        }
        if self.written == Some((paint, view)) {
            return;
        }
        use orb_shaders::layout::atmosphere as n;
        let mut b = orb_shaders::Block::new(n::WORDS);
        if let Some(p) = paint {
            b.f(n::SKY_COLOR, &p.sky_color)
                .f(n::HORIZON_COLOR, &p.horizon_color);
            b.f(n::FOG_COLOR, &p.fog_color);
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
        match (self.group, self.scatter, &self.sunward) {
            (Some((_, group)), Some(pipeline), Some(sunward))
                if self.enabled
                    && self.scatter_blend > 0.0
                    && gpu.pipeline_ready(pipeline.into()) =>
            {
                vec![
                    RenderCommand::SetPipeline(pipeline),
                    RenderCommand::SetBindGroup { index: 0, group },
                    RenderCommand::SetBindGroup {
                        index: 1,
                        group: sunward.group,
                    },
                    RenderCommand::Draw {
                        vertices: 0..3,
                        instances: 0..1,
                    },
                ]
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
    target: Attachments,
    layouts: &[BindGroupLayoutId],
    blend: Option<BlendState>,
) -> RenderPipelineId {
    Draw::new(shader, layouts, target, blend).make(gpu)
}

impl crate::release::Release for Atmosphere {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.uniform, self.group, self.sky, self.scatter, self.kept).release(gpu);
        if let Some(s) = self.sunward {
            (s.table, s.sampler, s.layout, s.group).release(gpu);
        }
    }
}

/// The scattering's sunward table (`scatter.wgsl`): the share of each
/// channel's light that reaches a point in the atmosphere from the sun, by
/// its height and the sun's angle from its zenith, as GL JS's three steps
/// toward the sun work it out. It holds no style value, so it is written
/// once, with the scattering's pipeline.
#[derive(Debug)]
struct Sunward {
    table: TextureId,
    sampler: SamplerId,
    layout: BindGroupLayoutId,
    group: BindGroupId,
}

impl Sunward {
    fn new(gpu: &mut dyn Gpu) -> Self {
        let [width, height] = SUNWARD_SIZE;
        let table = gpu.create_texture(&TextureDesc {
            label: "sunward",
            width,
            height,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            sample_count: 1,
            mip_levels: 1,
        });
        let region = TextureRegion {
            texture: table,
            x: 0,
            y: 0,
            layer: 0,
            width,
            height,
            level: 0,
        };
        gpu.write_texture(&region, bytemuck::cast_slice(&sunward_table()));
        let sampler = gpu.create_sampler(&SamplerDesc {
            label: "sunward",
            filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
        });
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "sunward",
            entries: &[
                BindGroupLayoutEntry::texture(
                    0,
                    ShaderStages::FRAGMENT,
                    true,
                    TextureViewDimension::D2,
                ),
                BindGroupLayoutEntry::sampler(1, ShaderStages::FRAGMENT, true),
            ],
        });
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "sunward",
            layout,
            entries: &[
                BindGroupEntry::texture(0, table, TextureViewDimension::D2),
                BindGroupEntry::sampler(1, sampler),
            ],
        });
        Self {
            table,
            sampler,
            layout,
            group,
        }
    }
}

/// The sunward table's texels, row by row, as RGBA half floats: at each
/// texel's height and cosine of the sun's zenith angle (`scatter.wgsl`'s
/// `scatter_sunward` lays them out, first and last texels at either end),
/// GL JS's march of three steps from there to the atmosphere's edge,
/// through the planet where the sun is below the horizon. Its constants
/// are `scatter.wgsl`'s `SCATTER_*`, and change with them.
fn sunward_table() -> Vec<u16> {
    const PLANET: f64 = 6371e3;
    const SHELL: f64 = 6471e3;
    const RAYLEIGH: [f64; 3] = [5.5e-6, 13.0e-6, 22.4e-6];
    const MIE: f64 = 21e-6;
    const RAYLEIGH_HEIGHT: f64 = 8e3;
    const MIE_HEIGHT: f64 = 1.2e3;
    let [width, height] = SUNWARD_SIZE;
    let mut texels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let row = f64::from(y) / f64::from(height - 1);
        let h = row * row * (SHELL - PLANET);
        let r = PLANET + h;
        let horizon = -(h * (2.0 * PLANET + h)).sqrt() / r;
        for x in 0..width {
            let past = (f64::from(x) / f64::from(width - 1) - 0.5) * 2.0;
            let past = past * past.abs();
            let mu = if past >= 0.0 {
                horizon + past * (1.0 - horizon)
            } else {
                horizon + past * (1.0 + horizon)
            };
            // The far side of the shell along the sun, in three steps.
            let b = r * mu;
            let step = ((b * b - r * r + SHELL * SHELL).max(0.0).sqrt() - b) / 3.0;
            let (mut rayleigh, mut mie) = (0.0, 0.0);
            for j in 0..3 {
                let t = step * (f64::from(j) + 0.5);
                let h = (r * r + t * t + 2.0 * r * t * mu).sqrt() - PLANET;
                rayleigh += (-h / RAYLEIGH_HEIGHT).exp() * step;
                mie += (-h / MIE_HEIGHT).exp() * step;
            }
            let light = RAYLEIGH.map(|k| half((-(k * rayleigh + MIE * mie)).exp() as f32));
            texels.extend(light);
            texels.push(half(1.0));
        }
    }
    texels
}

/// `x`, from 0 to 1, as the nearest half float.
fn half(x: f32) -> u16 {
    if x < 2f32.powi(-14) {
        // Below the smallest normal half: in steps of 2^-24.
        return (x * 2f32.powi(24)).round() as u16;
    }
    // Rounded to nearest, ties to even, on the 13 bits the half drops; a
    // carry moves into the exponent as it should.
    let bits = x.to_bits();
    let bits = bits + 0xfff + ((bits >> 13) & 1);
    ((bits >> 13) - ((127 - 15) << 10)) as u16
}

#[cfg(test)]
mod tests {
    use super::half;

    #[test]
    fn half_floats_round_to_nearest() {
        assert_eq!(half(0.0), 0);
        assert_eq!(half(1.0), 0x3c00);
        assert_eq!(half(0.5), 0x3800);
        // 1/3 lies nearer 0x3555 than 0x3554.
        assert_eq!(half(1.0 / 3.0), 0x3555);
        // Just under 1 rounds up into the next exponent.
        assert_eq!(half(1.0 - 2f32.powi(-13)), 0x3c00);
        // The smallest subnormal half, and a value rounding to it.
        assert_eq!(half(2f32.powi(-24)), 1);
        assert_eq!(half(0.6 * 2f32.powi(-24)), 1);
    }
}
