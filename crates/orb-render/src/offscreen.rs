//! Drawing apart from the map. A layer composited once, a
//! 3D layer, a heatmap's density and a terrain stack each draw into an
//! offscreen [`Target`]; one full-screen draw then composites it into the
//! map ([`Composites`]).
//!
//! A target is a list of colour attachments, each with a multisampled
//! attachment when it draws multisampled.

use crate::release::Release;
use orb_hal::*;
use orb_shaders::{COMPOSITE_ENCODE_SHADER, COMPOSITE_OVER_SHADER, COMPOSITE_REPLACE_SHADER};

use crate::pipeline::Draw;
use crate::{Blend, TargetFormat};

/// Colour attachments to draw into and sample.
#[derive(Debug)]
pub(crate) struct Target {
    colors: Vec<Attachment>,
}

#[derive(Debug, Clone, Copy)]
struct Attachment {
    color: TextureId,
    multisample: Option<TextureId>,
}

impl Target {
    /// Attachments of `formats`, `size` texels, drawn at `samples`.
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        label: &'static str,
        formats: &[TextureFormat],
        size: [u32; 2],
        samples: u32,
    ) -> Self {
        let texture = |gpu: &mut dyn Gpu, format, samples, usage| {
            gpu.create_texture(&TextureDesc {
                label,
                width: size[0],
                height: size[1],
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format,
                usage,
                sample_count: samples,
                mip_levels: 1,
                view_formats: &[],
            })
        };
        let colors = formats
            .iter()
            .map(|&format| Attachment {
                color: texture(
                    gpu,
                    format,
                    1,
                    TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                ),
                multisample: (samples > 1).then(|| {
                    texture(
                        gpu,
                        format,
                        samples,
                        TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                    )
                }),
            })
            .collect();
        Self { colors }
    }

    /// Attachment `i`'s texture, to sample.
    pub(crate) fn texture(&self, i: usize) -> TextureId {
        self.colors[i].color
    }

    /// Attachment `i` as a pass draws it: into the multisampled texture,
    /// resolved, when there is one.
    pub(crate) fn attachment(&self, i: usize, load: LoadOp<[f64; 4]>) -> ColorAttachment {
        let a = self.colors[i];
        ColorAttachment {
            target: a.multisample.unwrap_or(a.color),
            resolve: a.multisample.map(|_| a.color),
            store: if a.multisample.is_some() {
                StoreOp::Discard
            } else {
                StoreOp::Store
            },
            load,
        }
    }
}

/// A node drawn apart from the map: its target and composite once sized,
/// the opacity the composite applies (a 3D layer's, a heatmap's), and its
/// recorded draw.
#[derive(Debug)]
pub(crate) struct Apart {
    pub(crate) offscreen: Option<Offscreen>,
    pub(crate) opacity: f32,
    pub(crate) bundle: Option<RenderBundleId>,
}

impl Default for Apart {
    fn default() -> Self {
        Self {
            offscreen: None,
            opacity: 1.0,
            bundle: None,
        }
    }
}

impl Apart {
    /// Drops the recorded draw, so the next frame records it over the
    /// current bind groups.
    pub(crate) fn forget_bundle(&mut self, gpu: &mut dyn Gpu) {
        self.bundle.take().release(gpu);
    }

    pub(crate) fn set_opacity(&mut self, gpu: &mut dyn Gpu, opacity: f32) {
        self.opacity = opacity;
        if let Some(o) = &mut self.offscreen {
            o.set_opacity(gpu, opacity);
        }
    }
}

/// A target with its composite into the map, made again at another size
/// or sample count.
#[derive(Debug)]
pub(crate) struct Offscreen {
    pub(crate) target: Target,
    pub(crate) composite: Composite,
    made: ([u32; 2], u32),
}

impl Offscreen {
    /// Makes `slot` a `label` target of `formats` at `size` and `samples`,
    /// composited by `mode` (over `backdrop`) at `opacity`, unless it is
    /// one; whether it changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit(
        slot: &mut Option<Self>,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        (label, formats): (&'static str, &[TextureFormat]),
        (size, samples): ([u32; 2], u32),
        (mode, backdrop): (Mode, Option<TextureId>),
        opacity: f32,
    ) -> bool {
        if slot.as_ref().is_some_and(|o| o.made == (size, samples)) {
            return false;
        }
        if let Some(old) = slot.take() {
            old.release(gpu);
        }
        let target = Target::new(gpu, label, formats, size, samples);
        let composite = composites.composite(gpu, target.texture(0), backdrop, mode, opacity);
        *slot = Some(Self {
            target,
            composite,
            made: (size, samples),
        });
        true
    }

    pub(crate) fn set_opacity(&mut self, gpu: &mut dyn Gpu, opacity: f32) {
        Composites::set_opacity(gpu, &mut self.composite, opacity);
    }
}

/// How a composite combines its source with the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// The source at an opacity, by premultiplied blending.
    Over,
    /// Against the map beneath, read from a canvas, written in place.
    Blend(Blend),
    /// The source's red channel through a colour ramp.
    Ramp,
}

impl Mode {
    /// `composite.wgsl`'s mode number.
    fn number(self) -> f32 {
        match self {
            Mode::Over => 0.0,
            Mode::Blend(blend) => blend.mode(),
            Mode::Ramp => 7.0,
        }
    }

    /// Whether it writes over the map rather than blending.
    fn replaces(self) -> bool {
        matches!(self, Mode::Blend(b) if b != Blend::Normal)
    }
}

/// The composite pipelines and what their groups share.
#[derive(Debug)]
pub(crate) struct Composites {
    layout: BindGroupLayoutId,
    /// Made for the first composite that needs each.
    over: Option<RenderPipelineId>,
    replace: Option<RenderPipelineId>,
    target: TargetFormat,
    linear: SamplerId,
    /// Bound where a composite has no ramp or backdrop.
    blank: TextureId,
}

/// One composite: its uniform and bind group.
#[derive(Debug)]
pub(crate) struct Composite {
    mode: Mode,
    opacity: f32,
    uniform: BufferId,
    group: BindGroupId,
}

impl Composites {
    pub(crate) fn new(gpu: &mut dyn Gpu, target: TargetFormat) -> Self {
        use BindGroupLayoutEntry as E;
        let fs = ShaderStages::FRAGMENT;
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "composite",
            entries: &[
                E::uniform(0, fs),
                E::texture(1, fs, true, TextureViewDimension::D2),
                E::texture(2, fs, true, TextureViewDimension::D2),
                E::sampler(3, fs, true),
            ],
        });
        let linear = gpu.create_sampler(&SamplerDesc {
            label: "composite",
            filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
        });
        let blank = gpu.create_texture(&TextureDesc {
            label: "composite blank",
            width: 1,
            height: 1,
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        Self {
            layout,
            over: None,
            replace: None,
            target,
            linear,
            blank,
        }
    }

    /// Makes the pipeline of composites in `mode`, unless it is made.
    pub(crate) fn make(&mut self, gpu: &mut dyn Gpu, mode: Mode) {
        let target = self.target;
        let (slot, shader, blend) = if mode.replaces() {
            (&mut self.replace, &COMPOSITE_REPLACE_SHADER, None)
        } else {
            (
                &mut self.over,
                &COMPOSITE_OVER_SHADER,
                Some(BlendState::PremultipliedAlpha),
            )
        };
        if slot.is_none() {
            *slot = Some(Draw::new(shader, &[self.layout], target, blend).make(gpu));
        }
    }

    /// A composite of `source` in `mode` at `opacity`, with `aux` its ramp
    /// or the canvas beneath; makes the pipeline it needs.
    pub(crate) fn composite(
        &mut self,
        gpu: &mut dyn Gpu,
        source: TextureId,
        aux: Option<TextureId>,
        mode: Mode,
        opacity: f32,
    ) -> Composite {
        self.make(gpu, mode);
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "composite",
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        gpu.write_buffer(
            uniform,
            0,
            bytemuck::cast_slice(&[opacity, mode.number(), 0.0, 0.0]),
        );
        use TextureViewDimension::D2;
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "composite",
            layout: self.layout,
            entries: &[
                BindGroupEntry::buffer(0, uniform),
                BindGroupEntry::texture(1, source, D2),
                BindGroupEntry::texture(2, aux.unwrap_or(self.blank), D2),
                BindGroupEntry::sampler(3, self.linear),
            ],
        });
        Composite {
            mode,
            opacity,
            uniform,
            group,
        }
    }

    /// Writes a composite's opacity, if it changed; its draw stays
    /// recorded.
    pub(crate) fn set_opacity(gpu: &mut dyn Gpu, composite: &mut Composite, opacity: f32) {
        if composite.opacity != opacity {
            composite.opacity = opacity;
            gpu.write_buffer(
                composite.uniform,
                0,
                bytemuck::cast_slice(&[opacity, composite.mode.number(), 0.0, 0.0]),
            );
        }
    }

    /// Whether the pipeline of a composite in `mode` compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, mode: Mode) -> bool {
        let pipeline = if mode.replaces() {
            self.replace
        } else {
            self.over
        };
        pipeline.is_some_and(|p| gpu.pipeline_ready(p.into()))
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        [self.over, self.replace]
            .into_iter()
            .flatten()
            .map(PipelineId::from)
            .collect()
    }

    /// A composite's draw, for a bundle or the pass it opens.
    pub(crate) fn commands(&self, composite: &Composite) -> [RenderCommand; 3] {
        let pipeline = if composite.mode.replaces() {
            self.replace
        } else {
            self.over
        };
        [
            RenderCommand::SetPipeline(pipeline.expect("made with the composite")),
            RenderCommand::SetBindGroup {
                index: 0,
                group: composite.group,
            },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ]
    }
}

impl crate::release::Release for Apart {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.bundle, self.offscreen).release(gpu);
    }
}

impl crate::release::Release for Offscreen {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.target, self.composite).release(gpu);
    }
}

impl crate::release::Release for Target {
    fn release(self, gpu: &mut dyn Gpu) {
        for a in self.colors {
            (a.color, a.multisample).release(gpu);
        }
    }
}

impl crate::release::Release for Composite {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.group, self.uniform).release(gpu);
    }
}

impl crate::release::Release for Composites {
    fn release(self, gpu: &mut dyn Gpu) {
        (
            self.layout,
            self.over,
            self.replace,
            self.linear,
            self.blank,
        )
            .release(gpu);
    }
}

/// A linear-light map's own target: the renderer draws into an sRGB texture
/// in its format, which blends in linear light, and encodes it into the
/// host's target at the end of the frame, as hosts' targets aren't sRGB.
#[derive(Debug)]
pub(crate) struct LinearScene {
    /// The encode, drawing into the host's format.
    pipeline: RenderPipelineId,
    uniform: BufferId,
    /// The texture at its size, and the encode's group over it.
    target: Option<(TextureId, [u32; 2], BindGroupId)>,
}

impl LinearScene {
    /// The encode into targets of `host`, over `composites`' layout.
    pub(crate) fn new(gpu: &mut dyn Gpu, composites: &Composites, host: TargetFormat) -> Self {
        let pipeline =
            Draw::new(&COMPOSITE_ENCODE_SHADER, &[composites.layout], host, None).make(gpu);
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "linear scene",
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        gpu.write_buffer(uniform, 0, bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0]));
        Self {
            pipeline,
            uniform,
            target: None,
        }
    }

    /// The texture a frame of `size` texels draws into, in `composites`'
    /// format, made again at another size.
    pub(crate) fn texture(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &Composites,
        size: [u32; 2],
    ) -> TextureId {
        if let Some((texture, made, _)) = self.target
            && made == size
        {
            return texture;
        }
        if let Some((texture, _, group)) = self.target.take() {
            (group, texture).release(gpu);
        }
        let texture = gpu.create_texture(&TextureDesc {
            label: "linear scene",
            width: size[0],
            height: size[1],
            array_layers: 1,
            dimension: TextureViewDimension::D2,
            format: composites.target.color,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_levels: 1,
            view_formats: &[],
        });
        use TextureViewDimension::D2;
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "linear scene",
            layout: composites.layout,
            entries: &[
                BindGroupEntry::buffer(0, self.uniform),
                BindGroupEntry::texture(1, texture, D2),
                BindGroupEntry::texture(2, composites.blank, D2),
                BindGroupEntry::sampler(3, composites.linear),
            ],
        });
        self.target = Some((texture, size, group));
        texture
    }

    pub(crate) fn pipeline(&self) -> PipelineId {
        self.pipeline.into()
    }

    /// The pass encoding the frame's texture into `host`, once the encode
    /// compiled; until then `host` is cleared to `clear`, encoded.
    pub(crate) fn pass(&self, gpu: &mut dyn Gpu, host: TextureId, clear: [f64; 4]) -> Pass {
        let ready = gpu.pipeline_ready(self.pipeline.into());
        let commands = match self.target {
            Some((_, _, group)) if ready => vec![
                RenderCommand::SetPipeline(self.pipeline),
                RenderCommand::SetBindGroup { index: 0, group },
                RenderCommand::Draw {
                    vertices: 0..3,
                    instances: 0..1,
                },
            ],
            _ => Vec::new(),
        };
        Pass::Render(RenderPass {
            label: "encode",
            colors: vec![ColorAttachment {
                target: host,
                resolve: None,
                store: StoreOp::Store,
                load: LoadOp::Clear(clear),
            }],
            depth: None,
            commands,
        })
    }
}

impl crate::release::Release for LinearScene {
    fn release(self, gpu: &mut dyn Gpu) {
        if let Some((texture, _, group)) = self.target {
            (group, texture).release(gpu);
        }
        (self.pipeline, self.uniform).release(gpu);
    }
}
