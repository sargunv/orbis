//! Drawing apart from the map. A layer composited once, a
//! 3D layer and a heatmap's density each draw into an offscreen
//! [`Target`]; one full-screen draw then composites it into the map
//! ([`Composites`]).
//!
//! A target is a colour attachment, with a multisampled one when it draws
//! multisampled.

use crate::release::Release;
use orb_hal::*;
use orb_shaders::{COMPOSITE_OVER_SHADER, COMPOSITE_REPLACE_SHADER};

use crate::Blend;
use crate::pipeline::{Attachments, Draw};

/// A colour attachment to draw into and sample.
#[derive(Debug)]
pub(crate) struct Target {
    color: TextureId,
    multisample: Option<TextureId>,
}

impl Target {
    /// An attachment of `format`, `size` texels, drawn at `samples`, its
    /// samples kept between passes where `kept` (else transient).
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        label: &'static str,
        format: TextureFormat,
        size: [u32; 2],
        (samples, kept): (u32, bool),
    ) -> Self {
        let texture = |gpu: &mut dyn Gpu, samples, usage| {
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
            })
        };
        let color = texture(
            gpu,
            1,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        );
        let multisample = (samples > 1).then(|| {
            let usage = match kept {
                true => TextureUsages::RENDER_ATTACHMENT,
                false => TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
            };
            texture(gpu, samples, usage)
        });
        Self { color, multisample }
    }

    /// Its texture, to sample.
    pub(crate) fn texture(&self) -> TextureId {
        self.color
    }

    /// The attachment as a pass draws it: into the multisampled texture,
    /// resolved, when there is one.
    pub(crate) fn attachment(&self, load: LoadOp<[f64; 4]>) -> ColorAttachment {
        self.attachment_kept(load, false)
    }

    /// The attachment as a pass draws it that a later pass goes on
    /// drawing, where `keep`: its samples stored, and resolved only once
    /// no pass goes on (the target made `kept`).
    pub(crate) fn attachment_kept(&self, load: LoadOp<[f64; 4]>, keep: bool) -> ColorAttachment {
        let Self { color, multisample } = *self;
        ColorAttachment {
            target: multisample.unwrap_or(color),
            resolve: multisample.filter(|_| !keep).map(|_| color),
            store: if multisample.is_some() && !keep {
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
    /// Makes `slot` a `label` target of `format` at `size` and `samples`,
    /// composited by `mode` (over `backdrop`) at `opacity`, unless it is
    /// one; whether it changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit(
        slot: &mut Option<Self>,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        (label, format): (&'static str, TextureFormat),
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
        let target = Target::new(gpu, label, format, size, (samples, false));
        let composite = composites.composite(gpu, target.texture(), backdrop, mode, opacity);
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
    /// Those for the map's other kind of pass ([`retarget`](Self::retarget)).
    kept: [Option<RenderPipelineId>; 2],
    target: Attachments,
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
    pub(crate) fn new(gpu: &mut dyn Gpu, target: Attachments) -> Self {
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
        });
        Self {
            layout,
            over: None,
            replace: None,
            kept: [None; 2],
            target,
            linear,
            blank,
        }
    }

    /// Lets the pipelines kept for the other kind of pass go: they were
    /// made for passes the map's no longer are.
    pub(crate) fn forget_kept(&mut self, gpu: &mut dyn Gpu) {
        std::mem::take(&mut self.kept).release(gpu);
    }

    /// Composites into `target`'s passes from now on, the map's with or
    /// without their depth: the pipelines for those, kept since it last
    /// composited into them, take the place of these, which are kept in
    /// turn.
    pub(crate) fn retarget(&mut self, target: Attachments) {
        let current = [self.over, self.replace];
        [self.over, self.replace] = std::mem::replace(&mut self.kept, current);
        self.target = target;
    }

    /// A blank texture: the source of a composite that leaves the map
    /// beneath it as it is.
    pub(crate) fn blank(&self) -> TextureId {
        self.blank
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
    /// or the canvas beneath. It draws once its pipeline, the pipeline
    /// set's, compiled ([`ready`](Self::ready)).
    pub(crate) fn composite(
        &mut self,
        gpu: &mut dyn Gpu,
        source: TextureId,
        aux: Option<TextureId>,
        mode: Mode,
        opacity: f32,
    ) -> Composite {
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "composite",
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        write_uniform(gpu, uniform, mode, opacity);
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
            write_uniform(gpu, composite.uniform, composite.mode, opacity);
        }
    }

    /// The pipeline of composites in `mode`, once made.
    fn pipeline(&self, mode: Mode) -> Option<RenderPipelineId> {
        match mode.replaces() {
            true => self.replace,
            false => self.over,
        }
    }

    /// Whether the pipeline of a composite in `mode` compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, mode: Mode) -> bool {
        (self.pipeline(mode)).is_some_and(|p| gpu.pipeline_ready(p.into()))
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
        let pipeline = self.pipeline(composite.mode);
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

/// Writes a composite's uniform: its opacity and `composite.wgsl`'s mode.
fn write_uniform(gpu: &mut dyn Gpu, uniform: BufferId, mode: Mode, opacity: f32) {
    gpu.write_buffer(
        uniform,
        0,
        bytemuck::cast_slice(&[opacity, mode.number(), 0.0, 0.0]),
    );
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
        (self.color, self.multisample).release(gpu);
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
            self.kept,
            self.linear,
            self.blank,
        )
            .release(gpu);
    }
}
