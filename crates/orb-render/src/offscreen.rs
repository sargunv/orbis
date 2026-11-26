//! Drawing apart from the map. A layer composited once, a
//! 3D layer and a heatmap's density each draw into an offscreen
//! [`Target`]; one full-screen draw then composites it into the map
//! ([`Composites`]).
//!
//! A target is a colour attachment, with a multisampled one when it draws
//! multisampled.

use crate::release::Release;
use orb_hal::*;
use orb_shaders::{COMPOSITE_OVER_SHADER, COMPOSITE_REPLACE_SHADER, COMPOSITE_STACK_SHADER};

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
    /// The source through the coverage of a render target, a texel a
    /// logical pixel.
    Mask,
    /// A colour through the coverage of a render target.
    Solid,
    /// A render target's image, sampled to the map's pixels.
    Sampled,
}

impl Mode {
    /// `composite.wgsl`'s mode number.
    fn number(self) -> f32 {
        match self {
            Mode::Over => 0.0,
            Mode::Blend(blend) => blend.mode(),
            Mode::Ramp => 7.0,
            Mode::Mask => 8.0,
            Mode::Solid => 9.0,
            Mode::Sampled => 10.0,
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
    /// Stacks' ([`Stack`]).
    stack_layout: BindGroupLayoutId,
    /// Made for the first composite that needs each.
    over: Option<RenderPipelineId>,
    replace: Option<RenderPipelineId>,
    /// Stacks' by size ([`STACK_SIZES`]).
    stacks: [Option<RenderPipelineId>; 3],
    /// Those for the map's other kind of pass ([`retarget`](Self::retarget)).
    kept: [Option<RenderPipelineId>; 5],
    target: Attachments,
    linear: SamplerId,
    /// Bound where a composite has no ramp or backdrop.
    blank: TextureId,
}

/// One composite: its uniform and bind group.
#[derive(Debug)]
pub(crate) struct Composite {
    mode: Mode,
    /// It composites through a mask.
    masked: bool,
    /// It blends its colour rather than its source.
    colored: bool,
    /// What it samples, as one of a stack ([`Composite::layer`]): a
    /// sampled composite's source, a solid one's mask.
    sampled: Option<TextureId>,
    opacity: f32,
    /// A solid composite's colour, premultiplied.
    color: [f32; 4],
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
                E::texture(4, fs, true, TextureViewDimension::D2),
            ],
        });
        let stack_layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "composite stack",
            entries: &STACK_ENTRIES,
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
            stack_layout,
            over: None,
            replace: None,
            stacks: [None; 3],
            kept: [None; 5],
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
        let [a, b, c] = self.stacks;
        let current = [self.over, self.replace, a, b, c];
        let [over, replace, a, b, c] = std::mem::replace(&mut self.kept, current);
        (self.over, self.replace, self.stacks) = (over, replace, [a, b, c]);
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

    /// Makes the pipelines of stacks, unless they are made.
    pub(crate) fn make_stack(&mut self, gpu: &mut dyn Gpu) {
        let layouts = [self.stack_layout];
        let blend = Some(BlendState::PremultipliedAlpha);
        let draw = Draw::new(&COMPOSITE_STACK_SHADER, &layouts, self.target, blend);
        for (slot, size) in self.stacks.iter_mut().zip(STACK_SIZES) {
            slot.get_or_insert_with(|| {
                let constants: &[_] = &[(orb_shaders::STACK_LAYERS, size as f64)];
                Draw {
                    constants: [&[], constants],
                    ..draw
                }
                .make(gpu)
            });
        }
    }

    /// The pipelines of stacks, `None` for one not made.
    pub(crate) fn stack_needs(&self) -> impl Iterator<Item = Option<PipelineId>> + use<> {
        self.stacks.map(|p| p.map(Into::into)).into_iter()
    }

    /// A stack of `layers`, 2 to [`STACK`], bottom first.
    pub(crate) fn stack(&mut self, gpu: &mut dyn Gpu, layers: &[Layer]) -> Stack {
        assert!(
            (2..=STACK).contains(&layers.len()),
            "a stack of 2 to {STACK}"
        );
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "composite stack",
            size: (STACK_WORDS * 4) as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let textures: [BindGroupEntry; STACK] = std::array::from_fn(|i| {
            let texture = layers.get(i).map_or(self.blank, |l| l.texture);
            BindGroupEntry::texture(2 + i as u32, texture, TextureViewDimension::D2)
        });
        let mut entries = vec![
            BindGroupEntry::buffer(0, uniform),
            BindGroupEntry::sampler(1, self.linear),
        ];
        entries.extend(textures);
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "composite stack",
            layout: self.stack_layout,
            entries: &entries,
        });
        let stack = Stack {
            layers: layers.to_vec(),
            uniform,
            group,
        };
        stack.write(gpu);
        stack
    }

    /// Sets layer `i` of `stack` to `layer`, of the same texture, if it
    /// changed.
    pub(crate) fn set_layer(gpu: &mut dyn Gpu, stack: &mut Stack, i: usize, layer: Layer) {
        if stack.layers[i] != layer {
            debug_assert_eq!(
                stack.layers[i].texture, layer.texture,
                "a layer's texture changed"
            );
            stack.layers[i] = layer;
            stack.write(gpu);
        }
    }

    /// The pipeline of stacks of `layers`: the smallest that holds them.
    fn stack_pipeline(&self, layers: usize) -> RenderPipelineId {
        let size = STACK_SIZES.iter().position(|&s| s >= layers);
        let pipeline = size.and_then(|s| self.stacks[s]);
        pipeline.expect("made with the stack")
    }

    /// A stack's draw, for a bundle.
    pub(crate) fn stack_commands(&self, stack: &Stack) -> [RenderCommand; 3] {
        [
            RenderCommand::SetPipeline(self.stack_pipeline(stack.layers.len())),
            RenderCommand::SetBindGroup {
                index: 0,
                group: stack.group,
            },
            RenderCommand::Draw {
                vertices: 0..3,
                instances: 0..1,
            },
        ]
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
        self.composite_through(gpu, (source, aux, None), mode, opacity)
    }

    /// [`composite`](Self::composite) through the coverage of `mask`, a
    /// render target: a fill drawn through one, in `Mode::Mask` or
    /// `Mode::Solid`, or in a blend mode.
    pub(crate) fn composite_through(
        &mut self,
        gpu: &mut dyn Gpu,
        (source, aux, mask): (TextureId, Option<TextureId>, Option<TextureId>),
        mode: Mode,
        opacity: f32,
    ) -> Composite {
        let flags = (mode, mask.is_some(), false);
        let uniform = gpu.create_buffer(&BufferDesc {
            label: "composite",
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        write_uniform(gpu, uniform, flags, opacity, [0.0; 4]);
        use TextureViewDimension::D2;
        let group = gpu.create_bind_group(&BindGroupDesc {
            label: "composite",
            layout: self.layout,
            entries: &[
                BindGroupEntry::buffer(0, uniform),
                BindGroupEntry::texture(1, source, D2),
                BindGroupEntry::texture(2, aux.unwrap_or(self.blank), D2),
                BindGroupEntry::sampler(3, self.linear),
                BindGroupEntry::texture(4, mask.unwrap_or(self.blank), D2),
            ],
        });
        let sampled = match mode {
            Mode::Sampled => Some(source),
            Mode::Solid => mask,
            _ => None,
        };
        Composite {
            masked: flags.1,
            colored: false,
            sampled,
            mode,
            opacity,
            color: [0.0; 4],
            uniform,
            group,
        }
    }

    /// A composite of `color`, premultiplied, at `opacity` through the
    /// coverage of `mask`: a fill in a colour drawn through a render
    /// target, below opacity 1 or in a blend mode, then against
    /// `backdrop`. As [`Mode::Solid`] it covers every pixel the mask does,
    /// the sky included, which the fill's own draws, on the map's plane,
    /// don't.
    pub(crate) fn colour_through(
        &mut self,
        gpu: &mut dyn Gpu,
        (backdrop, mask): (Option<TextureId>, TextureId),
        blend: Blend,
        opacity: f32,
        color: [f32; 4],
    ) -> Composite {
        let mode = match blend {
            Blend::Normal => Mode::Solid,
            blend => Mode::Blend(blend),
        };
        let sources = (self.blank, backdrop, Some(mask));
        let mut composite = self.composite_through(gpu, sources, mode, opacity);
        composite.colored = true;
        composite.color = color;
        write_uniform(gpu, composite.uniform, composite.flags(), opacity, color);
        composite
    }

    /// Writes a composite's opacity, if it changed; its draw stays
    /// recorded.
    pub(crate) fn set_opacity(gpu: &mut dyn Gpu, composite: &mut Composite, opacity: f32) {
        if composite.opacity != opacity {
            composite.opacity = opacity;
            let c = &*composite;
            write_uniform(gpu, c.uniform, c.flags(), opacity, c.color);
        }
    }

    /// Writes a solid composite's colour, if it changed.
    pub(crate) fn set_color(gpu: &mut dyn Gpu, composite: &mut Composite, color: [f32; 4]) {
        if composite.color != color {
            composite.color = color;
            let c = &*composite;
            write_uniform(gpu, c.uniform, c.flags(), c.opacity, color);
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
        crate::awaits::compiled(gpu, [self.needs(mode)])
    }

    /// The pipeline of a composite in `mode`, `None` until made.
    pub(crate) fn needs(&self, mode: Mode) -> Option<PipelineId> {
        self.pipeline(mode).map(Into::into)
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        [self.over, self.replace]
            .into_iter()
            .chain(self.stacks)
            .flatten()
            .map(PipelineId::from)
            .collect()
    }

    /// The layout of composites' groups, for pipelines made apart of
    /// their shaders.
    pub(crate) fn layout(&self) -> BindGroupLayoutId {
        self.layout
    }

    /// A composite's draw, for a bundle or the pass it opens.
    pub(crate) fn commands(&self, composite: &Composite) -> [RenderCommand; 3] {
        let pipeline = self.pipeline(composite.mode);
        Self::commands_through(composite, pipeline.expect("made with the composite"))
    }

    /// A composite's draw through `pipeline`, one of its mode made for
    /// other attachments than the map's.
    pub(crate) fn commands_through(
        composite: &Composite,
        pipeline: RenderPipelineId,
    ) -> [RenderCommand; 3] {
        [
            RenderCommand::SetPipeline(pipeline),
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

impl Composite {
    /// Its mode, and whether it composites through a mask and blends its
    /// colour.
    fn flags(&self) -> (Mode, bool, bool) {
        (self.mode, self.masked, self.colored)
    }

    /// Its colour, premultiplied: a solid composite's.
    pub(crate) fn color(&self) -> [f32; 4] {
        self.color
    }

    /// What it draws as one layer of a stack: a sampled composite's
    /// source at its opacity, a solid one's colour through its mask.
    /// `None` for one that reads the map's pixels or the map beneath.
    pub(crate) fn layer(&self) -> Option<Layer> {
        let texture = self.sampled?;
        Some(match self.mode {
            Mode::Solid => Layer {
                texture,
                scale: 0.0,
                tint: self.color.map(|c| c * self.opacity),
            },
            _ => Layer {
                texture,
                scale: self.opacity,
                tint: [0.0; 4],
            },
        })
    }
}

/// The most layers a stack draws.
pub(crate) const STACK: usize = 8;

/// The sizes of stacks' pipelines, each of which samples that many
/// layers (`composite_stack.wgsl`'s `stack_layers`): a stack draws with the
/// smallest that holds it, the layers past its own blank.
const STACK_SIZES: [usize; 3] = [2, 4, STACK];

/// A stack's uniform: its count, then each layer's scale and tint.
const STACK_WORDS: usize = 4 + STACK * 8;

/// A stack's bind group layout: its uniform, the sampler, its textures.
const STACK_ENTRIES: [BindGroupLayoutEntry; 2 + STACK] = {
    let fs = ShaderStages::FRAGMENT;
    let mut entries = [BindGroupLayoutEntry::uniform(0, fs); 2 + STACK];
    entries[1] = BindGroupLayoutEntry::sampler(1, fs, true);
    let mut i = 0;
    while i < STACK {
        entries[2 + i] =
            BindGroupLayoutEntry::texture(2 + i as u32, fs, true, TextureViewDimension::D2);
        i += 1;
    }
    entries
};

/// One layer of a [`Stack`] ([`Composite::layer`]): its texture sampled
/// linearly, times `scale`, plus `tint` times the texel's coverage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Layer {
    texture: TextureId,
    scale: f32,
    tint: [f32; 4],
}

/// Composites in a row drawn as one: their layers put over one another in
/// the shader, the result over the map by premultiplied blending, as the
/// composites one by one would draw it (`composite_stack.wgsl`). Each of
/// those is a full-screen draw at the map's resolution, which a stack of
/// [`STACK`] pays once.
#[derive(Debug)]
pub(crate) struct Stack {
    layers: Vec<Layer>,
    uniform: BufferId,
    group: BindGroupId,
}

impl Stack {
    fn write(&self, gpu: &mut dyn Gpu) {
        let mut words = [0.0f32; STACK_WORDS];
        words[0] = self.layers.len() as f32;
        for (i, layer) in self.layers.iter().enumerate() {
            let at = 4 + i * 8;
            words[at] = layer.scale;
            words[at + 4..at + 8].copy_from_slice(&layer.tint);
        }
        gpu.write_buffer(self.uniform, 0, bytemuck::cast_slice(&words));
    }
}

impl crate::release::Release for Stack {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.group, self.uniform).release(gpu);
    }
}

/// Writes a composite's uniform: its opacity, `composite.wgsl`'s mode and a
/// solid or coloured one's colour.
fn write_uniform(
    gpu: &mut dyn Gpu,
    uniform: BufferId,
    (mode, masked, colored): (Mode, bool, bool),
    opacity: f32,
    color: [f32; 4],
) {
    let [r, g, b, a] = color;
    let words = [
        opacity,
        mode.number(),
        f32::from(u8::from(masked)),
        f32::from(u8::from(colored)),
        r,
        g,
        b,
        a,
    ];
    gpu.write_buffer(uniform, 0, bytemuck::cast_slice(&words));
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
            (self.layout, self.stack_layout),
            self.over,
            self.replace,
            self.stacks,
            self.kept,
            self.linear,
            self.blank,
        )
            .release(gpu);
    }
}
