//! Render targets (the `dev.sargunv.orbis:targets` preview): offscreen
//! images the size of the view, a texel a logical pixel, drawn every frame
//! before the map. Their sizes are logical pixels, and what reads them
//! (blurs, masks) is soft: a device's further pixels would cost a pass
//! their square for nothing that shows.
//!
//! A target is written by the nodes its layers route into it and by the
//! image-process layers that name it their destination, in style order.
//! One written by nodes, or by several writers, is *drawn*: a target in
//! the map's format, which a pass of its own clears and draws its writers
//! into, a process by compositing its output. One written by a single
//! process alone is that process's output, which nothing copies.
//!
//! A process runs its operations as passes over textures of the view's
//! size ([`Step`]): a kernel (a copy, one axis of a Gaussian blur, a
//! convolution), then pointwise operations; a blur too wide for its taps
//! runs on the image halved until it fits, and comes back. Its passes read
//! its source target, take turns in scratch textures shared by every
//! process (up to three of each size), and end in its output, which
//! composites into its destination target or, with none, into the map in
//! its layer's place.
//!
//! What reads a target runs after whatever writes it ([`Targets::order`]):
//! processes reading it, and the fills drawn through it, which composite
//! in the map's pass through its coverage: a colour (`Mode::Solid`), or a
//! patterned fill drawn apart first (`Mode::Mask`).

use orb_hal::*;
use orb_shaders::IMAGE_PROCESS_SHADER;

use crate::offscreen::{Composite, Composites, Mode, Target};
use crate::pipeline::{Attachments, Draw};
use crate::release::Release;
use crate::{FrameBlock, TargetFormat};

/// What a style draws through render targets, by node index and target
/// index ([`crate::Renderer::set_targets`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TargetPlan {
    /// How many targets the style declares.
    pub targets: u32,
    /// The target each node draws into rather than the map, by node.
    pub draws: Vec<Option<u32>>,
    /// The target each node (a fill) draws through, by node.
    pub masks: Vec<Option<u32>>,
    /// The image-process layers, in style order.
    pub processes: Vec<ProcessPlan>,
}

/// An image-process layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessPlan {
    /// Where it draws among the nodes, into its destination or the map:
    /// before node `at`.
    pub at: u32,
    /// The target it reads.
    pub source: u32,
    /// The target it draws into; `None` for the map.
    pub destination: Option<u32>,
    pub ops: Vec<ProcessOp>,
}

/// An image-process operation (`orb_style::Operation`) at the frame's zoom,
/// sizes in logical pixels, colours straight (not premultiplied).
#[derive(Clone, Debug, PartialEq)]
pub enum ProcessOp {
    /// `tolerance` 0–255.
    Select {
        color: [f32; 3],
        tolerance: f32,
    },
    Blur {
        size: f32,
    },
    /// `speed` logical pixels a second, 0 for still noise.
    Noise {
        size: f32,
        amount: f32,
        speed: f32,
    },
    /// `cutoff` 0–255.
    Threshold {
        cutoff: f32,
    },
    Invert,
    Sharpen {
        size: f32,
        amount: f32,
    },
    HighPass {
        size: f32,
    },
    /// A square kernel, row by row.
    Convolve {
        kernel: Vec<f32>,
    },
    /// Right and down.
    Offset {
        translate: [f32; 2],
    },
    Dilate {
        size: f32,
    },
    Erode {
        size: f32,
    },
    /// With target `target`'s image (an index past the plan's targets reads
    /// as nothing).
    Combine {
        target: u32,
        mode: CombineMode,
    },
    Opacity {
        value: f32,
    },
    Color {
        color: [f32; 4],
    },
    Luminance,
    /// Coverages 0–1 in order, up to [`MAX_STOPS`].
    Ramp {
        stops: Vec<(f32, [f32; 4])>,
    },
    /// Each channel's move, right and down, red, green then blue; scaled
    /// by the distance from the view's centre where `radial`.
    Split {
        channels: [[f32; 2]; 3],
        radial: bool,
    },
    /// Radial, or linear along `angle` degrees clockwise from top to
    /// bottom, about `center` (pixels right and down of the view's
    /// centre): 0 to 1 over `extent` pixels (a linear one's length, a
    /// radial one's radius), or by default over the view, corner to corner
    /// along the angle or to its corner farthest from the centre.
    Gradient {
        radial: bool,
        angle: f32,
        center: [f32; 2],
        extent: Option<f32>,
    },
    /// `width` 0–1.
    Bands {
        count: f32,
        width: f32,
    },
}

/// How [`ProcessOp::Combine`] puts another target's image with the one
/// processed (`orb_style::Combine`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineMode {
    Multiply,
    Subtract,
    Max,
    Min,
    Over,
    Under,
}

/// Pointwise operations one pass applies at most (`image_process.wgsl`'s
/// `ops`).
const MAX_POINTWISE: usize = 8;
/// A ramp's stops at most (`ramp_at`, `ramp_colors`).
pub const MAX_STOPS: usize = 8;
/// Linear taps a side a blur pass takes at most (`weights` holds the
/// centre and a pair for each).
const MAX_TAPS: usize = 31;
/// The widest Gaussian, standard deviation in texels, a blur pass takes
/// at its textures' size: its taps reach three of them, every texel on
/// the way. A wider blur runs on the image halved until it fits.
const MAX_SIGMA: f32 = (2 * MAX_TAPS) as f32 / 3.0;
/// Halvings a blur takes at most: past them, its taps reach no further.
const MAX_HALVINGS: u32 = 6;
/// Texels a side a dilation or erosion pass reaches at most.
const MAX_REACH: f32 = 64.0;
/// Vectors of four each pointwise operation takes in the uniform.
const OP_VECTORS: usize = 3;
/// The uniform's words (`Process` in `image_process.wgsl`).
const WORDS: usize = 4 + 4 + 64 + 4 + MAX_POINTWISE * OP_VECTORS * 4 + 12 + 8 + 32;
/// Where the uniform's pointwise operations, unprojection and ramp start.
const OPS_AT: usize = 76;
const UNPROJECT_AT: usize = OPS_AT + MAX_POINTWISE * OP_VECTORS * 4;
const RAMP_AT: usize = UNPROJECT_AT + 12;

/// What a pass samples its input through.
#[derive(Clone, Debug, PartialEq)]
enum Kernel {
    Copy,
    /// A Gaussian of standard deviation `size` along x.
    BlurX {
        size: f32,
    },
    /// The same along y, weighed against the blur's input (`combine` 1
    /// sharpen by `amount`, 2 high-pass; 0 the blur alone).
    BlurY {
        size: f32,
        combine: u32,
        amount: f32,
    },
    /// The input at half its size: each texel the mean of the four it
    /// covers, which a wide blur runs on.
    Halve,
    /// A blur run on a halved image brought back to the view's size,
    /// weighed against the blur's input as [`Kernel::BlurY`] is.
    Restore {
        combine: u32,
        amount: f32,
    },
    Convolve(Vec<f32>),
    /// The most (or least, `erode`) within `size` along one axis.
    Morph {
        size: f32,
        erode: bool,
        y: bool,
    },
    /// The input moved by `translate`.
    Offset {
        translate: [f32; 2],
    },
    /// Each channel of the input moved by its own offset.
    Split {
        channels: [[f32; 2]; 3],
        radial: bool,
    },
}

impl Kernel {
    /// What decides the pass's bindings and number: its kind, not its
    /// sizes.
    fn kind(&self) -> u32 {
        match self {
            Kernel::Copy => 0,
            Kernel::BlurX { .. } => 1,
            Kernel::BlurY { combine, .. } => 2 + combine,
            Kernel::Convolve(_) => 5,
            Kernel::Morph { .. } => 6,
            Kernel::Offset { .. } => 7,
            Kernel::Split { .. } => 8,
            Kernel::Halve => 9,
            Kernel::Restore { combine, .. } => 10 + combine,
        }
    }
}

/// A pointwise operation of a pass.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pointwise {
    Select {
        color: [f32; 3],
        tolerance: f32,
    },
    Noise {
        size: f32,
        amount: f32,
        speed: f32,
    },
    Threshold {
        cutoff: f32,
    },
    Invert,
    Gradient {
        radial: bool,
        angle: f32,
        center: [f32; 2],
        extent: Option<f32>,
    },
    Bands {
        count: f32,
        width: f32,
    },
    Opacity {
        value: f32,
    },
    Color {
        color: [f32; 4],
    },
    Luminance,
    /// The pass's ramp.
    Ramp,
    /// With the pass's other target.
    Combine {
        mode: CombineMode,
    },
}

/// One pass of a process.
#[derive(Clone, Debug, PartialEq)]
struct Step {
    kernel: Kernel,
    pointwise: Vec<Pointwise>,
    /// The ramp its `Ramp` reads, premultiplied.
    ramp: Vec<(f32, [f32; 4])>,
    /// The target its `Combine`s read.
    other: Option<u32>,
    /// How many times its output is halved from the view's size.
    halved: u32,
    /// It starts a blur: what a sharpen or high-pass weighs the blur
    /// against is its input.
    opens: bool,
}

impl Step {
    fn of(kernel: Kernel) -> Self {
        Self {
            kernel,
            pointwise: Vec::new(),
            ramp: Vec::new(),
            other: None,
            halved: 0,
            opens: false,
        }
    }

    fn halved(self, halved: u32) -> Self {
        Self { halved, ..self }
    }

    fn opening(self) -> Self {
        Self {
            opens: true,
            ..self
        }
    }

    /// Whether it reads the input of the blur it ends as well.
    fn reads_original(&self) -> bool {
        matches!(self.kernel, Kernel::BlurY { combine, .. } | Kernel::Restore { combine, .. }
            if combine != 0)
    }

    /// What its bindings and its place among the passes come of: steps of
    /// one shape differ only in their uniform.
    fn shape(&self) -> (u32, usize, Option<u32>, u32) {
        (
            self.kernel.kind(),
            self.pointwise.len(),
            self.other,
            self.halved,
        )
    }

    /// Whether `op` can join its pointwise operations: at the view's size,
    /// room left, and no second ramp or other target.
    fn takes(&self, op: &ProcessOp) -> bool {
        self.halved == 0
            && self.pointwise.len() < MAX_POINTWISE
            && match op {
                ProcessOp::Ramp { .. } => self.ramp.is_empty(),
                ProcessOp::Combine { target, .. } => self.other.is_none_or(|o| o == *target),
                _ => true,
            }
    }
}

/// Premultiplies a straight colour.
fn premultiply([r, g, b, a]: [f32; 4]) -> [f32; 4] {
    [r * a, g * a, b * a, a]
}

/// A process's operations as passes: each blur, dilation and erosion two
/// (a blur wider than [`MAX_SIGMA`] its halvings, two at the smaller size
/// and one back), pointwise operations on the pass before them, and a copy
/// where nothing else passes.
fn steps(ops: &[ProcessOp]) -> Vec<Step> {
    let mut steps: Vec<Step> = Vec::new();
    let blur = |steps: &mut Vec<Step>, size: f32, combine: u32, amount: f32| {
        if size <= 0.0 {
            return;
        }
        let halvings = halvings(size);
        if halvings == 0 {
            steps.push(Step::of(Kernel::BlurX { size }).opening());
            steps.push(Step::of(Kernel::BlurY {
                size,
                combine,
                amount,
            }));
            return;
        }
        // Halved, then blurred at that size, then brought back: what the
        // halving and the return blur by themselves is taken from the
        // blur at the smaller size, so that the whole is the Gaussian.
        for h in 1..=halvings {
            let halve = Step::of(Kernel::Halve).halved(h);
            steps.push(if h == 1 { halve.opening() } else { halve });
        }
        let size = small_sigma(size, halvings);
        let small = |kernel| Step::of(kernel).halved(halvings);
        steps.push(small(Kernel::BlurX { size }));
        steps.push(small(Kernel::BlurY {
            size,
            combine: 0,
            amount: 0.0,
        }));
        steps.push(Step::of(Kernel::Restore { combine, amount }));
    };
    for op in ops {
        let pointwise = match *op {
            ProcessOp::Select { color, tolerance } => Pointwise::Select { color, tolerance },
            ProcessOp::Noise {
                size,
                amount,
                speed,
            } => Pointwise::Noise {
                size,
                amount,
                speed,
            },
            ProcessOp::Gradient {
                radial,
                angle,
                center,
                extent,
            } => Pointwise::Gradient {
                radial,
                angle,
                center,
                extent,
            },
            ProcessOp::Bands { count, width } => Pointwise::Bands { count, width },
            ProcessOp::Threshold { cutoff } => Pointwise::Threshold { cutoff },
            ProcessOp::Invert => Pointwise::Invert,
            ProcessOp::Opacity { value } => Pointwise::Opacity { value },
            ProcessOp::Color { color } => Pointwise::Color { color },
            ProcessOp::Luminance => Pointwise::Luminance,
            ProcessOp::Ramp { .. } => Pointwise::Ramp,
            ProcessOp::Combine { mode, .. } => Pointwise::Combine { mode },
            ProcessOp::Blur { size } => {
                blur(&mut steps, size, 0, 0.0);
                continue;
            }
            ProcessOp::Sharpen { size, amount } => {
                blur(&mut steps, size, 1, amount);
                continue;
            }
            ProcessOp::HighPass { size } => {
                blur(&mut steps, size, 2, 0.0);
                continue;
            }
            ProcessOp::Convolve { ref kernel } => {
                steps.push(Step::of(Kernel::Convolve(kernel.clone())));
                continue;
            }
            ProcessOp::Dilate { size } | ProcessOp::Erode { size } => {
                let erode = matches!(op, ProcessOp::Erode { .. });
                if size > 0.0 {
                    for y in [false, true] {
                        steps.push(Step::of(Kernel::Morph { size, erode, y }));
                    }
                }
                continue;
            }
            ProcessOp::Offset { translate } => {
                steps.push(Step::of(Kernel::Offset { translate }));
                continue;
            }
            ProcessOp::Split { channels, radial } => {
                steps.push(Step::of(Kernel::Split { channels, radial }));
                continue;
            }
        };
        if !steps.last().is_some_and(|s| s.takes(op)) {
            steps.push(Step::of(Kernel::Copy));
        }
        let step = steps.last_mut().expect("pushed above");
        match op {
            ProcessOp::Ramp { stops } => {
                step.ramp = (stops.iter().take(MAX_STOPS))
                    .map(|&(at, color)| (at, premultiply(color)))
                    .collect();
            }
            ProcessOp::Combine { target, .. } => step.other = Some(*target),
            _ => {}
        }
        step.pointwise.push(pointwise);
    }
    if steps.is_empty() {
        steps.push(Step::of(Kernel::Copy));
    }
    steps
}

/// A texture a pass reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Texture {
    /// The process's source target.
    Source,
    /// A scratch texture: how many times it is halved from the view's
    /// size, and which of those.
    Scratch(u32, usize),
    /// The process's output.
    Output,
}

/// Each step's input, the original it weighs against (the input of the
/// blur it ends), and its output: scratch textures of its size in turn,
/// none a pass reads, the last step into the output.
fn wiring(steps: &[Step]) -> Vec<(Texture, Option<Texture>, Texture)> {
    let mut current = Texture::Source;
    let mut opened = Texture::Source;
    let mut wires = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        if step.opens {
            opened = current;
        }
        let original = step.reads_original().then_some(opened);
        let output = if i + 1 == steps.len() {
            Texture::Output
        } else {
            (0..3)
                .map(|s| Texture::Scratch(step.halved, s))
                .find(|t| *t != current && Some(*t) != original)
                .expect("three scratch textures suffice")
        };
        wires.push((current, original, output));
        current = output;
    }
    wires
}

/// How a target is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// Nothing writes it: it reads as transparent.
    Unused,
    /// A pass of its own draws its writers.
    Drawn,
    /// It is process `p`'s output.
    Output(usize),
}

/// What the frame runs before the map, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Run {
    /// A process's passes.
    Process(usize),
    /// A drawn target's pass.
    Draw(usize),
}

/// A process's pass as made: its uniform and bind group, the words last
/// written, and its output.
#[derive(Debug)]
struct Pass_ {
    uniform: BufferId,
    group: BindGroupId,
    written: [f32; WORDS],
    output: TextureId,
}

/// What the targets hold on the device at a size.
#[derive(Debug, Default)]
struct Made {
    size: [u32; 2],
    attachments: Option<Attachments>,
    /// By target, the drawn ones.
    drawn: Vec<Option<Target>>,
    /// By process.
    outputs: Vec<TextureId>,
    /// By how many times each is halved and its index among those.
    scratch: Vec<((u32, usize), TextureId)>,
    /// By process, by step.
    passes: Vec<Vec<Pass_>>,
    /// By process, its output's composite into its destination where that
    /// is drawn, or into the map.
    composites: Vec<Option<Composite>>,
    /// By target, a drawn one's recorded draws.
    bundles: Vec<Option<RenderBundleId>>,
    /// The drawn targets' depth, where the map's passes have one, which
    /// every pipeline they draw with declares.
    depth: Option<TextureId>,
    /// By node, a fill's composite of its colour through its target.
    solids: Vec<Option<Composite>>,
}

#[derive(Debug)]
pub(crate) struct Targets {
    plan: TargetPlan,
    roles: Vec<Role>,
    order: Vec<Run>,
    steps: Vec<Vec<Step>>,
    /// By process, whether its layer shows at the frame's zoom.
    visible: Vec<bool>,
    /// By node, a fill drawn through a target in a colour alone: that
    /// colour at its opacity, premultiplied.
    colors: Vec<Option<[f32; 4]>>,
    made: Made,
    /// The plan changed since the targets were made.
    stale: bool,
    pipeline: Option<RenderPipelineId>,
    layout: BindGroupLayoutId,
    sampler: SamplerId,
}

impl Targets {
    pub(crate) fn new(gpu: &mut dyn Gpu) -> Self {
        use BindGroupLayoutEntry as E;
        let fs = ShaderStages::FRAGMENT;
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "image process",
            entries: &[
                E::uniform(0, fs),
                E::texture(1, fs, true, TextureViewDimension::D2),
                E::texture(2, fs, true, TextureViewDimension::D2),
                E::sampler(3, fs, true),
                E::texture(4, fs, true, TextureViewDimension::D2),
            ],
        });
        let sampler = gpu.create_sampler(&SamplerDesc {
            label: "image process",
            filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            anisotropy: 1,
        });
        Self {
            plan: TargetPlan::default(),
            roles: Vec::new(),
            order: Vec::new(),
            steps: Vec::new(),
            visible: Vec::new(),
            colors: Vec::new(),
            made: Made::default(),
            stale: false,
            pipeline: None,
            layout,
            sampler,
        }
    }

    /// Whether the style draws through any target.
    pub(crate) fn any(&self) -> bool {
        self.plan.targets > 0
    }

    /// Takes `plan`, unless it is the one held; whether it changed. Makes
    /// the process pipeline for the first process.
    pub(crate) fn set_plan(&mut self, gpu: &mut dyn Gpu, plan: TargetPlan) -> bool {
        if plan == self.plan {
            return false;
        }
        let targets = plan.targets as usize;
        let mut writers = vec![(0u32, None); targets];
        for &t in plan.draws.iter().flatten() {
            writers[t as usize].0 += 1;
        }
        for (p, process) in plan.processes.iter().enumerate() {
            if let Some(t) = process.destination {
                let w = &mut writers[t as usize];
                w.0 += 1;
                w.1 = Some(p);
            }
        }
        self.roles = writers
            .iter()
            .map(|&(count, process)| match (count, process) {
                (0, _) => Role::Unused,
                (1, Some(p)) => Role::Output(p),
                _ => Role::Drawn,
            })
            .collect();
        self.steps = plan.processes.iter().map(|p| steps(&p.ops)).collect();
        self.visible = vec![true; plan.processes.len()];
        self.colors = vec![None; plan.masks.len()];
        self.plan = plan;
        self.order = self.order();
        self.stale = true;
        if !self.plan.processes.is_empty() && self.pipeline.is_none() {
            let target = TargetFormat {
                color: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            };
            let layouts = [self.layout];
            let draw = Draw::new(&IMAGE_PROCESS_SHADER, &layouts, target, None);
            self.pipeline = Some(draw.make(gpu));
        }
        true
    }

    /// The runs in an order where each comes after what it reads: a
    /// process after its source's writers, a drawn target after its
    /// processes. A cycle (which the runtime refuses) runs where it is
    /// first met.
    fn order(&self) -> Vec<Run> {
        let mut order = Vec::new();
        let mut seen = Vec::new();
        let runs = (0..self.plan.processes.len()).map(Run::Process).chain(
            (0..self.roles.len())
                .filter(|&t| self.roles[t] == Role::Drawn)
                .map(Run::Draw),
        );
        for run in runs {
            self.visit(run, &mut seen, &mut order);
        }
        order
    }

    fn visit(&self, run: Run, seen: &mut Vec<Run>, order: &mut Vec<Run>) {
        if seen.contains(&run) {
            return;
        }
        seen.push(run);
        match run {
            Run::Process(p) => {
                let others = self.steps[p].iter().filter_map(|s| s.other);
                let reads: Vec<u32> = std::iter::once(self.plan.processes[p].source)
                    .chain(others)
                    .collect();
                for t in reads {
                    if let Some(writer) = self.writer(t) {
                        self.visit(writer, seen, order);
                    }
                }
            }
            Run::Draw(t) => {
                let processes = self.plan.processes.iter().enumerate();
                for (p, _) in processes.filter(|(_, q)| q.destination == Some(t as u32)) {
                    self.visit(Run::Process(p), seen, order);
                }
            }
        }
        order.push(run);
    }

    /// The run that completes target `t`.
    fn writer(&self, t: u32) -> Option<Run> {
        match self.roles.get(t as usize)? {
            Role::Unused => None,
            Role::Drawn => Some(Run::Draw(t as usize)),
            Role::Output(p) => Some(Run::Process(*p)),
        }
    }

    /// Sets process `p`'s operations as the frame's zoom has them: passes
    /// of the same shapes take the new values in their uniforms; others
    /// are made anew with the targets.
    pub(crate) fn set_ops(&mut self, p: usize, ops: Vec<ProcessOp>) {
        let Some(process) = self.plan.processes.get_mut(p) else {
            return;
        };
        if process.ops == ops {
            return;
        }
        let steps = steps(&ops);
        let old = &self.steps[p];
        let same =
            steps.len() == old.len() && steps.iter().zip(old).all(|(a, b)| a.shape() == b.shape());
        process.ops = ops;
        self.steps[p] = steps;
        if !same {
            self.order = self.order();
            self.stale = true;
        }
    }

    /// Shows or hides a process: a hidden one's output is transparent.
    pub(crate) fn set_visible(&mut self, process: usize, visible: bool) {
        if let Some(v) = self.visible.get_mut(process) {
            *v = visible;
        }
    }

    /// Whether node `node` draws into or through a target.
    pub(crate) fn routes(&self, node: usize) -> bool {
        self.draws_into(node).is_some() || self.mask_of(node).is_some()
    }

    /// Whether node `node` draws nothing: into a target, which draws it in
    /// its own pass, or through one nothing writes.
    pub(crate) fn off_map(&self, node: usize) -> bool {
        self.draws_into(node).is_some()
            || self
                .mask_of(node)
                .is_some_and(|t| self.texture(t).is_none())
    }

    /// Sets the colour of fill `node`, drawn through a target, where it
    /// draws one alone (no pattern); whether it changed between a colour
    /// and none, which records the bundles again.
    pub(crate) fn set_color(
        &mut self,
        gpu: &mut dyn Gpu,
        node: usize,
        color: Option<[f32; 4]>,
    ) -> bool {
        let Some(slot) = self.colors.get_mut(node) else {
            return false;
        };
        let changed = slot.is_some() != color.is_some();
        *slot = color;
        if let (Some(color), Some(Some(composite))) = (color, self.made.solids.get_mut(node)) {
            Composites::set_color(gpu, composite, color);
        }
        changed
    }

    /// The composite of fill `node` in its colour through its target,
    /// where it draws a colour alone.
    pub(crate) fn solid(&self, node: usize) -> Option<&Composite> {
        self.colors.get(node)?.as_ref()?;
        self.made.solids.get(node)?.as_ref()
    }

    /// The target node `node` draws into, if any.
    pub(crate) fn draws_into(&self, node: usize) -> Option<u32> {
        self.plan.draws.get(node).copied().flatten()
    }

    /// The target node `node` draws through, if any.
    pub(crate) fn mask_of(&self, node: usize) -> Option<u32> {
        self.plan.masks.get(node).copied().flatten()
    }

    /// The texture target `t` reads as, once made: `None` where nothing
    /// writes it.
    pub(crate) fn texture(&self, t: u32) -> Option<TextureId> {
        match *self.roles.get(t as usize)? {
            Role::Unused => None,
            Role::Drawn => Some(self.made.drawn.get(t as usize)?.as_ref()?.texture()),
            Role::Output(p) => self.made.outputs.get(p).copied(),
        }
    }

    /// Whether the process pipeline compiled, where there is one.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        self.pipeline.is_none_or(|p| gpu.pipeline_ready(p.into()))
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.pipeline.into_iter().map(PipelineId::from).collect()
    }

    /// Makes the targets for frames of `size` pixels drawn into
    /// `attachments`, unless they are made for those; whether they were
    /// made anew, so that the bundles drawing into or from them record
    /// again.
    pub(crate) fn fit(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        size: [u32; 2],
        attachments: Attachments,
    ) -> bool {
        // Without targets nothing is made, at any size: only the plan that
        // dropped them lets theirs go.
        let made = &self.made;
        let fits = !self.any() || (made.size == size && made.attachments == Some(attachments));
        if !self.stale && fits {
            return false;
        }
        self.stale = false;
        std::mem::take(&mut self.made).release(gpu);
        if !self.any() {
            return true;
        }
        let color = |gpu: &mut dyn Gpu, label, halved| {
            let [width, height] = halved_size(size, halved);
            gpu.create_texture(&TextureDesc {
                label,
                width,
                height,
                array_layers: 1,
                dimension: TextureViewDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_levels: 1,
            })
        };
        let samples = (attachments.sample_count, false);
        let drawn: Vec<_> = (self.roles.iter())
            .map(|r| {
                (*r == Role::Drawn)
                    .then(|| Target::new(gpu, "render target", attachments.color, size, samples))
            })
            .collect();
        let depth = (attachments.depth)
            .filter(|_| drawn.iter().any(Option::is_some))
            .map(|format| {
                gpu.create_texture(&TextureDesc {
                    label: "render target depth",
                    width: size[0],
                    height: size[1],
                    array_layers: 1,
                    dimension: TextureViewDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TRANSIENT_ATTACHMENT,
                    sample_count: attachments.sample_count,
                    mip_levels: 1,
                })
            });
        let wirings: Vec<_> = self.steps.iter().map(|s| wiring(s)).collect();
        let mut wanted: Vec<(u32, usize)> = (wirings.iter().flatten())
            .filter_map(|&(_, _, out)| match out {
                Texture::Scratch(halved, i) => Some((halved, i)),
                _ => None,
            })
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        let scratch = (wanted.into_iter())
            .map(|(halved, i)| ((halved, i), color(gpu, "image process scratch", halved)))
            .collect();
        let outputs = (0..self.plan.processes.len())
            .map(|_| color(gpu, "image process output", 0))
            .collect();
        self.made = Made {
            size,
            attachments: Some(attachments),
            drawn,
            outputs,
            scratch,
            depth,
            ..Made::default()
        };
        let blank = composites.blank();
        for (p, wires) in wirings.iter().enumerate() {
            let source = self.texture(self.plan.processes[p].source);
            let resolve = |t: Texture| match t {
                Texture::Source => source.unwrap_or(blank),
                Texture::Scratch(halved, i) => {
                    let at = self.made.scratch.iter().find(|(k, _)| *k == (halved, i));
                    at.expect("made above").1
                }
                Texture::Output => self.made.outputs[p],
            };
            let passes = (wires.iter().zip(&self.steps[p]))
                .map(|(&(input, original, output), step)| {
                    let other = step.other.and_then(|t| self.texture(t)).unwrap_or(blank);
                    let uniform = gpu.create_buffer(&BufferDesc {
                        label: "image process",
                        size: (WORDS * 4) as u64,
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    });
                    let group = gpu.create_bind_group(&BindGroupDesc {
                        label: "image process",
                        layout: self.layout,
                        entries: &[
                            BindGroupEntry::buffer(0, uniform),
                            BindGroupEntry::texture(1, resolve(input), TextureViewDimension::D2),
                            BindGroupEntry::texture(
                                2,
                                original.map_or(blank, resolve),
                                TextureViewDimension::D2,
                            ),
                            BindGroupEntry::sampler(3, self.sampler),
                            BindGroupEntry::texture(4, other, TextureViewDimension::D2),
                        ],
                    });
                    Pass_ {
                        uniform,
                        group,
                        written: [f32::NAN; WORDS],
                        output: resolve(output),
                    }
                })
                .collect();
            self.made.passes.push(passes);
        }
        let drawn_out = (0..self.plan.processes.len())
            .map(|p| {
                let to_map = match self.plan.processes[p].destination {
                    None => true,
                    Some(t) => self.roles[t as usize] == Role::Drawn,
                };
                let output = self.made.outputs[p];
                to_map.then(|| composites.composite(gpu, output, None, Mode::Sampled, 1.0))
            })
            .collect();
        self.made.composites = drawn_out;
        let solids = (0..self.plan.masks.len())
            .map(|node| {
                let mask = self.texture(self.plan.masks[node]?)?;
                let sources = (blank, None, Some(mask));
                let mut solid = composites.composite_through(gpu, sources, Mode::Solid, 1.0);
                if let Some(color) = self.colors[node] {
                    Composites::set_color(gpu, &mut solid, color);
                }
                Some(solid)
            })
            .collect();
        self.made.solids = solids;
        self.made.bundles = vec![None; self.roles.len()];
        true
    }

    /// The draws of a process's output into the map, or into its drawn
    /// destination: `None` where it has none.
    pub(crate) fn composite(&self, process: usize) -> Option<&Composite> {
        self.made.composites.get(process)?.as_ref()
    }

    /// The processes that draw into the map before node `at`.
    pub(crate) fn drawing_on_map_at(&self, at: usize) -> impl Iterator<Item = usize> + '_ {
        (self.plan.processes.iter().enumerate())
            .filter(move |(_, p)| p.destination.is_none() && p.at as usize == at)
            .map(|(i, _)| i)
    }

    /// What drawn target `t` draws, in style order: its nodes, and its
    /// processes' outputs before node `at`.
    pub(crate) fn writers(&self, t: usize) -> Vec<Writer> {
        let mut writers: Vec<(usize, u8, Writer)> = (self.plan.draws.iter().enumerate())
            .filter(|(_, d)| **d == Some(t as u32))
            .map(|(n, _)| (n, 1, Writer::Node(n)))
            .collect();
        for (p, process) in self.plan.processes.iter().enumerate() {
            if process.destination == Some(t as u32) {
                writers.push((process.at as usize, 0, Writer::Process(p)));
            }
        }
        writers.sort_by_key(|&(at, kind, _)| (at, kind));
        writers.into_iter().map(|(_, _, w)| w).collect()
    }

    /// The drawn targets, by index.
    pub(crate) fn drawn(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.roles.len()).filter(|&t| self.roles[t] == Role::Drawn)
    }

    /// Replaces drawn target `t`'s recorded draws.
    pub(crate) fn set_bundle(&mut self, gpu: &mut dyn Gpu, t: usize, bundle: RenderBundleId) {
        if let Some(slot) = self.made.bundles.get_mut(t) {
            slot.replace(bundle).release(gpu);
        } else {
            bundle.release(gpu);
        }
    }

    /// The frame's passes before the map: each process's, where its
    /// pipeline compiled, and each drawn target's. Returns the draws.
    pub(crate) fn passes(
        &mut self,
        gpu: &mut dyn Gpu,
        frame: &FrameBlock,
        out: &mut Vec<orb_hal::Pass>,
    ) -> u32 {
        let depth = self.made.depth;
        let ready = self.ready(gpu);
        let mut draws = 0;
        for run in self.order.clone() {
            match run {
                Run::Draw(t) => {
                    let Some(target) = &self.made.drawn[t] else {
                        continue;
                    };
                    let bundle = self.made.bundles[t];
                    out.push(orb_hal::Pass::Render(RenderPass {
                        label: "render target",
                        colors: vec![target.attachment(LoadOp::Clear([0.0; 4]))],
                        depth: depth.map(|target| DepthAttachment {
                            target,
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Discard,
                        }),
                        commands: bundle
                            .map(|b| vec![RenderCommand::ExecuteBundles(vec![b])])
                            .unwrap_or_default(),
                    }));
                    draws += u32::from(bundle.is_some());
                }
                Run::Process(p) => {
                    let shown = self.visible[p] && ready;
                    if !shown {
                        out.push(clear(self.made.outputs[p]));
                        continue;
                    }
                    let pipeline = self.pipeline.expect("made with the first process");
                    for (s, step) in self.steps[p].iter().enumerate() {
                        let words = uniform(step, frame, self.made.size);
                        let pass = &mut self.made.passes[p][s];
                        if words != pass.written {
                            pass.written = words;
                            gpu.write_buffer(pass.uniform, 0, bytemuck::cast_slice(&words));
                        }
                        out.push(orb_hal::Pass::Render(RenderPass {
                            label: "image process",
                            commands: vec![
                                RenderCommand::SetPipeline(pipeline),
                                RenderCommand::SetBindGroup {
                                    index: 0,
                                    group: pass.group,
                                },
                                RenderCommand::Draw {
                                    vertices: 0..3,
                                    instances: 0..1,
                                },
                            ],
                            ..clear_pass(pass.output)
                        }));
                        draws += 1;
                    }
                }
            }
        }
        draws
    }
}

/// What reads a drawn target writes into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Writer {
    Node(usize),
    Process(usize),
}

fn clear_pass(texture: TextureId) -> RenderPass {
    RenderPass {
        label: "image process clear",
        colors: vec![ColorAttachment {
            target: texture,
            resolve: None,
            store: StoreOp::Store,
            load: LoadOp::Clear([0.0; 4]),
        }],
        depth: None,
        commands: Vec::new(),
    }
}

fn clear(texture: TextureId) -> orb_hal::Pass {
    orb_hal::Pass::Render(clear_pass(texture))
}

/// The size of a texture of the view's `size` halved `halved` times.
fn halved_size(size: [u32; 2], halved: u32) -> [u32; 2] {
    size.map(|s| s.div_ceil(1 << halved).max(1))
}

/// How many times a blur of standard deviation `sigma` texels halves the
/// image first, for the Gaussian to fit its taps at the smaller size:
/// none for one that fits at the view's.
fn halvings(sigma: f32) -> u32 {
    (0..MAX_HALVINGS)
        .find(|&h| sigma <= MAX_SIGMA * (1 << h) as f32)
        .unwrap_or(MAX_HALVINGS)
}

/// The standard deviation, in texels of the image halved `halvings`
/// times, of the blur there that with the halvings and the return makes a
/// Gaussian of `sigma` texels at the view's size. Each halving means two
/// texels, as a box of that many at its size (variance (4ʰ - 1) / 12 over
/// them all); the return interpolates between texels `2ʰ` apart, as a tent
/// that wide (variance 4ʰ / 6), on average over where pixels fall.
fn small_sigma(sigma: f32, halvings: u32) -> f32 {
    let scale = (1u32 << halvings) as f32;
    let added = (scale * scale - 1.0) / 12.0 + scale * scale / 6.0;
    (sigma * sigma - added).max(0.0).sqrt() / scale
}

/// A Gaussian of standard deviation `sigma` texels as linear taps: the
/// centre weight, then each tap's (offset, weight), offsets in texels,
/// cut off where the taps reach no further ([`MAX_SIGMA`]).
fn gaussian(sigma: f32) -> (f32, Vec<[f32; 2]>) {
    let radius = (sigma * 3.0).ceil().max(1.0);
    let n = (radius as usize).min(2 * MAX_TAPS);
    let w = |i: usize| {
        let x = i as f32;
        (-x * x / (2.0 * sigma * sigma)).exp()
    };
    let total = w(0) + 2.0 * (1..=n).map(w).sum::<f32>();
    let taps = (1..=n)
        .step_by(2)
        .map(|i| {
            let (a, b) = (w(i), if i < n { w(i + 1) } else { 0.0 });
            // A tap too far out to weigh anything stays where it is.
            let offset = if a + b > 0.0 {
                (i as f32 * a + (i + 1) as f32 * b) / (a + b)
            } else {
                i as f32
            };
            [offset, (a + b) / total]
        })
        .collect();
    (w(0) / total, taps)
}

/// A pass's uniform for `frame`, over textures of `size` pixels.
fn uniform(step: &Step, frame: &FrameBlock, size: [u32; 2]) -> [f32; WORDS] {
    let mut words = [0.0f32; WORDS];
    // A pass at a halved size samples texels of that size; one at the
    // view's (all that take pointwise operations) has its pixels.
    let size = halved_size(size, step.halved);
    let texel = [1.0 / size[0] as f32, 1.0 / size[1] as f32];
    let (kernel, step_uv, weights): ([f32; 4], [f32; 2], Vec<f32>) = match &step.kernel {
        Kernel::Copy => ([0.0; 4], [0.0; 2], Vec::new()),
        // Sampled between the four texels each covers: a move by nothing.
        Kernel::Halve => ([4.0, 0.0, 0.0, 0.0], [0.0; 2], Vec::new()),
        // Sampled between the halved texels: a blur of no taps.
        &Kernel::Restore { combine, amount } => {
            ([1.0, 0.0, combine as f32, amount], [0.0; 2], vec![1.0])
        }
        Kernel::BlurX { size } | Kernel::BlurY { size, .. } => {
            let (centre, taps) = gaussian(*size);
            let (combine, amount) = match step.kernel {
                Kernel::BlurY {
                    combine, amount, ..
                } => (combine, amount),
                _ => (0, 0.0),
            };
            let axis = match step.kernel {
                Kernel::BlurX { .. } => [texel[0], 0.0],
                _ => [0.0, texel[1]],
            };
            let mut weights = vec![centre];
            weights.extend(taps.iter().flatten());
            let kernel = [1.0, taps.len() as f32, combine as f32, amount];
            (kernel, axis, weights)
        }
        Kernel::Convolve(k) => {
            let side = (k.len() as f32).sqrt().round();
            let kernel = [2.0, side, 0.0, 0.0];
            (kernel, texel, k.clone())
        }
        &Kernel::Morph { size, erode, y } => {
            let reach = size.round().clamp(1.0, MAX_REACH);
            let axis = if y { [0.0, 1.0] } else { [1.0, 0.0] };
            (
                [3.0, reach, f32::from(u8::from(erode)), 0.0],
                axis,
                Vec::new(),
            )
        }
        Kernel::Offset { .. } => ([4.0, 0.0, 0.0, 0.0], [0.0; 2], Vec::new()),
        // Where each channel comes from, in uv: back along its move.
        &Kernel::Split { channels, radial } => {
            let mut from: Vec<f32> = (channels.iter())
                .flat_map(|[x, y]| [-x * texel[0], -y * texel[1]])
                .collect();
            from.push(f32::from(u8::from(radial)));
            ([5.0, 0.0, 0.0, 0.0], [0.0; 2], from)
        }
    };
    words[0..4].copy_from_slice(&kernel);
    words[4..6].copy_from_slice(&step_uv);
    // Where a moved pixel's colour comes from: back along the move.
    if let Kernel::Offset { translate } = step.kernel {
        words[6] = -translate[0] * texel[0];
        words[7] = -translate[1] * texel[1];
    }
    words[8..8 + weights.len().min(64)].copy_from_slice(&weights[..weights.len().min(64)]);
    words[72] = step.pointwise.len() as f32;
    // The view's pixels from its centre, for gradients across it.
    let half = [size[0] as f32 * 0.5, size[1] as f32 * 0.5];
    // Noise is fixed to the map: lattices of the whole zooms around the
    // camera's, each `size` logical pixels a point at its zoom, the finer
    // fading in towards the next zoom; positions camera-relative, with
    // each lattice's offset at the camera taken in f64 where it repeats.
    let zoom_floor = frame.zoom.floor();
    let fade = (frame.zoom - zoom_floor) as f32;
    let px_per_world = 1.0 / f64::from(frame.world_per_px.max(f32::MIN_POSITIVE));
    let floor_px_per_world = px_per_world * (zoom_floor - frame.zoom).exp2();
    // Moving noise drifts each lattice `speed` logical pixels a second at
    // the zoom its points are `size` apart: as many of its own points
    // each `size` pixels, on both octaves, so the finer drifts on as the
    // coarser once a whole zoom takes it there.
    let seconds = f64::from(frame.time) / 1000.0;
    for (i, op) in step.pointwise.iter().enumerate() {
        let none = [0.0; 4];
        let (a, b, c): ([f32; 4], [f32; 4], [f32; 4]) = match *op {
            Pointwise::Select { color, tolerance } => (
                [1.0, tolerance / 255.0, 0.0, 0.0],
                [color[0], color[1], color[2], 0.0],
                none,
            ),
            Pointwise::Noise {
                size,
                amount,
                speed,
            } => {
                let scale = floor_px_per_world / f64::from(size);
                // Each lattice's offset at the camera, slid `drift` of its
                // points along `direction`.
                let offset = |s: f64, direction: [f64; 2], drift: f64| {
                    let at = |k: usize| frame.camera[k] * s + direction[k] * drift;
                    [0, 1].map(|k| at(k).rem_euclid(256.0) as f32)
                };
                let [x0, y0] = offset(scale, [0.0; 2], 0.0);
                let [x1, y1] = offset(scale * 2.0, [0.0; 2], 0.0);
                if speed > 0.0 {
                    let drift = seconds * f64::from(speed) / f64::from(size);
                    // Two lattices each octave, apart, sliding two ways.
                    let (one, other) = ([0.8, 0.6], [-0.6, 0.8]);
                    let [x0, y0] = offset(scale, one, drift);
                    let [x1, y1] = offset(scale * 2.0, one, drift);
                    let [x2, y2] = offset(scale, other, drift + 128.0);
                    let [x3, y3] = offset(scale * 2.0, other, drift + 128.0);
                    (
                        [10.0, amount, fade, scale as f32],
                        [x0, y0, x1, y1],
                        [x2, y2, x3, y3],
                    )
                } else {
                    ([2.0, amount, fade, scale as f32], [x0, y0, x1, y1], none)
                }
            }
            Pointwise::Threshold { cutoff } => ([3.0, cutoff / 255.0, 0.0, 0.0], none, none),
            Pointwise::Invert => ([4.0, 0.0, 0.0, 0.0], none, none),
            Pointwise::Opacity { value } => ([5.0, value, 0.0, 0.0], none, none),
            Pointwise::Color { color } => ([6.0, 0.0, 0.0, 0.0], color, none),
            Pointwise::Luminance => ([7.0, 0.0, 0.0, 0.0], none, none),
            Pointwise::Ramp => ([8.0, 0.0, 0.0, 0.0], none, none),
            // A linear gradient's direction over its length, by default
            // the view's corner to corner along it; a radial one's scale
            // over its radius, by default to the view's corner farthest
            // from its centre.
            Pointwise::Gradient {
                radial,
                angle,
                center,
                extent,
            } => {
                let (sin, cos) = angle.to_radians().sin_cos();
                let direction = [-sin, cos];
                let corners =
                    2.0 * ((half[0] * direction[0]).abs() + (half[1] * direction[1]).abs());
                let farthest = (half[0] + center[0].abs()).hypot(half[1] + center[1].abs());
                let extent = extent.unwrap_or(if radial { farthest } else { corners });
                let per_px = 1.0 / extent.max(f32::MIN_POSITIVE);
                let along = if radial { [0.0; 2] } else { direction };
                (
                    [11.0, f32::from(u8::from(radial)), 0.0, 0.0],
                    [along[0] * per_px, along[1] * per_px, per_px, 0.0],
                    [center[0], center[1], 0.0, 0.0],
                )
            }
            Pointwise::Bands { count, width } => ([12.0, count, width, 0.0], none, none),
            Pointwise::Combine { mode } => {
                let mode = match mode {
                    CombineMode::Multiply => 0.0,
                    CombineMode::Subtract => 1.0,
                    CombineMode::Max => 2.0,
                    CombineMode::Min => 3.0,
                    CombineMode::Over => 4.0,
                    CombineMode::Under => 5.0,
                };
                ([9.0, mode, 0.0, 0.0], none, none)
            }
        };
        let at = OPS_AT + i * OP_VECTORS * 4;
        words[at..at + 4].copy_from_slice(&a);
        words[at + 4..at + 8].copy_from_slice(&b);
        words[at + 8..at + 12].copy_from_slice(&c);
    }
    if step
        .pointwise
        .iter()
        .any(|p| matches!(p, Pointwise::Noise { .. }))
    {
        words[UNPROJECT_AT..RAMP_AT].copy_from_slice(&unproject(&frame.view_proj));
    }
    words[73] = step.ramp.len() as f32;
    words[74..76].copy_from_slice(&half);
    let colors_at = RAMP_AT + MAX_STOPS;
    for (i, &(at, color)) in step.ramp.iter().enumerate() {
        words[RAMP_AT + i] = at;
        words[colors_at + i * 4..colors_at + i * 4 + 4].copy_from_slice(&color);
    }
    words
}

/// The rows of the map from normalized device coordinates to the
/// camera-relative world on the plane (z = 0), homogeneous, as vectors of
/// four: the inverse of the view's projection of the plane.
fn unproject(m: &[[f64; 4]; 4]) -> [f32; 12] {
    // Clip x, y and w of the plane's (x, y, 1), by row.
    let h = [
        [m[0][0], m[1][0], m[3][0]],
        [m[0][1], m[1][1], m[3][1]],
        [m[0][3], m[1][3], m[3][3]],
    ];
    let det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    if det.abs() < f64::MIN_POSITIVE {
        return [0.0; 12];
    }
    let c =
        |r0: usize, c0: usize, r1: usize, c1: usize| h[r0][c0] * h[r1][c1] - h[r0][c1] * h[r1][c0];
    let inv = [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ];
    let mut rows = [0.0f32; 12];
    for (r, row) in inv.iter().enumerate() {
        for (k, v) in row.iter().enumerate() {
            rows[r * 4 + k] = (v / det) as f32;
        }
    }
    rows
}

impl Release for Pass_ {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.group, self.uniform).release(gpu);
    }
}

impl Release for Made {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.bundles, self.composites, self.solids, self.passes).release(gpu);
        self.depth.release(gpu);
        let scratch: Vec<TextureId> = self.scratch.into_iter().map(|(_, t)| t).collect();
        (self.drawn, self.outputs, scratch).release(gpu);
    }
}

impl Release for Targets {
    fn release(self, gpu: &mut dyn Gpu) {
        self.made.release(gpu);
        (self.pipeline, self.layout, self.sampler).release(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations_become_passes() {
        let ops = [
            ProcessOp::Select {
                color: [1.0, 1.0, 0.0],
                tolerance: 16.0,
            },
            ProcessOp::Blur { size: 5.0 },
            ProcessOp::Noise {
                size: 2.0,
                amount: 0.25,
                speed: 0.0,
            },
            ProcessOp::Threshold { cutoff: 128.0 },
            ProcessOp::Blur { size: 0.0 },
            ProcessOp::HighPass { size: 2.0 },
        ];
        let passes = steps(&ops);
        let kernels: Vec<_> = passes
            .iter()
            .map(|s| (&s.kernel, s.pointwise.len()))
            .collect();
        assert_eq!(
            kernels,
            [
                (&Kernel::Copy, 1),
                (&Kernel::BlurX { size: 5.0 }, 0),
                (
                    &Kernel::BlurY {
                        size: 5.0,
                        combine: 0,
                        amount: 0.0
                    },
                    2
                ),
                (&Kernel::BlurX { size: 2.0 }, 0),
                (
                    &Kernel::BlurY {
                        size: 2.0,
                        combine: 2,
                        amount: 0.0
                    },
                    0
                ),
            ]
        );
        // The high-pass weighs the threshold's output against its blur:
        // neither is the scratch its axes write.
        assert_eq!(
            wiring(&passes),
            [
                (Texture::Source, None, Texture::Scratch(0, 0)),
                (Texture::Scratch(0, 0), None, Texture::Scratch(0, 1)),
                (Texture::Scratch(0, 1), None, Texture::Scratch(0, 0)),
                (Texture::Scratch(0, 0), None, Texture::Scratch(0, 1)),
                (
                    Texture::Scratch(0, 1),
                    Some(Texture::Scratch(0, 0)),
                    Texture::Output
                ),
            ]
        );
        assert_eq!(steps(&[]), [Step::of(Kernel::Copy)]);
        // A wide sharpen blurs the image halved, and weighs what comes
        // back against the image it halved; what follows applies on that
        // pass, at the view's size.
        let wide = steps(&[
            ProcessOp::Sharpen {
                size: 30.0,
                amount: 1.0,
            },
            ProcessOp::Invert,
        ]);
        let kernels: Vec<_> = wide.iter().map(|s| (s.kernel.kind(), s.halved)).collect();
        assert_eq!(kernels, [(9, 1), (1, 1), (2, 1), (11, 0)]);
        assert_eq!(wide[3].pointwise, [Pointwise::Invert]);
        assert_eq!(
            wiring(&wide),
            [
                (Texture::Source, None, Texture::Scratch(1, 0)),
                (Texture::Scratch(1, 0), None, Texture::Scratch(1, 1)),
                (Texture::Scratch(1, 1), None, Texture::Scratch(1, 0)),
                (
                    Texture::Scratch(1, 0),
                    Some(Texture::Source),
                    Texture::Output
                ),
            ]
        );
        // A split samples its input as a kernel; what follows applies on
        // its pass.
        let split = steps(&[
            ProcessOp::Split {
                channels: [[2.0, 0.0], [0.0; 2], [-2.0, 0.0]],
                radial: false,
            },
            ProcessOp::Gradient {
                radial: true,
                angle: 0.0,
                center: [0.0; 2],
                extent: None,
            },
            ProcessOp::Bands {
                count: 4.0,
                width: 0.5,
            },
        ]);
        assert_eq!(split.len(), 1);
        assert_eq!(split[0].pointwise.len(), 2);
    }

    #[test]
    fn gaussians_sum_to_one() {
        // A wider blur runs halved, its Gaussian within the taps' reach
        // there.
        for sigma in [21.0, 40.0, 64.0, 300.0] {
            let small = small_sigma(sigma, halvings(sigma));
            assert!(
                small <= MAX_SIGMA && small > MAX_SIGMA / 2.5,
                "{sigma}: {small} halved"
            );
        }
        for sigma in [0.01, 0.3, 1.0, 5.0, 20.0, MAX_SIGMA] {
            let (centre, taps) = gaussian(sigma);
            assert!(taps.len() <= MAX_TAPS, "{sigma}: {} taps", taps.len());
            let sum = centre + 2.0 * taps.iter().map(|t| t[1]).sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-4, "{sigma}: {sum}");
            // Variance about the centre, as the taps sample it: texels
            // resolve a kernel only a texel or more wide.
            if sigma < 1.0 {
                continue;
            }
            let variance = 2.0 * taps.iter().map(|t| t[0] * t[0] * t[1]).sum::<f32>();
            assert!(
                (variance.sqrt() / sigma - 1.0).abs() < 0.15,
                "{sigma}: σ {}",
                variance.sqrt()
            );
        }
    }

    #[test]
    fn the_plane_unprojects_where_it_projects() {
        // A tilted, turned view: world (x, y, 0) → clip.
        let (c, s) = (0.6f64.cos(), 0.6f64.sin());
        let m = [
            [0.002 * c, 0.001 * s, 0.0, 0.0004],
            [-0.002 * s, 0.001 * c, 0.0, 0.0002],
            [0.0, 0.0, 1.0, 0.0],
            [0.1, -0.2, 0.5, 1.0],
        ];
        let rows = unproject(&m);
        for world in [[0.0, 0.0], [120.0, -40.0], [-300.0, 250.0]] {
            let clip = |r: usize| m[0][r] * world[0] + m[1][r] * world[1] + m[3][r];
            let ndc = [clip(0) / clip(3), clip(1) / clip(3), 1.0];
            let h: Vec<f64> = (0..3)
                .map(|r| (0..3).map(|k| f64::from(rows[r * 4 + k]) * ndc[k]).sum())
                .collect();
            let back = [h[0] / h[2], h[1] / h[2]];
            assert!(
                (back[0] - world[0]).abs() < 1e-2 && (back[1] - world[1]).abs() < 1e-2,
                "{world:?} → {back:?}"
            );
        }
    }
}
