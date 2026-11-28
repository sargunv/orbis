//! Layer composites: `*-layer-opacity` and `*-layer-blend` (style-spec 26).
//! A style layer's nodes draw into a texture of their own. It composites
//! into the map once, at the layer's opacity, so its overlaps don't
//! accumulate (GL JS `draw_layer_opacity.ts`). A group is a run of
//! consecutive nodes of one layer below opacity 1 or in a blend mode other
//! than normal. Otherwise the layer draws as usual.
//!
//! Normal groups composite with fixed-function premultiplied blending,
//! inside the main pass. Every other mode needs the map beneath it. The
//! main pass splits at each such group into canvases, two textures used in
//! turn. Each canvas pass after the first opens with a full-screen
//! draw: the group's layer over the previous canvas, by W3C Compositing 1's
//! formula. Then the map's draws continue on top. The last pass draws
//! into the target, so a map without such groups keeps its one pass.
//!
//! On terrain, a draped layer's group is its own terrain layer, which
//! draws at its opacity in the map's pass. One in a blend mode other than
//! normal splits the pass there all the same: its composite of nothing
//! carries the canvas over, and the terrain layer draws over that, reading
//! the map beneath from the canvas before, and fading into the fog rather
//! than taking its colour, since what it blends with is fogged already. A
//! blended hillshade the terrain lights needs no group where its terrain
//! layer blends it into the stack below it; where it blends into the map
//! beneath, its terrain layer splits the pass as such a group does.

use crate::release::Release;
use std::ops::Range;

use orb_hal::*;

use crate::offscreen::{Composite, Composites, Mode, Target};
use crate::pipeline::{Attachments, Cache};

/// How a layer composited once combines with the map beneath it: W3C
/// Compositing 1's modes, as `fill-layer-blend`, `line-layer-blend` and
/// `raster-layer-blend` name them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Blend {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Color,
    Luminosity,
    PlusLighter,
}

impl Blend {
    /// The shader's mode number (`blend_mix` in `composite.wgsl`).
    pub(crate) fn mode(self) -> f32 {
        match self {
            Blend::Normal => 0.0,
            Blend::Multiply => 1.0,
            Blend::Screen => 2.0,
            Blend::Overlay => 3.0,
            Blend::Color => 4.0,
            Blend::Luminosity => 5.0,
            Blend::PlusLighter => 6.0,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct LayerComposites {
    pub(crate) groups: Vec<Group>,
    /// The two canvases the main pass splits into, while a group blends
    /// in a mode other than normal.
    canvases: Vec<Target>,
    /// What the targets were made for: the frame's size, the map's format
    /// and whether the map's passes split at 3D layers.
    made: Option<Made>,
}

/// What a group's targets are made for ([`LayerComposites::fit`]).
pub(crate) type Made = ([u32; 2], crate::TargetFormat, bool);

/// A run of one layer's nodes that composites once.
#[derive(Debug)]
pub(crate) struct Run {
    pub(crate) nodes: Range<usize>,
    pub(crate) opacity: f32,
    pub(crate) blend: Blend,
    /// What it composites when another draw draws the layer (a terrain
    /// layer, which leaves nothing to composite), or `None` for the
    /// group's own target.
    pub(crate) source: Option<TextureId>,
    /// The render target whose coverage it composites through (a fill
    /// drawn through a target).
    pub(crate) mask: Option<TextureId>,
    /// The colour it composites in place of a source, premultiplied: a
    /// fill in a colour alone drawn through a target, below opacity 1 or
    /// in a blend mode, which covers the whole mask as one drawn at
    /// opacity 1 normally does.
    pub(crate) color: Option<[f32; 4]>,
}

impl Run {
    /// What its group's target and composite are made for, the `backdrop`
    /// canvas it blends over included: all but its opacity and colour.
    fn key(&self, backdrop: Option<usize>) -> RunKey {
        RunKey {
            nodes: self.nodes.clone(),
            blend: self.blend,
            source: self.source,
            mask: self.mask,
            colored: self.color.is_some(),
            backdrop,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RunKey {
    nodes: Range<usize>,
    blend: Blend,
    source: Option<TextureId>,
    mask: Option<TextureId>,
    colored: bool,
    backdrop: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct Group {
    key: RunKey,
    /// Its own target, unless another pass draws the layer.
    layer: Option<Target>,
    composite: Composite,
}

impl Group {
    pub(crate) fn nodes(&self) -> Range<usize> {
        self.key.nodes.clone()
    }

    pub(crate) fn blend(&self) -> Blend {
        self.key.blend
    }

    /// Whether another draw draws the layer (a terrain layer), so the
    /// group records no bundle of its own.
    pub(crate) fn drawn_elsewhere(&self) -> bool {
        self.layer.is_none()
    }

    /// Its composite into the map's passes as `at` has them: in the main
    /// pass's bundle for normal blending, or opening the canvas pass after
    /// a split.
    pub(crate) fn composite_commands(
        &self,
        composites: &Composites,
        at: (&Cache, Attachments),
    ) -> [RenderCommand; 3] {
        (composites.commands(at, &self.composite)).expect("made before its group records")
    }

    fn mode(&self) -> Mode {
        match self.key.blend {
            Blend::Normal if self.key.mask.is_some() => Mode::Mask,
            Blend::Normal => Mode::Over,
            blend => Mode::Blend(blend),
        }
    }
}

impl LayerComposites {
    /// The pipelines the groups composite with into the map's passes as
    /// `target`, `None` for one not made.
    pub(crate) fn needs(
        &self,
        composites: &Composites,
        (cache, target): (&Cache, Attachments),
    ) -> Vec<Option<PipelineId>> {
        (self.groups.iter())
            .map(|g| composites.needs(cache, target, g.mode()))
            .collect()
    }

    /// Whether its targets are made for `made`.
    pub(crate) fn made_for(&self, made: Made) -> bool {
        self.made == Some(made)
    }

    /// Fits the groups to `runs`, each drawn into a target of the frame's
    /// size in the map's format of its own unless it has a source, as
    /// `made` says, the canvases keeping their samples between passes
    /// where the map's passes split at 3D layers: what it holds for the
    /// same runs at the same size stays, its opacity and colour written.
    /// Whether it made or let go of a target or composite, which the
    /// bundles bind.
    pub(crate) fn fit(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        runs: Vec<Run>,
        made: Made,
    ) -> bool {
        let mut changed = false;
        if self.made != Some(made) {
            changed = !self.groups.is_empty() || !self.canvases.is_empty();
            self.clear(gpu);
            self.made = Some(made);
        }
        let (size, format, split) = made;
        let target = |gpu: &mut dyn Gpu, label, kept| {
            Target::new(gpu, label, format.color, size, (format.sample_count, kept))
        };
        // The groups blending over them go with them.
        let canvases = runs.iter().any(|r| r.blend != Blend::Normal);
        if canvases == self.canvases.is_empty() {
            changed = true;
            std::mem::take(&mut self.canvases).release(gpu);
            if canvases {
                self.canvases = vec![
                    target(gpu, "map canvas", split),
                    target(gpu, "map canvas", split),
                ];
            }
        }
        let mut old = std::mem::take(&mut self.groups);
        let mut backdrops = 0;
        for run in runs {
            // The k-th blending group reads the canvas pass k drew.
            let backdrop = (run.blend != Blend::Normal).then(|| {
                backdrops += 1;
                backdrops - 1
            });
            let key = run.key(backdrop);
            if let Some(at) = old.iter().position(|g| g.key == key) {
                let mut group = old.swap_remove(at);
                Composites::set_opacity(gpu, &mut group.composite, run.opacity);
                if let Some(color) = run.color {
                    Composites::set_color(gpu, &mut group.composite, color);
                }
                self.groups.push(group);
                continue;
            }
            changed = true;
            let layer = (run.source.is_none()).then(|| target(gpu, "layer composite", false));
            let texture = run
                .source
                .unwrap_or_else(|| layer.as_ref().expect("made above").texture());
            let backdrop = backdrop.map(|k| self.canvas(k).texture());
            let mode = match run.blend {
                Blend::Normal if run.mask.is_some() => Mode::Mask,
                Blend::Normal => Mode::Over,
                blend => Mode::Blend(blend),
            };
            let composite = match (run.color, run.mask) {
                (Some(color), Some(mask)) => {
                    composites.colour_through(gpu, (backdrop, mask), run.blend, run.opacity, color)
                }
                _ => {
                    let sources = (texture, backdrop, run.mask);
                    composites.composite_through(gpu, sources, mode, run.opacity)
                }
            };
            self.groups.push(Group {
                key,
                layer,
                composite,
            });
        }
        changed |= !old.is_empty();
        old.release(gpu);
        changed
    }

    /// Writes a group's opacity, if it changed.
    pub(crate) fn set_opacity(&mut self, gpu: &mut dyn Gpu, node: usize, opacity: f32) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.key.nodes.contains(&node)) {
            Composites::set_opacity(gpu, &mut group.composite, opacity);
        }
    }

    /// Writes the colour of a group that composites one, if it changed.
    pub(crate) fn set_color(&mut self, gpu: &mut dyn Gpu, node: usize, color: [f32; 4]) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.key.nodes.contains(&node)) {
            Composites::set_color(gpu, &mut group.composite, color);
        }
    }

    /// The group node `i` starts, if any, by its index, and whether the
    /// node belongs to one.
    pub(crate) fn at(&self, i: usize) -> (Option<usize>, bool) {
        match self.groups.iter().position(|g| g.key.nodes.contains(&i)) {
            Some(g) => ((self.groups[g].key.nodes.start == i).then_some(g), true),
            None => (None, false),
        }
    }

    /// Each group's pass: its nodes' draws (`bundles`, by group, as the
    /// main recording made them) into its target, over `depth` where the
    /// map's passes have one.
    pub(crate) fn passes<'a>(
        &'a self,
        bundles: &'a [Option<RenderBundleId>],
        depth: Option<TextureId>,
    ) -> impl Iterator<Item = Pass> + 'a {
        (self.groups.iter().zip(bundles)).filter_map(move |(g, bundle)| {
            Some(Pass::Render(RenderPass {
                label: "layer composite",
                colors: vec![g.layer.as_ref()?.attachment(LoadOp::Clear([0.0; 4]))],
                depth: depth.map(|target| DepthAttachment {
                    target,
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Discard,
                }),
                commands: vec![RenderCommand::ExecuteBundles(vec![(*bundle)?])],
            }))
        })
    }

    /// The canvas the map's `k`th pass draws into where a later one reads
    /// it: the canvases take turns.
    pub(crate) fn canvas(&self, k: usize) -> &Target {
        &self.canvases[k % 2]
    }

    fn clear(&mut self, gpu: &mut dyn Gpu) {
        std::mem::take(&mut self.groups).release(gpu);
        std::mem::take(&mut self.canvases).release(gpu);
        self.made = None;
    }
}

impl crate::release::Release for Group {
    fn release(self, gpu: &mut dyn Gpu) {
        (self.composite, self.layer).release(gpu);
    }
}

impl crate::release::Release for LayerComposites {
    fn release(mut self, gpu: &mut dyn Gpu) {
        self.clear(gpu);
    }
}
