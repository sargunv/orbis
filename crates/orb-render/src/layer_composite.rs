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
}

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
}

#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) nodes: Range<usize>,
    pub(crate) blend: Blend,
    /// Its own target, unless another pass draws the layer.
    layer: Option<Target>,
    composite: Composite,
    pub(crate) bundle: Option<RenderBundleId>,
}

impl Group {
    /// Whether another draw draws the layer (a terrain layer), so the
    /// group records no bundle of its own.
    pub(crate) fn drawn_elsewhere(&self) -> bool {
        self.layer.is_none()
    }

    fn mode(&self) -> Mode {
        match self.blend {
            Blend::Normal => Mode::Over,
            blend => Mode::Blend(blend),
        }
    }
}

impl LayerComposites {
    /// Whether the pipelines the groups composite with compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu, composites: &Composites) -> bool {
        self.groups.iter().all(|g| composites.ready(gpu, g.mode()))
    }

    /// Replaces the groups with `runs`, each drawn into a target of `size`
    /// in the map's `format` of its own unless it has a source. The
    /// canvases keep their samples between passes where the map's passes
    /// split at 3D layers (`split`).
    pub(crate) fn regroup(
        &mut self,
        gpu: &mut dyn Gpu,
        composites: &mut Composites,
        runs: Vec<Run>,
        (size, format, split): ([u32; 2], crate::TargetFormat, bool),
    ) {
        self.clear(gpu);
        let target = |gpu: &mut dyn Gpu, label, kept| {
            Target::new(
                gpu,
                label,
                &[format.color],
                size,
                (format.sample_count, kept),
            )
        };
        if runs.iter().any(|r| r.blend != Blend::Normal) {
            self.canvases = vec![
                target(gpu, "map canvas", split),
                target(gpu, "map canvas", split),
            ];
        }
        let mut backdrops = 0;
        for Run {
            nodes,
            opacity,
            blend,
            source,
        } in runs
        {
            let layer = source
                .is_none()
                .then(|| target(gpu, "layer composite", false));
            let texture = source.unwrap_or_else(|| layer.as_ref().expect("made above").texture(0));
            // The k-th blending group reads the canvas pass k drew.
            let (mode, backdrop) = match blend {
                Blend::Normal => (Mode::Over, None),
                blend => {
                    backdrops += 1;
                    (
                        Mode::Blend(blend),
                        Some(self.canvases[(backdrops - 1) % 2].texture(0)),
                    )
                }
            };
            let composite = composites.composite(gpu, texture, backdrop, mode, opacity);
            self.groups.push(Group {
                nodes,
                blend,
                layer,
                composite,
                bundle: None,
            });
        }
    }

    /// Writes a group's opacity, if it changed.
    pub(crate) fn set_opacity(&mut self, gpu: &mut dyn Gpu, node: usize, opacity: f32) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.nodes.contains(&node)) {
            Composites::set_opacity(gpu, &mut group.composite, opacity);
        }
    }

    /// The group node `i` starts, if any, by its index, and whether the
    /// node belongs to one.
    pub(crate) fn at(&self, i: usize) -> (Option<usize>, bool) {
        match self.groups.iter().position(|g| g.nodes.contains(&i)) {
            Some(g) => ((self.groups[g].nodes.start == i).then_some(g), true),
            None => (None, false),
        }
    }

    /// A group's composite into the map: in the main pass's bundle for
    /// normal blending, or opening the canvas pass after a split.
    pub(crate) fn composite_commands(
        &self,
        composites: &Composites,
        group: &Group,
    ) -> [RenderCommand; 3] {
        composites.commands(&group.composite)
    }

    /// Each group's pass: its nodes' draws into its target, over `depth`
    /// where the map's passes have one.
    pub(crate) fn passes(&self, depth: Option<TextureId>) -> impl Iterator<Item = Pass> + '_ {
        self.groups.iter().filter_map(move |g| {
            Some(Pass::Render(RenderPass {
                label: "layer composite",
                colors: vec![g.layer.as_ref()?.attachment(0, LoadOp::Clear([0.0; 4]))],
                depth: depth.map(|target| DepthAttachment {
                    target,
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Discard,
                }),
                commands: vec![RenderCommand::ExecuteBundles(vec![g.bundle?])],
            }))
        })
    }

    /// The canvas the map's `k`th pass draws into where a later one reads
    /// it: the canvases take turns.
    pub(crate) fn canvas(&self, k: usize) -> &Target {
        &self.canvases[k % 2]
    }

    pub(crate) fn clear(&mut self, gpu: &mut dyn Gpu) {
        for g in self.groups.drain(..) {
            (g.bundle, g.composite, g.layer).release(gpu);
        }
        std::mem::take(&mut self.canvases).release(gpu);
    }
}

impl crate::release::Release for LayerComposites {
    fn release(mut self, gpu: &mut dyn Gpu) {
        self.clear(gpu);
    }
}
