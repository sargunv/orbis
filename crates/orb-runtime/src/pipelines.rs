//! A style's pipeline set: every pipeline it can reach on a renderer,
//! derived when the style loads or an edit changes its nodes, and asked
//! for at once, the first view's first. Frames choose among them and ask
//! for none, so a gesture that brings a layer in, or changes what a
//! layer's paint reads, never compiles one.

use std::sync::Arc;

use orb_hal::Gpu;
use orb_render::{Blend, Family, ImageAtlas, NodePaint, PipelineGroup, Renderer};
use orb_style::CameraInputs;

use crate::map::LoadedStyle;
use crate::paint::{Crossfade, PaintAt, camera_plan, drawn_at, layer_composite, node_paint};

/// What a set was made for: the renderer, the plan (its style's address)
/// and the renderer's nodes (`LoadedStyle::drawn`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SetKey(u64, usize, Vec<u32>);

impl SetKey {
    pub(crate) fn of(renderer: &Renderer, style: &LoadedStyle) -> Self {
        let plan = style.worker.plan();
        Self(
            renderer.id(),
            Arc::as_ptr(&plan.style) as usize,
            style.drawn.clone(),
        )
    }
}

/// Every pipeline a style can reach, by what draws together at a zoom.
pub(crate) struct PipelineSet {
    pub(crate) groups: Vec<PipelineGroup>,
}

impl PipelineSet {
    /// The set of the style's nodes: each painted at the camera's `zoom`
    /// (the first view, compiled first) and at every whole zoom its layer
    /// draws at, nearest first, a symbol layer's two halves together. The
    /// variants, sample count and terrain the pipelines are made in are
    /// the renderer's.
    pub(crate) fn for_style(
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        style: &mut LoadedStyle,
        (zoom, inputs): (f64, CameraInputs),
        atlas: &mut ImageAtlas,
    ) -> Self {
        let worker = Arc::clone(&style.worker);
        let plan = worker.plan();
        let camera = camera_plan(style, plan, inputs);
        let plan = camera.as_deref().unwrap_or(plan);
        let crossfade = Crossfade {
            from_scale: 1.0,
            mix: 1.0,
        };
        // The camera's zoom, then whole zooms nearest it.
        let mut zooms: Vec<f64> = (0..=24).map(f64::from).collect();
        zooms.sort_by(|a, b| (a - zoom).abs().total_cmp(&(b - zoom).abs()));
        zooms.insert(0, zoom);
        let mut groups: Vec<PipelineGroup> = Vec::new();
        for (index, node) in plan.nodes.iter().enumerate() {
            let layer = &plan.style.layers[node.layer];
            let from = groups.len();
            if node.family.is_image() || node.family == Family::Heatmap {
                // Their family alone picks their pipelines, and their
                // blend their composite's.
                for (n, &at) in zooms.iter().enumerate() {
                    let composite = composite_at(layer, at);
                    if drawn_at(layer, node, at)
                        && !groups[from..].iter().any(|g| g.composite == composite)
                    {
                        groups.push(PipelineGroup {
                            nodes: vec![(index as u32, None)],
                            composite,
                            first: n == 0,
                        });
                    }
                }
                continue;
            }
            // A symbol layer's icon node draws its text node's quads too,
            // with one pipeline both halves' paint at a zoom picks.
            let nodes = match node.family {
                Family::SymbolText => continue,
                Family::SymbolIcon => index..index + 2,
                _ => index..index + 1,
            };
            for i in nodes.clone() {
                let node = &plan.nodes[i];
                let state = style.nodes.entry(node.key).or_default();
                let properties = plan.paint_properties(node).len();
                if state.maxima.len() < properties {
                    state.maxima.resize(properties, 0.0);
                }
            }
            for (n, &at) in zooms.iter().enumerate() {
                if !drawn_at(layer, node, at) {
                    continue;
                }
                let composite = composite_at(layer, at);
                let nodes: Vec<_> = (nodes.clone())
                    .map(|i| {
                        let node = &plan.nodes[i];
                        let state = style.nodes.entry(node.key).or_default();
                        let lookup = style.states.lookup(i);
                        let paint = if node.family.is_symbol() {
                            crate::labels::paint(plan, node, at, &state.maxima, lookup)
                                .map(NodePaint::Symbol)
                        } else if node.family == Family::Circle {
                            crate::point::node_paint(plan, node, at, &state.maxima, lookup)
                        } else {
                            let at = PaintAt {
                                zoom: at,
                                bearing: inputs.bearing,
                                crossfade,
                            };
                            node_paint(gpu, renderer, (style, plan), i as u32, at, atlas)
                        };
                        (i as u32, paint)
                    })
                    .collect();
                if nodes.iter().all(|(_, paint)| paint.is_none()) {
                    continue;
                }
                // The renderer makes each pipeline once: only paint that
                // may pick others is kept.
                let same = |g: &PipelineGroup| {
                    g.composite == composite
                        && (g.nodes.iter().zip(&nodes)).all(|((_, a), (_, b))| match (a, b) {
                            (Some(a), Some(b)) => orb_render::same_pipelines(a, b),
                            (a, b) => a.is_none() && b.is_none(),
                        })
                };
                if !groups[from..].iter().any(same) {
                    groups.push(PipelineGroup {
                        nodes,
                        composite,
                        first: n == 0,
                    });
                }
            }
        }
        Self { groups }
    }
}

/// The blend a layer composites in at `zoom`, when it composites once:
/// below layer opacity 1, or blending other than normally.
fn composite_at(layer: &orb_style::Layer, zoom: f64) -> Option<Blend> {
    layer_composite(layer, zoom)
        .filter(|&(opacity, blend)| opacity < 1.0 || blend != Blend::Normal)
        .map(|(_, blend)| blend)
}
