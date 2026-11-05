//! Model layers (plan §9.4): a model node's paint.
//!
//! A model layer draws as one node per model its `model-id` can pick, each
//! with the layer's points that pick it (`style`, `tile`). The map loads
//! the document's models through [`Jobs::model`](crate::Jobs::model) and
//! uploads each at its index in the document's models.

use orb_hal::Gpu;
use orb_render::{Curve, ModelPaint, NodePaint, Renderer, StateLookup};

use crate::style::{NodePlan, PaintProperty, StylePlan};

/// Writes a model node's paint at `zoom`. `maxima` are its property rows'
/// maxima; `metres_per_px` the camera target's, for the cull's reach;
/// `state` where its feature-state lookups read.
pub(crate) fn paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (plan, node, index): (&StylePlan, &NodePlan, u32),
    zoom: f64,
    (maxima, state): (&[f32], StateLookup),
    (bounds, metres_per_px): (Option<[[f32; 3]; 2]>, f64),
) {
    let l = &plan.style.layers[node.layer];
    let mut curves = [Curve::Step; 12];
    let mut numbers = [0.0f32; 12];
    let mut color = [0.0; 4];
    let mut table_mask = 0;
    for (slot, property) in plan.paint_properties(node).iter().enumerate() {
        if property.per_feature() {
            table_mask |= 1 << slot;
            curves[slot] = crate::paint::curve(property);
            numbers[slot] = maxima.get(slot).copied().unwrap_or(0.0);
        } else if property.is_state_lookup() {
            // The value without state; features with state read theirs.
            curves[slot] = crate::paint::curve(property);
            match property {
                PaintProperty::Color(_) => color = property.default_color(zoom),
                p => numbers[slot] = p.default_number(zoom),
            }
        } else {
            match property {
                PaintProperty::Color(_) => color = property.color(zoom),
                p => numbers[slot] = p.number(zoom),
            }
        }
    }
    let triple = |at: usize| [numbers[at], numbers[at + 1], numbers[at + 2]];
    let (scale, translation) = (triple(3), triple(9));
    // The farthest a model reaches from its point, and above the ground.
    let [lo, hi] = bounds.unwrap_or_default();
    let largest = scale.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let radius = lo.iter().chain(&hi).fold(0.0f32, |m, v| m.max(v.abs())) * largest
        + translation.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    // A view's latitudes vary its metres per pixel a little: slack.
    let reach_px = (f64::from(radius) / metres_per_px * 1.1) as f32 + 1.0;
    let height = hi[2].max(lo[2]).max(0.0) * scale[2].abs() + translation[2];
    renderer.set_paint(
        gpu,
        index,
        &NodePaint::Model(ModelPaint {
            color,
            color_mix: numbers[1],
            emissive: numbers[2],
            scale,
            rotation: triple(6),
            translation,
            opacity: l.get(orb_style::MODEL_OPACITY).evaluate(zoom),
            table_mask,
            curves,
            reach_px,
            height,
            state,
        }),
    );
}
