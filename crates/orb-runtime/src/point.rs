//! Paint for the two point families. Geometry and property rows stay fixed
//! while the camera changes; zoom-dependent values live in node uniforms.

use crate::style::{NodePlan, PaintProperty, StylePlan};
use orb_hal::Gpu;
use orb_render::{CirclePaint, Curve, HeatmapPaint, NodePaint, Renderer, StateLookup, pack_color};
use orb_style::{LayerType, TranslateAnchor};

#[allow(clippy::too_many_arguments)]
pub(crate) fn paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    plan: &StylePlan,
    node: &NodePlan,
    index: u32,
    zoom: f64,
    maxima: &[f32],
    lookup: StateLookup,
) -> bool {
    let l = &plan.style.layers[node.layer];
    if l.ty == LayerType::Heatmap {
        let ramp = l
            .ramp(orb_style::HEATMAP_COLOR)
            .expect("heatmaps have a default ramp");
        let colors: Vec<u32> = (0..256)
            .map(|i| {
                let context = orb_expr::EvaluationContext {
                    heatmap_density: f64::from(i) / 255.0,
                    ..Default::default()
                };
                let color = ramp
                    .expression()
                    .evaluate_with(&context)
                    .ok()
                    .and_then(|v| v.as_color())
                    .map_or([0.0; 4], |c| c.premultiplied());
                pack_color(color)
            })
            .collect();
        renderer.set_heatmap_ramp(gpu, index, &colors);
    }
    let Some(paint) = node_paint(plan, node, zoom, maxima, lookup) else {
        return false;
    };
    renderer.set_paint(gpu, index, &paint);
    true
}

/// A circle or heatmap node's paint at `zoom`.
pub(crate) fn node_paint(
    plan: &StylePlan,
    node: &NodePlan,
    zoom: f64,
    maxima: &[f32],
    lookup: StateLookup,
) -> Option<NodePaint> {
    let l = &plan.style.layers[node.layer];
    if !matches!(l.ty, LayerType::Circle | LayerType::Heatmap) {
        return None;
    }
    let mut curves = [Curve::Step; 7];
    let mut numbers = [0.0; 7];
    let mut colors = [[0.0; 4]; 7];
    let mut table_mask = 0;
    let properties = plan.paint_properties(node);
    for (slot, property) in properties.iter().enumerate() {
        if property.per_feature() {
            table_mask |= 1 << slot;
            curves[slot] = crate::paint::curve(property);
            numbers[slot] = maxima[slot];
        } else if property.is_state_lookup() {
            curves[slot] = crate::paint::curve(property);
            match property {
                PaintProperty::Number(_) => numbers[slot] = property.default_number(zoom),
                p => colors[slot] = p.default_color(zoom),
            }
        } else {
            match property {
                PaintProperty::Number(p) => numbers[slot] = p.evaluate(zoom),
                p => colors[slot] = p.color(zoom),
            }
        }
    }
    let paint = match l.ty {
        LayerType::Circle => {
            use orb_style::{CIRCLE_PITCH_ALIGNMENT, CIRCLE_PITCH_SCALE};
            let map = |key| l.get(key).evaluate(zoom) == TranslateAnchor::Map;
            // Per feature (beyond the spec), from the rows; the quads have
            // the cells map-aligned circles need.
            let pitch_rows = plan.circle_pitch_rows(node);
            NodePaint::Circle(CirclePaint {
                color: colors[0],
                radius: numbers[1],
                blur: numbers[2],
                opacity: numbers[3],
                stroke_color: colors[4],
                stroke_width: numbers[5],
                stroke_opacity: numbers[6],
                translate: crate::paint::translate(plan, node, zoom, maxima.get(properties.len())),
                pitch_scale_map: map(CIRCLE_PITCH_SCALE),
                pitch_alignment_map: pitch_rows || map(CIRCLE_PITCH_ALIGNMENT),
                pitch_rows,
                table_mask,
                curves,
                state: lookup,
            })
        }
        LayerType::Heatmap => NodePaint::Heatmap(HeatmapPaint {
            radius: numbers[0],
            weight: numbers[1],
            intensity: l.get(orb_style::HEATMAP_INTENSITY).evaluate(zoom),
            opacity: l.get(orb_style::HEATMAP_OPACITY).evaluate(zoom),
            table_mask,
            curves: [curves[0], curves[1]],
        }),
        _ => unreachable!(),
    };
    Some(paint)
}
