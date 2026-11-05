//! Paint for the two point families. Geometry and property rows stay fixed
//! while the camera changes; zoom-dependent values live in node uniforms.

use crate::style::{NodePlan, PaintProperty, StylePlan};
use orb_hal::Gpu;
use orb_render::{
    CirclePaint, Curve, HeatmapPaint, NodePaint, Renderer, StateLookup, Translate, pack_color,
};
use orb_v8::{LayerKind, TranslateAnchor};

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
    let kind = &plan.style.layers[node.layer].kind;
    if !matches!(kind, LayerKind::Circle(_) | LayerKind::Heatmap(_)) {
        return false;
    }
    let mut curves = [Curve::Step; 7];
    let mut numbers = [0.0; 7];
    let mut colors = [[0.0; 4]; 7];
    let mut table_mask = 0;
    for (slot, property) in plan.paint_properties(node).iter().enumerate() {
        if property.per_feature() {
            table_mask |= 1 << slot;
            curves[slot] = property.curve();
            numbers[slot] = maxima[slot];
        } else if property.is_state_lookup() {
            curves[slot] = property.curve();
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
    let paint = match kind {
        LayerKind::Circle(c) => {
            let p = &c.paint;
            NodePaint::Circle(CirclePaint {
                color: colors[0],
                radius: numbers[1],
                blur: numbers[2],
                opacity: numbers[3],
                stroke_color: colors[4],
                stroke_width: numbers[5],
                stroke_opacity: numbers[6],
                translate: Translate {
                    px: p.translate.evaluate(zoom),
                    viewport: p.translate_anchor.evaluate(zoom) == TranslateAnchor::Viewport,
                },
                pitch_scale_map: p.pitch_scale.evaluate(zoom) == TranslateAnchor::Map,
                pitch_alignment_map: p.pitch_alignment.evaluate(zoom) == TranslateAnchor::Map,
                table_mask,
                curves,
                state: lookup,
            })
        }
        LayerKind::Heatmap(h) => {
            let colors: Vec<u32> = (0..256)
                .map(|i| {
                    let context = orb_expr::EvaluationContext {
                        heatmap_density: f64::from(i) / 255.0,
                        ..Default::default()
                    };
                    let color = h
                        .paint
                        .color
                        .expression()
                        .evaluate_with(&context)
                        .ok()
                        .and_then(|v| v.as_color())
                        .map_or([0.0; 4], |c| c.premultiplied());
                    pack_color(color)
                })
                .collect();
            renderer.set_heatmap_ramp(gpu, index, &colors);
            NodePaint::Heatmap(HeatmapPaint {
                radius: numbers[0],
                weight: numbers[1],
                intensity: h.paint.intensity.evaluate(zoom),
                opacity: h.paint.opacity.evaluate(zoom),
                table_mask,
                curves: [curves[0], curves[1]],
            })
        }
        _ => unreachable!(),
    };
    renderer.set_paint(gpu, index, &paint);
    true
}
