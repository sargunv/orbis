//! Paint for the two point families. Geometry and property rows stay fixed
//! while the camera changes; zoom-dependent values live in node uniforms.

use crate::style::{NodePlan, PaintProperty, StylePlan};
use orb_hal::Gpu;
use orb_render::{
    CirclePaint, Curve, HeatmapPaint, NodePaint, Renderer, StateLookup, Translate, pack_color,
};
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
    if !matches!(l.ty, LayerType::Circle | LayerType::Heatmap) {
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
    let paint = match l.ty {
        LayerType::Circle => {
            use orb_style::{
                CIRCLE_PITCH_ALIGNMENT, CIRCLE_PITCH_SCALE, CIRCLE_TRANSLATE,
                CIRCLE_TRANSLATE_ANCHOR,
            };
            NodePaint::Circle(CirclePaint {
                color: colors[0],
                radius: numbers[1],
                blur: numbers[2],
                opacity: numbers[3],
                stroke_color: colors[4],
                stroke_width: numbers[5],
                stroke_opacity: numbers[6],
                translate: Translate {
                    px: l.get(CIRCLE_TRANSLATE).evaluate(zoom),
                    viewport: l.get(CIRCLE_TRANSLATE_ANCHOR).evaluate(zoom)
                        == TranslateAnchor::Viewport,
                },
                pitch_scale_map: l.get(CIRCLE_PITCH_SCALE).evaluate(zoom) == TranslateAnchor::Map,
                pitch_alignment_map: l.get(CIRCLE_PITCH_ALIGNMENT).evaluate(zoom)
                    == TranslateAnchor::Map,
                table_mask,
                curves,
                state: lookup,
            })
        }
        LayerType::Heatmap => {
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
            NodePaint::Heatmap(HeatmapPaint {
                radius: numbers[0],
                weight: numbers[1],
                intensity: l.get(orb_style::HEATMAP_INTENSITY).evaluate(zoom),
                opacity: l.get(orb_style::HEATMAP_OPACITY).evaluate(zoom),
                table_mask,
                curves: [curves[0], curves[1]],
            })
        }
        _ => unreachable!(),
    };
    renderer.set_paint(gpu, index, &paint);
    true
}
