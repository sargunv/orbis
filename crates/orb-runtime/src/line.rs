//! Line texture keys are job data. GPU allocation and stable ids belong to
//! the shared scene atlas, just like named sprite dependencies.

use orb_hal::Gpu;
use orb_render::{ImageAtlas, ImageKey, NodePaint, PatternPaint, pack_color};
use orb_style::{
    Kind, LINE_CAP, LINE_DASHARRAY, LINE_GRADIENT, LINE_PATTERN, LINE_WIDTH, LayerType, LineCap,
};
use orb_tile::Layer;

use crate::map::Crossfade;
use crate::tile::ImageRef;

/// Whether a line layer's property rows carry dash or gradient textures.
pub(crate) fn has_textures(layer: &orb_style::Layer) -> bool {
    layer.ty == LayerType::Line
        && (layer.get(LINE_DASHARRAY).constant() != Some(None)
            || layer.ramp(LINE_GRADIENT).is_some())
}

pub(crate) fn textures(
    line: &orb_style::Layer,
    layer: &Layer,
    features: &[u32],
    zoom: f64,
    source_maxzoom: u8,
    props: &mut [u32],
    images: &mut Vec<ImageRef>,
) {
    // One row per feature, of the family's words and any input columns.
    let stride = props.len() / features.len().max(1);
    if line.get(LINE_PATTERN).constant() != Some(None) {
        return;
    }
    let dashed = line.get(LINE_DASHARRAY).constant() != Some(None);
    if !dashed && line.ramp(LINE_GRADIENT).is_none() {
        return;
    }
    let dash = line.get(LINE_DASHARRAY).bind(layer);
    let cap = line.get(LINE_CAP).bind(layer);
    let width = line.get(LINE_WIDTH).bind(layer);
    let gradient = line.ramp(LINE_GRADIENT).map(|g| g.expression());
    let nearest = gradient.is_some_and(|g| g.is_step());
    let max_length = features
        .iter()
        .map(|&f| {
            layer
                .geometry(f as usize)
                .parts()
                .map(|p| {
                    p.rings()
                        .map(|ring| {
                            ring.windows(2)
                                .map(|p| {
                                    (f64::from(p[1][0]) - f64::from(p[0][0]))
                                        .hypot(f64::from(p[1][1]) - f64::from(p[0][1]))
                                })
                                .sum::<f64>()
                        })
                        .fold(0.0, f64::max)
                })
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    // GL JS reserves resolution for overzooming a source's last canonical tile
    // through its default camera maxZoom (22), then clamps to texture limits.
    let overzoom = if zoom == f64::from(source_maxzoom) {
        2f64.powi(22 - i32::from(source_maxzoom)).max(1.0)
    } else {
        1.0
    };
    let resolution = if nearest {
        ((max_length / f64::from(layer.extent()) * 1024.0 * overzoom)
            .clamp(256.0, 16384.0)
            .ceil() as usize)
            .next_power_of_two()
    } else {
        256
    };
    for (slot, &feature) in features.iter().enumerate() {
        let base = slot * stride;
        props[base + 18] = width.evaluate(zoom, feature as usize).to_bits();
        if dashed {
            for (i, z) in [zoom - 1.0, zoom, zoom + 1.0].into_iter().enumerate() {
                if let Some(array) = dash.evaluate(z, feature as usize) {
                    images.push(ImageRef {
                        word: (base + 15 + i) as u32,
                        key: ImageKey::Dash {
                            array: array.iter().map(|v| v.to_bits()).collect(),
                            round: cap.evaluate(zoom, feature as usize) == LineCap::Round,
                        },
                    });
                }
            }
        }
        if let Some(expression) = gradient {
            let Some([start, end]) = orb_layout::line::feature_clips(layer, feature as usize)
            else {
                continue;
            };
            let colors = (0..resolution)
                .map(|i| {
                    let t = i as f64 / (resolution - 1) as f64;
                    let context = orb_expr::EvaluationContext {
                        line_progress: start * (1.0 - t) + end * t,
                        ..Default::default()
                    };
                    let color = expression
                        .evaluate_with(&context)
                        .ok()
                        .and_then(|v| v.as_color())
                        .map_or([0.0; 4], |c| c.premultiplied());
                    pack_color(color)
                })
                .collect();
            images.push(ImageRef {
                word: (base + 19) as u32,
                key: ImageKey::Ramp { colors, nearest },
            });
        }
    }
}

pub(crate) fn paint(
    gpu: &mut dyn Gpu,
    paint: NodePaint,
    line: &orb_style::Layer,
    atlas: &mut ImageAtlas,
    namespace: u64,
    zoom: f64,
    crossfade: Crossfade,
) -> NodePaint {
    let NodePaint::Line(mut paint) = paint else {
        return paint;
    };
    if line.ty != LayerType::Line {
        return NodePaint::Line(paint);
    }
    if line.get(LINE_PATTERN).constant() != Some(None) {
        return NodePaint::Line(paint);
    }
    paint.gradient = line.ramp(LINE_GRADIENT).is_some();
    let p = line.get(LINE_DASHARRAY);
    if p.constant() != Some(None) {
        // The pin only evaluates caps per feature for data-driven dashes.
        // Constant dashes use the cap's constant fallback for their SDF.
        let per_feature = matches!(p.kind(), Kind::Source | Kind::Composite);
        if per_feature {
            paint.table_mask |= 128;
        }
        let mut resolve = |z| {
            p.evaluate(z)
                .and_then(|array| {
                    atlas
                        .resolve_key(
                            gpu,
                            namespace,
                            &ImageKey::Dash {
                                array: array.iter().map(|v| v.to_bits()).collect(),
                                round: line.get(LINE_CAP).evaluate(zoom) == LineCap::Round,
                            },
                        )
                        .ok()
                })
                .unwrap_or(0)
        };
        paint.dash = Some(PatternPaint {
            from: if per_feature {
                0
            } else {
                resolve(
                    zoom.floor()
                        + if crossfade.from_scale > 1.0 {
                            -1.0
                        } else {
                            1.0
                        },
                )
            },
            to: if per_feature {
                0
            } else {
                resolve(zoom.floor())
            },
            from_scale: crossfade.from_scale,
            to_scale: 1.0,
            mix: crossfade.mix,
            floor_width: line.get(LINE_WIDTH).evaluate(zoom.floor()),
        });
    }
    NodePaint::Line(paint)
}
