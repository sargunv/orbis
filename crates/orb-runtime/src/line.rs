//! Line texture keys are job data. GPU allocation and stable ids belong to
//! the shared scene atlas, just like named sprite dependencies.

use orb_hal::Gpu;
use orb_render::{ImageAtlas, ImageKey, NodePaint, PatternPaint, pack_color};
use orb_tile::Layer;
use orb_v8::{Kind, LayerKind, LineCap};

use crate::map::Crossfade;
use crate::tile::ImageRef;

pub(crate) fn textures(
    line: &orb_v8::Line,
    layer: &Layer,
    features: &[u32],
    zoom: f64,
    source_maxzoom: u8,
    props: &mut [u32],
    images: &mut Vec<ImageRef>,
) {
    // One row per feature, of the family's words and any input columns.
    let stride = props.len() / features.len().max(1);
    if line.paint.pattern.as_constant() != Some(&None) {
        return;
    }
    let dashed = line.paint.dasharray.as_constant() != Some(&None);
    if !dashed && line.paint.gradient.is_none() {
        return;
    }
    let dash = line.paint.dasharray.bind(layer);
    let cap = line.layout.cap.bind(layer);
    let width = line.paint.width.bind(layer);
    let gradient = line.paint.gradient.as_ref().map(|g| g.expression());
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
    layer: &LayerKind,
    atlas: &mut ImageAtlas,
    namespace: u64,
    zoom: f64,
    crossfade: Crossfade,
) -> NodePaint {
    let (NodePaint::Line(mut paint), LayerKind::Line(line)) = (paint, layer) else {
        return paint;
    };
    if line.paint.pattern.as_constant() != Some(&None) {
        return NodePaint::Line(paint);
    }
    paint.gradient = line.paint.gradient.is_some();
    let p = &line.paint.dasharray;
    if p.as_constant() != Some(&None) {
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
                                round: line.layout.cap.evaluate(zoom) == LineCap::Round,
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
            floor_width: line.paint.width.evaluate(zoom.floor()),
        });
    }
    NodePaint::Line(paint)
}
