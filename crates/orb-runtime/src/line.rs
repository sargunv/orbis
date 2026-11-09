//! Line textures in the tile job: per feature, the keys of its dashes and
//! gradient ramp. The scene's shared atlas allocates them, as it does
//! named sprite images.

use orb_layout::{ImageKey, pack_color};
use orb_shaders::layout;
use orb_style::{
    LINE_CAP, LINE_DASHARRAY, LINE_GRADIENT, LINE_PATTERN, LINE_WIDTH, LayerType, LineCap,
};
use orb_tile::Layer;

use crate::tile::ImageRef;

/// Whether a line layer's property rows carry dash or gradient textures.
pub(crate) fn has_textures(layer: &orb_style::Layer) -> bool {
    layer.ty == LayerType::Line
        && (layer.get(LINE_DASHARRAY).constant() != Some(None)
            || layer.ramp(LINE_GRADIENT).is_some())
}

/// `overscaled` is the data tile an overzoom slice's lines lay out along
/// (`Layer::overscaled_lines`): a step gradient's resolution follows its
/// lines, as GL JS's overscaled tile's does.
#[allow(clippy::too_many_arguments)]
pub(crate) fn textures(
    line: &orb_style::Layer,
    layer: &Layer,
    features: &[u32],
    zoom: f64,
    source_maxzoom: u8,
    overscaled: Option<&Layer>,
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
    let lines = overscaled.unwrap_or(layer);
    let max_length = features
        .iter()
        .map(|&f| {
            let f = match overscaled {
                Some(_) => layer.data_feature(f as usize),
                None => f,
            };
            lines
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
    let overzoom = if zoom == f64::from(source_maxzoom) || overscaled.is_some() {
        2f64.powi(22 - i32::from(source_maxzoom)).max(1.0)
    } else {
        1.0
    };
    let resolution = if nearest {
        ((max_length / f64::from(lines.extent()) * 1024.0 * overzoom)
            .clamp(256.0, 16384.0)
            .ceil() as usize)
            .next_power_of_two()
    } else {
        256
    };
    for (slot, &feature) in features.iter().enumerate() {
        let base = slot * stride;
        props[base + layout::LINE_ROW_FLOOR_WIDTH] =
            width.evaluate(zoom, feature as usize).to_bits();
        if dashed {
            for (i, z) in [zoom - 1.0, zoom, zoom + 1.0].into_iter().enumerate() {
                if let Some(array) = dash.evaluate(z, feature as usize) {
                    images.push(ImageRef {
                        word: (base + layout::LINE_ROW_DASH + i) as u32,
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
                word: (base + layout::LINE_ROW_GRADIENT) as u32,
                key: ImageKey::Ramp { colors, nearest },
            });
        }
    }
}
