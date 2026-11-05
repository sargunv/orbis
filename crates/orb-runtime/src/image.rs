//! Image tiles on the scene side: raster and DEM tiles into their source's
//! texture page as one quad per render node, DEM neighbours for the
//! prepare pass, and the paint of raster, hillshade and color-relief nodes.

use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_layout::{Chunk, TileBox};
use orb_render::{
    ColorReliefPaint, ContourPaint, HillshadeMethod, HillshadePaint, IMAGE_EXTENT, ImageMesh,
    MAX_LIGHTS, NodePaint, RasterPaint, Renderer,
};
use orb_style::{Color, IlluminationAnchor, LayerType, PropertyRef, PropertyValue, Resampling};

use crate::style::{NodePlan, StylePlan};

/// What the map keeps of a resident image tile: the mesh that warps it
/// onto another plane or bends it on a globe, if any. The renderer holds its texels and quads.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidentImage {
    pub(crate) warped: Option<Arc<Warped>>,
    /// Whether the mesh is for a globe surface.
    pub(crate) globe: bool,
}

/// A tile's warp mesh as its nodes draw it (plan §9.2, `RasterWarp`).
#[derive(Debug, PartialEq)]
pub(crate) struct Warped {
    /// The square of the drawing plane its positions span `IMAGE_EXTENT`
    /// of.
    pub(crate) frame: orb_crs::WorldRect,
    pub(crate) mesh: Arc<ImageMesh>,
    /// Each side's refinement for its neighbours ([`orb_crs::Warp::mesh`]).
    pub(crate) edges: [u8; 4],
}

impl Warped {
    /// The tile's mesh from `warp`, its linear interpolation within a
    /// quarter of a pixel at 512 pixels per tile side, its sides refined
    /// by `edges`.
    pub(crate) fn new(warp: &orb_crs::Warp, tile: TileIndex, edges: [u8; 4]) -> Option<Self> {
        let mesh = warp.mesh(tile, 1.0 / 2048.0, 1.0, edges)?;
        Some(Self {
            edges,
            ..Self::of(mesh, None)
        })
    }

    /// A tile's own square (`frame`) as a grid of `cells` per side, for a
    /// globe to bend, with its top or bottom edge fanned out to the North
    /// or South Pole (caps) as `poles` ask: each cap triangle stretches its
    /// edge's texels to the pole, as GL JS's do.
    pub(crate) fn grid(frame: orb_crs::WorldRect, cells: u32, poles: [bool; 2]) -> Self {
        let n = cells + 1;
        let mut positions: Vec<glam::DVec2> = (0..n * n)
            .map(|k| glam::DVec2::new(f64::from(k % n), f64::from(k / n)) / f64::from(cells))
            .collect();
        let mut sources = positions.clone();
        let mut triangles: Vec<[u32; 3]> = (0..cells * cells)
            .flat_map(|c| {
                let k = c / cells * n + c % cells;
                [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
            })
            .collect();
        let pole_y = [
            orb_layout::fill::NORTH_POLE_Y,
            orb_layout::fill::SOUTH_POLE_Y,
        ];
        for ((cap, row), pole) in poles.into_iter().zip([0, cells]).zip(pole_y) {
            if !cap {
                continue;
            }
            // Positions in image units, where the shader reads the pole.
            let y = f64::from(pole) / f64::from(IMAGE_EXTENT);
            for i in 0..cells {
                let tip = positions.len() as u32;
                let middle = (f64::from(i) + 0.5) / f64::from(cells);
                positions.push(glam::DVec2::new(middle, y));
                sources.push(glam::DVec2::new(middle, f64::from(row) / f64::from(cells)));
                triangles.push([row * n + i, row * n + i + 1, tip]);
            }
        }
        Self::of(
            orb_crs::WarpMesh {
                frame,
                sources,
                positions,
                triangles,
            },
            None,
        )
    }

    /// An image source's mesh: its quad's warp at the vertices of a grid
    /// of `cells` per side over the image, in the tile `frame`, with the
    /// image's own positions as sources (GL JS warps per vertex too).
    pub(crate) fn quad(frame: orb_crs::WorldRect, quad: &orb_crs::ImageQuad, cells: u32) -> Self {
        let n = cells + 1;
        let sources: Vec<glam::DVec2> = (0..n * n)
            .map(|k| glam::DVec2::new(f64::from(k % n), f64::from(k / n)) / f64::from(cells))
            .collect();
        let size = frame.size();
        let positions = sources
            .iter()
            .map(|&uv| {
                let p = quad.at(uv);
                glam::DVec2::new((p.x - frame.min.x) / size.x, (frame.max.y - p.y) / size.y)
            })
            .collect();
        let triangles = (0..cells * cells)
            .flat_map(|c| {
                let k = c / cells * n + c % cells;
                [[k, k + 1, k + n + 1], [k, k + n + 1, k + n]]
            })
            .collect();
        let weights: Vec<f32> = sources.iter().map(|&uv| quad.weight(uv) as f32).collect();
        Self::of(
            orb_crs::WarpMesh {
                frame,
                positions,
                sources,
                triangles,
            },
            Some(&weights),
        )
    }

    fn of(mesh: orb_crs::WarpMesh, weights: Option<&[f32]>) -> Self {
        let quad = weights.is_some();
        let units = |v: f64| (v * f64::from(IMAGE_EXTENT)).round() as i16 as u16 as u32;
        let position = |v: f64| ((v * f64::from(IMAGE_EXTENT)) as f32).to_bits();
        let vertices = mesh
            .positions
            .iter()
            .zip(&mesh.sources)
            .enumerate()
            .flat_map(|(k, (p, s))| {
                let weight = weights.map_or(1.0, |w| w[k]);
                let image = units(s.x) | units(s.y) << 16;
                // Clipping is in the image's tile units, but an image
                // source's grid sits in its tile's own units.
                let clip = if quad {
                    units(p.x) | units(p.y) << 16
                } else {
                    image
                };
                [position(p.x), position(p.y), clip, weight.to_bits(), image]
            })
            .collect();
        let indices: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
        let point = |k: u32| mesh.positions[k as usize] * f64::from(IMAGE_EXTENT);
        let chunks = mesh
            .triangles
            .chunks(256)
            .enumerate()
            .map(|(c, triangles)| {
                let mut min = glam::DVec2::splat(f64::INFINITY);
                let mut max = glam::DVec2::splat(f64::NEG_INFINITY);
                for p in triangles.iter().flatten().map(|&k| point(k)) {
                    min = min.min(p);
                    max = max.max(p);
                }
                // Bounds round outward, so the cull keeps every pixel.
                let min = [min.x.floor() as i16, min.y.floor() as i16];
                let max = [max.x.ceil() as i16, max.y.ceil() as i16];
                Chunk {
                    sort_key: None,
                    first_index: (c * 256 * 3) as u32,
                    index_count: (triangles.len() * 3) as u32,
                    bounds: TileBox { min, max },
                }
            })
            .collect();
        Self {
            frame: mesh.frame,
            mesh: Arc::new(ImageMesh {
                vertices,
                indices,
                chunks,
            }),
            edges: [0; 4],
        }
    }
}

/// An image node's paint at `zoom` with the map rotated by `bearing`
/// degrees. A color-relief node's ramp is written first, once its source's
/// `unpack` vector is known; until then it has no paint.
pub(crate) fn paint(
    gpu: &mut dyn Gpu,
    renderer: &mut Renderer,
    (plan, node, index): (&StylePlan, &NodePlan, u32),
    zoom: f64,
    bearing: f64,
    unpack: Option<[f64; 4]>,
    ramp: &mut Option<u32>,
) -> Option<NodePaint> {
    use orb_style::{
        COLOR_RELIEF_COLOR, COLOR_RELIEF_OPACITY, COLOR_RELIEF_RESAMPLE, HILLSHADE_ACCENT_COLOR,
        HILLSHADE_EXAGGERATION, HILLSHADE_HIGHLIGHT_COLOR, HILLSHADE_ILLUMINATION_ALTITUDE,
        HILLSHADE_ILLUMINATION_ANCHOR, HILLSHADE_ILLUMINATION_DIRECTION, HILLSHADE_METHOD,
        HILLSHADE_RESAMPLE, HILLSHADE_SHADOW_COLOR, Key, RASTER_BRIGHTNESS_MAX,
        RASTER_BRIGHTNESS_MIN, RASTER_CONTRAST, RASTER_FADE_DURATION, RASTER_HUE_ROTATE,
        RASTER_OPACITY, RASTER_RESAMPLE, RASTER_RESAMPLING, RASTER_SATURATION,
    };
    let l = &plan.style.layers[node.layer];
    let nearest = |key: Key<Resampling>| l.get(key).evaluate(zoom) == Resampling::Nearest;
    let number = |key: Key<f32>| l.get(key).evaluate(zoom);
    Some(match l.ty {
        LayerType::Raster => {
            let image_source = matches!(
                plan.source_def(node.source).kind,
                orb_style::SourceKind::Image(_)
            );
            NodePaint::Raster(RasterPaint {
                opacity: number(RASTER_OPACITY),
                hue_rotate: number(RASTER_HUE_ROTATE),
                brightness_min: number(RASTER_BRIGHTNESS_MIN),
                brightness_max: number(RASTER_BRIGHTNESS_MAX),
                saturation: number(RASTER_SATURATION),
                contrast: number(RASTER_CONTRAST),
                nearest: nearest(RASTER_RESAMPLE) || nearest(RASTER_RESAMPLING),
                // Image sources are one texture without mip levels in GL JS,
                // and don't fade.
                mipmapped: !image_source,
                fade_duration: if image_source {
                    0.0
                } else {
                    number(RASTER_FADE_DURATION).max(0.0)
                },
            })
        }
        LayerType::Hillshade => {
            let directions = l.list(HILLSHADE_ILLUMINATION_DIRECTION);
            let altitudes = l.list(HILLSHADE_ILLUMINATION_ALTITUDE);
            let shadows = l.list(HILLSHADE_SHADOW_COLOR);
            let highlights = l.list(HILLSHADE_HIGHLIGHT_COLOR);
            let lights = [
                directions.len(),
                altitudes.len(),
                shadows.len(),
                highlights.len(),
            ]
            .into_iter()
            .max()
            .unwrap_or(1)
            .clamp(1, MAX_LIGHTS);
            // GL JS repeats each array's last value up to the longest.
            fn per_light<T: PropertyValue, U>(
                values: &[PropertyRef<'_, T>],
                zoom: f64,
                f: impl Fn(T) -> U,
            ) -> [U; MAX_LIGHTS] {
                std::array::from_fn(|i| f(values[i.min(values.len() - 1)].evaluate(zoom)))
            }
            let turn = if l.get(HILLSHADE_ILLUMINATION_ANCHOR).evaluate(zoom)
                == IlluminationAnchor::Viewport
            {
                bearing.to_radians() as f32
            } else {
                0.0
            };
            let color = |c: Color| c.premultiplied();
            NodePaint::Hillshade(HillshadePaint {
                accent: l.get(HILLSHADE_ACCENT_COLOR).evaluate(zoom).premultiplied(),
                shadows: per_light(&shadows, zoom, color),
                highlights: per_light(&highlights, zoom, color),
                altitudes: per_light(&altitudes, zoom, f32::to_radians),
                azimuths: per_light(&directions, zoom, |d| d.to_radians() + turn),
                exaggeration: number(HILLSHADE_EXAGGERATION),
                method: match l.get(HILLSHADE_METHOD).evaluate(zoom) {
                    orb_style::HillshadeMethod::Standard => HillshadeMethod::Standard,
                    orb_style::HillshadeMethod::Basic => HillshadeMethod::Basic,
                    orb_style::HillshadeMethod::Combined => HillshadeMethod::Combined,
                    orb_style::HillshadeMethod::Igor => HillshadeMethod::Igor,
                    orb_style::HillshadeMethod::Multidirectional => {
                        HillshadeMethod::Multidirectional
                    }
                },
                lights: lights as u32,
                nearest: nearest(HILLSHADE_RESAMPLE),
            })
        }
        LayerType::ColorRelief => {
            let unpack = unpack?;
            let size = match *ramp {
                Some(size) => size,
                None => {
                    let max = gpu.caps().limits.max_texture_dimension_2d as usize;
                    let (colors, elevations) = color_ramp(l.ramp(COLOR_RELIEF_COLOR), unpack, max);
                    renderer.set_ramp(gpu, index, &colors, &elevations);
                    *ramp.insert(colors.len() as u32)
                }
            };
            NodePaint::ColorRelief(ColorReliefPaint {
                opacity: number(COLOR_RELIEF_OPACITY),
                unpack: unpack.map(|v| v as f32),
                ramp_size: size,
                nearest: nearest(COLOR_RELIEF_RESAMPLE),
            })
        }
        LayerType::Contour => {
            use orb_style::{
                CONTOUR_COLOR, CONTOUR_INTERVAL, CONTOUR_MAJOR, CONTOUR_MAJOR_WIDTH,
                CONTOUR_OPACITY, CONTOUR_WIDTH,
            };
            let opacity = number(CONTOUR_OPACITY);
            NodePaint::Contour(ContourPaint {
                unpack: unpack?.map(|v| v as f32),
                color: l
                    .get(CONTOUR_COLOR)
                    .evaluate(zoom)
                    .premultiplied()
                    .map(|c| c * opacity),
                interval: number(CONTOUR_INTERVAL),
                major: number(CONTOUR_MAJOR).round().max(1.0),
                width: number(CONTOUR_WIDTH),
                major_width: number(CONTOUR_MAJOR_WIDTH),
            })
        }
        _ => unreachable!("image nodes are raster, hillshade, color-relief or contour layers"),
    })
}

/// GL JS's `_createColorRamp` and `getColorRampTextures`: per stop, its
/// colour as the texel GL JS uploads (unpremultiplied to bytes, then
/// premultiplied again) and its elevation packed with `unpack`. A ramp
/// that isn't a top-level `interpolate` is transparent; ramps longer than
/// `max` keep `max` evenly spaced stops.
fn color_ramp(
    ramp: Option<&orb_style::PropertyExpression>,
    unpack: [f64; 4],
    max: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut stops: Vec<(f64, [f32; 4])> = Vec::new();
    if let Some(expression) = ramp.map(|r| r.expression())
        && let Some(labels) = expression.interpolate_labels()
    {
        for &label in labels {
            let context = orb_expr::EvaluationContext {
                elevation: label,
                ..orb_expr::EvaluationContext::default()
            };
            let color = expression
                .evaluate_with(&context)
                .ok()
                .and_then(|v| v.as_color())
                .map_or([0.0; 4], |c| c.premultiplied());
            stops.push((label, color));
        }
    }
    if stops.is_empty() {
        stops.push((0.0, [0.0; 4]));
    }
    if stops.len() < 2 {
        stops.push((stops[0].0 + 1.0, stops[0].1));
    }
    if stops.len() > max {
        let step = (stops.len() - 1) as f64 / (max - 1) as f64;
        let mut remapped = Vec::with_capacity(max);
        let mut i = 0.0;
        while i < stops.len() as f64 - 0.5 {
            remapped.push(stops[i.round() as usize]);
            i += step;
        }
        stops = remapped;
    }
    let texel = |[r, g, b, a]: [f32; 4]| {
        let alpha = (a * 255.0).round();
        let channel = |c: f32| {
            let straight = if a > 0.0 {
                (c * 255.0 / a).round()
            } else {
                0.0
            };
            (straight * alpha / 255.0).round() as u32
        };
        channel(r) | channel(g) << 8 | channel(b) << 16 | (alpha as u32) << 24
    };
    stops
        .iter()
        .map(|&(elevation, color)| {
            let [r, g, b] = pack_dem(elevation, unpack);
            (texel(color), r | g << 8 | b << 16 | 0xff << 24)
        })
        .unzip()
}

/// GL JS's `packDEMData`: an elevation as the RGB bytes `unpack` decodes.
/// Channels whose factor is 0 pack to 0.
fn pack_dem(elevation: f64, [red, green, blue, base_shift]: [f64; 4]) -> [u32; 3] {
    let factors = [red, green, blue];
    let Some(min_scale) = factors
        .into_iter()
        .filter(|&f| f != 0.0)
        .min_by(f64::total_cmp)
    else {
        return [0; 3];
    };
    let scaled = ((elevation + base_shift) / min_scale).round();
    factors.map(|factor| {
        if factor == 0.0 {
            0
        } else {
            ((scaled * min_scale / factor).floor() as i64).rem_euclid(256) as u32
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_elevations_like_gl_js() {
        let mapbox = orb_style::DemEncoding::Mapbox.unpack([0.0; 4]);
        // (v + 10000) / 0.1 = 110000 = 1·65536 + 173·256 + 176.
        assert_eq!(pack_dem(1000.0, mapbox), [1, 173, 176]);
        assert_eq!(pack_dem(5.0, [0.0, 0.0, 0.0, 1.0]), [0, 0, 0]);
        // A zero factor packs to 0 and leaves the others' scale alone.
        assert_eq!(pack_dem(1000.0, [0.0, 25.6, 0.1, 3446.4]), [0, 173, 176]);
    }

    #[test]
    fn ramps_pad_and_quantize_like_gl_js() {
        let (colors, elevations) = color_ramp(None, [6553.6, 25.6, 0.1, 10000.0], 8192);
        assert_eq!(colors, [0, 0]);
        assert_eq!(elevations.len(), 2);
        let json = orb_expr::json::parse(
            r##"["interpolate", ["linear"], ["elevation"], 0, "rgba(255, 0, 0, 0.5)", 100, "#00f"]"##,
        )
        .unwrap();
        let spec = orb_expr::PropertySpec {
            ty: orb_expr::PropertyType::Color,
            default: None,
            zoom: false,
            feature: false,
            interpolated: true,
            tokens: false,
        };
        let ramp = orb_style::PropertyExpression::parse(&json, &spec).unwrap();
        let (colors, _) = color_ramp(Some(&ramp), [6553.6, 25.6, 0.1, 10000.0], 8192);
        // Half-transparent red: 255 straight, alpha 128, premultiplied 128.
        assert_eq!(colors, [128 | 128 << 24, 0xffff_0000]);
    }
}
