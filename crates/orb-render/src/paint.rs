//! Render node paint values and property rows, laid out as in `fill.wgsl`,
//! `line.wgsl`, `draw.wgsl` and the image family shaders.
//!
//! Colours are premultiplied RGBA in the style's colour space, as GL JS
//! blends them. A property whose `table_mask` bit is set comes from the
//! feature's property row instead: two words, its values at the tile's zoom
//! and the next, blended by the property's [`Curve`]. Its constant field then
//! holds the maximum over the rows, which culling uses for margins.

use crate::Family;

/// Words per feature in a fill row: color, opacity, three pattern image ids.
pub const FILL_ROW_WORDS: usize = 7;
/// Words per feature in a line property row: color, opacity, width, gap
/// width, offset, blur (pairs), three pattern ids, three dash ids, floor
/// width, gradient ramp id. The two triples cover layout zoom −1/0/+1.
pub const LINE_ROW_WORDS: usize = 20;
/// Symbol paint rows: opacity, colour, halo colour, halo width and halo
/// blur, each at the tile's zoom and the next.
pub const SYMBOL_ROW_WORDS: usize = 10;
/// Color, radius, blur, opacity, stroke color, width and opacity.
pub const CIRCLE_ROW_WORDS: usize = 14;
/// Radius and weight.
pub const HEATMAP_ROW_WORDS: usize = 4;
/// Colour, colour mix, emissive strength, then scale, rotation and
/// translation by component.
pub const MODEL_ROW_WORDS: usize = 24;

/// A model node's paint (`model.wgsl`'s `ModelNode`). Slots: colour,
/// colour mix, emissive strength, scale, rotation and translation by
/// component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelPaint {
    pub color: [f32; 4],
    pub color_mix: f32,
    pub emissive: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 3],
    pub translation: [f32; 3],
    /// Applied by the 3D composite.
    pub opacity: f32,
    pub table_mask: u32,
    pub curves: [Curve; 12],
    /// How far a model reaches from its point on screen and above the
    /// ground (metres), for culling.
    pub reach_px: f32,
    pub height: f32,
    pub state: StateLookup,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CirclePaint {
    pub color: [f32; 4],
    pub stroke_color: [f32; 4],
    pub radius: f32,
    pub blur: f32,
    pub opacity: f32,
    pub stroke_width: f32,
    pub stroke_opacity: f32,
    pub translate: Translate,
    pub pitch_scale_map: bool,
    pub pitch_alignment_map: bool,
    pub table_mask: u32,
    pub curves: [Curve; 7],
    pub state: StateLookup,
}

/// Where a node's feature-state lookups read (plan §10.2): bit `i` of
/// `mask` takes property slot `i` from a feature's state block when the
/// feature has state, else from the node's constant. The node's block of
/// state row `s` starts at word `base + s * stride` of the property arena;
/// its first word is non-zero once the feature has state, followed by two
/// words per masked slot, the values at `zoom` and `zoom + 1`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StateLookup {
    pub mask: u32,
    pub base: u32,
    pub stride: u32,
    pub zoom: f32,
}

impl StateLookup {
    fn words(self) -> [u32; 4] {
        [self.mask, self.base, self.stride, self.zoom.to_bits()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatmapPaint {
    pub radius: f32,
    pub weight: f32,
    pub intensity: f32,
    pub opacity: f32,
    pub table_mask: u32,
    pub curves: [Curve; 2],
}

/// How a property row's two values blend as the camera zooms past the
/// tile's zoom (GL JS's `interpolationFactor`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Curve {
    /// Always the first value: feature-only properties and `step` curves.
    #[default]
    Step,
    /// Exponential with this base; 1 is linear.
    Exponential(f32),
    CubicBezier([f32; 4]),
}

impl Curve {
    fn words(self) -> [u32; 8] {
        let (kind, params) = match self {
            Self::Step => (0, [0.0; 4]),
            Self::Exponential(base) => (1, [base, 0.0, 0.0, 0.0]),
            Self::CubicBezier(p) => (2, p),
        };
        let [a, b, c, d] = params.map(f32::to_bits);
        [a, b, c, d, kind, 0, 0, 0]
    }
}

/// A `*-translate` offset in pixels and its anchor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Translate {
    pub px: [f32; 2],
    /// Turn with the map's bearing (`translate-anchor: viewport`).
    pub viewport: bool,
}

impl Translate {
    fn words(self) -> [u32; 5] {
        [
            u32::from(self.viewport),
            self.px[0].to_bits(),
            self.px[1].to_bits(),
            0,
            0,
        ]
    }

    fn length(self) -> f32 {
        self.px[0].hypot(self.px[1])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillPaint {
    pub color: [f32; 4],
    pub opacity: f32,
    pub translate: Translate,
    /// Bit 0: color, bit 1: opacity from the property row.
    pub table_mask: u32,
    pub curves: [Curve; 2],
    pub state: StateLookup,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinePaint {
    pub color: [f32; 4],
    pub opacity: f32,
    pub translate: Translate,
    pub width: f32,
    pub gap_width: f32,
    pub offset: f32,
    pub blur: f32,
    /// Bits 0–5: color, opacity, width, gap width, offset, blur from the
    /// property row.
    pub table_mask: u32,
    pub curves: [Curve; 6],
    /// SDF dash crossfade, with row ids resolved by the scene.
    pub dash: Option<PatternPaint>,
    /// Sample the per-feature gradient ramp in property word 19.
    pub gradient: bool,
    pub state: StateLookup,
}

/// Raster paint as GL JS's `rasterUniformValues` derives it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RasterPaint {
    pub opacity: f32,
    /// Degrees.
    pub hue_rotate: f32,
    pub brightness_min: f32,
    pub brightness_max: f32,
    pub saturation: f32,
    pub contrast: f32,
    /// Filter by nearest texel rather than linearly.
    pub nearest: bool,
    /// Sample mip levels for the texture's footprint; `false` reads the
    /// full-size level alone, as GL JS samples image sources.
    pub mipmapped: bool,
    /// `raster-fade-duration`, milliseconds.
    pub fade_duration: f32,
}

/// The most light sources a hillshade node holds.
pub const MAX_LIGHTS: usize = 4;

/// Hillshade paint, after GL JS's `hillshadeUniformValues`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HillshadePaint {
    /// Premultiplied.
    pub accent: [f32; 4],
    /// Per light, premultiplied.
    pub shadows: [[f32; 4]; MAX_LIGHTS],
    pub highlights: [[f32; 4]; MAX_LIGHTS],
    /// Radians above the horizon, per light.
    pub altitudes: [f32; MAX_LIGHTS],
    /// Radians clockwise from north (with the map's bearing for viewport
    /// anchoring), per light.
    pub azimuths: [f32; MAX_LIGHTS],
    pub exaggeration: f32,
    pub method: HillshadeMethod,
    /// Lights in use, 1 to [`MAX_LIGHTS`].
    pub lights: u32,
    pub nearest: bool,
}

/// `hillshade-method`, numbered as GL JS's shader does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HillshadeMethod {
    Standard = 0,
    Combined = 1,
    Igor = 2,
    Multidirectional = 3,
    Basic = 4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorReliefPaint {
    pub opacity: f32,
    /// The source's DEM unpack vector.
    pub unpack: [f32; 4],
    /// Stops in the node's ramp (see [`Renderer::set_ramp`](crate::Renderer::set_ramp)).
    pub ramp_size: u32,
    pub nearest: bool,
}

/// Colour, height and base (pairs), three pattern ids, the feature's
/// centroid (two i16s), and 1 when the building is whole in the tile's
/// data, which only the placement holding its centroid draws.
pub const EXTRUSION_ROW_WORDS: usize = 11;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtrusionPaint {
    pub color: [f32; 4],
    pub height: f32,
    pub base: f32,
    pub opacity: f32,
    pub translate: Translate,
    pub vertical_gradient: bool,
    /// How much of its own colour it shows under the style's lights (0 lit
    /// fully, 1 unlit), which the frame block carries.
    pub emissive: f32,
    pub table_mask: u32,
    pub curves: [Curve; 3],
    pub state: StateLookup,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodePaint {
    Extrusion(ExtrusionPaint),
    ExtrusionPattern(ExtrusionPaint, PatternPaint),
    Fill(FillPaint),
    /// A solid background covers the entire viewport, including beyond source bounds.
    Background(FillPaint),
    Line(LinePaint),
    Circle(CirclePaint),
    Heatmap(HeatmapPaint),
    Raster(RasterPaint),
    Hillshade(HillshadePaint),
    ColorRelief(ColorReliefPaint),
    FillPattern(FillPaint, PatternPaint),
    LinePattern(LinePaint, PatternPaint),
    Symbol(SymbolPaint),
    Model(ModelPaint),
}

/// How a node draws its symbols' sizes per frame (GL JS `SizeData` with
/// `evaluateSizeForZoom`); a tile's layout size and covering zooms are in
/// its mesh header.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SymbolSize {
    Constant(f32),
    /// Per feature, from the placed symbol.
    #[default]
    Source,
    /// Per zoom: the curve over the first `count` `(zoom, size)` stops.
    Camera {
        curve: Curve,
        stops: [[f32; 2]; SymbolSize::MAX_STOPS],
        count: usize,
    },
    /// Per feature and zoom: the placed symbol's sizes at the tile's
    /// covering zooms, blended by the curve.
    Composite {
        curve: Curve,
    },
}

impl SymbolSize {
    /// At most this many camera stops are kept; GL JS has no limit.
    pub const MAX_STOPS: usize = 8;

    fn words(&self) -> Vec<u32> {
        let f = f32::to_bits;
        let (kind, size, curve, stops) = match self {
            Self::Constant(size) => (0, *size, Curve::Step, &[][..]),
            Self::Source => (1, 0.0, Curve::Step, &[][..]),
            Self::Camera {
                curve,
                stops,
                count,
            } => (2, 0.0, *curve, &stops[..(*count).min(Self::MAX_STOPS)]),
            Self::Composite { curve } => (3, 0.0, *curve, &[][..]),
        };
        let mut words = vec![kind, f(size), stops.len() as u32, 0];
        words.extend(curve.words());
        let mut table = [0u32; 16];
        for (i, &[zoom, size]) in stops.iter().enumerate() {
            table[i * 2] = f(zoom);
            table[i * 2 + 1] = f(size);
        }
        words.extend(table);
        words
    }
}

/// A symbol node's paint and drawing mode (`symbol.wgsl`'s `SymbolNode`).
/// Slots: opacity, colour, halo colour, halo width, halo blur.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymbolPaint {
    pub color: [f32; 4],
    pub halo_color: [f32; 4],
    pub opacity: f32,
    pub halo_width: f32,
    pub halo_blur: f32,
    pub table_mask: u32,
    pub curves: [Curve; 5],
    pub translate: Translate,
    pub size: SymbolSize,
    /// For icons fitted to variable-anchored text: the text's size.
    pub text_size: SymbolSize,
    pub is_text: bool,
    pub pitch_with_map: bool,
    pub rotate_with_map: bool,
    pub along_line: bool,
    /// Point labels rotated with the map but not pitched.
    pub rotate_in_shader: bool,
    pub variable_anchor: bool,
    /// The layer sets `icon-offset` (GL JS `u_is_offset`): sizes skip the
    /// perspective ratio.
    pub is_offset: bool,
    pub keep_upright: bool,
    /// `symbol-height-anchor: ground`: heights count from the terrain.
    pub height_anchor_ground: bool,
    /// Halos are drawn in a pass of their own, under every fill (GL JS
    /// `isGlyphOverlap`).
    pub separate_halo: bool,
    /// How far labels reach from their anchors, pixels, for culling.
    pub reach_px: f32,
    /// Images sample linearly even when their bucket needs no filtering:
    /// scaled icons, pitched views (GL JS `atlasInterpolation`).
    pub linear_icons: bool,
    pub state: StateLookup,
}

/// Pattern endpoints and scales, after GL JS CrossFadedProperty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatternPaint {
    pub from: u32,
    pub to: u32,
    pub from_scale: f32,
    pub to_scale: f32,
    pub mix: f32,
    pub floor_width: f32,
}

/// Packs a premultiplied colour into a property row word (rgba8).
pub fn pack_color(color: [f32; 4]) -> u32 {
    let [r, g, b, a] = color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u32);
    r | g << 8 | b << 16 | a << 24
}

impl NodePaint {
    /// Whether this paint fits a node of `family`.
    pub(crate) fn fits(&self, family: Family) -> bool {
        matches!(
            (self, family),
            (
                Self::Fill(_) | Self::Background(_) | Self::FillPattern(..),
                Family::Fill | Family::FillOutline
            ) | (Self::Line(_) | Self::LinePattern(..), Family::Line)
                | (
                    Self::Extrusion(_) | Self::ExtrusionPattern(..),
                    Family::FillExtrusion
                )
                | (Self::Circle(_), Family::Circle)
                | (Self::Heatmap(_), Family::Heatmap)
                | (Self::Raster(_), Family::Raster)
                | (Self::Hillshade(_), Family::Hillshade)
                | (Self::ColorRelief(_), Family::ColorRelief)
                | (Self::Symbol(_), Family::SymbolIcon | Family::SymbolText)
                | (Self::Model(_), Family::Model)
        )
    }

    pub(crate) fn uniform_words(&self, node: u32) -> Vec<u32> {
        let f = f32::to_bits;
        let (mut words, curves): (Vec<u32>, &[Curve]) = match self {
            Self::Fill(p) | Self::Background(p) | Self::FillPattern(p, _) => {
                let [r, g, b, a] = p.color.map(f);
                let mut words = vec![r, g, b, a, f(p.opacity), node, p.table_mask];
                words.extend(p.translate.words());
                (words, &p.curves)
            }
            Self::Extrusion(p) | Self::ExtrusionPattern(p, _) => {
                let mut words = p.color.map(f).to_vec();
                words.extend([f(p.height), f(p.base), node, p.table_mask]);
                words.extend([
                    f(p.translate.px[0]),
                    f(p.translate.px[1]),
                    u32::from(p.translate.viewport),
                    u32::from(p.vertical_gradient),
                ]);
                words.extend([f(p.opacity), 0, 0, f(p.emissive)]);
                (words, &p.curves)
            }
            Self::Line(p) | Self::LinePattern(p, _) => {
                let [r, g, b, a] = p.color.map(f);
                let mut words = vec![
                    r,
                    g,
                    b,
                    a,
                    f(p.opacity),
                    f(p.width),
                    f(p.gap_width),
                    f(p.offset),
                    f(p.blur),
                    node,
                    p.table_mask,
                ];
                words.extend(p.translate.words());
                (words, &p.curves)
            }
            Self::Circle(p) => {
                let mut words = p.color.map(f).to_vec();
                words.extend(p.stroke_color.map(f));
                words.extend([
                    f(p.radius),
                    f(p.blur),
                    f(p.opacity),
                    f(p.stroke_width),
                    f(p.stroke_opacity),
                    node,
                    p.table_mask,
                    u32::from(p.translate.viewport)
                        | u32::from(p.pitch_scale_map) << 1
                        | u32::from(p.pitch_alignment_map) << 2,
                    f(p.translate.px[0]),
                    f(p.translate.px[1]),
                    0,
                    0,
                ]);
                (words, &p.curves)
            }
            Self::Model(p) => {
                let mut words = p.color.map(f).to_vec();
                words.extend([f(p.color_mix), f(p.emissive), node, p.table_mask]);
                words.extend(p.scale.map(f));
                words.push(f(p.opacity));
                words.extend(p.rotation.map(f));
                words.push(0);
                words.extend(p.translation.map(f));
                words.push(0);
                (words, &p.curves)
            }
            Self::Heatmap(p) => (
                vec![
                    f(p.radius),
                    f(p.weight),
                    f(p.intensity),
                    f(p.opacity),
                    node,
                    p.table_mask,
                    0,
                    0,
                ],
                &p.curves,
            ),
            Self::Raster(p) => {
                // GL JS's `spinWeights`, `saturationFactor` and
                // `contrastFactor`.
                let angle = p.hue_rotate.to_radians();
                let (s, c) = angle.sin_cos();
                let root3 = 3f32.sqrt();
                let spin = [
                    (2.0 * c + 1.0) / 3.0,
                    (-root3 * s - c + 1.0) / 3.0,
                    (root3 * s - c + 1.0) / 3.0,
                    0.0,
                ];
                let saturation = if p.saturation > 0.0 {
                    1.0 - 1.0 / (1.001 - p.saturation)
                } else {
                    -p.saturation
                };
                let contrast = if p.contrast > 0.0 {
                    1.0 / (1.0 - p.contrast)
                } else {
                    1.0 + p.contrast
                };
                let mut words: Vec<u32> = spin.map(f).to_vec();
                words.extend([
                    f(p.opacity),
                    f(p.brightness_min),
                    f(p.brightness_max),
                    f(saturation),
                    f(contrast),
                    node,
                    u32::from(p.nearest),
                    u32::from(!p.mipmapped),
                    f(p.fade_duration),
                ]);
                (words, &[][..])
            }
            Self::Hillshade(p) => {
                let mut words: Vec<u32> = p.accent.map(f).to_vec();
                words.extend(p.shadows.iter().flatten().map(|&v| f(v)));
                words.extend(p.highlights.iter().flatten().map(|&v| f(v)));
                words.extend(p.altitudes.map(f));
                words.extend(p.azimuths.map(f));
                words.extend([
                    f(p.exaggeration),
                    p.method as u32,
                    p.lights,
                    node,
                    u32::from(p.nearest),
                    0,
                    0,
                    0,
                ]);
                (words, &[][..])
            }
            Self::ColorRelief(p) => {
                let mut words: Vec<u32> = p.unpack.map(f).to_vec();
                words.extend([f(p.opacity), p.ramp_size, node, u32::from(p.nearest)]);
                (words, &[][..])
            }
            Self::Symbol(p) => {
                let flags = [
                    p.is_text,
                    p.linear_icons,
                    p.pitch_with_map,
                    p.rotate_with_map,
                    p.along_line,
                    p.rotate_in_shader,
                    p.variable_anchor,
                    p.is_offset,
                    p.keep_upright,
                    p.translate.viewport,
                ]
                .iter()
                .enumerate()
                .fold(0u32, |w, (i, &b)| w | u32::from(b) << i)
                    | u32::from(p.height_anchor_ground) << 11;
                let mut words = p.color.map(f).to_vec();
                words.extend(p.halo_color.map(f));
                words.extend([
                    f(p.opacity),
                    f(p.halo_width),
                    f(p.halo_blur),
                    node,
                    p.table_mask,
                    flags,
                    // Glyph table, page sizes: the renderer fills these.
                    0,
                    0,
                    f(p.translate.px[0]),
                    f(p.translate.px[1]),
                    0,
                    0,
                ]);
                words.extend(p.size.words());
                words.extend(p.text_size.words());
                (words, &p.curves)
            }
        };
        for curve in curves {
            words.extend(curve.words());
        }
        if let Self::Circle(p) = self {
            words.extend(p.state.words());
        }
        if let Self::Model(p) = self {
            words.extend(p.state.words());
        }
        if let Self::Symbol(p) = self {
            words.extend(p.state.words());
        }
        if !matches!(
            self,
            Self::Fill(_)
                | Self::Background(_)
                | Self::Line(_)
                | Self::FillPattern(..)
                | Self::LinePattern(..)
                | Self::Extrusion(_)
                | Self::ExtrusionPattern(..)
        ) {
            return words;
        }
        let pattern = match self {
            Self::FillPattern(_, p) | Self::LinePattern(_, p) | Self::ExtrusionPattern(_, p) => {
                Some(p)
            }
            _ => None,
        };
        let at = match self {
            Self::Fill(_) | Self::Background(_) | Self::FillPattern(..) => 10,
            Self::Extrusion(_) | Self::ExtrusionPattern(..) => 13,
            _ => 14,
        };
        if let Some(p) = pattern {
            words[at] = p.from;
            words[at + 1] = p.to;
            words.extend([
                f(p.from_scale),
                f(p.to_scale),
                f(p.mix),
                1,
                f(p.floor_width),
                0,
                0,
                0,
            ]);
        } else {
            words.extend([f(1.0), f(1.0), f(1.0), 0, 0, 0, 0, 0]);
        }
        if let Self::Line(p) | Self::LinePattern(p, _) = self {
            let dash = p.dash.unwrap_or(PatternPaint {
                from: 0,
                to: 0,
                from_scale: 1.0,
                to_scale: 1.0,
                mix: 1.0,
                floor_width: 1.0,
            });
            words.extend([
                dash.from,
                dash.to,
                f(dash.from_scale),
                f(dash.mix),
                f(dash.floor_width),
                u32::from(p.dash.is_some()),
                u32::from(p.gradient),
                0,
            ]);
        }
        let state = match self {
            Self::Fill(p) | Self::Background(p) | Self::FillPattern(p, _) => p.state,
            Self::Line(p) | Self::LinePattern(p, _) => p.state,
            Self::Extrusion(p) | Self::ExtrusionPattern(p, _) => p.state,
            _ => unreachable!("returned above"),
        };
        words.extend(state.words());
        words
    }

    /// How far drawn pixels can reach past a chunk's bounds.
    pub(crate) fn cull_margin_px(&self) -> f32 {
        match self {
            Self::Circle(p) => p.radius + p.stroke_width + p.translate.length() + 1.0,
            Self::Heatmap(p) => {
                let amplitude = p.weight * p.intensity * 0.3989423;
                let scale = (-2.0 * (1.0 / 255.0 / 16.0 / amplitude.max(1e-20)).ln())
                    .max(0.0)
                    .sqrt()
                    / 3.0;
                p.radius * scale + 1.0
            }
            Self::Fill(p) | Self::Background(p) | Self::FillPattern(p, _) => p.translate.length(),
            Self::Extrusion(p) | Self::ExtrusionPattern(p, _) => p.translate.length(),
            Self::Line(p) | Self::LinePattern(p, _) => {
                let half_gap = p.gap_width / 2.0;
                let outset = if half_gap > 0.0 {
                    half_gap + p.width
                } else {
                    p.width / 2.0
                };
                outset + p.offset.abs() + p.blur + p.translate.length() + 1.0
            }
            Self::Raster(_) | Self::Hillshade(_) | Self::ColorRelief(_) => 0.0,
            Self::Symbol(p) => p.reach_px + p.translate.length(),
            Self::Model(p) => p.reach_px,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_fit_their_slot() {
        let line = NodePaint::Line(LinePaint {
            color: [0.0; 4],
            opacity: 1.0,
            translate: Translate::default(),
            width: 1.0,
            gap_width: 0.0,
            offset: 0.0,
            blur: 0.0,
            table_mask: 0,
            curves: [Curve::Step; 6],
            dash: None,
            gradient: false,
            state: StateLookup::default(),
        });
        // color, 8 scalars, anchor, translate and padding, then 32-byte
        // curves, pattern, dash and state lookup, as `LineNode`.
        assert_eq!(line.uniform_words(0).len() * 4, 64 + 6 * 32 + 32 + 32 + 16);
        assert!(line.uniform_words(0).len() * 4 <= crate::NODE_UNIFORM_SIZE as usize);
    }

    #[test]
    fn image_uniforms_match_their_structs() {
        let hillshade = NodePaint::Hillshade(HillshadePaint {
            accent: [0.0; 4],
            shadows: [[0.0; 4]; MAX_LIGHTS],
            highlights: [[0.0; 4]; MAX_LIGHTS],
            altitudes: [0.0; MAX_LIGHTS],
            azimuths: [0.0; MAX_LIGHTS],
            exaggeration: 0.5,
            method: HillshadeMethod::Standard,
            lights: 1,
            nearest: false,
        });
        // `HillshadeNode`: accent, 4 shadows, 4 highlights, altitudes,
        // azimuths, then 8 scalars.
        assert_eq!(hillshade.uniform_words(0).len() * 4, 208);
        let raster = NodePaint::Raster(RasterPaint {
            opacity: 1.0,
            hue_rotate: 0.0,
            brightness_min: 0.0,
            brightness_max: 1.0,
            saturation: 0.0,
            contrast: 0.0,
            nearest: false,
            mipmapped: true,
            fade_duration: 300.0,
        });
        assert_eq!(raster.uniform_words(0).len() * 4, 52);
        // No hue rotation is the identity.
        let words = raster.uniform_words(0);
        let spin: Vec<f32> = words[..3].iter().map(|&w| f32::from_bits(w)).collect();
        assert!((spin[0] - 1.0).abs() < 1e-6 && spin[1].abs() < 1e-6 && spin[2].abs() < 1e-6);
    }

    #[test]
    fn model_uniform_matches_its_struct() {
        let model = NodePaint::Model(ModelPaint {
            color: [1.0; 4],
            color_mix: 0.0,
            emissive: 0.0,
            scale: [1.0; 3],
            rotation: [0.0; 3],
            translation: [0.0; 3],
            opacity: 1.0,
            table_mask: 0,
            curves: [Curve::Step; 12],
            reach_px: 0.0,
            height: 0.0,
            state: StateLookup::default(),
        });
        // `ModelNode`: colour, four scalars, three vec3s each padded to 16
        // bytes (the scale's padding is the opacity), 12 curves, the state
        // lookup.
        let words = model.uniform_words(7);
        assert_eq!(words.len() * 4, 80 + 12 * 32 + 16);
        assert_eq!(words[6], 7);
        assert_eq!(f32::from_bits(words[11]), 1.0);
    }

    #[test]
    fn packs_colors_as_rgba8() {
        assert_eq!(pack_color([1.0, 0.0, 0.5, 1.0]), 0xff80_00ff);
    }
}
