//! Render node paint values and property rows, laid out as in `fill.wgsl`,
//! `line.wgsl`, `draw.wgsl` and the image family shaders.
//!
//! Colours are premultiplied RGBA in the style's colour space, as GL JS
//! blends them. A property whose `table_mask` bit is set comes from the
//! feature's property row instead: two words, its values at the tile's zoom
//! and the next, blended by the property's [`Curve`]. Its constant field then
//! holds the maximum over the rows, which culling uses for margins.

use orb_shaders::Block;

use crate::Family;

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

/// Where a node's feature-state lookups read: bit `i` of
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

/// A `*-translate` offset in pixels and its anchor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Translate {
    pub px: [f32; 2],
    /// Turn with the map's bearing (`translate-anchor: viewport`).
    pub viewport: bool,
}

impl Translate {
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

pub use orb_shaders::layout::MAX_LIGHTS;

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

    fn put(&self, b: &mut Block, at: usize) {
        use orb_shaders::layout::symbol_size as n;
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
        b.u(at + n::KIND, &[kind])
            .f(at + n::SIZE, &[size])
            .u(at + n::STOP_COUNT, &[stops.len() as u32]);
        b.curves(at + n::CURVE, &[curve])
            .f(at + n::STOPS, stops.as_flattened());
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
        )
    }

    /// The node's uniform, its struct in the family's shader.
    pub(crate) fn uniform_words(&self, node: u32) -> Vec<u32> {
        use orb_shaders::layout::*;
        let pattern = match self {
            Self::FillPattern(_, p) | Self::LinePattern(_, p) | Self::ExtrusionPattern(_, p) => {
                Some(p)
            }
            _ => None,
        };
        let mut b;
        match self {
            Self::Fill(p) | Self::Background(p) | Self::FillPattern(p, _) => {
                use fill_node as n;
                b = Block::new(n::WORDS);
                b.f(n::COLOR, &p.color).f(n::OPACITY, &[p.opacity]);
                b.u(n::INDEX, &[node]).u(n::TABLE_MASK, &[p.table_mask]);
                b.translate(n::TRANSLATE_ANCHOR, n::TRANSLATE, p.translate);
                b.curves(n::CURVES, &p.curves).state(n::STATE, p.state);
                b.pattern([n::IMAGE_FROM, n::IMAGE_TO, n::PATTERN], pattern);
            }
            Self::Extrusion(p) | Self::ExtrusionPattern(p, _) => {
                use extrusion_node as n;
                b = Block::new(n::WORDS);
                b.f(n::COLOR, &p.color)
                    .f(n::HEIGHT, &[p.height])
                    .f(n::BASE, &[p.base]);
                b.u(n::INDEX, &[node]).u(n::TABLE_MASK, &[p.table_mask]);
                b.translate(n::TRANSLATE_ANCHOR, n::TRANSLATE, p.translate);
                b.u(n::VERTICAL_GRADIENT, &[u32::from(p.vertical_gradient)]);
                b.f(n::OPACITY, &[p.opacity]).f(n::EMISSIVE, &[p.emissive]);
                b.curves(n::CURVES, &p.curves).state(n::STATE, p.state);
                b.pattern([n::IMAGE_FROM, n::IMAGE_TO, n::PATTERN], pattern);
            }
            Self::Line(p) | Self::LinePattern(p, _) => {
                use line_node as n;
                b = Block::new(n::WORDS);
                b.f(n::COLOR, &p.color)
                    .f(n::OPACITY, &[p.opacity])
                    .f(n::WIDTH, &[p.width]);
                b.f(n::GAP_WIDTH, &[p.gap_width])
                    .f(n::OFFSET, &[p.offset])
                    .f(n::BLUR, &[p.blur]);
                b.u(n::INDEX, &[node]).u(n::TABLE_MASK, &[p.table_mask]);
                b.translate(n::TRANSLATE_ANCHOR, n::TRANSLATE, p.translate);
                b.curves(n::CURVES, &p.curves).state(n::STATE, p.state);
                b.pattern([n::IMAGE_FROM, n::IMAGE_TO, n::PATTERN], pattern);
                let dash = p.dash.unwrap_or(PatternPaint {
                    from: 0,
                    to: 0,
                    from_scale: 1.0,
                    to_scale: 1.0,
                    mix: 1.0,
                    floor_width: 1.0,
                });
                b.u(n::DASH_FROM, &[dash.from]).u(n::DASH_TO, &[dash.to]);
                b.f(n::DASH_FROM_SCALE, &[dash.from_scale])
                    .f(n::DASH_FADE, &[dash.mix]);
                b.f(n::DASH_FLOOR_WIDTH, &[dash.floor_width]);
                b.u(n::DASHED, &[u32::from(p.dash.is_some())])
                    .u(n::GRADIENT, &[u32::from(p.gradient)]);
            }
            Self::Circle(p) => {
                use circle_node as n;
                b = Block::new(n::WORDS);
                b.f(n::COLOR, &p.color).f(n::STROKE_COLOR, &p.stroke_color);
                b.f(n::RADIUS, &[p.radius])
                    .f(n::BLUR, &[p.blur])
                    .f(n::OPACITY, &[p.opacity]);
                b.f(n::STROKE_WIDTH, &[p.stroke_width])
                    .f(n::STROKE_OPACITY, &[p.stroke_opacity]);
                b.u(n::INDEX, &[node]).u(n::TABLE_MASK, &[p.table_mask]);
                let flags = [
                    p.translate.viewport,
                    p.pitch_scale_map,
                    p.pitch_alignment_map,
                ];
                b.u(n::FLAGS, &[bits(&flags)])
                    .f(n::TRANSLATE, &p.translate.px);
                b.curves(n::CURVES, &p.curves).state(n::STATE, p.state);
            }
            Self::Heatmap(p) => {
                use heatmap_node as n;
                b = Block::new(n::WORDS);
                b.f(n::RADIUS, &[p.radius]).f(n::WEIGHT, &[p.weight]);
                b.f(n::INTENSITY, &[p.intensity])
                    .f(n::OPACITY, &[p.opacity]);
                b.u(n::INDEX, &[node])
                    .u(n::TABLE_MASK, &[p.table_mask])
                    .curves(n::CURVES, &p.curves);
            }
            Self::Raster(p) => {
                use raster_node as n;
                // GL JS's `spinWeights`, `saturationFactor` and
                // `contrastFactor`.
                let (s, c) = p.hue_rotate.to_radians().sin_cos();
                let root3 = 3f32.sqrt();
                let spin =
                    [2.0 * c + 1.0, -root3 * s - c + 1.0, root3 * s - c + 1.0].map(|w| w / 3.0);
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
                b = Block::new(n::WORDS);
                b.f(n::SPIN_WEIGHTS, &spin).f(n::OPACITY, &[p.opacity]);
                b.f(n::BRIGHTNESS_LOW, &[p.brightness_min])
                    .f(n::BRIGHTNESS_HIGH, &[p.brightness_max]);
                b.f(n::SATURATION_FACTOR, &[saturation])
                    .f(n::CONTRAST_FACTOR, &[contrast]);
                b.u(n::INDEX, &[node])
                    .u(n::NEAREST, &[u32::from(p.nearest)]);
                b.u(n::LEVEL0, &[u32::from(!p.mipmapped)])
                    .f(n::FADE_DURATION, &[p.fade_duration]);
            }
            Self::Hillshade(p) => {
                use hillshade_node as n;
                b = Block::new(n::WORDS);
                b.f(n::ACCENT, &p.accent)
                    .f(n::SHADOWS, p.shadows.as_flattened());
                b.f(n::HIGHLIGHTS, p.highlights.as_flattened());
                b.f(n::ALTITUDES, &p.altitudes).f(n::AZIMUTHS, &p.azimuths);
                b.f(n::EXAGGERATION, &[p.exaggeration])
                    .u(n::METHOD, &[p.method as u32]);
                b.u(n::LIGHTS, &[p.lights])
                    .u(n::INDEX, &[node])
                    .u(n::NEAREST, &[u32::from(p.nearest)]);
            }
            Self::ColorRelief(p) => {
                use color_relief_node as n;
                b = Block::new(n::WORDS);
                b.f(n::UNPACK, &p.unpack)
                    .f(n::OPACITY, &[p.opacity])
                    .u(n::RAMP_SIZE, &[p.ramp_size]);
                b.u(n::INDEX, &[node])
                    .u(n::NEAREST, &[u32::from(p.nearest)]);
            }
            Self::Symbol(p) => {
                use symbol_node as n;
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
                    false,
                    p.height_anchor_ground,
                ];
                b = Block::new(n::WORDS);
                b.f(n::FILL_COLOR, &p.color).f(n::HALO_COLOR, &p.halo_color);
                b.f(n::OPACITY, &[p.opacity])
                    .f(n::HALO_WIDTH, &[p.halo_width]);
                b.f(n::HALO_BLUR, &[p.halo_blur]).u(n::INDEX, &[node]);
                b.u(n::TABLE_MASK, &[p.table_mask])
                    .u(n::FLAGS, &[bits(&flags)]);
                // The glyph table and page sizes are the renderer's.
                b.f(n::TRANSLATE, &p.translate.px);
                p.size.put(&mut b, n::SIZE);
                p.text_size.put(&mut b, n::TEXT_SIZE);
                b.curves(n::CURVES, &p.curves).state(n::STATE, p.state);
            }
        }
        b.0
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
        }
    }
}

/// Bit `i` set for each `flags[i]` that is.
fn bits(flags: &[bool]) -> u32 {
    flags
        .iter()
        .enumerate()
        .fold(0, |w, (i, &b)| w | u32::from(b) << i)
}

/// What node paint writes beside plain members.
trait PaintBlock {
    fn curves(&mut self, at: usize, curves: &[Curve]) -> &mut Self;
    fn state(&mut self, at: usize, s: StateLookup) -> &mut Self;
    fn translate(&mut self, anchor: usize, at: usize, t: Translate) -> &mut Self;
    fn pattern(&mut self, at: [usize; 3], p: Option<&PatternPaint>) -> &mut Self;
}

impl PaintBlock for Block {
    fn curves(&mut self, at: usize, curves: &[Curve]) -> &mut Self {
        use orb_shaders::layout::curve as n;
        for (i, curve) in curves.iter().enumerate() {
            let (kind, params) = match *curve {
                Curve::Step => (0, [0.0; 4]),
                Curve::Exponential(base) => (1, [base, 0.0, 0.0, 0.0]),
                Curve::CubicBezier(p) => (2, p),
            };
            let at = at + i * n::WORDS;
            self.f(at + n::PARAMS, &params).u(at + n::KIND, &[kind]);
        }
        self
    }

    fn state(&mut self, at: usize, s: StateLookup) -> &mut Self {
        use orb_shaders::layout::state_lookup as n;
        self.u(at + n::MASK, &[s.mask])
            .u(at + n::BASE, &[s.base])
            .u(at + n::STRIDE, &[s.stride])
            .f(at + n::ZOOM, &[s.zoom])
    }

    fn translate(&mut self, anchor: usize, at: usize, t: Translate) -> &mut Self {
        self.u(anchor, &[u32::from(t.viewport)]).f(at, &t.px)
    }

    /// A pattern's image ids and cross-fade, or none: scale 1, off.
    fn pattern(&mut self, [from, to, at]: [usize; 3], p: Option<&PatternPaint>) -> &mut Self {
        use orb_shaders::layout::pattern as n;
        let enabled = u32::from(p.is_some());
        let p = p.copied().unwrap_or(PatternPaint {
            from: 0,
            to: 0,
            from_scale: 1.0,
            to_scale: 1.0,
            mix: 1.0,
            floor_width: 0.0,
        });
        self.u(from, &[p.from]).u(to, &[p.to]);
        self.f(at + n::FROM_SCALE, &[p.from_scale])
            .f(at + n::TO_SCALE, &[p.to_scale]);
        self.f(at + n::FADE, &[p.mix])
            .u(at + n::ENABLED, &[enabled]);
        self.f(at + n::FLOOR_WIDTH, &[p.floor_width])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_fit_their_slot() {
        use orb_shaders::layout::*;
        for words in [
            fill_node::WORDS,
            line_node::WORDS,
            extrusion_node::WORDS,
            circle_node::WORDS,
            heatmap_node::WORDS,
            raster_node::WORDS,
            hillshade_node::WORDS,
            color_relief_node::WORDS,
            symbol_node::WORDS,
        ] {
            assert!(words * 4 <= crate::NODE_UNIFORM_SIZE as usize);
        }
    }

    #[test]
    fn no_hue_rotation_is_the_identity() {
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
        let words = raster.uniform_words(0);
        let at = orb_shaders::layout::raster_node::SPIN_WEIGHTS;
        let spin: Vec<f32> = words[at..at + 3]
            .iter()
            .map(|&w| f32::from_bits(w))
            .collect();
        assert_eq!(spin, [1.0, 0.0, 0.0]);
    }
}
