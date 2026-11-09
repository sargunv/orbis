//! Symbol layout of one tile layer: GL JS `symbol_layout.ts`
//! (`performSymbolLayout`, `addFeature`, `addSymbol`) and the feature
//! preparation of `SymbolBucket.populate`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::ops::Range;

use orb_text::{
    Anchor, FontStack, GlyphMap, GlyphRequests, Justify, Label, ONE_EM, ShapeContext, ShapeOptions,
    Shaping, WordBreaker, WritingMode, shape_text, unicode,
};
use orb_tile::{Ancestor, GeometryType, Layer, Value};

use crate::anchors::{self, Anchor as SymbolAnchor, LabelExtent, Square};
use crate::bucket::{
    AnchorOffset, NONE, PlacedSymbol, Quad, SizeKind, SortKeyRange, SymbolBucket, SymbolInstance,
    SymbolStream, SymbolWritingMode, instance_flags, justification, quad_flags,
};
use crate::collision::{CollisionFeature, Shape, collision_feature};
use crate::geometry::{EXTENT, Point, js_max, js_min, js_round, to_i16, to_u16};
use crate::icon::{IconImage, IconTextFit, PositionedIcon, fit_icon_to_text, shape_icon};
use crate::key::stable_key;
use crate::lines::{Lines, clip_line_rounding};
use crate::merge_lines::merge_lines;
use crate::pole::pole_of_inaccessibility;
use crate::quads::{GlyphQuadContext, SymbolQuad, Texture, glyph_quads, icon_quads};
use crate::variable::{self, TextOffset, anchor_code};

/// GL JS `SIZE_PACK_FACTOR`.
const SIZE_PACK_FACTOR: f64 = 128.0;
/// GL JS `MAX_PACKED_SIZE`: 255 px × [`SIZE_PACK_FACTOR`].
const MAX_PACKED_SIZE: f64 = 255.0 * SIZE_PACK_FACTOR;
/// Glyphs are laid out at 24 px.
const GLYPH_SIZE: f64 = 24.0;

/// `symbol-placement`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    #[default]
    Point,
    Line,
    LineCenter,
}

/// `text-rotation-alignment`, `icon-rotation-alignment`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Alignment {
    /// Also what a data-driven `icon-rotation-alignment` means here, as in
    /// GL JS.
    #[default]
    Auto,
    Map,
    Viewport,
    ViewportGlyph,
}

impl Alignment {
    /// `auto` resolved for the placement (GL JS
    /// `SymbolStyleLayer.recalculate`).
    pub fn resolve(self, placement: Placement) -> Self {
        match self {
            Self::Auto if placement == Placement::Point => Self::Viewport,
            Self::Auto => Self::Map,
            other => other,
        }
    }
}

/// `text-justify`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextJustify {
    Auto,
    Left,
    #[default]
    Center,
    Right,
}

/// The layout properties GL JS reads once per layer: not data-driven.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerLayout {
    pub placement: Placement,
    /// `symbol-spacing`, pixels.
    pub spacing: f64,
    pub text_rotation_alignment: Alignment,
    pub icon_rotation_alignment: Alignment,
    pub text_keep_upright: bool,
    /// `text-max-angle`, degrees.
    pub text_max_angle: f64,
    /// `text-line-height`, ems.
    pub text_line_height: f64,
    /// `text-padding`, pixels.
    pub text_padding: f64,
    /// `text-writing-mode`, in order.
    pub text_writing_mode: Vec<WritingMode>,
    /// `text-variable-anchor`.
    pub text_variable_anchor: Option<Vec<Anchor>>,
    /// Whether the style sets `text-radial-offset`: then variable anchors
    /// use it rather than `text-offset`.
    pub text_radial_offset_set: bool,
    pub icon_text_fit: IconTextFit,
    /// `icon-text-fit-padding`, CSS order, pixels.
    pub icon_text_fit_padding: [f64; 4],
    /// `icon-rotate` when it is constant; `None` when data-driven.
    pub icon_rotate_constant: Option<f64>,
    pub text_size: SizeKind,
    pub icon_size: SizeKind,
}

impl Default for LayerLayout {
    /// The v8 defaults.
    fn default() -> Self {
        Self {
            placement: Placement::Point,
            spacing: 250.0,
            text_rotation_alignment: Alignment::Auto,
            icon_rotation_alignment: Alignment::Auto,
            text_keep_upright: true,
            text_max_angle: 45.0,
            text_line_height: 1.2,
            text_padding: 2.0,
            text_writing_mode: Vec::new(),
            text_variable_anchor: None,
            text_radial_offset_set: false,
            icon_text_fit: IconTextFit::None,
            icon_text_fit_padding: [0.0; 4],
            icon_rotate_constant: Some(0.0),
            text_size: SizeKind::Constant,
            icon_size: SizeKind::Constant,
        }
    }
}

impl LayerLayout {
    /// Text follows lines (GL JS `textAlongLine`).
    pub(crate) fn text_along_line(&self) -> bool {
        self.text_rotation_alignment != Alignment::Viewport && self.placement != Placement::Point
    }

    /// Point labels may be set vertically (GL JS `allowVerticalPlacement`).
    pub fn allow_vertical_placement(&self) -> bool {
        self.placement == Placement::Point
            && self.text_writing_mode.contains(&WritingMode::Vertical)
    }

    /// Whether a label needs vertical glyph forms (GL JS `populate`).
    pub fn needs_vertical_forms(&self, label: &Label) -> bool {
        ((self.text_along_line() && self.text_keep_upright) || self.allow_vertical_placement())
            && unicode::allows_vertical_writing_mode(&label.text())
    }
}

/// A size property of one feature, evaluated at the zooms layout needs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FeatureSize {
    /// At the tile's zoom.
    pub tile: f64,
    /// At the tile's zoom + 1: the size boxes are built for.
    pub layout: f64,
    /// `text-size` at zoom 18, which spaces line anchors the same at every
    /// zoom.
    pub max: f64,
    /// At the stops of [`SizeKind::Composite`].
    pub stops: [f64; 2],
}

impl FeatureSize {
    /// A size that is the same at every zoom.
    pub fn constant(size: f64) -> Self {
        Self {
            tile: size,
            layout: size,
            max: size,
            stops: [size; 2],
        }
    }
}

/// The data-driven layout properties of one feature, evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureLayout {
    pub text_font: FontStack,
    pub text_size: FeatureSize,
    pub icon_size: FeatureSize,
    /// `text-letter-spacing`, ems.
    pub text_letter_spacing: f64,
    pub text_anchor: Anchor,
    pub text_justify: TextJustify,
    /// `text-max-width`, ems.
    pub text_max_width: f64,
    /// `text-offset`, ems.
    pub text_offset: [f64; 2],
    /// `text-radial-offset`, ems.
    pub text_radial_offset: f64,
    /// `text-variable-anchor-offset`, ems.
    pub text_variable_anchor_offset: Option<Vec<(Anchor, [f64; 2])>>,
    /// `text-rotate`, degrees.
    pub text_rotate: f64,
    /// `icon-offset`, pixels at icon size 1.
    pub icon_offset: [f64; 2],
    pub icon_anchor: Anchor,
    /// `icon-rotate`, degrees.
    pub icon_rotate: f64,
    /// `icon-padding`, CSS order, pixels.
    pub icon_padding: [f64; 4],
    /// `symbol-height-offset`, meters.
    pub height_offset: f64,
}

impl Default for FeatureLayout {
    /// The v8 defaults, with an empty font stack.
    fn default() -> Self {
        Self {
            text_font: FontStack::new::<&str>(&[]),
            text_size: FeatureSize::constant(16.0),
            icon_size: FeatureSize::constant(1.0),
            text_letter_spacing: 0.0,
            text_anchor: Anchor::Center,
            text_justify: TextJustify::Center,
            text_max_width: 10.0,
            text_offset: [0.0; 2],
            text_radial_offset: 0.0,
            text_variable_anchor_offset: None,
            text_rotate: 0.0,
            icon_offset: [0.0; 2],
            icon_anchor: Anchor::Center,
            icon_rotate: 0.0,
            icon_padding: [2.0; 4],
            height_offset: 0.0,
        }
    }
}

/// A feature the layer draws, with its label, icon and layout values.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolFeature {
    /// Index of the feature in the tile layer.
    pub index: u32,
    /// The feature's id when it names the same feature in every tile: the
    /// promoted id, when the source has `promoteId` and the feature is not a
    /// cluster (GL JS `keyWithFeatureId`).
    pub stable_id: Option<Value>,
    /// The evaluated `text-field` with `text-transform` applied.
    pub text: Option<Label>,
    /// The evaluated `icon-image`.
    pub icon: Option<Box<str>>,
    /// `symbol-sort-key`, set for every feature when the layer sorts by it
    /// (GL JS `sortFeaturesByKey`), else for none.
    pub sort_key: Option<f64>,
    pub layout: FeatureLayout,
}

/// The images a tile's symbols use, numbered in first-insert order: the
/// tile-local image indices of [`Quad::texture`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileImages {
    index: HashMap<Box<str>, u32>,
    images: Vec<(Box<str>, IconImage)>,
}

impl TileImages {
    /// Adds an image, or replaces the one of that name; returns its index.
    pub fn insert(&mut self, name: &str, image: IconImage) -> u32 {
        if let Some(&i) = self.index.get(name) {
            self.images[i as usize].1 = image;
            return i;
        }
        let i = u32::try_from(self.images.len()).expect("image count fits u32");
        self.index.insert(name.into(), i);
        self.images.push((name.into(), image));
        i
    }

    pub fn get(&self, name: &str) -> Option<(u32, &IconImage)> {
        let &i = self.index.get(name)?;
        Some((i, &self.images[i as usize].1))
    }

    /// Image names by index.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.images.iter().map(|(n, _)| &**n)
    }
}

/// The tile a layer is laid out for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileParams<'a> {
    /// The source layer's name, part of every [`SymbolInstance::key`].
    pub source_layer: &'a str,
    /// Tile size over 512 px (GL JS `overscaling`).
    pub overscaling: f64,
    /// The device pixel ratio images are drawn at.
    pub pixel_ratio: f64,
    /// For an overzoom slice laying out in its ancestor, where the slice
    /// is in it: the layer is the ancestor's, laid out as GL JS lays it out
    /// at `overscaling` times the ancestor's scale, keeping the symbols
    /// whose anchors fall in the slice, in the slice's units.
    pub ancestor: Option<Ancestor>,
}

/// How the layer's units map to the tile's: scaled by `scale` (the
/// ancestor's size in tiles), then moved back by `offset`.
#[derive(Clone, Copy, Debug)]
struct Units {
    scale: f64,
    offset: Point,
}

impl Units {
    fn new(ancestor: Option<Ancestor>) -> Self {
        ancestor.map_or(
            Self {
                scale: 1.0,
                offset: Point::new(0.0, 0.0),
            },
            |a| Self {
                scale: f64::from(a.scale),
                offset: Point::new(f64::from(a.cell[0]) * EXTENT, f64::from(a.cell[1]) * EXTENT),
            },
        )
    }

    /// A vertex as GL JS loads it in the layer's tile, in this tile's units.
    fn point(self, p: [i32; 2], extent: u32) -> Point {
        let p = load_point(p, extent);
        if self.scale == 1.0 {
            return p;
        }
        Point::new(
            p.x * self.scale - self.offset.x,
            p.y * self.scale - self.offset.y,
        )
    }

    /// GL JS's tile: the layer's.
    fn square(self) -> Square {
        Square {
            min: Point::new(-self.offset.x, -self.offset.y),
            size: EXTENT * self.scale,
            grid: self.scale,
        }
    }
}

/// Glyph, image and word-break data for the tile.
#[derive(Clone, Copy, Debug)]
pub struct LayoutContext<'a> {
    pub glyphs: &'a GlyphMap,
    pub images: &'a TileImages,
    pub words: Option<&'a WordBreaker>,
}

/// Adds the glyphs `features` need to `requests` (GL JS
/// `calculateGlyphDependencies`, as `populate` calls it).
pub fn add_glyph_requests(
    layer: &Layer,
    features: &[SymbolFeature],
    style: &LayerLayout,
    requests: &mut GlyphRequests,
) {
    let allow_vertical_placement = style.allow_vertical_placement();
    // In the layer's own units: an overzoom slice's ancestor holds it.
    let units = Units::new(None);
    for feature in features {
        if is_skipped(layer, feature, style, units) {
            continue;
        }
        if let Some(label) = &feature.text {
            requests.add_label(
                label,
                &feature.layout.text_font,
                style.needs_vertical_forms(label),
                allow_vertical_placement,
            );
        }
    }
}

/// Point labels of points all outside the tile are dropped before layout.
fn is_skipped(layer: &Layer, feature: &SymbolFeature, style: &LayerLayout, units: Units) -> bool {
    let index = feature.index as usize;
    style.placement == Placement::Point
        && layer.geometry_type(index) == GeometryType::Point
        && layer.geometry(index).parts().all(|part| {
            part.rings()
                .flatten()
                .all(|&p| is_outside_tile(units.point(p, layer.extent())))
        })
}

fn is_outside_tile(p: Point) -> bool {
    p.x < 0.0 || p.x >= EXTENT || p.y < 0.0 || p.y >= EXTENT
}

/// The square of tile units a bucket's line vertices hold: three tiles
/// past the tile on every side.
const LINE_REACH: [f64; 2] = [-3.0 * EXTENT, 4.0 * EXTENT - 1.0];

fn fits_i16(p: Point) -> bool {
    let range = LINE_REACH[0]..=LINE_REACH[1];
    range.contains(&p.x) && range.contains(&p.y)
}

/// The stretch of `line` around `anchor` (in the tile) within
/// [`LINE_REACH`], ends cut where it leaves, and the anchor on it.
fn trim_line(line: &[Point], anchor: &SymbolAnchor) -> (Vec<Point>, SymbolAnchor) {
    let segment = anchor.segment.expect("an anchor along the line");
    let mut first = segment;
    while first > 0 && fits_i16(line[first]) {
        first -= 1;
    }
    let mut last = segment + 1;
    while last + 1 < line.len() && fits_i16(line[last]) {
        last += 1;
    }
    let mut points = line[first..=last].to_vec();
    // Where a segment from `from`, inside, to `to` leaves the square.
    let leave = |from: Point, to: Point| {
        let mut t: f64 = 1.0;
        for (a, b) in [(from.x, to.x), (from.y, to.y)] {
            for edge in LINE_REACH {
                if (b - edge) * (a - edge) < 0.0 {
                    t = t.min((edge - a) / (b - a));
                }
            }
        }
        Point::new(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t)
    };
    if !fits_i16(points[0]) {
        let from = if first == segment {
            anchor.point
        } else {
            points[1]
        };
        points[0] = leave(from, points[0]);
    }
    let n = points.len();
    if !fits_i16(points[n - 1]) {
        let from = if last == segment + 1 {
            anchor.point
        } else {
            points[n - 2]
        };
        points[n - 1] = leave(from, points[n - 1]);
    }
    let anchor = SymbolAnchor {
        segment: Some(segment - first),
        ..*anchor
    };
    (points, anchor)
}

/// Scales a vertex to [`EXTENT`] and rounds it, as GL JS `loadGeometry`.
fn load_point(p: [i32; 2], extent: u32) -> Point {
    let scale = EXTENT / f64::from(extent);
    let clamp = |v: f64| v.clamp(-32768.0, 32767.0);
    Point::new(
        clamp(js_round(f64::from(p[0]) * scale)),
        clamp(js_round(f64::from(p[1]) * scale)),
    )
}

/// A feature's geometry as GL JS sees it: one "line" per point, line
/// string or polygon ring, rings closed.
struct Geometry {
    kind: GeometryType,
    lines: Vec<Vec<Point>>,
    /// Polygons: each part's range of `lines`.
    parts: Vec<Range<usize>>,
}

fn load_geometry(layer: &Layer, index: usize, units: Units) -> Geometry {
    let extent = layer.extent();
    let kind = layer.geometry_type(index);
    let mut lines = Vec::new();
    let mut parts = Vec::new();
    for part in layer.geometry(index).parts() {
        let start = lines.len();
        for ring in part.rings() {
            let points = ring.iter().map(|&p| units.point(p, extent));
            match kind {
                GeometryType::Point => lines.extend(points.map(|p| vec![p])),
                GeometryType::LineString => lines.push(points.collect()),
                GeometryType::Polygon => {
                    let mut ring: Vec<Point> = points.collect();
                    if let Some(&first) = ring.first() {
                        ring.push(first);
                    }
                    lines.push(ring);
                }
            }
        }
        parts.push(start..lines.len());
    }
    Geometry { kind, lines, parts }
}

/// Lays out `features`, the features of `layer` a symbol layer draws, in
/// the layer's order (GL JS `populate` from the filter on, then
/// `performSymbolLayout`).
pub fn layout(
    layer: &Layer,
    features: &[SymbolFeature],
    style: &LayerLayout,
    tile: TileParams<'_>,
    context: LayoutContext<'_>,
) -> SymbolBucket {
    struct Prepared<'f> {
        feature: &'f SymbolFeature,
        geometry: Geometry,
    }
    let units = Units::new(tile.ancestor);
    let mut prepared: Vec<Prepared<'_>> = features
        .iter()
        .filter(|f| !is_skipped(layer, f, style, units))
        .map(|feature| Prepared {
            feature,
            geometry: load_geometry(layer, feature.index as usize, units),
        })
        .collect();
    prepared.retain(|p| !p.geometry.lines.is_empty());

    if style.placement == Placement::Line {
        let texts: Vec<String> = prepared
            .iter()
            .map(|p| p.feature.text.as_ref().map(Label::text).unwrap_or_default())
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let firsts: Vec<&[Point]> = prepared.iter().map(|p| &p.geometry.lines[0][..]).collect();
        let merged = merge_lines(&text_refs, &firsts);
        let mut slots: Vec<Option<Prepared<'_>>> = prepared.into_iter().map(Some).collect();
        prepared = merged
            .into_iter()
            .map(|m| {
                let mut p = slots[m.feature].take().expect("each feature merges once");
                if let Some(joined) = m.joined {
                    p.geometry.lines[0] = joined;
                }
                p
            })
            .collect();
    }

    if prepared.iter().any(|p| p.feature.sort_key.is_some()) {
        // A stable sort, as `Array.prototype.sort`. NaN keys compare equal
        // to everything in JavaScript, which is no order; they sort as 0.
        let key = |p: &Prepared<'_>| {
            let k = p.feature.sort_key.unwrap_or(0.0);
            if k.is_nan() { 0.0 } else { k }
        };
        prepared.sort_by(|a, b| {
            key(a)
                .partial_cmp(&key(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut builder = Builder::new(style, tile, context);
    for p in &prepared {
        builder.add_feature(p.feature, &p.geometry);
    }
    builder.bucket
}

/// A label's shapings (GL JS `ShapedTextOrientations`).
#[derive(Default)]
struct Shapings {
    /// Justifications in the order GL JS's object keys have them, each with
    /// its shaping, or with none when it shares the first, single-line one.
    horizontal: Vec<(Justify, Option<usize>)>,
    shapings: Vec<Shaping>,
    vertical: Option<Shaping>,
}

impl Shapings {
    /// The shaping collision uses (GL JS `getDefaultHorizontalShaping`).
    fn default_horizontal(&self) -> Option<&Shaping> {
        self.horizontal
            .first()
            .and_then(|&(_, i)| i.map(|i| &self.shapings[i]))
    }

    fn contains(&self, justify: Justify) -> bool {
        self.horizontal.iter().any(|&(j, _)| j == justify)
    }
}

fn anchor_justification(anchor: Anchor) -> Justify {
    match anchor {
        Anchor::Right | Anchor::TopRight | Anchor::BottomRight => Justify::Right,
        Anchor::Left | Anchor::TopLeft | Anchor::BottomLeft => Justify::Left,
        _ => Justify::Center,
    }
}

fn justification_slot(justify: Justify) -> usize {
    match justify {
        Justify::Right => justification::RIGHT,
        Justify::Center => justification::CENTER,
        Justify::Left => justification::LEFT,
    }
}

/// What `addFeature` hands every `addSymbol` of one feature.
struct FeatureSymbols<'a> {
    feature: &'a SymbolFeature,
    shapings: &'a Shapings,
    icon: Option<(u32, PositionedIcon<'a>)>,
    vertical_icon: Option<PositionedIcon<'a>>,
    sdf_icon: bool,
    text_offset: [f64; 2],
    text_box_scale: f64,
    icon_box_scale: f64,
    text_padding: f64,
    icon_padding: [f64; 4],
    /// The text's key part: the default horizontal shaping's text.
    key_text: &'a str,
    anchor_offsets: &'a [(Anchor, [f64; 2])],
}

struct Builder<'a> {
    style: &'a LayerLayout,
    tile: TileParams<'a>,
    context: LayoutContext<'a>,
    bucket: SymbolBucket,
    /// Anchors so far per label text, for `symbol-spacing` across features
    /// (GL JS `compareText`).
    compare_text: HashMap<String, Vec<Point>>,
    quads: Vec<SymbolQuad>,
    anchors: Vec<SymbolAnchor>,
    clipped: Lines,
    units: Units,
}

impl<'a> Builder<'a> {
    fn new(style: &'a LayerLayout, tile: TileParams<'a>, context: LayoutContext<'a>) -> Self {
        let bucket = SymbolBucket {
            text_size: style.text_size,
            icon_size: style.icon_size,
            tile_pixel_ratio: EXTENT / (512.0 * tile.overscaling),
            allow_vertical_placement: style.allow_vertical_placement(),
            ..SymbolBucket::default()
        };
        Self {
            style,
            tile,
            context,
            bucket,
            compare_text: HashMap::new(),
            quads: Vec::new(),
            anchors: Vec::new(),
            clipped: Lines::default(),
            units: Units::new(tile.ancestor),
        }
    }

    fn shape(&self, label: &Label, f: &FeatureLayout, options: ShapeOptions) -> Option<Shaping> {
        let images = self.context.images;
        let lookup = |name: &str| images.get(name).map(|(_, image)| image.inline());
        let context = ShapeContext {
            glyphs: self.context.glyphs,
            images: &lookup,
            words: self.context.words,
        };
        shape_text(label, &f.text_font, context, &options)
    }

    /// The loop body of GL JS `performSymbolLayout`.
    fn add_feature(&mut self, feature: &SymbolFeature, geometry: &Geometry) {
        let style = self.style;
        let f = &feature.layout;
        let line_height = style.text_line_height * ONE_EM;
        let text_along_line = style.text_along_line();
        let allow_vertical_placement = style.allow_vertical_placement();

        let anchor_offsets = self.variable_anchor_offsets(f);
        let mut shapings = Shapings::default();
        let mut text_offset = [0.0, 0.0];
        if let Some(label) = &feature.text {
            let unformatted = label.text();
            let spacing = f.text_letter_spacing * ONE_EM;
            let spacing = if unicode::allows_letter_spacing(&unformatted) {
                spacing
            } else {
                0.0
            };
            let variable = anchor_offsets.is_some();
            if !variable {
                let radial = f.text_radial_offset;
                text_offset = if radial != 0.0 && !radial.is_nan() {
                    variable::variable_offset(f.text_anchor, TextOffset::Radial(radial * ONE_EM))
                } else {
                    [f.text_offset[0] * ONE_EM, f.text_offset[1] * ONE_EM]
                };
            }
            let max_width = if style.placement == Placement::Point {
                f.text_max_width * ONE_EM
            } else {
                f64::INFINITY
            };
            let options = |anchor, justify, writing_mode, allow_vertical| ShapeOptions {
                max_width,
                line_height,
                anchor,
                justify,
                spacing,
                translate: text_offset,
                writing_mode,
                allow_vertical_placement: allow_vertical,
                layout_text_size: f.text_size.layout,
                layout_text_size_this_zoom: f.text_size.tile,
            };
            let vertical_point_shaping = |this: &Self| {
                (allow_vertical_placement && unicode::allows_vertical_writing_mode(&unformatted))
                    .then(|| {
                        this.shape(
                            label,
                            f,
                            options(f.text_anchor, Justify::Left, WritingMode::Vertical, true),
                        )
                    })
                    .flatten()
            };

            if !text_along_line && variable {
                let mut justifications: Vec<Justify> = Vec::new();
                if f.text_justify == TextJustify::Auto {
                    for &(anchor, _) in anchor_offsets.iter().flatten() {
                        let j = anchor_justification(anchor);
                        if !justifications.contains(&j) {
                            justifications.push(j);
                        }
                    }
                } else {
                    justifications.push(resolve_justify(f.text_justify, f.text_anchor));
                }
                let mut single_line = false;
                for justification in justifications {
                    if shapings.contains(justification) {
                        continue;
                    }
                    if single_line {
                        shapings.horizontal.push((justification, None));
                    } else if let Some(shaping) = self.shape(
                        label,
                        f,
                        options(
                            Anchor::Center,
                            justification,
                            WritingMode::Horizontal,
                            false,
                        ),
                    ) {
                        single_line = shaping.lines.len() == 1;
                        shapings
                            .horizontal
                            .push((justification, Some(shapings.shapings.len())));
                        shapings.shapings.push(shaping);
                    }
                }
                shapings.vertical = vertical_point_shaping(self);
            } else {
                let justify = if text_along_line {
                    Justify::Center
                } else {
                    resolve_justify(f.text_justify, f.text_anchor)
                };
                if let Some(shaping) = self.shape(
                    label,
                    f,
                    options(f.text_anchor, justify, WritingMode::Horizontal, false),
                ) {
                    shapings.horizontal.push((justify, Some(0)));
                    shapings.shapings.push(shaping);
                }
                shapings.vertical = vertical_point_shaping(self);
                if unicode::allows_vertical_writing_mode(&unformatted)
                    && text_along_line
                    && style.text_keep_upright
                {
                    shapings.vertical = self.shape(
                        label,
                        f,
                        options(f.text_anchor, justify, WritingMode::Vertical, false),
                    );
                }
            }
        }

        let mut icon = None;
        let mut sdf_icon = false;
        if let Some(name) = feature.icon.as_deref().filter(|n| !n.is_empty())
            && let Some((index, image)) = self.context.images.get(name)
        {
            icon = Some((index, shape_icon(image, f.icon_offset, f.icon_anchor)));
            sdf_icon = image.sdf;
            match self.bucket.sdf_icons {
                None => self.bucket.sdf_icons = Some(sdf_icon),
                Some(sdf) if sdf != sdf_icon => self.bucket.warnings.mixed_sdf_icons = true,
                Some(_) => {}
            }
            if image.pixel_ratio != self.tile.pixel_ratio
                || style.icon_rotate_constant.is_none_or(|r| r != 0.0)
            {
                self.bucket.icons_need_linear = true;
            }
        }

        let shaped_text = shapings.default_horizontal().or(shapings.vertical.as_ref());
        self.bucket.icons_in_text |= shaped_text.is_some_and(|s| s.icons_in_text);
        if shaped_text.is_some() || icon.is_some() {
            let anchor_offsets = anchor_offsets.as_deref().unwrap_or_default();
            self.add_anchors(
                feature,
                geometry,
                &shapings,
                icon,
                sdf_icon,
                text_offset,
                anchor_offsets,
            );
        }
    }

    /// GL JS `getTextVariableAnchorOffset`, in layout units; `None` without
    /// variable anchors.
    fn variable_anchor_offsets(&self, f: &FeatureLayout) -> Option<Vec<(Anchor, [f64; 2])>> {
        let mut out = Vec::new();
        if let Some(values) = &f.text_variable_anchor_offset {
            variable::from_anchor_offsets(values, &mut out);
        } else if let Some(anchors) = &self.style.text_variable_anchor {
            let offset = if self.style.text_radial_offset_set {
                TextOffset::Radial(f.text_radial_offset * ONE_EM)
            } else {
                TextOffset::Xy([f.text_offset[0] * ONE_EM, f.text_offset[1] * ONE_EM])
            };
            variable::from_anchors(anchors, offset, &mut out);
        } else {
            return None;
        }
        Some(out)
    }

    /// GL JS `addFeature`: the feature's anchors, then a symbol at each.
    #[allow(clippy::too_many_arguments)]
    fn add_anchors(
        &mut self,
        feature: &SymbolFeature,
        geometry: &Geometry,
        shapings: &Shapings,
        icon: Option<(u32, PositionedIcon<'a>)>,
        sdf_icon: bool,
        text_offset: [f64; 2],
        anchor_offsets: &[(Anchor, [f64; 2])],
    ) {
        let style = self.style;
        let f = &feature.layout;
        let tile_pixel_ratio = self.bucket.tile_pixel_ratio;
        let text_max_size = f.text_size.max;
        let default_horizontal = shapings.default_horizontal();
        let font_scale = f.text_size.layout / GLYPH_SIZE;
        let text_box_scale = tile_pixel_ratio * font_scale;
        let text_max_box_scale = tile_pixel_ratio * text_max_size / GLYPH_SIZE;
        let icon_box_scale = tile_pixel_ratio * f.icon_size.layout;
        let symbol_min_distance = tile_pixel_ratio * style.spacing;
        let text_padding = style.text_padding * tile_pixel_ratio;
        let icon_padding = f.icon_padding.map(|p| p * tile_pixel_ratio);
        let text_max_angle = style.text_max_angle / 180.0 * PI;
        let text_repeat_distance = symbol_min_distance / 2.0;

        let mut icon = icon;
        let mut vertical_icon = None;
        if let Some((_, shaped_icon)) = &mut icon
            && style.icon_text_fit != IconTextFit::None
        {
            if style.allow_vertical_placement()
                && let Some(vertical) = &shapings.vertical
            {
                vertical_icon = Some(fit_icon_to_text(
                    shaped_icon,
                    vertical,
                    style.icon_text_fit,
                    style.icon_text_fit_padding,
                    f.icon_offset,
                    font_scale,
                ));
            }
            if let Some(horizontal) = default_horizontal {
                *shaped_icon = fit_icon_to_text(
                    shaped_icon,
                    horizontal,
                    style.icon_text_fit,
                    style.icon_text_fit_padding,
                    f.icon_offset,
                    font_scale,
                );
            }
        }

        let extent = LabelExtent {
            text: shapings
                .vertical
                .as_ref()
                .or(default_horizontal)
                .map(|s| s.right - s.left),
            icon: icon.as_ref().map(|(_, i)| i.right - i.left),
        };
        let symbols = FeatureSymbols {
            feature,
            shapings,
            icon,
            vertical_icon,
            sdf_icon,
            text_offset,
            text_box_scale,
            icon_box_scale,
            text_padding,
            icon_padding,
            key_text: default_horizontal.map_or("", |s| s.text.as_str()),
            anchor_offsets,
        };

        match style.placement {
            Placement::Line => {
                let mut clipped = std::mem::take(&mut self.clipped);
                clipped.clear();
                let square = self.units.square();
                let min = square.min;
                clip_line_rounding(
                    geometry.lines.iter().map(Vec::as_slice),
                    [min.x, min.y, min.x + square.size, min.y + square.size],
                    |p| square.round(p),
                    &mut clipped,
                );
                // Laid out in an ancestor, the anchors that matter are the
                // tile's and those near enough to keep a repeated label
                // from it.
                let window = self.tile.ancestor.map(|_| {
                    let m = symbol_min_distance;
                    [Point::new(-m, -m), Point::new(EXTENT + m, EXTENT + m)]
                });
                let mut anchors = std::mem::take(&mut self.anchors);
                for line in clipped.iter() {
                    anchors.clear();
                    anchors::line_anchors(
                        line,
                        symbol_min_distance,
                        text_max_angle,
                        extent,
                        GLYPH_SIZE,
                        text_max_box_scale,
                        self.tile.overscaling * self.units.scale,
                        square,
                        window,
                        &mut anchors,
                    );
                    let mut shared_line = None;
                    for anchor in &anchors {
                        let too_close = default_horizontal.is_some_and(|s| {
                            self.anchor_is_too_close(&s.text, text_repeat_distance, anchor.point)
                        });
                        if !too_close {
                            self.add_symbol(&symbols, anchor, line, &mut shared_line);
                        }
                    }
                }
                self.anchors = anchors;
                self.clipped = clipped;
            }
            Placement::LineCenter => {
                for line in geometry.lines.iter().filter(|l| l.len() > 1) {
                    if let Some(anchor) = anchors::center_anchor(
                        line,
                        text_max_angle,
                        extent,
                        GLYPH_SIZE,
                        text_max_box_scale,
                        self.units.square(),
                    ) {
                        self.add_symbol(&symbols, &anchor, line, &mut None);
                    }
                }
            }
            Placement::Point => match geometry.kind {
                GeometryType::Polygon => {
                    for part in &geometry.parts {
                        let rings: Vec<&[Point]> = geometry.lines[part.clone()]
                            .iter()
                            .map(Vec::as_slice)
                            .collect();
                        if rings.is_empty() {
                            continue;
                        }
                        let pole = pole_of_inaccessibility(&rings, 16.0 * self.units.scale);
                        self.add_symbol(&symbols, &SymbolAnchor::at(pole), &[], &mut None);
                    }
                }
                GeometryType::LineString => {
                    for line in &geometry.lines {
                        let anchor = SymbolAnchor::at(line[0]);
                        self.add_symbol(&symbols, &anchor, line, &mut None);
                    }
                }
                GeometryType::Point => {
                    for points in &geometry.lines {
                        for &point in points {
                            let anchor = SymbolAnchor::at(point);
                            self.add_symbol(&symbols, &anchor, &[], &mut None);
                        }
                    }
                }
            },
        }
    }

    /// GL JS `anchorIsTooClose`: records the anchor unless another of the
    /// same text is within `repeat_distance`.
    fn anchor_is_too_close(&mut self, text: &str, repeat_distance: f64, anchor: Point) -> bool {
        let others = self.compare_text.entry(text.to_owned()).or_default();
        if others
            .iter()
            .rev()
            .any(|&o| anchor.dist(o) < repeat_distance)
        {
            return true;
        }
        others.push(anchor);
        false
    }

    /// The line vertices a symbol rides on (GL JS `addToLineVertexArray`,
    /// without the unused distances), shared by the anchors of one line.
    fn line_range(
        &mut self,
        anchor: &SymbolAnchor,
        line: &[Point],
        shared: &mut Option<[u32; 2]>,
    ) -> [u32; 2] {
        let start = self.bucket.lines.len() as u32;
        if anchor.segment.is_none() {
            return [start, 0];
        }
        *shared.get_or_insert_with(|| {
            self.bucket
                .lines
                .extend(line.iter().map(|p| [to_i16(p.x), to_i16(p.y)]));
            [start, line.len() as u32]
        })
    }

    /// GL JS `addSymbol`: one symbol instance and its placed symbols, for
    /// an anchor inside the tile.
    fn add_symbol(
        &mut self,
        s: &FeatureSymbols<'_>,
        anchor: &SymbolAnchor,
        line: &[Point],
        shared_line: &mut Option<[u32; 2]>,
    ) {
        if is_outside_tile(anchor.point) {
            return;
        }
        // An ancestor's line may reach past what the bucket's 16 bits hold.
        let trimmed;
        let (anchor, line_range) = if self.tile.ancestor.is_some()
            && anchor.segment.is_some()
            && !line.iter().all(|&p| fits_i16(p))
        {
            trimmed = trim_line(line, anchor);
            (
                &trimmed.1,
                self.line_range(&trimmed.1, &trimmed.0, &mut None),
            )
        } else {
            (anchor, self.line_range(anchor, line, shared_line))
        };
        let style = self.style;
        let f = &s.feature.layout;
        let text_along_line = style.text_along_line();
        let elevation = f.height_offset;
        if elevation > self.bucket.max_height_offset {
            self.bucket.max_height_offset = elevation;
        }
        let instance = self.bucket.instances.len() as u32;
        let text_padding = [s.text_padding; 4];

        let mut vertical_text_feature = None;
        let mut vertical_icon_feature = None;
        if style.allow_vertical_placement()
            && let Some(vertical) = &s.shapings.vertical
        {
            let rotation = f.text_rotate + 90.0;
            vertical_text_feature = Some(collision_feature(
                Shape::text(vertical),
                s.text_box_scale,
                text_padding,
                text_along_line,
                rotation,
            ));
            if let Some(vertical_icon) = &s.vertical_icon {
                vertical_icon_feature = Some(collision_feature(
                    Shape::icon(vertical_icon),
                    s.icon_box_scale,
                    s.icon_padding,
                    text_along_line,
                    rotation,
                ));
            }
        }

        let mut icon_feature = None;
        let mut placed_icon = NONE;
        let mut vertical_placed_icon = NONE;
        if let Some((image, icon)) = &s.icon {
            let has_icon_text_fit = style.icon_text_fit != IconTextFit::None;
            let size = self.size_vertex(false, f);
            icon_feature = Some(collision_feature(
                Shape::icon(icon),
                s.icon_box_scale,
                s.icon_padding,
                false,
                f.icon_rotate,
            ));
            let mut quads = std::mem::take(&mut self.quads);
            let mut add = |this: &mut Self, icon, mode| {
                quads.clear();
                let fit = has_icon_text_fit;
                icon_quads(icon, *image, f.icon_rotate, s.sdf_icon, fit, &mut quads);
                let offset = f.icon_offset;
                this.add_symbols(
                    false, &quads, size, offset, mode, anchor, line_range, instance, NONE,
                )
            };
            placed_icon = add(self, icon, SymbolWritingMode::None);
            if let Some(vertical_icon) = &s.vertical_icon {
                vertical_placed_icon = add(self, vertical_icon, SymbolWritingMode::Vertical);
            }
            self.quads = quads;
        }

        let mut text_feature = None;
        let mut placed_text = [NONE; 4];
        let horizontal_mode = if s.shapings.vertical.is_some() {
            SymbolWritingMode::Horizontal
        } else {
            SymbolWritingMode::HorizontalOnly
        };
        for &(justify, shaping) in &s.shapings.horizontal {
            let shaping = &s.shapings.shapings
                [shaping.expect("an aliased justification follows a single-line shaping")];
            if text_feature.is_none() {
                text_feature = Some(collision_feature(
                    Shape::text(shaping),
                    s.text_box_scale,
                    text_padding,
                    text_along_line,
                    f.text_rotate,
                ));
            }
            let single_line = shaping.lines.len() == 1;
            let placed = self.add_text(
                s,
                shaping,
                anchor,
                line_range,
                horizontal_mode,
                instance,
                placed_icon,
            );
            if single_line {
                for &(j, _) in &s.shapings.horizontal {
                    placed_text[justification_slot(j)] = placed;
                }
                break;
            }
            placed_text[justification_slot(justify)] = placed;
        }
        if let Some(vertical) = &s.shapings.vertical {
            placed_text[justification::VERTICAL] = self.add_text(
                s,
                vertical,
                anchor,
                line_range,
                SymbolWritingMode::Vertical,
                instance,
                vertical_placed_icon,
            );
        }

        let mut flags = 0;
        let mut boxes = [[0i16; 4]; 4];
        let features = [
            text_feature,
            vertical_text_feature,
            icon_feature,
            vertical_icon_feature,
        ];
        let box_flags = [
            instance_flags::TEXT_BOX,
            instance_flags::VERTICAL_TEXT_BOX,
            instance_flags::ICON_BOX,
            instance_flags::VERTICAL_ICON_BOX,
        ];
        let mut circle_diameter = -1.0;
        for (i, feature) in features.iter().enumerate() {
            match feature {
                Some(CollisionFeature::Box(b)) => {
                    boxes[i] = [to_i16(b.x1), to_i16(b.y1), to_i16(b.x2), to_i16(b.y2)];
                    flags |= box_flags[i];
                }
                Some(c) => {
                    if let Some(d) = c.circle_diameter() {
                        circle_diameter = js_max(d, circle_diameter);
                    }
                }
                None => {}
            }
        }
        if circle_diameter > -1.0 {
            flags |= instance_flags::RUNTIME_COLLISION_CIRCLES;
            circle_diameter *= f.text_size.layout / ONE_EM;
        }

        if let Some(sort_key) = s.feature.sort_key {
            let ranges = &mut self.bucket.sort_key_ranges;
            match ranges.last_mut() {
                Some(last) if last.sort_key == sort_key => last.end = instance + 1,
                _ => ranges.push(SortKeyRange {
                    sort_key,
                    start: instance,
                    end: instance + 1,
                }),
            }
        }

        let offsets_start = self.bucket.anchor_offsets.len() as u32;
        for &(anchor, offset) in s.anchor_offsets {
            self.bucket.anchor_offsets.push(AnchorOffset {
                anchor: anchor_code(anchor),
                offset: [offset[0] as f32, offset[1] as f32],
            });
        }
        let offsets_end = self.bucket.anchor_offsets.len() as u32;

        self.bucket.instances.push(SymbolInstance {
            key: stable_key(
                self.tile.source_layer,
                s.feature.stable_id.as_ref(),
                s.key_text,
            ),
            anchor: [to_i16(anchor.point.x), to_i16(anchor.point.y)],
            feature: s.feature.index,
            text_box_scale: s.text_box_scale as f32,
            collision_circle_diameter: circle_diameter as f32,
            height_offset: elevation as f32,
            text: placed_text,
            icon: [placed_icon, vertical_placed_icon],
            boxes,
            anchor_offsets: [offsets_start, offsets_end],
            flags,
        });
    }

    /// GL JS `addTextVertices`. Returns the placed symbol's index.
    #[allow(clippy::too_many_arguments)]
    fn add_text(
        &mut self,
        s: &FeatureSymbols<'_>,
        shaping: &Shaping,
        anchor: &SymbolAnchor,
        line: [u32; 2],
        writing_mode: SymbolWritingMode,
        instance: u32,
        associated_icon: u32,
    ) -> u32 {
        let f = &s.feature.layout;
        let mut quads = std::mem::take(&mut self.quads);
        quads.clear();
        glyph_quads(
            shaping,
            GlyphQuadContext {
                glyphs: self.context.glyphs,
                images: self.context.images,
                text_offset: s.text_offset,
                rotate: f.text_rotate,
                along_line: self.style.text_along_line(),
                allow_vertical_placement: self.style.allow_vertical_placement(),
            },
            &mut quads,
        );
        let size = self.size_vertex(true, f);
        let placed = self.add_symbols(
            true,
            &quads,
            size,
            s.text_offset,
            writing_mode,
            anchor,
            line,
            instance,
            associated_icon,
        );
        self.quads = quads;
        placed
    }

    /// The per-feature text or icon size GL JS bakes into vertices,
    /// noting sizes too large to pack.
    fn size_vertex(&mut self, text: bool, f: &FeatureLayout) -> Option<[f64; 2]> {
        let (kind, size, warning) = if text {
            let w = &mut self.bucket.warnings.text_size_too_large;
            (self.style.text_size, &f.text_size, w)
        } else {
            let w = &mut self.bucket.warnings.icon_size_too_large;
            (self.style.icon_size, &f.icon_size, w)
        };
        let size = size_vertex(kind, size);
        if size.is_some_and(|s| s[0] > MAX_PACKED_SIZE || s[1] > MAX_PACKED_SIZE) {
            *warning = true;
        }
        size
    }

    /// GL JS `SymbolBucket.addSymbols`. Returns the placed symbol's index.
    #[allow(clippy::too_many_arguments)]
    fn add_symbols(
        &mut self,
        text: bool,
        quads: &[SymbolQuad],
        size: Option<[f64; 2]>,
        line_offset: [f64; 2],
        writing_mode: SymbolWritingMode,
        anchor: &SymbolAnchor,
        line: [u32; 2],
        instance: u32,
        associated_icon: u32,
    ) -> u32 {
        let stream: &mut SymbolStream = if text {
            &mut self.bucket.text
        } else {
            &mut self.bucket.icon
        };
        let symbol = stream.placed.len() as u32;
        let first_quad = stream.quads.len() as u32;
        let packed = size.map_or([0, 0], |s| {
            // `Math.min(MAX_PACKED_SIZE, Math.round(v))`; an absent upper
            // size is NaN there, and stores as 0.
            let pack = |v: f64| to_u16(js_min(MAX_PACKED_SIZE, js_round(v)));
            [pack(s[0]), pack(s[1])]
        });
        let fixed = |v: f64| to_i16(js_round(v * 32.0));
        for quad in quads {
            let y = quad.glyph_offset[1];
            let (texture, image_flag) = match quad.texture {
                Texture::Glyph(id) => (id.0, 0),
                Texture::Image(index) => (index, quad_flags::IMAGE),
            };
            stream.quads.push(Quad {
                corners: [
                    [fixed(quad.tl.x), fixed(y + quad.tl.y)],
                    [fixed(quad.tr.x), fixed(y + quad.tr.y)],
                    [fixed(quad.bl.x), fixed(y + quad.bl.y)],
                    [fixed(quad.br.x), fixed(y + quad.br.y)],
                ],
                tex: [
                    to_u16(quad.tex.x),
                    to_u16(quad.tex.y),
                    to_u16(quad.tex.x + quad.tex.w),
                    to_u16(quad.tex.y + quad.tex.h),
                ],
                pixel_offsets: [
                    to_i16(quad.pixel_offset_tl.x * 16.0),
                    to_i16(quad.pixel_offset_tl.y * 16.0),
                    to_i16(quad.pixel_offset_br.x * 16.0),
                    to_i16(quad.pixel_offset_br.y * 16.0),
                ],
                min_font_scale: [
                    to_i16(quad.min_font_scale[0] * 256.0),
                    to_i16(quad.min_font_scale[1] * 256.0),
                ],
                texture,
                symbol,
                section: quad.section as u16,
                flags: image_flag | if quad.sdf { quad_flags::SDF } else { 0 },
            });
            stream.glyph_offsets.push(quad.glyph_offset[0] as f32);
        }
        stream.placed.push(PlacedSymbol {
            anchor: [to_i16(anchor.point.x), to_i16(anchor.point.y)],
            segment: anchor.segment.map_or(0, |s| to_u16(s as f64)),
            writing_mode: writing_mode as u8,
            _pad: 0,
            quads: [first_quad, quads.len() as u32],
            line,
            size: packed,
            line_offset: [line_offset[0] as f32, line_offset[1] as f32],
            instance,
            associated_icon,
        });
        symbol
    }
}

fn resolve_justify(justify: TextJustify, anchor: Anchor) -> Justify {
    match justify {
        TextJustify::Auto => anchor_justification(anchor),
        TextJustify::Left => Justify::Left,
        TextJustify::Center => Justify::Center,
        TextJustify::Right => Justify::Right,
    }
}

/// The per-feature size GL JS bakes into vertices, × 128: `[size, NaN]`
/// for source sizes, the two stops for composite ones, none otherwise.
fn size_vertex(kind: SizeKind, size: &FeatureSize) -> Option<[f64; 2]> {
    match kind {
        SizeKind::Source => Some([SIZE_PACK_FACTOR * size.tile, f64::NAN]),
        SizeKind::Composite { .. } => Some([
            SIZE_PACK_FACTOR * size.stops[0],
            SIZE_PACK_FACTOR * size.stops[1],
        ]),
        SizeKind::Constant | SizeKind::Camera => None,
    }
}
