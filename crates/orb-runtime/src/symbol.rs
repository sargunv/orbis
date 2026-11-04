//! The text stage of a tile job (decision 0009, step 2): one symbol
//! layer's features evaluated into `orb-symbol` inputs, its glyph requests,
//! and its layout.
//!
//! Everything an expression decides is evaluated here, on the job side, as
//! GL JS's `SymbolBucket.populate` and `performSymbolLayout` evaluate it;
//! `orb-symbol` only lays out.

use std::collections::HashMap;

use orb_expr::{FormattedSection, Kind};
use orb_symbol::{
    FeatureLayout, FeatureSize, IconImage, LayerLayout, LayoutContext, Placement, SizeKind,
    SymbolBucket, SymbolFeature, TextFit, TileImages, TileParams, quad_flags,
};
use orb_text::{Content, FontStack, GlyphKey, GlyphRequests, Label, Section, VerticalAlign};
use orb_tile::{Layer, ValueRef};
#[cfg(test)]
use orb_v8::LayerKind;
use orb_v8::{Property, Symbol};

/// Sprite metadata: what icon layout needs of every image, by name
/// (namespaced as the sprites expose them), at the pixel ratio the map
/// draws.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpriteIcons {
    /// The device pixel ratio images are drawn at.
    pub pixel_ratio: f64,
    pub images: HashMap<String, IconImage>,
}

impl SpriteIcons {
    /// The icons as one JSON document, for a job worker.
    pub fn to_json(&self) -> String {
        use orb_expr::json::{Json, JsonValue};
        let j = |value| Json::new(value, Default::default());
        let num = |n: f64| j(JsonValue::Number(n));
        let ranges = |r: &Option<Vec<[f64; 2]>>| match r {
            None => j(JsonValue::Null),
            Some(r) => j(JsonValue::Array(
                r.iter().flat_map(|&[a, b]| [num(a), num(b)]).collect(),
            )),
        };
        let fit = |f: Option<TextFit>| match f {
            None => j(JsonValue::Null),
            Some(f) => num(f as u8 as f64),
        };
        let mut names: Vec<&String> = self.images.keys().collect();
        names.sort();
        let images = names
            .into_iter()
            .map(|name| {
                let i = &self.images[name];
                j(JsonValue::Array(vec![
                    j(JsonValue::String(name.clone())),
                    num(f64::from(i.size[0])),
                    num(f64::from(i.size[1])),
                    num(i.pixel_ratio),
                    j(JsonValue::Bool(i.sdf)),
                    ranges(&i.stretch_x),
                    ranges(&i.stretch_y),
                    match i.content {
                        None => j(JsonValue::Null),
                        Some(c) => j(JsonValue::Array(c.iter().map(|&v| num(v)).collect())),
                    },
                    fit(i.text_fit_width),
                    fit(i.text_fit_height),
                ]))
            })
            .collect();
        j(JsonValue::Array(vec![
            num(self.pixel_ratio),
            j(JsonValue::Array(images)),
        ]))
        .to_string()
    }

    /// Reads [`to_json`](Self::to_json)'s document.
    pub fn parse(text: &str) -> Result<Self, String> {
        use orb_expr::json::Json;
        let bad = || "malformed sprite icons".to_string();
        let doc = orb_expr::json::parse(text).map_err(|e| e.to_string())?;
        let [ratio, images] = doc.as_array().ok_or_else(bad)? else {
            return Err(bad());
        };
        let num = |v: &Json| v.as_f64().ok_or_else(bad);
        let ranges = |v: &Json| -> Result<Option<Vec<[f64; 2]>>, String> {
            if v.is_null() {
                return Ok(None);
            }
            let items = v.as_array().ok_or_else(bad)?;
            items
                .as_chunks::<2>()
                .0
                .iter()
                .map(|[a, b]| Ok([num(a)?, num(b)?]))
                .collect::<Result<_, String>>()
                .map(Some)
        };
        let fit = |v: &Json| -> Result<Option<TextFit>, String> {
            Ok(match v.as_f64() {
                None => None,
                Some(1.0) => Some(TextFit::StretchOnly),
                Some(2.0) => Some(TextFit::Proportional),
                Some(_) => Some(TextFit::StretchOrShrink),
            })
        };
        let mut icons = Self {
            pixel_ratio: num(ratio)?,
            images: HashMap::new(),
        };
        for image in images.as_array().ok_or_else(bad)? {
            let [name, w, h, r, sdf, sx, sy, content, fw, fh] = image.as_array().ok_or_else(bad)?
            else {
                return Err(bad());
            };
            let content = match content.as_array() {
                Some([a, b, c, d]) => Some([num(a)?, num(b)?, num(c)?, num(d)?]),
                _ => None,
            };
            icons.images.insert(
                name.as_str().ok_or_else(bad)?.to_string(),
                IconImage {
                    size: [num(w)? as u32, num(h)? as u32],
                    pixel_ratio: num(r)?,
                    sdf: sdf.as_bool().ok_or_else(bad)?,
                    stretch_x: ranges(sx)?,
                    stretch_y: ranges(sy)?,
                    content,
                    text_fit_width: fit(fw)?,
                    text_fit_height: fit(fh)?,
                },
            );
        }
        Ok(icons)
    }

    /// The icons of loaded sprite images.
    pub(crate) fn add(&mut self, name: &str, size: [u32; 2], m: &orb_render::ImageMetadata) {
        let fit = |f: &Option<String>| {
            f.as_deref().map(|f| match f {
                "stretchOnly" => TextFit::StretchOnly,
                "proportional" => TextFit::Proportional,
                _ => TextFit::StretchOrShrink,
            })
        };
        let ranges =
            |r: &[[f32; 2]]| (!r.is_empty()).then(|| r.iter().map(|s| s.map(f64::from)).collect());
        self.images.insert(
            name.to_string(),
            IconImage {
                size,
                pixel_ratio: f64::from(m.pixel_ratio),
                sdf: m.sdf,
                stretch_x: ranges(&m.stretch_x),
                stretch_y: ranges(&m.stretch_y),
                content: m.content.map(|c| c.map(f64::from)),
                text_fit_width: fit(&m.text_fit_width),
                text_fit_height: fit(&m.text_fit_height),
            },
        );
    }
}

/// One symbol layer's features of a tile, evaluated, and the glyphs they
/// need.
#[derive(Debug)]
pub(crate) struct Prepared {
    pub(crate) style: LayerLayout,
    pub(crate) features: Vec<SymbolFeature>,
    /// By feature index, for features whose `format` sections override
    /// `text-color`: per section layout keeps, its premultiplied colour as
    /// a property row packs it.
    pub(crate) section_colors: HashMap<u32, Vec<Option<u32>>>,
    /// By feature index, what its symbol reads as (the semantic snapshot):
    /// its text sections before `text-transform` and shaping, or its icon.
    pub(crate) names: HashMap<u32, String>,
    pub(crate) requests: GlyphRequests,
}

/// What a symbol layer's tile job produces.
#[derive(Debug, Default)]
pub(crate) struct SymbolBuild {
    pub(crate) bucket: SymbolBucket,
    /// Per tile-local glyph id, what it resolved to.
    pub(crate) glyphs: Vec<Option<GlyphKey>>,
    /// Tile-local image indices' names.
    pub(crate) images: Vec<String>,
    /// Per symbol instance, the property-row slot of its feature.
    pub(crate) instance_rows: Vec<u32>,
    /// Per text quad with [`quad_flags::COLOR`], its colour; empty when no
    /// quad has one.
    pub(crate) text_colors: Vec<u32>,
    /// Per symbol instance, the byte range of its name in `name_bytes`.
    pub(crate) name_ranges: Vec<u32>,
    pub(crate) name_bytes: Vec<u8>,
}

fn anchor(a: orb_v8::Anchor) -> orb_text::Anchor {
    use orb_text::Anchor as T;
    use orb_v8::Anchor as V;
    match a {
        V::Center => T::Center,
        V::Left => T::Left,
        V::Right => T::Right,
        V::Top => T::Top,
        V::Bottom => T::Bottom,
        V::TopLeft => T::TopLeft,
        V::TopRight => T::TopRight,
        V::BottomLeft => T::BottomLeft,
        V::BottomRight => T::BottomRight,
    }
}

/// GL JS `getCoveringZoomStops`: the stops covering `[zoom, zoom + 1]`.
fn covering_stops(stops: &[f64], zoom: f64) -> [f64; 2] {
    let mut lower = 0;
    while lower < stops.len() && stops[lower] <= zoom {
        lower += 1;
    }
    let lower = lower.saturating_sub(1);
    let mut upper = lower;
    while upper < stops.len() && stops[upper] < zoom + 1.0 {
        upper += 1;
    }
    let upper = upper.min(stops.len() - 1);
    [stops[lower], stops[upper]]
}

/// How a size property is drawn (GL JS `getSizeData`).
pub(crate) fn size_kind(property: &Property<f64>, zoom: f64) -> SizeKind {
    match property.kind() {
        Kind::Constant => SizeKind::Constant,
        Kind::Source => SizeKind::Source,
        Kind::Camera => SizeKind::Camera,
        Kind::Composite => {
            let stops = property
                .as_expression()
                .and_then(|e| e.zoom_curve())
                .map_or(&[][..], |c| &c.stops);
            match stops {
                [] => SizeKind::Source,
                stops => SizeKind::Composite {
                    zooms: covering_stops(stops, zoom),
                },
            }
        }
    }
}

/// The layer-wide layout, after GL JS's `SymbolStyleLayer.recalculate`.
fn layer_layout(symbol: &Symbol, zoom: f64) -> LayerLayout {
    let l = &symbol.layout;
    let placement = match l.placement.evaluate(zoom) {
        orb_v8::SymbolPlacement::Point => Placement::Point,
        orb_v8::SymbolPlacement::Line => Placement::Line,
        orb_v8::SymbolPlacement::LineCenter => Placement::LineCenter,
    };
    let text_rotation_alignment = match l.text_rotation_alignment.evaluate(zoom) {
        orb_v8::TextRotationAlignment::Map => orb_symbol::Alignment::Map,
        orb_v8::TextRotationAlignment::Viewport => orb_symbol::Alignment::Viewport,
        orb_v8::TextRotationAlignment::ViewportGlyph => orb_symbol::Alignment::ViewportGlyph,
        orb_v8::TextRotationAlignment::Auto => orb_symbol::Alignment::Auto,
    }
    .resolve(placement);
    // A data-driven or `auto` icon alignment resolves for the placement.
    let icon_rotation_alignment = match l.icon_rotation_alignment.as_constant() {
        Some(orb_v8::Alignment::Map) => orb_symbol::Alignment::Map,
        Some(orb_v8::Alignment::Viewport) => orb_symbol::Alignment::Viewport,
        _ => orb_symbol::Alignment::Auto,
    }
    .resolve(placement);
    let text_writing_mode = {
        let modes = l.text_writing_mode.evaluate(zoom).unwrap_or_default();
        let mut deduped = Vec::new();
        for mode in modes {
            let mode = match mode {
                orb_v8::WritingMode::Horizontal => orb_text::WritingMode::Horizontal,
                orb_v8::WritingMode::Vertical => orb_text::WritingMode::Vertical,
            };
            if !deduped.contains(&mode) {
                deduped.push(mode);
            }
        }
        if placement == Placement::Point && deduped.is_empty() {
            deduped.push(orb_text::WritingMode::Horizontal);
        }
        deduped
    };
    LayerLayout {
        placement,
        spacing: l.spacing.evaluate(zoom),
        text_rotation_alignment,
        icon_rotation_alignment,
        text_keep_upright: l.text_keep_upright.evaluate(zoom),
        text_max_angle: l.text_max_angle.evaluate(zoom),
        text_line_height: l.text_line_height.evaluate(zoom),
        text_padding: l.text_padding.evaluate(zoom),
        text_writing_mode,
        text_variable_anchor: l
            .text_variable_anchor
            .evaluate(zoom)
            .map(|a| a.into_iter().map(anchor).collect()),
        text_radial_offset_set: l.text_radial_offset_set,
        icon_text_fit: match l.icon_text_fit.evaluate(zoom) {
            orb_v8::IconTextFit::None => orb_symbol::IconTextFit::None,
            orb_v8::IconTextFit::Width => orb_symbol::IconTextFit::Width,
            orb_v8::IconTextFit::Height => orb_symbol::IconTextFit::Height,
            orb_v8::IconTextFit::Both => orb_symbol::IconTextFit::Both,
        },
        icon_text_fit_padding: l.icon_text_fit_padding.evaluate(zoom),
        icon_rotate_constant: matches!(l.icon_rotate.kind(), Kind::Constant | Kind::Camera)
            .then(|| l.icon_rotate.evaluate(zoom)),
        text_size: size_kind(&l.text_size, zoom),
        icon_size: size_kind(&l.icon_size, zoom),
    }
}

/// A `formatted` value as a label, with `text-transform` and Arabic
/// shaping applied to each text section (GL JS `transformText`).
fn label(sections: &[FormattedSection], transform: orb_text::TextTransform) -> Label {
    Label {
        sections: sections
            .iter()
            .map(|s| {
                let section = match &s.image {
                    Some((name, _)) => Section::image(name),
                    None => Section::text(
                        &orb_text::transform_text(&s.text, transform),
                        s.scale,
                        s.font_stack.as_deref().map(FontStack::from_joined),
                    ),
                };
                section.with_vertical_align(match s.vertical_align.as_deref() {
                    Some("center") => VerticalAlign::Center,
                    Some("top") => VerticalAlign::Top,
                    _ => VerticalAlign::Bottom,
                })
            })
            .collect(),
    }
}

/// Evaluates a symbol layer's `features` (the filter's selection, in
/// order) at tile zoom `zoom`. `ids` is the column of stable ids, the
/// promoted id when the source has one.
pub(crate) fn prepare(
    symbol: &Symbol,
    layer: &Layer,
    features: &[u32],
    zoom: f64,
    ids: Option<&orb_tile::Column>,
) -> Prepared {
    let style = layer_layout(symbol, zoom);
    let l = &symbol.layout;
    // GL JS `populate`'s hasText.
    let has_text = (l.text_field.kind() != Kind::Constant
        || l.text_field
            .as_constant()
            .is_some_and(|s| s.iter().any(|s| !s.text.is_empty() || s.image.is_some())))
        && (l.text_font.kind() != Kind::Constant
            || l.text_font.as_constant().is_some_and(|f| !f.is_empty()));
    let sort_by_key = l.z_order.evaluate(zoom) != orb_v8::SymbolZOrder::ViewportY
        && matches!(l.sort_key.kind(), Kind::Source | Kind::Composite);

    let text_field = l.text_field.bind(layer);
    let text_transform = l.text_transform.bind(layer);
    let text_font = l.text_font.bind(layer);
    let icon_image = l.icon_image.bind(layer);
    let sort_key = l.sort_key.bind(layer);
    let text_size = l.text_size.bind(layer);
    let icon_size = l.icon_size.bind(layer);
    let letter_spacing = l.text_letter_spacing.bind(layer);
    let text_anchor = l.text_anchor.bind(layer);
    let text_justify = l.text_justify.bind(layer);
    let max_width = l.text_max_width.bind(layer);
    let text_offset = l.text_offset.bind(layer);
    let radial_offset = l.text_radial_offset.bind(layer);
    let anchor_offset = l.text_variable_anchor_offset.bind(layer);
    let text_rotate = l.text_rotate.bind(layer);
    let icon_offset = l.icon_offset.bind(layer);
    let icon_anchor = l.icon_anchor.bind(layer);
    let icon_rotate = l.icon_rotate.bind(layer);
    let icon_padding = l.icon_padding.bind(layer);
    let height_offset = l.height_offset.bind(layer);

    let sizes = |bound: &orb_v8::BoundProperty<'_, f64>, kind: SizeKind, row: usize| {
        let stops = match kind {
            SizeKind::Composite { zooms } => zooms.map(|z| bound.evaluate(z, row)),
            _ => [0.0; 2],
        };
        FeatureSize {
            tile: bound.evaluate(zoom, row),
            layout: bound.evaluate(zoom + 1.0, row),
            max: bound.evaluate(18.0, row),
            stops,
        }
    };

    let mut prepared = Vec::with_capacity(features.len());
    let mut section_colors = HashMap::new();
    let mut names = HashMap::new();
    for &index in features {
        let row = index as usize;
        let text = has_text.then(|| {
            let transform = match text_transform.evaluate(zoom, row) {
                orb_v8::TextTransform::None => orb_text::TextTransform::None,
                orb_v8::TextTransform::Uppercase => orb_text::TextTransform::Uppercase,
                orb_v8::TextTransform::Lowercase => orb_text::TextTransform::Lowercase,
            };
            let sections = text_field.evaluate(zoom, row);
            let name: String = sections
                .iter()
                .filter(|s| s.image.is_none())
                .map(|s| s.text.as_str())
                .collect();
            if !name.trim().is_empty() {
                names.insert(index, name);
            }
            if sections.iter().any(|s| s.text_color.is_some()) {
                // Layout drops image sections without a name.
                let colors = sections
                    .iter()
                    .filter(|s| s.image.as_ref().is_none_or(|(name, _)| !name.is_empty()))
                    .map(|s| {
                        s.text_color
                            .as_ref()
                            .map(|c| orb_render::pack_color(c.premultiplied()))
                    })
                    .collect();
                section_colors.insert(index, colors);
            }
            label(&sections, transform)
        });
        let icon: Option<Box<str>> = icon_image
            .evaluate(zoom, row)
            .filter(|name| !name.is_empty())
            .map(Into::into);
        if text.is_none() && icon.is_none() {
            continue;
        }
        if let Some(icon) = &icon {
            names.entry(index).or_insert_with(|| icon.to_string());
        }
        let layout = FeatureLayout {
            text_font: FontStack::new(&text_font.evaluate(zoom, row)),
            text_size: sizes(&text_size, style.text_size, row),
            icon_size: sizes(&icon_size, style.icon_size, row),
            text_letter_spacing: letter_spacing.evaluate(zoom, row),
            text_anchor: anchor(text_anchor.evaluate(zoom, row)),
            text_justify: match text_justify.evaluate(zoom, row) {
                orb_v8::TextJustify::Auto => orb_symbol::TextJustify::Auto,
                orb_v8::TextJustify::Left => orb_symbol::TextJustify::Left,
                orb_v8::TextJustify::Center => orb_symbol::TextJustify::Center,
                orb_v8::TextJustify::Right => orb_symbol::TextJustify::Right,
            },
            text_max_width: max_width.evaluate(zoom, row),
            text_offset: text_offset.evaluate(zoom, row),
            text_radial_offset: radial_offset.evaluate(zoom, row),
            text_variable_anchor_offset: anchor_offset
                .evaluate(zoom, row)
                .map(|o| o.into_iter().map(|(a, o)| (anchor(a), o)).collect()),
            text_rotate: text_rotate.evaluate(zoom, row),
            icon_offset: icon_offset.evaluate(zoom, row),
            icon_anchor: anchor(icon_anchor.evaluate(zoom, row)),
            icon_rotate: icon_rotate.evaluate(zoom, row),
            icon_padding: icon_padding.evaluate(zoom, row).0,
            height_offset: height_offset.evaluate(zoom, row),
        };
        prepared.push(SymbolFeature {
            index,
            stable_id: ids.and_then(|ids| ids.get(row)).map(ValueRef::to_owned),
            text,
            icon,
            sort_key: sort_by_key.then(|| sort_key.evaluate(zoom, row).unwrap_or(0.0)),
            layout,
        });
    }
    let mut requests = GlyphRequests::default();
    orb_symbol::add_glyph_requests(layer, &prepared, &style, &mut requests);
    Prepared {
        style,
        features: prepared,
        section_colors,
        names,
        requests,
    }
}

/// Whether any label has two neighbouring characters of a script written
/// without spaces, between which GL JS asks its word segmenter for a break.
pub(crate) fn needs_word_breaks(prepared: &Prepared) -> bool {
    prepared.features.iter().any(|f| {
        f.text
            .iter()
            .flat_map(|t| &t.sections)
            .any(|s| match &s.content {
                Content::Text(text) => text.chars().zip(text.chars().skip(1)).any(|(a, b)| {
                    orb_text::unicode::is_written_without_spaces(a)
                        && orb_text::unicode::is_written_without_spaces(b)
                }),
                Content::Image(_) => false,
            })
    })
}

/// The images a layer's features name that the sprites have, numbered for
/// layout.
fn tile_images(features: &[SymbolFeature], icons: &SpriteIcons) -> TileImages {
    let mut images = TileImages::default();
    let mut add = |name: &str| {
        if images.get(name).is_none()
            && let Some(image) = icons.images.get(name)
        {
            images.insert(name, image.clone());
        }
    };
    for feature in features {
        if let Some(icon) = &feature.icon {
            add(icon);
        }
        for section in feature.text.iter().flat_map(|t| &t.sections) {
            if let Content::Image(name) = &section.content {
                add(name);
            }
        }
    }
    images
}

/// Lays out a prepared layer. `slots` maps a feature's tile index to its
/// property-row slot.
pub(crate) fn layout(
    prepared: Prepared,
    layer: &Layer,
    source_layer: &str,
    glyphs: orb_text::GlyphMap,
    icons: &SpriteIcons,
    slots: &HashMap<u32, u32>,
    words: Option<&orb_text::WordBreaker>,
) -> SymbolBuild {
    let images = tile_images(&prepared.features, icons);
    let mut bucket = orb_symbol::layout(
        layer,
        &prepared.features,
        &prepared.style,
        TileParams {
            source_layer,
            // Orbis slices overzoomed tiles to their own extent, so every
            // tile lays out at its own scale (decision 0009).
            overscaling: 1.0,
            pixel_ratio: icons.pixel_ratio,
        },
        LayoutContext {
            glyphs: &glyphs,
            images: &images,
            words,
        },
    );
    let glyph_keys = (0..glyphs.len() as u32)
        .map(|id| glyphs.glyph(orb_text::GlyphId(id)).map(|g| g.key.clone()))
        .collect();
    let instance_rows = bucket
        .instances
        .iter()
        .flat_map(|i| {
            [
                slots.get(&i.feature).copied().unwrap_or(0),
                i.height_offset.to_bits(),
            ]
        })
        .collect();
    let text_colors = override_colors(&mut bucket, &prepared.section_colors);
    let (mut name_ranges, mut name_bytes) = (Vec::new(), Vec::new());
    for instance in &bucket.instances {
        let name = prepared
            .names
            .get(&instance.feature)
            .map_or("", String::as_str);
        name_ranges.extend([name_bytes.len() as u32, name.len() as u32]);
        name_bytes.extend_from_slice(name.as_bytes());
    }
    SymbolBuild {
        glyphs: glyph_keys,
        images: images.names().map(str::to_string).collect(),
        instance_rows,
        text_colors,
        name_ranges,
        name_bytes,
        bucket,
    }
}

/// Flags the text quads whose section overrides `text-color` and returns
/// the colour table, per text quad.
fn override_colors(
    bucket: &mut SymbolBucket,
    section_colors: &HashMap<u32, Vec<Option<u32>>>,
) -> Vec<u32> {
    if section_colors.is_empty() {
        return Vec::new();
    }
    let text = &mut bucket.text;
    let mut colors = vec![0; text.quads.len()];
    for (quad, color) in text.quads.iter_mut().zip(&mut colors) {
        let instance = text.placed[quad.symbol as usize].instance;
        let feature = bucket.instances[instance as usize].feature;
        if let Some(&Some(c)) = section_colors
            .get(&feature)
            .and_then(|s| s.get(usize::from(quad.section)))
        {
            quad.flags |= quad_flags::COLOR;
            *color = c;
        }
    }
    colors
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as Json;

    const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../orb-symbol/tests/data");

    fn num(v: &Json) -> f64 {
        match v {
            Json::String(s) => s.parse().unwrap(),
            v => v.as_f64().unwrap(),
        }
    }

    fn nums<const N: usize>(v: &Json) -> [f64; N] {
        let a = v.as_array().unwrap();
        std::array::from_fn(|i| num(&a[i]))
    }

    fn anchor_name(a: orb_text::Anchor) -> &'static str {
        use orb_text::Anchor as A;
        match a {
            A::Center => "center",
            A::Left => "left",
            A::Right => "right",
            A::Top => "top",
            A::Bottom => "bottom",
            A::TopLeft => "top-left",
            A::TopRight => "top-right",
            A::BottomLeft => "bottom-left",
            A::BottomRight => "bottom-right",
        }
    }

    fn check_size(case: &str, what: &str, size: &FeatureSize, v: &Json, kind: SizeKind) {
        let golden = FeatureSize {
            tile: num(&v["tile"]),
            layout: num(&v["layout"]),
            max: num(&v["max"]),
            stops: nums(&v["stops"]),
        };
        let mut size = *size;
        if !matches!(kind, SizeKind::Composite { .. }) {
            size.stops = golden.stops;
        }
        assert_eq!(size, golden, "{case}: {what}");
    }

    fn check_kind(case: &str, kind: SizeKind, v: &Json) {
        let expected = match v["kind"].as_str().unwrap() {
            "constant" => SizeKind::Constant,
            "source" => SizeKind::Source,
            "camera" => SizeKind::Camera,
            _ => SizeKind::Composite {
                zooms: [num(&v["minZoom"]), num(&v["maxZoom"])],
            },
        };
        assert_eq!(kind, expected, "{case}: size kind");
    }

    /// The runtime evaluates every GL JS whole-layout case's features
    /// exactly as `SymbolBucket.populate` and `performSymbolLayout` do.
    #[test]
    fn evaluates_layout_inputs_as_gl_js() {
        let cases: Json = serde_json::from_str(
            &std::fs::read_to_string(format!("{DATA}/layout-cases.json")).unwrap(),
        )
        .unwrap();
        for case in cases.as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let golden: Json = serde_json::from_str(
                &std::fs::read_to_string(format!("{DATA}/layout-{name}.json")).unwrap(),
            )
            .unwrap();
            let tile = case["tile"].as_str().unwrap();
            let bytes = std::fs::read(format!("{DATA}/tiles/{tile}")).unwrap();
            let frame = orb_tile::mvt::decode(&bytes).unwrap();
            let source_layer = case["sourceLayer"].as_str().unwrap();
            let layer = frame.layer(source_layer).unwrap();
            let mut source = serde_json::json!({"type": "vector", "tiles": ["test://{z}/{x}/{y}"]});
            if let Some(p) = case.get("promoteId") {
                source["promoteId"] = p.clone();
            }
            let mut style_layer = serde_json::json!({
                "id": "l", "type": "symbol", "source": "s", "source-layer": source_layer,
                "layout": case["layout"],
            });
            if let Some(f) = case.get("filter") {
                style_layer["filter"] = f.clone();
            }
            let style = serde_json::json!({
                "version": 8, "sources": {"s": source}, "layers": [style_layer],
            });
            let style = orb_v8::parse(&style.to_string()).unwrap_or_else(|e| panic!("{name}: {e}"));
            let LayerKind::Symbol(symbol) = &style.layers[0].kind else {
                panic!("a symbol layer")
            };
            let zoom = num(&case["zoom"]);
            let features: Vec<u32> = match &symbol.selection.filter {
                Some(filter) => filter.select(layer, zoom),
                None => (0..layer.len() as u32).collect(),
            };
            let ids = case
                .get("promoteId")
                .and_then(|p| layer.property(p.as_str().unwrap()));
            let prepared = prepare(symbol, layer, &features, zoom, ids);

            let g = &golden["layer"];
            let s = &prepared.style;
            check_kind(name, s.text_size, &g["textSize"]);
            check_kind(name, s.icon_size, &g["iconSize"]);
            let placement = match s.placement {
                Placement::Point => "point",
                Placement::Line => "line",
                Placement::LineCenter => "line-center",
            };
            assert_eq!(placement, g["placement"], "{name}: placement");
            let alignment = |a: orb_symbol::Alignment| match a {
                orb_symbol::Alignment::Map => "map",
                orb_symbol::Alignment::Viewport => "viewport",
                orb_symbol::Alignment::ViewportGlyph => "viewport-glyph",
                orb_symbol::Alignment::Auto => "auto",
            };
            assert_eq!(
                alignment(s.text_rotation_alignment),
                g["textRotationAlignment"],
                "{name}"
            );
            assert_eq!(
                alignment(s.icon_rotation_alignment),
                g["iconRotationAlignment"],
                "{name}"
            );
            assert_eq!(s.spacing, num(&g["spacing"]), "{name}: spacing");
            assert_eq!(
                s.text_keep_upright,
                g["keepUpright"].as_bool().unwrap(),
                "{name}"
            );
            assert_eq!(s.text_max_angle, num(&g["maxAngle"]), "{name}");
            assert_eq!(s.text_line_height, num(&g["lineHeight"]), "{name}");
            assert_eq!(s.text_padding, num(&g["textPadding"]), "{name}");
            let modes: Vec<&str> = s
                .text_writing_mode
                .iter()
                .map(|m| match m {
                    orb_text::WritingMode::Horizontal => "horizontal",
                    orb_text::WritingMode::Vertical => "vertical",
                })
                .collect();
            let golden_modes: Vec<&str> = g["writingModes"]
                .as_array()
                .map(|m| m.iter().map(|m| m.as_str().unwrap()).collect())
                .unwrap_or_default();
            assert_eq!(modes, golden_modes, "{name}: writing modes");
            let variable: Option<Vec<&str>> = s
                .text_variable_anchor
                .as_ref()
                .map(|a| a.iter().map(|&a| anchor_name(a)).collect());
            let golden_variable: Option<Vec<&str>> = g["variableAnchor"]
                .as_array()
                .map(|a| a.iter().map(|a| a.as_str().unwrap()).collect());
            assert_eq!(variable, golden_variable, "{name}: variable anchors");
            assert_eq!(
                s.text_radial_offset_set,
                g["radialOffsetSet"].as_bool().unwrap(),
                "{name}"
            );
            assert_eq!(
                s.icon_rotate_constant,
                (!g["iconRotateConstant"].is_null()).then(|| num(&g["iconRotateConstant"])),
                "{name}: icon-rotate constant"
            );
            let fit = match s.icon_text_fit {
                orb_symbol::IconTextFit::None => "none",
                orb_symbol::IconTextFit::Width => "width",
                orb_symbol::IconTextFit::Height => "height",
                orb_symbol::IconTextFit::Both => "both",
            };
            assert_eq!(fit, g["iconTextFit"], "{name}");
            assert_eq!(
                s.icon_text_fit_padding,
                nums(&g["iconTextFitPadding"]),
                "{name}"
            );

            let golden_features = golden["features"].as_array().unwrap();
            assert_eq!(
                prepared.features.len(),
                golden_features.len(),
                "{name}: features"
            );
            for (f, g) in prepared.features.iter().zip(golden_features) {
                let at = format!("{name} feature {}", f.index);
                assert_eq!(f64::from(f.index), num(&g["index"]), "{at}");
                let id = match &f.stable_id {
                    None => Json::Null,
                    Some(orb_tile::Value::Str(s)) => Json::String(s.clone()),
                    Some(orb_tile::Value::Json(s)) => serde_json::from_str(s).unwrap(),
                    Some(orb_tile::Value::F64(n)) => serde_json::json!(n),
                    Some(orb_tile::Value::I64(n)) => serde_json::json!(*n as f64),
                    Some(orb_tile::Value::U64(n)) => serde_json::json!(*n as f64),
                    Some(orb_tile::Value::Bool(b)) => Json::Bool(*b),
                };
                assert_eq!(id, g["id"], "{at}: id");
                match (&f.text, g["text"].as_array()) {
                    (None, None) => {}
                    (Some(label), Some(sections)) => {
                        assert_eq!(label.sections.len(), sections.len(), "{at}: sections");
                        for (s, g) in label.sections.iter().zip(sections) {
                            match &s.content {
                                Content::Text(t) => {
                                    assert_eq!(t, g["text"].as_str().unwrap(), "{at}");
                                    assert!(g["image"].is_null(), "{at}");
                                }
                                Content::Image(i) => {
                                    assert_eq!(i, g["image"].as_str().unwrap(), "{at}")
                                }
                            }
                            assert_eq!(
                                s.scale,
                                (!g["scale"].is_null()).then(|| num(&g["scale"])),
                                "{at}"
                            );
                            assert_eq!(
                                s.font_stack.as_ref().map(FontStack::as_str),
                                g["fontStack"].as_str(),
                                "{at}"
                            );
                            let align = match s.vertical_align {
                                VerticalAlign::Bottom => None,
                                VerticalAlign::Center => Some("center"),
                                VerticalAlign::Top => Some("top"),
                            };
                            let golden_align =
                                g["verticalAlign"].as_str().filter(|a| *a != "bottom");
                            assert_eq!(align, golden_align, "{at}");
                        }
                    }
                    other => panic!("{at}: text {other:?}"),
                }
                assert_eq!(f.icon.as_deref(), g["icon"].as_str(), "{at}: icon");
                assert_eq!(
                    f.sort_key,
                    (!g["sortKey"].is_null()).then(|| num(&g["sortKey"])),
                    "{at}"
                );
                let l = &f.layout;
                let font: Vec<&str> = g["textFont"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| s.as_str().unwrap())
                    .collect();
                assert_eq!(l.text_font, FontStack::new(&font), "{at}");
                check_size(&at, "text-size", &l.text_size, &g["textSize"], s.text_size);
                check_size(&at, "icon-size", &l.icon_size, &g["iconSize"], s.icon_size);
                assert_eq!(l.text_letter_spacing, num(&g["textLetterSpacing"]), "{at}");
                assert_eq!(anchor_name(l.text_anchor), g["textAnchor"], "{at}");
                let justify = match l.text_justify {
                    orb_symbol::TextJustify::Auto => "auto",
                    orb_symbol::TextJustify::Left => "left",
                    orb_symbol::TextJustify::Center => "center",
                    orb_symbol::TextJustify::Right => "right",
                };
                assert_eq!(justify, g["textJustify"], "{at}");
                assert_eq!(l.text_max_width, num(&g["textMaxWidth"]), "{at}");
                assert_eq!(l.text_offset, nums(&g["textOffset"]), "{at}");
                assert_eq!(l.text_radial_offset, num(&g["textRadialOffset"]), "{at}");
                let offsets = g["textVariableAnchorOffset"].as_array().map(|v| {
                    v.chunks(2)
                        .map(|p| (p[0].as_str().unwrap().to_string(), nums::<2>(&p[1])))
                        .collect::<Vec<_>>()
                });
                let mine = l.text_variable_anchor_offset.as_ref().map(|v| {
                    v.iter()
                        .map(|(a, o)| (anchor_name(*a).to_string(), *o))
                        .collect::<Vec<_>>()
                });
                assert_eq!(mine, offsets, "{at}: variable anchor offsets");
                assert_eq!(l.text_rotate, num(&g["textRotate"]), "{at}");
                assert_eq!(l.icon_offset, nums(&g["iconOffset"]), "{at}");
                assert_eq!(anchor_name(l.icon_anchor), g["iconAnchor"], "{at}");
                assert_eq!(l.icon_rotate, num(&g["iconRotate"]), "{at}");
                assert_eq!(l.icon_padding, nums(&g["iconPadding"]), "{at}");
                assert_eq!(l.height_offset, num(&g["heightOffset"]), "{at}");
            }
        }
    }

    #[test]
    fn covering_stops_follow_gl_js() {
        let stops = [10.0, 12.0, 14.0, 16.0];
        assert_eq!(covering_stops(&stops, 9.0), [10.0, 10.0]);
        assert_eq!(covering_stops(&stops, 12.0), [12.0, 14.0]);
        assert_eq!(covering_stops(&stops, 12.5), [12.0, 14.0]);
        assert_eq!(covering_stops(&stops, 13.0), [12.0, 14.0]);
        assert_eq!(covering_stops(&stops, 15.5), [14.0, 16.0]);
        assert_eq!(covering_stops(&stops, 20.0), [16.0, 16.0]);
    }
}

/// The tables of a symbol layer's tile, in [`ResultSymbols::tables`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolTable {
    Instances,
    TextPlaced,
    TextQuads,
    TextGlyphOffsets,
    IconPlaced,
    IconQuads,
    IconGlyphOffsets,
    Lines,
    AnchorOffsets,
    SortKeyRanges,
    /// Per instance, the property-row slot of its feature and its
    /// `symbol-height-offset` (f32 bits).
    InstanceRows,
    /// Per text quad, its `format` colour override (empty if none has one).
    TextColors,
    /// Per instance, the start and length of its name in [`NameBytes`](Self::NameBytes).
    NameRanges,
    /// The instances' names, UTF-8, padded to a word.
    NameBytes,
}

/// How many [`SymbolTable`]s a layer's tile has.
pub const SYMBOL_TABLES: usize = 14;

/// The scalars of a layer's bucket ([`orb_symbol::SymbolBucket`] without
/// its tables).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SymbolMeta {
    pub text_size: SizeKind,
    pub icon_size: SizeKind,
    pub tile_pixel_ratio: f64,
    pub sdf_icons: Option<bool>,
    pub icons_in_text: bool,
    pub icons_need_linear: bool,
    pub allow_vertical_placement: bool,
    pub max_height_offset: f64,
    pub warnings: orb_symbol::Warnings,
}

/// A symbol layer's share of a finished tile: its bucket's tables as word
/// streams, the glyph keys its quads name by tile-local glyph id, and its
/// images by tile-local index.
#[derive(Debug)]
pub struct ResultSymbols {
    /// The layer's key.
    pub layer: u32,
    pub meta: SymbolMeta,
    pub tables: [crate::Stream; SYMBOL_TABLES],
    /// By glyph id; `None` for requests nothing could draw.
    pub glyphs: Vec<Option<GlyphKey>>,
    pub images: Vec<String>,
}

impl ResultSymbols {
    pub fn table(&self, table: SymbolTable) -> &crate::Stream {
        &self.tables[table as usize]
    }

    /// The layer's tables copied into word streams.
    pub(crate) fn new(layer: u32, build: &SymbolBuild) -> Self {
        let b = &build.bucket;
        let words = |rows: &[u32]| crate::Stream::Core(rows.to_vec());
        let cast = |bytes: &[u8]| {
            crate::Stream::Core(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_ne_bytes(*w))
                    .collect(),
            )
        };
        let tables = [
            cast(bytemuck::cast_slice(&b.instances)),
            cast(bytemuck::cast_slice(&b.text.placed)),
            cast(bytemuck::cast_slice(&b.text.quads)),
            cast(bytemuck::cast_slice(&b.text.glyph_offsets)),
            cast(bytemuck::cast_slice(&b.icon.placed)),
            cast(bytemuck::cast_slice(&b.icon.quads)),
            cast(bytemuck::cast_slice(&b.icon.glyph_offsets)),
            cast(bytemuck::cast_slice(&b.lines)),
            cast(bytemuck::cast_slice(&b.anchor_offsets)),
            cast(bytemuck::cast_slice(&b.sort_key_ranges)),
            words(&build.instance_rows),
            words(&build.text_colors),
            words(&build.name_ranges),
            {
                let mut bytes = build.name_bytes.clone();
                bytes.resize(bytes.len().next_multiple_of(4), 0);
                cast(&bytes)
            },
        ];
        Self {
            layer,
            meta: SymbolMeta {
                text_size: b.text_size,
                icon_size: b.icon_size,
                tile_pixel_ratio: b.tile_pixel_ratio,
                sdf_icons: b.sdf_icons,
                icons_in_text: b.icons_in_text,
                icons_need_linear: b.icons_need_linear,
                allow_vertical_placement: b.allow_vertical_placement,
                max_height_offset: b.max_height_offset,
                warnings: b.warnings,
            },
            tables,
            glyphs: build.glyphs.clone(),
            images: build.images.clone(),
        }
    }
}

fn write_f64(words: &mut Vec<u32>, v: f64) {
    let bits = v.to_bits();
    words.extend([bits as u32, (bits >> 32) as u32]);
}

fn read_f64(next: &mut impl FnMut() -> Result<u32, String>) -> Result<f64, String> {
    let lo = u64::from(next()?);
    Ok(f64::from_bits(lo | u64::from(next()?) << 32))
}

fn write_size(words: &mut Vec<u32>, size: SizeKind) {
    let (tag, zooms) = match size {
        SizeKind::Constant => (0, [0.0; 2]),
        SizeKind::Source => (1, [0.0; 2]),
        SizeKind::Camera => (2, [0.0; 2]),
        SizeKind::Composite { zooms } => (3, zooms),
    };
    words.push(tag);
    write_f64(words, zooms[0]);
    write_f64(words, zooms[1]);
}

fn read_size(next: &mut impl FnMut() -> Result<u32, String>) -> Result<SizeKind, String> {
    let tag = next()?;
    let zooms = [read_f64(next)?, read_f64(next)?];
    Ok(match tag {
        0 => SizeKind::Constant,
        1 => SizeKind::Source,
        2 => SizeKind::Camera,
        3 => SizeKind::Composite { zooms },
        _ => return Err("malformed symbol size".into()),
    })
}

/// Appends a layer's header words: its key, scalars, table lengths, glyph
/// keys and image names. Its tables follow in the payload.
pub(crate) fn write_header(words: &mut Vec<u32>, s: &ResultSymbols) {
    let m = &s.meta;
    words.push(s.layer);
    write_size(words, m.text_size);
    write_size(words, m.icon_size);
    write_f64(words, m.tile_pixel_ratio);
    write_f64(words, m.max_height_offset);
    let flags = [
        m.sdf_icons.is_some(),
        m.sdf_icons == Some(true),
        m.icons_in_text,
        m.icons_need_linear,
        m.allow_vertical_placement,
        m.warnings.mixed_sdf_icons,
        m.warnings.text_size_too_large,
        m.warnings.icon_size_too_large,
    ];
    words.push(flags.iter().rev().fold(0, |w, &f| w << 1 | u32::from(f)));
    words.extend(s.tables.iter().map(|t| match t {
        crate::Stream::Core(w) => w.len() as u32,
        crate::Stream::Host { len, .. } => *len,
    }));
    words.push(s.glyphs.len() as u32);
    for key in &s.glyphs {
        let variant = |v: orb_text::Variant| v as u32;
        match key {
            None => words.push(0),
            Some(GlyphKey::Pbf { stack, code_point }) => {
                words.push(1);
                crate::jobs::write_text(words, stack.as_str());
                words.push(*code_point);
            }
            Some(GlyphKey::Font {
                font,
                variant: v,
                text,
            }) => {
                words.extend([2, font.0 as u32, (font.0 >> 32) as u32, variant(*v)]);
                crate::jobs::write_text(words, text);
            }
            Some(GlyphKey::Cluster {
                stack,
                variant: v,
                text,
            }) => {
                words.push(3);
                crate::jobs::write_text(words, stack.as_str());
                words.push(variant(*v));
                crate::jobs::write_text(words, text);
            }
        }
    }
    words.push(s.images.len() as u32);
    for name in &s.images {
        crate::jobs::write_text(words, name);
    }
}

/// Reads [`write_header`]'s words; tables become host streams from
/// `offset`, which advances past them.
pub(crate) fn read_header(
    next: &mut impl FnMut() -> Result<u32, String>,
    offset: &mut u64,
) -> Result<ResultSymbols, String> {
    let bad = || "malformed packed symbols".to_string();
    let layer = next()?;
    let text_size = read_size(next)?;
    let icon_size = read_size(next)?;
    let tile_pixel_ratio = read_f64(next)?;
    let max_height_offset = read_f64(next)?;
    let flags = next()?;
    let flag = |i: u32| flags >> i & 1 != 0;
    let meta = SymbolMeta {
        text_size,
        icon_size,
        tile_pixel_ratio,
        sdf_icons: flag(0).then_some(flag(1)),
        icons_in_text: flag(2),
        icons_need_linear: flag(3),
        allow_vertical_placement: flag(4),
        max_height_offset,
        warnings: orb_symbol::Warnings {
            mixed_sdf_icons: flag(5),
            text_size_too_large: flag(6),
            icon_size_too_large: flag(7),
        },
    };
    let mut lens = [0u32; SYMBOL_TABLES];
    for len in &mut lens {
        *len = next()?;
    }
    let tables = lens.map(|len| {
        let s = crate::Stream::Host {
            offset: *offset,
            len,
        };
        *offset += u64::from(len) * 4;
        s
    });
    let variant = |v: u32| match v {
        0 => Ok(orb_text::Variant::Default),
        1 => Ok(orb_text::Variant::Vertical),
        _ => Err(bad()),
    };
    let glyphs = (0..next()?)
        .map(|_| {
            Ok(match next()? {
                0 => None,
                1 => Some(GlyphKey::Pbf {
                    stack: FontStack::from_joined(&crate::jobs::read_text(next)?),
                    code_point: next()?,
                }),
                2 => {
                    let lo = u64::from(next()?);
                    let font = orb_text::FontId(lo | u64::from(next()?) << 32);
                    Some(GlyphKey::Font {
                        font,
                        variant: variant(next()?)?,
                        text: crate::jobs::read_text(next)?.into(),
                    })
                }
                3 => Some(GlyphKey::Cluster {
                    stack: FontStack::from_joined(&crate::jobs::read_text(next)?),
                    variant: variant(next()?)?,
                    text: crate::jobs::read_text(next)?.into(),
                }),
                _ => return Err(bad()),
            })
        })
        .collect::<Result<_, String>>()?;
    let images = (0..next()?)
        .map(|_| crate::jobs::read_text(next))
        .collect::<Result<_, _>>()?;
    Ok(ResultSymbols {
        layer,
        meta,
        tables,
        glyphs,
        images,
    })
}
