//! Layers.

use std::collections::HashSet;

use orb_expr::json::{Json, JsonValue, Member};
use orb_expr::{Color, Filter, Kind, PropertyExpression, Value};

use crate::Cx;
use crate::property::{
    Alignment, Anchor, AnchorOffsets, HeightAnchor, HillshadeMethod, IconTextFit,
    IlluminationAnchor, LayerBlend, LineCap, LineJoin, Overlap, Padding, Property, PropertyValue,
    Resampling, SymbolPlacement, SymbolZOrder, TextJustify, TextRotationAlignment, TextTransform,
    TranslateAnchor, Visibility, WritingMode,
};
use crate::source::{Source, SourceKind};
use crate::spec::{self, Prop};

/// A style layer.
#[derive(Clone, Debug)]
pub struct Layer {
    pub id: String,
    /// The layer draws from this zoom, inclusive.
    pub minzoom: f64,
    /// The layer draws up to this zoom, exclusive. GL JS ignores a
    /// `maxzoom` of 0, which reads as unset.
    pub maxzoom: f64,
    pub visibility: Visibility,
    pub kind: LayerKind,
    /// Paint properties' own `*-transition`, by property name; the rest
    /// take the style's.
    pub transitions: Vec<(String, crate::TransitionOverride)>,
}

impl Layer {
    /// Whether the layer draws at `zoom`, as GL JS's `isHidden` decides: a
    /// zero `minzoom` or `maxzoom` sets no bound, so default layers draw at
    /// the negative zooms a map can reach.
    pub fn is_visible_at(&self, zoom: f64) -> bool {
        let above_min = self.minzoom == 0.0 || zoom >= self.minzoom;
        let below_max = self.maxzoom == 0.0 || zoom < self.maxzoom;
        self.visibility == Visibility::Visible && above_min && below_max
    }
}

#[derive(Clone, Debug)]
pub enum LayerKind {
    Background(Background),
    Fill(Box<Fill>),
    FillExtrusion(Box<FillExtrusion>),
    Line(Box<Line>),
    Circle(Box<Circle>),
    Heatmap(Box<Heatmap>),
    Raster(Box<Raster>),
    Hillshade(Box<Hillshade>),
    ColorRelief(Box<ColorRelief>),
    Symbol(Box<Symbol>),
    /// A v8 layer type Orbis doesn't draw yet.
    Unsupported {
        ty: String,
    },
}

/// Which features a vector layer draws.
#[derive(Clone, Debug)]
pub struct Selection {
    pub source: String,
    /// Required for vector sources.
    pub source_layer: Option<String>,
    pub filter: Option<Filter>,
}

#[derive(Clone, Debug)]
pub struct Background {
    pub paint: BackgroundPaint,
}

#[derive(Clone, Debug)]
pub struct BackgroundPaint {
    pub color: Property<Color>,
    /// Shared-atlas image pattern.
    pub pattern: Property<Option<String>>,
    pub opacity: Property<f32>,
}

#[derive(Clone, Debug)]
pub struct Fill {
    pub selection: Selection,
    pub layout: FillLayout,
    pub paint: FillPaint,
}

#[derive(Clone, Debug)]
pub struct FillLayout {
    pub sort_key: Property<Option<f32>>,
}

#[derive(Clone, Debug)]
pub struct FillPaint {
    pub antialias: Property<bool>,
    pub opacity: Property<f32>,
    /// Applied to the layer as a whole, once (`fill-layer-opacity`).
    pub layer_opacity: Property<f32>,
    /// How the layer, as a whole, combines with what is beneath it.
    pub layer_blend: Property<LayerBlend>,
    pub color: Property<Color>,
    /// `None` draws the outline in `color`.
    pub outline_color: Property<Option<Color>>,
    pub translate: Property<[f32; 2]>,
    pub translate_anchor: Property<TranslateAnchor>,
    /// Shared-atlas image pattern.
    pub pattern: Property<Option<String>>,
}

/// Polygon roofs and walls, in metres above the surface.
#[derive(Clone, Debug)]
pub struct FillExtrusion {
    pub selection: Selection,
    /// `fill-extrusion-rounded-corner-distance`, metres.
    pub rounded_corner_distance: Property<f32>,
    pub paint: FillExtrusionPaint,
}

#[derive(Clone, Debug)]
pub struct FillExtrusionPaint {
    pub color: Property<Color>,
    pub opacity: Property<f32>,
    pub height: Property<f32>,
    pub base: Property<f32>,
    pub translate: Property<[f32; 2]>,
    pub translate_anchor: Property<TranslateAnchor>,
    pub vertical_gradient: Property<bool>,
    pub pattern: Property<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct Line {
    pub selection: Selection,
    pub layout: LineLayout,
    pub paint: LinePaint,
}

#[derive(Clone, Debug)]
pub struct LineLayout {
    pub cap: Property<LineCap>,
    pub join: Property<LineJoin>,
    pub miter_limit: Property<f32>,
    pub round_limit: Property<f32>,
    pub sort_key: Property<Option<f32>>,
}

#[derive(Clone, Debug)]
pub struct LinePaint {
    pub opacity: Property<f32>,
    /// Applied to the layer as a whole, once (`line-layer-opacity`).
    pub layer_opacity: Property<f32>,
    /// How the layer, as a whole, combines with what is beneath it.
    pub layer_blend: Property<LayerBlend>,
    pub color: Property<Color>,
    pub translate: Property<[f32; 2]>,
    pub translate_anchor: Property<TranslateAnchor>,
    pub width: Property<f32>,
    pub gap_width: Property<f32>,
    pub offset: Property<f32>,
    pub blur: Property<f32>,
    /// SDF dash lengths in line-width units.
    pub dasharray: Property<Option<Vec<f32>>>,
    /// Shared-atlas image pattern.
    pub pattern: Property<Option<String>>,
    /// A colour ramp over `line-progress`, sampled per clipped line.
    pub gradient: Option<PropertyExpression>,
}

/// A circle layer over a vector or GeoJSON source.
#[derive(Clone, Debug)]
pub struct Circle {
    pub selection: Selection,
    pub layout: CircleLayout,
    pub paint: CirclePaint,
}

#[derive(Clone, Debug)]
pub struct CircleLayout {
    pub sort_key: Property<Option<f32>>,
}

#[derive(Clone, Debug)]
pub struct CirclePaint {
    pub radius: Property<f32>,
    pub color: Property<Color>,
    pub blur: Property<f32>,
    pub opacity: Property<f32>,
    pub translate: Property<[f32; 2]>,
    pub translate_anchor: Property<TranslateAnchor>,
    pub pitch_scale: Property<TranslateAnchor>,
    pub pitch_alignment: Property<TranslateAnchor>,
    pub stroke_width: Property<f32>,
    pub stroke_color: Property<Color>,
    pub stroke_opacity: Property<f32>,
}

/// Gaussian density coloured through a ramp over `heatmap-density`.
#[derive(Clone, Debug)]
pub struct Heatmap {
    pub selection: Selection,
    pub paint: HeatmapPaint,
}

#[derive(Clone, Debug)]
pub struct HeatmapPaint {
    pub radius: Property<f32>,
    pub weight: Property<f32>,
    pub intensity: Property<f32>,
    pub color: PropertyExpression,
    pub opacity: Property<f32>,
}

/// A `raster` layer: the images of a `raster` source.
#[derive(Clone, Debug)]
pub struct Raster {
    pub source: String,
    pub paint: RasterPaint,
}

#[derive(Clone, Debug)]
pub struct RasterPaint {
    pub opacity: Property<f32>,
    /// Degrees.
    pub hue_rotate: Property<f32>,
    pub brightness_min: Property<f32>,
    pub brightness_max: Property<f32>,
    pub saturation: Property<f32>,
    pub contrast: Property<f32>,
    /// `resampling`; either it or `raster-resampling` set to `nearest`
    /// filters by nearest texel, as GL JS draws.
    pub resampling: Property<Resampling>,
    pub raster_resampling: Property<Resampling>,
    /// Milliseconds. Parsed; tiles are drawn as if it were 0.
    pub fade_duration: Property<f32>,
}

/// A `hillshade` layer over a `raster-dem` source.
#[derive(Clone, Debug)]
pub struct Hillshade {
    pub source: String,
    pub paint: HillshadePaint,
}

/// `numberArray` and `colorArray` properties hold one value per light
/// source: a literal array, or a single value or expression.
#[derive(Clone, Debug)]
pub struct HillshadePaint {
    /// Degrees clockwise from north, per light.
    pub illumination_direction: Vec<Property<f32>>,
    /// Degrees above the horizon, per light.
    pub illumination_altitude: Vec<Property<f32>>,
    pub illumination_anchor: Property<IlluminationAnchor>,
    pub exaggeration: Property<f32>,
    pub shadow_color: Vec<Property<Color>>,
    pub highlight_color: Vec<Property<Color>>,
    pub accent_color: Property<Color>,
    pub method: Property<HillshadeMethod>,
    pub resampling: Property<Resampling>,
}

/// A `color-relief` layer over a `raster-dem` source.
#[derive(Clone, Debug)]
pub struct ColorRelief {
    pub source: String,
    pub paint: ColorReliefPaint,
}

#[derive(Clone, Debug)]
pub struct ColorReliefPaint {
    pub opacity: Property<f32>,
    /// A colour ramp over `["elevation"]`; unset draws nothing.
    pub color: Option<PropertyExpression>,
    pub resampling: Property<Resampling>,
}

/// Labels and icons.
#[derive(Clone, Debug)]
pub struct Symbol {
    pub selection: Selection,
    pub layout: SymbolLayout,
    pub paint: SymbolPaint,
}

/// A symbol layer's layout. Numbers are `f64`: layout matches GL JS's
/// doubles.
#[derive(Clone, Debug)]
pub struct SymbolLayout {
    pub placement: Property<SymbolPlacement>,
    pub spacing: Property<f64>,
    pub avoid_edges: Property<bool>,
    pub sort_key: Property<Option<f64>>,
    pub z_order: Property<SymbolZOrder>,
    pub icon_allow_overlap: Property<bool>,
    /// Unset defers to `icon-allow-overlap`.
    pub icon_overlap: Property<Option<Overlap>>,
    pub icon_ignore_placement: Property<bool>,
    pub icon_optional: Property<bool>,
    pub icon_rotation_alignment: Property<Alignment>,
    pub icon_size: Property<f64>,
    pub icon_text_fit: Property<IconTextFit>,
    pub icon_text_fit_padding: Property<[f64; 4]>,
    pub icon_image: Property<Option<String>>,
    pub icon_rotate: Property<f64>,
    pub icon_padding: Property<Padding>,
    pub icon_keep_upright: Property<bool>,
    pub icon_offset: Property<[f64; 2]>,
    /// Whether the style sets `icon-offset`: GL JS then sizes symbols
    /// without the perspective ratio.
    pub icon_offset_set: bool,
    pub icon_anchor: Property<Anchor>,
    pub icon_pitch_alignment: Property<Alignment>,
    pub text_pitch_alignment: Property<Alignment>,
    pub text_rotation_alignment: Property<TextRotationAlignment>,
    pub text_field: Property<Vec<orb_expr::FormattedSection>>,
    pub text_font: Property<Vec<String>>,
    pub text_size: Property<f64>,
    pub text_max_width: Property<f64>,
    pub text_line_height: Property<f64>,
    pub text_letter_spacing: Property<f64>,
    pub text_justify: Property<TextJustify>,
    pub text_radial_offset: Property<f64>,
    /// Whether the style sets `text-radial-offset`: variable anchors then
    /// use it rather than `text-offset`.
    pub text_radial_offset_set: bool,
    pub text_variable_anchor: Property<Option<Vec<Anchor>>>,
    pub text_variable_anchor_offset: Property<Option<AnchorOffsets>>,
    pub text_anchor: Property<Anchor>,
    pub text_max_angle: Property<f64>,
    pub text_writing_mode: Property<Option<Vec<WritingMode>>>,
    pub text_rotate: Property<f64>,
    pub text_padding: Property<f64>,
    pub text_keep_upright: Property<bool>,
    pub text_transform: Property<TextTransform>,
    pub text_offset: Property<[f64; 2]>,
    pub text_allow_overlap: Property<bool>,
    /// Unset defers to `text-allow-overlap`.
    pub text_overlap: Property<Option<Overlap>>,
    pub text_ignore_placement: Property<bool>,
    pub text_optional: Property<bool>,
    pub height_offset: Property<f64>,
    pub height_anchor: Property<HeightAnchor>,
}

/// A symbol layer's paint: its icons' and its text's.
#[derive(Clone, Debug)]
pub struct SymbolPaint {
    pub icon: SymbolPartPaint,
    pub text: SymbolPartPaint,
}

/// The paint of a symbol layer's icons or text.
#[derive(Clone, Debug)]
pub struct SymbolPartPaint {
    pub opacity: Property<f32>,
    pub color: Property<Color>,
    pub halo_color: Property<Color>,
    pub halo_width: Property<f32>,
    pub halo_blur: Property<f32>,
    pub translate: Property<[f32; 2]>,
    pub translate_anchor: Property<TranslateAnchor>,
}

const LAYER_TYPES: &[&str] = &[
    "fill",
    "line",
    "symbol",
    "circle",
    "heatmap",
    "fill-extrusion",
    "raster",
    "hillshade",
    "color-relief",
    "background",
];

const LAYER_KEYS: &[&str] = &[
    "id",
    "type",
    "metadata",
    "source",
    "source-layer",
    "minzoom",
    "maxzoom",
    "filter",
    "layout",
    "paint",
];

/// A layer's `paint` or `layout` object.
struct Section<'a> {
    path: String,
    members: &'a [Member],
}

impl Section<'_> {
    fn get(&self, name: &str) -> Option<&Json> {
        self.members
            .iter()
            .rev()
            .find(|m| m.key == name)
            .map(|m| &m.value)
    }
}

fn section<'a>(
    cx: &mut Cx,
    layer: &'a Json,
    key: &str,
    layer_path: &str,
    props: Option<&[&Prop]>,
) -> Section<'a> {
    let path = format!("{layer_path}.{key}");
    let members = match layer.get(key) {
        None => &[][..],
        Some(json) => cx.object(json, &path).unwrap_or_default(),
    };
    if let Some(props) = props {
        let known = |name: &str| props.iter().any(|p| p.name == name);
        for m in members {
            let member_path = format!("{path}.{}", m.key);
            if known(&m.key) {
                continue;
            }
            match m.key.strip_suffix("-transition") {
                Some(base) if key == "paint" && known(base) => {}
                _ => cx.warn(
                    member_path,
                    m.key_span,
                    format!("unknown property \"{}\"", m.key),
                ),
            }
        }
    }
    Section { path, members }
}

/// The typed spec default of `prop`.
fn default_of<T: PropertyValue>(prop: &Prop) -> T {
    match prop.default_json() {
        None => T::UNSET.unwrap_or_else(|| panic!("{} has no default", prop.name)),
        Some(json) => {
            let parsed = PropertyExpression::parse(&json, &prop.spec())
                .unwrap_or_else(|e| panic!("{} default: {e:?}", prop.name));
            let value = parsed
                .expression()
                .constant()
                .expect("defaults are literals");
            T::from_value(value)
                .unwrap_or_else(|| panic!("{} default has the wrong type", prop.name))
        }
    }
}

/// Validates a constant against the property's enum values and range.
fn check_constant(cx: &mut Cx, prop: &Prop, json: &Json, value: &Value<'_>, path: &str) {
    if let orb_expr::PropertyType::Enum(values) = prop.ty
        && let Some(s) = value.as_str()
        && !values.contains(&s)
    {
        cx.error(
            path,
            json.span,
            format!("expected one of [{}], \"{s}\" found", values.join(", ")),
        );
    }
    // Ranges apply to the literal numbers of a plain value, as GL JS
    // validates them.
    let numbers: Vec<&Json> = match &json.value {
        JsonValue::Number(_) => vec![json],
        JsonValue::Array(items) if items.iter().all(|i| i.as_f64().is_some()) => {
            items.iter().collect()
        }
        _ => Vec::new(),
    };
    for n in numbers {
        let v = n.as_f64().expect("number");
        if let Some(min) = prop.minimum
            && v < min
        {
            cx.error(
                path,
                n.span,
                format!("{v} is less than the minimum value {min}"),
            );
        }
        if let Some(max) = prop.maximum
            && v > max
        {
            cx.error(
                path,
                n.span,
                format!("{v} is greater than the maximum value {max}"),
            );
        }
    }
}

fn property<T: PropertyValue>(cx: &mut Cx, section: &Section<'_>, prop: &Prop) -> Property<T> {
    let path = format!("{}.{}", section.path, prop.name);
    property_at(cx, section.get(prop.name), &path, prop)
}

/// A property's value from `json`, located at `path`.
pub(crate) fn property_at<T: PropertyValue>(
    cx: &mut Cx,
    json: Option<&Json>,
    path: &str,
    prop: &Prop,
) -> Property<T> {
    let default = default_of::<T>(prop);
    let Some(json) = json else {
        return Property::constant(default);
    };
    let path = path.to_string();
    match PropertyExpression::parse_with_state(json, &prop.spec(), &cx.state) {
        Err(errors) => {
            cx.errors_at(&path, errors);
            Property::constant(default)
        }
        // Constants are folded to literals, except images, which GL JS
        // leaves to resolve at evaluation with the style's images.
        Ok(parsed) if parsed.kind() == Kind::Constant && !parsed.expression().reads_images() => {
            let value = parsed
                .expression()
                .evaluate(0.0)
                .map(Value::into_owned)
                .unwrap_or(Value::Null);
            check_constant(cx, prop, json, &value, &path);
            Property::constant(T::from_value(&value).unwrap_or(default))
        }
        Ok(parsed) => Property::expression(parsed, default),
    }
}

/// `line-gradient`, a colour ramp evaluated over `line-progress`.
fn color_ramp(cx: &mut Cx, section: &Section<'_>, prop: &Prop) -> Option<PropertyExpression> {
    let json = section.get(prop.name)?;
    let path = format!("{}.{}", section.path, prop.name);
    PropertyExpression::parse_with_state(json, &prop.spec(), &cx.state)
        .map_err(|errors| cx.errors_at(&path, errors))
        .ok()
}

/// A `numberArray` or `colorArray` property: one value per element of a
/// literal array, else one value.
fn property_array<T: PropertyValue>(
    cx: &mut Cx,
    section: &Section<'_>,
    prop: &Prop,
) -> Vec<Property<T>> {
    let literal = section
        .get(prop.name)
        .and_then(Json::as_array)
        .filter(|items| {
            !items.is_empty()
                && items.iter().all(|item| match &item.value {
                    JsonValue::Number(_) => true,
                    JsonValue::String(s) => orb_expr::Color::parse(s).is_some(),
                    _ => false,
                })
        });
    let path = format!("{}.{}", section.path, prop.name);
    let Some(items) = literal else {
        return vec![property_at(cx, section.get(prop.name), &path, prop)];
    };
    items
        .iter()
        .enumerate()
        .map(|(i, item)| property_at(cx, Some(item), &format!("{path}[{i}]"), prop))
        .collect()
}

fn optional_string(cx: &mut Cx, layer: &Json, key: &str, path: &str) -> Option<String> {
    layer
        .get(key)
        .and_then(|v| cx.string(v, &format!("{path}.{key}")))
}

fn selection(
    cx: &mut Cx,
    json: &Json,
    path: &str,
    id: &str,
    sources: &[Source],
) -> Option<Selection> {
    let errors = cx.diagnostics.errors.len();
    let source = match json.get("source") {
        None => {
            cx.error(path, json.span, "missing required property \"source\"");
            None
        }
        Some(s) => cx.string(s, &format!("{path}.source")),
    };
    let source_layer = optional_string(cx, json, "source-layer", path);
    if let Some(source) = &source {
        let span = json.get("source").expect("present").span;
        match sources.iter().find(|s| s.id == *source) {
            None => cx.error(
                format!("{path}.source"),
                span,
                format!("source \"{source}\" not found"),
            ),
            Some(Source {
                kind: SourceKind::Vector(_),
                ..
            }) if source_layer.is_none() => cx.error(
                path,
                json.span,
                format!("layer \"{id}\" must specify a \"source-layer\""),
            ),
            Some(Source {
                kind: SourceKind::Raster(_) | SourceKind::RasterDem(_) | SourceKind::Image(_),
                ..
            }) => cx.error(
                format!("{path}.source"),
                span,
                format!("layer \"{id}\" requires a vector source"),
            ),
            Some(_) => {}
        }
    }
    let filter = match json.get("filter") {
        None => None,
        Some(f) if f.is_null() => None,
        Some(f) => {
            let filter_path = format!("{path}.filter");
            let mut warnings = Vec::new();
            let filter = Filter::parse_with_state(f, &mut warnings, &cx.state)
                .map_err(|errors| cx.errors_at(&filter_path, errors))
                .ok();
            cx.warnings_at(&filter_path, warnings);
            filter
        }
    };
    (cx.diagnostics.errors.len() == errors).then(|| Selection {
        source: source.expect("no errors"),
        source_layer,
        filter,
    })
}

/// The source of a raster, hillshade or color-relief layer, which must be
/// a `raster` (`dem: false`) or `raster-dem` source.
fn image_source(
    cx: &mut Cx,
    json: &Json,
    path: &str,
    id: &str,
    sources: &[Source],
    dem: bool,
) -> Option<String> {
    let Some(value) = json.get("source") else {
        cx.error(path, json.span, "missing required property \"source\"");
        return None;
    };
    let source = cx.string(value, &format!("{path}.source"))?;
    let fits = match sources.iter().find(|s| s.id == source).map(|s| &s.kind) {
        None => {
            cx.error(
                format!("{path}.source"),
                value.span,
                format!("source \"{source}\" not found"),
            );
            return None;
        }
        Some(SourceKind::Raster(_) | SourceKind::Image(_)) => !dem,
        Some(SourceKind::RasterDem(_)) => dem,
        // Video sources warn on their own.
        Some(SourceKind::Unsupported { .. }) => true,
        Some(_) => false,
    };
    if !fits {
        let want = if dem { "raster-dem" } else { "raster" };
        cx.error(
            format!("{path}.source"),
            value.span,
            format!("layer \"{id}\" requires a {want} source"),
        );
        return None;
    }
    Some(source)
}

pub(crate) fn parse_layer(
    cx: &mut Cx,
    json: &Json,
    index: usize,
    sources: &[Source],
    ids: &mut HashSet<String>,
) -> Option<Layer> {
    let path = format!("layers[{index}]");
    let members = cx.object(json, &path)?;
    let errors = cx.diagnostics.errors.len();

    let id = match json.get("id") {
        None => {
            cx.error(&path, json.span, "missing required property \"id\"");
            None
        }
        Some(id) => cx.string(id, &format!("{path}.id")),
    };
    if let Some(id) = &id
        && !ids.insert(id.clone())
    {
        cx.error(
            format!("{path}.id"),
            json.get("id").expect("present").span,
            format!("duplicate layer id \"{id}\""),
        );
    }
    let ty = match json.get("type") {
        None => {
            cx.error(&path, json.span, "missing required property \"type\"");
            None
        }
        Some(t) => match t.as_str() {
            Some(ty) if LAYER_TYPES.contains(&ty) => Some(ty),
            _ => {
                cx.error(
                    format!("{path}.type"),
                    t.span,
                    format!("expected one of [{}], {t} found", LAYER_TYPES.join(", ")),
                );
                None
            }
        },
    };
    for m in members {
        if !LAYER_KEYS.contains(&m.key.as_str()) {
            cx.warn(
                format!("{path}.{}", m.key),
                m.key_span,
                format!("unknown property \"{}\"", m.key),
            );
        }
    }
    let minzoom = json
        .get("minzoom")
        .and_then(|z| cx.number_in(z, &format!("{path}.minzoom"), 0.0, 24.0))
        .unwrap_or(0.0);
    let maxzoom = json
        .get("maxzoom")
        .and_then(|z| cx.number_in(z, &format!("{path}.maxzoom"), 0.0, 24.0))
        .filter(|&z| z != 0.0)
        .unwrap_or(f64::INFINITY);
    let (id, ty) = (id?, ty?);

    let typed = matches!(
        ty,
        "background"
            | "fill"
            | "fill-extrusion"
            | "line"
            | "circle"
            | "heatmap"
            | "raster"
            | "hillshade"
            | "color-relief"
            | "symbol"
    );
    let (layout_props, paint_props): (&[&Prop], &[&Prop]) = match ty {
        "background" => (spec::BACKGROUND_LAYOUT, spec::BACKGROUND_PAINT),
        "fill" => (spec::FILL_LAYOUT, spec::FILL_PAINT),
        "fill-extrusion" => (spec::EXTRUSION_LAYOUT, spec::EXTRUSION_PAINT),
        "line" => (spec::LINE_LAYOUT, spec::LINE_PAINT),
        "circle" => (spec::CIRCLE_LAYOUT, spec::CIRCLE_PAINT),
        "heatmap" => (spec::VISIBILITY_LAYOUT, spec::HEATMAP_PAINT),
        "raster" => (spec::VISIBILITY_LAYOUT, spec::RASTER_PAINT),
        "hillshade" => (spec::VISIBILITY_LAYOUT, spec::HILLSHADE_PAINT),
        "color-relief" => (spec::VISIBILITY_LAYOUT, spec::COLOR_RELIEF_PAINT),
        "symbol" => (spec::SYMBOL_LAYOUT, spec::SYMBOL_PAINT),
        _ => (&[], &[]),
    };
    let layout = section(cx, json, "layout", &path, typed.then_some(layout_props));
    let paint = section(cx, json, "paint", &path, typed.then_some(paint_props));
    let visibility: Visibility = property(cx, &layout, &spec::VISIBILITY)
        .as_constant()
        .copied()
        .unwrap_or(Visibility::Visible);

    let kind = match ty {
        "background" => LayerKind::Background(Background {
            paint: BackgroundPaint {
                color: property(cx, &paint, &spec::BACKGROUND_COLOR),
                pattern: property(cx, &paint, &spec::BACKGROUND_PATTERN),
                opacity: property(cx, &paint, &spec::BACKGROUND_OPACITY),
            },
        }),
        "fill" => {
            let selection = selection(cx, json, &path, &id, sources);
            LayerKind::Fill(Box::new(Fill {
                layout: FillLayout {
                    sort_key: property(cx, &layout, &spec::FILL_SORT_KEY),
                },
                paint: FillPaint {
                    antialias: property(cx, &paint, &spec::FILL_ANTIALIAS),
                    opacity: property(cx, &paint, &spec::FILL_OPACITY),
                    layer_opacity: property(cx, &paint, &spec::FILL_LAYER_OPACITY),
                    layer_blend: property(cx, &paint, &spec::FILL_LAYER_BLEND),
                    color: property(cx, &paint, &spec::FILL_COLOR),
                    outline_color: property(cx, &paint, &spec::FILL_OUTLINE_COLOR),
                    translate: property(cx, &paint, &spec::FILL_TRANSLATE),
                    translate_anchor: property(cx, &paint, &spec::FILL_TRANSLATE_ANCHOR),
                    pattern: property(cx, &paint, &spec::FILL_PATTERN),
                },
                selection: selection?,
            }))
        }
        "fill-extrusion" => LayerKind::FillExtrusion(Box::new(FillExtrusion {
            selection: selection(cx, json, &path, &id, sources)?,
            rounded_corner_distance: property(cx, &layout, &spec::EXTRUSION_ROUNDED_CORNER),
            paint: FillExtrusionPaint {
                color: property(cx, &paint, &spec::EXTRUSION_COLOR),
                opacity: property(cx, &paint, &spec::EXTRUSION_OPACITY),
                height: property(cx, &paint, &spec::EXTRUSION_HEIGHT),
                base: property(cx, &paint, &spec::EXTRUSION_BASE),
                translate: property(cx, &paint, &spec::EXTRUSION_TRANSLATE),
                translate_anchor: property(cx, &paint, &spec::EXTRUSION_TRANSLATE_ANCHOR),
                vertical_gradient: property(cx, &paint, &spec::EXTRUSION_VERTICAL_GRADIENT),
                pattern: property(cx, &paint, &spec::EXTRUSION_PATTERN),
            },
        })),
        "line" => {
            let selection = selection(cx, json, &path, &id, sources);
            LayerKind::Line(Box::new(Line {
                layout: LineLayout {
                    cap: property(cx, &layout, &spec::LINE_CAP),
                    join: property(cx, &layout, &spec::LINE_JOIN),
                    miter_limit: property(cx, &layout, &spec::LINE_MITER_LIMIT),
                    round_limit: property(cx, &layout, &spec::LINE_ROUND_LIMIT),
                    sort_key: property(cx, &layout, &spec::LINE_SORT_KEY),
                },
                paint: LinePaint {
                    opacity: property(cx, &paint, &spec::LINE_OPACITY),
                    layer_opacity: property(cx, &paint, &spec::LINE_LAYER_OPACITY),
                    layer_blend: property(cx, &paint, &spec::LINE_LAYER_BLEND),
                    color: property(cx, &paint, &spec::LINE_COLOR),
                    translate: property(cx, &paint, &spec::LINE_TRANSLATE),
                    translate_anchor: property(cx, &paint, &spec::LINE_TRANSLATE_ANCHOR),
                    width: property(cx, &paint, &spec::LINE_WIDTH),
                    gap_width: property(cx, &paint, &spec::LINE_GAP_WIDTH),
                    offset: property(cx, &paint, &spec::LINE_OFFSET),
                    blur: property(cx, &paint, &spec::LINE_BLUR),
                    dasharray: property(cx, &paint, &spec::LINE_DASHARRAY),
                    pattern: property(cx, &paint, &spec::LINE_PATTERN),
                    gradient: color_ramp(cx, &paint, &spec::LINE_GRADIENT),
                },
                selection: selection?,
            }))
        }
        "circle" => LayerKind::Circle(Box::new(Circle {
            selection: selection(cx, json, &path, &id, sources)?,
            layout: CircleLayout {
                sort_key: property(cx, &layout, &spec::CIRCLE_SORT_KEY),
            },
            paint: CirclePaint {
                radius: property(cx, &paint, &spec::CIRCLE_RADIUS),
                color: property(cx, &paint, &spec::CIRCLE_COLOR),
                blur: property(cx, &paint, &spec::CIRCLE_BLUR),
                opacity: property(cx, &paint, &spec::CIRCLE_OPACITY),
                translate: property(cx, &paint, &spec::CIRCLE_TRANSLATE),
                translate_anchor: property(cx, &paint, &spec::CIRCLE_TRANSLATE_ANCHOR),
                pitch_scale: property(cx, &paint, &spec::CIRCLE_PITCH_SCALE),
                pitch_alignment: property(cx, &paint, &spec::CIRCLE_PITCH_ALIGNMENT),
                stroke_width: property(cx, &paint, &spec::CIRCLE_STROKE_WIDTH),
                stroke_color: property(cx, &paint, &spec::CIRCLE_STROKE_COLOR),
                stroke_opacity: property(cx, &paint, &spec::CIRCLE_STROKE_OPACITY),
            },
        })),
        "heatmap" => LayerKind::Heatmap(Box::new(Heatmap {
            selection: selection(cx, json, &path, &id, sources)?,
            paint: HeatmapPaint {
                radius: property(cx, &paint, &spec::HEATMAP_RADIUS),
                weight: property(cx, &paint, &spec::HEATMAP_WEIGHT),
                intensity: property(cx, &paint, &spec::HEATMAP_INTENSITY),
                opacity: property(cx, &paint, &spec::HEATMAP_OPACITY),
                color: color_ramp(cx, &paint, &spec::HEATMAP_COLOR).unwrap_or_else(|| {
                    PropertyExpression::parse(
                        &spec::HEATMAP_COLOR.default_json().expect("default ramp"),
                        &spec::HEATMAP_COLOR.spec(),
                    )
                    .expect("valid default ramp")
                }),
            },
        })),
        "raster" => {
            let source = image_source(cx, json, &path, &id, sources, false);
            LayerKind::Raster(Box::new(Raster {
                paint: RasterPaint {
                    opacity: property(cx, &paint, &spec::RASTER_OPACITY),
                    hue_rotate: property(cx, &paint, &spec::RASTER_HUE_ROTATE),
                    brightness_min: property(cx, &paint, &spec::RASTER_BRIGHTNESS_MIN),
                    brightness_max: property(cx, &paint, &spec::RASTER_BRIGHTNESS_MAX),
                    saturation: property(cx, &paint, &spec::RASTER_SATURATION),
                    contrast: property(cx, &paint, &spec::RASTER_CONTRAST),
                    resampling: property(cx, &paint, &spec::RESAMPLING),
                    raster_resampling: property(cx, &paint, &spec::RASTER_RESAMPLING),
                    fade_duration: property(cx, &paint, &spec::RASTER_FADE_DURATION),
                },
                source: source?,
            }))
        }
        "hillshade" => {
            let source = image_source(cx, json, &path, &id, sources, true);
            let p = &paint;
            LayerKind::Hillshade(Box::new(Hillshade {
                paint: HillshadePaint {
                    illumination_direction: property_array(
                        cx,
                        p,
                        &spec::HILLSHADE_ILLUMINATION_DIRECTION,
                    ),
                    illumination_altitude: property_array(
                        cx,
                        p,
                        &spec::HILLSHADE_ILLUMINATION_ALTITUDE,
                    ),
                    illumination_anchor: property(cx, p, &spec::HILLSHADE_ILLUMINATION_ANCHOR),
                    exaggeration: property(cx, p, &spec::HILLSHADE_EXAGGERATION),
                    shadow_color: property_array(cx, p, &spec::HILLSHADE_SHADOW_COLOR),
                    highlight_color: property_array(cx, p, &spec::HILLSHADE_HIGHLIGHT_COLOR),
                    accent_color: property(cx, p, &spec::HILLSHADE_ACCENT_COLOR),
                    method: property(cx, p, &spec::HILLSHADE_METHOD),
                    resampling: property(cx, p, &spec::RESAMPLING),
                },
                source: source?,
            }))
        }
        "color-relief" => {
            let source = image_source(cx, json, &path, &id, sources, true);
            LayerKind::ColorRelief(Box::new(ColorRelief {
                paint: ColorReliefPaint {
                    opacity: property(cx, &paint, &spec::COLOR_RELIEF_OPACITY),
                    color: color_ramp(cx, &paint, &spec::COLOR_RELIEF_COLOR),
                    resampling: property(cx, &paint, &spec::RESAMPLING),
                },
                source: source?,
            }))
        }
        "symbol" => {
            let selection = selection(cx, json, &path, &id, sources);
            let l = &layout;
            let part = |cx: &mut Cx, p: [&Prop; 7]| SymbolPartPaint {
                opacity: property(cx, &paint, p[0]),
                color: property(cx, &paint, p[1]),
                halo_color: property(cx, &paint, p[2]),
                halo_width: property(cx, &paint, p[3]),
                halo_blur: property(cx, &paint, p[4]),
                translate: property(cx, &paint, p[5]),
                translate_anchor: property(cx, &paint, p[6]),
            };
            let paint = SymbolPaint {
                icon: part(
                    cx,
                    [
                        &spec::ICON_OPACITY,
                        &spec::ICON_COLOR,
                        &spec::ICON_HALO_COLOR,
                        &spec::ICON_HALO_WIDTH,
                        &spec::ICON_HALO_BLUR,
                        &spec::ICON_TRANSLATE,
                        &spec::ICON_TRANSLATE_ANCHOR,
                    ],
                ),
                text: part(
                    cx,
                    [
                        &spec::TEXT_OPACITY,
                        &spec::TEXT_COLOR,
                        &spec::TEXT_HALO_COLOR,
                        &spec::TEXT_HALO_WIDTH,
                        &spec::TEXT_HALO_BLUR,
                        &spec::TEXT_TRANSLATE,
                        &spec::TEXT_TRANSLATE_ANCHOR,
                    ],
                ),
            };
            LayerKind::Symbol(Box::new(Symbol {
                layout: SymbolLayout {
                    placement: property(cx, l, &spec::SYMBOL_PLACEMENT),
                    spacing: property(cx, l, &spec::SYMBOL_SPACING),
                    avoid_edges: property(cx, l, &spec::SYMBOL_AVOID_EDGES),
                    sort_key: property(cx, l, &spec::SYMBOL_SORT_KEY),
                    z_order: property(cx, l, &spec::SYMBOL_Z_ORDER),
                    icon_allow_overlap: property(cx, l, &spec::ICON_ALLOW_OVERLAP),
                    icon_overlap: property(cx, l, &spec::ICON_OVERLAP),
                    icon_ignore_placement: property(cx, l, &spec::ICON_IGNORE_PLACEMENT),
                    icon_optional: property(cx, l, &spec::ICON_OPTIONAL),
                    icon_rotation_alignment: property(cx, l, &spec::ICON_ROTATION_ALIGNMENT),
                    icon_size: property(cx, l, &spec::ICON_SIZE),
                    icon_text_fit: property(cx, l, &spec::ICON_TEXT_FIT),
                    icon_text_fit_padding: property(cx, l, &spec::ICON_TEXT_FIT_PADDING),
                    icon_image: property(cx, l, &spec::ICON_IMAGE),
                    icon_rotate: property(cx, l, &spec::ICON_ROTATE),
                    icon_padding: property(cx, l, &spec::ICON_PADDING),
                    icon_keep_upright: property(cx, l, &spec::ICON_KEEP_UPRIGHT),
                    icon_offset: property(cx, l, &spec::ICON_OFFSET),
                    icon_offset_set: l.get(spec::ICON_OFFSET.name).is_some(),
                    icon_anchor: property(cx, l, &spec::ICON_ANCHOR),
                    icon_pitch_alignment: property(cx, l, &spec::ICON_PITCH_ALIGNMENT),
                    text_pitch_alignment: property(cx, l, &spec::TEXT_PITCH_ALIGNMENT),
                    text_rotation_alignment: property(cx, l, &spec::TEXT_ROTATION_ALIGNMENT),
                    text_field: property(cx, l, &spec::TEXT_FIELD),
                    text_font: property(cx, l, &spec::TEXT_FONT),
                    text_size: property(cx, l, &spec::TEXT_SIZE),
                    text_max_width: property(cx, l, &spec::TEXT_MAX_WIDTH),
                    text_line_height: property(cx, l, &spec::TEXT_LINE_HEIGHT),
                    text_letter_spacing: property(cx, l, &spec::TEXT_LETTER_SPACING),
                    text_justify: property(cx, l, &spec::TEXT_JUSTIFY),
                    text_radial_offset: property(cx, l, &spec::TEXT_RADIAL_OFFSET),
                    text_radial_offset_set: l.get(spec::TEXT_RADIAL_OFFSET.name).is_some(),
                    text_variable_anchor: property(cx, l, &spec::TEXT_VARIABLE_ANCHOR),
                    text_variable_anchor_offset: property(
                        cx,
                        l,
                        &spec::TEXT_VARIABLE_ANCHOR_OFFSET,
                    ),
                    text_anchor: property(cx, l, &spec::TEXT_ANCHOR),
                    text_max_angle: property(cx, l, &spec::TEXT_MAX_ANGLE),
                    text_writing_mode: property(cx, l, &spec::TEXT_WRITING_MODE),
                    text_rotate: property(cx, l, &spec::TEXT_ROTATE),
                    text_padding: property(cx, l, &spec::TEXT_PADDING),
                    text_keep_upright: property(cx, l, &spec::TEXT_KEEP_UPRIGHT),
                    text_transform: property(cx, l, &spec::TEXT_TRANSFORM),
                    text_offset: property(cx, l, &spec::TEXT_OFFSET),
                    text_allow_overlap: property(cx, l, &spec::TEXT_ALLOW_OVERLAP),
                    text_overlap: property(cx, l, &spec::TEXT_OVERLAP),
                    text_ignore_placement: property(cx, l, &spec::TEXT_IGNORE_PLACEMENT),
                    text_optional: property(cx, l, &spec::TEXT_OPTIONAL),
                    height_offset: property(cx, l, &spec::SYMBOL_HEIGHT_OFFSET),
                    height_anchor: property(cx, l, &spec::SYMBOL_HEIGHT_ANCHOR),
                },
                paint,
                selection: selection?,
            }))
        }
        other => {
            if let Some(source) = json.get("source")
                && let Some(name) = source.as_str()
                && !sources.iter().any(|s| s.id == name)
            {
                cx.error(
                    format!("{path}.source"),
                    source.span,
                    format!("source \"{name}\" not found"),
                );
            }
            cx.warn(
                format!("{path}.type"),
                json.get("type").expect("present").span,
                format!("layer type \"{other}\" is not supported yet; the layer is skipped"),
            );
            LayerKind::Unsupported {
                ty: other.to_string(),
            }
        }
    };
    let transitions = paint
        .members
        .iter()
        .filter_map(|m| {
            let name = m.key.strip_suffix("-transition")?;
            match crate::paint::parse_transition(&m.value) {
                Some(t) => Some((name.to_string(), t)),
                None => {
                    cx.warn(
                        format!("{}.{}", paint.path, m.key),
                        m.key_span,
                        "expected {\"duration\", \"delay\"} in milliseconds",
                    );
                    None
                }
            }
        })
        .collect();
    (cx.diagnostics.errors.len() == errors).then_some(Layer {
        id,
        minzoom,
        maxzoom,
        visibility,
        kind,
        transitions,
    })
}
