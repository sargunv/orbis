//! The v8 spec entries of the properties Orbis types (`v8.json`): one
//! table of every layer property, which [`PropertyId`]s index. Each entry
//! has its layer type, section, value type, default and whether it may be
//! data-driven; parsing, validation, the diff and the renderer all read it.

use orb_expr::json::{self, Json};
use orb_expr::{PropertyExpression, PropertySpec, PropertyType, Type};

use std::marker::PhantomData;

use orb_expr::Color;

use crate::property::{
    Alignment, Anchor, AnchorOffsets, Enum, HeightAnchor, HillshadeMethod, IconTextFit,
    IlluminationAnchor, LayerBlend, LineCap, LineJoin, Overlap, Padding, Resampling,
    SymbolPlacement, SymbolZOrder, TextJustify, TextRotationAlignment, TextTransform,
    TranslateAnchor, Visibility, WritingMode,
};

/// One property's spec entry.
#[derive(Debug)]
pub struct Prop {
    pub name: &'static str,
    pub ty: PropertyType,
    /// The spec default as JSON text.
    pub default: Option<&'static str>,
    /// `zoom` is an expression parameter.
    pub zoom: bool,
    /// Data-driven.
    pub feature: bool,
    pub interpolated: bool,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    /// Strings expand `{token}`s.
    pub tokens: bool,
    /// The layer type of a layer property.
    pub layer: LayerType,
    /// A `paint` property, else `layout`.
    pub paint: bool,
    /// One value per light source (`hillshade-*`), given as an array.
    pub list: bool,
    /// A colour ramp over `heatmap-density`, `line-progress` or
    /// `elevation`, kept as an expression.
    pub ramp: bool,
}

impl Prop {
    /// The fields a root-level property (light, sky, terrain) leaves at
    /// their defaults.
    pub const ROOT: Self = prop("", PropertyType::Number, None);

    pub fn spec(&self) -> PropertySpec {
        PropertySpec {
            ty: self.ty.clone(),
            default: self.default_json(),
            zoom: self.zoom,
            feature: self.feature,
            interpolated: self.interpolated,
            tokens: self.tokens,
        }
    }

    pub fn default_json(&self) -> Option<Json> {
        self.default
            .map(|text| json::parse(text).expect("spec defaults are JSON"))
    }
}

pub const fn prop(name: &'static str, ty: PropertyType, default: Option<&'static str>) -> Prop {
    Prop {
        name,
        ty,
        default,
        zoom: true,
        feature: false,
        interpolated: false,
        minimum: None,
        maximum: None,
        tokens: false,
        layer: LayerType::Background,
        paint: false,
        list: false,
        ramp: false,
    }
}

impl Prop {
    const fn zoom_independent(mut self) -> Self {
        self.zoom = false;
        self
    }
    const fn data_driven(mut self) -> Self {
        self.feature = true;
        self
    }
    const fn interpolated(mut self) -> Self {
        self.interpolated = true;
        self
    }
    const fn tokens(mut self) -> Self {
        self.tokens = true;
        self
    }
    const fn at(mut self, layer: LayerType, paint: bool) -> Self {
        self.layer = layer;
        self.paint = paint;
        self
    }
    const fn list(mut self) -> Self {
        self.list = true;
        self
    }
    const fn ramp(mut self) -> Self {
        self.ramp = true;
        self
    }
    const fn range(mut self, minimum: f64, maximum: Option<f64>) -> Self {
        self.minimum = Some(minimum);
        self.maximum = maximum;
        self
    }
}

const NUMBER: PropertyType = PropertyType::Number;
const COLOR: PropertyType = PropertyType::Color;
const IMAGE: PropertyType = PropertyType::ResolvedImage;
const TRANSLATE: PropertyType = PropertyType::Array(Type::Number, Some(2));

const fn enumeration<E: Enum>() -> PropertyType {
    PropertyType::Enum(E::VALUES)
}

/// `visibility`: a constant, which only `global-state` could vary.
pub const VISIBILITY: Prop = prop(
    "visibility",
    enumeration::<Visibility>(),
    Some("\"visible\""),
)
.zoom_independent();

const OFFSET: PropertyType = PropertyType::Array(Type::Number, Some(2));
const ANCHOR: PropertyType = enumeration::<Anchor>();
const ANCHORS: PropertyType = PropertyType::Array(Type::String, None);

/// A v8 layer type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerType {
    Background,
    Fill,
    FillExtrusion,
    Line,
    Circle,
    Heatmap,
    Raster,
    Hillshade,
    ColorRelief,
    Symbol,
}

impl LayerType {
    /// The v8 types, in the spec's order.
    pub const V8: [Self; 10] = [
        Self::Fill,
        Self::Line,
        Self::Symbol,
        Self::Circle,
        Self::Heatmap,
        Self::FillExtrusion,
        Self::Raster,
        Self::Hillshade,
        Self::ColorRelief,
        Self::Background,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Background => "background",
            Self::Fill => "fill",
            Self::FillExtrusion => "fill-extrusion",
            Self::Line => "line",
            Self::Circle => "circle",
            Self::Heatmap => "heatmap",
            Self::Raster => "raster",
            Self::Hillshade => "hillshade",
            Self::ColorRelief => "color-relief",
            Self::Symbol => "symbol",
        }
    }

    /// A v8 type by its name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::V8.into_iter().find(|t| t.name() == name)
    }

    /// The type's properties: a contiguous run of the table.
    pub fn properties(self) -> impl Iterator<Item = PropertyId> {
        PROPS
            .iter()
            .enumerate()
            .filter(move |(_, p)| p.layer == self)
            .map(|(i, _)| PropertyId(i as u16))
    }

    /// The type's first property, from which a layer indexes its own.
    pub(crate) fn first(self) -> u16 {
        PROPS
            .iter()
            .position(|p| p.layer == self)
            .expect("every type has properties") as u16
    }

    /// Whether the layer draws a source's features (and has a source
    /// layer and filter), rather than its images or none.
    pub fn draws_features(self) -> bool {
        !matches!(
            self,
            Self::Background | Self::Raster | Self::Hillshade | Self::ColorRelief
        )
    }
}

/// One layer property: an index into the property table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PropertyId(u16);

impl PropertyId {
    #[inline]
    pub fn prop(self) -> &'static Prop {
        &PROPS[self.0 as usize]
    }

    #[inline]
    pub(crate) fn index(self) -> u16 {
        self.0
    }

    /// The v8 name.
    pub fn name(self) -> &'static str {
        self.prop().name
    }

    #[inline]
    pub fn layer_type(self) -> LayerType {
        self.prop().layer
    }

    /// A `paint` property, else `layout`.
    pub fn is_paint(self) -> bool {
        self.prop().paint
    }

    pub fn spec(self) -> PropertySpec {
        self.prop().spec()
    }

    /// The default, `Null` for properties without one.
    #[inline]
    pub fn default_value(self) -> &'static orb_expr::Value<'static> {
        static DEFAULTS: std::sync::OnceLock<Vec<orb_expr::Value<'static>>> =
            std::sync::OnceLock::new();
        &DEFAULTS.get_or_init(|| {
            PROPS
                .iter()
                .map(|p| match p.default_json().filter(|_| !p.ramp) {
                    None => orb_expr::Value::Null,
                    Some(json) => PropertyExpression::parse(&json, &p.spec())
                        .unwrap_or_else(|e| panic!("{} default: {e:?}", p.name))
                        .expression()
                        .constant()
                        .expect("defaults are literals")
                        .clone(),
                })
                .collect()
        })[self.0 as usize]
    }

    /// Whether a paint change eases from the prior value:
    /// interpolated numbers, colours and pairs.
    pub fn transitions(self) -> bool {
        let p = self.prop();
        p.paint && p.interpolated && !p.list && !p.ramp
    }

    /// `slot`'s value at `zoom` without a feature, as this property.
    pub fn value(self, slot: &crate::Slot, zoom: f64) -> orb_expr::Value<'static> {
        crate::PropertyRef::of(slot, self).evaluate(zoom)
    }

    /// A `layer_type` layer's `paint` or `layout` property `name`.
    pub fn find(layer_type: LayerType, paint: bool, name: &str) -> Option<Self> {
        PROPS
            .iter()
            .position(|p| p.layer == layer_type && p.paint == paint && p.name == name)
            .map(|i| Self(i as u16))
    }
}

/// A property and the type it is read as.
#[derive(Debug)]
pub struct Key<T>(PropertyId, PhantomData<fn() -> T>);

impl<T> Clone for Key<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Key<T> {}

impl<T> Key<T> {
    const fn new(index: u16) -> Self {
        Self(PropertyId(index), PhantomData)
    }

    pub const fn id(self) -> PropertyId {
        self.0
    }
}

/// The type of a colour ramp, which stays an expression.
#[derive(Debug)]
pub enum Ramp {}

/// The type of a per-light list of `T`.
#[derive(Debug)]
pub struct List<T>(PhantomData<T>);

macro_rules! section {
    (paint) => {
        true
    };
    (layout) => {
        false
    };
}

macro_rules! properties {
    ($($layer:ident { $($(#[$doc:meta])* $section:ident $name:ident: $ty:ty = $prop:expr;)* })*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[repr(u16)]
        enum Index { $($($name,)*)* }

        /// Every layer property, by [`PropertyId`].
        pub(crate) static PROPS: &[Prop] = &[
            $($($prop.at(LayerType::$layer, section!($section)),)*)*
        ];

        $($($(#[$doc])* pub const $name: Key<$ty> = Key::new(Index::$name as u16);)*)*
    };
}

properties! {
    Background {
        paint BACKGROUND_COLOR: Color = prop("background-color", COLOR, Some("\"#000000\"")).interpolated();
        paint BACKGROUND_PATTERN: Option<String> = prop("background-pattern", IMAGE, None);
        paint BACKGROUND_OPACITY: f32 = prop("background-opacity", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
    }
    Fill {
        layout FILL_SORT_KEY: Option<f32> = prop("fill-sort-key", NUMBER, None).data_driven();
        paint FILL_ANTIALIAS: bool = prop("fill-antialias", PropertyType::Boolean, Some("true"));
        paint FILL_OPACITY: f32 = prop("fill-opacity", NUMBER, Some("1")).data_driven().interpolated().range(0.0, Some(1.0));
        /// The layer composited once at this opacity (style-spec 26).
        paint FILL_LAYER_OPACITY: f32 = prop("fill-layer-opacity", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
        /// How the layer composites with what is beneath it (style-spec 26).
        paint FILL_LAYER_BLEND: LayerBlend = prop("fill-layer-blend", enumeration::<LayerBlend>(), Some("\"normal\"")).zoom_independent();
        paint FILL_COLOR: Color = prop("fill-color", COLOR, Some("\"#000000\"")).data_driven().interpolated();
        paint FILL_OUTLINE_COLOR: Option<Color> = prop("fill-outline-color", COLOR, None).data_driven().interpolated();
        paint FILL_TRANSLATE: [f32; 2] = prop("fill-translate", TRANSLATE, Some("[0, 0]")).interpolated();
        paint FILL_TRANSLATE_ANCHOR: TranslateAnchor = prop("fill-translate-anchor", enumeration::<TranslateAnchor>(), Some("\"map\""));
        paint FILL_PATTERN: Option<String> = prop("fill-pattern", IMAGE, None).data_driven();
    }
    FillExtrusion {
        layout EXTRUSION_ROUNDED_CORNER: f32 = prop("fill-extrusion-rounded-corner-distance", NUMBER, Some("0")).range(0.0, None).zoom_independent();
        paint EXTRUSION_COLOR: Color = prop("fill-extrusion-color", COLOR, Some("\"#000000\"")).data_driven().interpolated();
        paint EXTRUSION_OPACITY: f32 = prop("fill-extrusion-opacity", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
        paint EXTRUSION_HEIGHT: f32 = prop("fill-extrusion-height", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint EXTRUSION_BASE: f32 = prop("fill-extrusion-base", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint EXTRUSION_TRANSLATE: [f32; 2] = prop("fill-extrusion-translate", TRANSLATE, Some("[0,0]")).interpolated();
        paint EXTRUSION_TRANSLATE_ANCHOR: TranslateAnchor = prop("fill-extrusion-translate-anchor", enumeration::<TranslateAnchor>(), Some("\"map\""));
        paint EXTRUSION_VERTICAL_GRADIENT: bool = prop("fill-extrusion-vertical-gradient", PropertyType::Boolean, Some("true"));
        paint EXTRUSION_PATTERN: Option<String> = prop("fill-extrusion-pattern", IMAGE, None).data_driven();
    }
    Line {
        layout LINE_CAP: LineCap = prop("line-cap", enumeration::<LineCap>(), Some("\"butt\"")).data_driven();
        layout LINE_JOIN: LineJoin = prop("line-join", enumeration::<LineJoin>(), Some("\"miter\"")).data_driven();
        layout LINE_MITER_LIMIT: f32 = prop("line-miter-limit", NUMBER, Some("2")).data_driven().interpolated();
        layout LINE_ROUND_LIMIT: f32 = prop("line-round-limit", NUMBER, Some("1.05")).data_driven().interpolated();
        layout LINE_SORT_KEY: Option<f32> = prop("line-sort-key", NUMBER, None).data_driven();
        paint LINE_OPACITY: f32 = prop("line-opacity", NUMBER, Some("1")).data_driven().interpolated().range(0.0, Some(1.0));
        /// The layer composited once at this opacity (style-spec 26).
        paint LINE_LAYER_OPACITY: f32 = prop("line-layer-opacity", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
        /// How the layer composites with what is beneath it (style-spec 26).
        paint LINE_LAYER_BLEND: LayerBlend = prop("line-layer-blend", enumeration::<LayerBlend>(), Some("\"normal\"")).zoom_independent();
        paint LINE_COLOR: Color = prop("line-color", COLOR, Some("\"#000000\"")).data_driven().interpolated();
        paint LINE_TRANSLATE: [f32; 2] = prop("line-translate", TRANSLATE, Some("[0, 0]")).interpolated();
        paint LINE_TRANSLATE_ANCHOR: TranslateAnchor = prop("line-translate-anchor", enumeration::<TranslateAnchor>(), Some("\"map\""));
        paint LINE_WIDTH: f32 = prop("line-width", NUMBER, Some("1")).data_driven().interpolated().range(0.0, None);
        paint LINE_GAP_WIDTH: f32 = prop("line-gap-width", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint LINE_OFFSET: f32 = prop("line-offset", NUMBER, Some("0")).data_driven().interpolated();
        paint LINE_BLUR: f32 = prop("line-blur", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint LINE_DASHARRAY: Option<Vec<f32>> = prop("line-dasharray", PropertyType::Array(Type::Number, None), None).data_driven().range(0.0, None);
        paint LINE_PATTERN: Option<String> = prop("line-pattern", IMAGE, None).data_driven();
        /// `line-gradient` varies with `line-progress`, not zoom or the feature.
        paint LINE_GRADIENT: Ramp = prop("line-gradient", COLOR, None).zoom_independent().interpolated().ramp();
    }
    Circle {
        layout CIRCLE_SORT_KEY: Option<f32> = prop("circle-sort-key", NUMBER, None).data_driven();
        paint CIRCLE_RADIUS: f32 = prop("circle-radius", NUMBER, Some(r#"5"#)).data_driven().interpolated().range(0.0, None);
        paint CIRCLE_COLOR: Color = prop("circle-color", COLOR, Some(r##""#000000""##)).data_driven().interpolated();
        paint CIRCLE_BLUR: f32 = prop("circle-blur", NUMBER, Some(r#"0"#)).data_driven().interpolated().range(0.0, None);
        paint CIRCLE_OPACITY: f32 = prop("circle-opacity", NUMBER, Some(r#"1"#)).data_driven().interpolated().range(0.0, Some(1.0));
        paint CIRCLE_TRANSLATE: [f32; 2] = prop("circle-translate", TRANSLATE, Some(r#"[0, 0]"#)).interpolated();
        paint CIRCLE_TRANSLATE_ANCHOR: TranslateAnchor = prop("circle-translate-anchor", enumeration::<TranslateAnchor>(), Some(r#""map""#));
        paint CIRCLE_PITCH_SCALE: TranslateAnchor = prop("circle-pitch-scale", enumeration::<TranslateAnchor>(), Some(r#""map""#));
        paint CIRCLE_PITCH_ALIGNMENT: TranslateAnchor = prop("circle-pitch-alignment", enumeration::<TranslateAnchor>(), Some(r#""viewport""#));
        paint CIRCLE_STROKE_WIDTH: f32 = prop("circle-stroke-width", NUMBER, Some(r#"0"#)).data_driven().interpolated().range(0.0, None);
        paint CIRCLE_STROKE_COLOR: Color = prop("circle-stroke-color", COLOR, Some(r##""#000000""##)).data_driven().interpolated();
        paint CIRCLE_STROKE_OPACITY: f32 = prop("circle-stroke-opacity", NUMBER, Some(r#"1"#)).data_driven().interpolated().range(0.0, Some(1.0));
    }
    Heatmap {
        paint HEATMAP_RADIUS: f32 = prop("heatmap-radius", NUMBER, Some(r#"30"#)).data_driven().interpolated().range(1.0, None);
        paint HEATMAP_WEIGHT: f32 = prop("heatmap-weight", NUMBER, Some(r#"1"#)).data_driven().interpolated().range(0.0, None);
        paint HEATMAP_INTENSITY: f32 = prop("heatmap-intensity", NUMBER, Some(r#"1"#)).interpolated().range(0.0, None);
        paint HEATMAP_COLOR: Ramp = prop("heatmap-color", COLOR, Some(r#"["interpolate",["linear"],["heatmap-density"],0,"rgba(0, 0, 255, 0)",0.1,"royalblue",0.3,"cyan",0.5,"lime",0.7,"yellow",1,"red"]"#)).interpolated().zoom_independent().ramp();
        paint HEATMAP_OPACITY: f32 = prop("heatmap-opacity", NUMBER, Some(r#"1"#)).interpolated().range(0.0, Some(1.0));
    }
    Raster {
        paint RASTER_OPACITY: f32 = prop("raster-opacity", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
        paint RASTER_HUE_ROTATE: f32 = prop("raster-hue-rotate", NUMBER, Some("0")).interpolated();
        paint RASTER_BRIGHTNESS_MIN: f32 = prop("raster-brightness-min", NUMBER, Some("0")).interpolated().range(0.0, Some(1.0));
        paint RASTER_BRIGHTNESS_MAX: f32 = prop("raster-brightness-max", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
        paint RASTER_SATURATION: f32 = prop("raster-saturation", NUMBER, Some("0")).interpolated().range(-1.0, Some(1.0));
        paint RASTER_CONTRAST: f32 = prop("raster-contrast", NUMBER, Some("0")).interpolated().range(-1.0, Some(1.0));
        /// `resampling`, which raster, hillshade and color-relief layers share.
        paint RASTER_RESAMPLE: Resampling = prop("resampling", enumeration::<Resampling>(), Some("\"linear\""));
        paint RASTER_RESAMPLING: Resampling = prop("raster-resampling", enumeration::<Resampling>(), Some("\"linear\""));
        /// Parsed and validated; tiles appear without fading.
        paint RASTER_FADE_DURATION: f32 = prop("raster-fade-duration", NUMBER, Some("300")).interpolated().range(0.0, None);
    }
    Hillshade {
        paint HILLSHADE_ILLUMINATION_DIRECTION: List<f32> = prop("hillshade-illumination-direction", NUMBER, Some("335")).interpolated().range(0.0, Some(359.0)).list();
        paint HILLSHADE_ILLUMINATION_ALTITUDE: List<f32> = prop("hillshade-illumination-altitude", NUMBER, Some("45")).interpolated().range(0.0, Some(90.0)).list();
        paint HILLSHADE_ILLUMINATION_ANCHOR: IlluminationAnchor = prop("hillshade-illumination-anchor", enumeration::<IlluminationAnchor>(), Some("\"viewport\""));
        paint HILLSHADE_EXAGGERATION: f32 = prop("hillshade-exaggeration", NUMBER, Some("0.5")).interpolated().range(0.0, Some(1.0));
        paint HILLSHADE_SHADOW_COLOR: List<Color> = prop("hillshade-shadow-color", COLOR, Some("\"#000000\"")).interpolated().list();
        paint HILLSHADE_HIGHLIGHT_COLOR: List<Color> = prop("hillshade-highlight-color", COLOR, Some("\"#FFFFFF\"")).interpolated().list();
        paint HILLSHADE_ACCENT_COLOR: Color = prop("hillshade-accent-color", COLOR, Some("\"#000000\"")).interpolated();
        paint HILLSHADE_METHOD: HillshadeMethod = prop("hillshade-method", enumeration::<HillshadeMethod>(), Some("\"standard\""));
        /// `resampling`, which raster, hillshade and color-relief layers share.
        paint HILLSHADE_RESAMPLE: Resampling = prop("resampling", enumeration::<Resampling>(), Some("\"linear\""));
    }
    ColorRelief {
        paint COLOR_RELIEF_OPACITY: f32 = prop("color-relief-opacity", NUMBER, Some("1")).interpolated().range(0.0, Some(1.0));
        /// A colour ramp over `["elevation"]`, not zoom or the feature.
        paint COLOR_RELIEF_COLOR: Ramp = prop("color-relief-color", COLOR, None).zoom_independent().interpolated().ramp();
        /// `resampling`, which raster, hillshade and color-relief layers share.
        paint COLOR_RELIEF_RESAMPLE: Resampling = prop("resampling", enumeration::<Resampling>(), Some("\"linear\""));
    }
    Symbol {
        layout SYMBOL_PLACEMENT: SymbolPlacement = prop("symbol-placement", enumeration::<SymbolPlacement>(), Some("\"point\""));
        layout SYMBOL_SPACING: f64 = prop("symbol-spacing", NUMBER, Some("250")).interpolated().range(1.0, None);
        layout SYMBOL_AVOID_EDGES: bool = prop("symbol-avoid-edges", PropertyType::Boolean, Some("false"));
        layout SYMBOL_SORT_KEY: Option<f64> = prop("symbol-sort-key", NUMBER, None).data_driven();
        layout SYMBOL_Z_ORDER: SymbolZOrder = prop("symbol-z-order", enumeration::<SymbolZOrder>(), Some("\"auto\""));
        layout ICON_ALLOW_OVERLAP: bool = prop("icon-allow-overlap", PropertyType::Boolean, Some("false"));
        layout ICON_OVERLAP: Option<Overlap> = prop("icon-overlap", enumeration::<Overlap>(), None);
        layout ICON_IGNORE_PLACEMENT: bool = prop("icon-ignore-placement", PropertyType::Boolean, Some("false"));
        layout ICON_OPTIONAL: bool = prop("icon-optional", PropertyType::Boolean, Some("false"));
        layout ICON_ROTATION_ALIGNMENT: Alignment = prop("icon-rotation-alignment", enumeration::<Alignment>(), Some("\"auto\"")).data_driven();
        layout ICON_SIZE: f64 = prop("icon-size", NUMBER, Some("1")).data_driven().interpolated().range(0.0, None);
        layout ICON_TEXT_FIT: IconTextFit = prop("icon-text-fit", enumeration::<IconTextFit>(), Some("\"none\""));
        layout ICON_TEXT_FIT_PADDING: [f64; 4] = prop("icon-text-fit-padding", PropertyType::Array(Type::Number, Some(4)), Some("[0, 0, 0, 0]")).interpolated();
        layout ICON_IMAGE: Option<String> = prop("icon-image", IMAGE, None).data_driven().tokens();
        layout ICON_ROTATE: f64 = prop("icon-rotate", NUMBER, Some("0")).data_driven().interpolated();
        layout ICON_PADDING: Padding = prop("icon-padding", PropertyType::Padding, Some("[2]")).data_driven().interpolated();
        layout ICON_KEEP_UPRIGHT: bool = prop("icon-keep-upright", PropertyType::Boolean, Some("false"));
        layout ICON_OFFSET: [f64; 2] = prop("icon-offset", OFFSET, Some("[0, 0]")).data_driven().interpolated();
        layout ICON_ANCHOR: Anchor = prop("icon-anchor", ANCHOR, Some("\"center\"")).data_driven();
        layout ICON_PITCH_ALIGNMENT: Alignment = prop("icon-pitch-alignment", enumeration::<Alignment>(), Some("\"auto\""));
        layout TEXT_PITCH_ALIGNMENT: Alignment = prop("text-pitch-alignment", enumeration::<Alignment>(), Some("\"auto\""));
        layout TEXT_ROTATION_ALIGNMENT: TextRotationAlignment = prop("text-rotation-alignment", enumeration::<TextRotationAlignment>(), Some("\"auto\""));
        layout TEXT_FIELD: Vec<orb_expr::FormattedSection> = prop("text-field", PropertyType::Formatted, Some("\"\"")).data_driven().tokens();
        layout TEXT_FONT: Vec<String> = prop("text-font", PropertyType::Array(Type::String, None), Some("[\"Open Sans Regular\", \"Arial Unicode MS Regular\"]")).data_driven();
        layout TEXT_SIZE: f64 = prop("text-size", NUMBER, Some("16")).data_driven().interpolated().range(0.0, None);
        layout TEXT_MAX_WIDTH: f64 = prop("text-max-width", NUMBER, Some("10")).data_driven().interpolated().range(0.0, None);
        layout TEXT_LINE_HEIGHT: f64 = prop("text-line-height", NUMBER, Some("1.2")).interpolated();
        layout TEXT_LETTER_SPACING: f64 = prop("text-letter-spacing", NUMBER, Some("0")).data_driven().interpolated();
        layout TEXT_JUSTIFY: TextJustify = prop("text-justify", enumeration::<TextJustify>(), Some("\"center\"")).data_driven();
        layout TEXT_RADIAL_OFFSET: f64 = prop("text-radial-offset", NUMBER, Some("0")).data_driven().interpolated();
        layout TEXT_VARIABLE_ANCHOR: Option<Vec<Anchor>> = prop("text-variable-anchor", ANCHORS, None);
        layout TEXT_VARIABLE_ANCHOR_OFFSET: Option<AnchorOffsets> = prop("text-variable-anchor-offset", PropertyType::VariableAnchorOffsetCollection, None).data_driven().interpolated();
        layout TEXT_ANCHOR: Anchor = prop("text-anchor", ANCHOR, Some("\"center\"")).data_driven();
        layout TEXT_MAX_ANGLE: f64 = prop("text-max-angle", NUMBER, Some("45")).interpolated();
        layout TEXT_WRITING_MODE: Option<Vec<WritingMode>> = prop("text-writing-mode", PropertyType::Array(Type::String, None), None);
        layout TEXT_ROTATE: f64 = prop("text-rotate", NUMBER, Some("0")).data_driven().interpolated();
        layout TEXT_PADDING: f64 = prop("text-padding", NUMBER, Some("2")).interpolated().range(0.0, None);
        layout TEXT_KEEP_UPRIGHT: bool = prop("text-keep-upright", PropertyType::Boolean, Some("true"));
        layout TEXT_TRANSFORM: TextTransform = prop("text-transform", enumeration::<TextTransform>(), Some("\"none\"")).data_driven();
        layout TEXT_OFFSET: [f64; 2] = prop("text-offset", OFFSET, Some("[0, 0]")).data_driven().interpolated();
        layout TEXT_ALLOW_OVERLAP: bool = prop("text-allow-overlap", PropertyType::Boolean, Some("false"));
        layout TEXT_OVERLAP: Option<Overlap> = prop("text-overlap", enumeration::<Overlap>(), None);
        layout TEXT_IGNORE_PLACEMENT: bool = prop("text-ignore-placement", PropertyType::Boolean, Some("false"));
        layout TEXT_OPTIONAL: bool = prop("text-optional", PropertyType::Boolean, Some("false"));
        layout SYMBOL_HEIGHT_OFFSET: f64 = prop("symbol-height-offset", NUMBER, Some("0")).data_driven().interpolated();
        layout SYMBOL_HEIGHT_ANCHOR: HeightAnchor = prop("symbol-height-anchor", enumeration::<HeightAnchor>(), Some("\"ground\""));
        paint ICON_OPACITY: f32 = prop("icon-opacity", NUMBER, Some("1")).data_driven().interpolated().range(0.0, Some(1.0));
        paint ICON_COLOR: Color = prop("icon-color", COLOR, Some("\"#000000\"")).data_driven().interpolated();
        paint ICON_HALO_COLOR: Color = prop("icon-halo-color", COLOR, Some("\"rgba(0, 0, 0, 0)\"")).data_driven().interpolated();
        paint ICON_HALO_WIDTH: f32 = prop("icon-halo-width", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint ICON_HALO_BLUR: f32 = prop("icon-halo-blur", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint ICON_TRANSLATE: [f32; 2] = prop("icon-translate", TRANSLATE, Some("[0, 0]")).interpolated();
        paint ICON_TRANSLATE_ANCHOR: TranslateAnchor = prop("icon-translate-anchor", enumeration::<TranslateAnchor>(), Some("\"map\""));
        paint TEXT_OPACITY: f32 = prop("text-opacity", NUMBER, Some("1")).data_driven().interpolated().range(0.0, Some(1.0));
        paint TEXT_COLOR: Color = prop("text-color", COLOR, Some("\"#000000\"")).data_driven().interpolated();
        paint TEXT_HALO_COLOR: Color = prop("text-halo-color", COLOR, Some("\"rgba(0, 0, 0, 0)\"")).data_driven().interpolated();
        paint TEXT_HALO_WIDTH: f32 = prop("text-halo-width", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint TEXT_HALO_BLUR: f32 = prop("text-halo-blur", NUMBER, Some("0")).data_driven().interpolated().range(0.0, None);
        paint TEXT_TRANSLATE: [f32; 2] = prop("text-translate", TRANSLATE, Some("[0, 0]")).interpolated();
        paint TEXT_TRANSLATE_ANCHOR: TranslateAnchor = prop("text-translate-anchor", enumeration::<TranslateAnchor>(), Some("\"map\""));
    }
}

/// The paint a symbol layer's icons and its text each have.
#[derive(Clone, Copy, Debug)]
pub struct SymbolPart {
    pub opacity: Key<f32>,
    pub color: Key<Color>,
    pub halo_color: Key<Color>,
    pub halo_width: Key<f32>,
    pub halo_blur: Key<f32>,
    pub translate: Key<[f32; 2]>,
    pub translate_anchor: Key<TranslateAnchor>,
}

pub const ICON_PART: SymbolPart = SymbolPart {
    opacity: ICON_OPACITY,
    color: ICON_COLOR,
    halo_color: ICON_HALO_COLOR,
    halo_width: ICON_HALO_WIDTH,
    halo_blur: ICON_HALO_BLUR,
    translate: ICON_TRANSLATE,
    translate_anchor: ICON_TRANSLATE_ANCHOR,
};

pub const TEXT_PART: SymbolPart = SymbolPart {
    opacity: TEXT_OPACITY,
    color: TEXT_COLOR,
    halo_color: TEXT_HALO_COLOR,
    halo_width: TEXT_HALO_WIDTH,
    halo_blur: TEXT_HALO_BLUR,
    translate: TEXT_TRANSLATE,
    translate_anchor: TEXT_TRANSLATE_ANCHOR,
};
