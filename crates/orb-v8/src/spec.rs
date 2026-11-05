//! The v8 spec entries of the properties Orbis types (`v8.json`).

use orb_expr::json::{self, Json};
use orb_expr::{PropertySpec, PropertyType, Type};

use crate::property::{
    Alignment, Anchor, Enum, HeightAnchor, HillshadeMethod, IconTextFit, IlluminationAnchor,
    LayerBlend, LineCap, LineJoin, Overlap, Resampling, SymbolPlacement, SymbolZOrder, TextJustify,
    TextRotationAlignment, TextTransform, TranslateAnchor, Visibility,
};

/// One property's spec entry.
#[derive(Debug)]
pub(crate) struct Prop {
    pub(crate) name: &'static str,
    pub(crate) ty: PropertyType,
    /// The spec default as JSON text.
    pub(crate) default: Option<&'static str>,
    /// `zoom` is an expression parameter.
    pub(crate) zoom: bool,
    /// Data-driven.
    pub(crate) feature: bool,
    pub(crate) interpolated: bool,
    pub(crate) minimum: Option<f64>,
    pub(crate) maximum: Option<f64>,
    /// Strings expand `{token}`s.
    pub(crate) tokens: bool,
}

impl Prop {
    pub(crate) fn spec(&self) -> PropertySpec {
        PropertySpec {
            ty: self.ty.clone(),
            default: self.default_json(),
            zoom: self.zoom,
            feature: self.feature,
            interpolated: self.interpolated,
            tokens: self.tokens,
        }
    }

    pub(crate) fn default_json(&self) -> Option<Json> {
        self.default
            .map(|text| json::parse(text).expect("spec defaults are JSON"))
    }
}

const fn prop(name: &'static str, ty: PropertyType, default: Option<&'static str>) -> Prop {
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
pub(crate) const VISIBILITY: Prop = prop(
    "visibility",
    enumeration::<Visibility>(),
    Some("\"visible\""),
)
.zoom_independent();

pub(crate) const BACKGROUND_COLOR: Prop =
    prop("background-color", COLOR, Some("\"#000000\"")).interpolated();
pub(crate) const BACKGROUND_PATTERN: Prop = prop("background-pattern", IMAGE, None);
pub(crate) const BACKGROUND_OPACITY: Prop = prop("background-opacity", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));

pub(crate) const FILL_SORT_KEY: Prop = prop("fill-sort-key", NUMBER, None).data_driven();
pub(crate) const FILL_ANTIALIAS: Prop = prop("fill-antialias", PropertyType::Boolean, Some("true"));
pub(crate) const FILL_OPACITY: Prop = prop("fill-opacity", NUMBER, Some("1"))
    .data_driven()
    .interpolated()
    .range(0.0, Some(1.0));
/// The layer composited once at this opacity (style-spec 26).
pub(crate) const FILL_LAYER_OPACITY: Prop = prop("fill-layer-opacity", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));
/// How the layer composites with what is beneath it (style-spec 26).
pub(crate) const FILL_LAYER_BLEND: Prop = prop(
    "fill-layer-blend",
    enumeration::<LayerBlend>(),
    Some("\"normal\""),
)
.zoom_independent();
pub(crate) const FILL_COLOR: Prop = prop("fill-color", COLOR, Some("\"#000000\""))
    .data_driven()
    .interpolated();
pub(crate) const FILL_OUTLINE_COLOR: Prop = prop("fill-outline-color", COLOR, None)
    .data_driven()
    .interpolated();
pub(crate) const FILL_TRANSLATE: Prop =
    prop("fill-translate", TRANSLATE, Some("[0, 0]")).interpolated();
pub(crate) const FILL_TRANSLATE_ANCHOR: Prop = prop(
    "fill-translate-anchor",
    enumeration::<TranslateAnchor>(),
    Some("\"map\""),
);
pub(crate) const FILL_PATTERN: Prop = prop("fill-pattern", IMAGE, None).data_driven();

pub(crate) const LINE_CAP: Prop =
    prop("line-cap", enumeration::<LineCap>(), Some("\"butt\"")).data_driven();
pub(crate) const LINE_JOIN: Prop =
    prop("line-join", enumeration::<LineJoin>(), Some("\"miter\"")).data_driven();
pub(crate) const LINE_MITER_LIMIT: Prop = prop("line-miter-limit", NUMBER, Some("2"))
    .data_driven()
    .interpolated();
pub(crate) const LINE_ROUND_LIMIT: Prop = prop("line-round-limit", NUMBER, Some("1.05"))
    .data_driven()
    .interpolated();
pub(crate) const LINE_SORT_KEY: Prop = prop("line-sort-key", NUMBER, None).data_driven();

pub(crate) const LINE_OPACITY: Prop = prop("line-opacity", NUMBER, Some("1"))
    .data_driven()
    .interpolated()
    .range(0.0, Some(1.0));
/// The layer composited once at this opacity (style-spec 26).
pub(crate) const LINE_LAYER_OPACITY: Prop = prop("line-layer-opacity", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));
/// How the layer composites with what is beneath it (style-spec 26).
pub(crate) const LINE_LAYER_BLEND: Prop = prop(
    "line-layer-blend",
    enumeration::<LayerBlend>(),
    Some("\"normal\""),
)
.zoom_independent();
pub(crate) const LINE_COLOR: Prop = prop("line-color", COLOR, Some("\"#000000\""))
    .data_driven()
    .interpolated();
pub(crate) const LINE_TRANSLATE: Prop =
    prop("line-translate", TRANSLATE, Some("[0, 0]")).interpolated();
pub(crate) const LINE_TRANSLATE_ANCHOR: Prop = prop(
    "line-translate-anchor",
    enumeration::<TranslateAnchor>(),
    Some("\"map\""),
);
pub(crate) const LINE_WIDTH: Prop = prop("line-width", NUMBER, Some("1"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const LINE_GAP_WIDTH: Prop = prop("line-gap-width", NUMBER, Some("0"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const LINE_OFFSET: Prop = prop("line-offset", NUMBER, Some("0"))
    .data_driven()
    .interpolated();
pub(crate) const LINE_BLUR: Prop = prop("line-blur", NUMBER, Some("0"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const LINE_DASHARRAY: Prop = prop(
    "line-dasharray",
    PropertyType::Array(Type::Number, None),
    None,
)
.data_driven()
.range(0.0, None);
pub(crate) const LINE_PATTERN: Prop = prop("line-pattern", IMAGE, None).data_driven();
/// `line-gradient` varies with `line-progress`, not zoom or the feature.
pub(crate) const LINE_GRADIENT: Prop = prop("line-gradient", COLOR, None)
    .zoom_independent()
    .interpolated();

pub(crate) const RASTER_OPACITY: Prop = prop("raster-opacity", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const RASTER_HUE_ROTATE: Prop =
    prop("raster-hue-rotate", NUMBER, Some("0")).interpolated();
pub(crate) const RASTER_BRIGHTNESS_MIN: Prop = prop("raster-brightness-min", NUMBER, Some("0"))
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const RASTER_BRIGHTNESS_MAX: Prop = prop("raster-brightness-max", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const RASTER_SATURATION: Prop = prop("raster-saturation", NUMBER, Some("0"))
    .interpolated()
    .range(-1.0, Some(1.0));
pub(crate) const RASTER_CONTRAST: Prop = prop("raster-contrast", NUMBER, Some("0"))
    .interpolated()
    .range(-1.0, Some(1.0));
/// `resampling`, which raster, hillshade and color-relief layers share.
pub(crate) const RESAMPLING: Prop = prop(
    "resampling",
    enumeration::<Resampling>(),
    Some("\"linear\""),
);
pub(crate) const RASTER_RESAMPLING: Prop = prop(
    "raster-resampling",
    enumeration::<Resampling>(),
    Some("\"linear\""),
);
/// Parsed and validated; tiles appear without fading.
pub(crate) const RASTER_FADE_DURATION: Prop = prop("raster-fade-duration", NUMBER, Some("300"))
    .interpolated()
    .range(0.0, None);

pub(crate) const HILLSHADE_ILLUMINATION_DIRECTION: Prop =
    prop("hillshade-illumination-direction", NUMBER, Some("335"))
        .interpolated()
        .range(0.0, Some(359.0));
pub(crate) const HILLSHADE_ILLUMINATION_ALTITUDE: Prop =
    prop("hillshade-illumination-altitude", NUMBER, Some("45"))
        .interpolated()
        .range(0.0, Some(90.0));
pub(crate) const HILLSHADE_ILLUMINATION_ANCHOR: Prop = prop(
    "hillshade-illumination-anchor",
    enumeration::<IlluminationAnchor>(),
    Some("\"viewport\""),
);
pub(crate) const HILLSHADE_EXAGGERATION: Prop = prop("hillshade-exaggeration", NUMBER, Some("0.5"))
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const HILLSHADE_SHADOW_COLOR: Prop =
    prop("hillshade-shadow-color", COLOR, Some("\"#000000\"")).interpolated();
pub(crate) const HILLSHADE_HIGHLIGHT_COLOR: Prop =
    prop("hillshade-highlight-color", COLOR, Some("\"#FFFFFF\"")).interpolated();
pub(crate) const HILLSHADE_ACCENT_COLOR: Prop =
    prop("hillshade-accent-color", COLOR, Some("\"#000000\"")).interpolated();
pub(crate) const HILLSHADE_METHOD: Prop = prop(
    "hillshade-method",
    enumeration::<HillshadeMethod>(),
    Some("\"standard\""),
);

pub(crate) const COLOR_RELIEF_OPACITY: Prop = prop("color-relief-opacity", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));
/// A colour ramp over `["elevation"]`, not zoom or the feature.
pub(crate) const COLOR_RELIEF_COLOR: Prop = prop("color-relief-color", COLOR, None)
    .zoom_independent()
    .interpolated();

pub(crate) const BACKGROUND_LAYOUT: &[&Prop] = &[&VISIBILITY];
/// Raster, hillshade and color-relief layers lay out nothing else.
pub(crate) const VISIBILITY_LAYOUT: &[&Prop] = &[&VISIBILITY];
pub(crate) const BACKGROUND_PAINT: &[&Prop] =
    &[&BACKGROUND_COLOR, &BACKGROUND_PATTERN, &BACKGROUND_OPACITY];
pub(crate) const FILL_LAYOUT: &[&Prop] = &[&FILL_SORT_KEY, &VISIBILITY];
pub(crate) const FILL_PAINT: &[&Prop] = &[
    &FILL_ANTIALIAS,
    &FILL_OPACITY,
    &FILL_LAYER_OPACITY,
    &FILL_LAYER_BLEND,
    &FILL_COLOR,
    &FILL_OUTLINE_COLOR,
    &FILL_TRANSLATE,
    &FILL_TRANSLATE_ANCHOR,
    &FILL_PATTERN,
];
pub(crate) const LINE_LAYOUT: &[&Prop] = &[
    &LINE_CAP,
    &LINE_JOIN,
    &LINE_MITER_LIMIT,
    &LINE_ROUND_LIMIT,
    &LINE_SORT_KEY,
    &VISIBILITY,
];
pub(crate) const LINE_PAINT: &[&Prop] = &[
    &LINE_OPACITY,
    &LINE_LAYER_OPACITY,
    &LINE_LAYER_BLEND,
    &LINE_COLOR,
    &LINE_TRANSLATE,
    &LINE_TRANSLATE_ANCHOR,
    &LINE_WIDTH,
    &LINE_GAP_WIDTH,
    &LINE_OFFSET,
    &LINE_BLUR,
    &LINE_DASHARRAY,
    &LINE_PATTERN,
    &LINE_GRADIENT,
];
pub(crate) const RASTER_PAINT: &[&Prop] = &[
    &RASTER_OPACITY,
    &RASTER_HUE_ROTATE,
    &RASTER_BRIGHTNESS_MIN,
    &RASTER_BRIGHTNESS_MAX,
    &RASTER_SATURATION,
    &RASTER_CONTRAST,
    &RESAMPLING,
    &RASTER_RESAMPLING,
    &RASTER_FADE_DURATION,
];
pub(crate) const HILLSHADE_PAINT: &[&Prop] = &[
    &HILLSHADE_ILLUMINATION_DIRECTION,
    &HILLSHADE_ILLUMINATION_ALTITUDE,
    &HILLSHADE_ILLUMINATION_ANCHOR,
    &HILLSHADE_EXAGGERATION,
    &HILLSHADE_SHADOW_COLOR,
    &HILLSHADE_HIGHLIGHT_COLOR,
    &HILLSHADE_ACCENT_COLOR,
    &HILLSHADE_METHOD,
    &RESAMPLING,
];
pub(crate) const COLOR_RELIEF_PAINT: &[&Prop] =
    &[&COLOR_RELIEF_OPACITY, &COLOR_RELIEF_COLOR, &RESAMPLING];

pub(crate) const CIRCLE_SORT_KEY: Prop = prop("circle-sort-key", NUMBER, None).data_driven();
pub(crate) const CIRCLE_RADIUS: Prop = prop("circle-radius", NUMBER, Some(r#"5"#))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const CIRCLE_COLOR: Prop = prop("circle-color", COLOR, Some(r##""#000000""##))
    .data_driven()
    .interpolated();
pub(crate) const CIRCLE_BLUR: Prop = prop("circle-blur", NUMBER, Some(r#"0"#))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const CIRCLE_OPACITY: Prop = prop("circle-opacity", NUMBER, Some(r#"1"#))
    .data_driven()
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const CIRCLE_TRANSLATE: Prop =
    prop("circle-translate", TRANSLATE, Some(r#"[0, 0]"#)).interpolated();
pub(crate) const CIRCLE_TRANSLATE_ANCHOR: Prop = prop(
    "circle-translate-anchor",
    enumeration::<TranslateAnchor>(),
    Some(r#""map""#),
);
pub(crate) const CIRCLE_PITCH_SCALE: Prop = prop(
    "circle-pitch-scale",
    enumeration::<TranslateAnchor>(),
    Some(r#""map""#),
);
pub(crate) const CIRCLE_PITCH_ALIGNMENT: Prop = prop(
    "circle-pitch-alignment",
    enumeration::<TranslateAnchor>(),
    Some(r#""viewport""#),
);
pub(crate) const CIRCLE_STROKE_WIDTH: Prop = prop("circle-stroke-width", NUMBER, Some(r#"0"#))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const CIRCLE_STROKE_COLOR: Prop =
    prop("circle-stroke-color", COLOR, Some(r##""#000000""##))
        .data_driven()
        .interpolated();
pub(crate) const CIRCLE_STROKE_OPACITY: Prop = prop("circle-stroke-opacity", NUMBER, Some(r#"1"#))
    .data_driven()
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const HEATMAP_RADIUS: Prop = prop("heatmap-radius", NUMBER, Some(r#"30"#))
    .data_driven()
    .interpolated()
    .range(1.0, None);
pub(crate) const HEATMAP_WEIGHT: Prop = prop("heatmap-weight", NUMBER, Some(r#"1"#))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const HEATMAP_INTENSITY: Prop = prop("heatmap-intensity", NUMBER, Some(r#"1"#))
    .interpolated()
    .range(0.0, None);
pub(crate) const HEATMAP_OPACITY: Prop = prop("heatmap-opacity", NUMBER, Some(r#"1"#))
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const HEATMAP_COLOR: Prop = prop("heatmap-color", COLOR, Some(r#"["interpolate",["linear"],["heatmap-density"],0,"rgba(0, 0, 255, 0)",0.1,"royalblue",0.3,"cyan",0.5,"lime",0.7,"yellow",1,"red"]"#)).interpolated().zoom_independent();
pub(crate) const CIRCLE_LAYOUT: &[&Prop] = &[&VISIBILITY, &CIRCLE_SORT_KEY];
pub(crate) const CIRCLE_PAINT: &[&Prop] = &[
    &CIRCLE_RADIUS,
    &CIRCLE_COLOR,
    &CIRCLE_BLUR,
    &CIRCLE_OPACITY,
    &CIRCLE_TRANSLATE,
    &CIRCLE_TRANSLATE_ANCHOR,
    &CIRCLE_PITCH_SCALE,
    &CIRCLE_PITCH_ALIGNMENT,
    &CIRCLE_STROKE_WIDTH,
    &CIRCLE_STROKE_COLOR,
    &CIRCLE_STROKE_OPACITY,
];
pub(crate) const HEATMAP_PAINT: &[&Prop] = &[
    &HEATMAP_RADIUS,
    &HEATMAP_WEIGHT,
    &HEATMAP_INTENSITY,
    &HEATMAP_COLOR,
    &HEATMAP_OPACITY,
];

// The v8 flat extrusion properties. Opacity applies to the whole layer.
pub(crate) const EXTRUSION_COLOR: Prop = prop("fill-extrusion-color", COLOR, Some("\"#000000\""))
    .data_driven()
    .interpolated();
pub(crate) const EXTRUSION_OPACITY: Prop = prop("fill-extrusion-opacity", NUMBER, Some("1"))
    .interpolated()
    .range(0.0, Some(1.0));
pub(crate) const EXTRUSION_HEIGHT: Prop = prop("fill-extrusion-height", NUMBER, Some("0"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const EXTRUSION_BASE: Prop = prop("fill-extrusion-base", NUMBER, Some("0"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const EXTRUSION_TRANSLATE: Prop =
    prop("fill-extrusion-translate", TRANSLATE, Some("[0,0]")).interpolated();
pub(crate) const EXTRUSION_TRANSLATE_ANCHOR: Prop = prop(
    "fill-extrusion-translate-anchor",
    enumeration::<TranslateAnchor>(),
    Some("\"map\""),
);
pub(crate) const EXTRUSION_VERTICAL_GRADIENT: Prop = prop(
    "fill-extrusion-vertical-gradient",
    PropertyType::Boolean,
    Some("true"),
);
pub(crate) const EXTRUSION_PATTERN: Prop =
    prop("fill-extrusion-pattern", IMAGE, None).data_driven();
pub(crate) const EXTRUSION_PAINT: &[&Prop] = &[
    &EXTRUSION_COLOR,
    &EXTRUSION_OPACITY,
    &EXTRUSION_HEIGHT,
    &EXTRUSION_BASE,
    &EXTRUSION_TRANSLATE,
    &EXTRUSION_TRANSLATE_ANCHOR,
    &EXTRUSION_VERTICAL_GRADIENT,
    &EXTRUSION_PATTERN,
];

pub(crate) const EXTRUSION_ROUNDED_CORNER: Prop =
    prop("fill-extrusion-rounded-corner-distance", NUMBER, Some("0"))
        .range(0.0, None)
        .zoom_independent();
pub(crate) const EXTRUSION_LAYOUT: &[&Prop] = &[&VISIBILITY, &EXTRUSION_ROUNDED_CORNER];
const OFFSET: PropertyType = PropertyType::Array(Type::Number, Some(2));
const ANCHOR: PropertyType = enumeration::<Anchor>();
const ANCHORS: PropertyType = PropertyType::Array(Type::String, None);

pub(crate) const SYMBOL_PLACEMENT: Prop = prop(
    "symbol-placement",
    enumeration::<SymbolPlacement>(),
    Some("\"point\""),
);
pub(crate) const SYMBOL_SPACING: Prop = prop("symbol-spacing", NUMBER, Some("250"))
    .interpolated()
    .range(1.0, None);
pub(crate) const SYMBOL_AVOID_EDGES: Prop =
    prop("symbol-avoid-edges", PropertyType::Boolean, Some("false"));
pub(crate) const SYMBOL_SORT_KEY: Prop = prop("symbol-sort-key", NUMBER, None).data_driven();
pub(crate) const SYMBOL_Z_ORDER: Prop = prop(
    "symbol-z-order",
    enumeration::<SymbolZOrder>(),
    Some("\"auto\""),
);
pub(crate) const ICON_ALLOW_OVERLAP: Prop =
    prop("icon-allow-overlap", PropertyType::Boolean, Some("false"));
pub(crate) const ICON_OVERLAP: Prop = prop("icon-overlap", enumeration::<Overlap>(), None);
pub(crate) const ICON_IGNORE_PLACEMENT: Prop = prop(
    "icon-ignore-placement",
    PropertyType::Boolean,
    Some("false"),
);
pub(crate) const ICON_OPTIONAL: Prop = prop("icon-optional", PropertyType::Boolean, Some("false"));
pub(crate) const ICON_ROTATION_ALIGNMENT: Prop = prop(
    "icon-rotation-alignment",
    enumeration::<Alignment>(),
    Some("\"auto\""),
)
.data_driven();
pub(crate) const ICON_SIZE: Prop = prop("icon-size", NUMBER, Some("1"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const ICON_TEXT_FIT: Prop = prop(
    "icon-text-fit",
    enumeration::<IconTextFit>(),
    Some("\"none\""),
);
pub(crate) const ICON_TEXT_FIT_PADDING: Prop = prop(
    "icon-text-fit-padding",
    PropertyType::Array(Type::Number, Some(4)),
    Some("[0, 0, 0, 0]"),
)
.interpolated();
pub(crate) const ICON_IMAGE: Prop = prop("icon-image", IMAGE, None).data_driven().tokens();
pub(crate) const ICON_ROTATE: Prop = prop("icon-rotate", NUMBER, Some("0"))
    .data_driven()
    .interpolated();
pub(crate) const ICON_PADDING: Prop = prop("icon-padding", PropertyType::Padding, Some("[2]"))
    .data_driven()
    .interpolated();
pub(crate) const ICON_KEEP_UPRIGHT: Prop =
    prop("icon-keep-upright", PropertyType::Boolean, Some("false"));
pub(crate) const ICON_OFFSET: Prop = prop("icon-offset", OFFSET, Some("[0, 0]"))
    .data_driven()
    .interpolated();
pub(crate) const ICON_ANCHOR: Prop = prop("icon-anchor", ANCHOR, Some("\"center\"")).data_driven();
pub(crate) const ICON_PITCH_ALIGNMENT: Prop = prop(
    "icon-pitch-alignment",
    enumeration::<Alignment>(),
    Some("\"auto\""),
);
pub(crate) const TEXT_PITCH_ALIGNMENT: Prop = prop(
    "text-pitch-alignment",
    enumeration::<Alignment>(),
    Some("\"auto\""),
);
pub(crate) const TEXT_ROTATION_ALIGNMENT: Prop = prop(
    "text-rotation-alignment",
    enumeration::<TextRotationAlignment>(),
    Some("\"auto\""),
);
pub(crate) const TEXT_FIELD: Prop = prop("text-field", PropertyType::Formatted, Some("\"\""))
    .data_driven()
    .tokens();
pub(crate) const TEXT_FONT: Prop = prop(
    "text-font",
    PropertyType::Array(Type::String, None),
    Some("[\"Open Sans Regular\", \"Arial Unicode MS Regular\"]"),
)
.data_driven();
pub(crate) const TEXT_SIZE: Prop = prop("text-size", NUMBER, Some("16"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const TEXT_MAX_WIDTH: Prop = prop("text-max-width", NUMBER, Some("10"))
    .data_driven()
    .interpolated()
    .range(0.0, None);
pub(crate) const TEXT_LINE_HEIGHT: Prop =
    prop("text-line-height", NUMBER, Some("1.2")).interpolated();
pub(crate) const TEXT_LETTER_SPACING: Prop = prop("text-letter-spacing", NUMBER, Some("0"))
    .data_driven()
    .interpolated();
pub(crate) const TEXT_JUSTIFY: Prop = prop(
    "text-justify",
    enumeration::<TextJustify>(),
    Some("\"center\""),
)
.data_driven();
pub(crate) const TEXT_RADIAL_OFFSET: Prop = prop("text-radial-offset", NUMBER, Some("0"))
    .data_driven()
    .interpolated();
pub(crate) const TEXT_VARIABLE_ANCHOR: Prop = prop("text-variable-anchor", ANCHORS, None);
pub(crate) const TEXT_VARIABLE_ANCHOR_OFFSET: Prop = prop(
    "text-variable-anchor-offset",
    PropertyType::VariableAnchorOffsetCollection,
    None,
)
.data_driven()
.interpolated();
pub(crate) const TEXT_ANCHOR: Prop = prop("text-anchor", ANCHOR, Some("\"center\"")).data_driven();
pub(crate) const TEXT_MAX_ANGLE: Prop = prop("text-max-angle", NUMBER, Some("45")).interpolated();
pub(crate) const TEXT_WRITING_MODE: Prop = prop(
    "text-writing-mode",
    PropertyType::Array(Type::String, None),
    None,
);
pub(crate) const TEXT_ROTATE: Prop = prop("text-rotate", NUMBER, Some("0"))
    .data_driven()
    .interpolated();
pub(crate) const TEXT_PADDING: Prop = prop("text-padding", NUMBER, Some("2"))
    .interpolated()
    .range(0.0, None);
pub(crate) const TEXT_KEEP_UPRIGHT: Prop =
    prop("text-keep-upright", PropertyType::Boolean, Some("true"));
pub(crate) const TEXT_TRANSFORM: Prop = prop(
    "text-transform",
    enumeration::<TextTransform>(),
    Some("\"none\""),
)
.data_driven();
pub(crate) const TEXT_OFFSET: Prop = prop("text-offset", OFFSET, Some("[0, 0]"))
    .data_driven()
    .interpolated();
pub(crate) const TEXT_ALLOW_OVERLAP: Prop =
    prop("text-allow-overlap", PropertyType::Boolean, Some("false"));
pub(crate) const TEXT_OVERLAP: Prop = prop("text-overlap", enumeration::<Overlap>(), None);
pub(crate) const TEXT_IGNORE_PLACEMENT: Prop = prop(
    "text-ignore-placement",
    PropertyType::Boolean,
    Some("false"),
);
pub(crate) const TEXT_OPTIONAL: Prop = prop("text-optional", PropertyType::Boolean, Some("false"));
pub(crate) const SYMBOL_HEIGHT_OFFSET: Prop = prop("symbol-height-offset", NUMBER, Some("0"))
    .data_driven()
    .interpolated();
pub(crate) const SYMBOL_HEIGHT_ANCHOR: Prop = prop(
    "symbol-height-anchor",
    enumeration::<HeightAnchor>(),
    Some("\"ground\""),
);

pub(crate) const SYMBOL_LAYOUT: &[&Prop] = &[
    &SYMBOL_PLACEMENT,
    &SYMBOL_SPACING,
    &SYMBOL_AVOID_EDGES,
    &SYMBOL_SORT_KEY,
    &SYMBOL_Z_ORDER,
    &ICON_ALLOW_OVERLAP,
    &ICON_OVERLAP,
    &ICON_IGNORE_PLACEMENT,
    &ICON_OPTIONAL,
    &ICON_ROTATION_ALIGNMENT,
    &ICON_SIZE,
    &ICON_TEXT_FIT,
    &ICON_TEXT_FIT_PADDING,
    &ICON_IMAGE,
    &ICON_ROTATE,
    &ICON_PADDING,
    &ICON_KEEP_UPRIGHT,
    &ICON_OFFSET,
    &ICON_ANCHOR,
    &ICON_PITCH_ALIGNMENT,
    &TEXT_PITCH_ALIGNMENT,
    &TEXT_ROTATION_ALIGNMENT,
    &TEXT_FIELD,
    &TEXT_FONT,
    &TEXT_SIZE,
    &TEXT_MAX_WIDTH,
    &TEXT_LINE_HEIGHT,
    &TEXT_LETTER_SPACING,
    &TEXT_JUSTIFY,
    &TEXT_RADIAL_OFFSET,
    &TEXT_VARIABLE_ANCHOR,
    &TEXT_VARIABLE_ANCHOR_OFFSET,
    &TEXT_ANCHOR,
    &TEXT_MAX_ANGLE,
    &TEXT_WRITING_MODE,
    &TEXT_ROTATE,
    &TEXT_PADDING,
    &TEXT_KEEP_UPRIGHT,
    &TEXT_TRANSFORM,
    &TEXT_OFFSET,
    &TEXT_ALLOW_OVERLAP,
    &TEXT_OVERLAP,
    &TEXT_IGNORE_PLACEMENT,
    &TEXT_OPTIONAL,
    &SYMBOL_HEIGHT_OFFSET,
    &SYMBOL_HEIGHT_ANCHOR,
    &VISIBILITY,
];

/// The paint of a symbol layer's icons (`icon-*`) or text (`text-*`).
macro_rules! symbol_paint {
    ($opacity:ident, $color:ident, $halo_color:ident, $halo_width:ident, $halo_blur:ident,
     $translate:ident, $translate_anchor:ident, $prefix:literal) => {
        pub(crate) const $opacity: Prop = prop(concat!($prefix, "-opacity"), NUMBER, Some("1"))
            .data_driven()
            .interpolated()
            .range(0.0, Some(1.0));
        pub(crate) const $color: Prop =
            prop(concat!($prefix, "-color"), COLOR, Some("\"#000000\""))
                .data_driven()
                .interpolated();
        pub(crate) const $halo_color: Prop = prop(
            concat!($prefix, "-halo-color"),
            COLOR,
            Some("\"rgba(0, 0, 0, 0)\""),
        )
        .data_driven()
        .interpolated();
        pub(crate) const $halo_width: Prop =
            prop(concat!($prefix, "-halo-width"), NUMBER, Some("0"))
                .data_driven()
                .interpolated()
                .range(0.0, None);
        pub(crate) const $halo_blur: Prop = prop(concat!($prefix, "-halo-blur"), NUMBER, Some("0"))
            .data_driven()
            .interpolated()
            .range(0.0, None);
        pub(crate) const $translate: Prop =
            prop(concat!($prefix, "-translate"), TRANSLATE, Some("[0, 0]")).interpolated();
        pub(crate) const $translate_anchor: Prop = prop(
            concat!($prefix, "-translate-anchor"),
            enumeration::<TranslateAnchor>(),
            Some("\"map\""),
        );
    };
}

symbol_paint!(
    ICON_OPACITY,
    ICON_COLOR,
    ICON_HALO_COLOR,
    ICON_HALO_WIDTH,
    ICON_HALO_BLUR,
    ICON_TRANSLATE,
    ICON_TRANSLATE_ANCHOR,
    "icon"
);
symbol_paint!(
    TEXT_OPACITY,
    TEXT_COLOR,
    TEXT_HALO_COLOR,
    TEXT_HALO_WIDTH,
    TEXT_HALO_BLUR,
    TEXT_TRANSLATE,
    TEXT_TRANSLATE_ANCHOR,
    "text"
);

pub(crate) const SYMBOL_PAINT: &[&Prop] = &[
    &ICON_OPACITY,
    &ICON_COLOR,
    &ICON_HALO_COLOR,
    &ICON_HALO_WIDTH,
    &ICON_HALO_BLUR,
    &ICON_TRANSLATE,
    &ICON_TRANSLATE_ANCHOR,
    &TEXT_OPACITY,
    &TEXT_COLOR,
    &TEXT_HALO_COLOR,
    &TEXT_HALO_WIDTH,
    &TEXT_HALO_BLUR,
    &TEXT_TRANSLATE,
    &TEXT_TRANSLATE_ANCHOR,
];

/// The spec of a typed layer's paint (`paint: true`) or layout property, if
/// Orbis models it.
pub(crate) fn find(layer_type: &str, paint: bool, name: &str) -> Option<PropertySpec> {
    find_prop(layer_type, paint, name).map(Prop::spec)
}

/// [`find`]'s property itself.
pub(crate) fn find_prop(layer_type: &str, paint: bool, name: &str) -> Option<&'static Prop> {
    let props: &[&Prop] = match (layer_type, paint) {
        ("background", false) => BACKGROUND_LAYOUT,
        ("background", true) => BACKGROUND_PAINT,
        ("fill", false) => FILL_LAYOUT,
        ("fill", true) => FILL_PAINT,
        ("line", false) => LINE_LAYOUT,
        ("line", true) => LINE_PAINT,
        ("circle", false) => CIRCLE_LAYOUT,
        ("circle", true) => CIRCLE_PAINT,
        ("heatmap", true) => HEATMAP_PAINT,
        ("raster", true) => RASTER_PAINT,
        ("hillshade", true) => HILLSHADE_PAINT,
        ("color-relief", true) => COLOR_RELIEF_PAINT,
        ("fill-extrusion", true) => EXTRUSION_PAINT,
        ("fill-extrusion", false) => EXTRUSION_LAYOUT,
        ("symbol", false) => SYMBOL_LAYOUT,
        ("symbol", true) => SYMBOL_PAINT,
        (_, false) => VISIBILITY_LAYOUT,
        _ => &[],
    };
    props.iter().find(|p| p.name == name).copied()
}
