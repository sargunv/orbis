//! Typed property values.

use orb_expr::{Bound, Color, Kind, PropertyExpression, Value};

/// A value type a property can hold.
pub trait PropertyValue: Clone + Sized {
    /// The typed form of an evaluated value, or `None` if it doesn't fit,
    /// in which case the property falls back to its default.
    fn from_value(value: &Value<'_>) -> Option<Self>;

    /// The value of a property the spec gives no default, if this type can
    /// represent one.
    const UNSET: Option<Self> = None;
}

impl PropertyValue for f32 {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        value.as_f64().map(|n| n as f32)
    }
}

/// Symbol layout reads numbers at full precision, as GL JS's doubles.
impl PropertyValue for f64 {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        value.as_f64()
    }
}

impl PropertyValue for bool {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        value.as_bool()
    }
}

impl PropertyValue for Color {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        value.as_color()
    }
}

impl PropertyValue for String {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        match value {
            Value::String(s) | Value::Image { name: s, .. } => Some(s.to_string()),
            _ => None,
        }
    }
}

impl<const N: usize> PropertyValue for [f32; N] {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        let values = value.as_array()?;
        if values.len() != N {
            return None;
        }
        let mut out = [0.0; N];
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = value.as_f64()? as f32;
        }
        Some(out)
    }
}

impl<const N: usize> PropertyValue for [f64; N] {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        let values = items(value)?;
        if values.len() != N {
            return None;
        }
        let mut out = [0.0; N];
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = value.as_f64()?;
        }
        Some(out)
    }
}

/// The items of an array or typed array (`padding`, `numberArray`, …).
fn items<'v>(value: &'v Value<'_>) -> Option<&'v [Value<'static>]> {
    match value {
        Value::Array(items) | Value::TypedArray(_, items) => Some(items),
        _ => None,
    }
}

impl PropertyValue for Vec<String> {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        items(value)?
            .iter()
            .map(|v| v.as_str().map(str::to_string))
            .collect()
    }
}

/// A `formatted` value: its sections. A plain string is one section.
impl PropertyValue for Vec<orb_expr::FormattedSection> {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        match value {
            Value::Formatted(sections) => Some(sections.to_vec()),
            Value::String(text) => Some(vec![orb_expr::FormattedSection::plain(text)]),
            _ => None,
        }
    }
}

/// A `padding`: top, right, bottom, left, expanded from 1–4 values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Padding(pub [f64; 4]);

impl PropertyValue for Padding {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        let values: Vec<f64> = match value {
            Value::Number(n) => vec![*n],
            v => items(v)?.iter().map(Value::as_f64).collect::<Option<_>>()?,
        };
        Some(Self(match values[..] {
            [a] => [a; 4],
            [a, b] => [a, b, a, b],
            [a, b, c] => [a, b, c, b],
            [a, b, c, d] => [a, b, c, d],
            _ => return None,
        }))
    }
}

/// A `variableAnchorOffsetCollection`: anchors with their offsets in ems.
pub type AnchorOffsets = Vec<(Anchor, [f64; 2])>;

impl PropertyValue for AnchorOffsets {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        let items = items(value)?;
        if items.len() % 2 != 0 {
            return None;
        }
        items
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[anchor, offset]| {
                Some((
                    Anchor::from_name(anchor.as_str()?)?,
                    <[f64; 2]>::from_value(offset)?,
                ))
            })
            .collect()
    }
}

impl<E: Enum> PropertyValue for Vec<E> {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        items(value)?
            .iter()
            .map(|v| E::from_name(v.as_str()?))
            .collect()
    }
}

impl PropertyValue for Vec<f32> {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        value
            .as_array()?
            .iter()
            .map(|v| v.as_f64().map(|n| n as f32))
            .collect()
    }
}

/// Properties without a spec default: `None` when unset or when the value
/// evaluates to nothing.
impl<T: PropertyValue> PropertyValue for Option<T> {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        T::from_value(value).map(Some)
    }

    const UNSET: Option<Self> = Some(None);
}

/// A string enumeration from the v8 spec.
pub trait Enum: Copy + Sized + 'static {
    /// The v8 names, in declaration order.
    const VALUES: &'static [&'static str];
    fn from_name(name: &str) -> Option<Self>;
    fn name(self) -> &'static str;
}

macro_rules! v8_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant),+
        }

        impl Enum for $name {
            const VALUES: &'static [&'static str] = &[$($text),+];

            fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($text => Some(Self::$variant),)+
                    _ => None,
                }
            }

            fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }

        impl PropertyValue for $name {
            fn from_value(value: &Value<'_>) -> Option<Self> {
                Self::from_name(value.as_str()?)
            }
        }
    };
}

v8_enum!(
    /// `visibility`.
    Visibility { Visible = "visible", None = "none" }
);
v8_enum!(
    /// `fill-layer-blend`, `line-layer-blend`: how the layer, composited
    /// once, combines with what is beneath it (W3C compositing modes).
    LayerBlend {
        Normal = "normal",
        Multiply = "multiply",
        Color = "color",
        Luminosity = "luminosity",
        PlusLighter = "plus-lighter",
        Screen = "screen",
        Overlay = "overlay"
    }
);
v8_enum!(
    /// `*-translate-anchor`.
    TranslateAnchor { Map = "map", Viewport = "viewport" }
);
v8_enum!(
    /// `line-cap`.
    LineCap { Butt = "butt", Round = "round", Square = "square" }
);
v8_enum!(
    /// `line-join`.
    LineJoin { Bevel = "bevel", Round = "round", Miter = "miter" }
);
v8_enum!(
    /// `symbol-placement`.
    SymbolPlacement { Point = "point", Line = "line", LineCenter = "line-center" }
);
v8_enum!(
    /// `symbol-z-order`.
    SymbolZOrder { Auto = "auto", ViewportY = "viewport-y", Source = "source" }
);
v8_enum!(
    /// `text-overlap`, `icon-overlap`.
    Overlap { Never = "never", Always = "always", Cooperative = "cooperative" }
);
v8_enum!(
    /// `icon-rotation-alignment` and the `*-pitch-alignment`s.
    Alignment { Map = "map", Viewport = "viewport", Auto = "auto" }
);
v8_enum!(
    /// `text-rotation-alignment`.
    TextRotationAlignment {
        Map = "map",
        Viewport = "viewport",
        ViewportGlyph = "viewport-glyph",
        Auto = "auto",
    }
);
v8_enum!(
    /// `icon-text-fit`.
    IconTextFit { None = "none", Width = "width", Height = "height", Both = "both" }
);
v8_enum!(
    /// `text-anchor`, `icon-anchor` and `text-variable-anchor` items.
    Anchor {
        Center = "center",
        Left = "left",
        Right = "right",
        Top = "top",
        Bottom = "bottom",
        TopLeft = "top-left",
        TopRight = "top-right",
        BottomLeft = "bottom-left",
        BottomRight = "bottom-right",
    }
);
v8_enum!(
    /// `text-justify`.
    TextJustify { Auto = "auto", Left = "left", Center = "center", Right = "right" }
);
v8_enum!(
    /// `text-writing-mode` items.
    WritingMode { Horizontal = "horizontal", Vertical = "vertical" }
);
v8_enum!(
    /// `text-transform`.
    TextTransform { None = "none", Uppercase = "uppercase", Lowercase = "lowercase" }
);
v8_enum!(
    /// `symbol-height-anchor`.
    HeightAnchor { Ground = "ground", Absolute = "absolute" }
);
v8_enum!(
    /// A vector source's tile `scheme`.
    Scheme { Xyz = "xyz", Tms = "tms" }
);
v8_enum!(
    /// A vector source's tile `encoding`.
    Encoding { Mvt = "mvt", Mlt = "mlt" }
);
v8_enum!(
    /// How a `raster-dem` source's colours encode elevation.
    DemEncoding { Mapbox = "mapbox", Terrarium = "terrarium", Custom = "custom" }
);
v8_enum!(
    /// `resampling` and `raster-resampling`: how image texels are filtered.
    Resampling { Linear = "linear", Nearest = "nearest" }
);
v8_enum!(
    /// `hillshade-illumination-anchor`.
    IlluminationAnchor { Map = "map", Viewport = "viewport" }
);
v8_enum!(
    /// `hillshade-method`.
    HillshadeMethod {
        Standard = "standard",
        Basic = "basic",
        Combined = "combined",
        Igor = "igor",
        Multidirectional = "multidirectional",
    }
);

/// A paint or layout property value: a constant, or an expression with its
/// classification and the spec default it falls back to.
#[derive(Clone, Debug)]
pub struct Property<T> {
    value: Inner<T>,
}

#[derive(Clone, Debug)]
enum Inner<T> {
    Constant(T),
    /// Boxed: most properties are constants and stay small.
    Expression {
        expression: Box<PropertyExpression>,
        default: T,
    },
}

impl<T: PropertyValue> Property<T> {
    pub fn constant(value: T) -> Self {
        Self {
            value: Inner::Constant(value),
        }
    }

    /// A property backed by an expression: one that isn't constant, or
    /// one that reads `image`.
    pub(crate) fn expression(expression: PropertyExpression, default: T) -> Self {
        debug_assert!(
            expression.kind() != Kind::Constant || expression.expression().reads_images()
        );
        Self {
            value: Inner::Expression {
                expression: Box::new(expression),
                default,
            },
        }
    }

    pub fn kind(&self) -> Kind {
        match &self.value {
            Inner::Constant(_) => Kind::Constant,
            Inner::Expression { expression, .. } => expression.kind(),
        }
    }

    /// The value, if it is the same everywhere.
    pub fn as_constant(&self) -> Option<&T> {
        match &self.value {
            Inner::Constant(v) => Some(v),
            Inner::Expression { .. } => None,
        }
    }

    /// What an expression falls back to: the spec default (a constant's
    /// value for a constant).
    pub fn default_value(&self) -> T {
        match &self.value {
            Inner::Constant(v) => v.clone(),
            Inner::Expression { default, .. } => default.clone(),
        }
    }

    /// The expression of a camera, source or composite value, for its zoom
    /// curve and `interpolation_factor`.
    pub fn as_expression(&self) -> Option<&PropertyExpression> {
        match &self.value {
            Inner::Constant(_) => None,
            Inner::Expression { expression, .. } => Some(expression),
        }
    }

    /// The value at `zoom` without a feature: exact for constant and camera
    /// values. A source or composite value reads no properties, so it
    /// usually falls back to the default.
    pub fn evaluate(&self, zoom: f64) -> T {
        match &self.value {
            Inner::Constant(v) => v.clone(),
            Inner::Expression {
                expression,
                default,
            } => fit(expression.expression().evaluate(zoom).ok(), default),
        }
    }

    /// The value at `zoom` for a feature with `state` that reads no feature
    /// properties: exact for constant, camera and state-only values.
    pub fn evaluate_state(&self, zoom: f64, state: &[(String, Value<'static>)]) -> T {
        match &self.value {
            Inner::Constant(v) => v.clone(),
            Inner::Expression {
                expression,
                default,
            } => {
                let context = orb_expr::EvaluationContext {
                    zoom,
                    feature_state: state,
                    ..Default::default()
                };
                fit(
                    expression.expression().evaluate_with(&context).ok(),
                    default,
                )
            }
        }
    }

    /// Whether the value reads `feature-state`, but no feature properties:
    /// plan §11.3's `StateLookup` class.
    pub fn is_state_lookup(&self) -> bool {
        self.as_expression()
            .is_some_and(|e| e.reads_feature_state() && !e.reads_feature_properties())
    }

    /// Binds the property to a tile layer, to evaluate it per feature.
    pub fn bind<'a>(&'a self, layer: &'a orb_tile::Layer) -> BoundProperty<'a, T> {
        BoundProperty {
            property: self,
            bound: self.as_expression().map(|e| e.expression().bind(layer)),
        }
    }

    /// Binds the property to a tile layer with the style's images, which
    /// `image` expressions find available.
    pub fn bind_with_images<'a>(
        &'a self,
        layer: &'a orb_tile::Layer,
        images: &'a [String],
    ) -> BoundProperty<'a, T> {
        BoundProperty {
            property: self,
            bound: self
                .as_expression()
                .map(|e| e.expression().bind(layer).with_images(images)),
        }
    }
}

/// GL JS's fallback: `null`, `NaN`, an evaluation error or a value of the
/// wrong shape yields the default.
fn fit<T: PropertyValue>(value: Option<Value<'_>>, default: &T) -> T {
    match value {
        None | Some(Value::Null) => default.clone(),
        Some(Value::Number(n)) if n.is_nan() => default.clone(),
        Some(v) => T::from_value(&v).unwrap_or_else(|| default.clone()),
    }
}

/// A property bound to one tile layer.
#[derive(Debug)]
pub struct BoundProperty<'a, T> {
    property: &'a Property<T>,
    bound: Option<Bound<'a>>,
}

impl<T: PropertyValue> BoundProperty<'_, T> {
    /// The value for feature `row` at `zoom`.
    pub fn evaluate(&self, zoom: f64, row: usize) -> T {
        match (&self.property.value, &self.bound) {
            (Inner::Expression { default, .. }, Some(bound)) => {
                fit(bound.evaluate(zoom, row).ok(), default)
            }
            (Inner::Constant(v), _) => v.clone(),
            (Inner::Expression { .. }, None) => unreachable!("expressions bind"),
        }
    }
}
