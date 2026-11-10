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

/// Any value, untyped: transitions ease between them.
impl PropertyValue for Value<'static> {
    fn from_value(value: &Value<'_>) -> Option<Self> {
        Some(value.clone().into_owned())
    }

    const UNSET: Option<Self> = Some(Value::Null);
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
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

/// A stored property value: a constant, an expression, or for properties
/// that take one value per light (`hillshade-*`), a list of them. Its type
/// and default come from where it is stored: a layer's
/// [`PropertyId`](crate::PropertyId), or a typed [`Property`].
#[derive(Clone, Debug)]
pub struct Slot {
    value: Inner,
    /// The document gives the value.
    set: bool,
}

#[derive(Clone, Debug)]
enum Inner {
    Constant(Value<'static>),
    /// Boxed: most properties are constants and stay small.
    Expression(Box<PropertyExpression>),
    List(Box<[Slot]>),
}

impl Slot {
    pub fn constant(value: Value<'static>, set: bool) -> Self {
        Self {
            value: Inner::Constant(value),
            set,
        }
    }

    /// A slot backed by an expression: one that isn't constant, or one
    /// that reads `image`.
    pub fn expression(expression: PropertyExpression) -> Self {
        debug_assert!(
            expression.kind() != Kind::Constant || expression.expression().reads_images()
        );
        Self {
            value: Inner::Expression(Box::new(expression)),
            set: true,
        }
    }

    /// A colour ramp's expression, whatever its kind.
    pub(crate) fn expression_any(expression: PropertyExpression, set: bool) -> Self {
        Self {
            value: Inner::Expression(Box::new(expression)),
            set,
        }
    }

    pub fn list(items: Vec<Slot>) -> Self {
        Self {
            value: Inner::List(items.into()),
            set: true,
        }
    }

    #[inline]
    pub fn kind(&self) -> Kind {
        match &self.value {
            Inner::Constant(_) => Kind::Constant,
            Inner::Expression(e) => e.kind(),
            Inner::List(items) => items.first().map_or(Kind::Constant, Slot::kind),
        }
    }

    /// Whether the document gives the value, rather than the default.
    pub fn is_set(&self) -> bool {
        self.set
    }

    /// The value, if it is the same everywhere.
    #[inline]
    pub(crate) fn as_constant(&self) -> Option<&Value<'static>> {
        match &self.value {
            Inner::Constant(v) => Some(v),
            _ => None,
        }
    }

    /// The expression of a camera, source or composite value.
    #[inline]
    pub fn as_expression(&self) -> Option<&PropertyExpression> {
        match &self.value {
            Inner::Expression(e) => Some(e),
            _ => None,
        }
    }

    /// The items of a list; a single value is a list of one.
    #[inline]
    pub fn items(&self) -> &[Slot] {
        match &self.value {
            Inner::List(items) => items,
            _ => std::slice::from_ref(self),
        }
    }

    /// Replaces the value with a constant, as a transition overlays one.
    pub fn set_constant(&mut self, value: Value<'static>) {
        self.value = Inner::Constant(value);
    }
}

/// A property read as type `T`, with the default it falls back to.
#[derive(Clone, Debug)]
pub struct PropertyRef<'a, T> {
    slot: &'a Slot,
    default: Fallback<T>,
}

/// A property's default: the table's, looked up and read as `T` only
/// when a value falls back to it, or a typed property's own.
#[derive(Clone, Debug)]
enum Fallback<T> {
    Table(crate::PropertyId),
    Typed(T),
}

impl<T: PropertyValue> Fallback<T> {
    fn get(&self) -> T {
        match self {
            Self::Table(id) => T::from_value(id.default_value())
                .or(T::UNSET)
                .expect("table defaults fit their type"),
            Self::Typed(value) => value.clone(),
        }
    }
}

impl<'a, T: PropertyValue> PropertyRef<'a, T> {
    /// A slot read with a typed default.
    pub fn new(slot: &'a Slot, default: T) -> Self {
        Self {
            slot,
            default: Fallback::Typed(default),
        }
    }

    /// Property `id`'s slot, read with the table's default.
    pub fn of(slot: &'a Slot, id: crate::PropertyId) -> Self {
        Self {
            slot,
            default: Fallback::Table(id),
        }
    }

    pub fn slot(&self) -> &'a Slot {
        self.slot
    }

    /// The property it reads, unless it was read with a typed default.
    pub fn id(&self) -> Option<crate::PropertyId> {
        match self.default {
            Fallback::Table(id) => Some(id),
            Fallback::Typed(_) => None,
        }
    }

    pub fn kind(&self) -> Kind {
        self.slot.kind()
    }

    pub fn is_set(&self) -> bool {
        self.slot.is_set()
    }

    /// The value, if it is the same everywhere.
    pub fn constant(&self) -> Option<T> {
        self.slot.as_constant().map(|v| self.fit(Some(v)))
    }

    /// The expression of a camera, source or composite value, for its zoom
    /// curve and `interpolation_factor`.
    pub fn as_expression(&self) -> Option<&'a PropertyExpression> {
        self.slot.as_expression()
    }

    /// GL JS's fallback: `null`, `NaN`, an evaluation error or a value of
    /// the wrong shape yields the default.
    fn fit(&self, value: Option<&Value<'_>>) -> T {
        match value {
            None | Some(Value::Null) => self.default.get(),
            Some(Value::Number(n)) if n.is_nan() => self.default.get(),
            Some(v) => T::from_value(v).unwrap_or_else(|| self.default.get()),
        }
    }

    /// The value at `zoom` without a feature: exact for constant and camera
    /// values. A source or composite value reads no properties, so it
    /// usually falls back to the default.
    pub fn evaluate(&self, zoom: f64) -> T {
        self.evaluate_state(zoom, &[], None)
    }

    /// The value at `zoom` without a feature, under `camera`: exact for
    /// constant and camera values.
    pub fn evaluate_camera(&self, zoom: f64, camera: orb_expr::CameraInputs) -> T {
        let slot = self.slot.items().first().unwrap_or(self.slot);
        match &slot.value {
            Inner::Constant(v) => self.fit(Some(v)),
            Inner::Expression(expression) => {
                let context = orb_expr::EvaluationContext {
                    zoom,
                    pitch: camera.pitch,
                    bearing: camera.bearing,
                    meters_per_pixel: camera.meters_per_pixel,
                    ..Default::default()
                };
                let value = expression.expression().evaluate_with(&context);
                self.fit(value.ok().as_ref())
            }
            Inner::List(_) => unreachable!("a list's items are values"),
        }
    }

    /// The value at `zoom` for a feature with `state`: exact for constant,
    /// camera and state-only values, and with the feature's captured `data`
    /// for values that read both state and data.
    pub fn evaluate_state(
        &self,
        zoom: f64,
        state: &[(String, Value<'static>)],
        data: Option<&orb_expr::FeatureData>,
    ) -> T {
        let slot = self.slot.items().first().unwrap_or(self.slot);
        match &slot.value {
            Inner::Constant(v) => self.fit(Some(v)),
            Inner::Expression(expression) => {
                let context = orb_expr::EvaluationContext {
                    zoom,
                    feature_state: state,
                    ..Default::default()
                };
                let expression = expression.expression();
                let value = match data {
                    Some(data) => expression.evaluate_data(&context, data),
                    None => expression.evaluate_with(&context),
                };
                self.fit(value.ok().as_ref())
            }
            Inner::List(_) => unreachable!("a list's items are values"),
        }
    }

    /// Whether the value reads `feature-state`, but no feature properties:
    /// a `StateLookup`.
    pub fn is_state_lookup(&self) -> bool {
        self.as_expression()
            .is_some_and(|e| e.reads_feature_state() && !e.reads_feature_properties())
    }

    /// An owned copy, to hold apart from its layer.
    pub fn to_owned(&self) -> Property<T> {
        Property::from_slot(self.slot.clone(), self.default.get())
    }

    /// Binds the property to a tile layer, to evaluate it per feature.
    pub fn bind(&self, layer: &'a orb_tile::Layer) -> BoundProperty<'a, T> {
        BoundProperty {
            constant: self.constant(),
            bound: self.as_expression().map(|e| e.expression().bind(layer)),
            default: self.default.get(),
        }
    }

    /// Binds the property to a tile layer with the style's images, which
    /// `image` expressions find available.
    pub fn bind_with_images(
        &self,
        layer: &'a orb_tile::Layer,
        images: &'a [String],
    ) -> BoundProperty<'a, T> {
        BoundProperty {
            constant: self.constant(),
            bound: self
                .as_expression()
                .map(|e| e.expression().bind(layer).with_images(images)),
            default: self.default.get(),
        }
    }
}

impl<T: PropertyValue + Into<f64>> PropertyRef<'_, T> {
    /// The largest value at `zoom` over every feature, the default that
    /// a `null` or failed evaluation falls back to included: `None` when
    /// the expression sets no bound (a bare `get`).
    pub fn max_value(&self, zoom: f64) -> Option<f64> {
        match self.kind() {
            Kind::Constant | Kind::Camera => Some(self.evaluate(zoom).into()),
            Kind::Source | Kind::Composite => {
                let expression = self.as_expression()?.expression();
                let max = expression.max_number(zoom)?;
                Some(max.max(self.default.get().into()))
            }
        }
    }
}

/// A typed property held outside a layer (the light, the sky, the
/// terrain): a slot and its default.
#[derive(Clone, Debug)]
pub struct Property<T> {
    slot: Slot,
    default: T,
}

impl<T: PropertyValue> Property<T> {
    /// The same value everywhere (held as the default of an unset slot).
    pub fn constant(value: T) -> Self {
        Self {
            slot: Slot::constant(Value::Null, false),
            default: value,
        }
    }

    pub fn from_slot(slot: Slot, default: T) -> Self {
        Self { slot, default }
    }

    pub fn get(&self) -> PropertyRef<'_, T> {
        PropertyRef::new(&self.slot, self.default.clone())
    }

    pub fn kind(&self) -> Kind {
        self.slot.kind()
    }

    pub fn evaluate(&self, zoom: f64) -> T {
        self.get().evaluate(zoom)
    }

    pub fn evaluate_camera(&self, zoom: f64, camera: orb_expr::CameraInputs) -> T {
        self.get().evaluate_camera(zoom, camera)
    }

    pub fn as_expression(&self) -> Option<&PropertyExpression> {
        self.slot.as_expression()
    }
}

/// GL JS's fallback: `null`, `NaN`, an evaluation error or a value of the
/// wrong shape yields the default.
pub fn fit<T: PropertyValue>(value: Option<Value<'_>>, default: &T) -> T {
    match value {
        None | Some(Value::Null) => default.clone(),
        Some(Value::Number(n)) if n.is_nan() => default.clone(),
        Some(v) => T::from_value(&v).unwrap_or_else(|| default.clone()),
    }
}

/// A property bound to one tile layer.
#[derive(Debug)]
pub struct BoundProperty<'a, T> {
    constant: Option<T>,
    bound: Option<Bound<'a>>,
    default: T,
}

impl<T> BoundProperty<'_, T> {
    /// The canonical tile the layer is in, for expressions that read a
    /// feature's geometry (`within`, `distance`).
    pub fn with_tile(mut self, tile: Option<orb_expr::CanonicalTile>) -> Self {
        self.bound = self.bound.map(|b| b.with_tile(tile));
        self
    }
}

impl<T: PropertyValue> BoundProperty<'_, T> {
    /// The value for feature `row` at `zoom`.
    pub fn evaluate(&self, zoom: f64, row: usize) -> T {
        match (&self.constant, &self.bound) {
            (Some(v), _) => v.clone(),
            (None, Some(bound)) => fit(bound.evaluate(zoom, row).ok(), &self.default),
            (None, None) => self.default.clone(),
        }
    }
}
