//! Expression types and runtime values, after the style-spec's `types.ts` and
//! `values.ts`.

use std::borrow::Cow;
use std::fmt;

use crate::color::Color;
use crate::js;
use crate::json::{Json, JsonValue};

/// The static type of an expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Null,
    Error,
    Collator,
    Formatted,
    ProjectionDefinition,
    Padding,
    NumberArray,
    ColorArray,
    VariableAnchorOffsetCollection,

    Number,
    String,
    Boolean,
    Color,
    Object,
    /// Any value; checked at run time where a narrower type is needed.
    Value,
    /// An image name, as `*-pattern` properties take.
    ResolvedImage,
    /// An array with an item type and an optional fixed length.
    Array(Box<Type>, Option<u32>),
}

impl Type {
    pub fn array(item: Type, len: Option<u32>) -> Self {
        Self::Array(Box::new(item), len)
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Collator => "collator",
            Self::Formatted => "formatted",
            Self::ProjectionDefinition => "projectionDefinition",
            Self::Padding => "padding",
            Self::NumberArray => "numberArray",
            Self::ColorArray => "colorArray",
            Self::VariableAnchorOffsetCollection => "variableAnchorOffsetCollection",

            Self::Null => "null",
            Self::Number => "number",
            Self::String => "string",
            Self::Boolean => "boolean",
            Self::Color => "color",
            Self::Object => "object",
            Self::Value => "value",
            Self::ResolvedImage => "resolvedImage",
            Self::Array(..) => "array",
        }
    }
}

/// `typeToString`.
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Array(item, Some(n)) => write!(f, "array<{item}, {n}>"),
            Self::Array(item, None) if **item == Self::Value => f.write_str("array"),
            Self::Array(item, None) => write!(f, "array<{item}>"),
            other => f.write_str(other.kind()),
        }
    }
}

/// `checkSubtype`: `Ok` if `t` may be used where `expected` is wanted.
pub(crate) fn check_subtype(expected: &Type, t: &Type) -> Result<(), String> {
    let ok = match (expected, t) {
        (_, Type::Error) => true,
        (Type::Array(ei, en), Type::Array(ti, tn)) => {
            ((*tn == Some(0) && **ti == Type::Value) || check_subtype(ei, ti).is_ok())
                && (en.is_none() || en == tn)
        }
        (Type::Array(..), _) => false,
        // Every type we model is a member of `value`.
        (Type::Value, Type::Collator) => false,
        (Type::Value, _) => true,
        (e, t) => e.kind() == t.kind(),
    };
    if ok {
        Ok(())
    } else {
        Err(format!("Expected {expected} but found {t} instead."))
    }
}

/// A runtime value.
///
/// Strings, arrays and objects borrow from the expression's literals or the
/// tile's columns where they can, so evaluating a filter or a `match` on a
/// string property does not allocate.
#[derive(Clone, Debug, PartialEq)]
pub enum Value<'a> {
    Null,
    Bool(bool),
    Number(f64),
    String(Cow<'a, str>),
    Color(Color),
    /// An image name, from a coercion to `resolvedImage`.
    Image {
        name: Cow<'a, str>,
        available: bool,
    },
    Collator(crate::Collator),
    TypedArray(Type, Cow<'a, [Value<'static>]>),
    Formatted(Cow<'a, [FormattedSection]>),
    Array(Cow<'a, [Value<'static>]>),
    /// Members in insertion order, as a JavaScript object keeps them.
    Object(Cow<'a, [(String, Value<'static>)]>),
}

impl<'a> Value<'a> {
    /// The value's type, as `typeOf`. Array item types follow GL JS: a
    /// common item type when every item has the same non-array type,
    /// `value` otherwise.
    pub(crate) fn type_of(&self) -> Type {
        match self {
            Self::Null => Type::Null,
            Self::Bool(_) => Type::Boolean,
            Self::Number(_) => Type::Number,
            Self::String(_) => Type::String,
            Self::Color(_) => Type::Color,
            Self::Image { .. } => Type::ResolvedImage,
            Self::Collator(_) => Type::Collator,
            Self::Formatted(_) => Type::Formatted,
            Self::TypedArray(ty, _) => ty.clone(),
            Self::Object(_) => Type::Object,
            Self::Array(items) => {
                let mut item: Option<Type> = None;
                for v in items.iter() {
                    let t = v.type_of();
                    match &item {
                        None => item = Some(t),
                        // GL JS compares type objects by identity, and every
                        // array type is a fresh object.
                        Some(prev) if *prev == t && !matches!(t, Type::Array(..)) => {}
                        Some(_) => {
                            item = Some(Type::Value);
                            break;
                        }
                    }
                }
                Type::array(item.unwrap_or(Type::Value), Some(items.len() as u32))
            }
        }
    }

    /// `typeOf(v).kind`, without building array types.
    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Color(_) => "color",
            Self::Image { .. } => "resolvedImage",
            Self::Collator(_) => "collator",
            Self::Formatted(_) => "formatted",
            Self::TypedArray(ty, _) => ty.kind(),
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_color(&self) -> Option<Color> {
        match self {
            Self::Color(c) => Some(*c),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value<'static>]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The same value, borrowing what `self` owns.
    pub(crate) fn reborrow(&self) -> Value<'_> {
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(*b),
            Self::Number(n) => Value::Number(*n),
            Self::Color(c) => Value::Color(*c),
            Self::String(s) => Value::String(Cow::Borrowed(s)),
            Self::Image { name, available } => Value::Image {
                name: Cow::Borrowed(name),
                available: *available,
            },
            Self::Collator(c) => Value::Collator(c.clone()),
            Self::Formatted(s) => Value::Formatted(Cow::Borrowed(s)),
            Self::TypedArray(ty, a) => Value::TypedArray(ty.clone(), Cow::Borrowed(a)),
            Self::Array(a) => Value::Array(Cow::Borrowed(a)),
            Self::Object(o) => Value::Object(Cow::Borrowed(o)),
        }
    }

    pub fn into_owned(self) -> Value<'static> {
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(b),
            Self::Number(n) => Value::Number(n),
            Self::Color(c) => Value::Color(c),
            Self::String(s) => Value::String(Cow::Owned(s.into_owned())),
            Self::Image { name, available } => Value::Image {
                name: Cow::Owned(name.into_owned()),
                available,
            },
            Self::Collator(c) => Value::Collator(c),
            Self::Formatted(s) => Value::Formatted(Cow::Owned(s.into_owned())),
            Self::TypedArray(ty, a) => Value::TypedArray(ty, Cow::Owned(a.into_owned())),
            Self::Array(a) => Value::Array(Cow::Owned(a.into_owned())),
            Self::Object(o) => Value::Object(Cow::Owned(o.into_owned())),
        }
    }

    /// The value of a JSON literal.
    pub fn from_json(json: &Json) -> Value<'static> {
        match &json.value {
            JsonValue::Null => Value::Null,
            JsonValue::Bool(b) => Value::Bool(*b),
            JsonValue::Number(n) => Value::Number(*n),
            JsonValue::String(s) => Value::String(Cow::Owned(s.clone())),
            JsonValue::Array(items) => {
                Value::Array(Cow::Owned(items.iter().map(Value::from_json).collect()))
            }
            JsonValue::Object(members) => Value::Object(Cow::Owned(
                members
                    .iter()
                    .map(|m| (m.key.clone(), Value::from_json(&m.value)))
                    .collect(),
            )),
        }
    }

    /// JavaScript truthiness, as `Boolean(v)`.
    pub(crate) fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(b) => *b,
            Self::Number(n) => *n != 0.0 && !n.is_nan(),
            Self::String(s) => !s.is_empty(),
            _ => true,
        }
    }

    /// Strict equality (`===`). Colours, arrays and objects are compared by
    /// reference in GL JS, so two of them are never equal here.
    pub(crate) fn strict_eq(&self, other: &Value<'_>) -> bool {
        match (self, other) {
            (Self::Null, Value::Null) => true,
            (Self::Bool(a), Value::Bool(b)) => a == b,
            (Self::Number(a), Value::Number(b)) => a == b,
            (Self::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }

    /// `Number(v)`.
    pub(crate) fn to_number(&self) -> f64 {
        match self {
            Self::Null => 0.0,
            Self::Bool(b) => f64::from(u8::from(*b)),
            Self::Number(n) => *n,
            Self::String(s) => js::string_to_number(s),
            Self::Array(_) => js::string_to_number(&self.to_js_string()),
            _ => f64::NAN,
        }
    }

    /// JavaScript's `String(v)`, as `Array.prototype.join` applies it to items.
    pub(crate) fn to_js_string(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Array(items) => items
                .iter()
                .map(|v| match v {
                    Value::Null => String::new(),
                    v => v.to_js_string(),
                })
                .collect::<Vec<_>>()
                .join(","),
            Self::Object(_) => "[object Object]".into(),
            other => other.to_string(),
        }
    }

    /// `JSON.stringify(v)`.
    pub fn to_json_string(&self) -> String {
        let mut s = String::new();
        self.write_json(&mut s).expect("write to String");
        s
    }

    fn write_json(&self, f: &mut impl fmt::Write) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Number(n) => js::write_json_number(f, *n),
            Self::String(s) => js::write_json_string(f, s),
            Self::Image { name, available } => {
                f.write_str("{\"name\":")?;
                js::write_json_string(f, name)?;
                write!(f, ",\"available\":{available}}}")
            }
            Self::Collator(_) => f.write_str("{}"),
            Self::TypedArray(_, a) => {
                f.write_str("{\"values\":")?;
                Value::Array(Cow::Borrowed(a)).write_json(f)?;
                f.write_str("}")
            }
            Self::Formatted(sections) => {
                f.write_str("{\"sections\":[")?;
                for (i, section) in sections.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    section.as_value().write_json(f)?;
                }
                f.write_str("]}")
            }
            Self::Color(c) => {
                let [r, g, b, a] = c.premultiplied();
                f.write_str("{\"r\":")?;
                js::write_json_number(f, f64::from(r))?;
                f.write_str(",\"g\":")?;
                js::write_json_number(f, f64::from(g))?;
                f.write_str(",\"b\":")?;
                js::write_json_number(f, f64::from(b))?;
                f.write_str(",\"a\":")?;
                js::write_json_number(f, f64::from(a))?;
                f.write_str("}")
            }
            Self::Array(items) => {
                f.write_str("[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    v.write_json(f)?;
                }
                f.write_str("]")
            }
            Self::Object(members) => {
                f.write_str("{")?;
                for (i, (k, v)) in members.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    js::write_json_string(f, k)?;
                    f.write_str(":")?;
                    v.write_json(f)?;
                }
                f.write_str("}")
            }
        }
    }
}

/// `valueToString`, which `to-string` and `concat` use: `null` is empty,
/// colours are `rgba(…)`, arrays and objects are JSON.
impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => Ok(()),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Number(n) => js::write_number(f, *n),
            Self::String(s) | Self::Image { name: s, .. } => f.write_str(s),
            Self::Collator(_) => f.write_str("[object Object]"),
            Self::TypedArray(_, a) => Value::Array(Cow::Borrowed(a)).write_json(f),
            Self::Formatted(sections) => {
                for s in sections.iter() {
                    f.write_str(&s.text)?;
                }
                Ok(())
            }
            Self::Color(c) => write!(f, "{c}"),
            Self::Array(_) | Self::Object(_) => self.write_json(f),
        }
    }
}

/// A value crosses between workers as its JSON text.
impl serde::Serialize for Value<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_json_string())
    }
}

impl<'de> serde::Deserialize<'de> for Value<'static> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        crate::json::parse(&text)
            .map(|json| Value::from_json(&json))
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    fn type_of(text: &str) -> String {
        Value::from_json(&json::parse(text).unwrap())
            .type_of()
            .to_string()
    }

    #[test]
    fn types_of_arrays_follow_gl_js() {
        assert_eq!(type_of("[1, 2]"), "array<number, 2>");
        assert_eq!(type_of("[1, \"a\"]"), "array<value, 2>");
        assert_eq!(type_of("[]"), "array<value, 0>");
        assert_eq!(type_of("[[1]]"), "array<array<number, 1>, 1>");
        assert_eq!(type_of("[[1], [2]]"), "array<value, 2>");
        assert_eq!(type_of("[{}, {}]"), "array<object, 2>");
    }

    #[test]
    fn subtyping() {
        let num2 = Type::array(Type::Number, Some(2));
        assert!(check_subtype(&Type::array(Type::Number, None), &num2).is_ok());
        assert!(check_subtype(&num2, &Type::array(Type::Number, None)).is_err());
        assert!(check_subtype(&num2, &Type::array(Type::Value, Some(0))).is_err());
        assert!(
            check_subtype(
                &Type::array(Type::String, None),
                &Type::array(Type::Value, Some(0))
            )
            .is_ok()
        );
        assert!(check_subtype(&Type::Value, &num2).is_ok());
        assert_eq!(
            check_subtype(&Type::String, &Type::Number).unwrap_err(),
            "Expected string but found number instead."
        );
    }

    #[test]
    fn strings_follow_value_to_string() {
        let v = Value::from_json(&json::parse(r#"[1, null, "a", {"b": true}]"#).unwrap());
        assert_eq!(v.to_string(), r#"[1,null,"a",{"b":true}]"#);
        assert_eq!(v.to_js_string(), "1,,a,[object Object]");
        assert_eq!(Value::Null.to_string(), "");
        assert_eq!(Value::Number(1e21).to_string(), "1e+21");
    }
}

/// One styled text or image run in a `format` value.
#[derive(Clone, Debug, PartialEq)]
pub struct FormattedSection {
    pub text: String,
    pub image: Option<(String, bool)>,
    pub scale: Option<f64>,
    pub font_stack: Option<String>,
    pub text_color: Option<Color>,
    pub vertical_align: Option<String>,
}
impl FormattedSection {
    /// A text section without overrides, as a plain string formats.
    pub fn plain(text: &str) -> Self {
        Self::text(text.to_string())
    }

    pub(crate) fn text(text: String) -> Self {
        Self {
            text,
            image: None,
            scale: None,
            font_stack: None,
            text_color: None,
            vertical_align: None,
        }
    }
    pub(crate) fn as_value(&self) -> Value<'static> {
        let string = |s: &Option<String>| {
            s.as_ref()
                .map_or(Value::Null, |s| Value::String(s.clone().into()))
        };
        Value::Object(
            vec![
                ("text".into(), Value::String(self.text.clone().into())),
                (
                    "image".into(),
                    self.image
                        .as_ref()
                        .map_or(Value::Null, |(n, a)| Value::Image {
                            name: n.clone().into(),
                            available: *a,
                        }),
                ),
                (
                    "scale".into(),
                    self.scale.map_or(Value::Null, Value::Number),
                ),
                ("fontStack".into(), string(&self.font_stack)),
                (
                    "textColor".into(),
                    self.text_color.map_or(Value::Null, Value::Color),
                ),
                ("verticalAlign".into(), string(&self.vertical_align)),
            ]
            .into(),
        )
    }
}
