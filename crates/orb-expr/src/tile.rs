//! Evaluation against the rows of an `orb_tile::Layer`.

use std::borrow::Cow;

use orb_tile::{Column, GeometryType, Layer, ValueRef};

use crate::eval::{EvalError, Evaluator, Feature};
use crate::ir::{Ir, KeyId};
use crate::value::Value;

/// An expression bound to one tile layer: its property keys are resolved to
/// columns once, so evaluating a row does no string lookups.
#[derive(Debug)]
pub struct Bound<'a> {
    ir: &'a Ir,
    layer: &'a Layer,
    columns: Vec<Option<&'a Column>>,
    images: &'a [String],
}

impl<'a> Bound<'a> {
    pub(crate) fn new(ir: &'a Ir, layer: &'a Layer) -> Self {
        let columns = ir.keys.iter().map(|k| layer.property(k)).collect();
        Self {
            ir,
            layer,
            columns,
            images: &[],
        }
    }

    /// The images `image` expressions find available when evaluated with
    /// [`evaluate`](Self::evaluate); none by default.
    pub fn with_images(mut self, images: &'a [String]) -> Self {
        self.images = images;
        self
    }

    /// Evaluates the expression for feature `row` at `zoom`.
    pub fn evaluate(&self, zoom: f64, row: usize) -> Result<Value<'a>, EvalError> {
        self.evaluate_with(
            &crate::EvaluationContext {
                zoom,
                available_images: self.images,
                ..crate::EvaluationContext::default()
            },
            row,
        )
    }

    /// Evaluates a row with explicit state, geometry, locale and globals.
    /// Literal property keys still address the columns resolved at bind time.
    pub fn evaluate_with(
        &self,
        context: &crate::EvaluationContext<'a>,
        row: usize,
    ) -> Result<Value<'a>, EvalError> {
        Evaluator {
            context,
            ir: self.ir,
            feature: Some(&Row {
                layer: self.layer,
                columns: &self.columns,
                row,
            }),
        }
        .eval_root()
    }
}

/// What paint expressions read of one tile feature, held apart from its
/// tile: the scene re-evaluates paint that reads both feature state and
/// feature data from it when the feature's state changes.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureData {
    pub id: Value<'static>,
    pub geometry_type: Option<GeometryType>,
    pub properties: Vec<(String, Value<'static>)>,
}

impl FeatureData {
    /// Captures what `expressions` read of feature `row` of `layer`: the
    /// properties under their literal keys, or all of them when one reads
    /// a computed key or `properties`.
    pub fn capture<'e>(
        layer: &Layer,
        row: usize,
        expressions: impl IntoIterator<Item = &'e crate::Expression> + Clone,
    ) -> Self {
        let dynamic = expressions.clone().into_iter().any(|e| {
            e.ir.nodes.iter().any(|n| {
                matches!(
                    n.op,
                    crate::ir::Op::Properties
                        | crate::ir::Op::GetDynamic(_)
                        | crate::ir::Op::HasDynamic(_)
                )
            })
        });
        let properties = if dynamic {
            layer
                .properties()
                .filter_map(|(k, c)| Some((k.to_string(), value(c.get(row)?).into_owned())))
                .collect()
        } else {
            let mut keys: Vec<&str> = expressions
                .into_iter()
                .flat_map(|e| e.ir.keys.iter().map(|k| &**k))
                .collect();
            keys.sort_unstable();
            keys.dedup();
            keys.into_iter()
                .filter_map(|k| {
                    Some((
                        k.to_string(),
                        value(layer.property(k)?.get(row)?).into_owned(),
                    ))
                })
                .collect()
        };
        Self {
            id: layer
                .ids()
                .get(row)
                .map_or(Value::Null, |v| value(v).into_owned()),
            geometry_type: Some(layer.geometry_type(row)),
            properties,
        }
    }

    /// The data as JSON text, `[id, geometry type, properties]`, for
    /// messages between workers.
    pub fn to_json_string(&self) -> String {
        Value::Array(Cow::Owned(vec![
            self.id.clone(),
            self.geometry_type.map_or(Value::Null, |t| {
                Value::String(Cow::Borrowed(geometry_type(t)))
            }),
            Value::Object(Cow::Owned(self.properties.clone())),
        ]))
        .to_json_string()
    }

    /// Reads [`to_json_string`](Self::to_json_string)'s text.
    pub(crate) fn from_json_str(text: &str) -> Option<Self> {
        let Value::Array(items) = Value::from_json(&crate::json::parse(text).ok()?) else {
            return None;
        };
        let [id, ty, Value::Object(properties)] =
            <[Value<'static>; 3]>::try_from(items.into_owned()).ok()?
        else {
            return None;
        };
        Some(Self {
            id,
            geometry_type: match ty.as_str() {
                Some("Point") => Some(GeometryType::Point),
                Some("LineString") => Some(GeometryType::LineString),
                Some("Polygon") => Some(GeometryType::Polygon),
                _ => None,
            },
            properties: properties.into_owned(),
        })
    }

    /// Fills in keys `other` has that this lacks: captures of one feature
    /// for several nodes.
    pub fn merge(&mut self, other: &Self) {
        for (k, v) in &other.properties {
            if !self.properties.iter().any(|(key, _)| key == k) {
                self.properties.push((k.clone(), v.clone()));
            }
        }
    }
}

/// Captured data crosses between workers as its JSON text.
impl serde::Serialize for FeatureData {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_json_string())
    }
}

impl<'de> serde::Deserialize<'de> for FeatureData {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Self::from_json_str(&text).ok_or_else(|| serde::de::Error::custom("malformed feature data"))
    }
}

impl crate::Expression {
    /// Evaluates for a captured feature.
    pub fn evaluate_data<'a>(
        &'a self,
        context: &crate::EvaluationContext<'a>,
        data: &'a FeatureData,
    ) -> Result<Value<'a>, EvalError> {
        Evaluator {
            context,
            ir: &self.ir,
            feature: Some(&data),
        }
        .eval_root()
    }
}

impl<'a> Feature<'a> for &'a FeatureData {
    fn get(&self, _: KeyId, name: &str) -> Option<Value<'a>> {
        self.get_by_name(name)
    }

    fn get_by_name(&self, name: &str) -> Option<Value<'a>> {
        self.properties
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
    }

    fn properties(&self) -> Vec<(String, Value<'static>)> {
        self.properties.clone()
    }

    fn id(&self) -> Value<'a> {
        self.id.clone()
    }

    fn geometry_type(&self) -> Option<&'a str> {
        self.geometry_type.map(geometry_type)
    }
}

struct Row<'s, 'a> {
    layer: &'a Layer,
    columns: &'s [Option<&'a Column>],
    row: usize,
}

impl Value<'_> {
    /// A tile property's value, as expressions read it.
    pub fn from_tile(v: ValueRef<'_>) -> Value<'_> {
        value(v)
    }

    /// The value as a tile property: numbers, strings and booleans as
    /// themselves, other values as JSON; `null` is no property.
    pub fn to_tile(&self) -> Option<orb_tile::Value> {
        Some(match self {
            Value::Null => return None,
            Value::Bool(b) => orb_tile::Value::Bool(*b),
            Value::Number(n) => orb_tile::Value::F64(*n),
            Value::String(s) => orb_tile::Value::Str(s.to_string()),
            other => orb_tile::Value::Json(other.to_json_string()),
        })
    }
}

pub(crate) fn value(v: ValueRef<'_>) -> Value<'_> {
    match v {
        ValueRef::Bool(b) => Value::Bool(b),
        // GL JS reads tile integers as JavaScript numbers.
        ValueRef::I64(n) => Value::Number(n as f64),
        ValueRef::U64(n) => Value::Number(n as f64),
        ValueRef::F64(n) => Value::Number(n),
        ValueRef::Str(s) => Value::String(Cow::Borrowed(s)),
        ValueRef::Json(s) => crate::json::parse(s).map_or(Value::Null, |j| Value::from_json(&j)),
    }
}

impl<'a> Feature<'a> for Row<'_, 'a> {
    fn get(&self, key: KeyId, _: &str) -> Option<Value<'a>> {
        self.columns[key as usize]
            .and_then(|c| c.get(self.row))
            .map(value)
    }

    fn get_by_name(&self, name: &str) -> Option<Value<'a>> {
        self.layer.property(name)?.get(self.row).map(value)
    }

    fn properties(&self) -> Vec<(String, Value<'static>)> {
        self.layer
            .properties()
            .filter_map(|(k, c)| Some((k.to_string(), value(c.get(self.row)?).into_owned())))
            .collect()
    }

    fn id(&self) -> Value<'a> {
        self.layer.ids().get(self.row).map_or(Value::Null, value)
    }

    fn geometry_type(&self) -> Option<&'a str> {
        Some(geometry_type(self.layer.geometry_type(self.row)))
    }
}

/// The `geometry-type` of a tile feature.
pub(crate) fn geometry_type(t: GeometryType) -> &'static str {
    match t {
        GeometryType::Point => "Point",
        GeometryType::LineString => "LineString",
        GeometryType::Polygon => "Polygon",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GeoJSON's nested values read back as arrays and objects.
    #[test]
    fn json_properties_are_arrays_and_objects() {
        assert_eq!(
            value(ValueRef::Json("[1,-1]")),
            Value::Array(Cow::Owned(vec![Value::Number(1.0), Value::Number(-1.0)]))
        );
        assert_eq!(
            value(ValueRef::Json(r#"{"a":"b"}"#)),
            Value::Object(Cow::Owned(vec![(
                "a".into(),
                Value::String(Cow::Borrowed("b"))
            )]))
        );
    }
}
