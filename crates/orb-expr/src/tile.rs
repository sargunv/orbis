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

struct Row<'s, 'a> {
    layer: &'a Layer,
    columns: &'s [Option<&'a Column>],
    row: usize,
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
