//! The complete pinned style-spec expression conformance suite. Native and
//! wasm tests share compilation, inputs, error paths and output comparisons.
//! See `tests/data/SOURCE.md` for the upstream version and regeneration.
//! With the `conformance` feature, [`check`] runs one fixture for
//! `orb-fixtures`, which runs MapLibre Native's copy of the suite.

use crate::eval::{Evaluator, Feature};
use crate::ir::KeyId;
use crate::json::{self, Json, JsonValue, Member};
use crate::legacy::lower_function;
use crate::parse::{Annotation, Parser};
use crate::property::{PropertyExpression, PropertySpec, PropertyType};
use crate::value::{Type, Value};
use crate::{Expression, Kind};

#[cfg(test)]
pub(crate) const FIXTURES: &str = include_str!("../tests/data/expression-fixtures.json");
const ROOT_KEY: &str = "layers[0].paint.some-property";

pub(crate) struct JsonFeature<'a> {
    properties: &'a [Member],
    id: Option<&'a Json>,
    geometry_type: Option<&'a str>,
}

impl<'a> Feature<'a> for JsonFeature<'a> {
    fn get(&self, _: KeyId, name: &str) -> Option<Value<'a>> {
        self.get_by_name(name)
    }

    fn get_by_name(&self, name: &str) -> Option<Value<'a>> {
        self.properties
            .iter()
            .rev()
            .find(|m| m.key == name)
            .map(|m| Value::from_json(&m.value))
    }

    fn properties(&self) -> Vec<(String, Value<'static>)> {
        self.properties
            .iter()
            .map(|m| (m.key.clone(), Value::from_json(&m.value)))
            .collect()
    }

    fn id(&self) -> Value<'a> {
        self.id.map_or(Value::Null, Value::from_json)
    }

    fn geometry_type(&self) -> Option<&'a str> {
        self.geometry_type
    }
}

/// GL JS's `getCompletePropertySpec`, then the property's parsing spec.
pub(crate) fn fixture_spec(fixture: &Json) -> (PropertySpec, Option<Type>) {
    let spec = fixture.get("propertySpec");
    let field = |k: &str| spec.and_then(|s| s.get(k));
    let item = |name: Option<&str>| match name {
        Some("number") => Type::Number,
        Some("string") => Type::String,
        Some("boolean") => Type::Boolean,
        _ => Type::Value,
    };
    let ty = match field("type").and_then(Json::as_str) {
        Some("number") => Some(PropertyType::Number),
        Some("string") => Some(PropertyType::String),
        Some("boolean") => Some(PropertyType::Boolean),
        Some("color") => Some(PropertyType::Color),
        Some("resolvedImage") => Some(PropertyType::ResolvedImage),
        Some("formatted") => Some(PropertyType::Formatted),
        Some("projectionDefinition") => Some(PropertyType::ProjectionDefinition),
        Some("padding") => Some(PropertyType::Padding),
        Some("numberArray") => Some(PropertyType::NumberArray),
        Some("colorArray") => Some(PropertyType::ColorArray),
        Some("variableAnchorOffsetCollection") => {
            Some(PropertyType::VariableAnchorOffsetCollection)
        }

        Some("enum") => {
            let values: Vec<&'static str> = field("values")
                .and_then(Json::as_object)
                .unwrap_or_default()
                .iter()
                .map(|m| &*Box::leak(m.key.clone().into_boxed_str()))
                .collect();
            Some(PropertyType::Enum(Box::leak(values.into_boxed_slice())))
        }
        Some("array") => Some(PropertyType::Array(
            item(field("value").and_then(Json::as_str)),
            field("length").and_then(Json::as_f64).map(|n| n as u32),
        )),
        // GL JS expects nothing of types it doesn't map, like `null`.
        _ => None,
    };
    let property_type = field("property-type")
        .and_then(Json::as_str)
        .unwrap_or("data-driven");
    let expression = field("expression");
    let parameters: Vec<&str> = match expression {
        Some(e) => e
            .get("parameters")
            .and_then(Json::as_array)
            .unwrap_or_default()
            .iter()
            .filter_map(Json::as_str)
            .collect(),
        None => vec!["zoom", "feature"],
    };
    let interpolated = match expression {
        Some(e) => e
            .get("interpolated")
            .and_then(Json::as_bool)
            .unwrap_or(false),
        None => true,
    };
    let expected = ty.as_ref().map(|ty| {
        PropertySpec {
            ty: ty.clone(),
            default: None,
            zoom: false,
            feature: false,
            state: false,
            camera: false,
            elevation: false,
            interpolated: false,
            tokens: false,
        }
        .expected_type()
    });
    let spec = PropertySpec {
        // Only legacy functions read the type without an expected type, and
        // their fixtures all declare one.
        ty: ty.unwrap_or(PropertyType::Number),
        default: field("default").map(|d| d.to_string().into()),
        zoom: parameters.contains(&"zoom"),
        feature: matches!(property_type, "data-driven" | "cross-faded-data-driven"),
        // GL JS evaluates state wherever data is allowed; only the style
        // validator rejects it in layout.
        state: matches!(property_type, "data-driven" | "cross-faded-data-driven"),
        camera: false,
        elevation: parameters.contains(&"elevation"),
        interpolated,
        tokens: field("tokens").and_then(Json::as_bool).unwrap_or(false),
    };
    (spec, expected)
}

/// `createPropertyExpression`, with `convertFunction` for legacy functions,
/// against a spec whose type may be absent.
pub(crate) fn compile(
    expression: &Json,
    spec: &PropertySpec,
    expected: Option<&Type>,
) -> Result<PropertyExpression, Vec<crate::Diagnostic>> {
    let annotation =
        (spec.ty == PropertyType::String && expected.is_some()).then_some(Annotation::Coerce);
    let mut parser = Parser::default();
    let root = if expression.as_object().is_some() {
        let lowered = lower_function(expression, spec)?;
        parser.parse(&lowered, "", expected, annotation)
    } else {
        parser.parse(expression, "", expected, annotation)
    };
    let Some(root) = root else {
        return Err(parser.errors);
    };
    PropertyExpression::classify(
        Expression {
            ir: parser.ir.compact(root),
        },
        spec,
    )
    .map_err(|message| {
        vec![crate::Diagnostic {
            path: String::new(),
            span: expression.span,
            message: message.into(),
        }]
    })
}

/// The reference truncates outputs to six significant figures
/// (`stripPrecision`), which can itself lose a unit to rounding, so a match
/// is within two units of the sixth.
fn numbers_match(a: f64, b: f64) -> bool {
    let magnitude = a.abs().max(b.abs());
    a == b || (a - b).abs() <= 2.0 * 10f64.powf(magnitude.log10().ceil() - 6.0)
}

/// Whose expectations a fixture holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reference {
    /// maplibre-style-spec's, which GL JS's harness checks: evaluation
    /// errors name the property and the failing subexpression's path.
    StyleSpec,
    /// MapLibre Native's copy (`metrics/integration/expression-tests`):
    /// evaluation errors are the bare message, and formatted sections
    /// predate `verticalAlign`. Native's runner also checks serialization
    /// and a round trip through it, which Orbis doesn't have.
    Native,
}

/// Compares an output with the fixture's, numbers to about six significant
/// figures as the reference runner does.
fn matches(
    actual: &Value<'_>,
    expected: &Json,
    color_as_array: bool,
    reference: Reference,
) -> bool {
    let matches = |a: &Value<'_>, e: &Json| matches(a, e, false, reference);
    match (actual, &expected.value) {
        (Value::Null, JsonValue::Null) => true,
        (Value::Bool(a), JsonValue::Bool(b)) => a == b,
        (Value::Number(a), JsonValue::Number(b)) => numbers_match(*a, *b),
        (Value::String(a), JsonValue::String(b)) => a == b,
        (Value::Color(c), JsonValue::Array(items)) if color_as_array => {
            items.len() == 4
                && c.premultiplied()
                    .iter()
                    .zip(items)
                    .all(|(a, b)| b.as_f64().is_some_and(|b| numbers_match(f64::from(*a), b)))
        }
        (Value::Array(a), JsonValue::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| matches(a, b))
        }
        (Value::Object(a), JsonValue::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(k, v)| {
                    b.iter()
                        .find(|m| m.key == *k)
                        .is_some_and(|m| matches(v, &m.value))
                })
        }
        (Value::Formatted(_), _) if reference == Reference::Native => {
            let text = actual
                .to_json_string()
                .replace(r#","verticalAlign":null"#, "");
            json::parse(&text).is_ok_and(|j| matches(&Value::from_json(&j), expected))
        }
        (Value::Image { .. } | Value::Formatted(_) | Value::TypedArray(..), _) => {
            json::parse(&actual.to_json_string())
                .is_ok_and(|j| matches(&Value::from_json(&j), expected))
        }
        (Value::Color(c), JsonValue::Object(_)) => json::parse(&Value::Color(*c).to_json_string())
            .is_ok_and(|j| matches(&Value::from_json(&j), expected)),
        _ => false,
    }
}

/// The VM's output for one fixture input, `[globals, feature]`.
pub(crate) fn evaluate(
    parsed: &PropertyExpression,
    fixture: &Json,
    input: &Json,
) -> Result<Value<'static>, crate::EvalError> {
    let input = input.as_array().expect("input pair");
    let globals = &input[0];
    let feature = &input[1];
    let canonical = globals.get("canonicalID").and_then(|j| {
        Some(crate::CanonicalTile {
            z: j.get("z")?.as_f64()? as u8,
            x: j.get("x")?.as_f64()? as u32,
            y: j.get("y")?.as_f64()? as u32,
        })
    });
    let geometry = canonical.and_then(|t| {
        feature
            .get("geometry")
            .and_then(|g| crate::geometry::fixture_geometry(g, t))
    });
    let global_state = state(
        fixture
            .get("globalState")
            .or_else(|| globals.get("globalState")),
    );
    let feature_state = state(feature.get("featureState"));
    let images = globals
        .get("availableImages")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(Json::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let num = |key| globals.get(key).and_then(Json::as_f64).unwrap_or(0.0);
    let context = crate::EvaluationContext {
        zoom: zoom(globals),
        line_progress: num("lineProgress"),
        heatmap_density: num("heatmapDensity"),
        elevation: num("elevation"),
        accumulated: globals
            .get("accumulated")
            .map_or(Value::Null, Value::from_json),
        global_state: &global_state,
        feature_state: &feature_state,
        available_images: &images,
        geometry: geometry.as_ref(),
        ..crate::EvaluationContext::default()
    };
    Evaluator {
        context: &context,
        ir: &parsed.expression().ir,
        feature: Some(&json_feature(feature)),
    }
    .eval_root()
    .map(Value::into_owned)
}

/// A fixture input's zoom, NaN if it has none.
pub(crate) fn zoom(globals: &Json) -> f64 {
    globals
        .get("zoom")
        .and_then(Json::as_f64)
        .unwrap_or(f64::NAN)
}

/// A fixture's state object as key-value pairs.
pub(crate) fn state(j: Option<&Json>) -> Vec<(String, Value<'static>)> {
    j.map(Value::from_json)
        .and_then(|v| {
            if let Value::Object(o) = v {
                Some(o.into_owned())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

pub(crate) fn json_feature(feature: &Json) -> JsonFeature<'_> {
    JsonFeature {
        properties: feature
            .get("properties")
            .and_then(Json::as_object)
            .unwrap_or_default(),
        id: feature.get("id"),
        geometry_type: feature
            .get("geometry")
            .and_then(|g| g.get("type"))
            .and_then(Json::as_str),
    }
}

#[cfg(test)]
fn run_fixture(name: &str, fixture: &Json) -> Result<(), String> {
    check(name, fixture, Reference::StyleSpec)
}

fn field<'a>(name: &str, json: &'a Json, key: &str) -> Result<&'a Json, String> {
    json.get(key).ok_or_else(|| format!("{name}: no {key}"))
}

/// Runs one fixture (a parsed `test.json`): compiles its expression for its
/// property spec, checks the compiled result or errors, then evaluates each
/// input and checks its output, as `reference`'s runner does. Why it
/// failed, if it did.
pub fn check(name: &str, fixture: &Json, reference: Reference) -> Result<(), String> {
    let (spec, expected_type) = fixture_spec(fixture);
    let expression = field(name, fixture, "expression")?;
    let expected = field(name, fixture, "expected")?;
    let compiled = field(name, expected, "compiled")?;
    let result = compile(expression, &spec, expected_type.as_ref());

    let want = compiled.get("result").and_then(Json::as_str);
    let parsed = match (&result, want) {
        (Err(errors), Some("error")) => {
            let want: Vec<(String, String)> = compiled
                .get("errors")
                .and_then(Json::as_array)
                .unwrap_or_default()
                .iter()
                .map(|e| {
                    let s = |k| {
                        e.get(k)
                            .and_then(Json::as_str)
                            .unwrap_or_default()
                            .to_string()
                    };
                    (s("key"), s("error"))
                })
                .collect();
            let got: Vec<(String, String)> = errors
                .iter()
                .map(|e| (e.path.clone(), e.message.clone()))
                .collect();
            return if got == want {
                Ok(())
            } else {
                Err(format!("{name}: errors\n  want {want:?}\n  got  {got:?}"))
            };
        }
        (Ok(parsed), Some("success")) => parsed,
        (Err(errors), _) => {
            let got: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            return Err(format!("{name}: expected success, got {got:?}"));
        }
        (Ok(_), _) => return Err(format!("{name}: expected errors, compiled")),
    };

    let kind = parsed.kind();
    let want_feature_constant = compiled.get("isFeatureConstant").and_then(Json::as_bool);
    let want_zoom_constant = compiled.get("isZoomConstant").and_then(Json::as_bool);
    let want_type = compiled.get("type").and_then(Json::as_str);
    let ty = parsed.expression().ty().to_string();
    let got = (
        Some(matches!(kind, Kind::Constant | Kind::Camera)),
        Some(matches!(kind, Kind::Constant | Kind::Source)),
        Some(ty.as_str()),
    );
    if got != (want_feature_constant, want_zoom_constant, want_type) {
        return Err(format!(
            "{name}: compiled\n  want {:?}\n  got  {got:?}",
            (want_feature_constant, want_zoom_constant, want_type)
        ));
    }

    let outputs = expected
        .get("outputs")
        .and_then(Json::as_array)
        .unwrap_or_default();
    let inputs = fixture
        .get("inputs")
        .and_then(Json::as_array)
        .unwrap_or_default();
    if inputs.len() != outputs.len() {
        return Err(format!(
            "{name}: {} inputs but {} outputs",
            inputs.len(),
            outputs.len()
        ));
    }
    let color = *parsed.expression().ty() == Type::Color;
    for (i, (input, want)) in inputs.iter().zip(outputs).enumerate() {
        let result = evaluate(parsed, fixture, input);
        let ok = match (&result, want.get("error").and_then(Json::as_str)) {
            (Err(e), Some(message)) => {
                let got = match reference {
                    Reference::StyleSpec => {
                        format!("{ROOT_KEY}{}: {e}", parsed.expression().error_path(e))
                    }
                    Reference::Native => e.to_string(),
                };
                if got == message {
                    true
                } else {
                    return Err(format!(
                        "{name}: input {i}\n  want error {message}\n  got  {got}"
                    ));
                }
            }
            (Ok(v), None) => matches(v, want, color, reference),
            (Ok(v), Some(_)) => {
                return Err(format!("{name}: input {i}: want {want}, got {v:?}"));
            }
            (Err(e), None) => {
                return Err(format!("{name}: input {i}: want {want}, got error {e}"));
            }
        };
        if !ok {
            let got = result.map(|v| format!("{v:?}"));
            return Err(format!("{name}: input {i}\n  want {want}\n  got  {got:?}"));
        }
    }
    Ok(())
}

/// Runs the entire pinned suite through the CPU evaluator. Shared by native
/// tests and wasm tests in Chrome; future backends can call the same cases.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn style_spec_expression_fixtures() {
    let fixtures = json::parse(FIXTURES).expect("fixture bundle parses");
    let fixtures = fixtures.as_object().expect("bundle is an object");
    let failures: Vec<String> = fixtures
        .iter()
        .filter_map(|m| run_fixture(&m.key, &m.value).err())
        .collect();
    #[cfg(not(target_arch = "wasm32"))]
    println!(
        "CPU expression fixtures: {}/{} passed",
        fixtures.len() - failures.len(),
        fixtures.len()
    );
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!(
        "CPU expression fixtures: {}/{} passed",
        fixtures.len() - failures.len(),
        fixtures.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {} fixtures failed:\n{}",
        failures.len(),
        fixtures.len(),
        failures.join("\n")
    );
    assert_eq!(fixtures.len(), 583);
}
