//! Legacy syntax, lowered to expressions in one place (plan §11.2, S3).
//!
//! Legacy filters become GL JS's internal `filter-*` operators
//! (`feature_filter/index.ts`), which keep legacy semantics: a missing
//! property or a type mismatch makes a test false instead of an error.
//! Legacy functions become expression JSON, as the style-spec's
//! `convertFunction` builds it, and are then parsed like any expression.

use crate::Diagnostic;
use crate::ir::{CmpOp, LegacyFilter, NodeId, Op, Subject};
use crate::json::{Json, JsonValue, Span};
use crate::parse::Parser;
use crate::property::{PropertySpec, PropertyType};
use crate::value::{Type, Value};

const LEGACY_OPS: &[&str] = &[
    "==", "!=", ">", ">=", "<", "<=", "in", "!in", "all", "any", "none", "has", "!has",
];
const GEOMETRY_TYPES: &[&str] = &["Point", "LineString", "Polygon"];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Syntax {
    Expression,
    Legacy,
    /// Reads the same either way, so it is no evidence of either syntax.
    Neutral,
}

fn classify(filter: &Json) -> Syntax {
    let items = match &filter.value {
        JsonValue::Bool(_) => return Syntax::Neutral,
        JsonValue::Array(items) if !items.is_empty() => items,
        _ => return Syntax::Legacy,
    };
    let is_array = |i: usize| items.get(i).is_some_and(|j| j.as_array().is_some());
    match items[0].as_str() {
        Some("has") => {
            if items.len() < 2 || matches!(items[1].as_str(), Some("$id" | "$type")) {
                Syntax::Legacy
            } else if items.len() == 2 {
                Syntax::Neutral
            } else {
                Syntax::Expression
            }
        }
        // Legacy `in` tests a property against a set; the expression tests
        // for a substring. The scalar form stays legacy.
        Some("in") => {
            if items.len() >= 3 && (items[1].as_str().is_none() || is_array(2)) {
                Syntax::Expression
            } else {
                Syntax::Legacy
            }
        }
        Some("!in" | "!has" | "none") => Syntax::Legacy,
        Some("==" | "!=" | ">" | ">=" | "<" | "<=") => {
            if items.len() != 3 || is_array(1) || is_array(2) {
                Syntax::Expression
            } else {
                Syntax::Legacy
            }
        }
        Some("any" | "all") => {
            let mut legacy = false;
            for child in &items[1..] {
                match classify(child) {
                    Syntax::Expression => return Syntax::Expression,
                    Syntax::Legacy => legacy = true,
                    Syntax::Neutral => {}
                }
            }
            if legacy {
                Syntax::Legacy
            } else {
                Syntax::Neutral
            }
        }
        _ => Syntax::Expression,
    }
}

/// GL JS's `isExpressionFilter`.
pub(crate) fn is_expression_filter(filter: &Json) -> bool {
    classify(filter) != Syntax::Legacy
}

fn child_key(key: &str, i: usize) -> String {
    format!("{key}[{i}]")
}

/// The warning GL JS logs for legacy syntax inside an expression filter.
pub(crate) fn mixed_filter_warning(filter: &Json) -> Option<Diagnostic> {
    let (path, legacy) = find_mixed(filter, String::new())?;
    Some(Diagnostic {
        path,
        span: legacy.span,
        message: mixed_message(legacy),
    })
}

fn find_mixed(filter: &Json, path: String) -> Option<(String, &Json)> {
    let items = filter.as_array().filter(|a| !a.is_empty())?;
    let check = |i: usize| -> Option<(String, &Json)> {
        let child = items.get(i)?;
        child.as_array()?;
        let path = child_key(&path, i);
        if !is_expression_filter(child) {
            Some((path, child))
        } else {
            find_mixed(child, path)
        }
    };
    match items[0].as_str() {
        Some("all" | "any" | "none") => (1..items.len()).find_map(check),
        Some("!") => check(1),
        Some("case") => (1..items.len().saturating_sub(1))
            .step_by(2)
            .find_map(check),
        _ => None,
    }
}

fn mixed_message(filter: &Json) -> String {
    let items = filter.as_array().unwrap_or_default();
    let op = items.first().and_then(Json::as_str).unwrap_or_default();
    if matches!(op, "<" | "<=" | ">" | ">=") && items.get(1).and_then(Json::as_str) == Some("$type")
    {
        return format!("\"$type\" cannot be use with operator \"{op}\"");
    }
    match suggestion(filter) {
        Some(s) => format!(
            "Mixing deprecated filter syntax with expression syntax is not supported. Replace {filter} with {s}."
        ),
        None => format!(
            "Mixing deprecated filter syntax with expression syntax is not supported. Convert {filter} to expression syntax."
        ),
    }
}

fn suggestion(filter: &Json) -> Option<Json> {
    let items = filter.as_array()?;
    let b = Build(filter.span);
    let op = items.first()?.as_str()?;
    let key = items.get(1).and_then(Json::as_str);
    let subject = |key: &str| match key {
        "$type" => b.array(vec![b.string("geometry-type")]),
        "$id" => b.array(vec![b.string("id")]),
        _ => b.array(vec![b.string("get"), b.string(key)]),
    };
    match op {
        "==" | "!=" | "<" | "<=" | ">" | ">=" => {
            if items.len() != 3 {
                return None;
            }
            Some(b.array(vec![b.string(op), subject(key?), items[2].clone()]))
        }
        "in" | "!in" => {
            let expr = b.array(vec![
                b.string("in"),
                subject(key?),
                b.array(vec![b.string("literal"), b.array(items[2..].to_vec())]),
            ]);
            Some(if op == "!in" {
                b.array(vec![b.string("!"), expr])
            } else {
                expr
            })
        }
        "has" | "!has" => {
            let key = key.filter(|k| items.len() == 2 && !matches!(*k, "$type" | "$id"))?;
            let expr = b.array(vec![b.string("has"), b.string(key)]);
            Some(if op == "!has" {
                b.array(vec![b.string("!"), expr])
            } else {
                expr
            })
        }
        _ => None,
    }
}

/// Lowers a legacy filter (one [`is_expression_filter`] rejects) to IR.
pub(crate) fn lower_filter(p: &mut Parser, filter: &Json, key: &str) -> Option<NodeId> {
    let Some(items) = filter.as_array() else {
        p.error(
            key,
            filter.span,
            format!("array expected, {} found", filter.type_name()),
        );
        return None;
    };
    let Some(op) = items.first().and_then(Json::as_str) else {
        let found = items
            .first()
            .map_or_else(|| "nothing".to_string(), Json::to_string);
        p.error(
            key,
            filter.span,
            format!("expected one of [{}], {found} found", LEGACY_OPS.join(", ")),
        );
        return None;
    };
    if !LEGACY_OPS.contains(&op) {
        p.error(
            child_key(key, 0),
            items[0].span,
            format!(
                "expected one of [{}], \"{op}\" found",
                LEGACY_OPS.join(", ")
            ),
        );
        return None;
    }
    if items.len() <= 1 {
        return Some(literal_bool(p, op != "any", key));
    }
    let boolean = |p: &mut Parser, op: Op| p.ir.push(op, Type::Boolean, key);
    match op {
        "==" | "!=" | "<" | "<=" | ">" | ">=" => {
            if items.len() != 3 {
                p.error(
                    key,
                    filter.span,
                    format!("filter array for operator \"{op}\" must have 3 elements"),
                );
                return None;
            }
            let subject = subject(p, &items[1], &child_key(key, 1))?;
            let cmp = match op {
                "==" => CmpOp::Eq,
                "!=" => CmpOp::Ne,
                "<" => CmpOp::Lt,
                "<=" => CmpOp::Le,
                ">" => CmpOp::Gt,
                _ => CmpOp::Ge,
            };
            if subject == Subject::Type && cmp.is_order() {
                p.error(
                    key,
                    filter.span,
                    format!("\"$type\" cannot be use with operator \"{op}\""),
                );
                return None;
            }
            let value = filter_value(p, subject, &items[2], &child_key(key, 2))?;
            Some(boolean(
                p,
                Op::Legacy(LegacyFilter::Compare {
                    subject,
                    op: cmp,
                    value,
                }),
            ))
        }
        "any" | "all" | "none" => {
            let mut children = Vec::new();
            for (i, child) in items.iter().enumerate().skip(1) {
                let child_key = child_key(key, i);
                let mut id = lower_filter(p, child, &child_key)?;
                if op == "none" {
                    id = p.ir.push(Op::Not(id), Type::Boolean, &child_key);
                }
                children.push(id);
            }
            Some(boolean(
                p,
                if op == "any" {
                    Op::Any(children)
                } else {
                    Op::All(children)
                },
            ))
        }
        "in" | "!in" => {
            let subject = subject(p, &items[1], &child_key(key, 1))?;
            let mut values = Vec::new();
            for (i, v) in items.iter().enumerate().skip(2) {
                values.push(filter_value(p, subject, v, &child_key(key, i))?);
            }
            let id = if values.is_empty() {
                literal_bool(p, false, key)
            } else {
                boolean(p, Op::Legacy(LegacyFilter::In { subject, values }))
            };
            Some(if op == "!in" {
                boolean(p, Op::Not(id))
            } else {
                id
            })
        }
        _ => {
            if items.len() != 2 {
                p.error(
                    key,
                    filter.span,
                    format!("filter array for \"{op}\" operator must have 2 elements"),
                );
                return None;
            }
            let subject = subject(p, &items[1], &child_key(key, 1))?;
            let id = if subject == Subject::Type {
                literal_bool(p, true, key)
            } else {
                boolean(p, Op::Legacy(LegacyFilter::Has(subject)))
            };
            Some(if op == "!has" {
                boolean(p, Op::Not(id))
            } else {
                id
            })
        }
    }
}

fn literal_bool(p: &mut Parser, b: bool, key: &str) -> NodeId {
    p.ir.push(Op::Literal(Value::Bool(b)), Type::Boolean, key)
}

fn subject(p: &mut Parser, key: &Json, path: &str) -> Option<Subject> {
    match key.as_str() {
        Some("$type") => Some(Subject::Type),
        Some("$id") => Some(Subject::Id),
        Some(name) => Some(Subject::Property(p.ir.intern_key(name))),
        None => {
            p.error(
                path,
                key.span,
                format!("string expected, {} found", key.type_name()),
            );
            None
        }
    }
}

fn filter_value(
    p: &mut Parser,
    subject: Subject,
    value: &Json,
    path: &str,
) -> Option<Value<'static>> {
    if subject == Subject::Type {
        let name = value.as_str();
        if !name.is_some_and(|n| GEOMETRY_TYPES.contains(&n)) {
            p.error(
                path,
                value.span,
                format!(
                    "expected one of [{}], {value} found",
                    GEOMETRY_TYPES.join(", ")
                ),
            );
            return None;
        }
    } else if !matches!(
        value.value,
        JsonValue::String(_) | JsonValue::Number(_) | JsonValue::Bool(_)
    ) {
        p.error(
            path,
            value.span,
            format!(
                "string, number, or boolean expected, {} found",
                value.type_name()
            ),
        );
        return None;
    }
    Some(Value::from_json(value))
}

/// Builds synthetic JSON, every node carrying the span it was lowered from.
#[derive(Clone, Copy)]
struct Build(Span);

impl Build {
    fn json(self, value: JsonValue) -> Json {
        Json::new(value, self.0)
    }
    fn string(self, s: &str) -> Json {
        self.json(JsonValue::String(s.into()))
    }
    fn number(self, n: f64) -> Json {
        self.json(JsonValue::Number(n))
    }
    fn array(self, items: Vec<Json>) -> Json {
        self.json(JsonValue::Array(items))
    }
    /// `convertLiteral`: objects, arrays and `null` need `literal`.
    fn literal(self, v: &Json) -> Json {
        match v.value {
            JsonValue::Null | JsonValue::Array(_) | JsonValue::Object(_) => {
                Build(v.span).array(vec![Build(v.span).string("literal"), v.clone()])
            }
            _ => v.clone(),
        }
    }
}

/// Strict equality of JSON scalars, as `===` on parsed JSON.
fn json_strict_eq(a: &Json, b: &Json) -> bool {
    match (&a.value, &b.value) {
        (JsonValue::Null, JsonValue::Null) => true,
        (JsonValue::Bool(a), JsonValue::Bool(b)) => a == b,
        (JsonValue::Number(a), JsonValue::Number(b)) => a == b,
        (JsonValue::String(a), JsonValue::String(b)) => a == b,
        _ => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FunctionType {
    Identity,
    Exponential,
    Interval,
    Categorical,
}

/// A legacy function's parameters, validated.
struct Function<'a> {
    json: &'a Json,
    ty: Option<FunctionType>,
    property: Option<&'a Json>,
    base: Option<f64>,
    default: Option<&'a Json>,
    /// `interpolate`, `interpolate-hcl` or `interpolate-lab`.
    interpolate: &'static str,
    stops: Option<Vec<(&'a Json, &'a Json)>>,
}

fn error(errors: &mut Vec<Diagnostic>, path: &str, span: Span, message: impl Into<String>) {
    errors.push(Diagnostic {
        path: path.into(),
        span,
        message: message.into(),
    });
}

impl<'a> Function<'a> {
    fn read(json: &'a Json, errors: &mut Vec<Diagnostic>) -> Option<Self> {
        let start = errors.len();
        let ty = json.get("type").and_then(|t| {
            let ty = match t.as_str() {
                Some("identity") => FunctionType::Identity,
                Some("exponential") => FunctionType::Exponential,
                Some("interval") => FunctionType::Interval,
                Some("categorical") => FunctionType::Categorical,
                _ => {
                    error(
                        errors,
                        ".type",
                        t.span,
                        format!(
                            "expected one of [identity, exponential, interval, categorical], {t} found"
                        ),
                    );
                    return None;
                }
            };
            Some(ty)
        });
        let property = json.get("property");
        if let Some(p) = property
            && p.as_str().is_none()
        {
            error(
                errors,
                ".property",
                p.span,
                format!("string expected, {} found", p.type_name()),
            );
        }
        let base = json.get("base").and_then(|b| {
            if b.as_f64().is_none() {
                error(
                    errors,
                    ".base",
                    b.span,
                    format!("number expected, {} found", b.type_name()),
                );
            }
            b.as_f64()
        });
        let interpolate = match json.get("colorSpace") {
            None => "interpolate",
            Some(c) => match c.as_str() {
                Some("rgb") => "interpolate",
                Some("hcl") => "interpolate-hcl",
                Some("lab") => "interpolate-lab",
                _ => {
                    error(
                        errors,
                        ".colorSpace",
                        c.span,
                        format!("expected one of [rgb, lab, hcl], {c} found"),
                    );
                    "interpolate"
                }
            },
        };
        let stops = json.get("stops").and_then(|stops| {
            let Some(items) = stops.as_array() else {
                error(
                    errors,
                    ".stops",
                    stops.span,
                    format!("array expected, {} found", stops.type_name()),
                );
                return None;
            };
            if items.is_empty() {
                error(
                    errors,
                    ".stops",
                    stops.span,
                    "array must have at least one stop",
                );
                return None;
            }
            let mut pairs = Vec::new();
            for (i, stop) in items.iter().enumerate() {
                match stop.as_array() {
                    Some([input, output]) => pairs.push((input, output)),
                    _ => error(
                        errors,
                        &format!(".stops[{i}]"),
                        stop.span,
                        "array with 2 elements expected",
                    ),
                }
            }
            Some(pairs)
        });
        match (ty, &stops) {
            (Some(FunctionType::Identity), _) if property.is_none() => {
                error(
                    errors,
                    "",
                    json.span,
                    "missing required property \"property\"",
                );
            }
            (Some(t), None) if t != FunctionType::Identity && json.get("stops").is_none() => {
                error(errors, "", json.span, "missing required property \"stops\"");
            }
            (None, None) if json.get("stops").is_none() => {
                error(errors, "", json.span, "missing required property \"stops\"");
            }
            _ => {}
        }
        (errors.len() == start).then_some(Self {
            json,
            ty,
            property,
            base,
            default: json.get("default"),
            interpolate,
            stops,
        })
    }

    /// `getFunctionType`.
    fn ty(&self, spec: &PropertySpec) -> FunctionType {
        self.ty.unwrap_or(if spec.interpolated {
            FunctionType::Exponential
        } else {
            FunctionType::Interval
        })
    }
}

/// `appendStopPair`: skips duplicate inputs; a step drops its first input.
fn append_stop(curve: &mut Vec<Json>, input: Json, output: Json, is_step: bool) {
    if curve.len() > 3 && json_strict_eq(&input, &curve[curve.len() - 2]) {
        return;
    }
    if !(is_step && curve.len() == 2) {
        curve.push(input);
    }
    curve.push(output);
}

/// `fixupDegenerateStepCurve`: a one-stop step gets a no-op second stop.
///
/// The reference pushes `expression[3]` after pushing the new label, which
/// repeats the label `0` as the output; GL JS's runtime (`createFunction`)
/// treats a one-stop function as constant, so this repeats the output.
fn fix_degenerate_step(b: Build, curve: &mut Vec<Json>) {
    if curve.first().and_then(Json::as_str) == Some("step") && curve.len() == 3 {
        let output = curve[2].clone();
        curve.push(b.number(0.0));
        curve.push(output);
    }
}

/// `convertTokenString`: `"a {b} c"` → `["concat", "a ", ["get", "b"], " c"]`;
/// a string without tokens stays a string.
pub(crate) fn convert_token_string(span: Span, s: &str) -> Json {
    let b = Build(span);
    let mut parts = vec![b.string("concat")];
    let mut pos = 0;
    let mut search = 0;
    while let Some(open) = s[search..].find('{').map(|i| search + i) {
        let Some(close) = s[open + 1..].find(['{', '}']).map(|i| open + 1 + i) else {
            break;
        };
        if s.as_bytes()[close] == b'{' || close == open + 1 {
            search = close.max(open + 1);
            continue;
        }
        if open > pos {
            parts.push(b.string(&s[pos..open]));
        }
        parts.push(b.array(vec![b.string("get"), b.string(&s[open + 1..close])]));
        pos = close + 1;
        search = pos;
    }
    if parts.len() == 1 {
        return b.string(s);
    }
    if pos < s.len() {
        parts.push(b.string(&s[pos..]));
    } else if parts.len() == 2 {
        return b.array(vec![b.string("to-string"), parts.pop().expect("one part")]);
    }
    b.array(parts)
}

/// Lowers a legacy function to expression JSON (`convertFunction`).
pub(crate) fn lower_function(json: &Json, spec: &PropertySpec) -> Result<Json, Vec<Diagnostic>> {
    let mut errors = Vec::new();
    let f = Function::read(json, &mut errors).ok_or_else(|| errors.clone())?;
    let b = Build(json.span);
    let Some(stops) = &f.stops else {
        return Ok(identity(&f, spec, b));
    };

    let zoom_and_feature = matches!(
        stops[0].0.value,
        JsonValue::Object(_) | JsonValue::Array(_) | JsonValue::Null
    );
    let feature = zoom_and_feature || f.property.is_some();
    let stops: Vec<(Json, Json)> = stops
        .iter()
        .map(|(input, output)| {
            let output = match output.as_str() {
                Some(s) if !feature && spec.tokens => convert_token_string(output.span, s),
                _ => b.literal(output),
            };
            ((*input).clone(), output)
        })
        .collect();

    let result = if zoom_and_feature {
        zoom_and_property(&f, spec, b, stops, &mut errors)
    } else if feature {
        property_function(&f, spec, b, false, stops, &mut errors)
    } else {
        zoom_function(&f, spec, b, stops, &mut errors)
    };
    match result {
        Some(json) if errors.is_empty() => Ok(json),
        _ => Err(errors),
    }
}

fn type_name(spec: &PropertySpec) -> &'static str {
    match &spec.ty {
        PropertyType::Number => "number",
        PropertyType::String => "string",
        PropertyType::Boolean => "boolean",
        PropertyType::Color => "to-color",
        PropertyType::ResolvedImage => "resolvedImage",
        PropertyType::Formatted => "formatted",
        PropertyType::ProjectionDefinition => "projectionDefinition",
        PropertyType::Padding => "padding",
        PropertyType::NumberArray => "numberArray",
        PropertyType::ColorArray => "colorArray",
        PropertyType::VariableAnchorOffsetCollection => "variableAnchorOffsetCollection",

        PropertyType::Enum(_) => "string",
        PropertyType::Array(..) => "array",
    }
}

fn identity(f: &Function<'_>, spec: &PropertySpec, b: Build) -> Json {
    let property = f.property.cloned().unwrap_or_else(|| b.string(""));
    let get = b.array(vec![b.string("get"), property]);
    match (f.default, &spec.ty) {
        (None, PropertyType::String) => b.array(vec![b.string("string"), get]),
        (None, _) => get,
        (Some(default), PropertyType::Enum(values)) => b.array(vec![
            b.string("match"),
            get.clone(),
            b.array(values.iter().map(|v| b.string(v)).collect()),
            get,
            default.clone(),
        ]),
        (Some(default), ty) => {
            let mut expr = vec![b.string(type_name(spec))];
            if let PropertyType::Array(item, len) = ty {
                expr.push(b.string(item.kind()));
                expr.push(match len {
                    Some(n) => b.number(f64::from(*n)),
                    None => b.json(JsonValue::Null),
                });
            }
            expr.push(get);
            expr.push(b.literal(default));
            b.array(expr)
        }
    }
}

fn zoom_and_property(
    f: &Function<'_>,
    spec: &PropertySpec,
    b: Build,
    stops: Vec<(Json, Json)>,
    errors: &mut Vec<Diagnostic>,
) -> Option<Json> {
    // Group the stops by zoom, in order of first appearance.
    let mut groups: Vec<(f64, Vec<(Json, Json)>)> = Vec::new();
    for (i, (input, output)) in stops.into_iter().enumerate() {
        let zoom = input.get("zoom").and_then(Json::as_f64);
        let value = input.get("value");
        let (Some(zoom), Some(value)) = (zoom, value) else {
            error(
                errors,
                &format!(".stops[{i}][0]"),
                input.span,
                "object with numeric \"zoom\" and a \"value\" expected",
            );
            continue;
        };
        let pair = (value.clone(), output);
        match groups.iter_mut().find(|(z, _)| *z == zoom) {
            Some((_, group)) => group.push(pair),
            None => groups.push((zoom, vec![pair])),
        }
    }
    if !errors.is_empty() {
        return None;
    }
    // The zoom dimension interpolates linearly for interpolatable
    // properties and steps otherwise.
    let is_step = !spec.interpolated;
    let mut curve = if is_step {
        vec![b.string("step"), b.array(vec![b.string("zoom")])]
    } else {
        vec![
            b.string(f.interpolate),
            b.array(vec![b.string("linear")]),
            b.array(vec![b.string("zoom")]),
        ]
    };
    for (zoom, group) in groups {
        let output = property_function(f, spec, b, true, group, errors)?;
        append_stop(&mut curve, b.number(zoom), output, is_step);
    }
    if is_step {
        fix_degenerate_step(b, &mut curve);
    }
    Some(b.array(curve))
}

/// `getFallback`: the function's default, else the property's.
fn fallback(
    f: &Function<'_>,
    spec: &PropertySpec,
    b: Build,
    errors: &mut Vec<Diagnostic>,
) -> Option<Json> {
    match f.default.or(spec.default.as_ref()) {
        Some(default) => Some(b.literal(default)),
        None if spec.ty == PropertyType::ResolvedImage => Some(b.string("")),
        None => {
            error(
                errors,
                "",
                f.json.span,
                "a categorical function needs a \"default\" for a property without one",
            );
            None
        }
    }
}

/// `convertPropertyFunction`. A `nested` function is one zoom level of a
/// zoom-and-property function, whose parameters GL JS rebuilds without
/// `base` or `colorSpace`.
fn property_function(
    f: &Function<'_>,
    spec: &PropertySpec,
    b: Build,
    nested: bool,
    stops: Vec<(Json, Json)>,
    errors: &mut Vec<Diagnostic>,
) -> Option<Json> {
    let (base, interpolate) = if nested {
        (None, "interpolate")
    } else {
        (f.base, f.interpolate)
    };
    let property = f.property.cloned().unwrap_or_else(|| b.string(""));
    let get = b.array(vec![b.string("get"), property]);
    let with_default = |curve: Json| match f.default {
        None => curve,
        Some(default) => b.array(vec![
            b.string("case"),
            b.array(vec![
                b.string("=="),
                b.array(vec![b.string("typeof"), get.clone()]),
                b.string("number"),
            ]),
            curve,
            b.literal(default),
        ]),
    };
    match f.ty(spec) {
        FunctionType::Categorical if matches!(stops[0].0.value, JsonValue::Bool(_)) => {
            let mut expr = vec![b.string("case")];
            for (input, output) in stops {
                expr.push(b.array(vec![b.string("=="), get.clone(), input]));
                expr.push(output);
            }
            expr.push(fallback(f, spec, b, errors)?);
            Some(b.array(expr))
        }
        FunctionType::Categorical => {
            let mut expr = vec![b.string("match"), get];
            for (input, output) in stops {
                append_stop(&mut expr, input, output, false);
            }
            expr.push(fallback(f, spec, b, errors)?);
            Some(b.array(expr))
        }
        FunctionType::Interval => {
            let mut expr = vec![
                b.string("step"),
                b.array(vec![b.string("number"), get.clone()]),
            ];
            for (input, output) in stops {
                append_stop(&mut expr, input, output, true);
            }
            fix_degenerate_step(b, &mut expr);
            Some(with_default(b.array(expr)))
        }
        FunctionType::Exponential => {
            let mut expr = vec![
                b.string(interpolate),
                interpolation(b, base),
                b.array(vec![b.string("number"), get.clone()]),
            ];
            for (input, output) in stops {
                append_stop(&mut expr, input, output, false);
            }
            Some(with_default(b.array(expr)))
        }
        FunctionType::Identity => {
            error(
                errors,
                ".type",
                f.json.span,
                "identity functions take no \"stops\"",
            );
            None
        }
    }
}

fn interpolation(b: Build, base: Option<f64>) -> Json {
    match base.unwrap_or(1.0) {
        1.0 => b.array(vec![b.string("linear")]),
        base => b.array(vec![b.string("exponential"), b.number(base)]),
    }
}

fn zoom_function(
    f: &Function<'_>,
    spec: &PropertySpec,
    b: Build,
    stops: Vec<(Json, Json)>,
    errors: &mut Vec<Diagnostic>,
) -> Option<Json> {
    let zoom = b.array(vec![b.string("zoom")]);
    let (mut curve, is_step) = match f.ty(spec) {
        FunctionType::Interval => (vec![b.string("step"), zoom], true),
        FunctionType::Exponential => (
            vec![b.string(f.interpolate), interpolation(b, f.base), zoom],
            false,
        ),
        other => {
            let name = if other == FunctionType::Categorical {
                "categorical"
            } else {
                "identity"
            };
            error(
                errors,
                ".type",
                f.json.get("type").map_or(f.json.span, |t| t.span),
                format!("Unknown zoom function type \"{name}\""),
            );
            return None;
        }
    };
    for (input, output) in stops {
        append_stop(&mut curve, input, output, is_step);
    }
    fix_degenerate_step(b, &mut curve);
    Some(b.array(curve))
}
