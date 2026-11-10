// Style expressions parsed and evaluated without a map, over orb-expr. A
// result is written as MapLibre Native writes one: numbers as doubles,
// colors as `["rgba", r, g, b, a]`, formatted text as a `format`
// expression.

use std::sync::Arc;

use orb_expr::json::{self, Json};
use orb_expr::{CanonicalTile, EvaluationContext, Expression, FeatureGeometry, Type, Value};

use crate::base::{Failure, Result, boundary, fresh_out, put, sized, text};
use crate::handles::{self, Kind, Object};
use crate::sys::*;

pub(crate) struct ExpressionHandle {
    expression: Expression,
}

fn expression(handle: mln_style_expression) -> Result<Arc<ExpressionHandle>> {
    match handles::lookup(handle.0, Kind::Expression)? {
        Object::Expression(e) => Ok(e),
        _ => unreachable!(),
    }
}

fn item_type(kind: mln_style_expression_type_kind) -> Result<Type> {
    Ok(match kind {
        MLN_STYLE_EXPRESSION_TYPE_VALUE => Type::Value,
        MLN_STYLE_EXPRESSION_TYPE_NUMBER => Type::Number,
        MLN_STYLE_EXPRESSION_TYPE_STRING => Type::String,
        MLN_STYLE_EXPRESSION_TYPE_BOOLEAN => Type::Boolean,
        MLN_STYLE_EXPRESSION_TYPE_COLOR => Type::Color,
        MLN_STYLE_EXPRESSION_TYPE_OBJECT => Type::Object,
        MLN_STYLE_EXPRESSION_TYPE_FORMATTED => Type::Formatted,
        MLN_STYLE_EXPRESSION_TYPE_IMAGE => Type::ResolvedImage,
        MLN_STYLE_EXPRESSION_TYPE_PADDING => Type::Padding,
        MLN_STYLE_EXPRESSION_TYPE_VARIABLE_ANCHOR_OFFSET_COLLECTION => {
            Type::VariableAnchorOffsetCollection
        }
        _ => {
            return Err(Failure::argument(format!(
                "{kind} names no expression type an array can hold"
            )));
        }
    })
}

fn expected_type(t: &mln_style_expression_type) -> Result<Type> {
    if t.kind == MLN_STYLE_EXPRESSION_TYPE_ARRAY {
        let item = item_type(t.item_kind)?;
        return Ok(Type::array(
            item,
            (t.item_count > 0).then_some(t.item_count),
        ));
    }
    item_type(t.kind)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_style_expression_create(
    expression: mln_buffer_view,
    options: *const mln_style_expression_options,
    out_expression: *mut mln_style_expression,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            fresh_out(out_expression, "out_expression")?;
            if expression.data.is_null() {
                return Err(Failure::argument("expression is null"));
            }
            let mut expected = None;
            if !options.is_null() {
                let o = sized(options, "options")?;
                if o.fields & !MLN_STYLE_EXPRESSION_OPTION_EXPECTED_TYPE != 0 {
                    return Err(Failure::argument("options.fields has unknown bits"));
                }
                if o.fields & MLN_STYLE_EXPRESSION_OPTION_EXPECTED_TYPE != 0 {
                    expected = Some(expected_type(&o.expected_type)?);
                }
            }
            let source = text(expression, "expression")?;
            let json = json::parse(&source).map_err(|e| {
                Failure::argument(format!("expression: the expression is not valid JSON: {e}"))
            })?;
            let parsed = Expression::parse_value(&json, expected.as_ref()).map_err(|errors| {
                let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
                Failure::argument(format!("expression: {}", messages.join("; ")))
            })?;
            if contains_collator(parsed.ty()) {
                return Err(Failure::argument(
                    "expression: a collator result has no JSON form",
                ));
            }
            let id = handles::insert(Object::Expression(Arc::new(ExpressionHandle {
                expression: parsed,
            })));
            put(out_expression, mln_style_expression(id), "out_expression")
        })
    }
}

fn contains_collator(ty: &Type) -> bool {
    match ty {
        Type::Collator => true,
        Type::Array(item, _) => contains_collator(item),
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_style_expression_release(expression: mln_style_expression) {
    if handles::lookup(expression.0, Kind::Expression).is_ok() {
        handles::remove(expression.0);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_style_expression_get_info(
    expression: mln_style_expression,
    out_info: *mut mln_style_expression_info,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let e = self::expression(expression)?;
            let size = sized(out_info.cast_const(), "out_info")?.size;
            let reads = e.expression.reads();
            out_info.write(mln_style_expression_info {
                size,
                is_feature_constant: !reads.properties && !reads.feature_state,
                is_zoom_constant: !reads.zoom,
            });
            Ok(())
        })
    }
}

/// A JSON object of the inputs, as key-value pairs.
fn object(view: mln_buffer_view, what: &str) -> Result<Vec<(String, Value<'static>)>> {
    if view.data.is_null() && view.size != 0 {
        return Err(Failure::argument(format!("{what} is null")));
    }
    // SAFETY: the caller's view.
    let source = unsafe { text(view, what) }?;
    if source.is_empty() {
        return Ok(Vec::new());
    }
    let json = json::parse(&source)
        .map_err(|e| Failure::argument(format!("{what} is not valid JSON: {e}")))?;
    match Value::from_json(&json) {
        Value::Object(members) => Ok(members.into_owned()),
        _ => Err(Failure::argument(format!(
            "{what} must contain a JSON object"
        ))),
    }
}

fn finite(v: f64, what: &str) -> Result<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(Failure::argument(format!("{what} must be finite")))
    }
}

fn unavailable(what: &str) -> Failure {
    Failure::argument(format!(
        "expression: the {what} the expression reads is unavailable"
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_style_expression_evaluate(
    expression: mln_style_expression,
    inputs: *const mln_style_expression_inputs,
    out_json: *mut std::ffi::c_char,
    json_capacity: usize,
    out_json_length: *mut usize,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let e = self::expression(expression)?;
            if out_json_length.is_null() {
                return Err(Failure::argument("out_json_length is null"));
            }
            if out_json.is_null() && json_capacity != 0 {
                return Err(Failure::argument("out_json is null"));
            }
            // SAFETY: plain C data.
            let mut i: mln_style_expression_inputs = std::mem::zeroed();
            if !inputs.is_null() {
                i = *sized(inputs, "inputs")?;
            }
            let known = MLN_STYLE_EXPRESSION_INPUT_ZOOM
                | MLN_STYLE_EXPRESSION_INPUT_COLOR_RAMP_PARAMETER
                | MLN_STYLE_EXPRESSION_INPUT_CANONICAL_TILE_ID;
            if i.fields & !known != 0 {
                return Err(Failure::argument("inputs.fields has unknown bits"));
            }
            let has = |bit| i.fields & bit != 0;
            let zoom = has(MLN_STYLE_EXPRESSION_INPUT_ZOOM)
                .then(|| finite(i.zoom, "inputs.zoom"))
                .transpose()?;
            let ramp = has(MLN_STYLE_EXPRESSION_INPUT_COLOR_RAMP_PARAMETER)
                .then(|| finite(i.color_ramp_parameter, "inputs.color_ramp_parameter"))
                .transpose()?;
            let tile = if has(MLN_STYLE_EXPRESSION_INPUT_CANONICAL_TILE_ID) {
                let t = i.canonical_tile_id;
                if t.z > 30 || u64::from(t.x) >= 1u64 << t.z || u64::from(t.y) >= 1u64 << t.z {
                    return Err(Failure::argument("inputs.canonical_tile_id names no tile"));
                }
                Some(CanonicalTile {
                    z: t.z as u8,
                    x: t.x,
                    y: t.y,
                })
            } else {
                None
            };
            if i.feature.data.is_null() && i.feature.size != 0 {
                return Err(Failure::argument("inputs.feature is null"));
            }
            let feature_text = text(i.feature, "inputs.feature")?;
            let feature = if feature_text.is_empty() {
                None
            } else {
                let json = json::parse(&feature_text).map_err(|e| {
                    Failure::argument(format!("inputs.feature is not valid JSON: {e}"))
                })?;
                if json.get("type").and_then(Json::as_str) != Some("Feature") {
                    return Err(Failure::argument(
                        "inputs.feature must be a GeoJSON feature",
                    ));
                }
                Some(json)
            };
            let feature_state = object(i.feature_state, "inputs.feature_state")?;
            let global_state = object(i.global_state, "inputs.global_state")?;
            let mut images = Vec::new();
            if i.available_image_count > 0 {
                if i.available_images.is_null() {
                    return Err(Failure::argument("inputs.available_images is null"));
                }
                for (n, id) in
                    std::slice::from_raw_parts(i.available_images, i.available_image_count)
                        .iter()
                        .enumerate()
                {
                    if id.data.is_null() || id.size == 0 {
                        return Err(Failure::argument(format!(
                            "inputs.available_images[{n}] must not be empty"
                        )));
                    }
                    images.push(text(*id, "inputs.available_images")?);
                }
            }

            let reads = e.expression.reads();
            if reads.zoom && zoom.is_none() {
                return Err(unavailable("zoom level"));
            }
            if reads.ramp && ramp.is_none() {
                return Err(unavailable("color ramp parameter"));
            }
            if reads.properties && feature.is_none() {
                return Err(unavailable("feature"));
            }
            let geometry = match (&feature, tile) {
                (Some(f), Some(tile)) => f
                    .get("geometry")
                    .and_then(|g| FeatureGeometry::from_geojson(g, tile)),
                _ => None,
            };
            if reads.geometry && geometry.is_none() {
                return Err(unavailable("feature geometry in a canonical tile"));
            }
            let ramp = ramp.unwrap_or(0.0);
            let context = EvaluationContext {
                zoom: zoom.unwrap_or(0.0),
                line_progress: ramp,
                heatmap_density: ramp,
                elevation: ramp,
                feature_state: &feature_state,
                global_state: &global_state,
                available_images: &images,
                geometry: geometry.as_ref(),
                ..EvaluationContext::default()
            };
            let result = match &feature {
                Some(f) => e.expression.evaluate_geojson(&context, f),
                None => e.expression.evaluate_with(&context),
            }
            .map_err(|err| Failure::argument(format!("expression: {err}")))?;
            let mut out = String::new();
            write_value(&mut out, &result);
            put(out_json_length, out.len(), "out_json_length")?;
            if json_capacity > out.len() {
                std::ptr::copy_nonoverlapping(out.as_ptr(), out_json.cast::<u8>(), out.len());
                *out_json.add(out.len()) = 0;
            }
            Ok(())
        })
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push_str(
        &json::Json::new(
            json::JsonValue::String(s.to_string()),
            json::Span::default(),
        )
        .to_string(),
    );
}

fn write_number(out: &mut String, n: f64) {
    if n.is_finite() {
        out.push_str(&format!("{n:?}"));
    } else {
        out.push_str("null");
    }
}

fn write_color(out: &mut String, c: &orb_expr::Color) {
    let [r, g, b, a] = c.straight();
    out.push_str("[\"rgba\"");
    for v in [r * 255.0, g * 255.0, b * 255.0, a] {
        out.push(',');
        write_number(out, f64::from(v));
    }
    out.push(']');
}

/// A value as Native writes one.
fn write_value(out: &mut String, v: &Value<'_>) {
    match v {
        Value::Null | Value::Collator(_) => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => write_number(out, *n),
        Value::String(s) => write_string(out, s),
        Value::Color(c) => write_color(out, c),
        Value::Image { name, available } => {
            out.push_str("{\"name\":");
            write_string(out, name);
            out.push_str(&format!(",\"available\":{available}}}"));
        }
        Value::TypedArray(_, items) | Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(members) => {
            out.push('{');
            for (i, (k, v)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, k);
                out.push(':');
                write_value(out, v);
            }
            out.push('}');
        }
        Value::Formatted(sections) => {
            out.push_str("[\"format\"");
            for s in sections.iter() {
                out.push(',');
                match &s.image {
                    Some((name, _)) => {
                        out.push_str("[\"image\",");
                        write_string(out, name);
                        out.push(']');
                    }
                    None => write_string(out, &s.text),
                }
                let mut options = Vec::new();
                if let Some(scale) = s.scale {
                    let mut o = String::from("\"font-scale\":");
                    write_number(&mut o, scale);
                    options.push(o);
                }
                if let Some(fonts) = &s.font_stack {
                    let mut o = String::from("\"text-font\":[\"literal\",[");
                    for (i, font) in fonts.split(',').enumerate() {
                        if i > 0 {
                            o.push(',');
                        }
                        write_string(&mut o, font.trim());
                    }
                    o.push_str("]]");
                    options.push(o);
                }
                if let Some(color) = &s.text_color {
                    let mut o = String::from("\"text-color\":");
                    write_color(&mut o, color);
                    options.push(o);
                }
                if let Some(align) = &s.vertical_align {
                    let mut o = String::from("\"vertical-align\":");
                    write_string(&mut o, align);
                    options.push(o);
                }
                if !options.is_empty() {
                    out.push_str(",{");
                    out.push_str(&options.join(","));
                    out.push('}');
                }
            }
            out.push(']');
        }
    }
}
