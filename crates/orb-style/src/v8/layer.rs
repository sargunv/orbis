//! Layers: a v8 layer's source and selection, and its properties in the
//! order of the property table ([`PropertyId`]).

use crate::*;
use std::collections::HashSet;

use orb_expr::json::{Json, JsonValue, Member};
use orb_expr::{Filter, Kind, PropertyExpression, Value};

use crate::v8::Cx;

const LAYER_KEYS: &[&str] = &[
    "id",
    "type",
    "metadata",
    "source",
    "source-layer",
    "minzoom",
    "maxzoom",
    "filter",
    "layout",
    "paint",
];

/// A layer's `paint` or `layout` object.
struct Section<'a> {
    path: String,
    members: &'a [Member],
}

impl Section<'_> {
    fn get(&self, name: &str) -> Option<&Json> {
        self.members
            .iter()
            .rev()
            .find(|m| m.key == name)
            .map(|m| &m.value)
    }
}

fn section<'a>(
    cx: &mut Cx,
    layer: &'a Json,
    key: &str,
    layer_path: &str,
    ty: LayerType,
) -> Section<'a> {
    let path = format!("{layer_path}.{key}");
    let members = match layer.get(key) {
        None => &[][..],
        Some(json) => cx.object(json, &path).unwrap_or_default(),
    };
    {
        let paint = key == "paint";
        let known = |name: &str| {
            PropertyId::find(ty, paint, name).is_some() || (!paint && name == "visibility")
        };
        for m in members {
            let member_path = format!("{path}.{}", m.key);
            if known(&m.key) {
                continue;
            }
            match m.key.strip_suffix("-transition") {
                Some(base) if paint && known(base) => {}
                _ => cx.warn(
                    member_path,
                    m.key_span,
                    format!("unknown property \"{}\"", m.key),
                ),
            }
        }
    }
    Section { path, members }
}

/// Validates a constant against the property's enum values and range.
/// A property's slot from its JSON (`None`: unset), with diagnostics at
/// `path`; an invalid value is the default.
pub(crate) fn property_at(cx: &mut Cx, json: Option<&Json>, path: &str, prop: &Prop) -> Slot {
    if prop.ramp
        && json.is_none()
        && let Some(default) = prop.default_json()
    {
        let ramp = PropertyExpression::parse(&default, &prop.spec()).expect("valid default ramp");
        return Slot::expression_any(ramp, false);
    }
    let Some(json) = json else {
        return Slot::constant(Value::Null, false);
    };
    let path = path.to_string();
    if prop.ramp {
        return match PropertyExpression::parse_with_state(json, &prop.spec(), &cx.state) {
            Ok(parsed) => Slot::expression_any(parsed, true),
            Err(errors) => {
                cx.errors_at(&path, errors);
                Slot::constant(Value::Null, false)
            }
        };
    }
    if prop.list
        && let Some(items) = json.as_array().filter(|items| {
            !items.is_empty()
                && items.iter().all(|item| match &item.value {
                    JsonValue::Number(_) => true,
                    JsonValue::String(s) => orb_expr::Color::parse(s).is_some(),
                    _ => false,
                })
        })
    {
        let items = items
            .iter()
            .enumerate()
            .map(|(i, item)| property_at_one(cx, item, &format!("{path}[{i}]"), prop))
            .collect();
        return Slot::list(items);
    }
    property_at_one(cx, json, &path, prop)
}

/// [`property_at`] for a typed property outside a layer.
pub(crate) fn typed_property_at<T: PropertyValue>(
    cx: &mut Cx,
    json: Option<&Json>,
    path: &str,
    prop: &Prop,
) -> crate::v8::Property<T> {
    let default = match prop.default_json() {
        None => T::UNSET.unwrap_or_else(|| panic!("{} has no default", prop.name)),
        Some(json) => {
            let parsed = PropertyExpression::parse(&json, &prop.spec())
                .unwrap_or_else(|e| panic!("{} default: {e:?}", prop.name));
            T::from_value(
                parsed
                    .expression()
                    .constant()
                    .expect("defaults are literals"),
            )
            .unwrap_or_else(|| panic!("{} default has the wrong type", prop.name))
        }
    };
    crate::v8::Property::from_slot(property_at(cx, json, path, prop), default)
}

fn property_at_one(cx: &mut Cx, json: &Json, path: &str, prop: &Prop) -> Slot {
    match PropertyExpression::parse_with_state(json, &prop.spec(), &cx.state) {
        Err(errors) => {
            cx.errors_at(path, errors);
            Slot::constant(Value::Null, false)
        }
        // Constants are folded to literals, except images, which GL JS
        // leaves to resolve at evaluation with the style's images.
        Ok(parsed) if parsed.kind() == Kind::Constant && !parsed.expression().reads_images() => {
            let value = parsed
                .expression()
                .evaluate(0.0)
                .map(Value::into_owned)
                .unwrap_or(Value::Null);
            check_constant(cx, prop, json, &value, path);
            Slot::constant(value, true)
        }
        Ok(parsed) => Slot::expression(parsed),
    }
}

fn check_constant(cx: &mut Cx, prop: &Prop, json: &Json, value: &Value<'_>, path: &str) {
    if let orb_expr::PropertyType::Enum(values) = prop.ty
        && let Some(s) = value.as_str()
        && !values.contains(&s)
    {
        cx.error(
            path,
            json.span,
            format!("expected one of [{}], \"{s}\" found", values.join(", ")),
        );
    }
    // Ranges apply to the literal numbers of a plain value, as GL JS
    // validates them.
    let numbers: Vec<&Json> = match &json.value {
        JsonValue::Number(_) => vec![json],
        JsonValue::Array(items) if items.iter().all(|i| i.as_f64().is_some()) => {
            items.iter().collect()
        }
        _ => Vec::new(),
    };
    for n in numbers {
        let v = n.as_f64().expect("number");
        if let Some(min) = prop.minimum
            && v < min
        {
            cx.error(
                path,
                n.span,
                format!("{v} is less than the minimum value {min}"),
            );
        }
        if let Some(max) = prop.maximum
            && v > max
        {
            cx.error(
                path,
                n.span,
                format!("{v} is greater than the maximum value {max}"),
            );
        }
    }
}

fn optional_string(cx: &mut Cx, layer: &Json, key: &str, path: &str) -> Option<String> {
    layer
        .get(key)
        .and_then(|v| cx.string(v, &format!("{path}.{key}")))
}

/// A feature layer's source, source layer and filter.
fn selection(
    cx: &mut Cx,
    json: &Json,
    path: &str,
    id: &str,
    sources: &[std::sync::Arc<Source>],
) -> Option<(String, Option<String>, Option<Filter>)> {
    let errors = cx.diagnostics.errors.len();
    let source = match json.get("source") {
        None => {
            cx.error(path, json.span, "missing required property \"source\"");
            None
        }
        Some(s) => cx.string(s, &format!("{path}.source")),
    };
    let source_layer = optional_string(cx, json, "source-layer", path);
    if let Some(source) = &source {
        let span = json.get("source").expect("present").span;
        match sources.iter().find(|s| s.id == *source).map(|s| &**s) {
            None => cx.error(
                format!("{path}.source"),
                span,
                format!("source \"{source}\" not found"),
            ),
            Some(Source {
                kind: SourceKind::Vector(_),
                ..
            }) if source_layer.is_none() => cx.error(
                path,
                json.span,
                format!("layer \"{id}\" must specify a \"source-layer\""),
            ),
            Some(Source {
                kind: SourceKind::Raster(_) | SourceKind::RasterDem(_) | SourceKind::Image(_),
                ..
            }) => cx.error(
                format!("{path}.source"),
                span,
                format!("layer \"{id}\" requires a vector source"),
            ),
            Some(_) => {}
        }
    }
    let filter = match json.get("filter") {
        None => None,
        Some(f) if f.is_null() => None,
        Some(f) => {
            let filter_path = format!("{path}.filter");
            let mut warnings = Vec::new();
            let filter = Filter::parse_with_state(f, &mut warnings, &cx.state)
                .map_err(|errors| cx.errors_at(&filter_path, errors))
                .ok();
            cx.warnings_at(&filter_path, warnings);
            filter
        }
    };
    (cx.diagnostics.errors.len() == errors)
        .then(|| (source.expect("no errors"), source_layer, filter))
}

/// The source of a raster, hillshade or color-relief layer, which must be
/// a `raster` (`dem: false`) or `raster-dem` source.
fn image_source(
    cx: &mut Cx,
    json: &Json,
    path: &str,
    id: &str,
    sources: &[std::sync::Arc<Source>],
    dem: bool,
) -> Option<String> {
    let Some(value) = json.get("source") else {
        cx.error(path, json.span, "missing required property \"source\"");
        return None;
    };
    let source = cx.string(value, &format!("{path}.source"))?;
    let fits = match sources.iter().find(|s| s.id == source).map(|s| &s.kind) {
        None => {
            cx.error(
                format!("{path}.source"),
                value.span,
                format!("source \"{source}\" not found"),
            );
            return None;
        }
        Some(SourceKind::Raster(_) | SourceKind::Image(_)) => !dem,
        Some(SourceKind::RasterDem(_)) => dem,
        Some(_) => false,
    };
    if !fits {
        let want = if dem { "raster-dem" } else { "raster" };
        cx.error(
            format!("{path}.source"),
            value.span,
            format!("layer \"{id}\" requires a {want} source"),
        );
        return None;
    }
    Some(source)
}

pub(crate) fn parse_layer(
    cx: &mut Cx,
    json: &Json,
    index: usize,
    sources: &[std::sync::Arc<Source>],
    ids: &mut HashSet<String>,
) -> Option<Layer> {
    let path = format!("layers[{index}]");
    let members = cx.object(json, &path)?;
    let errors = cx.diagnostics.errors.len();

    let id = match json.get("id") {
        None => {
            cx.error(&path, json.span, "missing required property \"id\"");
            None
        }
        Some(id) => cx.string(id, &format!("{path}.id")),
    };
    if let Some(id) = &id
        && !ids.insert(id.clone())
    {
        cx.error(
            format!("{path}.id"),
            json.get("id").expect("present").span,
            format!("duplicate layer id \"{id}\""),
        );
    }
    let ty = match json.get("type") {
        None => {
            cx.error(&path, json.span, "missing required property \"type\"");
            None
        }
        Some(t) => match t.as_str().and_then(LayerType::from_name) {
            Some(ty) => Some(ty),
            None => {
                let names: Vec<&str> = LayerType::V8.iter().map(|t| t.name()).collect();
                cx.error(
                    format!("{path}.type"),
                    t.span,
                    format!("expected one of [{}], {t} found", names.join(", ")),
                );
                None
            }
        },
    };
    for m in members {
        if !LAYER_KEYS.contains(&m.key.as_str()) {
            cx.warn(
                format!("{path}.{}", m.key),
                m.key_span,
                format!("unknown property \"{}\"", m.key),
            );
        }
    }
    let minzoom = json
        .get("minzoom")
        .and_then(|z| cx.number_in(z, &format!("{path}.minzoom"), 0.0, 24.0))
        .unwrap_or(0.0);
    let maxzoom = json
        .get("maxzoom")
        .and_then(|z| cx.number_in(z, &format!("{path}.maxzoom"), 0.0, 24.0))
        .filter(|&z| z != 0.0)
        .unwrap_or(f64::INFINITY);
    let (id, ty) = (id?, ty?);

    let layout = section(cx, json, "layout", &path, ty);
    let paint = section(cx, json, "paint", &path, ty);
    let visibility = {
        let slot = property_at(
            cx,
            layout.get(crate::VISIBILITY.name),
            &format!("{}.visibility", layout.path),
            &crate::VISIBILITY,
        );
        slot.as_constant()
            .and_then(Visibility::from_value)
            .unwrap_or(Visibility::Visible)
    };
    let props: Vec<Slot> = ty
        .properties()
        .map(|id| {
            let prop = id.prop();
            let section = if prop.paint { &paint } else { &layout };
            let path = format!("{}.{}", section.path, prop.name);
            property_at(cx, section.get(prop.name), &path, prop)
        })
        .collect();
    let (source, source_layer, filter) = match ty {
        LayerType::Background => (None, None, None),
        LayerType::Raster | LayerType::Hillshade | LayerType::ColorRelief => {
            let dem = ty != LayerType::Raster;
            (image_source(cx, json, &path, &id, sources, dem), None, None)
        }
        _ => match selection(cx, json, &path, &id, sources) {
            Some((source, layer, filter)) => (Some(source), layer, filter),
            None => (None, None, None),
        },
    };
    let transitions = paint
        .members
        .iter()
        .filter_map(|m| {
            let name = m.key.strip_suffix("-transition")?;
            match crate::v8::paint::parse_transition(&m.value) {
                Some(t) => Some((name.to_string(), t)),
                None => {
                    cx.warn(
                        format!("{}.{}", paint.path, m.key),
                        m.key_span,
                        "expected {\"duration\", \"delay\"} in milliseconds",
                    );
                    None
                }
            }
        })
        .collect();
    let mut layer = Layer::new(id, ty, props);
    layer.source = source;
    layer.source_layer = source_layer;
    layer.filter = filter;
    layer.minzoom = minzoom;
    layer.maxzoom = maxzoom;
    layer.visibility = visibility;
    layer.transitions = transitions;
    (cx.diagnostics.errors.len() == errors).then_some(layer)
}
