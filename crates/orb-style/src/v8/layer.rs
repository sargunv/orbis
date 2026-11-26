//! Layers: a v8 layer's source and selection, and its properties in the
//! order of the property table ([`PropertyId`]).

use crate::*;

use orb_expr::json::{Json, JsonValue, Member, Span};
use orb_expr::{Diagnostic, Filter, Kind, PropertyExpression, Value};

use crate::v8::{Cx, target};

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
            let transition = m.key.strip_suffix("-transition").filter(|_| paint);
            if !known(&m.key) && !transition.is_some_and(known) {
                cx.warn_unknown(&path, m);
                continue;
            }
            // A property beyond the approved spec warns while its preview
            // is off, and is then left at its default.
            let name = transition.filter(|n| known(n)).unwrap_or(&m.key);
            if let Some(preview) =
                PropertyId::find(ty, paint, name).and_then(|id| id.prop().preview)
            {
                let what = format!("\"{}\"", m.key);
                cx.preview(preview, format!("{path}.{}", m.key), m.key_span, what);
            }
        }
    }
    Section { path, members }
}

/// A property's slot from its JSON (`None`: unset), with diagnostics at
/// `path`; a value that doesn't fit the property draws its default,
/// reported by [`Cx::invalid`].
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
    match value_at(cx, json, path, prop) {
        Ok(slot) => slot,
        Err(diagnostics) => {
            cx.invalid(path, diagnostics);
            property_at(cx, None, path, prop)
        }
    }
}

/// The slot of a property set to `json`, or why the value doesn't fit it.
fn value_at(cx: &mut Cx, json: &Json, path: &str, prop: &Prop) -> Result<Slot, Vec<Diagnostic>> {
    if prop.ramp {
        return PropertyExpression::parse_with_state(json, &prop.spec(), &cx.state)
            .map(|parsed| Slot::expression_any(parsed, true))
            .map_err(|errors| located(path, errors));
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
        return items
            .iter()
            .enumerate()
            .map(|(i, item)| property_at_one(cx, item, &format!("{path}[{i}]"), prop))
            .collect::<Result<_, _>>()
            .map(Slot::list);
    }
    property_at_one(cx, json, path, prop)
}

/// `orb_expr` diagnostics, whose paths are relative to `prefix`, made
/// absolute.
fn located(prefix: &str, diagnostics: Vec<Diagnostic>) -> Vec<Diagnostic> {
    diagnostics
        .into_iter()
        .map(|d| Diagnostic {
            path: format!("{prefix}{}", d.path),
            ..d
        })
        .collect()
}

/// [`property_at`] for a typed property outside a layer.
pub(crate) fn typed_property_at<T: PropertyValue>(
    cx: &mut Cx,
    json: Option<&Json>,
    path: &str,
    prop: &Prop,
) -> crate::v8::Property<T> {
    let default = match prop.default_value() {
        None => T::UNSET.unwrap_or_else(|| panic!("{} has no default", prop.name)),
        Some(value) => T::from_value(&value)
            .unwrap_or_else(|| panic!("{} default has the wrong type", prop.name)),
    };
    crate::v8::Property::from_slot(property_at(cx, json, path, prop), default)
}

fn property_at_one(
    cx: &mut Cx,
    json: &Json,
    path: &str,
    prop: &Prop,
) -> Result<Slot, Vec<Diagnostic>> {
    let parsed =
        PropertyExpression::parse_with_state(json, &prop.spec(), &cx.state).or_else(|errors| {
            // Inputs beyond the spec's that Orbis honours: drawn, with a
            // warning that GL JS and Native reject the style.
            let extended = (prop.orbis_inputs() != prop.spec_inputs())
                .then(|| {
                    let spec = prop.spec_with(prop.orbis_inputs());
                    PropertyExpression::parse_with_state(json, &spec, &cx.state).ok()
                })
                .flatten();
            match extended {
                Some(parsed) => {
                    // Elevation read where it isn't honoured parses but
                    // doesn't vary per point, so it goes unnamed.
                    let beyond = parsed
                        .reads()
                        .intersect(prop.orbis_inputs())
                        .minus(prop.spec_inputs());
                    if !beyond.is_empty() {
                        cx.not_portable(path, json.span, prop, beyond);
                    }
                    Ok(parsed)
                }
                None => Err(errors),
            }
        });
    match parsed {
        Err(errors) => Err(located(path, errors)),
        // Constants are folded to literals, except images, which GL JS
        // leaves to resolve at evaluation with the style's images.
        Ok(parsed) if parsed.kind() == Kind::Constant && !parsed.expression().reads_images() => {
            let value = parsed
                .expression()
                .evaluate(0.0)
                .map(Value::into_owned)
                .unwrap_or(Value::Null);
            let errors = check_constant(prop, json, &value, path);
            if errors.is_empty() {
                Ok(Slot::constant(value, true))
            } else {
                Err(errors)
            }
        }
        Ok(parsed) => {
            // The spec's validator lets data-driven paint read state, but
            // values laid out per tile (patterns, dashes) and values read
            // per feature beyond the spec never see it.
            if parsed.reads().contains(Inputs::STATE)
                && !prop.orbis_inputs().contains(Inputs::STATE)
            {
                let per_tile = if prop.phase() == Phase::Tile {
                    "per tile, "
                } else {
                    ""
                };
                cx.warn(
                    path,
                    json.span,
                    format!(
                        "\"{}\" is evaluated {per_tile}without feature state",
                        prop.name
                    ),
                );
            }
            Ok(Slot::expression(parsed))
        }
    }
}

/// What makes a constant invalid: a value outside the property's enum
/// values or range.
fn check_constant(prop: &Prop, json: &Json, value: &Value<'_>, path: &str) -> Vec<Diagnostic> {
    let mut errors = Vec::new();
    let mut error = |span, message| {
        errors.push(Diagnostic {
            path: path.to_string(),
            span,
            message,
        });
    };
    if let orb_expr::PropertyType::Enum(values) = prop.ty
        && let Some(s) = value.as_str()
        && !values.contains(&s)
    {
        error(
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
            error(n.span, format!("{v} is less than the minimum value {min}"));
        }
        if let Some(max) = prop.maximum
            && v > max
        {
            error(
                n.span,
                format!("{v} is greater than the maximum value {max}"),
            );
        }
    }
    errors
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
    (id, ty): (&str, LayerType),
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
            None => missing_source(cx, path, span, source),
            Some(Source {
                kind: SourceKind::Vector(v),
                ..
            }) if source_layer.is_none() && !v.tileset.host() => cx.error(
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
            let filter = Filter::parse_symbol(f, &mut warnings, &cx.state)
                .map_err(|errors| cx.errors_at(&filter_path, errors))
                .ok();
            cx.warnings_at(&filter_path, warnings);
            filter.and_then(|filter| camera_filter(cx, filter, &filter_path, f.span, (id, ty)))
        }
    };
    (cx.diagnostics.errors.len() == errors)
        .then(|| (source.expect("no errors"), source_layer, filter))
}

/// A filter that reads the camera, which only a symbol layer's may: its
/// symbols decide at placement. Reading `pitch` or `bearing` isn't
/// portable; the distance from the view's centre needs its preview, or
/// the filter's camera conditions are ignored.
fn camera_filter(
    cx: &mut Cx,
    filter: Filter,
    path: &str,
    span: Span,
    (id, ty): (&str, LayerType),
) -> Option<Filter> {
    let Some(camera) = filter.camera() else {
        return Some(filter);
    };
    if ty != LayerType::Symbol {
        // Mapbox GL JS's message for the layer types whose filters don't
        // read the camera.
        let message = format!(
            "[\"{}\"] expression is not supported in a filter for a {} layer with id: {id}",
            camera.operator(),
            ty.name(),
        );
        cx.error(path, span, message);
        return None;
    }
    if camera.reads_center_distance()
        && !cx.preview(
            Preview::DistanceFromCenter,
            path,
            span,
            "the camera part of this filter",
        )
    {
        return Some(filter.without_camera());
    }
    if camera.reads_camera() {
        cx.filter_not_portable(path, span);
    }
    Some(filter)
}

/// A layer names a source the style doesn't have (yet): it draws nothing
/// until one by that name is added, as in MapLibre Native. GL JS refuses
/// the style.
fn missing_source(cx: &mut Cx, path: &str, span: Span, source: &str) {
    cx.warn(
        format!("{path}.source"),
        span,
        format!(
            "source \"{source}\" not found; the layer draws once it is added, which GL JS refuses"
        ),
    );
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
            missing_source(cx, path, value.span, &source);
            return Some(source);
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
    ids: &mut rustc_hash::FxHashSet<String>,
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
            Some(ty) => match ty.preview() {
                Some(preview) if !cx.previews.contains(preview) => {
                    cx.error(
                        format!("{path}.type"),
                        t.span,
                        format!("{t} needs the \"{preview}\" preview, which is off"),
                    );
                    None
                }
                _ => Some(ty),
            },
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
    let target_keys = [target::DESTINATION, target::SOURCE, target::OPERATIONS];
    for m in members {
        if target_keys.contains(&m.key.as_str()) {
            let what = format!("\"{}\"", m.key);
            cx.preview(
                Preview::Targets,
                format!("{path}.{}", m.key),
                m.key_span,
                what,
            );
        } else if !LAYER_KEYS.contains(&m.key.as_str()) {
            cx.warn_unknown(&path, m);
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
    let targets = targets(cx, json, &path, ty);
    let mut props: Vec<Slot> = ty
        .properties()
        .map(|id| {
            let prop = id.prop();
            let section = if prop.paint { &paint } else { &layout };
            let json = (section.get(prop.name))
                .filter(|_| prop.preview.is_none_or(|p| cx.previews.contains(p)));
            // Only a property the layer sets has diagnostics to locate.
            let path = json.map_or_else(String::new, |_| format!("{}.{}", section.path, prop.name));
            property_at(cx, json, &path, prop)
        })
        .collect();
    // A fill drawn through a target has no features: what would vary by
    // feature draws its default.
    if targets.source.is_some() {
        for (id, slot) in ty.properties().zip(&mut props) {
            let per_feature = slot
                .as_expression()
                .is_some_and(|e| matches!(e.kind(), Kind::Source | Kind::Composite));
            if per_feature {
                let section = if id.is_paint() { &paint } else { &layout };
                cx.warn(
                    format!("{}.{}", section.path, id.name()),
                    json.span,
                    "a fill drawn through a target has no features; this draws its default",
                );
                *slot = Slot::constant(Value::Null, false);
            }
        }
    }
    let (source, source_layer, filter) = match ty {
        _ if targets.source.is_some() => {
            for key in ["source", "source-layer", "filter"] {
                if let Some(m) = members.iter().find(|m| m.key == key) {
                    let message = format!("\"{key}\" is ignored where a layer reads a target");
                    cx.warn(format!("{path}.{key}"), m.key_span, message);
                }
            }
            (None, None, None)
        }
        LayerType::ImageProcess => (None, None, None),
        LayerType::Background => (None, None, None),
        LayerType::Raster | LayerType::Hillshade | LayerType::ColorRelief => {
            let dem = ty != LayerType::Raster;
            (image_source(cx, json, &path, &id, sources, dem), None, None)
        }
        _ => match selection(cx, json, &path, (&id, ty), sources) {
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
    layer.layout_group = Some(layout_group(json));
    layer.destination_target = targets.destination;
    layer.source_target = targets.source;
    layer.operations = targets.operations;
    (cx.diagnostics.errors.len() == errors).then_some(layer)
}

/// A layer's render target keys, as it may have them.
#[derive(Default)]
struct Targets {
    destination: Option<String>,
    source: Option<String>,
    operations: Operations,
}

/// The render target keys of a `ty` layer, while their preview is on: a
/// destination on any layer but extrusions and heatmaps, which draw apart,
/// and fills drawn through a target; a source on fills and image-process
/// layers, which need one; operations on image-process layers.
fn targets(cx: &mut Cx, json: &Json, path: &str, ty: LayerType) -> Targets {
    if !cx.previews.contains(Preview::Targets) {
        return Targets::default();
    }
    let ignored = |cx: &mut Cx, key: &str, why: &str| {
        if let Some(value) = json.get(key) {
            cx.warn(
                format!("{path}.{key}"),
                value.span,
                format!("{why}; ignored"),
            );
        }
    };
    let source = match ty {
        LayerType::Fill | LayerType::ImageProcess => {
            optional_string(cx, json, target::SOURCE, path)
        }
        _ => {
            let why = "only fill and image-process layers read a target";
            ignored(cx, target::SOURCE, why);
            None
        }
    };
    if ty == LayerType::ImageProcess && source.is_none() && json.get(target::SOURCE).is_none() {
        let message = format!("missing required property \"{}\"", target::SOURCE);
        cx.error(path, json.span, message);
    }
    let destination = match ty {
        LayerType::FillExtrusion | LayerType::Heatmap => {
            ignored(
                cx,
                target::DESTINATION,
                "extrusions and heatmaps draw apart from targets",
            );
            None
        }
        LayerType::Fill if source.is_some() => {
            ignored(
                cx,
                target::DESTINATION,
                "a fill drawn through a target draws on the map",
            );
            None
        }
        _ => optional_string(cx, json, target::DESTINATION, path),
    };
    let operations = match (ty, json.get(target::OPERATIONS)) {
        (LayerType::ImageProcess, Some(ops)) => {
            target::parse_operations(cx, ops, &format!("{path}.{}", target::OPERATIONS))
        }
        (_, Some(_)) => {
            ignored(
                cx,
                target::OPERATIONS,
                "only image-process layers have operations",
            );
            Operations::default()
        }
        (_, None) => Operations::default(),
    };
    Targets {
        destination,
        source,
        operations,
    }
}

/// A hash of what GL JS's `groupByLayout` keys a layer by: its type,
/// source, source layer, zoom range, filter and layout, objects compared
/// whatever their key order.
fn layout_group(layer: &Json) -> u64 {
    use std::hash::{Hash, Hasher};
    fn hash(json: Option<&Json>, h: &mut impl Hasher) {
        let Some(json) = json else {
            return 0u8.hash(h);
        };
        match &json.value {
            JsonValue::Null => 1u8.hash(h),
            JsonValue::Bool(b) => (2u8, b).hash(h),
            JsonValue::Number(n) => (3u8, n.to_bits()).hash(h),
            JsonValue::String(s) => (4u8, s).hash(h),
            JsonValue::Array(items) => {
                (5u8, items.len()).hash(h);
                for item in items {
                    hash(Some(item), h);
                }
            }
            JsonValue::Object(members) => {
                let mut keys: Vec<&str> = members.iter().map(|m| m.key.as_str()).collect();
                keys.sort_unstable();
                keys.dedup();
                (6u8, keys.len()).hash(h);
                for key in keys {
                    key.hash(h);
                    hash(json.get(key), h);
                }
            }
        }
    }
    let mut h = rustc_hash::FxHasher::default();
    for key in [
        "type",
        "source",
        "source-layer",
        "minzoom",
        "maxzoom",
        "filter",
        "layout",
    ] {
        hash(layer.get(key), &mut h);
    }
    h.finish()
}
