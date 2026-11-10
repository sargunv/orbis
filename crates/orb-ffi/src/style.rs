// Style edits and reads: sources, layers, their properties, images, the
// light, the default transition and the global state, over Orbis's style
// document and its transactions.
//
// MapLibre Native reports a missing or duplicate id as its own status, so
// each command looks the ids up in the document before it edits it.
// Values read back as Native writes them: every number as a double, and a
// color property's color as `["rgba", r, g, b, a]`.

use std::collections::{HashMap, HashSet};

use orb_api::{Op, StyleImage, Transaction};
use orb_expr::json::{self, Json, JsonValue};
use orb_expr::{Color, PropertyType};
use orb_style::{LayerType, PropertyId};

use crate::base::boundary;
use crate::base::{Failure, Result, sized, text};
use crate::delivery::Value;
use crate::map::{MapEntry, command, entry, job, post};
use crate::runtime::State;
use crate::sys::*;

/// What the C API keeps about a map's style beyond Orbis's document.
#[derive(Default)]
pub(crate) struct StyleState {
    /// The images set through the API, as set, for reading back.
    images: HashMap<String, ImageCopy>,
    /// Sources marked volatile.
    volatile: HashSet<String>,
    /// Whether symbols fade as they're placed; `None` is the default, on.
    placement_transitions: Option<bool>,
}

impl StyleState {
    /// Forgets what belonged to the style a new one replaces.
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone)]
struct ImageCopy {
    width: u32,
    height: u32,
    /// Premultiplied, tightly packed rows.
    pixels: Vec<u8>,
    options: mln_style_image_options,
    stretch_x: Vec<mln_image_stretch>,
    stretch_y: Vec<mln_image_stretch>,
}

// SAFETY: the options' stretch pointers are cleared; the copies own theirs.
unsafe impl Send for ImageCopy {}

// MARK: - JSON as Native writes it

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_number(out: &mut String, n: f64) {
    if n.is_finite() {
        out.push_str(&format!("{n:?}"));
    } else {
        out.push_str("null");
    }
}

/// `json` with every number a double, and a color string in a color
/// property as Native's `["rgba", …]`.
fn write_native(out: &mut String, json: &Json, color: bool) {
    match &json.value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(n) => write_number(out, *n),
        JsonValue::String(s) => match color.then(|| Color::parse(s)).flatten() {
            Some(c) => write_color(out, c),
            None => write_string(out, s),
        },
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_native(out, item, false);
            }
            out.push(']');
        }
        JsonValue::Object(members) => {
            out.push('{');
            for (i, m) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(out, &m.key);
                out.push(':');
                write_native(out, &m.value, false);
            }
            out.push('}');
        }
    }
}

fn write_color(out: &mut String, c: Color) {
    let [r, g, b, a] = c.straight();
    out.push_str("[\"rgba\"");
    for v in [
        (f64::from(r) * 255.0).round(),
        (f64::from(g) * 255.0).round(),
        (f64::from(b) * 255.0).round(),
        f64::from(a),
    ] {
        out.push(',');
        write_number(out, v);
    }
    out.push(']');
}

fn native(json: &Json, color: bool) -> String {
    let mut out = String::new();
    write_native(&mut out, json, color);
    out
}

/// A layer as Native writes it: its properties' values as
/// [`write_native`] does.
fn native_layer(layer: &Json) -> String {
    let ty = layer
        .get("type")
        .and_then(Json::as_str)
        .and_then(LayerType::from_name);
    let Some(members) = layer.as_object() else {
        return native(layer, false);
    };
    let mut out = String::from("{");
    for (i, m) in members.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_string(&mut out, &m.key);
        out.push(':');
        let paint = m.key == "paint";
        match (m.value.as_object(), ty) {
            (Some(props), Some(ty)) if paint || m.key == "layout" => {
                out.push('{');
                for (j, p) in props.iter().enumerate() {
                    if j > 0 {
                        out.push(',');
                    }
                    write_string(&mut out, &p.key);
                    out.push(':');
                    write_native(&mut out, &p.value, is_color(ty, paint, &p.key));
                }
                out.push('}');
            }
            _ => write_native(&mut out, &m.value, false),
        }
    }
    out.push('}');
    out
}

fn is_color(ty: LayerType, paint: bool, name: &str) -> bool {
    PropertyId::find(ty, paint, name).is_some_and(|p| p.prop().ty == PropertyType::Color)
}

/// A value from the caller's JSON text.
fn parse_json(text: &str, what: &str) -> Result<Json> {
    json::parse(text).map_err(|e| Failure::argument(format!("{what} is not valid JSON: {e}")))
}

// MARK: - Lookups

fn doc(entry: &MapEntry) -> Result<&orb_api::StyleDoc> {
    entry
        .map
        .style()
        .ok_or_else(|| Failure::state("the style is not loaded"))
}

fn layer<'a>(entry: &'a MapEntry, id: &str) -> Result<&'a Json> {
    doc(entry)?.layer(id).ok_or_else(|| {
        Failure::new(
            MLN_STATUS_NOT_FOUND,
            format!("layer \"{id}\" does not exist"),
        )
    })
}

fn source<'a>(entry: &'a MapEntry, id: &str) -> Result<&'a Json> {
    doc(entry)?.source(id).ok_or_else(|| {
        Failure::new(
            MLN_STATUS_NOT_FOUND,
            format!("source \"{id}\" does not exist"),
        )
    })
}

fn source_type(source: &Json) -> &str {
    source.get("type").and_then(Json::as_str).unwrap_or("")
}

/// Applies `ops` as one edit; a failure is the caller's argument.
fn edit(entry: &mut MapEntry, ops: Vec<Op>) -> Result {
    doc(entry)?;
    let mut tx = Transaction::new();
    for op in ops {
        tx.push(op);
    }
    entry
        .map
        .edit(&tx)
        .map(|_| ())
        .map_err(|e| Failure::new(status_of(&e), e.message()))
}

fn status_of(e: &orb_api::Error) -> mln_status {
    match e {
        orb_api::Error::InvalidState(_) => MLN_STATUS_INVALID_STATE,
        _ => MLN_STATUS_INVALID_ARGUMENT,
    }
}

/// A caller's id: present and UTF-8.
unsafe fn id(view: mln_buffer_view, what: &str) -> Result<String> {
    if view.data.is_null() && view.size != 0 {
        return Err(Failure::argument(format!("{what} is null")));
    }
    // SAFETY: the caller's view.
    let id = unsafe { text(view, what) }?;
    if id.is_empty() {
        return Err(Failure::argument(format!("{what} must not be empty")));
    }
    Ok(id)
}

/// A caller's optional text: null or empty is `None`.
unsafe fn optional(view: mln_buffer_view, what: &str) -> Result<Option<String>> {
    if view.data.is_null() && view.size != 0 {
        return Err(Failure::argument(format!("{what} is null")));
    }
    // SAFETY: the caller's view.
    let t = unsafe { text(view, what) }?;
    Ok((!t.is_empty()).then_some(t))
}

/// A caller's JSON text, which must be there.
unsafe fn json_text(view: mln_buffer_view, what: &str) -> Result<String> {
    if view.data.is_null() {
        return Err(Failure::argument(format!("{what} is null")));
    }
    // SAFETY: the caller's view.
    unsafe { text(view, what) }
}

// MARK: - Global state

command!(
    mln_map_set_global_state_property(name: mln_buffer_view, value: mln_buffer_view),
    false,
    true,
    |state, id| {
        let name = id(name, "property_name")?;
        let value = json_text(value, "value")?;
        Ok(job(move |entry| {
            doc(entry)?;
            let json = parse_json(&value, "the global state value")?;
            let value = (!json.is_null()).then(|| orb_expr::Value::from_json(&json));
            edit(entry, vec![Op::SetGlobalState { key: name, value }])
        }))
    }
);

command!(mln_map_get_global_state(), true, false, |state, id| {
    Ok(|state: &mut State, id: u64| {
        let entry = entry(state, id);
        let mut out = String::from("{");
        if let Some(doc) = entry.map.style() {
            for (i, (key, value)) in doc.global_state().iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(&mut out, key);
                out.push(':');
                out.push_str(&value.to_json_string());
            }
        }
        out.push('}');
        Ok(Value::Text(out))
    })
});

// MARK: - Sources

command!(
    mln_map_add_style_source_json(source_id: mln_buffer_view, json: mln_buffer_view),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        let json = json_text(json, "source_json")?;
        Ok(job(move |entry| add_source(entry, &source_id, &json)))
    }
);

fn add_source(entry: &mut MapEntry, id: &str, json: &str) -> Result {
    if doc(entry)?.source(id).is_some() {
        return Err(Failure::new(
            MLN_STATUS_ALREADY_EXISTS,
            format!("source \"{id}\" already exists"),
        ));
    }
    let source = parse_json(json, &format!("source \"{id}\""))?;
    if source.as_object().is_none() {
        return Err(Failure::argument(format!(
            "source \"{id}\" is not a JSON object"
        )));
    }
    edit(
        entry,
        vec![Op::AddSource {
            id: id.to_string(),
            source,
        }],
    )
}

command!(
    mln_map_remove_style_source(source_id: mln_buffer_view),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        Ok(job(move |entry| {
            source(entry, &source_id)?;
            let user = doc(entry)?.layer_ids().find(|l| {
                doc(entry)
                    .ok()
                    .and_then(|d| d.layer(l))
                    .and_then(|j| j.get("source"))
                    .and_then(Json::as_str)
                    == Some(source_id.as_str())
            });
            if let Some(user) = user {
                return Err(Failure::state(format!(
                    "source \"{source_id}\" is used by layer \"{user}\""
                )));
            }
            edit(
                entry,
                vec![Op::RemoveSource {
                    id: source_id.clone(),
                }],
            )?;
            entry.style.volatile.remove(&source_id);
            Ok(())
        }))
    }
);

command!(
    mln_map_set_style_source_volatile(source_id: mln_buffer_view, is_volatile: bool),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        Ok(job(move |entry| {
            source(entry, &source_id)?;
            if is_volatile {
                entry.style.volatile.insert(source_id);
            } else {
                entry.style.volatile.remove(&source_id);
            }
            Ok(())
        }))
    }
);

/// A source's info and the text it lends.
struct SourceInfo {
    info: mln_style_source_info,
}

/// The texts source infos point at, kept with them.
#[derive(Default)]
struct Lent {
    texts: Vec<String>,
    views: Vec<Box<[mln_buffer_view]>>,
}

// SAFETY: the views point into the texts, which move with them.
unsafe impl Send for Lent {}

impl Lent {
    fn text(&mut self, text: &str) -> mln_buffer_view {
        // A string's bytes stay put as the vector grows.
        let mut text = text.to_string();
        let view = crate::base::lend(&mut text);
        self.texts.push(text);
        view
    }
}

fn source_info(entry: &MapEntry, id: &str, json: &Json, lent: &mut Lent) -> SourceInfo {
    // SAFETY: plain C data.
    let mut info: mln_style_source_info = unsafe { std::mem::zeroed() };
    let ty = source_type(json);
    info.id = lent.text(id);
    info.type_ = match ty {
        "vector" => MLN_STYLE_SOURCE_TYPE_VECTOR,
        "raster" => MLN_STYLE_SOURCE_TYPE_RASTER,
        "raster-dem" => MLN_STYLE_SOURCE_TYPE_RASTER_DEM,
        "geojson" => MLN_STYLE_SOURCE_TYPE_GEO_JSON,
        "image" => MLN_STYLE_SOURCE_TYPE_IMAGE,
        "video" => MLN_STYLE_SOURCE_TYPE_VIDEO,
        _ => MLN_STYLE_SOURCE_TYPE_UNKNOWN,
    };
    info.is_volatile = entry.style.volatile.contains(id);
    info.attribution = lent.text(
        json.get("attribution")
            .and_then(Json::as_str)
            .unwrap_or_default(),
    );
    let url = match ty {
        "geojson" => json.get("data").and_then(Json::as_str),
        _ => json.get("url").and_then(Json::as_str),
    };
    info.url = lent.text(url.unwrap_or_default());
    if let Some(tiles) = json.get("tiles").and_then(Json::as_array) {
        let views: Box<[mln_buffer_view]> = tiles
            .iter()
            .map(|t| lent.text(t.as_str().unwrap_or_default()))
            .collect();
        info.fields |= MLN_STYLE_SOURCE_INFO_TILE_JSON;
        info.tile_json = mln_style_source_tile_info {
            tile_urls: views.as_ptr(),
            tile_url_count: views.len(),
            min_zoom: json.get("minzoom").and_then(Json::as_f64).unwrap_or(0.0),
            max_zoom: json.get("maxzoom").and_then(Json::as_f64).unwrap_or(22.0),
            scheme: if json.get("scheme").and_then(Json::as_str) == Some("tms") {
                MLN_STYLE_TILE_SCHEME_TMS
            } else {
                MLN_STYLE_TILE_SCHEME_XYZ
            },
        };
        lent.views.push(views);
    }
    if let Some([w, s, e, n]) = json.get("bounds").and_then(Json::as_array).and_then(|b| {
        let b: Vec<f64> = b.iter().filter_map(Json::as_f64).collect();
        <[f64; 4]>::try_from(b).ok()
    }) {
        info.fields |= MLN_STYLE_SOURCE_INFO_BOUNDS;
        info.bounds = mln_lat_lng_bounds {
            southwest: mln_lat_lng {
                latitude: s,
                longitude: w,
            },
            northeast: mln_lat_lng {
                latitude: n,
                longitude: e,
            },
        };
    }
    if matches!(ty, "vector" | "raster" | "raster-dem") {
        info.fields |= MLN_STYLE_SOURCE_INFO_TILE_SIZE;
        info.tile_size = json.get("tileSize").and_then(Json::as_f64).unwrap_or(512.0) as u32;
    }
    match ty {
        "vector" => {
            info.fields |= MLN_STYLE_SOURCE_INFO_VECTOR_ENCODING;
            info.vector_encoding = if json.get("encoding").and_then(Json::as_str) == Some("mlt") {
                MLN_STYLE_VECTOR_TILE_ENCODING_MLT
            } else {
                MLN_STYLE_VECTOR_TILE_ENCODING_MVT
            };
        }
        "raster-dem" => {
            info.fields |= MLN_STYLE_SOURCE_INFO_RASTER_ENCODING;
            info.raster_encoding =
                if json.get("encoding").and_then(Json::as_str) == Some("terrarium") {
                    MLN_STYLE_RASTER_DEM_ENCODING_TERRARIUM
                } else {
                    MLN_STYLE_RASTER_DEM_ENCODING_MAPBOX
                };
        }
        "geojson" => {
            info.fields |= MLN_STYLE_SOURCE_INFO_GEO_JSON;
            info.geo_json = geo_json_options(json, lent);
        }
        _ => {}
    }
    SourceInfo { info }
}

fn geo_json_options(json: &Json, lent: &mut Lent) -> mln_geo_json_source_options {
    let number = |key: &str, default: f64| json.get(key).and_then(Json::as_f64).unwrap_or(default);
    let flag = |key: &str| json.get(key).and_then(Json::as_bool).unwrap_or(false);
    let max_zoom = number("maxzoom", 18.0);
    mln_geo_json_source_options {
        size: std::mem::size_of::<mln_geo_json_source_options>() as u32,
        fields: MLN_GEO_JSON_SOURCE_OPTION_MIN_ZOOM
            | MLN_GEO_JSON_SOURCE_OPTION_MAX_ZOOM
            | MLN_GEO_JSON_SOURCE_OPTION_TOLERANCE
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_MAX_ZOOM
            | MLN_GEO_JSON_SOURCE_OPTION_BUFFER
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_RADIUS
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_MIN_POINTS
            | MLN_GEO_JSON_SOURCE_OPTION_LINE_METRICS
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER,
        min_zoom: number("minzoom", 0.0),
        max_zoom,
        tolerance: number("tolerance", 0.375),
        cluster_max_zoom: number("clusterMaxZoom", max_zoom - 1.0),
        cluster_properties: lent.text(
            &json
                .get("clusterProperties")
                .map(|p| native(p, false))
                .unwrap_or_default(),
        ),
        buffer: number("buffer", 128.0) as u32,
        cluster_radius: number("clusterRadius", 50.0) as u32,
        cluster_min_points: number("clusterMinPoints", 2.0) as u64,
        line_metrics: flag("lineMetrics"),
        cluster: flag("cluster"),
    }
}

command!(
    mln_map_get_style_source(source_id: mln_buffer_view),
    true,
    false,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let Some(json) = entry.map.style().and_then(|d| d.source(&source_id)) else {
                return Ok(Value::None);
            };
            let mut lent = Lent::default();
            let info = source_info(entry, &source_id, json, &mut lent).info;
            Ok(Value::array(&[info], Some(Box::new(lent))))
        })
    }
);

command!(mln_map_list_style_sources(), true, false, |state, id| {
    Ok(|state: &mut State, id: u64| {
        let entry = entry(state, id);
        let mut lent = Lent::default();
        let mut infos = Vec::new();
        if let Some(doc) = entry.map.style() {
            for source_id in doc.source_ids() {
                if let Some(json) = doc.source(source_id) {
                    infos.push(source_info(entry, source_id, json, &mut lent).info);
                }
            }
        }
        Ok(Value::array(&infos, Some(Box::new(lent))))
    })
});

// MARK: - Layers

/// The layer types a source type can feed.
fn feeds(source_type: &str, layer_type: &str) -> bool {
    match source_type {
        "vector" | "geojson" => matches!(
            layer_type,
            "fill" | "line" | "symbol" | "circle" | "heatmap" | "fill-extrusion"
        ),
        "raster" | "image" | "video" => layer_type == "raster",
        "raster-dem" => matches!(layer_type, "hillshade" | "color-relief"),
        _ => true,
    }
}

command!(
    mln_map_add_style_layer_json(json: mln_buffer_view, before: mln_buffer_view),
    false,
    true,
    |state, id| {
        let json = json_text(json, "layer_json")?;
        let before = optional(before, "before_layer_id")?;
        Ok(job(move |entry| {
            doc(entry)?;
            let layer = parse_json(&json, "the layer")?;
            let layer_id = layer
                .get("id")
                .and_then(Json::as_str)
                .ok_or_else(|| Failure::argument("the layer has no id"))?
                .to_string();
            if doc(entry)?.layer(&layer_id).is_some() {
                return Err(Failure::new(
                    MLN_STATUS_ALREADY_EXISTS,
                    format!("layer \"{layer_id}\" already exists"),
                ));
            }
            if let Some(before) = &before
                && doc(entry)?.layer(before).is_none()
            {
                return Err(Failure::new(
                    MLN_STATUS_NOT_FOUND,
                    format!("before_layer_id \"{before}\" does not exist"),
                ));
            }
            let layer_type = layer.get("type").and_then(Json::as_str).unwrap_or("");
            if let Some(source_id) = layer.get("source").and_then(Json::as_str) {
                let Some(source) = doc(entry)?.source(source_id) else {
                    return Err(Failure::new(
                        MLN_STATUS_NOT_FOUND,
                        format!(
                            "layer \"{layer_id}\" uses source \"{source_id}\", which does not exist"
                        ),
                    ));
                };
                if !feeds(source_type(source), layer_type) {
                    return Err(Failure::argument(format!(
                        "source \"{source_id}\" does not support {layer_type} layer \"{layer_id}\""
                    )));
                }
            }
            edit(entry, vec![Op::AddLayer { layer, before }])
        }))
    }
);

command!(
    mln_map_remove_style_layer(layer_id: mln_buffer_view),
    false,
    true,
    |state, id| {
        let layer_id = id(layer_id, "layer_id")?;
        Ok(job(move |entry| {
            layer(entry, &layer_id)?;
            edit(entry, vec![Op::RemoveLayer { id: layer_id }])
        }))
    }
);

command!(
    mln_map_move_style_layer(layer_id: mln_buffer_view, before: mln_buffer_view),
    false,
    true,
    |state, id| {
        let layer_id = id(layer_id, "layer_id")?;
        let before = optional(before, "before_layer_id")?;
        Ok(job(move |entry| {
            layer(entry, &layer_id)?;
            if let Some(b) = &before {
                if doc(entry)?.layer(b).is_none() {
                    return Err(Failure::new(
                        MLN_STATUS_NOT_FOUND,
                        format!("before_layer_id \"{b}\" does not exist"),
                    ));
                }
                // Before itself: where it is.
                if *b == layer_id {
                    return Ok(());
                }
            }
            edit(
                entry,
                vec![Op::MoveLayer {
                    id: layer_id,
                    before,
                }],
            )
        }))
    }
);

fn layer_info(id: &str, json: &Json, lent: &mut Lent) -> mln_style_layer_info {
    let text = |key: &str| json.get(key).and_then(Json::as_str).unwrap_or_default();
    mln_style_layer_info {
        id: lent.text(id),
        type_: lent.text(text("type")),
        source_id: lent.text(text("source")),
        source_layer: lent.text(text("source-layer")),
        min_zoom: json
            .get("minzoom")
            .and_then(Json::as_f64)
            .unwrap_or(f64::NEG_INFINITY),
        max_zoom: json
            .get("maxzoom")
            .and_then(Json::as_f64)
            .unwrap_or(f64::INFINITY),
        visibility: if json
            .get("layout")
            .and_then(|l| l.get("visibility"))
            .and_then(Json::as_str)
            == Some("none")
        {
            MLN_STYLE_LAYER_VISIBILITY_NONE
        } else {
            MLN_STYLE_LAYER_VISIBILITY_VISIBLE
        },
    }
}

command!(
    mln_map_get_style_layer(layer_id: mln_buffer_view),
    true,
    false,
    |state, id| {
        let layer_id = id(layer_id, "layer_id")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let Some(json) = entry.map.style().and_then(|d| d.layer(&layer_id)) else {
                return Ok(Value::None);
            };
            let mut lent = Lent::default();
            let info = layer_info(&layer_id, json, &mut lent);
            Ok(Value::array(&[info], Some(Box::new(lent))))
        })
    }
);

command!(mln_map_list_style_layers(), true, false, |state, id| {
    Ok(|state: &mut State, id: u64| {
        let entry = entry(state, id);
        let mut lent = Lent::default();
        let mut infos = Vec::new();
        if let Some(doc) = entry.map.style() {
            for layer_id in doc.layer_ids() {
                if let Some(json) = doc.layer(layer_id) {
                    infos.push(layer_info(layer_id, json, &mut lent));
                }
            }
        }
        Ok(Value::array(&infos, Some(Box::new(lent))))
    })
});

command!(
    mln_map_get_style_layer_json(layer_id: mln_buffer_view),
    true,
    false,
    |state, id| {
        let layer_id = id(layer_id, "layer_id")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            Ok(entry
                .map
                .style()
                .and_then(|d| d.layer(&layer_id))
                .map_or(Value::None, |json| Value::Text(native_layer(json))))
        })
    }
);

/// Where a layer property lives: a root key, or a section's property.
enum Slot {
    Root,
    Paint,
    Layout,
}

fn slot(layer_id: &str, json: &Json, name: &str) -> Result<(Slot, bool)> {
    if matches!(
        name,
        "source" | "source-layer" | "minzoom" | "maxzoom" | "filter"
    ) {
        return Ok((Slot::Root, false));
    }
    let type_name = json.get("type").and_then(Json::as_str).unwrap_or("");
    let ty = LayerType::from_name(type_name).ok_or_else(|| {
        Failure::argument(format!(
            "property \"{name}\" of layer \"{layer_id}\": {type_name} layers have no properties"
        ))
    })?;
    if let Some(p) = PropertyId::find(ty, true, name) {
        return Ok((Slot::Paint, p.prop().ty == PropertyType::Color));
    }
    if let Some(p) = PropertyId::find(ty, false, name) {
        return Ok((Slot::Layout, p.prop().ty == PropertyType::Color));
    }
    if name == "visibility" {
        return Ok((Slot::Layout, false));
    }
    Err(Failure::argument(format!(
        "property \"{name}\" of layer \"{layer_id}\" isn't a property of {type_name} layers"
    )))
}

command!(
    mln_map_set_style_layer_property(
        layer_id: mln_buffer_view,
        name: mln_buffer_view,
        value: mln_buffer_view,
    ),
    false,
    true,
    |state, id| {
        let layer_id = id(layer_id, "layer_id")?;
        let name = id(name, "property_name")?;
        let value = json_text(value, "value")?;
        Ok(job(move |entry| {
            let json = layer(entry, &layer_id)?.clone();
            let (slot, _) = slot(&layer_id, &json, &name)?;
            let parsed = json::parse(&value).map_err(|_| {
                Failure::argument(format!(
                    "property \"{name}\" of layer \"{layer_id}\" is not valid JSON"
                ))
            })?;
            let value = (!parsed.is_null()).then_some(parsed);
            let ops = match slot {
                Slot::Paint => vec![Op::SetPaint {
                    layer: layer_id,
                    name,
                    value,
                }],
                Slot::Layout => vec![Op::SetLayout {
                    layer: layer_id,
                    name,
                    value,
                }],
                Slot::Root => root_ops(entry, &layer_id, &json, &name, value)?,
            };
            edit(entry, ops)
        }))
    }
);

/// The edits that set a layer's root key: zoom range and filter have their
/// own; a source or source layer replaces the layer in place.
fn root_ops(
    entry: &MapEntry,
    layer_id: &str,
    json: &Json,
    name: &str,
    value: Option<Json>,
) -> Result<Vec<Op>> {
    let zoom = |key: &str| json.get(key).and_then(Json::as_f64);
    let number = |v: &Option<Json>| -> Result<Option<f64>> {
        match v {
            None => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or_else(|| {
                Failure::argument(format!(
                    "property \"{name}\" of layer \"{layer_id}\" isn't a number"
                ))
            }),
        }
    };
    Ok(match name {
        "minzoom" => vec![Op::SetZoomRange {
            layer: layer_id.into(),
            min: number(&value)?,
            max: zoom("maxzoom"),
        }],
        "maxzoom" => vec![Op::SetZoomRange {
            layer: layer_id.into(),
            min: zoom("minzoom"),
            max: number(&value)?,
        }],
        "filter" => vec![Op::SetFilter {
            layer: layer_id.into(),
            filter: value,
        }],
        _ => {
            let type_name = json.get("type").and_then(Json::as_str).unwrap_or("");
            if !LayerType::from_name(type_name).is_some_and(LayerType::draws_features)
                && !(name == "source" && json.get("source").is_some())
            {
                return Err(Failure::argument(format!(
                    "{type_name} layer \"{layer_id}\" does not take a {name}"
                )));
            }
            let text = match &value {
                None if name == "source" => {
                    return Err(Failure::argument(format!(
                        "property \"source\" of layer \"{layer_id}\" can't be reset"
                    )));
                }
                None => None,
                Some(v) => Some(
                    v.as_str()
                        .ok_or_else(|| {
                            Failure::argument(format!(
                                "property \"{name}\" of layer \"{layer_id}\" isn't a string"
                            ))
                        })?
                        .to_string(),
                ),
            };
            let mut members = json.as_object().unwrap_or_default().to_vec();
            members.retain(|m| m.key != name);
            if let Some(text) = text.filter(|t| !t.is_empty()) {
                let at = members
                    .iter()
                    .position(|m| m.key == "type")
                    .map_or(members.len(), |i| i + 1);
                let at = if name == "source-layer" {
                    members
                        .iter()
                        .position(|m| m.key == "source")
                        .map_or(at, |i| i + 1)
                } else {
                    at
                };
                members.insert(
                    at,
                    json::Member {
                        key: name.into(),
                        key_span: json::Span::default(),
                        value: Json::new(JsonValue::String(text), json::Span::default()),
                    },
                );
            }
            let layer = Json::new(JsonValue::Object(members), json::Span::default());
            let doc = doc(entry)?;
            let ids: Vec<&str> = doc.layer_ids().collect();
            let before = ids
                .iter()
                .position(|l| *l == layer_id)
                .and_then(|i| ids.get(i + 1))
                .map(|l| (*l).to_string());
            vec![
                Op::RemoveLayer {
                    id: layer_id.into(),
                },
                Op::AddLayer { layer, before },
            ]
        }
    })
}

command!(
    mln_map_get_style_layer_property(layer_id: mln_buffer_view, name: mln_buffer_view),
    true,
    false,
    |state, id| {
        let layer_id = id(layer_id, "layer_id")?;
        let name = id(name, "property_name")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let json = layer(entry, &layer_id)?;
            // A property the layer's type doesn't have is unset.
            let Ok((slot, color)) = slot(&layer_id, json, &name) else {
                return Ok(Value::None);
            };
            let value = match slot {
                Slot::Root => json.get(&name),
                Slot::Paint => json.get("paint").and_then(|p| p.get(&name)),
                Slot::Layout => json.get("layout").and_then(|p| p.get(&name)),
            };
            Ok(value.map_or(Value::None, |v| Value::Text(native(v, color))))
        })
    }
);

// MARK: - Light and transition

/// The style's root member `key`, as its document has it now.
fn root(entry: &MapEntry, key: &str) -> Result<Option<Json>> {
    let doc = doc(entry)?;
    let root = json::parse(&doc.to_json()).map_err(|e| Failure::native(e.to_string()))?;
    Ok(root.get(key).cloned())
}

fn set_light(entry: &mut MapEntry, light: Json, what: &str) -> Result {
    if light.as_object().is_none() {
        return Err(Failure::argument(format!("the {what} is not an object")));
    }
    edit(
        entry,
        vec![Op::SetRoot {
            key: "light".into(),
            value: Some(light),
        }],
    )
    .map_err(|f| Failure::new(f.status, format!("the {what} is invalid: {}", f.message)))
}

command!(
    mln_map_set_style_light_json(light: mln_buffer_view),
    false,
    true,
    |state, id| {
        let light = json_text(light, "light_json")?;
        Ok(job(move |entry| {
            doc(entry)?;
            let light = parse_json(&light, "the style light")?;
            set_light(entry, light, "style light")
        }))
    }
);

command!(
    mln_map_set_style_light_property(name: mln_buffer_view, value: mln_buffer_view),
    false,
    true,
    |state, id| {
        let name = id(name, "property_name")?;
        let value = json_text(value, "value")?;
        Ok(job(move |entry| {
            let what = format!("style light property \"{name}\"");
            let value = parse_json(&value, &what)?;
            let mut members = root(entry, "light")?
                .and_then(|l| l.as_object().map(<[_]>::to_vec))
                .unwrap_or_default();
            members.retain(|m| m.key != name);
            if !value.is_null() {
                members.push(json::Member {
                    key: name.clone(),
                    key_span: json::Span::default(),
                    value,
                });
            }
            set_light(
                entry,
                Json::new(JsonValue::Object(members), json::Span::default()),
                &what,
            )
        }))
    }
);

command!(
    mln_map_get_style_light_property(name: mln_buffer_view),
    true,
    false,
    |state, id| {
        let name = id(name, "property_name")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let color = name == "color";
            Ok(root(entry, "light")?
                .and_then(|l| l.get(&name).cloned())
                .map_or(Value::None, |v| Value::Text(native(&v, color))))
        })
    }
);

command!(
    mln_map_set_style_transition_options(options: *const mln_style_transition_options),
    false,
    true,
    |state, id| {
        if options.is_null() {
            return Err(Failure::argument("options must not be null"));
        }
        if ((*options).size as usize) < std::mem::size_of::<mln_style_transition_options>() {
            return Err(Failure::argument("options.size is too small"));
        }
        let o = *sized(options, "options")?;
        Ok(job(move |entry| {
            let known = MLN_STYLE_TRANSITION_OPTION_DURATION
                | MLN_STYLE_TRANSITION_OPTION_DELAY
                | MLN_STYLE_TRANSITION_OPTION_ENABLE_PLACEMENT_TRANSITIONS;
            if o.fields & !known != 0 {
                return Err(Failure::argument("options.fields has unknown bits"));
            }
            let has = |bit| o.fields & bit != 0;
            for (bit, v, what) in [
                (
                    MLN_STYLE_TRANSITION_OPTION_DURATION,
                    o.duration_ms,
                    "duration_ms",
                ),
                (MLN_STYLE_TRANSITION_OPTION_DELAY, o.delay_ms, "delay_ms"),
            ] {
                if has(bit) && !(v.is_finite() && v >= 0.0) {
                    return Err(Failure::argument(format!(
                        "options.{what} must be finite and not negative"
                    )));
                }
            }
            let mut members = root(entry, "transition")?
                .and_then(|t| t.as_object().map(<[_]>::to_vec))
                .unwrap_or_default();
            for (bit, key, v) in [
                (
                    MLN_STYLE_TRANSITION_OPTION_DURATION,
                    "duration",
                    o.duration_ms,
                ),
                (MLN_STYLE_TRANSITION_OPTION_DELAY, "delay", o.delay_ms),
            ] {
                if has(bit) {
                    members.retain(|m| m.key != key);
                    members.push(json::Member {
                        key: key.into(),
                        key_span: json::Span::default(),
                        value: Json::new(JsonValue::Number(v), json::Span::default()),
                    });
                }
            }
            if has(MLN_STYLE_TRANSITION_OPTION_DURATION) || has(MLN_STYLE_TRANSITION_OPTION_DELAY) {
                edit(
                    entry,
                    vec![Op::SetRoot {
                        key: "transition".into(),
                        value: Some(Json::new(JsonValue::Object(members), json::Span::default())),
                    }],
                )?;
            } else {
                doc(entry)?;
            }
            if has(MLN_STYLE_TRANSITION_OPTION_ENABLE_PLACEMENT_TRANSITIONS) {
                entry.style.placement_transitions = Some(o.enable_placement_transitions);
            }
            Ok(())
        }))
    }
);

command!(
    mln_map_get_style_transition_options(),
    true,
    false,
    |state, id| {
        Ok(|state: &mut State, id: u64| {
            let entry = entry(state, id);
            // SAFETY: plain C data.
            let mut o: mln_style_transition_options = std::mem::zeroed();
            o.size = std::mem::size_of::<mln_style_transition_options>() as u32;
            o.fields = MLN_STYLE_TRANSITION_OPTION_ENABLE_PLACEMENT_TRANSITIONS;
            o.enable_placement_transitions = entry.style.placement_transitions.unwrap_or(true);
            o.duration_ms = 300.0;
            if entry.map.style().is_some()
                && let Some(t) = root(entry, "transition")?
            {
                if let Some(d) = t.get("duration").and_then(Json::as_f64) {
                    o.fields |= MLN_STYLE_TRANSITION_OPTION_DURATION;
                    o.duration_ms = d;
                }
                if let Some(d) = t.get("delay").and_then(Json::as_f64) {
                    o.fields |= MLN_STYLE_TRANSITION_OPTION_DELAY;
                    o.delay_ms = d;
                }
            }
            Ok(Value::of(o))
        })
    }
);

// MARK: - Images

/// Straight RGBA8 rows from premultiplied ones, tightly packed.
fn unpremultiply(pixels: &[u8]) -> Vec<u8> {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            let a = p[3];
            if a == 0 {
                [0, 0, 0, 0]
            } else {
                let c =
                    |v: u8| ((u32::from(v) * 255 + u32::from(a) / 2) / u32::from(a)).min(255) as u8;
                [c(p[0]), c(p[1]), c(p[2]), a]
            }
        })
        .collect()
}

/// The image's rows, tightly packed, after checking its layout.
unsafe fn image_pixels(image: *const mln_premultiplied_rgba8_image) -> Result<(u32, u32, Vec<u8>)> {
    // SAFETY: the caller's pointer.
    let i = unsafe { sized(image, "image") }?;
    if i.width == 0 || i.height == 0 {
        return Err(Failure::argument("the image is empty"));
    }
    let row = i.width as usize * 4;
    if (i.stride as usize) < row {
        return Err(Failure::argument(
            "the image's stride is shorter than a row",
        ));
    }
    let needed = i.stride as usize * (i.height as usize - 1) + row;
    if i.pixels.is_null() || i.byte_length < needed {
        return Err(Failure::argument(
            "the image's pixels are shorter than its rows",
        ));
    }
    // SAFETY: the caller's `byte_length` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(i.pixels, i.byte_length) };
    let mut out = Vec::with_capacity(row * i.height as usize);
    for y in 0..i.height as usize {
        out.extend_from_slice(&bytes[y * i.stride as usize..][..row]);
    }
    Ok((i.width, i.height, out))
}

unsafe fn stretches(
    p: *const mln_image_stretch,
    count: usize,
    extent: u32,
    what: &str,
) -> Result<Vec<mln_image_stretch>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if p.is_null() {
        return Err(Failure::argument(format!("{what} is null")));
    }
    // SAFETY: the caller's `count` stretches.
    let s = unsafe { std::slice::from_raw_parts(p, count) }.to_vec();
    let mut last = 0.0f32;
    for r in &s {
        if !(r.from.is_finite() && r.to.is_finite()) || r.to <= r.from {
            return Err(Failure::argument(format!(
                "{what}: intervals must have a positive width"
            )));
        }
        if r.from < last {
            return Err(Failure::argument(format!(
                "{what}: intervals must be in increasing order without overlap"
            )));
        }
        if r.from < 0.0 || r.to > extent as f32 {
            return Err(Failure::argument(format!(
                "{what}: intervals must lie within the image"
            )));
        }
        last = r.to;
    }
    Ok(s)
}

fn options_json(o: &ImageCopy) -> String {
    let opts = &o.options;
    let has = |bit| opts.fields & bit != 0;
    let mut parts = Vec::new();
    if has(MLN_STYLE_IMAGE_OPTION_PIXEL_RATIO) {
        parts.push(format!("\"pixelRatio\":{}", opts.pixel_ratio));
    }
    if has(MLN_STYLE_IMAGE_OPTION_SDF) {
        parts.push(format!("\"sdf\":{}", opts.sdf));
    }
    let ranges = |s: &[mln_image_stretch]| {
        s.iter()
            .map(|r| format!("[{},{}]", r.from, r.to))
            .collect::<Vec<_>>()
            .join(",")
    };
    if !o.stretch_x.is_empty() {
        parts.push(format!("\"stretchX\":[{}]", ranges(&o.stretch_x)));
    }
    if !o.stretch_y.is_empty() {
        parts.push(format!("\"stretchY\":[{}]", ranges(&o.stretch_y)));
    }
    if has(MLN_STYLE_IMAGE_OPTION_CONTENT) {
        let c = opts.content;
        parts.push(format!(
            "\"content\":[{},{},{},{}]",
            c.left, c.top, c.right, c.bottom
        ));
    }
    let fit = |f: mln_style_image_text_fit| match f {
        MLN_STYLE_IMAGE_TEXT_FIT_STRETCH_ONLY => "stretchOnly",
        MLN_STYLE_IMAGE_TEXT_FIT_PROPORTIONAL => "proportional",
        _ => "stretchOrShrink",
    };
    if has(MLN_STYLE_IMAGE_OPTION_TEXT_FIT_WIDTH) {
        parts.push(format!("\"textFitWidth\":\"{}\"", fit(opts.text_fit_width)));
    }
    if has(MLN_STYLE_IMAGE_OPTION_TEXT_FIT_HEIGHT) {
        parts.push(format!(
            "\"textFitHeight\":\"{}\"",
            fit(opts.text_fit_height)
        ));
    }
    format!("{{{}}}", parts.join(","))
}

command!(
    mln_map_set_style_image(
        image_id: mln_buffer_view,
        image: *const mln_premultiplied_rgba8_image,
        options: *const mln_style_image_options,
    ),
    false,
    true,
    |state, id| {
        let image_id = id(image_id, "image_id")?;
        let (width, height, pixels) = image_pixels(image)?;
        let mut copy = ImageCopy {
            width,
            height,
            pixels,
            // SAFETY: plain C data.
            options: std::mem::zeroed(),
            stretch_x: Vec::new(),
            stretch_y: Vec::new(),
        };
        if !options.is_null() {
            let o = *sized(options, "options")?;
            let known = MLN_STYLE_IMAGE_OPTION_PIXEL_RATIO
                | MLN_STYLE_IMAGE_OPTION_SDF
                | MLN_STYLE_IMAGE_OPTION_STRETCH_X
                | MLN_STYLE_IMAGE_OPTION_STRETCH_Y
                | MLN_STYLE_IMAGE_OPTION_CONTENT
                | MLN_STYLE_IMAGE_OPTION_TEXT_FIT_WIDTH
                | MLN_STYLE_IMAGE_OPTION_TEXT_FIT_HEIGHT
                | MLN_STYLE_IMAGE_OPTION_KEEP_UNUSED;
            if o.fields & !known != 0 {
                return Err(Failure::argument("options.fields has unknown bits"));
            }
            let has = |bit| o.fields & bit != 0;
            if has(MLN_STYLE_IMAGE_OPTION_PIXEL_RATIO)
                && !(o.pixel_ratio.is_finite() && o.pixel_ratio > 0.0)
            {
                return Err(Failure::argument("options.pixel_ratio must be positive"));
            }
            if has(MLN_STYLE_IMAGE_OPTION_STRETCH_X) {
                copy.stretch_x =
                    stretches(o.stretch_x, o.stretch_x_count, width, "options.stretch_x")?;
            }
            if has(MLN_STYLE_IMAGE_OPTION_STRETCH_Y) {
                copy.stretch_y =
                    stretches(o.stretch_y, o.stretch_y_count, height, "options.stretch_y")?;
            }
            if has(MLN_STYLE_IMAGE_OPTION_CONTENT) {
                let c = o.content;
                let ok = [c.left, c.top, c.right, c.bottom]
                    .iter()
                    .all(|v| v.is_finite())
                    && c.left <= c.right
                    && c.top <= c.bottom
                    && c.left >= 0.0
                    && c.top >= 0.0
                    && c.right <= width as f32
                    && c.bottom <= height as f32;
                if !ok {
                    return Err(Failure::argument(
                        "options.content must lie within the image",
                    ));
                }
            }
            for (bit, fit, what) in [
                (
                    MLN_STYLE_IMAGE_OPTION_TEXT_FIT_WIDTH,
                    o.text_fit_width,
                    "text_fit_width",
                ),
                (
                    MLN_STYLE_IMAGE_OPTION_TEXT_FIT_HEIGHT,
                    o.text_fit_height,
                    "text_fit_height",
                ),
            ] {
                if has(bit) && fit > MLN_STYLE_IMAGE_TEXT_FIT_PROPORTIONAL {
                    return Err(Failure::argument(format!("options.{what} names no fit")));
                }
            }
            copy.options = o;
            copy.options.stretch_x = std::ptr::null();
            copy.options.stretch_y = std::ptr::null();
        }
        Ok(job(move |entry| {
            doc(entry)?;
            let image = StyleImage::with_options(
                copy.width,
                copy.height,
                unpremultiply(&copy.pixels),
                &options_json(&copy),
            )
            .map_err(Failure::argument)?;
            if entry.map.has_image(&image_id) {
                entry.map.remove_image(&image_id).map_err(Failure::from)?;
            }
            entry
                .map
                .add_image(&image_id, image)
                .map_err(Failure::from)?;
            entry.style.images.insert(image_id, copy);
            Ok(())
        }))
    }
);

command!(
    mln_map_remove_style_image(image_id: mln_buffer_view),
    false,
    true,
    |state, id| {
        let image_id = id(image_id, "image_id")?;
        Ok(job(move |entry| {
            doc(entry)?;
            if !entry.map.has_image(&image_id) {
                return Err(Failure::new(
                    MLN_STATUS_NOT_FOUND,
                    format!("style image \"{image_id}\" does not exist"),
                ));
            }
            entry.map.remove_image(&image_id).map_err(Failure::from)?;
            entry.style.images.remove(&image_id);
            Ok(())
        }))
    }
);

/// An image's info and the pixels and stretches it points at.
struct ImageKeep {
    _copy: ImageCopy,
}

command!(
    mln_map_get_style_image(image_id: mln_buffer_view),
    true,
    false,
    |state, id| {
        let image_id = id(image_id, "image_id")?;
        Ok(move |state: &mut State, id: u64| {
            let entry = entry(state, id);
            let Some(copy) = entry.style.images.get(&image_id).cloned() else {
                return Ok(Value::None);
            };
            let o = copy.options;
            let has = |bit| o.fields & bit != 0;
            // SAFETY: plain C data.
            let mut info: mln_style_image_info = std::mem::zeroed();
            info.width = copy.width;
            info.height = copy.height;
            info.pixels = crate::base::view(&copy.pixels);
            info.stretch_x = copy.stretch_x.as_ptr();
            info.stretch_x_count = copy.stretch_x.len();
            info.stretch_y = copy.stretch_y.as_ptr();
            info.stretch_y_count = copy.stretch_y.len();
            if has(MLN_STYLE_IMAGE_OPTION_CONTENT) {
                info.fields |= MLN_STYLE_IMAGE_INFO_CONTENT;
                info.content = o.content;
            }
            if has(MLN_STYLE_IMAGE_OPTION_TEXT_FIT_WIDTH) {
                info.fields |= MLN_STYLE_IMAGE_INFO_TEXT_FIT_WIDTH;
                info.text_fit_width = o.text_fit_width;
            }
            if has(MLN_STYLE_IMAGE_OPTION_TEXT_FIT_HEIGHT) {
                info.fields |= MLN_STYLE_IMAGE_INFO_TEXT_FIT_HEIGHT;
                info.text_fit_height = o.text_fit_height;
            }
            info.pixel_ratio = if has(MLN_STYLE_IMAGE_OPTION_PIXEL_RATIO) {
                o.pixel_ratio
            } else {
                1.0
            };
            info.sdf = has(MLN_STYLE_IMAGE_OPTION_SDF) && o.sdf;
            info.keep_unused = has(MLN_STYLE_IMAGE_OPTION_KEEP_UNUSED) && o.keep_unused;
            Ok(Value::array(
                &[info],
                Some(Box::new(ImageKeep { _copy: copy })),
            ))
        })
    }
);

// MARK: - GeoJSON sources

/// A GeoJSON source's JSON with `options` and no data.
unsafe fn geo_json_source(
    options: *const mln_geo_json_source_options,
    attribution: &str,
) -> Result<String> {
    let mut members = vec![
        "\"type\":\"geojson\"".to_string(),
        "\"data\":{\"type\":\"FeatureCollection\",\"features\":[]}".to_string(),
    ];
    if !attribution.is_empty() {
        let mut a = String::from("\"attribution\":");
        write_string(&mut a, attribution);
        members.push(a);
    }
    if !options.is_null() {
        // SAFETY: every sized struct starts with its u32 size.
        if (unsafe { *options.cast::<u32>() } as usize)
            < std::mem::size_of::<mln_geo_json_source_options>()
        {
            return Err(Failure::argument("options.size is too small"));
        }
        // SAFETY: the caller's options, of their full size.
        let o = unsafe { *options };
        let known = MLN_GEO_JSON_SOURCE_OPTION_MIN_ZOOM
            | MLN_GEO_JSON_SOURCE_OPTION_MAX_ZOOM
            | MLN_GEO_JSON_SOURCE_OPTION_TOLERANCE
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_MAX_ZOOM
            | MLN_GEO_JSON_SOURCE_OPTION_BUFFER
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_RADIUS
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_MIN_POINTS
            | MLN_GEO_JSON_SOURCE_OPTION_LINE_METRICS
            | MLN_GEO_JSON_SOURCE_OPTION_CLUSTER;
        if o.fields & !known != 0 {
            return Err(Failure::argument("options.fields has unknown bits"));
        }
        let has = |bit| o.fields & bit != 0;
        let zoom = |v: f64, what: &str, max: f64| -> Result<f64> {
            if !(v.is_finite() && v.fract() == 0.0 && (0.0..=max).contains(&v)) {
                return Err(Failure::argument(format!(
                    "options.{what} must be an integer within [0, {max}]"
                )));
            }
            Ok(v)
        };
        let min = has(MLN_GEO_JSON_SOURCE_OPTION_MIN_ZOOM)
            .then(|| zoom(o.min_zoom, "min_zoom", 24.0))
            .transpose()?;
        let max = has(MLN_GEO_JSON_SOURCE_OPTION_MAX_ZOOM)
            .then(|| zoom(o.max_zoom, "max_zoom", 24.0))
            .transpose()?;
        if min.unwrap_or(0.0) > max.unwrap_or(18.0) {
            return Err(Failure::argument(
                "options.min_zoom must be less than or equal to max_zoom",
            ));
        }
        let mut push = |key: &str, value: String| members.push(format!("\"{key}\":{value}"));
        if let Some(min) = min {
            push("minzoom", format!("{min}"));
        }
        if let Some(max) = max {
            push("maxzoom", format!("{max}"));
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_TOLERANCE) {
            if !(o.tolerance.is_finite() && o.tolerance >= 0.0) {
                return Err(Failure::argument(
                    "options.tolerance must be finite and not negative",
                ));
            }
            push("tolerance", format!("{}", o.tolerance));
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_MAX_ZOOM) {
            push(
                "clusterMaxZoom",
                format!("{}", zoom(o.cluster_max_zoom, "cluster_max_zoom", 30.0)?),
            );
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_BUFFER) {
            if o.buffer > 512 {
                return Err(Failure::argument("options.buffer must be at most 512"));
            }
            push("buffer", format!("{}", o.buffer));
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_RADIUS) {
            if o.cluster_radius > 4095 {
                return Err(Failure::argument(
                    "options.cluster_radius must be at most 4095",
                ));
            }
            push("clusterRadius", format!("{}", o.cluster_radius));
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_CLUSTER_MIN_POINTS) {
            if o.cluster_min_points > u64::from(u32::MAX) && usize::BITS <= 32 {
                return Err(Failure::argument(
                    "options.cluster_min_points must be at most 4294967295",
                ));
            }
            push("clusterMinPoints", format!("{}", o.cluster_min_points));
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_LINE_METRICS) {
            push("lineMetrics", format!("{}", o.line_metrics));
        }
        if has(MLN_GEO_JSON_SOURCE_OPTION_CLUSTER) {
            push("cluster", format!("{}", o.cluster));
        }
        if o.cluster_properties.data.is_null() && o.cluster_properties.size != 0 {
            return Err(Failure::argument("options.cluster_properties is null"));
        }
        // SAFETY: the caller's view.
        let properties = unsafe { text(o.cluster_properties, "options.cluster_properties") }?;
        if !properties.is_empty() {
            let json = json::parse(&properties).map_err(|e| {
                Failure::argument(format!("options.cluster_properties is not valid JSON: {e}"))
            })?;
            let Some(properties) = json.as_object() else {
                return Err(Failure::argument(
                    "options.cluster_properties must contain a JSON object",
                ));
            };
            // Each property is an expression, or a map and a reduce one.
            if properties.iter().any(|m| m.value.as_array().is_none()) {
                return Err(Failure::argument(
                    "the GeoJSON source options' cluster properties must be expressions",
                ));
            }
            push("clusterProperties", json.to_string());
        }
    }
    Ok(format!("{{{}}}", members.join(",")))
}

command!(
    mln_map_add_geo_json_source(
        source_id: mln_buffer_view,
        options: *const mln_geo_json_source_options,
        attribution: mln_buffer_view,
    ),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        let attribution = optional(attribution, "attribution")?.unwrap_or_default();
        let json = geo_json_source(options, &attribution)?;
        Ok(job(move |entry| add_source(entry, &source_id, &json)))
    }
);

/// Checks that `id` names a GeoJSON source.
fn geo_json(entry: &MapEntry, id: &str) -> Result {
    if source_type(source(entry, id)?) != "geojson" {
        return Err(Failure::argument(format!(
            "source \"{id}\" is not a GeoJSON source"
        )));
    }
    Ok(())
}

/// Checks data a clustered source takes: a feature collection of points.
fn check_clustered(data: &Json) -> Result {
    match data.get("type").and_then(Json::as_str) {
        Some("FeatureCollection") => {}
        Some("Feature") => {
            return Err(Failure::argument(
                "a clustered source requires a feature collection, not a single feature",
            ));
        }
        _ => {
            return Err(Failure::argument(
                "a clustered source requires a feature collection, not a bare geometry",
            ));
        }
    }
    let features = data
        .get("features")
        .and_then(Json::as_array)
        .unwrap_or_default();
    for (i, f) in features.iter().enumerate() {
        let kind = f
            .get("geometry")
            .and_then(|g| g.get("type"))
            .and_then(Json::as_str)
            .unwrap_or("null");
        if kind != "Point" {
            let name = match kind {
                "LineString" => "line string",
                "MultiLineString" => "multi line string",
                "Polygon" => "polygon",
                "MultiPolygon" => "multi polygon",
                "MultiPoint" => "multi point",
                "GeometryCollection" => "geometry collection",
                _ => "missing geometry",
            };
            return Err(Failure::argument(format!(
                "a clustered source requires a point geometry on every feature: feature {i} \
                 has a {name}"
            )));
        }
    }
    Ok(())
}

const GEOJSON_TYPES: [&str; 9] = [
    "FeatureCollection",
    "Feature",
    "Point",
    "MultiPoint",
    "LineString",
    "MultiLineString",
    "Polygon",
    "MultiPolygon",
    "GeometryCollection",
];

command!(
    mln_map_set_geo_json_source_data(source_id: mln_buffer_view, data: mln_buffer_view),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        let data = json_text(data, "data")?;
        Ok(job(move |entry| {
            geo_json(entry, &source_id)?;
            if data.contains('\0') {
                return Err(Failure::argument("the GeoJSON data contains a NUL"));
            }
            let json = parse_json(&data, "the GeoJSON data")?;
            if !json
                .get("type")
                .and_then(Json::as_str)
                .is_some_and(|t| GEOJSON_TYPES.contains(&t))
            {
                return Err(Failure::argument("the data isn't GeoJSON"));
            }
            let clustered = source(entry, &source_id)?
                .get("cluster")
                .and_then(Json::as_bool)
                .unwrap_or(false);
            if clustered {
                check_clustered(&json)?;
            }
            edit(
                entry,
                vec![Op::SetSourceData {
                    id: source_id,
                    data: json,
                }],
            )
        }))
    }
);

command!(
    mln_map_set_geo_json_source_url(source_id: mln_buffer_view, url: mln_buffer_view),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        let url = id(url, "url")?;
        Ok(job(move |entry| {
            geo_json(entry, &source_id)?;
            edit(
                entry,
                vec![Op::SetSourceData {
                    id: source_id,
                    data: Json::new(JsonValue::String(url), json::Span::default()),
                }],
            )
        }))
    }
);

command!(
    mln_map_set_geo_json_source_synchronous_tiling(source_id: mln_buffer_view, enabled: bool),
    false,
    true,
    |state, id| {
        let source_id = id(source_id, "source_id")?;
        // Orbis tiles on its job threads and shows new data once its
        // tiles are built, so the hint changes nothing.
        let _ = enabled;
        Ok(job(move |entry| geo_json(entry, &source_id)))
    }
);
