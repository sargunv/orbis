//! Feature queries in a tile's owner worker: full properties
//! and geometry from the worker's decoded `TileFrame`, as GeoJSON, with no
//! re-decode on the render side and no re-encode.

use orb_style::{Encoding, SourceKind};
use orb_tile::{GeometryType, Layer, ValueRef};

use crate::worker::{TileRequest, TileWorker};

/// Which features of a tile a query wants.
#[derive(Clone, Debug, PartialEq, orb_wire::Wire)]
pub enum FeatureSelect {
    /// These features of a source layer, by index, each with the style
    /// layer that drew it, whose `paint` and `layout` the answer gives as
    /// evaluated for it (GL JS's `evaluateProperties`), those a v8 filter,
    /// as JSON, selects when there is one.
    Indices {
        layer: String,
        features: Vec<Drawn>,
        camera: QueryCamera,
        filter: Option<String>,
    },
    /// Every feature of a source layer (every layer if `None`) that a v8
    /// filter, as JSON, selects (all of them without one).
    Filter {
        layer: Option<String>,
        filter: Option<String>,
    },
}

/// A feature a rendered query answers, and what drew it.
#[derive(Clone, Debug, PartialEq, orb_wire::Wire)]
pub struct Drawn {
    /// The feature's index in its source layer.
    pub feature: u32,
    /// The key of the style layer that drew it.
    pub layer: u32,
    /// Its feature state, a JSON object, if it has any.
    pub state: Option<String>,
}

/// The camera a rendered query's layers are evaluated at.
#[derive(Clone, Copy, Debug, PartialEq, orb_wire::Wire)]
pub struct QueryCamera {
    pub zoom: f64,
    pub pitch: f64,
    pub bearing: f64,
}

/// A query to a tile's owner worker: the tile, as it was loaded, and what
/// to read from it.
#[derive(Clone, Debug, PartialEq, orb_wire::Wire)]
pub struct FeatureRequest {
    /// The tile's load request (`nodes` and `rows_only` unused).
    pub tile: TileRequest,
    pub select: FeatureSelect,
    /// The style revision the drawn tile's features were laid out with:
    /// they are read from the source's data as of then, not newer data
    /// a `setSourceData` brought since. `None` for the newest.
    pub revision: Option<u64>,
}

/// One feature read from a tile.
#[derive(Clone, Debug, PartialEq, Eq, orb_wire::Wire)]
pub struct TileFeature {
    /// The source layer's name.
    pub layer: String,
    /// The feature's index in it.
    pub index: u32,
    /// The feature's id (promoted where the source says), as JSON.
    pub id: Option<String>,
    /// GL JS's `String(id)`: the feature-state key.
    pub(crate) state_id: Option<String>,
    /// `"geometry":…,"properties":…`, a GeoJSON feature's members.
    pub body: String,
    /// By the key of a style layer that drew it, that layer's `"paint":…,
    /// "layout":…` members, evaluated for the feature.
    pub evaluated: Vec<(u32, String)>,
}

/// Called once with a feature query's answer, on any thread.
pub type FeaturesDone = Box<dyn FnOnce(Result<Vec<TileFeature>, String>) + Send + 'static>;

impl TileWorker {
    /// Reads features from the tile's cached `TileFrame`, decoding `bytes`
    /// (in `encoding`) when the frame left the cache.
    pub fn features(
        &self,
        bytes: &[u8],
        encoding: Encoding,
        request: &FeatureRequest,
    ) -> Result<Vec<TileFeature>, String> {
        let frame = self.frame_at(bytes, encoding, &request.tile, request.revision)?;
        let tile = request.tile.tile;
        let index = self
            .plan()
            .source(request.tile.source)
            .ok_or("the source was removed")?;
        let def = self.plan().source_def(index);
        let promote = match &def.kind {
            SourceKind::Vector(v) => v.promote_id.as_ref(),
            _ => None,
        };
        let tms = self
            .surface()
            .source(&def.id, matches!(def.kind, SourceKind::GeoJson(_)));
        let read = |layer: &Layer, features: &mut dyn Iterator<Item = u32>| {
            let promoted = promote
                .and_then(|p| p.key(layer.name()))
                .and_then(|key| layer.property(key));
            let ids = promoted.unwrap_or(layer.ids());
            features
                .filter(|&f| (f as usize) < layer.len())
                .map(|f| {
                    let id = ids.get(f as usize);
                    TileFeature {
                        layer: layer.name().to_string(),
                        index: f,
                        id: id.map(|v| {
                            let mut out = String::new();
                            write_value(&mut out, v);
                            out
                        }),
                        state_id: id.map(state_key),
                        body: body(layer, f as usize, tile, tms),
                        evaluated: Vec::new(),
                    }
                })
                .collect::<Vec<_>>()
        };
        Ok(match &request.select {
            FeatureSelect::Indices {
                layer,
                features,
                camera,
                filter,
            } => match frame.layer(layer) {
                Some(l) => {
                    let mut indices: Vec<u32> = features.iter().map(|d| d.feature).collect();
                    indices.sort_unstable();
                    indices.dedup();
                    let canonical = self.canonical(index, tile);
                    // At the tile's zoom, as GL JS and Native filter a
                    // rendered query, not the camera's.
                    if let Some(filter) = filter.as_deref().map(parse_filter).transpose()? {
                        indices = filter.select_rows(l, request.tile.zoom, canonical, indices);
                    }
                    let mut read = read(l, &mut indices.into_iter());
                    let icons = self.icons();
                    let plan = self.plan();
                    for drawn in features {
                        let Ok(at) = read.binary_search_by_key(&drawn.feature, |f| f.index) else {
                            continue;
                        };
                        let feature = &mut read[at];
                        let Some(style_layer) = (plan.layer_keys.iter())
                            .position(|&k| k == drawn.layer)
                            .map(|i| &plan.style.layers[i])
                        else {
                            continue;
                        };
                        if feature.evaluated.iter().any(|(k, _)| *k == drawn.layer) {
                            continue;
                        }
                        let state = drawn.state.as_deref().map_or_else(Vec::new, parse_state);
                        let context = orb_expr::EvaluationContext {
                            zoom: camera.zoom,
                            pitch: camera.pitch,
                            bearing: camera.bearing,
                            feature_state: &state,
                            available_images: icons.names(),
                            ..Default::default()
                        };
                        let members = evaluated(
                            style_layer,
                            &context,
                            (l, drawn.feature as usize, canonical),
                        );
                        feature.evaluated.push((drawn.layer, members));
                    }
                    read
                }
                None => Vec::new(),
            },
            FeatureSelect::Filter { layer, filter } => {
                let filter = filter.as_deref().map(parse_filter).transpose()?;
                let zoom = request.tile.zoom;
                let canonical = self.canonical(index, tile);
                frame
                    .layers
                    .iter()
                    .filter(|l| layer.as_deref().is_none_or(|name| l.name() == name))
                    .flat_map(|l| {
                        let selected = match &filter {
                            Some(filter) => filter.select(l, zoom, canonical),
                            None => (0..l.len() as u32).collect(),
                        };
                        read(l, &mut selected.into_iter())
                    })
                    .collect()
            }
        })
    }
}

/// A query's v8 filter, from its JSON.
fn parse_filter(json: &str) -> Result<orb_expr::Filter, String> {
    let json = orb_expr::json::parse(json).map_err(|e| e.to_string())?;
    orb_expr::Filter::parse(&json, &mut Vec::new()).map_err(|e| {
        e.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    })
}

/// A feature state's JSON object, as values.
fn parse_state(json: &str) -> Vec<(String, orb_expr::Value<'static>)> {
    match orb_expr::json::parse(json).map(|j| orb_expr::Value::from_json(&j)) {
        Ok(orb_expr::Value::Object(members)) => members.into_owned(),
        _ => Vec::new(),
    }
}

/// A style layer's `"paint":{…},"layout":{…}` for one feature, as GL JS's
/// `evaluateProperties` gives a queried feature's layer: each property
/// the style sets, evaluated at the camera with the feature's data and
/// state. Colour ramps, which have no value of their own, are left out.
fn evaluated(
    style_layer: &orb_style::Layer,
    context: &orb_expr::EvaluationContext<'_>,
    feature: (&Layer, usize, Option<orb_expr::CanonicalTile>),
) -> String {
    let mut sections = [String::new(), String::new()];
    for (id, slot) in style_layer.slots() {
        let prop = id.prop();
        if !slot.is_set() || prop.ramp || prop.list {
            continue;
        }
        let value: orb_expr::Value<'static> =
            orb_style::PropertyRef::of(slot, id).evaluate_feature(context, feature);
        let out = &mut sections[usize::from(prop.paint)];
        if !out.is_empty() {
            out.push(',');
        }
        write_string(out, id.name());
        out.push(':');
        out.push_str(&value.to_json_string());
    }
    let [layout, paint] = sections;
    format!("\"paint\":{{{paint}}},\"layout\":{{{layout}}}")
}

/// GL JS's `String(id)`, the feature-state key of an id.
pub(crate) fn state_key(value: ValueRef<'_>) -> String {
    match value {
        ValueRef::Str(s) | ValueRef::Json(s) => s.to_string(),
        ValueRef::Bool(b) => b.to_string(),
        ValueRef::I64(n) => orb_expr::js_number_string(n as f64),
        ValueRef::U64(n) => orb_expr::js_number_string(n as f64),
        ValueRef::F64(n) => orb_expr::js_number_string(n),
    }
}

/// A feature's GeoJSON geometry and properties, in longitude and latitude
/// from the tile's coordinates, as GL JS's `toGeoJSON` gives them: through
/// the tile's matrix set, or `WebMercatorQuad`'s formula without one.
fn body(
    layer: &Layer,
    feature: usize,
    tile: orb_crs::TileIndex,
    tms: Option<&orb_crs::TileMatrixSet>,
) -> String {
    let n = f64::from(1u32 << tile.z);
    let extent = f64::from(layer.extent());
    let to_world = tms.map(|tms| (tms, tms.data_to_world(tile, layer.extent())));
    let point = |out: &mut String, p: [i32; 2]| {
        let (lon, lat) = match &to_world {
            Some((tms, a)) => {
                let world = a.transform_point2(glam::DVec2::new(f64::from(p[0]), f64::from(p[1])));
                let g = tms.projection().inverse(world);
                (g.lon, g.lat)
            }
            None => {
                let x = (f64::from(tile.col) + f64::from(p[0]) / extent) / n;
                let y = (f64::from(tile.row) + f64::from(p[1]) / extent) / n;
                let lat = (std::f64::consts::PI * (1.0 - 2.0 * y))
                    .sinh()
                    .atan()
                    .to_degrees();
                (x * 360.0 - 180.0, lat)
            }
        };
        out.push('[');
        out.push_str(&orb_expr::js_number_string(lon));
        out.push(',');
        out.push_str(&orb_expr::js_number_string(lat));
        out.push(']');
    };
    let ring = |out: &mut String, ring: &[[i32; 2]], close: bool| {
        out.push('[');
        for (i, p) in ring.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            point(out, *p);
        }
        if close && let Some(first) = ring.first() {
            out.push(',');
            point(out, *first);
        }
        out.push(']');
    };
    let geometry = layer.geometry(feature);
    let parts: Vec<_> = geometry.parts().collect();
    let mut out = String::from("\"geometry\":{\"type\":");
    match layer.geometry_type(feature) {
        GeometryType::Point => {
            let points: Vec<[i32; 2]> = parts
                .iter()
                .flat_map(|p| p.rings())
                .flatten()
                .copied()
                .collect();
            if points.len() == 1 {
                out.push_str("\"Point\",\"coordinates\":");
                point(&mut out, points[0]);
            } else {
                out.push_str("\"MultiPoint\",\"coordinates\":");
                ring(&mut out, &points, false);
            }
        }
        GeometryType::LineString => {
            let lines: Vec<&[[i32; 2]]> = parts.iter().filter_map(|p| p.rings().next()).collect();
            if lines.len() == 1 {
                out.push_str("\"LineString\",\"coordinates\":");
                ring(&mut out, lines[0], false);
            } else {
                out.push_str("\"MultiLineString\",\"coordinates\":[");
                for (i, line) in lines.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    ring(&mut out, line, false);
                }
                out.push(']');
            }
        }
        GeometryType::Polygon => {
            let polygon = |out: &mut String, part: &orb_tile::Part<'_>| {
                out.push('[');
                for (i, r) in part.rings().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    ring(out, r, true);
                }
                out.push(']');
            };
            if parts.len() == 1 {
                out.push_str("\"Polygon\",\"coordinates\":");
                polygon(&mut out, &parts[0]);
            } else {
                out.push_str("\"MultiPolygon\",\"coordinates\":[");
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    polygon(&mut out, part);
                }
                out.push(']');
            }
        }
    }
    out.push_str("},\"properties\":{");
    let mut first = true;
    for (name, column) in layer.properties() {
        let Some(value) = column.get(feature) else {
            continue;
        };
        if !first {
            out.push(',');
        }
        first = false;
        write_string(&mut out, name);
        out.push(':');
        write_value(&mut out, value);
    }
    out.push('}');
    out
}

/// A property value as JSON. Nested GeoJSON values are JSON already.
pub(crate) fn write_value(out: &mut String, value: ValueRef<'_>) {
    match value {
        ValueRef::Bool(b) => out.push_str(if b { "true" } else { "false" }),
        ValueRef::I64(n) => out.push_str(&n.to_string()),
        ValueRef::U64(n) => out.push_str(&n.to_string()),
        ValueRef::F64(n) if n.is_finite() => out.push_str(&orb_expr::js_number_string(n)),
        ValueRef::F64(_) => out.push_str("null"),
        ValueRef::Str(s) => write_string(out, s),
        ValueRef::Json(json) => out.push_str(json),
    }
}

/// A JSON string literal.
pub(crate) fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                out.push_str("\\u00");
                out.push(HEX[(c as usize) >> 4] as char);
                out.push(HEX[(c as usize) & 15] as char);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_escape() {
        let mut out = String::new();
        write_string(&mut out, "a\"b\\c\nd\u{1}");
        assert_eq!(out, r#""a\"b\\c\nd\u0001""#);
    }
}
