//! Feature queries in a tile's owner worker (plan §13): full properties
//! and geometry from the worker's decoded `TileFrame`, as GeoJSON, with no
//! re-decode on the render side and no re-encode.

use orb_style::{Encoding, SourceKind};
use orb_tile::{GeometryType, Layer, ValueRef};

use crate::jobs::{TileRequest, TileWorker};

/// Which features of a tile a query wants.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FeatureSelect {
    /// These features of a source layer, by index.
    Indices { layer: String, features: Vec<u32> },
    /// Every feature of a source layer (every layer if `None`) that a v8
    /// filter, as JSON, selects (all of them without one).
    Filter {
        layer: Option<String>,
        filter: Option<String>,
    },
}

/// A query to a tile's owner worker: the tile, as it was loaded, and what
/// to read from it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TileFeature {
    /// The source layer's name.
    pub layer: String,
    /// The feature's index in it.
    pub index: u32,
    /// The feature's id (promoted where the source says), as JSON.
    pub id: Option<String>,
    /// GL JS's `String(id)`: the feature-state key.
    pub state_id: Option<String>,
    /// `"geometry":…,"properties":…`, a GeoJSON feature's members.
    pub body: String,
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
                    }
                })
                .collect::<Vec<_>>()
        };
        Ok(match &request.select {
            FeatureSelect::Indices { layer, features } => match frame.layer(layer) {
                Some(l) => read(l, &mut features.iter().copied()),
                None => Vec::new(),
            },
            FeatureSelect::Filter { layer, filter } => {
                let filter = match filter {
                    Some(json) => {
                        let json = orb_expr::json::parse(json).map_err(|e| e.to_string())?;
                        let mut warnings = Vec::new();
                        Some(orb_expr::Filter::parse(&json, &mut warnings).map_err(|e| {
                            e.iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join("; ")
                        })?)
                    }
                    None => None,
                };
                let zoom = request.tile.zoom;
                frame
                    .layers
                    .iter()
                    .filter(|l| layer.as_deref().is_none_or(|name| l.name() == name))
                    .flat_map(|l| {
                        let selected = match &filter {
                            Some(filter) => filter.select(l, zoom),
                            None => (0..l.len() as u32).collect(),
                        };
                        read(l, &mut selected.into_iter())
                    })
                    .collect()
            }
        })
    }
}

/// GL JS's `String(id)`, the feature-state key of an id.
fn state_key(value: ValueRef<'_>) -> String {
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
