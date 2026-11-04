//! Conversion from the shared JSON parser to typed GeoJSON features.

use orb_expr::json::{self, Json, JsonValue};
use orb_tile::{
    Value,
    geojson::{Feature, Geometry},
};

/// Parses GeoJSON with `orb_expr::json`, including geometry/feature collections.
/// Null properties are absent; objects and arrays are kept as JSON text,
/// which expressions read back as arrays and objects.
pub fn parse_geojson(text: &str) -> Result<Vec<Feature>, String> {
    convert(&json::parse(text).map_err(|e| e.to_string())?)
}

pub(crate) fn convert(doc: &Json) -> Result<Vec<Feature>, String> {
    match ty(doc)? {
        "FeatureCollection" => array(doc.get("features"))?.iter().map(feature).collect(),
        "Feature" => Ok(vec![feature(doc)?]),
        _ => Ok(vec![Feature {
            id: None,
            properties: Vec::new(),
            geometry: Some(geometry(doc)?),
        }]),
    }
}

fn ty(doc: &Json) -> Result<&str, String> {
    doc.get("type")
        .and_then(Json::as_str)
        .ok_or_else(|| "GeoJSON object needs a type".into())
}
fn array(value: Option<&Json>) -> Result<&[Json], String> {
    value
        .and_then(Json::as_array)
        .ok_or_else(|| "GeoJSON coordinates or collection must be an array".into())
}
fn value(json: &Json) -> Option<Value> {
    match &json.value {
        JsonValue::Null => None,
        JsonValue::Bool(v) => Some(Value::Bool(*v)),
        JsonValue::Number(v) => Some(Value::F64(*v)),
        JsonValue::String(v) => Some(Value::Str(v.clone())),
        _ => Some(Value::Json(json.to_string())),
    }
}
fn feature(doc: &Json) -> Result<Feature, String> {
    if ty(doc)? != "Feature" {
        return Err("GeoJSON FeatureCollection must contain Features".into());
    }
    let mut properties = Vec::new();
    if let Some(props) = doc.get("properties").filter(|v| !v.is_null()) {
        for m in props
            .as_object()
            .ok_or("GeoJSON properties must be an object or null")?
        {
            if let Some(v) = value(&m.value) {
                properties.push((m.key.clone(), v));
            }
        }
    }
    let id = doc
        .get("id")
        .filter(|v| v.as_str().is_some() || v.as_f64().is_some())
        .and_then(value);
    let geometry = doc
        .get("geometry")
        .filter(|v| !v.is_null())
        .map(geometry)
        .transpose()?;
    Ok(Feature {
        id,
        properties,
        geometry,
    })
}
fn position(doc: &Json) -> Result<[f64; 2], String> {
    let coords = doc.as_array().ok_or("GeoJSON position must be an array")?;
    let number = |i| {
        coords
            .get(i)
            .and_then(Json::as_f64)
            .ok_or_else(|| "GeoJSON position needs two numbers".to_string())
    };
    Ok([number(0)?, number(1)?])
}
fn positions(doc: &Json) -> Result<Vec<[f64; 2]>, String> {
    array(Some(doc))?.iter().map(position).collect()
}
fn lines(doc: &Json) -> Result<Vec<Vec<[f64; 2]>>, String> {
    array(Some(doc))?.iter().map(positions).collect()
}
fn geometry(doc: &Json) -> Result<Geometry, String> {
    let ty = ty(doc)?;
    if ty == "GeometryCollection" {
        return Ok(Geometry::Collection(
            array(doc.get("geometries"))?
                .iter()
                .map(geometry)
                .collect::<Result<_, _>>()?,
        ));
    }
    let c = doc
        .get("coordinates")
        .ok_or("GeoJSON geometry needs coordinates")?;
    // Empty geometries are valid and contribute no features to the index.
    if c.as_array().is_some_and(|a| a.is_empty()) {
        return Ok(Geometry::Collection(vec![]));
    }
    Ok(match ty {
        "Point" => Geometry::Points(vec![position(c)?]),
        "MultiPoint" => Geometry::Points(positions(c)?),
        "LineString" => Geometry::Lines(vec![positions(c)?]),
        "MultiLineString" => Geometry::Lines(lines(c)?),
        "Polygon" => Geometry::Polygons(vec![lines(c)?]),
        "MultiPolygon" => Geometry::Polygons(
            array(Some(c))?
                .iter()
                .map(lines)
                .collect::<Result<_, _>>()?,
        ),
        _ => return Err(format!("invalid GeoJSON geometry type {ty:?}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn property_values_and_null_geometry() {
        let features=parse_geojson(r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":null,"properties":null},{"type":"Feature","id":"a","geometry":{"type":"Point","coordinates":[1,2,3]},"properties":{"n":1,"b":true,"null":null,"array":[1,2],"object":{"a":1}}}]}"#).unwrap();
        assert_eq!(features.len(), 2);
        assert_eq!(features[1].id, Some(Value::Str("a".into())));
        assert_eq!(features[1].properties.len(), 4);
        assert_eq!(features[1].properties[2].1, Value::Json("[1,2]".into()));
        assert_eq!(
            features[1].properties[3].1,
            Value::Json(r#"{"a":1}"#.into())
        );
    }
    #[test]
    fn rejects_bad_geojson() {
        for bad in [
            "{}",
            r#"{"type":"FeatureCollection","features":[{}]}"#,
            r#"{"type":"Point","coordinates":[1]}"#,
        ] {
            assert!(parse_geojson(bad).is_err());
        }
    }
}
