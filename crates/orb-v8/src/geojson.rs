//! Conversion from the shared JSON parser to typed GeoJSON features.

use orb_expr::json::{self, Json, JsonValue};

use crate::doc::set_member;
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

/// `source` with `diffs` applied to its parsed inline features.
pub fn apply_diffs(
    source: &orb_style::Source,
    diffs: &[std::sync::Arc<Json>],
) -> Result<orb_style::Source, String> {
    let mut source = source.clone();
    let orb_style::SourceKind::GeoJson(geojson) = &mut source.kind else {
        return Err(format!(
            "the source \"{}\" is not a GeoJSON source",
            source.id
        ));
    };
    let orb_style::GeoJsonData::Inline(features) = &mut geojson.data else {
        return Err(format!(
            "the source \"{}\" has no inline data to update",
            source.id
        ));
    };
    for diff in diffs {
        Diff::parse(diff)?.apply(features, geojson.options.promote_id.as_deref())?;
    }
    Ok(source)
}

/// GL JS's `GeoJSONSourceDiff` (`updateData`): features removed, added
/// and updated by id (or by the `promoteId` property), applied to a
/// source's data without sending it whole.
#[derive(Clone, Debug)]
pub(crate) struct Diff {
    remove_all: bool,
    remove: Vec<Key>,
    add: Vec<Json>,
    update: Vec<Update>,
}

#[derive(Clone, Debug)]
struct Update {
    id: Key,
    geometry: Option<Json>,
    remove_all_properties: bool,
    remove_properties: Vec<String>,
    set_properties: Vec<(String, Json)>,
}

/// A feature id: a number and the string of it are different features.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    Number(u64),
    String(String),
}

impl Key {
    fn of_json(json: &Json) -> Option<Self> {
        match &json.value {
            JsonValue::Number(n) => Some(Self::Number((n + 0.0).to_bits())),
            JsonValue::String(s) => Some(Self::String(s.clone())),
            _ => None,
        }
    }

    fn of_value(value: &Value) -> Option<Self> {
        match value {
            Value::F64(n) => Some(Self::Number((n + 0.0).to_bits())),
            Value::I64(n) => Some(Self::Number((*n as f64 + 0.0).to_bits())),
            Value::U64(n) => Some(Self::Number((*n as f64 + 0.0).to_bits())),
            Value::Str(s) => Some(Self::String(s.clone())),
            _ => None,
        }
    }
}

/// Features by id in their order, as GL JS's updateable map keeps them: a
/// feature added again keeps its place, a new one goes last.
struct Updateable<T> {
    order: Vec<Option<(Key, T)>>,
    at: std::collections::HashMap<Key, usize>,
}

impl<T> Updateable<T> {
    fn new(features: Vec<T>, key: impl Fn(&T) -> Option<Key>) -> Result<Self, String> {
        let mut map = Self {
            order: Vec::new(),
            at: std::collections::HashMap::new(),
        };
        for feature in features {
            let key = key(&feature).ok_or(
                "updating a GeoJSON source's data needs every feature to have an id \
                 (or its promoteId property)",
            )?;
            map.set(key, feature);
        }
        Ok(map)
    }

    fn set(&mut self, key: Key, feature: T) {
        match self.at.get(&key) {
            Some(&i) => self.order[i] = Some((key, feature)),
            None => {
                self.at.insert(key.clone(), self.order.len());
                self.order.push(Some((key, feature)));
            }
        }
    }

    fn remove(&mut self, key: &Key) {
        if let Some(i) = self.at.remove(key) {
            self.order[i] = None;
        }
    }

    fn get_mut(&mut self, key: &Key) -> Option<&mut T> {
        let i = *self.at.get(key)?;
        self.order[i].as_mut().map(|(_, f)| f)
    }

    fn into_features(self) -> Vec<T> {
        self.order.into_iter().flatten().map(|(_, f)| f).collect()
    }
}

impl Diff {
    /// `{"removeAll": bool, "remove": [ids], "add": [features], "update":
    /// [{"id", "newGeometry", "removeAllProperties", "removeProperties",
    /// "addOrUpdateProperties": [{"key", "value"}]}]}`.
    pub(crate) fn parse(json: &Json) -> Result<Self, String> {
        if json.as_object().is_none() {
            return Err("a GeoJSON diff must be an object".into());
        }
        let list = |key: &str| -> Result<&[Json], String> {
            match json.get(key) {
                None => Ok(&[]),
                Some(v) => v
                    .as_array()
                    .ok_or_else(|| format!("{key} must be an array")),
            }
        };
        let id = |v: &Json| Key::of_json(v).ok_or_else(|| "ids are numbers or strings".to_string());
        let mut update = Vec::new();
        for u in list("update")? {
            let mut remove_properties = Vec::new();
            for p in u
                .get("removeProperties")
                .and_then(Json::as_array)
                .unwrap_or(&[])
            {
                remove_properties.push(p.as_str().ok_or("property names are strings")?.to_string());
            }
            let mut set_properties = Vec::new();
            for p in u
                .get("addOrUpdateProperties")
                .and_then(Json::as_array)
                .unwrap_or(&[])
            {
                let key = p
                    .get("key")
                    .and_then(Json::as_str)
                    .ok_or("properties need a key")?;
                let value = p
                    .get("value")
                    .cloned()
                    .unwrap_or_else(|| Json::new(JsonValue::Null, p.span));
                set_properties.push((key.to_string(), value));
            }
            if let Some(g) = u.get("newGeometry") {
                geometry(g)?;
            }
            update.push(Update {
                id: id(u.get("id").ok_or("an update needs an id")?)?,
                geometry: u.get("newGeometry").cloned(),
                remove_all_properties: u
                    .get("removeAllProperties")
                    .and_then(Json::as_bool)
                    .unwrap_or(false),
                remove_properties,
                set_properties,
            });
        }
        for f in list("add")? {
            feature(f)?;
        }
        Ok(Self {
            remove_all: json
                .get("removeAll")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            remove: list("remove")?.iter().map(id).collect::<Result<_, _>>()?,
            add: list("add")?.to_vec(),
            update,
        })
    }

    fn apply_to<T>(
        &self,
        map: &mut Updateable<T>,
        key: impl Fn(&T) -> Option<Key>,
        convert: impl Fn(&Json) -> Result<T, String>,
        mut update: impl FnMut(&mut T, &Update) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.remove_all {
            map.order.clear();
            map.at.clear();
        }
        for id in &self.remove {
            map.remove(id);
        }
        for json in &self.add {
            let feature = convert(json)?;
            if let Some(id) = key(&feature) {
                map.set(id, feature);
            }
        }
        for u in &self.update {
            if let Some(feature) = map.get_mut(&u.id) {
                update(feature, u)?;
            }
        }
        Ok(())
    }

    /// The diff applied to a source's `data` (a feature collection, or a
    /// feature), as a feature collection.
    pub(crate) fn apply_json(&self, data: &Json, promote_id: Option<&str>) -> Result<Json, String> {
        let features = match data.get("type").and_then(Json::as_str) {
            Some("FeatureCollection") => array(data.get("features"))?.to_vec(),
            Some("Feature") => vec![data.clone()],
            _ => return Err("only a feature collection or a feature can be updated".into()),
        };
        let key = |f: &Json| match promote_id {
            Some(p) => f
                .get("properties")
                .and_then(|ps| ps.get(p))
                .and_then(Key::of_json),
            None => f.get("id").and_then(Key::of_json),
        };
        let mut map = Updateable::new(features, key)?;
        self.apply_to(
            &mut map,
            key,
            |j| Ok(j.clone()),
            |feature, u| {
                let mut members = feature.as_object().map(<[_]>::to_vec).unwrap_or_default();
                if let Some(g) = &u.geometry {
                    set_member(&mut members, "geometry", Some(g.clone()));
                }
                let mut properties = match feature.get("properties").and_then(Json::as_object) {
                    Some(p) if !u.remove_all_properties => p.to_vec(),
                    _ => Vec::new(),
                };
                properties.retain(|m| !u.remove_properties.contains(&m.key));
                for (name, value) in &u.set_properties {
                    set_member(&mut properties, name, Some(value.clone()));
                }
                let properties = Json::new(JsonValue::Object(properties), feature.span);
                set_member(&mut members, "properties", Some(properties));
                *feature = Json::new(JsonValue::Object(members), feature.span);
                Ok(())
            },
        )?;
        let mut collection = Vec::new();
        let kind = Json::new(JsonValue::String("FeatureCollection".into()), data.span);
        set_member(&mut collection, "type", Some(kind));
        let features = Json::new(JsonValue::Array(map.into_features()), data.span);
        set_member(&mut collection, "features", Some(features));
        Ok(Json::new(JsonValue::Object(collection), data.span))
    }

    /// The diff applied to a source's parsed features.
    pub(crate) fn apply(
        &self,
        features: &mut Vec<Feature>,
        promote_id: Option<&str>,
    ) -> Result<(), String> {
        let key = |f: &Feature| match promote_id {
            Some(p) => f
                .properties
                .iter()
                .find(|(k, _)| k == p)
                .and_then(|(_, v)| Key::of_value(v)),
            None => f.id.as_ref().and_then(Key::of_value),
        };
        let mut map = Updateable::new(std::mem::take(features), key)?;
        self.apply_to(&mut map, key, feature, |feature, u| {
            if let Some(g) = &u.geometry {
                feature.geometry = (!g.is_null()).then(|| geometry(g)).transpose()?;
            }
            if u.remove_all_properties {
                feature.properties.clear();
            }
            feature
                .properties
                .retain(|(k, _)| !u.remove_properties.contains(k));
            for (name, json) in &u.set_properties {
                feature.properties.retain(|(k, _)| k != name);
                if let Some(v) = value(json) {
                    feature.properties.push((name.clone(), v));
                }
            }
            Ok(())
        })?;
        *features = map.into_features();
        Ok(())
    }
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
