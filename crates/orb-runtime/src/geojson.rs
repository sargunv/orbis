//! A GeoJSON source on the job side: its data tiled (geojson-vt) or, with
//! `cluster: true`, its points clustered (supercluster), and the answers to
//! cluster queries (GL JS `getClusterChildren`, `getClusterLeaves` and
//! `getClusterExpansionZoom`).

use std::sync::Arc;

use orb_expr::{EvaluationContext, Expression, FeatureData};
use orb_style::{ClusterProperty, GeoJsonSource};
use orb_tile::geojson::{self, Aggregate, Clusters, Feature, Member, Properties};
use orb_tile::{TileFrame, Value};

use crate::features::{write_string, write_value};

/// What a cluster query asks of a clustered source.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ClusterQuery {
    /// The cluster's members one zoom in, as GeoJSON features.
    Children(u64),
    /// Up to `limit` of its points, past the first `offset`.
    Leaves { id: u64, limit: u32, offset: u32 },
    /// The zoom at which it breaks apart, as a number.
    ExpansionZoom(u64),
}

/// A cluster query, sent to a worker with any of the source's tiles: the
/// worker that loads it has the source's data.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ClusterRequest {
    pub tile: crate::TileRequest,
    pub query: ClusterQuery,
}

#[derive(Debug)]
pub(crate) enum Index {
    Tiles(geojson::Index),
    Clusters {
        clusters: Clusters,
        /// The source's `maxzoom`: higher tiles overzoom it.
        max_zoom: u8,
    },
}

impl Index {
    pub(crate) fn new(
        features: Vec<Feature>,
        source: &GeoJsonSource,
        plane: Option<Arc<orb_crs::TileMatrixSet>>,
    ) -> Result<Self, String> {
        let options = geojson::Options {
            plane,
            ..source.options.clone()
        };
        let Some(cluster) = &source.cluster else {
            return geojson::Index::new(features, options).map(Self::Tiles);
        };
        if options.plane.is_some() {
            return Err("GeoJSON clusters draw on Web Mercator's plane only".into());
        }
        let aggregate = (!cluster.properties.is_empty())
            .then(|| ExpressionAggregate::new(&cluster.properties))
            .transpose()?
            .map(|a| Arc::new(a) as Arc<dyn Aggregate>);
        Ok(Self::Clusters {
            clusters: Clusters::new(features, cluster.options.clone(), aggregate),
            max_zoom: options.max_zoom,
        })
    }

    /// Tile `z/x/y`, x wrapping; `None` past the source's `maxzoom`.
    pub(crate) fn tile(&mut self, z: u8, x: i32, y: u32) -> Option<Arc<TileFrame>> {
        match self {
            Self::Tiles(index) => index.tile(z, x, y),
            Self::Clusters { clusters, max_zoom } => {
                (z <= *max_zoom).then(|| Arc::new(clusters.tile(z, x, y)))
            }
        }
    }

    /// The answer to a cluster query: a JSON array of GeoJSON features, or
    /// a number.
    pub(crate) fn answer(&self, query: &ClusterQuery) -> Result<String, String> {
        let Self::Clusters { clusters, .. } = self else {
            return Err("the source isn't clustered".into());
        };
        let mut out = String::from("[");
        let push = |out: &mut String, write: &dyn Fn(&mut String)| {
            if out.len() > 1 {
                out.push(',');
            }
            write(out);
        };
        match *query {
            ClusterQuery::Children(id) => {
                for member in clusters.children(id)? {
                    match member {
                        Member::Cluster {
                            id,
                            position,
                            properties,
                        } => push(&mut out, &|out| {
                            write_feature(out, Some(&Value::F64(id as f64)), &properties, position)
                        }),
                        Member::Point(feature) => push(&mut out, &|out| point(out, feature)),
                    }
                }
            }
            ClusterQuery::Leaves { id, limit, offset } => {
                for feature in clusters.leaves(id, limit as usize, offset as usize)? {
                    push(&mut out, &|out| point(out, feature));
                }
            }
            ClusterQuery::ExpansionZoom(id) => return Ok(clusters.expansion_zoom(id)?.to_string()),
        }
        out.push(']');
        Ok(out)
    }
}

/// An app's tile of a source it feeds, given as GeoJSON: a JSON object of
/// source layer names and GeoJSON documents, each tiled at `tile` (on Web
/// Mercator's grid) as a GeoJSON source's data is.
pub(crate) fn host_frame(text: &str, tile: orb_crs::TileIndex) -> Result<TileFrame, String> {
    let doc = orb_expr::json::parse(text).map_err(|e| e.to_string())?;
    let layers = doc
        .as_object()
        .ok_or("an app's GeoJSON tile is an object of source layers")?;
    let mut frame = TileFrame::default();
    for layer in layers {
        let features = orb_style::v8::geojson_features(&layer.value)?;
        let options = geojson::Options {
            max_zoom: tile.z,
            layer: layer.key.clone(),
            ..geojson::Options::default()
        };
        let tiled = geojson::Index::new(features, options)?.tile(tile.z, tile.col as i32, tile.row);
        // The index is gone: the tile is this frame's alone.
        if let Some(tiled) = tiled {
            frame.layers.extend(Arc::unwrap_or_clone(tiled).layers);
        }
    }
    Ok(frame)
}

fn point(out: &mut String, feature: &Feature) {
    let position = match &feature.geometry {
        Some(geojson::Geometry::Points(points)) => points.first().copied(),
        _ => None,
    };
    write_feature(
        out,
        feature.id.as_ref(),
        &feature.properties,
        position.unwrap_or_default(),
    );
}

fn write_feature(
    out: &mut String,
    id: Option<&Value>,
    properties: &Properties,
    [lng, lat]: [f64; 2],
) {
    out.push_str("{\"type\":\"Feature\",");
    if let Some(id) = id {
        out.push_str("\"id\":");
        write_value(out, id.as_ref());
        out.push(',');
    }
    out.push_str("\"properties\":{");
    for (i, (key, value)) in properties.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_string(out, key);
        out.push(':');
        write_value(out, value.as_ref());
    }
    out.push_str("},\"geometry\":{\"type\":\"Point\",\"coordinates\":[");
    out.push_str(&orb_expr::js_number_string(lng));
    out.push(',');
    out.push_str(&orb_expr::js_number_string(lat));
    out.push_str("]}}");
}

/// `clusterProperties` as GL JS evaluates them: `map` on a point's
/// properties; `reduce` on the other's properties, `["accumulated"]` the
/// value so far.
#[derive(Debug)]
struct ExpressionAggregate {
    properties: Vec<(String, Expression, Expression)>,
}

impl ExpressionAggregate {
    fn new(properties: &[ClusterProperty]) -> Result<Self, String> {
        let parse = |json| {
            Expression::parse(json, None).map_err(|e| {
                e.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; ")
            })
        };
        Ok(Self {
            properties: properties
                .iter()
                .map(|p| Ok((p.name.clone(), parse(&p.map)?, parse(&p.reduce)?)))
                .collect::<Result<_, String>>()?,
        })
    }
}

fn data(properties: &[(String, Value)]) -> FeatureData {
    FeatureData {
        id: orb_expr::Value::Null,
        geometry_type: Some(orb_tile::GeometryType::Point),
        properties: properties
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    orb_expr::Value::from_tile(v.as_ref()).into_owned(),
                )
            })
            .collect(),
    }
}

impl Aggregate for ExpressionAggregate {
    fn map(&self, properties: &[(String, Value)]) -> Properties {
        let feature = data(properties);
        let context = EvaluationContext::default();
        self.properties
            .iter()
            .filter_map(|(name, map, _)| {
                let value = map.evaluate_data(&context, &feature).ok()?;
                Some((name.clone(), value.to_tile()?))
            })
            .collect()
    }

    fn reduce(&self, accumulated: &mut Properties, other: &[(String, Value)]) {
        let feature = data(other);
        for (name, _, reduce) in &self.properties {
            let at = accumulated.iter().position(|(k, _)| k == name);
            let so_far = at.map_or(orb_expr::Value::Null, |i| {
                orb_expr::Value::from_tile(accumulated[i].1.as_ref()).into_owned()
            });
            let context = EvaluationContext {
                accumulated: so_far,
                ..EvaluationContext::default()
            };
            let value = reduce
                .evaluate_data(&context, &feature)
                .ok()
                .and_then(|v| v.to_tile());
            match (at, value) {
                (Some(i), Some(v)) => accumulated[i].1 = v,
                (Some(i), None) => {
                    accumulated.remove(i);
                }
                (None, Some(v)) => accumulated.push((name.clone(), v)),
                (None, None) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(json: &str) -> GeoJsonSource {
        let style = orb_style::v8::parse(&format!(
            r#"{{"version":8,"sources":{{"s":{json}}},"layers":[]}}"#
        ))
        .unwrap();
        match &style.sources[0].kind {
            orb_style::SourceKind::GeoJson(s) => s.clone(),
            _ => panic!("a GeoJSON source"),
        }
    }

    #[test]
    fn clusters_sum_their_points_and_answer_queries() {
        let points: Vec<String> = (0..8)
            .map(|i| {
                format!(
                    r#"{{"type":"Feature","id":{i},"properties":{{"n":{i}}},"geometry":{{"type":"Point","coordinates":[{},0]}}}}"#,
                    f64::from(i) * 0.001
                )
            })
            .collect();
        let s = source(&format!(
            r#"{{"type":"geojson","cluster":true,
                "clusterProperties":{{"sum":["+",["get","n"]],"most":[["max",["accumulated"],["get","most"]],["get","n"]]}},
                "data":{{"type":"FeatureCollection","features":[{}]}}}}"#,
            points.join(",")
        ));
        let orb_style::GeoJsonData::Inline(features) = s.data.clone() else {
            panic!()
        };
        let mut index = Index::new(features, &s, None).unwrap();
        let world = index.tile(0, 0, 0).unwrap();
        let layer = &world.layers[0];
        assert_eq!(layer.len(), 1);
        let get = |key: &str| {
            layer
                .property(key)
                .and_then(|c| c.get(0))
                .map(|v| format!("{v:?}"))
        };
        assert_eq!(get("point_count").as_deref(), Some("F64(8.0)"));
        assert_eq!(get("sum").as_deref(), Some("F64(28.0)"));
        assert_eq!(get("most").as_deref(), Some("F64(7.0)"));
        let id = match layer.ids().get(0) {
            Some(orb_tile::ValueRef::F64(id)) => id as u64,
            other => panic!("{other:?}"),
        };
        let leaves = index
            .answer(&ClusterQuery::Leaves {
                id,
                limit: 3,
                offset: 0,
            })
            .unwrap();
        let leaves = orb_expr::json::parse(&leaves).unwrap();
        assert_eq!(leaves.as_array().unwrap().len(), 3);
        let zoom: u8 = index
            .answer(&ClusterQuery::ExpansionZoom(id))
            .unwrap()
            .parse()
            .unwrap();
        assert!(zoom > 0);
        let children = index.answer(&ClusterQuery::Children(id)).unwrap();
        assert!(children.starts_with("[{\"type\":\"Feature\""), "{children}");
        assert!(index.answer(&ClusterQuery::Children(1)).is_err());
        // Past the source's maxzoom, tiles overzoom.
        assert!(index.tile(19, 0, 0).is_none());
    }
}
