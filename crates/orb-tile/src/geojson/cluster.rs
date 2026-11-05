//! GeoJSON point clustering, ported from `supercluster` 8.0.1 (ISC, see
//! LICENSE in this directory), as GL JS clusters a `geojson` source with
//! `cluster: true`: a tree of points per zoom, each zoom's points merged
//! within `radius` into clusters of the next zoom down.

use std::sync::Arc;

use super::kdbush::KdBush;
use super::{Feature, Geometry, LAYER_NAME};
use crate::{GeometryType, LayerBuilder, TileFrame, Value};

/// Clustering options, in GL JS's units: tile units of `extent` per tile.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterOptions {
    pub min_zoom: u8,
    /// The last zoom with clusters (GL JS `clusterMaxZoom`, by default the
    /// source's `maxzoom` less one).
    pub max_zoom: u8,
    /// The fewest points a cluster takes (GL JS `clusterMinPoints`).
    pub min_points: usize,
    /// Tile units: GL JS's `clusterRadius` pixels × 16.
    pub radius: f64,
    pub extent: u32,
    pub generate_id: bool,
}

impl Default for ClusterOptions {
    fn default() -> Self {
        Self {
            min_zoom: 0,
            max_zoom: 17,
            min_points: 2,
            radius: 50.0 * 16.0,
            extent: 8192,
            generate_id: false,
        }
    }
}

pub type Properties = Vec<(String, Value)>;

/// A source's `clusterProperties`: what each point contributes, and how a
/// cluster folds its members' contributions together.
pub trait Aggregate: std::fmt::Debug + Send + Sync {
    /// A point's contribution, from its properties.
    fn map(&self, properties: &[(String, Value)]) -> Properties;
    /// Folds `other`, a point's contribution or a cluster's properties,
    /// into `accumulated`.
    fn reduce(&self, accumulated: &mut Properties, other: &[(String, Value)]);
}

// The layout of a point in a zoom's data: position, the zoom it was last
// clustered at, its id (a point's index, or a cluster's id), its parent
// cluster's id, how many points it holds and, with an aggregate, its
// cluster properties' index.
const ZOOM: usize = 2;
const ID: usize = 3;
const PARENT: usize = 4;
const NUM: usize = 5;
const PROP: usize = 6;

#[derive(Debug)]
struct Tree {
    index: KdBush,
    data: Vec<f64>,
}

/// A clustered GeoJSON source.
#[derive(Debug)]
pub struct Clusters {
    options: ClusterOptions,
    features: Vec<Feature>,
    /// By zoom, `min_zoom..=max_zoom + 1`; the last holds every point.
    trees: Vec<Tree>,
    stride: usize,
    aggregate: Option<Arc<dyn Aggregate>>,
    cluster_properties: Vec<Properties>,
}

/// A cluster's or a point's place in the answer to a cluster query.
#[derive(Debug, PartialEq)]
pub enum Member<'a> {
    Cluster {
        id: u64,
        /// Longitude and latitude.
        position: [f64; 2],
        properties: Properties,
    },
    Point(&'a Feature),
}

fn lng_x(lng: f64) -> f64 {
    lng / 360.0 + 0.5
}

fn lat_y(lat: f64) -> f64 {
    let sin = lat.to_radians().sin();
    let y = 0.5 - 0.25 * ((1.0 + sin) / (1.0 - sin)).ln() / std::f64::consts::PI;
    y.clamp(0.0, 1.0)
}

fn x_lng(x: f64) -> f64 {
    (x - 0.5) * 360.0
}

fn y_lat(y: f64) -> f64 {
    let y2 = (180.0 - y * 360.0).to_radians();
    360.0 * y2.exp().atan() / std::f64::consts::PI - 90.0
}

/// A point feature's coordinates; other geometry isn't clustered.
fn position(feature: &Feature) -> Option<[f64; 2]> {
    match &feature.geometry {
        Some(Geometry::Points(points)) if points.len() == 1 => Some(points[0]),
        _ => None,
    }
}

/// JavaScript's `Math.round`: halves go up.
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

impl Clusters {
    pub fn new(
        features: Vec<Feature>,
        options: ClusterOptions,
        aggregate: Option<Arc<dyn Aggregate>>,
    ) -> Self {
        let stride = if aggregate.is_some() { 7 } else { 6 };
        let mut data = Vec::new();
        for (i, feature) in features.iter().enumerate() {
            let Some([lng, lat]) = position(feature) else {
                continue;
            };
            let (x, y) = (f64::from(lng_x(lng) as f32), f64::from(lat_y(lat) as f32));
            data.extend([x, y, f64::INFINITY, i as f64, -1.0, 1.0]);
            if aggregate.is_some() {
                data.push(0.0);
            }
        }
        let mut clusters = Self {
            options,
            features,
            trees: Vec::new(),
            stride,
            aggregate,
            cluster_properties: Vec::new(),
        };
        let (min, max) = (clusters.options.min_zoom, clusters.options.max_zoom);
        let mut trees = vec![clusters.tree(data)];
        for z in (min..=max).rev() {
            let next = clusters.cluster(trees.last_mut().expect("a tree"), z);
            trees.push(clusters.tree(next));
        }
        trees.reverse();
        clusters.trees = trees;
        clusters
    }

    fn tree(&self, data: Vec<f64>) -> Tree {
        let points = data.chunks(self.stride).map(|p| [p[0], p[1]]);
        Tree {
            index: KdBush::new(points, 64),
            data,
        }
    }

    fn tree_at(&self, zoom: u8) -> Option<&Tree> {
        self.trees
            .get(usize::from(zoom.checked_sub(self.options.min_zoom)?))
    }

    /// supercluster's `_cluster`: merges `tree`'s points into the next zoom
    /// down's.
    fn cluster(&mut self, tree: &mut Tree, zoom: u8) -> Vec<f64> {
        let stride = self.stride;
        let r = self.options.radius / (f64::from(self.options.extent) * f64::from(zoom).exp2());
        let zoom = f64::from(zoom);
        let data = &mut tree.data;
        let mut next = Vec::new();
        let count = self.features.len() as f64;
        for i in (0..data.len()).step_by(stride) {
            if data[i + ZOOM] <= zoom {
                continue;
            }
            data[i + ZOOM] = zoom;
            let (x, y) = (data[i], data[i + 1]);
            let neighbours = tree.index.within([x, y], r);
            let origin = data[i + NUM];
            let mut points = origin;
            for &n in &neighbours {
                let k = n as usize * stride;
                if data[k + ZOOM] > zoom {
                    points += data[k + NUM];
                }
            }
            if points > origin && points >= self.options.min_points as f64 {
                let (mut wx, mut wy) = (x * origin, y * origin);
                let mut properties: Option<(Properties, usize)> = None;
                let id = ((i / stride) as f64) * 32.0 + (zoom + 1.0) + count;
                for &n in &neighbours {
                    let k = n as usize * stride;
                    if data[k + ZOOM] <= zoom {
                        continue;
                    }
                    data[k + ZOOM] = zoom;
                    let points2 = data[k + NUM];
                    wx += data[k] * points2;
                    wy += data[k + 1] * points2;
                    data[k + PARENT] = id;
                    if let Some(aggregate) = self.aggregate.clone() {
                        let (accumulated, _) = properties.get_or_insert_with(|| {
                            let own = self.mapped(data, i);
                            (own, self.cluster_properties.len())
                        });
                        let other = self.mapped(data, k);
                        aggregate.reduce(accumulated, &other);
                    }
                }
                data[i + PARENT] = id;
                next.extend([wx / points, wy / points, f64::INFINITY, id, -1.0, points]);
                if self.aggregate.is_some() {
                    let index = match properties {
                        Some((properties, index)) => {
                            self.cluster_properties.push(properties);
                            index as f64
                        }
                        None => -1.0,
                    };
                    next.push(index);
                }
            } else {
                next.extend_from_slice(&data[i..i + stride]);
                if points > 1.0 {
                    for &n in &neighbours {
                        let k = n as usize * stride;
                        if data[k + ZOOM] <= zoom {
                            continue;
                        }
                        data[k + ZOOM] = zoom;
                        next.extend_from_slice(&data[k..k + stride]);
                    }
                }
            }
        }
        next
    }

    /// supercluster's `_map`: a cluster's properties, or a point's
    /// contribution.
    fn mapped(&self, data: &[f64], i: usize) -> Properties {
        if data[i + NUM] > 1.0 {
            return self.cluster_properties[data[i + PROP] as usize].clone();
        }
        let aggregate = self.aggregate.as_ref().expect("an aggregate");
        aggregate.map(&self.features[data[i + ID] as usize].properties)
    }

    /// A cluster's properties, as GL JS writes them: its aggregates, then
    /// `cluster`, `cluster_id`, `point_count` and
    /// `point_count_abbreviated`.
    fn properties(&self, data: &[f64], i: usize) -> Properties {
        let count = data[i + NUM];
        let abbreviated = if count >= 10_000.0 {
            Value::Str(format!("{}k", js_round(count / 1000.0)))
        } else if count >= 1000.0 {
            Value::Str(format!("{}k", js_round(count / 100.0) / 10.0))
        } else {
            Value::F64(count)
        };
        let mut properties = match data.get(i + PROP).filter(|_| self.stride > PROP) {
            Some(&p) if p >= 0.0 => self.cluster_properties[p as usize].clone(),
            _ => Vec::new(),
        };
        properties.extend([
            ("cluster".into(), Value::Bool(true)),
            ("cluster_id".into(), Value::F64(data[i + ID])),
            ("point_count".into(), Value::F64(count)),
            ("point_count_abbreviated".into(), abbreviated),
        ]);
        properties
    }

    /// Tile `z/x/y` (x wrapping around the world): clusters and points as
    /// point features of [`LAYER_NAME`], with a tile of margin.
    pub fn tile(&self, z: u8, x: i32, y: u32) -> TileFrame {
        let mut layer = LayerBuilder::new(LAYER_NAME, self.options.extent);
        let zoom = z.clamp(self.options.min_zoom, self.options.max_zoom + 1);
        if let Some(tree) = self.tree_at(zoom)
            && y < 1u32 << z.min(31)
        {
            let z2 = f64::from(z).exp2();
            let x = f64::from(x.rem_euclid(1i32 << z.min(30)));
            let y = f64::from(y);
            let p = self.options.radius / f64::from(self.options.extent);
            let (top, bottom) = ((y - p) / z2, (y + 1.0 + p) / z2);
            let ids = tree
                .index
                .range([(x - p) / z2, top], [(x + 1.0 + p) / z2, bottom]);
            self.add_tile_features(&mut layer, tree, &ids, (x, y, z2));
            if x == 0.0 {
                let ids = tree.index.range([1.0 - p / z2, top], [1.0, bottom]);
                self.add_tile_features(&mut layer, tree, &ids, (z2, y, z2));
            }
            if x == z2 - 1.0 {
                let ids = tree.index.range([0.0, top], [p / z2, bottom]);
                self.add_tile_features(&mut layer, tree, &ids, (-1.0, y, z2));
            }
        }
        TileFrame {
            layers: vec![layer.finish()],
        }
    }

    fn add_tile_features(
        &self,
        layer: &mut LayerBuilder,
        tree: &Tree,
        ids: &[u32],
        (x, y, z2): (f64, f64, f64),
    ) {
        let extent = f64::from(self.options.extent);
        for &i in ids {
            let k = i as usize * self.stride;
            let data = &tree.data;
            let cluster = data[k + NUM] > 1.0;
            let (properties, px, py, id) = if cluster {
                let id = Some(Value::F64(data[k + ID]));
                (self.properties(data, k), data[k], data[k + 1], id)
            } else {
                let feature = &self.features[data[k + ID] as usize];
                let [lng, lat] = position(feature).expect("a point");
                let id = if self.options.generate_id {
                    Some(Value::F64(data[k + ID]))
                } else {
                    feature.id.clone()
                };
                (feature.properties.clone(), lng_x(lng), lat_y(lat), id)
            };
            layer.begin_feature(None, GeometryType::Point);
            if let Some(id) = id {
                layer.set_id(id);
            }
            for (key, value) in properties {
                layer.set_property(&key, value);
            }
            layer.begin_part();
            layer.begin_ring();
            layer.push_vertex([
                js_round(extent * (px * z2 - x)) as i32,
                js_round(extent * (py * z2 - y)) as i32,
            ]);
        }
    }

    /// The zoom a cluster was made at, and its data's index there.
    fn origin(&self, id: u64) -> Result<(u8, usize), String> {
        let missing = || "No cluster with the specified id.".to_string();
        let offset = id
            .checked_sub(self.features.len() as u64)
            .ok_or_else(missing)?;
        Ok(((offset % 32) as u8, (offset >> 5) as usize))
    }

    /// A cluster's members one zoom in: clusters and points (GL JS
    /// `getClusterChildren`).
    pub fn children(&self, id: u64) -> Result<Vec<Member<'_>>, String> {
        let missing = || "No cluster with the specified id.".to_string();
        let (zoom, origin) = self.origin(id)?;
        let tree = self.tree_at(zoom).ok_or_else(missing)?;
        let data = &tree.data;
        let k = origin * self.stride;
        if k >= data.len() {
            return Err(missing());
        }
        let r =
            self.options.radius / (f64::from(self.options.extent) * (f64::from(zoom) - 1.0).exp2());
        let mut children = Vec::new();
        for n in tree.index.within([data[k], data[k + 1]], r) {
            let k = n as usize * self.stride;
            if data[k + PARENT] != id as f64 {
                continue;
            }
            children.push(if data[k + NUM] > 1.0 {
                Member::Cluster {
                    id: data[k + ID] as u64,
                    position: [x_lng(data[k]), y_lat(data[k + 1])],
                    properties: self.properties(data, k),
                }
            } else {
                Member::Point(&self.features[data[k + ID] as usize])
            });
        }
        if children.is_empty() {
            return Err(missing());
        }
        Ok(children)
    }

    /// Up to `limit` of a cluster's points, past the first `offset` (GL JS
    /// `getClusterLeaves`).
    pub fn leaves(&self, id: u64, limit: usize, offset: usize) -> Result<Vec<&Feature>, String> {
        let mut leaves = Vec::new();
        self.append_leaves(&mut leaves, id, limit, offset, 0)?;
        Ok(leaves)
    }

    fn append_leaves<'a>(
        &'a self,
        leaves: &mut Vec<&'a Feature>,
        id: u64,
        limit: usize,
        offset: usize,
        mut skipped: usize,
    ) -> Result<usize, String> {
        for child in self.children(id)? {
            match child {
                Member::Cluster { id, properties, .. } => {
                    let count = properties
                        .iter()
                        .find(|(k, _)| k == "point_count")
                        .and_then(|(_, v)| match v {
                            Value::F64(n) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0);
                    if skipped + count <= offset {
                        skipped += count;
                    } else {
                        skipped = self.append_leaves(leaves, id, limit, offset, skipped)?;
                    }
                }
                Member::Point(feature) if skipped < offset => {
                    let _ = feature;
                    skipped += 1;
                }
                Member::Point(feature) => leaves.push(feature),
            }
            if leaves.len() == limit {
                break;
            }
        }
        Ok(skipped)
    }

    /// The zoom at which a cluster breaks apart (GL JS
    /// `getClusterExpansionZoom`).
    pub fn expansion_zoom(&self, mut id: u64) -> Result<u8, String> {
        let (zoom, _) = self.origin(id)?;
        let mut zoom = i32::from(zoom) - 1;
        while zoom <= i32::from(self.options.max_zoom) {
            let children = self.children(id)?;
            zoom += 1;
            match children.as_slice() {
                [Member::Cluster { id: only, .. }] => id = *only,
                _ => break,
            }
        }
        Ok(zoom.max(0) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(lng: f64, lat: f64, i: usize) -> Feature {
        Feature {
            id: Some(Value::F64(i as f64)),
            properties: vec![("n".into(), Value::F64(i as f64))],
            geometry: Some(Geometry::Points(vec![[lng, lat]])),
        }
    }

    /// Ten points close together and one far away.
    fn sample() -> Vec<Feature> {
        let mut features: Vec<Feature> = (0..10)
            .map(|i| point(10.0 + i as f64 * 0.01, 20.0, i))
            .collect();
        features.push(point(-100.0, -40.0, 10));
        features
    }

    #[derive(Debug)]
    struct Sum;

    impl Aggregate for Sum {
        fn map(&self, properties: &[(String, Value)]) -> Properties {
            let n = properties
                .iter()
                .find(|(k, _)| k == "n")
                .map(|p| p.1.clone());
            vec![("sum".into(), n.unwrap_or(Value::F64(0.0)))]
        }

        fn reduce(&self, accumulated: &mut Properties, other: &[(String, Value)]) {
            let (Value::F64(a), Some((_, Value::F64(b)))) = (&accumulated[0].1, other.first())
            else {
                return;
            };
            accumulated[0].1 = Value::F64(a + b);
        }
    }

    fn feature_props(frame: &TileFrame) -> Vec<Vec<(String, String)>> {
        let layer = &frame.layers[0];
        (0..layer.len())
            .map(|row| {
                layer
                    .properties()
                    .filter_map(|(k, c)| Some((k.to_string(), format!("{:?}", c.get(row)?))))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn close_points_cluster_until_they_part() {
        let clusters = Clusters::new(sample(), ClusterOptions::default(), Some(Arc::new(Sum)));
        let world = clusters.tile(0, 0, 0);
        let props = feature_props(&world);
        assert_eq!(props.len(), 2, "{props:?}");
        let cluster = props
            .iter()
            .find(|p| p.iter().any(|(k, _)| k == "cluster"))
            .expect("a cluster");
        let get = |k: &str| cluster.iter().find(|p| p.0 == k).map(|p| p.1.clone());
        assert_eq!(get("point_count").as_deref(), Some("F64(10.0)"));
        assert_eq!(get("sum").as_deref(), Some("F64(45.0)"));
        // At the cluster's expansion zoom its children are more than one.
        let id = (0..world.layers[0].len())
            .find_map(|row| match world.layers[0].ids().get(row) {
                Some(crate::ValueRef::F64(id)) if id >= 11.0 => Some(id as u64),
                _ => None,
            })
            .expect("the cluster's id");
        let zoom = clusters.expansion_zoom(id).unwrap();
        assert!(zoom > 0, "{zoom}");
        let mut leaves: Vec<_> = clusters
            .leaves(id, usize::MAX, 0)
            .unwrap()
            .iter()
            .map(|f| match &f.id {
                Some(Value::F64(n)) => *n as usize,
                _ => usize::MAX,
            })
            .collect();
        leaves.sort_unstable();
        assert_eq!(leaves, (0..10).collect::<Vec<_>>());
        assert_eq!(clusters.leaves(id, 3, 8).unwrap().len(), 2);
        assert!(clusters.children(id).unwrap().len() > 1 || zoom > 1);
        assert!(clusters.children(3).is_err());
        // Past the last cluster zoom, the points stand alone.
        assert_eq!(clusters.tile(18, 0, 0).layers[0].len(), 0);
        let (x, y) = (lng_x(10.05) * 262_144.0, lat_y(20.0) * 262_144.0);
        let full = clusters.tile(18, x as i32, y as u32);
        assert_eq!(full.layers[0].len(), 1);
        assert!(
            feature_props(&full)
                .iter()
                .all(|p| p.iter().all(|(k, _)| k != "cluster"))
        );
    }
}
