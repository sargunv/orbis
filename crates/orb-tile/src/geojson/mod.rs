//! GeoJSON tiling, ported from `@maplibre/geojson-vt` 6.3.0
//! (f7fce808d7c14810ca805b383d5873b833ebf591). See LICENSE in this directory.
//!
//! JSON is parsed by `orb-style`'s v8 front end using `orb_expr::json`;
//! this module consumes typed features so the tile/expr dependency remains
//! acyclic. Coordinates are longitude/latitude degrees. Output goes
//! straight to `TileFrame`.

mod clip;
mod cluster;
mod convert;
mod kdbush;
mod tile;

pub use cluster::{Aggregate, ClusterOptions, Clusters, Member, Properties};

use std::collections::HashMap;
use std::sync::Arc;

use crate::{TileFrame, Value};

/// The source-layer name used by GL JS's GeoJSON wrapper.
pub const LAYER_NAME: &str = "_geojsonTileLayer";

/// Tiling options, in tile units. Defaults match GL JS's GeoJSONSource:
/// extent 8192, buffer 128 pixels × 16, tolerance .375 pixels × 16.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub max_zoom: u8,
    pub tolerance: f64,
    pub extent: u32,
    pub buffer: f64,
    pub line_metrics: bool,
    pub generate_id: bool,
    pub promote_id: Option<String>,
    pub index_max_zoom: u8,
    pub index_max_points: usize,
    /// The plane to tile in: a one-tile quadtree whose zoom-0 tile is the
    /// index's unit square. `None` is Web Mercator, as GL JS tiles.
    pub plane: Option<std::sync::Arc<orb_crs::TileMatrixSet>>,
    /// The tiles' layer: [`LAYER_NAME`], or a source layer an app's tile
    /// names.
    pub layer: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_zoom: 18,
            tolerance: 6.0,
            extent: 8192,
            buffer: 2048.0,
            line_metrics: false,
            generate_id: false,
            promote_id: None,
            index_max_zoom: 5,
            index_max_points: 100_000,
            plane: None,
            layer: LAYER_NAME.into(),
        }
    }
}

/// GeoJSON geometry. Multi-geometries keep their parts in the same feature;
/// geometry collections expand into features with the parent's id/properties.
#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    Points(Vec<[f64; 2]>),
    Lines(Vec<Vec<[f64; 2]>>),
    Polygons(Vec<Vec<Vec<[f64; 2]>>>),
    Collection(Vec<Geometry>),
}

/// One source feature, before projection. Null geometry is allowed and still
/// occupies an index for `generate_id`.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    pub id: Option<Value>,
    pub properties: Vec<(String, Value)>,
    pub geometry: Option<Geometry>,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    xy: [f64; 2],
    importance: f64,
}

#[derive(Clone, Debug)]
struct Line {
    points: Vec<Point>,
    size: f64,
    start: f64,
    end: f64,
}

#[derive(Clone, Debug)]
enum Shape {
    Points(Vec<Point>),
    Lines(Vec<Line>),
    Polygons(Vec<Vec<Line>>),
}

#[derive(Clone, Debug)]
struct Projected {
    id: Option<Value>,
    tags: Arc<Vec<(String, Value)>>,
    shape: Shape,
    min: [f64; 2],
    max: [f64; 2],
}

impl Projected {
    fn new(id: Option<Value>, tags: Arc<Vec<(String, Value)>>, shape: Shape) -> Self {
        let mut feature = Self {
            id,
            tags,
            shape,
            min: [f64::INFINITY; 2],
            max: [f64::NEG_INFINITY; 2],
        };
        let mut min = feature.min;
        let mut max = feature.max;
        feature.each_point(|p| {
            for axis in 0..2 {
                min[axis] = min[axis].min(p.xy[axis]);
                max[axis] = max[axis].max(p.xy[axis]);
            }
        });
        feature.min = min;
        feature.max = max;
        feature
    }

    fn each_point(&self, mut f: impl FnMut(&Point)) {
        match &self.shape {
            Shape::Points(points) => points.iter().for_each(f),
            Shape::Lines(lines) => lines.iter().flat_map(|l| &l.points).for_each(f),
            Shape::Polygons(polygons) => {
                for p in polygons {
                    for l in p {
                        for point in &l.points {
                            f(point);
                        }
                    }
                }
            }
        }
    }

    fn count(&self) -> usize {
        let mut count = 0;
        self.each_point(|_| count += 1);
        count
    }
}

type Key = (u8, u32, u32);

#[derive(Debug)]
struct Tile {
    source: Option<Vec<Projected>>,
    frame: TileFrame,
}

/// A reusable spatial index. Initial tiling stops at `index_max_zoom` or
/// `index_max_points`; requests drill down from retained parent geometry.
#[derive(Debug)]
pub struct Index {
    options: Options,
    tiles: HashMap<Key, Tile>,
}

impl Index {
    pub fn new(features: Vec<Feature>, options: Options) -> Result<Self, String> {
        if options.max_zoom > 24 {
            return Err("maxZoom should be in the 0-24 range".into());
        }
        if options.promote_id.is_some() && options.generate_id {
            return Err("promoteId and generateId cannot be used together".into());
        }
        if options.extent == 0
            || !options.buffer.is_finite()
            || options.buffer < 0.0
            || !options.tolerance.is_finite()
            || options.tolerance < 0.0
        {
            return Err(
                "extent must be positive; buffer and tolerance must be finite and nonnegative"
                    .into(),
            );
        }
        let features = convert::convert(features, &options);
        let features = clip::wrap(features, &options);
        let mut index = Self {
            options,
            tiles: HashMap::new(),
        };
        if !features.is_empty() {
            index.split(features, (0, 0, 0), None);
        }
        Ok(index)
    }

    /// Returns a tile, wrapping x around the world. Missing tiles are empty.
    /// Requests above `max_zoom` are absent; the runtime overzooms the parent.
    pub fn tile(&mut self, z: u8, x: i32, y: u32) -> Option<TileFrame> {
        if z > self.options.max_zoom || y >= 1u32 << z {
            return None;
        }
        let x = x.rem_euclid(1i32 << z) as u32;
        let key = (z, x, y);
        if !self.tiles.contains_key(&key) {
            let mut parent = key;
            while parent.0 > 0 {
                parent = (parent.0 - 1, parent.1 >> 1, parent.2 >> 1);
                if let Some(tile) = self.tiles.get_mut(&parent) {
                    let features = tile.source.take()?;
                    self.split(features, parent, Some(key));
                    break;
                }
            }
        }
        self.tiles.get(&key).map(|tile| tile.frame.clone())
    }

    fn split(&mut self, features: Vec<Projected>, key: Key, target: Option<Key>) {
        let mut stack = vec![(features, key)];
        while let Some((features, key @ (z, x, y))) = stack.pop() {
            let count: usize = features.iter().map(Projected::count).sum();
            self.tiles.entry(key).or_insert_with(|| Tile {
                frame: tile::create(&features, key, &self.options),
                source: None,
            });
            let stop = z == self.options.max_zoom
                || match target {
                    None => {
                        z >= self.options.index_max_zoom || count <= self.options.index_max_points
                    }
                    Some((tz, tx, ty)) => z == tz || x != tx >> (tz - z) || y != ty >> (tz - z),
                };
            if stop {
                self.tiles.get_mut(&key).expect("inserted").source = Some(features);
                continue;
            }
            if features.is_empty() {
                continue;
            }
            let scale = f64::from(1u32 << z);
            let k = 0.5 * self.options.buffer / f64::from(self.options.extent);
            let x = f64::from(x);
            let y = f64::from(y);
            let left = clip::clip(
                &features,
                (x - k) / scale,
                (x + 0.5 + k) / scale,
                0,
                &self.options,
            );
            let right = clip::clip(
                &features,
                (x + 0.5 - k) / scale,
                (x + 1.0 + k) / scale,
                0,
                &self.options,
            );
            for (side, col) in [(left, key.1 * 2), (right, key.1 * 2 + 1)] {
                for row in 0..2 {
                    let offset = f64::from(row) * 0.5;
                    let child = clip::clip(
                        &side,
                        (y + offset - k) / scale,
                        (y + offset + 0.5 + k) / scale,
                        1,
                        &self.options,
                    );
                    stack.push((child, (z + 1, col, key.2 * 2 + row)));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
