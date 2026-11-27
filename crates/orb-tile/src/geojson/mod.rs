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

use rustc_hash::FxHashMap as HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
    shape: Arc<Shape>,
    min: [f64; 2],
    max: [f64; 2],
}

impl Projected {
    fn new(id: Option<Value>, tags: Arc<Vec<(String, Value)>>, shape: Shape) -> Self {
        let mut feature = Self {
            id,
            tags,
            shape: Arc::new(shape),
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
        match &*self.shape {
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

/// The tile one zoom out.
fn parent((z, x, y): Key) -> Key {
    (z - 1, x >> 1, y >> 1)
}

#[derive(Debug)]
struct Tile {
    frame: Arc<TileFrame>,
    /// The features its children are cut from, until they are. A request
    /// below takes them, holding the lock while it cuts, so requests for
    /// that subtree wait for its children and requests elsewhere don't.
    source: Mutex<Option<Vec<Projected>>>,
}

/// A reusable spatial index. Initial tiling stops at `index_max_zoom` or
/// `index_max_points`; requests drill down from retained parent geometry,
/// a zoom at a time, from any number of threads at once. Each tile is cut
/// from its parent's features alone, so what it holds doesn't depend on
/// the order tiles are asked for.
#[derive(Debug)]
pub struct Index {
    options: Options,
    tiles: RwLock<HashMap<Key, Arc<Tile>>>,
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
        let index = Self {
            options,
            tiles: RwLock::default(),
        };
        let mut tiles = HashMap::default();
        let mut stack = vec![((0, 0, 0), features)];
        while let Some((key, features)) = stack.pop() {
            let count: usize = features.iter().map(Projected::count).sum();
            let tile = index.made(key, features);
            let deeper =
                key.0 < index.options.index_max_zoom && count > index.options.index_max_points;
            if deeper && let Some(features) = lock(&tile.source).take() {
                stack.extend(index.cut(&features, key));
            }
            tiles.insert(key, Arc::new(tile));
        }
        *write(&index.tiles) = tiles;
        Ok(index)
    }

    /// Returns a tile, wrapping x around the world. Missing tiles are empty.
    /// Requests above `max_zoom` are absent; the runtime overzooms the parent.
    pub fn tile(&self, z: u8, x: i32, y: u32) -> Option<Arc<TileFrame>> {
        if z > self.options.max_zoom || y >= 1u32 << z {
            return None;
        }
        let key = (z, x.rem_euclid(1i32 << z) as u32, y);
        loop {
            // The tile, or the nearest ancestor there is.
            let (at, tile) = {
                let tiles = read(&self.tiles);
                let mut at = key;
                loop {
                    if let Some(tile) = tiles.get(&at) {
                        break (at, Arc::clone(tile));
                    }
                    if at.0 == 0 {
                        return None;
                    }
                    at = parent(at);
                }
            };
            if at == key {
                return Some(Arc::clone(&tile.frame));
            }
            let mut source = lock(&tile.source);
            let Some(features) = source.as_deref() else {
                // Another request cut its children while this one waited,
                // or it has none: it is empty, and so is all below it.
                drop(source);
                let mut child = key;
                while child.0 > at.0 + 1 {
                    child = parent(child);
                }
                if read(&self.tiles).contains_key(&child) {
                    continue;
                }
                return None;
            };
            let children = self
                .cut(features, at)
                .map(|(key, features)| (key, Arc::new(self.made(key, features))));
            // In before the source's lock goes, for the requests it holds,
            // and the features go only once the children are in, so a cut
            // that panics leaves them for the next request to cut again.
            write(&self.tiles).extend(children);
            *source = None;
        }
    }

    /// A tile of `features`, keeping them for its children where it has
    /// any to cut.
    fn made(&self, key: Key, features: Vec<Projected>) -> Tile {
        let frame = Arc::new(tile::create(&features, key, &self.options));
        let keep = key.0 < self.options.max_zoom && !features.is_empty();
        Tile {
            frame,
            source: Mutex::new(keep.then_some(features)),
        }
    }

    /// The features of each of tile `key`'s children.
    fn cut(&self, features: &[Projected], key @ (z, x, y): Key) -> [(Key, Vec<Projected>); 4] {
        let scale = f64::from(1u32 << z);
        let k = 0.5 * self.options.buffer / f64::from(self.options.extent);
        let (left, top) = (f64::from(x), f64::from(y));
        let side = |from: f64, to: f64| {
            clip::clip(
                features,
                (left + from - k) / scale,
                (left + to + k) / scale,
                0,
                &self.options,
            )
        };
        let sides = [side(0.0, 0.5), side(0.5, 1.0)];
        std::array::from_fn(|i| {
            let (col, row) = ((i / 2) as u32, (i % 2) as u32);
            let offset = f64::from(row) * 0.5;
            let child = clip::clip(
                &sides[col as usize],
                (top + offset - k) / scale,
                (top + offset + 0.5 + k) / scale,
                1,
                &self.options,
            );
            ((z + 1, key.1 * 2 + col, key.2 * 2 + row), child)
        })
    }
}

/// The lock's data, whether or not a cut that held it panicked: a cut
/// adds its children and only then drops its features, so a tile a panic
/// left behind still holds them and the next request cuts again.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
