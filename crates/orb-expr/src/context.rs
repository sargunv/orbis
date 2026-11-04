//! Explicit inputs shared by every expression backend.
use crate::{BUILTIN_LOCALE, Locale, Value};

/// Evaluation inputs that vary independently of tile properties.
#[derive(Debug)]
pub struct EvaluationContext<'a> {
    pub zoom: f64,
    pub line_progress: f64,
    pub heatmap_density: f64,
    pub elevation: f64,
    pub accumulated: Value<'a>,
    pub feature_state: &'a [(String, Value<'static>)],
    pub global_state: &'a [(String, Value<'static>)],
    pub available_images: &'a [String],
    pub geometry: Option<&'a FeatureGeometry>,
    pub locale: &'a dyn Locale,
    pub is_supported_script: Option<fn(&str) -> bool>,
}
impl Default for EvaluationContext<'_> {
    fn default() -> Self {
        Self {
            zoom: 0.0,
            line_progress: 0.0,
            heatmap_density: 0.0,
            elevation: 0.0,
            accumulated: Value::Null,
            feature_state: &[],
            global_state: &[],
            available_images: &[],
            geometry: None,
            locale: &BUILTIN_LOCALE,
            is_supported_script: None,
        }
    }
}

/// Feature coordinates in a tile's local units, with the tile's world location.
/// The extent is explicit because vector-tile producers use different extents.
#[derive(Clone, Debug)]
pub struct FeatureGeometry {
    pub kind: orb_tile::GeometryType,
    pub parts: Vec<Vec<[f64; 2]>>,
    pub tile: CanonicalTile,
    pub extent: f64,
}

/// The canonical tile location used to locate feature geometry.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalTile {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl FeatureGeometry {
    /// Captures the geometry for a bound row. Prepare this only for
    /// geometry-dependent expressions, and reuse it for all such properties.
    pub fn from_layer(layer: &orb_tile::Layer, row: usize, tile: CanonicalTile) -> Self {
        let kind = layer.geometry_type(row);
        let mut parts = Vec::new();
        for part in layer.geometry(row).parts() {
            for r in part.rings() {
                let mut points = r.iter().map(|p| p.map(f64::from)).collect::<Vec<_>>();
                if kind == orb_tile::GeometryType::Polygon
                    && let Some(&first) = points.first()
                {
                    points.push(first);
                }
                parts.push(points);
            }
        }
        Self {
            kind,
            parts,
            tile,
            extent: f64::from(layer.extent()),
        }
    }
}
