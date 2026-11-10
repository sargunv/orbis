//! Explicit inputs shared by every expression backend.
use crate::Value;

/// Evaluation inputs that vary independently of tile properties.
#[derive(Debug)]
pub struct EvaluationContext<'a> {
    pub zoom: f64,
    /// The camera's pitch and bearing, degrees: `["pitch"]` and
    /// `["bearing"]`.
    pub pitch: f64,
    pub bearing: f64,
    /// Ground metres per logical pixel at the camera's target, unpitched:
    /// `["meters-per-pixel"]`. NaN for Web Mercator's on the equator at
    /// `zoom`.
    pub meters_per_pixel: f64,
    pub line_progress: f64,
    pub heatmap_density: f64,
    pub elevation: f64,
    pub accumulated: Value<'a>,
    pub feature_state: &'a [(String, Value<'static>)],
    pub global_state: &'a [(String, Value<'static>)],
    pub available_images: &'a [String],
    pub geometry: Option<&'a FeatureGeometry>,
    pub is_supported_script: Option<fn(&str) -> bool>,
}
impl Default for EvaluationContext<'_> {
    fn default() -> Self {
        Self {
            zoom: 0.0,
            pitch: 0.0,
            bearing: 0.0,
            meters_per_pixel: f64::NAN,
            line_progress: 0.0,
            heatmap_density: 0.0,
            elevation: 0.0,
            accumulated: Value::Null,
            feature_state: &[],
            global_state: &[],
            available_images: &[],
            geometry: None,
            is_supported_script: None,
        }
    }
}

impl EvaluationContext<'_> {
    /// `["meters-per-pixel"]`: the context's, else Web Mercator's on the
    /// equator at its zoom, 512 pixels a tile.
    pub fn meters_per_pixel(&self) -> f64 {
        if self.meters_per_pixel.is_nan() {
            EQUATOR_METRES / 512.0 / self.zoom.exp2()
        } else {
            self.meters_per_pixel
        }
    }
}

/// The WGS84 equator's length in metres.
const EQUATOR_METRES: f64 = 2.0 * std::f64::consts::PI * 6_378_137.0;

/// What the camera gives a value beside its zoom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraInputs {
    /// Degrees.
    pub pitch: f64,
    pub bearing: f64,
    /// Ground metres per logical pixel at the target, unpitched.
    pub meters_per_pixel: f64,
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
    pub(crate) fn from_layer(layer: &orb_tile::Layer, row: usize, tile: CanonicalTile) -> Self {
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
