//! What placement sees of the camera and of each tile: an immutable
//! snapshot taken on the scene side.

use crate::geometry::{EXTENT, Point};

/// A column-major 4×4 matrix.
pub type Mat4 = [f64; 16];

/// The camera placement runs against (GL JS's transform, in the parts
/// placement reads).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Viewport size, logical pixels.
    pub width: f64,
    pub height: f64,
    pub zoom: f64,
    /// Clockwise map rotation, radians.
    pub bearing: f64,
    /// Radians from straight down.
    pub pitch: f64,
    /// Distance from the camera to the target, logical pixels.
    pub camera_to_center_distance: f64,
}

/// A tile's identity across placements: its zoom, column, row and world
/// copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
    pub wrap: i32,
}

impl TileId {
    /// The ancestor at zoom `z`.
    pub(crate) fn scaled_to(self, z: u8) -> Self {
        let d = self.z - z;
        Self {
            z,
            x: self.x >> d,
            y: self.y >> d,
            wrap: self.wrap,
        }
    }

    pub(crate) fn is_child_of(self, parent: Self) -> bool {
        parent.z < self.z && self.wrap == parent.wrap && self.scaled_to(parent.z) == parent
    }
}

/// One tile as drawn this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileView {
    pub id: TileId,
    /// Tile units ([`EXTENT`] per side) and metres up → clip space.
    pub matrix: Mat4,
}

impl TileView {
    /// GL JS `pixelsToTileUnits(tile, 1, zoom)`.
    pub(crate) fn tile_units_per_pixel(&self, camera: &Camera) -> f64 {
        EXTENT / (512.0 * 2f64.powf(camera.zoom - f64::from(self.id.z)))
    }
}

/// A projected point: clip `x, y` over `w`, and `w`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Projected {
    pub(crate) point: Point,
    pub(crate) w: f64,
}

/// `m × (x, y, z, 1)` divided by `w` (GL JS `projectWithMatrix`).
pub(crate) fn project(m: &Mat4, x: f64, y: f64, z: Option<f64>) -> Projected {
    let z = z.unwrap_or(0.0);
    let px = m[0] * x + m[4] * y + m[8] * z + m[12];
    let py = m[1] * x + m[5] * y + m[9] * z + m[13];
    let w = m[3] * x + m[7] * y + m[11] * z + m[15];
    Projected {
        point: Point::new(px / w, py / w),
        w,
    }
}

/// GL JS `getTileSkewVectors`, without roll.
pub(crate) fn tile_skew_vectors(camera: &Camera) -> (Point, Point) {
    let cos_pitch = libm::cos(camera.pitch);
    let (sin_b, cos_b) = (libm::sin(camera.bearing), libm::cos(camera.bearing));
    let unit = |x: f64, y: f64| {
        let len = libm::hypot(x, y);
        if len < 1.0e-9 {
            Point::new(0.0, 0.0)
        } else {
            Point::new(x / len, y / len)
        }
    };
    let south = unit(-sin_b, cos_b);
    let east = unit(cos_b * cos_pitch, sin_b * cos_pitch);
    (east, south)
}

/// The matrix from tile units to a pitched label plane (GL JS
/// `getPitchedLabelPlaneMatrix`): a 2D affine `[a, b, c, d]` (column
/// major, no translation).
pub(crate) fn pitched_label_plane(
    rotate_with_map: bool,
    camera: &Camera,
    tile_units_per_pixel: f64,
) -> [f64; 4] {
    let mut m = [1.0, 0.0, 0.0, 1.0];
    if !rotate_with_map {
        let (east, south) = tile_skew_vectors(camera);
        // The inverse of [east south].
        let det = east.x * south.y - south.x * east.y;
        if det != 0.0 {
            m = [south.y / det, -east.y / det, -south.x / det, east.x / det];
        } else {
            m = [0.0; 4];
        }
    }
    m.map(|v| v / tile_units_per_pixel)
}

/// The terrain symbols stand on: its elevation in metres, with
/// exaggeration, at tile units of a drawn tile (GL JS's `getElevation`
/// for a tile).
pub trait Ground: Send + Sync + std::fmt::Debug {
    fn elevation(&self, tile: TileId, x: f64, y: f64) -> f64;

    /// A compact copy of these elevations as bytes, for placement on
    /// another thread (the web's placement worker).
    fn pyramid(&self) -> Vec<u8>;
}

/// A symbol's elevation in metres (GL JS `getSymbolElevation`): its height
/// offset, over the terrain's elevation `ground` when anchored to the
/// ground. `None` at the datum, for the flat fast path.
pub(crate) fn symbol_elevation(
    ground: Option<f64>,
    height_offset: f64,
    height_anchor_ground: bool,
) -> Option<f64> {
    if !height_anchor_ground {
        return Some(height_offset);
    }
    match ground {
        Some(ground) => Some(ground + height_offset),
        None => (height_offset != 0.0).then_some(height_offset),
    }
}
