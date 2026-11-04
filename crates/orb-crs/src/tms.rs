use std::fmt;

use glam::{DAffine2, DMat2, DVec2};

use crate::{Projection, WorldRect};

/// A tile in a tile matrix set: zoom, column (left to right) and row (top to
/// bottom).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileIndex {
    pub z: u8,
    pub col: u32,
    pub row: u32,
}

impl TileIndex {
    pub const fn new(z: u8, col: u32, row: u32) -> Self {
        Self { z, col, row }
    }

    /// The four children of a quad tile, in row-major order.
    pub fn children(self) -> [Self; 4] {
        let (z, c, r) = (self.z + 1, self.col * 2, self.row * 2);
        [
            Self::new(z, c, r),
            Self::new(z, c + 1, r),
            Self::new(z, c, r + 1),
            Self::new(z, c + 1, r + 1),
        ]
    }

    /// The parent of a quad tile, or `None` at zoom 0.
    pub fn parent(self) -> Option<Self> {
        (self.z > 0).then(|| Self::new(self.z - 1, self.col / 2, self.row / 2))
    }
}

/// A tiling of a projection's plane (OGC 2D TileMatrixSet). Every matrix is a
/// quad subdivision of the zoom-0 matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct TileMatrixSet {
    id: &'static str,
    projection: Projection,
    /// World position of the top-left corner of the zoom-0 matrix.
    origin: DVec2,
    /// Width and height of one zoom-0 tile in world units.
    z0_span: f64,
    /// Columns and rows of the zoom-0 matrix.
    z0_size: (u32, u32),
    max_zoom: u8,
}

/// An identifier that names no known tile matrix set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownTileMatrixSet(pub String);

impl fmt::Display for UnknownTileMatrixSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown tile matrix set `{}`", self.0)
    }
}

impl std::error::Error for UnknownTileMatrixSet {}

impl TileMatrixSet {
    /// A registered tile matrix set by its OGC identifier.
    pub fn by_id(id: &str) -> Result<Self, UnknownTileMatrixSet> {
        match id {
            "WebMercatorQuad" => {
                let projection =
                    Projection::from_descriptor("web-mercator").expect("built-in projection");
                let bounds = projection.world_bounds();
                Ok(Self {
                    id: "WebMercatorQuad",
                    projection,
                    origin: DVec2::new(bounds.min.x, bounds.max.y),
                    z0_span: bounds.size().x,
                    z0_size: (1, 1),
                    max_zoom: 30,
                })
            }
            _ => Err(UnknownTileMatrixSet(id.to_string())),
        }
    }

    pub fn id(&self) -> &str {
        self.id
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    pub fn max_zoom(&self) -> u8 {
        self.max_zoom
    }

    /// World width and height of one tile at zoom `z`.
    pub fn tile_span(&self, z: u8) -> f64 {
        self.z0_span / f64::from(1u32 << z)
    }

    /// Columns and rows of the matrix at zoom `z`.
    pub fn matrix_size(&self, z: u8) -> (u32, u32) {
        (self.z0_size.0 << z, self.z0_size.1 << z)
    }

    /// The tiles of the zoom-0 matrix.
    pub fn roots(&self) -> impl Iterator<Item = TileIndex> + '_ {
        let (cols, rows) = self.z0_size;
        (0..rows).flat_map(move |row| (0..cols).map(move |col| TileIndex::new(0, col, row)))
    }

    /// The world rectangle a tile covers.
    pub fn tile_bounds(&self, tile: TileIndex) -> WorldRect {
        let span = self.tile_span(tile.z);
        let min_x = self.origin.x + f64::from(tile.col) * span;
        let max_y = self.origin.y - f64::from(tile.row) * span;
        WorldRect {
            min: DVec2::new(min_x, max_y - span),
            max: DVec2::new(min_x + span, max_y),
        }
    }

    /// The affine map from a tile's data units, `extent` per side with y down
    /// from the top-left corner, into world space.
    pub fn data_to_world(&self, tile: TileIndex, extent: u32) -> DAffine2 {
        let bounds = self.tile_bounds(tile);
        let scale = self.tile_span(tile.z) / f64::from(extent);
        DAffine2::from_mat2_translation(
            DMat2::from_diagonal(DVec2::new(scale, -scale)),
            DVec2::new(bounds.min.x, bounds.max.y),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_mercator_quad_bounds() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let world = tms.projection().world_bounds();
        assert_eq!(tms.tile_bounds(TileIndex::new(0, 0, 0)), world);
        let ne = tms.tile_bounds(TileIndex::new(1, 1, 0));
        assert_eq!(ne.min, DVec2::ZERO);
        assert_eq!(ne.max, world.max);
    }

    #[test]
    fn data_to_world_maps_corners() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let tile = TileIndex::new(3, 5, 2);
        let bounds = tms.tile_bounds(tile);
        let m = tms.data_to_world(tile, 4096);
        assert_eq!(
            m.transform_point2(DVec2::ZERO),
            DVec2::new(bounds.min.x, bounds.max.y)
        );
        let br = m.transform_point2(DVec2::splat(4096.0));
        assert!(
            (br - DVec2::new(bounds.max.x, bounds.min.y))
                .abs()
                .max_element()
                < 1e-6
        );
    }

    #[test]
    fn parent_and_children_agree() {
        let t = TileIndex::new(4, 7, 9);
        assert!(t.children().iter().all(|c| c.parent() == Some(t)));
        assert_eq!(TileIndex::new(0, 0, 0).parent(), None);
    }
}
