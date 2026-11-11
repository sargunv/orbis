//! Projections, tile matrix sets and tile cover.
//!
//! A [`Projection`] maps geodetic positions into a plane's world space, in
//! projected metres with y up. A [`TileMatrixSet`] tiles that plane; tile
//! payloads are in tile-local data units which [`TileMatrixSet::data_to_world`]
//! maps into world space. Both are values chosen by descriptor or identifier,
//! so nothing outside this crate hard-codes a CRS.

#![forbid(unsafe_code)]

mod cover;
mod epsg;
mod family;
mod geo;
mod projection;
mod quad;
pub mod sphere;
mod surface;
mod tms;
mod warp;

pub use cover::{CoverDecision, CoveredTile};
pub use geo::{Geodetic, MERCATOR_MAX_LATITUDE, style_parallel, wgs84};
pub use projection::{Descriptor, Projection, UnknownProjection};
pub use quad::ImageQuad;
pub use surface::{Globe, Surface};
pub use tms::{
    TileIndex, TileMatrix, TileMatrixSet, TileMatrixSetDefinition, UnknownTileMatrixSet,
};
pub use warp::{Footprint, MAX_EDGE_REFINE, Node, Patch, Series, Warp, WarpGrid, WarpMesh};

use glam::DVec2;

/// An axis-aligned rectangle in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRect {
    pub min: DVec2,
    pub max: DVec2,
}

impl WorldRect {
    /// The smallest rectangle holding `points`; inside out (`min` above
    /// `max`) when there are none.
    pub fn around(points: impl IntoIterator<Item = DVec2>) -> Self {
        let (min, max) = points
            .into_iter()
            .fold((DVec2::INFINITY, DVec2::NEG_INFINITY), |(min, max), p| {
                (min.min(p), max.max(p))
            });
        Self { min, max }
    }

    pub fn size(&self) -> DVec2 {
        self.max - self.min
    }

    pub fn center(&self) -> DVec2 {
        (self.min + self.max) * 0.5
    }

    #[must_use]
    pub fn translated(&self, offset: DVec2) -> Self {
        Self {
            min: self.min + offset,
            max: self.max + offset,
        }
    }

    /// The point of the rectangle closest to `p`.
    pub fn clamp(&self, p: DVec2) -> DVec2 {
        p.clamp(self.min, self.max)
    }
}
