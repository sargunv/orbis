//! Projections, tile matrix sets and tile cover.
//!
//! A [`Projection`] maps geodetic positions into a plane's world space, in
//! projected metres with y up. A [`TileMatrixSet`] tiles that plane; tile
//! payloads are in tile-local data units which [`TileMatrixSet::data_to_world`]
//! maps into world space. Both are values chosen by descriptor or identifier,
//! so nothing outside this crate hard-codes a CRS.

#![forbid(unsafe_code)]

mod cover;
mod projection;
mod tms;

pub use cover::{CoverDecision, CoveredTile};
pub use projection::{Projection, UnknownProjection};
pub use tms::{TileIndex, TileMatrixSet, UnknownTileMatrixSet};

use glam::DVec2;

/// An axis-aligned rectangle in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRect {
    pub min: DVec2,
    pub max: DVec2,
}

impl WorldRect {
    pub fn size(&self) -> DVec2 {
        self.max - self.min
    }

    pub fn center(&self) -> DVec2 {
        (self.min + self.max) * 0.5
    }

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
