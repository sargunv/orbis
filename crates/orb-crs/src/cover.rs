use std::ops::RangeInclusive;

use glam::DVec2;

use crate::{TileIndex, TileMatrixSet, WorldRect};

/// A tile chosen by [`TileMatrixSet::cover`], in one copy of a wrapping world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CoveredTile {
    pub index: TileIndex,
    /// Which world copy, counted in wrap periods east of the primary one.
    pub wrap: i32,
}

/// What [`TileMatrixSet::cover`] does with a visited tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoverDecision {
    /// Not visible: drop it and its descendants.
    Skip,
    /// Visible at a sufficient level of detail.
    Accept,
    /// Visible but too coarse: visit its children.
    Refine,
}

impl TileMatrixSet {
    /// Covers a view by depth-first traversal from the zoom-0 matrix of each
    /// world copy in `wraps`. `decide` sees each tile with its bounds shifted
    /// into its world copy. Tiles asked to refine at `max_zoom` are accepted.
    ///
    /// Projections that don't wrap only have copy 0.
    pub fn cover(
        &self,
        wraps: RangeInclusive<i32>,
        max_zoom: u8,
        mut decide: impl FnMut(TileIndex, WorldRect) -> CoverDecision,
    ) -> Vec<CoveredTile> {
        self.cover_with(
            wraps,
            self.projection().wrap_period(),
            max_zoom,
            |index, offset| {
                decide(
                    index,
                    self.tile_bounds(index).translated(DVec2::new(offset, 0.0)),
                )
            },
        )
    }

    /// [`cover`](Self::cover) for tiles drawn on another plane: `decide`
    /// sees each tile with the x offset of its copy of the drawing plane,
    /// whose wrap `period` (if it wraps) sets the copies.
    pub fn cover_with(
        &self,
        wraps: RangeInclusive<i32>,
        period: Option<f64>,
        max_zoom: u8,
        mut decide: impl FnMut(TileIndex, f64) -> CoverDecision,
    ) -> Vec<CoveredTile> {
        let wraps = if period.is_some() { wraps } else { 0..=0 };
        let max_zoom = max_zoom.min(self.max_zoom());

        let mut out = Vec::new();
        let mut stack = Vec::new();
        for wrap in wraps {
            let offset = f64::from(wrap) * period.unwrap_or(0.0);
            stack.extend(self.roots());
            while let Some(index) = stack.pop() {
                match decide(index, offset) {
                    CoverDecision::Skip => {}
                    CoverDecision::Refine if index.z < max_zoom => stack.extend(index.children()),
                    CoverDecision::Accept | CoverDecision::Refine => {
                        out.push(CoveredTile { index, wrap })
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_a_region_at_uniform_zoom() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        // The north-east quadrant of the primary world at zoom 2.
        let region = WorldRect {
            min: DVec2::splat(1.0),
            max: tms.projection().world_bounds().max - 1.0,
        };
        let mut tiles = tms.cover(-1..=1, 2, |index, bounds| {
            let overlaps = bounds.min.cmplt(region.max).all() && bounds.max.cmpgt(region.min).all();
            match (overlaps, index.z) {
                (false, _) => CoverDecision::Skip,
                (true, 2) => CoverDecision::Accept,
                (true, _) => CoverDecision::Refine,
            }
        });
        tiles.sort();
        let mut expected = [(2, 0), (3, 0), (2, 1), (3, 1)].map(|(col, row)| CoveredTile {
            index: TileIndex::new(2, col, row),
            wrap: 0,
        });
        expected.sort();
        assert_eq!(tiles, expected);
    }

    #[test]
    fn refine_stops_at_max_zoom() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let tiles = tms.cover(0..=0, 1, |_, _| CoverDecision::Refine);
        assert_eq!(tiles.len(), 4);
        assert!(tiles.iter().all(|t| t.index.z == 1));
    }
}
