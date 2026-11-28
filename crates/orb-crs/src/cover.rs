use crate::{TileIndex, TileMatrixSet};

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
    /// Covers a view by depth-first traversal from the first matrix,
    /// through the children each matrix holds: on a drawing plane that
    /// wraps (`wrap`: copies on either side, and the period between them),
    /// in each copy `-copies..=copies`, else in its one copy. `decide` sees
    /// each tile with its copy's x offset. Tiles asked to refine at
    /// `max_zoom` are accepted.
    pub fn cover(
        &self,
        wrap: Option<(i32, f64)>,
        max_zoom: u8,
        mut decide: impl FnMut(TileIndex, f64) -> CoverDecision,
    ) -> Vec<CoveredTile> {
        let max_zoom = max_zoom.min(self.max_zoom());
        let mut out = Vec::new();
        let mut stack = Vec::new();
        let (copies, period) = wrap.unwrap_or((0, 0.0));
        for wrap in -copies..=copies {
            let offset = f64::from(wrap) * period;
            stack.extend(self.roots());
            while let Some(index) = stack.pop() {
                match decide(index, offset) {
                    CoverDecision::Skip => {}
                    CoverDecision::Refine if index.z < max_zoom => {
                        stack.extend(index.children().into_iter().filter(|&c| self.contains(c)))
                    }
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
    fn covers_a_region_at_uniform_zoom_in_its_copy() {
        let tms = TileMatrixSet::web_mercator();
        let half = tms.tile_span(0) / 2.0;
        // The north-east quadrant of the primary world at zoom 2.
        let mut tiles = tms.cover(Some((1, 2.0 * half)), 2, |index, offset| {
            let b = tms
                .tile_area(index)
                .translated(glam::DVec2::new(offset, 0.0));
            let overlaps = b.max.x > 0.0 && b.min.x < half && b.max.y > 0.0;
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
        let tms = TileMatrixSet::web_mercator();
        let tiles = tms.cover(None, 1, |_, _| CoverDecision::Refine);
        assert_eq!(tiles.len(), 4);
        assert!(tiles.iter().all(|t| t.index.z == 1));
    }
}
