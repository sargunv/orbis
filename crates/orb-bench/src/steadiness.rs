//! How still the terrain's cover holds from one frame to the next, from
//! the terrain each frame drew ([`TerrainProbe`]): for a grid of points
//! across the screen, the ground under each in the earlier frame, and in
//! the later one whether that ground's terrain tile changed zoom (and
//! whether to a coarser one), and how far the ground near the eye moved
//! on screen because its elevation changed, beyond what the camera's move
//! explains. Cover tiles that gave way to an ancestor count as drops.

use glam::DVec2;
use orb_api::TileMatrixSet;
use orb_camera::Elevation;
use orb_crs::CoveredTile;
use orb_runtime::TerrainProbe;

/// The grid of screen points sampled: columns and rows.
pub(crate) const GRID: [u32; 2] = [48, 30];
/// The rows of the grid nearest the eye whose ground's wobble counts: the
/// bottom quarter of the screen.
const NEAR_ROWS: u32 = GRID[1] / 4;

/// What one frame changed of the frame before.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Change {
    /// Ground points under the earlier frame's grid, on terrain tiles of
    /// both frames.
    pub(crate) points: u32,
    /// Of those, the ones whose terrain tile changed zoom.
    pub(crate) rezoomed: u32,
    /// Of those, the ones whose terrain tile is coarser.
    pub(crate) coarser: u32,
    /// Cover tiles that an ancestor took the place of.
    pub(crate) drops: u32,
    /// The most a near ground point moved on screen as its elevation
    /// changed, logical pixels.
    pub(crate) wobble_px: f64,
}

impl Change {
    /// The share of the screen's ground whose terrain changed zoom.
    pub(crate) fn rezoomed_share(&self) -> f64 {
        f64::from(self.rezoomed) / f64::from(self.points.max(1))
    }
}

/// The zoom of the terrain tile of `tiles` under `world`.
fn zoom_at(tms: &TileMatrixSet, tiles: &[CoveredTile], world: DVec2) -> Option<u8> {
    let period = tms.projection().wrap_period().unwrap_or(0.0);
    tiles
        .iter()
        .find(|t| {
            let b = tms.tile_bounds(t.index);
            let x = world.x - f64::from(t.wrap) * period;
            (b.min.x..b.max.x).contains(&x) && (b.min.y..b.max.y).contains(&world.y)
        })
        .map(|t| t.index.z)
}

/// What frame `b` changed of frame `a`.
pub(crate) fn change(a: &TerrainProbe, b: &TerrainProbe) -> Change {
    let mut out = Change::default();
    let viewport = a.view.viewport();
    let size = DVec2::new(viewport.width, viewport.height);
    for row in 0..GRID[1] {
        for column in 0..GRID[0] {
            let px = (DVec2::new(f64::from(column), f64::from(row)) + 0.5)
                / DVec2::new(f64::from(GRID[0]), f64::from(GRID[1]))
                * size;
            let Some((world, height)) = a.view.unproject_terrain(px, &*a.elevation) else {
                continue;
            };
            let (Some(from), Some(to)) = (
                zoom_at(&a.tms, &a.tiles, world),
                zoom_at(&b.tms, &b.tiles, world),
            ) else {
                continue;
            };
            out.points += 1;
            out.rezoomed += u32::from(from != to);
            out.coarser += u32::from(to < from);
            if row >= GRID[1] - NEAR_ROWS {
                let now = Elevation::at(&*b.elevation, world).unwrap_or(height);
                if let (Some(was), Some(is)) = (
                    b.view.project_3d(world, height),
                    b.view.project_3d(world, now),
                ) {
                    out.wobble_px = out.wobble_px.max(was.distance(is));
                }
            }
        }
    }
    let kept: std::collections::HashSet<CoveredTile> = b.tiles.iter().copied().collect();
    out.drops = (a.tiles.iter())
        .filter(|t| !kept.contains(t))
        .filter(|t| {
            let mut at = t.index.parent();
            while let Some(parent) = at {
                if kept.contains(&CoveredTile {
                    index: parent,
                    wrap: t.wrap,
                }) {
                    return true;
                }
                at = parent.parent();
            }
            false
        })
        .count() as u32;
    out
}
