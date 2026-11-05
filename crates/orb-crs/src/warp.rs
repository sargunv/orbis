//! A tile matrix set's tiles on another plane (plan §9.2, the `RasterWarp`
//! plan): where each tile lands, for cover, and an error-bounded mesh that
//! carries the tile's image there.

use std::collections::HashMap;

use glam::DVec2;

use crate::{Projection, TileIndex, TileMatrixSet, WorldRect};

/// Tiles of `from` drawn on the plane of `to`.
#[derive(Clone, Debug, PartialEq)]
pub struct Warp {
    from: TileMatrixSet,
    to: Projection,
}

/// Where a tile lands on the target plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Footprint {
    /// The tile's bounding rectangle in the target plane's primary world
    /// copy (a wrapping plane's footprint may extend past its edges).
    pub bounds: WorldRect,
    /// Target world units per tile side at the tile's centre: its local
    /// scale, for level of detail.
    pub span: f64,
}

/// A tile's mesh on the target plane.
#[derive(Clone, Debug, PartialEq)]
pub struct WarpMesh {
    /// The square `positions` are relative to, its top-left corner at the
    /// mesh's top-left extremes.
    pub frame: WorldRect,
    /// Target-plane positions as fractions of the frame's side, x right,
    /// y down.
    pub positions: Vec<DVec2>,
    /// Source positions as fractions of the tile, x right, y down.
    pub sources: Vec<DVec2>,
    pub triangles: Vec<[u32; 3]>,
}

/// A tile on the target plane (or, as `Series<DVec3>`, on a sphere) as
/// quadratic patches over an even grid.
#[derive(Clone, Debug, PartialEq)]
pub struct Series<V = DVec2> {
    /// The tile centre's position on the target; patches are relative to
    /// it.
    pub anchor: V,
    /// Patches per side.
    pub cells: u32,
    /// Row by row, top to bottom, relative to `anchor`.
    pub patches: Vec<Patch<V>>,
    /// The longest straight segment, as a fraction of the tile's side, whose
    /// image stays within the tolerance of its curve: the tile's geometry
    /// is subdivided to it.
    pub max_segment: f64,
    /// Target units per tile side at the centre.
    pub scale: f64,
}

/// A position a series interpolates: on a plane or in space.
pub trait Node:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<f64, Output = Self>
    + std::ops::AddAssign
{
    const ZERO: Self;
    fn length(self) -> f64;
    fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }
}

impl Node for DVec2 {
    const ZERO: Self = DVec2::ZERO;
    fn length(self) -> f64 {
        DVec2::length(self)
    }
}

impl Node for glam::DVec3 {
    const ZERO: Self = glam::DVec3::ZERO;
    fn length(self) -> f64 {
        glam::DVec3::length(self)
    }
}

impl<V: Node> Series<V> {
    /// Fits the fewest of 1, 4, … 256 equal patches whose quadratics stay
    /// within `limit` of `map` (tile fractions, y down, to positions), or
    /// `None` where `map` doesn't reach or the tile needs more. `map`
    /// returns positions continuous across the tile (unwrapped).
    pub(crate) fn fit(
        map: impl Fn(DVec2) -> Option<V>,
        limit: f64,
        scale: f64,
        max_cells: u32,
    ) -> Option<Self> {
        let anchor = map(DVec2::splat(0.5))?;
        let mut cells = 1u32;
        loop {
            let side = 1.0 / f64::from(cells);
            let mut patches = Vec::with_capacity((cells * cells) as usize);
            let mut error: f64 = 0.0;
            for j in 0..cells {
                for i in 0..cells {
                    let center = (DVec2::new(f64::from(i), f64::from(j)) + 0.5) * side;
                    let patch = Patch::through(&map, center, side, anchor)?;
                    // Its worst point among a 6×6 sample between its nodes.
                    for b in 0..6 {
                        for a in 0..6 {
                            let st = DVec2::new(f64::from(a), f64::from(b)) / 2.5 - 1.0;
                            let exact = map(center + st * side / 2.0)?;
                            error = error.max((anchor + patch.eval(st)).distance(exact));
                        }
                    }
                    patches.push(patch);
                }
            }
            if error <= limit {
                // Second derivatives in tile fractions: a patch's in its
                // `[-1, 1]` parameters, times (2 / side)².
                let curvature = patches
                    .iter()
                    .map(|p| p.curvature() * 4.0 / (side * side))
                    .fold(0.0, f64::max);
                return Some(Self {
                    anchor,
                    cells,
                    patches,
                    // A chord of `len` bows by `curvature·len²/8`; keep that
                    // within the limit.
                    max_segment: if curvature > 0.0 {
                        (8.0 * limit / curvature).sqrt().min(1.0)
                    } else {
                        1.0
                    },
                    scale,
                });
            }
            if cells >= max_cells {
                return None;
            }
            cells *= 2;
        }
    }

    /// The patch holding a tile position (fractions) and the position's
    /// parameters in it, each in `[-1, 1]`.
    pub fn locate(&self, uv: DVec2) -> (&Patch<V>, DVec2) {
        let cells = f64::from(self.cells);
        let cell = (uv * cells)
            .floor()
            .clamp(DVec2::ZERO, DVec2::splat(cells - 1.0));
        let patch = &self.patches[(cell.y as u32 * self.cells + cell.x as u32) as usize];
        (patch, (uv * cells - cell) * 2.0 - 1.0)
    }

    /// Cells per side of the grid to cut the tile's geometry along: the
    /// fewest, a power of two, whose cells' diagonals are within
    /// `max_segment` (at most 256).
    pub fn grid(&self) -> u32 {
        let cells = (std::f64::consts::SQRT_2 / self.max_segment)
            .log2()
            .ceil()
            .max(0.0);
        1u32 << (cells as u32).min(8)
    }

    /// A tile position (fractions) on the target.
    pub fn position(&self, uv: DVec2) -> V {
        let (patch, st) = self.locate(uv);
        self.anchor + patch.eval(st)
    }

    /// The derivatives of the position along a tile's x and y (fractions)
    /// at `uv`.
    pub fn derivatives(&self, uv: DVec2) -> [V; 2] {
        let (patch, st) = self.locate(uv);
        // Parameters run over a patch's side: 2 · cells per tile side.
        patch
            .derivatives(st)
            .map(|d| d * (2.0 * f64::from(self.cells)))
    }
}

impl Series<DVec2> {
    /// The tile's linear approximation at its centre, from data units
    /// (`extent` per side, y down) to the target plane.
    pub fn linear(&self, extent: f64) -> glam::DAffine2 {
        let [ds, dt] = self.derivatives(DVec2::splat(0.5));
        let center = self.position(DVec2::splat(0.5));
        glam::DAffine2::from_mat2_translation(
            glam::DMat2::from_cols(ds / extent, dt / extent),
            center - (ds + dt) * 0.5,
        )
    }
}

/// A biquadratic patch: positions at its 3×3 nodes (row by row, top to
/// bottom; corners, edge midpoints and centre), interpolated by quadratic
/// Lagrange polynomials in parameters `s` (right) and `t` (down), each in
/// `[-1, 1]`. Neighbouring patches share their edges' three nodes, so they
/// meet exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch<V = DVec2> {
    pub points: [V; 9],
}

/// Quadratic Lagrange basis at nodes -1, 0, 1, and its derivative.
fn basis(s: f64) -> ([f64; 3], [f64; 3]) {
    (
        [s * (s - 1.0) / 2.0, 1.0 - s * s, s * (s + 1.0) / 2.0],
        [s - 0.5, -2.0 * s, s + 0.5],
    )
}

impl<V: Node> Patch<V> {
    /// The patch over the square of `side` about `center` (tile fractions)
    /// through `map`'s positions at its corners, edge midpoints and
    /// centre, relative to `anchor`.
    fn through(
        map: impl Fn(DVec2) -> Option<V>,
        center: DVec2,
        side: f64,
        anchor: V,
    ) -> Option<Self> {
        let mut points = [V::ZERO; 9];
        for (k, point) in points.iter_mut().enumerate() {
            let offset = DVec2::new((k % 3) as f64 - 1.0, (k / 3) as f64 - 1.0) * side / 2.0;
            *point = map(center + offset)? - anchor;
        }
        Some(Self { points })
    }

    pub fn eval(&self, st: DVec2) -> V {
        let ((bs, _), (bt, _)) = (basis(st.x), basis(st.y));
        let mut p = V::ZERO;
        for (k, point) in self.points.iter().enumerate() {
            p += *point * (bs[k % 3] * bt[k / 3]);
        }
        p
    }

    /// The derivatives along `s` and `t` at `st`.
    pub fn derivatives(&self, st: DVec2) -> [V; 2] {
        let ((bs, ds), (bt, dt)) = (basis(st.x), basis(st.y));
        let (mut ps, mut pt) = (V::ZERO, V::ZERO);
        for (k, point) in self.points.iter().enumerate() {
            ps += *point * (ds[k % 3] * bt[k / 3]);
            pt += *point * (bs[k % 3] * dt[k / 3]);
        }
        [ps, pt]
    }

    /// The largest second difference across its rows and columns: its
    /// second derivative's bound in its parameters.
    fn curvature(&self) -> f64 {
        let p = &self.points;
        (0..3)
            .flat_map(|i| {
                [
                    p[3 * i] - p[3 * i + 1] * 2.0 + p[3 * i + 2],
                    p[i] - p[3 + i] * 2.0 + p[6 + i],
                ]
            })
            .map(Node::length)
            .fold(0.0, f64::max)
    }
}

/// The most patches per side of a planar series.
const MAX_SERIES_CELLS: u32 = 16;

/// Samples per side when finding a footprint.
const FOOTPRINT_STEPS: u32 = 8;
/// The finest mesh grid, cells per side, and the steps along tile edges.
const MAX_CELLS: u32 = 64;
/// Vertex positions per tile side: tile edges at `MAX_CELLS` steps, and
/// cell centres of the finest grid.
const LATTICE: u32 = 2 * MAX_CELLS;

impl Warp {
    pub fn new(from: TileMatrixSet, to: Projection) -> Self {
        Self { from, to }
    }

    pub fn source(&self) -> &TileMatrixSet {
        &self.from
    }

    /// A position in a tile (fractions, y down) on the target plane, or
    /// `None` where either projection doesn't reach.
    pub fn map(&self, tile: TileIndex, uv: DVec2) -> Option<DVec2> {
        let b = self.from.tile_bounds(tile);
        let world = DVec2::new(b.min.x + uv.x * b.size().x, b.max.y - uv.y * b.size().y);
        let mut g = self.from.projection().inverse(world);
        if !g.lon.is_finite() || !g.lat.is_finite() || !self.to.contains(g) {
            return None;
        }
        // On a plane cut along a meridian, a tile's points on the cut (Web
        // Mercator's ±180° edges) take the side of the tile just inside them.
        if let Some(lon0) = self.to.seam() {
            let wrap = |lon: f64| (lon - lon0 + 540.0).rem_euclid(360.0) - 180.0;
            if (wrap(g.lon).abs() - 180.0).abs() < 1e-6 {
                let inside = uv + (DVec2::splat(0.5) - uv) * 1e-3;
                let world = DVec2::new(
                    b.min.x + inside.x * b.size().x,
                    b.max.y - inside.y * b.size().y,
                );
                let near = self.from.projection().inverse(world);
                g.lon = lon0 + 180.0f64.copysign(wrap(near.lon));
            }
        }
        let p = self.to.forward(g);
        p.is_finite().then_some(p)
    }

    /// `x` in the world copy nearest `reference`, on a wrapping plane.
    fn unwrap(&self, x: f64, reference: f64) -> f64 {
        let near = DVec2::new(reference, 0.0);
        self.to.nearest_copy(DVec2::new(x, 0.0), near).x
    }

    /// Whether the tile holds a pole that a wrapping target plane spreads
    /// along its edge (Web Mercator's), so the tile spans every copy.
    fn spans_wrap(&self, tile: TileIndex) -> bool {
        self.to.wrap_period().is_some()
            && [90.0, -90.0].into_iter().any(|lat| {
                let pole = crate::Geodetic::new(0.0, lat);
                let from = self.from.projection();
                if !from.contains(pole) {
                    return false;
                }
                let p = from.forward(pole);
                let b = self.from.tile_bounds(tile);
                p.cmpge(b.min).all() && p.cmple(b.max).all()
            })
    }

    /// Where a tile lands, or `None` if no part of it reaches the target
    /// plane.
    pub fn footprint(&self, tile: TileIndex) -> Option<Footprint> {
        let steps = FOOTPRINT_STEPS;
        let samples: Vec<_> = (0..=steps)
            .flat_map(|j| (0..=steps).map(move |i| (i, j)))
            .filter_map(|(i, j)| {
                self.map(
                    tile,
                    DVec2::new(f64::from(i), f64::from(j)) / f64::from(steps),
                )
            })
            .collect();
        let reference = self
            .map(tile, DVec2::splat(0.5))
            .or(samples.first().copied())?;
        let mut bounds = WorldRect {
            min: DVec2::splat(f64::INFINITY),
            max: DVec2::splat(f64::NEG_INFINITY),
        };
        for p in &samples {
            let p = DVec2::new(self.unwrap(p.x, reference.x), p.y);
            bounds.min = bounds.min.min(p);
            bounds.max = bounds.max.max(p);
        }
        if self.spans_wrap(tile) {
            let period = self.to.wrap_period().expect("wraps");
            bounds.min.x = bounds.min.x.min(reference.x - period / 2.0);
            bounds.max.x = bounds.max.x.max(reference.x + period / 2.0);
            let world = self.to.world_bounds();
            bounds.min.y = bounds.min.y.min(world.min.y).max(world.min.y);
            bounds.max.y = bounds.max.y.max(world.max.y).min(world.max.y);
        }
        Some(Footprint {
            bounds,
            span: self.local_span(tile).unwrap_or(bounds.size().max_element()),
        })
    }

    /// The tile's side on the target plane at its centre: the square root
    /// of the local area scale times the tile's side.
    fn local_span(&self, tile: TileIndex) -> Option<f64> {
        let h = 1.0 / 64.0;
        let c = DVec2::splat(0.5);
        let dx =
            self.map(tile, c + DVec2::new(h, 0.0))? - self.map(tile, c - DVec2::new(h, 0.0))?;
        let dy =
            self.map(tile, c + DVec2::new(0.0, h))? - self.map(tile, c - DVec2::new(0.0, h))?;
        let (dx, dy) = (
            DVec2::new(self.unwrap(dx.x, 0.0), dx.y),
            DVec2::new(self.unwrap(dy.x, 0.0), dy.y),
        );
        let area = dx.perp_dot(dy).abs() / (2.0 * h * 2.0 * h);
        (area > 0.0 && area.is_finite()).then(|| area.sqrt())
    }

    /// The tile's position on the target plane as quadratic patches (the
    /// `GpuPerVertex` plan's precision series, plan §9.3): the fewest of
    /// 1, 4, … 256 equal patches whose quadratics stay within `tolerance` of
    /// the tile's side of the true position, or `None` where the tile
    /// leaves the target plane's domain, crosses a wrapping plane's seam or
    /// needs more.
    pub fn series(&self, tile: TileIndex, tolerance: f64) -> Option<Series> {
        let footprint = self.footprint(tile)?;
        if self.spans_wrap(tile) {
            return None;
        }
        let anchor = self.map(tile, DVec2::splat(0.5))?;
        Series::fit(
            |uv| {
                let p = self.map(tile, uv)?;
                Some(DVec2::new(self.unwrap(p.x, anchor.x), p.y))
            },
            tolerance * footprint.span,
            footprint.span,
            MAX_SERIES_CELLS,
        )
    }

    /// The tile's mesh: a grid fine enough that its linear interpolation
    /// strays from the true position by at most `tolerance` of the tile's
    /// side (to a 64-cell limit), without the triangles that leave the
    /// target plane's world bounds grown by `reach` times their size.
    pub fn mesh(&self, tile: TileIndex, tolerance: f64, reach: f64) -> Option<WarpMesh> {
        let footprint = self.footprint(tile)?;
        let reference = footprint.bounds.center().x;
        let world = self.to.world_bounds();
        let allowed = WorldRect {
            min: world.min - world.size() * reach,
            max: world.max + world.size() * reach,
        };
        let limit = tolerance * footprint.span;
        let mut cells = 4;
        let grid = loop {
            let n = cells + 1;
            let uvs: Vec<DVec2> = (0..n)
                .flat_map(|j| (0..n).map(move |i| DVec2::new(f64::from(i), f64::from(j))))
                .map(|p| p / f64::from(cells))
                .collect();
            let grid: Vec<Option<DVec2>> = uvs
                .iter()
                .map(|&uv| {
                    self.map(tile, uv)
                        .map(|p| DVec2::new(self.unwrap(p.x, reference), p.y))
                        .filter(|p| p.cmpge(allowed.min).all() && p.cmple(allowed.max).all())
                })
                .collect();
            if cells >= MAX_CELLS {
                break grid;
            }
            // The worst cell: its edges' midpoints against their ends'
            // mean, and its centre against its corners'. (A cell can bend
            // so that its centre stays near the corners' mean while its
            // edges bow out, as a parallel does on a polar plane.)
            let mut error: f64 = 0.0;
            let mut check = |uv: DVec2, ends: &[DVec2]| {
                if let Some(exact) = self.map(tile, uv) {
                    let mean = ends.iter().copied().sum::<DVec2>() / ends.len() as f64;
                    let exact = DVec2::new(self.unwrap(exact.x, mean.x), exact.y);
                    error = error.max(exact.distance(mean));
                }
            };
            let step = 1.0 / f64::from(cells);
            for j in 0..cells {
                for i in 0..cells {
                    let at = |di: u32, dj: u32| grid[((j + dj) * n + i + di) as usize];
                    let (Some(a), Some(b), Some(c), Some(d)) =
                        (at(0, 0), at(1, 0), at(0, 1), at(1, 1))
                    else {
                        continue;
                    };
                    let corner = DVec2::new(f64::from(i), f64::from(j)) * step;
                    check(corner + DVec2::new(0.5, 0.0) * step, &[a, b]);
                    check(corner + DVec2::new(0.0, 0.5) * step, &[a, c]);
                    check(corner + DVec2::new(1.0, 0.5) * step, &[b, d]);
                    check(corner + DVec2::new(0.5, 1.0) * step, &[c, d]);
                    check(corner + DVec2::splat(0.5) * step, &[a, b, c, d]);
                }
            }
            if error <= limit {
                break grid;
            }
            cells *= 2;
        };
        // Vertices sit on a lattice of `LATTICE` steps per side: the grid's
        // corners, boundary cells' centres, and every `MAX_CELLS`th of the
        // tile's edges, so that neighbours share their edge vertices
        // whatever grid each chose (no T-junction cracks).
        let unit = LATTICE / cells;
        let edge_step = LATTICE / MAX_CELLS;
        let mut lattice: HashMap<(u32, u32), Option<u32>> = HashMap::new();
        let mut positions: Vec<DVec2> = Vec::new();
        let mut sources = Vec::new();
        let mut vertex = |x: u32, y: u32| -> Option<u32> {
            *lattice.entry((x, y)).or_insert_with(|| {
                let uv = DVec2::new(f64::from(x), f64::from(y)) / f64::from(LATTICE);
                let on_grid = x.is_multiple_of(unit) && y.is_multiple_of(unit);
                let p = if on_grid {
                    grid[((y / unit) * (cells + 1) + x / unit) as usize]
                } else {
                    self.map(tile, uv)
                        .map(|p| DVec2::new(self.unwrap(p.x, reference), p.y))
                        .filter(|p| p.cmpge(allowed.min).all() && p.cmple(allowed.max).all())
                };
                p.map(|p| {
                    positions.push(p);
                    sources.push(uv);
                    (positions.len() - 1) as u32
                })
            })
        };
        let mut polygons: Vec<Vec<Option<u32>>> = Vec::new();
        let mut fans: Vec<(Option<u32>, Vec<Option<u32>>)> = Vec::new();
        for j in 0..cells {
            for i in 0..cells {
                let (x0, y0, x1, y1) = (i * unit, j * unit, (i + 1) * unit, (j + 1) * unit);
                let boundary =
                    unit > edge_step && (x0 == 0 || y0 == 0 || x1 == LATTICE || y1 == LATTICE);
                if !boundary {
                    let [a, b, c, d] =
                        [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].map(|(x, y)| vertex(x, y));
                    polygons.push(vec![a, b, c]);
                    polygons.push(vec![c, b, d]);
                    continue;
                }
                // The cell's outline clockwise, with the tile edge's
                // vertices along the sides on it, as a fan from its centre.
                let mut ring = Vec::new();
                let side = |on_edge: bool| if on_edge { edge_step } else { unit };
                let mut x = x0;
                while x < x1 {
                    ring.push(vertex(x, y0));
                    x += side(y0 == 0);
                }
                let mut y = y0;
                while y < y1 {
                    ring.push(vertex(x1, y));
                    y += side(x1 == LATTICE);
                }
                let mut x = x1;
                while x > x0 {
                    ring.push(vertex(x, y1));
                    x -= side(y1 == LATTICE);
                }
                let mut y = y1;
                while y > y0 {
                    ring.push(vertex(x0, y));
                    y -= side(x0 == 0);
                }
                fans.push((vertex(x0 + unit / 2, y0 + unit / 2), ring));
            }
        }
        for (center, ring) in fans {
            for k in 0..ring.len() {
                polygons.push(vec![center, ring[k], ring[(k + 1) % ring.len()]]);
            }
        }
        let period = self.to.wrap_period();
        let mut triangles = Vec::new();
        for tri in polygons {
            let Some(ids) = tri.into_iter().collect::<Option<Vec<u32>>>() else {
                continue;
            };
            let points: Vec<DVec2> = ids.iter().map(|&k| positions[k as usize]).collect();
            let spread = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max)
                - points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            if let Some(period) = period.filter(|period| spread > period / 2.0) {
                // Across the wrapping plane's seam: its own copies of the
                // corners, near the first. A triangle around the pole has
                // no such copy; the plane draws the pole as an edge, so it
                // is degenerate there anyway.
                let first = points[0].x;
                let xs: Vec<f64> = points.iter().map(|p| self.unwrap(p.x, first)).collect();
                let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if hi - lo > period / 2.0 {
                    continue;
                }
                let base = positions.len() as u32;
                for (&k, (x, p)) in ids.iter().zip(xs.iter().zip(&points)) {
                    positions.push(DVec2::new(*x, p.y));
                    sources.push(sources[k as usize]);
                }
                triangles.push([base, base + 1, base + 2]);
                continue;
            }
            triangles.push([ids[0], ids[1], ids[2]]);
        }
        if triangles.is_empty() {
            return None;
        }
        let mut frame = WorldRect {
            min: DVec2::splat(f64::INFINITY),
            max: DVec2::splat(f64::NEG_INFINITY),
        };
        for p in &positions {
            frame.min = frame.min.min(*p);
            frame.max = frame.max.max(*p);
        }
        let side = frame.size().max_element().max(f64::MIN_POSITIVE);
        let frame = WorldRect {
            min: DVec2::new(frame.min.x, frame.max.y - side),
            max: DVec2::new(frame.min.x + side, frame.max.y),
        };
        let positions = positions
            .into_iter()
            .map(|p| DVec2::new(p.x - frame.min.x, frame.max.y - p.y) / side)
            .collect();
        Some(WarpMesh {
            frame,
            positions,
            sources,
            triangles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn polar() -> TileMatrixSet {
        TileMatrixSet::by_id("UPSArcticWGS84Quad").unwrap()
    }

    fn mercator() -> TileMatrixSet {
        TileMatrixSet::by_id("WebMercatorQuad").unwrap()
    }

    /// Every mesh vertex lands where the projections put its source point.
    fn check_mesh(warp: &Warp, tile: TileIndex, tolerance: f64) -> WarpMesh {
        let mesh = warp.mesh(tile, tolerance, 1.0).unwrap();
        let size = mesh.frame.size();
        for (p, uv) in mesh.positions.iter().zip(&mesh.sources) {
            let world = DVec2::new(
                mesh.frame.min.x + p.x * size.x,
                mesh.frame.max.y - p.y * size.y,
            );
            let expected = warp.map(tile, *uv).unwrap();
            let dx = warp.unwrap(expected.x, world.x) - world.x;
            assert!(dx.abs() < 1e-6 * size.x && (expected.y - world.y).abs() < 1e-6 * size.y);
        }
        mesh
    }

    #[test]
    fn polar_tiles_on_web_mercator() {
        let warp = Warp::new(polar(), *mercator().projection());
        // A tile of the polar grid around Svalbard, away from the pole.
        let sval = polar()
            .projection()
            .forward(crate::Geodetic::new(15.0, 78.0));
        let tile = polar()
            .tiles_in(
                WorldRect {
                    min: sval,
                    max: sval,
                },
                6,
            )
            .next()
            .unwrap();
        let footprint = warp.footprint(tile).unwrap();
        let center = mercator()
            .projection()
            .forward(crate::Geodetic::new(15.0, 78.0));
        assert!(
            footprint.bounds.min.cmple(center).all() && footprint.bounds.max.cmpge(center).all()
        );
        let mesh = check_mesh(&warp, tile, 1.0 / 2048.0);
        assert!(mesh.triangles.len() >= 32);
        // The tile holding the pole spreads over the whole Mercator width.
        let pole = polar()
            .projection()
            .forward(crate::Geodetic::new(0.0, 90.0));
        let tile = polar()
            .tiles_in(
                WorldRect {
                    min: pole,
                    max: pole,
                },
                3,
            )
            .next()
            .unwrap();
        let footprint = warp.footprint(tile).unwrap();
        let width = mercator().projection().world_bounds().size().x;
        assert!(footprint.bounds.size().x >= width * 0.999);
        let mesh = check_mesh(&warp, tile, 1.0 / 2048.0);
        // Seam triangles keep every edge shorter than half the world.
        let size = mesh.frame.size();
        for t in &mesh.triangles {
            let xs = t.map(|k| mesh.positions[k as usize].x * size.x);
            let spread = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                - xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            assert!(spread < width / 2.0);
        }
    }

    /// Tile edges, which neighbours share vertex for vertex, keep within
    /// the tolerance between their vertices.
    #[test]
    fn edges_keep_within_the_tolerance() {
        let warp = Warp::new(mercator(), *polar().projection());
        for tile in [
            TileIndex::new(1, 0, 0),
            TileIndex::new(2, 1, 0),
            TileIndex::new(3, 3, 1),
            TileIndex::new(4, 7, 3),
        ] {
            let limit = warp.footprint(tile).unwrap().span / 2048.0;
            let mesh = warp.mesh(tile, 1.0 / 2048.0, 1.0).unwrap();
            let edge: Vec<f64> = mesh
                .sources
                .iter()
                .filter(|s| s.y == 1.0)
                .map(|s| s.x)
                .collect();
            assert!(edge.len() >= 65, "{tile:?}: {} edge vertices", edge.len());
            let at = |u: f64| warp.map(tile, DVec2::new(u, 1.0)).unwrap();
            for k in 0..64 {
                let (a, b) = (f64::from(k) / 64.0, f64::from(k + 1) / 64.0);
                let mid = at((a + b) / 2.0);
                assert!(
                    mid.distance((at(a) + at(b)) / 2.0) <= limit,
                    "{tile:?} at {a}"
                );
            }
        }
    }

    #[test]
    fn series_stay_within_the_tolerance() {
        let warp = Warp::new(mercator(), *polar().projection());
        for tile in [
            TileIndex::new(2, 1, 0),
            TileIndex::new(4, 7, 3),
            TileIndex::new(10, 600, 300),
        ] {
            let series = warp.series(tile, 1.0 / 2048.0).unwrap();
            let limit = series.scale / 2048.0;
            let side = 1.0 / f64::from(series.cells);
            for k in 0..=20 {
                for l in 0..=20 {
                    let uv = DVec2::new(f64::from(k), f64::from(l)) / 20.0;
                    let p = series.position(uv);
                    assert!(
                        p.distance(warp.map(tile, uv).unwrap()) <= limit * 1.5,
                        "{tile:?} {uv}"
                    );
                }
            }
            // Patches share their edges: both sides of a seam agree.
            for k in 1..series.cells {
                let u = f64::from(k) * side;
                for v in [0.1, 0.5, 0.9] {
                    let (left, right) = (DVec2::new(u - 1e-12, v), DVec2::new(u + 1e-12, v));
                    let gap = series.position(left).distance(series.position(right));
                    assert!(
                        gap < 1e-3,
                        "{tile:?} cells {} gap {gap} at {u},{v}",
                        series.cells
                    );
                }
            }
            assert!(series.max_segment > 0.0 && series.max_segment <= 1.0);
        }
        // A deep tile needs one patch.
        assert_eq!(
            warp.series(TileIndex::new(12, 2400, 1200), 1.0 / 2048.0)
                .unwrap()
                .cells,
            1
        );
    }

    #[test]
    fn mercator_tiles_on_a_polar_plane() {
        let warp = Warp::new(mercator(), *polar().projection());
        // Mercator's north-east quarter at zoom 1 maps to a quarter wedge
        // around the pole; a fine enough mesh bounds the error.
        let tile = TileIndex::new(1, 1, 0);
        let mesh = check_mesh(&warp, tile, 1.0 / 2048.0);
        assert!(mesh.triangles.len() > 100);
        let footprint = warp.footprint(tile).unwrap();
        assert!(footprint.span > 0.0);
        // Same-plane tiles map one to one.
        let same = Warp::new(mercator(), *mercator().projection());
        let tile = TileIndex::new(5, 10, 12);
        let mesh = check_mesh(&same, tile, 1.0 / 2048.0);
        // A 4×4 grid: four inner cells, and twelve boundary cells fanned
        // to 64 steps per tile edge (plus their 32 inner sides).
        assert_eq!(mesh.triangles.len(), 4 * 2 + 4 * 64 + 32);
        let footprint = same.footprint(tile).unwrap();
        assert!((footprint.span - mercator().tile_span(5)).abs() < 1e-6 * footprint.span);
    }

    /// Web Mercator's tiles on Equal Earth: those on the antimeridian keep
    /// their edges on their own side of the cut.
    #[test]
    fn tiles_on_the_seam_keep_their_side() {
        let ee = Projection::from_descriptor("EPSG:8857").unwrap();
        let warp = Warp::new(mercator(), ee);
        // Each tile's west and east edges at its middle row, on the cut.
        for (tile, west_cut, east_cut) in [
            (TileIndex::new(0, 0, 0), true, true),
            (TileIndex::new(3, 0, 3), true, false),
            (TileIndex::new(3, 7, 4), false, true),
        ] {
            let series = warp.series(tile, 1.0 / 2048.0).unwrap();
            let [left, right] = [0.0, 1.0].map(|u| series.position(DVec2::new(u, 0.5)));
            assert!(left.x < right.x, "{tile:?}: {left} .. {right}");
            let on_cut = |p: DVec2| (ee.inverse(p).lon.abs() - 180.0).abs() < 1e-3;
            assert_eq!(on_cut(left), west_cut, "{tile:?} west {left}");
            assert_eq!(on_cut(right), east_cut, "{tile:?} east {right}");
        }
    }
}
