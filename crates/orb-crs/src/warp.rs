//! A tile matrix set's tiles on another plane (the `RasterWarp` plan):
//! where each tile lands, for cover, and an error-bounded mesh that
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

/// A tile's grid on the target plane ([`Warp::grid`]): what its mesh
/// takes of the projections, so that [`Warp::mesh_on`] maps only the
/// points refined edges add.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WarpGrid {
    /// Cells per side.
    cells: u32,
    /// The x positions unwrap near: the footprint's centre.
    reference: f64,
    /// How far past the world bounds, in their sizes, positions are kept.
    reach: f64,
    /// Target positions by lattice point (`LATTICE` steps per side, x
    /// right, y down), `None` outside the domain or the kept bounds.
    points: HashMap<(u32, u32), Option<DVec2>>,
}

/// A tile on the target plane (or, as `Series<DVec3>`, on a sphere) as
/// quadratic patches over an even grid of its window.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Series<V = DVec2> {
    /// The window's centre's position on the target; patches are relative
    /// to it.
    pub anchor: V,
    /// The rectangle of the tile the patches span, its corners in tile
    /// fractions (y down): the whole tile, or the part the target draws
    /// where it only clamps the rest (Web Mercator past ±85.05°). Past a
    /// side short of the tile's edge, positions stay on that side, as the
    /// target's own clamp puts them.
    pub window: [DVec2; 2],
    /// Patches per side, along x and y.
    pub cells: [u32; 2],
    /// Row by row, top to bottom, relative to `anchor`.
    pub patches: Vec<Patch<V>>,
    /// The largest cell, its sides as fractions of the tile's, within which
    /// straight segments' images stay within the tolerance of their curves:
    /// the tile's geometry is cut along a grid of such cells
    /// ([`grid`](Self::grid)).
    pub max_cell: DVec2,
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

/// A map's positions on a lattice of `steps` per side, x then y
/// (fractions, y down), each mapped when first asked for.
struct Lattice<V> {
    steps: [u32; 2],
    points: Vec<Option<V>>,
}

impl<V: Node> Lattice<V> {
    fn new() -> Self {
        Self {
            steps: [0; 2],
            points: Vec::new(),
        }
    }

    /// Makes this a lattice of `steps` per side, multiples of the current
    /// one's, keeping the points it had.
    fn refine(&mut self, steps: [u32; 2]) {
        let n = steps[0] + 1;
        let mut points = vec![None; (n * (steps[1] + 1)) as usize];
        if self.steps[0] > 0 {
            let m = self.steps[0] + 1;
            let ratio = [steps[0] / self.steps[0], steps[1] / self.steps[1]];
            for q in 0..=self.steps[1] {
                for p in 0..m {
                    points[(q * ratio[1] * n + p * ratio[0]) as usize] =
                        self.points[(q * m + p) as usize];
                }
            }
        }
        *self = Self { steps, points };
    }

    /// The position of point `(p, q)`, or `None` where `map` doesn't reach.
    fn at(&mut self, map: &impl Fn(DVec2) -> Option<V>, p: u32, q: u32) -> Option<V> {
        let slot = &mut self.points[(q * (self.steps[0] + 1) + p) as usize];
        if slot.is_none() {
            let steps = DVec2::new(f64::from(self.steps[0]), f64::from(self.steps[1]));
            *slot = Some(map(DVec2::new(f64::from(p), f64::from(q)) / steps)?);
        }
        *slot
    }
}

/// The whole tile, as a series' window.
pub(crate) const WHOLE: [DVec2; 2] = [DVec2::ZERO, DVec2::ONE];

impl<V: Node> Series<V> {
    /// Fits the fewest equal patches over `window`, at most `max_patches`
    /// and doubling along whichever side strays, whose quadratics stay
    /// within `limit` of `map` (tile fractions, y down, to positions), or
    /// `None` where `map` doesn't reach or the window needs more. `map`
    /// returns positions continuous across the window (unwrapped).
    pub(crate) fn fit(
        map: impl Fn(DVec2) -> Option<V>,
        window: [DVec2; 2],
        limit: f64,
        scale: f64,
        max_patches: u32,
    ) -> Option<Self> {
        let size = window[1] - window[0];
        let map = |w: DVec2| map(window[0] + w * size);
        let anchor = map(DVec2::splat(0.5))?;
        // Patches' nodes lie at halves of a cell and their error samples at
        // fifths: neighbours share their edges' points, and each level's
        // points are among the next one's, so each is mapped once.
        let (mut nodes, mut samples) = (Lattice::new(), Lattice::new());
        let mut cells = [1u32; 2];
        loop {
            let [cx, cy] = cells;
            nodes.refine([2 * cx, 2 * cy]);
            samples.refine([5 * cx, 5 * cy]);
            let mut patches = Vec::with_capacity((cx * cy) as usize);
            // The worst error along rows of nodes (where only the quadratic
            // across x interpolates), along columns, and anywhere.
            let (mut across, mut down, mut error) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..cy {
                for i in 0..cx {
                    let mut points = [V::ZERO; 9];
                    for (k, point) in (0..).zip(&mut points) {
                        *point = nodes.at(&map, 2 * i + k % 3, 2 * j + k / 3)? - anchor;
                    }
                    let patch = Patch { points };
                    // Its worst point among a 6×6 sample between its nodes.
                    for b in 0..6 {
                        for a in 0..6 {
                            let st = DVec2::new(f64::from(a), f64::from(b)) / 2.5 - 1.0;
                            let exact = samples.at(&map, 5 * i + a, 5 * j + b)?;
                            let e = (anchor + patch.eval(st)).distance(exact);
                            error = error.max(e);
                            if b % 5 == 0 {
                                across = across.max(e);
                            }
                            if a % 5 == 0 {
                                down = down.max(e);
                            }
                        }
                    }
                    patches.push(patch);
                }
            }
            if error <= limit {
                // Second derivatives in tile fractions along each axis: a
                // patch's in its `[-1, 1]` parameters, times (2 / side)².
                let side = size / DVec2::new(f64::from(cx), f64::from(cy));
                let curvature = (patches.iter())
                    .map(|p| p.curvature() * 4.0 / (side * side))
                    .fold(DVec2::ZERO, DVec2::max);
                // A chord across a cell of sides `h` bows by at most
                // `(c.x·h.x² + c.y·h.y²) / 8`: each axis keeps half the limit.
                let max_cell = (4.0 * limit / curvature).powf(0.5).min(DVec2::ONE);
                return Some(Self {
                    anchor,
                    window,
                    cells,
                    patches,
                    max_cell,
                    scale,
                });
            }
            // Split the side that strays; both where neither does alone.
            let (x, y) = match (across > limit, down > limit) {
                (false, false) => (true, true),
                split => split,
            };
            cells = [cx << u32::from(x), cy << u32::from(y)];
            if cells[0] * cells[1] > max_patches {
                return None;
            }
        }
    }

    /// The patch holding a tile position (fractions) and the position's
    /// parameters in it, each in `[-1, 1]` within the tile.
    pub fn locate(&self, uv: DVec2) -> (&Patch<V>, DVec2) {
        let [lo, hi] = self.window;
        // Sides at the tile's edges extrapolate (geometry past the tile);
        // those short of it hold positions on them.
        let cut = |bound: f64, edge: f64, past: f64| if bound == edge { past } else { bound };
        let uv = uv.clamp(
            DVec2::new(cut(lo.x, 0.0, f64::MIN), cut(lo.y, 0.0, f64::MIN)),
            DVec2::new(cut(hi.x, 1.0, f64::MAX), cut(hi.y, 1.0, f64::MAX)),
        );
        let cells = self.cells_f64();
        let at = (uv - lo) / (hi - lo) * cells;
        let cell = at.floor().clamp(DVec2::ZERO, cells - 1.0);
        let patch = &self.patches[(cell.y as u32 * self.cells[0] + cell.x as u32) as usize];
        (patch, (at - cell) * 2.0 - 1.0)
    }

    fn cells_f64(&self) -> DVec2 {
        DVec2::new(f64::from(self.cells[0]), f64::from(self.cells[1]))
    }

    /// Cells per side of the grid to cut the tile's geometry along, x then
    /// y: the fewest, powers of two, within [`max_cell`](Self::max_cell)
    /// (at most 256).
    pub fn grid(&self) -> [u32; 2] {
        let cells = |max: f64| 1u32 << ((1.0 / max).log2().ceil().max(0.0) as u32).min(8);
        [cells(self.max_cell.x), cells(self.max_cell.y)]
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
        // Parameters run over a patch's side: 2 · cells per window side.
        let per = 2.0 * self.cells_f64() / (self.window[1] - self.window[0]);
        let [ds, dt] = patch.derivatives(st);
        [ds * per.x, dt * per.y]
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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// The largest second difference along its rows (x) and its columns
    /// (y): its second derivatives' bounds in its parameters.
    fn curvature(&self) -> DVec2 {
        let p = &self.points;
        (0..3)
            .map(|i| {
                DVec2::new(
                    (p[3 * i] - p[3 * i + 1] * 2.0 + p[3 * i + 2]).length(),
                    (p[i] - p[3 + i] * 2.0 + p[6 + i]).length(),
                )
            })
            .fold(DVec2::ZERO, DVec2::max)
    }
}

/// The most patches of a planar series.
const MAX_SERIES_PATCHES: u32 = 256;

/// Samples per side when finding a footprint.
const FOOTPRINT_STEPS: u32 = 8;
/// The finest mesh grid, cells per side, and the steps along tile edges.
const MAX_CELLS: u32 = 64;
/// How many times finer than a tile a neighbour may be and still share
/// its edge's vertices, as a power of two ([`Warp::mesh`]'s `edges`).
pub const MAX_EDGE_REFINE: u8 = 4;
/// Vertex positions per tile side: tile edges at `MAX_CELLS` steps (finer
/// beside finer tiles), and cell centres of the finest grid.
const LATTICE: u32 = (2 * MAX_CELLS) << MAX_EDGE_REFINE;

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
        let mut g = self.geodetic(tile, uv)?;
        // On a plane cut along a meridian, a tile's points on the cut (Web
        // Mercator's ±180° edges) take the side of the tile just inside them.
        if let Some(lon0) = self.to.seam() {
            let wrap = |lon: f64| (lon - lon0 + 540.0).rem_euclid(360.0) - 180.0;
            if (wrap(g.lon).abs() - 180.0).abs() < 1e-6 {
                let inside = uv + (DVec2::splat(0.5) - uv) * 1e-3;
                let near = self
                    .from
                    .projection()
                    .inverse(self.from.tile_point(tile, inside));
                g.lon = lon0 + 180.0f64.copysign(wrap(near.lon));
            }
        }
        let p = self.to.forward(g);
        p.is_finite().then_some(p)
    }

    /// A position in a tile (fractions, y down) on the ground, if the
    /// target plane contains it.
    fn geodetic(&self, tile: TileIndex, uv: DVec2) -> Option<crate::Geodetic> {
        let g = self
            .from
            .projection()
            .inverse(self.from.tile_point(tile, uv));
        (g.lon.is_finite() && g.lat.is_finite() && self.to.contains(g)).then_some(g)
    }

    /// Whether the target plane puts a position in a tile on its edge
    /// rather than projecting it.
    fn clamped(&self, tile: TileIndex, uv: DVec2) -> bool {
        self.geodetic(tile, uv).is_some_and(|g| self.to.clamps(g))
    }

    /// The rectangle of the tile (fractions, y down) the target plane
    /// projects: the whole tile, less the bands along its sides that it
    /// clamps onto its edge (a geographic tile's polar rows on Web
    /// Mercator), found to within 2⁻³⁰ of the tile's side. `None` if it
    /// clamps all of it. Positions it doesn't contain are left to the
    /// fit, which gives up on them.
    fn window(&self, tile: TileIndex) -> Option<[DVec2; 2]> {
        const STEPS: u32 = 16;
        let mut window = WHOLE;
        // Rows first, along the whole width, then columns along the rows
        // kept: each side's band, from the side's edge inward.
        for (axis, side) in [(1, 0), (1, 1), (0, 0), (0, 1)] {
            let drawn = |window: &[DVec2; 2], t: f64| {
                (0..=STEPS).all(|k| {
                    let across = window[0][1 - axis]
                        + (window[1][1 - axis] - window[0][1 - axis]) * f64::from(k)
                            / f64::from(STEPS);
                    let mut uv = DVec2::ZERO;
                    uv[axis] = t;
                    uv[1 - axis] = across;
                    !self.clamped(tile, uv)
                })
            };
            let (edge, inner) = (window[side][axis], window[1 - side][axis]);
            let at = |k: u32| edge + (inner - edge) * f64::from(k) / f64::from(STEPS);
            let first = (0..=STEPS).find(|&k| drawn(&window, at(k)))?;
            if first == 0 {
                continue;
            }
            // Between a line not drawn throughout and one that is.
            let (mut out, mut kept) = (at(first - 1), at(first));
            while (kept - out).abs() > 1.0 / f64::from(1u32 << 30) {
                let mid = (out + kept) / 2.0;
                if drawn(&window, mid) {
                    kept = mid;
                } else {
                    out = mid;
                }
            }
            window[side][axis] = kept;
        }
        Some(window)
    }

    /// `x` in the world copy nearest `reference`, on a wrapping plane.
    fn unwrap(&self, x: f64, reference: f64) -> f64 {
        let near = DVec2::new(reference, 0.0);
        self.to.nearest_copy(DVec2::new(x, 0.0), near).x
    }

    /// Whether the tile holds a pole that a wrapping target plane spreads
    /// along its edge (Web Mercator's), so the tile spans every copy. Only
    /// a source plane whose pole is one point (a polar set's) holds one: a
    /// geographic plane's pole is its top edge, and Web Mercator's own
    /// clamps the pole onto its edge.
    fn spans_wrap(&self, tile: TileIndex) -> bool {
        self.to.wrap_period().is_some()
            && [90.0, -90.0].into_iter().any(|lat: f64| {
                let pole = crate::Geodetic::new(0.0, lat);
                let from = self.from.projection();
                if !from.contains(pole) {
                    return false;
                }
                let p = from.forward(pole);
                let elsewhere = from.forward(crate::Geodetic::new(90.0, lat));
                if p.distance(elsewhere) > 1e-9 * from.world_bounds().size().max_element() {
                    return false;
                }
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
        let mut bounds = WorldRect::around(
            samples
                .iter()
                .map(|p| DVec2::new(self.unwrap(p.x, reference.x), p.y)),
        );
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

    /// The tile's side on the target plane at the centre of its area: the
    /// square root of the local area scale times the tile's side.
    fn local_span(&self, tile: TileIndex) -> Option<f64> {
        let h = 1.0 / 64.0;
        let c = crate::tms::area_center(&self.from, tile);
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
    /// `GpuPerVertex` plan's precision series): the fewest equal patches,
    /// at most 256, over the part of the tile the target projects
    /// ([`Series::window`]) whose quadratics stay within `tolerance` of the
    /// tile's side of the true position, or `None` where that part leaves
    /// the target plane's domain, crosses a wrapping plane's seam or needs
    /// more.
    pub fn series(&self, tile: TileIndex, tolerance: f64) -> Option<Series> {
        let footprint = self.footprint(tile)?;
        if self.spans_wrap(tile) {
            return None;
        }
        let window = self.window(tile)?;
        let anchor = self.map(tile, (window[0] + window[1]) / 2.0)?;
        Series::fit(
            |uv| {
                let p = self.map(tile, uv)?;
                Some(DVec2::new(self.unwrap(p.x, anchor.x), p.y))
            },
            window,
            tolerance * footprint.span,
            footprint.span,
            MAX_SERIES_PATCHES,
        )
    }

    /// The tile's mesh: [`grid`](Self::grid) then [`mesh_on`](Self::mesh_on).
    pub fn mesh(
        &self,
        tile: TileIndex,
        tolerance: f64,
        reach: f64,
        edges: [u8; 4],
    ) -> Option<WarpMesh> {
        self.mesh_on(tile, &self.grid(tile, tolerance, reach)?, edges)
    }

    /// The costly part of the tile's mesh: a grid fine enough that its
    /// linear interpolation strays from the true position by at most
    /// `tolerance` of the tile's side (to a 64-cell limit), keeping what
    /// lies within the target plane's world bounds grown by `reach` times
    /// their size, with every position its mesh takes without refined
    /// edges. `None` if no part of the tile reaches the target plane.
    pub fn grid(&self, tile: TileIndex, tolerance: f64, reach: f64) -> Option<WarpGrid> {
        let footprint = self.footprint(tile)?;
        let reference = footprint.bounds.center().x;
        let limit = tolerance * footprint.span;
        // Each point's exact position, mapped once: every search's corners
        // and midpoints lie on the finest grid's corners.
        let n = MAX_CELLS + 1;
        let mut exact: Vec<Option<Option<DVec2>>> = vec![None; (n * n) as usize];
        let mut at = |i: u32, j: u32| {
            *exact[(j * n + i) as usize].get_or_insert_with(|| {
                let uv = DVec2::new(f64::from(i), f64::from(j)) / f64::from(MAX_CELLS);
                self.map(tile, uv)
            })
        };
        let mut cells = 4;
        while cells < MAX_CELLS {
            // The worst cell: its edges' midpoints against their ends'
            // mean, and its centre against its corners'. (A cell can bend
            // so that its centre stays near the corners' mean while its
            // edges bow out, as a parallel does on a polar plane.)
            let unit = MAX_CELLS / cells;
            let half = unit / 2;
            let mut error: f64 = 0.0;
            for j in 0..cells {
                for i in 0..cells {
                    let (x, y) = (i * unit, j * unit);
                    let near = |p| self.kept(reference, reach, p);
                    let (Some(a), Some(b), Some(c), Some(d)) = (
                        near(at(x, y)),
                        near(at(x + unit, y)),
                        near(at(x, y + unit)),
                        near(at(x + unit, y + unit)),
                    ) else {
                        continue;
                    };
                    for ((mx, my), ends) in [
                        ((x + half, y), &[a, b][..]),
                        ((x, y + half), &[a, c][..]),
                        ((x + unit, y + half), &[b, d][..]),
                        ((x + half, y + unit), &[c, d][..]),
                        ((x + half, y + half), &[a, b, c, d][..]),
                    ] {
                        if let Some(p) = at(mx, my) {
                            let mean = ends.iter().copied().sum::<DVec2>() / ends.len() as f64;
                            error = error.max(self.unwrap_near(p, mean.x).distance(mean));
                        }
                    }
                }
            }
            if error <= limit {
                break;
            }
            cells *= 2;
        }
        let mut grid = WarpGrid {
            cells,
            reference,
            reach,
            points: HashMap::new(),
        };
        // What the mesh takes without refined edges: the grid's corners,
        // and boundary cells' centres and the edge steps along their
        // sides, keyed by lattice point.
        let (scale, unit) = (LATTICE / MAX_CELLS, MAX_CELLS / cells);
        let mut wanted = Vec::new();
        for j in (0..=MAX_CELLS).step_by(unit as usize) {
            for i in (0..=MAX_CELLS).step_by(unit as usize) {
                wanted.push((i, j));
            }
        }
        if unit > 1 {
            for k in 0..=MAX_CELLS {
                wanted.extend([(k, 0), (MAX_CELLS, k), (k, MAX_CELLS), (0, k)]);
            }
            let (first, last) = (unit / 2, MAX_CELLS - unit / 2);
            for k in 0..cells {
                let middle = k * unit + unit / 2;
                wanted.extend([
                    (middle, first),
                    (last, middle),
                    (middle, last),
                    (first, middle),
                ]);
            }
        }
        for (i, j) in wanted {
            let p = self.kept(reference, reach, at(i, j));
            grid.points.insert((i * scale, j * scale), p);
        }
        Some(grid)
    }

    /// A position on the target, in the world copy nearest `reference`.
    fn unwrap_near(&self, p: DVec2, reference: f64) -> DVec2 {
        DVec2::new(self.unwrap(p.x, reference), p.y)
    }

    /// An exact position as a mesh takes it: in the world copy nearest
    /// `reference`, if within the world bounds grown by `reach` times
    /// their size.
    fn kept(&self, reference: f64, reach: f64, p: Option<DVec2>) -> Option<DVec2> {
        let world = self.to.world_bounds();
        let (min, max) = (
            world.min - world.size() * reach,
            world.max + world.size() * reach,
        );
        p.map(|p| self.unwrap_near(p, reference))
            .filter(|p| p.cmpge(min).all() && p.cmple(max).all())
    }

    /// The tile's mesh on its `grid`. Its edges take 64 steps, each side
    /// (top, right, bottom, left in tile space) `2^edges[side]` times as
    /// many: as many as a neighbour that much finer takes along it, so the
    /// two share every vertex there and no T-junction cracks open. Only
    /// the points refined edges add are mapped here.
    pub fn mesh_on(&self, tile: TileIndex, grid: &WarpGrid, edges: [u8; 4]) -> Option<WarpMesh> {
        let cells = grid.cells;
        let place = |x: u32, y: u32| match grid.points.get(&(x, y)) {
            Some(&p) => p,
            None => {
                let uv = DVec2::new(f64::from(x), f64::from(y)) / f64::from(LATTICE);
                self.kept(grid.reference, grid.reach, self.map(tile, uv))
            }
        };
        // Vertices sit on a lattice of `LATTICE` steps per side: the grid's
        // corners, boundary cells' centres, and every `MAX_CELLS`th of the
        // tile's edges, so that neighbours share their edge vertices
        // whatever grid each chose (no T-junction cracks).
        let unit = LATTICE / cells;
        let step = edges.map(|e| LATTICE / (MAX_CELLS << e.min(MAX_EDGE_REFINE)));
        let mut lattice: HashMap<(u32, u32), Option<u32>> = HashMap::new();
        let mut positions: Vec<DVec2> = Vec::new();
        let mut sources = Vec::new();
        let mut vertex = |x: u32, y: u32| -> Option<u32> {
            *lattice.entry((x, y)).or_insert_with(|| {
                let uv = DVec2::new(f64::from(x), f64::from(y)) / f64::from(LATTICE);
                place(x, y).map(|p| {
                    positions.push(p);
                    sources.push(uv);
                    (positions.len() - 1) as u32
                })
            })
        };
        let mut polygons: Vec<[Option<u32>; 3]> = Vec::with_capacity((2 * cells * cells) as usize);
        let mut fans: Vec<(Option<u32>, Vec<Option<u32>>)> = Vec::new();
        for j in 0..cells {
            for i in 0..cells {
                let (x0, y0, x1, y1) = (i * unit, j * unit, (i + 1) * unit, (j + 1) * unit);
                let [top, right, bottom, left] = [y0 == 0, x1 == LATTICE, y1 == LATTICE, x0 == 0];
                let boundary = [top, right, bottom, left]
                    .into_iter()
                    .zip(step)
                    .any(|(on, step)| on && unit > step);
                if !boundary {
                    let [a, b, c, d] =
                        [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].map(|(x, y)| vertex(x, y));
                    polygons.extend([[a, b, c], [c, b, d]]);
                    continue;
                }
                // The cell's outline clockwise, with the tile edge's
                // vertices along the sides on it, as a fan from its centre.
                let mut ring = Vec::new();
                let side = |on_edge: bool, side: usize| if on_edge { step[side] } else { unit };
                let mut x = x0;
                while x < x1 {
                    ring.push(vertex(x, y0));
                    x += side(top, 0);
                }
                let mut y = y0;
                while y < y1 {
                    ring.push(vertex(x1, y));
                    y += side(right, 1);
                }
                let mut x = x1;
                while x > x0 {
                    ring.push(vertex(x, y1));
                    x -= side(bottom, 2);
                }
                let mut y = y1;
                while y > y0 {
                    ring.push(vertex(x0, y));
                    y -= side(left, 3);
                }
                fans.push((vertex(x0 + unit / 2, y0 + unit / 2), ring));
            }
        }
        for (center, ring) in fans {
            for k in 0..ring.len() {
                polygons.push([center, ring[k], ring[(k + 1) % ring.len()]]);
            }
        }
        let period = self.to.wrap_period();
        let mut triangles = Vec::new();
        for tri in polygons {
            let [Some(a), Some(b), Some(c)] = tri else {
                continue;
            };
            let ids = [a, b, c];
            let points = ids.map(|k| positions[k as usize]);
            let spread = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max)
                - points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
            if let Some(period) = period.filter(|period| spread > period / 2.0) {
                // Across the wrapping plane's seam: its own copies of the
                // corners, near the first. A triangle around the pole has
                // no such copy; the plane draws the pole as an edge, so it
                // is degenerate there anyway.
                let first = points[0].x;
                let xs = points.map(|p| self.unwrap(p.x, first));
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
            triangles.push(ids);
        }
        if triangles.is_empty() {
            return None;
        }
        let frame = WorldRect::around(positions.iter().copied());
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
        let mesh = warp.mesh(tile, tolerance, 1.0, [0; 4]).unwrap();
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
            let mesh = warp.mesh(tile, 1.0 / 2048.0, 1.0, [0; 4]).unwrap();
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

    /// A side refined once takes the points its two finer neighbours
    /// take along it.
    #[test]
    fn refined_edges_take_their_finer_neighbours_points() {
        let warp = Warp::new(mercator(), *polar().projection());
        let bottom = |tile, edges| {
            let mesh = warp.mesh(tile, 1.0 / 2048.0, 1.0, edges).unwrap();
            let mut xs: Vec<f64> = mesh
                .sources
                .iter()
                .filter(|s| s.y == 1.0)
                .map(|s| s.x)
                .collect();
            xs.sort_by(f64::total_cmp);
            xs.dedup();
            xs
        };
        let steps = |n: u32| {
            (0..=n)
                .map(|k| f64::from(k) / f64::from(n))
                .collect::<Vec<_>>()
        };
        assert_eq!(bottom(TileIndex::new(4, 7, 3), [0; 4]), steps(64));
        assert_eq!(bottom(TileIndex::new(3, 3, 1), [0, 0, 1, 0]), steps(128));
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
            for axis in 0..2 {
                let cells = series.cells[axis];
                for k in 1..cells {
                    let t = f64::from(k) / f64::from(cells);
                    for along in [0.1, 0.5, 0.9] {
                        let mut at = DVec2::splat(along);
                        at[axis] = t;
                        let mut nudge = DVec2::ZERO;
                        nudge[axis] = 1e-12;
                        let gap = series
                            .position(at - nudge)
                            .distance(series.position(at + nudge));
                        assert!(gap < 1e-3, "{tile:?} cells {cells} gap {gap} at {at}");
                    }
                }
            }
            assert!(series.max_cell.cmpgt(DVec2::ZERO).all());
            assert!(series.max_cell.cmple(DVec2::ONE).all());
        }
        // A deep tile needs one patch.
        assert_eq!(
            warp.series(TileIndex::new(12, 2400, 1200), 1.0 / 2048.0)
                .unwrap()
                .cells,
            [1, 1]
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

    /// Web Mercator's top row holds no pole (it stops at 85.05°), so on a
    /// geographic plane its tiles stay within their own longitudes and
    /// have patches to draw through.
    #[test]
    fn mercator_top_rows_hold_no_pole() {
        let geographic = Projection::from_descriptor("EPSG:4326").unwrap();
        let warp = Warp::new(mercator(), geographic);
        for tile in [
            TileIndex::new(1, 0, 0),
            TileIndex::new(1, 1, 0),
            TileIndex::new(3, 4, 0),
        ] {
            let footprint = warp.footprint(tile).unwrap();
            assert!(footprint.bounds.size().x < 180.5, "{tile:?}");
            assert!(warp.series(tile, 1.0 / 2048.0).is_some(), "{tile:?}");
        }
        // A polar set's tile around the pole does span every copy.
        let pole = polar()
            .projection()
            .forward(crate::Geodetic::new(0.0, 90.0));
        let tile = polar()
            .tiles_in(
                WorldRect {
                    min: pole,
                    max: pole,
                },
                2,
            )
            .next()
            .unwrap();
        let warp = Warp::new(polar(), geographic);
        assert!(warp.spans_wrap(tile));
        assert!(warp.series(tile, 1.0 / 2048.0).is_none());
    }

    /// A geographic set's polar rows on Web Mercator: their poles are
    /// edges, not points, so they stay within their own longitudes, and
    /// their series fit the part Web Mercator draws, holding the rest on
    /// its edge.
    #[test]
    fn geographic_polar_rows_draw_on_web_mercator() {
        let crs84 = TileMatrixSet::by_id("WorldCRS84Quad").unwrap();
        let warp = Warp::new(crs84.clone(), *mercator().projection());
        let world = mercator().projection().world_bounds();
        for z in 0..4u8 {
            let last = (1 << z) - 1;
            for (col, row) in (0..2u32 << z).flat_map(|c| [(c, 0), (c, last)]) {
                let tile = TileIndex::new(z, col, row);
                assert!(!warp.spans_wrap(tile), "{tile:?}");
                let footprint = warp.footprint(tile).unwrap();
                assert!(footprint.bounds.size().x < world.size().x / 1.9, "{tile:?}");
                let series = warp.series(tile, 1.0 / 2048.0).unwrap();
                // Longitude is linear on both planes: one column.
                assert_eq!(series.cells[0], 1, "{tile:?}");
                let limit = series.scale / 2048.0;
                let [lo, hi] = series.window;
                for k in 0..=20 {
                    for l in 0..=20 {
                        let uv = DVec2::new(f64::from(k), f64::from(l)) / 20.0;
                        let exact = warp.map(tile, uv).unwrap();
                        let p = series.position(uv);
                        if uv.y >= lo.y && uv.y <= hi.y {
                            let p = DVec2::new(warp.unwrap(p.x, exact.x), p.y);
                            assert!(p.distance(exact) <= limit * 1.5, "{tile:?} {uv}");
                        } else {
                            // Past 85.05°, on the edge Web Mercator clamps to.
                            let edge = if uv.y < lo.y {
                                world.max.y
                            } else {
                                world.min.y
                            };
                            assert!((p.y - edge).abs() <= limit, "{tile:?} {uv}: {p}");
                        }
                    }
                }
            }
        }
        // The window ends where Web Mercator does.
        let series = warp.series(TileIndex::new(2, 5, 0), 1.0 / 2048.0).unwrap();
        let top = crs84
            .projection()
            .inverse(crs84.tile_point(TileIndex::new(2, 5, 0), series.window[0]));
        assert!((top.lat - 85.051_128_779_806_59).abs() < 1e-6, "{top:?}");
    }
}
