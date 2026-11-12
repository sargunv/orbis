//! The line family: CPU extrusion of line strings and closed polygon outlines.
//!
//! Vertex layout, 4 u32 words (16 bytes), and a fifth with `lineMetrics`:
//! - Word 0: `x as u16 | (y as u16) << 16`, signed i16 tile units.
//! - Word 1: signed i8 `extrude_x`, `extrude_y`, `offset_x`, `offset_y`,
//!   in successive bytes. Both vectors are scaled by 63 and rounded like
//!   JavaScript's `Math.round`. Offset normals are bounded join miters,
//!   easing toward segment normals near hairpins.
//! - Word 2: flags: `round` in bit 0 and `up` in bit 1; bits 2–31 store
//!   half the distance in normalized 8192 units. An overzoom slice's lines
//!   run on from its data tile's (`layout_overscaled`), past 15 bits.
//! - Word 3: feature slot (position in the selection).
//! - Word 4, only from a source with `lineMetrics` (GL JS's `a_uv_x`): f32
//!   progress within this tiled line, before clip-range remapping, which
//!   only `line-gradient` reads. With metrics, word 2 stores global
//!   progress × 32767, quantized by two, as GL JS does; word 4 retains
//!   precision for per-clip ramps.
//!
//! Lines are clipped to `[-extent, 2 · extent]²`, which holds any source
//! buffer (GL JS's GeoJSON buffer reaches at most a tile) and fits i16: a
//! stroke drawn from a neighbour's line in the buffer reaches into the tile,
//! as GL JS's unclipped buckets do. The fragment shader must discard outside
//! the actual tile square; what lies well past it chunks apart
//! ([`MeshBuilder::set_focus`]), for the cull to drop. Clipped ends use butt caps. Sharp-corner vertices use MapLibre's 512-pixel tile at overscaling 1.
//!
//! Extrusion ported from MapLibre GL JS 2b68611a4e2d4f7f076f6c6ebf758e2d53f91471, src/data/bucket/line_bucket.ts
//! (addLine, addCurrentVertex, addHalfVertex and the fakeround pie slices):
//! <https://github.com/maplibre/maplibre-gl-js/blob/2b68611a4e2d4f7f076f6c6ebf758e2d53f91471/src/data/bucket/line_bucket.ts>.
//! BSD-3-Clause; copyright and terms in `LICENSE-MapLibre` alongside this module.
//! Distance follows the pinned shader's two-tile-unit quantization.

use orb_tile::js_round;
use std::ops::{Add, Mul, Sub};

use glam::DVec2;
use orb_tile::{Ancestor, GeometryType, Layer};

use crate::clip::{ClippedLine, clip_polyline};
use crate::fill::pack_position;
use crate::{Mesh, MeshBuilder};

/// Words per line vertex.
pub const STRIDE: usize = 4;
/// Words per line vertex from a source with `lineMetrics`: the progress
/// `line-gradient` reads follows.
pub const METRICS_STRIDE: usize = 5;
/// Half the line distance, in word 2's bits 2–31.
const DISTANCE_MASK: u32 = u32::MAX >> 2;
const EXTRUDE_SCALE: f64 = 63.0;
/// The farthest a vertex's extrusion or offset normal reaches along either
/// axis, in half widths (or offsets): a saturated byte over
/// [`EXTRUDE_SCALE`]. Miters and caps reach past the half width, so a
/// cull's margin for a line is this many outsets.
pub const MAX_EXTRUDE: f32 = i8::MAX as f32 / EXTRUDE_SCALE as f32;
const COS_HALF_SHARP_CORNER: f64 = 0.7933533402912352; // cos(75° / 2)
// Offset normals use the same signed bytes as extrusion. Near hairpins,
// ease away from the bisector so offsets do not run past neighbouring points.
const MAX_OFFSET_MITER_LENGTH: f64 = 2.0;
const OFFSET_FALLBACK_START: f64 = 3.0;
const OFFSET_FALLBACK_END: f64 = 8.0;

/// Geometric layout properties evaluated for one feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineLayout {
    pub join: LineJoin,
    pub cap: LineCap,
    pub miter_limit: f32,
    pub round_limit: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineJoin {
    Bevel,
    Round,
    Miter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineCap {
    Butt,
    Round,
    Square,
}

/// Lays out line strings and every polygon ring among `features` (indices
/// into `layer`), skipping points. Each feature's slot is its position in
/// `features`. Polygon rings are closed and have no caps. `parameters` is
/// called with the slot of each non-point feature.
///
/// # Panics
/// Panics if the extent plus clipping margin cannot fit signed i16 positions.
pub fn layout(
    layer: &Layer,
    features: &[u32],
    parameters: impl FnMut(usize) -> LineLayout,
) -> Mesh {
    layout_in_extent(layer, features, parameters, layer.extent(), false)
}

/// Rescales source positions before rounding sharp-corner subdivisions and
/// line distances. GL JS buckets use extent 8192 regardless of source extent.
/// The returned positions and chunk bounds use `output_extent`. With
/// `metrics`, vertices carry their progress ([`METRICS_STRIDE`]).
pub fn layout_in_extent(
    layer: &Layer,
    features: &[u32],
    parameters: impl FnMut(usize) -> LineLayout,
    output_extent: u32,
    metrics: bool,
) -> Mesh {
    let extent = i32::try_from(output_extent).expect("extent fits i32");
    let scale = f64::from(output_extent) / f64::from(layer.extent());
    let place = |p: &[i32; 2]| p.map(|v| js_round(f64::from(v) * scale) as i64);
    let frame = Frame {
        extent,
        sharp_corner_offset: 15.0 * f64::from(extent) / 512.0,
        overscale: 1.0,
        step: None,
        metrics,
    };
    lay_out(layer, features, parameters, place, &frame)
}

/// Lays out an overzoom slice's lines, as GL JS lays out the data tile it
/// overscales: `features` of the data tile's `layer` (one per slot),
/// along their whole lines there, in the slice's units (extent 8192) at
/// `ancestor`, its place in that tile. Line distance runs on from the
/// data tile's line starts, reset where GL JS resets it in the data tile,
/// and `lineMetrics` progress runs along its whole lines, so neither
/// starts again at the slice's edges. `parameters` is called with a slot.
/// With `cells`, lines get a vertex wherever they cross a grid of `cells`
/// (across, down) over the slice, as `TileFrame::subdivided` gives the slice's
/// own, for a curved warp. With `metrics`, vertices carry their progress.
pub fn layout_overscaled(
    layer: &Layer,
    features: &[u32],
    parameters: impl FnMut(usize) -> LineLayout,
    ancestor: Ancestor,
    cells: Option<[u32; 2]>,
    metrics: bool,
) -> Mesh {
    let overscale = f64::from(ancestor.scale);
    let grid = 8192.0 / f64::from(layer.extent());
    let offset = ancestor.cell.map(|c| i64::from(c) * 8192);
    let scale = i64::from(ancestor.scale);
    // GL JS's bucket rounds the data tile's vertices to its grid first.
    let place = |p: &[i32; 2]| {
        let p = p.map(|v| js_round(f64::from(v) * grid) as i64 * scale);
        [p[0] - offset[0], p[1] - offset[1]]
    };
    let frame = Frame {
        extent: 8192,
        // GL JS's sharp corners at its overscaling, in slice units.
        sharp_corner_offset: if overscale <= 16.0 {
            15.0 * 8192.0 / 512.0
        } else {
            0.0
        },
        overscale,
        step: cells
            .filter(|c| c.iter().any(|&c| c > 1))
            .map(|c| c.map(|c| 8192.0 / f64::from(c))),
        metrics,
    };
    lay_out(layer, features, parameters, place, &frame)
}

/// The units a layout's lines land in.
struct Frame {
    /// Units per tile side.
    extent: i32,
    sharp_corner_offset: f64,
    /// Units per unit of the tile the lines were laid out in by GL JS,
    /// whose units its distance resets count: an overzoom slice's scale in
    /// its data tile, 1 otherwise.
    overscale: f64,
    /// The grid lines get a vertex at, if any, this far apart (x then y).
    step: Option<[f64; 2]>,
    /// Whether vertices carry their progress ([`METRICS_STRIDE`]).
    metrics: bool,
}

fn lay_out(
    layer: &Layer,
    features: &[u32],
    mut parameters: impl FnMut(usize) -> LineLayout,
    place: impl Fn(&[i32; 2]) -> [i64; 2],
    frame: &Frame,
) -> Mesh {
    let extent = frame.extent;
    let margin = extent;
    assert!(
        extent <= i32::from(i16::MAX) - margin,
        "extent {extent} plus margin exceeds i16 positions"
    );
    let stride = if frame.metrics {
        METRICS_STRIDE
    } else {
        STRIDE
    };
    let mut builder = MeshBuilder::new(stride);
    // What lies past the square by more than a stroke usually reaches
    // (32 px at the tile's zoom) goes in chunks of its own, which the cull
    // drops unless the node's widths reach that far.
    let near = (extent / 16) as i16;
    let far = (extent as i16).saturating_add(near);
    builder.set_focus(crate::TileBox {
        min: [-near; 2],
        max: [far; 2],
    });
    let distance_scale = 8192.0 / f64::from(extent);
    let (mut scaled, mut lines) = (Vec::new(), Vec::new());
    for (slot, &feature) in features.iter().enumerate() {
        let feature = feature as usize;
        let kind = layer.geometry_type(feature);
        if kind == GeometryType::Point {
            continue;
        }
        let params = parameters(slot);
        let clips = feature_clips(layer, feature);
        // GL JS resets distance past half its 15 bits, without metrics.
        let reset = (clips.is_none()).then_some(16384.0 / distance_scale * frame.overscale);
        for part in layer.geometry(feature).parts() {
            for ring in part.rings() {
                let ring = match kind {
                    GeometryType::Polygon => layer.outline(ring),
                    _ => ring.into(),
                };
                scaled.clear();
                scaled.extend(ring.iter().map(&place));
                scaled.dedup();
                let closed = kind == GeometryType::Polygon;
                let total_distance: f64 = scaled.windows(2).map(|p| length(p[0], p[1])).sum();
                // GL JS's resets, unless the line is too short for any, as
                // most are. A ring's distance starts with its closing edge,
                // and runs at most twice its length.
                let longest = match (scaled.first(), scaled.last()) {
                    (Some(&a), Some(&b)) if closed => 2.0 * (total_distance + length(b, a)),
                    _ => total_distance,
                };
                let resets = reset
                    .filter(|&at| longest > at)
                    .map(|at| Resets::along(&scaled, closed, at, frame.sharp_corner_offset));
                clip_polyline(&scaled, closed, -margin, extent + margin, &mut lines);
                for line in &mut lines {
                    if let Some(step) = frame.step {
                        subdivide(line, step);
                    }
                    let start = resets
                        .as_ref()
                        .map_or(line.start_distance, |r| r.distance(line.start_distance));
                    Extrusion {
                        builder: &mut builder,
                        slot: slot as u32,
                        edges: [None; 2],
                        distance_scale,
                        overscale: frame.overscale,
                        reset,
                        distance: start,
                        distance_point: None,
                        clips,
                        total_distance,
                        last: None,
                        at_clip: false,
                        metrics: frame.metrics,
                    }
                    .add_line(line, &params, frame.sharp_corner_offset);
                }
            }
        }
    }
    builder.finish()
}

/// Adds a vertex to `line` wherever it crosses a grid line `step` apart.
fn subdivide(line: &mut ClippedLine, step: [f64; 2]) {
    let n = line.points.len();
    let mut points = Vec::with_capacity(n);
    let mut crossings = Vec::new();
    let mut seam = None;
    for i in 0..n {
        if line.seam.is_some_and(|(at, _)| at == i) {
            seam = Some(points.len());
        }
        points.push(line.points[i]);
        let next = match line.points.get(i + 1) {
            Some(&next) => next,
            None if line.closed => line.points[0],
            None => break,
        };
        // Clipped already: every line it crosses counts.
        let every = [f64::NEG_INFINITY, f64::INFINITY];
        orb_tile::grid_crossings(line.points[i], next, step, every, &mut crossings);
        points.extend_from_slice(&crossings);
    }
    line.points = points;
    line.seam = line.seam.zip(seam).map(|((_, d), at)| (at, d));
}

fn length(a: [i64; 2], b: [i64; 2]) -> f64 {
    (b[0] as f64 - a[0] as f64).hypot(b[1] as f64 - a[1] as f64)
}

/// Where GL JS resets a line's distance as it lays the whole line out: at
/// each vertex the distance since the last reset passes `at`. A piece of
/// the line clipped out of it starts with GL JS's distance there.
struct Resets(Vec<f64>);

impl Resets {
    /// GL JS's resets along `line` (deduplicated; a ring may repeat its
    /// first vertex), whose distance starts at its first vertex with the
    /// closing edge's length when it is a ring, as GL JS's does. It checks
    /// at each vertex and at the points it adds `sharp_offset` either side
    /// of a sharp corner.
    fn along(line: &[[i64; 2]], closed: bool, at: f64, sharp_offset: f64) -> Self {
        let n = line.len() - usize::from(closed && line.len() > 1 && line.first() == line.last());
        let ring = closed && n > 1;
        let len = n + usize::from(ring);
        let point = |i: usize| line[i % n].map(|v| v as f64);
        let direction = |a: [f64; 2], b: [f64; 2]| {
            let d = [b[0] - a[0], b[1] - a[1]];
            let m = d[0].hypot(d[1]);
            [d[0] / m, d[1] / m]
        };
        let mut distance = if ring {
            length(line[n - 1], line[0])
        } else {
            0.0
        };
        let mut resets = vec![0.0];
        let mut last = 0.0;
        let mut check = |distance: f64| {
            if distance - last > at {
                last = distance;
                resets.push(distance);
            }
        };
        for i in 0..len {
            let prev = if i > 0 {
                Some(point(i - 1))
            } else {
                ring.then(|| point(n - 1))
            };
            let next = if i + 1 < len {
                Some(point(i + 1))
            } else {
                ring.then(|| point(1))
            };
            let p = point(i);
            let sharp = match (prev, next) {
                (Some(a), Some(b)) if sharp_offset > 0.0 => {
                    let (d1, d2) = (direction(a, p), direction(p, b));
                    let cos = d1[0] * d2[0] + d1[1] * d2[1];
                    ((1.0 + cos) / 2.0).max(0.0).sqrt() < COS_HALF_SHARP_CORNER
                }
                _ => false,
            };
            let before = if i > 0 {
                length(line[i - 1], line[i % n])
            } else {
                0.0
            };
            if sharp && i > 0 && before > 2.0 * sharp_offset {
                check(distance + before - sharp_offset);
            }
            distance += before;
            check(distance);
            if let Some(b) = next.filter(|_| sharp && i + 1 < len) {
                let after = (b[0] - p[0]).hypot(b[1] - p[1]);
                if after > 2.0 * sharp_offset {
                    check(distance + sharp_offset);
                }
            }
        }
        Self(resets)
    }

    /// GL JS's distance at `distance` along the whole line, as
    /// [`ClippedLine::start_distance`] counts it.
    fn distance(&self, distance: f64) -> f64 {
        let i = self.0.partition_point(|&r| r <= distance);
        distance - self.0[i.saturating_sub(1)]
    }
}

/// GeoJSON-vt's range along the original line. Absent for ordinary vector
/// features and GeoJSON sources without `lineMetrics`.
pub fn feature_clips(layer: &Layer, feature: usize) -> Option<[f64; 2]> {
    let number = |key| match layer.property(key)?.get(feature)? {
        orb_tile::ValueRef::F64(v) => Some(v),
        orb_tile::ValueRef::I64(v) => Some(v as f64),
        orb_tile::ValueRef::U64(v) => Some(v as f64),
        _ => None,
    };
    let clips = [
        number("geojsonvt_clip_start")?,
        number("geojsonvt_clip_end")?,
    ];
    (clips[0].is_finite() && clips[1].is_finite() && clips[1] > clips[0]).then_some(clips)
}

type Point = DVec2;

fn point([x, y]: [i32; 2]) -> Point {
    Point::new(x.into(), y.into())
}

/// GL JS's `Point` operations that glam's differ from in their last bit.
trait JsPoint {
    /// Divided by its length (glam multiplies by its reciprocal); a zero
    /// vector stays zero.
    fn unit(self) -> Self;
    fn js_round(self) -> Self;
}

impl JsPoint for Point {
    fn unit(self) -> Self {
        let mag = self.length();
        if mag == 0.0 { self } else { self / mag }
    }

    fn js_round(self) -> Self {
        Point::new(js_round(self.x), js_round(self.y))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Join {
    Miter,
    Bevel,
    FakeRound,
    FlipBevel,
    Butt,
    Square,
    Round,
}

impl From<LineJoin> for Join {
    fn from(join: LineJoin) -> Self {
        match join {
            LineJoin::Bevel => Self::Bevel,
            LineJoin::Round => Self::Round,
            LineJoin::Miter => Self::Miter,
        }
    }
}

impl From<LineCap> for Join {
    fn from(cap: LineCap) -> Self {
        match cap {
            LineCap::Butt => Self::Butt,
            LineCap::Round => Self::Round,
            LineCap::Square => Self::Square,
        }
    }
}

struct Extrusion<'a> {
    builder: &'a mut MeshBuilder,
    slot: u32,
    // Last vertex on each side of the strip (GL JS's e1/e2).
    edges: [Option<u32>; 2],
    distance_scale: f64,
    /// GL JS's units per unit of the tile it lays the line out in, for
    /// `lineMetrics` progress packed as a distance there ([`Frame`]).
    overscale: f64,
    /// The distance past which GL JS starts it again, if it does.
    reset: Option<f64>,
    distance: f64,
    distance_point: Option<Point>,
    clips: Option<[f64; 2]>,
    total_distance: f64,
    /// The last pair added, as `add_current_vertex` took it.
    last: Option<(Point, Point, f64, f64, bool, Point)>,
    /// Whether the vertex being added is where the line was clipped: GL
    /// JS has no vertex there to reset its distance at.
    at_clip: bool,
    /// Whether vertices carry their progress ([`METRICS_STRIDE`]).
    metrics: bool,
}

impl Extrusion<'_> {
    /// Adds `line`, its distance starting from `self.distance`.
    fn add_line(&mut self, line: &ClippedLine, params: &LineLayout, sharp_offset: f64) {
        let points = &line.points;
        let len = points.len() + usize::from(line.closed);
        let miter_limit = if params.join == LineJoin::Bevel {
            1.05
        } else {
            f64::from(params.miter_limit)
        };
        let mut current: Option<Point> = None;
        let mut next_normal: Option<Point> = None;
        if line.closed {
            let last = point(points[points.len() - 1]);
            current = Some(last);
            self.distance_point = Some(last);
            next_normal = Some(point(points[0]).sub(last).unit().perp());
        }
        for i in 0..len {
            self.at_clip = (i == 0 && line.start_clipped) || (i == len - 1 && line.end_clipped);
            let next = if i == len - 1 {
                line.closed.then(|| point(points[1]))
            } else {
                Some(point(points[(i + 1) % points.len()]))
            };
            let prev = current;
            let p = point(points[i % points.len()]);
            let normal = next.map_or_else(
                || next_normal.expect("last segment normal"),
                |n| n.sub(p).unit().perp(),
            );
            let prev_normal = next_normal.unwrap_or(normal);
            next_normal = Some(normal);
            let mut join_normal = prev_normal.add(normal).unit();
            let cos_angle = prev_normal.dot(normal);
            let cos_half_angle = join_normal.dot(normal);
            let miter_length = if cos_half_angle == 0.0 {
                f64::INFINITY
            } else {
                1.0 / cos_half_angle
            };
            let approx_angle = 2.0 * (2.0 - 2.0 * cos_half_angle).sqrt();
            let middle = prev.is_some() && next.is_some();
            let sharp = cos_half_angle < COS_HALF_SHARP_CORNER && middle;
            let turns_left = prev_normal.x * normal.y - prev_normal.y * normal.x > 0.0;
            let offset_normal = join_normal.mul(miter_length.min(MAX_OFFSET_MITER_LENGTH));
            let offset_fallback = ((miter_length - OFFSET_FALLBACK_START)
                / (OFFSET_FALLBACK_END - OFFSET_FALLBACK_START))
                .clamp(0.0, 1.0);

            if sharp && i > 0 {
                let delta = p.sub(prev.expect("middle vertex"));
                let distance = delta.length();
                if distance > 2.0 * sharp_offset {
                    let before = p.sub(delta.mul(sharp_offset / distance).js_round());
                    self.add_current_vertex(before, prev_normal, 0.0, 0.0, false, prev_normal);
                }
            }

            let mut join = if middle {
                params.join.into()
            } else if line.closed
                || (i == 0 && line.start_clipped)
                || (i == len - 1 && line.end_clipped)
            {
                Join::Butt
            } else {
                params.cap.into()
            };
            if middle && join == Join::Round {
                if miter_length < f64::from(params.round_limit) {
                    join = Join::Miter;
                } else if miter_length <= 2.0 {
                    join = Join::FakeRound;
                }
            }
            if join == Join::Miter && miter_length > miter_limit {
                join = Join::Bevel;
            }
            if join == Join::Bevel {
                if miter_length > 2.0 {
                    join = Join::FlipBevel;
                }
                if miter_length < miter_limit {
                    join = Join::Miter;
                }
            }

            match join {
                Join::Miter => self.add_current_vertex(
                    p,
                    join_normal.mul(miter_length),
                    0.0,
                    0.0,
                    false,
                    offset_normal,
                ),
                Join::FlipBevel => {
                    join_normal = if miter_length > 100.0 {
                        normal.mul(-1.0)
                    } else {
                        let bevel_length = miter_length * prev_normal.add(normal).length()
                            / prev_normal.sub(normal).length();
                        join_normal
                            .perp()
                            .mul(bevel_length * if turns_left { -1.0 } else { 1.0 })
                    };
                    let fallback = |normal: Point| {
                        offset_normal.add(normal.sub(offset_normal).mul(offset_fallback))
                    };
                    self.add_current_vertex(p, join_normal, 0.0, 0.0, false, fallback(join_normal));
                    self.add_current_vertex(
                        p,
                        join_normal.mul(-1.0),
                        0.0,
                        0.0,
                        false,
                        fallback(join_normal.mul(-1.0)),
                    );
                }
                Join::Bevel | Join::FakeRound => {
                    let offset = -(miter_length * miter_length - 1.0).sqrt();
                    let (a, b) = if turns_left {
                        (offset, 0.0)
                    } else {
                        (0.0, offset)
                    };
                    if prev.is_some() {
                        self.add_current_vertex(p, prev_normal, a, b, false, offset_normal);
                    }
                    if join == Join::FakeRound {
                        let n =
                            js_round(approx_angle * 180.0 / std::f64::consts::PI / 20.0) as usize;
                        for m in 1..n {
                            let mut t = m as f64 / n as f64;
                            if t != 0.5 {
                                let t2 = t - 0.5;
                                let a = 1.0904
                                    + cos_angle
                                        * (-3.2452 + cos_angle * (3.55645 - cos_angle * 1.43519));
                                let b = 0.848013 + cos_angle * (-1.06021 + cos_angle * 0.215638);
                                t += t * t2 * (t - 1.0) * (a * t2 * t2 + b);
                            }
                            let extrude = normal
                                .sub(prev_normal)
                                .mul(t)
                                .add(prev_normal)
                                .unit()
                                .mul(if turns_left { -1.0 } else { 1.0 });
                            self.add_half_vertex(
                                p,
                                extrude,
                                offset_normal.mul(if turns_left { -1.0 } else { 1.0 }),
                                false,
                                turns_left,
                            );
                        }
                    }
                    if next.is_some() {
                        self.add_current_vertex(p, normal, -a, -b, false, offset_normal);
                    }
                }
                Join::Butt => {
                    self.add_current_vertex(p, join_normal, 0.0, 0.0, false, offset_normal)
                }
                Join::Square => {
                    let offset = if prev.is_some() { 1.0 } else { -1.0 };
                    self.add_current_vertex(p, join_normal, offset, offset, false, offset_normal);
                }
                Join::Round => {
                    let fallback = |normal: Point| {
                        offset_normal.add(normal.sub(offset_normal).mul(offset_fallback))
                    };
                    if prev.is_some() {
                        self.add_current_vertex(
                            p,
                            prev_normal,
                            0.0,
                            0.0,
                            false,
                            fallback(prev_normal),
                        );
                        self.add_current_vertex(
                            p,
                            prev_normal,
                            1.0,
                            1.0,
                            true,
                            fallback(prev_normal),
                        );
                    }
                    if next.is_some() {
                        self.add_current_vertex(p, normal, -1.0, -1.0, true, fallback(normal));
                        self.add_current_vertex(p, normal, 0.0, 0.0, false, fallback(normal));
                    }
                }
            }
            if let Some((at, distance)) = line.seam
                && at == i
                && let Some((p, normal, left, right, round, offset)) = self.last
            {
                // A ring's distance starts again at its first vertex.
                self.distance = distance;
                self.add_current_vertex(p, normal, left, right, round, offset);
            }
            current = Some(p);
            if sharp && i < len - 1 {
                let delta = next.expect("middle vertex").sub(p);
                let distance = delta.length();
                if distance > 2.0 * sharp_offset {
                    let after = p.add(delta.mul(sharp_offset / distance).js_round());
                    self.add_current_vertex(after, normal, 0.0, 0.0, false, normal);
                    current = Some(after);
                }
            }
        }
    }

    fn add_current_vertex(
        &mut self,
        p: Point,
        normal: Point,
        end_left: f64,
        end_right: f64,
        round: bool,
        offset_normal: Point,
    ) {
        if let Some(prev) = self.distance_point {
            self.distance += p.sub(prev).length();
        }
        self.distance_point = Some(p);
        let left = Point {
            x: normal.x + normal.y * end_left,
            y: normal.y - normal.x * end_left,
        };
        let right = Point {
            x: -normal.x + normal.y * end_right,
            y: -normal.y - normal.x * end_right,
        };
        self.add_half_vertex(p, left, offset_normal, round, false);
        self.add_half_vertex(p, right, offset_normal.mul(-1.0), round, true);
        self.last = Some((p, normal, end_left, end_right, round, offset_normal));
        if !self.at_clip && self.reset.is_some_and(|at| self.distance > at) {
            self.distance = 0.0;
            self.add_current_vertex(p, normal, end_left, end_right, round, offset_normal);
        }
    }

    fn progress(&self) -> f64 {
        if self.total_distance > 0.0 {
            self.distance / self.total_distance
        } else {
            0.0
        }
    }

    fn scaled_distance(&self) -> f64 {
        self.clips
            .map_or(self.distance * self.distance_scale, |[start, end]| {
                (start + (end - start) * self.progress()) * 32767.0 * self.overscale
            })
    }

    fn add_half_vertex(&mut self, p: Point, extrude: Point, offset: Point, round: bool, up: bool) {
        let position = [p.x as i16, p.y as i16];
        // A byte holds extrusions up to about twice the half width; a
        // longer miter (a `line-miter-limit` above 2) saturates, as
        // MapLibre Native clamps it. GL JS's Uint8 storage wraps instead,
        // folding the join's far side back over the line.
        let x = js_round(EXTRUDE_SCALE * extrude.x) as i8 as u8;
        let y = js_round(EXTRUDE_SCALE * extrude.y) as i8 as u8;
        let ox = js_round(EXTRUDE_SCALE * offset.x) as i8 as u8;
        let oy = js_round(EXTRUDE_SCALE * offset.y) as i8 as u8;
        let flags = u32::from(round)
            | u32::from(up) << 1
            | ((self.scaled_distance() / 2.0) as u32 & DISTANCE_MASK) << 2;
        let data = u32::from(x) | u32::from(y) << 8 | u32::from(ox) << 16 | u32::from(oy) << 24;
        let words = [
            pack_position(position[0], position[1]),
            data,
            flags,
            self.slot,
            (self.progress() as f32).to_bits(),
        ];
        let stride = if self.metrics { METRICS_STRIDE } else { STRIDE };
        let vertex = self.builder.push_vertex(position, &words[..stride]);
        if let [Some(e1), Some(e2)] = self.edges {
            self.builder.push_triangle(e1, vertex, e2);
        }
        self.edges[usize::from(up)] = Some(vertex);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_CHUNK_TRIANGLES;
    use orb_tile::LayerBuilder;

    fn params(join: LineJoin, cap: LineCap) -> LineLayout {
        LineLayout {
            join,
            cap,
            miter_limit: 2.0,
            round_limit: 1.05,
        }
    }

    fn feature(builder: &mut LayerBuilder, kind: GeometryType, rings: &[&[[i32; 2]]]) {
        builder.begin_feature(None, kind);
        builder.begin_part();
        for ring in rings {
            builder.begin_ring();
            for &point in *ring {
                builder.push_vertex(point);
            }
        }
    }

    fn line(points: &[[i32; 2]], params: &LineLayout) -> Mesh {
        let mut b = LayerBuilder::new("roads", 4096);
        feature(&mut b, GeometryType::LineString, &[points]);
        layout(&b.finish(), &[0], |_| *params)
    }

    fn positions(mesh: &Mesh) -> Vec<[i16; 2]> {
        mesh.vertices
            .as_chunks::<STRIDE>()
            .0
            .iter()
            .map(|v| [v[0] as u16 as i16, (v[0] >> 16) as u16 as i16])
            .collect()
    }

    // [extrude_x, extrude_y, flags], interpreted using the shader's format.
    fn data(mesh: &Mesh) -> Vec<[i16; 3]> {
        mesh.vertices
            .as_chunks::<STRIDE>()
            .0
            .iter()
            .map(|v| {
                let w = v[1];
                [
                    i16::from(w as u8 as i8),
                    i16::from((w >> 8) as u8 as i8),
                    (v[2] & 3) as i16,
                ]
            })
            .collect()
    }

    fn counts(mesh: &Mesh, vertices: usize, triangles: usize) {
        assert_eq!(mesh.vertices.len(), vertices * STRIDE);
        assert_eq!(mesh.vertex_indices().count(), triangles * 3);
        assert!(mesh.vertex_indices().all(|i| (i as usize) < vertices));
    }

    fn offsets(mesh: &Mesh) -> Vec<[i8; 2]> {
        mesh.vertices
            .as_chunks::<STRIDE>()
            .0
            .iter()
            .map(|v| [(v[1] >> 16) as u8 as i8, (v[1] >> 24) as u8 as i8])
            .collect()
    }

    #[test]
    fn miters_longer_than_a_byte_holds_saturate() {
        // MapLibre Native's `line-miter-limit/global-state`: a 53° peak
        // with a limit of 5 mitres 2.24 half widths up, past the 127 / 63
        // a byte holds. Its tips stop there, straight off the vertex,
        // where GL JS's wrapped byte points it down.
        let peak = line(
            &[[0, 200], [100, 0], [200, 200]],
            &LineLayout {
                miter_limit: 5.0,
                ..params(LineJoin::Miter, LineCap::Butt)
            },
        );
        let tips: Vec<_> = positions(&peak)
            .iter()
            .zip(data(&peak))
            .filter(|(p, _)| **p == [100, 0])
            .map(|(_, d)| [d[0], d[1]])
            .collect();
        assert!(tips.contains(&[0, -128]), "{tips:?}");
        assert!(tips.contains(&[0, 127]), "{tips:?}");
    }

    #[test]
    fn offset_normals_follow_joins_instead_of_cap_extrusion() {
        let square = line(
            &[[10, 10], [20, 10]],
            &params(LineJoin::Miter, LineCap::Square),
        );
        assert_eq!(offsets(&square), [[0, 63], [0, -63], [0, 63], [0, -63]]);

        let points = [[10, 10], [20, 10], [20, 20]];
        let bevel = line(&points, &params(LineJoin::Bevel, LineCap::Butt));
        assert_eq!(
            &offsets(&bevel)[2..6],
            &[[-63, 63], [63, -63], [-63, 63], [63, -63]]
        );
        let round = line(&points, &params(LineJoin::Round, LineCap::Butt));
        // Every outer pie slice follows the same join miter as its neighbours.
        assert_eq!(&offsets(&round)[4..7], &[[63, -63]; 3]);

        let acute = line(
            &[[10, 10], [20, 10], [16, 13]],
            &params(LineJoin::Bevel, LineCap::Butt),
        );
        // sqrt(10) miter: capped at 2, with fallback (sqrt(10)-3)/5.
        assert_eq!(
            &offsets(&acute)[2..6],
            &[[-115, 41], [115, -41], [-116, 37], [116, -37]]
        );
        let hairpin = line(
            &[[10, 10], [20, 10], [10, 10]],
            &params(LineJoin::Miter, LineCap::Butt),
        );
        assert_eq!(
            &offsets(&hairpin)[2..6],
            &[[0, 63], [0, -63], [0, -63], [0, 63]]
        );
    }

    #[test]
    fn parameters_and_vertices_follow_selection_slots() {
        let mut b = LayerBuilder::new("roads", 4096);
        for y in [10, 20] {
            feature(&mut b, GeometryType::LineString, &[&[[10, y], [20, y]]]);
        }
        let mut evaluated = Vec::new();
        let mesh = layout(&b.finish(), &[1, 0], |slot| {
            evaluated.push(slot);
            params(
                LineJoin::Miter,
                if slot == 0 {
                    LineCap::Round
                } else {
                    LineCap::Butt
                },
            )
        });
        assert_eq!(evaluated, [0, 1]);
        counts(&mesh, 12, 8);
        let vertices = mesh.vertices.as_chunks::<STRIDE>().0;
        assert!(vertices[..8].iter().all(|v| v[3] == 0));
        assert!(vertices[8..].iter().all(|v| v[3] == 1));
    }

    #[test]
    fn straight_segment_caps_match_gl_js() {
        // GL JS: horizontal normal (0,1); left=(endLeft,1),
        // right=(endRight,-1); offset normals stay perpendicular to the line.
        // addHalfVertex emits one triangle per vertex after the first pair.
        let points = [[10, 10], [20, 10]];
        let butt = line(&points, &params(LineJoin::Miter, LineCap::Butt));
        counts(&butt, 4, 2);
        assert_eq!(
            data(&butt),
            [[0, 63, 0], [0, -63, 2], [0, 63, 0], [0, -63, 2]]
        );

        let square = line(&points, &params(LineJoin::Miter, LineCap::Square));
        counts(&square, 4, 2);
        assert_eq!(
            data(&square),
            [[-63, 63, 0], [-63, -63, 2], [63, 63, 0], [63, -63, 2]]
        );

        let round = line(&points, &params(LineJoin::Miter, LineCap::Round));
        counts(&round, 8, 6);
        assert_eq!(
            data(&round),
            [
                [-63, 63, 1],
                [-63, -63, 3],
                [0, 63, 0],
                [0, -63, 2],
                [0, 63, 0],
                [0, -63, 2],
                [63, 63, 1],
                [63, -63, 3],
            ]
        );
        assert_eq!(
            positions(&round),
            [
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [20, 10],
                [20, 10],
                [20, 10],
                [20, 10]
            ]
        );
    }

    #[test]
    fn right_angle_joins_match_gl_js() {
        // Short segments avoid the independently tested sharp-corner inserts.
        // Normals (0,1),(-1,0) give miter sqrt(2), join vector (-1,1).
        let points = [[10, 10], [20, 10], [20, 20]];
        let miter = line(&points, &params(LineJoin::Miter, LineCap::Butt));
        counts(&miter, 6, 4);
        assert_eq!(
            data(&miter),
            [
                [0, 63, 0],
                [0, -63, 2],
                [-63, 63, 0],
                [63, -63, 2],
                [-63, 0, 0],
                [63, 0, 2]
            ]
        );

        // Bevel's effective miter limit is 1.05, offset=-sqrt(2-1)=-1.
        let bevel = line(&points, &params(LineJoin::Bevel, LineCap::Butt));
        counts(&bevel, 8, 6);
        assert_eq!(
            &data(&bevel)[2..6],
            &[[-63, 63, 0], [0, -63, 2], [-63, 63, 0], [63, 0, 2]]
        );

        // GL JS's approxAngle gives n=round(87.7/20)=4: three outer
        // pie vertices, approximately 22.5°, 45°, 67.5° into the join.
        let round = line(&points, &params(LineJoin::Round, LineCap::Butt));
        counts(&round, 11, 9);
        assert_eq!(
            &data(&round)[2..9],
            &[
                [-63, 63, 0],
                [0, -63, 2],
                [24, -58, 2],
                [45, -45, 2],
                [58, -24, 2],
                [-63, 63, 0],
                [63, 0, 2],
            ]
        );
    }

    #[test]
    fn right_turn_round_fan_uses_the_other_side() {
        let mesh = line(
            &[[10, 20], [20, 20], [20, 10]],
            &params(LineJoin::Round, LineCap::Butt),
        );
        counts(&mesh, 11, 9);
        assert_eq!(
            &data(&mesh)[2..9],
            &[
                [0, 63, 0],
                [-63, -63, 2],
                [24, 58, 0],
                [45, 45, 0],
                [58, 24, 0],
                [63, 0, 0],
                [-63, -63, 2],
            ]
        );
    }

    #[test]
    fn limits_select_miter_or_bevel_like_gl_js() {
        let points = [[10, 10], [20, 10], [20, 20]];
        let bevel = line(&points, &params(LineJoin::Bevel, LineCap::Butt));
        let mut p = params(LineJoin::Miter, LineCap::Butt);
        p.miter_limit = 1.0;
        assert_eq!(line(&points, &p), bevel, "sqrt(2) exceeds miter limit");
        p = params(LineJoin::Round, LineCap::Butt);
        p.round_limit = 1.5;
        assert_eq!(
            line(&points, &p),
            line(&points, &params(LineJoin::Miter, LineCap::Butt)),
            "sqrt(2) below round limit becomes miter"
        );
        // A nearly straight bevel becomes a miter below its fixed 1.05 limit.
        let shallow = [[10, 10], [20, 10], [30, 11]];
        assert_eq!(
            line(&shallow, &params(LineJoin::Bevel, LineCap::Butt)),
            line(&shallow, &params(LineJoin::Miter, LineCap::Butt))
        );
    }

    #[test]
    fn hairpin_uses_flipbevel_without_infinite_extrusion() {
        // Opposite normals cancel: miter=Infinity, flipbevel uses -nextNormal
        // and then nextNormal. This branch must never multiply 0 by Infinity.
        let mesh = line(
            &[[10, 10], [20, 10], [10, 10]],
            &params(LineJoin::Miter, LineCap::Butt),
        );
        counts(&mesh, 8, 6);
        assert_eq!(
            &data(&mesh)[2..6],
            &[[0, 63, 0], [0, -63, 2], [0, -63, 0], [0, 63, 2]]
        );

        // Finite miter >2: joinNormal.perp() times bevelLength, not a miter.
        // Normals (0,1),(-.6,-.8): bisector=(-3,1)/sqrt(10),
        // miter=sqrt(10), bevelLength=sqrt(10)/3, flipped => (1/3,1).
        let finite = line(
            &[[10, 10], [20, 10], [16, 13]],
            &params(LineJoin::Bevel, LineCap::Butt),
        );
        counts(&finite, 8, 6);
        assert_eq!(
            &data(&finite)[2..6],
            &[[21, 63, 0], [-21, -63, 2], [-21, -63, 0], [21, 63, 2]]
        );
    }

    #[test]
    fn acute_round_join_uses_round_caps_on_both_segments() {
        let mesh = line(
            &[[10, 10], [20, 10], [16, 13]],
            &params(LineJoin::Round, LineCap::Butt),
        );
        counts(&mesh, 12, 10);
        assert_eq!(
            &data(&mesh)[2..10],
            &[
                [0, 63, 0],
                [0, -63, 2],
                [63, 63, 1],
                [63, -63, 3],
                [13, -88, 1],
                [88, 13, 3],
                [-38, -50, 0],
                [38, 50, 2],
            ]
        );
    }

    #[test]
    fn sharp_corners_insert_points_fifteen_pixels_away() {
        // 4096/512 * 15 = 120 tile units. GL JS inserts a pair 120
        // before and after a 90° corner if each segment exceeds 240.
        let mesh = line(
            &[[0, 0], [1000, 0], [1000, 1000]],
            &params(LineJoin::Miter, LineCap::Butt),
        );
        counts(&mesh, 10, 8);
        assert_eq!(
            positions(&mesh),
            [
                [0, 0],
                [0, 0],
                [880, 0],
                [880, 0],
                [1000, 0],
                [1000, 0],
                [1000, 120],
                [1000, 120],
                [1000, 1000],
                [1000, 1000]
            ]
        );
        assert_eq!(
            &data(&mesh)[2..8],
            &[
                [0, 63, 0],
                [0, -63, 2],
                [-63, 63, 0],
                [63, -63, 2],
                [-63, 0, 0],
                [63, 0, 2]
            ]
        );
    }

    #[test]
    fn polygon_rings_close_without_caps_including_holes() {
        let outer = [[10, 10], [20, 10], [20, 20], [10, 20]];
        let hole = [[12, 12], [12, 18], [18, 18], [18, 12]];
        let mut b = LayerBuilder::new("boundaries", 4096);
        feature(&mut b, GeometryType::Polygon, &[&outer, &hole]);
        let layer = b.finish();
        let mesh = layout(&layer, &[0], |_| params(LineJoin::Miter, LineCap::Round));
        counts(&mesh, 20, 16);
        let positions = positions(&mesh);
        assert_eq!(&positions[0..2], &positions[8..10]);
        assert_eq!(&positions[10..12], &positions[18..20]);
        assert!(data(&mesh).iter().all(|v| v[2] & 1 == 0));
        assert!(mesh.vertex_indices().take(24).all(|i| i < 10));
        assert!(mesh.vertex_indices().skip(24).all(|i| i >= 10));
        assert_eq!(
            mesh,
            layout(&layer, &[0], |_| params(LineJoin::Miter, LineCap::Square))
        );
    }

    #[test]
    fn clipping_uses_margin_and_suppresses_only_synthetic_caps() {
        let mesh = line(
            &[[-5000, 50], [1000, 50], [9000, 50]],
            &params(LineJoin::Miter, LineCap::Round),
        );
        // GL JS resets the distance at the last vertex, past the clip, so
        // no pair repeats where the line is clipped.
        counts(&mesh, 6, 4);
        assert_eq!(
            positions(&mesh),
            [
                [-4096, 50],
                [-4096, 50],
                [1000, 50],
                [1000, 50],
                [8192, 50],
                [8192, 50],
            ]
        );
        assert!(data(&mesh).iter().all(|v| v[2] & 1 == 0));
        let one_end = line(
            &[[-5000, 50], [1000, 50]],
            &params(LineJoin::Miter, LineCap::Round),
        );
        counts(&one_end, 6, 4);
        assert_eq!(&data(&one_end)[..2], &[[0, 63, 0], [0, -63, 2]]);
        assert_eq!(&data(&one_end)[4..], &[[63, 63, 1], [63, -63, 3]]);
    }

    #[test]
    fn what_lies_well_past_the_square_chunks_apart() {
        // Along y = 50 from far west of the tile into it: the segments
        // west of x = −256 (a sixteenth of 4096) stand apart.
        let mesh = line(
            &[
                [-4000, 50],
                [-3000, 50],
                [-2000, 50],
                [-1000, 50],
                [1000, 50],
            ],
            &params(LineJoin::Miter, LineCap::Butt),
        );
        let bounds: Vec<_> = mesh
            .chunks
            .iter()
            .map(|c| (c.bounds.min, c.bounds.max))
            .collect();
        assert_eq!(
            bounds,
            [([-4000, 50], [-1000, 50]), ([-1000, 50], [1000, 50])]
        );
    }

    #[test]
    fn clipped_polygon_is_open_and_does_not_follow_clip_edges() {
        let mut b = LayerBuilder::new("boundaries", 4096);
        feature(
            &mut b,
            GeometryType::Polygon,
            &[&[[-5000, 10], [10, 10], [10, 20], [-5000, 20]]],
        );
        let mesh = layout(&b.finish(), &[0], |_| {
            params(LineJoin::Miter, LineCap::Round)
        });
        // Four positions and two sharp-corner insertions; GL JS resets the
        // distance at the last vertex, past the clip.
        counts(&mesh, 12, 10);
        let positions = positions(&mesh);
        assert_eq!(positions.first(), Some(&[-4096, 10]));
        assert_eq!(positions.last(), Some(&[-4096, 20]));
        // There is no artificial vertical edge joining the two clipped ends.
        assert!(
            mesh.vertex_indices()
                .collect::<Vec<_>>()
                .as_chunks::<3>()
                .0
                .iter()
                .all(|t| {
                    let p = t.map(|i| positions[i as usize]);
                    !(p.iter().all(|q| q[0] == -4096) && p.iter().any(|q| q[1] != p[0][1]))
                })
        );
        assert!(data(&mesh).iter().all(|v| v[2] & 1 == 0));
    }

    #[test]
    fn feature_slots_duplicates_and_degenerate_geometry() {
        let mut b = LayerBuilder::new("roads", 4096);
        feature(&mut b, GeometryType::LineString, &[&[[1, 1], [2, 1]]]);
        feature(&mut b, GeometryType::Point, &[&[[5, 5]]]);
        feature(
            &mut b,
            GeometryType::LineString,
            &[&[[10, 10], [10, 10], [20, 10], [20, 10], [20, 20], [20, 20]]],
        );
        feature(
            &mut b,
            GeometryType::LineString,
            &[&[], &[[1, 1]], &[[1, 1], [1, 1]]],
        );
        let mesh = layout(&b.finish(), &[1, 3, 2, 0], |_| {
            params(LineJoin::Miter, LineCap::Butt)
        });
        counts(&mesh, 10, 6);
        assert!(
            mesh.vertices[..6 * STRIDE]
                .as_chunks::<STRIDE>()
                .0
                .iter()
                .all(|v| v[3] == 2)
        );
        assert!(
            mesh.vertices[6 * STRIDE..]
                .as_chunks::<STRIDE>()
                .0
                .iter()
                .all(|v| v[3] == 3)
        );
        assert_eq!(
            data(&mesh)[..6],
            data(&line(
                &[[10, 10], [20, 10], [20, 20]],
                &params(LineJoin::Miter, LineCap::Butt)
            ))
        );
    }

    #[test]
    fn long_line_splits_chunks_without_breaking_the_strip() {
        let points: Vec<_> = (0..600).map(|x| [x, 100]).collect();
        let mesh = line(&points, &params(LineJoin::Miter, LineCap::Butt));
        counts(&mesh, 1200, 1198);
        assert!(
            mesh.chunks
                .iter()
                .all(|c| c.index_count <= MAX_CHUNK_TRIANGLES * 3)
        );
        let positions = positions(&mesh);
        for chunk in &mesh.chunks {
            for i in chunk.first_index..chunk.first_index + chunk.index_count {
                let index = chunk.first_vertex + u32::from(crate::mesh::index_at(&mesh.indices, i));
                let p = positions[index as usize];
                assert!((chunk.bounds.min[0]..=chunk.bounds.max[0]).contains(&p[0]));
                assert!((chunk.bounds.min[1]..=chunk.bounds.max[1]).contains(&p[1]));
            }
        }
    }

    #[test]
    fn metrics_preserve_local_precision_and_global_packed_progress() {
        let mut b = LayerBuilder::new("roads", 8192);
        feature(
            &mut b,
            GeometryType::LineString,
            &[&[[0, 0], [1000, 0], [1000, 1000]]],
        );
        b.set_property("geojsonvt_clip_start", orb_tile::Value::F64(0.25));
        b.set_property("geojsonvt_clip_end", orb_tile::Value::F64(0.75));
        let layer = b.finish();
        let butt = |_| params(LineJoin::Miter, LineCap::Butt);
        let mesh = layout_in_extent(&layer, &[0], butt, 8192, true);
        let vertices = mesh.vertices.as_chunks::<METRICS_STRIDE>().0;
        let first = vertices.first().unwrap();
        let last = vertices.last().unwrap();
        assert_eq!(first[2] >> 2, (0.25 * 32767.0 / 2.0) as u32);
        assert_eq!(last[2] >> 2, (0.75 * 32767.0 / 2.0) as u32);
        assert_eq!(f32::from_bits(first[4]), 0.0);
        assert_eq!(f32::from_bits(last[4]), 1.0);
        let corner = vertices
            .iter()
            .find(|v| v[0] == pack_position(1000, 0))
            .unwrap();
        assert_eq!(f32::from_bits(corner[4]), 0.5);
        assert_eq!(corner[2] >> 2, (0.5 * 32767.0 / 2.0) as u32);
        // Without metrics, vertices end at the feature slot.
        let mesh = layout(&layer, &[0], butt);
        assert_eq!(mesh.vertices.len() / STRIDE, vertices.len());
    }

    #[test]
    fn clipped_metrics_keep_distance_before_the_layout_clip() {
        let mut b = LayerBuilder::new("roads", 4096);
        feature(
            &mut b,
            GeometryType::LineString,
            &[&[[-5000, 0], [1000, 0]]],
        );
        b.set_property("geojsonvt_clip_start", orb_tile::Value::F64(0.0));
        b.set_property("geojsonvt_clip_end", orb_tile::Value::F64(1.0));
        let butt = |_| params(LineJoin::Miter, LineCap::Butt);
        let mesh = layout_in_extent(&b.finish(), &[0], butt, 8192, true);
        let first = &mesh.vertices[..METRICS_STRIDE];
        assert_eq!(f32::from_bits(first[4]), 904.0 / 6000.0);
        assert_eq!(first[2] >> 2, (904.0 / 6000.0 * 32767.0 / 2.0) as u32);
    }

    /// Distances (word 2) and progress (word 4) at each position, of a
    /// mesh with metrics.
    fn distances(mesh: &Mesh) -> Vec<([i16; 2], u32, f32)> {
        let words = mesh.vertices.as_chunks::<METRICS_STRIDE>().0;
        (words.iter())
            .map(|v| {
                let p = [v[0] as u16 as i16, (v[0] >> 16) as u16 as i16];
                (p, (v[2] >> 2) * 2, f32::from_bits(v[4]))
            })
            .collect()
    }

    /// An overzoom slice's lines lay out along the data tile's whole lines:
    /// at the data tile's vertices, each slice has the data tile's
    /// distance in its units, reset where the data tile's is, and its
    /// `lineMetrics` progress.
    #[test]
    fn overscaled_slices_run_on_along_the_data_tiles_lines() {
        // A spiral and a 12-gon in 30° turns, too gentle for sharp corners,
        // whose points differ between overscalings. Slices of four cut the
        // polygon's ring, through its first vertex for some.
        let around = |i: i32, r: f64| {
            let a = f64::from(i) * std::f64::consts::PI / 6.0;
            [4096.0 + r * a.cos(), 4096.0 + r * a.sin()].map(|v| v.round() as i32)
        };
        let spiral: Vec<_> = (0..18)
            .map(|i| around(i, 3800.0 - 120.0 * f64::from(i)))
            .collect();
        let gon: Vec<_> = (0..12).map(|i| around(i, 3800.0)).collect();
        let cases = [
            (GeometryType::LineString, &spiral, false, 2),
            (GeometryType::LineString, &spiral, true, 2),
            (GeometryType::Polygon, &gon, false, 4),
        ];
        for (kind, ring, metrics, scale) in cases {
            let mut b = LayerBuilder::new("roads", 8192);
            feature(&mut b, kind, &[ring]);
            if metrics {
                b.set_property("geojsonvt_clip_start", orb_tile::Value::F64(0.0));
                b.set_property("geojsonvt_clip_end", orb_tile::Value::F64(1.0));
            }
            let layer = b.finish();
            let butt = params(LineJoin::Miter, LineCap::Butt);
            let whole = distances(&layout_in_extent(&layer, &[0], |_| butt, 8192, true));
            // Without metrics, the whole line's distance resets once.
            let resets = whole.iter().filter(|&&(_, d, _)| d == 0).count();
            assert_eq!(
                resets,
                if metrics { 2 } else { 4 } - 2 * usize::from(kind == GeometryType::Polygon)
            );
            let mut matched = 0;
            for cell in (0..scale * scale).map(|c| [c % scale, c / scale]) {
                let ancestor = Ancestor { scale, cell };
                let slice = layout_overscaled(&layer, &[0], |_| butt, ancestor, None, true);
                for (q, d, progress) in distances(&slice) {
                    let p = [0, 1].map(|i| i32::from(q[i]) + (cell[i] * 8192) as i32);
                    if p.iter().any(|v| v % scale as i32 != 0) {
                        continue;
                    }
                    let p = p.map(|v| (v / scale as i32) as i16);
                    let at: Vec<_> = whole.iter().filter(|w| w.0 == p).collect();
                    if at.is_empty() {
                        continue;
                    }
                    matched += 1;
                    assert!(
                        at.iter().any(|w| d.abs_diff(w.1 * scale) <= 2 * scale
                            && (progress - w.2).abs() < 1e-3),
                        "{kind:?} {p:?} in {cell:?}: {d} {progress}, whole {at:?}"
                    );
                }
            }
            assert!(matched >= ring.len() * 2, "{matched}");
        }
    }

    #[test]
    fn ordinary_long_lines_duplicate_a_pair_at_distance_reset() {
        let points: Vec<_> = (0..40)
            .map(|i| [if i % 2 == 0 { 0 } else { 1000 }, 100])
            .collect();
        let mesh = line(&points, &params(LineJoin::Miter, LineCap::Butt));
        let vertices = mesh.vertices.as_chunks::<STRIDE>().0;
        assert!(
            vertices
                .windows(4)
                .any(|v| v[0][0] == v[2][0] && v[0][2] >> 2 > 8192 && v[2][2] >> 2 == 0)
        );
        assert!(vertices.iter().all(|v| v[2] >> 2 <= 16383));
    }
}
