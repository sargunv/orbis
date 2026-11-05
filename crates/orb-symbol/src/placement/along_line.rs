//! Glyphs along a projected line (GL JS `projection.ts`): what placement
//! needs to place collision circles for line labels. The GPU symbol pass
//! walks lines the same way to draw them.

use crate::bucket::PlacedSymbol;
use crate::geometry::Point;

use super::view::{Camera, Ground, TileId, TileView, symbol_elevation};

/// The plane labels are laid out in: screen pixels, or a pitched map
/// plane in pixels.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LabelPlane {
    Screen,
    /// Tile units → pitched label plane, a 2D affine.
    Pitched([f64; 4]),
}

/// GL JS `SymbolProjectionContext` with its `ProjectionCache`, for one
/// placed symbol.
pub(crate) struct LineContext<'a> {
    /// The symbol's line: its vertices of the bucket's lines.
    pub(crate) lines: &'a [[i16; 2]],
    pub(crate) plane: LabelPlane,
    pub(crate) tile: &'a TileView,
    pub(crate) camera: &'a Camera,
    pub(crate) anchor: Point,
    pub(crate) translation: [f64; 2],
    pub(crate) elevation: Elevation<'a>,
    projections: Vec<Option<Point>>,
    offsets: Vec<Option<Point>>,
    cached_anchor: Option<Point>,
}

/// Where a line label's points stand: GL JS's `elevationAt` over its
/// projection context.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Elevation<'a> {
    pub(crate) ground: Option<(&'a dyn Ground, TileId)>,
    pub(crate) height_offset: f64,
    pub(crate) anchor_ground: bool,
}

impl Elevation<'_> {
    #[cfg(test)]
    pub(crate) const FLAT: Self = Self {
        ground: None,
        height_offset: 0.0,
        anchor_ground: true,
    };

    pub(crate) fn at(&self, x: f64, y: f64) -> Option<f64> {
        let ground = self.ground.map(|(g, tile)| g.elevation(tile, x, y));
        symbol_elevation(ground, self.height_offset, self.anchor_ground)
    }
}

/// A point projected into the label plane, and how far in front of the
/// camera it is.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanePoint {
    pub(crate) point: Point,
    pub(crate) w: f64,
}

impl<'a> LineContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        lines: &'a [[i16; 2]],
        plane: LabelPlane,
        tile: &'a TileView,
        camera: &'a Camera,
        anchor: Point,
        translation: [f64; 2],
        elevation: Elevation<'a>,
    ) -> Self {
        Self {
            lines,
            plane,
            tile,
            camera,
            anchor,
            translation,
            elevation,
            projections: vec![None; lines.len()],
            offsets: vec![None; lines.len()],
            cached_anchor: None,
        }
    }

    fn vertex(&self, index: usize) -> Point {
        let v = self.lines[index];
        Point::new(f64::from(v[0]), f64::from(v[1]))
    }

    /// GL JS `projectTileCoordinatesToLabelPlane`.
    pub(crate) fn to_label_plane(&self, x: f64, y: f64) -> PlanePoint {
        let (x, y) = (x + self.translation[0], y + self.translation[1]);
        match self.plane {
            LabelPlane::Pitched(m) => PlanePoint {
                point: Point::new(m[0] * x + m[2] * y, m[1] * x + m[3] * y),
                w: 1.0,
            },
            LabelPlane::Screen => self.to_screen(x, y),
        }
    }

    /// Tile units to screen pixels, untranslated (GL JS
    /// `projectTileCoordinates` as `projectPathSpecialProjection` uses it).
    pub(crate) fn to_screen(&self, x: f64, y: f64) -> PlanePoint {
        let p = self.tile.project(x, y, self.elevation.at(x, y));
        PlanePoint {
            point: Point::new(
                (p.point.x * 0.5 + 0.5) * self.camera.width,
                (-p.point.y * 0.5 + 0.5) * self.camera.height,
            ),
            w: p.w,
        }
    }

    /// GL JS `projectLineVertexToLabelPlane`.
    fn line_vertex(&mut self, index: usize, synthetic: &Synthetic) -> Point {
        if let Some(p) = self.projections[index] {
            return p;
        }
        let current = self.vertex(index);
        let projection = self.to_label_plane(current.x, current.y);
        if projection.w > 0.0 {
            self.projections[index] = Some(projection.point);
            return projection.point;
        }
        // Behind the camera: a vertex along the line far enough for the
        // glyph instead, not cached.
        let previous = if synthetic.distance_from_anchor == 0.0 {
            self.anchor
        } else {
            self.vertex((index as isize - synthetic.direction) as usize)
        };
        let minimum = synthetic.abs_offset_x - synthetic.distance_from_anchor + 1.0;
        self.truncated_segment(previous, current, synthetic.previous_vertex, minimum)
    }

    /// GL JS `projectTruncatedLineSegmentToLabelPlane`.
    fn truncated_segment(
        &self,
        previous_tile: Point,
        current_tile: Point,
        previous_projected: Point,
        minimum_length: f64,
    ) -> Point {
        let unit = unit(sub(previous_tile, current_tile));
        let unit_vertex = previous_tile.add(unit);
        let projected = self.to_label_plane(unit_vertex.x, unit_vertex.y).point;
        let segment = sub(previous_projected, projected);
        let length = mag(segment);
        previous_projected.add(scale(segment, minimum_length / length))
    }

    /// GL JS `findOffsetIntersectionPoint`.
    #[allow(clippy::too_many_arguments)]
    fn offset_intersection(
        &mut self,
        index: usize,
        prev_to_current_normal: Point,
        current: Point,
        line: (usize, usize),
        offset_previous: Point,
        line_offset_y: f64,
        synthetic: &Synthetic,
    ) -> Point {
        if let Some(p) = self.offsets[index] {
            return p;
        }
        let offset_current = current.add(prev_to_current_normal);
        let next_index = index as isize + synthetic.direction;
        if next_index < line.0 as isize || next_index >= line.1 as isize {
            self.offsets[index] = Some(offset_current);
            return offset_current;
        }
        let next = self.line_vertex(next_index as usize, synthetic);
        let current_to_next_normal =
            offset_normal(sub(next, current), line_offset_y, synthetic.direction);
        let begin = current.add(current_to_next_normal);
        let end = next.add(current_to_next_normal);
        let doubles_back = self.doubles_back(index, synthetic.direction, line);
        let p = if doubles_back {
            offset_current
        } else {
            line_intersection(offset_previous, offset_current, begin, end).unwrap_or(offset_current)
        };
        self.offsets[index] = Some(p);
        p
    }

    /// GL JS `isLineDoublingBack`.
    fn doubles_back(&self, index: usize, direction: isize, line: (usize, usize)) -> bool {
        let (previous, next) = (index as isize - direction, index as isize + direction);
        if previous.min(next) < line.0 as isize || previous.max(next) >= line.1 as isize {
            return false;
        }
        let (p, c, n) = (
            self.vertex(previous as usize),
            self.vertex(index),
            self.vertex(next as usize),
        );
        let (in_x, in_y) = (c.x - p.x, c.y - p.y);
        let (out_x, out_y) = (n.x - c.x, n.y - c.y);
        in_x * out_y - in_y * out_x == 0.0 && in_x * out_x + in_y * out_y < 0.0
    }
}

struct Synthetic {
    distance_from_anchor: f64,
    previous_vertex: Point,
    direction: isize,
    abs_offset_x: f64,
}

/// A glyph placed on the line: the path from the anchor to it, ending at
/// the glyph.
#[derive(Clone, Debug)]
pub(crate) struct PlacedGlyph {
    pub(crate) path: Vec<Point>,
}

fn sub(a: Point, b: Point) -> Point {
    Point::new(a.x - b.x, a.y - b.y)
}

fn scale(a: Point, k: f64) -> Point {
    Point::new(a.x * k, a.y * k)
}

fn mag(a: Point) -> f64 {
    libm::sqrt(a.x * a.x + a.y * a.y)
}

fn unit(a: Point) -> Point {
    scale(a, 1.0 / mag(a))
}

/// GL JS `transformToOffsetNormal`: `unit(v).perp() × offset × direction`.
fn offset_normal(v: Point, offset: f64, direction: isize) -> Point {
    let u = unit(v);
    scale(Point::new(-u.y, u.x), offset * direction as f64)
}

/// GL JS `findLineIntersection`.
fn line_intersection(a1: Point, a2: Point, b1: Point, b2: Point) -> Option<Point> {
    let denominator = (b2.y - b1.y) * (a2.x - a1.x) - (b2.x - b1.x) * (a2.y - a1.y);
    if denominator == 0.0 {
        return None;
    }
    let ua = ((b2.x - b1.x) * (a1.y - b1.y) - (b2.y - b1.y) * (a1.x - b1.x)) / denominator;
    Some(Point::new(
        a1.x + ua * (a2.x - a1.x),
        a1.y + ua * (a2.y - a1.y),
    ))
}

/// GL JS `placeGlyphAlongLine`, without rotation (collision needs none).
#[allow(clippy::too_many_arguments)]
fn place_glyph(
    cx: &mut LineContext<'_>,
    offset_x: f64,
    line_offset_x: f64,
    line_offset_y: f64,
    flip: bool,
    anchor_segment: usize,
    line: (usize, usize),
) -> Option<PlacedGlyph> {
    let combined = if flip {
        offset_x - line_offset_x
    } else {
        offset_x + line_offset_x
    };
    let mut direction: isize = if combined > 0.0 { 1 } else { -1 };
    if flip {
        direction = -direction;
    }
    let mut index = if direction > 0 {
        (line.0 + anchor_segment) as isize
    } else {
        (line.0 + anchor_segment + 1) as isize
    };
    let anchor = match cx.cached_anchor {
        Some(a) => a,
        None => {
            let a = cx.to_label_plane(cx.anchor.x, cx.anchor.y).point;
            cx.cached_anchor = Some(a);
            a
        }
    };
    let mut current = anchor;
    let mut previous = anchor;
    let mut offset_intersection: Option<Point> = None;
    let mut offset_previous: Option<Point> = None;
    let mut distance = 0.0;
    let mut segment_distance = 0.0;
    let abs_offset_x = combined.abs();
    let mut path = Vec::new();
    let mut segment = Point::new(0.0, 0.0);
    while distance + segment_distance <= abs_offset_x {
        index += direction;
        if index < line.0 as isize || index >= line.1 as isize {
            return None;
        }
        distance += segment_distance;
        previous = current;
        offset_previous = offset_intersection;
        let synthetic = Synthetic {
            distance_from_anchor: distance,
            previous_vertex: previous,
            direction,
            abs_offset_x,
        };
        current = cx.line_vertex(index as usize, &synthetic);
        if line_offset_y == 0.0 {
            path.push(previous);
            segment = sub(current, previous);
        } else {
            let prev_to_current = sub(current, previous);
            let normal = if mag(prev_to_current) == 0.0 {
                // The anchor sits on the vertex: look one vertex ahead.
                let next = cx.line_vertex((index + direction) as usize, &synthetic);
                offset_normal(sub(next, current), line_offset_y, direction)
            } else {
                offset_normal(prev_to_current, line_offset_y, direction)
            };
            let start = *offset_previous.get_or_insert(previous.add(normal));
            let p = cx.offset_intersection(
                index as usize,
                normal,
                current,
                line,
                start,
                line_offset_y,
                &synthetic,
            );
            offset_intersection = Some(p);
            path.push(start);
            segment = sub(p, start);
        }
        segment_distance = mag(segment);
    }
    let t = (abs_offset_x - distance) / segment_distance;
    let p = scale(segment, t).add(offset_previous.unwrap_or(previous));
    path.push(p);
    Some(PlacedGlyph { path })
}

/// GL JS `placeFirstAndLastGlyph`, unflipped and not rotated to the line.
pub(crate) fn first_and_last_glyph(
    cx: &mut LineContext<'_>,
    font_scale: f64,
    glyph_offsets: &[f32],
    line_offset: [f64; 2],
    symbol: &PlacedSymbol,
) -> Option<(PlacedGlyph, PlacedGlyph)> {
    let [start, count] = symbol.quads.map(|v| v as usize);
    if count == 0 {
        return None;
    }
    // The context holds the symbol's own line.
    let line = (0, cx.lines.len());
    let segment = usize::from(symbol.segment);
    let first_offset = f64::from(glyph_offsets[start]);
    let last_offset = f64::from(glyph_offsets[start + count - 1]);
    let first = place_glyph(
        cx,
        font_scale * first_offset,
        line_offset[0],
        line_offset[1],
        false,
        segment,
        line,
    )?;
    let last = place_glyph(
        cx,
        font_scale * last_offset,
        line_offset[0],
        line_offset[1],
        false,
        segment,
        line,
    )?;
    Some((first, last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_glyphs_along_a_straight_line() {
        // The label plane is the tile plane at one pixel per unit.
        let camera = Camera {
            width: 512.0,
            height: 512.0,
            zoom: 0.0,
            bearing: 0.0,
            pitch: 0.0,
            roll: 0.0,
            camera_to_center_distance: 1.0,
        };
        let identity = TileView {
            id: TileId {
                z: 0,
                x: 0,
                y: 0,
                wrap: 0,
            },
            matrix: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            globe: None,
        };
        let lines = [[0, 0], [100, 0], [200, 0]];
        let plane = LabelPlane::Pitched([1.0, 0.0, 0.0, 1.0]);
        let mut cx = LineContext::new(
            &lines,
            plane,
            &identity,
            &camera,
            Point::new(50.0, 0.0),
            [0.0; 2],
            Elevation::FLAT,
        );
        let at = |g: &PlacedGlyph| *g.path.last().unwrap();
        let g = place_glyph(&mut cx, 80.0, 0.0, 0.0, false, 0, (0, 3)).unwrap();
        assert_eq!((at(&g).x, at(&g).y), (130.0, 0.0));
        assert_eq!(g.path.len(), 3);
        let back = place_glyph(&mut cx, -30.0, 0.0, 0.0, false, 0, (0, 3)).unwrap();
        assert_eq!(at(&back).x, 20.0);
        // Off the end of the line.
        assert!(place_glyph(&mut cx, 200.0, 0.0, 0.0, false, 0, (0, 3)).is_none());
        // Offset below the line.
        let below = place_glyph(&mut cx, 20.0, 0.0, 10.0, false, 0, (0, 3)).unwrap();
        assert_eq!((at(&below).x, at(&below).y), (70.0, 10.0));
    }
}
