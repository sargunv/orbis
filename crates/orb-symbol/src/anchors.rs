//! Label anchors along lines: GL JS `get_anchors.ts` and
//! `check_max_angle.ts`, line by line.

use std::collections::VecDeque;
use std::f64::consts::PI;

use crate::geometry::{Point, js_max, js_round};

/// GL JS's tile in a tile's units: the square anchors fall in and the grid
/// they round to. An overzoom slice laying out its labels in its ancestor
/// (`orb_tile::Ancestor`) has the ancestor's, `grid` of its units to one
/// of the ancestor's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Square {
    pub(crate) min: Point,
    pub(crate) size: f64,
    pub(crate) grid: f64,
}

impl Square {
    /// The tile's own square, `extent` units wide.
    #[cfg(test)]
    pub(crate) const fn tile(extent: f64) -> Self {
        Self {
            min: Point::new(0.0, 0.0),
            size: extent,
            grid: 1.0,
        }
    }

    fn contains(self, x: f64, y: f64) -> bool {
        x >= self.min.x
            && x < self.min.x + self.size
            && y >= self.min.y
            && y < self.min.y + self.size
    }

    fn on_edge(self, p: Point) -> bool {
        let max = Point::new(self.min.x + self.size, self.min.y + self.size);
        p.x == self.min.x || p.x == max.x || p.y == self.min.y || p.y == max.y
    }

    /// `Point._round` on the square's grid.
    pub(crate) fn round(self, p: Point) -> Point {
        if self.grid == 1.0 {
            return p.round();
        }
        let g = self.grid;
        Point::new(js_round(p.x / g) * g, js_round(p.y / g) * g)
    }
}

/// Where a symbol sits, in tile units (GL JS `Anchor`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Anchor {
    pub(crate) point: Point,
    /// The angle of the line at the anchor; 0 off lines.
    pub(crate) angle: f64,
    /// The line segment the anchor is on, for anchors along lines.
    pub(crate) segment: Option<usize>,
}

impl Anchor {
    pub(crate) fn at(point: Point) -> Self {
        Self {
            point,
            angle: 0.0,
            segment: None,
        }
    }
}

/// The along-line extent a label needs, in layout units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LabelExtent {
    /// The text box, `right - left`, if there is text.
    pub(crate) text: Option<f64>,
    /// The icon box, `right - left`, if there is an icon.
    pub(crate) icon: Option<f64>,
}

impl LabelExtent {
    fn angle_window_size(self, glyph_size: f64, box_scale: f64) -> f64 {
        if self.text.is_some() {
            3.0 / 5.0 * glyph_size * box_scale
        } else {
            0.0
        }
    }

    fn length(self) -> f64 {
        js_max(self.text.unwrap_or(0.0), self.icon.unwrap_or(0.0))
    }
}

/// JavaScript's `!x` for a number: zero or NaN.
fn is_falsy(x: f64) -> bool {
    x == 0.0 || x.is_nan()
}

fn line_length(line: &[Point]) -> f64 {
    let mut length = 0.0;
    for pair in line.windows(2) {
        length += pair[0].dist(pair[1]);
    }
    length
}

/// `interpolates.number`.
fn interpolate(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// One anchor at the middle of the line (`symbol-placement: line-center`),
/// if the label fits there (GL JS `getCenterAnchor`).
pub(crate) fn center_anchor(
    line: &[Point],
    max_angle: f64,
    label: LabelExtent,
    glyph_size: f64,
    box_scale: f64,
    square: Square,
) -> Option<Anchor> {
    let angle_window_size = label.angle_window_size(glyph_size, box_scale);
    let label_length = label.length() * box_scale;
    let mut prev_distance = 0.0;
    let center_distance = line_length(line) / 2.0;

    for (i, pair) in line.windows(2).enumerate() {
        let (a, b) = (pair[0], pair[1]);
        let segment_distance = a.dist(b);
        if prev_distance + segment_distance > center_distance {
            let t = (center_distance - prev_distance) / segment_distance;
            let point = square.round(Point::new(
                interpolate(a.x, b.x, t),
                interpolate(a.y, b.y, t),
            ));
            let anchor = Anchor {
                point,
                angle: b.angle_to(a),
                segment: Some(i),
            };
            let fits = is_falsy(angle_window_size)
                || check_max_angle(line, &anchor, label_length, angle_window_size, max_angle);
            return fits.then_some(anchor);
        }
        prev_distance += segment_distance;
    }
    None
}

/// Anchors every `spacing` along the line where the label fits and its
/// corners stay under `max_angle` (GL JS `getAnchors`), in `square`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn line_anchors(
    line: &[Point],
    mut spacing: f64,
    max_angle: f64,
    label: LabelExtent,
    glyph_size: f64,
    box_scale: f64,
    overscaling: f64,
    square: Square,
    out: &mut Vec<Anchor>,
) {
    let angle_window_size = label.angle_window_size(glyph_size, box_scale);
    let shaped_label_length = label.length();
    let label_length = shaped_label_length * box_scale;

    let is_line_continued = square.on_edge(line[0]);

    if spacing - label_length < spacing / 4.0 {
        spacing = label_length + spacing / 4.0;
    }

    let fixed_extra_offset = glyph_size * 2.0;
    let offset = if !is_line_continued {
        ((shaped_label_length / 2.0 + fixed_extra_offset) * box_scale * overscaling) % spacing
    } else {
        (spacing / 2.0 * overscaling) % spacing
    };

    let sample = Resample {
        line,
        spacing,
        angle_window_size,
        max_angle,
        label_length,
        is_line_continued,
        square,
    };
    sample.run(offset, false, out);
}

struct Resample<'a> {
    line: &'a [Point],
    spacing: f64,
    angle_window_size: f64,
    max_angle: f64,
    label_length: f64,
    is_line_continued: bool,
    square: Square,
}

impl Resample<'_> {
    fn run(&self, offset: f64, place_at_middle: bool, out: &mut Vec<Anchor>) {
        let start = out.len();
        let half_label_length = self.label_length / 2.0;
        let line_length = line_length(self.line);
        let mut distance = 0.0;
        let mut marked_distance = offset - self.spacing;

        for (i, pair) in self.line.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let segment_dist = a.dist(b);
            let angle = b.angle_to(a);

            while marked_distance + self.spacing < distance + segment_dist {
                marked_distance += self.spacing;
                let t = (marked_distance - distance) / segment_dist;
                let x = interpolate(a.x, b.x, t);
                let y = interpolate(a.y, b.y, t);

                if self.square.contains(x, y)
                    && marked_distance - half_label_length >= 0.0
                    && marked_distance + half_label_length <= line_length
                {
                    let anchor = Anchor {
                        point: self.square.round(Point::new(x, y)),
                        angle,
                        segment: Some(i),
                    };
                    if is_falsy(self.angle_window_size)
                        || check_max_angle(
                            self.line,
                            &anchor,
                            self.label_length,
                            self.angle_window_size,
                            self.max_angle,
                        )
                    {
                        out.push(anchor);
                    }
                }
            }
            distance += segment_dist;
        }

        if !place_at_middle && out.len() == start && !self.is_line_continued {
            self.run(distance / 2.0, true, out);
        }
    }
}

/// Whether no part of a label at `anchor` bends more than `max_angle`
/// within any `window_size` of its length (GL JS `checkMaxAngle`).
pub(crate) fn check_max_angle(
    line: &[Point],
    anchor: &Anchor,
    label_length: f64,
    window_size: f64,
    max_angle: f64,
) -> bool {
    let Some(segment) = anchor.segment else {
        return true;
    };
    if label_length == 0.0 {
        return true;
    }

    let mut p = anchor.point;
    let mut index = segment as isize + 1;
    let mut anchor_distance = 0.0;

    while anchor_distance > -label_length / 2.0 {
        index -= 1;
        if index < 0 {
            return false;
        }
        anchor_distance -= line[index as usize].dist(p);
        p = line[index as usize];
    }

    let mut index = index as usize;
    anchor_distance += line[index].dist(line[index + 1]);
    index += 1;

    let mut recent_corners: VecDeque<(f64, f64)> = VecDeque::new();
    let mut recent_angle_delta = 0.0;

    while anchor_distance < label_length / 2.0 {
        let prev = line[index - 1];
        let current = line[index];
        let Some(&next) = line.get(index + 1) else {
            return false;
        };

        let mut angle_delta = prev.angle_to(current) - current.angle_to(next);
        angle_delta = (((angle_delta + 3.0 * PI) % (PI * 2.0)) - PI).abs();

        recent_corners.push_back((anchor_distance, angle_delta));
        recent_angle_delta += angle_delta;

        while anchor_distance - recent_corners[0].0 > window_size {
            let (_, delta) = recent_corners.pop_front().expect("a corner is recent");
            recent_angle_delta -= delta;
        }

        if recent_angle_delta > max_angle {
            return false;
        }

        index += 1;
        anchor_distance += current.dist(next);
    }
    true
}
