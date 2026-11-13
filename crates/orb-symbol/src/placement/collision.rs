//! GL JS `collision_index.ts`: whether a symbol's boxes or circles fit in
//! the viewport among those already placed, and placing them.

use crate::bucket::PlacedSymbol;
use crate::geometry::{EXTENT, Point, js_max, js_min};
use crate::lines::{Lines, clip_line};
use crate::placement::along_line::{
    Elevation, LabelPlane, LineCache, LineContext, first_and_last_glyph,
};
use crate::placement::grid::{GridIndex, GridKey, Group, Overlap};
use crate::placement::view::{
    Camera, Ground, TileView, pitched_label_plane, symbol_elevation, tile_skew_vectors,
};
use orb_text::ONE_EM;

/// Pixels around the viewport placed as if inside it, so that symbols
/// entering the view change their neighbours offscreen.
pub(crate) const VIEWPORT_PADDING: f64 = 100.0;

/// A collision box around a tile-unit anchor, in tile units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CollisionBox {
    pub(crate) anchor: Point,
    pub(crate) x1: f64,
    pub(crate) y1: f64,
    pub(crate) x2: f64,
    pub(crate) y2: f64,
}

/// GL JS `PlacedBox`: where a box landed in the padded viewport.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PlacedBox {
    pub(crate) bounds: [f64; 4],
    pub(crate) placeable: bool,
    pub(crate) offscreen: bool,
}

/// GL JS `PlacedCircles`.
#[derive(Clone, Debug, Default)]
pub(crate) struct PlacedCircles {
    /// `x, y, radius` per circle.
    pub(crate) circles: Vec<[f64; 3]>,
    pub(crate) offscreen: bool,
    pub(crate) collision_detected: bool,
}

/// How a symbol's shapes are placed (the arguments GL JS threads through
/// `placeCollisionBox`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoxPlacement<'a> {
    pub(crate) tile: &'a TileView,
    pub(crate) overlap: Overlap,
    pub(crate) pitch_with_map: bool,
    pub(crate) rotate_with_map: bool,
    /// `*-translate` in tile units.
    pub(crate) translation: [f64; 2],
    pub(crate) group: Group,
    pub(crate) height_offset: f64,
    pub(crate) height_anchor_ground: bool,
    pub(crate) ground: Option<&'a dyn Ground>,
    /// Every shape must lie inside the viewport, unpadded: a tile scope's
    /// tile, for `symbol-avoid-edges` (MapLibre Native's `avoidEdges`).
    pub(crate) within_view: bool,
}

struct Projected {
    x: f64,
    y: f64,
    perspective_ratio: f64,
    w: f64,
}

/// The padded viewport placement projects into: the camera, and where
/// the screen and the collision grid end.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Viewport {
    pub(crate) camera: Camera,
    screen_right: f64,
    screen_bottom: f64,
    grid_right: f64,
    grid_bottom: f64,
}

/// What line labels' circles are worked out in, kept across labels.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    line: LineCache,
    first: Vec<Point>,
    last: Vec<Point>,
    path: Vec<Point>,
    screen: Vec<Point>,
    clipped: Lines,
    distances: Vec<f64>,
}

/// The placed shapes of a placement in the padded viewport's grid.
#[derive(Debug)]
pub(crate) struct CollisionIndex {
    grid: GridIndex,
    pub(crate) view: Viewport,
    scratch: Scratch,
}

/// Viewport pixels per tile unit at a tile's zoom (GL JS
/// `textPixelRatio`): a tile spans 512 of them, overscaled or not.
const PX_PER_TILE_UNIT: f64 = 512.0 / EXTENT;

/// Below this perspective ratio, near the horizon, nothing is placed.
const PERSPECTIVE_RATIO_CUTOFF: f64 = 0.6;

impl CollisionIndex {
    pub(crate) fn new(camera: Camera) -> Self {
        let view = Viewport::new(camera);
        Self {
            grid: GridIndex::new(view.grid_right, view.grid_bottom, 25.0),
            view,
            scratch: Scratch::default(),
        }
    }

    /// GL JS `placeCollisionBox`. `shift` is a variable anchor's, in layout
    /// pixels at the box's scale.
    pub(crate) fn place_box(
        &mut self,
        b: &CollisionBox,
        p: &BoxPlacement<'_>,
        shift: Option<Point>,
    ) -> PlacedBox {
        let mut placed = self.view.place_box(b, p, shift);
        let [x1, y1, x2, y2] = placed.bounds;
        if placed.placeable
            && p.overlap != Overlap::Always
            && self.grid.hit_test(x1, y1, x2, y2, p.overlap, p.group)
        {
            placed.placeable = false;
            placed.offscreen = false;
        }
        placed
    }

    /// GL JS `placeCollisionCircles` for a line label's centred placed
    /// symbol, without collision debug.
    pub(crate) fn place_circles(
        &mut self,
        p: &BoxPlacement<'_>,
        label: &LineLabel<'_>,
    ) -> PlacedCircles {
        let grid = &mut self.grid;
        self.view
            .place_circles(p, label, &mut self.scratch, |x, y, radius| {
                grid.hit_test_circle(x, y, radius, p.overlap, p.group)
            })
    }

    /// Adds a placed box; one that ignores placement collides with nothing.
    pub(crate) fn insert_box(&mut self, bounds: [f64; 4], key: GridKey, ignore_placement: bool) {
        if !ignore_placement {
            self.grid
                .insert(key, bounds[0], bounds[1], bounds[2], bounds[3]);
        }
    }

    /// Adds placed circles; those that ignore placement collide with
    /// nothing.
    pub(crate) fn insert_circles(
        &mut self,
        circles: &[[f64; 3]],
        key: GridKey,
        ignore_placement: bool,
    ) {
        if ignore_placement {
            return;
        }
        for &[x, y, r] in circles {
            self.grid.insert_circle(key, x, y, r);
        }
    }
}

/// A line label's centred placed symbol, as its collision circles need it.
pub(crate) struct LineLabel<'a> {
    pub(crate) symbol: &'a PlacedSymbol,
    /// The bucket's lines and per-quad glyph offsets.
    pub(crate) lines: &'a [[i16; 2]],
    pub(crate) glyph_offsets: &'a [f32],
    pub(crate) font_size: f64,
    pub(crate) circle_diameter: f64,
    /// `text-padding`, pixels.
    pub(crate) padding: f64,
}

impl Viewport {
    pub(crate) fn new(camera: Camera) -> Self {
        Self {
            screen_right: camera.width + VIEWPORT_PADDING,
            screen_bottom: camera.height + VIEWPORT_PADDING,
            grid_right: camera.width + 2.0 * VIEWPORT_PADDING,
            grid_bottom: camera.height + 2.0 * VIEWPORT_PADDING,
            camera,
        }
    }

    /// GL JS `projectAndGetPerspectiveRatio`, in padded viewport pixels.
    fn project(&self, tile: &TileView, x: f64, y: f64, elevation: Option<f64>) -> Projected {
        let p = tile.project(x, y, elevation);
        Projected {
            x: (p.point.x + 1.0) / 2.0 * self.camera.width + VIEWPORT_PADDING,
            y: (-p.point.y + 1.0) / 2.0 * self.camera.height + VIEWPORT_PADDING,
            perspective_ratio: 0.5 + 0.5 * (self.camera.camera_to_center_distance / p.w),
            w: p.w,
        }
    }

    /// Where a tile-unit anchor projects, in padded viewport pixels, on
    /// the ground if there is one.
    pub(crate) fn anchor_px(
        &self,
        tile: &TileView,
        anchor: [i16; 2],
        ground: Option<&dyn super::Ground>,
    ) -> Point {
        let (x, y) = (f64::from(anchor[0]), f64::from(anchor[1]));
        let elevation = ground.map(|g| g.elevation(tile.id, x, y));
        let p = self.project(tile, x, y, elevation);
        Point::new(p.x, p.y)
    }

    /// Whether a tile-unit anchor lies in the viewport, without its
    /// padding: right and bottom edges excluded, so that tiles of a grid
    /// share each anchor out.
    pub(crate) fn in_view(&self, tile: &TileView, anchor: [i16; 2]) -> bool {
        let p = self.project(tile, f64::from(anchor[0]), f64::from(anchor[1]), None);
        (VIEWPORT_PADDING..self.camera.width + VIEWPORT_PADDING).contains(&p.x)
            && (VIEWPORT_PADDING..self.camera.height + VIEWPORT_PADDING).contains(&p.y)
    }

    pub(crate) fn is_offscreen(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
        x2 < VIEWPORT_PADDING
            || x1 >= self.screen_right
            || y2 < VIEWPORT_PADDING
            || y1 > self.screen_bottom
    }

    fn is_inside_grid(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
        x2 >= 0.0 && x1 < self.grid_right && y2 >= 0.0 && y1 < self.grid_bottom
    }

    /// Whether bounds lie wholly inside the unpadded viewport (Native's
    /// `isInsideTile`).
    fn is_inside_view(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
        x1 >= VIEWPORT_PADDING
            && y1 >= VIEWPORT_PADDING
            && x2 < self.camera.width + VIEWPORT_PADDING
            && y2 < self.camera.height + VIEWPORT_PADDING
    }

    /// Where a box lands and whether it may be placed, before collisions
    /// (see [`CollisionIndex::place_box`]).
    fn place_box(&self, b: &CollisionBox, p: &BoxPlacement<'_>, shift: Option<Point>) -> PlacedBox {
        let x = b.anchor.x + p.translation[0];
        let y = b.anchor.y + p.translation[1];
        let ground = p.ground.map(|g| g.elevation(p.tile.id, x, y));
        let elevation = symbol_elevation(ground, p.height_offset, p.height_anchor_ground);
        let projected = self.project(p.tile, x, y, elevation);
        let tile_to_viewport = PX_PER_TILE_UNIT * projected.perspective_ratio;
        let bounds = if !p.pitch_with_map && !p.rotate_with_map {
            let px = projected.x + shift.map_or(0.0, |s| s.x * tile_to_viewport);
            let py = projected.y + shift.map_or(0.0, |s| s.y * tile_to_viewport);
            [
                px + b.x1 * tile_to_viewport,
                py + b.y1 * tile_to_viewport,
                px + b.x2 * tile_to_viewport,
                py + b.y2 * tile_to_viewport,
            ]
        } else {
            self.project_box(b, tile_to_viewport, p, &projected, shift, elevation)
        };
        let [x1, y1, x2, y2] = bounds;
        let unplaceable = projected.perspective_ratio < PERSPECTIVE_RATIO_CUTOFF
            || !self.is_inside_grid(x1, y1, x2, y2)
            || (p.within_view && !self.is_inside_view(x1, y1, x2, y2));
        let placeable = !unplaceable;
        PlacedBox {
            bounds,
            placeable,
            offscreen: placeable && self.is_offscreen(x1, y1, x2, y2),
        }
    }

    /// GL JS `_projectCollisionBox`: a map-rotated or map-pitched box's
    /// bounds through eight points of its outline.
    fn project_box(
        &self,
        b: &CollisionBox,
        tile_to_viewport: f64,
        p: &BoxPlacement<'_>,
        projected: &Projected,
        shift: Option<Point>,
        elevation: Option<f64>,
    ) -> [f64; 4] {
        let (mut east, mut south) = (Point::new(1.0, 0.0), Point::new(0.0, 1.0));
        let anchor_x = b.anchor.x + p.translation[0];
        let anchor_y = b.anchor.y + p.translation[1];
        if p.rotate_with_map && !p.pitch_with_map {
            let projected_east = self.project(p.tile, anchor_x + 1.0, anchor_y, elevation);
            let (dx, dy) = (
                projected_east.x - projected.x,
                projected_east.y - projected.y,
            );
            let angle =
                crate::math::atan(dy / dx) + if dx < 0.0 { std::f64::consts::PI } else { 0.0 };
            let (sin, cos) = (crate::math::sin(angle), crate::math::cos(angle));
            east = Point::new(cos, sin);
            south = Point::new(-sin, cos);
        } else if !p.rotate_with_map && p.pitch_with_map {
            (east, south) = tile_skew_vectors(&self.camera);
        }
        let (mut base_x, mut base_y) = (projected.x, projected.y);
        let mut multiplier = tile_to_viewport;
        if p.pitch_with_map {
            base_x = anchor_x;
            base_y = anchor_y;
            let zoom_fraction = self.camera.zoom - f64::from(p.tile.id.z);
            multiplier = 2f64.powf(-zoom_fraction);
            if shift.is_none() {
                let distance_ratio = projected.w / self.camera.camera_to_center_distance;
                multiplier *= (0.5 + 0.5 * distance_ratio).clamp(0.0, 4.0);
            }
        }
        if let Some(s) = shift {
            base_x += east.x * s.x * multiplier + south.x * s.y * multiplier;
            base_y += east.y * s.x * multiplier + south.y * s.y * multiplier;
        }
        let (x_min, x_max) = (b.x1 * multiplier, b.x2 * multiplier);
        let (y_min, y_max) = (b.y1 * multiplier, b.y2 * multiplier);
        let (x_half, y_half) = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0);
        let offsets = [
            (x_min, y_min),
            (x_half, y_min),
            (x_max, y_min),
            (x_max, y_half),
            (x_max, y_max),
            (x_half, y_max),
            (x_min, y_max),
            (x_min, y_half),
        ];
        let mut out = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (ox, oy) in offsets {
            let mut q = Point::new(
                base_x + east.x * ox + south.x * oy,
                base_y + east.y * ox + south.y * oy,
            );
            if p.pitch_with_map {
                let r = self.project(p.tile, q.x, q.y, elevation);
                q = Point::new(r.x, r.y);
            }
            // GL JS `getAABB` uses Math.min and Math.max.
            out = [
                js_min(out[0], q.x),
                js_min(out[1], q.y),
                js_max(out[2], q.x),
                js_max(out[3], q.y),
            ];
        }
        out
    }

    /// GL JS `placeCollisionCircles` for a line label's centred placed
    /// symbol, without collision debug: `hits` tests a circle against
    /// those placed, unless the label may overlap anything.
    pub(crate) fn place_circles(
        &self,
        p: &BoxPlacement<'_>,
        label: &LineLabel<'_>,
        scratch: &mut Scratch,
        mut hits: impl FnMut(f64, f64, f64) -> bool,
    ) -> PlacedCircles {
        let (symbol, tile) = (label.symbol, p.tile);
        let anchor = Point::new(f64::from(symbol.anchor[0]), f64::from(symbol.anchor[1]));
        // GL JS takes the anchor's perspective on the terrain alone.
        let anchor_elevation = p.ground.map(|g| g.elevation(tile.id, anchor.x, anchor.y));
        let projected_anchor = self.project(tile, anchor.x, anchor.y, anchor_elevation);
        let perspective_ratio = projected_anchor.perspective_ratio;
        let plane_font_size = if p.pitch_with_map {
            label.font_size / perspective_ratio
        } else {
            label.font_size * perspective_ratio
        };
        let font_scale = plane_font_size / ONE_EM;
        let line_offset = [
            f64::from(symbol.line_offset[0]) * font_scale,
            f64::from(symbol.line_offset[1]) * font_scale,
        ];
        let units_per_pixel = tile.tile_units_per_pixel(&self.camera);
        let plane = if p.pitch_with_map {
            LabelPlane::Pitched(pitched_label_plane(
                p.rotate_with_map,
                &self.camera,
                units_per_pixel,
            ))
        } else {
            LabelPlane::Screen
        };
        let line = &label.lines[symbol.line[0] as usize..][..symbol.line[1] as usize];
        let Scratch {
            line: cache,
            first,
            last,
            path,
            screen,
            clipped,
            distances,
        } = scratch;
        let mut cx = LineContext::new(
            line,
            plane,
            tile,
            &self.camera,
            anchor,
            p.translation,
            Elevation {
                ground: p.ground.map(|g| (g, tile.id)),
                height_offset: p.height_offset,
                anchor_ground: p.height_anchor_ground,
            },
            cache,
        );
        let mut in_grid = false;
        let mut entirely_offscreen = true;
        let mut placed = Vec::new();
        if first_and_last_glyph(
            &mut cx,
            font_scale,
            label.glyph_offsets,
            line_offset,
            symbol,
            (first, last),
        ) {
            let radius = label.circle_diameter * 0.5 * perspective_ratio + label.padding;
            let (min, max) = (
                Point::new(-VIEWPORT_PADDING, -VIEWPORT_PADDING),
                Point::new(self.screen_right, self.screen_bottom),
            );
            path.clear();
            path.extend(first[1..].iter().rev());
            path.extend_from_slice(&last[1..]);
            let circle_distance = radius * 2.5;
            if let LabelPlane::Pitched(m) = plane {
                // Back to tile units, then to the screen; nothing if any
                // point is behind the camera.
                let det = m[0] * m[3] - m[2] * m[1];
                let inverse = [m[3] / det, -m[1] / det, -m[2] / det, m[0] / det];
                screen.clear();
                let mut behind = false;
                for v in path.iter() {
                    let tile_x = inverse[0] * v.x + inverse[2] * v.y;
                    let tile_y = inverse[1] * v.x + inverse[3] * v.y;
                    let q = cx.to_screen(tile_x, tile_y);
                    behind |= q.w <= 0.0;
                    screen.push(q.point);
                }
                std::mem::swap(path, screen);
                if behind {
                    path.clear();
                }
            }
            clipped.clear();
            let mut whole = false;
            if !path.is_empty() {
                let mut lo = path[0];
                let mut hi = path[0];
                for v in &path[1..] {
                    lo = Point::new(js_min(lo.x, v.x), js_min(lo.y, v.y));
                    hi = Point::new(js_max(hi.x, v.x), js_max(hi.y, v.y));
                }
                if lo.x >= min.x && hi.x <= max.x && lo.y >= min.y && hi.y <= max.y {
                    whole = true;
                } else if !(hi.x < min.x || lo.x > max.x || hi.y < min.y || lo.y > max.y) {
                    clip_line([&path[..]], min.x, min.y, max.x, max.y, clipped);
                }
            }
            let whole = whole.then_some(&path[..]);
            for segment in whole.into_iter().chain(clipped.iter()) {
                let interpolator = PathInterpolator::new(segment, radius * 0.25, distances);
                let circles = if interpolator.length <= 0.5 * radius {
                    1
                } else {
                    (interpolator.padded_length / circle_distance).ceil() as usize + 1
                };
                for i in 0..circles {
                    let t = i as f64 / js_max((circles - 1) as f64, 1.0);
                    let at = interpolator.lerp(t);
                    let (x, y) = (at.x + VIEWPORT_PADDING, at.y + VIEWPORT_PADDING);
                    placed.push([x, y, radius]);
                    let (x1, y1, x2, y2) = (x - radius, y - radius, x + radius, y + radius);
                    entirely_offscreen &= self.is_offscreen(x1, y1, x2, y2);
                    in_grid |= self.is_inside_grid(x1, y1, x2, y2);
                    if (p.within_view && !self.is_inside_view(x1, y1, x2, y2))
                        || (p.overlap != Overlap::Always && hits(x, y, radius))
                    {
                        return PlacedCircles {
                            circles: Vec::new(),
                            offscreen: false,
                            collision_detected: true,
                        };
                    }
                }
            }
        }
        if !in_grid || perspective_ratio < PERSPECTIVE_RATIO_CUTOFF {
            placed.clear();
        }
        PlacedCircles {
            circles: placed,
            offscreen: entirely_offscreen,
            collision_detected: false,
        }
    }
}

/// GL JS `PathInterpolator`.
struct PathInterpolator<'a> {
    points: &'a [Point],
    distances: &'a [f64],
    length: f64,
    padded_length: f64,
    padding: f64,
}

impl<'a> PathInterpolator<'a> {
    /// Over `points`, with their distances along the path kept in
    /// `distances`.
    fn new(points: &'a [Point], padding: f64, distances: &'a mut Vec<f64>) -> Self {
        distances.clear();
        distances.push(0.0);
        for i in 1..points.len() {
            distances.push(distances[i - 1] + points[i].dist(points[i - 1]));
        }
        let length = *distances.last().expect("one distance");
        let padding = js_min(padding, length * 0.5);
        Self {
            points,
            distances,
            length,
            padded_length: length - padding * 2.0,
            padding,
        }
    }

    fn lerp(&self, t: f64) -> Point {
        if self.points.len() == 1 {
            return self.points[0];
        }
        let t = t.clamp(0.0, 1.0);
        let mut current = 1;
        let mut current_distance = self.distances[current];
        let target = t * self.padded_length + self.padding;
        while current_distance < target && current < self.distances.len() {
            current += 1;
            current_distance = self.distances.get(current).copied().unwrap_or(f64::NAN);
        }
        let previous = current - 1;
        let segment_length = current_distance - self.distances[previous];
        let segment_t = if segment_length > 0.0 {
            (target - self.distances[previous]) / segment_length
        } else {
            0.0
        };
        let (a, b) = (
            self.points[previous],
            self.points[current.min(self.points.len() - 1)],
        );
        Point::new(
            a.x * (1.0 - segment_t) + b.x * segment_t,
            a.y * (1.0 - segment_t) + b.y * segment_t,
        )
    }
}
