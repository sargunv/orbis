//! Vector overzoom, following pinned GL JS's `sliceVectorTileLayer` and
//! `clipGeometry`, before layout.
//!
//! Polygons and points keep GL JS's buffer past the child's square. Lines
//! keep a whole tile, the margin line layout keeps (`orb_layout::line`): a
//! wide stroke crossing the child's edge needs its line to go on well past
//! it, or the end of the stroke shows inside the tile. (GL JS's buffer is
//! 28 px at z19.8; a 140 px road crossing a tile edge at 57° lost a corner
//! on each side of it.) Such a line splits where it crosses the buffer's
//! square: the pieces inside are GL JS's slice of it, so dash phase and
//! line-label anchors start where GL JS's do, and the pieces outside only
//! carry the stroke on across the edge.
//!
//! A polygon whole in the source tile (`Layer::is_whole`) stays whole in a
//! child its bounds meet, unclipped, while it reaches no further than one
//! child past it: a building crossing the child's edge is then drawn once,
//! entire, by the child owning its centroid, not clipped by every child
//! per fragment, which turns off the GPU's hidden-surface removal.
//!
//! GL JS slices only so far below a source's last tile, then lays deeper
//! tiles out from that slice, overscaled. A slice deeper than where its
//! labels lay out carries that ancestor's slice for them (`Labels`), so
//! line labels run on across slice edges as GL JS's do.

use std::borrow::Cow;

use crate::{GeometryType, Layer, LayerBuilder, TileFrame};

type Point = [f64; 2];

/// Where an overzoom slice lays out its labels: an ancestor's slice of the
/// source tile, the tile GL JS lays the slice out with (at `overscaling`
/// its scale), so line labels run on along the ancestor's lines across the
/// slice's edges rather than starting again at each.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Labels {
    /// The ancestor's slice, in its own units: the slice's features, in
    /// order, with their geometry there (none for those it lacks).
    pub(crate) layer: Layer,
    pub(crate) ancestor: Ancestor,
}

/// An overzoom slice's place in the ancestor its labels lay out in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ancestor {
    /// The ancestor's size in slices, a power of two.
    pub scale: u32,
    /// The slice's column and row among the ancestor's `scale × scale`.
    pub cell: [u32; 2],
}

impl TileFrame {
    /// Slices a descendant at relative `(col, row)` and `depth` into normalized
    /// extent 8192. Intersections stay in f64 until the layout grid is rounded.
    ///
    /// Labels lay out from the ancestor `labels_depth` (at most `depth`)
    /// below this tile: past it, each layer with lines or polygons carries
    /// the ancestor's slice ([`Layer::labelled`]) and keeps every feature
    /// that has geometry in either, so the two share feature indices.
    pub fn overzoom(&self, depth: u8, col: u32, row: u32, labels_depth: u8) -> Self {
        assert!(depth <= 30 && col < 1u32 << depth && row < 1u32 << depth);
        assert!(labels_depth <= depth);
        Self {
            layers: self
                .layers
                .iter()
                .map(|l| {
                    let shapes = l.types.iter().any(|&t| t != GeometryType::Point);
                    if labels_depth < depth && shapes {
                        labelled_slice(l, depth, col, row, labels_depth)
                    } else {
                        slice(l, depth, col, row, None).0
                    }
                })
                .filter(|l| !l.is_empty())
                .collect(),
        }
    }
}

impl Layer {
    /// What labels lay out from, with the ancestor it is in when that isn't
    /// this tile: an overzoom slice past its labels' ancestor gives that
    /// ancestor's slice; another gives itself without the pieces of lines
    /// it keeps past its buffer for their strokes, GL JS's slice. Feature
    /// indices stay.
    pub fn labelled(&self) -> (Cow<'_, Layer>, Option<Ancestor>) {
        match self.ancestor_labels() {
            Some((layer, ancestor)) => (Cow::Borrowed(layer), Some(ancestor)),
            None => (self.within_buffer(), None),
        }
    }

    /// The ancestor's slice labels lay out from, if they lay out in one.
    pub fn ancestor_labels(&self) -> Option<(&Layer, Ancestor)> {
        self.labels.as_deref().map(|l| (&l.layer, l.ancestor))
    }

    /// The layer without the pieces of lines an overzoom slice keeps past
    /// its buffer for their strokes.
    fn within_buffer(&self) -> Cow<'_, Layer> {
        let (b, e) = (self.clip_buffer(), self.extent() as i32);
        let inside = |r: &[[i32; 2]]| {
            r.len() < 2
                || [0, 1]
                    .iter()
                    .all(|&i| (-b..=e + b).contains(&((r[0][i] + r[1][i]) / 2)))
        };
        let lines = |f| self.geometry_type(f) == GeometryType::LineString;
        let outer = (0..self.len()).filter(|&f| lines(f)).any(|f| {
            self.geometry(f)
                .parts()
                .any(|p| p.rings().any(|r| !inside(r)))
        });
        if self.clip.is_none() || !outer {
            return Cow::Borrowed(self);
        }
        let mut builder = LayerBuilder::new(self.name(), self.extent());
        builder.clipped_at(b);
        for f in 0..self.len() {
            copy_feature(&mut builder, self, f, |r| !lines(f) || inside(r));
        }
        Cow::Owned(builder.finish())
    }
}

/// Adds feature `f` of `from` to `builder`, with the rings `keep` keeps.
fn copy_feature(
    builder: &mut LayerBuilder,
    from: &Layer,
    f: usize,
    keep: impl Fn(&[[i32; 2]]) -> bool,
) {
    builder.begin_feature(None, from.geometry_type(f));
    if let Some(id) = from.ids().get(f) {
        builder.set_id(id.to_owned());
    }
    for (key, column) in from.properties() {
        if let Some(value) = column.get(f) {
            builder.set_property(key, value.to_owned());
        }
    }
    for part in from.geometry(f).parts() {
        let mut rings = part.rings().filter(|r| keep(r)).peekable();
        if rings.peek().is_some() {
            builder.begin_part();
        }
        for ring in rings {
            builder.begin_ring();
            ring.iter().for_each(|&p| builder.push_vertex(p));
        }
    }
}

/// [`slice`] at `depth`, carrying its labels' ancestor's slice
/// `labels_depth` below the source tile. The ancestor slices the source as
/// GL JS does (lines end at its buffer); at the source tile itself it is
/// the source layer, as GL JS overscales a source's last tile whole.
fn labelled_slice(layer: &Layer, depth: u8, col: u32, row: u32, labels_depth: u8) -> Layer {
    let up = depth - labels_depth;
    let (ancestor, ancestor_kept): (Cow<'_, Layer>, Vec<u32>) = if labels_depth == 0 {
        (Cow::Borrowed(layer), (0..layer.len() as u32).collect())
    } else {
        let (sliced, kept) = slice(layer, labels_depth, col >> up, row >> up, None);
        (Cow::Owned(sliced.within_buffer().into_owned()), kept)
    };
    let (mut child, kept) = slice(layer, depth, col, row, Some(&ancestor_kept));
    let mut builder = LayerBuilder::new(layer.name(), ancestor.extent());
    builder.clipped_at(ancestor.clip_buffer());
    let mut at = 0;
    for &f in &kept {
        while ancestor_kept.get(at).is_some_and(|&a| a < f) {
            at += 1;
        }
        if ancestor_kept.get(at) == Some(&f) {
            copy_feature(&mut builder, &ancestor, at, |_| true);
        } else {
            copy_feature(&mut builder, layer, f as usize, |_| false);
        }
    }
    let mask = (1u32 << up) - 1;
    child.labels = Some(Box::new(Labels {
        layer: builder.finish(),
        ancestor: Ancestor {
            scale: 1 << up,
            cell: [col & mask, row & mask],
        },
    }));
    child
}

/// The slice of `layer` at `depth`, and the indices of the source features
/// it has, in order. Features in `keep` (sorted) stay even where nothing of
/// them is left, with no geometry.
fn slice(layer: &Layer, depth: u8, col: u32, row: u32, keep: Option<&[u32]>) -> (Layer, Vec<u32>) {
    let scale = 2f64.powi(i32::from(depth)) * 8192.0 / f64::from(layer.extent());
    let offset = [f64::from(col) * 8192.0, f64::from(row) * 8192.0];
    let buffer = (128.0 * f64::from(layer.extent()) / 4096.0)
        .round()
        .max(128.0)
        * 8192.0
        / f64::from(layer.extent());
    let mut builder = LayerBuilder::new(layer.name(), 8192);
    builder.clipped_at(buffer.round() as i32);
    let source_buffer = layer.clip_buffer();
    let place = |p: &[i32; 2]| {
        [
            f64::from(p[0]) * scale - offset[0],
            f64::from(p[1]) * scale - offset[1],
        ]
    };
    let mut kept = Vec::new();
    let mut keep = keep.unwrap_or_default().iter().copied().peekable();
    for feature in 0..layer.len() {
        let kind = layer.geometry_type(feature);
        let parts = slice_feature(layer, feature, source_buffer, buffer, place);
        let keep = keep.next_if_eq(&(feature as u32)).is_some();
        if parts.is_empty() && !keep {
            continue;
        }
        kept.push(feature as u32);
        builder.begin_feature(None, kind);
        if let Some(id) = layer.ids().get(feature) {
            builder.set_id(id.to_owned());
        }
        for (key, column) in layer.properties() {
            if let Some(value) = column.get(feature) {
                builder.set_property(key, value.to_owned());
            }
        }
        for part in parts {
            builder.begin_part();
            for ring in part {
                builder.begin_ring();
                let end = ring.len()
                    - usize::from(
                        kind == GeometryType::Polygon
                            && ring.len() > 1
                            && ring.first() == ring.last(),
                    );
                for point in &ring[..end] {
                    builder.push_vertex(point.map(|v| (v + 0.5).floor() as i32));
                }
            }
        }
    }
    (builder.finish(), kept)
}

/// A feature's parts in a slice `buffer` past the square, placed by
/// `place`; none when nothing of it is left.
fn slice_feature(
    layer: &Layer,
    feature: usize,
    source_buffer: i32,
    buffer: f64,
    place: impl Fn(&[i32; 2]) -> Point,
) -> Vec<Vec<Vec<Point>>> {
    let kind = layer.geometry_type(feature);
    let whole = (kind == GeometryType::Polygon && layer.is_whole(feature, source_buffer))
        .then(|| {
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for part in layer.geometry(feature).parts() {
                for p in part.rings().flatten() {
                    let p = place(p);
                    lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                    hi = [hi[0].max(p[0]), hi[1].max(p[1])];
                }
            }
            (lo, hi)
        })
        .filter(|&(lo, hi)| {
            lo.iter()
                .chain(&hi)
                .all(|v| (-8192.0..=16384.0).contains(v))
        });
    if let Some((lo, hi)) = whole
        && (lo.iter().any(|&v| v > 8192.0 + buffer) || hi.iter().any(|&v| v < -buffer))
    {
        return Vec::new();
    }
    let reach = if kind == GeometryType::LineString {
        8192.0
    } else {
        buffer
    };
    let mut parts = Vec::new();
    for part in layer.geometry(feature).parts() {
        let mut rings = Vec::new();
        for ring in part.rings() {
            let mut points: Vec<Point> = ring.iter().map(&place).collect();
            if kind == GeometryType::Polygon && !points.is_empty() {
                points.push(points[0]);
            }
            let mut geometry = vec![points];
            for axis in (0..2).filter(|_| whole.is_none()) {
                geometry = clip(geometry, kind, -reach, 8192.0 + reach, axis);
            }
            if kind == GeometryType::LineString {
                let (lo, hi) = (-buffer, 8192.0 + buffer);
                rings.extend(geometry.into_iter().flat_map(|l| split(&l, lo, hi)));
            } else {
                rings.extend(geometry);
            }
        }
        if !rings.is_empty() {
            parts.push(rings);
        }
    }
    parts
}

/// Splits a line where it crosses the square `lo..=hi`, at points shared
/// by the pieces on each side, so their strokes meet seamlessly.
fn split(line: &[Point], lo: f64, hi: f64) -> Vec<Vec<Point>> {
    let mut pieces = Vec::new();
    let mut piece: Vec<Point> = line.first().into_iter().copied().collect();
    for pair in line.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        // The part of the segment inside the square (Liang–Barsky).
        let (mut t0, mut t1) = (0f64, 1f64);
        for axis in 0..2 {
            let d = b[axis] - a[axis];
            for (p, q) in [(-d, a[axis] - lo), (d, hi - a[axis])] {
                if p == 0.0 && q < 0.0 {
                    (t0, t1) = (1.0, 0.0);
                } else if p < 0.0 {
                    t0 = t0.max(q / p);
                } else if p > 0.0 {
                    t1 = t1.min(q / p);
                }
            }
        }
        for t in [t0, t1]
            .into_iter()
            .filter(|&t| t0 < t1 && t > 0.0 && t < 1.0)
        {
            let at = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            piece.push(at);
            pieces.push(std::mem::replace(&mut piece, vec![at]));
        }
        piece.push(b);
    }
    pieces.push(piece);
    pieces.retain(|p| p.len() > 1);
    pieces
}

fn clip(
    geometry: Vec<Vec<Point>>,
    kind: GeometryType,
    min: f64,
    max: f64,
    axis: usize,
) -> Vec<Vec<Point>> {
    if kind == GeometryType::Point {
        return geometry
            .into_iter()
            .flatten()
            .filter(|p| p[axis] >= min && p[axis] <= max)
            .map(|p| vec![p])
            .collect();
    }
    let polygon = kind == GeometryType::Polygon;
    let mut out = Vec::new();
    for line in geometry {
        let mut slice = Vec::new();
        let intersection = |a: Point, b: Point, edge: f64| {
            let t = (edge - a[axis]) / (b[axis] - a[axis]);
            let mut p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            p[axis] = edge;
            p
        };
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut exited = false;
            if a[axis] < min {
                if b[axis] >= min {
                    slice.push(intersection(a, b, min));
                }
            } else if a[axis] > max {
                if b[axis] <= max {
                    slice.push(intersection(a, b, max));
                }
            } else {
                slice.push(a);
            }
            if b[axis] < min && a[axis] >= min {
                slice.push(intersection(a, b, min));
                exited = true;
            }
            if b[axis] > max && a[axis] <= max {
                slice.push(intersection(a, b, max));
                exited = true;
            }
            if !polygon && exited && !slice.is_empty() {
                out.push(std::mem::take(&mut slice));
            }
        }
        if let Some(&last) = line.last()
            && last[axis] >= min
            && last[axis] <= max
        {
            slice.push(last);
        }
        if polygon && !slice.is_empty() && slice.first() != slice.last() {
            slice.push(slice[0]);
        }
        if !slice.is_empty() {
            out.push(slice);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Value, ValueRef};
    #[test]
    fn child_clips_before_rounding_and_retains_columns() {
        let mut b = LayerBuilder::new("roads", 4096);
        b.begin_feature(Some(42), GeometryType::LineString);
        b.set_property("class", Value::Str("primary".into()));
        b.begin_part();
        b.begin_ring();
        b.push_vertex([-512, 100]);
        b.push_vertex([6144, 103]);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[0, 0], [4096, 0], [4096, 4096], [0, 4096]] {
            b.push_vertex(p);
        }
        let child = TileFrame {
            layers: vec![b.finish()],
        }
        .overzoom(2, 1, 0, 2);
        let l = &child.layers[0];
        assert_eq!(l.extent(), 8192);
        // The line goes on a tile past each side of the square, clipped
        // between vertices, in three pieces split where it crosses the
        // buffer's square; the polygon stops at the source buffer.
        assert_eq!(
            &l.vertices[..6],
            [
                [-8192, 802],
                [-256, 805],
                [-256, 805],
                [8448, 809],
                [8448, 809],
                [16384, 813]
            ]
        );
        assert_eq!(l.geometry(0).parts().next().unwrap().rings().count(), 3);
        assert_eq!(l.clip_buffer(), 256);
        assert!(
            l.vertices[6..]
                .iter()
                .flatten()
                .all(|&v| (-256..=8448).contains(&v))
        );
        assert_eq!(l.ids().get(0), Some(ValueRef::U64(42)));
        assert_eq!(
            l.property("class").unwrap().get(0),
            Some(ValueRef::Str("primary"))
        );
    }

    #[test]
    fn slices_past_their_labels_ancestor_carry_its_slice() {
        let mut b = LayerBuilder::new("roads", 4096);
        for (name, line) in [
            // Across the slice and its ancestor.
            ("through", [[0, 300], [4096, 300]]),
            // In the ancestor, two slices away from the slice.
            ("beside", [[0, 900], [4096, 900]]),
            // Outside the ancestor.
            ("away", [[0, 3000], [4096, 3000]]),
        ] {
            b.begin_feature(None, GeometryType::LineString);
            b.set_property("name", Value::Str(name.into()));
            b.begin_part();
            b.begin_ring();
            line.iter().for_each(|&p| b.push_vertex(p));
        }
        let source = TileFrame {
            layers: vec![b.finish()],
        };
        // The slice 4 zooms down at (5, 1), its labels in the ancestor 2
        // zooms down at (1, 0).
        let child = source.overzoom(4, 5, 1, 2);
        let l = &child.layers[0];
        let name = |l: &Layer, f| match l.property("name").unwrap().get(f) {
            Some(ValueRef::Str(s)) => s.to_owned(),
            other => panic!("{other:?}"),
        };
        // The slice keeps what its ancestor has, with no geometry where
        // nothing of it is left, so the two share feature indices.
        assert_eq!(l.len(), 2);
        assert_eq!(
            (name(l, 0), name(l, 1)),
            ("through".into(), "beside".into())
        );
        assert_eq!(l.geometry(1).parts().count(), 0);
        let (labels, ancestor) = l.labelled();
        assert_eq!(
            ancestor,
            Some(Ancestor {
                scale: 4,
                cell: [1, 1]
            })
        );
        assert_eq!(labels.len(), 2);
        assert_eq!(name(&labels, 1), "beside");
        // In the ancestor's units, cut at its buffer.
        let ring = labels
            .geometry(1)
            .parts()
            .next()
            .unwrap()
            .rings()
            .next()
            .unwrap();
        assert_eq!(ring, [[-256, 7200], [8448, 7200]]);
        // At the labels' own zoom, a slice is its own.
        assert!(source.overzoom(2, 1, 0, 2).layers[0].labelled().1.is_none());
    }
}
