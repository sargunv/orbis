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
//! A polygon whole in the source tile stays whole in a child its bounds
//! meet, unclipped, while it reaches no further than one child past it and
//! past the child holding its centroid (`slice_feature`): a building
//! crossing the child's edge is then drawn once, entire, by the child
//! owning its centroid, not clipped by every child per fragment, which
//! turns off the GPU's hidden-surface removal. Each polygon of a
//! multipolygon decides alone.
//!
//! GL JS slices only so far below a source's last tile, then lays deeper
//! tiles out from that slice, overscaled. A slice deeper than where its
//! labels lay out carries that ancestor's slice for them (`Labels`), so
//! line labels run on across slice edges as GL JS's do. Where that
//! ancestor is the data tile itself, as it always is for GeoJSON (GL JS
//! and MapLibre Native overscale a GeoJSON source's last tile whole), its
//! slices share the data tile's layer, and their lines lay out along it
//! too ([`Layer::overscaled_lines`]): dash, pattern and gradient run on
//! across slice edges as along the data tile's lines.
//!
//! A slice keeps only the features it meets; [`Layer::data_feature`]
//! gives each one's index in the data tile.

use std::borrow::Cow;
use std::sync::Arc;

use crate::{GeometryType, Layer, LayerBuilder, TileFrame};

type Point = [f64; 2];

/// Where an overzoom slice lays out its labels: an ancestor's slice of the
/// source tile, the tile GL JS lays the slice out with (at `overscaling`
/// its scale), so line labels run on along the ancestor's lines across the
/// slice's edges rather than starting again at each.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Labels {
    layer: Overscaled,
    ancestor: Ancestor,
}

/// The tile an overzoom slice's labels lay out in.
#[derive(Clone, Debug, PartialEq)]
enum Overscaled {
    /// The data tile's layer, at this index of its frame, which its slices
    /// share.
    Data(Arc<TileFrame>, usize),
    /// An ancestor's slice of it, in its own units, cut at its buffer.
    Slice(Box<Layer>),
}

impl Labels {
    fn layer(&self) -> &Layer {
        match &self.layer {
            Overscaled::Data(frame, at) => &frame.layers[*at],
            Overscaled::Slice(layer) => layer,
        }
    }

    /// Heap bytes it holds of its own: none of a shared data tile.
    pub(crate) fn bytes(&self) -> usize {
        match &self.layer {
            Overscaled::Data(..) => 0,
            Overscaled::Slice(layer) => layer.bytes(),
        }
    }
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
    /// the ancestor's slice ([`Layer::labelled`]), or this tile's layer
    /// itself when `labels_depth` is 0.
    pub fn overzoom(self: &Arc<Self>, depth: u8, col: u32, row: u32, labels_depth: u8) -> Self {
        assert!(depth <= 30 && col < 1u32 << depth && row < 1u32 << depth);
        assert!(labels_depth <= depth);
        Self {
            layers: self
                .layers
                .iter()
                .enumerate()
                .map(|(at, l)| {
                    let mut child = slice(l, depth, col, row);
                    let shapes = l.types.iter().any(|&t| t != GeometryType::Point);
                    if labels_depth < depth && shapes {
                        let labels = labels(self, at, depth, [col, row], labels_depth);
                        child.labels = Some(Box::new(labels));
                    }
                    child
                })
                .filter(|l| {
                    !l.is_empty() || l.labels.as_ref().is_some_and(|l| !l.layer().is_empty())
                })
                .collect(),
        }
    }
}

impl Layer {
    /// What labels lay out from, with the ancestor it is in when that isn't
    /// this tile: an overzoom slice past its labels' ancestor gives that
    /// ancestor's slice; another gives itself without the pieces of lines
    /// it keeps past its buffer for their strokes, GL JS's slice.
    pub fn labelled(&self) -> (Cow<'_, Layer>, Option<Ancestor>) {
        match self.ancestor_labels() {
            Some((layer, ancestor)) => (Cow::Borrowed(layer), Some(ancestor)),
            None => (self.within_buffer(), None),
        }
    }

    /// The data tile's layer an overzoom slice's lines lay out along, with
    /// the slice's place in it, when GL JS overscales the data tile whole
    /// (its labels' ancestor is the data tile): its features are this
    /// layer's [`data_feature`](Self::data_feature)s. Line distance runs
    /// on along each whole line there, across the slice's edges.
    pub fn overscaled_lines(&self) -> Option<(&Layer, Ancestor)> {
        let labels = self.labels.as_deref()?;
        matches!(labels.layer, Overscaled::Data(..)).then(|| (labels.layer(), labels.ancestor))
    }

    /// Polygon ring `ring` of this layer as GL JS's slice clips it. A
    /// slice keeps a whole polygon unclipped for its fill
    /// ([`Layer::is_whole`]), but an outline starts where GL JS's clipped
    /// ring does, and that sets its dash, pattern and gradient phase.
    pub fn outline<'a>(&self, ring: &'a [[i32; 2]]) -> Cow<'a, [[i32; 2]]> {
        let Some(b) = self.clip else {
            return Cow::Borrowed(ring);
        };
        let (lo, hi) = (-b, self.extent() as i32 + b);
        if ring.iter().flatten().all(|v| (lo..=hi).contains(v)) {
            return Cow::Borrowed(ring);
        }
        let mut points: Vec<Point> = ring.iter().map(|p| p.map(f64::from)).collect();
        if ring.first() != ring.last() {
            points.push(points[0]);
        }
        let mut geometry = vec![points];
        for axis in 0..2 {
            geometry = clip(geometry, GeometryType::Polygon, lo.into(), hi.into(), axis);
        }
        let mut clipped: Vec<[i32; 2]> = geometry
            .into_iter()
            .next()
            .unwrap_or_default()
            .iter()
            .map(|p| p.map(|v| (v + 0.5).floor() as i32))
            .collect();
        if clipped.len() > 1 && clipped.first() == clipped.last() {
            clipped.pop();
        }
        Cow::Owned(clipped)
    }

    /// The ancestor's slice labels lay out from, if they lay out in one.
    pub fn ancestor_labels(&self) -> Option<(&Layer, Ancestor)> {
        self.labels.as_deref().map(|l| (l.layer(), l.ancestor))
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
        builder.clipped_like(self);
        for f in 0..self.len() {
            copy_feature(&mut builder, self, f, |r| !lines(f) || inside(r));
        }
        let mut layer = builder.finish();
        layer.data_features.clone_from(&self.data_features);
        Cow::Owned(layer)
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

/// The labels of `frame`'s layer `at` sliced at `depth` and `cell`: its
/// ancestor's slice `labels_depth` below the source tile, which slices the
/// source as GL JS does (lines end at its buffer), or at the source tile
/// itself the source layer, as GL JS overscales a source's last tile whole.
fn labels(
    frame: &Arc<TileFrame>,
    at: usize,
    depth: u8,
    cell: [u32; 2],
    labels_depth: u8,
) -> Labels {
    let up = depth - labels_depth;
    let layer = if labels_depth == 0 {
        Overscaled::Data(Arc::clone(frame), at)
    } else {
        let sliced = slice(
            &frame.layers[at],
            labels_depth,
            cell[0] >> up,
            cell[1] >> up,
        );
        Overscaled::Slice(Box::new(sliced.within_buffer().into_owned()))
    };
    let mask = (1u32 << up) - 1;
    Labels {
        layer,
        ancestor: Ancestor {
            scale: 1 << up,
            cell: cell.map(|c| c & mask),
        },
    }
}

/// The slice of `layer` at `depth`, with the features it meets.
fn slice(layer: &Layer, depth: u8, col: u32, row: u32) -> Layer {
    let scale = 2f64.powi(i32::from(depth)) * 8192.0 / f64::from(layer.extent());
    let offset = [f64::from(col) * 8192.0, f64::from(row) * 8192.0];
    let buffer = (128.0 * f64::from(layer.extent()) / 4096.0)
        .round()
        .max(128.0)
        * 8192.0
        / f64::from(layer.extent());
    let mut builder = LayerBuilder::new(layer.name(), 8192);
    builder.clipped_at(buffer.round() as i32).whole_within(8192);
    let source_buffer = layer.clip_buffer();
    let place = |p: &[i32; 2]| {
        [
            f64::from(p[0]) * scale - offset[0],
            f64::from(p[1]) * scale - offset[1],
        ]
    };
    // The data tile's square in the slice's units, and how far past it
    // the source clipped.
    let data = Data {
        min: [-offset[0], -offset[1]],
        size: 2f64.powi(i32::from(depth)) * 8192.0,
        buffer: f64::from(source_buffer) * scale,
    };
    let mut kept = Vec::new();
    for feature in 0..layer.len() {
        let kind = layer.geometry_type(feature);
        let parts = slice_feature(layer, feature, (source_buffer, &data), buffer, place);
        if parts.is_empty() {
            continue;
        }
        kept.push(layer.data_feature(feature));
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
                for point in ring {
                    builder.push_vertex(point);
                }
            }
        }
    }
    let mut sliced = builder.finish();
    sliced.data_features = Some(kept);
    sliced
}

/// A slice's data tile, in the slice's units: its square and its source's
/// clip buffer.
struct Data {
    min: [f64; 2],
    size: f64,
    buffer: f64,
}

/// A feature's parts in a slice `buffer` past the square, placed by
/// `place` and rounded to the slice's grid; none when nothing of it is
/// left.
///
/// A polygon whole in its data tile (no edge on the source's clip line,
/// `source_buffer`) stays unclipped while every slice at this depth that
/// holds it would keep it so too, as [`Layer::is_whole`] decides for one:
/// it reaches no further than one slice past this slice and past the slice
/// holding its centroid, and that slice's data tile, when another, holds
/// it whole as well. The slice holding its centroid then draws it whole,
/// and every other leaves it to that one. Each polygon of a feature
/// decides alone.
fn slice_feature(
    layer: &Layer,
    feature: usize,
    (source_buffer, data): (i32, &Data),
    buffer: f64,
    place: impl Fn(&[i32; 2]) -> Point,
) -> Vec<Vec<Vec<[i32; 2]>>> {
    let kind = layer.geometry_type(feature);
    let extent = layer.extent() as i32;
    let round = |p: &Point| p.map(|v| (v + 0.5).floor() as i32);
    let reach = if kind == GeometryType::LineString {
        8192.0
    } else {
        buffer
    };
    let mut parts = Vec::new();
    for part in layer.geometry(feature).parts() {
        if kind == GeometryType::Polygon && !crate::part_cut(&part, source_buffer, extent) {
            let rings: Vec<Vec<[i32; 2]>> = part
                .rings()
                .map(|ring| {
                    let mut ring: Vec<[i32; 2]> = ring.iter().map(|p| round(&place(p))).collect();
                    if ring.len() > 1 && ring.first() == ring.last() {
                        ring.pop();
                    }
                    ring
                })
                .collect();
            let owner = crate::owner_keeps(rings.iter().map(Vec::as_slice), 8192, 8192);
            if let Some(owner) = owner
                && in_owner_data(&rings, owner, data)
            {
                let outside = |p: &[i32; 2]| {
                    let b = buffer.round() as i32;
                    [p[0] < -b, p[0] > 8192 + b, p[1] < -b, p[1] > 8192 + b]
                };
                // Kept where its bounds meet the buffer's square.
                let away = (rings.iter().flatten())
                    .map(outside)
                    .reduce(|a, b| std::array::from_fn(|i| a[i] && b[i]))
                    .is_some_and(|a| a.iter().any(|&o| o));
                if !away {
                    parts.push(rings);
                }
                continue;
            }
        }
        let mut rings = Vec::new();
        for ring in part.rings() {
            let mut points: Vec<Point> = ring.iter().map(&place).collect();
            if kind == GeometryType::Polygon && !points.is_empty() {
                points.push(points[0]);
            }
            let mut geometry = vec![points];
            for axis in 0..2 {
                geometry = clip(geometry, kind, -reach, 8192.0 + reach, axis);
            }
            let pieces = if kind == GeometryType::LineString {
                let (lo, hi) = (-buffer, 8192.0 + buffer);
                geometry
                    .into_iter()
                    .flat_map(|l| split(&l, lo, hi))
                    .collect()
            } else {
                geometry
            };
            rings.extend(pieces.into_iter().map(|ring| {
                let end = ring.len()
                    - usize::from(
                        kind == GeometryType::Polygon
                            && ring.len() > 1
                            && ring.first() == ring.last(),
                    );
                ring[..end].iter().map(round).collect::<Vec<_>>()
            }));
        }
        if !rings.is_empty() {
            parts.push(rings);
        }
    }
    parts
}

/// Whether the data tile of the slice holding a polygon's centroid
/// (`owner`, relative to this slice) holds `rings` whole: this slice's own
/// does, and a neighbour's does where they reach no further past it than
/// the source clipped.
fn in_owner_data(rings: &[Vec<[i32; 2]>], owner: [i64; 2], data: &Data) -> bool {
    // The owner's data tile, relative to this slice's.
    let tile =
        [0, 1].map(|a| (((owner[a] as f64 + 0.5) * 8192.0 - data.min[a]) / data.size).floor());
    if tile == [0.0; 2] {
        return true;
    }
    rings.iter().flatten().all(|p| {
        (0..2).all(|a| {
            let lo = data.min[a] + tile[a] * data.size - data.buffer;
            let hi = data.min[a] + (tile[a] + 1.0) * data.size + data.buffer;
            (lo..=hi).contains(&f64::from(p[a]))
        })
    })
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
        let child = Arc::new(TileFrame {
            layers: vec![b.finish()],
        })
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

    /// Buildings larger than a slice, at z15, z16 and z19 below a z14 data
    /// tile (#143): every slice whose square a building meets draws its
    /// part there, itself (cut, or whole and owning its centroid) or by
    /// leaving it to the slice beside it holding the centroid, which keeps
    /// it whole too. Layout splits a multipolygon into its polygons.
    fn bounds(points: &[[i32; 2]]) -> ([i32; 2], [i32; 2]) {
        (points.iter()).fold(([i32::MAX; 2], [i32::MIN; 2]), |(lo, hi), p| {
            (
                [0, 1].map(|a| lo[a].min(p[a])),
                [0, 1].map(|a| hi[a].max(p[a])),
            )
        })
    }

    #[test]
    fn every_slice_a_building_meets_draws_its_part() {
        let rect =
            |x0: i32, y0: i32, x1: i32, y1: i32| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        // A long slab across slices at every depth.
        let slab = vec![rect(1000, 1900, 3100, 2100)];
        // A building whose vertices crowd its east end, which pulls its
        // centroid into a slice that would cut it.
        let mut crowded = rect(900, 2600, 2200, 2700);
        crowded.splice(2..2, (1..20).map(|i| [2200, 2600 + 5 * i]));
        // A block's buildings merged into one multipolygon, as
        // OpenMapTiles merges them: no slice holds every part.
        let block = vec![
            rect(1950, 3000, 2150, 3100),
            rect(3000, 3000, 3200, 3150),
            rect(400, 3000, 500, 3100),
            rect(60, 3300, 3900, 3400),
        ];
        let mut b = LayerBuilder::new("building", 4096);
        b.clipped_at(64);
        let buildings = [slab, vec![crowded], block];
        for parts in &buildings {
            b.begin_feature(None, GeometryType::Polygon);
            for ring in parts {
                b.begin_part();
                b.begin_ring();
                ring.iter().for_each(|&p| b.push_vertex(p));
            }
        }
        let source = Arc::new(TileFrame {
            layers: vec![b.finish()],
        });
        for depth in [1u8, 2, 5] {
            let n = 1u32 << depth;
            let cell = 4096 / n as i32;
            let slices: Vec<Vec<Layer>> = (0..n)
                .map(|row| {
                    (0..n)
                        .map(|col| source.overzoom(depth, col, row, 0).layers.remove(0))
                        .collect()
                })
                .collect();
            // How slice `(col, row)` draws polygon `part` of `feature`:
            // `None` without it, else whether whole and its centroid's
            // slice, relative to this one.
            let drawn = |col: i64, row: i64, feature: usize, part: &[[i32; 2]]| {
                let layer = slices.get(row as usize)?.get(col as usize)?;
                let f = (0..layer.len()).find(|&f| layer.data_feature(f) as usize == feature)?;
                // The slice's piece of `part`: within its bounds, in data
                // units.
                let (lo, hi) = bounds(part);
                let piece = layer.geometry(f).parts().find(|p| {
                    p.rings().flatten().all(|v| {
                        (0..2).all(|a| {
                            let o = [col, row][a] as i32 * cell;
                            let d = o + v[a] * cell / 8192;
                            (lo[a] - 1..=hi[a] + 1).contains(&d)
                        })
                    })
                })?;
                let mut one = LayerBuilder::new("building", 8192);
                one.clipped_like(layer);
                one.begin_feature(None, GeometryType::Polygon);
                one.begin_part();
                for ring in piece.rings() {
                    one.begin_ring();
                    ring.iter().for_each(|&p| one.push_vertex(p));
                }
                let one = one.finish();
                let whole = one.is_whole(0, one.clip_buffer());
                let owner = crate::owner_keeps(piece.rings(), 8192, i32::MAX / 4);
                Some((whole, owner.expect("a polygon")))
            };
            for (feature, parts) in buildings.iter().enumerate() {
                for part in parts {
                    let (lo, hi) = bounds(part);
                    for row in lo[1] / cell..=(hi[1] - 1) / cell {
                        for col in lo[0] / cell..=(hi[0] - 1) / cell {
                            let (col, row) = (i64::from(col), i64::from(row));
                            let at = format!(
                                "z{} slice {col},{row}, feature {feature} {part:?}",
                                14 + depth
                            );
                            let (whole, owner) = drawn(col, row, feature, part)
                                .unwrap_or_else(|| panic!("{at}: dropped"));
                            if !whole || owner == [0, 0] {
                                continue;
                            }
                            let (c, r) = (col + owner[0], row + owner[1]);
                            let there = drawn(c, r, feature, part);
                            assert_eq!(
                                there,
                                Some((true, [0, 0])),
                                "{at}: left to slice {c},{r}, which doesn't draw it whole"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn whole_polygons_outline_as_gl_js_slices_them() {
        let mut b = LayerBuilder::new("buildings", 4096);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[1900, 100], [2200, 100], [2200, 200], [1900, 200]] {
            b.push_vertex(p);
        }
        let child = Arc::new(TileFrame {
            layers: vec![b.finish()],
        })
        .overzoom(1, 0, 0, 1);
        let l = &child.layers[0];
        let ring = l
            .geometry(0)
            .parts()
            .next()
            .unwrap()
            .rings()
            .next()
            .unwrap();
        // The fill keeps the building whole across the slice's edge.
        assert_eq!(ring, [[7600, 400], [8800, 400], [8800, 800], [7600, 800]]);
        // Its outline is GL JS's ring, cut at the slice's buffer.
        assert_eq!(
            *l.outline(ring),
            [[7600, 400], [8448, 400], [8448, 800], [7600, 800]]
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
        let source = Arc::new(TileFrame {
            layers: vec![b.finish()],
        });
        // The slice 4 zooms down at (5, 1), its labels in the ancestor 2
        // zooms down at (1, 0).
        let child = source.overzoom(4, 5, 1, 2);
        let l = &child.layers[0];
        let name = |l: &Layer, f| match l.property("name").unwrap().get(f) {
            Some(ValueRef::Str(s)) => s.to_owned(),
            other => panic!("{other:?}"),
        };
        // The slice keeps what it meets, with each one's index in the
        // data tile.
        assert_eq!(l.len(), 1);
        assert_eq!(name(l, 0), "through");
        assert_eq!(l.data_feature(0), 0);
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
        assert_eq!(labels.data_feature(1), 1);
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
        assert!(l.overscaled_lines().is_none());
        // At the labels' own zoom, a slice is its own.
        assert!(source.overzoom(2, 1, 0, 2).layers[0].labelled().1.is_none());
    }

    #[test]
    fn slices_of_a_wholly_overscaled_tile_share_it() {
        let mut b = LayerBuilder::new("roads", 4096);
        for line in [[[0, 300], [4096, 300]], [[0, 3000], [4096, 3000]]] {
            b.begin_feature(None, GeometryType::LineString);
            b.begin_part();
            b.begin_ring();
            line.iter().for_each(|&p| b.push_vertex(p));
        }
        let source = Arc::new(TileFrame {
            layers: vec![b.finish()],
        });
        // Its labels and lines lay out in the data tile itself.
        let child = source.overzoom(2, 1, 3, 0);
        let l = &child.layers[0];
        assert_eq!(l.len(), 1);
        assert_eq!(l.data_feature(0), 1);
        let (data, ancestor) = l.overscaled_lines().unwrap();
        assert!(std::ptr::eq(data, &source.layers[0]));
        assert_eq!(
            ancestor,
            Ancestor {
                scale: 4,
                cell: [1, 3]
            }
        );
        assert_eq!(l.labelled().0.len(), 2);
        // Only what the slice holds counts towards it.
        assert!(child.bytes() < source.bytes());
    }
}
