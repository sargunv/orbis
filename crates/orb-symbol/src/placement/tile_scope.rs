//! Tile scope's second step (see [`Scope::Tiles`](super::Scope::Tiles)):
//! after each tile has placed its own labels, a label that overlaps one of
//! another tile's that ranks above it is dropped. Whether a label is dropped
//! depends only on the tiles' own placements, never on which labels the
//! step drops, so it reads no farther than the label's neighbours.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::grid::{
    boxes_collide, circle_and_rect_collide, circles_collide, in_group, overlap_allowed,
};
use super::place::{Record, RecordedLabel, RecordedShape, ScopeTile, ShapeGeometry};

/// Cells of the lookup, in pixels.
const CELL: f64 = 64.0;

/// A placed shape: its tile, label and shape indices.
type Placed = (usize, usize, usize);

/// The labels to drop, with whether their text and icon still show (as
/// `*-overlap: always` keeps them).
pub(crate) fn outranked(tiles: &[ScopeTile], records: &[Record]) -> Vec<(u32, bool, bool)> {
    // Every blocking shape by the cells its bounds cover, in a frame shared
    // by the tiles. Only the lookup uses it: collisions are tested between
    // two tiles' own views, offset by whole tiles, so they come out the same
    // whichever tile is drawn.
    let mut cells: HashMap<(i64, i64), Vec<Placed>> = HashMap::new();
    for (t, record) in records.iter().enumerate() {
        for (l, label) in record.labels.iter().enumerate() {
            for s in label.shapes.clone() {
                let shape = &record.shapes[s];
                if !shape.blocks {
                    continue;
                }
                for cell in covered(&tiles[t], shape) {
                    cells.entry(cell).or_default().push((t, l, s));
                }
            }
        }
    }
    let mut dropped = Vec::new();
    for (t, record) in records.iter().enumerate() {
        for label in &record.labels {
            let hit = label.shapes.clone().any(|s| {
                let shape = &record.shapes[s];
                covered(&tiles[t], shape).any(|cell| {
                    cells.get(&cell).is_some_and(|entries| {
                        entries.iter().any(|&(u, m, e)| {
                            u != t
                                && outranks(&tiles[u], &records[u].labels[m], &tiles[t], label)
                                && collides(
                                    shape,
                                    &records[u].shapes[e],
                                    offset(&tiles[t], &tiles[u]),
                                )
                        })
                    })
                })
            });
            if hit {
                dropped.push((label.id, label.always_text, label.always_icon));
            }
        }
    }
    dropped
}

/// Where tile `b`'s view lies in tile `a`'s, pixels.
fn offset(a: &ScopeTile, b: &ScopeTile) -> [f64; 2] {
    [
        (b.col - a.col) as f64 * a.camera.width,
        (b.row - a.row) as f64 * a.camera.height,
    ]
}

/// The lookup cells a shape covers, with a cell of margin.
fn covered(tile: &ScopeTile, shape: &RecordedShape) -> impl Iterator<Item = (i64, i64)> + use<> {
    let [x1, y1, x2, y2] = match shape.geometry {
        ShapeGeometry::Box(b) => b,
        ShapeGeometry::Circle([x, y, r]) => [x - r, y - r, x + r, y + r],
    };
    let (ox, oy) = (
        tile.col as f64 * tile.camera.width,
        tile.row as f64 * tile.camera.height,
    );
    let cell = |v: f64, o: f64| ((v + o) / CELL).floor() as i64;
    let (cx1, cx2) = (cell(x1, ox) - 1, cell(x2, ox) + 1);
    let (cy1, cy2) = (cell(y1, oy) - 1, cell(y2, oy) + 1);
    (cx1..=cx2).flat_map(move |x| (cy1..=cy2).map(move |y| (x, y)))
}

/// Whether `a` of tile `ta` ranks above `b` of tile `tb`: by layer, sort
/// key, tile (row-major) and the order its tile placed it in.
fn outranks(ta: &ScopeTile, a: &RecordedLabel, tb: &ScopeTile, b: &RecordedLabel) -> bool {
    let order = a
        .layer
        .cmp(&b.layer)
        .then(a.sort_key.total_cmp(&b.sort_key))
        .then((ta.row, ta.col).cmp(&(tb.row, tb.col)))
        .then(a.order.cmp(&b.order));
    order == Ordering::Less
}

/// Whether `query` would have been blocked by `placed`, which lies
/// `offset` pixels away in the query's view.
fn collides(query: &RecordedShape, placed: &RecordedShape, offset: [f64; 2]) -> bool {
    if !in_group(query.group, placed.key) || overlap_allowed(query.overlap, placed.key.overlap) {
        return false;
    }
    let [dx, dy] = offset;
    match (query.geometry, placed.geometry) {
        (ShapeGeometry::Box(a), ShapeGeometry::Box(b)) => {
            boxes_collide(a, &[b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy])
        }
        (ShapeGeometry::Box(a), ShapeGeometry::Circle([x, y, r])) => {
            circle_and_rect_collide(x + dx, y + dy, r, a[0], a[1], a[2], a[3])
        }
        (ShapeGeometry::Circle([x, y, r]), ShapeGeometry::Box(b)) => {
            circle_and_rect_collide(x, y, r, b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        }
        (ShapeGeometry::Circle([x1, y1, r1]), ShapeGeometry::Circle([x2, y2, r2])) => {
            circles_collide(x1, y1, r1, x2 + dx, y2 + dy, r2)
        }
    }
}
