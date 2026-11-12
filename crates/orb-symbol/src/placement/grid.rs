//! GL JS `grid_index.ts`: placed boxes and circles in viewport cells, for
//! collision queries.

/// `text-overlap`, `icon-overlap` (GL JS `OverlapMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Overlap {
    #[default]
    Never,
    Always,
    Cooperative,
}

/// What a placed shape is, as collision queries see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GridKey {
    pub(crate) overlap: Overlap,
    /// The collision group: a source, unless sources collide with each
    /// other.
    pub(crate) group: u32,
}

/// GL JS `overlapAllowed`.
pub(crate) fn overlap_allowed(a: Overlap, b: Overlap) -> bool {
    a == Overlap::Always || !(a == Overlap::Never || b == Overlap::Never)
}

/// A query's filter on the keys it may hit (GL JS's collision group
/// predicate): `None` hits every group.
pub(crate) type Group = Option<u32>;

pub(crate) fn in_group(group: Group, key: GridKey) -> bool {
    group.is_none_or(|g| key.group == g)
}

#[derive(Debug)]
pub(crate) struct GridIndex {
    box_cells: Vec<Vec<u32>>,
    circle_cells: Vec<Vec<u32>>,
    box_keys: Vec<GridKey>,
    circle_keys: Vec<GridKey>,
    bboxes: Vec<f64>,
    circles: Vec<f64>,
    x_cells: usize,
    y_cells: usize,
    width: f64,
    height: f64,
    x_scale: f64,
    y_scale: f64,
    /// Per box and circle uid, the query that last saw it.
    box_seen: Vec<u32>,
    circle_seen: Vec<u32>,
    query: u32,
}

impl GridIndex {
    pub(crate) fn new(width: f64, height: f64, cell_size: f64) -> Self {
        let x_cells = (width / cell_size).ceil().max(1.0) as usize;
        let y_cells = (height / cell_size).ceil().max(1.0) as usize;
        Self {
            box_cells: vec![Vec::new(); x_cells * y_cells],
            circle_cells: vec![Vec::new(); x_cells * y_cells],
            box_keys: Vec::new(),
            circle_keys: Vec::new(),
            bboxes: Vec::new(),
            circles: Vec::new(),
            x_cells,
            y_cells,
            width,
            height,
            x_scale: x_cells as f64 / width,
            y_scale: y_cells as f64 / height,
            box_seen: Vec::new(),
            circle_seen: Vec::new(),
            query: 0,
        }
    }

    fn cells(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> impl Iterator<Item = usize> + use<> {
        let cx = |x: f64| ((x * self.x_scale).floor().max(0.0) as usize).min(self.x_cells - 1);
        let cy = |y: f64| ((y * self.y_scale).floor().max(0.0) as usize).min(self.y_cells - 1);
        let (cx1, cy1, cx2, cy2) = (cx(x1), cy(y1), cx(x2), cy(y2));
        let stride = self.x_cells;
        (cx1..=cx2).flat_map(move |x| (cy1..=cy2).map(move |y| stride * y + x))
    }

    pub(crate) fn insert(&mut self, key: GridKey, x1: f64, y1: f64, x2: f64, y2: f64) {
        let uid = self.box_keys.len() as u32;
        for cell in self.cells(x1, y1, x2, y2) {
            self.box_cells[cell].push(uid);
        }
        self.box_keys.push(key);
        self.bboxes.extend([x1, y1, x2, y2]);
        self.box_seen.push(0);
    }

    pub(crate) fn insert_circle(&mut self, key: GridKey, x: f64, y: f64, radius: f64) {
        let uid = self.circle_keys.len() as u32;
        for cell in self.cells(x - radius, y - radius, x + radius, y + radius) {
            self.circle_cells[cell].push(uid);
        }
        self.circle_keys.push(key);
        self.circles.extend([x, y, radius]);
        self.circle_seen.push(0);
    }

    fn next_query(&mut self) -> u32 {
        self.query += 1;
        self.query
    }

    /// Whether a box hits anything it may not overlap (GL JS `hitTest`).
    pub(crate) fn hit_test(
        &mut self,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        overlap: Overlap,
        group: Group,
    ) -> bool {
        // A box over the whole grid collides with everything.
        if x1 <= 0.0 && y1 <= 0.0 && self.width <= x2 && self.height <= y2 {
            return true;
        }
        self.hits(
            [x1, y1, x2, y2],
            overlap,
            group,
            |b| boxes_collide([x1, y1, x2, y2], b),
            |c| circle_and_rect_collide(c[0], c[1], c[2], x1, y1, x2, y2),
        )
    }

    /// Whether a circle hits anything it may not overlap (GL JS
    /// `hitTestCircle`).
    pub(crate) fn hit_test_circle(
        &mut self,
        x: f64,
        y: f64,
        radius: f64,
        overlap: Overlap,
        group: Group,
    ) -> bool {
        self.hits(
            [x - radius, y - radius, x + radius, y + radius],
            overlap,
            group,
            |b| circle_and_rect_collide(x, y, radius, b[0], b[1], b[2], b[3]),
            |c| circles_collide(c[0], c[1], c[2], x, y, radius),
        )
    }

    /// Whether a shape with these bounds meets, by `meets_box` and
    /// `meets_circle`, a placed box or circle it may not overlap. Each is
    /// tested once, however many of its cells the bounds cover.
    fn hits(
        &mut self,
        [x1, y1, x2, y2]: [f64; 4],
        overlap: Overlap,
        group: Group,
        meets_box: impl Fn(&[f64]) -> bool,
        meets_circle: impl Fn(&[f64]) -> bool,
    ) -> bool {
        if x2 < 0.0 || x1 > self.width || y2 < 0.0 || y1 > self.height {
            return false;
        }
        let query = self.next_query();
        let blocks =
            |key: &GridKey| in_group(group, *key) && !overlap_allowed(overlap, key.overlap);
        for cell in self.cells(x1, y1, x2, y2) {
            for &uid in &self.box_cells[cell] {
                let uid = uid as usize;
                if std::mem::replace(&mut self.box_seen[uid], query) == query {
                    continue;
                }
                if meets_box(&self.bboxes[uid * 4..][..4]) && blocks(&self.box_keys[uid]) {
                    return true;
                }
            }
            for &uid in &self.circle_cells[cell] {
                let uid = uid as usize;
                if std::mem::replace(&mut self.circle_seen[uid], query) == query {
                    continue;
                }
                if meets_circle(&self.circles[uid * 3..][..3]) && blocks(&self.circle_keys[uid]) {
                    return true;
                }
            }
        }
        false
    }
}

/// Whether two boxes meet, edges included. Boxes computed through
/// different projections should still meet.
pub(crate) fn boxes_collide(a: [f64; 4], b: &[f64]) -> bool {
    const EPSILON: f64 = 1e-6;
    a[0] <= b[2] + EPSILON
        && a[1] <= b[3] + EPSILON
        && a[2] >= b[0] - EPSILON
        && a[3] >= b[1] - EPSILON
}

pub(crate) fn circles_collide(x1: f64, y1: f64, r1: f64, x2: f64, y2: f64, r2: f64) -> bool {
    let (dx, dy) = (x2 - x1, y2 - y1);
    let both = r1 + r2;
    both * both > dx * dx + dy * dy
}

pub(crate) fn circle_and_rect_collide(
    cx: f64,
    cy: f64,
    r: f64,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
) -> bool {
    let half_w = (x2 - x1) / 2.0;
    let dist_x = (cx - (x1 + half_w)).abs();
    if dist_x > half_w + r {
        return false;
    }
    let half_h = (y2 - y1) / 2.0;
    let dist_y = (cy - (y1 + half_h)).abs();
    if dist_y > half_h + r {
        return false;
    }
    if dist_x <= half_w || dist_y <= half_h {
        return true;
    }
    let (dx, dy) = (dist_x - half_w, dist_y - half_h);
    dx * dx + dy * dy <= r * r
}
