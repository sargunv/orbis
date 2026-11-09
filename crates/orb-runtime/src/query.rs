//! Picking, independent of the camera. Tile jobs write each
//! node's hit-testable features in tile-local data units ([`MeshHits`]); a
//! tile's [`TileMapping`] places them in world space. A frame records the
//! tiles it drew and their clip regions ([`DrawnTile`]), placement commits
//! the labels they show ([`LabelHit`], anchored in world space), and the
//! style each node's pixel tolerances ([`HitLayers`]). A query unprojects
//! its point or box with the camera it is asked with, culls features by
//! their world bounds, and projects only the candidates into pixels for
//! the exact test, so a camera change costs nothing until something picks.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use glam::{DVec2, Vec2};
use orb_camera::{CameraState, View};
use orb_crs::{TileIndex, TileMatrixSet};
use orb_style::{Kind, LayerType, Property, TranslateAnchor};

use crate::state::FeatureState;

/// The topmost feature under a point in the last published frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub layer: String,
    pub source: String,
    pub source_layer: Option<String>,
    pub id: Option<String>,
    pub generation: u64,
}

/// A hit-testable feature's id, index and parts, unpacked, for tests.
#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(crate) struct HitFeature {
    pub id: Option<String>,
    pub feature: u32,
    pub parts: Vec<Vec<Vec<[i32; 2]>>>,
}

/// [`NodeHit::kind`] of a symbol node's feature: its id, by feature
/// index, for the labels placement shows of it.
pub(crate) const LABEL: u32 = 3;

/// Where a hit's feature lives, for queries to its owner worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct FeatureKey {
    /// The plan source's key.
    pub(crate) source: u32,
    pub(crate) tile: TileIndex,
    /// The feature's index in its source layer.
    pub(crate) feature: u32,
}

/// Words, read in order; malformed input is an error, never a panic.
struct Reader<'a> {
    words: &'a [u32],
    at: usize,
}

const MALFORMED: &str = "malformed hit data";

impl<'a> Reader<'a> {
    fn new(words: &'a [u32]) -> Self {
        Self { words, at: 0 }
    }
    fn next(&mut self) -> Result<u32, String> {
        let word = *self.words.get(self.at).ok_or(MALFORMED)?;
        self.at += 1;
        Ok(word)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u32], String> {
        let end = self.at.checked_add(n).ok_or(MALFORMED)?;
        let words = self.words.get(self.at..end).ok_or(MALFORMED)?;
        self.at = end;
        Ok(words)
    }
    fn text(&mut self) -> Result<String, String> {
        read_text(&mut || self.next())
    }
    fn done(&self) -> bool {
        self.at == self.words.len()
    }
}

/// A tile's hit-testable geometry of one source layer, in tile-local data
/// units as the tile job wrote it: each feature its nodes pick, once, so
/// that neither the render thread nor a UI thread decodes them to pick.
/// Polygons' rings are closed, and points only those inside the half-open
/// tile, as lines and circles pick them.
///
/// Words, per feature: its length in words, then `min_x, min_y, max_x,
/// max_y` (i32), its index in the data tile (an overzoom slice's differs),
/// `has_id, [id], parts, {rings, {points, {x, y}}}`.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HitGeometry {
    /// The layer's data units per tile side.
    extent: u32,
    /// How far past the tile square its polygons answer queries, in data
    /// units: the layer's clip buffer. An overzoom slice keeps a polygon
    /// whole past its buffer to draw it once (`orb_tile`'s overzoom), but
    /// it answers as clipped there, as GL JS's child tiles do.
    clip: i32,
    words: Vec<u32>,
    /// Records by feature index, and whether they hold its parts.
    #[serde(skip)]
    added: HashMap<(u32, bool), u32>,
}

impl HitGeometry {
    /// Feature `feature` of the tile's `layer`, with `id` (formatted as GL
    /// JS keys feature state, once), and its parts if `with_parts`: its
    /// record's first word, added unless it is here.
    /// A label's feature, without parts, may come from the layer its labels
    /// lay out from (`orb_tile::Layer::labelled`).
    pub(crate) fn add(
        &mut self,
        layer: &orb_tile::Layer,
        feature: u32,
        id: Option<orb_tile::ValueRef<'_>>,
        with_parts: bool,
    ) -> u32 {
        if let Some(&at) = self.added.get(&(feature, with_parts)) {
            return at;
        }
        let at = self.words.len() as u32;
        if with_parts && self.extent == 0 {
            self.extent = layer.extent();
            self.clip = layer.clip_buffer();
        }
        let (extent, f, words) = (self.extent as i32, feature as usize, &mut self.words);
        words.extend([
            0,
            i32::MAX as u32,
            i32::MAX as u32,
            i32::MIN as u32,
            i32::MIN as u32,
            layer.data_feature(feature as usize),
        ]);
        words.push(u32::from(id.is_some()));
        if let Some(id) = id {
            write_text(words, &crate::features::state_key(id));
        }
        let parts: Vec<_> = layer.geometry(f).parts().filter(|_| with_parts).collect();
        words.push(parts.len() as u32);
        let kind = layer.geometry_type(f);
        for part in parts {
            words.push(part.rings().count() as u32);
            for ring in part.rings() {
                let points: Vec<[i32; 2]> = match kind {
                    orb_tile::GeometryType::Point => ring
                        .iter()
                        .filter(|p| p.iter().all(|v| (0..extent).contains(v)))
                        .copied()
                        .collect(),
                    orb_tile::GeometryType::Polygon => {
                        ring.iter().chain(ring.first()).copied().collect()
                    }
                    _ => ring.to_vec(),
                };
                words.push(points.len() as u32);
                for p in points {
                    for (i, v) in p.into_iter().enumerate() {
                        let (min, max) = (at as usize + 1 + i, at as usize + 3 + i);
                        words[min] = (words[min] as i32).min(v) as u32;
                        words[max] = (words[max] as i32).max(v) as u32;
                    }
                    words.extend(p.map(|v| v as u32));
                }
            }
        }
        words[at as usize] = words.len() as u32 - at - 1;
        self.added.insert((feature, with_parts), at);
        at
    }

    /// The record at `at`, after its length word.
    fn record(&self, at: u32) -> Result<&[u32], String> {
        let at = at as usize;
        let len = *self.words.get(at).ok_or(MALFORMED)? as usize;
        self.words
            .get(at + 1..at + 1 + len)
            .ok_or_else(|| MALFORMED.into())
    }

    pub fn bytes(&self) -> u64 {
        self.words.len() as u64 * 4
    }
}

/// One render node's hit-testable features in one tile: for each, its
/// record in the tile's [`HitGeometry`] of the node's source layer and the
/// node's dimensions of it.
///
/// Words: the feature count; the largest radius, stroke and height (or
/// base) of any feature (f32); the features' extent;
/// their bounds `min_x, min_y, max_x, max_y` (i32); then per feature
/// `geometry, feature, kind, radius, stroke, height, base`.
///
/// Records cross between workers as their words, the tile's geometry once
/// beside them; a malformed one is skipped when picked.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MeshHits {
    words: Vec<u32>,
    /// Its geometry's index in the tile's ([`link`](Self::link)).
    layer: u32,
    #[serde(skip)]
    geometry: Arc<HitGeometry>,
    /// Per feature: its record's first word and world bounds, worked out
    /// on first use.
    #[serde(skip)]
    bounds: OnceLock<Vec<Bounds>>,
    /// Records by feature index, worked out on first use.
    #[serde(skip)]
    by_feature: OnceLock<HashMap<u32, u32>>,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    record: u32,
    min: DVec2,
    max: DVec2,
}

const MESH_HEADER: usize = 9;
const NODE_RECORD: usize = 7;

/// One feature of a [`MeshHits`], unpacked but for its geometry.
struct FeatureRef<'a> {
    bounds: &'a [u32],
    id: Option<&'a [u32]>,
    feature: u32,
    /// Its index in the data tile.
    data_feature: u32,
    kind: u32,
    extent: u32,
    /// Its layer's clip buffer ([`HitGeometry`]'s).
    clip: i32,
    radius: f32,
    stroke: f32,
    height: f32,
    base: f32,
    geometry: &'a [u32],
}

impl FeatureRef<'_> {
    /// The box, in data units, a polygon of it answers within when it
    /// reaches past it: the square and its layer's clip buffer.
    fn clip_box(&self) -> Option<(f64, f64)> {
        let (lo, hi) = (-self.clip, self.extent as i32 + self.clip);
        let b = |i: usize| self.bounds[i] as i32;
        (self.kind == 0 && (b(0) < lo || b(1) < lo || b(2) > hi || b(3) > hi))
            .then(|| (f64::from(lo), f64::from(hi)))
    }

    /// Whether its bounds meet the tile's half-open square, as GL JS's
    /// `FeatureIndex::insert` asks before it indexes a feature: a copy held
    /// wholly in a tile's buffer answers only from the tile it lies in.
    fn meets_square(&self) -> bool {
        let b = |i: usize| self.bounds[i] as i32;
        let extent = self.extent as i32;
        b(0) < extent && b(1) < extent && b(2) >= 0 && b(3) >= 0
    }

    fn id(&self) -> Option<String> {
        let mut r = Reader::new(self.id?);
        r.text().ok()
    }

    /// Per part, its rings' points.
    fn parts(&self) -> Result<Vec<Vec<&[u32]>>, String> {
        let mut r = Reader::new(self.geometry);
        let mut parts = Vec::new();
        for _ in 0..r.next()? {
            let mut rings = Vec::new();
            for _ in 0..r.next()? {
                let n = r.next()? as usize;
                rings.push(r.take(n.checked_mul(2).ok_or(MALFORMED)?)?);
            }
            parts.push(rings);
        }
        if !r.done() {
            return Err(MALFORMED.into());
        }
        Ok(parts)
    }
}

/// Closed polygon ring `ring` clipped to the box `lo..hi` on both axes
/// (Sutherland–Hodgman), closed; empty when nothing of it is inside.
fn clip_ring(ring: &[DVec2], lo: f64, hi: f64) -> Vec<DVec2> {
    let mut points = ring.to_vec();
    if points.len() > 1 && points.first() == points.last() {
        points.pop();
    }
    for axis in 0..2 {
        for (bound, keep_above) in [(lo, true), (hi, false)] {
            let inside = |p: DVec2| (p[axis] >= bound) == keep_above || p[axis] == bound;
            let mut out = Vec::with_capacity(points.len() + 4);
            for (i, &b) in points.iter().enumerate() {
                let a = points[(i + points.len() - 1) % points.len()];
                if inside(a) != inside(b) {
                    let t = (bound - a[axis]) / (b[axis] - a[axis]);
                    out.push(a + (b - a) * t);
                }
                if inside(b) {
                    out.push(b);
                }
            }
            points = out;
            if points.is_empty() {
                return points;
            }
        }
    }
    points.push(points[0]);
    points
}

fn point(words: &[u32; 2]) -> [i32; 2] {
    [words[0] as i32, words[1] as i32]
}

impl MeshHits {
    /// A node's hits, their records in the tile's `layer`th geometry.
    pub(crate) fn new(layer: u32) -> Self {
        let (lo, hi) = (i32::MAX as u32, i32::MIN as u32);
        Self {
            words: vec![0, 0, 0, 0, 0, lo, lo, hi, hi],
            layer,
            ..Self::default()
        }
    }

    /// Adds feature `feature` of its source layer, its record at `at` in
    /// `geometry`, of `kind` (0: polygons, 1: lines, 2: points, [`LABEL`]:
    /// a label's feature, which labels pick from placement), with the
    /// node's `[radius, stroke, height, base]` of it: a circle's radius
    /// and stroke width; half a line's width, its gap width, and its
    /// offset's size and offset; an extrusion's height and base.
    pub(crate) fn push(
        &mut self,
        geometry: &HitGeometry,
        at: u32,
        feature: u32,
        kind: u32,
        dimensions: [f32; 4],
    ) {
        let w = &mut self.words;
        w[0] += 1;
        let [radius, stroke, height, base] = dimensions;
        for (i, v) in [radius, stroke, height.max(base)].into_iter().enumerate() {
            w[1 + i] = f32::from_bits(w[1 + i]).max(v).to_bits();
        }
        w[4] = geometry.extent;
        if let Ok(record) = geometry.record(at) {
            for i in 0..2 {
                w[5 + i] = (w[5 + i] as i32).min(record[i] as i32) as u32;
                w[7 + i] = (w[7 + i] as i32).max(record[2 + i] as i32) as u32;
            }
        }
        w.extend([at, feature, kind]);
        w.extend(dimensions.map(f32::to_bits));
    }

    /// Points its records at the tile's hit geometries, as the tile
    /// arrives.
    pub(crate) fn link(&mut self, geometries: &[Arc<HitGeometry>]) {
        if let Some(geometry) = geometries.get(self.layer as usize) {
            self.geometry = Arc::clone(geometry);
        }
    }

    /// Bytes its records take, with its share of the tile's geometry.
    pub fn bytes(&self) -> u64 {
        self.words.len() as u64 * 4
            + self.geometry.bytes() / Arc::strong_count(&self.geometry) as u64
    }

    pub fn is_empty(&self) -> bool {
        self.words.first().is_none_or(|&n| n == 0)
    }

    fn header(&self, i: usize) -> f32 {
        self.words.get(i).map_or(0.0, |&w| f32::from_bits(w))
    }

    /// Its features, by their records' first words.
    fn features(&self) -> impl Iterator<Item = (u32, FeatureRef<'_>)> {
        (0..self.words.first().copied().unwrap_or(0) as usize)
            .map(|i| (MESH_HEADER + i * NODE_RECORD) as u32)
            .filter_map(|at| Some((at, self.feature(at).ok()?)))
    }

    /// The feature whose record starts at `at`; malformed geometry is an
    /// error, never a panic.
    fn feature(&self, at: u32) -> Result<FeatureRef<'_>, String> {
        let at = at as usize;
        let node = self.words.get(at..at + NODE_RECORD).ok_or(MALFORMED)?;
        let record = self.geometry.record(node[0])?;
        let mut r = Reader::new(record);
        let bounds = r.take(4)?;
        let data_feature = r.next()?;
        let id = if r.next()? != 0 {
            let start = r.at;
            let len = r.next()? as usize;
            r.take(len.div_ceil(4))?;
            Some(&record[start..r.at])
        } else {
            None
        };
        let f = |i: usize| f32::from_bits(node[i]);
        Ok(FeatureRef {
            bounds,
            id,
            feature: node[1],
            data_feature,
            kind: node[2],
            extent: self.geometry.extent,
            clip: self.geometry.clip,
            radius: f(3),
            stroke: f(4),
            height: f(5),
            base: f(6),
            geometry: &record[r.at..],
        })
    }

    /// The feature at index `feature` of the layer its node laid out.
    fn label(&self, feature: u32) -> Option<FeatureRef<'_>> {
        let by_feature = self
            .by_feature
            .get_or_init(|| self.features().map(|(at, f)| (f.feature, at)).collect());
        self.feature(*by_feature.get(&feature)?).ok()
    }

    /// The features, unpacked.
    #[cfg(test)]
    pub(crate) fn decode(&self) -> Vec<HitFeature> {
        self.features()
            .filter_map(|(_, f)| {
                let parts = f
                    .parts()
                    .ok()?
                    .into_iter()
                    .map(|rings| {
                        rings
                            .into_iter()
                            .map(|ring| ring.as_chunks::<2>().0.iter().map(point).collect())
                            .collect()
                    })
                    .collect();
                Some(HitFeature {
                    id: f.id(),
                    feature: f.feature,
                    parts,
                })
            })
            .collect()
    }

    /// The features' world bounds in the primary copy, if the tile maps
    /// them exactly from its data units' (an affine tile).
    fn mesh_bounds(&self, mapping: &TileMapping) -> Option<(DVec2, DVec2)> {
        let extent = *self.words.get(4)?;
        let local = self.words.get(5..9)?;
        (extent != 0 && matches!(mapping, TileMapping::Affine { .. }))
            .then(|| affine_bounds(mapping, extent, local))
    }

    /// Each feature's world bounds in the primary copy, worked out once:
    /// from its bounds in data units on an affine tile, from its points on
    /// a warped one.
    fn bounds(&self, mapping: &TileMapping) -> &[Bounds] {
        self.bounds.get_or_init(|| {
            let mut out = Vec::new();
            for (at, f) in self.features() {
                let (min, max) = match mapping {
                    TileMapping::Affine { .. } => affine_bounds(mapping, f.extent, f.bounds),
                    TileMapping::Warped(_) => {
                        let transform = mapping.mapper(f.extent);
                        let (mut min, mut max) = (DVec2::INFINITY, DVec2::NEG_INFINITY);
                        for ring in f.parts().unwrap_or_default().into_iter().flatten() {
                            for p in ring.as_chunks::<2>().0 {
                                let [x, y] = point(p);
                                let world = transform(DVec2::new(f64::from(x), f64::from(y)));
                                min = min.min(world);
                                max = max.max(world);
                            }
                        }
                        (min, max)
                    }
                };
                out.push(Bounds {
                    record: at,
                    min,
                    max,
                });
            }
            out
        })
    }
}

/// Data-unit bounds `[min_x, min_y, max_x, max_y]` (i32 words) through an
/// affine mapping: exact, since it only scales and flips its axes. Empty
/// bounds stay empty.
fn affine_bounds(mapping: &TileMapping, extent: u32, local: &[u32]) -> (DVec2, DVec2) {
    let [x0, y0, x1, y1] = [0, 1, 2, 3].map(|i| f64::from(local[i] as i32));
    if x0 > x1 || y0 > y1 {
        return (DVec2::INFINITY, DVec2::NEG_INFINITY);
    }
    let transform = mapping.mapper(extent);
    let (a, b) = (transform(DVec2::new(x0, y0)), transform(DVec2::new(x1, y1)));
    (a.min(b), a.max(b))
}

/// How a tile's data units land on the surface.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TileMapping {
    /// [`TileMatrixSet::data_to_world`] at any extent: the tile's left and
    /// top edges and its side.
    Affine { left: f64, top: f64, span: f64 },
    /// A warped tile's series.
    Warped(Arc<orb_crs::Series>),
}

impl TileMapping {
    pub(crate) fn of(
        tms: &TileMatrixSet,
        tile: TileIndex,
        series: Option<&Arc<orb_crs::Series>>,
    ) -> Self {
        match series {
            Some(series) => Self::Warped(Arc::clone(series)),
            None => {
                let bounds = tms.tile_bounds(tile);
                Self::Affine {
                    left: bounds.min.x,
                    top: bounds.max.y,
                    span: tms.tile_span(tile.z),
                }
            }
        }
    }

    /// Data units (`extent` per side) to world, in the primary copy.
    fn mapper(&self, extent: u32) -> impl Fn(DVec2) -> DVec2 + '_ {
        let affine = match self {
            &Self::Affine { left, top, span } => {
                let scale = span / f64::from(extent);
                Some(glam::DAffine2::from_mat2_translation(
                    glam::DMat2::from_diagonal(DVec2::new(scale, -scale)),
                    DVec2::new(left, top),
                ))
            }
            Self::Warped(_) => None,
        };
        move |p| match (affine, self) {
            (Some(a), _) => a.transform_point2(p),
            (None, Self::Warped(s)) => s.position(p / f64::from(extent)),
            (None, Self::Affine { .. }) => unreachable!("affine mappings have their transform"),
        }
    }
}

/// A tile's nodes' features, by node key.
pub(crate) type Meshes = Vec<(u32, Arc<MeshHits>)>;

/// A tile's picking geometry: where it lands and its nodes' features.
#[derive(Clone, Debug)]
pub(crate) struct TileHits {
    pub(crate) mapping: TileMapping,
    /// Per node key, its features.
    pub(crate) meshes: Meshes,
}

/// A tile a frame drew: each world copy, with the regions it draws in.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DrawnTile {
    /// The plan source's key.
    pub(crate) source: u32,
    pub(crate) tile: TileIndex,
    /// The tile its data comes from: an overzoom slice's ancestor at its
    /// source's maxzoom, whose features queries answer with.
    pub(crate) data: TileIndex,
    pub(crate) copies: Vec<DrawnCopy>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DrawnCopy {
    pub(crate) wrap: i32,
    /// Its regions' corners in world space, primary copy.
    pub(crate) clips: Vec<[DVec2; 4]>,
}

impl DrawnTile {
    /// A tile's draws, as frames place them.
    pub(crate) fn new(
        source: u32,
        (tile, data): (TileIndex, TileIndex),
        draws: &[&orb_source::Draw],
        tms: &TileMatrixSet,
        series: Option<&orb_crs::Series>,
    ) -> Self {
        let corners = |region: TileIndex| match series {
            None => {
                let b = tms.tile_bounds(region);
                [
                    b.min,
                    DVec2::new(b.max.x, b.min.y),
                    b.max,
                    DVec2::new(b.min.x, b.max.y),
                ]
            }
            Some(series) => {
                let depth = region.z - tile.z;
                let n = f64::from(1u32 << depth);
                let x = f64::from(region.col - (tile.col << depth)) / n;
                let y = f64::from(region.row - (tile.row << depth)) / n;
                // Tile fractions run down; corners as the affine case's.
                [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
                    .map(|(dx, dy)| series.position(DVec2::new(x + dx / n, y + dy / n)))
            }
        };
        Self {
            source,
            tile,
            data,
            copies: draws
                .iter()
                .map(|d| DrawnCopy {
                    wrap: d.wrap,
                    clips: d.regions.iter().map(|&r| corners(r)).collect(),
                })
                .collect(),
        }
    }
}

/// A label or icon the last placement commit shows, anchored in world
/// space; its boxes are in pixels at the anchor's scale.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LabelHit {
    pub(crate) node: u32,
    pub(crate) key: FeatureKey,
    /// Its anchor in world space, in the copy placement drew.
    pub(crate) anchor: DVec2,
    /// Its text and icon boxes `[x1, y1, x2, y2]`, shown ones only, in
    /// tile units at 512 pixels a tile side from its anchor.
    pub(crate) boxes: Vec<[f64; 4]>,
    /// A line label's collision circles along its line, `[dx, dy, radius]`
    /// in pixels from its anchor, as placement placed them.
    pub(crate) circles: Vec<[f64; 3]>,
}

/// What a node's features take from its layer's paint to pick.
#[derive(Clone, Debug)]
enum HitPaint {
    Fill {
        translate: Property<[f32; 2]>,
        anchor: Property<TranslateAnchor>,
    },
    Line {
        width: Property<f32>,
        gap: Property<f32>,
        offset: Property<f32>,
        translate: Property<[f32; 2]>,
        anchor: Property<TranslateAnchor>,
    },
    Circle {
        radius: Property<f32>,
        stroke: Property<f32>,
        /// `circle-pitch-scale` and `circle-pitch-alignment`.
        scale: Property<TranslateAnchor>,
        alignment: Property<TranslateAnchor>,
        translate: Property<[f32; 2]>,
        anchor: Property<TranslateAnchor>,
    },
    Extrusion {
        base: Property<f32>,
        height: Property<f32>,
        translate: Property<[f32; 2]>,
        anchor: Property<TranslateAnchor>,
    },
    Other,
}

/// Bound per feature: the value is the feature's, laid out with the tile.
fn per_feature(kind: Kind) -> bool {
    matches!(kind, Kind::Source | Kind::Composite)
}

impl HitPaint {
    fn of(l: &orb_style::Layer) -> Self {
        use orb_style::{
            CIRCLE_PITCH_ALIGNMENT, CIRCLE_PITCH_SCALE, CIRCLE_RADIUS, CIRCLE_STROKE_WIDTH,
            CIRCLE_TRANSLATE, CIRCLE_TRANSLATE_ANCHOR, EXTRUSION_BASE, EXTRUSION_HEIGHT,
            EXTRUSION_TRANSLATE, EXTRUSION_TRANSLATE_ANCHOR, FILL_TRANSLATE, FILL_TRANSLATE_ANCHOR,
            LINE_GAP_WIDTH, LINE_OFFSET, LINE_TRANSLATE, LINE_TRANSLATE_ANCHOR, LINE_WIDTH,
        };
        match l.ty {
            LayerType::Fill => Self::Fill {
                translate: l.get(FILL_TRANSLATE).to_owned(),
                anchor: l.get(FILL_TRANSLATE_ANCHOR).to_owned(),
            },
            LayerType::Line => Self::Line {
                width: l.get(LINE_WIDTH).to_owned(),
                gap: l.get(LINE_GAP_WIDTH).to_owned(),
                offset: l.get(LINE_OFFSET).to_owned(),
                translate: l.get(LINE_TRANSLATE).to_owned(),
                anchor: l.get(LINE_TRANSLATE_ANCHOR).to_owned(),
            },
            LayerType::Circle => Self::Circle {
                radius: l.get(CIRCLE_RADIUS).to_owned(),
                stroke: l.get(CIRCLE_STROKE_WIDTH).to_owned(),
                scale: l.get(CIRCLE_PITCH_SCALE).to_owned(),
                alignment: l.get(CIRCLE_PITCH_ALIGNMENT).to_owned(),
                translate: l.get(CIRCLE_TRANSLATE).to_owned(),
                anchor: l.get(CIRCLE_TRANSLATE_ANCHOR).to_owned(),
            },
            LayerType::FillExtrusion => Self::Extrusion {
                base: l.get(EXTRUSION_BASE).to_owned(),
                height: l.get(EXTRUSION_HEIGHT).to_owned(),
                translate: l.get(EXTRUSION_TRANSLATE).to_owned(),
                anchor: l.get(EXTRUSION_TRANSLATE_ANCHOR).to_owned(),
            },
            _ => Self::Other,
        }
    }

    /// The dimensions picking reads of the paint.
    fn dimensions(&self) -> Vec<&Property<f32>> {
        match self {
            Self::Circle { radius, stroke, .. } => vec![radius, stroke],
            Self::Line {
                width, gap, offset, ..
            } => vec![width, gap, offset],
            Self::Extrusion { base, height, .. } => vec![base, height],
            _ => Vec::new(),
        }
    }

    /// Whether a dimension reads feature state: features' extents are
    /// known only when picked.
    fn reads_state(&self) -> bool {
        self.dimensions()
            .iter()
            .any(|p| p.as_expression().is_some_and(|e| e.reads_feature_state()))
    }

    fn translate(&self) -> Option<(&Property<[f32; 2]>, &Property<TranslateAnchor>)> {
        match self {
            Self::Fill { translate, anchor }
            | Self::Line {
                translate, anchor, ..
            }
            | Self::Circle {
                translate, anchor, ..
            }
            | Self::Extrusion {
                translate, anchor, ..
            } => Some((translate, anchor)),
            Self::Other => None,
        }
    }
}

/// A node's layer, as picking reads it.
#[derive(Clone, Debug)]
struct HitLayer {
    /// The layer's index in the style: hits above lower ones.
    order: usize,
    id: String,
    source: String,
    source_layer: Option<String>,
    minzoom: f64,
    maxzoom: f64,
    visible: bool,
    paint: HitPaint,
}

impl HitLayer {
    /// [`orb_style::Layer::is_visible_at`].
    fn visible_at(&self, zoom: f64) -> bool {
        let above_min = self.minzoom == 0.0 || zoom >= self.minzoom;
        let below_max = self.maxzoom == 0.0 || zoom < self.maxzoom;
        self.visible && above_min && below_max
    }

    fn hit(&self, id: Option<String>, generation: u64) -> Hit {
        Hit {
            layer: self.id.clone(),
            source: self.source.clone(),
            source_layer: self.source_layer.clone(),
            id,
            generation,
        }
    }
}

/// The nodes that draw from a source: those picking can report.
fn picked(plan: &crate::style::StylePlan) -> impl Iterator<Item = &crate::style::NodePlan> {
    plan.nodes.iter().filter(|n| n.source < plan.sources.len())
}

/// Every render node's layer, by node key.
#[derive(Clone, Debug, Default)]
pub(crate) struct HitLayers {
    nodes: HashMap<u32, HitLayer>,
}

impl HitLayers {
    pub(crate) fn of(plan: &crate::style::StylePlan) -> Self {
        let nodes = picked(plan)
            .map(|node| {
                let layer = &plan.style.layers[node.layer];
                (
                    node.key,
                    HitLayer {
                        order: node.layer,
                        id: layer.id.clone(),
                        source: plan.sources[node.source].id.clone(),
                        source_layer: plan.source_layer(node).map(str::to_string),
                        minzoom: layer.minzoom,
                        maxzoom: layer.maxzoom,
                        visible: layer.visibility == orb_style::Visibility::Visible,
                        paint: HitPaint::of(layer),
                    },
                )
            })
            .collect();
        Self { nodes }
    }
}

/// What a query reads: the layers, the drawn tiles with their geometry,
/// the labels and the terrain they stand on.
pub(crate) struct Scene<'a> {
    pub(crate) layers: &'a HitLayers,
    pub(crate) drawn: &'a [DrawnTile],
    /// Per drawn tile, its geometry, if it has any.
    pub(crate) tiles: Vec<Option<&'a TileHits>>,
    pub(crate) labels: &'a [LabelHit],
    /// The surface's wrap period, 0 if it doesn't wrap.
    pub(crate) period: f64,
    pub(crate) generation: u64,
    /// The terrain's elevations, metres, as the frame drew them.
    pub(crate) elevation: Option<&'a dyn orb_camera::Elevation>,
    /// Feature states, for paint that reads them.
    pub(crate) states: Option<&'a crate::state::FeatureStates>,
}

/// A dimension at the query's zoom: under the feature's state where it
/// reads state (with its captured data where it mixes in properties),
/// else `fallback` (the layer's value, or the feature's at layout).
fn live(p: &Property<f32>, zoom: f64, state: FeatureState<'_>, fallback: f32) -> f32 {
    let p_ref = p.get();
    if !p.as_expression().is_some_and(|e| e.reads_feature_state()) {
        return fallback;
    }
    match state {
        Some((state, data)) if p_ref.is_state_lookup() || data.is_some() => {
            p_ref.evaluate_state(zoom, state, data)
        }
        None if p_ref.is_state_lookup() => p_ref.evaluate_state(zoom, &[], None),
        _ => fallback,
    }
}

/// A feature a query may report, in the order that puts upper ones last:
/// its layer, then tiles before labels, then its tile's rank (GL JS's
/// tile order, [`tile_rank`]) or a label's ([`label_rank`]), then the
/// drawing order (a label's feature index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Order {
    layer: usize,
    label: bool,
    tile: u64,
    mesh: u32,
    copy: u32,
    feature: u32,
}

/// A layer's values at the query's camera.
#[derive(Clone, Copy, Debug)]
struct LayerAt {
    /// Circles' radius and stroke, lines' half width; `None` where bound
    /// per feature.
    radius: Option<f32>,
    stroke: Option<f32>,
    /// Extrusions' base and height in metres; `None` where per feature.
    base: Option<f32>,
    height: Option<f32>,
    /// Lines' gap width and offset in pixels; `None` where per feature.
    gap: Option<f32>,
    offset: Option<f32>,
    /// Circles: whether they scale with the map under pitch, and whether
    /// they lie on it.
    circle_pitch: (bool, bool),
    /// The translation in world units.
    shift: DVec2,
}

/// One query's pixels in world space, and what the camera asks of layers.
struct Query<'a> {
    view: &'a View,
    camera: &'a CameraState,
    zoom: f64,
    elevation: Option<&'a dyn orb_camera::Elevation>,
}

impl Query<'_> {
    /// Where a world position `height` world units above the ground draws,
    /// in logical pixels, as the renderer places it: on the terrain, on the
    /// globe or its low-zoom plane. `None` behind the eye or the horizon.
    fn project(&self, world: DVec2, height: f64) -> Option<Vec2> {
        self.project_depth(world, height).map(|(p, _)| p)
    }

    /// [`project`](Self::project) with the position's depth, 0 at the near
    /// plane and 1 at the far one.
    fn project_depth(&self, world: DVec2, height: f64) -> Option<(Vec2, f64)> {
        let ground = self
            .elevation
            .and_then(|e| e.at(world))
            .map_or(0.0, |metres| metres * self.view.height_scale());
        self.view
            .project_depth(world, ground + height)
            .map(|(p, depth)| (p.as_vec2(), depth))
    }

    /// Pixels per pixel at the view's centre, at a world position: the
    /// perspective ratio GL JS scales labels by.
    fn local_scale(&self, world: DVec2) -> Option<f64> {
        let step = self.view.world_per_px();
        let (a, b) = (
            self.project(world, 0.0)?,
            self.project(world + DVec2::new(step, 0.0), 0.0)?,
        );
        Some(f64::from(a.distance(b)))
    }

    fn layer(&self, layer: &HitLayer) -> LayerAt {
        let zoom = self.zoom;
        let uniform = |p: &Property<f32>| (!per_feature(p.kind())).then(|| p.evaluate(zoom));
        // Extrusions bind their base and height per feature unless constant
        // or zoom-only.
        let level = |p: &Property<f32>| {
            matches!(p.kind(), Kind::Constant | Kind::Camera).then(|| p.evaluate(zoom))
        };
        let (radius, stroke, base, height) = match &layer.paint {
            HitPaint::Circle { radius, stroke, .. } => {
                (uniform(radius), uniform(stroke), None, None)
            }
            HitPaint::Line { width, .. } => (uniform(width).map(|w| w / 2.0), None, None, None),
            HitPaint::Extrusion { base, height, .. } => (None, None, level(base), level(height)),
            _ => (None, None, None, None),
        };
        let (gap, offset) = match &layer.paint {
            HitPaint::Line { gap, offset, .. } => (uniform(gap), uniform(offset)),
            _ => (Some(0.0), Some(0.0)),
        };
        let circle_pitch = match &layer.paint {
            HitPaint::Circle {
                scale, alignment, ..
            } => (
                scale.evaluate(zoom) == TranslateAnchor::Map,
                alignment.evaluate(zoom) == TranslateAnchor::Map,
            ),
            _ => (false, false),
        };
        let shift = match layer.paint.translate() {
            Some((translate, anchor)) => {
                let [x, y] = translate.evaluate(zoom);
                let mut shift = DVec2::new(f64::from(x), f64::from(y));
                if anchor.evaluate(zoom) == TranslateAnchor::Viewport {
                    let (sin, cos) = self.camera.bearing.to_radians().sin_cos();
                    shift =
                        DVec2::new(shift.x * cos - shift.y * sin, shift.x * sin + shift.y * cos);
                }
                shift.y = -shift.y;
                shift * self.view.world_per_px()
            }
            None => DVec2::ZERO,
        };
        LayerAt {
            radius,
            stroke,
            base,
            height,
            gap,
            offset,
            circle_pitch,
            shift,
        }
    }

    /// The feature's tolerance in pixels, its line offset in pixels, and
    /// an extrusion's base and top in world units, under its state.
    fn feature(
        &self,
        layer: &HitLayer,
        at: &LayerAt,
        f: &FeatureRef<'_>,
        state: FeatureState<'_>,
    ) -> (f32, f32, Option<(f64, f64)>) {
        let zoom = self.zoom;
        let (radius, offset) = match &layer.paint {
            HitPaint::Circle { radius, stroke, .. } => (
                live(radius, zoom, state, at.radius.unwrap_or(f.radius))
                    + live(stroke, zoom, state, at.stroke.unwrap_or(f.stroke)),
                0.0,
            ),
            // GL JS `getLineWidth`: a gap's line is two strokes about it.
            HitPaint::Line {
                width, gap, offset, ..
            } => {
                let width = live(width, zoom, state, at.radius.unwrap_or(f.radius) * 2.0);
                let gap = live(gap, zoom, state, at.gap.unwrap_or(f.stroke));
                let total = if gap > 0.0 { gap + 2.0 * width } else { width };
                (
                    total / 2.0,
                    live(offset, zoom, state, at.offset.unwrap_or(f.base)),
                )
            }
            HitPaint::Extrusion { .. } => (0.0, 0.0),
            _ => (f.radius, 0.0),
        };
        let elevation = match &layer.paint {
            HitPaint::Extrusion { base, height, .. } => {
                let scale = self.view.height_scale();
                Some((
                    f64::from(live(base, zoom, state, at.base.unwrap_or(f.base))) * scale,
                    f64::from(live(height, zoom, state, at.height.unwrap_or(f.height))) * scale,
                ))
            }
            _ => None,
        };
        (radius, offset, elevation)
    }

    /// The largest tolerance and top any of a mesh's features can have.
    fn mesh_bound(&self, layer: &HitLayer, at: &LayerAt, mesh: &MeshHits) -> (f32, f64) {
        let (radius, stroke, height) = (mesh.header(1), mesh.header(2), mesh.header(3));
        match &layer.paint {
            HitPaint::Circle { .. } => (
                at.radius.unwrap_or(radius) + at.stroke.unwrap_or(stroke),
                0.0,
            ),
            HitPaint::Line { .. } => {
                let width = at.radius.unwrap_or(radius) * 2.0;
                let gap = at.gap.unwrap_or(stroke);
                let total = if gap > 0.0 { gap + 2.0 * width } else { width };
                (total / 2.0 + at.offset.map_or(height, f32::abs), 0.0)
            }
            HitPaint::Extrusion { .. } => {
                let top = at.height.unwrap_or(height).max(at.base.unwrap_or(height));
                (0.0, f64::from(top) * self.view.height_scale())
            }
            _ => (radius, 0.0),
        }
    }

    /// World bounds of everything at heights `0..=top` that projects into
    /// the pixel box `min..max` grown by `pad`; `None` when part of it
    /// sees above the horizon.
    fn ground(&self, min: DVec2, max: DVec2, pad: f64, top: f64) -> Option<(DVec2, DVec2)> {
        let (min, max) = (min - pad, max + pad);
        let corners = [min, DVec2::new(max.x, min.y), max, DVec2::new(min.x, max.y)];
        let (mut lo, mut hi) = (DVec2::INFINITY, DVec2::NEG_INFINITY);
        // On terrain, features stand anywhere in its range of heights.
        let (low, high) = match self.elevation {
            None => (0.0, 0.0),
            Some(e) => {
                let world = self.view.projection().world_bounds();
                let (low, high) = e.range(world, 0)?;
                let scale = self.view.height_scale();
                (low.min(0.0) * scale, high.max(0.0) * scale)
            }
        };
        for z in [low, top + high] {
            for c in corners {
                let world = self.view.unproject_at(c, z)?;
                lo = lo.min(world);
                hi = hi.max(world);
            }
        }
        // Rounding in the inverse: a little more, never less.
        let slack = (hi - lo).max_element().max(1.0) * 1e-6;
        Some((lo - slack, hi + slack))
    }
}

impl Scene<'_> {
    /// Candidates whose projection may touch the pixel box `min..max`,
    /// upper ones first, each with what its exact test needs.
    fn candidates(&self, q: &Query<'_>, min: DVec2, max: DVec2) -> Vec<(Order, Candidate<'_>)> {
        let mut layers: HashMap<u32, LayerAt> = HashMap::new();
        let layer_at = |node: u32| -> Option<&HitLayer> {
            let layer = self.layers.nodes.get(&node)?;
            layer.visible_at(q.zoom).then_some(layer)
        };
        // The furthest any feature reaches in pixels and how high.
        let (mut pad, mut top, mut unbounded) = (0f32, 0f64, false);
        for tile in self.tiles.iter().flatten() {
            for (node, mesh) in &tile.meshes {
                let Some(layer) = layer_at(*node).filter(|l| !matches!(l.paint, HitPaint::Other))
                else {
                    continue;
                };
                let at = layers.entry(*node).or_insert_with(|| q.layer(layer));
                let (radius, height) = q.mesh_bound(layer, at, mesh);
                pad = pad.max(radius);
                top = top.max(height);
                unbounded |= layer.paint.reads_state();
            }
        }
        // A pixel's worth for the projection's rounding to f32.
        let ground = if unbounded {
            None
        } else {
            q.ground(min, max, f64::from(pad) + 1.0, top)
        };
        let mut out = Vec::new();
        for (i, (drawn, tile)) in self.drawn.iter().zip(&self.tiles).enumerate() {
            let Some(tile) = tile else {
                continue;
            };
            for (m, (node, mesh)) in tile.meshes.iter().enumerate() {
                let Some(&at) = layers.get(node) else {
                    continue;
                };
                let layer = &self.layers.nodes[node];
                let outside = |(min, max): (DVec2, DVec2), offset: DVec2| {
                    ground.is_some_and(|(lo, hi)| {
                        (max + offset).cmplt(lo).any() || (min + offset).cmpgt(hi).any()
                    })
                };
                let offsets: Vec<DVec2> = drawn
                    .copies
                    .iter()
                    .map(|c| DVec2::new(f64::from(c.wrap) * self.period, 0.0) + at.shift)
                    .collect();
                if let Some(b) = mesh.mesh_bounds(&tile.mapping)
                    && offsets.iter().all(|&o| outside(b, o))
                {
                    continue;
                }
                let bounds = mesh.bounds(&tile.mapping);
                for (d, copy) in drawn.copies.iter().enumerate() {
                    let offset = offsets[d];
                    for (f, b) in bounds.iter().enumerate() {
                        if outside((b.min, b.max), offset) {
                            continue;
                        }
                        out.push((
                            Order {
                                layer: layer.order,
                                label: false,
                                tile: tile_rank(drawn.tile, i),
                                mesh: m as u32,
                                copy: d as u32,
                                feature: f as u32,
                            },
                            Candidate::Feature {
                                layer,
                                drawn,
                                copy,
                                mapping: &tile.mapping,
                                mesh,
                                record: b.record,
                                at,
                            },
                        ));
                    }
                }
            }
        }
        for label in self.labels.iter() {
            let Some(layer) = self.layers.nodes.get(&label.node) else {
                continue;
            };
            out.push((
                Order {
                    layer: layer.order,
                    label: true,
                    tile: label_rank(label.key.tile),
                    mesh: 0,
                    copy: 0,
                    feature: label.key.feature,
                },
                Candidate::Label { layer, label },
            ));
        }
        out.sort_unstable_by_key(|c| std::cmp::Reverse(c.0));
        out
    }

    /// The candidate in pixels, for its exact test, with its hit and key.
    fn project(&self, q: &Query<'_>, c: &Candidate<'_>) -> Option<(ScreenHit, Hit, FeatureKey)> {
        match *c {
            Candidate::Feature {
                layer,
                drawn,
                copy,
                mapping,
                mesh,
                record,
                at,
            } => {
                let f = mesh.feature(record).ok()?;
                if !f.meets_square() {
                    return None;
                }
                let id = f.id();
                let state = match (self.states, &id) {
                    (Some(states), Some(id)) if layer.paint.reads_state() => {
                        states.feature(drawn.source, layer.source_layer.as_deref(), id)
                    }
                    _ => None,
                };
                let (radius, offset, elevation) = q.feature(layer, &at, &f, state);
                // A building answers from every drawn copy holding it,
                // clipped to the copy's regions, as GL JS's tiles answer:
                // the copies' pieces make up what's drawn, whether a copy
                // draws it whole or clipped (the extrusion shader's
                // `extrusion_drawn`).
                let screen = ScreenHit::project(
                    &f,
                    layer.source_layer.is_some(),
                    mapping,
                    copy,
                    q,
                    self.period,
                    (f64::from(radius), f64::from(offset)),
                    elevation,
                    (at.shift, at.circle_pitch),
                )?;
                // An overzoom slice's feature answers as its data tile's,
                // once, as MapLibre Native's overscaled tiles answer.
                let key = FeatureKey {
                    source: drawn.source,
                    tile: drawn.data,
                    feature: f.data_feature,
                };
                Some((screen, layer.hit(id, self.generation), key))
            }
            Candidate::Label { layer, label } => {
                let anchor = q.project(label.anchor, 0.0)?.as_dvec2();
                // GL JS: tile units at 512 px per tile side, scaled by the
                // perspective ratio.
                let scale =
                    512.0 / orb_symbol::EXTENT * (0.5 + 0.5 * q.local_scale(label.anchor)?);
                let boxes = label
                    .boxes
                    .iter()
                    .map(|b| {
                        [
                            anchor.x + b[0] * scale,
                            anchor.y + b[1] * scale,
                            anchor.x + b[2] * scale,
                            anchor.y + b[3] * scale,
                        ]
                    })
                    .collect();
                let circles = label
                    .circles
                    .iter()
                    .map(|&[dx, dy, r]| [anchor.x + dx, anchor.y + dy, r])
                    .collect();
                let (id, key) = self.label_feature(label)?;
                Some((
                    ScreenHit::label(boxes, circles),
                    layer.hit(id, self.generation),
                    key,
                ))
            }
        }
    }

    /// A label's feature id and where it lives in its data tile, from its
    /// tile's records of the symbol layer's features (on its icon node or
    /// its text node).
    fn label_feature(&self, label: &LabelHit) -> Option<(Option<String>, FeatureKey)> {
        let key = label.key;
        let at = self
            .drawn
            .iter()
            .position(|d| d.source == key.source && d.tile == key.tile)?;
        let f = self.tiles[at]?
            .meshes
            .iter()
            .filter(|(node, _)| *node / 2 == label.node / 2)
            .find_map(|(_, mesh)| mesh.label(key.feature))?;
        let key = FeatureKey {
            tile: self.drawn[at].data,
            feature: f.data_feature,
            ..key
        };
        Some((f.id(), key))
    }

    /// The topmost feature under `point`, in pixels of `view`: the first
    /// a point query answers, the nearest extrusion among them where one
    /// is uppermost.
    pub(crate) fn hit_test(
        &self,
        view: &View,
        camera: &CameraState,
        zoom: f64,
        point: DVec2,
    ) -> Option<Hit> {
        let hits = self.query(view, camera, zoom, point, point, None);
        hits.into_iter().next().map(|(hit, ..)| hit)
    }

    /// Every feature drawn touching the pixel box `min..max` (a point when
    /// they are equal), topmost first, once per layer and feature, with its
    /// layer's index in the style.
    pub(crate) fn query(
        &self,
        view: &View,
        camera: &CameraState,
        zoom: f64,
        min: DVec2,
        max: DVec2,
        layers: Option<&[String]>,
    ) -> Vec<(Hit, usize, FeatureKey)> {
        if !(min.is_finite() && max.is_finite()) {
            return Vec::new();
        }
        let q = Query {
            view,
            camera,
            zoom,
            elevation: self.elevation,
        };
        let mut seen = std::collections::HashSet::new();
        let (mut flat, mut solid) = (Vec::new(), Vec::new());
        for (order, c) in self.candidates(&q, min, max) {
            let layer = match c {
                Candidate::Feature { layer, .. } | Candidate::Label { layer, .. } => layer,
            };
            if layers.is_some_and(|l| !l.contains(&layer.id)) {
                continue;
            }
            let Some((screen, hit, key)) = self.project(&q, &c) else {
                continue;
            };
            if screen.intersects(min, max) && seen.insert((order.layer, key)) {
                match screen.depth(min, max) {
                    Some(depth) => solid.push((depth, (hit, order.layer, key))),
                    None => flat.push((hit, order.layer, key)),
                }
            }
        }
        let extrusions = self
            .layers
            .nodes
            .values()
            .filter(|l| matches!(l.paint, HitPaint::Extrusion { .. }))
            .map(|l| l.order)
            .collect();
        stack(flat, solid, extrusions, |hit| hit.1)
    }
}

/// Answers topmost first, as GL JS's `_flattenAndSortRenderedFeatures`
/// stacks them: `flat` is topmost first already, and extrusions come
/// nearest first across their layers, each where the layers above it
/// end, unless a lower layer's extrusion is nearer still. `extrusions`
/// are the style's extrusion layers; `layer` gives an answer's.
fn stack<T>(
    flat: Vec<T>,
    mut solid: Vec<(f64, T)>,
    mut extrusions: Vec<usize>,
    layer: impl Fn(&T) -> usize,
) -> Vec<T> {
    solid.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut solid = solid.into_iter().map(|(_, hit)| hit).peekable();
    let mut flat = flat.into_iter().peekable();
    extrusions.sort_unstable_by(|a, b| b.cmp(a));
    extrusions.dedup();
    let mut out = Vec::new();
    for at in extrusions {
        out.extend(std::iter::from_fn(|| flat.next_if(|f| layer(f) > at)));
        out.extend(std::iter::from_fn(|| solid.next_if(|s| layer(s) >= at)));
    }
    out.extend(flat);
    out.extend(solid);
    out
}

enum Candidate<'a> {
    Feature {
        layer: &'a HitLayer,
        drawn: &'a DrawnTile,
        copy: &'a DrawnCopy,
        mapping: &'a TileMapping,
        mesh: &'a MeshHits,
        record: u32,
        at: LayerAt,
    },
    Label {
        layer: &'a HitLayer,
        label: &'a LabelHit,
    },
}

/// A feature or label in a query's pixels, for its exact test.
#[derive(Debug)]
struct ScreenHit {
    /// Whether region clips apply: features of vector tiles.
    clipped: bool,
    /// 0 polygons, 1 lines, 2 points, 3 a label's box.
    kind: u32,
    radius: f32,
    parts: Vec<Vec<Vec<Vec2>>>,
    /// An extrusion's faces' depths (0 near, 1 far): per part, its outer
    /// ring's points'. Empty for everything flat.
    depths: Vec<Vec<f64>>,
    /// A line label's collision circles: centre and radius.
    circles: Vec<(Vec2, f32)>,
    /// The regions its tile draws in, and how far past them in pixels a
    /// query still finds it: GL JS queries every tile within the layer's
    /// query radius (its translation and width) of the query, and a
    /// tile's features run into its buffer, so a feature near a tile
    /// edge can answer from both tiles.
    clips: Vec<Vec<Vec2>>,
    clip_pad: f32,
    min: Vec2,
    max: Vec2,
}

impl ScreenHit {
    #[allow(clippy::too_many_arguments)]
    fn project(
        feature: &FeatureRef<'_>,
        clipped: bool,
        mapping: &TileMapping,
        copy: &DrawnCopy,
        q: &Query<'_>,
        period: f64,
        (radius, offset): (f64, f64),
        extrusion: Option<(f64, f64)>,
        (shift, (scale_with_map, on_map)): (DVec2, (bool, bool)),
    ) -> Option<Self> {
        if feature.kind == 2 && q.camera.pitch > 0.0 && (scale_with_map || on_map) {
            return Self::pitched_circles(
                feature,
                clipped,
                mapping,
                copy,
                q,
                period,
                radius,
                shift,
                (scale_with_map, on_map),
            );
        }
        let transform = mapping.mapper(feature.extent);
        // A line's offset in its data units, as GL JS offsets it in tile
        // units before projecting.
        let offset = match mapping {
            &TileMapping::Affine { span, .. } if offset != 0.0 && feature.kind == 1 => {
                offset * q.view.world_per_px() * f64::from(feature.extent) / span
            }
            _ => 0.0,
        };
        let wrap = DVec2::new(f64::from(copy.wrap) * period, 0.0);
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let project = |p: DVec2, height: f64| q.project_depth(transform(p) + wrap + shift, height);
        let clip_box = feature.clip_box();
        // An extrusion's faces carry depths, its top's even when it has
        // no height, as GL JS sorts every extrusion by depth.
        let elevation = extrusion.unwrap_or_default();
        let solid = elevation.1 > elevation.0;
        let (mut parts, mut depths) = (Vec::new(), Vec::new());
        // Lines under pitch, as polygons.
        let mut buffered = Vec::new();
        for part in feature.parts().ok()? {
            let part: Vec<Vec<DVec2>> = part
                .iter()
                .map(|ring| {
                    let ring: Vec<DVec2> = ring
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|p| DVec2::new(f64::from(p[0] as i32), f64::from(p[1] as i32)))
                        .collect();
                    match clip_box {
                        Some((lo, hi)) => clip_ring(&ring, lo, hi),
                        None => ring,
                    }
                })
                .filter(|ring| !ring.is_empty())
                .collect();
            if part.is_empty() {
                continue;
            }
            let mut rings = Vec::new();
            for line in &part {
                let line = if offset != 0.0 {
                    offset_line(line, offset)
                } else {
                    line.clone()
                };
                if feature.kind == 1 && q.camera.pitch > 0.0 {
                    // GL JS tests lines in tile space: under pitch their
                    // width widens toward the eye. The line buffered on
                    // the map, projected.
                    let world: Vec<DVec2> =
                        line.iter().map(|p| transform(*p) + wrap + shift).collect();
                    for polygon in buffer_line(&world, radius * q.view.world_per_px()) {
                        let ring = polygon
                            .iter()
                            .map(|&p| q.project(p, elevation.1))
                            .collect::<Option<Vec<_>>>()?;
                        buffered.push(vec![ring]);
                    }
                    continue;
                }
                let (points, depth): (Vec<Vec2>, Vec<f64>) = line
                    .iter()
                    .map(|p| project(*p, elevation.1))
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .unzip();
                if extrusion.is_some() && rings.is_empty() {
                    depths.push(depth);
                }
                rings.push(points);
            }
            parts.push(rings);
            if solid {
                for ring in &part {
                    for (a, b) in ring
                        .iter()
                        .zip(ring.iter().cycle().skip(1))
                        .take(ring.len())
                    {
                        let (face, depth) = [
                            project(*a, elevation.0)?,
                            project(*b, elevation.0)?,
                            project(*b, elevation.1)?,
                            project(*a, elevation.1)?,
                        ]
                        .into_iter()
                        .unzip();
                        parts.push(vec![face]);
                        depths.push(depth);
                    }
                }
            }
        }
        let clip_pad = (shift.length() / q.view.world_per_px()) as f32 + radius as f32;
        let (kind, radius) = if buffered.is_empty() {
            (feature.kind, radius)
        } else {
            parts = buffered;
            (0, 0.0)
        };
        for point in parts.iter().flatten().flatten() {
            min = min.min(*point);
            max = max.max(*point);
        }
        let radius = radius as f32;
        min -= Vec2::splat(radius);
        max += Vec2::splat(radius);
        let clips = copy
            .clips
            .iter()
            .map(|corners| {
                corners
                    .iter()
                    .filter_map(|&p| q.project(p + wrap, 0.0))
                    .collect()
            })
            .collect();
        Some(Self {
            clipped,
            kind,
            radius,
            parts,
            depths,
            circles: Vec::new(),
            min,
            max,
            clips,
            clip_pad,
        })
    }

    /// Circles under pitch, as GL JS queries them: with
    /// `circle-pitch-scale: map` a viewport-aligned circle's radius scales
    /// by the perspective at its centre; a map-aligned one is a circle on
    /// the ground, `radius` pixels at the view's centre (or at its own,
    /// with `circle-pitch-scale: viewport`), projected.
    #[allow(clippy::too_many_arguments)]
    fn pitched_circles(
        feature: &FeatureRef<'_>,
        clipped: bool,
        mapping: &TileMapping,
        copy: &DrawnCopy,
        q: &Query<'_>,
        period: f64,
        radius: f64,
        shift: DVec2,
        (scale_with_map, on_map): (bool, bool),
    ) -> Option<Self> {
        const SIDES: usize = 32;
        let transform = mapping.mapper(feature.extent);
        let wrap = DVec2::new(f64::from(copy.wrap) * period, 0.0);
        let (mut parts, mut circles) = (Vec::new(), Vec::new());
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for ring in feature.parts().ok()?.into_iter().flatten() {
            for p in ring.as_chunks::<2>().0 {
                let [x, y] = point(p);
                let world = transform(DVec2::new(f64::from(x), f64::from(y))) + wrap + shift;
                let ratio = q.local_scale(world)?;
                if on_map {
                    let scale = if scale_with_map { 1.0 } else { 1.0 / ratio };
                    // Circumscribed, so the polygon holds the circle.
                    let r = radius * q.view.world_per_px() * scale
                        / (std::f64::consts::PI / SIDES as f64).cos();
                    let ring = (0..SIDES)
                        .map(|i| {
                            let a = i as f64 * std::f64::consts::TAU / SIDES as f64;
                            q.project(world + DVec2::new(a.cos(), a.sin()) * r, 0.0)
                        })
                        .collect::<Option<Vec<_>>>()?;
                    for &v in &ring {
                        min = min.min(v);
                        max = max.max(v);
                    }
                    parts.push(vec![ring]);
                } else {
                    let r = (radius * ratio) as f32;
                    let c = q.project(world, 0.0)?;
                    min = min.min(c - r);
                    max = max.max(c + r);
                    circles.push((c, r));
                }
            }
        }
        let clips = copy
            .clips
            .iter()
            .map(|corners| {
                corners
                    .iter()
                    .filter_map(|&p| q.project(p + wrap, 0.0))
                    .collect()
            })
            .collect();
        Some(Self {
            clipped,
            kind: 0,
            radius: 0.0,
            parts,
            depths: Vec::new(),
            circles,
            min,
            max,
            clips,
            clip_pad: (shift.length() / q.view.world_per_px() + radius) as f32,
        })
    }

    /// A label's text and icon boxes `[x1, y1, x2, y2]` and collision
    /// circles `[x, y, radius]`, in pixels.
    fn label(boxes: Vec<[f64; 4]>, circles: Vec<[f64; 3]>) -> Self {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut parts = Vec::new();
        for b in boxes {
            let [x1, y1, x2, y2] = b.map(|v| v as f32);
            parts.push(vec![vec![
                Vec2::new(x1, y1),
                Vec2::new(x2, y1),
                Vec2::new(x2, y2),
                Vec2::new(x1, y2),
            ]]);
            min = min.min(Vec2::new(x1, y1));
            max = max.max(Vec2::new(x2, y2));
        }
        let circles: Vec<(Vec2, f32)> = circles
            .into_iter()
            .map(|[x, y, r]| (Vec2::new(x as f32, y as f32), r as f32))
            .collect();
        for &(c, r) in &circles {
            min = min.min(c - r);
            max = max.max(c + r);
        }
        Self {
            clipped: false,
            kind: LABEL,
            radius: 0.0,
            parts,
            depths: Vec::new(),
            circles,
            clips: Vec::new(),
            clip_pad: 0.0,
            min,
            max,
        }
    }

    fn contains(&self, p: DVec2) -> bool {
        let p = p.as_vec2();
        if !p.is_finite() || !p.cmpge(self.min).all() || !p.cmple(self.max).all() {
            return false;
        }
        let near = |clip: &[Vec2]| {
            inside(p, clip)
                || (0..clip.len())
                    .any(|i| distance(p, clip[i], clip[(i + 1) % clip.len()]) <= self.clip_pad)
        };
        if self.clipped && !self.clips.is_empty() && !self.clips.iter().any(|clip| near(clip)) {
            return false;
        }
        if self
            .circles
            .iter()
            .any(|&(c, r)| c.distance_squared(p) <= r * r)
        {
            return true;
        }
        self.parts.iter().any(|part| match self.kind {
            2 => part
                .iter()
                .flatten()
                .any(|q| q.distance_squared(p) <= self.radius * self.radius),
            1 => part.iter().any(|ring| {
                ring.windows(2)
                    .any(|pair| distance(p, pair[0], pair[1]) <= self.radius)
            }),
            LABEL => part.first().is_some_and(|outer| inside(p, outer)),
            _ => polygon_meets(part, p, p),
        })
    }

    /// Whether the feature drawn touches the box `min..max` (GL JS's box
    /// queries); a point query is a box with no area.
    fn intersects(&self, min: DVec2, max: DVec2) -> bool {
        if min == max {
            return self.contains(min);
        }
        let (min, max) = (min.as_vec2(), max.as_vec2());
        if self.max.cmplt(min).any() || self.min.cmpgt(max).any() {
            return false;
        }
        if self.clipped
            && !self.clips.is_empty()
            && !self.clips.iter().any(|clip| {
                polygon_meets(std::slice::from_ref(clip), min, max)
                    || ring_touches(clip, min, max, self.clip_pad, true)
            })
        {
            return false;
        }
        if self
            .circles
            .iter()
            .any(|&(c, r)| point_box_distance(c, min, max) <= r)
        {
            return true;
        }
        self.parts.iter().any(|part| match self.kind {
            2 => part
                .iter()
                .flatten()
                .any(|&q| point_box_distance(q, min, max) <= self.radius),
            1 => part
                .iter()
                .any(|ring| ring_touches(ring, min, max, self.radius, false)),
            _ => polygon_meets(part, min, max),
        })
    }

    /// How near the query `min..max` meets an extrusion, as GL JS's
    /// `checkIntersection` takes it: the nearest of the faces it meets, a
    /// point at its depth on the face, a box at the face's nearest corner.
    /// `None` for flat features.
    fn depth(&self, min: DVec2, max: DVec2) -> Option<f64> {
        let (min, max) = (min.as_vec2(), max.as_vec2());
        self.parts
            .iter()
            .zip(&self.depths)
            .filter(|(part, _)| polygon_meets(part, min, max))
            .map(|(part, depth)| match (min == max, part.first()) {
                (true, Some(outer)) => plane_depth(min, outer, depth),
                _ => depth.iter().copied().fold(f64::INFINITY, f64::min),
            })
            .reduce(f64::min)
    }
}

/// Whether a polygon (its outer ring, then its holes) meets the box
/// `min..max`, or holds the point when they are equal.
fn polygon_meets(part: &[Vec<Vec2>], min: Vec2, max: Vec2) -> bool {
    let Some(outer) = part.first() else {
        return false;
    };
    if min == max {
        return inside(min, outer) && !part[1..].iter().any(|hole| inside(min, hole));
    }
    let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
    outer
        .iter()
        .any(|q| q.cmpge(min).all() && q.cmple(max).all())
        || corners.iter().any(|&c| inside(c, outer))
        || part
            .iter()
            .any(|ring| ring_touches(ring, min, max, 0.0, true))
}

/// Whether a ring's edges (closing it if `closed`) come within `radius`
/// of the box `min..max`.
fn ring_touches(ring: &[Vec2], min: Vec2, max: Vec2, radius: f32, closed: bool) -> bool {
    let n = ring.len();
    let segments = if closed { n } else { n.saturating_sub(1) };
    (0..segments).any(|i| segment_box_distance(ring[i], ring[(i + 1) % n], min, max) <= radius)
        || (n == 1 && point_box_distance(ring[0], min, max) <= radius)
}

/// The depth at `p` of the plane through a planar face, from its points
/// and their depths: barycentric over its first triangle with area, as
/// GL JS's `getIntersectionDistance` takes it.
fn plane_depth(p: Vec2, face: &[Vec2], depth: &[f64]) -> f64 {
    let Some(b) = (1..face.len()).find(|&i| face[i] != face[0]) else {
        return f64::INFINITY;
    };
    let (a, p) = (face[0].as_dvec2(), p.as_dvec2());
    let (ab, ap) = (face[b].as_dvec2() - a, p - a);
    for c in b + 1..face.len() {
        let ac = face[c].as_dvec2() - a;
        let denom = ab.dot(ab) * ac.dot(ac) - ab.dot(ac) * ab.dot(ac);
        let v = (ac.dot(ac) * ap.dot(ab) - ab.dot(ac) * ap.dot(ac)) / denom;
        let w = (ab.dot(ab) * ap.dot(ac) - ab.dot(ac) * ap.dot(ab)) / denom;
        let d = depth[0] * (1.0 - v - w) + depth[b] * v + depth[c] * w;
        if d.is_finite() {
            return d;
        }
    }
    f64::INFINITY
}

/// A tile's rank among a layer's features, highest first: GL JS answers a
/// query tile by tile in `compareTileId` order (lower zooms first, then
/// rows and columns from the last), so a feature two tiles hold comes
/// from each in that order. `drawn` breaks ties between world copies.
fn tile_rank(tile: TileIndex, drawn: usize) -> u64 {
    let z = u64::from(31u8.saturating_sub(tile.z));
    (z << 58) | (u64::from(tile.row) << 30) | (u64::from(tile.col) << 2) | (drawn as u64 & 3)
}

/// A label's tile's rank among its layer's labels, highest first: GL JS
/// answers labels bucket by bucket in `sortTilesIn` order (lower zooms
/// first, then rows and columns from the first), each bucket's features
/// last first.
fn label_rank(tile: TileIndex) -> u64 {
    let key = (u64::from(tile.z) << 58) | (u64::from(tile.row) << 29) | u64::from(tile.col);
    u64::MAX - key
}

/// A line widened by `half` on each side, as polygons: a quadrilateral a
/// segment and a circumscribed octagon a vertex (round joins and caps).
fn buffer_line(line: &[DVec2], half: f64) -> Vec<Vec<DVec2>> {
    let mut out = Vec::new();
    for pair in line.windows(2) {
        let n = (pair[1] - pair[0]).normalize_or_zero().perp() * half;
        out.push(vec![pair[0] + n, pair[1] + n, pair[1] - n, pair[0] - n]);
    }
    let r = half / (std::f64::consts::PI / 8.0).cos();
    for &p in line {
        out.push(
            (0..8)
                .map(|i| p + DVec2::from_angle(f64::from(i) * std::f64::consts::FRAC_PI_4) * r)
                .collect(),
        );
    }
    out
}

/// GL JS `offsetLine`: a line moved `offset` to its right (y down), with
/// mitred joins.
fn offset_line(line: &[DVec2], offset: f64) -> Vec<DVec2> {
    let perp = |v: DVec2| {
        let u = v.normalize_or_zero();
        DVec2::new(-u.y, u.x)
    };
    (0..line.len())
        .map(|i| {
            let b = line[i];
            let a_to_b = if i == 0 {
                DVec2::ZERO
            } else {
                perp(b - line[i - 1])
            };
            let b_to_c = if i + 1 == line.len() {
                DVec2::ZERO
            } else {
                perp(line[i + 1] - b)
            };
            let mut extrude = (a_to_b + b_to_c).normalize_or_zero();
            let cos_half = extrude.dot(b_to_c);
            if cos_half != 0.0 {
                extrude /= cos_half;
            }
            b + extrude * offset
        })
        .collect()
}

fn point_box_distance(p: Vec2, min: Vec2, max: Vec2) -> f32 {
    (p.clamp(min, max) - p).length()
}

/// The distance between a segment and a box: zero when they meet.
fn segment_box_distance(a: Vec2, b: Vec2, min: Vec2, max: Vec2) -> f32 {
    let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
    let crosses = |p: Vec2, q: Vec2, r: Vec2, s: Vec2| {
        let cross = |u: Vec2, v: Vec2| u.x * v.y - u.y * v.x;
        let (d1, d2) = (cross(q - p, r - p), cross(q - p, s - p));
        let (d3, d4) = (cross(s - r, p - r), cross(s - r, q - r));
        (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
    };
    let inside_box = |p: Vec2| p.cmpge(min).all() && p.cmple(max).all();
    if inside_box(a)
        || inside_box(b)
        || (0..4).any(|i| crosses(a, b, corners[i], corners[(i + 1) % 4]))
    {
        return 0.0;
    }
    let ends = point_box_distance(a, min, max).min(point_box_distance(b, min, max));
    corners
        .iter()
        .map(|&c| distance(c, a, b))
        .fold(ends, f32::min)
}

fn distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let d = b - a;
    let t = if d.length_squared() == 0.0 {
        0.0
    } else {
        ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0)
    };
    p.distance(a + d * t)
}
fn inside(p: Vec2, ring: &[Vec2]) -> bool {
    let mut inside = false;
    for (a, b) in ring
        .iter()
        .zip(ring.iter().cycle().skip(1))
        .take(ring.len())
    {
        if distance(p, *a, *b) < 1e-7 {
            return true;
        }
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
    }
    inside
}

/// What a map's labels for picking were worked out for: placement
/// commits, the style version and the drawn list's revision.
pub(crate) type LabelsKey = (u64, u64, u64);

/// A map's picking state: the last frame's view and drawn tiles, the layer
/// table it drew with, and the labels of the commit it showed.
#[derive(Debug, Default)]
pub(crate) struct Picking {
    /// The last frame's view and camera.
    pub(crate) view: Option<(View, CameraState)>,
    pub(crate) drawn: Vec<DrawnTile>,
    /// Counts changes to `drawn`.
    pub(crate) drawn_rev: u64,
    /// The style generation the last frame shows in full.
    pub(crate) generation: u64,
    /// The layer table of the style version the last frame drew.
    pub(crate) layers: Arc<HitLayers>,
    pub(crate) layers_version: Option<u64>,
    /// The labels for (commits, style version, drawn revision).
    pub(crate) labels: Option<(LabelsKey, Arc<Vec<LabelHit>>)>,
}

impl Picking {
    /// Records a frame: its view, drawn tiles and generation.
    pub(crate) fn drew(
        &mut self,
        view: View,
        camera: CameraState,
        drawn: Vec<DrawnTile>,
        generation: u64,
    ) {
        self.view = Some((view, camera));
        if drawn != self.drawn {
            self.drawn = drawn;
            self.drawn_rev += 1;
        }
        self.generation = generation;
    }
}

impl crate::Map {
    /// Synchronous, topmost hit in CSS pixels of the last rendered frame.
    pub fn hit_test(&self, point: DVec2) -> Option<Hit> {
        self.with_scene(|scene, view, camera, zoom| scene.hit_test(view, camera, zoom, point))
            .flatten()
    }

    /// Runs `f` over the last frame's picking scene.
    pub(crate) fn with_scene<R>(
        &self,
        f: impl FnOnce(&Scene<'_>, &View, &CameraState, f64) -> R,
    ) -> Option<R> {
        let labels = self.label_hits();
        let picking = self.picking.borrow();
        let (view, camera) = picking.view.as_ref()?;
        let tiles: Vec<Option<TileHits>> = picking
            .drawn
            .iter()
            .map(|d| self.tile_hits(d.source, d.tile))
            .collect();
        let elevation = self.elevation();
        let scene = Scene {
            layers: &picking.layers,
            drawn: &picking.drawn,
            tiles: tiles.iter().map(Option::as_ref).collect(),
            labels: &labels,
            period: self.surface.period(),
            generation: picking.generation,
            elevation: elevation.as_deref(),
            states: self.style.as_ref().map(|s| &s.states),
        };
        Some(f(
            &scene,
            view,
            camera,
            camera.zoom(self.surface.projection()),
        ))
    }

    /// A resident tile's picking geometry, if it has any.
    pub(crate) fn tile_hits(&self, source: u32, tile: TileIndex) -> Option<TileHits> {
        let style = self.style.as_ref()?;
        let index = style
            .worker
            .plan()
            .sources
            .iter()
            .position(|s| s.key == source)?;
        let crate::resident::SourceState::Ready {
            pyramid, endpoint, ..
        } = &style.sources[index]
        else {
            return None;
        };
        let resident = pyramid.get(tile)?;
        let meshes: Vec<_> = resident
            .meshes
            .iter()
            .filter(|m| !m.hits.is_empty())
            .map(|m| (m.node, Arc::clone(&m.hits)))
            .collect();
        (!meshes.is_empty()).then(|| TileHits {
            mapping: TileMapping::of(&endpoint.tms, tile, resident.series.as_ref()),
            meshes,
        })
    }

    /// The labels the last commit shows, worked out once per commit; the
    /// last live frame's while the map draws still output.
    pub(crate) fn label_hits(&self) -> Arc<Vec<LabelHit>> {
        let Some(style) = &self.style else {
            return Arc::default();
        };
        if self.mode != crate::RenderMode::Continuous {
            return self
                .picking
                .borrow()
                .labels
                .as_ref()
                .map(|(_, labels)| Arc::clone(labels))
                .unwrap_or_default();
        }
        let key = (
            self.labels.commits(),
            self.version,
            self.picking.borrow().drawn_rev,
        );
        if let Some((k, labels)) = &self.picking.borrow().labels
            && *k == key
        {
            return Arc::clone(labels);
        }
        let labels = Arc::new(
            self.labels
                .label_hits(style.worker.plan(), &self.bucket_planes(style)),
        );
        self.picking.borrow_mut().labels = Some((key, Arc::clone(&labels)));
        labels
    }
}

/// A string as words: its length, then its UTF-8 bytes padded to a word.
fn write_text(words: &mut Vec<u32>, text: &str) {
    words.push(text.len() as u32);
    for chunk in text.as_bytes().chunks(4) {
        let mut bytes = [0; 4];
        bytes[..chunk.len()].copy_from_slice(chunk);
        words.push(u32::from_le_bytes(bytes));
    }
}

fn read_text(next: &mut dyn FnMut() -> Result<u32, String>) -> Result<String, String> {
    let len = next()? as usize;
    let mut bytes = Vec::new();
    for _ in 0..len.div_ceil(4) {
        bytes.extend_from_slice(&next()?.to_le_bytes());
    }
    bytes.truncate(len);
    String::from_utf8(bytes).map_err(|_| "malformed packed string".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A polygon an overzoom slice keeps whole answers as clipped at its
    /// layer's buffer: a square reaching past the box keeps what's in it.
    #[test]
    fn rings_clip_to_the_box() {
        let ring: Vec<DVec2> = [[-100.0, 10.0], [50.0, 10.0], [50.0, 300.0], [-100.0, 300.0]]
            .into_iter()
            .chain([[-100.0, 10.0]])
            .map(DVec2::from)
            .collect();
        let clipped = clip_ring(&ring, -20.0, 220.0);
        assert_eq!(clipped.first(), clipped.last());
        let (lo, hi) = clipped.iter().fold(
            (DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        assert_eq!((lo, hi), (DVec2::new(-20.0, 10.0), DVec2::new(50.0, 220.0)));
        // Wholly outside, nothing.
        let away: Vec<DVec2> = ring.iter().map(|p| *p + DVec2::new(1000.0, 0.0)).collect();
        assert!(clip_ring(&away, -20.0, 220.0).is_empty());
    }

    #[test]
    fn polygon_edges_holes_and_segments() {
        let outer = vec![
            Vec2::ZERO,
            Vec2::new(10., 0.),
            Vec2::splat(10.),
            Vec2::new(0., 10.),
        ];
        assert!(inside(Vec2::splat(5.), &outer));
        assert!(inside(Vec2::ZERO, &outer));
        assert!(!inside(Vec2::splat(11.), &outer));
        assert_eq!(
            distance(Vec2::new(5., 2.), Vec2::ZERO, Vec2::new(10., 0.)),
            2.
        );
    }

    /// A point meets a face at the face's depth under it; a box at the
    /// face's nearest corner. Leading repeated points don't count.
    #[test]
    fn face_depths() {
        let face = [Vec2::ZERO, Vec2::ZERO, Vec2::new(10., 0.), Vec2::splat(10.)];
        let depth = [0.2, 0.2, 0.4, 0.8];
        let at = plane_depth(Vec2::new(5., 5.), &face, &depth);
        assert!((at - 0.5).abs() < 1e-9, "{at}");
        let hit = ScreenHit {
            clipped: false,
            kind: 0,
            radius: 0.0,
            parts: vec![vec![face.to_vec()]],
            depths: vec![depth.to_vec()],
            circles: Vec::new(),
            clips: Vec::new(),
            clip_pad: 0.0,
            min: Vec2::ZERO,
            max: Vec2::splat(10.),
        };
        let depth = |min, max| hit.depth(DVec2::splat(min), DVec2::splat(max));
        assert_eq!(depth(9.0, 9.5), Some(0.2));
        assert_eq!(depth(20.0, 21.0), None);
    }

    /// GL JS's stacking: with extrusion layers 3 and 1 around a line
    /// layer 2, above a fill layer 0, a nearer extrusion of layer 1 waits
    /// under the line but still comes before a farther one of layer 3.
    #[test]
    fn extrusions_stack_by_depth() {
        let order = |solid: Vec<(f64, (char, usize))>| -> String {
            let flat = vec![('l', 2), ('f', 0)];
            stack(flat, solid, vec![1, 3, 1], |h| h.1)
                .into_iter()
                .map(|h| h.0)
                .collect()
        };
        assert_eq!(order(vec![(0.2, ('a', 3)), (0.4, ('b', 1))]), "albf");
        assert_eq!(order(vec![(0.4, ('a', 3)), (0.2, ('b', 1))]), "lbaf");
    }

    /// A tile's geometry holds each feature once, polygons closed and
    /// points in the tile, and crosses workers beside its nodes' records.
    #[test]
    fn mesh_hits_round_trip() {
        let mut b = orb_tile::LayerBuilder::new("l", 4096);
        b.begin_feature(Some(7), orb_tile::GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([1, 2]);
        b.push_vertex([-3, 4]);
        b.begin_feature(None, orb_tile::GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[0, 0], [10, 0], [10, 10]] {
            b.push_vertex(p);
        }
        let layer = b.finish();
        let mut geometry = HitGeometry::default();
        let point = geometry.add(&layer, 0, Some(orb_tile::ValueRef::Str("seven")), true);
        let polygon = geometry.add(&layer, 1, None, true);
        assert_eq!(geometry.add(&layer, 1, None, true), polygon);
        let mut hits = MeshHits::new(0);
        hits.push(&geometry, point, 0, 2, [5.0, 0.0, 0.0, 0.0]);
        hits.push(&geometry, polygon, 1, 0, [0.0; 4]);
        hits.link(&[Arc::new(geometry)]);
        let decoded = hits.decode();
        assert_eq!(decoded[0].id.as_deref(), Some("seven"));
        assert_eq!(decoded[0].parts, [vec![vec![[1, 2]]]]);
        assert_eq!(
            decoded[1].parts,
            [vec![vec![[0, 0], [10, 0], [10, 10], [0, 0]]]]
        );
        assert_eq!(hits.header(1), 5.0);
        let mut copy: MeshHits = crate::wire::decode(&crate::wire::encode(&hits)).unwrap();
        copy.link(&[Arc::clone(&hits.geometry)]);
        assert_eq!(copy.decode(), decoded);
        copy.words.truncate(MESH_HEADER + NODE_RECORD + 3);
        assert_eq!(copy.decode().len(), 1, "a malformed record is skipped");
    }
}
