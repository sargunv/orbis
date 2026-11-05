//! Picking, independent of the camera (decision 0024). Tile jobs write each
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

use crate::jobs::{read_text, write_text};

/// The topmost feature under a point in the last published frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub layer: String,
    pub source: String,
    pub source_layer: Option<String>,
    pub id: Option<String>,
    pub generation: u64,
}

/// A hit-testable feature as a tile job builds it; [`MeshHits`] packs a
/// node's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HitFeature {
    pub id: Option<String>,
    /// The feature's index in its source layer, for queries.
    pub feature: u32,
    /// 0: polygons, 1: lines, 2: points.
    pub kind: u32,
    pub extent: u32,
    /// Circles: the radius; lines: half the width.
    pub radius: f32,
    /// Circles: the stroke width, bound apart from the radius.
    pub stroke: f32,
    pub height: f32,
    pub base: f32,
    pub parts: Vec<Vec<Vec<[i32; 2]>>>,
}

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
    fn f32(&mut self) -> Result<f32, String> {
        self.next().map(f32::from_bits)
    }
    fn text(&mut self) -> Result<String, String> {
        read_text(&mut || self.next())
    }
    fn done(&self) -> bool {
        self.at == self.words.len()
    }
}

/// One render node's hit-testable features in one tile, packed in
/// tile-local data units as the tile job wrote them, with their bounds, so
/// that neither the render thread nor a UI thread decodes them to pick.
///
/// Words: the feature count; the largest radius, stroke and height (or
/// base) of any feature (f32); the features' extent (0 when they differ);
/// their bounds `min_x, min_y, max_x, max_y` (i32); then per feature its
/// length in words and `min_x, min_y, max_x, max_y, has_id, [id], feature,
/// kind, extent, radius, stroke, height, base, parts, {rings, {points,
/// {x, y}}}`.
#[derive(Debug)]
pub struct MeshHits {
    words: Vec<u32>,
    /// Per feature: its record's first word and world bounds, worked out
    /// on first use.
    bounds: OnceLock<Vec<Bounds>>,
}

impl MeshHits {
    /// Bytes its records take.
    pub fn bytes(&self) -> u64 {
        self.words.len() as u64 * 4
    }
}

impl Default for MeshHits {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl PartialEq for MeshHits {
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    record: u32,
    min: DVec2,
    max: DVec2,
}

const MESH_HEADER: usize = 9;
/// A record's bounds, before the feature.
const RECORD_BOUNDS: usize = 4;

/// One feature of a [`MeshHits`], unpacked but for its geometry.
struct FeatureRef<'a> {
    id: Option<&'a [u32]>,
    feature: u32,
    kind: u32,
    extent: u32,
    radius: f32,
    stroke: f32,
    height: f32,
    base: f32,
    geometry: &'a [u32],
}

impl<'a> FeatureRef<'a> {
    fn parse(record: &'a [u32]) -> Result<Self, String> {
        let mut r = Reader::new(record);
        r.take(RECORD_BOUNDS)?;
        let id = if r.next()? != 0 {
            let start = r.at;
            let len = r.next()? as usize;
            r.take(len.div_ceil(4))?;
            Some(&record[start..r.at])
        } else {
            None
        };
        Ok(Self {
            id,
            feature: r.next()?,
            kind: r.next()?,
            extent: r.next()?,
            radius: r.f32()?,
            stroke: r.f32()?,
            height: r.f32()?,
            base: r.f32()?,
            geometry: &record[r.at..],
        })
    }

    fn id(&self) -> Option<String> {
        let mut r = Reader::new(self.id?);
        r.text().ok()
    }

    /// Per part, its rings' points.
    fn parts(&self) -> Result<Vec<Vec<&'a [u32]>>, String> {
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

fn point(words: &[u32; 2]) -> [i32; 2] {
    [words[0] as i32, words[1] as i32]
}

impl MeshHits {
    /// Packs a node's features.
    pub fn new(features: &[HitFeature]) -> Self {
        let mut words = vec![0; MESH_HEADER];
        words[0] = features.len() as u32;
        let (mut radius, mut stroke, mut height) = (0f32, 0f32, 0f32);
        let extent = features.first().map_or(0, |f| f.extent);
        let extent = if features.iter().all(|f| f.extent == extent) {
            extent
        } else {
            0
        };
        let (mut lo, mut hi) = ([i32::MAX; 2], [i32::MIN; 2]);
        for h in features {
            let start = words.len();
            let (mut min, mut max) = ([i32::MAX; 2], [i32::MIN; 2]);
            for p in h.parts.iter().flatten().flatten() {
                for i in 0..2 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
            for i in 0..2 {
                lo[i] = lo[i].min(min[i]);
                hi[i] = hi[i].max(max[i]);
            }
            words.push(0);
            words.extend([min[0], min[1], max[0], max[1]].map(|v| v as u32));
            words.push(u32::from(h.id.is_some()));
            if let Some(id) = &h.id {
                write_text(&mut words, id);
            }
            words.extend([
                h.feature,
                h.kind,
                h.extent,
                h.radius.to_bits(),
                h.stroke.to_bits(),
                h.height.to_bits(),
                h.base.to_bits(),
                h.parts.len() as u32,
            ]);
            for part in &h.parts {
                words.push(part.len() as u32);
                for ring in part {
                    words.push(ring.len() as u32);
                    for p in ring {
                        words.extend(p.map(|v| v as u32));
                    }
                }
            }
            words[start] = (words.len() - start - 1) as u32;
            radius = radius.max(h.radius);
            stroke = stroke.max(h.stroke);
            height = height.max(h.height).max(h.base);
        }
        words[1] = radius.to_bits();
        words[2] = stroke.to_bits();
        words[3] = height.to_bits();
        words[4] = extent;
        for (i, v) in [lo[0], lo[1], hi[0], hi[1]].into_iter().enumerate() {
            words[5 + i] = v as u32;
        }
        Self {
            words,
            bounds: OnceLock::new(),
        }
    }

    /// Takes packed words, as [`words`](Self::words) gave them, checking
    /// their framing; a feature malformed within its record is skipped
    /// when picked.
    pub fn from_words(words: Vec<u32>) -> Result<Self, String> {
        let mut r = Reader::new(&words);
        let count = r.next()?;
        r.take(MESH_HEADER - 1)?;
        for _ in 0..count {
            let len = r.next()? as usize;
            if len < RECORD_BOUNDS {
                return Err(MALFORMED.into());
            }
            r.take(len)?;
        }
        if !r.done() {
            return Err(MALFORMED.into());
        }
        Ok(Self {
            words,
            bounds: OnceLock::new(),
        })
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn is_empty(&self) -> bool {
        self.words.first().is_none_or(|&n| n == 0)
    }

    fn header(&self, i: usize) -> f32 {
        self.words.get(i).map_or(0.0, |&w| f32::from_bits(w))
    }

    /// Each feature's record, after its length word, with that word's index.
    fn records(&self) -> Result<Vec<(u32, &[u32])>, String> {
        let mut r = Reader::new(&self.words);
        let count = r.next()?;
        r.take(MESH_HEADER - 1)?;
        let mut records = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let at = r.at as u32;
            let len = r.next()? as usize;
            records.push((at, r.take(len)?));
        }
        if !r.done() {
            return Err(MALFORMED.into());
        }
        Ok(records)
    }

    fn record(&self, at: u32) -> &[u32] {
        let at = at as usize;
        &self.words[at + 1..at + 1 + self.words[at] as usize]
    }

    /// The features, unpacked.
    pub fn decode(&self) -> Vec<HitFeature> {
        self.records()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, record)| {
                let f = FeatureRef::parse(record).ok()?;
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
                    kind: f.kind,
                    extent: f.extent,
                    radius: f.radius,
                    stroke: f.stroke,
                    height: f.height,
                    base: f.base,
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
            for (at, record) in self.records().unwrap_or_default() {
                let Ok(f) = FeatureRef::parse(record) else {
                    continue;
                };
                let (min, max) = match mapping {
                    TileMapping::Affine { .. } => {
                        affine_bounds(mapping, f.extent, &record[..RECORD_BOUNDS])
                    }
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
        tile: TileIndex,
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
    /// Its text and icon boxes `[x1, y1, x2, y2]`, shown ones only.
    pub(crate) boxes: Vec<[f64; 4]>,
    /// A line label's collision circle radius, for one with no box.
    pub(crate) circle: Option<f64>,
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
        translate: Property<[f32; 2]>,
        anchor: Property<TranslateAnchor>,
    },
    Circle {
        radius: Property<f32>,
        stroke: Property<f32>,
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
            CIRCLE_RADIUS, CIRCLE_STROKE_WIDTH, CIRCLE_TRANSLATE, CIRCLE_TRANSLATE_ANCHOR,
            EXTRUSION_BASE, EXTRUSION_HEIGHT, EXTRUSION_TRANSLATE, EXTRUSION_TRANSLATE_ANCHOR,
            FILL_TRANSLATE, FILL_TRANSLATE_ANCHOR, LINE_TRANSLATE, LINE_TRANSLATE_ANCHOR,
            LINE_WIDTH,
        };
        match l.ty {
            LayerType::Fill => Self::Fill {
                translate: l.get(FILL_TRANSLATE).to_owned(),
                anchor: l.get(FILL_TRANSLATE_ANCHOR).to_owned(),
            },
            LayerType::Line => Self::Line {
                width: l.get(LINE_WIDTH).to_owned(),
                translate: l.get(LINE_TRANSLATE).to_owned(),
                anchor: l.get(LINE_TRANSLATE_ANCHOR).to_owned(),
            },
            LayerType::Circle => Self::Circle {
                radius: l.get(CIRCLE_RADIUS).to_owned(),
                stroke: l.get(CIRCLE_STROKE_WIDTH).to_owned(),
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
}

/// A feature a query may report, in the order that puts upper ones last:
/// its layer, then tiles before labels, then the frame's drawing order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Order {
    layer: usize,
    label: bool,
    tile: u32,
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
        let ground = self
            .elevation
            .and_then(|e| e.at(world))
            .map_or(0.0, |metres| metres * self.view.height_scale());
        self.view
            .project_3d(world, ground + height)
            .map(|p| p.as_vec2())
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
            shift,
        }
    }

    /// The feature's tolerance in pixels and its base and top in world units.
    fn feature(&self, layer: &HitLayer, at: &LayerAt, f: &FeatureRef<'_>) -> (f32, (f64, f64)) {
        let radius = match &layer.paint {
            HitPaint::Circle { .. } => {
                at.radius.unwrap_or(f.radius) + at.stroke.unwrap_or(f.stroke)
            }
            HitPaint::Line { .. } => at.radius.unwrap_or(f.radius),
            _ => f.radius,
        };
        let elevation = match &layer.paint {
            HitPaint::Extrusion { .. } => {
                let scale = self.view.height_scale();
                (
                    f64::from(at.base.unwrap_or(f.base)) * scale,
                    f64::from(at.height.unwrap_or(f.height)) * scale,
                )
            }
            _ => (0.0, 0.0),
        };
        (radius, elevation)
    }

    /// The largest tolerance and top any of a mesh's features can have.
    fn mesh_bound(&self, layer: &HitLayer, at: &LayerAt, mesh: &MeshHits) -> (f32, f64) {
        let (radius, stroke, height) = (mesh.header(1), mesh.header(2), mesh.header(3));
        match &layer.paint {
            HitPaint::Circle { .. } => (
                at.radius.unwrap_or(radius) + at.stroke.unwrap_or(stroke),
                0.0,
            ),
            HitPaint::Line { .. } => (at.radius.unwrap_or(radius), 0.0),
            HitPaint::Extrusion { .. } => {
                let top = at.height.unwrap_or(height).max(at.base.unwrap_or(height));
                (radius, f64::from(top) * self.view.height_scale())
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
        let (mut pad, mut top) = (0f32, 0f64);
        for tile in self.tiles.iter().flatten() {
            for (node, mesh) in &tile.meshes {
                let Some(layer) = layer_at(*node) else {
                    continue;
                };
                let at = layers.entry(*node).or_insert_with(|| q.layer(layer));
                let (radius, height) = q.mesh_bound(layer, at, mesh);
                pad = pad.max(radius);
                top = top.max(height);
            }
        }
        // A pixel's worth for the projection's rounding to f32.
        let ground = q.ground(min, max, f64::from(pad) + 1.0, top);
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
                                tile: i as u32,
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
        for (i, label) in self.labels.iter().enumerate() {
            let Some(layer) = self.layers.nodes.get(&label.node) else {
                continue;
            };
            out.push((
                Order {
                    layer: layer.order,
                    label: true,
                    tile: i as u32,
                    mesh: 0,
                    copy: 0,
                    feature: 0,
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
                let f = FeatureRef::parse(mesh.record(record)).ok()?;
                let (radius, elevation) = q.feature(layer, &at, &f);
                let screen = ScreenHit::project(
                    &f,
                    layer.source_layer.is_some(),
                    mapping,
                    copy,
                    q,
                    self.period,
                    f64::from(radius),
                    elevation,
                    at.shift,
                )?;
                let key = FeatureKey {
                    source: drawn.source,
                    tile: drawn.tile,
                    feature: f.feature,
                };
                Some((screen, layer.hit(f.id(), self.generation), key))
            }
            Candidate::Label { layer, label } => {
                let anchor = q.project(label.anchor, 0.0)?.as_dvec2();
                // GL JS: tile units at 512 px per tile side, scaled by the
                // perspective ratio.
                let scale =
                    512.0 / orb_symbol::EXTENT * (0.5 + 0.5 * q.local_scale(label.anchor)?);
                let mut bounds = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for b in &label.boxes {
                    bounds = [
                        bounds[0].min(anchor.x + b[0] * scale),
                        bounds[1].min(anchor.y + b[1] * scale),
                        bounds[2].max(anchor.x + b[2] * scale),
                        bounds[3].max(anchor.y + b[3] * scale),
                    ];
                }
                if !bounds.iter().all(|b| b.is_finite()) {
                    // A line label collides by circles along its line; its
                    // anchor's stands for them.
                    let r = label.circle?;
                    bounds = [anchor.x - r, anchor.y - r, anchor.x + r, anchor.y + r];
                }
                Some((
                    ScreenHit::label(bounds),
                    layer.hit(None, self.generation),
                    label.key,
                ))
            }
        }
    }

    /// The topmost feature under `point`, in pixels of `view`.
    pub(crate) fn hit_test(
        &self,
        view: &View,
        camera: &CameraState,
        zoom: f64,
        point: DVec2,
    ) -> Option<Hit> {
        if !point.is_finite() {
            return None;
        }
        let q = Query {
            view,
            camera,
            zoom,
            elevation: self.elevation,
        };
        self.candidates(&q, point, point)
            .iter()
            .filter_map(|(_, c)| self.project(&q, c))
            .find(|(screen, ..)| screen.contains(point))
            .map(|(_, hit, _)| hit)
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
        let mut out = Vec::new();
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
                out.push((hit, order.layer, key));
            }
        }
        out
    }
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
    clips: Vec<Vec<Vec2>>,
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
        radius: f64,
        elevation: (f64, f64),
        shift: DVec2,
    ) -> Option<Self> {
        let transform = mapping.mapper(feature.extent);
        let wrap = DVec2::new(f64::from(copy.wrap) * period, 0.0);
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let project = |p: [i32; 2], height: f64| {
            q.project(
                transform(DVec2::new(p[0] as f64, p[1] as f64)) + wrap + shift,
                height,
            )
        };
        let mut parts = Vec::new();
        for part in feature.parts().ok()? {
            let mut rings = Vec::new();
            for ring in &part {
                let points = ring
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| project(point(p), elevation.1))
                    .collect::<Option<Vec<_>>>()?;
                rings.push(points);
            }
            parts.push(rings);
            if elevation.1 > elevation.0 {
                for ring in &part {
                    let ring: Vec<[i32; 2]> = ring.as_chunks::<2>().0.iter().map(point).collect();
                    for (a, b) in ring
                        .iter()
                        .zip(ring.iter().cycle().skip(1))
                        .take(ring.len())
                    {
                        parts.push(vec![vec![
                            project(*a, elevation.0)?,
                            project(*b, elevation.0)?,
                            project(*b, elevation.1)?,
                            project(*a, elevation.1)?,
                        ]]);
                    }
                }
            }
        }
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
                    .filter_map(|&p| q.project(p + wrap + shift, 0.0))
                    .collect()
            })
            .collect();
        Some(Self {
            clipped,
            kind: feature.kind,
            radius,
            parts,
            min,
            max,
            clips,
        })
    }

    /// A label's or icon's box `[x1, y1, x2, y2]` in pixels.
    fn label(bounds: [f64; 4]) -> Self {
        let [x1, y1, x2, y2] = bounds.map(|v| v as f32);
        let ring = vec![
            Vec2::new(x1, y1),
            Vec2::new(x2, y1),
            Vec2::new(x2, y2),
            Vec2::new(x1, y2),
        ];
        Self {
            clipped: false,
            kind: 3,
            radius: 0.0,
            parts: vec![vec![ring]],
            clips: Vec::new(),
            min: Vec2::new(x1, y1),
            max: Vec2::new(x2, y2),
        }
    }

    fn contains(&self, p: DVec2) -> bool {
        let p = p.as_vec2();
        if !p.is_finite() || !p.cmpge(self.min).all() || !p.cmple(self.max).all() {
            return false;
        }
        if self.clipped && !self.clips.is_empty() && !self.clips.iter().any(|clip| inside(p, clip))
        {
            return false;
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
            3 => part.first().is_some_and(|outer| inside(p, outer)),
            _ => {
                part.first().is_some_and(|outer| inside(p, outer))
                    && !part.iter().skip(1).any(|hole| inside(p, hole))
            }
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
        let in_box = |q: Vec2| q.cmpge(min).all() && q.cmple(max).all();
        let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
        let edges_touch = |ring: &[Vec2], radius: f32, closed: bool| {
            let n = ring.len();
            let segments = if closed { n } else { n.saturating_sub(1) };
            (0..segments).any(|i| {
                let (a, b) = (ring[i], ring[(i + 1) % n]);
                segment_box_distance(a, b, min, max) <= radius
            }) || (n == 1 && point_box_distance(ring[0], min, max) <= radius)
        };
        if self.clipped
            && !self.clips.is_empty()
            && !self.clips.iter().any(|clip| {
                clip.iter().copied().any(in_box)
                    || corners.iter().any(|&c| inside(c, clip))
                    || edges_touch(clip, 0.0, true)
            })
        {
            return false;
        }
        self.parts.iter().any(|part| match self.kind {
            2 => part
                .iter()
                .flatten()
                .any(|&q| point_box_distance(q, min, max) <= self.radius),
            1 => part
                .iter()
                .any(|ring| edges_touch(ring, self.radius, false)),
            _ => {
                let Some(outer) = part.first() else {
                    return false;
                };
                outer.iter().copied().any(in_box)
                    || corners.iter().any(|&c| inside(c, outer))
                    || part.iter().any(|ring| edges_touch(ring, 0.0, true))
            }
        })
    }
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
        let crate::map::SourceState::Ready {
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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn mesh_hits_round_trip() {
        let features = vec![
            HitFeature {
                id: Some("seven".into()),
                feature: 3,
                kind: 2,
                extent: 4096,
                radius: 5.0,
                stroke: 2.0,
                parts: vec![vec![vec![[1, 2], [-3, 4]]]],
                ..HitFeature::default()
            },
            HitFeature {
                feature: 4,
                kind: 0,
                extent: 4096,
                height: 30.0,
                parts: vec![vec![vec![[0, 0], [10, 0], [10, 10]], vec![]]],
                ..HitFeature::default()
            },
        ];
        let hits = MeshHits::new(&features);
        assert_eq!(hits.decode(), features);
        let copy = MeshHits::from_words(hits.words().to_vec()).unwrap();
        assert_eq!(copy, hits);
        assert_eq!(
            (hits.header(1), hits.header(2), hits.header(3)),
            (5.0, 2.0, 30.0)
        );
        let mut words = hits.words().to_vec();
        words.pop();
        assert!(MeshHits::from_words(words).is_err());
    }
}
